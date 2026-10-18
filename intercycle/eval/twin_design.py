"""The EVAL design and the re-run rule of the press-shop twin (PLAN_TWIN §5 "Design" and "Rules"), shared by the EVAL
runner (intercycle.eval.twin_eval) and its truth slot (twin.eval_metrics), so both score the same attempt per cell.

Isaac-safe imports only (stdlib, yaml, pydantic via twin.schemas): twin.eval_metrics is a twin module and the Isaac
dependency scan follows it here. Nothing here reads a log or opens truth; a run is known by its summary.json, its
directory name and classify_invalid's committed list.

  * Design: on the test split the design is explicit (the world list is required; days default to TEST_DAYS; the CPU
    double has all CONDITIONS) and, for the CPU double, the sealed design K_TEST x TEST_DAYS x CONDITIONS is enforced
    unless allow_partial_design (an API argument for synthetic tests only; `Design.fixed` is then False). A test run
    outside the design is refused. On the dev split worlds and days default to the runs present.
  * Runs: every directory under the runs root holding a run file (RUN_MARKERS), with or without summary.json, plus a
    placeholder per invalid-listed id with no directory. A run without a usable summary is placed in its design cell
    by its invalid-list record ({"run_id", "world", "day", "zone", "condition"[, "split", "engine"]}) or by its run_id
    (w<W>_d<D>_<zone>_<condition>[__a<k>]).
  * Re-runs: attempt k = the `__a<k>` suffix of the run_id (none = 0). Per (world, day, zone) the first attempt whose
    runs are all valid (none listed invalid, none with status infra_invalid) is scored; without one, the last attempt
    is scored and the cell is unresolved.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from twin.schemas.runs import Summary

CONDITIONS = ("no_map", "static", "learned", "reactive", "oracle")
REQUIRED = ("static", "learned", "reactive")
K_TEST = 16                        # test worlds (PLAN_TWIN §5 Design)
TEST_DAYS = (9, 10)                # scored days
# files that mark a run directory (a crashed container may leave any of them, or nothing but its id in the invalid list)
RUN_MARKERS = ("summary.json", "log.parquet", "log.jsonl", "obs_v2.jsonl", "plant_summary.json", "episode.json",
               "sim.log", "held_map.npz")
CELL_KEYS = ("world", "day", "zone", "condition")
_ATTEMPT = re.compile(r"__a(\d+)$")
_RUN_ID = re.compile(r"^w(\d+)_d(\d+)_(.+)_(" + "|".join(CONDITIONS) + r")(?:__a(\d+))?$")

Cell = tuple[int, int, str, str]          # (world, day, zone, condition)


class EvalError(ValueError):
    """The EVAL inputs are inconsistent (fail closed)."""


def parse_run_id(rid: str) -> tuple[Cell, int] | None:
    """(cell, attempt) of a run_id of the form w<W>_d<D>_<zone>_<condition>[__a<k>], else None."""
    m = _RUN_ID.match(rid)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)), int(m.group(5) or 0)


@dataclass
class Run:
    run_dir: Path
    run_id: str
    summary: Summary | None
    rows: list[dict] = field(default_factory=list)     # the log rows (twin_eval.load_run; empty here)
    problems: list[str] = field(default_factory=list)
    cell: Cell | None = None                # design cell of a run without a usable summary (set by `index_cells`)

    @property
    def attempt(self) -> int:
        m = _ATTEMPT.search(self.run_id)
        return int(m.group(1)) if m else 0

    @property
    def key(self) -> Cell | None:
        s = self.summary
        return self.cell if s is None else (s.world, s.day, s.zone, s.condition)


def read_summary(run_dir: str | Path) -> tuple[Summary | None, str, list[str]]:
    """(summary or None, run_id, problems) of one run directory. Problems are recorded, never raised."""
    d = Path(run_dir)
    problems: list[str] = []
    summary, rid = None, d.name
    try:
        raw = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rid = str(raw.get("run_id", rid))
        summary = Summary.model_validate(raw)
    except Exception as e:  # noqa: BLE001 - an unreadable summary is an error mission
        problems.append(f"summary.json unreadable or invalid: {e!r}"[:400])
    named = parse_run_id(rid)
    if summary is not None and named is not None:
        own = (summary.world, summary.day, summary.zone, summary.condition)
        if named[0] != own:
            problems.append(f"run_id {rid} names cell {named[0]}, the summary says {own}")
    return summary, rid, problems


def find_runs(root: str | Path) -> list[Path]:
    """Every directory under root that holds a run file (RUN_MARKERS), with or without summary.json."""
    out: set[Path] = set()
    for m in RUN_MARKERS:
        out |= {p.parent for p in Path(root).rglob(m)}
    return sorted(out)


def mission_orders(tasks_path: str | Path) -> dict[str, list[str]]:
    t = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8"))
    if not isinstance(t, Mapping) or t.get("contract") != "twin.tasks/1":
        raise EvalError(f"{tasks_path}: not a twin.tasks/1 file")
    return {str(z["zone"]): [str(c) for c in z["mission_order"]] for z in t["zones"]}


def load_invalid(path: str | Path | None) -> dict[str, dict | None]:
    """classify_invalid's committed list: a JSON list, or {"invalid": [...]} / {"runs": [...]}, of run ids or of
    records {"run_id", "world", "day", "zone", "condition"[, "split", "engine"]} that place a crashed run in its
    design cell. Returns run id -> its record (None for a bare id)."""
    if path is None:
        return {}
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(body, Mapping):
        body = body.get("invalid", body.get("runs"))
    if not isinstance(body, list):
        raise EvalError(f"{path}: expected a list of run ids or records")
    out: dict[str, dict | None] = {}
    for e in body:
        if isinstance(e, str):
            rid, rec = e, None
        elif isinstance(e, Mapping) and isinstance(e.get("run_id"), str):
            rid, rec = str(e["run_id"]), dict(e)
            have = [k for k in CELL_KEYS if k in rec]
            if have and len(have) != len(CELL_KEYS):
                raise EvalError(f"{path}: record {rid} names only {have} of the cell keys {CELL_KEYS}")
        else:
            raise EvalError(f"{path}: expected a run id or a record with a run_id, got {e!r}")
        if rid in out:
            raise EvalError(f"{path}: run id {rid} is listed twice")
        out[rid] = rec
    return out


def placeholder_runs(invalid: Mapping[str, dict | None], runs: Sequence[Run]) -> list[Run]:
    """A Run for every invalid-listed id without a run directory (the container died before writing anything)."""
    seen = {r.run_id for r in runs} | {r.run_dir.name for r in runs}
    return [Run(Path(rid), rid, None, [], ["no run directory (listed infrastructure-invalid)"])
            for rid in sorted(invalid) if rid not in seen]


def summary_runs(runs_root: str | Path, invalid: Mapping[str, dict | None]) -> list[Run]:
    """The runs under runs_root known by their summaries only (no log rows), plus the invalid placeholders."""
    runs = []
    for d in find_runs(runs_root):
        s, rid, problems = read_summary(d)
        runs.append(Run(d, rid, s, [], problems))
    return runs + placeholder_runs(invalid, runs)


def placed(r: Run, invalid: Mapping[str, dict | None]) -> tuple[Cell, Any, Any] | None:
    """(cell, split, engine) of a run without a usable summary: from its invalid record, else from its run_id; None if
    neither places it. split / engine are None unless the record names them."""
    rec = invalid.get(r.run_id) or invalid.get(r.run_dir.name)
    if rec is not None and all(k in rec for k in CELL_KEYS):
        cell = (int(rec["world"]), int(rec["day"]), str(rec["zone"]), str(rec["condition"]))
        return cell, rec.get("split"), rec.get("engine")
    named = parse_run_id(r.run_id) or parse_run_id(r.run_dir.name)
    return None if named is None else (named[0], None, None)


def is_bad(r: Run, invalid: Mapping[str, dict | None]) -> bool:
    """Infrastructure-invalid: listed by classify_invalid (by run_id or directory name), or status infra_invalid."""
    return (r.run_id in invalid or r.run_dir.name in invalid
            or (r.summary is not None and r.summary.status == "infra_invalid"))


@dataclass(frozen=True)
class Design:
    """The cells EVAL scores: worlds x days x zones (tasks.yaml) x conditions. fixed = the sealed design was enforced
    (test split, CPU double: K_TEST worlds, TEST_DAYS, all CONDITIONS)."""
    worlds: tuple[int, ...]
    days: tuple[int, ...]
    conditions: tuple[str, ...]
    fixed: bool

    def as_dict(self) -> dict:
        return {"worlds": list(self.worlds), "days": list(self.days), "conditions": list(self.conditions),
                "fixed": self.fixed, "k_expected": K_TEST, "test_days": list(TEST_DAYS)}


def make_design(runs: Sequence[Run], invalid: Mapping[str, dict | None], *, split: str, engine: str,
                worlds: Sequence[int] | None = None, days: Sequence[int] | None = None,
                conditions: Sequence[str] | None = None, allow_partial_design: bool = False) -> Design:
    """The design of the evaluation (module doc)."""
    bad_c = sorted(set(conditions or ()) - set(CONDITIONS))
    if bad_c:
        raise EvalError(f"unknown conditions {bad_c}")
    if split == "test":
        if not worlds:
            raise EvalError("split test needs the design's world list (--worlds): worlds are never taken from the "
                            "runs present (an unsynced world would be dropped)")
        wl = [int(w) for w in worlds]
        ws = tuple(sorted(set(wl)))
        if len(ws) != len(wl):
            raise EvalError(f"--worlds repeats a world: {wl}")
        ds = tuple(sorted(set(int(d) for d in days))) if days else TEST_DAYS
        cs = tuple(c for c in CONDITIONS if c in set(conditions)) if conditions else (
            CONDITIONS if engine == "double" else REQUIRED)
        sealed = engine == "double"
        if sealed and not allow_partial_design:
            if len(ws) != K_TEST:
                raise EvalError(f"the sealed design has K = {K_TEST} test worlds, got {len(ws)}: {list(ws)}")
            if ds != TEST_DAYS:
                raise EvalError(f"the sealed design scores days {TEST_DAYS}, got {list(ds)}")
            if cs != CONDITIONS:
                raise EvalError(f"the sealed design has all conditions {CONDITIONS}, got {list(cs)}")
        missing = [c for c in REQUIRED if c not in cs]
        if missing:
            raise EvalError(f"conditions {missing} are required (H1 / H2)")
        return Design(ws, ds, cs, bool(sealed and not allow_partial_design))
    cells = []
    for r in runs:
        s = r.summary
        if s is not None:
            if s.world_split == split and s.engine == engine:
                cells.append((s.world, s.day, s.zone, s.condition))
        else:
            p = placed(r, invalid)
            if p is not None and p[1] in (None, split) and p[2] in (None, engine):
                cells.append(p[0])
    ws = tuple(sorted(set(worlds) if worlds is not None else {c[0] for c in cells}))
    ds = tuple(sorted(set(days) if days is not None else {c[1] for c in cells}))
    present = {c[3] for c in cells}
    cs = tuple(conditions) if conditions else tuple(c for c in CONDITIONS if c in present or c in REQUIRED)
    if not ws or not ds:
        raise EvalError(f"no {engine} runs on {split} worlds")
    return Design(ws, ds, cs, False)


@dataclass
class Cells:
    """Runs indexed by design cell and attempt, and the attempt scored per (world, day, zone)."""
    by_cell: dict[tuple[int, int, str], dict[int, dict[str, Run]]]
    chosen: dict[tuple[int, int, str], int | None]          # scored attempt (None: no run at all)
    first: dict[tuple[int, int, str], int | None]           # attempt 0 / the earliest attempt
    unresolved: list[str]                                   # cells without a clean attempt
    unplaceable: list[str]                                  # run dirs whose summary and id give no design cell
    excluded_non_primary: list[str]                         # reset-sweep / short dev runs (never pooled)

    def runs_of(self, cell: tuple[int, int, str]) -> dict[str, Run]:
        """condition -> run of the scored attempt of a (world, day, zone)."""
        a = self.chosen.get(cell)
        return {} if a is None else dict(self.by_cell[cell][a])


def index_cells(runs: Iterable[Run], zones: Iterable[str], invalid: Mapping[str, dict | None], *, split: str,
                engine: str, design: Design) -> Cells:
    """Place every run of the evaluated split and engine in its design cell and pick the scored attempt per cell."""
    zs = set(zones)
    by_cell: dict[tuple[int, int, str], dict[int, dict[str, Run]]] = defaultdict(lambda: defaultdict(dict))
    unplaceable, non_primary = [], []
    strict = split == "test"
    for r in runs:
        s = r.summary
        if s is not None:
            if s.world_split != split or s.engine != engine:
                continue
            if s.variant != "primary" or s.reset_override_s is not None:
                non_primary.append(r.run_id)                # descriptive runs never enter the primary metric
                continue
            key = (s.world, s.day, s.zone, s.condition)
        else:
            p = placed(r, invalid)
            if p is None:
                unplaceable.append(str(r.run_dir))
                continue
            key, sp, en = p
            if (sp is not None and sp != split) or (en is not None and en != engine):
                continue
            r.cell = key
        w, d, z, c = key
        if w not in design.worlds or d not in design.days or c not in design.conditions or z not in zs:
            if strict:
                raise EvalError(f"run {r.run_id} ({r.run_dir}) is outside the test design: cell {key}, design "
                                f"{design.as_dict()}, zones {sorted(zs)}")
            continue
        att = by_cell[(w, d, z)][r.attempt]
        if c in att:
            raise EvalError(f"two runs for {key} attempt {r.attempt}: {att[c].run_dir} and {r.run_dir}")
        att[c] = r
    chosen: dict[tuple[int, int, str], int | None] = {}
    first: dict[tuple[int, int, str], int | None] = {}
    unresolved = []
    for w in design.worlds:
        for d in design.days:
            for z in sorted(zs):
                atts = by_cell.get((w, d, z), {})
                order = sorted(atts)
                clean = [a for a in order if not any(is_bad(r, invalid) for r in atts[a].values())]
                chosen[(w, d, z)] = clean[0] if clean else (order[-1] if order else None)
                first[(w, d, z)] = order[0] if order else None
                if order and not clean:                     # infrastructure-invalid and never cleanly re-run
                    unresolved.append(f"w{w}_d{d}_{z}")
    return Cells(dict(by_cell), chosen, first, unresolved, unplaceable, non_primary)
