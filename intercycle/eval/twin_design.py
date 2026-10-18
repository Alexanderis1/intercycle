"""The EVAL design and the re-run rule of the press-shop twin (PLAN_TWIN §5 "Design" and "Rules"), shared by the EVAL
runner (intercycle.eval.twin_eval) and its truth slot (twin.eval_metrics), so both score the same attempt per cell.

Isaac-safe imports only (stdlib, yaml, pydantic via twin.schemas): twin.eval_metrics is a twin module and the Isaac
dependency scan follows it here. Nothing here reads a log or opens truth; a run is known by its summary.json, its
directory name and classify_invalid's committed list.

  * classify_invalid's committed list (one JSON file, PLAN_TWIN §5 Rules): `invalid` = the infrastructure-invalid runs
    (re-run, never scored as failures); `crashed` = runs that crashed AFTER the robot's first command (no summary.json,
    a closed log): valid failures, finished, so the truth slot may open truth with them present. Entries are run ids or
    records {"run_id", "world", "day", "zone", "condition"[, "split", "engine"]}; both are matched by run_id or by
    directory name (`listed`). A run is never in both.

  * Design: on the test split the design is explicit (the world list is required; days default to TEST_DAYS; the CPU
    double has the cut design's SEALED_CONDITIONS = static, learned, reactive, USER DECISION Sun ~07:00: no oracle, no
    no-map) and, for the CPU double, the sealed design K_TEST x TEST_DAYS x SEALED_CONDITIONS is enforced (any other
    condition set, e.g. one with oracle or no_map, is refused, and a test run of such a condition is outside the design)
    unless allow_partial_design (an API argument for synthetic tests only; `Design.fixed` is then False). A test run
    outside the design is refused. On the dev split worlds and days default to the runs present.
  * Runs: every directory under the runs root holding a run file (RUN_MARKERS), with or without summary.json, plus a
    placeholder per invalid-listed id with no directory. A run without a usable summary is placed in its design cell
    by its invalid-list record ({"run_id", "world", "day", "zone", "condition"[, "split", "engine"]}) or by its run_id
    (w<W>_d<D>_<zone>_<condition>[__a<k>]).
  * Re-runs: attempt k = the `__a<k>` suffix of the run_id (none = 0). A re-run repeats ALL conditions: an attempt
    k > 0 that lacks a design condition is a partial re-run and never clean. Per (world, day, zone) the first clean
    attempt (every run valid: none listed invalid, none with status infra_invalid; complete if k > 0) is scored; without
    one, the last attempt is scored descriptively and the cell is unresolved (twin_eval then claims no gate).
  * Test split (fail closed, PLAN_TWIN §5 Rules: "logs are missing" is infrastructure-invalid, so a run the list does
    not name must be there): every attempt of every design (world, day, zone) must hold every design condition (an
    invalid-listed id without a directory counts), a cell with no run at all is refused, a partial re-run is refused,
    and a run without a usable summary must be listed invalid or crashed. Any of these raises EvalError.
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
# USER DECISION (Sun ~07:00 CEST, split freeze, CUT design): the sealed CPU double runs exactly these 3 conditions
# (no oracle, no no-map): 3 x K_TEST x TEST_DAYS x 2 zones = 192 missions. H1-H3 need learned, reactive and static.
SEALED_CONDITIONS = REQUIRED
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


def attempt_of(rid: str) -> int:
    """Attempt k of a run_id: its `__a<k>` suffix (none = attempt 0)."""
    m = _ATTEMPT.search(rid)
    return int(m.group(1)) if m else 0


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
    log_problem: str | None = None          # why the log is missing / unreadable (twin_eval.load_run; None here)

    @property
    def attempt(self) -> int:
        return attempt_of(self.run_id)

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


def _records(body: Any, path: str | Path, what: str) -> dict[str, dict | None]:
    if not isinstance(body, list):
        raise EvalError(f"{path}: expected a list of {what} run ids or records")
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


def load_classified(path: str | Path | None) -> tuple[dict[str, dict | None], dict[str, dict | None]]:
    """(invalid, crashed) of classify_invalid's committed list (module doc): a JSON list (invalid ids only), or
    {"invalid": [...] (or "runs"), "crashed": [...]}. Each maps run id -> its record (None for a bare id)."""
    if path is None:
        return {}, {}
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    crashed: dict[str, dict | None] = {}
    if isinstance(body, Mapping):
        crashed = _records(body.get("crashed", []), path, "crashed")
        body = body.get("invalid", body.get("runs", []))
    invalid = _records(body, path, "invalid")
    both = sorted(set(invalid) & set(crashed))
    if both:
        raise EvalError(f"{path}: runs listed both invalid and crashed: {both}")
    return invalid, crashed


def load_invalid(path: str | Path | None) -> dict[str, dict | None]:
    """The infrastructure-invalid part of classify_invalid's committed list (`load_classified`)."""
    return load_classified(path)[0]


def load_crashed(path: str | Path | None) -> dict[str, dict | None]:
    """The crashed-after-the-first-command part of classify_invalid's committed list (`load_classified`)."""
    return load_classified(path)[1]


def listed(run_id: str, dir_name: str, ids: Mapping[str, Any] | Iterable[str]) -> bool:
    """A run is in a classify_invalid list when its run_id or its directory name is."""
    return run_id in ids or dir_name in ids


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
    """(cell, split, engine) of a run without a usable summary: from its record (invalid or crashed list), else from
    its run_id; None if neither places it. split / engine are None unless the record names them."""
    rec = invalid.get(r.run_id) or invalid.get(r.run_dir.name)
    if rec is not None and all(k in rec for k in CELL_KEYS):
        cell = (int(rec["world"]), int(rec["day"]), str(rec["zone"]), str(rec["condition"]))
        return cell, rec.get("split"), rec.get("engine")
    named = parse_run_id(r.run_id) or parse_run_id(r.run_dir.name)
    return None if named is None else (named[0], None, None)


def is_bad(r: Run, invalid: Mapping[str, dict | None]) -> bool:
    """Infrastructure-invalid: listed by classify_invalid (by run_id or directory name), or status infra_invalid."""
    return (listed(r.run_id, r.run_dir.name, invalid)
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
            SEALED_CONDITIONS if engine == "double" else REQUIRED)
        sealed = engine == "double"
        if sealed and not allow_partial_design:
            if len(ws) != K_TEST:
                raise EvalError(f"the sealed design has K = {K_TEST} test worlds, got {len(ws)}: {list(ws)}")
            if ds != TEST_DAYS:
                raise EvalError(f"the sealed design scores days {TEST_DAYS}, got {list(ds)}")
            if cs != SEALED_CONDITIONS:
                raise EvalError(f"the sealed (cut) design has exactly the conditions {SEALED_CONDITIONS}, got {list(cs)}")
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
    unresolved: list[str]                                   # cells without a clean attempt (no gate claimed)
    unplaceable: list[str]                                  # run dirs whose summary and id give no design cell
    excluded_non_primary: list[str]                         # reset-sweep / short dev runs (never pooled)

    def runs_of(self, cell: tuple[int, int, str]) -> dict[str, Run]:
        """condition -> run of the scored attempt of a (world, day, zone)."""
        a = self.chosen.get(cell)
        return {} if a is None else dict(self.by_cell[cell][a])


def index_cells(runs: Iterable[Run], zones: Iterable[str], invalid: Mapping[str, dict | None], *, split: str,
                engine: str, design: Design, crashed: Mapping[str, dict | None] | None = None) -> Cells:
    """Place every run of the evaluated split and engine in its design cell and pick the scored attempt per cell
    (module doc: re-runs; the test split's completeness and classification checks raise EvalError)."""
    zs = set(zones)
    cr = dict(crashed or {})
    records = {**cr, **invalid}
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
            p = placed(r, records)
            if p is None:
                unplaceable.append(str(r.run_dir))
                continue
            key, sp, en = p
            if (sp is not None and sp != split) or (en is not None and en != engine):
                continue
            r.cell = key
            if strict and not is_bad(r, invalid) and not listed(r.run_id, r.run_dir.name, cr):
                raise EvalError(f"run {r.run_id} ({r.run_dir}) has no usable summary and is in neither the invalid nor "
                                f"the crashed list of classify_invalid ({'; '.join(r.problems)[:200]}): a run still "
                                f"running or unclassified is never scored")
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
    need = set(design.conditions)
    for w in design.worlds:
        for d in design.days:
            for z in sorted(zs):
                atts = by_cell.get((w, d, z), {})
                order = sorted(atts)
                where = f"w{w}_d{d}_{z}"
                if strict and not order:
                    raise EvalError(f"design cell {where} has no run at all and none is listed invalid (logs missing "
                                    f"= infrastructure-invalid: classify_invalid must list it and the cell re-run)")
                for a in order:
                    lack = sorted(need - set(atts[a]))
                    if strict and lack:
                        what = ("design runs absent" if a == order[0] else
                                "partial re-run (a re-run repeats ALL conditions of the cell)")
                        raise EvalError(f"{where} attempt {a}: {what}: no run for {lack}, and none is listed invalid")
                clean = [a for a in order if not any(is_bad(r, invalid) for r in atts[a].values())
                         and (a == order[0] or not need - set(atts[a]))]
                chosen[(w, d, z)] = clean[0] if clean else (order[-1] if order else None)
                first[(w, d, z)] = order[0] if order else None
                if order and not clean:                     # infrastructure-invalid and never cleanly re-run
                    unresolved.append(f"w{w}_d{d}_{z}")
    return Cells(dict(by_cell), chosen, first, unresolved, unplaceable, non_primary)
