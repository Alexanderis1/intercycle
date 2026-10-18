"""EVAL runner for the press-shop twin (PLAN_TWIN §5 "EVAL", §4 S1b item 4; WEAKNESSES E-W1, E-W2, E-W4, E-W7,
E-W11, E-W12).

    python -m intercycle.eval.twin_eval --runs RUNS_ROOT --tasks twin/tasks.yaml [--split test] [--engine double]
        --worlds W1,...,W16 [--days 9,10] [--invalid results/sealed/invalid.json]
        [--h3-counts H3.json --margin-file twin/m7_margin.json] [--recall frames.json --r0 R0] --out OUT.json

Inputs are the closed per-mission files the plant writes (`runs/<id>/summary.json`, `log.parquet` or `log.jsonl`) and,
for gate 3, the per-tick miss counts that `python -m twin.eval_metrics` computes offline after the run (the only
truth reader of EVAL, in the truth allowlist's EVAL slot; this module opens no truth file).

Design (fail closed, PLAN_TWIN §5 "Design"): on the test split the design is explicit, never taken from the runs that
happen to be present: the world list (--worlds, required), the scored days (default 9 and 10) and the conditions (the
CPU double: all five). For the double on the test split the sealed design is enforced: K_TEST = 16 worlds, days {9, 10}
and all five conditions (5 x 16 x 2 x 2 = 320 missions); anything else is refused. Every absent (world, day, zone,
condition) is scored as a missing failure, and a test run outside the design is refused. (`allow_partial_design`, an
API argument only, lets the synthetic tests use fewer worlds; the report then says `"fixed": false`.) On the dev split
worlds and days default to the runs present.

Scoring, per mission (one zone, one condition, one (world, day)):
  * Y = checkpoints of the zone's mission_order (tasks.yaml) whose capture passed Inspect (a `checkpoint_pass` row of
    an attempted checkpoint: the plant's summary rule, so a pass row without a `checkpoint_attempt` row never counts),
    set to 0 if the mission had any robot-caused press stop (`press_trip`), hold (`press_hold`) or incident; fixed
    T = 720 s. Declined and unattempted checkpoints score 0. The log rows and the summary counts must agree (fail
    closed: a disagreement makes the mission an error): trips, holds, incidents, checkpoints_planned, and attempted /
    passed / declined against the distinct mission_order checkpoints of the attempt / pass / decline rows (the
    plant's rule: passed counts attempted checkpoints only, declined only checkpoints never attempted). A run_id of the
    form w<W>_d<D>_<zone>_<condition>[__a<k>] must name the summary's own cell.
  * errors: a non-infrastructure error (summary status "error", an inconsistent summary or log, a crash after the
    robot's first command) is a failure: Y = 0 with the full T, and the remaining T (T - time of the first error) is
    added to the press time lost (H2). Errored missions are never dropped. On the test split a run whose log is missing
    or unreadable, a design run that is absent, or a run without a usable summary that classify_invalid lists neither
    invalid nor crashed is refused (EvalError): "logs are missing" is infrastructure-invalid (PLAN_TWIN §5 Rules), so
    the list must name it and the cell be re-run; it is never a free failure of its condition. (The dev split keeps
    descriptive missing-failure scoring.)
  * infrastructure-invalid runs (`scripts/classify_invalid.py`'s committed list, or summary status "infra_invalid")
    are re-run, and a re-run repeats ALL conditions of the (world, day, zone): for each cell the first clean attempt
    (all runs valid; a re-run holding every design condition) is scored; a partial re-run is refused on the test split
    (intercycle.eval.twin_design). Attempt k of a run is the `__a<k>` suffix of its run_id (none = attempt 0). A cell
    without a clean attempt is unresolved: it is scored on its last attempt with the invalid runs as failures for the
    report only, and while any cell is unresolved no gate is evaluated or claimed (status "not evaluable"). A crashed
    run has no summary.json: every directory holding a run file (RUN_MARKERS) is found, and a run without a usable
    summary is placed in its design cell by its invalid / crashed record ({"run_id", "world", "day", "zone",
    "condition"[, "split", "engine"]}) or by its run_id (w<W>_d<D>_<zone>_<condition>[__a<k>]); an id of the invalid
    list with no directory at all is placed the same way. A run of the `crashed` list (crashed after the robot's first
    command) is scored as an error (a failure).
  * per-checkpoint outcome (E-W12): tripped > held > passed > failed > declined > unattempted, from the rows that
    carry the checkpoint.

Statistics (fixed sequence, each one-sided at ALPHA = 0.05; E-W1 / E-W2):
  1. primary: per-world value V_w(c) = sum Y / sum T per hour pooled over both test days and both zones; d_w =
     V_w(learned) - V_w(reactive) (CRN pairs inside the world); exact one-sided sign-flip over the worlds (H1: > 0);
  2. H2 non-inferiority: L_w(c) = robot-caused press time lost (stops + holds incl. reset, plus the remaining T of
     errored missions) per mission-hour [s/h]; e_w = L_w(learned) - L_w(static) - DELTA (DELTA = 30 s/h); exact
     sign-flip, H1: mean e < 0;
  3. H3 map unsafe-free share: per world, the unweighted mean over its test-day units of the unit miss share at the
     frozen M-W7 margin (`intercycle.eval.conformal_margin.stream_units` + `per_world_miss_share`, the estimand the
     calibration used); e_w = share_w - (ALPHA_TARGET + 0.01), ALPHA_TARGET = 0.05 (user decision); sign-flip,
     H1: mean e < 0. The counts must be at the frozen margin of --margin-file (margin_m and cell_m), units use that
     file's unit.episode_s, and only streams of the evaluated split, the design worlds and the scored days are
     accepted (any other stream is refused). Each stream must name the run_id and attempt it came from, and they must
     be the learned run this evaluation scored for that (world, day, zone) (a clean one): counts built from another
     runs root, invalid list or attempt are refused. A world without a scored unit, or missing the stream of any of
     its (day, zone) cells, makes gate 3 not evaluable (not rejected: fail closed);
  4. H4 blind test person recall >= r0: with fewer than MIN_GROUPS test groups in any camera (the small-group rule)
     it passes only if the lowest per-group recall and the pooled recall are both >= r0, and the frame-level
     Clopper-Pearson interval is reported as "ignores clustering (only k groups)"; otherwise the one-sided 95 %
     group-bootstrap lower bound of the pooled recall must be >= r0.
  A hypothesis is claimed only if it and every earlier one rejected.
Reported outside the gate: learned - static on the primary with the world-clustered 95 % t interval over the world
differences (and the README flag when its point estimate is <= 0); robot safety outcomes per condition as Poisson rates
per mission-hour with exact (Garwood) 95 % intervals (E-W11: descriptive only); the share of the oracle gain
sum_w (L - R) / sum_w (O - R), reported only if sum_w (O - R) > 0 and the 2.5 % world-bootstrap quantile of that sum
is > 0, otherwise the two paired differences with t intervals; the per-checkpoint outcome table; first-attempt values
and the re-run count per condition.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from intercycle.eval.conformal_margin import UnitCounts, per_world_miss_share, stream_units
from intercycle.eval.twin_design import (CELL_KEYS, CONDITIONS, K_TEST, REQUIRED, RUN_MARKERS, TEST_DAYS, Cell,
                                         Design, EvalError, Run, attempt_of, find_runs, index_cells, is_bad, listed,
                                         load_classified, load_invalid, make_design, mission_orders, parse_run_id,
                                         placed, placeholder_runs, read_summary)
from intercycle.margin_inflate import Margin, load_margin
from twin.schemas.common import MISSION_T_S
from twin.schemas.runs import CheckpointOutcome, MissionMetric, PrimaryMetric, require_primary, world_values

__all__ = ["CELL_KEYS", "CONDITIONS", "K_TEST", "REQUIRED", "RUN_MARKERS", "TEST_DAYS", "Cell", "Design", "EvalError",
           "Run", "find_runs", "is_bad", "load_classified", "load_invalid", "make_design", "mission_orders",
           "parse_run_id", "placed", "placeholder_runs", "run_eval", "main"]

CONTRACT = "twin.eval_report/1"
ALPHA = 0.05
NI_DELTA_S_PER_H = 30.0            # H2 margin, fixed in PLAN_TWIN §5, not amendable
ALPHA_TARGET = 0.05                # S7b's conformal alpha (user decision)
H3_SLACK = 0.01
MIN_GROUPS = 5                     # small-group rule threshold (test groups per camera)
BOOT_B = 10000
BOOT_SEED = 20261018
MAX_SIGNFLIP_K = 20
H3_CONDITION = "learned"           # the condition whose robot holds the observer's map (= twin.eval_metrics')
OUTCOME_ORDER = ("tripped", "held", "passed", "failed", "declined", "unattempted")
_TIE = 1e-9


# ------------------------------------------------------------------------------------------------ statistics


def signflip_p(d: Sequence[float] | np.ndarray, alternative: str) -> float:
    """Exact one-sided sign-flip p-value of the mean of d over all 2^K sign vectors (K <= MAX_SIGNFLIP_K).

    alternative "greater": H1 mean > 0, p = P(sum eps*d >= sum d); "less": H1 mean < 0, p = P(sum eps*d <= sum d).
    Ties count against H1 (relative tolerance 1e-9)."""
    x = np.asarray(d, float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise EvalError("sign-flip needs a non-empty vector of finite per-world values")
    K = len(x)
    if K > MAX_SIGNFLIP_K:
        raise EvalError(f"exact sign-flip enumeration is limited to {MAX_SIGNFLIP_K} worlds, got {K}")
    if alternative not in ("greater", "less"):
        raise EvalError(f"alternative must be greater or less, got {alternative!r}")
    signs = 1 - 2 * ((np.arange(2 ** K)[:, None] >> np.arange(K)[None, :]) & 1)       # (2^K, K) of +-1
    null = signs @ x
    obs = float(x.sum())
    tol = _TIE * max(1.0, float(np.abs(x).sum()))
    hits = null >= obs - tol if alternative == "greater" else null <= obs + tol
    return float(hits.mean())


def t_interval(d: Sequence[float] | np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): Student t interval over the per-world values (one value per world cluster)."""
    x = np.asarray(d, float)
    m = float(x.mean())
    if len(x) < 2:
        return m, math.nan, math.nan
    h = float(stats.t.ppf(0.5 + level / 2, len(x) - 1)) * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h


def poisson_exact(k: int, exposure_h: float, level: float = 0.95) -> dict:
    """Rate per hour with the exact (Garwood) interval: chi2 quantiles of 2k and 2k + 2 degrees of freedom."""
    if exposure_h <= 0:
        raise EvalError("Poisson exposure must be > 0")
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.chi2.ppf(a / 2, 2 * k)) / 2
    hi = float(stats.chi2.ppf(1 - a / 2, 2 * k + 2)) / 2
    return {"count": int(k), "exposure_h": exposure_h, "rate_per_h": k / exposure_h, "lo": lo / exposure_h,
            "hi": hi / exposure_h, "level": level, "method": "exact Poisson (Garwood)"}


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def world_bootstrap(per_world: np.ndarray, fn, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """fn(resampled rows) for B resamples of the worlds (rows of per_world) with replacement."""
    rng = np.random.default_rng(seed)
    K = len(per_world)
    idx = rng.integers(0, K, size=(B, K))
    return np.array([fn(per_world[i]) for i in idx])


# ------------------------------------------------------------------------------------------------ runs


def read_log(run_dir: Path) -> list[dict]:
    """The plant's log rows: log.parquet (CPU double) or log.jsonl (Isaac python)."""
    p, j = run_dir / "log.parquet", run_dir / "log.jsonl"
    if p.exists():
        import pyarrow.parquet as pq

        return pq.read_table(p).to_pylist()
    if j.exists():
        return [json.loads(line) for line in j.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"{run_dir}: no log.parquet or log.jsonl")


def load_run(run_dir: str | Path) -> Run:
    """Read one closed mission (summary.json is written last by the plant). Problems are recorded, never raised: the
    mission is then scored as an error."""
    d = Path(run_dir)
    summary, rid, problems = read_summary(d)
    log_problem = None
    try:
        rows = read_log(d)
    except Exception as e:  # noqa: BLE001
        rows = []
        log_problem = f"log missing or unreadable: {e!r}"[:400]
        problems.append(log_problem)
    return Run(d, rid, summary, rows, problems, log_problem=log_problem)


def load_runs(runs_root: str | Path, invalid: Mapping[str, dict | None]) -> list[Run]:
    """Every run under runs_root (incl. crashed ones without summary.json) plus a placeholder per invalid-listed id
    without a directory."""
    runs = [load_run(d) for d in find_runs(runs_root)]
    return runs + placeholder_runs(invalid, runs)


# ------------------------------------------------------------------------------------------------ mission scoring


@dataclass
class Scored:
    metric: MissionMetric
    counts: dict[str, int]                 # trips, holds, incidents, contacts (from the log rows)
    error_reason: str | None
    attempt: int
    run_dir: str | None
    infra_invalid: bool = False            # scored although infrastructure-invalid (an unresolved cell)


def _kinds(rows: list[dict]) -> dict[str, set[str]]:
    kinds: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("checkpoint") is not None:
            kinds[str(r["checkpoint"])].add(str(r.get("kind")))
    return kinds


def _outcomes(rows: list[dict], planned: list[str]) -> list[CheckpointOutcome]:
    kinds = _kinds(rows)
    out = []
    for cp in planned:
        k = kinds.get(cp, set())
        if "press_trip" in k:
            o = "tripped"
        elif "press_hold" in k:
            o = "held"
        elif "checkpoint_pass" in k and "checkpoint_attempt" in k:      # the plant's rule: passes of attempts only
            o = "passed"
        elif k & {"checkpoint_attempt", "checkpoint_fail"}:
            o = "failed"
        elif "checkpoint_decline" in k:
            o = "declined"
        else:
            o = "unattempted"
        out.append(CheckpointOutcome(checkpoint=cp, outcome=o))  # type: ignore[arg-type]
    return out


def log_checkpoint_counts(rows: list[dict], planned: list[str]) -> tuple[int, int, int]:
    """(attempted, passed, declined) of the plant's summary rule, recomputed from the log rows over the mission_order:
    attempted = checkpoints with an attempt row, passed = attempted ones with a pass row, declined = checkpoints with a
    decline row that were never attempted."""
    kinds = _kinds(rows)
    att = {cp for cp in planned if "checkpoint_attempt" in kinds.get(cp, set())}
    pas = {cp for cp in att if "checkpoint_pass" in kinds.get(cp, set())}
    dec = {cp for cp in planned if "checkpoint_decline" in kinds.get(cp, set())} - att
    return len(att), len(pas), len(dec)


def score_run(run: Run, planned_by_zone: Mapping[str, list[str]], *, infra_invalid: bool = False,
              key: Cell | None = None, crashed: bool = False) -> Scored:
    """Score one mission (module doc). `key` = (world, day, zone, condition) for a run whose summary is unusable."""
    s = run.summary
    k = run.key or key
    if k is None:
        raise EvalError(f"{run.run_dir}: no usable summary and no design cell given")
    world, day, zone, cond = k
    planned = planned_by_zone.get(zone)
    if planned is None:
        raise EvalError(f"zone {zone} has no mission_order in tasks.yaml")
    rows = run.rows
    n = {kind: sum(1 for r in rows if r.get("kind") == kind)
         for kind in ("press_trip", "press_hold", "incident", "contact")}
    reasons = list(run.problems)
    if infra_invalid:
        reasons.append("infrastructure-invalid run without a clean re-run")
    if crashed:
        reasons.append("crashed after the robot's first command (classify_invalid crashed list)")
    if s is not None:
        require_primary(s)
        if s.status != "ok":
            reasons.append(f"status {s.status}: {s.status_reason}")
        if (s.robot_caused_press_stops, s.holds, s.incidents) != (n["press_trip"], n["press_hold"], n["incident"]):
            reasons.append(f"summary counts (trips, holds, incidents) = {(s.robot_caused_press_stops, s.holds, s.incidents)}"
                           f" != log rows {(n['press_trip'], n['press_hold'], n['incident'])}")
        if s.checkpoints_planned != len(planned):
            reasons.append(f"checkpoints_planned {s.checkpoints_planned} != tasks.yaml mission_order {len(planned)}")
        got = log_checkpoint_counts(rows, planned)
        if (s.attempted, s.passed, s.declined) != got:
            reasons.append(f"summary counts (attempted, passed, declined) = {(s.attempted, s.passed, s.declined)}"
                           f" != log rows {got}")
    outs = _outcomes(rows, planned)
    error = bool(reasons)
    lost = float(s.press_time_lost_s) if s is not None else 0.0
    if error:
        errs = [float(r["t"]) for r in rows if r.get("kind") == "error" and r.get("t") is not None]
        ends = [float(r["t"]) for r in rows if r.get("kind") == "mission_end" and r.get("t") is not None]
        t_err = min(errs) if errs else (min(ends) if ends and s is not None and s.status == "error" else 0.0)
        lost += max(0.0, MISSION_T_S - min(max(t_err, 0.0), MISSION_T_S))
    trip, hold, inc = n["press_trip"] > 0, n["press_hold"] > 0, n["incident"] > 0
    y = 0 if (trip or hold or inc or error) else sum(o.outcome == "passed" for o in outs)
    m = MissionMetric(run_id=run.run_id, world=world, day=day, zone=zone, condition=cond,  # type: ignore[arg-type]
                      T_s=MISSION_T_S, Y=y, robot_caused_press_stop=trip, hold=hold, incident=inc, error=error,
                      press_time_lost_s=round(lost, 6), checkpoints=outs)
    return Scored(m, {"trips": n["press_trip"], "holds": n["press_hold"], "incidents": n["incident"],
                      "contacts": n["contact"]}, "; ".join(reasons) or None, run.attempt, str(run.run_dir),
                  infra_invalid)


def missing_mission(key: Cell, planned: list[str], why: str) -> Scored:
    world, day, zone, cond = key
    m = MissionMetric(run_id=f"missing:w{world}_d{day}_{zone}_{cond}", world=world, day=day, zone=zone,
                      condition=cond, T_s=MISSION_T_S, Y=0, robot_caused_press_stop=False,  # type: ignore[arg-type]
                      hold=False, incident=False, error=True, press_time_lost_s=MISSION_T_S,
                      checkpoints=[CheckpointOutcome(checkpoint=c, outcome="unattempted") for c in planned])
    return Scored(m, {"trips": 0, "holds": 0, "incidents": 0, "contacts": 0}, why, -1, None)


# ------------------------------------------------------------------------------------------------ design and re-runs


@dataclass
class Selection:
    scored: list[Scored]                    # one per design cell x condition (the scored attempt)
    first: list[Scored]                     # attempt 0 (infra-invalid runs as failures), for the report
    reruns: dict[str, int]                  # condition -> cells whose run of that condition is a re-run's
    unresolved: list[str]                   # cells without a clean attempt
    unplaceable: list[str]                  # run dirs whose summary and id give no design cell
    excluded_non_primary: list[str] = field(default_factory=list)   # reset-sweep / short dev runs (never pooled)


def select(runs: Iterable[Run], planned_by_zone: Mapping[str, list[str]], invalid: Mapping[str, dict | None], *,
           split: str, engine: str, design: Design, crashed: Mapping[str, dict | None] | None = None) -> Selection:
    """Score every design cell x condition on its scored attempt (intercycle.eval.twin_design.index_cells) and on its
    first attempt; an absent run is a missing failure (dev split only: the test split refuses it, as it refuses a
    valid-looking run without a usable log)."""
    cr = dict(crashed or {})
    cells = index_cells(runs, planned_by_zone, invalid, split=split, engine=engine, design=design, crashed=cr)
    if split == "test":
        nolog = [f"{r.run_id} ({r.log_problem})" for atts in cells.by_cell.values() for att in atts.values()
                 for r in att.values() if r.log_problem is not None and not is_bad(r, invalid)]
        if nolog:
            raise EvalError("runs whose logs are missing or unreadable are not in classify_invalid's invalid list "
                            "(logs missing = infrastructure-invalid: list them and re-run their cells): "
                            + "; ".join(nolog[:10]))
    scored, first = [], []
    reruns = {c: 0 for c in design.conditions}
    for (w, d, z), chosen in cells.chosen.items():
        atts = cells.by_cell.get((w, d, z), {})
        a0 = cells.first[(w, d, z)]
        for c in design.conditions:
            key = (w, d, z, c)
            for a, sink in ((chosen, scored), (a0, first)):
                r = atts[a].get(c) if a is not None else None
                if r is None:
                    sink.append(missing_mission(key, planned_by_zone[z], "missing run"))
                else:
                    sink.append(score_run(r, planned_by_zone, infra_invalid=is_bad(r, invalid), key=key,
                                          crashed=r.summary is None and listed(r.run_id, r.run_dir.name, cr)))
            if chosen is not None and chosen != a0 and c in atts[chosen]:
                reruns[c] += 1
    return Selection(scored, first, reruns, cells.unresolved, cells.unplaceable, cells.excluded_non_primary)


# ------------------------------------------------------------------------------------------------ per-world values


def per_world(missions: Sequence[MissionMetric], what: str) -> dict[tuple[int, str], float]:
    """(world, condition) -> pooled value per mission-hour: "Y" = sum Y / sum T * 3600 (inspections per hour),
    "lost" = sum press_time_lost_s / sum T * 3600 (s per mission-hour)."""
    acc: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for m in missions:
        a = acc[(m.world, m.condition)]
        a[0] += m.Y if what == "Y" else m.press_time_lost_s
        a[1] += m.T_s
    return {k: v[0] / v[1] * 3600.0 for k, v in acc.items()}


def paired(values: Mapping[tuple[int, str], float], a: str, b: str) -> tuple[list[int], np.ndarray]:
    ws = sorted({w for w, _ in values})
    miss = [w for w in ws if (w, a) not in values or (w, b) not in values]
    if miss:
        raise EvalError(f"worlds {miss} lack {a} or {b}")
    return ws, np.array([values[(w, a)] - values[(w, b)] for w in ws])


# ------------------------------------------------------------------------------------------------ H3 and H4


def load_frozen_margin(path: str | Path) -> tuple[Margin, float]:
    """The frozen M-W7 margin (intercycle.margin_inflate.load_margin: frozen, status ok, the inflation rule) and its
    calibration unit length unit.episode_s (required: units must be cut as in the calibration)."""
    margin = load_margin(path)
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    ep = (body.get("unit") or {}).get("episode_s")
    if isinstance(ep, bool) or not isinstance(ep, (int, float)) or not math.isfinite(ep) or ep <= 0:
        raise EvalError(f"{path}: unit.episode_s must be a number > 0 (the calibration's unit), got {ep!r}")
    return margin, float(ep)


def h3_world_shares(count_files: Sequence[dict], margin_m: float, *, split: str, worlds: Sequence[int],
                    days: Sequence[int], zones: Sequence[str], episode_s: float,
                    cell_m: float | None = None) -> dict:
    """Per-world unit-mean miss share at the frozen margin from twin.eval_metrics' per-tick counts (one dict per
    (world, day, zone) stream: t, slice_id, occ, miss at margin_m, valid, world, day, zone, split, cell_m).

    Refuses (EvalError) a stream of another split, of a day that is not scored, of a world or zone outside the design,
    at another margin or cell size, or a repeated (world, day, zone). A world missing the stream of any of its
    (day, zone) cells gets share NaN (not evaluable). Returns {"shares": {world: share}, "missing_streams": [...]}."""
    ws, ds, zs = set(int(w) for w in worlds), set(int(d) for d in days), set(str(z) for z in zones)
    parts, seen = [], set()
    for c in count_files:
        name = c.get("stream", "?")
        if not math.isclose(float(c["margin_m"]), margin_m, abs_tol=1e-9):
            raise EvalError(f"H3 stream {name}: tick counts at margin {c['margin_m']} != frozen margin {margin_m}")
        if cell_m is not None and not math.isclose(float(c["cell_m"]), cell_m, abs_tol=1e-12):
            raise EvalError(f"H3 stream {name}: grid cell {c['cell_m']} != the frozen margin's cell {cell_m}")
        if str(c.get("split")) != split:
            raise EvalError(f"H3 stream {name}: split {c.get('split')!r} is not the evaluated split {split!r}")
        w, d, z = int(c["world"]), int(c["day"]), str(c["zone"])
        if d not in ds:
            raise EvalError(f"H3 stream {name}: day {d} is not a scored day {sorted(ds)} (gate 3 is on test days)")
        if w not in ws:
            raise EvalError(f"H3 stream {name}: world {w} is not a design world {sorted(ws)}")
        if z not in zs:
            raise EvalError(f"H3 stream {name}: zone {z} is not a tasks.yaml zone {sorted(zs)}")
        if (w, d, z) in seen:
            raise EvalError(f"two H3 streams for (world, day, zone) = {(w, d, z)}")
        seen.add((w, d, z))
        miss = np.asarray(c["miss"], np.int64).reshape(-1, 1)
        parts.append(stream_units(np.asarray(c["t"], float), np.asarray(c["slice_id"], np.int64),
                                  np.asarray(c["occ"], np.int64), miss, np.asarray(c["valid"], bool),
                                  world=w, day=d, margins_m=(margin_m,), episode_s=episode_s))
    missing = [(w, d, z) for w in sorted(ws) for d in sorted(ds) for z in sorted(zs) if (w, d, z) not in seen]
    got = per_world_miss_share(UnitCounts.concat(parts), margin_m) if parts else {}
    incomplete = {w for w, _, _ in missing}
    shares = {w: (math.nan if w in incomplete else float(got.get(w, math.nan))) for w in sorted(ws)}
    return {"shares": shares, "missing_streams": [f"w{w}_d{d}_{z}" for w, d, z in missing], "n_streams": len(seen),
            "margin_m": margin_m, "episode_s": episode_s}


def load_h3_counts(path: str | Path) -> list[dict]:
    """twin.eval_metrics' output JSON ({"margin_m": m, "cell_m": c, "streams": [...]}) -> its stream count dicts."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("contract") != "twin.h3_tick_counts/1":
        raise EvalError(f"{path}: not a twin.h3_tick_counts/1 file")
    return [dict(s, margin_m=body["margin_m"]) for s in body["streams"]]


def h4_recall(frames: Sequence[Mapping[str, Any]], r0: float, *, min_groups: int = MIN_GROUPS) -> dict:
    """Blind test person recall vs r0 (module doc). frames: {"camera", "group", "tp", "n_gt"} per labelled frame."""
    tp: dict[str, int] = defaultdict(int)
    gt: dict[str, int] = defaultdict(int)
    cams: dict[Any, set[str]] = defaultdict(set)
    for f in frames:
        g = f"{f['camera']}:{f['group']}"
        if not 0 <= int(f["tp"]) <= int(f["n_gt"]):
            raise EvalError(f"frame of group {g}: tp must lie in 0..n_gt")
        tp[g] += int(f["tp"])
        gt[g] += int(f["n_gt"])
        cams[f["camera"]].add(g)
    groups = sorted(g for g in gt if gt[g] > 0)
    if not groups:
        raise EvalError("no labelled person in the test frames")
    K, N = sum(tp[g] for g in groups), sum(gt[g] for g in groups)
    pooled = K / N
    per_group = {g: {"recall": tp[g] / gt[g], "n_gt": gt[g]} for g in groups}
    n_groups = {str(c): len([g for g in gs if gt[g] > 0]) for c, gs in cams.items()}
    small = any(v < min_groups for v in n_groups.values())
    out: dict[str, Any] = {"r0": r0, "pooled_recall": pooled, "n_gt": N, "per_group": per_group,
                           "groups_per_camera": n_groups, "small_group_rule": small}
    if small:
        lo, hi = clopper_pearson(K, N)
        k = len(groups)
        out.update(rule="lowest per-group recall and pooled recall both >= r0",
                   lowest_group_recall=min(v["recall"] for v in per_group.values()),
                   clopper_pearson_95=[lo, hi], ci_label=f"ignores clustering (only {k} groups)")
        out["reject"] = bool(out["lowest_group_recall"] >= r0 and pooled >= r0)
    else:
        arr = np.array([[tp[g], gt[g]] for g in groups], float)
        boot = world_bootstrap(arr, lambda a: a[:, 0].sum() / a[:, 1].sum())
        lo = float(np.quantile(boot, ALPHA))
        out.update(rule="one-sided 95 % group-bootstrap lower bound >= r0", bootstrap_lower_95=lo)
        out["reject"] = bool(lo >= r0)
    return out


# ------------------------------------------------------------------------------------------------ report


def _gate(name: str, p: float | None, reject: bool, prev: bool, **kw) -> dict:
    return {"hypothesis": name, "p": p, "alpha": ALPHA, "reject": reject, "tested_in_sequence": prev,
            "claimed": bool(prev and reject), **kw}


def _blocked(g: dict, unresolved: Sequence[str]) -> dict:
    """A gate while cells are unresolved (infrastructure-invalid, never cleanly re-run): not evaluable, no p, never
    rejected or claimed; the per-world values stay as a descriptive record of the failure scoring."""
    out = {k: v for k, v in g.items() if k not in ("p", "reject", "claimed", "status")}
    return dict(out, p=None, reject=False, claimed=False, status="not evaluable: unresolved cells",
                unresolved_cells=list(unresolved))


def evaluate(sel: Selection, *, engine: str, split: str, h3: Mapping[str, Any] | None = None,
             h4: dict | None = None, design: Design | None = None) -> dict:
    missions = [s.metric for s in sel.scored]
    pm = PrimaryMetric(contract="twin.primary_metric/1", engine=engine, split=split,  # type: ignore[arg-type]
                       missions=missions, per_world=world_values(missions))
    V, L = per_world(missions, "Y"), per_world(missions, "lost")
    ws, d1 = paired(V, "learned", "reactive")
    p1 = signflip_p(d1, "greater")
    gates = [_gate("H1 primary: learned - best reactive, inspections per mission-hour", p1, p1 <= ALPHA, True,
                   worlds=ws, d_w=d1.tolist(), mean=float(d1.mean()))]
    _, d2 = paired(L, "learned", "static")
    e2 = d2 - NI_DELTA_S_PER_H
    p2 = signflip_p(e2, "less")
    gates.append(_gate(f"H2 non-inferiority: press time lost per mission-hour, learned - static <= {NI_DELTA_S_PER_H}"
                       " s/h", p2, p2 <= ALPHA, gates[-1]["claimed"], d_w=d2.tolist(), delta=NI_DELTA_S_PER_H))
    thr = ALPHA_TARGET + H3_SLACK
    if h3 is None:
        gates.append(_gate("H3 map unsafe-free share", None, False, gates[-1]["claimed"], status="no input"))
    else:
        sh = [float(h3["shares"].get(w, math.nan)) for w in ws]
        if any(not math.isfinite(v) for v in sh):
            gates.append(_gate("H3 map unsafe-free share", None, False, gates[-1]["claimed"], status="not evaluable",
                               shares=sh, threshold=thr, missing_streams=list(h3.get("missing_streams", []))))
        else:
            e3 = np.array(sh) - thr
            p3 = signflip_p(e3, "less")
            gates.append(_gate(f"H3 map mean miss share < alpha_target + {H3_SLACK} = {thr}", p3, p3 <= ALPHA,
                               gates[-1]["claimed"], shares=sh, threshold=thr,
                               margin_m=h3.get("margin_m"), episode_s=h3.get("episode_s")))
    if h4 is None:
        gates.append(_gate("H4 blind test person recall >= r0", None, False, gates[-1]["claimed"], status="no input"))
    else:
        gates.append(_gate("H4 blind test person recall >= r0", None, bool(h4["reject"]), gates[-1]["claimed"],
                           **{k: v for k, v in h4.items() if k != "reject"}))
    if sel.unresolved:                                   # fail closed: an unresolved cell blocks every claim
        gates = [_blocked(g, sel.unresolved) for g in gates]

    _, ds = paired(V, "learned", "static")
    m, lo, hi = t_interval(ds)
    outside: dict[str, Any] = {"learned_minus_static": {
        "mean_per_h": m, "ci95_world_clustered": [lo, hi], "d_w": ds.tolist(),
        "static_matched_or_beat_learned": bool(m <= 0)}}
    conds = sorted({mm.condition for mm in missions})
    expo = {c: sum(mm.T_s for mm in missions if mm.condition == c) / 3600.0 for c in conds}
    outside["safety_poisson"] = {c: {k: poisson_exact(sum(s.counts[k] for s in sel.scored if s.metric.condition == c),
                                                      expo[c]) for k in ("trips", "holds", "incidents", "contacts")}
                                 for c in conds}
    if "oracle" in conds:
        _, lr = paired(V, "learned", "reactive")
        _, orr = paired(V, "oracle", "reactive")
        den = float(orr.sum())
        boot = world_bootstrap(np.stack([lr, orr], 1), lambda a: (a[:, 0].sum(), a[:, 1].sum()))
        den_lo = float(np.quantile(boot[:, 1], 0.025))
        if den > 0 and den_lo > 0:
            ratio = boot[:, 0] / boot[:, 1]
            outside["oracle_share"] = {"reported": True, "share": float(lr.sum()) / den,
                                       "ci95_world_bootstrap": [float(np.quantile(ratio, 0.025)),
                                                                float(np.quantile(ratio, 0.975))],
                                       "sum_O_minus_R": den, "sum_O_minus_R_boot_lo": den_lo}
        else:
            outside["oracle_share"] = {"reported": False, "sum_O_minus_R": den, "sum_O_minus_R_boot_lo": den_lo,
                                       "learned_minus_reactive": dict(zip(("mean", "lo", "hi"), t_interval(lr))),
                                       "oracle_minus_reactive": dict(zip(("mean", "lo", "hi"), t_interval(orr)))}
    table: dict[str, dict[str, dict[str, int]]] = {}
    for mm in missions:
        for co in mm.checkpoints:
            row = table.setdefault(mm.condition, {}).setdefault(co.checkpoint, {o: 0 for o in OUTCOME_ORDER})
            row[co.outcome] += 1
    first = [s.metric for s in sel.first]
    return {"contract": CONTRACT, "engine": engine, "split": split, "alpha": ALPHA,
            "design": None if design is None else design.as_dict(), "gatekeeping": gates,
            "reported_outside_gate": outside, "checkpoint_outcomes": table,
            "first_attempt_per_world": [w.model_dump() for w in world_values(first)],
            "reruns_per_condition": sel.reruns, "unresolved_cells": sel.unresolved,
            "unplaceable_runs": sel.unplaceable, "excluded_non_primary_runs": sel.excluded_non_primary,
            "errored_missions": [{"run_id": s.metric.run_id, "reason": s.error_reason} for s in sel.scored
                                 if s.metric.error],
            "primary_metric": pm.model_dump(mode="json")}


def check_h3_provenance(count_files: Sequence[Mapping[str, Any]], sel: Selection) -> None:
    """Every H3 stream must name (run_id, attempt) = the clean learned run Selection scored for its (world, day,
    zone); refuse any other (counts from another runs root, invalid list or attempt)."""
    chosen = {(s.metric.world, s.metric.day, s.metric.zone): s for s in sel.scored
              if s.metric.condition == H3_CONDITION}
    for c in count_files:
        name = c.get("stream", "?")
        key = (int(c["world"]), int(c["day"]), str(c["zone"]))
        rid, att = c.get("run_id"), c.get("attempt")
        if not isinstance(rid, str) or isinstance(att, bool) or not isinstance(att, int):
            raise EvalError(f"H3 stream {name}: no run_id / attempt (the counts must name the run they came from)")
        if att != attempt_of(rid):
            raise EvalError(f"H3 stream {name}: attempt {att} is not the attempt of run_id {rid}")
        s = chosen.get(key)
        if s is None:
            raise EvalError(f"H3 stream {name}: no scored {H3_CONDITION} mission for {key}")
        if s.infra_invalid or s.run_dir is None:
            raise EvalError(f"H3 stream {name}: the scored {H3_CONDITION} mission of {key} ({s.metric.run_id}) is "
                            f"not a clean run")
        if (rid, att) != (s.metric.run_id, s.attempt):
            raise EvalError(f"H3 stream {name}: from run {rid} attempt {att}, but EVAL scored {s.metric.run_id} "
                            f"attempt {s.attempt} for {key}")


def run_eval(runs_root: str | Path, tasks: str | Path, *, split: str = "test", engine: str = "double",
             invalid: str | Path | None = None, h3_counts: str | Path | None = None,
             margin_file: str | Path | None = None, recall_frames: str | Path | None = None, r0: float | None = None,
             conditions: Sequence[str] | None = None, worlds: Sequence[int] | None = None,
             days: Sequence[int] | None = None, allow_partial_design: bool = False) -> dict:
    planned = mission_orders(tasks)
    inv, crashed = load_classified(invalid)
    runs = load_runs(runs_root, inv)
    design = make_design(runs, {**crashed, **inv}, split=split, engine=engine, worlds=worlds, days=days,
                         conditions=conditions, allow_partial_design=allow_partial_design)
    sel = select(runs, planned, inv, split=split, engine=engine, design=design, crashed=crashed)
    h3 = None
    if h3_counts is not None:
        if margin_file is None:
            raise EvalError("--h3-counts needs --margin-file (the frozen M-W7 margin the counts must be at)")
        margin, episode_s = load_frozen_margin(margin_file)
        counts = load_h3_counts(h3_counts)
        h3 = h3_world_shares(counts, margin.margin_m, split=split, worlds=design.worlds,
                             days=design.days, zones=sorted(planned), episode_s=episode_s, cell_m=margin.cell_m)
        check_h3_provenance(counts, sel)
    h4 = None
    if recall_frames is not None:
        if r0 is None:
            raise EvalError("--recall needs --r0 (set from the 20 blind dev frames only)")
        h4 = h4_recall(json.loads(Path(recall_frames).read_text(encoding="utf-8")), r0)
    return evaluate(sel, engine=engine, split=split, h3=h3, h4=h4, design=design)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EVAL runner for the press-shop twin (PLAN_TWIN §5).")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--engine", default="double", choices=("double", "isaac"))
    ap.add_argument("--invalid", help="classify_invalid's committed list ({\"invalid\": [...], \"crashed\": [...]})")
    ap.add_argument("--h3-counts", help="python -m twin.eval_metrics output (twin.h3_tick_counts/1)")
    ap.add_argument("--margin-file", help="the frozen M-W7 margin JSON (twin/m7_margin.json); required with --h3-counts")
    ap.add_argument("--recall", help="blind test person-recall frames JSON")
    ap.add_argument("--r0", type=float)
    ap.add_argument("--worlds", type=_ints, help="the design's world indices (required on the test split)")
    ap.add_argument("--days", type=_ints, help="scored days (default 9,10 on the test split)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run_eval(a.runs, a.tasks, split=a.split, engine=a.engine, invalid=a.invalid, h3_counts=a.h3_counts,
                   margin_file=a.margin_file, recall_frames=a.recall, r0=a.r0, worlds=a.worlds, days=a.days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    for g in rep["gatekeeping"]:
        print(f"{g['hypothesis']}: p={g['p']} reject={g['reject']} claimed={g['claimed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
