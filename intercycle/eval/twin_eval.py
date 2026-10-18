"""EVAL runner for the press-shop twin (PLAN_TWIN §5 "EVAL", §4 S1b item 4; WEAKNESSES E-W1, E-W2, E-W4, E-W7,
E-W11, E-W12).

    python -m intercycle.eval.twin_eval --runs RUNS_ROOT --tasks twin/tasks.yaml [--split test] [--engine double]
        [--invalid results/sealed/invalid.json] [--h3-counts DIR] [--recall frames.json --r0 R0] --out OUT.json

Inputs are the closed per-mission files the plant writes (`runs/<id>/summary.json`, `log.parquet` or `log.jsonl`) and,
for gate 3, the per-tick miss counts that `python -m twin.eval_metrics` computes offline after the run (the only
truth reader of EVAL, in the truth allowlist's EVAL slot; this module opens no truth file).

Scoring, per mission (one zone, one condition, one (world, day)):
  * Y = checkpoints of the zone's mission_order (tasks.yaml) whose capture passed Inspect (`checkpoint_pass` rows),
    set to 0 if the mission had any robot-caused press stop (`press_trip`), hold (`press_hold`) or incident; fixed
    T = 720 s. Declined and unattempted checkpoints score 0. The log rows and the summary counts must agree (fail
    closed: a disagreement makes the mission an error).
  * errors: a non-infrastructure error (summary status "error", an unreadable / inconsistent summary or log, a
    missing run) is a failure: Y = 0 with the full T, and the remaining T (T - time of the first error) is added to
    the press time lost (H2). Errored or missing missions are never dropped.
  * infrastructure-invalid runs (`scripts/classify_invalid.py`'s committed list, or summary status "infra_invalid")
    are re-run, and a re-run repeats ALL conditions of the (world, day, zone): for each cell the first attempt whose
    runs are all valid is scored. Attempt k of a run is the `__a<k>` suffix of its run_id (none = attempt 0). A cell
    without a clean attempt is scored on its last attempt with the invalid runs as failures (flagged).
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
     H1: mean e < 0. A world without a scored unit makes gate 3 not evaluable (not rejected: fail closed);
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
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

from intercycle.eval.conformal_margin import UnitCounts, per_world_miss_share, stream_units
from twin.schemas.common import MISSION_T_S
from twin.schemas.runs import CheckpointOutcome, MissionMetric, PrimaryMetric, Summary, require_primary, world_values

CONTRACT = "twin.eval_report/1"
ALPHA = 0.05
NI_DELTA_S_PER_H = 30.0            # H2 margin, fixed in PLAN_TWIN §5, not amendable
ALPHA_TARGET = 0.05                # S7b's conformal alpha (user decision)
H3_SLACK = 0.01
MIN_GROUPS = 5                     # small-group rule threshold (test groups per camera)
CONDITIONS = ("no_map", "static", "learned", "reactive", "oracle")
REQUIRED = ("static", "learned", "reactive")
BOOT_B = 10000
BOOT_SEED = 20261018
MAX_SIGNFLIP_K = 20
OUTCOME_ORDER = ("tripped", "held", "passed", "failed", "declined", "unattempted")
_ATTEMPT = re.compile(r"__a(\d+)$")
_TIE = 1e-9


class EvalError(ValueError):
    """The EVAL inputs are inconsistent (fail closed)."""


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


@dataclass
class Run:
    run_dir: Path
    run_id: str
    summary: Summary | None
    rows: list[dict]
    problems: list[str] = field(default_factory=list)

    @property
    def attempt(self) -> int:
        m = _ATTEMPT.search(self.run_id)
        return int(m.group(1)) if m else 0

    @property
    def key(self) -> tuple[int, int, str, str] | None:
        s = self.summary
        return None if s is None else (s.world, s.day, s.zone, s.condition)


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
    problems: list[str] = []
    summary, rid = None, d.name
    try:
        raw = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rid = str(raw.get("run_id", rid))
        summary = Summary.model_validate(raw)
    except Exception as e:  # noqa: BLE001 - an unreadable summary is an error mission
        problems.append(f"summary.json unreadable or invalid: {e!r}"[:400])
    try:
        rows = read_log(d)
    except Exception as e:  # noqa: BLE001
        rows = []
        problems.append(f"log unreadable: {e!r}"[:400])
    return Run(d, rid, summary, rows, problems)


def find_runs(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("summary.json"))


def mission_orders(tasks_path: str | Path) -> dict[str, list[str]]:
    t = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8"))
    if not isinstance(t, Mapping) or t.get("contract") != "twin.tasks/1":
        raise EvalError(f"{tasks_path}: not a twin.tasks/1 file")
    return {str(z["zone"]): [str(c) for c in z["mission_order"]] for z in t["zones"]}


def load_invalid(path: str | Path | None) -> set[str]:
    """classify_invalid's committed list: a JSON list of run ids, or {"invalid": [...]} / {"runs": [...]}."""
    if path is None:
        return set()
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(body, Mapping):
        body = body.get("invalid", body.get("runs"))
    if not isinstance(body, list) or not all(isinstance(r, str) for r in body):
        raise EvalError(f"{path}: expected a list of run ids")
    return set(body)


# ------------------------------------------------------------------------------------------------ mission scoring


@dataclass
class Scored:
    metric: MissionMetric
    counts: dict[str, int]                 # trips, holds, incidents, contacts (from the log rows)
    error_reason: str | None
    attempt: int
    run_dir: str | None


def _outcomes(rows: list[dict], planned: list[str]) -> list[CheckpointOutcome]:
    kinds: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("checkpoint") is not None:
            kinds[str(r["checkpoint"])].add(str(r.get("kind")))
    out = []
    for cp in planned:
        k = kinds.get(cp, set())
        if "press_trip" in k:
            o = "tripped"
        elif "press_hold" in k:
            o = "held"
        elif "checkpoint_pass" in k:
            o = "passed"
        elif k & {"checkpoint_attempt", "checkpoint_fail"}:
            o = "failed"
        elif "checkpoint_decline" in k:
            o = "declined"
        else:
            o = "unattempted"
        out.append(CheckpointOutcome(checkpoint=cp, outcome=o))  # type: ignore[arg-type]
    return out


def score_run(run: Run, planned_by_zone: Mapping[str, list[str]], *, infra_invalid: bool = False,
              key: tuple[int, int, str, str] | None = None) -> Scored:
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
    if s is not None:
        require_primary(s)
        if s.status != "ok":
            reasons.append(f"status {s.status}: {s.status_reason}")
        if (s.robot_caused_press_stops, s.holds, s.incidents) != (n["press_trip"], n["press_hold"], n["incident"]):
            reasons.append(f"summary counts (trips, holds, incidents) = {(s.robot_caused_press_stops, s.holds, s.incidents)}"
                           f" != log rows {(n['press_trip'], n['press_hold'], n['incident'])}")
        if s.checkpoints_planned != len(planned):
            reasons.append(f"checkpoints_planned {s.checkpoints_planned} != tasks.yaml mission_order {len(planned)}")
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
                      "contacts": n["contact"]}, "; ".join(reasons) or None, run.attempt, str(run.run_dir))


def missing_mission(key: tuple[int, int, str, str], planned: list[str], why: str) -> Scored:
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
    reruns: dict[str, int]                  # condition -> cells scored on a re-run
    unresolved: list[str]                   # cells without a clean attempt
    unplaceable: list[str]                  # run dirs whose summary gives no design cell
    excluded_non_primary: list[str] = field(default_factory=list)   # reset-sweep / short dev runs (never pooled)


def select(runs: Iterable[Run], planned_by_zone: Mapping[str, list[str]], invalid: set[str], *, split: str,
           engine: str, conditions: Sequence[str], worlds: Sequence[int] | None = None,
           days: Sequence[int] | None = None) -> Selection:
    by_cell: dict[tuple[int, int, str], dict[int, dict[str, Run]]] = defaultdict(lambda: defaultdict(dict))
    unplaceable, non_primary = [], []
    for r in runs:
        s = r.summary
        if s is None:
            unplaceable.append(str(r.run_dir))
            continue
        if s.world_split != split or s.engine != engine:
            continue
        if s.variant != "primary" or s.reset_override_s is not None:
            non_primary.append(r.run_id)                    # descriptive runs never enter the primary metric
            continue
        att = by_cell[(s.world, s.day, s.zone)][r.attempt]
        if s.condition in att:
            raise EvalError(f"two runs for {(s.world, s.day, s.zone, s.condition)} attempt {r.attempt}: "
                            f"{att[s.condition].run_dir} and {r.run_dir}")
        att[s.condition] = r
    ws = sorted(set(worlds) if worlds is not None else {c[0] for c in by_cell})
    ds = sorted(set(days) if days is not None else {c[1] for c in by_cell})
    zs = sorted(planned_by_zone)
    if not ws or not ds:
        raise EvalError(f"no {engine} runs on {split} worlds")
    scored, first, unresolved = [], [], []
    reruns = {c: 0 for c in conditions}

    def bad(r: Run) -> bool:
        return r.run_id in invalid or (r.summary is not None and r.summary.status == "infra_invalid")

    for w, d, z in itertools.product(ws, ds, zs):
        atts = by_cell.get((w, d, z), {})
        order = sorted(atts)
        clean = [a for a in order if not any(bad(r) for r in atts[a].values())]
        chosen = clean[0] if clean else (order[-1] if order else None)
        if order and not clean:                         # infrastructure-invalid and never cleanly re-run
            unresolved.append(f"w{w}_d{d}_{z}")
        for c in conditions:
            key = (w, d, z, c)
            for a, sink in ((chosen, scored), (order[0] if order else None, first)):
                r = atts[a].get(c) if a is not None else None
                if r is None:
                    sink.append(missing_mission(key, planned_by_zone[z], "missing run"))
                else:
                    sink.append(score_run(r, planned_by_zone, infra_invalid=bad(r), key=key))
            if chosen is not None and order and chosen != order[0]:
                reruns[c] += 1
    return Selection(scored, first, reruns, unresolved, unplaceable, non_primary)


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


def h3_world_shares(count_files: Sequence[dict], margin_m: float) -> dict[int, float]:
    """Per-world unit-mean miss share at the frozen margin from twin.eval_metrics' per-tick counts (one dict per
    (world, day, zone) stream: t, slice_id, occ, miss[:, 0] at margin_m, valid, world, day, episode_s)."""
    parts = []
    for c in count_files:
        if not math.isclose(float(c["margin_m"]), margin_m, abs_tol=1e-9):
            raise EvalError(f"tick counts at margin {c['margin_m']} != frozen margin {margin_m}")
        miss = np.asarray(c["miss"], np.int64).reshape(-1, 1)
        parts.append(stream_units(np.asarray(c["t"], float), np.asarray(c["slice_id"], np.int64),
                                  np.asarray(c["occ"], np.int64), miss, np.asarray(c["valid"], bool),
                                  world=int(c["world"]), day=int(c["day"]), margins_m=(margin_m,),
                                  episode_s=float(c.get("episode_s", MISSION_T_S))))
    if not parts:
        raise EvalError("no H3 tick counts")
    return per_world_miss_share(UnitCounts.concat(parts), margin_m)


def load_h3_counts(path: str | Path) -> list[dict]:
    """twin.eval_metrics' output JSON ({"margin_m": m, "streams": [...]}) -> its stream count dicts."""
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


def evaluate(sel: Selection, *, engine: str, split: str, h3_shares: Mapping[int, float] | None = None,
             h4: dict | None = None) -> dict:
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
    if h3_shares is None:
        gates.append(_gate("H3 map unsafe-free share", None, False, gates[-1]["claimed"], status="no input"))
    else:
        sh = [h3_shares.get(w, math.nan) for w in ws]
        if any(not math.isfinite(v) for v in sh):
            gates.append(_gate("H3 map unsafe-free share", None, False, gates[-1]["claimed"], status="not evaluable",
                               shares=sh, threshold=thr))
        else:
            e3 = np.array(sh) - thr
            p3 = signflip_p(e3, "less")
            gates.append(_gate(f"H3 map mean miss share < alpha_target + {H3_SLACK} = {thr}", p3, p3 <= ALPHA,
                               gates[-1]["claimed"], shares=sh, threshold=thr))
    if h4 is None:
        gates.append(_gate("H4 blind test person recall >= r0", None, False, gates[-1]["claimed"], status="no input"))
    else:
        gates.append(_gate("H4 blind test person recall >= r0", None, bool(h4["reject"]), gates[-1]["claimed"],
                           **{k: v for k, v in h4.items() if k != "reject"}))

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
    return {"contract": CONTRACT, "engine": engine, "split": split, "alpha": ALPHA, "gatekeeping": gates,
            "reported_outside_gate": outside, "checkpoint_outcomes": table,
            "first_attempt_per_world": [w.model_dump() for w in world_values(first)],
            "reruns_per_condition": sel.reruns, "unresolved_cells": sel.unresolved,
            "unplaceable_runs": sel.unplaceable, "excluded_non_primary_runs": sel.excluded_non_primary,
            "errored_missions": [{"run_id": s.metric.run_id, "reason": s.error_reason} for s in sel.scored
                                 if s.metric.error],
            "primary_metric": pm.model_dump(mode="json")}


def run_eval(runs_root: str | Path, tasks: str | Path, *, split: str = "test", engine: str = "double",
             invalid: str | Path | None = None, h3_counts: str | Path | None = None,
             recall_frames: str | Path | None = None, r0: float | None = None,
             conditions: Sequence[str] | None = None, worlds: Sequence[int] | None = None,
             days: Sequence[int] | None = None) -> dict:
    planned = mission_orders(tasks)
    runs = [load_run(d) for d in find_runs(runs_root)]
    present = {r.summary.condition for r in runs if r.summary is not None}
    conds = list(conditions) if conditions else [c for c in CONDITIONS if c in present or c in REQUIRED]
    sel = select(runs, planned, load_invalid(invalid), split=split, engine=engine, conditions=conds, worlds=worlds,
                 days=days)
    shares = None
    if h3_counts is not None:
        counts = load_h3_counts(h3_counts)
        shares = h3_world_shares(counts, float(counts[0]["margin_m"])) if counts else None
    h4 = None
    if recall_frames is not None:
        if r0 is None:
            raise EvalError("--recall needs --r0 (set from the 20 blind dev frames only)")
        h4 = h4_recall(json.loads(Path(recall_frames).read_text(encoding="utf-8")), r0)
    return evaluate(sel, engine=engine, split=split, h3_shares=shares, h4=h4)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EVAL runner for the press-shop twin (PLAN_TWIN §5).")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--engine", default="double", choices=("double", "isaac"))
    ap.add_argument("--invalid", help="classify_invalid's committed list of infrastructure-invalid run ids")
    ap.add_argument("--h3-counts", help="python -m twin.eval_metrics output (twin.h3_tick_counts/1)")
    ap.add_argument("--recall", help="blind test person-recall frames JSON")
    ap.add_argument("--r0", type=float)
    ap.add_argument("--worlds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--days", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run_eval(a.runs, a.tasks, split=a.split, engine=a.engine, invalid=a.invalid, h3_counts=a.h3_counts,
                   recall_frames=a.recall, r0=a.r0, worlds=a.worlds, days=a.days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    for g in rep["gatekeeping"]:
        print(f"{g['hypothesis']}: p={g['p']} reject={g['reject']} claimed={g['claimed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
