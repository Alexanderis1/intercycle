"""EVAL metrics: the truth reads of EVAL, after the run only (PLAN_TWIN §2 truth guards: the EVAL metrics module slot of
twin.truth_guard.PLAN_READERS; §5 gate 3).

    python -m twin.eval_metrics --runs RUNS_ROOT --margin-file twin/m7_margin.json --tasks twin/tasks.yaml \
        --site twin/site.yaml --worlds W1,...,W16 [--days 9,10] [--invalid LIST.json] \
        --out results/sealed/h3_tick_counts.json                       # streams built from the run dirs
    python -m twin.eval_metrics --runs RUNS_ROOT --margin-file ... --out ... STREAM.npz [...]   # prebuilt streams

Gate 3 (map unsafe-free share) compares the observer map the robot held with the realised occupancy, so it needs
truth. This module:
  1. refuses to open any truth file until every run under RUNS_ROOT is finished: summary.json present and valid JSON
     (the plant writes it last) and the log closed (log.parquet with its PAR1 footer, or log.jsonl). The two records
     of classify_invalid's committed list (intercycle.eval.twin_design.load_classified; matched by run_id or directory
     name, as twin_design matches) count as finished too: an `invalid` run (infrastructure-invalid, re-run, never
     scored) in any state, and a `crashed` run (crashed after the robot's first command: no summary.json) whose log is
     closed. Any other run refuses;
  2. builds one stream per (world, day, zone) of the design from the run directories (`h3_from_runs`): the scored
     attempt of the cell's `learned` run (the condition whose robot holds the observer's map; the attempt rule is
     intercycle.eval.twin_design's, shared with twin_eval), its `realised_state.jsonl` opened through `twin.truth_guard.open_truth`
     (allowlisted caller, every opening appended to results/truth_access.log), and its HELD_MAP record (truth-free,
     written by the observer side during the run). A cell without a clean learned run or without these files gets
     no stream: twin_eval then marks the world not evaluable (fail closed). Prebuilt stream .npz files (the file name
     carries the realised_state token) are accepted too;
  3. writes per-tick counts at the frozen M-W7 margin: occ = truly occupied cells, miss = those the inflated
     prediction does not cover (`intercycle.margin_inflate.inflate`, the rule the robot applies and the calibration
     cross-checks), valid = a prediction existed at t - age_s, summed over the stream's grids, with the run_id and
     attempt of the run each stream came from (twin_eval refuses a stream that is not the learned run it scored). Only
     streams of the evaluated split and the scored days are accepted (any other is refused). No grid leaves this
     module.
intercycle.eval.twin_eval turns the counts into units and the per-world unit-mean miss share with
intercycle.eval.conformal_margin (stream_units, per_world_miss_share): the estimand the margin was calibrated on.

HELD_MAP (`runs/<id>/held_map.npz`, written by the observer side, no truth): what the robot held, per press grid
  press_ids (M,) str; res () float = the L1 grid cell (= the margin's cell_m); x0, z0 (M,) float grid origins (cell
  (i, j) has its centre at (x0 + (i + 0.5) res, z0 + (j + 0.5) res)); nx, nz (M,) int; t_pred (Tp,) observation time
  of the data behind each held prediction; pred_<k> (Tp, nx_k, nz_k) bool, press k's prediction (occupied OR UNKNOWN,
  un-inflated: ObserverLive.pred_l1 / calibration_pred); t_use (T,) the robot-use ticks; age_s (T,) the measured
  fast-layer age at each tick; slice_id (T,) the map time slice of each tick. All times are plant day seconds (the
  realised stream's and Observation.t_cap's clock); every t_use must be a tick of the run's realised stream.
TRUTH GRID (`truth_grid`, the rule S7b's dev recorder must share): press k's grid cells whose centre lies inside the
  ram plate square (the footprint's vertex mean +- twin.plant.RAM_HALF_M, the plant's press-ram mover) are truly
  occupied at a tick when the press is running, its ram is below top dead centre (ram > 0) or a stroke started in
  the tick; only the presses of the mission's zone are scored. People and forklifts are not L1 truth (they are the
  traffic layers' and the lidar's).

Isaac-safe imports only (numpy, stdlib, repo code); never imported by the robot side, observer_live or the broadcaster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from intercycle.eval import twin_design as TD
from intercycle.margin_inflate import Margin, inflate, load_margin
from twin.plant import RAM_HALF_M
from twin.schemas import load
from twin.schemas.common import TICK_HZ
from twin.schemas.runs import RealisedTick, parse_realised
from twin.schemas.site import Site
from twin.truth_guard import open_truth

CONTRACT = "twin.h3_tick_counts/1"
TOKEN = "realised_state"
REALISED = "realised_state.jsonl"
HELD_MAP = "held_map.npz"
STREAM_KEYS = ("t", "t_pred", "age_s", "slice_id", "world", "day", "zone", "cell_m", "split", "run_id")
HELD_KEYS = ("press_ids", "res", "x0", "z0", "nx", "nz", "t_pred", "t_use", "age_s", "slice_id")
RUNNING = "running"
H3_CONDITION = "learned"
_EPS = 1e-9


class NotClosed(RuntimeError):
    """A run is not closed yet: truth may not be read."""


NO_SUMMARY = "no summary.json"


def log_closed(run_dir: str | os.PathLike) -> str | None:
    """None if the run's log is closed (log.parquet with its PAR1 footer, or log.jsonl), else why not."""
    d = Path(run_dir)
    p = d / "log.parquet"
    if p.is_file():
        b = p.read_bytes()
        if len(b) < 12 or b[:4] != b"PAR1" or b[-4:] != b"PAR1":
            return "log.parquet is not closed (no PAR1 footer)"
        return None
    if (d / "log.jsonl").is_file():
        return None
    return "no log.parquet or log.jsonl"


def run_closed(run_dir: str | os.PathLike) -> str | None:
    """None if the run is closed (summary.json valid JSON and the log closed), else why not."""
    d = Path(run_dir)
    s = d / "summary.json"
    if not s.is_file():
        return NO_SUMMARY
    try:
        json.loads(s.read_text(encoding="utf-8"))
    except ValueError as e:
        return f"summary.json is not valid JSON ({e})"
    return log_closed(d)


def _run_id(run_dir: Path) -> str:
    """The run's id: summary.json's run_id when readable, else the directory name (twin_design.read_summary's rule)."""
    try:
        rid = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("run_id")
        return str(rid) if rid is not None else run_dir.name
    except (OSError, ValueError, AttributeError):
        return run_dir.name


def run_dirs(root: str | os.PathLike) -> list[Path]:
    """Every directory under root holding a run file (intercycle.eval.twin_design.RUN_MARKERS)."""
    return TD.find_runs(Path(root))


def assert_runs_closed(root: str | os.PathLike, invalid: Iterable[str] = (), crashed: Iterable[str] = ()) -> int:
    """Fail closed unless every run under root is finished (module doc, step 1: closed, or listed invalid, or listed
    crashed with no summary.json and a closed log; lists matched by run_id or directory name). Returns the number of
    finished runs that are not listed invalid."""
    inv, cr = set(invalid), set(crashed)
    bad, n = [], 0
    for d in run_dirs(root):
        rid = _run_id(d)
        if TD.listed(rid, d.name, inv):
            continue                                       # infrastructure-invalid: re-run, never scored
        why = run_closed(d)
        if why == NO_SUMMARY and TD.listed(rid, d.name, cr):
            why = log_closed(d)                            # crashed after the first command: finished if its log is
            if why is not None:
                why = f"listed crashed, but {why}"
        if why is None:
            n += 1
        else:
            bad.append(f"{d}: {why}")
    if bad:
        raise NotClosed("truth stays closed until every run is closed: " + "; ".join(bad[:10]))
    if n == 0:
        raise NotClosed(f"no closed run under {root}")
    return n


# ------------------------------------------------------------------------------------------------ streams


def stream_grids(d: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(truth (T, a, b), pred (Tp, a, b))] of a stream: one grid (`truth`, `pred`) or n_grids (`truth_<i>`,
    `pred_<i>`), each pair on its own grid."""
    if "n_grids" in d:
        n = int(d["n_grids"])
        if n < 1:
            raise ValueError("a stream needs at least one grid")
        return [(np.asarray(d[f"truth_{i}"], bool), np.asarray(d[f"pred_{i}"], bool)) for i in range(n)]
    return [(np.asarray(d["truth"], bool), np.asarray(d["pred"], bool))]


def check_stream(d: Mapping[str, Any], where: str) -> None:
    miss = [k for k in STREAM_KEYS if k not in d]
    if miss or not ("n_grids" in d or ("truth" in d and "pred" in d)):
        raise ValueError(f"{where}: missing keys {miss or ['truth/pred or n_grids']}")
    T, Tp = len(d["t"]), len(d["t_pred"])
    if len(d["age_s"]) != T or len(d["slice_id"]) != T:
        raise ValueError(f"{where}: t, age_s and slice_id must have one entry per tick")
    for i, (tr, pr) in enumerate(stream_grids(d)):
        if tr.ndim != 3 or tr.shape[0] != T:
            raise ValueError(f"{where}: truth grid {i} must be (len(t), a, b)")
        if pr.ndim != 3 or pr.shape[0] != Tp or pr.shape[1:] != tr.shape[1:]:
            raise ValueError(f"{where}: pred grid {i} must be (len(t_pred), a, b) on its truth grid")


def load_stream(path: str | os.PathLike) -> dict:
    """One prebuilt truth stream (module doc), opened through open_truth (logged)."""
    p = Path(path)
    if TOKEN not in p.name:
        raise ValueError(f"{p}: a truth stream's file name must carry {TOKEN!r}")
    with open_truth(p, "rb") as f:
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    check_stream(d, str(p))
    return d


def align_by_age(t_use: np.ndarray, age_s: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    """= intercycle.eval.conformal_margin.align_by_age: the last prediction with t_pred <= t - age (-1 if none)."""
    t_use, age_s, t_pred = (np.asarray(a, float) for a in (t_use, age_s, t_pred))
    if np.any(~np.isfinite(age_s)) or np.any(age_s < 0):
        raise ValueError("age_s must be finite and >= 0")
    if len(t_pred) > 1 and np.any(np.diff(t_pred) < 0):
        raise ValueError("t_pred must be sorted ascending")
    return np.searchsorted(t_pred, t_use - age_s + _EPS, side="right").astype(np.int64) - 1


def tick_counts(truth: np.ndarray, pred: np.ndarray, idx: np.ndarray, margin: Margin, cell_m: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per tick: occ = truly occupied cells, miss = those outside inflate(pred[idx]) (0, 0 where idx < 0)."""
    truth = np.asarray(truth, bool)
    pred = np.asarray(pred, bool)
    occ = np.zeros(len(truth), np.int64)
    miss = np.zeros(len(truth), np.int64)
    for k in np.unique(idx[idx >= 0]):
        cov = inflate(pred[int(k)], margin, cell_m=cell_m)
        rows = np.nonzero(idx == k)[0]
        tr = truth[rows]
        occ[rows] = tr.sum(axis=(1, 2))
        miss[rows] = (tr & ~cov[None]).sum(axis=(1, 2))
    return occ, miss


def stream_counts(stream: str | os.PathLike | Mapping[str, Any], margin: Margin) -> dict:
    """Per-tick counts of one stream (a prebuilt .npz path or a stream dict from `build_stream`), summed over its
    grids."""
    if isinstance(stream, Mapping):
        d, name = stream, str(stream.get("stream", "<built>"))
        check_stream(d, name)
    else:
        d, name = load_stream(stream), Path(stream).name
    cell = float(d["cell_m"])
    idx = align_by_age(d["t"], d["age_s"], d["t_pred"])
    occ = np.zeros(len(d["t"]), np.int64)
    miss = np.zeros(len(d["t"]), np.int64)
    for tr, pr in stream_grids(d):
        o, m = tick_counts(tr, pr, idx, margin, cell)
        occ += o
        miss += m
    rid = str(np.asarray(d["run_id"]).item())
    return {"stream": name, "world": int(d["world"]), "day": int(d["day"]), "zone": str(d["zone"]),
            "split": str(d["split"]), "run_id": rid, "attempt": TD.attempt_of(rid), "cell_m": cell,
            "t": np.asarray(d["t"], float).tolist(),
            "slice_id": np.asarray(d["slice_id"]).astype(int).tolist(), "occ": occ.tolist(), "miss": miss.tolist(),
            "valid": (idx >= 0).tolist()}


def h3_tick_counts(streams: Sequence[str | os.PathLike | Mapping[str, Any]], runs_root: str | os.PathLike,
                   margin: Margin, invalid: Iterable[str] = (), *, split: str, days: Sequence[int],
                   crashed: Iterable[str] = ()) -> dict:
    """The gate-3 counts file (module doc). One stream per (world, day, zone); truth only after every run finished.
    Refuses a stream of another split or of a day that is not scored."""
    assert_runs_closed(runs_root, invalid, crashed)
    ds = {int(x) for x in days}
    out, seen = [], set()
    for p in streams:
        c = stream_counts(p, margin)
        if c["split"] != split:
            raise ValueError(f"{c['stream']}: split {c['split']!r} is not the evaluated split {split!r}")
        if c["day"] not in ds:
            raise ValueError(f"{c['stream']}: day {c['day']} is not a scored day {sorted(ds)}")
        key = (c["world"], c["day"], c["zone"])
        if key in seen:
            raise ValueError(f"two streams for (world, day, zone) = {key}")
        if not math.isclose(c["cell_m"], margin.cell_m, abs_tol=1e-12):
            raise ValueError(f"{c['stream']}: grid cell {c['cell_m']} != the margin's cell {margin.cell_m}")
        seen.add(key)
        out.append(c)
    return {"contract": CONTRACT, "margin_m": margin.margin_m, "cell_m": margin.cell_m, "split": split,
            "days": sorted(ds), "streams": out}


# ------------------------------------------------------------------------------------------------ stream builder


def load_held_map(run_dir: str | os.PathLike) -> dict:
    """The run's HELD_MAP record (module doc), checked. Not truth: plain np.load."""
    p = Path(run_dir) / HELD_MAP
    with np.load(p, allow_pickle=False) as z:
        h = {k: z[k] for k in z.files}
    miss = [k for k in HELD_KEYS if k not in h]
    if miss:
        raise ValueError(f"{p}: missing keys {miss}")
    M = len(h["press_ids"])
    for k in ("x0", "z0", "nx", "nz"):
        if len(h[k]) != M:
            raise ValueError(f"{p}: {k} needs one entry per press")
    T, Tp = len(h["t_use"]), len(h["t_pred"])
    if len(h["age_s"]) != T or len(h["slice_id"]) != T:
        raise ValueError(f"{p}: t_use, age_s and slice_id must have one entry per robot-use tick")
    for k in range(M):
        pr = h.get(f"pred_{k}")
        if pr is None or pr.ndim != 3 or pr.shape != (Tp, int(h["nx"][k]), int(h["nz"][k])):
            raise ValueError(f"{p}: pred_{k} must be (len(t_pred), nx[{k}], nz[{k}])")
    return h


def truth_grid(tick: RealisedTick, press_id: str, centre: tuple[float, float], x0: float, z0: float, nx: int,
               nz: int, res: float) -> np.ndarray:
    """[nx, nz] bool: the module's TRUTH GRID rule for one press at one realised tick."""
    pr = next((p for p in tick.presses if p.id == press_id), None)
    if pr is None:
        raise ValueError(f"tick t={tick.t}: press {press_id} is not in the realised stream")
    g = np.zeros((nx, nz), bool)
    if not (pr.state == RUNNING or pr.ram > 0 or pr.stroke):
        return g
    cx = x0 + (np.arange(nx) + 0.5) * res
    cz = z0 + (np.arange(nz) + 0.5) * res
    ix = np.abs(cx - centre[0]) <= RAM_HALF_M + _EPS
    iz = np.abs(cz - centre[1]) <= RAM_HALF_M + _EPS
    g[np.ix_(ix, iz)] = True
    return g


def build_stream(run_dir: str | os.PathLike, site: Site, margin: Margin) -> dict:
    """One (world, day, zone) stream dict from a closed run: its summary (identity), its realised_state.jsonl (truth,
    through open_truth) and its HELD_MAP record. Call only after `assert_runs_closed` (h3_from_runs does)."""
    d = Path(run_dir)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    world, day, zone, split, rid = int(s["world"]), int(s["day"]), str(s["zone"]), str(s["world_split"]), s["run_id"]
    h = load_held_map(d)
    res = float(h["res"])
    if not math.isclose(res, margin.cell_m, abs_tol=1e-12):
        raise ValueError(f"{d / HELD_MAP}: grid cell {res} != the margin's cell {margin.cell_m}")
    ids = [str(x) for x in h["press_ids"]]
    mine = [p for p in site.presses if p.zone == zone]
    if not mine:
        raise ValueError(f"zone {zone} has no press in the site")
    lost = [p.id for p in mine if p.id not in ids]
    if lost:
        raise ValueError(f"{d / HELD_MAP}: no held grid for the zone's presses {lost}")
    with open_truth(d / REALISED, "r") as f:
        header, ticks_it = parse_realised(f)
        ticks = {round(tk.t * TICK_HZ): tk for tk in ticks_it}
    if (header.world, header.day, header.run_id) != (world, day, rid):
        raise ValueError(f"{d / REALISED}: header (world, day, run_id) = {(header.world, header.day, header.run_id)}"
                         f" != summary {(world, day, rid)}")
    t_use = np.asarray(h["t_use"], float)
    use = []
    for t in t_use:
        k = round(float(t) * TICK_HZ)
        if abs(float(t) * TICK_HZ - k) > 1e-6 or k not in ticks:
            raise ValueError(f"{d / HELD_MAP}: robot-use tick {t} is not a tick of the realised stream")
        use.append(ticks[k])
    out: dict[str, Any] = {"stream": f"{rid}:{REALISED}", "t": t_use, "t_pred": np.asarray(h["t_pred"], float),
                           "age_s": np.asarray(h["age_s"], float), "slice_id": np.asarray(h["slice_id"], np.int64),
                           "world": world, "day": day, "zone": zone, "cell_m": res, "split": split,
                           "n_grids": len(mine), "run_id": rid}
    for i, p in enumerate(mine):
        k = ids.index(p.id)
        fp = np.asarray(p.footprint.value, float)
        centre = (float(fp[:, 0].mean()), float(fp[:, 1].mean()))
        x0, z0, nx, nz = float(h["x0"][k]), float(h["z0"][k]), int(h["nx"][k]), int(h["nz"][k])
        out[f"truth_{i}"] = (np.stack([truth_grid(tk, p.id, centre, x0, z0, nx, nz, res) for tk in use])
                             if use else np.zeros((0, nx, nz), bool))
        out[f"pred_{i}"] = np.asarray(h[f"pred_{k}"], bool)
    check_stream(out, str(d))
    return out


def h3_from_runs(runs_root: str | os.PathLike, tasks: str | os.PathLike, site_path: str | os.PathLike,
                 margin: Margin, *, worlds: Sequence[int] | None, days: Sequence[int] | None = None,
                 split: str = "test", engine: str = "double", invalid: str | os.PathLike | None = None,
                 allow_partial_design: bool = False, conditions: Sequence[str] | None = None) -> dict:
    """The gate-3 counts file built from the run directories (module doc, step 2). The design and the scored attempt
    per cell come from intercycle.eval.twin_design (the rule twin_eval scores the primary with); truth is opened only
    after every run under runs_root is closed."""
    inv, crashed = TD.load_classified(None if invalid is None else Path(invalid))
    assert_runs_closed(runs_root, inv, crashed)           # before any truth opening
    planned = TD.mission_orders(Path(tasks))
    runs = TD.summary_runs(Path(runs_root), inv)
    design = TD.make_design(runs, {**crashed, **inv}, split=split, engine=engine, worlds=worlds, days=days,
                            conditions=conditions, allow_partial_design=allow_partial_design)
    if H3_CONDITION not in design.conditions:
        raise ValueError(f"the design has no {H3_CONDITION} condition")
    cells = TD.index_cells(runs, planned, inv, split=split, engine=engine, design=design, crashed=crashed)
    site = load(Site, Path(site_path))
    streams, skipped = [], []
    for w, d, z in sorted(cells.chosen):
        r = cells.runs_of((w, d, z)).get(H3_CONDITION)
        where = f"w{w}_d{d}_{z}"
        if r is None or r.summary is None or TD.is_bad(r, inv):
            skipped.append({"cell": where, "reason": f"no clean {H3_CONDITION} run"})
            continue
        try:
            streams.append(build_stream(r.run_dir, site, margin))
        except (OSError, ValueError, KeyError) as e:
            skipped.append({"cell": where, "run_id": r.run_id, "reason": repr(e)[:300]})
    body = h3_tick_counts(streams, runs_root, margin, inv, split=split, days=design.days, crashed=crashed)
    body.update(design=design.as_dict(), skipped=skipped)
    return body


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EVAL gate-3 per-tick miss counts (after the run only).")
    ap.add_argument("streams", nargs="*", help="prebuilt stream .npz files (else built from the runs with --site)")
    ap.add_argument("--runs", required=True, help="runs root: every run must be closed")
    ap.add_argument("--margin-file", help="the frozen M-W7 margin JSON (twin/m7_margin.json)")
    ap.add_argument("--margin-m", type=float, help="the disclosed fallback margin (with --cell-m) instead")
    ap.add_argument("--cell-m", type=float)
    ap.add_argument("--invalid", help="classify_invalid's committed list ({\"invalid\": [...], \"crashed\": [...]})")
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--engine", default="double", choices=("double", "isaac"))
    ap.add_argument("--days", type=_ints, help="scored days (default 9,10 on the test split)")
    ap.add_argument("--worlds", type=_ints, help="the design's world indices (with --site)")
    ap.add_argument("--tasks", help="twin/tasks.yaml (with --site)")
    ap.add_argument("--site", help="twin/site.yaml: build the streams from the run directories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.margin_file:
        margin = load_margin(a.margin_file)
    elif a.margin_m is not None and a.cell_m is not None:
        margin = Margin(a.margin_m, a.cell_m)
    else:
        ap.error("give --margin-file, or --margin-m with --cell-m")
    if a.site:
        if a.streams or not a.tasks:
            ap.error("--site builds the streams from the runs: give --tasks and no stream files")
        rep = h3_from_runs(a.runs, a.tasks, a.site, margin, worlds=a.worlds, days=a.days, split=a.split,
                           engine=a.engine, invalid=a.invalid)
    else:
        if not a.streams:
            ap.error("give stream files, or --site with --tasks to build them from the runs")
        days = a.days or (list(TD.TEST_DAYS) if a.split == "test" else None)
        if not days:
            ap.error("--days is required on the dev split")
        inv, crashed = TD.load_classified(a.invalid)
        rep = h3_tick_counts(a.streams, a.runs, margin, inv, split=a.split, days=days, crashed=crashed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rep['streams'])} streams at margin {margin.margin_m} m -> {a.out}"
          + (f"; skipped {len(rep['skipped'])} cells" if rep.get("skipped") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
