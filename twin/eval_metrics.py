"""EVAL metrics: the truth reads of EVAL, after the run only (PLAN_TWIN §2 truth guards: the EVAL metrics module slot of
twin.truth_guard.PLAN_READERS; §5 gate 3).

    python -m twin.eval_metrics --runs RUNS_ROOT --margin-file twin/m7_margin.json [--invalid LIST.json] \
        --out results/sealed/h3_tick_counts.json STREAM.npz [STREAM.npz ...]

Gate 3 (map unsafe-free share) compares the observer map the robot held with the realised occupancy, so it needs
truth. This module:
  1. refuses to open any truth file until every run under RUNS_ROOT is closed: summary.json present and valid JSON
     (the plant writes it last) and the log closed (log.parquet with its PAR1 footer, or log.jsonl); a run directory
     without summary.json must be in classify_invalid's committed list (a crashed container), otherwise it refuses;
  2. opens each (world, day, zone) truth stream through `twin.truth_guard.open_truth` (allowlisted caller, every
     opening appended to results/truth_access.log). A stream is an .npz whose file name carries the realised_state
     token: t, truth (T, ny, nx) bool realised occupancy at the robot-use ticks, t_pred, pred (Tp, ny, nx) bool map
     prediction (occupied or UNKNOWN), age_s (measured fast-layer age per tick), slice_id, world (index), day, zone,
     cell_m, split;
  3. writes per-tick counts at the frozen M-W7 margin: occ = truly occupied cells, miss = those the inflated
     prediction does not cover (`intercycle.margin_inflate.inflate`, the rule the robot applies and the calibration
     cross-checks), valid = a prediction existed at t - age_s. No grid leaves this module.
intercycle.eval.twin_eval turns the counts into units and the per-world unit-mean miss share with
intercycle.eval.conformal_margin (stream_units, per_world_miss_share): the estimand the margin was calibrated on.

Isaac-safe imports only (numpy, stdlib, repo code); never imported by the robot side, observer_live or the broadcaster.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from intercycle.margin_inflate import Margin, inflate, load_margin
from twin.truth_guard import open_truth

CONTRACT = "twin.h3_tick_counts/1"
TOKEN = "realised_state"
RUN_MARKERS = ("summary.json", "log.parquet", "log.jsonl", "obs_v2.jsonl", "plant_summary.json")
STREAM_KEYS = ("t", "truth", "t_pred", "pred", "age_s", "slice_id", "world", "day", "zone", "cell_m", "split")
_EPS = 1e-9


class NotClosed(RuntimeError):
    """A run is not closed yet: truth may not be read."""


def run_closed(run_dir: str | os.PathLike) -> str | None:
    """None if the run is closed, else why not."""
    d = Path(run_dir)
    s = d / "summary.json"
    if not s.is_file():
        return "no summary.json"
    try:
        json.loads(s.read_text(encoding="utf-8"))
    except ValueError as e:
        return f"summary.json is not valid JSON ({e})"
    p = d / "log.parquet"
    if p.is_file():
        b = p.read_bytes()
        if len(b) < 12 or b[:4] != b"PAR1" or b[-4:] != b"PAR1":
            return "log.parquet is not closed (no PAR1 footer)"
        return None
    if (d / "log.jsonl").is_file():
        return None
    return "no log.parquet or log.jsonl"


def run_dirs(root: str | os.PathLike) -> list[Path]:
    """Every directory under root holding a run file."""
    out = set()
    for m in RUN_MARKERS:
        out |= {p.parent for p in Path(root).rglob(m)}
    return sorted(out)


def assert_runs_closed(root: str | os.PathLike, invalid: Iterable[str] = ()) -> int:
    """Fail closed unless every run under root is closed (crashed dirs without summary.json must be listed invalid).
    Returns the number of closed runs."""
    inv = set(invalid)
    bad, n = [], 0
    for d in run_dirs(root):
        why = run_closed(d)
        if why is None:
            n += 1
        elif not (why == "no summary.json" and d.name in inv):
            bad.append(f"{d}: {why}")
    if bad:
        raise NotClosed("truth stays closed until every run is closed: " + "; ".join(bad[:10]))
    if n == 0:
        raise NotClosed(f"no closed run under {root}")
    return n


def load_stream(path: str | os.PathLike) -> dict:
    """One truth stream (module doc), opened through open_truth (logged)."""
    p = Path(path)
    if TOKEN not in p.name:
        raise ValueError(f"{p}: a truth stream's file name must carry {TOKEN!r}")
    with open_truth(p, "rb") as f:
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    miss = [k for k in STREAM_KEYS if k not in d]
    if miss:
        raise ValueError(f"{p}: missing keys {miss}")
    T = len(d["t"])
    if d["truth"].shape[0] != T or len(d["age_s"]) != T or len(d["slice_id"]) != T:
        raise ValueError(f"{p}: t, truth, age_s and slice_id must have one entry per tick")
    if d["pred"].shape[0] != len(d["t_pred"]) or d["pred"].shape[1:] != d["truth"].shape[1:]:
        raise ValueError(f"{p}: pred must be (len(t_pred), ny, nx) on the truth grid")
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


def stream_counts(path: str | os.PathLike, margin: Margin) -> dict:
    d = load_stream(path)
    cell = float(d["cell_m"])
    idx = align_by_age(d["t"], d["age_s"], d["t_pred"])
    occ, miss = tick_counts(d["truth"], d["pred"], idx, margin, cell)
    return {"stream": Path(path).name, "world": int(d["world"]), "day": int(d["day"]), "zone": str(d["zone"]),
            "split": str(d["split"]), "cell_m": cell, "t": d["t"].astype(float).tolist(),
            "slice_id": d["slice_id"].astype(int).tolist(), "occ": occ.tolist(), "miss": miss.tolist(),
            "valid": (idx >= 0).tolist()}


def h3_tick_counts(streams: Sequence[str | os.PathLike], runs_root: str | os.PathLike, margin: Margin,
                   invalid: Iterable[str] = ()) -> dict:
    """The gate-3 counts file (module doc). One stream per (world, day, zone); truth only after every run closed."""
    assert_runs_closed(runs_root, invalid)
    out, seen = [], set()
    for p in streams:
        c = stream_counts(p, margin)
        key = (c["world"], c["day"], c["zone"])
        if key in seen:
            raise ValueError(f"two streams for (world, day, zone) = {key}")
        if c["cell_m"] != margin.cell_m:
            raise ValueError(f"{p}: grid cell {c['cell_m']} != the margin's cell {margin.cell_m}")
        seen.add(key)
        out.append(c)
    return {"contract": CONTRACT, "margin_m": margin.margin_m, "cell_m": margin.cell_m, "streams": out}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EVAL gate-3 per-tick miss counts (after the run only).")
    ap.add_argument("streams", nargs="+")
    ap.add_argument("--runs", required=True, help="runs root: every run must be closed")
    ap.add_argument("--margin-file", help="the frozen M-W7 margin JSON (twin/m7_margin.json)")
    ap.add_argument("--margin-m", type=float, help="the disclosed fallback margin (with --cell-m) instead")
    ap.add_argument("--cell-m", type=float)
    ap.add_argument("--invalid", help="classify_invalid's committed list (crashed run dirs without summary.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.margin_file:
        margin = load_margin(a.margin_file)
    elif a.margin_m is not None and a.cell_m is not None:
        margin = Margin(a.margin_m, a.cell_m)
    else:
        ap.error("give --margin-file, or --margin-m with --cell-m")
    inv: list[str] = []
    if a.invalid:
        body = json.loads(Path(a.invalid).read_text(encoding="utf-8"))
        inv = list(body.get("invalid", body.get("runs", [])) if isinstance(body, dict) else body)
    rep = h3_tick_counts(a.streams, a.runs, margin, inv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rep['streams'])} streams at margin {margin.margin_m} m -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
