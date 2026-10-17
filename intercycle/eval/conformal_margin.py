"""M-W7 safety margin: offline Learn-then-Test calibration at the real fast-layer latency (lane S7b, PLAN_TWIN §4).

What it fixes (docs/WEAKNESSES.md M-W7, the old `map_metrics.conformal_margin`):
  * zero latency -> the truth the robot meets is paired with the map prediction the robot actually HOLDS at that instant,
    i.e. the prediction built from data `age` seconds old, where `age` is the real, measured fast-layer age per tick
    (`align_by_age`). Calibrating at age 0 is optimistic (movers keep moving while the message is in flight);
  * a mean-only guarantee -> Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021, arXiv 2110.01052) with the
    Hoeffding-Bentkus p-value (Bates et al. 2021, arXiv 2101.02703) and fixed-sequence testing from the largest margin
    down: P(R(margin_hat) <= alpha) >= 1 - delta over the draw of the calibration set;
  * correlated episodes -> the p-value uses a Kish effective sample size from the intra-class correlation of the unit
    losses within (world, day) clusters (`n_eff_mode="kish"`, a disclosed heuristic: LTT itself assumes iid units);
  * 4 hand-picked windows -> every unit of every calibration day is used, units never cross a time-slice boundary, and
    `calibrate` refuses a calibration set that misses a required slice;
  * unweighted vs weighted means -> ONE estimand, used by calibration and by evaluation alike: the risk is the
    UNWEIGHTED mean over units of each unit's pooled miss share (`unit_mean_miss_share`). EVAL computes the gate-3
    per-world miss share with that same function.

Definitions
  * unit = one episode: consecutive ticks of one (world, day) stream inside one time slice and one `episode_s` block
    (default 720 s = one robot mission, PLAN_TWIN §5 "fixed denominator T = 720 s");
  * truly occupied cell-instant = a True cell of the truth grid at a robot-use tick (the caller decides what is truth:
    moving hazards' footprints, the press ram, ...);
  * covered at margin m (metres) = the Euclidean distance from the cell centre to the nearest cell the robot's map
    marks occupied/UNKNOWN in the prediction it holds is <= m (a disc dilation of the prediction by m);
  * unit miss share(m) = uncovered / truly occupied cell-instants of the unit; units with no truly occupied
    cell-instant carry no loss and are left out of both calibration and evaluation (their count is reported);
  * risk R(m) = E[unit miss share(m)], the unweighted mean over units.

Isolation (PLAN_TWIN §2 truth guards): this module reads truth (dev worlds only). It must never be imported by
twin/observer_live.py, the robot policy or the broadcaster; tests/test_conformal_margin_isolation.py checks their import
graphs. They read only the frozen JSON constant this module writes (`write_margin_file`, key `margin_m`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage, stats

SCHEMA = "intercycle.m7_margin/v1"
DEFAULT_EPISODE_S = 720.0
DEFAULT_MARGINS_M = tuple(round(0.05 * k, 2) for k in range(0, 81))  # 0 .. 4 m in 5 cm steps
DEFAULT_OUT = "twin/m7_margin.json"
DEV_SEED_MAX = 999  # PLAN_TWIN §5: dev seeds come from 0-999
_EPS = 1e-9


# ------------------------------------------------------------------------------------------------ alignment

def align_by_age(t_use: np.ndarray, age_s: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    """Index of the prediction the robot HOLDS at each robot-use tick.

    t_use[j]  robot-use instant (the truth tick), seconds;
    age_s[j]  the real fast-layer age at t_use[j]: t_use[j] minus the observation time of the newest data behind the map
              message the robot holds (measured in the dev run, never assumed);
    t_pred[k] observation time of the data behind prediction k (sorted ascending).
    Returns idx[j] = the last k with t_pred[k] <= t_use[j] - age_s[j] (the newest prediction that data could have
    produced), or -1 when no prediction existed yet (that tick is then left out and counted)."""
    t_use = np.asarray(t_use, float)
    age_s = np.asarray(age_s, float)
    t_pred = np.asarray(t_pred, float)
    if t_use.shape != age_s.shape:
        raise ValueError("t_use and age_s must have the same shape")
    if np.any(~np.isfinite(age_s)) or np.any(age_s < 0):
        raise ValueError("age_s must be finite and >= 0")
    if len(t_pred) > 1 and np.any(np.diff(t_pred) < 0):
        raise ValueError("t_pred must be sorted ascending")
    return np.searchsorted(t_pred, t_use - age_s + _EPS, side="right").astype(np.int64) - 1


# ------------------------------------------------------------------------------------------------ per-tick counts

def _distance_cells(pred2d: np.ndarray) -> np.ndarray:
    """Euclidean distance (in cells, centre to centre) from every cell to the nearest predicted cell; inf if none."""
    if not pred2d.any():
        return np.full(pred2d.shape, np.inf)
    if pred2d.all():
        return np.zeros(pred2d.shape)
    return np.asarray(ndimage.distance_transform_edt(~pred2d), float)


def tick_counts(truth: np.ndarray, pred: np.ndarray, idx: np.ndarray, cell_m: float, margins_m: "Sequence[float] | np.ndarray",
                chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per robot-use tick j: occ[j] = truly occupied cells, miss[j, k] = those farther than margins_m[k] from the
    prediction pred[idx[j]] (disc dilation). Ticks with idx < 0 get occ = miss = 0 and must be masked by the caller.

    truth (T, ny, nx) bool at the robot-use ticks; pred (Tp, ny, nx) bool (occupied OR UNKNOWN, as the robot treats it)."""
    truth = np.asarray(truth, bool)
    pred = np.asarray(pred, bool)
    idx = np.asarray(idx, np.int64)
    margins = np.asarray(margins_m, float)
    if truth.ndim != 3 or pred.ndim != 3 or truth.shape[1:] != pred.shape[1:]:
        raise ValueError("truth and pred must be (T, ny, nx) stacks on the same grid")
    if len(idx) != len(truth):
        raise ValueError("idx must have one entry per truth tick")
    if np.any(np.diff(margins) <= 0) or margins[0] < 0:
        raise ValueError("margins_m must be >= 0 and strictly increasing")
    if np.any(idx >= len(pred)):
        raise ValueError("idx points past the prediction stack")
    T, M = len(truth), len(margins)
    margin_cells = margins / float(cell_m)
    occ = np.zeros(T, np.int64)
    miss = np.zeros((T, M), np.int64)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        ii = idx[s:e]
        ok = np.nonzero(ii >= 0)[0]
        if not len(ok):
            continue
        uniq, inv = np.unique(ii[ok], return_inverse=True)
        D = np.stack([_distance_cells(pred[k]) for k in uniq])
        tt, yy, xx = np.nonzero(truth[s:e][ok])
        d = D[inv[tt], yy, xx]
        # b = number of margins strictly below d (with a float tolerance): the cell is missed at margin k iff k < b
        b = np.searchsorted(margin_cells, d - _EPS, side="left")
        hist = np.zeros((len(ok), M + 1), np.int64)
        np.add.at(hist, (tt, b), 1)
        rows = s + ok
        occ[rows] = hist.sum(1)
        # miss[:, k] = sum_{b > k} hist[:, b]
        miss[rows] = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return occ, miss


# ------------------------------------------------------------------------------------------------ units

@dataclass
class UnitCounts:
    """Per-unit (episode) counts over one or more (world, day) streams. miss[u, k] at margins_m[k]."""
    world: np.ndarray
    day: np.ndarray
    slice_id: np.ndarray
    t0: np.ndarray
    n_ticks: np.ndarray
    occ: np.ndarray
    miss: np.ndarray
    margins_m: np.ndarray
    dropped_ticks: int = 0

    @staticmethod
    def concat(parts: Iterable["UnitCounts"]) -> "UnitCounts":
        parts = list(parts)
        if not parts:
            raise ValueError("no units")
        m0 = parts[0].margins_m
        for p in parts[1:]:
            if p.margins_m.shape != m0.shape or not np.allclose(p.margins_m, m0):
                raise ValueError("all parts must use the same margin grid")
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])  # noqa: E731
        return UnitCounts(cat("world"), cat("day"), cat("slice_id"), cat("t0"), cat("n_ticks"), cat("occ"),
                          np.concatenate([p.miss for p in parts], axis=0), m0.copy(),
                          int(sum(p.dropped_ticks for p in parts)))

    @property
    def scored(self) -> np.ndarray:
        """Units that carry a loss (at least one truly occupied cell-instant)."""
        return self.occ > 0

    def losses(self) -> np.ndarray:
        """(U_scored, M) unit miss shares in [0, 1], non-increasing along the margin axis."""
        s = self.scored
        return self.miss[s] / self.occ[s][:, None]

    def clusters(self) -> np.ndarray:
        """(world, day) cluster label of every scored unit."""
        s = self.scored
        _, lab = np.unique(np.stack([self.world[s], self.day[s]], 1), axis=0, return_inverse=True)
        return lab.reshape(-1)

    def subset(self, mask: np.ndarray) -> "UnitCounts":
        return UnitCounts(self.world[mask], self.day[mask], self.slice_id[mask], self.t0[mask], self.n_ticks[mask],
                          self.occ[mask], self.miss[mask], self.margins_m.copy(), 0)

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.world, self.day, self.slice_id, self.t0, self.n_ticks, self.occ, self.miss, self.margins_m):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def stream_units(t: np.ndarray, slice_id: np.ndarray, occ: np.ndarray, miss: np.ndarray, valid: np.ndarray, *,
                 world: int, day: int, margins_m: "Sequence[float] | np.ndarray", episode_s: float = DEFAULT_EPISODE_S) -> UnitCounts:
    """Group the per-tick counts of one (world, day) stream into units: a new unit starts when the time slice changes
    or a new `episode_s` block (counted from the stream's first tick) starts. Invalid ticks (no prediction held) are
    left out and counted in `dropped_ticks`."""
    t = np.asarray(t, float)
    slice_id = np.asarray(slice_id, np.int64)
    valid = np.asarray(valid, bool)
    if not (len(t) == len(slice_id) == len(occ) == len(miss) == len(valid)):
        raise ValueError("per-tick arrays must have the same length")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    dropped = int((~valid).sum())
    keep = np.nonzero(valid)[0]
    margins = np.asarray(margins_m, float)
    if not len(keep):
        e = np.zeros(0)
        return UnitCounts(e.astype(np.int64), e.astype(np.int64), e.astype(np.int64), e, e.astype(np.int64),
                          e.astype(np.int64), np.zeros((0, len(margins)), np.int64), margins, dropped)
    block = np.floor((t[keep] - t[0]) / float(episode_s)).astype(np.int64)
    sl = slice_id[keep]
    new = np.r_[True, (block[1:] != block[:-1]) | (sl[1:] != sl[:-1])]
    uid = np.cumsum(new) - 1
    U = int(uid[-1]) + 1
    starts = np.nonzero(new)[0]
    return UnitCounts(world=np.full(U, int(world), np.int64), day=np.full(U, int(day), np.int64),
                      slice_id=sl[starts].astype(np.int64), t0=t[keep][starts],
                      n_ticks=np.bincount(uid, minlength=U).astype(np.int64),
                      occ=np.bincount(uid, weights=np.asarray(occ)[keep], minlength=U).astype(np.int64),
                      miss=np.stack([np.bincount(uid, weights=np.asarray(miss)[keep, k], minlength=U)
                                     for k in range(len(margins))], 1).astype(np.int64),
                      margins_m=margins, dropped_ticks=dropped)


def units_from_stream(t: np.ndarray, truth: np.ndarray, t_pred: np.ndarray, pred: np.ndarray, age_s: np.ndarray,
                      slice_id: np.ndarray, *, world: int, day: int, cell_m: float,
                      margins_m: "Sequence[float] | np.ndarray" = DEFAULT_MARGINS_M, episode_s: float = DEFAULT_EPISODE_S) -> UnitCounts:
    """The clean input interface for one (world, day) dev stream.

    t (T,) robot-use ticks [s]; truth (T, ny, nx) bool truth occupancy AT those ticks; t_pred (Tp,) observation time of the
    data behind each map prediction; pred (Tp, ny, nx) bool the map's predicted occupancy (occupied or UNKNOWN) from that
    data; age_s (T,) the real fast-layer age at each tick; slice_id (T,) the map time slice of each tick.
    Pass age_s = 0 for the (optimistic) zero-latency calibration."""
    idx = align_by_age(t, age_s, t_pred)
    occ, miss = tick_counts(truth, pred, idx, cell_m, margins_m)
    return stream_units(t, slice_id, occ, miss, idx >= 0, world=world, day=day, margins_m=margins_m,
                        episode_s=episode_s)


def units_from_aligned(t: np.ndarray, truth: np.ndarray, pred_used: np.ndarray, slice_id: np.ndarray, *, world: int,
                       day: int, cell_m: float, margins_m: "Sequence[float] | np.ndarray" = DEFAULT_MARGINS_M,
                       episode_s: float = DEFAULT_EPISODE_S) -> UnitCounts:
    """Same as `units_from_stream` when the caller already paired each truth tick with the prediction the robot used
    at that tick (e.g. the robot-side mask after its own phase advance). The age shift is then the caller's job."""
    idx = np.arange(len(t), dtype=np.int64)
    occ, miss = tick_counts(truth, pred_used, idx, cell_m, margins_m)
    return stream_units(t, slice_id, occ, miss, np.ones(len(t), bool), world=world, day=day, margins_m=margins_m,
                        episode_s=episode_s)


# ------------------------------------------------------------------------------------------------ estimand (shared with EVAL)

def unit_mean_miss_share(counts: UnitCounts, margin_m: float) -> float:
    """THE risk estimand: unweighted mean over scored units of the unit's pooled miss share at `margin_m` (must be on
    the grid). EVAL's gate-3 per-world miss share must call this on that world's test-day units."""
    k = _margin_index(counts.margins_m, margin_m)
    L = counts.losses()
    return float(L[:, k].mean()) if len(L) else math.nan


def per_world_miss_share(counts: UnitCounts, margin_m: float) -> dict[int, float]:
    out = {}
    for w in np.unique(counts.world):
        out[int(w)] = unit_mean_miss_share(counts.subset(counts.world == w), margin_m)
    return out


def _margin_index(margins: np.ndarray, m: float) -> int:
    k = int(np.argmin(np.abs(margins - m)))
    if abs(margins[k] - m) > 1e-6:
        raise ValueError(f"margin {m} is not on the calibration grid")
    return k


# ------------------------------------------------------------------------------------------------ Learn-then-Test

def hb_pvalue(rhat: float, n: float, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R > alpha, from the mean `rhat` of n iid losses in [0, 1] (Bates et al. 2021,
    Prop. 3.5; used by LTT). Non-integer n (an effective sample size) is floored, which is conservative."""
    n_int = int(math.floor(n + _EPS))
    if n_int <= 0:
        return 1.0
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    a = min(max(rhat, 0.0), alpha)
    h1 = (a * math.log(a / alpha) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - alpha))
    hoeffding = math.exp(-n_int * h1)
    k = int(math.ceil(n_int * max(rhat, 0.0) - _EPS))
    bentkus = math.e * float(stats.binom.cdf(k, n_int, alpha))
    return float(min(1.0, hoeffding, bentkus))


def icc_oneway(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA intra-class correlation (unbalanced groups), clamped to [0, 1]. One group -> 1 (fully
    dependent, conservative); zero total variance -> 0 (no evidence of clustering)."""
    y = np.asarray(y, float)
    g = np.asarray(groups)
    labs, inv = np.unique(g, return_inverse=True)
    C, N = len(labs), len(y)
    if N == 0:
        return 0.0
    if C < 2:
        return 1.0
    if C == N:
        return 0.0
    nc = np.bincount(inv, minlength=C).astype(float)
    mc = np.bincount(inv, weights=y, minlength=C) / nc
    grand = y.mean()
    msb = float((nc * (mc - grand) ** 2).sum() / (C - 1))
    msw = float(((y - mc[inv]) ** 2).sum() / (N - C))
    n0 = (N - (nc ** 2).sum() / N) / (C - 1)
    den = msb + (n0 - 1) * msw
    if den <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (msb - msw) / den)))


def kish_n_eff(y: np.ndarray, groups: np.ndarray) -> float:
    """Kish effective sample size N / (1 + (m_bar - 1) rho), m_bar = sum n_c^2 / N (unequal cluster sizes)."""
    y = np.asarray(y, float)
    N = len(y)
    if N == 0:
        return 0.0
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    nc = np.bincount(inv).astype(float)
    m_bar = float((nc ** 2).sum() / N)
    return float(N / (1.0 + (m_bar - 1.0) * icc_oneway(y, groups)))


def ltt_fixed_sequence(losses: np.ndarray, margins_m: "Sequence[float] | np.ndarray", alpha: float, delta: float,
                       clusters: np.ndarray | None = None, n_eff_mode: str = "kish") -> dict:
    """Learn-then-Test with fixed-sequence testing from the LARGEST margin down (the order is fixed before seeing data;
    the loss is non-increasing in the margin). H0_m: R(m) > alpha is rejected when p_m <= delta; testing stops at the
    first non-rejection. The chosen margin is the smallest rejected one; FWER <= delta gives
    P(R(chosen) <= alpha) >= 1 - delta. Returns chosen index / margin (None when even the largest margin fails)."""
    L = np.asarray(losses, float)
    margins = np.asarray(margins_m, float)
    if L.ndim != 2 or L.shape[1] != len(margins):
        raise ValueError("losses must be (n_units, n_margins)")
    if np.any(L < -_EPS) or np.any(L > 1 + _EPS):
        raise ValueError("losses must lie in [0, 1]")
    if n_eff_mode not in ("kish", "none"):
        raise ValueError("n_eff_mode must be 'kish' or 'none'")
    if n_eff_mode == "kish" and clusters is None:
        raise ValueError("kish n_eff needs the cluster label of every unit")
    n = L.shape[0]
    rows = []
    for k in range(len(margins)):
        rhat = float(L[:, k].mean()) if n else math.nan
        ne = kish_n_eff(L[:, k], clusters) if n_eff_mode == "kish" else float(n)  # type: ignore[arg-type]
        rows.append(dict(margin_m=float(margins[k]), risk_cal=rhat, n_eff=ne,
                         p_value=hb_pvalue(rhat, ne, alpha) if n else 1.0, rejected=False))
    chosen = None
    for k in range(len(margins) - 1, -1, -1):
        if rows[k]["p_value"] <= delta:
            rows[k]["rejected"] = True
            chosen = k
        else:
            break
    return dict(alpha=float(alpha), delta=float(delta), n_units=int(n), n_eff_mode=n_eff_mode,
                chosen_index=chosen, margin_m=None if chosen is None else float(margins[chosen]), curve=rows)


# ------------------------------------------------------------------------------------------------ calibration

def calibrate(counts: UnitCounts, alpha: float, delta: float, *, n_eff_mode: str = "kish",
              required_slices: Sequence[int] | None = None, min_units_per_slice: int = 1,
              zero_latency: UnitCounts | None = None, cell_m: float | None = None, episode_s: float | None = None,
              age_summary: dict | None = None, provenance: dict | None = None) -> dict:
    """Calibrate the M-W7 margin on dev-world units (all time slices). Returns the content of the frozen constant file.

    required_slices: the time slices the robot can run in; calibration refuses if any has fewer than
    `min_units_per_slice` scored units ("calibrated over all time slices"). zero_latency: the same streams counted with
    age 0, reported only (shows the zero-latency margin is smaller, M-W7)."""
    s = counts.scored
    if not s.any():
        raise ValueError("no scored unit (no truly occupied cell-instant) in the calibration set")
    sl_scored = counts.slice_id[s]
    per_slice_n = {int(k): int((sl_scored == k).sum()) for k in np.unique(counts.slice_id)}
    if required_slices is not None:
        missing = [int(k) for k in required_slices if per_slice_n.get(int(k), 0) < min_units_per_slice]
        if missing:
            raise ValueError(f"calibration does not cover every time slice: too few scored units in slices {missing}")
    L = counts.losses()
    res = ltt_fixed_sequence(L, counts.margins_m, alpha, delta, counts.clusters(), n_eff_mode)
    out: dict = dict(schema=SCHEMA, status="ok" if res["margin_m"] is not None else "no_valid_margin",
                     alpha=float(alpha), delta=float(delta), margin_m=res["margin_m"],
                     margin_cells=None if res["margin_m"] is None or cell_m is None else res["margin_m"] / cell_m,
                     cell_m=cell_m, estimand="unweighted mean over units of the unit-pooled miss share",
                     unit=dict(episode_s=episode_s, kind="episode within one (world, day) and one time slice"),
                     method=dict(name="learn_then_test", pvalue="hoeffding_bentkus", fwer="fixed_sequence_desc",
                                 n_eff_mode=n_eff_mode, cluster="(world, day)"),
                     n_units_total=int(len(counts.occ)), n_units_scored=int(s.sum()),
                     n_units_unscored=int((~s).sum()), dropped_ticks=int(counts.dropped_ticks),
                     worlds=sorted(int(w) for w in np.unique(counts.world)),
                     days=sorted(int(d) for d in np.unique(counts.day)),
                     units_per_slice=per_slice_n, curve=res["curve"], input_sha256=counts.sha256())
    if res["margin_m"] is not None:
        k = res["chosen_index"]
        out["risk_cal_at_margin"] = float(L[:, k].mean())
        out["risk_per_slice_at_margin"] = {int(sid): float(L[sl_scored == sid, k].mean())
                                           for sid in np.unique(sl_scored)}
        out["leave_one_world_out"] = _lowo(counts, alpha, delta, n_eff_mode)
    if zero_latency is not None:
        z = ltt_fixed_sequence(zero_latency.losses(), zero_latency.margins_m, alpha, delta, zero_latency.clusters(),
                               n_eff_mode)
        out["margin_m_zero_latency"] = z["margin_m"]
    if age_summary is not None:
        out["fast_layer_age_s"] = age_summary
    if provenance is not None:
        out["provenance"] = provenance
    return out


def _lowo(counts: UnitCounts, alpha: float, delta: float, n_eff_mode: str) -> list[dict]:
    """Descriptive: calibrate on all worlds but one, report the held-out world's risk at that margin."""
    rows = []
    worlds = np.unique(counts.world)
    if len(worlds) < 2:
        return rows
    for w in worlds:
        cal, test = counts.subset(counts.world != w), counts.subset(counts.world == w)
        if not cal.scored.any() or not test.scored.any():
            continue
        r = ltt_fixed_sequence(cal.losses(), cal.margins_m, alpha, delta, cal.clusters(), n_eff_mode)
        rows.append(dict(held_out_world=int(w), margin_m=r["margin_m"],
                         held_out_risk=None if r["margin_m"] is None else unit_mean_miss_share(test, r["margin_m"])))
    return rows


def age_summary(age_s: np.ndarray, source: str) -> dict:
    a = np.asarray(age_s, float)
    return dict(source=source, n=int(len(a)), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def write_margin_file(result: dict, path: str | os.PathLike, overwrite: bool = False) -> Path:
    """Write the frozen margin constant (JSON). Refuses a result without a valid margin (the plan's fallback, a fixed
    latency-p99 margin, is the lead's decision, not this module's) and refuses to overwrite a frozen file."""
    if result.get("status") != "ok" or result.get("margin_m") is None:
        raise ValueError("no valid margin: LTT rejected no margin on the grid; do not freeze (lead decides fallback)")
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists: the margin is frozen")
    body = dict(result, frozen=True, frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ dev-only stream files

def _truth_log(path: Path, purpose: str, log: str | os.PathLike | None) -> None:
    lp = Path(log or os.environ.get("INTERCYCLE_TRUTH_LOG", "results/truth_access.log"))
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\tintercycle.eval.conformal_margin\t"
                f"{path.as_posix()}\t{purpose}\n")


def load_dev_stream(path: str | os.PathLike, truth_log: str | os.PathLike | None = None) -> dict:
    """Load one (world, day) calibration stream (.npz: t, truth, t_pred, pred, age_s, slice_id, world_seed, day,
    cell_m, split). Dev worlds only: refuses split != 'dev', a seed outside 0-999, and any sealed run. Logs the access."""
    if os.environ.get("INTERCYCLE_SEALED_RUN") == "1":
        raise RuntimeError("M-W7 calibration never runs inside a sealed run")
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    split = str(d.get("split", ""))
    seed = int(d["world_seed"])
    if split != "dev":
        raise ValueError(f"{p}: split={split!r}; M-W7 calibration uses dev worlds only")
    if not (0 <= seed <= DEV_SEED_MAX):
        raise ValueError(f"{p}: world_seed {seed} is not a dev seed (0-{DEV_SEED_MAX})")
    _truth_log(p, "M-W7 margin calibration (dev world)", truth_log)
    return d


def calibrate_files(paths: Sequence[str | os.PathLike], alpha: float, delta: float, *,
                    margins_m: "Sequence[float] | np.ndarray" = DEFAULT_MARGINS_M, episode_s: float = DEFAULT_EPISODE_S,
                    required_slices: Sequence[int] | None = None, n_eff_mode: str = "kish",
                    truth_log: str | os.PathLike | None = None) -> dict:
    parts, zparts, ages, cells = [], [], [], set()
    for path in paths:
        d = load_dev_stream(path, truth_log)
        w, dy, cm = int(d["world_seed"]), int(d["day"]), float(d["cell_m"])
        for age, acc in ((d["age_s"], parts), (np.zeros_like(d["age_s"]), zparts)):
            acc.append(units_from_stream(d["t"], d["truth"], d["t_pred"], d["pred"], age, d["slice_id"], world=w,
                                         day=dy, cell_m=cm, margins_m=margins_m, episode_s=episode_s))
        ages.append(np.asarray(d["age_s"], float))
        cells.add(float(d["cell_m"]))
    if len(cells) != 1:
        raise ValueError(f"streams use different cell sizes {sorted(cells)}")
    res = calibrate(UnitCounts.concat(parts), alpha, delta, n_eff_mode=n_eff_mode, required_slices=required_slices,
                    zero_latency=UnitCounts.concat(zparts), cell_m=cells.pop(), episode_s=float(episode_s),
                    age_summary=age_summary(np.concatenate(ages), "measured per tick in the dev runs (age_s)"),
                    provenance=dict(stream_files=[Path(p).name for p in paths]))
    return res


def _parse_margins(spec: str) -> tuple[float, ...]:
    a, b, step = (float(x) for x in spec.split(":"))
    n = int(round((b - a) / step))
    return tuple(round(a + step * k, 6) for k in range(n + 1))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M-W7: calibrate and freeze the conformal safety margin on dev worlds.")
    ap.add_argument("streams", nargs="+", help="dev (world, day) stream .npz files")
    ap.add_argument("--alpha", type=float, required=True, help="risk level (frozen in PREREG v1)")
    ap.add_argument("--delta", type=float, required=True, help="confidence level (frozen in PREREG v1)")
    ap.add_argument("--episode-s", type=float, default=DEFAULT_EPISODE_S)
    ap.add_argument("--margins-m", default="0:4:0.05", help="start:stop:step in metres")
    ap.add_argument("--slices", default=None, help="comma-separated time slices that must all be covered")
    ap.add_argument("--n-eff", choices=("kish", "none"), default="kish")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    req = [int(x) for x in a.slices.split(",")] if a.slices else None
    res = calibrate_files(a.streams, a.alpha, a.delta, margins_m=_parse_margins(a.margins_m), episode_s=a.episode_s,
                          required_slices=req, n_eff_mode=a.n_eff)
    print(json.dumps({k: res.get(k) for k in ("status", "alpha", "delta", "margin_m", "margin_m_zero_latency",
                                              "n_units_scored", "risk_cal_at_margin")}, indent=1))
    if res["status"] != "ok":
        return 2
    write_margin_file(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
