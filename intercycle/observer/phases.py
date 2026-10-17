"""Unsupervised phase discovery from camera observations (DIRECTION §4). Never imports ground truth (tested).

Three levels, all learned from `mapping.Observation` ticks only (no PLC, no truth, no calendar):

1. **Machine sub-phases.** Per machine zone a per-tick feature vector: motion energy (moving machine-part detections)
   plus a coarse 4 x 4 occupancy grid of all machine-part detections over the zone (where the arms are). Then
   period (`learn_cycles.learn` on the motion signal, ACF fallback) -> cycle motif (matrix profile, `stumpy`;
   numpy fallback) -> DTW alignment of every cycle to the motif and DBA averaging -> a cycle template -> optimal
   change points on the template (dynamic programming, the `ruptures.Dynp` cost; 4-8 segments chosen by a
   penalised cost) -> one sub-phase per segment, rotated so the cycle starts at the motion onset; the low-motion
   segment is the explicit **idle** state. Online decoding is a left-to-right HMM over template positions (each
   sub-phase expanded into a chain of positions, i.e. a segmental HMM with learned durations) plus an explicit
   **blocked** state for idles longer than the cycle allows (andon, break): the forward filter is causal.
2. **Line production state.** Per tick, trailing-window features: line indexing (skid motion, time since the last
   index in learned periods), share of line machines and of off-line cells in a non-idle sub-phase, people count,
   people moving, crowding, vehicles. Clustered (k-means++ on standardised features, k by silhouette) and the
   clusters are named by rules on their centroids (indexing -> running; stopped with cells working -> andon;
   stopped with a crowd -> break; stopped otherwise -> shift_change). The calendar is never an input.

Interface (what workstream M calls)::

    from intercycle.observer.phases import PhaseModel
    pm = PhaseModel.fit(observations, scn)            # iterable of mapping.Observation (training window, >= ~1 h)
    pm.update(obs)                                    # online: feed every tick (causal forward filter)
    pm.phase_bin(machine_id: str, t: float) -> int    # 0..n_bins(mid)-2 = discovered sub-phases in cycle order
                                                      # (the idle one is n_bins-2), n_bins(mid)-1 = blocked / stopped,
                                                      # -1 = unknown (no tick yet at t, or no model)
    pm.n_bins(machine_id: str) -> int                 # sub-phases + 1 blocked bin
    pm.bin_names(machine_id: str) -> list[str]        # "p0".."pK-1" ("idle" for the idle one), "blocked"
    pm.line_state(t: float) -> str                    # suggested name: "running" | "andon" | "break" | "shift_change"
                                                      # ("unknown" before data); an operator label overrides it
    pm.line_cluster(t: float) -> int                  # discovered cluster id (key a map on this), -1 unknown
    pm.line_catalogue() -> list[dict]                 # per cluster: "State A".., auto description, suggestions, label
    pm.set_line_label(cluster_id, name)               # operator names a cluster once (persists; carried over on
    pm.refit_line(T, X, F)                            #   refit by Hungarian matching of centroids)
    pm.save(path) / PhaseModel.load(path, scn)

`phase_bin` and `line_state` answer for the latest tick fed with t_cap <= t (they keep a short history, ~2 h, for
look-ups slightly in the past). `PhaseModel.fit_features` / `decode_features` work on precomputed feature arrays
(`Features.extract`), which is what the offline evaluation script uses.
"""
from __future__ import annotations

import math
import pickle
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from intercycle.observer.mapping import floor_ok_of
from intercycle.scenario import Scenario

PART, SKID, PERSON, VEHICLE = 2, 3, 0, 1          # mapping.py class codes (kept in sync by a test)
GRID = (4, 4)                                     # occupancy grid per machine zone (x, z)
MARGIN = 1.0                                      # zone margin (m), as CameraShim.machine_of
N_MFEAT = 1 + GRID[0] * GRID[1]                   # motion energy + occupancy grid
FLOW_NAMES = ("skid_moving", "skids", "people", "people_moving", "crowd", "vehicles", "vehicles_moving")
LINE_STATES = ("running", "andon", "break", "shift_change")
K_MIN, K_MAX = 4, 8                               # sub-phases per machine (incl. idle)
HIST_S = 7200.0


# ================================================================================================ features

@dataclass
class Layout:
    """What the observer knows about the site: machine zones, which are line stations, cameras (no truth)."""
    mids: list[str]
    rects: np.ndarray            # (M, 4) x0, z0, x1, z1
    on_line: np.ndarray          # (M,) bool
    cam_of_machine: np.ndarray   # (M,) camera covering the zone centre

    @classmethod
    def from_scenario(cls, scn: Scenario) -> "Layout":
        cams = np.array(scn.cameras, float)
        cen = np.array([[m.x, m.z] for m in scn.machines], float)
        return cls([m.id for m in scn.machines], np.array([m.rect for m in scn.machines], float),
                   np.array([m.on_line for m in scn.machines]),
                   np.hypot(cen[:, None, 0] - cams[None, :, 0], cen[:, None, 1] - cams[None, :, 1]).argmin(1))


class Features:
    """Per-tick features from one Observation: machine block (M, N_MFEAT) (NaN rows while the zone's camera is
    blacked out) and a flow vector (len(FLOW_NAMES),)."""

    def __init__(self, layout: Layout):
        self.lay = layout
        R = layout.rects
        self.x0, self.z0 = R[:, 0] - MARGIN, R[:, 1] - MARGIN
        self.wx = (R[:, 2] - R[:, 0] + 2 * MARGIN) / GRID[0]
        self.wz = (R[:, 3] - R[:, 1] + 2 * MARGIN) / GRID[1]

    def extract(self, obs) -> tuple[float, np.ndarray, np.ndarray]:
        M = len(self.lay.mids)
        X = np.zeros((M, N_MFEAT), np.float32)
        cls, mach = obs.cls, obs.machine
        sel = (cls == PART) & (mach >= 0)
        if sel.any():
            m = mach[sel].astype(np.int64)
            np.add.at(X[:, 0], m[obs.moving[sel]], 1.0)
            gx = np.clip(((obs.x[sel] - self.x0[m]) / self.wx[m]).astype(np.int64), 0, GRID[0] - 1)
            gz = np.clip(((obs.z[sel] - self.z0[m]) / self.wz[m]).astype(np.int64), 0, GRID[1] - 1)
            np.add.at(X, (m, 1 + gx * GRID[1] + gz), 1.0)
            if sel.sum() > 20 and obs.moving[sel].all():
                # frame-differencing glitch: every machine part on the floor flagged moving at once (robot bases never
                # move, so a real frame always has still parts). The shim does this when any body group changes size
                # (e.g. the lifter carriage appears); the frame carries no usable machine evidence -> missing.
                X[:] = np.nan
        X[~floor_ok_of(obs)[self.lay.cam_of_machine]] = np.nan     # M-W10: a frozen / blind camera's features are missing
        sk = cls == SKID
        pe = cls == PERSON
        ve = cls == VEHICLE
        crowd = 0.0
        if pe.any():                                       # most people in any 4 m cell
            key = np.floor(obs.x[pe] / 4.0).astype(np.int64) * 1000 + np.floor(obs.z[pe] / 4.0).astype(np.int64)
            crowd = float(np.unique(key, return_counts=True)[1].max())
        F = np.array([(sk & obs.moving).sum(), sk.sum(), pe.sum(), (pe & obs.moving).sum(), crowd, ve.sum(),
                      (ve & obs.moving).sum()], np.float32)
        return float(obs.t_cap), X, F


def extract_all(observations: Iterable, layout: Layout):
    fe = Features(layout)
    T, X, F = [], [], []
    for o in observations:
        t, x, f = fe.extract(o)
        T.append(t)
        X.append(x)
        F.append(f)
    return np.array(T), np.stack(X), np.stack(F)


# ================================================================================================ numeric helpers

def _ffill(Z: np.ndarray) -> np.ndarray:
    """Forward-fill NaN rows (then back-fill the head)."""
    Z = Z.copy()
    bad = np.isnan(Z[:, 0])
    if not bad.any():
        return Z
    idx = np.where(~bad, np.arange(len(Z)), 0)
    np.maximum.accumulate(idx, out=idx)
    Z = Z[idx]
    first = np.flatnonzero(~bad)
    if len(first):
        Z[: first[0]] = Z[first[0]]
    return Z


def acf_period(x: np.ndarray, dt: float, p_min: float = 5.0, p_max: float = 300.0) -> float:
    """Fundamental period of a real signal from its autocorrelation (shortest peak within 15 % of the best)."""
    x = np.nan_to_num(x - np.nanmean(x))
    n = len(x)
    maxlag = min(int(1.05 * p_max / dt), n // 2)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: maxlag + 1]
    r = r / max(r[0], 1e-12) * n / (n - np.arange(maxlag + 1))
    k0 = max(int(p_min / dt), 2)
    pk = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:])) + 1
    pk = pk[pk >= k0]
    if not len(pk):
        return math.nan
    k = int(pk[np.argmax(r[pk] >= 0.85 * r[pk].max())])
    a, b, d = r[k - 1], r[k], r[k + 1]
    den = a - 2 * b + d
    return (k + float(np.clip(0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0, -1, 1))) * dt


def sliding_sqdist(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """D[i] = sum_j ||Z[i + j] - Q[j]||^2 for every start i (non-normalised, multi-dimensional), via FFT."""
    from scipy.signal import fftconvolve
    m = len(Q)
    z2 = np.convolve((Z * Z).sum(1), np.ones(m), "valid")
    cr = sum(fftconvolve(Z[:, d], Q[::-1, d], "valid") for d in range(Z.shape[1]))
    return np.maximum(z2 - 2 * cr + (Q * Q).sum(), 0.0)


MOTIF_ENGINE = "auto"            # "auto" (stumpy if installed), "stumpy" or "numpy"; stumpy's numba JIT costs ~80-110 s once per process


def matrix_profile_motif(s: np.ndarray, m: int, valid: np.ndarray, max_periods: int = 60, engine: str | None = None) -> int:
    """Start of the best cycle motif of the 1-d series s (window m): argmin of the z-normalised matrix profile over
    `valid` starts, computed on the first `max_periods` periods. stumpy (BSD-3) when installed, else numpy."""
    s = np.asarray(s, float)[: max_periods * m]
    engine = engine or MOTIF_ENGINE
    mp = None
    if engine in ("auto", "stumpy"):
        try:
            import stumpy
            mp = stumpy.stump(s, m)[:, 0].astype(float)
        except ImportError:
            if engine == "stumpy":
                raise
    if mp is None:                                            # numpy: z-normalised distance from sampled starts
        n = len(s) - m + 1
        cand = np.arange(0, n, max(1, n // 400))
        mu = np.convolve(s, np.ones(m) / m, "valid")
        sd = np.sqrt(np.maximum(np.convolve(s * s, np.ones(m) / m, "valid") - mu * mu, 1e-12))
        mp = np.full(n, np.inf)
        for i in cand:
            q = s[i:i + m]
            qn = (q - q.mean()) / max(q.std(), 1e-6)
            c = np.correlate(s, qn, "valid") / m
            d = np.sqrt(np.maximum(2 * m * (1 - (c - 0) / sd), 0))
            d[max(0, i - m // 2): i + m // 2 + 1] = np.inf
            mp[i] = d.min()
    mp[~valid[: len(mp)]] = np.inf
    if not np.isfinite(mp).any():
        return int(np.argmax(valid)) if valid.any() else 0
    return int(np.argmin(mp))


def _dtw_core_py(C):
    m, L = C.shape
    D = np.full((m, L), np.inf)
    B = np.zeros((m, L), np.int8)
    D[0, 0] = C[0, 0]
    st = np.empty((3, m))
    for j in range(1, L):
        prev = D[:, j - 1]
        st[0] = prev                                # stay
        st[1, 0] = np.inf
        st[1, 1:] = prev[:-1]                       # advance 1
        st[2, :2] = np.inf
        st[2, 2:] = prev[:-2]                       # advance 2 (skip a position)
        k = st.argmin(0)
        D[:, j] = st[k, np.arange(m)] + C[:, j]
        B[:, j] = k
    return D, B


try:                                                # numba ships with stumpy; plain numpy otherwise
    import numba as _nb

    @_nb.njit(cache=False)
    def _dtw_core_nb(C):
        m, L = C.shape
        D = np.full((m, L), np.inf)
        B = np.zeros((m, L), np.int8)
        D[0, 0] = C[0, 0]
        for j in range(1, L):
            for i in range(m):
                best, k = D[i, j - 1], 0
                if i >= 1 and D[i - 1, j - 1] < best:
                    best, k = D[i - 1, j - 1], 1
                if i >= 2 and D[i - 2, j - 1] < best:
                    best, k = D[i - 2, j - 1], 2
                D[i, j] = best + C[i, j]
                B[i, j] = k
        return D, B
except Exception:                                   # pragma: no cover
    _dtw_core_nb = None


def _dtw_open_end(C: np.ndarray, lo: int, hi: int):
    """DTW on a local cost matrix C (m template positions x L frames), template start pinned to frame 0, template end
    free in frames [lo, hi). Steps per frame: repeat the position, advance one, or skip one.
    Returns (end frame j*, template position per frame 0..j*)."""
    m, L = C.shape
    D, B = (_dtw_core_nb if _dtw_core_nb is not None else _dtw_core_py)(np.ascontiguousarray(C, dtype=np.float64))
    hi = min(hi, L)
    lo = min(max(lo, 0), hi - 1)
    js = np.arange(lo, hi)
    j = int(js[np.argmin(D[m - 1, lo:hi] / (js + 1))])
    path = np.empty(j + 1, np.int64)
    i = m - 1
    for jj in range(j, -1, -1):
        path[jj] = i
        i = max(i - int(B[i, jj]), 0)
    return j, path


def optimal_partition(Y: np.ndarray, k: int, w: np.ndarray | None = None) -> tuple[float, list[int]]:
    """Exact change points of a (n, d) sequence into k contiguous segments minimising the summed squared deviation from
    the segment means (the `ruptures.Dynp` L2 cost). Returns (cost, segment starts)."""
    n = len(Y)
    S1 = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Y, 0)])
    S2 = np.r_[0.0, np.cumsum((Y * Y).sum(1))]

    def cost(a, b):                                # segment [a, b)
        s = S1[b] - S1[a]
        return S2[b] - S2[a] - (s * s).sum(-1) / np.maximum(b - a, 1)

    D = np.full((k + 1, n + 1), np.inf)
    A = np.zeros((k + 1, n + 1), np.int64)
    D[0, 0] = 0.0
    ends = np.arange(n + 1)
    for q in range(1, k + 1):
        for b in range(q, n + 1):
            a = np.arange(q - 1, b)
            v = D[q - 1, a] + cost(a, b)
            i = int(np.argmin(v))
            D[q, b], A[q, b] = v[i], a[i]
    starts, b = [], n
    for q in range(k, 0, -1):
        a = int(A[q, b])
        starts.append(a)
        b = a
    return float(D[k, n]), starts[::-1]


# ================================================================================================ one machine

@dataclass
class MachinePhases:
    """Discovered cycle structure of one machine and its online decoder (left-to-right HMM over template positions)."""
    mid: str
    hz: float
    period_s: float
    mu: np.ndarray               # (D,) feature standardisation (of log1p counts)
    sd: np.ndarray
    W: np.ndarray                # (D, k) PCA projection
    template: np.ndarray         # (m, k) cycle template, position 0 = motion onset
    seg_of_pos: np.ndarray       # (m,) sub-phase index per template position (last sub-phase = idle)
    var: np.ndarray              # (n_seg, k) emission variance per sub-phase
    n_seg: int
    idle_vec: np.ndarray         # (k,) look of the machine when stopped
    diag: dict = field(default_factory=dict)

    def project(self, X: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.maximum(X, 0)) - self.mu) / self.sd) @ self.W

    @property
    def n_bins(self) -> int:
        return self.n_seg + 1                       # + blocked

    @property
    def idle_seg(self) -> int:
        return self.n_seg - 1

    @property
    def blocked_bin(self) -> int:
        return self.n_seg

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, mid: str, T: np.ndarray, X: np.ndarray, hz: float, k_dims: int = 6, period_s: float | None = None,
            n_seg: int | None = None) -> "MachinePhases":
        diag: dict = {}
        valid = ~np.isnan(X[:, 0])
        Lg = np.log1p(np.maximum(np.nan_to_num(X), 0))
        mu = Lg[valid].mean(0)
        sd = np.maximum(Lg[valid].std(0), 0.05)
        Zs = (Lg - mu) / sd
        ev, V = np.linalg.eigh(np.cov(Zs[valid].T))
        o = np.argsort(ev)[::-1][:k_dims]
        W = V[:, o] * np.sign(V[np.abs(V[:, o]).argmax(0), o])
        diag["pca_var"] = (ev[o] / ev.sum()).round(3).tolist()
        Z = Zs @ W
        Z[~valid] = np.nan
        Zf = _ffill(Z)
        e = np.nan_to_num(X[:, 0])
        # ---- period: learn_cycles on the motion signal; ACF of the latent signal as fallback / cross-check
        p_acf = acf_period(Zf[:, 0], 1.0 / hz)
        p_lc = math.nan
        try:
            from intercycle.observer import learn_cycles as LC
            mod = LC.learn(T, (e > 0).astype(float), window=(T[0], T[0] + 5400.0), latency_s=0.0).model
            if mod.confidence >= LC.MIN_CONF:
                p_lc = float(mod.period_s)
        except Exception:
            pass
        diag.update(period_acf=round(p_acf, 2), period_learn_cycles=round(p_lc, 2))
        if period_s is None:
            period_s = p_lc if np.isfinite(p_lc) and (not np.isfinite(p_acf) or abs(p_lc / p_acf - 1) < 0.1) else p_acf
        m = int(round(period_s * hz))
        # ---- cycle motif: matrix profile of the first latent dimension over moving, fully observed windows
        # activity: frame-differencing motion, or a change of the look well above the frame-to-frame noise (a lifter
        # carriage moves vertically: it appears / disappears in plan view without "moving")
        chg = np.r_[0.0, ((Zf[1:] - Zf[:-1]) ** 2).sum(1)]
        chg_thr = 4.0 * float(np.median(chg[chg > 0])) if (chg > 0).any() else np.inf
        actv = _activity(Zf, e, chg_thr)
        roll = np.convolve(actv, np.ones(m) / m, "valid")
        okw = (roll > 0.05) & (np.convolve(valid, np.ones(m) / m, "valid") > 0.95)
        i0 = matrix_profile_motif(Zf[:, 0], m, okw)
        tpl = Zf[i0:i0 + m].copy()
        # ---- the look of the stopped machine: the densest configuration among the still frames (mode)
        busy5 = np.convolve(actv, np.ones(5) / 5, "same") >= 0.5                  # debounced: isolated dropout flips ignored
        long_still = [np.arange(a, b) for a, b in _runs_of(~busy5) if b - a > 1.5 * m]   # real stops (andon, break)
        long_still = np.concatenate(long_still) if long_still else np.zeros(0, np.int64)
        long_still = long_still[valid[long_still]]
        diag["stop_frames"] = int(len(long_still))
        idle_vec = _mode_look(Zf[long_still]) if len(long_still) >= 20 else _mode_look(Zf[valid & ~actv])
        stop_ref = idle_vec
        # ---- anchors (distance-profile minima) + DTW barycentre averaging; rotate once to the motion onset
        mot = np.convolve(actv, np.ones(m), "valid")                  # active frames per window start
        min_act = 0.5 * float(mot[i0])
        anchors = _anchors(sliding_sqdist(Zf, tpl), m, mot, min_act)
        et = np.zeros(m)
        aligned = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        for it in range(4):
            tpl, et, aligned = _dba(Zf, valid, e, anchors, tpl, m)
            off = _onset(tpl, et, stop_ref) if it < 3 else 0
            if off:                                                    # rotate to the motion onset, re-anchor
                tpl, et = np.roll(tpl, -off, 0), np.roll(et, -off)
                anchors = _anchors(sliding_sqdist(Zf, tpl), m, mot, min_act)
            if it >= 1 and not off:
                break
        diag["n_anchor_pass"] = it + 1
        if idle_vec is None:
            idle_vec = tpl[int(np.argmin(et))]
        # ---- sub-phases: idle = trailing stopped-like positions; optimal change points on the active part
        idle_pos = _idle_tail(tpl, et, stop_ref)
        act = _seg_space(tpl[:idle_pos], et[:idle_pos])
        costs, parts = {}, {}
        for kk in range(K_MIN - 1, K_MAX):
            if kk <= len(act):
                costs[kk + 1], parts[kk + 1] = optimal_partition(act, kk)
        if n_seg is None:
            n_seg = _choose_k(costs)
        starts = parts[n_seg] + [idle_pos]
        seg = np.zeros(m, np.int64)
        for s_i, a in enumerate(starts):
            seg[a:] = s_i
        diag.update(costs={int(k): round(v, 1) for k, v in costs.items()}, n_cycles=int(len(anchors)),
                    min_act=min_act, chg_thr=chg_thr,
                    seg_starts_s=[round(a / hz, 1) for a in starts], motif_start_s=round(float(T[i0] - T[0]), 1),
                    period_s=round(float(period_s), 3))
        # ---- emission variances per sub-phase from the aligned training frames
        fr, pos = aligned
        var = np.ones((n_seg, Z.shape[1]))
        for s_i in range(n_seg):
            sel = seg[pos] == s_i
            if sel.sum() > 10:
                var[s_i] = np.maximum((Zf[fr[sel]] - tpl[pos[sel]]).var(0), np.maximum(0.1 * ev[o], 0.1))   # floor: frame noise
        return cls(mid, hz, float(period_s), mu, sd, W, tpl, seg, var, int(n_seg), idle_vec, diag)

    # ---------------------------------------------------------------- offline labelling (DTW, non-causal)
    def label_offline(self, X: np.ndarray) -> np.ndarray:
        """Per-frame bin from anchor detection + open-end DTW of every cycle to the template; frames after the end
        of an aligned cycle (a stop) are `blocked_bin`. Non-causal: used to score the discovered segmentation."""
        valid = ~np.isnan(X[:, 0])
        Z = self.project(np.nan_to_num(X))
        Z[~valid] = np.nan
        Zf = _ffill(Z)
        m = len(self.template)
        mot = np.convolve(_activity(Zf, np.nan_to_num(X[:, 0]), self.diag["chg_thr"]), np.ones(m), "valid")
        anchors = _anchors(sliding_sqdist(Zf, self.template), m, mot, self.diag["min_act"])
        out = np.full(len(X), self.blocked_bin, np.int64)
        vpos = self.var[self.seg_of_pos]
        for a, b in zip(anchors, np.r_[anchors[1:], len(X)]):
            L = b - a
            if L < 0.5 * m:
                continue
            hi = min(L, int(1.15 * m) + 1)
            Cm = ((self.template[:, None, :] - Zf[None, a:a + hi, :]) ** 2 / vpos[:, None, :]).sum(-1)
            j, path = _dtw_open_end(Cm, int(0.85 * m), hi)
            out[a:a + j + 1] = self.seg_of_pos[path]
        return out


def _activity(Zf: np.ndarray, e: np.ndarray, chg_thr: float) -> np.ndarray:
    """Per frame: something moves (frame differencing) or the look changes beyond the frame-to-frame noise."""
    chg = np.r_[0.0, ((Zf[1:] - Zf[:-1]) ** 2).sum(1)]
    return (e > 0) | (chg > chg_thr)


def _mode_look(Y: np.ndarray, n_max: int = 1500, seed: int = 0) -> np.ndarray | None:
    """The densest configuration of a set of latent frames (mean of the neighbours of the point with the most
    neighbours within the 5 % distance quantile): the machine's resting look."""
    if len(Y) < 20:
        return None
    rng = np.random.default_rng(seed)
    S = Y[rng.choice(len(Y), min(n_max, len(Y)), replace=False)]
    q = (S * S).sum(1)
    D = np.maximum(q[:, None] + q[None, :] - 2 * S @ S.T, 0.0)
    r = np.quantile(D[np.triu_indices(len(S), 1)], 0.05)
    nb = D <= max(r, 1e-9)
    return S[nb[int(nb.sum(1).argmax())]].mean(0)


def _runs_of(z: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, np.asarray(z, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _anchors(Dp: np.ndarray, m: int, mot: np.ndarray | None = None, min_mot: float = 0.0) -> np.ndarray:
    """Cycle anchors: local minima of the distance profile, >= 0.6 m apart, that match the motif well and contain a
    cycle's worth of motion (`mot[i]` moving frames in the window, at least `min_mot`): a stop is never a cycle."""
    from scipy.ndimage import minimum_filter1d
    w = max(3, int(0.6 * m))
    if mot is not None:
        Dp = np.where(mot[: len(Dp)] >= min_mot, Dp, np.inf)
    loc = np.flatnonzero(np.isfinite(Dp) & (Dp == minimum_filter1d(Dp, 2 * w + 1, mode="nearest")))
    if not len(loc):
        return loc
    loc = loc[Dp[loc] < 3.0 * np.median(Dp[loc]) + 1e-9]
    keep = [int(loc[0])]
    for i in loc[1:]:
        if i - keep[-1] >= w:
            keep.append(int(i))
    return np.array(keep, np.int64)


def _dba(Zf, valid, e, anchors, tpl, m):
    """One DTW barycentre-averaging pass: align every anchor-to-anchor cycle to tpl, average the frames per position."""
    k = Zf.shape[1]
    acc, acc_e, cnt = np.zeros((m, k)), np.zeros(m), np.zeros(m)
    fr_all, pos_all = [], []
    var = np.maximum(np.nanvar(Zf, 0), 1e-3)
    for a, b in zip(anchors[:-1], anchors[1:]):
        L = b - a
        if L < 0.8 * m:
            continue
        hi = min(L, int(1.15 * m) + 1)
        Cm = ((tpl[:, None, :] - Zf[None, a:a + hi, :]) ** 2 / var).sum(-1)
        j, path = _dtw_open_end(Cm, int(0.85 * m), hi)
        fr = np.arange(a, a + j + 1)
        ok = valid[fr]
        np.add.at(acc, path[ok], Zf[fr[ok]])
        np.add.at(acc_e, path[ok], e[fr[ok]])
        np.add.at(cnt, path[ok], 1)
        fr_all.append(fr[ok])
        pos_all.append(path[ok])
    cat = lambda v: np.concatenate(v) if v else np.zeros(0, np.int64)
    fr, pos = cat(fr_all), cat(pos_all)
    # robust barycentre: per-position median (a mean lets outlier frames, e.g. partial frame-differencing glitches,
    # build a spurious feature that attracts more outliers on the next pass)
    new, et = tpl.copy(), np.zeros(m)
    if len(fr):
        o = np.argsort(pos, kind="stable")
        ps, fs = pos[o], fr[o]
        cut = np.searchsorted(ps, np.arange(m + 1))
        for p in range(m):
            a, b = cut[p], cut[p + 1]
            if b > a:
                new[p] = np.median(Zf[fs[a:b]], 0)
                et[p] = np.median(e[fs[a:b]])
    return new, et, (fr, pos)


def _smooth_c(x: np.ndarray, w: int) -> np.ndarray:
    """Circular moving average."""
    return np.convolve(np.r_[x[-w:], x, x[:w]], np.ones(w) / w, "same")[w:-w]


def _otsu(x: np.ndarray) -> float:
    """Otsu threshold of a 1-d sample (maximises the between-class variance)."""
    v = np.sort(x)
    best, thr = -1.0, float(v[0])
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best:
            best, thr = w, 0.5 * (v[i - 1] + v[i])
    return thr


def _stopped_like(tpl, et, stop_ref) -> np.ndarray:
    """Template positions that look like the stopped machine: low motion and, when long stops were observed, on the
    near side of the Otsu split of log-distance to the stopped look (`stop_ref`)."""
    lowe = _smooth_c(et, 3) <= 0.15 * max(float(et.max()), 1e-9)
    if stop_ref is None:
        return lowe
    ld = np.log10(((tpl - stop_ref) ** 2).sum(1) + 1e-3)
    return lowe & (ld <= _otsu(ld))


def _onset(tpl, et, stop_ref) -> int:
    """Rotation that puts position 0 at the motion onset: the end of the longest circular stopped-like run."""
    s = _stopped_like(tpl, et, stop_ref)
    m = len(s)
    if s.all() or not s.any():
        return int(np.argmax(np.diff(np.r_[et[-1], et])))
    best, best_end, run = 0, 0, 0
    for i, v in enumerate(np.r_[s, s]):
        run = run + 1 if v else 0
        if best < run <= m:
            best, best_end = run, i
    return int((best_end + 1) % m)


def _idle_tail(tpl, et, stop_ref) -> int:
    """First position of the trailing stopped-like run (the idle sub-phase), leaving room for the active part."""
    s = _stopped_like(tpl, et, stop_ref)
    m = len(s)
    i = m
    while i > 0 and s[i - 1]:
        i -= 1
    return int(min(max(i, K_MAX), m - 2))


def _choose_k(costs: dict[int, float]) -> int:
    """Smallest number of sub-phases after which one more segment gains < 10 % of the first gain (elbow)."""
    ks = sorted(costs)
    c = np.array([costs[k] for k in ks])
    g = -np.diff(c)
    if not len(g) or g[0] <= 0:
        return ks[0]
    for i in range(len(g)):
        if g[i] < 0.10 * g[0]:
            return ks[i]
    return ks[-1]


class HMMDecoder:
    """Causal forward filter of a left-to-right HMM over the template positions of one machine plus a blocked state.

    Position p: stay / advance one / skip one (0.15 / 0.70 / 0.15, one position per tick on average); the last
    position wraps to 0 (next cycle; an idle position may also jump to 0, the idle being cut short by jitter) or enters BLOCKED (the idle lasts longer than the cycle allows: andon, break,
    micro-stop); BLOCKED leaves to position 0 (motion onset). Emissions: diagonal Gaussians, mean = template position,
    variance = that sub-phase's residual variance; BLOCKED looks like the stopped machine. A tick with no observation
    (camera blacked out) only propagates. The output bin is the most probable sub-phase (sum over its positions)."""

    P_STAY, P_ADV, P_SKIP = 0.15, 0.70, 0.15
    P_WRAP, P_BLOCK, P_LEAVE, P_ONSET, P_EPS = 0.45, 0.40, 0.02, 0.01, 1e-4

    def __init__(self, mp: MachinePhases):
        self.mp = mp
        m = len(mp.template)
        self.m = m
        idle_look = np.median(mp.template[mp.seg_of_pos == mp.idle_seg], 0)     # blocked looks like the in-cycle idle
        self.mean = np.vstack([mp.template, idle_look[None, :]])
        v = np.vstack([mp.var[mp.seg_of_pos], mp.var[mp.idle_seg][None, :]])
        self.ivar = 1.0 / v
        self.lnorm = -0.5 * np.log(v).sum(1)
        self.seg = np.r_[mp.seg_of_pos, mp.blocked_bin]
        self.idle_mask = np.zeros(m, bool)
        self.idle_mask[:m - 1] = mp.seg_of_pos[:m - 1] == mp.idle_seg
        self.alpha = np.full(m + 1, 1.0 / (m + 1))
        self.bin = -1

    def run(self, Z: np.ndarray) -> np.ndarray:
        """Decode a (n, k) latent sequence (NaN rows = no observation) causally; returns the bin per tick."""
        pr = np.array([self.P_STAY, self.P_ADV, self.P_SKIP, self.P_WRAP, self.P_BLOCK, self.P_LEAVE, self.P_ONSET, self.P_EPS])
        out = np.empty(len(Z), np.int64)
        self.alpha = _hmm_run(self.alpha, np.ascontiguousarray(Z, dtype=np.float64), self.mean, self.ivar, self.lnorm,
                              self.idle_mask, self.seg, int(self.mp.n_bins), pr, out)
        if len(out):
            self.bin = int(out[-1])
        return out

    def step(self, z: np.ndarray | None) -> int:
        Z = np.full((1, self.mean.shape[1]), np.nan) if z is None else np.asarray(z, float).reshape(1, -1)
        return int(self.run(Z)[0])


def _hmm_run_py(alpha, Z, mean, ivar, lnorm, idle_mask, seg, n_bins, pr, out):
    """Forward filter over a block of ticks (see HMMDecoder). Returns the final alpha; writes the bin per tick to out."""
    P_STAY, P_ADV, P_SKIP, P_WRAP, P_BLOCK, P_LEAVE, P_ONSET, P_EPS = pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7]
    m = len(alpha) - 1
    k = Z.shape[1]
    a = alpha.copy()
    nw = np.zeros(m + 1)
    ll = np.zeros(m + 1)
    acc = np.zeros(n_bins)
    for t in range(Z.shape[0]):
        for i in range(m + 1):
            nw[i] = 0.0
        tot_pos = 0.0
        for i in range(m):
            v = a[i]
            nw[i] += P_STAY * v
            if i + 1 < m:
                nw[i + 1] += P_ADV * v
            if i + 2 < m:
                nw[i + 2] += P_SKIP * v
            if i < m - 1:
                tot_pos += v
        last = a[m - 1]
        early_sum = 0.0
        for i in range(m):
            if idle_mask[i]:
                e = P_ONSET * a[i]
                nw[i] -= e
                early_sum += e
        nw[0] += P_WRAP * last + P_SKIP * a[m - 2] + P_LEAVE * a[m] + early_sum
        nw[m] = P_BLOCK * last + (1.0 - P_LEAVE) * a[m] + P_EPS * tot_pos
        for i in range(m - 1):
            nw[i] *= 1.0 - P_EPS
        s = 0.0
        for i in range(m + 1):
            s += nw[i]
        if s > 0:
            for i in range(m + 1):
                nw[i] /= s
        else:
            for i in range(m + 1):
                nw[i] = 1.0 / (m + 1)
        ok = True
        for d in range(k):
            if not np.isfinite(Z[t, d]):
                ok = False
        if ok:
            mx = -np.inf
            for i in range(m + 1):
                q = 0.0
                for d in range(k):
                    r = Z[t, d] - mean[i, d]
                    q += r * r * ivar[i, d]
                ll[i] = lnorm[i] - 0.5 * q
                if ll[i] > mx:
                    mx = ll[i]
            s = 0.0
            for i in range(m + 1):
                nw[i] *= np.exp(ll[i] - mx)
                s += nw[i]
            if s > 0:
                for i in range(m + 1):
                    nw[i] /= s
            else:
                for i in range(m + 1):
                    nw[i] = 1.0 / (m + 1)
        for i in range(m + 1):
            a[i] = nw[i]
        for b in range(n_bins):
            acc[b] = 0.0
        for i in range(m + 1):
            acc[seg[i]] += a[i]
        best = 0
        for b in range(1, n_bins):
            if acc[b] > acc[best]:
                best = b
        out[t] = best
    return a


try:
    _hmm_run = _nb.njit(cache=False)(_hmm_run_py)
except Exception:                                   # pragma: no cover  (no numba: same code, interpreted)
    _hmm_run = _hmm_run_py


# ================================================================================================ line states

LINE_FEATS = ("index_overdue", "line_working", "line_blocked", "cells_working", "people_excess", "people_moving", "crowd_excess")


class _Win:
    """Trailing window with a running sum (mean in O(1))."""
    __slots__ = ("q", "n", "s")

    def __init__(self, n: int):
        self.q, self.n, self.s = deque(), max(int(n), 1), 0.0

    def add(self, v: float) -> float:
        self.q.append(v)
        self.s += v
        if len(self.q) > self.n:
            self.s -= self.q.popleft()
        return self.s / len(self.q)


class LineFeatureTracker:
    """Causal per-tick line features from the machine bins and the flow vector (trailing windows). People and crowd are
    taken relative to their own trailing 1 h median (sampled once a minute), so the shift's head count does not split
    'running' into day / night clusters."""

    def __init__(self, on_line: np.ndarray, idle_bins: np.ndarray, blocked_bins: np.ndarray, line_period_s: float, hz: float):
        self.on_line, self.idle, self.blk = on_line, idle_bins, blocked_bins
        self.P, self.hz = line_period_s, hz
        n30, n90 = int(30 * hz), int(90 * hz)
        self.lw, self.cw, self.pe, self.pm = _Win(n30), _Win(n90), _Win(n30), _Win(n30)
        self.cr = deque(maxlen=n30)
        self.last_index: float | None = None
        self.t0: float | None = None
        self.base = {"pe": deque(maxlen=60), "cr": deque(maxlen=60)}
        self.bmed = (0.0, 0.0)
        self.next_base: float | None = None

    def step(self, t: float, bins: np.ndarray, F: np.ndarray) -> np.ndarray:
        if self.t0 is None:
            self.t0 = t
        if F[0] > 0:
            self.last_index = t
        age = (t - (self.last_index if self.last_index is not None else self.t0)) / self.P
        known = bins >= 0
        working = known & (bins != self.idle) & (bins != self.blk)
        ol = self.on_line
        lw = self.lw.add(float(working[ol].mean()) if (ol & known).any() else 0.0)
        cw = self.cw.add(float(working[~ol].mean()) if (~ol & known).any() else 0.0)
        pe = self.pe.add(float(F[2]))
        pm = self.pm.add(float(F[3]) / max(float(F[2]), 1.0))
        self.cr.append(float(F[4]))
        cr = max(self.cr)
        blocked = float((bins[ol] == self.blk[ol]).mean())
        if self.next_base is None or t >= self.next_base:
            self.base["pe"].append(pe)
            self.base["cr"].append(cr)
            self.bmed = (float(np.median(self.base["pe"])), float(np.median(self.base["cr"])))
            self.next_base = t + 60.0
        return np.array([math.log1p(max(age - 1.15, 0.0)), lw, blocked, cw, pe - self.bmed[0], pm, cr - self.bmed[1]])


def _kmeans(Y: np.ndarray, k: int, rng: np.random.Generator, iters: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """k-means++ (Lloyd). Returns (centres, labels, inertia)."""
    C = [Y[rng.integers(len(Y))]]
    for _ in range(1, k):
        d = np.min(((Y[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(Y[rng.choice(len(Y), p=d / d.sum())] if d.sum() > 0 else Y[rng.integers(len(Y))])
    C = np.array(C)
    for _ in range(iters):
        lab = ((Y[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        Cn = np.array([Y[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    lab = ((Y[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    return C, lab, float(((Y - C[lab]) ** 2).sum())


def _silhouette(Y: np.ndarray, lab: np.ndarray) -> float:
    q = (Y * Y).sum(1)
    D = np.sqrt(np.maximum(q[:, None] + q[None, :] - 2 * Y @ Y.T, 0.0))
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    A = np.stack([D[:, lab == k].mean(1) for k in ks], 1)
    own = np.searchsorted(ks, lab)
    n_own = np.array([(lab == k).sum() for k in ks])[own]
    a = A[np.arange(len(Y)), own] * n_own / np.maximum(n_own - 1, 1)
    A[np.arange(len(Y)), own] = np.inf
    b = A.min(1)
    return float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12)))


@dataclass
class LineStates:
    """Production states discovered by clustering line features.

    The cluster id is the identity (what a map should be keyed on); a name never changes the clustering. Each cluster
    is shown as "State A", "State B", ... with an automatic description of its centroid until an operator names it
    once (`set_label`); labels persist with the model and carry over to a refit by Hungarian matching of centroids
    (`carry_labels`). `names` holds the *suggested* names from `_name_clusters`; those rules encode this simulator's
    design (andon = line overdue while the off-line cells work; break = a gathering of people, because the simulator
    spawns a break crowd) and are a suggestion only, as is a Gemma suggestion (`suggest`, marked unverified)."""
    mu: np.ndarray
    sd: np.ndarray
    centres: np.ndarray          # (k, f) standardised
    names: list[str]             # suggested (rule) name per cluster
    raw_centres: np.ndarray      # (k, f) in feature units (for the report)
    diag: dict = field(default_factory=dict)
    typical_s: list = field(default_factory=list)      # median episode length per cluster on the training window (s)
    share: list = field(default_factory=list)          # share of training ticks per cluster
    labels: dict = field(default_factory=dict)         # cluster id -> operator label (persists, carried over on refit)
    suggestions: dict = field(default_factory=dict)    # cluster id -> {"name", "confidence", "source", "verified": False}

    # ---------------------------------------------------------------- identity / naming
    def set_stats(self, cl: np.ndarray, hz: float) -> None:
        """Episode statistics of the training cluster sequence (for the descriptions)."""
        k = len(self.centres)
        cut = np.flatnonzero(cl[1:] != cl[:-1]) + 1
        st, en = np.r_[0, cut], np.r_[cut, len(cl)]
        self.typical_s = [float(np.median([(b - a) / hz for a, b in zip(st, en) if cl[a] == c] or [0.0])) for c in range(k)]
        self.share = (np.bincount(cl, minlength=k) / max(len(cl), 1)).tolist()

    def display(self, c: int) -> str:
        lab = getattr(self, "labels", {}).get(int(c))
        return lab if lab else f"State {chr(ord('A') + int(c))}"

    def describe(self, c: int) -> str:
        """Automatic description of a cluster's centroid (what an operator sees before naming it)."""
        r = dict(zip(LINE_FEATS, self.raw_centres[int(c)]))
        ty = (getattr(self, "typical_s", None) or [math.nan] * len(self.centres))[int(c)]
        sh = (getattr(self, "share", None) or [math.nan] * len(self.centres))[int(c)]
        dur = f"{ty / 60:.0f} min" if ty >= 120 else f"{ty:.0f} s"
        return (f"{100 * (1 - r['line_working']):.0f} % line stations idle ({100 * r['line_blocked']:.0f} % blocked), "
                f"{100 * (1 - r['cells_working']):.0f} % cells idle, line index {max(math.expm1(r['index_overdue']), 0.0):.1f} periods overdue, "
                f"people {r['people_excess']:+.1f} / crowding {r['crowd_excess']:+.1f} vs the 1 h baseline, "
                f"{100 * r['people_moving']:.0f} % of people walking; typical episode {dur}, {100 * sh:.0f} % of time")

    def set_label(self, c: int, name: str) -> None:
        if not hasattr(self, "labels") or self.labels is None:
            self.labels = {}
        self.labels[int(c)] = str(name)

    def suggest(self, c: int, name: str, confidence: float, source: str = "gemma") -> None:
        """Record a model suggestion for a cluster's name. It is never applied automatically (unverified)."""
        if not hasattr(self, "suggestions") or self.suggestions is None:
            self.suggestions = {}
        self.suggestions[int(c)] = {"name": str(name), "confidence": float(confidence), "source": source, "verified": False}

    def carry_labels(self, old: "LineStates", max_dist: float = 2.0) -> dict:
        """Copy operator labels (and suggestions) from a previous model: clusters are matched one-to-one by the Hungarian
        method on centroid distance (in this model's standardised units); pairs farther than `max_dist` are not matched.
        Returns {new id: old id}."""
        from scipy.optimize import linear_sum_assignment
        A = (old.raw_centres - self.mu) / self.sd
        D = np.sqrt(((self.centres[:, None, :] - A[None]) ** 2).sum(-1))
        r, c = linear_sum_assignment(D)
        match = {int(i): int(j) for i, j in zip(r, c) if D[i, j] <= max_dist}
        for i, j in match.items():
            if getattr(old, "labels", {}).get(j):
                self.set_label(i, old.labels[j])
            if getattr(old, "suggestions", {}).get(j):
                self.suggestions = getattr(self, "suggestions", None) or {}
                self.suggestions[i] = dict(old.suggestions[j])
        return match

    def catalogue(self) -> list[dict]:
        """One row per cluster: id, display name, description, suggested names (rule / model), operator label."""
        return [{"id": c, "display": self.display(c), "description": self.describe(c),
                 "suggested_rule": self.names[c], "suggested_model": getattr(self, "suggestions", {}).get(c),
                 "label": getattr(self, "labels", {}).get(c)} for c in range(len(self.centres))]

    @classmethod
    def fit(cls, Fl: np.ndarray, seed: int = 0, ks=(4, 5, 6), n_sub: int = 2000) -> "LineStates":
        mu, sd = Fl.mean(0), np.maximum(Fl.std(0), 1e-3)
        Y = (Fl - mu) / sd
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(Y), min(n_sub, len(Y)), replace=False)
        best = None
        sil = {}
        for k in ks:
            runs = [_kmeans(Y, k, np.random.default_rng([seed, k, r])) for r in range(4)]
            C, lab, inertia = min(runs, key=lambda x: x[2])
            sil[k] = _silhouette(Y[sub], lab[sub])
            if best is None or sil[k] > best[0]:
                best = (sil[k], k, C, lab)
        _, k, C, lab = best
        raw = C * sd + mu
        names = _name_clusters(raw, np.bincount(lab, minlength=k) / len(lab))
        return cls(mu, sd, C, names, raw, dict(silhouette={int(a): round(b, 3) for a, b in sil.items()}, k=int(k),
                                                 sizes=np.bincount(lab, minlength=k).tolist()))

    def cluster(self, f: np.ndarray) -> np.ndarray:
        Y = (np.atleast_2d(f) - self.mu) / self.sd
        return ((Y[:, None, :] - self.centres[None]) ** 2).sum(-1).argmin(1)

    def state(self, f: np.ndarray) -> np.ndarray:
        """State name per tick: the operator label of the tick's cluster when it is one of LINE_STATES, else the
        suggested (rule) name. Inside a planned-stop cluster (line and cells stopped) the tick's own flow features then
        suggest break (a gathering of people) vs shift_change (no gathering) - a rule that encodes the simulator."""
        f = np.atleast_2d(f)
        labs = getattr(self, "labels", {}) or {}
        nm = [labs[c] if labs.get(c) in LINE_STATES else self.names[c] for c in range(len(self.centres))]
        out = np.array(nm, dtype=object)[self.cluster(f)]
        stop = (out == "break") | (out == "shift_change")
        if stop.any():
            out[stop] = np.where(_gathering(f[stop]), "break", "shift_change")
        return out.astype(str)


PEOPLE_EXCESS, CROWD_EXCESS = 3.0, 1.0           # extra people / extra crowding over the trailing 1 h median = a gathering


def _gathering(f: np.ndarray) -> np.ndarray:
    i = {n: j for j, n in enumerate(LINE_FEATS)}
    f = np.atleast_2d(f)
    return (f[:, i["people_excess"]] >= PEOPLE_EXCESS) | (f[:, i["crowd_excess"]] >= CROWD_EXCESS)


def _name_clusters(raw: np.ndarray, share: np.ndarray) -> list[str]:
    """Name clusters from their centroids (no calendar): most line stations working (not blocked) -> running; line
    blocked with the off-line cells still working -> andon; line and cells stopped -> a planned-stop cluster, named
    break if its centroid shows a gathering, else shift_change (refined per tick in `LineStates.state`)."""
    i = {n: j for j, n in enumerate(LINE_FEATS)}
    run = raw[:, i["line_blocked"]] < 0.5
    if not run.any():
        run = raw[:, i["line_blocked"]] == raw[:, i["line_blocked"]].min()
    w = share[run] / share[run].sum()
    cells_run = float((raw[run, i["cells_working"]] * w).sum())
    gath = _gathering(raw)
    out = []
    for j in range(len(raw)):
        if run[j]:
            out.append("running")
        elif raw[j, i["cells_working"]] >= 0.5 * cells_run:
            out.append("andon")
        else:
            out.append("break" if gath[j] else "shift_change")
    return out


# ================================================================================================ the model M calls

class PhaseModel:
    """Discovered machine sub-phases + line production states, with an online (causal) decoder. See module docstring."""

    def __init__(self, layout: Layout, machines: dict[str, MachinePhases], line: LineStates | None, hz: float):
        self.layout, self.machines, self.line, self.hz = layout, machines, line, hz
        self._fe = Features(layout)
        self.reset()

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, observations: Iterable, scn: Scenario, hz: float = 2.0, **kw) -> "PhaseModel":
        lay = Layout.from_scenario(scn)
        T, X, F = extract_all(observations, lay)
        return cls.fit_features(T, X, F, lay, hz, **kw)

    @classmethod
    def fit_features(cls, T, X, F, layout: Layout, hz: float = 2.0, fit_line: bool = True, n_seg: dict | None = None) -> "PhaseModel":
        ms = {}
        for i, mid in enumerate(layout.mids):
            try:
                ms[mid] = MachinePhases.fit(mid, T, X[:, i], hz, n_seg=(n_seg or {}).get(mid))
            except Exception as ex:                       # a machine that never moved in the window: no model
                ms[mid] = None
                print(f"phases: no model for {mid}: {ex!r}")
        pm = cls(layout, ms, None, hz)
        if fit_line:
            bins, Fl = pm._decode_machines(T, X, F)
            pm.line = LineStates.fit(Fl)
            pm.line.set_stats(pm.line.cluster(Fl), hz)
            pm.reset()
        return pm

    def refit_line(self, T, X, F) -> dict:
        """Refit the line-state clusters on new data, carrying operator labels over (Hungarian on centroids)."""
        old = self.line
        _, Fl = self._decode_machines(T, X, F)
        self.line = LineStates.fit(Fl)
        self.line.set_stats(self.line.cluster(Fl), self.hz)
        match = self.line.carry_labels(old) if old is not None else {}
        self.reset()
        return match

    # ---------------------------------------------------------------- online
    def reset(self) -> None:
        self._dec = {mid: (mp and HMMDecoder(mp)) for mid, mp in self.machines.items()}
        lay = self.layout
        idle = np.array([mp.idle_seg if mp else -9 for mp in self.machines.values()])
        blk = np.array([mp.blocked_bin if mp else -9 for mp in self.machines.values()])
        per = [mp.period_s for mid, mp in self.machines.items() if mp and lay.on_line[lay.mids.index(mid)]]
        self._lft = LineFeatureTracker(lay.on_line, idle, blk, float(np.median(per)) if per else 60.0, self.hz)
        self._t: deque = deque()
        self._bins: deque = deque()
        self._state: deque = deque()
        self._cl: deque = deque()

    def update(self, obs) -> None:
        """Feed one Observation (in time order)."""
        self.update_features(*self._fe.extract(obs))

    def update_features(self, t: float, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        bins = np.full(len(self.layout.mids), -1, np.int64)
        for i, mid in enumerate(self.layout.mids):
            d = self._dec[mid]
            if d is not None:
                x = X[i]
                bins[i] = d.step(None if np.isnan(x[0]) else d.mp.project(x[None, :])[0])
        fl = self._lft.step(t, bins, F)
        st = str(self.line.state(fl)[0]) if self.line is not None else "unknown"
        self._t.append(t)
        self._bins.append(bins)
        self._state.append(st)
        self._cl.append(int(self.line.cluster(fl)[0]) if self.line is not None else -1)
        while self._t and self._t[0] < t - HIST_S:
            self._t.popleft()
            self._bins.popleft()
            self._state.popleft()
            self._cl.popleft()
        self._last_fl = fl
        return bins

    def _idx(self, t: float) -> int:
        if not self._t or t < self._t[0]:
            return -1
        return bisect_right(self._t, t) - 1

    def phase_bin(self, machine_id: str, t: float) -> int:
        k = self._idx(t)
        return -1 if k < 0 else int(self._bins[k][self.layout.mids.index(machine_id)])

    def n_bins(self, machine_id: str) -> int:
        mp = self.machines.get(machine_id)
        return mp.n_bins if mp else 0

    def bin_names(self, machine_id: str) -> list[str]:
        mp = self.machines.get(machine_id)
        if not mp:
            return []
        return [("idle" if s == mp.idle_seg else f"p{s}") for s in range(mp.n_seg)] + ["blocked"]

    def line_state(self, t: float) -> str:
        """Suggested state name (operator label if set). For a map, key on `line_cluster` instead."""
        k = self._idx(t)
        return "unknown" if k < 0 else self._state[k]

    def line_cluster(self, t: float) -> int:
        """Discovered line-state cluster id at t (the identity a map should be keyed on); -1 unknown."""
        k = self._idx(t)
        return -1 if k < 0 else int(self._cl[k])

    def n_line_clusters(self) -> int:
        return len(self.line.centres) if self.line is not None else 0

    def line_catalogue(self) -> list[dict]:
        """Per cluster: id, display ("State A" or the operator label), auto description, suggestions, label."""
        return self.line.catalogue() if self.line is not None else []

    def set_line_label(self, cluster_id: int, name: str) -> None:
        """Operator names a cluster once; persists with save() and carries over on refit_line()."""
        self.line.set_label(cluster_id, name)

    # ---------------------------------------------------------------- batch
    def _decode_machines(self, T, X, F):
        self.reset()
        n, M = len(T), len(self.layout.mids)
        bins = np.full((n, M), -1, np.int64)
        for i, mid in enumerate(self.layout.mids):
            d = self._dec[mid]
            if d is not None:
                Z = d.mp.project(np.nan_to_num(X[:, i]))
                Z[np.isnan(X[:, i, 0])] = np.nan
                bins[:, i] = d.run(Z)
        Fl = np.array([self._lft.step(T[k], bins[k], F[k]) for k in range(n)])
        return bins, Fl

    def decode_features(self, T, X, F):
        """Batch causal decode: (bins (n, M), line state names (n,), line features (n, f), cluster ids (n,))."""
        bins, Fl = self._decode_machines(T, X, F)
        cl = self.line.cluster(Fl) if self.line is not None else np.zeros(len(T), np.int64)
        names = self.line.state(Fl) if self.line is not None else np.full(len(T), "unknown")
        self.reset()
        return bins, names, Fl, cl

    # ---------------------------------------------------------------- persistence
    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"layout": self.layout, "machines": self.machines, "line": self.line, "hz": self.hz}, f)

    @classmethod
    def load(cls, path, scn: Scenario | None = None) -> "PhaseModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        lay = Layout.from_scenario(scn) if scn is not None else d["layout"]
        return cls(lay, d["machines"], d["line"], d["hz"])


SEG_MODE = "whiten+e3"            # change-point space: "pca" | "whiten" (per-dim unit variance) | "whiten+e<w>" (+ motion energy, weight w)
# Motion energy is the channel that separates still sub-phases (weld dwell, tip dress) from moving ones (swing, return);
# the 4 x 4 occupancy grid is coarse (dresser and robot base share a cell). w = 3 chosen on the dev window (seed 0).


def _seg_space(tpl: np.ndarray, et: np.ndarray) -> np.ndarray:
    """Template frames in the space the change-point cost is computed in."""
    if SEG_MODE == "pca":
        return tpl
    Y = (tpl - tpl.mean(0)) / np.maximum(tpl.std(0), 1e-6)
    if SEG_MODE.startswith("whiten+e"):
        e = np.log1p(np.maximum(et, 0))
        w = float(SEG_MODE[8:] or 1.0)
        Y = np.c_[Y, w * (e - e.mean()) / max(e.std(), 1e-6)]
    return Y
