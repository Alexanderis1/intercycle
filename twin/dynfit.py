"""DYN v1 building blocks (lane S3a, PLAN_TWIN §5 "DYN/DAYGEN (resampling is the default)").

Pure functions over arrays and the PERC tables; no truth is opened or written here (the day generator,
`twin/daygen.py`, is the producer). Three users:

* `scripts/dyn_fit.py` fits `twin/dynamics.json`, the press tables (`twin/dyn/press_v1.json`) and the path library
  (`$INTERCYCLE_DATA/dyn/path_library_v1.npz`) from TRAIN clips only;
* `twin/daygen.py` resamples the library and the press tables into a day;
* the DYN validation (`twin/daygen.py --validate`, `scripts/dyn_validate.py`) runs twin output through the
  validation camera (`PixelCamera`) and applies the SAME track statistic (`observed_pieces` + `track_speed`) and the
  SAME stroke-interval rule (`stroke_intervals`) to twin, train and dev.

Floor frame: (x, z) metres of `twin/site.yaml` (`project_tracks` writes them as columns x, y; y is z here).
numpy only (Isaac-side rule, tests/test_isaac_deps.py); the DataFrame handling lives in scripts/dyn_fit.py.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Division-model lens (same math as intercycle.framing.undistort / distort, which this Isaac-side module cannot
# import: intercycle/framing.py has lazy scipy / cv2 imports; tests/test_dynfit.py checks the two agree).
CENTER = ((1920 - 1) / 2.0, (1080 - 1) / 2.0)


def undistort(pts: np.ndarray, k1: float, center: Sequence[float] = CENTER) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    c = np.asarray(center, float)
    d = p - c
    return c + d / (1.0 + k1 * (d**2).sum(1, keepdims=True))


def distort(pts: np.ndarray, k1: float, center: Sequence[float] = CENTER) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    c = np.asarray(center, float)
    d = p - c
    r_u = np.sqrt((d**2).sum(1))
    out = p.copy()
    nz = r_u > 1e-12
    if abs(k1) < 1e-18:
        return out
    disc = 1.0 - 4.0 * k1 * r_u[nz] ** 2
    if np.any(disc < 0):
        raise ValueError("point outside the valid range of the division model")
    r_d = 2.0 * r_u[nz] / (1.0 + np.sqrt(disc))
    out[nz] = c + d[nz] * (r_d / r_u[nz])[:, None]
    return out

SMOOTH_N = 7                    # rolling-median window (samples; 0.58 s at 12 fps) applied before any statistic
DECIM_S = 0.25                  # library segments keep one point per >= 0.25 s (the engine runs at 2 Hz)
MIN_SEG_S = 1.0                 # shortest library segment
MIN_STAT_S = 2.0                # shortest track piece that enters the speed statistic
SPEED_LAG_S = 1.0               # speed = floor displacement over 1 s of the smoothed track
MAX_GAP_S = 1.0                 # a track is cut where it has no detection for longer than this
JUMP_MPS = {"person": 6.0, "forklift": 8.0}    # a smoothed step faster than this is an ID switch: cut there
GROUP_OVERLAP_S = 1.0           # two person pieces form a group when they overlap >= 1 s ...
GROUP_DIST_M = 1.5              # ... at a median distance <= 1.5 m over the overlap
RUN_MIN_STROKES = 2             # P4 rule: a press runs in a clip with >= 2 strokes
JUDGE_S = 4.0                   # a clip judges P(running) when the press ROI is observed (not MISSING) >= 4 s
MODE_LO, MODE_HI = 0.5, 1.5     # main-mode intervals: [0.5, 1.5] x the median interval; longer = a pause


# ------------------------------------------------------------------------------------------------ geometry

def inside(x: np.ndarray, z: np.ndarray, poly: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised even-odd test (same rule as `twin.schemas.site.point_in_polygon`)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    out = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if z0 == z1:
            continue
        cross = (z0 > z) != (z1 > z)
        xc = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
        out ^= cross & (x < xc)
    return out & np.isfinite(x) & np.isfinite(z)


def rolling_median(a: np.ndarray, n: int = SMOOTH_N) -> np.ndarray:
    """Centred rolling median by index; the window shrinks at the ends (numpy only: this module runs Isaac-side)."""
    a = np.asarray(a, float)
    if len(a) == 0:
        return a.copy()
    h = n // 2
    pad = np.r_[np.full(h, np.nan), a, np.full(n - 1 - h, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    with np.errstate(all="ignore"):
        return np.nanmedian(win, axis=1)


# ------------------------------------------------------------------------------------------------ tracks

def observed_pieces(t: np.ndarray, x: np.ndarray, z: np.ndarray, poly, cls: str,
                    min_s: float = MIN_STAT_S) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One raw floor track -> its smoothed pieces on the zone's visible floor.

    Rows are sorted by t, non-finite rows dropped, x / z smoothed (`rolling_median`), then the track is cut where it
    leaves the floor polygon, where it has no detection for > MAX_GAP_S, and where a smoothed step is faster than
    JUMP_MPS[cls] (an ID switch). Pieces shorter than ``min_s`` are dropped. Used on train, dev and twin tracks alike.
    """
    t = np.asarray(t, float)
    o = np.argsort(t, kind="stable")
    t, x, z = t[o], np.asarray(x, float)[o], np.asarray(z, float)[o]
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(z)
    t, x, z = t[ok], x[ok], z[ok]
    if len(t) < 2:
        return []
    xs, zs = rolling_median(x), rolling_median(z)
    ins = inside(xs, zs, poly)
    dt = np.diff(t)
    step = np.hypot(np.diff(xs), np.diff(zs)) / np.maximum(dt, 1e-6)
    cut = (dt > MAX_GAP_S) | (step > JUMP_MPS.get(cls, 6.0))
    out = []
    start = None
    for i in range(len(t)):
        if start is not None and (not ins[i] or cut[i - 1]):
            out.append((start, i))
            start = None
        if start is None and ins[i]:
            start = i
    if start is not None:
        out.append((start, len(t)))
    res = []
    for a, b in out:
        if b - a >= 2 and t[b - 1] - t[a] >= min_s:
            res.append((t[a:b], xs[a:b], zs[a:b]))
    return res


def track_speed(t: np.ndarray, x: np.ndarray, z: np.ndarray, lag: float = SPEED_LAG_S) -> float:
    """Median over the piece of |p(t_j) - p(t_i)| / (t_j - t_i), t_j the first sample >= t_i + lag. NaN if the piece
    is shorter than ``lag``."""
    t = np.asarray(t, float)
    j = np.searchsorted(t, t + lag - 1e-9)
    ok = j < len(t)
    if not ok.any():
        return math.nan
    i = np.nonzero(ok)[0]
    j = j[ok]
    d = np.hypot(x[j] - x[i], z[j] - z[i]) / (t[j] - t[i])
    return float(np.median(d))


def decimate(t: np.ndarray, step: float = DECIM_S) -> np.ndarray:
    """Indices that keep one sample per >= ``step`` seconds, always the first and the last."""
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= step - 1e-9:
            keep.append(i)
    if keep[-1] != len(t) - 1:
        if len(keep) > 1 and t[-1] - t[keep[-1]] < 1e-6:
            keep[-1] = len(t) - 1
        else:
            keep.append(len(t) - 1)
    return np.asarray(keep, int)


def link_groups(pieces: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Group label per piece: pieces overlapping >= GROUP_OVERLAP_S at a median distance <= GROUP_DIST_M are
    linked; groups are the connected components (people walking or working together)."""
    n = len(pieces)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        ta, xa, za = pieces[a]
        for b in range(a + 1, n):
            tb, xb, zb = pieces[b]
            lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
            if hi - lo < GROUP_OVERLAP_S:
                continue
            g = np.linspace(lo, hi, max(2, int((hi - lo) * 4) + 1))
            d = np.hypot(np.interp(g, ta, xa) - np.interp(g, tb, xb), np.interp(g, ta, za) - np.interp(g, tb, zb))
            if float(np.median(d)) <= GROUP_DIST_M:
                parent[find(a)] = find(b)
    roots = [find(i) for i in range(n)]
    lab = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.asarray([lab[r] for r in roots], int)


# ------------------------------------------------------------------------------------------------ presses

def stroke_intervals(t: np.ndarray, stroke: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-clip inter-stroke intervals of one press clip.

    Complete: consecutive strokes with no MISSING frame between them. Right-censored (Kaplan-Meier): the time from a
    stroke to the first MISSING frame after it when that comes before the next stroke, and from the last stroke to
    the clip end. Returns (complete, censored) in seconds.
    """
    t = np.asarray(t, float)
    stroke = np.asarray(stroke, bool)
    missing = np.asarray(missing, bool)
    st = np.nonzero(stroke)[0]
    comp, cens = [], []
    miss_idx = np.nonzero(missing)[0]
    for k, i in enumerate(st):
        nxt = st[k + 1] if k + 1 < len(st) else None
        m = miss_idx[np.searchsorted(miss_idx, i + 1):]
        first_miss = int(m[0]) if len(m) else None
        if nxt is not None and (first_miss is None or first_miss > nxt):
            comp.append(t[nxt] - t[i])
        elif first_miss is not None and (nxt is None or first_miss < nxt):
            cens.append(t[first_miss] - t[i])
        else:
            cens.append(t[-1] - t[i])
    return np.asarray(comp, float), np.asarray(cens, float)


def kaplan_meier(complete: np.ndarray, censored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product-limit estimate. Returns (event times, S(t) just after each event time)."""
    d = np.r_[np.asarray(complete, float), np.asarray(censored, float)]
    e = np.r_[np.ones(len(complete), bool), np.zeros(len(censored), bool)]
    if not e.any():
        return np.zeros(0), np.zeros(0)
    o = np.lexsort((~e, d))                      # at ties, events before censorings
    d, e = d[o], e[o]
    times, surv = [], []
    s = 1.0
    n_at = len(d)
    i = 0
    while i < len(d):
        j = i
        while j < len(d) and d[j] == d[i]:
            j += 1
        ev = int(e[i:j].sum())
        if ev:
            s *= 1.0 - ev / n_at
            times.append(float(d[i]))
            surv.append(s)
        n_at -= j - i
        i = j
    return np.asarray(times), np.asarray(surv)


def km_survival_at(times: np.ndarray, surv: np.ndarray, x: float) -> float:
    k = np.searchsorted(times, x, side="right")
    return 1.0 if k == 0 else float(surv[k - 1])


def km_sample_tail(rng: np.random.Generator, times: np.ndarray, surv: np.ndarray, lo: float) -> float:
    """One draw from the KM distribution conditional on T > lo, restricted to the observed event support (the mass
    the KM leaves beyond its last event, i.e. pauses longer than any clip shows, is the long-stop process's)."""
    s_lo = km_survival_at(times, surv, lo)
    m = times > lo
    tt, ss = times[m], surv[m]
    if len(tt) == 0:
        return float(lo)
    s_end = float(ss[-1])
    u = s_lo - rng.random() * (s_lo - s_end)
    k = int(np.searchsorted(-ss, -u, side="left"))
    return float(tt[min(k, len(tt) - 1)])


def beta_posterior(k: int, n: int, a0: float = 1.0, b0: float = 1.0) -> tuple[float, float]:
    """Beta(a0 + k, b0 + n - k): uniform prior on P(press running) updated by k running clips out of n judged."""
    return a0 + k, b0 + (n - k)


def impulse_fwhm(t: np.ndarray, flow_v: np.ndarray, stroke: np.ndarray, missing: np.ndarray,
                 half_s: float = 0.6) -> list[np.ndarray]:
    """|flow_v| profiles (on the clip's frame grid) in +-half_s around each stroke with no MISSING frame in the
    window; the caller averages them and takes the full width at half maximum."""
    t = np.asarray(t, float)
    f = np.abs(np.asarray(flow_v, float))
    if len(t) < 3:
        return []
    dt = float(np.median(np.diff(t)))
    h = int(round(half_s / dt))
    out = []
    for i in np.nonzero(np.asarray(stroke, bool))[0]:
        a, b = i - h, i + h + 1
        if a < 0 or b > len(t) or np.asarray(missing, bool)[a:b].any() or not np.isfinite(f[a:b]).all():
            continue
        out.append(f[a:b])
    return out


def fwhm_s(profile: np.ndarray, dt: float) -> float:
    p = np.asarray(profile, float)
    base = float(np.min(p))
    peak = float(np.max(p))
    if peak <= base:
        return math.nan
    above = np.nonzero(p >= base + 0.5 * (peak - base))[0]
    return float((above[-1] - above[0] + 1) * dt)


# ------------------------------------------------------------------------------------------------ camera (validation)

@dataclass
class PixelCamera:
    """Validation camera: floor -> pixels -> real-camera noise -> floor through the same CAL chain as the real tracks.

    floor (x, z) -> undistorted px by H^-1 -> distorted px (division model k1) -> foot-point jitter N(0, sigma_u/v)
    and a two-state (seen / missed) Markov chain per track at ``fps`` (p_miss_start = P(seen -> missed),
    p_miss_end = P(missed -> seen)) -> `project_tracks`-equivalent back-projection (undistort k1, H). Detections that
    fall outside the image are missed. All parameters are fitted on TRAIN tracks (scripts/dyn_fit.py).
    This is NOT stat_v1 (twin/camera.py, lane S3b); it stands in for it in the DYN validation only.
    """
    H: np.ndarray
    k1: float
    sigma_u: float
    sigma_v: float
    p_miss_start: float
    p_miss_end: float
    fps: float = 12.0
    size: tuple[int, int] = (1920, 1080)

    def to_px(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        q = np.c_[x, z, np.ones(len(x))] @ np.linalg.inv(np.asarray(self.H, float)).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        out = np.full_like(uv, np.nan)
        ok = np.isfinite(uv).all(1)
        if ok.any():
            try:
                out[ok] = distort(uv[ok], self.k1)
            except ValueError:
                for i in np.nonzero(ok)[0]:
                    try:
                        out[i] = distort(uv[i:i + 1], self.k1)[0]
                    except ValueError:
                        pass
        return out

    def to_floor(self, uv: np.ndarray) -> np.ndarray:
        und = undistort(uv, self.k1) if self.k1 != 0.0 else np.asarray(uv, float)
        p = np.c_[und, np.ones(len(und))] @ np.asarray(self.H, float).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, :2] / p[:, 2:3]

    def observe(self, rng: np.random.Generator, t: np.ndarray, x: np.ndarray, z: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One true track sampled at the camera's frame times -> the detected rows (t, x, z) of its observed track."""
        n = len(t)
        if n == 0:
            return t, x, z
        uv = self.to_px(np.asarray(x, float), np.asarray(z, float))
        uv = uv + np.c_[rng.normal(0.0, self.sigma_u, n), rng.normal(0.0, self.sigma_v, n)]
        seen = np.empty(n, bool)
        s = rng.random() >= self.p_miss_start / max(self.p_miss_start + self.p_miss_end, 1e-12)
        u = rng.random(n)
        for i in range(n):
            seen[i] = s
            s = (u[i] >= self.p_miss_start) if s else (u[i] < self.p_miss_end)
        W, Hh = self.size
        ok = seen & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
        fl = self.to_floor(uv[ok])
        return np.asarray(t, float)[ok], fl[:, 0], fl[:, 1]


def fit_pixel_jitter(tracks: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, int]:
    """Robust per-axis sd (px, 1.4826 x MAD) of raw foot points around their rolling median (SMOOTH_N). ``tracks``:
    one time-sorted (u, v) pair per track (track ids are only unique within a clip)."""
    du, dv = [], []
    for u, v in tracks:
        if len(u) < SMOOTH_N:
            continue
        du.append(np.asarray(u, float) - rolling_median(u))
        dv.append(np.asarray(v, float) - rolling_median(v))
    if not du:
        return math.nan, math.nan, 0
    a, b = np.concatenate(du), np.concatenate(dv)

    def mad(r: np.ndarray) -> float:
        return float(1.4826 * np.median(np.abs(r - np.median(r))))

    return mad(a), mad(b), int(len(a))


def fit_miss_chain(n_obs_frames: int, n_gap_runs: int, n_gap_frames: int) -> tuple[float, float]:
    """(P(seen -> missed), P(missed -> seen)) per sampled frame from the within-track gap runs (`track_gaps`)."""
    if n_obs_frames <= 0:
        return math.nan, math.nan
    p_start = n_gap_runs / n_obs_frames
    p_end = n_gap_runs / n_gap_frames if n_gap_frames else 1.0
    return float(p_start), float(min(p_end, 1.0))


# ------------------------------------------------------------------------------------------------ statistics

def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D (sup |F_a - F_b|)."""
    a = np.sort(np.asarray(a, float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, float)[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return math.nan
    g = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, g, side="right") / len(a)
                               - np.searchsorted(b, g, side="right") / len(b))))
