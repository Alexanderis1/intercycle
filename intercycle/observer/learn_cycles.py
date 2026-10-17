"""Learn machine cycles from a camera's binary motion signal (t, y). Never imports ground truth (tested).

Pipeline: a Schmitt trigger plus min-duration debounce cleans the signal; its autocorrelation picks the
fundamental period (a duty-cycle wave has strong harmonics); Lomb-Scargle (5-300 s) at that period (or
its harmonics) gives the false-alarm probability. The idle pulses are then tracked cycle by cycle (rise
+ busy time or fall + period -> expected next fall), which rejects occlusion dropouts and isolates long
idles (andon, breaks). Period, busy fraction, jitter (edge noise removed), drift and stop rate are read
off the tracked transitions; confidence is the weakest of FAP, fold contrast, jitter, cycle count,
coverage and period consistency. Line stations can be pooled (learn_line / line_forecast).

Conventions: y=1 ACTIVE (motion), y=0 IDLE. "fall" = ACTIVE->IDLE (idle window starts),
"rise" = IDLE->ACTIVE (next cycle starts). Edge times are corrected for the camera latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import LombScargle

from intercycle.types import MachineModel, MachineNow, State

LATENCY_S = 0.2        # camera pipeline latency (device calibration constant)
P_MIN, P_MAX = 5.0, 300.0
DEBOUNCE_S = 4.0       # runs shorter than this are dropouts / flips; shortest detectable idle is ~5 s
FLIP_S = 2.0           # active runs shorter than this are false flips (flips are isolated, dropouts come in bursts)
FAP_OK = 1e-3          # Lomb-Scargle false-alarm probability below which a period counts as real
MIN_CONF = 4           # robot ignores models below this


@dataclass(frozen=True)
class Edges:
    """Validated transitions (sorted, latency-corrected, seconds on the caller's time base)."""
    fall: np.ndarray   # ACTIVE->IDLE: idle window starts
    rise: np.ndarray   # IDLE->ACTIVE: next cycle starts


@dataclass
class Learned:
    model: MachineModel
    edges: Edges
    idle_s: float                      # typical idle window length
    act_s: float                       # typical active length
    diag: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Forecast:
    state: State
    next_fall: float       # absolute time the next idle window starts (after the current idle if idle now)
    idle_left: float       # remaining idle length if IDLE else 0
    idle_len: float        # predicted idle window length
    period_s: float
    ttl_s: float
    confidence: int

    def fall_after(self, t: float) -> float:
        """First predicted idle start >= t (extrapolating by whole periods)."""
        if t <= self.next_fall:
            return self.next_fall
        return self.next_fall + np.ceil((t - self.next_fall) / self.period_s) * self.period_s


# --------------------------------------------------------------------------- signal handling

def _prep(t, y, window=None) -> tuple[np.ndarray, np.ndarray, float]:
    t, y = np.asarray(t, float), np.asarray(y, float)
    if window is not None:
        m = (t >= window[0]) & (t < window[1])
        t, y = t[m], y[m]
    if len(t) < 3:
        return t, y, 0.5
    dt = float(np.median(np.diff(t)))
    if np.abs(np.diff(t) - dt).max() > 0.05 * dt:  # irregular: zero-order hold onto a uniform grid
        g = t[0] + dt * np.arange(int((t[-1] - t[0]) / dt) + 1)
        y = y[np.clip(np.searchsorted(t, g, side="right") - 1, 0, len(t) - 1)]
        t = g
    return t, (y > 0.5).astype(float), dt


def _lomb_scargle(t: np.ndarray, y: np.ndarray, p0: float | None = None) -> tuple[float, float]:
    """Lomb-Scargle over 5..300 s: (period, false-alarm probability (Baluev)). With p0, the peak is taken
    at p0 or its first harmonics (+-4 %), i.e. the evidence that the candidate period is real."""
    t0 = t - t[0]
    lo, hi = 1.0 / P_MAX, 1.0 / P_MIN
    ls = LombScargle(t0, y - y.mean())
    f, p = ls.autopower(minimum_frequency=lo, maximum_frequency=hi, samples_per_peak=10)
    ok = np.ones(len(f), bool)
    if p0 is not None:
        ok = np.any([np.abs(f * p0 / m - 1) < 0.04 for m in (1, 2, 3)], axis=0)
        if not ok.any():
            ok = np.ones(len(f), bool)
    i = int(np.flatnonzero(ok)[np.argmax(p[ok])])
    fap = float(ls.false_alarm_probability(p[i], minimum_frequency=lo, maximum_frequency=hi))
    return 1.0 / float(f[i]), max(fap, 1e-300)


def _acf_period(y: np.ndarray, dt: float) -> tuple[float, float]:
    """Fundamental period from the autocorrelation: the shortest local maximum within 15 % of the best.
    (A duty-cycle wave has strong harmonics, so the strongest spectral line may not be the fundamental.)"""
    n = len(y)
    x = y - y.mean()
    maxlag = min(int(1.05 * P_MAX / dt), n // 2)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: maxlag + 1]
    r = r / max(r[0], 1e-12) * n / (n - np.arange(maxlag + 1))
    k0 = max(int(P_MIN / dt), 2)
    pk = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:])) + 1
    pk = pk[pk >= k0]
    if len(pk) == 0:
        return 0.0, 0.0
    best = r[pk].max()
    k = int(pk[np.argmax(r[pk] >= 0.85 * best)])
    a, b, d = r[k - 1], r[k], r[k + 1]
    den = a - 2 * b + d
    off = 0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0
    return (k + float(np.clip(off, -1, 1))) * dt, float(b)


def _runs(z: np.ndarray):
    chg = np.flatnonzero(np.diff(z)) + 1
    st = np.r_[0, chg]
    return st, np.r_[chg, len(z)], z[st]


def _schmitt(y: np.ndarray, w: int = 3, hi: float = 0.6, lo: float = 0.4) -> np.ndarray:
    """Schmitt trigger on a w-sample moving mean (edge-padded): isolated false flips vanish."""
    s = np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    z = np.empty(len(y), np.int8)
    st = int(s[0] >= 0.5)
    for i, v in enumerate(s):
        if st == 0 and v >= hi:
            st = 1
        elif st == 1 and v <= lo:
            st = 0
        z[i] = st
    return z


def _debounce(z: np.ndarray, m_idle: int, m_act: int) -> np.ndarray:
    """Merge interior runs shorter than their minimum into the neighbours, most-undersized first.
    Idle runs need m_idle samples (occlusion bursts read as short idles), active runs m_act (false flips)."""
    z = z.copy()
    mins = np.array([m_idle, m_act])
    while True:
        st, en, v = _runs(z)
        r = (en - st)[1:-1] / mins[v[1:-1]]
        if r.size == 0 or r.min() >= 1:
            return z
        k = int(np.argmin(r)) + 1
        z[st[k]:en[k]] = 1 - v[k]


def _clean(y: np.ndarray, dt: float, debounce_s: float = DEBOUNCE_S) -> np.ndarray:
    """Schmitt trigger + debounce: the signal with flips and occlusion bursts removed."""
    return _debounce(_schmitt(y), max(int(round(debounce_s / dt)), 2), max(int(round(FLIP_S / dt)), 2))


def _tail(y: np.ndarray) -> tuple[int, int]:
    """Raw state at the end of the signal: (value, number of trailing samples), isolated flips suppressed."""
    z = _schmitt(y[-40:])
    st, en, v = _runs(z)
    return int(v[-1]), int(en[-1] - st[-1])


# --------------------------------------------------------------------------- transition tracking

def _chain(ia: np.ndarray, ib: np.ndarray, dur: np.ndarray, act: float, p0: float, w: float, t1: float,
           lo: int, hi: int) -> list[int]:
    """Best pulse chain among pulses lo..hi-1. The next fall is expected `act` after the previous rise
    (survives a stop) or `p0` after the previous fall (survives a rise bent by an occlusion burst), each
    within +-w; the longest pulse in either window wins (dropouts are short). -1 marks a missed cycle."""
    ss = lambda x, side="left": int(np.searchsorted(ia, x, side))
    best: list[int] = []
    for s in range(lo, hi):
        seq, cur, miss = [s], s, 0
        e1, e2 = ib[s] + act, ia[s] + p0
        while np.isfinite(ib[cur]) and e1 - w <= t1:
            a = np.union1d(np.arange(max(ss(e1 - w), cur + 1), min(ss(e1 + w, "right"), hi)),
                           np.arange(max(ss(e2 - w), cur + 1), min(ss(e2 + w, "right"), hi))).astype(int)
            if len(a):
                cur, miss = int(a[np.argmax(dur[a])]), 0
                seq.append(cur)
                e1, e2 = ib[cur] + act, ia[cur] + p0
            else:
                miss += 1
                if miss > 2:
                    break
                seq.append(-1)
                e1 += p0
                e2 += p0
        while seq[-1] < 0:
            seq.pop()
        if sum(k >= 0 for k in seq) > sum(k >= 0 for k in best):
            best = seq
    return best


def _track(ia: np.ndarray, ib: np.ndarray, act: float, p0: float, w: float, t0: float, t1: float) -> list[int]:
    """Follow idle pulses cycle by cycle (ia/ib = idle start/end, ib NaN while still open). Takes the best
    chain, then recursively the best chains before and after it (a rise bent by a long occlusion can break
    a chain); segments are separated by a -1."""
    dur = np.where(np.isfinite(ib), ib - ia, np.inf)

    def rec(lo: int, hi: int, first: bool) -> list[int]:
        if hi - lo < (1 if first else 4):
            return []
        c = _chain(ia, ib, dur, act, p0, w, t1, lo, hi)
        n = sum(k >= 0 for k in c)
        if n < (1 if first else 4):
            return []
        a, b = [k for k in c if k >= 0][0], [k for k in c if k >= 0][-1]
        left, right = rec(lo, a, False), rec(b + 1, hi, False)
        return (left + [-1] if left else []) + c + ([-1] + right if right else [])

    return rec(0, len(ia), True)


def _pulses(t: np.ndarray, z: np.ndarray, latency_s: float):
    """Idle pulses (start, end) and active-run lengths from a debounced state array."""
    st, en, v = _runs(z)
    tb = lambda i: 0.5 * (t[i - 1] + t[i]) - latency_s
    ia, ib = [], []
    for k in range(1, len(st)):
        if v[k] == 0:
            ia.append(tb(st[k]))
            ib.append(tb(en[k]) if en[k] < len(z) else np.nan)
    act_len = [(en[k] - st[k]) for k in range(1, len(st) - 1) if v[k] == 1]
    return np.array(ia), np.array(ib), np.array(act_len, float)


def _act_mode(ia: np.ndarray, ib: np.ndarray, p0: float, m_s: float) -> float:
    """Typical active length: mode of rise->next-fall gaps over the next few pulses. Regular cycles pile up
    at one value, occlusion fragments and stops scatter, so this survives a noisy pulse list."""
    g = []
    for b in ib[np.isfinite(ib)]:
        k = int(np.searchsorted(ia, b + m_s, "left"))
        g += [x for x in ia[k:k + 4] - b if x <= p0 - m_s]
    if not g:
        return 0.0
    g = np.array(g)
    h, e = np.histogram(g, bins=np.arange(0.0, p0 + 1.0, 1.0))
    k = int(np.argmax(np.convolve(h, np.ones(3), mode="same")))
    near = g[np.abs(g - (e[k] + 0.5)) <= 1.5]
    return float(np.median(near)) if len(near) else float(e[k] + 0.5)


def _rows(t: np.ndarray, z: np.ndarray, dt: float, period_s: float | None, act_s: float | None,
          latency_s: float, debounce_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Tracked idle pulses as aligned rows (fall, rise); NaN marks a missed edge."""
    m = max(int(round(debounce_s / dt)), 2)
    z = z.copy()
    st, en, v = _runs(z)
    if len(st) > 1 and en[-1] - st[-1] < (m if v[-1] == 0 else 2):  # unconfirmed final run: not an edge yet
        z[st[-1]:] = 1 - v[-1]
    ia, ib, act_len = _pulses(t, z, latency_s)
    if len(ia) == 0:
        return np.array([]), np.array([])
    if act_s is not None:
        act = act_s
    elif period_s is not None:
        act = _act_mode(ia, ib, period_s, debounce_s)
    else:
        act = float(np.median(act_len)) * dt if len(act_len) else 0.0
    if act < debounce_s:
        return np.array([]), np.array([])
    p0 = period_s if period_s is not None else act + float(np.nanmedian(ib - ia))
    w = max(6.0, 0.15 * act)
    seq = _track(ia, ib, act, p0, w, t[0], t[-1])
    F = np.array([ia[k] if k >= 0 else np.nan for k in seq])
    R = np.array([ib[k] if k >= 0 else np.nan for k in seq])
    return F, R


def detect_edges(t, y, period_s: float | None = None, act_s: float | None = None,
                 latency_s: float = LATENCY_S, debounce_s: float = DEBOUNCE_S) -> Edges:
    """Debounced, cycle-consistent transitions of a binary signal. period_s / act_s are optional hints."""
    t, y, dt = _prep(t, y)
    if len(t) < 20:
        return Edges(np.array([]), np.array([]))
    F, R = _rows(t, _clean(y, dt, debounce_s), dt, period_s, act_s, latency_s, debounce_s)
    return Edges(F[np.isfinite(F)], R[np.isfinite(R)])


# --------------------------------------------------------------------------- statistics

def _mad_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def _period_stats(cyc: np.ndarray, tc: np.ndarray, dt: float, q2: float) -> dict:
    """Period, jitter and drift from normal cycle lengths `cyc` (window k starts at tc[k]).
    Edge noise (quantisation, occlusion) makes consecutive cycle lengths anti-correlated (MA(1)):
    var(cyc) = sigma_c^2 + 2 sigma_e^2 and cov(cyc_k, cyc_k+1) = -sigma_e^2, so sigma_c^2 = var + 2 cov1,
    clamped by the pure-quantisation correction q2 (= dt^2/6)."""
    m = float(np.median(cyc))
    sg = max(_mad_sigma(cyc), dt)
    # period: mean over all but grossly wrong cycles. Edge noise cancels between neighbouring cycles (the sum
    # telescopes), so symmetric trimming here would bias it; only variance and drift use the trimmed set.
    P = float(cyc[np.abs(cyc - m) <= max(8 * sg, 6.0)].mean())
    keep = np.abs(cyc - m) <= max(2.5 * sg, 3 * dt)
    c, tk = cyc[keep], tc[keep]
    v = float(c.var(ddof=1)) if len(c) > 1 else 0.0
    nxt = np.abs(tk[1:] - (tk[:-1] + c[:-1])) < 1.0          # truly consecutive windows
    cov1 = float(np.mean((c[:-1] - P)[nxt] * (c[1:] - P)[nxt])) if nxt.sum() >= 5 else -q2 / 2
    sig = float(np.sqrt(max(min(v + 2 * cov1, v - q2), 0.0)))
    out = dict(period=P, sigma_s=sig, jitter=sig / P, n=int(keep.sum()), keep=keep, drift_per_h=0.0, drift_z=0.0)
    if len(c) >= 8:
        A = np.vstack([tk - tk.mean(), np.ones(len(tk))]).T
        coef, *_ = np.linalg.lstsq(A, c, rcond=None)
        resid = c - A @ coef
        se = np.sqrt(resid.var(ddof=2) / max(((tk - tk.mean()) ** 2).sum(), 1e-9))
        out["drift_per_h"] = float(coef[0] * 3600 / P)
        out["drift_z"] = float(abs(coef[0]) / max(se, 1e-12))
    return out


def _confidence(fap: float, contrast: float, jitter: float, n: int, cov: float, cons: float) -> int:
    """0..15, the weakest of: periodicity evidence, fold contrast, jitter, cycle count, coverage, consistency."""
    s = [np.clip((-np.log10(fap) - 2) / 10, 0, 1), np.clip((contrast - 0.35) / 0.4, 0, 1),
         np.clip((0.05 - jitter) / 0.03, 0, 1), np.clip((n - 4) / 12, 0, 1),
         np.clip((cov - 0.5) / 0.4, 0, 1), np.clip((0.06 - cons) / 0.04, 0, 1)]
    c = int(round(15 * min(s)))
    return min(c, MIN_CONF - 1) if (jitter >= 0.05 or min(s) <= 0) else c


def _null(index: int, y: np.ndarray, p: float, fap: float) -> Learned:
    mod = MachineModel(index, float(p), float(np.clip(y.mean() if len(y) else 0.0, 0, 1)), 0.5, 0.0, 0.0, 0)
    return Learned(mod, Edges(np.array([]), np.array([])), 0.0, 0.0, dict(fap=fap, reason="no periodicity"))


def learn(t, y, window: tuple[float, float] | None = None, index: int = 0, latency_s: float = LATENCY_S,
          debounce_s: float = DEBOUNCE_S) -> Learned:
    """Learn one machine from its motion signal. Returns the slow-layer MachineModel plus tracked edges and diagnostics."""
    t, y, dt = _prep(t, y, window)
    if len(t) < 60 or y.std() < 1e-9:
        return _null(index, y, 0.0, 1.0)
    zc = _clean(y, dt, debounce_s)
    p0, acf = _acf_period(zc.astype(float), dt)
    if p0 <= 0:
        return _null(index, y, 0.0, 1.0)
    p_ls, fap = _lomb_scargle(t, y, p0 if p0 > 0 else None)
    diag = dict(fap=fap, ls_period=p_ls, acf_period=p0, acf_score=acf)
    if fap > FAP_OK:
        L = _null(index, y, p0, fap)
        L.diag.update(diag)
        return L
    F, R = _rows(t, zc, dt, p0, None, latency_s, debounce_s)
    if len(F) < 5:
        L = _null(index, y, p0, fap)
        L.diag.update(diag, reason="no cycle chain")
        return L
    if not (np.isfinite(R - F).any() and np.isfinite(F[1:] - R[:-1]).any()):
        L = _null(index, y, p0, fap)
        L.diag.update(diag, reason="no complete cycle")
        return L
    u = R - F                                           # idle lengths (NaN: unobserved)
    w_act = F[1:] - R[:-1]                              # active length of the window starting at rise k
    cyc = R[1:] - R[:-1]                                # rise-to-rise cycle length
    idle0, act0 = float(np.nanmedian(u)), float(np.nanmedian(w_act))
    tol_u = max(4 * _mad_sigma(u[np.isfinite(u)]), 2.5)
    tol_w = max(4 * _mad_sigma(w_act[np.isfinite(w_act)]), 1.2)
    long_ = np.isfinite(u) & (u > idle0 + tol_u)
    wb = np.r_[np.nan, w_act]                           # active before idle i
    wa = np.r_[w_act, np.nan]                           # active after idle i
    ok = lambda x: ~(np.isfinite(x) & (x < act0 - tol_w))
    stop = long_ & ok(wb) & ok(wa)                      # a real stop leaves both neighbouring active phases intact
    bad = long_ & ~stop                                 # otherwise an occlusion burst bent one edge
    normal = np.isfinite(w_act) & np.isfinite(cyc) & ~long_[1:] & ~bad[:-1]
    if normal.sum() < 4:
        L = _null(index, y, p0, fap)
        L.diag.update(diag, reason="too few clean cycles")
        return L
    ps = _period_stats(cyc[normal], R[:-1][normal], dt, dt * dt / 6)
    P = ps["period"]
    kept = np.flatnonzero(normal)[ps["keep"]]
    busy = float(np.clip(np.median(w_act[kept]) / P, 0, 1))
    span = max(np.nanmax(R), np.nanmax(F)) - min(np.nanmin(F), np.nanmin(R))
    stops_h = float(stop.sum()) / max(span / 3600, 1e-9)
    # fold contrast of the tracked square wave against the raw signal
    s = t - latency_s
    lo, hi = np.nanmin(F), np.nanmax(R)
    in_idle = np.zeros(len(t), bool)
    for a, b in zip(F, R):
        if np.isfinite(a) and np.isfinite(b):
            in_idle |= (s >= a) & (s < b)
    sel = (s >= lo) & (s < hi)
    contrast = float(y[sel & ~in_idle].mean() - y[sel & in_idle].mean()) if (sel & in_idle).any() and (sel & ~in_idle).any() else 0.0
    excess = float(np.nansum((u - idle0)[long_]))
    cov = min(1.0, int(np.isfinite(u).sum()) / max((t[-1] - t[0] - excess) / P, 1e-9))
    cons = abs(P - p0) / p0
    drift = ps["drift_per_h"] if ps["drift_z"] > 2 else 0.0
    conf = _confidence(fap, contrast, ps["jitter"], ps["n"], cov, cons)
    mod = MachineModel(index, P, busy, float(min(ps["jitter"], 0.5)), float(drift), float(stops_h), conf)
    diag.update(n_cycles=ps["n"], n_stops=int(stop.sum()), contrast=contrast, cov=cov, cons=cons,
                drift_raw=ps["drift_per_h"], drift_z=ps["drift_z"], sigma_s=ps["sigma_s"], tol_u=tol_u, cyc=cyc[normal])
    return Learned(mod, Edges(F[np.isfinite(F)], R[np.isfinite(R)]), P * (1 - busy), P * busy, diag)


def learn_line(signals: list[tuple[np.ndarray, np.ndarray]], window: tuple[float, float] | None = None,
               indices: list[int] | None = None, latency_s: float = LATENCY_S) -> list[Learned]:
    """Line stations share one takt: pool every station's rises (minus its offset) into one clean index
    clock. Period, jitter, drift and stop rate then come from that clock (dropouts at single stations
    are voted out); busy fractions are measured against it. Falls back to per-station learning if the
    group is too small or inconsistent. Returns Learned in input order; `offset_s` is in diag."""
    idx = indices or list(range(len(signals)))
    L = [learn(t, y, window, i, latency_s) for (t, y), i in zip(signals, idx)]
    good = [k for k, l in enumerate(L) if l.model.confidence >= MIN_CONF]
    if len(good) < 3:
        return L
    pm = float(np.median([L[k].model.period_s for k in good]))
    grp = [k for k in good if abs(L[k].model.period_s / pm - 1) < 0.03]
    if len(grp) < 3:
        return L
    ref = max(grp, key=lambda k: len(L[k].edges.rise))
    rr = L[ref].edges.rise
    off = {}
    for k in grp:
        r = L[k].edges.rise
        d = r[:, None] - rr[None, :]
        d = d[np.arange(len(r)), np.abs(d).argmin(axis=1)]
        d = d[np.abs(d) < 0.4 * pm]
        off[k] = float(np.median(d)) if len(d) else 0.0
    pts = sorted((r - off[k], k) for k in grp for r in L[k].edges.rise)
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] > 0.25 * pm:
            clusters.append(cur)
            cur = []
        cur.append(p)
    clusters.append(cur)
    clusters = [c for c in clusters if len({k for _, k in c}) >= max(3, len(grp) // 2)]
    T = np.array([np.median([x for x, _ in c]) for c in clusters])
    if len(T) < 8:
        return L
    dt = float(np.median(np.diff(np.asarray(signals[0][0], float)[:100])))
    cyc = np.diff(T)
    m, s = float(np.median(cyc)), _mad_sigma(cyc)
    stop = cyc > m + max(5 * s, 0.1 * m)
    ps = _period_stats(cyc[~stop], T[:-1][~stop], dt, dt * dt / 6 * 0.25)
    P = ps["period"]
    stops_h = float(stop.sum()) / max((T[-1] - T[0]) / 3600, 1e-9)
    excess = float((cyc[stop] - m).sum())
    cov = min(1.0, (ps["n"] + int(stop.sum())) / max((T[-1] - T[0] - excess) / P, 1e-9))
    drift = ps["drift_per_h"] if ps["drift_z"] > 2 else 0.0
    out = list(L)
    for k in grp:
        l = L[k]
        o = off[k]
        fall = l.edges.fall
        b = []
        for j in np.flatnonzero(~stop):
            tr = T[j] + o
            f = fall[np.searchsorted(fall, tr):][:1]
            if len(f) and 0 < f[0] - tr < cyc[j]:
                b.append((f[0] - tr) / cyc[j])
        busy = float(np.median(b)) if len(b) >= 4 else l.model.busy_fraction
        conf = _confidence(l.diag["fap"], l.diag["contrast"], ps["jitter"], ps["n"], cov, abs(P - l.diag["acf_period"]) / l.diag["acf_period"])
        mod = MachineModel(l.model.index, P, busy, float(min(ps["jitter"], 0.5)), float(drift), stops_h, conf)
        edges = Edges(fall, np.sort(T + o))
        out[k] = Learned(mod, edges, P * (1 - busy), P * busy,
                         {**l.diag, "pooled": True, "offset_s": o, "n_cycles": ps["n"], "n_stops": int(stop.sum()),
                          "sigma_s": ps["sigma_s"], "drift_raw": ps["drift_per_h"], "drift_z": ps["drift_z"]})
    return out


# --------------------------------------------------------------------------- nowcast

def ttl_s(period_s: float, jitter: float, stops_per_h: float, tol_frac: float = 0.025,
          ttl_max: float = 300.0, ttl_min: float = 2.0) -> float:
    """How long a nowcast stays valid. Cycle boundaries random-walk: after time tau the phase error has
    sigma_e = jitter * sqrt(period * tau). A Monte-Carlo of that walk (scripts/eval_learner.py --ttl)
    gives the time for the 95 % error to reach tol_frac * period; the TTL is half of that:
        tau95 = period * (tol_frac / (1.96 * jitter))**2,  ttl = tau95 / 2   (~300 s at 0.5 %, ~50 s at
    1 %, ~12 s at 2 %, ~2 s at >=5 % for a 60 s cycle). Stops are Poisson: cap the TTL so that a stop
    inside it has < 0.3 % chance (10 / stops_per_h seconds), i.e. <= 10 s whenever stops_per_h > 1."""
    sig = max(float(jitter), 1e-4)
    ttl = 0.5 * period_s * (tol_frac / (1.96 * sig)) ** 2
    if stops_per_h > 0:
        ttl = min(ttl, 10.0 / stops_per_h)
    return float(np.clip(ttl, ttl_min, ttl_max))


def forecast(model: MachineModel, t, y, t_now: float, last_transitions: Edges | None = None,
             latency_s: float = LATENCY_S) -> Forecast:
    """Where the machine is in its cycle at t_now, using only samples with t <= t_now."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    m = t <= t_now
    t, y = t[m], (y[m] > 0.5).astype(float)
    P, busy = model.period_s, model.busy_fraction
    act, idle = busy * P, (1 - busy) * P
    sig = max(1.4 * model.jitter * P, 0.5)
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.5
    ttl = ttl_s(P, model.jitter, model.stops_per_h)
    if len(t) < 20 or P <= 0:
        return Forecast(State.UNKNOWN, t_now, 0.0, 0.0, max(P, 1.0), 0.0, 0)
    tv, tn = _tail(y)
    if model.confidence < MIN_CONF:                      # no usable cycle model: report the instantaneous state only
        st = State.ACTIVE if (tv == 1 and tn >= 2) else State.IDLE
        return Forecast(st, t_now, 0.0, 0.0, max(P, 1.0), min(ttl, 2.0), min(model.confidence, 3))
    if last_transitions is None:
        h = t_now - max(6 * P, 200.0)
        k = t >= h
        last_transitions = detect_edges(t[k], y[k], P, act, latency_s)
    fall = last_transitions.fall[last_transitions.fall <= t_now]
    rise = last_transitions.rise[last_transitions.rise <= t_now]
    rL = rise[-1] if len(rise) else -np.inf
    fL = fall[-1] if len(fall) else -np.inf
    if not (np.isfinite(rL) or np.isfinite(fL)):
        return Forecast(State.UNKNOWN, t_now, 0.0, 0.0, P, 0.0, 0)
    pen = 0
    if rL > fL or (tv == 1 and tn >= 2):                 # ACTIVE: confirmed rise, or motion for >= 2 samples
        if rL > fL:
            r_est = rL
        else:                                            # rise in progress, not yet debounced
            r_est = t[-tn] - 0.5 * dt - latency_s
            pen += 3 if t_now - fL < idle - 3 * sig else 0
        F = r_est + act
        if tv == 0 and tn * dt < DEBOUNCE_S:             # no motion for a moment: dropout, or the fall is under way
            pen += 4 if t_now > F - 3 * sig else 1
        if t_now > F + 3 * sig + DEBOUNCE_S and tv == 1:  # should have stopped by now
            pen += 8
            F = t_now
        state, left, nxt = State.ACTIVE, 0.0, F
        ilen = idle
    else:                                                # IDLE since fL
        age = t_now - fL
        if age > idle + 3 * max(sig, 0.5):
            state, left, nxt, ilen = State.STOPPED, 0.0, t_now, 0.0
        else:
            state, left, nxt = State.IDLE, max(idle - age, 0.0), fL + P
            ilen = left
            if tv == 1 and tn >= 1:                       # a lone motion sample: possible rise, ambiguous
                pen += 2
    conf = max(model.confidence - pen, 0)
    if state == State.STOPPED:
        conf = min(conf, 8)
    if pen:
        ttl = min(ttl, 5.0)
    if state == State.STOPPED:
        ttl = min(ttl, 10.0)
    return Forecast(state, float(nxt), float(left), float(ilen), P, float(ttl), int(conf))


def _now(model: MachineModel, f: Forecast, t_now: float, seq: int) -> MachineNow:
    if f.state == State.ACTIVE:
        nin, nlen = max(f.next_fall - t_now, 0.0), f.idle_len
    elif f.state == State.IDLE:
        nin, nlen = 0.0, f.idle_left
    else:
        nin, nlen = 0.0, 0.0
    return MachineNow(model.index, f.state, float(nin), float(nlen), f.ttl_s, f.confidence, seq & 0xFFFF)


def nowcast(model: MachineModel, t, y, t_now: float, last_transitions: Edges | None = None,
            seq: int = 0, latency_s: float = LATENCY_S) -> MachineNow:
    """Fast layer for one machine at time t_now (uses samples t <= t_now only). `last_transitions` are
    recent validated edges (default: detected from the last few cycles of the signal)."""
    return _now(model, forecast(model, t, y, t_now, last_transitions, latency_s), t_now, seq)


def line_forecast(models: list[MachineModel], signals: list[tuple[np.ndarray, np.ndarray]], offsets: list[float],
                  t_now: float, latency_s: float = LATENCY_S) -> list[Forecast]:
    """Forecasts for stations that share one takt, using the whole line. The newest cycle start is the median
    of every station's rise minus its offset (an occlusion at one camera is voted out); stations that have
    already fallen in this cycle reveal its length (fall = start + offset + busy * length), so a station
    still to fall knows its idle start to a fraction of a second. Falls back to the per-station forecast
    where the line view disagrees by > 3 s (stop under way, missing edges). offsets: learn_line diag['offset_s']."""
    ed, fc = [], []
    for md, (t, y) in zip(models, signals):
        t, y = np.asarray(t, float), (np.asarray(y, float) > 0.5).astype(float)
        m = t <= t_now
        t, y = t[m], y[m]
        k = t >= t_now - max(6 * md.period_s, 200.0)
        e = detect_edges(t[k], y[k], md.period_s, md.busy_fraction * md.period_s, latency_s) if k.sum() > 20 else Edges(np.array([]), np.array([]))
        ed.append(e)
        fc.append(forecast(md, t, y, t_now, e, latency_s))
    n = len(models)
    pts = sorted((r - offsets[i], i) for i, e in enumerate(ed) for r in e.rise[-3:] if models[i].confidence >= MIN_CONF)
    if n < 3 or not pts:
        return fc
    P = float(np.median([m.period_s for m in models if m.confidence >= MIN_CONF] or [models[0].period_s]))
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] > 0.25 * P:
            clusters.append(cur)
            cur = []
        cur.append(p)
    clusters.append(cur)
    clusters = [c for c in clusters if len({i for _, i in c}) >= 3]
    if not clusters:
        return fc
    T = float(np.median([x for x, _ in clusters[-1]]))
    est = []                                             # cycle length implied by stations that already fell
    for i, e in enumerate(ed):
        f = e.fall[(e.fall > T + offsets[i]) & (e.fall <= t_now)]
        b = models[i].busy_fraction
        if len(f) and b >= 0.5 and models[i].confidence >= MIN_CONF:
            est.append((f[0] - T - offsets[i]) / b)
    L = float(np.median(est)) if est and abs(np.median(est) / P - 1) < 0.08 else P
    out = []
    for i, f in enumerate(fc):
        md = models[i]
        if f.state == State.ACTIVE:
            F = T + offsets[i] + md.busy_fraction * L
        elif f.state == State.IDLE:
            fl = ed[i].fall[ed[i].fall <= t_now]
            if len(fl) and fl[-1] >= T + offsets[i] - 3.0:  # fell in this cycle: the next fall is one cycle on
                F = T + L + offsets[i] + md.busy_fraction * md.period_s
            else:                                          # still in the tail of the previous idle: falls this cycle
                F = T + offsets[i] + md.busy_fraction * L
        else:
            out.append(f)
            continue
        if abs(F - f.next_fall) < 3.0 and md.confidence >= MIN_CONF:
            f = Forecast(f.state, float(F), f.idle_left, f.idle_len, f.period_s, f.ttl_s, f.confidence)
        out.append(f)
    return out


def line_nowcast(models: list[MachineModel], signals: list[tuple[np.ndarray, np.ndarray]], offsets: list[float],
                 t_now: float, seq: int = 0, latency_s: float = LATENCY_S) -> list[MachineNow]:
    """Fast layer for all stations of the line at once (see line_forecast)."""
    return [_now(m, f, t_now, seq) for m, f in zip(models, line_forecast(models, signals, offsets, t_now, latency_s))]
