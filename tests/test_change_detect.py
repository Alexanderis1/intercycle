"""M-W4 (safety direction): in-zone change detection + forgetting per observed tick (observer/change_detect.py).

Synthetic streams at 2 Hz, camera detection 0.68 per occupied cell and tick (MEASURED_SHIM's 32 % miss), learned
background hits. Each test prints its numbers (run with -s): detection delays in ticks / seconds, false alarms per hour.
On the code before this lane (map counts without decay, observer/change.py blind inside zone + 1 m) every scenario below
stays FREE for hours: there is no in-zone detector, and p reaches p_min only after ~9 h (always occupied) / ~30 h.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from intercycle.observer import change_detect as CD
from intercycle.observer.change_detect import ChangeConfig, ChangeLayer

HZ = 2.0
DET = 0.68                                   # 1 - MEASURED_SHIM p_drop


def _bg_stream(rng, N, kind_p=(0.6, 0.25, 0.15), P=120):
    """Stationary mixed stream: iid low-rate cells, rare bursty cells (excursions of a few seconds), periodic envelope
    cells. Returns tick(t) -> hits."""
    typ = rng.choice(3, N, p=list(kind_p))
    p_iid = rng.uniform(0, 0.03, N)
    q = rng.uniform(1 / 6000, 1 / 600, N)    # excursions every 5-50 min
    r = rng.uniform(0.5, 0.9, N)             # mean excursion 1-5 s
    ph = rng.integers(0, P, N)
    w = rng.integers(10, 60, N)
    state = {"occ": np.zeros(N, bool)}

    def tick(t):
        u = rng.random(N)
        occ = np.where(state["occ"], u < r, u < q) & (typ == 1)
        state["occ"] = occ
        per = ((t + ph) % P) < w
        truth = np.where(typ == 2, per, occ)
        return (truth & (rng.random(N) < DET)) | ((typ == 0) & (rng.random(N) < p_iid)) | (rng.random(N) < 1e-4)
    return tick, typ


def _run_change(kind, frac=0.5, seed=1, N=400, learn=3 * 3600 * 2, after=40 * 60 * 2, nsl=1, P=120, cfg=None):
    """Learn a stationary background, then change cells 50..79 at tick `learn`. Returns (delay per changed cell in
    ticks, NaN = never; false-alarm cells elsewhere; layer)."""
    rng = np.random.default_rng(seed)
    p_bg = rng.uniform(0, 0.02, N)
    L = ChangeLayer(N, nsl, cfg)
    obs = np.ones(N, bool)
    new = np.arange(50, 80)
    ph0 = rng.integers(0, P, N)
    delays = np.full(N, np.nan)
    for t in range(learn + after):
        truth = np.zeros(N, bool)
        if kind == "program":                # envelope cells 50..79: occupied in phase [0, 0.5) before, [0.5, 1) after
            u = (t % P) / P
            truth[new] = (u < 0.5) if t < learn else (u >= 0.5)
        elif t >= learn:
            if kind == "always":
                truth[new] = True
            elif kind == "arm":              # a new arm sweeping the cells for `frac` of its own cycle (own period)
                truth[new] = ((t + ph0[new]) % 100) < frac * 100
        hits = (rng.random(N) < p_bg) | (truth & (rng.random(N) < DET))
        r = L.step((t % P) * nsl // P, hits, obs)
        for c in r.alarmed:
            if np.isnan(delays[c]):
                delays[c] = t - learn
    d = delays[new]
    fa = int(np.isfinite(np.delete(delays, new)).sum())
    return d, fa, L


# ---------------------------------------------------------------------------------------------- forgetting
def test_counts_decay_only_on_observed_ticks():
    L = ChangeLayer(4, n_slices=2, cfg=ChangeConfig(half_life_ticks=100.0))
    obs = np.array([True, True, False, True])
    for _ in range(50):
        L.step(0, np.array([True, False, True, False]), obs)
    n0, k0 = L.n.copy(), L.k.copy()
    assert n0[0, 2] == 0 and k0[0, 2] == 0, "an unobserved cell learns nothing (its hits are not evidence)"
    # 10 000 ticks with no camera: nothing decays, nothing changes
    for _ in range(10_000):
        L.step(0, np.zeros(4, bool), np.zeros(4, bool))
    assert np.array_equal(L.n, n0) and np.array_equal(L.k, k0)
    # ticks of slice 1 do not decay slice 0
    for _ in range(500):
        L.step(1, np.zeros(4, bool), np.ones(4, bool))
    assert np.array_equal(L.n[0], n0[0]) and np.array_equal(L.k[0], k0[0])
    # observed ticks of slice 0 decay it by exactly lam per tick: hits halve after half_life_ticks observed ticks
    k_before = L.k[0, 0]
    for _ in range(100):
        L.step(0, np.zeros(4, bool), np.ones(4, bool))
    assert L.k[0, 0] == pytest.approx(0.5 * k_before, rel=1e-9)
    # ... and a cell observed only on half of those ticks decays only on its observed ticks
    L2 = ChangeLayer(2, cfg=ChangeConfig(half_life_ticks=100.0))
    for _ in range(40):
        L2.step(0, np.ones(2, bool), np.ones(2, bool))
    k2 = L2.k[0].copy()
    for j in range(200):
        L2.step(0, np.zeros(2, bool), np.array([True, j % 2 == 0]))
    assert L2.k[0, 0] == pytest.approx(0.25 * k2[0], rel=1e-9)
    assert L2.k[0, 1] == pytest.approx(0.5 * k2[1], rel=1e-9)


def test_default_half_life_is_24h_of_wall_clock_spread_over_the_slices():
    assert ChangeLayer(3).half_life_ticks == 172_800.0
    assert ChangeLayer(3, n_slices=17).half_life_ticks == pytest.approx(172_800.0 / 17)
    assert CD.half_life_for(24, 48) == pytest.approx(3600.0)


def test_forgetting_absorbs_a_newly_occupied_cell_and_lifts_unknown():
    """Analytic check of the decay, then the full cycle: change -> UNKNOWN -> counts absorb it -> UNKNOWN lifted, the
    map now says occupied (p >= p_min). Without decay the same cell needs ~9 h against a 1-week denominator."""
    hl = 2000.0
    cfg = ChangeConfig(half_life_ticks=hl, clear_ticks=600)
    L = ChangeLayer(1, cfg=cfg)
    lam = 2 ** (-1 / hl)
    n_sat = 1 / (1 - lam)
    rng = np.random.default_rng(3)
    for _ in range(int(8 * hl)):                         # saturated empty history
        L.step(0, np.array([rng.random() < 0.002]), np.ones(1, bool))
    t_unknown = t_absorb = t_lift = None
    for t in range(int(3 * hl)):
        L.step(0, np.array([rng.random() < DET]), np.ones(1, bool))
        if t_unknown is None and L.unknown[0]:
            t_unknown = t
        if t_absorb is None and L.p(0)[0] >= cfg.p_min:
            t_absorb = t
        if t_unknown is not None and t_lift is None and not L.unknown[0]:
            t_lift = t
    expect = math.log(1 - cfg.p_min / DET) / math.log(lam)      # p(t) = DET (1 - lam^t) once n is saturated
    print(f"\nforgetting: UNKNOWN after {t_unknown} ticks, absorbed (p >= p_min) after {t_absorb} (analytic "
          f"{expect:.0f}), UNKNOWN lifted after {t_lift} ticks (half-life {hl:.0f}, clear {cfg.clear_ticks})")
    assert t_unknown is not None and t_unknown <= 120
    assert t_absorb is not None and abs(t_absorb - expect) <= 0.25 * expect + 20
    assert t_lift is not None and t_absorb <= t_lift <= t_absorb + cfg.clear_ticks + 5
    assert L.p(0)[0] >= cfg.p_min, "after the lift the map itself broadcasts the cell occupied"
    # the defect without forgetting: a week of empty visits at 2 Hz, always occupied from now on
    week = 7 * 24 * 3600 * HZ
    t_nodecay = cfg.p_min * week / (DET - cfg.p_min)
    t_decay_24h = math.log(1 - cfg.p_min / DET) / math.log(2 ** (-1 / CD.DAY_TICKS))
    print(f"time to p >= p_min for a newly always-occupied cell: no decay {t_nodecay / HZ / 3600:.1f} h, "
          f"24 h half-life {t_decay_24h / HZ / 3600:.1f} h; CUSUM: {t_unknown / HZ:.0f} s")
    assert t_nodecay / HZ / 3600 > 8.0 and t_decay_24h < t_nodecay / 3


def test_unknown_is_never_lifted_and_cusum_frozen_while_unobserved():
    cfg = ChangeConfig(clear_ticks=100)
    L = ChangeLayer(1, cfg=cfg)
    rng = np.random.default_rng(4)
    for _ in range(4000):
        L.step(0, np.array([rng.random() < 0.005]), np.ones(1, bool))
    t = 0
    while not L.unknown[0]:
        L.step(0, np.array([rng.random() < DET]), np.ones(1, bool))
        t += 1
        assert t < 200
    S = L.S.copy()
    for _ in range(10 * cfg.clear_ticks):                 # camera down: no lift, S frozen
        r = L.step(0, np.zeros(1, bool), np.zeros(1, bool))
        assert not len(r.lifted)
    assert L.unknown[0] and np.array_equal(L.S, S)
    lifted_at = None
    for j in range(2 * cfg.clear_ticks):                  # observed, no hits: lifted after clear_ticks observed ticks
        if len(L.step(0, np.zeros(1, bool), np.ones(1, bool)).lifted):
            lifted_at = j + 1
            break
    assert lifted_at == cfg.clear_ticks


# ---------------------------------------------------------------------------------------------- detection
@pytest.mark.parametrize("kind,frac,bound_s", [("always", 1.0, 90), ("arm", 0.5, 360), ("arm", 0.3, 600)])
def test_change_becomes_unknown_within_a_bounded_number_of_ticks(kind, frac, bound_s):
    """A newly always-occupied cell / a new arm sweeping cells for 50 % or 30 % of its cycle, in cells the map has
    learned as FREE (background p 0-2 %): every changed cell UNKNOWN within the bound, no false alarm elsewhere."""
    d, fa, _ = _run_change(kind, frac)
    print(f"\n{kind} {frac:.0%}: all {len(d)} cells UNKNOWN: {np.isfinite(d).all()}; delay median "
          f"{np.nanmedian(d) / HZ:.0f} s, p90 {np.nanquantile(d, 0.9) / HZ:.0f} s, max {np.nanmax(d) / HZ:.0f} s; "
          f"false-alarm cells {fa} over 370 cells x 40 min")
    assert np.isfinite(d).all(), "a changed cell stayed FREE"
    assert np.nanmax(d) <= bound_s * HZ
    assert fa == 0


def test_program_change_in_phase_slices_becomes_unknown():
    """16 phase slices: the cells' envelope moves from phases [0, 0.5) to [0.5, 1) (a robot program change). The new
    phases had p ~ background there, so the hits are in-zone hits in FREE slices of the same cells."""
    d, fa, L = _run_change("program", nsl=16, learn=3 * 3600 * 2, after=20 * 60 * 2)
    print(f"\nprogram change: all {len(d)} cells UNKNOWN: {np.isfinite(d).all()}; delay median "
          f"{np.nanmedian(d) / HZ:.0f} s, max {np.nanmax(d) / HZ:.0f} s; false-alarm cells {fa}")
    assert np.isfinite(d).all() and np.nanmax(d) <= 300 * HZ
    assert fa == 0


def test_false_alarm_rate_on_a_stationary_stream():
    """2000 cells (iid low-rate, bursty rare excursions with detection gaps, periodic envelope cells), learned for 2 h,
    then 8 h of the same process: false alarms per hour."""
    rng = np.random.default_rng(11)
    N = 2000
    tick, typ = _bg_stream(rng, N)
    L = ChangeLayer(N)
    obs = np.ones(N, bool)
    learn, test_h = int(2 * 3600 * HZ), 8
    for t in range(learn):
        L.step(0, tick(t), obs)
    mon = L.monitored(0)
    n_ev = 0
    L.n_alarms = 0
    for t in range(learn, learn + int(test_h * 3600 * HZ)):
        if len(L.step(0, tick(t), obs).alarmed):
            n_ev += 1
    print(f"\nstationary: {mon.sum()} monitored cells of {N}; {L.n_alarms} false-alarm cells, {n_ev} alarm ticks in "
          f"{test_h} h = {n_ev / test_h:.3f} false alarms/h ({n_ev / test_h / mon.sum() * 1000:.4f} per 1000 monitored "
          f"cells per h); max S {L.S.max():.1f} vs h {L.cfg.h}")
    assert mon[typ != 2].mean() > 0.95, "low-rate cells are monitored"
    assert n_ev <= 1


def test_hits_on_unobserved_cells_are_ignored_and_shapes_checked():
    L = ChangeLayer(6, shape=(2, 3), cfg=ChangeConfig(dilate=1))
    L.step(0, np.ones(6, bool), np.zeros(6, bool))
    assert L.n.sum() == 0 and L.k.sum() == 0
    with pytest.raises(ValueError):
        ChangeLayer(5, shape=(2, 3))
    with pytest.raises(IndexError):
        L.step(1, np.zeros(6, bool), np.ones(6, bool))
    L.unknown[0] = True
    assert L.unknown_mask().reshape(2, 3).tolist() == [[True, True, False], [True, True, False]]


def test_change_detect_modules_do_not_import_truth():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "intercycle" / "observer"
    for f in ("change_detect.py", "change_detect_l1.py"):
        tree = ast.parse((root / f).read_text(encoding="utf-8"))
        mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        mods += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not [m for m in mods if m.startswith(("intercycle.truth", "twin"))], (f, mods)


def test_drift_helper_matches_the_sign_of_the_cusum():
    cfg = ChangeConfig()
    assert CD.drift_per_tick(cfg, 0.5, 0.01, 0.01, 0.5) < 0          # the null itself drifts down
    assert CD.drift_per_tick(cfg, 0.5, 0.01, 0.5, 0.9) > 0           # a new long-stay occupant climbs
