"""Learner tests on SYNTHETIC camera signals (45 min from 07:00, truth only used for scoring)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from intercycle.observer import learn_cycles as LC
from intercycle.observer import learn_traffic, signals
from intercycle.scenario import load
from intercycle.truth import machines as truth
from intercycle.truth import traffic
from intercycle.types import State, Window

SCN = load()
T0 = 7 * 3600.0
WIN = (T0, T0 + 2700.0)
SEEDS = range(10)
IDS = [m.id for m in SCN.machines]
LINE = [m.id for m in SCN.machines if m.on_line]
CELLS = [m.id for m in SCN.machines if not m.on_line]


@lru_cache(maxsize=None)
def sig(mid: str, seed: int):
    return signals.machine_signal(SCN, mid, seed, T0, T0 + 2700 + 1000)


@lru_cache(maxsize=None)
def learned(seed: int) -> dict:
    out = dict(zip(LINE, LC.learn_line([sig(i, seed) for i in LINE], WIN, [IDS.index(i) for i in LINE])))
    for i in CELLS:
        out[i] = LC.learn(*sig(i, seed), window=WIN, index=IDS.index(i))
    return out


def p_true(mid: str) -> float:
    m = SCN.machine(mid)
    return SCN.line.takt_s if m.on_line else float(m.period_s)


def test_signal_shim_statistics():
    ws = [Window(t, t + 30, t + 60) for t in np.arange(0, 6000, 60.0)]
    t, y = signals.camera(ws, 0, 6000, seed=1)
    assert np.allclose(np.diff(t), 0.5)
    clean_t, clean_y = signals.camera(ws, 0, 6000, seed=1, p_occlude=0, p_flip=0)
    assert clean_y[(clean_t >= 120.5) & (clean_t < 149.9)].min() == 1      # active 120..150, seen 0.2 s late
    assert clean_y[(clean_t >= 150.5) & (clean_t < 179.9)].max() == 0
    assert clean_y[(clean_t > 119.9) & (clean_t < 120.2)].max() == 0        # latency: not yet visible
    assert clean_y[(clean_t > 149.9) & (clean_t < 150.2)].min() == 1        # ... and still visible after the stop
    act = clean_y == 1
    assert 0.03 < (y[act] == 0).mean() < 0.14                               # occlusion dropouts ~8 %
    assert 0.015 < (y[~act] == 1).mean() < 0.05                             # false flips ~3 %


def test_period_error_line_and_cells():
    for group in (LINE, CELLS):
        err = np.array([100 * abs(learned(s)[i].model.period_s / p_true(i) - 1) for i in group for s in SEEDS])
        assert (err < 0.3).mean() >= 0.85, np.sort(err)[-5:]
        assert err.max() < 0.6 and err.mean() < 0.2


def test_busy_fraction_error_below_4_points():
    for s in SEEDS:
        for i in IDS:
            assert abs(learned(s)[i].model.busy_fraction - SCN.machine(i).busy) < 0.04, (s, i)


def test_confidence_high_for_clean_machines():
    c = np.array([learned(s)[i].model.confidence for s in SEEDS for i in IDS])
    assert (c >= 4).all() and np.median(c) >= 14


def test_s50_framing_idle_is_six_seconds():
    for s in SEEDS:
        L = learned(s)["S50"]
        assert abs(L.model.busy_fraction - 0.9) < 0.02
        assert abs(L.model.period_s * (1 - L.model.busy_fraction) - 6.0) < 1.2
        assert L.model.confidence >= 8
        assert abs(L.idle_s - 6.0) < 1.2


def test_jitter_and_stop_rate_are_sensible():
    j = np.array([learned(s)[i].model.jitter for s in SEEDS for i in LINE])
    assert 0.006 < j.mean() < 0.014                                          # truth 1 %
    st = np.mean([learned(s)[i].model.stops_per_h for s in SEEDS for i in LINE])
    assert 2.5 < st < 6                                                      # truth 4 / h (andon)
    d = np.array([learned(s)[i].model.drift_per_h for s in SEEDS for i in IDS])
    assert np.abs(d).max() < 0.02                                            # 45 min cannot resolve 0.1 %/h; must not hallucinate


def test_pure_noise_has_low_confidence():
    rng = np.random.default_rng(5)
    t = T0 + np.arange(5400) / 2.0
    runs = np.concatenate([np.full(int(rng.exponential(20) * 2) + 1, k % 2) for k in range(400)])[:5400]
    cases = {"white": rng.integers(0, 2, 5400), "sticky": runs, "zeros+flips": (rng.random(5400) < 0.03).astype(int),
             "ones": np.ones(5400, int), "sparse": (rng.random(5400) < 0.2).astype(int)}
    for name, y in cases.items():
        assert LC.learn(t, y).model.confidence < 4, name


def test_high_jitter_caps_confidence_below_4():
    rng = np.random.default_rng(3)
    for jit in (0.06, 0.10):
        ws, t = [], 0.0
        while t < 3300:
            p = 60 * (1 + jit * rng.standard_normal())
            ws.append(Window(t, t + 0.7 * p, t + p))
            t += p
        tt, y = signals.camera(ws, 0, 3000, seed=2)
        m = LC.learn(tt, y).model
        assert m.confidence < 4 and m.jitter >= 0.04, (jit, m)
    ws = [Window(t, t + 42, t + 60) for t in np.arange(0, 3300, 60.0)]      # sanity: no jitter -> confident
    assert LC.learn(*signals.camera(ws, 0, 3000, seed=2)).model.confidence >= 12


def test_ttl_rule_follows_the_monte_carlo_anchors():
    f = lambda j, s=0.0: LC.ttl_s(60, j, s)
    assert f(0.003) >= 200 and 150 <= f(0.005) <= 300
    assert 40 <= f(0.01) <= 70 and 8 <= f(0.02) <= 15 and 1.5 <= f(0.05) <= 3 and f(0.2) <= 3
    assert all(f(a) >= f(b) for a, b in [(0.002, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05)])
    assert all(f(0.003, s) <= 10 for s in (1.01, 2, 4, 20))
    assert LC.ttl_s(120, 0.01, 0) > LC.ttl_s(60, 0.01, 0)


def _next_idle_errors(h: float, ids: list[str], seeds) -> np.ndarray:
    errs = []
    for s in seeds:
        mods = learned(s)
        ws = {i: truth.windows(SCN, i, s, T0, T0 + 2700 + 1300) for i in ids}
        for k in range(6):
            t_now = WIN[1] + 150 * k
            for i in ids:
                f = LC.forecast(mods[i].model, *sig(i, s), t_now)
                falls = np.array([w.is_ for w in ws[i]])
                pred = f.fall_after(t_now + h)
                errs.append(abs(pred - falls[np.abs(falls - pred).argmin()]))
    return np.array(errs)


def test_nowcast_next_idle_error_at_30s_horizon():
    e = _next_idle_errors(30, IDS, SEEDS)
    assert (e < 1.5).mean() >= 0.78, (e < 1.5).mean()
    assert np.median(e) < 0.9
    per_seed = [np.median(_next_idle_errors(30, IDS, [s])) for s in SEEDS]
    assert np.mean(np.array(per_seed) < 1.5) >= 0.9


def test_line_forecast_is_no_worse_than_per_station():
    errs_line, errs_one = [], []
    for s in list(SEEDS)[:5]:
        mods = [learned(s)[i].model for i in LINE]
        offs = [learned(s)[i].diag.get("offset_s", 0.0) for i in LINE]
        ws = {i: truth.windows(SCN, i, s, T0, T0 + 2700 + 1300) for i in LINE}
        for k in range(6):
            t_now = WIN[1] + 150 * k
            fcs = LC.line_forecast(mods, [sig(i, s) for i in LINE], offs, t_now)
            for i, f, m in zip(LINE, fcs, mods):
                falls = np.array([w.is_ for w in ws[i]])
                g = LC.forecast(m, *sig(i, s), t_now)
                for fc, acc in ((f, errs_line), (g, errs_one)):
                    pred = fc.fall_after(t_now + 15)
                    acc.append(abs(pred - falls[np.abs(falls - pred).argmin()]))
    assert np.mean(np.array(errs_line) < 1.0) >= np.mean(np.array(errs_one) < 1.0) - 0.02


def test_nowcast_fields_and_states():
    s, i = 0, "S10"
    L = learned(s)[i]
    ws = truth.windows(SCN, i, s, T0, T0 + 2700 + 1300)
    seen, bad = set(), 0
    for t_now in np.arange(WIN[1] + 5, WIN[1] + 600, 7.0):
        n = LC.nowcast(L.model, *sig(i, s), t_now, seq=int(t_now))
        seen.add(n.state)
        assert n.index == IDS.index(i) and 0 <= n.confidence <= 15 and n.ttl_s >= 0 and n.next_idle_in_s >= 0
        act = truth.is_active(ws, t_now)
        if n.state == State.IDLE:
            bad += act                                   # camera latency + debounce: a rise is seen ~1 s late
            assert n.next_idle_in_s == 0 and n.next_idle_len_s <= (1 - 0.8) * 60 + 3
        if n.state == State.ACTIVE:
            assert 0 <= n.next_idle_in_s <= 60 and abs(n.next_idle_len_s - 12) < 2
    assert State.ACTIVE in seen and State.IDLE in seen and bad <= 3


def test_nowcast_uses_only_the_past():
    s, i = 1, "S20"
    t, y = sig(i, s)
    t_now = WIN[1] + 100
    a = LC.nowcast(learned(s)[i].model, t, y, t_now)
    y2 = y.copy()
    y2[t > t_now] = 1 - y2[t > t_now]                    # corrupt the future
    assert a == LC.nowcast(learned(s)[i].model, t, y2, t_now)


def test_low_confidence_model_reports_instant_state_only():
    m = LC.learn(T0 + np.arange(5400) / 2.0, np.random.default_rng(1).integers(0, 2, 5400)).model
    n = LC.nowcast(m, T0 + np.arange(5400) / 2.0, np.ones(5400), T0 + 2000)
    assert n.confidence < 4 and n.state == State.ACTIVE


def test_stopped_when_idle_outlasts_the_prediction():
    s, i = 0, "S10"
    ws = truth.windows(SCN, i, s, T0, T0 + 2700 + 1300)
    stop = next(w for w in ws if w.kind == "andon" and w.bs > T0 + 300 and w.ie - w.is_ > 50)
    t, y = signals.machine_signal(SCN, i, s, T0, stop.ie + 100)
    mod = learned(s)[i].model
    n = LC.nowcast(mod, t, y, stop.is_ + (1 - mod.busy_fraction) * mod.period_s + 15)
    assert n.state == State.STOPPED and n.next_idle_len_s == 0
    n = LC.nowcast(mod, t, y, stop.ie + 40)
    assert n.state == State.ACTIVE


def test_traffic_levels_match_truth_and_shape():
    days = 3
    t, c = signals.occupancy(SCN, 7, 0, days * 86400.0)
    assert c.shape == (len(SCN.zones), days * 288)
    lv = learn_traffic.learn_traffic(t, c)
    ref = traffic.levels(SCN)
    assert len(lv) == len(SCN.zones) and all(len(r) == 24 and min(r) >= 0 and max(r) <= 15 for r in lv)
    err = np.array([[abs(lv[k][h] - ref[z.id][h]) for h in range(24)] for k, z in enumerate(SCN.zones)])
    assert err.mean() < 0.6 and err.max() <= 2
    assert all(max(r) == 15 for r in lv)
    ids = [z.id for z in SCN.zones]
    people = lv[ids.index("Z_AISLE_N")]
    assert people[6] >= 14 and people[14] >= 14 and people[22] > people[23] + 2   # shift-change peaks (22:00 vs the night plateau)
    assert people[9] <= people[8] - 1 and people[11] <= people[10] - 1          # breaks 09:00, 11:30
    dh = lv[ids.index("Z_DATA_HALL")]
    assert min(dh[8:18]) >= 12 and max(dh[0:5]) <= 7                           # technicians 07-19
    south = lv[ids.index("Z_AISLE_S")]                                          # AGVs keep it busy day and night
    assert min(south) >= 5 and min(south) >= min(people) + 2


def test_traffic_single_day_still_works():
    t, c = signals.occupancy(SCN, 3, 0, 86400.0)
    lv = learn_traffic.learn_traffic(t, c)
    ref = traffic.levels(SCN)
    err = [abs(lv[k][h] - ref[z.id][h]) for k, z in enumerate(SCN.zones) for h in range(24)]
    assert np.mean(err) < 1.0
    assert traffic.level("Z_MARKET", 6, SCN) == ref["Z_MARKET"][6]
