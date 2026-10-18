"""lane observer-fix-fix1: a press whose cycle gets SHORTER, the misfit hold covers every state, kept models age out."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from intercycle.observer import learn_cycles as LC
from intercycle.types import State
from tests.test_observer_fix import _obs, _strokes
from tests.test_observer_live import SPEC, T0, margin_file, stream
from twin import observer_live as OL


def _starts(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Truth stroke starts at tick resolution (the 0 -> 1 ticks of the realised activity)."""
    return t[1:][(~a[:-1]) & a[1:]]


def _run_change(tmp_path, p0: float, p1: float, seed: int):
    rng = np.random.default_rng(seed)
    n = int(20 * 60 * 2)
    t = T0 + 0.5 * np.arange(n) + 0.13
    t_change = T0 + 13 * 60.0
    a0 = _strokes(t, lambda s: p0 if s < t_change else p1, 0.4, 0.002, rng)
    a1 = _strokes(t, lambda s: 2.5, 0.4, 0.002, rng)
    st = _starts(t, a0)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    windows = []                                     # (issue time, state, window end, confidence) for press 0
    for j in range(n):
        ob.ingest(_obs(float(t[j]), (float(a0[j]), float(a1[j]))))
        tq = float(t[j]) + 0.2
        m = ob.fast(tq).machines[0]
        windows.append((tq, m.state, tq + m.next_idle_len_s, m.confidence))
    return ob, windows, st, t_change


def _bad_idle(windows, st, after: float) -> tuple[int, int]:
    """(confident IDLE windows issued after `after`, of those the ones containing a truth stroke start)."""
    n = bad = 0
    for tq, s, end, c in windows:
        if tq <= after or s != State.IDLE or c < LC.MIN_CONF:
            continue
        n += 1
        bad += bool(((st > tq) & (st < end)).any())
    return n, bad


@pytest.mark.parametrize("p0,p1,seed", [(5.0, 1.6, 21), (4.0, 2.5, 22)])
def test_a_shorter_cycle_never_sends_a_confident_idle_window_over_a_stroke(tmp_path, p0, p1, seed):
    """Blocker (fix round): after the period gets SHORTER the stale model's IDLE windows (left = idle - age) contain
    the next stroke start. From the first stroke start after the change that a sent confident IDLE window contained
    (the first contradiction the press can show), no confident IDLE window may contain a stroke start."""
    ob, windows, st, t_change = _run_change(tmp_path, p0, p1, seed)
    n_before, bad_before = _bad_idle([w for w in windows if w[0] < t_change], st, -np.inf)
    assert n_before > 50 and bad_before == 0, (n_before, bad_before)
    first = np.inf                                   # first contradiction: a stroke start inside a sent IDLE window
    for tq, s, end, c in windows:
        if tq > t_change and s == State.IDLE and c >= LC.MIN_CONF:
            hit = st[(st > tq) & (st < end)]
            if len(hit):
                first = min(first, float(hit[0]))
    assert np.isfinite(first)
    n_after, bad_after = _bad_idle(windows, st, first)
    assert bad_after == 0, (bad_after, n_after)
    assert ob.n_misfit[0] >= 1
    assert ob.n_misfit[1] == 0                       # the unchanged press is never flagged


def test_misfit_hold_sends_unknown_for_every_state(tmp_path):
    """Major: while a press is in its misfit hold, IDLE and ACTIVE nowcasts are UNKNOWN too, not only STOPPED."""
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    obs = stream(minutes=12.0, seed=7)
    for o in obs:
        ob.ingest(o)
    t = obs[-1].t_cap
    f = ob.fast(t + 0.2)
    assert any(m.state in (State.IDLE, State.ACTIVE) and m.confidence >= LC.MIN_CONF for m in f.machines)
    ob._misfit_t[:] = t
    f = ob.fast(t + 0.2)
    assert all(m.state == State.UNKNOWN and m.confidence == 0 for m in f.machines)


def test_unchanged_presses_raise_no_misfit(tmp_path):
    """The shorter-cycle test does not fire on presses that keep their period (stub stream: 1 % jitter, MISSING
    bursts)."""
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    for o in stream(minutes=25.0, seed=8):
        ob.ingest(o)
    assert ob.n_misfit.tolist() == [0, 0], ob.n_misfit


def test_kept_model_decays_after_keep_max_and_a_mostly_missing_press_that_changes_is_held(tmp_path):
    """Major: a confident model kept over mostly-MISSING re-fits decays below MIN_CONF keep_max_s after its last
    confident fit, and a press that changes period while mostly MISSING contradicts it on its few observed strokes."""
    cfg = OL.ObserverLiveConfig(margin_path=margin_file(tmp_path))
    ob = OL.ObserverLive(SPEC, cfg)
    obs = stream(minutes=20.0, seed=5)
    for o in obs:
        ob.ingest(o)
    assert ob.models[0].confidence >= LC.MIN_CONF
    fit_t = float(ob._fit_t[0])
    t = obs[-1].t_cap
    rng = np.random.default_rng(9)
    n = int(2 * (cfg.keep_max_s + 2 * cfg.learn_every_s))
    tt = t + 0.5 * np.arange(1, n + 1)
    P_new = 0.6 * ob.models[0].period_s
    a = _strokes(tt, lambda s: P_new, 0.4, 0.002, rng)
    seen = (np.arange(n) % 400) < 60                  # 15 % observed: 30 s seen every 200 s
    kept_seen_conf, held = False, False
    for j in range(n):
        o = obs[j % len(obs)]
        zm0 = float(a[j]) if seen[j] else np.nan
        ob.ingest(replace(o, t=float(tt[j]) + 0.2, t_cap=float(tt[j]),
                          zone_motion=np.array([zm0, o.zone_motion[1]], np.float32),
                          zone_vis=np.array([1.0 if seen[j] else 0.5, o.zone_vis[1]], np.float32)))
        if seen[j] and j % 7 == 0:
            m = ob.fast(float(tt[j]) + 0.2).machines[0]
            if m.state in (State.IDLE, State.ACTIVE) and m.confidence >= LC.MIN_CONF and ob.models[0].period_s > P_new * 1.3:
                kept_seen_conf = kept_seen_conf or ob.misfit(0, float(tt[j]))
            held = held or ob.misfit(0, float(tt[j]))
    assert held and ob.n_misfit[0] >= 1                # the few observed strokes contradict the kept model
    assert not kept_seen_conf
    # the kept model never outlives keep_max_s after its last confident fit
    if float(ob._fit_t[0]) == fit_t:
        assert ob.models[0].confidence < LC.MIN_CONF and ob.n_kept_decay[0] >= 1
