"""lane observer-fix: twin/observer_live.py fixes of the pre-freeze observer probe (spec / run / faults findings)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from intercycle.observer import learn_cycles as LC
from intercycle.observer import mapping as M
from intercycle.types import State
from tests.test_observer_live import SPEC, T0, margin_file, stream, write_jsonl
from twin import observer_live as OL


def _obs(t: float, zm, zv=(1.0, 1.0), cam_ok=(True, True), floor_ok=(True, True)) -> M.Observation:
    e = np.zeros(0)
    return M.Observation(t=t + 0.2, t_cap=t, cls=np.zeros(0, np.int8), x=e, z=e, r=e, heading=e,
                         moving=np.zeros(0, bool), machine=np.zeros(0, np.int16), cam_ok=np.array(cam_ok, bool),
                         zone_motion=np.array(zm, np.float32), zone_vis=np.array(zv, np.float32),
                         cam_floor_ok=np.array(floor_ok, bool))


def _strokes(t: np.ndarray, period_of, busy: float, jitter: float, rng) -> np.ndarray:
    """(n,) bool: the ram moves (busy share of each cycle); cycle k lasts period_of(start of k) x (1 + jitter N)."""
    s, out = [T0 - rng.uniform(0, 2.0)], []
    while s[-1] < t[-1] + 10:
        s.append(s[-1] + period_of(s[-1]) * (1 + jitter * rng.standard_normal()))
    s = np.array(s)
    k = np.searchsorted(s, t, side="right") - 1
    for j, tt in enumerate(t):
        P = s[k[j] + 1] - s[k[j]]
        out.append(tt - s[k[j]] < busy * P)
    return np.array(out)


def test_sealed_twin_run_builds_the_observer_with_measured_press_constants(tmp_path, monkeypatch):
    """Blocker: the twin-run config (broadcast_live.learned_observer / slice.step_observer, both via
    config_for_camera) carries the probe-P4 press constants, so INTERCYCLE_SEALED_RUN=1 no longer refuses it."""
    monkeypatch.setenv(OL.SEALED_ENV, "1")
    cfg = OL.config_for_camera("stat_v1", margin_file(tmp_path))
    c = cfg.consts["press"]
    assert c.placeholder == () and c.p_min == 1.288 and c.debounce_s == 0.25 and "P4" in c.source
    ob = OL.ObserverLive(SPEC, cfg)
    cc = ob.tags["cycle_consts"]
    assert isinstance(cc, dict)
    pc = cc["press"]
    assert isinstance(pc, dict)
    assert ob.tags["sealed"] and pc["placeholder"] == []
    # the default config keeps the placeholder and is still refused in a sealed run
    try:
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    except LC.PlaceholderConstantError:
        pass
    else:
        raise AssertionError("placeholder constants accepted in a sealed run")


def test_p_min_admits_every_twin_press_period():
    """The measured p_min (0.8 x P4 p10) lies below every press period the twin generator can draw."""
    import json
    from pathlib import Path
    d = json.loads((Path(OL.__file__).parent / "dynamics.json").read_text(encoding="utf-8"))
    found: list[float] = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.endswith(".stroke_period_s") and isinstance(v, dict):
                    found.append(float(v["sweep"]["lo"]))
                else:
                    walk(v)
    walk(d)
    assert len(found) >= 6
    lo = min(found)
    assert OL.press_consts().p_min <= lo


def test_unseen_zone_or_invalid_camera_is_missing_whatever_zone_motion_says(tmp_path):
    """Major (faults): zone_motion 0 at zone_vis < ZONE_VIS_MIN, or with a camera whose floor view is invalid, is
    MISSING, never observed idle (so never a STOPPED / STOP_BIN for a dead camera)."""
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    assert np.isnan(ob.motion(_obs(T0, (0.0, 0.0), zv=(0.01, 0.5)))).all()
    assert np.isnan(ob.motion(_obs(T0, (0.0, 1.0), cam_ok=(False, True), floor_ok=(False, True)))).all()
    assert np.isnan(ob.motion(_obs(T0, (0.0, 1.0), floor_ok=(True, False)))).all()
    assert ob.motion(_obs(T0, (0.0, 1.0))).tolist() == [0.0, 1.0]


def test_a_gap_before_the_mission_keeps_the_learned_models(tmp_path):
    """Blocker: learning, then the mission one hour later. The first mission tick must not re-fit on the (almost
    empty) post-gap window and drop the confident models; the fast layer works from the mission's start."""
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    for o in stream(minutes=25.0, seed=3):
        ob.ingest(o)
    conf0 = [m.confidence for m in ob.models]
    assert min(conf0) >= LC.MIN_CONF
    shift = 25 * 60.0 + 3600.0
    mission = [replace(o, t=o.t + shift, t_cap=o.t_cap + shift) for o in stream(minutes=8.0, seed=4)]
    unknown, n = 0, 0
    for o in mission:
        ob.ingest(o)
        if o.t_cap - mission[0].t_cap > 30 and len(ob._t) % 4 == 0:
            f = ob.fast(o.t)
            n += 2
            unknown += sum(m.state == State.UNKNOWN for m in f.machines)
        assert min(m.confidence for m in ob.models) >= LC.MIN_CONF, o.t_cap - mission[0].t_cap
    assert unknown / n < 0.5, (unknown, n)


def test_relearn_on_a_mostly_missing_window_keeps_the_confident_model(tmp_path):
    """Blocker (run): a non-confident re-fit on a window less than half observed (a MISSING-dominated press) keeps
    the last confident model instead of overwriting it with a null model."""
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    obs = stream(minutes=20.0, seed=5)
    for o in obs:
        ob.ingest(o)
    before = ob.models[0]
    assert before.confidence >= LC.MIN_CONF
    t = obs[-1].t_cap
    for j in range(1, 2 * 1250):                                    # 21 min of press 0 MISSING (press 1 observed)
        o = obs[j % len(obs)]
        ob.ingest(replace(o, t=t + 0.5 * j + 0.2, t_cap=t + 0.5 * j,
                          zone_motion=np.array([np.nan, o.zone_motion[1]], np.float32),
                          zone_vis=np.array([0.5, o.zone_vis[1]], np.float32)))
    assert ob.models[0].confidence >= LC.MIN_CONF                   # never overwritten by a null model
    assert abs(ob.models[0].period_s - before.period_s) < 0.1
    assert ob.t_last is not None
    assert ob.fast(ob.t_last + 0.2).machines[0].state == State.UNKNOWN    # MISSING now: still UNKNOWN


def test_cycle_change_never_reads_a_running_press_stopped(tmp_path):
    """Blocker (faults): after the period grows past the learned idle + 3 sigma, every new idle would read as a
    confident STOPPED (and STOP_BIN on a running press). The misfit guard suppresses STOPPED once the press has
    contradicted a would-be stop."""
    rng = np.random.default_rng(11)
    n = int(20 * 60 * 2)
    t = T0 + 0.5 * np.arange(n) + 0.13
    t_change = T0 + 13 * 60.0
    a0 = _strokes(t, lambda s: 1.6 if s < t_change else 5.0, 0.4, 0.002, rng)
    a1 = _strokes(t, lambda s: 2.5, 0.4, 0.002, rng)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    bad = 0
    for j in range(n):
        ob.ingest(_obs(float(t[j]), (float(a0[j]), float(a1[j]))))
        if t[j] > t_change:
            m = ob.fast(float(t[j]) + 0.2).machines[0]
            bad += m.state == State.STOPPED and m.confidence >= LC.MIN_CONF
    assert ob.n_misfit[0] >= 1
    assert bad <= 1, bad                    # at most the first would-be stop, before the press contradicts it


def test_advance_drops_duplicated_and_reordered_lines(tmp_path):
    """Minor (faults): one duplicated line in obs_v2.jsonl no longer kills the observer."""
    obs = stream(minutes=1.0, seed=6)
    obs = obs[:50] + [obs[49], obs[30]] + obs[50:]
    p = tmp_path / "obs_v2.jsonl"
    write_jsonl(p, obs)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)), obs_path=p)
    assert ob.advance(obs[-1].t) == len(obs) - 2
    assert ob.n_dup == 2 and ob.diagnostics()["n_dup"] == 2


def test_stop_bin_disclosure_states_it_never_opens():
    """Major (spec / run): the disclosure says the STOPPED TTL is pinned at the 2 s floor and STOP_BIN never opens,
    and that a restart inside the horizon is not guarded."""
    assert "never opens at all" in OL.STOP_ENTRY_TAG and "2 s floor" in OL.STOP_ENTRY_TAG
    assert "restart" in OL.STOP_ENTRY_TAG
    assert "packed once per mission" in OL.MISFIT_TAG
