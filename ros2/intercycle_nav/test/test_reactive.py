"""E-W4 reactive baseline (reactive.py): frozen config space, own-camera estimator, persistence within a world, planner
and mask integration. Pure logic, no ROS.

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test/test_reactive.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))

from intercycle import scenario as S  # noqa: E402

from intercycle_nav import masks as K  # noqa: E402
from intercycle_nav import reactive as R  # noqa: E402
from intercycle_nav import timing as T  # noqa: E402

DT = 0.2                     # the camera gate's sample period (intercycle_eval.camera.GATE_DT)


def press_stream(t0: float, t1: float, busy_s: float, still_s: float, phase0: float = 0.0, drop_every: int = 0):
    """Gate samples (t, visible, motion_seen) of a machine that moves busy_s then stands still still_s."""
    out = []
    period = busy_s + still_s
    for k, t in enumerate(np.arange(t0, t1, DT)):
        ph = (t - t0 + phase0) % period
        vis = not (drop_every and k % drop_every == drop_every - 1)
        out.append((float(t), vis, vis and ph < busy_s))
    return out


def feed(est, mid, samples):
    for t, v, m in samples:
        est.observe(t, mid, v, m, source="camera_model (R1 geometric, R3 gate)")


# ------------------------------------------------------------------------------------------------ config space
def test_config_is_frozen_hashable_and_canonical():
    c = R.ReactiveConfig()
    with pytest.raises(Exception):
        c.quantile = 0.5  # type: ignore[misc]
    assert hash(c) == hash(R.ReactiveConfig())
    assert c.sha256() == R.ReactiveConfig.from_dict(json.loads(c.to_json())).sha256()
    assert len(c.sha256()) == 64 and c.sha256() != replace(c, quantile=0.2).sha256()
    with pytest.raises(ValueError):
        R.ReactiveConfig.from_dict({"quantile": 0.15})              # outside the frozen space
    with pytest.raises(ValueError):
        R.ReactiveConfig.from_dict({"lookahead_s": 3.0})            # not a parameter
    with pytest.raises(ValueError):
        R.validate(replace(c, min_at_risk=True))                    # type must match too (True == 1 in python)


def test_candidate_set_is_fixed_distinct_and_in_space():
    a, b = R.candidate_configs(), R.candidate_configs()
    assert a == b and len(a) == R.N_CANDIDATES == 8
    assert a[0] == R.ReactiveConfig() and len({c.sha256() for c in a}) == 8
    for c in a:
        R.validate(c)
    assert R.space_sha256() == R.space_sha256()


def test_space_hash_pinned():
    """PREREG v1 freezes PARAM_SPACE (PLAN_TWIN §5). Changing the space, the default or the candidate seed changes this
    hash; after PREREG v1 that needs an amendment, never a silent edit."""
    assert R.space_sha256() == "9ef8b7f5bd8a9a820368b2615d9dcb86847ff4aedf9bb6c28534f424fb054594"


# ------------------------------------------------------------------------------------------------ estimator
def test_unseen_machine_is_unknown_and_stale_view_expires():
    est = R.OwnCameraEstimator(R.ReactiveConfig(), ["P1", "P2"], world_id="dev0")
    assert est.windows(0.0) == {"P1": None, "P2": None}
    feed(est, "P1", press_stream(0.0, 20.0, busy_s=30.0, still_s=30.0))   # moving the whole time
    assert est.windows(20.0)["P1"] == [] and est.windows(20.0)["P2"] is None
    assert est.windows(20.0 + R.FRESH_S + 0.5)["P1"] is None              # the robot looked away: unknown again


def test_window_from_own_history_only_after_enough_runs():
    cfg = R.ReactiveConfig()                        # min_still 2 s, q 0.10, min_at_risk 3, margin 2 s
    est = R.OwnCameraEstimator(cfg, ["P1"], world_id="dev0")
    busy, still = 8.0, 30.0
    samples = press_stream(0.0, 400.0, busy, still, drop_every=20)          # 5 % unseen samples
    opened = []
    for t, v, m in samples:
        est.observe(t, "P1", v, m)
        w = est.windows(t)["P1"]
        if w:
            opened.append((t, w[0]))
    runs = est.tracks["P1"].still_runs
    assert len(runs) >= 3
    assert opened, "no window ever opened"
    # the first window only after min_at_risk complete still runs were seen (first run's start is seen at t = 8 s)
    t_first = opened[0][0]
    n_before = sum(1 for k in range(10) if busy + k * (busy + still) + still <= t_first)
    assert n_before >= cfg.min_at_risk
    for t, (a, b) in opened:
        ph = t % (busy + still)
        assert ph >= busy + cfg.min_still_s - 1e-6          # still for at least min_still_s
        true_end = t - ph + busy + still
        assert b <= true_end - cfg.extra_margin_s + 2 * DT  # never past the real restart (identical runs)
        assert a <= t < b


def test_debounce_ignores_a_single_flicker_and_gap_censors():
    cfg = R.ReactiveConfig(debounce_s=0.4, max_gap_s=3.0)
    tr = R.MachineTrack(cfg)
    for t in np.arange(0.0, 5.0, DT):
        tr.observe(float(t), True, True)
    for t in np.arange(5.0, 20.0, DT):
        tr.observe(float(t), True, bool(abs(t - 12.0) < 1e-6))    # one false-motion sample at 12 s
    assert tr.cur == R.STILL and tr.onset == pytest.approx(5.0)
    for t in np.arange(20.0, 30.0, DT):
        tr.observe(float(t), True, True)
    assert tr.still_runs[-1] == (pytest.approx(15.0), False)
    tr.observe(40.0, False, False)                           # 10 s unseen > max_gap_s
    assert tr.cur == R.UNKNOWN and tr.busy_runs[-1][1] is True


def test_left_censored_run_is_not_timed_unless_configured():
    hist = [(30.0, False)] * 5
    for flag, expect in ((False, []), (True, "window")):
        tr = R.MachineTrack(R.ReactiveConfig(use_left_censored=flag))
        tr.still_runs = list(hist)
        for t in np.arange(100.0, 110.0, DT):                # arrives while the machine is already still
            tr.observe(float(t), True, False)
        w = tr.windows(109.9)
        assert w is not None
        assert (w == []) if expect == [] else (len(w) == 1 and w[0][0] == pytest.approx(100.0))


def test_km_conditional_quantile_censoring_and_cap():
    runs = [(3.0, False)] * 8 + [(120.0, False), (200.0, True)]
    assert R.km_window_length(runs, 0.0, 0.10, 3) == pytest.approx(3.0)       # short gaps dominate at e = 0
    assert R.km_window_length(runs, 10.0, 0.10, 2) == pytest.approx(120.0)    # past the gaps only the long stops remain
    assert R.km_window_length(runs, 10.0, 0.10, 3) is None                    # too few at risk
    assert R.km_window_length([(50.0, True), (80.0, True)], 10.0, 0.1, 2) == pytest.approx(80.0)   # capped, fail closed
    assert R.km_window_length([], 0.0, 0.1, 1) is None


# ------------------------------------------------------------------------------------------------ persistence
def test_persisted_within_the_world_and_refused_elsewhere(tmp_path):
    cfg = R.ReactiveConfig()
    p = str(tmp_path / "w3" / "reactive_state.json")
    assert R.load_state(p) is None
    first = R.OwnCameraEstimator.from_state(R.load_state(p), cfg, ["P1"], "w3")
    assert first.tracks["P1"].still_runs == [] and first.missions == []    # no camera history before the first mission
    feed(first, "P1", press_stream(0.0, 200.0, 8.0, 30.0))
    first.finish_mission(200.0, "w3-d9-z9")
    R.save_state(p, first)
    second = R.OwnCameraEstimator.from_state(R.load_state(p), cfg, ["P1"], "w3")
    assert second.tracks["P1"].still_runs == first.tracks["P1"].still_runs and second.missions == ["w3-d9-z9"]
    assert second.tracks["P1"].cur == R.UNKNOWN                              # a new mission starts unseen
    with pytest.raises(ValueError):
        R.OwnCameraEstimator.from_state(R.load_state(p), cfg, ["P1"], "w4")
    with pytest.raises(ValueError):
        R.OwnCameraEstimator.from_state(R.load_state(p), replace(cfg, quantile=0.2), ["P1"], "w3")
    with pytest.raises(ValueError):
        R.OwnCameraEstimator(cfg, ["P1"], world_id="")


# ------------------------------------------------------------------------------------------------ planner + masks
def test_planner_explores_unknown_and_waits_on_observed():
    core = T.TimingCore.from_scenario(S.load())
    tgs = {t.id: t for t in core.graph.targets}
    t0, xy = 1000.0, (4.0, 32.0)
    unknown = core.plan_target(tgs["HEM1"], t0, xy, {}, "reactive")
    assert unknown is not None and unknown.inside_xy is not None             # never seen: go and look
    seen = core.plan_target(tgs["HEM1"], t0, xy, {"HEM1": []}, "reactive")   # seen, no window now: still go and wait
    assert seen is not None and seen.inside_xy is not None
    assert core.plan_target(tgs["HEM1"], t0, xy, {"HEM1": []}, "learned") is None   # learned: [] means no window
    w ={"HEM1": [(t0 + 100.0, t0 + 130.0)], "SEAL1": [(t0 + 300.0, t0 + 330.0)]}
    a = core.choose([tgs["HEM1"], tgs["SEAL1"]], t0, xy, w, "reactive")
    b = core.choose([tgs["HEM1"], tgs["SEAL1"]], t0, xy, w, "learned")
    assert a is not None and b is not None
    assert (a.target.id, a.t_depart_wait, a.t_goal) == (b.target.id, b.t_depart_wait, b.t_goal)   # same SIPP planner


def test_all_targets_seen_without_window_still_plan_the_nearest_boundary():
    """Audit fix 1: every remaining machine seen with [] (moving or too little history) must give a plan, never None
    (None -> the robot stands still and skips the target after max_wait without ever waiting at its boundary)."""
    core = T.TimingCore.from_scenario(S.load())
    tgs = [t for t in core.graph.mission() if t.kind == "machine"]
    t0, xy = 1000.0, (4.0, 32.0)
    w = {t.machine: [] for t in tgs}
    a = core.choose(tgs, t0, xy, w, "reactive")
    b = core.choose(tgs, t0, xy, {}, "none")                   # no window logic at all: the nearest boundary first
    assert a is not None and a.inside_xy is not None
    assert b is not None and a.target.id == b.target.id and a.t_arrive_wait == pytest.approx(b.t_arrive_wait)
    assert core.choose(tgs, t0, xy, w, "learned") is None       # learned keeps its rule: [] = no idle window known


def test_reactive_masks_follow_the_learned_zone_rule():
    site = K.Site.from_scenario()
    grid = K.Grid(-2.0, -2.0, 0.25, 496, 208)
    mid = site.ids[0]
    x0, y0, x1, y1 = site.rects[mid]
    c = grid.box_slice((x0 + x1) / 2, (y0 + y1) / 2, (x0 + x1) / 2 + 0.01, (y0 + y1) / 2 + 0.01)
    t = 50.0
    for wins, held, closed in (({}, None, True), ({mid: [(40.0, 53.0)]}, None, True),     # window ends inside H = 6 s
                               ({mid: [(40.0, 80.0)]}, None, False), ({}, mid, False)):
        keep, floor, info = R.reactive_masks(grid, site, wins, t, 6.0, held)
        assert bool(keep[c].any()) is closed and np.array_equal(keep, floor)
        assert info["zones"][mid] == ("closed" if closed else "open")
    keep, _, _ = R.reactive_masks(grid, site, {}, t, 6.0, None)
    assert all(keep[grid.box_slice(*r)].all() for r in site.rects.values())    # nothing known: every zone keep-out
    # audit fix 5: a closed zone is the BARE rect, exactly as learned's keepout_floor (no static margin)
    p = K.MaskParams()
    snap = K.Snapshot(t, True, 1, True, 0.0, 0, [K.MachineView(m, i, "UNKNOWN", None, None) for i, m in enumerate(site.ids)])
    learned_floor = K.keepout_floor(grid, "learned", site, snap, object(), p)       # every zone closed (no window)
    assert np.array_equal(keep, learned_floor)


def test_snapshot_json_shape_matches_what_the_planner_reads():
    est = R.OwnCameraEstimator(R.ReactiveConfig(), ["P1"], "w0")
    js = R.snapshot_json(5.0, est)
    assert js["condition"] == "reactive" and js["machines"]["P1"]["windows"] is None
    assert json.loads(json.dumps(js)) == js


def test_reactive_imports_no_truth():
    code = ("import sys; sys.path.insert(0, r'%s'); import intercycle_nav.reactive, intercycle_nav.timing; "
            "from intercycle_nav.masks import assert_no_truth; assert_no_truth(); print('ok')" % PKG)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
