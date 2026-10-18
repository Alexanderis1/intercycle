"""A-lite CPU double (PLAN_TWIN §7 item 5, intercycle/alite/episode.py): the dry-run episode runs for every condition,
and the learned robot fails closed (no model, no phase origin, no parked footprint, an UNKNOWN box on the zone)."""
import json
from typing import Any

import numpy as np
import pytest

from intercycle.alite import episode as E
from intercycle.scenario import load

SCN = load()
DRY: dict[str, Any] = dict(learn_s=1800.0, mission_starts_h=(6.75,), mission_s=300.0, mission_day=0)   # = scripts/alite_run.py DRY


@pytest.fixture(scope="module")
def learned():
    ep = E.Episode(SCN, 801, "learned", **DRY)
    return ep, ep.run()


def _check_row(r: dict, cond: str):
    json.dumps(r)                                     # the runner writes it as JSON
    assert r["simulated"] is True and r["condition"] == cond and r["n_missions"] == 1
    tot = r["totals"]
    assert tot["captures_passed"] <= tot["captures_attempted"]
    assert tot["entries_clean"] <= tot["entries_attempted"]
    assert r["targets"][0], "the robot reached no target in the dry-run mission"


def test_dry_run_learned_runs_and_learns(learned):
    ep, r = learned
    _check_row(r, "learned")
    assert ep.obs.models is not None and ep.obs.origin is not None
    assert r["totals"]["captures_attempted"] >= 1


@pytest.mark.parametrize("cond", ["reactive", "static"])
def test_short_episode_runs_for_the_baselines(cond):
    r = E.run(SCN, 801, cond, learn_s=900.0, mission_starts_h=(6.5,), mission_s=180.0, mission_day=0)
    _check_row(r, cond)
    if cond == "static":
        assert r["totals"]["entries_attempted"] == 0
        assert r["totals"]["entries_declined"] == r["totals"]["captures_attempted"]


def test_bad_condition_and_mission_before_learning_end_are_refused():
    with pytest.raises(ValueError):
        E.Episode(SCN, 801, "oracle", **DRY)
    with pytest.raises(ValueError):
        E.Episode(SCN, 801, "learned", learn_s=3600.0, mission_starts_h=(6.5,), mission_s=60.0, mission_day=0)


def test_before_learning_the_map_never_says_parked():
    ep = E.Episode(SCN, 801, "learned", **DRY)
    t = ep.t_learn0 + 60.0
    ep.advance(t)
    assert ep.obs.models is None
    assert ep.parked_envelope("HEM1") is None
    f, ti = ep.fast_at(t)
    assert ep._map_parked("HEM1", t, ti, f[ep.idx["HEM1"]]) is False


def test_learned_fails_closed_without_phase_origin(learned, monkeypatch):
    ep, _ = learned
    t = ep.starts[0] + ep.mission_s
    f, ti = ep.fast_at(t)
    monkeypatch.setattr(ep.obs, "origin", None)
    assert all(ep._map_parked(m, t, ti, f[ep.idx[m]]) is False for m in ("HEM1", "S20", "SEAL1"))


def test_learned_declines_entry_when_an_unknown_box_touches_the_zone(learned, monkeypatch):
    ep, _ = learned
    x0, z0, x1, z1 = SCN.machine("HEM1").rect
    monkeypatch.setattr(ep, "unknown_at_start", [(x1 - 0.2, z0, x1 + 1.0, z1)])
    t = ep.starts[0] + ep.mission_s
    res = E.MissionResult(t)
    assert ep._unknown_in_zone("HEM1")
    assert ep.entry("HEM1", t, t + 600.0, res) == (t, "declined")
    assert res.trips == 0 and res.holds == 0


def test_the_robot_gets_unknown_boxes_at_mission_start_only(learned, monkeypatch):
    ep, _ = learned
    x0, z0, x1, z1 = SCN.machine("SEAL1").rect
    monkeypatch.setattr(ep, "unknown_at_start", [])
    monkeypatch.setattr(ep.obs.map, "unknown_rects", [(x0, z0, x1, z1)])   # appears on the map during a mission
    assert not ep._unknown_in_zone("SEAL1")
    monkeypatch.setattr(ep, "mission_s", 1.0)
    ep.mission(ep.starts[0] + 320.0)                   # the next mission start hands the boxes over
    assert ep._unknown_in_zone("SEAL1")


def test_static_never_enters():
    ep = E.Episode(SCN, 801, "static", **DRY)
    res = E.MissionResult(0.0)
    assert ep.entry("S20", 25000.0, 26000.0, res) == (25000.0, "declined")


def test_own_camera_occlusion_never_confirms_clear():
    cam = E.OwnCamera(801, "HEM1", 1000.0, 2000.0, 0.2)
    cam.occ[:] = True
    assert cam.moving(lambda t: False, 1500.0)
    cam.occ[:] = False
    cam.miss[:] = False
    cam.false[:] = False
    assert not cam.moving(lambda t: False, 1500.0)
    assert cam.moving(lambda t: t > 1499.5, 1500.0)


def test_occlusion_is_never_evidence_of_a_transfer():
    """E-W6: the walkway rule counts a transfer as seen only on visible motion, never on an occluded reading."""
    cam = E.OwnCamera(801, "WALKWAY", 1000.0, 2000.0, 0.2)
    cam.miss[:] = False
    cam.false[:] = False
    cam.occ[:] = True
    assert cam.moving(lambda t: False, 1500.0) and not cam.saw_motion(lambda t: True, 1500.0)
    cam.occ[:] = False
    assert cam.saw_motion(lambda t: True, 1500.0)
    assert not cam.saw_motion(lambda t: t < 1499.5, 1500.0)     # needs all of the last 3 samples


def test_mission_score_is_zero_after_any_trip_or_hold():
    m = E.MissionResult(0.0, targets=[{"target": "S10", "capture": "passed", "entry": "clean"},
                                      {"target": "HEM1", "capture": "passed", "entry": "declined"}])
    s = m.summary()
    assert (s["captures_passed"], s["entries_clean"], s["entries_declined"], s["Y"]) == (2, 1, 1, 3)
    m.holds = 1
    assert m.summary()["Y"] == 0


def test_reactive_idle_estimate_uses_the_last_runs():
    assert E._idle_estimate([1.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0]) == pytest.approx(float(np.median([50, 50, 2, 2, 2])))
