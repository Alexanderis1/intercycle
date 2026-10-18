"""R-W13: fixed ISO 13855 protective field (ros2/intercycle_sim/intercycle_sim/safety_scanner.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"))

from intercycle_sim import movers as MV  # noqa: E402
from intercycle_sim import safety_scanner as SS  # noqa: E402

GO = (0.5, 0.0, 0.0)
STOP = (0.0, 0.0, 0.0)


def person(x, y, r=0.3, h=1.75):
    return {"circles": np.array([[x, y, r, h]]), "boxes": np.zeros((0, 6))}


def test_iso13855_distance_from_stated_k_t_c():
    assert SS.iso13855_c_mm(300.0) == pytest.approx(1080.0)
    assert SS.iso13855_c_mm(1000.0) == pytest.approx(850.0)          # never below 850 mm
    sp = SS.FieldSpec()
    assert sp.k_mps == pytest.approx(1.6 + math.hypot(0.7, 0.3))
    assert sp.t_s == pytest.approx(0.08 + 0.02 + math.hypot(0.7, 0.3) / 1.5) == pytest.approx(0.6077, abs=1e-4)
    assert sp.c_m == pytest.approx(1.08)
    assert sp.s_m == pytest.approx(2.5152, abs=1e-3)                  # 2.3616 * 0.6077 + 1.08
    assert sp.s_m > 2.362 * 0.60 + 1.08                              # the old rounded-down T gave a smaller S
    rec = sp.record()
    assert rec["s_m"] == pytest.approx(sp.s_m, abs=1e-4) and rec["sources"]["t_stop"] == "ASSUMED"


def test_person_in_field_stops_a_moving_robot_and_releases_after_clear():
    f = SS.ProtectiveField()
    reach = 0.50 + f.spec.s_m                                        # field half-length ahead of the body centre
    assert f.gate(0.0, 0.02, 0.0, 0.0, 0.0, GO, 0.5, person(reach + 0.3 + 0.2, 0.0)) == GO     # just outside
    assert f.gate(0.02, 0.02, 0.0, 0.0, 0.0, GO, 0.5, person(reach + 0.3 - 0.2, 0.0)) == STOP  # intrudes
    assert f.n_stops == 1 and f.active
    t = 0.04
    while t < 1.0:                                                   # field clear, but not yet for release_s
        assert f.gate(t, 0.02, 0.0, 0.0, 0.0, GO, 0.0, person(10.0, 0.0)) == STOP
        t += 0.02
    t2 = t
    while f.active and t2 < 3.0:
        f.gate(t2, 0.02, 0.0, 0.0, 0.0, GO, 0.0, person(10.0, 0.0))
        t2 += 0.02
    assert not f.active and t2 - 0.04 >= f.spec.release_s - 0.03
    assert f.n_stops == 1 and f.stopped_s == pytest.approx(t2 - 0.02, abs=0.05)
    s = f.summary()
    assert s["n_protective_stops"] == 1 and s["events"][0]["kinds"] == ["circle"]


def test_field_is_around_the_whole_body_and_follows_heading():
    f = SS.ProtectiveField()
    side = 0.32 + f.spec.s_m
    assert f.intruders(0.0, 0.0, 0.0, person(0.0, -(side + 0.1)))          # beside the robot (r 0.3 overlaps)
    assert not f.intruders(0.0, 0.0, 0.0, person(0.0, -(side + 0.4)))
    # rotate the robot 90 deg: the long axis now points along y
    long_ = 0.50 + f.spec.s_m
    assert f.intruders(0.0, 0.0, math.pi / 2, person(0.0, long_ + 0.2))
    assert not f.intruders(0.0, 0.0, 0.0, person(0.0, long_ + 0.2))


def test_standing_robot_is_not_stopped_but_cannot_start():
    f = SS.ProtectiveField()
    near = person(1.5, 0.0)
    assert f.gate(0.0, 0.02, 0.0, 0.0, 0.0, STOP, 0.0, near) == STOP and f.n_stops == 0 and not f.active
    assert f.gate(0.02, 0.02, 0.0, 0.0, 0.0, GO, 0.0, near) == STOP and f.n_stops == 1     # blocked start counts


def test_vehicle_boxes_sat_and_low_bodies():
    f = SS.ProtectiveField()
    reach = 0.50 + f.spec.s_m
    # forklift box (hl 0.95, hw 0.55) rotated 45 deg, its corner ((0.95 + 0.55) / sqrt 2 = 1.06 m) 0.2 m into the field
    box = {"circles": np.zeros((0, 4)), "boxes": np.array([[reach + 0.86, 0.0, 0.95, 0.55, math.pi / 4, 2.2]])}
    assert f.intruders(0.0, 0.0, 0.0, box)
    # a long thin box 5 cm outside: its bounding circle would overlap, the exact test does not
    thin = {"circles": np.zeros((0, 4)), "boxes": np.array([[reach + 1.0, 0.0, 0.95, 0.1, 0.0, 2.2]])}
    assert not f.intruders(0.0, 0.0, 0.0, thin)
    far = {"circles": np.zeros((0, 4)), "boxes": np.array([[reach + 2.5, 0.0, 0.95, 0.55, math.pi / 4, 2.2]])}
    assert not f.intruders(0.0, 0.0, 0.0, far)
    low = {"circles": np.array([[1.0, 0.0, 0.3, 0.2]]), "boxes": np.zeros((0, 6))}   # below the 0.3 m plane
    assert not f.intruders(0.0, 0.0, 0.0, low)


def test_only_people_and_vehicles_are_scanned():
    parts = MV.parts_for([{"id": "p1", "cls": "person"}, {"id": "f1", "cls": "forklift"},
                          {"id": "MC_arm_00", "cls": "cyl:0.4:2.0"}, {"id": "t1", "cls": "tugger_train"}])
    idx = SS.scanned_part_index(parts)
    assert [parts[k].mover_id for k in idx] == ["p1", "f1", "f1", "t1", "t1", "t1", "t1"]


def test_twin_runs_get_the_scanner_by_default_and_cannot_turn_it_off():
    """Audit fix 7: no twin run goes ahead without the R-W13 field."""
    assert SS.scanner_enabled({}, {"twin": True}) is True                   # twin world: on by default
    assert SS.scanner_enabled({"twin": True}, {}) is True                   # twin cfg: on by default
    assert SS.scanner_enabled({"twin": True, "safety_scanner": True}, None) is True
    with pytest.raises(ValueError):
        SS.scanner_enabled({"safety_scanner": False}, {"twin": True})
    with pytest.raises(ValueError):
        SS.scanner_enabled({"twin": True, "safety_scanner": False}, None)
    assert SS.scanner_enabled({}, {}) is False                              # body shop / A-lite: unchanged default
    assert SS.scanner_enabled({"safety_scanner": True}, None) is True


def test_assert_scanner_ran_for_the_runners():
    f = SS.ProtectiveField()
    assert SS.assert_scanner_ran({"safety_scanner": f.summary()})["n_protective_stops"] == 0
    for bad in ({}, {"safety_scanner": None}, {"safety_scanner": {"n_protective_stops": 0}}):
        with pytest.raises(ValueError):
            SS.assert_scanner_ran(bad)


def test_runner_uses_the_twin_default():
    """The sim runner decides through scanner_enabled with the world's world.json (no ROS / Isaac needed to check)."""
    src = (Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim" / "intercycle_sim" / "runner.py").read_text(
        encoding="utf-8")
    assert "if scanner_enabled(cfg, wmeta):" in src and 'os.path.join(wd, "world.json")' in src
    assert 'cfg.get("safety_scanner", False)' not in src
