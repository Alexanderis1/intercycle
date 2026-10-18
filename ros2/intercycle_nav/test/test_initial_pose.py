"""Audit fix 4 (R-W4 start-pose noise): AMCL's initial pose carries the covariance of the start-pose draw, not zero.

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test/test_initial_pose.py
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO / "ros2" / "intercycle_sim"))

from intercycle_nav import timing as T  # noqa: E402


def test_covariance_matches_the_uniform_start_draw():
    c = T.initial_pose_covariance()
    assert len(c) == 36
    assert c[0] == pytest.approx(0.5 ** 2 / 3) == pytest.approx(0.0833, abs=1e-4)
    assert c[7] == pytest.approx(c[0])
    assert c[35] == pytest.approx(math.radians(10.0) ** 2 / 3) == pytest.approx(0.0102, abs=1e-4)
    assert all(v == 0.0 for i, v in enumerate(c) if i not in (0, 7, 35))
    assert c[0] > 0 and c[35] > 0                               # non-zero: the point of the fix
    with pytest.raises(ValueError):
        T.initial_pose_covariance(-0.1, 10.0)


def test_covariance_defaults_equal_the_simulator_start_noise():
    from intercycle_sim import loc_noise as L

    assert T.INIT_SPREAD_XY_M == L.START_DXY
    assert math.radians(T.INIT_SPREAD_YAW_DEG) == pytest.approx(L.START_DPSI)
    # the variance of the real draw (perturb_start, no boxes) agrees with the covariance sent to AMCL
    err = np.array([L.perturb_start((0.0, 0.0, 0.0), seed=s)["error"] for s in range(4000)])
    c = T.initial_pose_covariance()
    assert np.var(err[:, 0]) == pytest.approx(c[0], rel=0.08)
    assert np.var(err[:, 1]) == pytest.approx(c[7], rel=0.08)
    assert np.var(err[:, 2]) == pytest.approx(c[35], rel=0.08)


def test_planner_publishes_the_covariance_after_nav2_is_active():
    """No ROS on this host: check the planner's code path statically. run() must call spread_initial_pose() after
    waitUntilNav2Active() (BasicNavigator's own loop re-sends a zero covariance until then), and the message it
    publishes on initial_pose_pub is a PoseWithCovarianceStamped whose covariance is initial_pose_covariance(...)."""
    src = (PKG / "intercycle_nav" / "timing_planner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    run = ast.get_source_segment(src, fns["run"]) or ""
    assert run.index("waitUntilNav2Active()") < run.index("self.spread_initial_pose()")
    msg = ast.get_source_segment(src, fns["initial_pose_msg"]) or ""
    assert "PoseWithCovarianceStamped()" in msg and "m.pose.covariance = " in msg and "self.init_cov" in msg
    spread = ast.get_source_segment(src, fns["spread_initial_pose"]) or ""
    assert "initial_pose_pub.publish(" in spread
    init = ast.get_source_segment(src, fns["__init__"]) or ""
    assert "self.init_cov = initial_pose_covariance(" in init


def test_spread_defaults_to_zero_unless_the_episode_exports_it():
    """Fix round 2: an exact-start run (body-shop R3 batch) keeps AMCL on the exact pose; the spread comes from the
    episode's IC_INIT_SPREAD_* env (the start-noise range only when start noise was applied)."""
    src = (PKG / "intercycle_nav" / "timing_planner.py").read_text(encoding="utf-8")
    init = ast.get_source_segment(src, {n.name: n for n in ast.walk(ast.parse(src))
                                        if isinstance(n, ast.FunctionDef)}["__init__"]) or ""
    assert 'os.environ.get("IC_INIT_SPREAD_XY_M", "0")' in init
    assert 'os.environ.get("IC_INIT_SPREAD_YAW_DEG", "0")' in init
    assert "INIT_SPREAD_XY_M" not in init.replace("IC_INIT_SPREAD_XY_M", "")   # no 0.5 m default any more
    from intercycle_sim import episode as E

    assert E.planner_env({"spread": {"xy_m": 0.0, "yaw_deg": 0.0}, "belief": [4.0, 32.0, 0.0]})[
        "IC_INIT_SPREAD_XY_M"] == "0.0"


def test_ack_uses_the_components_actually_spread():
    from intercycle_nav import initial_pose as IP

    yaw_only = T.initial_pose_covariance(0.0, 10.0)
    got = [0.0] * 36
    got[35] = 0.009
    assert IP.spread_components(yaw_only) == [35]
    assert IP.ack_ok(yaw_only, got)                       # before: cov[0] > 1e-6 was required, so it never passed
    assert not IP.ack_ok(yaw_only, [0.0] * 36)
    full = T.initial_pose_covariance()
    got_xy = [0.0] * 36
    got_xy[0] = got_xy[7] = 0.05
    assert not IP.ack_ok(full, got_xy)                    # yaw was sent but AMCL's answer has no yaw spread
    got_xy[35] = 0.008
    assert IP.ack_ok(full, got_xy)
    assert IP.spread_components(T.initial_pose_covariance(0.0, 0.0)) == []
    assert not IP.ack_ok(T.initial_pose_covariance(0.0, 0.0), got_xy)
    src = (PKG / "intercycle_nav" / "timing_planner.py").read_text(encoding="utf-8")
    assert "ack_ok(self.init_cov, cov)" in src and "cov[0] > 1e-6" not in src
