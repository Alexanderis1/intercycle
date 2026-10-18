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
