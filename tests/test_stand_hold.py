"""R-W6: stand mode holds against the robot's own pose estimate, never the true pose (intercycle_sim/stand.py)."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "ros2" / "intercycle_sim"
sys.path.insert(0, str(SIM))

from intercycle_sim import sensors  # noqa: E402
from intercycle_sim.stand import StandHold  # noqa: E402

DT = 0.02
ZERO = (0.0, 0.0, 0.0)


def test_hold_starts_after_settle_and_corrects_toward_the_estimate_hold_pose():
    h = StandHold(DT)
    est = (1.0, 2.0, 0.0)
    for _ in range(9):                                   # < 0.2 s of zero command: no hold yet
        assert h.step(ZERO, est) == ZERO and h.hold is None
    h.step(ZERO, est)
    assert h.hold == est and h.n_holds == 1
    vx, vy, wz = h.step(ZERO, (1.1, 2.0, 0.0))           # the estimate says the base slid 10 cm forward
    assert vx == pytest.approx(-0.1) and vy == 0.0 and wz == 0.0
    assert h.step(ZERO, (1.01, 2.0, 0.0)) == ZERO        # inside the 2 cm deadband
    assert h.step((0.3, 0.0, 0.0), est) == (0.3, 0.0, 0.0) and h.hold is None     # a command releases it


def test_no_estimate_means_no_hold_never_truth():
    h = StandHold(DT)
    for _ in range(50):
        assert h.step(ZERO, None) == ZERO
    assert h.hold is None and h.n_holds == 0


def test_odometry_drift_moves_the_true_base():
    """Kinematic check: a base that is truly still while its odometry drifts gets pushed by the hold, so the TRUE pose
    drifts (the realism R-W6 asks for); with a perfect estimate it would not move at all."""
    rng = np.random.default_rng(4)
    od = sensors.OdomModel(rng=rng, v_bias_rw=0.02)       # exaggerated bias walk so 60 s shows it
    h = StandHold(DT)
    x = y = psi = 0.0
    vb = np.zeros(3)
    for _ in range(int(60 / DT)):
        o = od.step(DT, vb[0], vb[1], vb[2])
        cmd = h.step(ZERO, (o["x"], o["y"], o["psi"]))
        vb = np.array(cmd)
        c, s = math.cos(psi), math.sin(psi)
        x += (c * vb[0] - s * vb[1]) * DT
        y += (s * vb[0] + c * vb[1]) * DT
        psi += vb[2] * DT
    assert math.hypot(x, y) > 0.02                        # drift shows up in the true pose


def test_isaac_backend_stand_mode_reads_no_true_pose():
    src = (SIM / "intercycle_sim" / "isaac_backend.py").read_text(encoding="utf-8")
    body = re.search(r"def _stand\(self.*?(?=\n    def )", src, re.S)
    assert body is not None
    assert "self.state()" not in body.group(0) and "root_pos_w" not in body.group(0)
    assert "_stand_hold.step" in body.group(0)
    runner = (SIM / "intercycle_sim" / "runner.py").read_text(encoding="utf-8")
    assert 'be.set_estimate(fr["odom"]["x"], fr["odom"]["y"], fr["odom"]["psi"])' in runner
