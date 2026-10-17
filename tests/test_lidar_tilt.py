"""R-W5: the 2D lidar's scan plane tilts with the true body roll / pitch (ros2/intercycle_sim/intercycle_sim/sensors.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"))

from intercycle_sim import movers as M  # noqa: E402
from intercycle_sim import sensors  # noqa: E402

WALL = [{"id": "w", "cls": "wall", "cx": 5.0, "cy": 0.0, "w": 0.2, "d": 10.0, "yaw": 0.0, "h": 4.0}]
AHEAD = 180                                   # index of angle 0 (angles start at -pi)


def rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body -> world rotation, z-y-x (yaw, pitch, roll) as quat_to_R of an Isaac quaternion."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def lidar(boxes=WALL, **kw):
    return sensors.LidarModel(sensors.BoxCaster(boxes), n=360, sigma=0.0, dropout=0.0, mount=(0.0, 0.0, 0.2), **kw)


def test_level_body_gives_the_old_rays_exactly():
    lid = lidar()
    parts = M.parts_for([{"id": "p1", "cls": "person"}])
    prims = M.primitives(parts, M.part_poses(parts, {"p1": (2.0, 1.0, 0.0)}))
    for yaw in (0.0, 0.7, -2.0):
        a = lid.true_ranges(0.0, 0.0, 0.6, yaw, prims)
        b = lid.true_ranges(0.0, 0.0, 0.6, yaw, prims, R_wb=rot(0.0, 0.0, yaw))
        assert np.array_equal(a, b)
    assert np.allclose(sensors.quat_to_R(quat(0.03, -0.02, 0.4)), rot(0.03, -0.02, 0.4))


def test_pitched_down_rays_hit_the_floor_and_up_rays_pass_over():
    lid = lidar(boxes=[])
    pitch = math.radians(3.0)                  # nose down = positive pitch about +y
    r = lid.true_ranges(0.0, 0.0, 0.6, 0.0, None, R_wb=rot(0.0, pitch, 0.0))
    oz = 0.6 + 0.2 * math.cos(pitch)                                       # the mount tilts with the body too
    assert r[AHEAD] == pytest.approx(oz / math.sin(pitch), rel=1e-6)      # floor return 15.3 m ahead
    assert r[0] == lid.rmax                                                # behind the plane rises: no return
    assert lidar(boxes=[], levelled=True).true_ranges(0.0, 0.0, 0.6, 0.0, None, R_wb=rot(0.0, pitch, 0.0))[AHEAD] == \
        lid.rmax                                                            # the old levelled mount sees nothing


def test_tilt_changes_what_low_and_high_bodies_return():
    low = [{"id": "pallet", "cls": "x", "cx": 4.0, "cy": 0.0, "w": 1.0, "d": 1.0, "yaw": 0.0, "h": 0.7}]
    lid = lidar(boxes=low)
    assert lid.true_ranges(0.0, 0.0, 0.6, 0.0, None)[AHEAD] == lid.rmax    # level plane at 0.8 m: above the pallet
    pitch = math.radians(2.0)
    r = lid.true_ranges(0.0, 0.0, 0.6, 0.0, None, R_wb=rot(0.0, pitch, 0.0))[AHEAD]
    ox = 0.2 * math.sin(pitch)
    assert r == pytest.approx((3.5 - ox) / math.cos(pitch), rel=1e-6)      # now the pallet's front face (z ~ 0.68 m)
    # a wall top at 0.9 m, 10 m away: pitched up 2 deg the ray is at 0.8 + 9.9 tan 2 = 1.15 m there and passes over
    short = [{"id": "w", "cls": "x", "cx": 10.0, "cy": 0.0, "w": 0.2, "d": 10.0, "yaw": 0.0, "h": 0.9}]
    lid2 = lidar(boxes=short)
    assert lid2.true_ranges(0.0, 0.0, 0.6, 0.0, None)[AHEAD] == pytest.approx(9.9)
    assert lid2.true_ranges(0.0, 0.0, 0.6, 0.0, None, R_wb=rot(0.0, -pitch, 0.0))[AHEAD] == lid2.rmax


def test_tilted_movers_and_rolled_scan():
    lid = lidar(boxes=[])
    parts = M.parts_for([{"id": "p1", "cls": "person"}])                 # r 0.3, 1.75 m tall
    prims = M.primitives(parts, M.part_poses(parts, {"p1": (3.0, 0.0, 0.0)}))
    roll = math.radians(4.0)
    r = lid.true_ranges(0.0, 0.0, 0.6, 0.0, prims, R_wb=rot(roll, 0.0, 0.0))
    assert r[AHEAD] == pytest.approx(2.7, abs=1e-3)                       # roll leaves the forward ray level
    side = r[90]                                                           # -90 deg (right): rolled down -> floor
    assert side == pytest.approx((0.6 + 0.2 * math.cos(roll)) / math.sin(roll), rel=1e-6)
    # elevated primitive (zmin column): a ram 1.0-2.0 m above the floor is passed under by a level plane at 0.8 m
    ram = {"circles": np.array([[3.0, 0.0, 0.3, 2.0, 1.0]]), "boxes": np.zeros((0, 6))}
    assert lid.true_ranges(0.0, 0.0, 0.6, 0.0, ram)[AHEAD] == lid.rmax     # level: passes under
    up = math.radians(5.0)
    ox = -0.2 * math.sin(up)
    assert lid.true_ranges(0.0, 0.0, 0.6, 0.0, ram, R_wb=rot(0.0, -up, 0.0))[AHEAD] == \
        pytest.approx((2.7 - ox) / math.cos(up), rel=1e-6)                 # pitched up 5 deg: z = 1.04 m there


def test_sensor_suite_passes_the_true_attitude():
    cast = sensors.BoxCaster(WALL)
    suite = sensors.SensorSuite(cast, 0.02, seed=0, lidar_kw={"sigma": 0.0, "dropout": 0.0})
    lvl = suite.frame(0.0, [0.0, 0.0, 0.6], quat(0, 0, 0), [0, 0, 0], [0, 0, 0], [0, 0, 0], None)["scan"]["ranges"]
    suite2 = sensors.SensorSuite(cast, 0.02, seed=0, lidar_kw={"sigma": 0.0, "dropout": 0.0})
    tl = suite2.frame(0.0, [0.0, 0.0, 0.6], quat(0, math.radians(8.0), 0), [0, 0, 0], [0, 0, 0], [0, 0, 0],
                      None)["scan"]["ranges"]
    assert lvl[AHEAD] == pytest.approx(4.9 - 0.30, abs=1e-6)               # default mount 0.30 m ahead
    p8 = math.radians(8.0)
    ox = 0.30 * math.cos(p8) + 0.20 * math.sin(p8)
    assert tl[AHEAD] == pytest.approx((4.9 - ox) / math.cos(p8), rel=1e-6)  # pitched 8 deg down: lower on the wall
