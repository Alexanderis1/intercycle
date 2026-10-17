"""R1: pure-python pieces of the legged-robot sim bridge (world mesh, sensor models, lockstep protocol)."""
from __future__ import annotations

import math
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"))

from intercycle_sim import movers as M  # noqa: E402
from intercycle_sim import protocol, sensors, world  # noqa: E402


@pytest.fixture(scope="module")
def files():
    return world.build_world_files()


def test_world_files(files, tmp_path):
    world.write_world(files, str(tmp_path))
    boxes = world.load_boxes(str(tmp_path / "boxes.json"))
    V, F = world.load_mesh(str(tmp_path / "mesh.npz"))
    assert len(boxes) > 100
    assert V.shape == (8 * len(boxes), 3) and F.shape == (12 * len(boxes), 3)
    assert F.max() < len(V)
    usda = files["factory_l0.usda"].decode()
    assert usda.startswith("#usda 1.0") and "PhysicsCollisionAPI" in usda
    head = files["map.pgm"][:20].split(b"\n")
    assert head[0] == b"P5" and head[1] == b"1240 520"
    assert b"resolution: 0.1" in files["map.yaml"]


def test_mesh_triangles_point_outward():
    V, F = world.boxes_to_mesh([{"cx": 0.0, "cy": 0.0, "w": 2.0, "d": 1.0, "yaw": 0.3, "h": 1.0}])
    c = V.mean(0)
    for f in F:
        a, b, d = V[f]
        n = np.cross(b - a, d - a)
        assert n @ ((a + b + d) / 3 - c) > 0


def test_map_matches_boxes(files):
    # a wall box must be occupied in the PGM; aisle centre free
    pgm = files["map.pgm"]
    hdr_len = len(b"P5\n1240 520\n255\n")
    img = np.frombuffer(pgm[hdr_len:], np.uint8).reshape(520, 1240)

    def px(x, y):
        return img[519 - int((y + 2.0) / 0.1), int((x + 2.0) / 0.1)]

    assert px(0.15, 24.0) == 0          # west wall
    assert px(40.0, 32.0) == 254        # north aisle


def test_lidar_static_and_dynamic():
    boxes = [{"id": "w", "cls": "wall", "cx": 5.0, "cy": 0.0, "w": 0.2, "d": 10.0, "yaw": 0.0, "h": 4.0}]
    cast = sensors.BoxCaster(boxes)
    lid = sensors.LidarModel(cast, n=360, sigma=0.0, dropout=0.0, mount=(0.0, 0.0, 0.2))
    r = lid.true_ranges(0.0, 0.0, 0.6, 0.0, None)
    i0 = 180                                       # angle 0 (ahead) is index n/2 (angles start at -pi)
    assert abs(lid.angles[i0]) < 1e-9
    assert r[i0] == pytest.approx(4.9, abs=1e-6)
    assert r[0] == lid.rmax                        # behind: nothing
    # a person 2 m ahead occludes the wall
    parts = M.parts_for([{"id": "p1", "cls": "person"}])
    prims = M.primitives(parts, M.part_poses(parts, {"p1": (2.0, 0.0, 0.0)}))
    r2 = lid.true_ranges(0.0, 0.0, 0.6, 0.0, prims)
    assert r2[i0] == pytest.approx(1.7, abs=1e-6)
    # rotate the robot by +90 deg: the wall is now on the right (angle -90 deg -> index 90)
    r3 = lid.true_ranges(0.0, 0.0, 0.6, math.pi / 2, None)
    assert r3[90] == pytest.approx(4.9, abs=1e-6)


def test_lidar_ignores_low_bodies_and_noise_limits():
    boxes = [{"id": "low", "cls": "x", "cx": 3.0, "cy": 0.0, "w": 1.0, "d": 1.0, "yaw": 0.0, "h": 0.3}]
    lid = sensors.LidarModel(sensors.BoxCaster(boxes), n=36, sigma=0.01, dropout=0.0)
    out = lid.scan(0.0, 0.0, 0.6, 0.0, None)
    assert all(math.isinf(v) for v in out)


def test_ray_box_rotated():
    b = np.array([[3.0, 0.0, 0.5, 0.5, math.pi / 4, 2.0]])
    r = sensors.ray_boxes(0.0, 0.0, np.array([1.0]), np.array([0.0]), b, 0.5, 20.0)
    assert r[0] == pytest.approx(3.0 - 0.5 * math.sqrt(2), abs=1e-6)


def test_odom_drifts_but_tracks():
    od = sensors.OdomModel(rng=np.random.default_rng(0))
    for _ in range(500):
        o = od.step(0.02, 0.5, 0.0, 0.0)
    assert 3.5 < o["x"] < 6.5 and abs(o["y"]) < 1.5


def test_imu_gravity_at_rest():
    imu = sensors.ImuModel(rng=np.random.default_rng(0))
    for _ in range(3):
        m = imu.step(0.02, np.eye(3), np.zeros(3), np.zeros(3))
    assert m["acc"][2] == pytest.approx(9.81, abs=0.3)


def test_camera_model_occlusion_and_frustum():
    boxes = [{"id": "w", "cls": "wall", "cx": 2.0, "cy": 0.0, "w": 0.2, "d": 0.6, "yaw": 0.0, "h": 4.0}]
    cam = sensors.CameraModel(sensors.BoxCaster(boxes), latency_s=0.0, p_detect=1.0)
    xyz = np.array((0.0, 0.0, 1.0))
    obs = cam.observe(0.0, xyz, 0.0, 0.0, [{"id": "a", "xyz": (4.0, 0.0, 1.0)}, {"id": "b", "xyz": (4.0, 2.0, 1.0)},
                                           {"id": "c", "xyz": (-4.0, 0.0, 1.0)}])
    by = {o["id"]: o for o in obs}
    assert by["a"]["in_frustum"] and by["a"]["occluded"]
    assert by["b"]["in_frustum"] and not by["b"]["occluded"] and by["b"]["detected"]
    assert not by["c"]["in_frustum"]


def test_protocol_roundtrip(tmp_path):
    path = str(tmp_path / "s.sock") if hasattr(socket, "AF_UNIX") else None
    if path is None or sys.platform == "win32":
        a, b = socket.socketpair()
    else:
        srv = protocol.listen(path)
        a = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        a.connect(path)
        b, _ = srv.accept()
    msg = {"type": "frame", "t": 0.02, "scan": {"ranges": [1.0, float("inf")]}}

    def echo():
        m = protocol.recv_msg(b)
        protocol.send_msg(b, {"type": "cmd", "vx": 0.5, "vy": 0.0, "wz": 0.1, "age": 0.0, "echo": m["t"]})

    th = threading.Thread(target=echo)
    th.start()
    protocol.send_msg(a, msg)
    rep = protocol.recv_msg(a)
    th.join()
    assert rep["vx"] == 0.5 and rep["echo"] == 0.02
    a.close()
    b.close()


def test_sensor_suite_frame_has_no_truth():
    suite = sensors.SensorSuite(sensors.BoxCaster([]), 0.02, seed=1, start=(4.0, 32.0, 0.0))
    fr = suite.frame(0.02, [4.0, 32.0, 0.6], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], None)
    assert set(fr) == {"t", "imu", "odom", "scan"}
    assert set(fr["odom"]) == {"x", "y", "psi", "vx", "vy", "wz", "v_var", "w_var"}
    assert fr["odom"]["x"] != 4.0                    # odom frame starts at 0, not at the true world pose
