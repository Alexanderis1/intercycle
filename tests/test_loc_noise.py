"""R-W4 (kept part): start-pose noise and unmapped clutter (ros2/intercycle_sim/intercycle_sim/loc_noise.py)."""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"))

from intercycle_sim import loc_noise as L  # noqa: E402
from intercycle_sim import sensors, world  # noqa: E402

RES = 0.1


def room_files():
    """20 x 10 m room, a wall across x = 10 with one 1.6 m door at y 4.2-5.8."""
    boxes = [{"id": "s", "cls": "wall", "cx": 10.0, "cy": 0.05, "w": 20.0, "d": 0.1, "yaw": 0.0, "h": 4.0},
             {"id": "n", "cls": "wall", "cx": 10.0, "cy": 9.95, "w": 20.0, "d": 0.1, "yaw": 0.0, "h": 4.0},
             {"id": "w", "cls": "wall", "cx": 0.05, "cy": 5.0, "w": 0.1, "d": 10.0, "yaw": 0.0, "h": 4.0},
             {"id": "e", "cls": "wall", "cx": 19.95, "cy": 5.0, "w": 0.1, "d": 10.0, "yaw": 0.0, "h": 4.0},
             {"id": "m1", "cls": "wall", "cx": 10.0, "cy": 2.1, "w": 0.2, "d": 4.2, "yaw": 0.0, "h": 4.0},
             {"id": "m2", "cls": "wall", "cx": 10.0, "cy": 7.9, "w": 0.2, "d": 4.2, "yaw": 0.0, "h": 4.0}]
    nx, ny = 200, 100
    occ = np.zeros((nx, ny), bool)
    cx, cy = (np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES
    for b in boxes:
        occ |= (np.abs(cx[:, None] - b["cx"]) <= b["w"] / 2) & (np.abs(cy[None, :] - b["cy"]) <= b["d"] / 2)
    V, F = world.boxes_to_mesh(boxes)
    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=V, faces=F)
    return {"boxes.json": json.dumps(boxes).encode(), "mesh.npz": buf.getvalue(),
            "factory_l0.usda": world.mesh_to_usda(V, F).encode(), "map.pgm": world.occupancy_pgm(occ),
            "map.yaml": world.map_yaml(RES, 0.0, 0.0), "world.json": json.dumps({"n_boxes": len(boxes)}).encode()}


KEEP = [(2.0, 5.0), (17.0, 5.0), (9.0, 5.0), (11.0, 5.0)]       # dock, a far target, both sides of the door


def test_perturb_start_is_seeded_bounded_and_collision_free():
    boxes = json.loads(room_files()["boxes.json"])
    a, b = L.perturb_start((2.0, 5.0, 0.3), 7, boxes), L.perturb_start((2.0, 5.0, 0.3), 7, boxes)
    assert a == b and a["belief"] == [2.0, 5.0, 0.3]
    errs = np.array([L.perturb_start((2.0, 5.0, 0.3), s, boxes)["error"] for s in range(200)])
    assert (np.abs(errs[:, :2]) <= 0.5).all() and (np.abs(errs[:, 2]) <= math.radians(10.0) + 1e-12).all()
    assert np.abs(errs[:, :2]).max() > 0.4 and np.abs(errs[:, 2]).max() > math.radians(8.0)    # the full range is used
    near_wall = L.perturb_start((0.7, 5.0, 0.0), 3, boxes)
    assert L._box_gap(near_wall["true"][0], near_wall["true"][1], boxes) > L.FOOT_HALF[1]
    for s in range(100):                                         # every accepted start: footprint clear at its heading
        st = L.perturb_start((0.75, 5.0, math.pi / 4), s, boxes)
        assert not L.footprint_hits(*st["true"], boxes, clear_m=L.FOOT_CLEAR_M - 1e-9)
    with pytest.raises(RuntimeError):
        L.perturb_start((10.0, 2.0, 0.0), 1, boxes)            # inside the wall: no valid pose, fail closed


def test_clutter_is_in_the_world_but_not_in_the_map():
    f0 = room_files()
    f1, items = L.add_unmapped_clutter(f0, seed=11, n=8, keep_clear=KEEP)
    assert f1["map.pgm"] == f0["map.pgm"] and f1["map.yaml"] == f0["map.yaml"]
    b0, b1 = json.loads(f0["boxes.json"]), json.loads(f1["boxes.json"])
    assert len(b1) == len(b0) + 8 and all(b["cls"].startswith("clutter_") for b in b1[len(b0):])
    V, F = world.load_mesh(io.BytesIO(f1["mesh.npz"]))      # type: ignore[arg-type]
    assert V.shape[0] == 8 * len(b1) and F.shape[0] == 12 * len(b1)
    assert f1["factory_l0.usda"].count(b"(") > f0["factory_l0.usda"].count(b"(")
    meta = json.loads(f1["world.json"])
    assert meta["unmapped_clutter"]["in_map"] is False and meta["n_boxes"] == len(b1)
    assert all(it["h"] > 0.85 for it in items)                 # above the 2D lidar plane (about 0.8 m)
    for it in items:
        r = math.hypot(it["w"], it["d"]) / 2
        assert all(math.hypot(it["cx"] - x, it["cy"] - y) >= L.CLEAR_R + r - 1e-9 for x, y in KEEP)
    again, _ = L.add_unmapped_clutter(f0, seed=11, n=8, keep_clear=KEEP)
    other, _ = L.add_unmapped_clutter(f0, seed=12, n=8, keep_clear=KEEP)
    assert again["boxes.json"] == f1["boxes.json"] and other["boxes.json"] != f1["boxes.json"]


def test_clutter_never_cuts_the_way_and_respects_avoid_rects():
    f0 = room_files()
    zone = (12.0, 1.0, 16.0, 9.0)
    f1, items = L.add_unmapped_clutter(f0, seed=5, n=10, keep_clear=KEEP, avoid_rects=[zone])
    occ, res, x0, y0 = L.read_map(f0)
    grid = occ.copy()
    for it in items:
        ii, jj = L._footprint_cells(it["cx"], it["cy"], it["w"] / 2, it["d"] / 2, it["yaw"], res, x0, y0, grid.shape)
        grid[ii, jj] = True
        r = math.hypot(it["w"], it["d"]) / 2
        assert not (zone[0] - r <= it["cx"] <= zone[2] + r and zone[1] - r <= it["cy"] <= zone[3] + r)
    assert L._connected(grid, res, x0, y0, KEEP)                # the door stays passable for the robot disc
    with pytest.raises(ValueError):
        L.add_unmapped_clutter(f0, seed=5, n=1, keep_clear=[])


def test_lidar_sees_the_clutter_the_map_does_not_have():
    f0 = room_files()
    f1, items = L.add_unmapped_clutter(f0, seed=11, n=8, keep_clear=KEEP)
    it = items[0]
    ox, oy = KEEP[0]
    psi = math.atan2(it["cy"] - oy, it["cx"] - ox)
    lid0 = sensors.LidarModel(sensors.BoxCaster(json.loads(f0["boxes.json"])), n=360, sigma=0.0, dropout=0.0,
                              mount=(0.0, 0.0, 0.2))
    lid1 = sensors.LidarModel(sensors.BoxCaster(json.loads(f1["boxes.json"])), n=360, sigma=0.0, dropout=0.0,
                              mount=(0.0, 0.0, 0.2))
    r0 = lid0.true_ranges(ox, oy, 0.6, psi, None)[180]
    r1 = lid1.true_ranges(ox, oy, 0.6, psi, None)[180]
    assert r1 < math.hypot(it["cx"] - ox, it["cy"] - oy) + 1e-6 and r1 < r0     # a return where the map is free


def test_oriented_footprint_catches_the_corner_a_centre_check_misses():
    """Audit fix 8: corner radius hypot(0.50, 0.32) = 0.594 m > the old centre clearance 0.55 m."""
    wall = [{"id": "w", "cls": "wall", "cx": -0.05, "cy": 0.0, "w": 0.1, "d": 10.0, "yaw": 0.0, "h": 4.0}]   # face x = 0
    corner = math.atan2(0.32, 0.50)
    x = 0.57                                                    # centre 0.57 m from the face: passed the old check
    assert L._box_gap(x, 0.0, wall) > 0.55
    assert L.footprint_hits(x, 0.0, math.pi + corner, wall, clear_m=0.0)   # a corner points at the wall: overlap
    assert not L.footprint_hits(x, 0.0, math.pi / 2, wall, clear_m=0.0)    # side-on: 0.32 m half-width, clear
    assert L.footprint_hits(0.53, 0.0, 0.0, wall, clear_m=0.05)            # 0.03 m gap < the 0.05 m clearance
    assert not L.footprint_hits(0.56, 0.0, 0.0, wall, clear_m=0.05)

