"""S5a item 2 (G1): the Isaac smoke on compile(site_v0) - world bytes, mission cfg and the walked verdict."""
from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("isaac_smoke_twin", REPO / "scripts" / "isaac_smoke_twin.py")
assert _SPEC is not None and _SPEC.loader is not None
S = importlib.util.module_from_spec(_SPEC)
sys.modules["isaac_smoke_twin"] = S
_SPEC.loader.exec_module(S)


@pytest.fixture(scope="module")
def z9():
    return S.compile_zone("z9")


def test_world_bytes_match_the_committed_twin_zone(z9):
    """The bytes shipped to Isaac are world.py's files for compile(site_v0) zone z9, the same as twin/z9/."""
    w = S.zone_world_bytes(z9)
    assert {"boxes.json", "factory_l0.usda", "mesh.npz", "map.pgm", "map.yaml", "world.json"} <= set(w)
    d = REPO / "twin" / "z9"
    assert json.loads(w["boxes.json"]) == json.loads((d / "boxes.json").read_bytes())
    assert w["map.pgm"] == (d / "map.pgm").read_bytes()
    assert w["map.yaml"].replace(b"\r\n", b"\n") == (d / "map.yaml").read_bytes().replace(b"\r\n", b"\n")
    assert w["factory_l0.usda"].replace(b"\r\n", b"\n") == (d / "static.usda").read_bytes().replace(b"\r\n", b"\n")
    a, b = np.load(io.BytesIO(w["mesh.npz"])), np.load(d / "mesh.npz")
    assert np.array_equal(a["vertices"], b["vertices"]) and np.array_equal(a["faces"], b["faces"])
    meta = json.loads(w["world.json"])
    assert meta["dock"] == [1.0, 1.0] and meta["n_boxes"] > 0 and not meta["empty"]


def test_mission_goes_dock_checkpoint_dock_in_the_ros_frame(z9):
    cfg = S.mission_cfg(z9, "cp_P9a_side", tag="t")
    sx, sy, spsi = cfg["start"]
    assert (sx, sy) == (1.0, 1.0) and spsi == pytest.approx(math.pi / 2)          # plan yaw 0 faces +z = ROS +y
    (cx, cy, cpsi), (bx, by, bpsi) = cfg["goal"]
    assert (cx, cy) == z9.graph.pos["P_cp_P9a_side"]                                   # ROS x = plan x, y = plan z
    assert cpsi == pytest.approx(math.pi)                                             # plan yaw -pi/2 faces -x
    assert (bx, by, bpsi) == (sx, sy, spsi)
    assert cfg["nav2"] and cfg["mode"] == "nav" and cfg["twin"] is True
    assert not cfg.get("movers") and not cfg.get("mover_source")                     # static scene, no movers
    assert cfg["seconds"] >= 2 * cfg["timeout_s"]


def test_mission_refuses_a_checkpoint_of_another_zone(z9):
    with pytest.raises(KeyError):
        S.mission_cfg(z9, "cp_P14a_ram", tag="t")


def _rows(points):
    return [{"t": 0.1 * k, "x": x, "y": y, "z": 0.6} for k, (x, y) in enumerate(points)]


def _res(path_len=22.0, fell=False, legs=("SUCCEEDED", "SUCCEEDED"), error=None):
    return {"error": error,
            "sim": {"path_length_m": path_len, "fell": fell, "end_reason": "stop_file", "sim_s": 80.0, "rtf": 0.8,
                    "n_contact_steps": 0},
            "goal": {"result": "SUCCEEDED" if all(r == "SUCCEEDED" for r in legs) else "FAILED",
                     "legs": [{"result": r} for r in legs]}}


CP, DOCK = (9.2, 7.25), (1.0, 1.0)
GOOD = _rows([(1.0, 1.0), (5.0, 3.0), (9.15, 7.2), (5.0, 3.0), (1.1, 1.05)])


def test_verdict_walked():
    v = S.smoke_verdict(_res(), GOOD, CP, DOCK)
    assert v["walked"], v
    assert v["checks"]["reached_checkpoint"] and v["checks"]["back_at_dock"]


@pytest.mark.parametrize("res,rows,failed", [
    (_res(fell=True), GOOD, "no_fall"),
    (_res(path_len=3.0), GOOD, "path_length"),
    (_res(legs=("SUCCEEDED", "FAILED")), GOOD, "nav2_legs_succeeded"),
    (_res(legs=("SUCCEEDED",)), GOOD, "nav2_legs_succeeded"),
    (_res(), _rows([(1.0, 1.0), (8.0, 6.0), (1.0, 1.0)]), "reached_checkpoint"),
    (_res(), _rows([(1.0, 1.0), (9.2, 7.25), (4.0, 4.0)]), "back_at_dock"),
    (_res(error="RuntimeError('boom')"), GOOD, "no_episode_error"),
    (_res(), [], "truth_log"),
])
def test_verdict_fails_closed(res, rows, failed):
    v = S.smoke_verdict(res, rows, CP, DOCK)
    assert not v["walked"] and not v["checks"][failed], v


def test_verdict_fails_closed_without_a_sim_summary():
    v = S.smoke_verdict({"error": None, "sim": None, "goal": None}, GOOD, CP, DOCK)
    assert not v["walked"]
