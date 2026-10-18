"""twin/tasks.yaml (lane S2 TASKS): generated from site.yaml, 8 checkpoints (5 in z9, 3 in z14), zoom / thermal /
defect rate / detectability / window, train-only provenance, and the TwinSiteGraph compiles from it."""
from __future__ import annotations

import math
import sys
from typing import cast
from pathlib import Path

import numpy as np
import pytest

from twin import compile as C
from twin import inspect_model as IM
from twin import robot_camera as RC
from twin import tasks_gen as TG
from twin.provenance import assert_train_only, source_clips
from twin.robot_camera import DORI_INSPECT, FX
from twin.schemas import load
from twin.schemas.site import Site
from twin.schemas.tasks import Tasks

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "twin"
if str(ROOT / "ros2" / "intercycle_sim") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2" / "intercycle_sim"))


@pytest.fixture(scope="module")
def tasks() -> Tasks:
    return load(Tasks, TWIN / "tasks.yaml")


def test_tasks_yaml_is_current():
    assert TG.main(["--check"]) == 0


def test_eight_checkpoints_five_in_z9_three_in_z14(tasks):
    by: dict = {}
    for c in tasks.checkpoints:
        by.setdefault(c.zone, []).append(c)
    assert len(tasks.checkpoints) == 8 and len(by["z9"]) == 5 and len(by["z14"]) == 3


def test_inspection_model(tasks):
    i = tasks.inspection
    assert i.defect_rate == 0.25 and i.zoom_max == 20.0
    assert "1000 px/m" in i.detectability and "cos(theta)" in i.detectability
    assert i.capture_model and i.thermal_model
    assert i.window is not None and set(i.window.kinds) == {"press_stop", "stroke_gap"}
    assert "inside_s + margin_s" in i.window.definition
    assert any(c.zoom == 20.0 for c in tasks.checkpoints) and any(c.thermal for c in tasks.checkpoints)
    for z in ("z9", "z14"):
        assert any(c.inside_danger for c in tasks.checkpoints if c.zone == z)


def test_every_checkpoint_has_generated_candidates_and_provenance(tasks):
    for c in tasks.checkpoints:
        assert c.candidate_poses and all(p.source == "generated" for p in c.candidate_poses)
        assert c.source == "assumed" or c.clips


def test_provenance_is_train_only(tasks):
    assert assert_train_only(source_clips(tasks=tasks)) > 0


def test_graph_compiles_from_tasks_yaml(tasks):
    out = C.compile_paths(TWIN / "site.yaml", TWIN / "dynamics.json", TWIN / "tasks.yaml")
    for z in ("z9", "z14"):
        g = out[z].graph
        order = next(zt.mission_order for zt in tasks.zones if zt.zone == z)
        assert [t.id for t in g.mission()] == order
        for t in g.mission():
            cp = next(c for c in tasks.checkpoints if c.id == t.id)
            assert t.kind == ("machine" if cp.inside_danger else "point")


class _Zero:
    """Zero aim noise (the E-W3 oracle dev-check is geometric)."""

    def normal(self, *a, **k):
        return 0.0


def _gen(stub) -> np.random.Generator:
    return cast(np.random.Generator, stub)


class _Fixed:
    def __init__(self, v: float):
        self.v = v

    def normal(self, *a, **k):
        return self.v


class _Cast3D:
    """BoxCaster with exact 3-D rays (what the Warp mesh caster computes); the plant's numpy fallback is `cast`."""

    def __init__(self, bc):
        self.bc = bc

    def cast(self, origin, dirs, rmax):
        return self.bc.cast3d(origin, dirs, rmax)


@pytest.fixture(scope="module")
def compiled():
    return C.compile_paths(TWIN / "site.yaml", TWIN / "dynamics.json", TWIN / "tasks.yaml")


def _casters(rects):
    from intercycle_sim.sensors import BoxCaster  # pyright: ignore[reportMissingImports]
    from intercycle_sim.world import boxes_from_rects  # pyright: ignore[reportMissingImports]

    bc = BoxCaster(boxes_from_rects(rects))
    return bc, _Cast3D(bc)


def test_capture_passes_from_every_candidate_pose_against_the_compiled_static_world(tasks, compiled):
    """E-W3 oracle dev-check, offline: robot_camera.capture (zero aim noise, open movers, settled, still) from every
    candidate pose, camera at the real mount (cam_pose with the plan yaw converted, base z 0 and 0.3 m), against the
    compiled static boxes (press boxes 2 m + unseen-floor walls 4 m), with both the plant's numpy caster and an exact
    3-D caster: every capture passes Inspect (written formula, 3-D range and cos(theta)), and every thermal checkpoint
    passes the thermal capture."""
    from intercycle.robot.twin_sitegraph import plan_yaw_to_psi

    for c in tasks.checkpoints:
        casters = _casters(compiled[c.zone].rects)
        xyz = np.array(c.target_xzh, float)
        n = np.array((c.normal_xz[0], c.normal_xz[1], 0.0))
        for cand in c.candidate_poses:
            p = cand.pose
            for rz in (0.0, 0.3):
                cam = RC.cam_pose(p.x, p.z, rz, plan_yaw_to_psi(p.yaw))
                for cs in casters:
                    rec = RC.capture(0.0, cam, xyz, n, np.zeros(3), 0.0, cs, None, _gen(_Zero()), zoom=c.zoom)
                    assert rec["pass"], (c.id, p, rz, rec)
                    r = float(np.linalg.norm(xyz - cam))
                    los = (xyz - cam) / r
                    pxm = FX * c.zoom / r * float(-los @ n)
                    assert pxm >= DORI_INSPECT and abs(pxm - rec["px_per_m"]) < 0.2, (c.id, pxm, rec)
                    if c.thermal:
                        th = IM.thermal_capture(0.0, cam, xyz, n, np.zeros(3), 0.0, cs, None, _gen(_Zero()))
                        assert th["pass"], (c.id, p, rz, th)


def test_capture_check_is_not_vacuous_a_target_inside_its_press_box_is_occluded(tasks, compiled):
    from intercycle.robot.twin_sitegraph import plan_yaw_to_psi

    for c in tasks.checkpoints:
        bc, c3 = _casters(compiled[c.zone].rects)
        p = c.candidate_poses[0].pose
        cam = RC.cam_pose(p.x, p.z, 0.0, plan_yaw_to_psi(p.yaw))
        n = np.array((c.normal_xz[0], c.normal_xz[1], 0.0))
        inner = np.array(c.target_xzh, float) - 0.3 * n
        for cs in (bc, c3):
            rec = RC.capture(0.0, cam, inner, n, np.zeros(3), 0.0, cs, None, _gen(_Zero()), zoom=c.zoom)
            assert not rec["checks"]["unoccluded"] and not rec["pass"], c.id


def test_in_image_rule_uses_the_z1_frame_with_aim_error_then_re_centres(tasks):
    """Detectability condition (1): the frame check is the Z = 1 frame at the aim error; at Z = 20 a 2.5 deg error
    per axis (outside the zoomed 3.5 x 2.0 deg frame) still passes in_image (the PTZ re-centres before zooming)."""
    assert "Z = 1 frame" in tasks.inspection.detectability and "re-centres" in tasks.inspection.detectability
    cam = np.array((0.0, 0.0, 0.35))
    tgt = np.array((5.0, 0.0, 0.35))
    n = np.array((-1.0, 0.0, 0.0))
    rec = RC.capture(0.0, cam, tgt, n, np.zeros(3), 0.0, None, None, _gen(_Fixed(math.radians(2.5))), zoom=20.0)
    assert rec["checks"]["in_image"] and rec["pass"]
    rec = RC.capture(0.0, cam, tgt, n, np.zeros(3), 0.0, None, None, _gen(_Fixed(math.radians(40.0))), zoom=20.0)
    assert not rec["checks"]["in_image"] and not rec["pass"]


def test_window_definition_matches_the_runtime_timing(tasks, compiled):
    w = tasks.inspection.window
    assert w is not None and w.margin_s == C.SAFETY_MARGIN_S
    for z in ("z9", "z14"):
        ins = compiled[z].scenario.inspection
        assert ins.safety_margin_s == w.margin_s
        # inside_s = entry_m / v + dwell_s + entry_m / v (the written definition)
        assert ins.inside_s == pytest.approx(ins.entry_m / ins.robot_speed_mps + ins.dwell_s
                                             + ins.entry_m / ins.robot_speed_mps)
    assert "inside_s + margin_s" in w.definition and "entry_m / v + dwell_s + entry_m / v" in w.definition


def test_generation_is_deterministic():
    s = load(Site, TWIN / "site.yaml")
    assert TG.render(s) == TG.render(s)
