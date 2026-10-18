"""twin/tasks.yaml (lane S2 TASKS): generated from site.yaml, 8 checkpoints (5 in z9, 3 in z14), zoom / thermal /
defect rate / detectability / window, train-only provenance, and the TwinSiteGraph compiles from it."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from twin import compile as C
from twin import tasks_gen as TG
from twin.provenance import assert_train_only, source_clips
from twin.robot_camera import DORI_INSPECT, FX
from twin.schemas import load
from twin.schemas.site import Site
from twin.schemas.tasks import Tasks

TWIN = Path(__file__).resolve().parents[1] / "twin"


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


def test_chosen_pose_reaches_dori_inspect_at_the_checkpoint_zoom(tasks):
    for c in tasks.checkpoints:
        p = c.candidate_poses[0].pose
        r = math.dist((p.x, p.z), c.target_xzh[:2])
        assert FX * c.zoom / r >= DORI_INSPECT, c.id


def test_generation_is_deterministic():
    s = load(Site, TWIN / "site.yaml")
    assert TG.render(s) == TG.render(s)
