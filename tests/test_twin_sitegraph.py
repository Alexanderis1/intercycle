"""S1b item 3: the robot-side TwinSiteGraph (intercycle/robot/twin_sitegraph.py) built from tasks.yaml.

* parity with twin/compile.py's validated graph (nodes, edges, targets, capture poses, mission order);
* any approach normal: the wait pose is generated along the target normal out of the danger rect, also along x or a
  diagonal, and the timing core keeps it;
* the planner's graph: twin scenario path -> TwinSiteGraph (the body-shop SiteGraph is never built), a twin-compiled
  scenario without its inputs fails closed, the body-shop scenario keeps its SiteGraph.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from intercycle import scenario as S
from intercycle.robot import sitegraph as SG
from intercycle.robot import twin_sitegraph as TS
from twin import compile as C
from twin.schemas.dynamics import Dynamics
from twin.schemas.prereg import SplitRow
from twin.schemas.site import Site
from twin.schemas.tasks import Tasks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2" / "intercycle_nav"))

from intercycle_nav.timing import TimingCore  # noqa: E402

TWIN = REPO / "twin"
TRAIN0 = {"0_tr0": SplitRow(clip_id="0_tr0", group_id="g", split="train", sha256="0" * 64)}
SITE0 = yaml.safe_load((TWIN / "site_v0.yaml").read_text(encoding="utf-8"))
DYN0 = json.loads((TWIN / "dynamics_v0.json").read_text(encoding="utf-8"))
TASKS0 = yaml.safe_load((TWIN / "tasks_v0.yaml").read_text(encoding="utf-8"))


def _world(tmp_path: Path, site: dict, tasks: dict, dyn: dict = DYN0, split=TRAIN0) -> dict[str, Path]:
    """Compile the contracts and write, per zone, a world dir with scenario.yaml and inputs.json (absolute paths)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    sp, tp = tmp_path / "site.yaml", tmp_path / "tasks.yaml"
    sp.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    tp.write_text(yaml.safe_dump(tasks, sort_keys=False), encoding="utf-8")
    zones = C.compile_site(Site.model_validate(site), Dynamics.model_validate(dyn), Tasks.model_validate(tasks),
                           _split_for_tests=split)
    out = {}
    for zid, cz in zones.items():
        d = tmp_path / zid
        d.mkdir()
        (d / "scenario.yaml").write_text(C.scenario_yaml(cz.scenario), encoding="utf-8")
        rec = {k: {"path": str(p), "sha256": TS.text_sha256(p), "stub": False} for k, p in (("site", sp), ("tasks", tp))}
        (d / "inputs.json").write_text(json.dumps(rec), encoding="utf-8")
        out[zid] = d / "scenario.yaml"
    out["_compiled"] = zones  # type: ignore[assignment]
    return out


def _same_graph(a, b) -> None:
    assert a.pos.keys() == b.pos.keys()
    for n in a.pos:
        assert a.pos[n] == pytest.approx(b.pos[n], abs=1e-9), n
    assert {k: set(v) for k, v in a.adj.items()} == {k: set(v) for k, v in b.adj.items()}
    for k, v in a.adj.items():
        for n, w in v.items():
            assert w == pytest.approx(b.adj[k][n], abs=1e-9)
    assert a.targets == b.targets
    assert [t.id for t in a.mission()] == [t.id for t in b.mission()]
    assert a.inspect == b.inspect and a.yaw == b.yaw


# ------------------------------------------------------------------------------------------------ parity


def test_committed_v0_worlds_load_and_match_compile():
    zones = C.compile_site(Site.model_validate(SITE0), Dynamics.model_validate(DYN0), Tasks.model_validate(TASKS0),
                           _split_for_tests=TRAIN0)
    for zid, cz in zones.items():
        g = TS.load_twin_graph(TWIN / zid / "scenario.yaml")
        _same_graph(g, cz.graph)
        zt = next(z for z in TASKS0["zones"] if z["zone"] == zid)
        assert [t.id for t in g.mission()] == zt["mission_order"]            # mission order from tasks.yaml


def test_l_shaped_floor_matches_compile(tmp_path):
    site = copy.deepcopy(SITE0)
    site["zones"][0]["floor_polygon"]["value"] = [[0.0, 0.0], [14.0, 0.0], [14.0, 4.0], [10.5, 4.0], [10.5, 10.0],
                                                  [0.0, 10.0]]
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][2]["candidate_poses"][0]["pose"] = {"x": 12.3, "z": 2.0, "yaw": math.pi / 2}
    w = _world(tmp_path, site, t)
    for zid, cz in w["_compiled"].items():                     # type: ignore[union-attr]
        _same_graph(TS.load_twin_graph(w[zid]), cz.graph)


# ------------------------------------------------------------------------------------------------ any approach normal


def _press_cp(tasks: dict, pose: tuple[float, float, float], normal: tuple[float, float]) -> dict:
    t = copy.deepcopy(tasks)
    cp = t["checkpoints"][0]
    assert cp["id"] == "cp_P9a_ram" and cp["inside_danger"]
    cp["candidate_poses"][0]["pose"] = {"x": pose[0], "z": pose[1], "yaw": pose[2]}
    cp["normal_xz"] = list(normal)
    return t


@pytest.mark.parametrize("pose, normal", [((7.6, 5.2, -math.pi / 2), (1.0, 0.0)),        # approach from +x
                                          ((7.6, 5.0, -3 * math.pi / 4), (1.0, -1.0))])  # diagonal approach
def test_wait_pose_follows_any_approach_normal(tmp_path, pose, normal):
    w = _world(tmp_path, SITE0, _press_cp(TASKS0, pose, normal))
    g = TS.load_twin_graph(w["z9"])
    _same_graph(g, w["_compiled"]["z9"].graph)                 # type: ignore[index]
    tg = next(t for t in g.targets if t.id == "cp_P9a_ram")
    r = g.scn.machine("P9a").rect
    ix, iz = g.inspect_pose(tg)
    wx, wz = g.pos[tg.node]
    L = math.hypot(*normal)
    step = ((wx - ix), (wz - iz))
    assert step[0] / math.hypot(*step) == pytest.approx(normal[0] / L)
    assert step[1] / math.hypot(*step) == pytest.approx(normal[1] / L)
    assert max(r[0] - wx, wx - r[2], r[1] - wz, wz - r[3]) > 0                 # outside the danger rect
    t_exit = TS.ray_exit(ix, iz, normal[0] / L, normal[1] / L, r)
    assert math.hypot(*step) == pytest.approx(t_exit + TS.WAIT_OFFSET_M)
    core = TimingCore.from_graph(g)
    assert g.pos[tg.node] == pytest.approx((wx, wz), abs=1e-9)               # the core keeps the generated pose
    assert core.inside[tg.id] == f"IN_{tg.id}" and g.pos[core.inside[tg.id]] == pytest.approx((ix, iz))
    assert core.face_psi(tg) == pytest.approx(TS.plan_yaw_to_psi(pose[2]))
    tp = core.plan_target(tg, 0.0, g.pos["DOCK"], {"P9a": [(0.0, 3600.0)]}, "learned")
    assert tp is not None and tp.inside_xy == pytest.approx((ix, iz)) and tp.wait_xy == pytest.approx((wx, wz))


def test_two_checkpoints_on_one_press_get_their_own_inside_nodes(tmp_path):
    t = copy.deepcopy(TASKS0)
    extra = copy.deepcopy(t["checkpoints"][0])
    extra["id"] = "cp_P9a_ram2"
    extra["candidate_poses"][0]["pose"] = {"x": 5.5, "z": 5.3, "yaw": 0.0}
    t["checkpoints"].append(extra)
    t["zones"][0]["mission_order"].append("cp_P9a_ram2")
    g = TS.load_twin_graph(_world(tmp_path, SITE0, t)["z9"])
    core = TimingCore.from_graph(g)
    assert core.inside["cp_P9a_ram"] != core.inside["cp_P9a_ram2"]
    for tg in g.mission():
        win = {"P9a": [(0.0, 3600.0)]}
        for cond in ("learned", "static", "reactive", "none"):
            assert core.plan_target(tg, 0.0, g.pos["DOCK"], win, cond) is not None, (tg.id, cond)


def test_yaw_conversion():
    assert TS.plan_yaw_to_psi(0.0) == pytest.approx(math.pi / 2)              # plan +z = ROS +y
    assert TS.plan_yaw_to_psi(math.pi / 2) == pytest.approx(0.0)              # plan +x = ROS +x
    assert TS.plan_yaw_to_psi(-math.pi / 2) == pytest.approx(math.pi)


# ------------------------------------------------------------------------------------------------ fail closed


def test_unreachable_or_blocked_poses_fail_closed(tmp_path):
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][0]["candidate_poses"][0]["pose"]["z"] = 3.0              # capture pose outside the danger rect
    w = _world(tmp_path / "ok", SITE0, TASKS0)
    tasks_p = Path(json.loads((w["z9"].parent / "inputs.json").read_text())["tasks"]["path"])
    tasks_p.write_text(yaml.safe_dump(t), encoding="utf-8")
    with pytest.raises(TS.TwinGraphError, match="sha256"):                    # edited after compile: refused
        TS.load_twin_graph(w["z9"])
    scn = S.load(w["z9"])
    with pytest.raises(TS.TwinGraphError, match="not inside the danger rect"):
        TS.TwinSiteGraph(scn, t, "z9", SITE0["zones"][0]["floor_polygon"]["value"],
                         [TS.aabb(SITE0["presses"][0]["footprint"]["value"])])
    with pytest.raises(TS.TwinGraphError, match="not reachable"):
        TS.TwinSiteGraph(scn, TASKS0, "z9", SITE0["zones"][0]["floor_polygon"]["value"],
                         [TS.aabb(SITE0["presses"][0]["footprint"]["value"]), (-1.0, 6.0, 15.0, 6.5)])


def test_twin_scenario_without_inputs_never_gets_the_body_shop_graph(tmp_path, monkeypatch):
    w = _world(tmp_path, SITE0, TASKS0)
    scn = S.load(w["z9"])
    assert TS.is_twin_scenario(scn) and not TS.is_twin_scenario(S.load())

    def boom(*a, **k):
        raise AssertionError("body-shop SiteGraph built in twin mode")

    monkeypatch.setattr(SG, "SiteGraph", boom)
    core = TimingCore.from_scenario(scn, scenario_path=str(w["z9"]))
    assert isinstance(core.graph, TS.TwinSiteGraph)
    assert [t.id for t in core.graph.mission()] == ["cp_P9a_ram", "cp_P9a_side", "cp_z9_panel"]
    with pytest.raises(TS.TwinGraphError, match="refusing the body-shop graph"):
        TimingCore.from_scenario(scn)
    (w["z9"].parent / "inputs.json").unlink()
    with pytest.raises(TS.TwinGraphError):
        TimingCore.from_scenario(scn, scenario_path=str(w["z9"]))


def test_body_shop_keeps_its_site_graph():
    core = TimingCore.from_scenario(S.load())
    assert type(core.graph) is SG.SiteGraph
    old = SG.SiteGraph(S.load())                       # the previous along-z push-out, recomputed here
    for tg in old.targets:
        if tg.kind != "machine":
            continue
        ix, iz = old.inspect_pose(tg)
        wx, wz = old.pos[tg.node]
        assert tg.machine is not None
        m = old.scn.machine(tg.machine)
        edge = m.rect[1] if iz > wz else m.rect[3]
        assert core.graph.pos[tg.node] == (wx, edge - math.copysign(core.wait_offset_m, iz - wz))
        assert core.inside[tg.machine] == f"IN_{tg.machine}"
