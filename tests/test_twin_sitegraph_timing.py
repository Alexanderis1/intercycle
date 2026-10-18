"""timing_planner in twin mode plans on the zone's compiled TwinSiteGraph (twin/<zone>/sitegraph.json written by
twin/build_worlds.py), never on the body-shop SiteGraph: same nodes, edges, targets and mission order as
twin.compile's graph; the compiled wait poses are final; the robot faces each checkpoint's target point; a wrong
contract or a graph naming machines the scenario lacks fails closed."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("intercycle_nav.timing")

from intercycle import scenario as S  # noqa: E402
from intercycle_nav import timing as T  # noqa: E402
from twin import compile as C  # noqa: E402


def _zone(z: str):
    d = REPO / "twin" / z
    return S.load(str(d / "scenario.yaml")), d / "sitegraph.json"


def _compiled(z: str):
    inp = json.loads((REPO / "twin" / z / "inputs.json").read_text(encoding="utf-8"))
    paths = {k: REPO / v["path"] for k, v in inp.items()}
    return C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])[z]


@pytest.mark.parametrize("z", ["z9", "z14"])
def test_graph_file_is_the_compiled_twin_graph(z):
    scn, gp = _zone(z)
    g = _compiled(z).graph
    core = T.TimingCore.from_graph_file(scn, str(gp))
    gd = core.graph
    assert isinstance(gd, T.GraphData) and gd.zone == z and gd.wait_poses_final
    assert [t.id for t in gd.mission()] == [t.id for t in g.mission()]
    for n, p in g.pos.items():                              # compiled wait poses kept exactly (not pushed out)
        assert gd.pos[n] == pytest.approx(tuple(p))
    for tg in g.targets:
        assert gd.inspect_pose(next(t for t in gd.targets if t.id == tg.id)) == pytest.approx(tuple(g.inspect_pose(tg)))
    assert set(gd.adj) == set(g.adj) | {f"IN_{t.machine}" for t in g.targets if t.kind == "machine"}
    for a, nb in g.adj.items():
        assert set(nb) <= set(gd.adj[a])


def test_z14_plans_and_faces_the_checkpoint_targets():
    scn, gp = _zone("z14")
    core = T.TimingCore.from_graph_file(scn, str(gp))
    g = core.graph
    by_id = {t.id: t for t in g.targets}
    press = by_id["cp_P14a_ram"]
    point = by_id["cp_z14_panel"]
    assert press.kind == "machine" and press.machine == "P14a" and point.machine is None
    for tg in (press, point):
        wx, wz = g.pos[tg.node]
        fx, fz = g.face[tg.id]
        assert core.face_psi(tg) == pytest.approx(math.atan2(fz - wz, fx - wx))
    dock = g.pos["DOCK"]
    p = core.plan_target(point, 0.0, dock, {}, "learned")
    assert p is not None
    # the press with no known idle window (learned, no windows) has no plan: fail closed
    assert core.plan_target(press, 0.0, dock, {"P14a": None}, "learned") is None


def test_graph_file_fails_closed(tmp_path):
    scn, gp = _zone("z14")
    d = json.loads(gp.read_text(encoding="utf-8"))
    bad = dict(d, contract="twin.sitegraph/0")
    (tmp_path / "a.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        T.TimingCore.from_graph_file(scn, str(tmp_path / "a.json"))
    other = dict(d, targets=[dict(t, machine="P9a") if t["machine"] else t for t in d["targets"]])
    (tmp_path / "b.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        T.TimingCore.from_graph_file(scn, str(tmp_path / "b.json"))
    order = dict(d, mission_order=d["mission_order"] + ["cp_nope"])
    (tmp_path / "c.json").write_text(json.dumps(order), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        T.TimingCore.from_graph_file(scn, str(tmp_path / "c.json"))


def test_twin_scenario_is_a_twin_zone_for_the_planner():
    """timing_planner detects a twin zone by its missing line (takt 0) and then requires sitegraph:= (no body-shop
    graph on a twin zone); the body-shop scenario has a line."""
    scn, _ = _zone("z14")
    assert float(scn.line.takt_s) == 0.0
    assert float(S.load().line.takt_s) > 0.0
