"""Collision geometry kernel: static map, bodies, lidar, and the site graph against the plant."""
from __future__ import annotations

import math
import timeit
from collections import deque

import numpy as np
import pytest
from scipy import ndimage

from intercycle.robot.sitegraph import SiteGraph
from intercycle.scenario import load
from intercycle.sim import geometry as G
from intercycle.truth.agents import agents
from intercycle.types import MoverClass

ROBOT_R = 0.42


@pytest.fixture(scope="module")
def scn():
    return load()


@pytest.fixture(scope="module")
def sm(scn):
    return G.StaticMap(scn)


@pytest.fixture(scope="module")
def graph(scn):
    return SiteGraph(scn)


def strip_free(sm, x0, z0, x1, z1) -> bool:
    i0, i1 = int(math.ceil((x0 - G.X0) / G.RES)), int(math.floor((x1 - G.X0) / G.RES))
    j0, j1 = int(math.ceil((z0 - G.Z0) / G.RES)), int(math.floor((z1 - G.Z0) / G.RES))
    return not sm.grid[i0:i1, j0:j1].any()


def best_us(fn, n: int, repeat: int = 7) -> float:
    return min(timeit.repeat(fn, number=n, repeat=repeat)) / n * 1e6


def bad_nodes_of(sm, g) -> set[str]:
    return {n for n, (x, z) in g.pos.items()
            if any(G.static_hit(sm, G.footprint("robot", x, z, yaw)) for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2))}


def bad_edges_of(sm, g) -> set[tuple[str, str]]:
    return {(a, b) for a, nb in g.adj.items() for b in nb if a < b and G.segment_hit(sm, g.pos[a], g.pos[b], "robot", 0.25) is not None}


# ------------------------------------------------------------------------------------------------ performance
def test_clearance_is_fast(sm):
    pts = np.array([[10.0, 32.0], [50.0, 16.0], [100.0, 24.0]])
    us = best_us(lambda: sm.clearance(pts), 2000)
    assert us < 50, f"clearance(3 pts) {us:.1f} us"


def test_lidar_is_fast(sm):
    rng = np.random.default_rng(0)
    dyn = np.column_stack([rng.uniform(10, 40, 40), rng.uniform(28, 36, 40), np.full(40, 0.4)])
    ms = best_us(lambda: G.lidar(sm, 20.0, 32.0, 0.5, dyn, 180, 12.0), 100, 5) / 1000
    assert ms < 2.0, f"lidar(180 rays, 40 circles) {ms:.2f} ms"


# ------------------------------------------------------------------------------------------------ circle algebra
def test_overlaps_and_min_gap_simple_cases():
    a = np.array([[0.0, 0.0, 1.0]])
    assert G.min_gap(a, np.array([[3.0, 0.0, 1.0]])) == pytest.approx(1.0)
    assert G.min_gap(a, np.array([[2.0, 0.0, 1.0]])) == pytest.approx(0.0)
    assert G.min_gap(a, np.array([[1.5, 0.0, 1.0]])) == pytest.approx(-0.5)
    assert not G.overlaps(a, np.array([[3.0, 0.0, 1.0]]))
    assert not G.overlaps(a, np.array([[2.0, 0.0, 1.0]]))               # touching is not penetrating
    assert G.overlaps(a, np.array([[1.9, 0.0, 1.0]]))
    assert not G.overlaps(a, np.array([[2.3, 0.0, 1.0]]), margin=0.2)
    assert G.overlaps(a, np.array([[2.1, 0.0, 1.0]]), margin=0.2)       # inside the 0.2 m margin
    two = np.array([[10.0, 0.0, 0.5], [0.0, 3.0, 0.5]])                 # nearest pair decides
    assert G.min_gap(a, two) == pytest.approx(3.0 - 1.0 - 0.5)
    assert G.min_gap(a, np.zeros((0, 3))) == math.inf
    assert not G.overlaps(np.zeros((0, 3)), a)


def test_footprint_geometry():
    fwd = lambda c, yaw: (np.sin(yaw), np.cos(yaw))
    r = G.footprint("robot", 10.0, 20.0, 0.0)                           # yaw 0 -> forward is +z
    assert r.shape == (3, 3)
    assert r[:, 1].max() + 0 == pytest.approx(20.0 + 0.12)
    assert (r[:, 1] + r[:, 2]).max() - 20.0 == pytest.approx(0.54, abs=0.02)        # half length 0.525 (+ margin)
    assert (r[:, 0] + r[:, 2]).max() - 10.0 == pytest.approx(0.42)                  # half width 0.4 (+ margin)
    r = G.footprint("robot", 10.0, 20.0, math.pi / 2)                   # yaw pi/2 -> forward is +x
    assert (r[:, 0] + r[:, 2]).max() - 10.0 == pytest.approx(0.54, abs=0.02)
    p = G.footprint("person", 1.0, 2.0, 1.0)
    assert p.shape == (1, 3) and p[0, 2] == 0.3
    # tugger train: tractor + 3 carts trail straight behind the heading
    t = G.footprint("tugger", 50.0, 8.0, math.pi / 2)                   # heading +x, carts at smaller x
    assert (t[:, 0] + t[:, 2]).max() - 50.0 == pytest.approx(0.7)
    assert 50.0 - (t[:, 0] - t[:, 2]).min() == pytest.approx(0.7 + 0.2 + 3 * 1.6 + 2 * 0.2, abs=0.2)   # ~6.1 m behind
    assert np.allclose(t[:, 1], 8.0)
    # forklift: forks reach 2.05 m ahead of the body centre; body 1.9 long
    f = G.footprint("forklift", 0.0, 0.0, 0.0)
    assert (f[:, 1] + f[:, 2]).max() == pytest.approx(2.05)
    assert (f[:, 1] - f[:, 2]).min() == pytest.approx(-0.95)
    # weld arm discs
    assert G.footprint("weld_arm_active", 5.0, 6.0, 0.0).tolist() == [[5.0, 6.0, 2.2]]
    assert G.footprint("weld_arm_idle", 5.0, 6.0, 0.0).tolist() == [[5.0, 6.0, 0.7]]
    agv = G.footprint("agv", 0.0, 0.0, 0.0)
    assert (agv[:, 1] + agv[:, 2]).max() == pytest.approx(0.8) and agv[:, 2].max() == 0.5
    with pytest.raises(KeyError):
        G.footprint("hovercraft", 0, 0, 0)


# ------------------------------------------------------------------------------------------------ static map
def test_map_structure(sm, scn):
    assert sm.grid.shape == (1240, 520) and sm.grid.dtype == bool
    assert sm.dist.shape == sm.grid.shape and sm.dist.dtype == np.float32
    assert (sm.dist[sm.grid] == 0).all() and (sm.dist[~sm.grid] > 0).all()
    rs = sm.rects()
    assert {"id", "cls", "cx", "cz", "w", "d", "yaw"} <= set(rs[0])
    assert len({r["id"] for r in rs}) == len(rs)
    classes = {r["cls"] for r in rs}
    assert {"wall", "conveyor", "robot_base", "tip_dresser", "curtain_post", "lifter_post", "fixture_table", "rack_market",
            "rack_dh", "crah", "column"} <= classes
    assert sum(r["cls"] == "rack_dh" for r in rs) == 36 and sum(r["cls"] == "crah" for r in rs) == 4
    assert sum(r["cls"] == "curtain_post" for r in rs) == 4 * len(scn.machines)
    assert len(sm.arm_bases) == 2 * 7 + 2 * 2                                           # 7 weld/framing/respot stations + 2 cells


def test_clearance_semantics(sm):
    assert sm.clearance(np.array([[3.0, 32.0]]))[0] == pytest.approx(2.7, abs=0.06)      # west wall inner face at x = 0.3
    assert sm.clearance(np.array([[0.1, 32.0]]))[0] == 0.0                              # inside the wall
    assert sm.clearance(np.array([[-50.0, 32.0], [500.0, 500.0]])).tolist() == [0.0, 0.0]   # outside the world
    c = sm.clearance(np.array([[12.0, 24.0], [30.0, 22.0], [30.0, 21.5]]))              # conveyor z 22.8..25.2
    assert c[0] == 0.0 and c[1] == pytest.approx(0.8, abs=0.06) and c[2] == pytest.approx(1.3, abs=0.06)
    assert sm.free(3.0, 32.0, 1.0) and not sm.free(3.0, 32.0, 3.0)
    assert not sm.free(12.0, 24.0, 0.0)
    assert sm.free(40.0, 32.0, 1.9)                                                      # mid north aisle
    # the pure-Python small-N path, the numpy path and free() agree
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0, 120, 200), rng.uniform(0, 48, 200)])
    cv = sm.clearance(pts)                                                               # > 10 points: numpy path
    cs = np.concatenate([sm.clearance(pts[i:i + 5]) for i in range(0, 200, 5)])          # <= 10 points: scalar path
    assert np.allclose(cv, cs, atol=1e-9)
    for (x, z), c in zip(pts, cv):
        assert c == 0.0 and not sm.free(x, z, 0.0) or (sm.free(x, z, c - 1e-6) and not sm.free(x, z, c + 1e-3))


def test_robot_inside_rack_collides(sm):
    rack = next(r for r in sm.rects() if r["cls"] == "rack_dh")
    assert G.static_hit(sm, G.footprint("robot", rack["cx"], rack["cz"], 0.3))
    assert G.static_gap(sm, G.footprint("robot", rack["cx"], rack["cz"], 0.3)) < 0
    mid_aisle = G.footprint("robot", 104.0, 20.0, 0.0)                                  # between rows 0 and 1
    assert not G.static_hit(sm, mid_aisle) and G.static_gap(sm, mid_aisle) > 2.0
    # a robot half a metre from a rack face is fine; nudged into it, it is not
    z_face = rack["cz"] - rack["d"] / 2
    assert not G.static_hit(sm, G.footprint("robot", rack["cx"], z_face - 0.6, 0.0))
    assert G.static_hit(sm, G.footprint("robot", rack["cx"], z_face - 0.3, 0.0))
    # a robot on the conveyor, in a wall, on a light-curtain post
    assert G.static_hit(sm, G.footprint("robot", 30.0, 24.0, 0.0))
    assert G.static_hit(sm, G.footprint("robot", 0.5, 32.0, 0.0))
    post = next(r for r in sm.rects() if r["cls"] == "curtain_post")
    assert G.static_hit(sm, G.footprint("person", post["cx"], post["cz"], 0.0))
    assert not G.static_hit(sm, np.zeros((0, 3)))


def test_aisles_and_walkway_stay_free(sm):
    assert strip_free(sm, 0.4, 30.2, 95.4, 33.8), "north aisle z 30..34 must keep 3.6 m clear"
    assert strip_free(sm, 0.4, 14.2, 95.4, 17.8), "south aisle z 14..18 must keep 3.6 m clear"
    assert strip_free(sm, 0.4, 6.2, 95.4, 9.8), "transport aisle z 6..10 must keep 3.6 m clear"
    assert strip_free(sm, 0.4, 8.0, 4.0, 34.0), "cross connection at x~2"
    assert strip_free(sm, 88.2, 8.0, 91.8, 34.0), "cross connection at x~90"
    assert strip_free(sm, 51.75, 8.0, 53.25, 32.0), "walkway at x=52.5 (1.5 m between curtain posts, 2 m conveyor gap)"
    assert strip_free(sm, 51.5, 22.8, 53.5, 25.2), "the conveyor gap itself"
    assert not strip_free(sm, 51.0, 22.8, 51.4, 25.2) and not strip_free(sm, 53.6, 22.8, 54.0, 25.2), "conveyor either side"
    # the conveyor is one continuous 2.4 m body on each side of the gap (x 6..86)
    for x in np.arange(6.1, 51.4, 0.5).tolist() + np.arange(53.6, 85.9, 0.5).tolist():
        assert sm.grid[sm.cell(x, 22.9)] and sm.grid[sm.cell(x, 25.1)] and not sm.grid[sm.cell(x, 22.6)] and not sm.grid[sm.cell(x, 25.4)]
    assert not sm.grid[sm.cell(5.8, 24.0)] and not sm.grid[sm.cell(86.2, 24.0)]
    # no column sits in an aisle, and the 12 x 12 m grid is present
    cols = [r for r in sm.rects() if r["cls"] == "column"]
    assert len(cols) == 14 and all(r["cz"] in (12.0, 36.0) for r in cols)


def test_wall_openings(sm):
    for z in (24.0, 23.0, 25.0, 8.0, 7.2, 8.8, 32.0, 31.2, 32.8):                       # door 3 m; aisle openings 2 m
        assert sm.free(96.0, z, 0.2), z
    assert sm.free(96.0, 24.0, 1.4) and sm.free(96.0, 8.0, 0.9) and sm.free(96.0, 32.0, 0.9)
    for z in (20.0, 27.0, 5.0, 15.0, 40.0):
        assert not sm.free(96.0, z, 0.1), z
    assert not sm.free(0.1, 20.0, 0.0) and not sm.free(50.0, 0.1, 0.0) and not sm.free(50.0, 47.9, 0.0) and not sm.free(119.9, 24.0, 0.0)
    # robot passes through the door
    assert G.segment_hit(sm, (90.0, 24.0), (100.0, 24.0), "robot") is None
    assert G.segment_hit(sm, (90.0, 20.0), (100.0, 20.0), "robot") is not None              # but not through the wall


def test_data_hall_layout(sm):
    rs = sm.rects()
    racks = [r for r in rs if r["cls"] == "rack_dh"]
    rows = sorted({r["cz"] for r in racks})
    assert rows == [16.0, 24.0, 32.0]
    for z in rows:
        row = sorted((r for r in racks if r["cz"] == z), key=lambda r: r["cx"])
        assert len(row) == 12 and all(abs(r["w"] - 0.6) < 1e-9 and abs(r["d"] - 1.1) < 1e-9 for r in row)
        assert all(b["cx"] - a["cx"] == pytest.approx(0.6) for a, b in zip(row, row[1:]))          # contiguous
        assert row[0]["cx"] - 0.3 == pytest.approx(101.0)
        assert row[0]["cx"] - 0.3 - 96.15 >= 2.0 and 118.0 - (row[-1]["cx"] + 0.3) >= 2.0               # row-end gaps
    assert min(b - a - 1.1 for a, b in zip(rows, rows[1:])) >= 1.2                                  # cold/hot aisles
    crah = [r for r in rs if r["cls"] == "crah"]
    assert all(r["cx"] == pytest.approx(118.6) and r["w"] == 1.0 and r["d"] == 2.6 for r in crah)
    # technicians' routes in the data hall are free
    for z in (20.0, 28.0):
        assert G.segment_hit(sm, (100.0, z), (116.0, z), "person") is None


def test_machine_furniture(sm, scn):
    rs = {r["id"]: r for r in sm.rects()}
    for m in scn.machines:
        x0, z0, x1, z1 = m.rect
        for k in range(4):
            p = rs[f"{m.id}_LC{k}"]
            assert x0 < p["cx"] < x1 and z0 < p["cz"] < z1 and p["w"] == 0.12
    assert rs["S10_BASE0"]["w"] == 0.9 and rs["S10_TIP0"]["w"] == 0.5
    assert (rs["S10_BASE0"]["cx"], rs["S10_BASE0"]["cz"]) == pytest.approx((10.2, 20.7))
    assert (rs["S10_BASE1"]["cx"], rs["S10_BASE1"]["cz"]) == pytest.approx((13.8, 27.3))
    assert "S80_FRAME0" in rs and "S80_BASE0" not in rs
    for cell in ("HEM1", "SEAL1"):
        t = rs[f"{cell}_TABLE"]
        assert (t["w"], t["d"]) == (3.0, 1.6)
        m = scn.machine(cell)
        assert (t["cx"], t["cz"]) == (m.x, m.z)
    # market racks: 5 m bays, 1.2 m deep rows at z 1.2 / 3.8, with gaps, cut at the forklift cross aisles
    mk = [r for r in sm.rects() if r["cls"] == "rack_market"]
    assert {r["cz"] for r in mk} == {1.2, 3.8} and all(abs(r["d"] - 1.2) < 1e-9 and r["w"] <= 5.0 + 1e-9 for r in mk)
    for x in (2.0, 52.5, 88.0, 90.0):
        assert strip_free(sm, x - 0.6, 0.4, x + 0.6, 5.0), x
    assert sum(1 for r in mk if abs(r["w"] - 5.0) < 1e-9) >= 20


# ------------------------------------------------------------------------------------------------ site graph vs plant
def test_wait_points_clear_of_static_geometry(sm, graph):
    waits = [n for n in graph.pos if n.startswith("W_")]
    assert len(waits) == 10
    for n in waits:
        x, z = graph.pos[n]
        assert sm.clearance(np.array([[x, z]]))[0] >= 0.45, (n, x, z)


def test_non_datahall_graph_is_collision_free(sm, graph):
    """Everything on the shop floor (aisles, crossings, wait nodes, door) must be drivable by the robot footprint."""
    bad_n = {n for n in bad_nodes_of(sm, graph) if not n.startswith("R")}
    bad_e = {(a, b) for a, b in bad_edges_of(sm, graph) if not (a.startswith("R") or b.startswith("R"))}
    assert not bad_n, bad_n
    assert not bad_e, bad_e


def test_sitegraph_nodes_free(sm, graph):
    """Known issue (report to lead): rack stops R00/R10/R20 at x=104 lie inside the racks (rows span x 101..108.2)."""
    bad = bad_nodes_of(sm, graph)
    assert bad <= {n for n in graph.pos if n.startswith("R")}, bad


def test_sitegraph_edges_free(sm, graph):
    bad = bad_edges_of(sm, graph)
    assert all(a.startswith("R") or b.startswith("R") for a, b in bad), bad


def test_sitegraph_fully_collision_free(sm, graph):
    assert not bad_nodes_of(sm, graph) and not bad_edges_of(sm, graph)


def patched_graph(scn) -> SiteGraph:
    """The suggested fix: rack stops in the aisle 1.6 m in front of each row's south face, reached via waypoints
    DA<r> on the west side of the row ends (x=99). Same node names R<r><k> as before."""
    g = SiteGraph(scn)
    v = scn.inspection.robot_speed_mps
    for n in [n for n in g.pos if n.startswith("R")]:
        g.pos.pop(n)
        g.adj.pop(n)
    for nb in g.adj.values():
        for n in [n for n in nb if n.startswith("R")]:
            nb.pop(n)
    g.targets[:] = [t for t in g.targets if t.kind != "rack"]
    from intercycle.robot.sitegraph import Target
    for r in range(scn.data_hall_targets["rows"]):
        zr = 16.0 + 8.0 * r - 1.6
        g._node(f"DA{r}", 99.0, zr)
        g._edge("DH", f"DA{r}", v)
        prev = f"DA{r}"
        for k, x in enumerate((103.0, 106.6)):
            g._node(f"R{r}{k}", x, zr)
            g._edge(prev, f"R{r}{k}", v)
            g.targets.append(Target(f"R{r}{k}", "rack", f"R{r}{k}", None, 6.0))
            prev = f"R{r}{k}"
    return g


def test_proposed_rack_stops_fix(sm, scn):
    g = patched_graph(scn)
    assert not bad_nodes_of(sm, g)
    assert not bad_edges_of(sm, g)
    for r in range(3):
        for k in range(2):
            x, z = g.pos[f"R{r}{k}"]
            assert sm.clearance(np.array([[x, z]]))[0] >= 1.0                       # camera standoff from the racks
            assert 101.0 <= x <= 108.2                                              # in front of the row, not past its end


def test_all_nodes_reachable_through_free_space(sm, scn, graph):
    """Flood fill over cells where the robot centre has >= 0.42 m clearance: the dock reaches every node that is itself free."""
    free = sm.dist >= ROBOT_R
    lab, _ = ndimage.label(free)
    for g in (graph, patched_graph(scn)):
        bad = bad_nodes_of(sm, g)
        comp = lab[sm.cell(*g.pos["DOCK"])]
        assert comp > 0
        for n, (x, z) in g.pos.items():
            if n in bad:
                continue
            assert lab[sm.cell(x, z)] == comp, f"{n} at ({x}, {z}) is not reachable from the dock"
    # graph connectivity itself (BFS over edges)
    seen, q = {"DOCK"}, deque(["DOCK"])
    while q:
        for m in graph.adj[q.popleft()]:
            if m not in seen:
                seen.add(m)
                q.append(m)
    assert seen == set(graph.pos)
    # the data hall is reachable from the door through the 3 m opening for a robot-sized disc
    assert lab[sm.cell(90.0, 24.0)] == lab[sm.cell(100.0, 24.0)]


def route_hits(sm, scn):
    name = {MoverClass.AGV: "agv", MoverClass.TUGGER: "tugger", MoverClass.FORKLIFT: "forklift", MoverClass.PERSON: "person"}
    hits, checked = {}, 0
    for a in agents(scn, 3):
        cls = name[a.cls]
        for p, q in zip(a._pts, a._pts[1:]):
            L = math.dist(p, q)
            if L < 1e-9:
                continue
            if cls == "tugger":            # carts trail straight behind the route start: test from one train length on
                if L <= 6.5:               # corner of the one-way loop: shorter than the train, the approximation doesn't apply
                    continue
                u = ((q[0] - p[0]) / L, (q[1] - p[1]) / L)
                p = (p[0] + 6.5 * u[0], p[1] + 6.5 * u[1])
            hit = G.segment_hit(sm, p, q, cls, 0.1)
            checked += 1
            if hit is not None:
                hits[(a.id, p, q)] = hit
    return hits, checked


def test_mover_routes_are_clear(sm, scn):
    """AGV loop, tugger, forklift, person and technician routes from truth/agents.py never touch static geometry."""
    hits, checked = route_hits(sm, scn)
    assert checked > 60
    assert not hits, hits


def test_forklift_corner_at_west_wall(sm, scn):
    hits, _ = route_hits(sm, scn)
    assert not hits


def test_tugger_standing_at_a_route_end_is_known_issue(sm):
    """Documented limitation: with carts approximated as straight behind the heading, a train standing at the west end of
    its route (3, 8) pointing east has its carts through the wall.  Arrival heading (carts toward the interior) is fine."""
    assert not G.static_hit(sm, G.footprint("tugger", 3.0, 8.0, -math.pi / 2))           # arrived heading west
    assert not G.static_hit(sm, G.footprint("tugger", 20.0, 8.0, math.pi / 2))
    assert G.static_hit(sm, G.footprint("tugger", 3.0, 8.0, math.pi / 2))                # pointing east: carts in the wall


# ------------------------------------------------------------------------------------------------ lidar
def test_lidar_wall_distance(sm):
    empty = np.zeros((0, 3))
    r = G.lidar(sm, 3.0, 32.0, -math.pi / 2, empty)               # facing west: wall inner face at x = 0.3
    assert r.shape == (180,)
    assert r[0] == pytest.approx(2.7, abs=0.08)
    r = G.lidar(sm, 3.0, 32.0, math.pi / 2, empty)                # facing east: ray 90 (180 deg round) looks at the wall
    assert r[90] == pytest.approx(2.7, abs=0.08)
    assert r[0] == pytest.approx(12.0)                            # ray 0 runs down the aisle: no return within rmax
    # rays at +-45 deg to the wall: range = 2.7 / cos(45 deg)
    r = G.lidar(sm, 3.0, 32.0, -math.pi / 2, empty, n=360)
    assert r[45] == pytest.approx(2.7 / math.cos(math.radians(45)), abs=0.12)
    assert r[360 - 45] == pytest.approx(2.7 / math.cos(math.radians(45)), abs=0.12)
    # facing south from the north aisle at x=40: the first thing is the conveyor's north face at z = 25.2
    r = G.lidar(sm, 40.0, 32.0, math.pi, empty)
    assert r[0] == pytest.approx(32.0 - 25.2, abs=0.08)


def test_lidar_matches_exact_sphere_tracing(sm):
    rng = np.random.default_rng(7)
    empty = np.zeros((0, 3))
    tot = bad = 0
    for _ in range(60):
        x, z = rng.uniform(3, 118), rng.uniform(3, 46)
        if not sm.free(x, z, 0.4):
            continue
        yaw = rng.uniform(0, 2 * math.pi)
        a = G.lidar(sm, x, z, yaw, empty)
        b = G.lidar(sm, x, z, yaw, empty, exact=True)
        tot += a.size
        bad += int((np.abs(a - b) > 0.1).sum())
    assert tot > 5000 and bad / tot < 0.02, (bad, tot)


def test_lidar_dynamic_bodies_and_occlusion(sm):
    x, z = 20.0, 32.0                                              # north aisle, facing +x along it
    yaw = math.pi / 2
    person = np.array([[x + 5.0, z, 0.3]])
    r = G.lidar(sm, x, z, yaw, person)
    assert r[0] == pytest.approx(4.7, abs=1e-6)                    # analytic: centre distance minus radius
    for i in (1, 179):                                             # +-2 deg still graze the circle: analytic chord
        a = math.radians(2.0)
        assert r[i] == pytest.approx(5.0 * math.cos(a) - math.sqrt(0.3 ** 2 - (5.0 * math.sin(a)) ** 2), abs=1e-6)
    assert r[2] == pytest.approx(12.0) and r[178] == pytest.approx(12.0)      # +-4 deg miss it
    # a person behind a wall is invisible (static occlusion), one in front of the wall is seen first
    hidden = np.array([[-1.0, 32.0, 0.3]])
    assert G.lidar(sm, 3.0, 32.0, -math.pi / 2, hidden)[0] == pytest.approx(2.7, abs=0.08)
    front = np.array([[1.5, 32.0, 0.3]])
    assert G.lidar(sm, 3.0, 32.0, -math.pi / 2, front)[0] == pytest.approx(1.2, abs=1e-6)
    # two bodies on the same ray: the nearer wins, whatever the order
    two = np.array([[x + 8.0, z, 0.5], [x + 3.0, z, 0.3]])
    assert G.lidar(sm, x, z, yaw, two)[0] == pytest.approx(2.7, abs=1e-6)
    assert G.lidar(sm, x, z, yaw, two[::-1])[0] == pytest.approx(2.7, abs=1e-6)
    # a body beyond rmax is not returned; a body off to the side does not leak into ray 0
    far = np.array([[x + 20.0, z, 0.3], [x, z + 30.0, 0.3]])
    assert np.array_equal(G.lidar(sm, x, z, yaw, far, rmax=12.0), G.lidar(sm, x, z, yaw, np.zeros((0, 3)), rmax=12.0))
    side = np.array([[x + 4.0, z + 1.5, 0.3]])
    assert G.lidar(sm, x, z, yaw, side)[0] == pytest.approx(12.0)
    # a ray's range is a real angular occlusion: the shadow of a 0.3 m circle at 5 m is about +-3.4 deg
    r = G.lidar(sm, x, z, yaw, person, n=360)
    shadow = np.flatnonzero(r < G.lidar(sm, x, z, yaw, np.zeros((0, 3)), n=360) - 1e-6)
    assert shadow.tolist() == [0, 1, 2, 3, 357, 358, 359]
    # output carries only ranges
    assert r.dtype == np.float64 and r.ndim == 1
    # a sensor inside a body (own body left in `dyn` by mistake) reads 0
    assert (G.lidar(sm, x, z, yaw, np.array([[x, z, 0.5]])) == 0.0).all()


def test_lidar_in_data_hall_sees_racks_not_through_them(sm):
    empty = np.zeros((0, 3))
    r = G.lidar(sm, 104.0, 20.0, 0.0, empty)                       # facing +z (north) from the aisle between rows 0 and 1
    assert r[0] == pytest.approx(24.0 - 0.55 - 20.0, abs=0.08)     # row 1's south face at z = 23.45
    r = G.lidar(sm, 104.0, 20.0, math.pi, empty)                   # facing -z: row 0's north face at z = 16.55
    assert r[0] == pytest.approx(20.0 - 16.55, abs=0.08)
