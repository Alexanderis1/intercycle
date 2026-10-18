"""twin/compile.py (PLAN_TWIN §2, §4 S1a; R-W10): site.yaml -> per-zone Scenario, rects + extent, danger rects and a
SiteGraph from tasks.yaml; provenance against the active split first (fail closed)."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from intercycle import scenario as scn_mod
from intercycle.sim import geometry as G
from twin import compile as C
from twin.provenance import ProvenanceError
from twin.schemas.dynamics import Dynamics
from twin.schemas.prereg import SplitRow
from twin.schemas.site import Site, point_in_polygon
from twin.schemas.tasks import Tasks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2" / "intercycle_nav"))

from intercycle_nav.timing import TimingCore  # noqa: E402

SHA = "0" * 64


def _raw(name: str) -> dict:
    p = REPO / "twin" / name
    text = p.read_text(encoding="utf-8")
    return yaml.safe_load(text) if p.suffix == ".yaml" else json.loads(text)


SITE0, DYN0, TASKS0 = _raw("site_v0.yaml"), _raw("dynamics_v0.json"), _raw("tasks_v0.yaml")


def split_of(**ids: str) -> dict[str, SplitRow]:
    return {c: SplitRow(clip_id=c, group_id=f"g_{c}", split=s, sha256=SHA) for c, s in ids.items()}  # type: ignore[arg-type]


TRAIN0 = split_of(**{"0_tr0": "train"})


def compiled(site: dict | None = None, dyn: dict | None = None, tasks: dict | None = None, split=None):
    return C.compile_site(Site.model_validate(site or SITE0), Dynamics.model_validate(dyn or DYN0),
                          Tasks.model_validate(tasks or TASKS0), split=TRAIN0 if split is None else split)


@pytest.fixture(scope="module")
def v0():
    return compiled()


# ------------------------------------------------------------------------------------------------ constants


def test_grid_matches_geometry():
    assert C.GRID_RES == G.RES


def test_default_inputs_fall_back_to_v0_stubs(tmp_path):
    d = C.default_inputs(tmp_path)
    assert (d["site"].name, d["dynamics"].name, d["tasks"].name) == ("site_v0.yaml", "dynamics_v0.json", "tasks_v0.yaml")
    (tmp_path / "tasks.yaml").write_text("x", encoding="utf-8")
    assert C.default_inputs(tmp_path)["tasks"].name == "tasks.yaml"


# ------------------------------------------------------------------------------------------------ scenario


def test_presses_are_off_line_machines_with_the_danger_rect(v0):
    assert set(v0) == {"z9", "z14"}
    for zid, cz in v0.items():
        presses = [p for p in SITE0["presses"] if p["zone"] == zid]
        assert [m.id for m in cz.scenario.machines] == [p["id"] for p in presses]
        for m, p in zip(cz.scenario.machines, presses):
            assert m.on_line is False and m.kind == "press"
            xs, zs = zip(*p["danger_polygon"]["value"])
            d = p["dilation_m"]["value"]
            want = (min(xs) - d, min(zs) - d, max(xs) + d, max(zs) + d)
            assert np.allclose(m.rect, want) and np.allclose(cz.danger_rects[p["id"]], want)
            assert m.busy == DYN0["observed"][f"press.{p['id']}.p_running"]["value"]
            assert m.period_s == DYN0["observed"][f"press.{p['id']}.stroke_period_s"]["value"]


def test_scenario_has_no_body_shop_structure(v0):
    for cz in v0.values():
        s = cz.scenario
        assert s.shifts == [] and s.breaks == [] and s.line_stops() == []      # P2: no time-of-day structure
        assert s.traffic == {} and s.ble.broadcasters == []
        assert s.line.takt_s == 0.0
        assert all(not v for k, v in s.model_dump().items() if k.startswith("data_"))


def test_scenario_yaml_round_trips(v0, tmp_path):
    for zid, cz in v0.items():
        p = tmp_path / f"{zid}.yaml"
        p.write_text(C.scenario_yaml(cz.scenario), encoding="utf-8")
        assert scn_mod.load(p) == cz.scenario


def test_dock_and_entry(v0):
    for zid, cz in v0.items():
        zt = next(z for z in TASKS0["zones"] if z["zone"] == zid)
        assert cz.scenario.robot["dock"] == [zt["dock"]["x"], zt["dock"]["z"]]
        g = cz.graph
        want = max(math.dist(g.pos[t.node], g.inspect[t.id]) for t in g.targets if t.kind == "machine")
        assert cz.scenario.inspection.entry_m == pytest.approx(want)


# ------------------------------------------------------------------------------------------------ rects + extent


def test_extent_is_snapped_with_a_ring_margin(v0):
    for zid, cz in v0.items():
        z = next(z for z in SITE0["zones"] if z["id"] == zid)
        xs, zs = zip(*z["floor_polygon"]["value"])
        x0, z0, x1, z1 = cz.extent
        m = C.EXTENT_MARGIN_CELLS * C.GRID_RES
        assert x0 <= min(xs) - m + 1e-9 and z0 <= min(zs) - m + 1e-9
        assert x1 >= max(xs) + m - 1e-9 and z1 >= max(zs) + m - 1e-9
        for v in cz.extent:
            assert abs(v / C.GRID_RES - round(v / C.GRID_RES)) < 1e-6
        G.StaticMap(None, rects=cz.rects, extent=cz.extent)                   # accepted by the geometry kernel


def test_rects_are_footprints_and_unseen_walls_not_danger(v0):
    for zid, cz in v0.items():
        cls = sorted({r["cls"] for r in cz.rects})
        assert cls == ["press", "wall"]
        assert all(r["yaw"] == 0.0 for r in cz.rects)                         # axis-aligned
        feet = [r for r in cz.rects if r["cls"] == "press"]
        assert len(feet) == len(cz.danger_rects)
        for r in feet:
            # the danger rect is a hazard zone, not a body: no static rect equals it
            assert all(not np.allclose((r["cx"] - r["w"] / 2, r["cz"] - r["d"] / 2, r["cx"] + r["w"] / 2,
                                        r["cz"] + r["d"] / 2), d) for d in cz.danger_rects.values())


L_FLOOR = [[0.0, 0.0], [14.0, 0.0], [14.0, 4.0], [8.0, 4.0], [8.0, 10.0], [0.0, 10.0]]
SLANT_FLOOR = [[0.0, 0.0], [14.0, 0.0], [12.0, 10.0], [0.03, 10.0]]


@pytest.mark.parametrize("floor", [L_FLOOR, SLANT_FLOOR], ids=["L", "slanted"])
def test_unseen_cells_are_exactly_the_cells_not_inside(floor):
    zone = Site.model_validate(SITE0).zones[0].model_copy(update={"floor_polygon": Site.model_validate(SITE0).zones[0]
                                                                  .floor_polygon.model_copy(update={"value": floor})})
    ext = C.zone_extent(zone)
    poly = [(float(p[0]), float(p[1])) for p in floor]
    mask = C.unseen_cells(poly, ext)
    h = C.GRID_RES / 2
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            cx, cz = ext[0] + (i + 0.5) * C.GRID_RES, ext[1] + (j + 0.5) * C.GRID_RES
            corners = [(cx + a * (h - 1e-7), cz + b * (h - 1e-7)) for a in (-1, 1) for b in (-1, 1)]
            if not mask[i, j]:            # a visible cell lies entirely inside the visible floor
                assert all(point_in_polygon(x, z, poly) for x, z in corners), (i, j)
            elif point_in_polygon(cx, cz, poly):
                # a wall cell whose centre is inside must be cut by an edge (some corner region outside)
                fine = [(cx + a * h * k / 4, cz + b * h * k / 4) for a in (-1, 1) for b in (-1, 1) for k in range(1, 5)]
                assert not all(point_in_polygon(x, z, poly) for x, z in fine), (i, j)
    # the merged rects cover exactly the unseen cells
    rects = C.unseen_floor_rects(poly, ext, "U")
    cover = np.zeros_like(mask)
    for r in rects:
        i0 = round((r["cx"] - r["w"] / 2 - ext[0]) / C.GRID_RES)
        i1 = round((r["cx"] + r["w"] / 2 - ext[0]) / C.GRID_RES)
        j0 = round((r["cz"] - r["d"] / 2 - ext[1]) / C.GRID_RES)
        j1 = round((r["cz"] + r["d"] / 2 - ext[1]) / C.GRID_RES)
        assert not cover[i0:i1, j0:j1].any(), "merged rects overlap"
        cover[i0:i1, j0:j1] = True
    assert (cover == mask).all()


def test_grid_aligned_floor_edges_keep_their_visible_cells():
    rect = [(0.0, 0.0), (14.0, 0.0), (14.0, 10.0), (0.0, 10.0)]
    ext = (-0.2, -0.2, 14.2, 10.2)
    mask = C.unseen_cells(rect, ext)
    assert mask.sum() == mask.size - 140 * 100                               # only the 2-cell margin is unseen


# ------------------------------------------------------------------------------------------------ site graph


def test_graph_targets_follow_tasks(v0):
    for zid, cz in v0.items():
        zt = next(z for z in TASKS0["zones"] if z["zone"] == zid)
        assert [t.id for t in cz.graph.mission()] == zt["mission_order"]
        for cp in (c for c in TASKS0["checkpoints"] if c["zone"] == zid):
            t = next(t for t in cz.graph.targets if t.id == cp["id"])
            assert t.kind == ("machine" if cp["inside_danger"] else "point")
            assert t.machine == (cp["press"] if cp["inside_danger"] else None)
            assert t.dwell_s == cp["dwell_s"]


def test_wait_pose_is_generated_outside_the_danger_rect_along_the_normal(v0):
    for cz in v0.values():
        for t in cz.graph.targets:
            if t.kind != "machine":
                continue
            r = cz.danger_rects[t.machine]
            ix, iz = cz.graph.inspect_pose(t)
            wx, wz = cz.graph.pos[t.node]
            assert r[0] < ix < r[2] and r[1] < iz < r[3]
            cp = next(c for c in TASKS0["checkpoints"] if c["id"] == t.id)
            n = np.array(cp["normal_xz"], float) / np.linalg.norm(cp["normal_xz"])
            step = np.array((wx - ix, wz - iz))
            assert np.allclose(step / np.linalg.norm(step), n)                # along the target normal
            gap = max(r[0] - wx, wx - r[2], r[1] - wz, wz - r[3])
            assert gap == pytest.approx(C.WAIT_OFFSET_M)


def test_graph_nodes_and_edges_are_clear_in_the_static_map(v0):
    for cz in v0.values():
        sm = G.StaticMap(None, rects=cz.rects, extent=cz.extent)
        g = cz.graph
        for n, (x, z) in g.pos.items():
            assert sm.free(x, z, 0.45), n
            assert all(not (r[0] < x < r[2] and r[1] < z < r[3]) for r in cz.danger_rects.values()), n
        for a, nb in g.adj.items():
            for b, w in nb.items():
                assert g.adj[b][a] == w
                assert G.segment_hit(sm, g.pos[a], g.pos[b], "robot") is None, (a, b)
                p, q = np.array(g.pos[a]), np.array(g.pos[b])
                assert all(C._seg_rect_dist(p, q, r) >= C.ROBOT_R - 1e-9 for r in cz.danger_rects.values())


def test_graph_drives_the_timing_core(v0):
    for cz in v0.values():
        core = TimingCore(cz.graph)
        core._build()
        for t in cz.graph.targets:
            dock = cz.graph.pos["DOCK"]
            windows = {t.machine: [(0.0, 3600.0)]} if t.machine else {}
            assert core.plan_target(t, 0.0, dock, windows, "learned") is not None, t.id
            assert core.plan_target(t, 0.0, dock, windows, "static") is not None, t.id


# ------------------------------------------------------------------------------------------------ fail closed


def test_capture_pose_outside_the_danger_rect_is_refused():
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][0]["candidate_poses"][0]["pose"]["z"] = 3.0
    with pytest.raises(C.CompileError, match="not inside the danger rect"):
        compiled(tasks=t)


def test_unreachable_target_is_refused():
    s = copy.deepcopy(SITE0)
    s["presses"][0]["footprint"]["value"] = [[-1.0, 6.0], [15.0, 6.0], [15.0, 6.5], [-1.0, 6.5]]   # a wall across z9
    with pytest.raises(C.CompileError, match="not reachable"):
        compiled(site=s)


def test_pose_on_unseen_floor_is_refused():
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][2]["candidate_poses"][0]["pose"]["x"] = 13.8
    with pytest.raises(C.CompileError, match="unseen floor"):
        compiled(tasks=t)


def test_inconsistent_ids_are_refused():
    t = copy.deepcopy(TASKS0)
    t["site_id"] = "other"
    with pytest.raises(C.CompileError, match="site_id"):
        compiled(tasks=t)
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][0]["press"] = "P14a"
    with pytest.raises(C.CompileError, match="not a press of zone"):
        compiled(tasks=t)
    d = copy.deepcopy(DYN0)
    del d["observed"]["press.P9a.p_running"]
    with pytest.raises(C.CompileError, match="press.P9a.p_running"):
        compiled(dyn=d)


def _with_clips():
    s = copy.deepcopy(SITE0)
    s["cal_sources"] = ["0_trS"]
    s["scale"] = {"value": s["scale"]["value"], "source": "cal_v1", "clips": ["0_trS"]}
    d = copy.deepcopy(DYN0)
    d["observed"]["person.speed_mps"].update(source="fitted", n=12, clips=["1_trD"])
    t = copy.deepcopy(TASKS0)
    t["checkpoints"][2].update(source="traced_v0", clips=["2_trT"])
    return s, d, t


def test_provenance_all_train_compiles():
    s, d, t = _with_clips()
    out = compiled(s, d, t, split=split_of(**{"0_trS": "train", "1_trD": "train", "2_trT": "train"}))
    assert set(out) == {"z9", "z14"}


@pytest.mark.parametrize("bad", ["0_trS", "1_trD", "2_trT"])
@pytest.mark.parametrize("kind", ["dev", "test", "missing"])
def test_provenance_refuses_non_train_ids(bad, kind):
    s, d, t = _with_clips()
    ids = {"0_trS": "train", "1_trD": "train", "2_trT": "train"}
    if kind == "missing":
        del ids[bad]
    else:
        ids[bad] = kind
    with pytest.raises(ProvenanceError, match=bad):
        compiled(s, d, t, split=split_of(**ids))


def test_provenance_runs_before_any_geometry():
    s, d, t = _with_clips()
    t["checkpoints"][0]["candidate_poses"][0]["pose"]["z"] = 3.0          # would be a CompileError
    with pytest.raises(ProvenanceError):
        compiled(s, d, t, split=split_of(**{"0_trS": "dev", "1_trD": "train", "2_trT": "train"}))


# ------------------------------------------------------------------------------------------------ cameras


def test_affine_placeholder_camera_is_at_the_principal_point_floor_point(v0):
    assert np.allclose(v0["z9"].scenario.cameras[0], (7.0, 5.0))
    assert np.allclose(v0["z14"].scenario.cameras[0], (34.0, 4.0))


def _framing(H: np.ndarray, K: np.ndarray):
    fr = Site.model_validate(SITE0).framings[0]
    return fr.model_copy(update={"K": fr.K.model_copy(update={"value": K.tolist()}),
                                 "H_img_to_floor": fr.H_img_to_floor.model_copy(update={"value": H.tolist()})})


def _K(f: float) -> np.ndarray:
    return np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])


def _cam_H(f: float, Cw, tilt_deg: float, pan_deg: float = 0.0) -> np.ndarray:
    """H_img_to_floor of a pinhole camera at Cw (floor frame x, z_plan, height), panned and tilted down."""
    t, p = math.radians(tilt_deg), math.radians(pan_deg)
    fwd = np.array([math.sin(p) * math.cos(t), math.cos(p) * math.cos(t), -math.sin(t)])
    xc = np.array([math.cos(p), -math.sin(p), 0.0])
    R = np.stack([xc, np.cross(fwd, xc), fwd])                       # world -> camera (rows: camera axes)
    H = np.linalg.inv(_K(f) @ np.column_stack([R[:, 0], R[:, 1], -R @ np.asarray(Cw, float)]))
    return H / H[2, 2]


CAM = (7.0, 2.0, 4.0)


def test_f_range_is_the_p6_range():
    from intercycle import framing
    assert C.F_RANGE_PX == framing.F_RANGE and C.F_RANGE_WIDTH_PX == framing.W


def test_perspective_camera_centre_is_recovered():
    H = _cam_H(1330.0, (3.0, 4.0, 4.0), 35)
    assert np.allclose(C.camera_floor_xz(_framing(H, _K(1330.0))), (3.0, 4.0), atol=1e-6)
    skew = H.copy()
    skew[0, 1] += 0.5 * abs(H[0, 0]) + 0.01
    skew[2, 0] += 1e-3
    with pytest.raises(C.CompileError, match="pinhole|focal"):
        C.camera_floor_xz(_framing(skew, _K(1330.0)))


@pytest.mark.parametrize("ratio", [0.80, 0.85, 0.90, 1.0, 1.05, 1.075])
@pytest.mark.parametrize("tilt, pan", [(45, 0), (45, 30), (30, -20), (60, 10)])
def test_f_is_fitted_from_H_when_the_prior_is_off(ratio, tilt, pan):
    """K holds the P6 prior f = 1330; the true f differs. The camera centre and f are recovered exactly (1e-6 m) as
    long as the true f is in the pre-registered 1020-1430 px."""
    f_true = 1330.0 * ratio
    fit = C.camera_fit(_framing(_cam_H(f_true, CAM, tilt, pan), _K(1330.0)))
    assert fit.f_source == "fitted" and fit.f_px == pytest.approx(f_true, rel=1e-9)
    assert np.allclose(fit.xz, CAM[:2], atol=1e-6) and fit.height_m == pytest.approx(CAM[2], abs=1e-6)
    assert fit.residual < 1e-9


@pytest.mark.parametrize("ratio", [0.85, 1.15])
def test_f_off_by_15_percent_inside_the_range_is_recovered(ratio):
    """Prior 1230 px, true f +-15 % (1045.5 / 1414.5 px, both inside 1020-1430 px)."""
    fit = C.camera_fit(_framing(_cam_H(1230.0 * ratio, CAM, 45, 30), _K(1230.0)))
    assert fit.f_px == pytest.approx(1230.0 * ratio, rel=1e-9)
    assert np.allclose(fit.xz, CAM[:2], atol=1e-6)


@pytest.mark.parametrize("ratio", [0.75, 1.10, 1.15, 1.25])
def test_f_outside_the_p6_range_fails_closed(ratio):
    with pytest.raises(C.CompileError, match="outside the pre-registered 1020-1430 px"):
        C.camera_fit(_framing(_cam_H(1330.0 * ratio, CAM, 45, 30), _K(1330.0)))


def test_f_range_scales_with_the_image_width():
    fr = _framing(_cam_H(1330.0 * 1.10 / 2, CAM, 45, 30), _K(1330.0))
    half = fr.model_copy(update={"image_size_px": fr.image_size_px.model_copy(update={"value": (960, 540)})})
    # a 960 px wide frame: range 510-715 px; f = 731.5 px is outside, f = 0.9 * 665 inside
    with pytest.raises(C.CompileError, match="outside the pre-registered 510-715 px"):
        C.camera_fit(half)
    K2 = np.array([[665.0, 0, 960], [0, 665.0, 540], [0, 0, 1.0]])
    ok = half.model_copy(update={"K": half.K.model_copy(update={"value": K2.tolist()}),
                                 "H_img_to_floor": half.H_img_to_floor.model_copy(
                                     update={"value": _cam_H(665.0 * 0.9, CAM, 45, 30).tolist()})})
    assert C.camera_fit(ok).f_px == pytest.approx(598.5)


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (u, v), (x, z) in zip(src, dst):
        A += [[u, v, 1, 0, 0, 0, -x * u, -x * v, -x], [0, 0, 0, u, v, 1, -z * u, -z * v, -z]]
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    return H / H[2, 2]


@pytest.mark.parametrize("ratio", [0.85, 1.0])
def test_four_noisy_clicks_place_the_camera_within_15_cm(ratio):
    """CAL v0 fits H from 4 clicked floor points. With 1 px click noise (sigma) and the true f 15 % below the prior,
    the camera nadir is within 0.15 m in 95 % of 100 draws (probe: p95 0.08 m at 45 deg tilt)."""
    H = _cam_H(1330.0 * ratio, CAM, 45, 30)
    img = np.array([[300.0, 700.0], [1600.0, 700.0], [1700.0, 1000.0], [200.0, 1000.0]])
    flo = np.array([(p[:2] / p[2]) for p in (H @ np.c_[img, np.ones(4)].T).T])
    rng = np.random.default_rng(7)
    err = [math.dist(C.camera_fit(_framing(_dlt(img + rng.normal(0, 1.0, img.shape), flo), _K(1330.0))).xz, CAM[:2])
           for _ in range(100)]
    assert np.percentile(err, 95) < 0.15


def test_near_affine_H_uses_the_prior_f():
    fit = C.camera_fit(_framing(_cam_H(1330.0, CAM, 89.5, 0), _K(1330.0)))
    assert fit.f_source == "prior" and fit.f_px == 1330.0
    assert np.allclose(fit.xz, CAM[:2], atol=1e-3)


def test_skewed_K_is_refused():
    K = _K(1330.0)
    K[0, 1] = 5.0
    with pytest.raises(C.CompileError, match="K must be"):
        C.camera_fit(_framing(_cam_H(1330.0, CAM, 45), K))


def test_compiled_zones_record_the_camera_fits(v0):
    for zid, cz in v0.items():
        z = next(z for z in SITE0["zones"] if z["id"] == zid)
        assert list(cz.cameras) == z["framings"]
        assert [c.xz for c in cz.cameras.values()] == cz.scenario.cameras
        assert all(c.f_source == "affine" for c in cz.cameras.values())          # the site_v0 placeholder H
