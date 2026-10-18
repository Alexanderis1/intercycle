"""twin/<zone>/{boxes.json, static.usda, mesh.npz, map.pgm, map.yaml} via intercycle_sim/world.py (PLAN_TWIN §2):
unseen floor is a wall in both the USD and the map; the committed files are what a fresh build of their recorded
inputs produces."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from twin import build_worlds as B
from twin import compile as C
from twin.schemas.prereg import SplitRow
from twin.schemas.site import point_in_polygon

REPO = Path(__file__).resolve().parents[1]
TWIN = REPO / "twin"
TRAIN0 = {"0_tr0": SplitRow(clip_id="0_tr0", group_id="g", split="train", sha256="0" * 64)}
V0 = {"site": TWIN / "site_v0.yaml", "dynamics": TWIN / "dynamics_v0.json", "tasks": TWIN / "tasks_v0.yaml"}
L_FLOOR = [[0.0, 0.0], [14.0, 0.0], [14.0, 4.0], [10.5, 4.0], [10.5, 10.0], [0.0, 10.0]]


def read_map(d: Path) -> tuple[np.ndarray, float, float, float]:
    """map.pgm + map.yaml -> occupied[nx, nz] (i along +x, j along +z), res, x0, z0."""
    meta = yaml.safe_load((d / "map.yaml").read_text(encoding="utf-8"))
    raw = (d / "map.pgm").read_bytes()
    m = re.match(rb"P5\n(\d+) (\d+)\n255\n", raw)
    assert m, "map.pgm is not a binary PGM"
    w, h = int(m.group(1)), int(m.group(2))
    img = np.frombuffer(raw[m.end():], np.uint8).reshape(h, w)
    occ = (img == 0)[::-1].T                                    # row 0 = max z
    assert set(np.unique(img)) <= {0, 254}
    return occ, float(meta["resolution"]), float(meta["origin"][0]), float(meta["origin"][1])


def read_usd_boxes(d: Path) -> np.ndarray:
    """static.usda points -> (N, 5) boxes (x0, z0, x1, z1, top); every box is 8 points (world.boxes_to_mesh)."""
    text = (d / "static.usda").read_text(encoding="utf-8")
    m = re.search(r"point3f\[\] points = \[(.*?)\]", text, re.S)
    assert m, "no points in static.usda"
    pts = np.array([[float(v) for v in t.split(",")] for t in re.findall(r"\(([^()]*)\)", m.group(1))])
    assert len(pts) % 8 == 0
    boxes = pts.reshape(-1, 8, 3)
    mesh = np.load(d / "mesh.npz")
    assert np.allclose(mesh["vertices"], pts, atol=1e-3), "mesh.npz and static.usda differ"
    # ROS frame: x = plan x, y = plan z
    return np.stack([boxes[:, :, 0].min(1), boxes[:, :, 1].min(1), boxes[:, :, 0].max(1), boxes[:, :, 1].max(1),
                     boxes[:, :, 2].max(1)], axis=1)


def check_unseen_is_wall(d: Path, floor: list[list[float]], feet: list[tuple[float, float, float, float]]) -> int:
    """Every grid cell centre outside the visible floor is occupied in the map AND inside a USD box at least 1 m tall;
    every visible cell clear of the press footprints is free in the map and in no box. Returns the unseen count."""
    occ, res, x0, z0 = read_map(d)
    boxes = read_usd_boxes(d)
    poly = [(float(p[0]), float(p[1])) for p in floor]
    tol = 1e-3
    n_unseen = 0
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            x, z = x0 + (i + 0.5) * res, z0 + (j + 0.5) * res
            in_box = (boxes[:, 0] - tol <= x) & (x <= boxes[:, 2] + tol) & (boxes[:, 1] - tol <= z) & \
                     (z <= boxes[:, 3] + tol)
            if not point_in_polygon(x, z, poly):
                n_unseen += 1
                assert occ[i, j], f"unseen floor ({x:.2f}, {z:.2f}) is free in the map"
                assert (in_box & (boxes[:, 4] >= 1.0)).any(), f"unseen floor ({x:.2f}, {z:.2f}) has no USD wall"
                continue
            h = res / 2 + tol
            near_foot = any(f[0] - h <= x <= f[2] + h and f[1] - h <= z <= f[3] + h for f in feet)
            cell_inside = all(point_in_polygon(x + a * res / 2 * 0.999, z + b * res / 2 * 0.999, poly)
                              for a in (-1, 1) for b in (-1, 1))
            if cell_inside and not near_foot:
                assert not occ[i, j], f"visible floor ({x:.2f}, {z:.2f}) is occupied in the map"
                assert not in_box.any(), f"visible floor ({x:.2f}, {z:.2f}) is inside a USD box"
    return n_unseen


@pytest.fixture(scope="module")
def built_v0(tmp_path_factory):
    out = tmp_path_factory.mktemp("worlds")
    return B.build(V0["site"], V0["dynamics"], V0["tasks"], out, _split_for_tests=TRAIN0), out


def _site_feet(site: dict, zone: str) -> list[tuple[float, float, float, float]]:
    out = []
    for p in site["presses"]:
        if p["zone"] == zone:
            xs, zs = zip(*p["footprint"]["value"])
            out.append((min(xs), min(zs), max(xs), max(zs)))
    return out


def test_build_writes_every_file(built_v0):
    written, _ = built_v0
    assert set(written) == {"z9", "z14"}
    zones = C.compile_paths(V0["site"], V0["dynamics"], V0["tasks"], _split_for_tests=TRAIN0)
    for zid, d in written.items():
        assert sorted(p.name for p in d.iterdir()) == sorted(B.FILES)
        meta = json.loads((d / "world.json").read_text(encoding="utf-8"))
        assert meta["extent"] == list(zones[zid].extent)
        assert meta["dock"] == zones[zid].scenario.robot["dock"]
        inputs = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
        assert inputs["site"]["path"] == "twin/site_v0.yaml"
        assert inputs["tasks"]["sha256"] == B.text_sha256(V0["tasks"])
        assert all(v["stub"] for v in inputs.values())                    # built on the v0 stubs: flagged
        boxes = json.loads((d / "boxes.json").read_text(encoding="utf-8"))
        assert {b["cls"] for b in boxes} == {"press", "wall"}
        cams = json.loads((d / "cameras.json").read_text(encoding="utf-8"))
        assert [c["framing"] for c in cams] == list(zones[zid].cameras)
        assert [tuple(c["xz"]) for c in cams] == [tuple(x) for x in zones[zid].scenario.cameras]
        assert all(c["f_source"] == "affine" and c["f_px"] is None for c in cams)   # site_v0 placeholder H


# ------------------------------------------------------------------------------------------------ CLI fails closed


def _use_split_root(monkeypatch, root: Path) -> None:
    from twin.schemas import prereg as P
    real = P.active_split_path
    monkeypatch.setattr(P, "active_split_path", lambda repo=None: real(root))


def test_cli_refuses_missing_inputs_and_stubs_without_the_flag(tmp_path, monkeypatch):
    # The v1 contract files exist in twin/ now, so "missing inputs" is exercised on an empty tmp twin dir.
    empty = tmp_path / "twin_empty"
    empty.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(B, "default_inputs", lambda allow_stubs=False: C.default_inputs(empty, allow_stubs=allow_stubs))
    with pytest.raises(C.CompileError, match="stubs are not allowed"):
        B.main(["--out", str(out)])
    tmp_path = out
    with pytest.raises(C.CompileError, match="is the G0 v0 stub; pass --allow-stubs"):
        B.main(["--site", str(V0["site"]), "--dynamics", str(V0["dynamics"]), "--tasks", str(V0["tasks"]),
                "--out", str(tmp_path)])
    assert not any(tmp_path.iterdir())


def test_cli_without_an_active_split_fails_closed(tmp_path, monkeypatch):
    _use_split_root(monkeypatch, tmp_path / "norepo")
    with pytest.raises(FileNotFoundError, match="active split .* does not exist"):
        B.main(["--allow-stubs", "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_cli_with_a_dev_source_in_the_active_split_fails_closed(tmp_path, monkeypatch):
    from twin.provenance import ProvenanceError
    site = yaml.safe_load(V0["site"].read_text(encoding="utf-8"))
    site["cal_sources"] = ["0_trS"]
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(site), encoding="utf-8")
    root = tmp_path / "repo"
    (root / "prereg").mkdir(parents=True)
    (root / "prereg" / "split.csv").write_text("clip_id,group_id,split,sha256\n0_trS,g,dev," + "0" * 64 + "\n",
                                               encoding="utf-8")
    _use_split_root(monkeypatch, root)
    with pytest.raises(ProvenanceError, match="0_trS"):
        B.main(["--allow-stubs", "--site", str(p), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_unseen_floor_is_a_wall_in_both_v0(built_v0):
    written, _ = built_v0
    site = yaml.safe_load(V0["site"].read_text(encoding="utf-8"))
    for zid, d in written.items():
        floor = next(z for z in site["zones"] if z["id"] == zid)["floor_polygon"]["value"]
        assert check_unseen_is_wall(d, floor, _site_feet(site, zid)) > 0


def test_unseen_floor_is_a_wall_in_both_l_shaped(tmp_path):
    site = yaml.safe_load(V0["site"].read_text(encoding="utf-8"))
    site = copy.deepcopy(site)
    site["zones"][0]["floor_polygon"]["value"] = L_FLOOR
    p = tmp_path / "site_L.yaml"
    p.write_text(yaml.safe_dump(site), encoding="utf-8")
    tasks = yaml.safe_load(V0["tasks"].read_text(encoding="utf-8"))
    for cp in tasks["checkpoints"]:                      # every z9 capture pose stays on the L's visible floor
        assert cp["zone"] != "z9" or point_in_polygon(cp["candidate_poses"][0]["pose"]["x"],
                                                      cp["candidate_poses"][0]["pose"]["z"], [(q[0], q[1]) for q in L_FLOOR])
    written = B.build(p, V0["dynamics"], V0["tasks"], tmp_path / "out", _split_for_tests=TRAIN0)
    n = check_unseen_is_wall(written["z9"], L_FLOOR, _site_feet(site, "z9"))
    assert n >= 3.5 * 6 / 0.01 - 1                       # the 3.5 x 6 m notch is wall (plus the ring)


def _committed() -> list[Path]:
    return sorted(d for d in TWIN.iterdir() if d.is_dir() and (d / "inputs.json").exists())


def test_committed_worlds_exist():
    assert {d.name for d in _committed()} >= {"z9", "z14"}


@pytest.mark.parametrize("d", _committed(), ids=lambda d: d.name)
def test_committed_world_matches_a_fresh_build_and_walls_unseen_floor(d, tmp_path):
    inputs = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
    paths = {k: REPO / v["path"] for k, v in inputs.items()}
    for k, p in paths.items():
        assert B.text_sha256(p) == inputs[k]["sha256"], f"{d.name}: {p} changed since the build; re-run the build"
    fresh = B.build(paths["site"], paths["dynamics"], paths["tasks"], tmp_path, _split_for_tests=TRAIN0)[d.name]
    for name in ("scenario.yaml", "boxes.json", "static.usda", "map.yaml", "world.json", "inputs.json", "cameras.json",
                 "map.pgm"):
        a, b = (d / name).read_bytes(), (fresh / name).read_bytes()
        if name != "map.pgm":
            a, b = a.replace(b"\r\n", b"\n"), b.replace(b"\r\n", b"\n")
        assert a == b, f"{d.name}/{name} differs from a fresh build"
    m0, m1 = np.load(d / "mesh.npz"), np.load(fresh / "mesh.npz")
    assert np.array_equal(m0["vertices"], m1["vertices"]) and np.array_equal(m0["faces"], m1["faces"])
    site = yaml.safe_load(paths["site"].read_text(encoding="utf-8"))
    floor = next(z for z in site["zones"] if z["id"] == d.name)["floor_polygon"]["value"]
    assert check_unseen_is_wall(d, floor, _site_feet(site, d.name)) > 0


def test_check_catches_a_free_unseen_cell(built_v0, tmp_path):
    """Self-test: a map with a hole in the unseen ring fails the check."""
    written, _ = built_v0
    d = tmp_path / "broken"
    d.mkdir()
    src = written["z9"]
    for name in B.FILES:
        (d / name).write_bytes((src / name).read_bytes())
    raw = bytearray((d / "map.pgm").read_bytes())
    hdr = re.match(rb"P5\n(\d+) (\d+)\n255\n", bytes(raw))
    assert hdr
    raw[hdr.end()] = 254                                  # top-left corner cell (unseen) -> free
    (d / "map.pgm").write_bytes(bytes(raw))
    site = yaml.safe_load(V0["site"].read_text(encoding="utf-8"))
    with pytest.raises(AssertionError, match="free in the map"):
        check_unseen_is_wall(d, site["zones"][0]["floor_polygon"]["value"], _site_feet(site, "z9"))
