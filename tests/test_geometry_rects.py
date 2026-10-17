"""S1b: StaticMap(scn, rects=, extent=) and world.build_world_files(rects=, extent=) for the press-shop twin (R-W10).

1. Body-shop regression: with rects=None and extent=None the static map, its queries and every world file are
   bit-identical to the outputs of the code before this change (golden sha256 values computed on commit 8c52e12).
2. Twin mode: compile.py passes axis-aligned rects plus an extent; the raster, the queries and the USD/map follow them,
   and unseen floor (given as wall rects, plus the ring outside the extent) is a wall in both the USD and the map.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"))

from intercycle.scenario import load  # noqa: E402
from intercycle.sim import geometry as G  # noqa: E402
from intercycle_sim import world as W  # noqa: E402


def _h(a) -> str:
    if isinstance(a, np.ndarray):
        return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()
    if isinstance(a, bytes):
        return hashlib.sha256(a).hexdigest()
    return hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()


def fingerprint_map(sm) -> dict[str, str]:
    """sha256 of every array the map holds and of a fixed set of query results."""
    rng = np.random.default_rng(1234)
    pts = np.c_[rng.uniform(-3.0, 123.0, 400), rng.uniform(-3.0, 51.0, 400)]
    few = pts[:6]
    n = 180
    a = np.arange(n) * (2 * math.pi / n)
    ux, uz = np.sin(a), np.cos(a)
    poses = [(4.0, 32.0), (50.0, 16.0), (100.0, 24.0), (60.0, 33.0), (0.5, 0.5)]
    dyn = np.array([[5.0, 33.0, 0.3], [51.0, 17.5, 0.5]])
    return {
        "grid": _h(sm.grid), "dist": _h(sm.dist), "sdf": _h(sm._sdf), "cf": _h(sm._cf), "occ": _h(sm._occ),
        "rects": _h(sm.rects()), "arms": _h([list(b) for b in sm.arm_bases]),
        "clear_many": _h(sm.clearance(pts)), "clear_few": _h(sm.clearance(few)),
        "free": _h([sm.free(float(x), float(z), 0.42) for x, z in few]),
        "cell": _h([sm.cell(float(x), float(z)) for x, z in few]),
        "trace": _h(np.stack([sm.trace(x, z, ux, uz, 12.0) for x, z in poses])),
        "sphere": _h(np.stack([sm.trace_sphere(x, z, ux, uz, 12.0) for x, z in poses])),
        "lidar": _h(np.stack([G.lidar(sm, x, z, 0.3, dyn) for x, z in poses])),
        "seg": _h([G.segment_hit(sm, (4.0, 32.0), (60.0, 32.0)), G.segment_hit(sm, (4.0, 32.0), (4.0, 0.0))]),
    }


def fingerprint_world(files: dict[str, bytes]) -> dict[str, str]:
    """sha256 per file; mesh.npz is hashed on its decoded arrays (zip/zlib bytes may differ between library builds)."""
    out = {k: _h(v) for k, v in sorted(files.items()) if k != "mesh.npz"}
    d = np.load(io.BytesIO(files["mesh.npz"]))
    out["mesh.npz"] = _h(np.concatenate((d["vertices"].ravel().view(np.uint32), d["faces"].ravel().view(np.uint32))))
    return out


# Golden values: the body-shop outputs of intercycle/sim/geometry.py and world.py at commit 8c52e12 (before rects/extent).
GOLD_MAP = {
    "grid": "23ca9dd12a6225d165b74082a2ab60fa0cae9bb4f74aa2461a6963457b881973",
    "dist": "f48a36a145e5b3790c6899a34b25210934178244974f12b0115075663c81d8ad",
    "sdf": "0519079e21c07eff92262013f3dba90702e95c3d5f474cf3254624a8410ccfb6",
    "cf": "94aadbf8c2a18a33bd353cac5a16e986bd83c9398050c0a6d97fdaa621b22c89",
    "occ": "1e54b69fb2eb69a81923af42330c313b7025ed0ec44f550508418a70b3dfe1b8",
    "rects": "98dbfbc3f73298a40e771c52300f44329932f63b924125a2590673212484b903",
    "arms": "8833375673ab145a3dccfc05368a6825851ae846d7643b619ce47734d031dd36",
    "clear_many": "26646d0e58d5d310eab4b006bdbdf7a16b41cc2ba07d3b8b1e13f34340889ef1",
    "clear_few": "e15dd70aa1298ba000637f016ff914d26cd288d775d398aa86dd82b727db7a37",
    "free": "467bb2abb6ed0d905c225f2ec4cbf2de201cf0832914d44a6f12f0480bdbc921",
    "cell": "cbf478b2b9ad3dc99b29c9f1d7c23cee794a52850ee2e5134aae03021131b54a",
    "trace": "98a39bc97c05d885a87d5755d3de3b1d8adc591ca33fe25a35410f7bb4428805",
    "sphere": "89396800dca9546e467ef7313205d634cf015579e791a86ba4e86d0c536625a0",
    "lidar": "1722b1e298654b7d478f3e3f4a2b906aa76bcd60273b8eb230e5ff18e89f0a71",
    "seg": "1fb89d6b44ce2bc227000905e338efa28ca7a6ce4640f7a79be9ec033f1ceb28",
}
GOLD_WORLD = {
    "boxes.json": "289faedc38ec420bd7f85b7ed98c547ec9425da0b3ff39b457601bb0e263c33c",
    "factory_l0.usda": "ba97aa7ccfb580bd3f0bcea24b61091d19e08f6e22efe45a6450593a7177eaee",
    "map.pgm": "6948fa21e82ef32a63b8ed0ebfd39e5ff50c4d0fb79d70a514c9ad8d58ad27a0",
    "map.yaml": "15208cfe7478c1b8acba8ebba78b9023946289a96963389e7916fc452f3f130f",
    "world.json": "7dfa45aa00813cb71cfc9803fb12632d5c2fb834dcebb47cbed1f6ed9d8df5d9",
    "mesh.npz": "5e041d3cc30e8fb02da7111355e63b1977e76ccceb17980832629d5d02073ad2",
}
GOLD_WORLD_EMPTY = {
    "boxes.json": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "factory_l0.usda": "744757f65f9089b33aab0ec57b3e32a343fd2c7d7d6a72ecc446717dbe5a72c2",
    "map.pgm": "ae6507963d42f9943f56fc2a2748ee6acb5a8c4d909b7ba897d7e7ef46e1e1db",
    "map.yaml": "15208cfe7478c1b8acba8ebba78b9023946289a96963389e7916fc452f3f130f",
    "world.json": "327182cc49c55572f9b135272ad7239cb69f106ac0854501555de13307ca2f7f",
    "mesh.npz": "c9dda96f84e8c682be0dbedfa8de284476fd7068ce632bd79ceeecb2add6be4c",
}


@pytest.fixture(scope="module")
def scn():
    return load()


# ------------------------------------------------------------------------------------------- body-shop regression
def test_bodyshop_static_map_bit_identical(scn):
    assert fingerprint_map(G.StaticMap(scn)) == GOLD_MAP


def test_bodyshop_static_map_explicit_default_extent_bit_identical(scn):
    assert fingerprint_map(G.StaticMap(scn, extent=(G.X0, G.Z0, G.X1, G.Z1))) == GOLD_MAP


def test_bodyshop_world_files_bit_identical():
    assert fingerprint_world(W.build_world_files()) == GOLD_WORLD
    assert fingerprint_world(W.build_world_files(empty=True)) == GOLD_WORLD_EMPTY


# ------------------------------------------------------------------------------------------- twin mode
EXT = (-3.0, -1.5, 17.0, 8.5)                       # 20 x 10 m, origin not at 0 (200 x 100 cells)
TWIN_RECTS = [
    {"id": "UNSEEN_0", "cls": "wall", "cx": 14.5, "cz": 3.5, "w": 5.0, "d": 10.0},       # unseen floor x 12..17
    {"id": "P1", "cls": "press", "cx": 2.0, "cz": 3.5, "w": 2.0, "d": 2.0, "yaw": 0.0},  # press footprint x 1..3
]
SCENARIO = str(Path(__file__).resolve().parents[1] / "scenario" / "bodyshop.yaml")


@pytest.fixture(scope="module")
def twin():
    return G.StaticMap(None, rects=TWIN_RECTS, extent=EXT)


def test_twin_raster_follows_extent(twin):
    assert (twin.nx, twin.nz) == (200, 100) and twin.grid.shape == (200, 100) and twin.dist.shape == (200, 100)
    assert twin.extent == EXT
    assert twin.cell(-3.0 + 0.05, -1.5 + 0.05) == (0, 0)
    assert twin.cell(16.95, 8.45) == (199, 99)
    assert twin.grid[:2, :].all() and twin.grid[-2:, :].all() and twin.grid[:, :2].all() and twin.grid[:, -2:].all()
    assert not twin.grid[70:140, 5:95].any()                    # free floor x 4..11 between the press and unseen floor
    assert twin.grid[151:198, 2:98].all()                       # unseen floor x 12.1..16.8 is solid
    assert twin.arm_bases == []


def test_module_constants_untouched(twin):
    assert (G.NX, G.NZ, G.X0, G.Z0, G.X1, G.Z1) == (1240, 520, -2.0, -2.0, 122.0, 50.0)


def test_twin_queries(twin):
    # (7.5, 3.5): press face at x=3 (4.5 m), unseen-floor wall face at x=12 (4.5 m), ring faces at z=-1.3 / 8.3 (4.8 m)
    c = float(twin.clearance(np.array([[7.5, 3.5]]))[0])
    assert c == pytest.approx(4.5, abs=0.05)
    many = twin.clearance(np.tile([[7.5, 3.5]], (20, 1)))      # numpy path agrees with the scalar path
    assert np.allclose(many, c, atol=1e-9)
    assert float(twin.clearance(np.array([[14.0, 3.5]]))[0]) == 0.0         # inside unseen floor
    assert float(twin.clearance(np.array([[2.0, 3.5]]))[0]) == 0.0          # inside the press
    assert float(twin.clearance(np.array([[-10.0, 3.5]]))[0]) == 0.0        # outside the world
    assert twin.free(7.5, 3.5, 0.42) and not twin.free(11.8, 3.5, 0.42)
    # ray 0 straight ahead, yaw pi/2 = +x: unseen floor at 5 m; ray n/2 = -x: press face at 4 m
    none = np.zeros((0, 3))
    r = G.lidar(twin, 7.0, 3.5, math.pi / 2, none, n=180, rmax=12.0)
    assert r[0] == pytest.approx(5.0, abs=0.06) and r[90] == pytest.approx(4.0, abs=0.06)
    rs = G.lidar(twin, 7.0, 3.5, math.pi / 2, none, n=180, rmax=12.0, exact=True)
    assert rs[0] == pytest.approx(5.0, abs=0.04) and rs[90] == pytest.approx(4.0, abs=0.04)
    assert G.segment_hit(twin, (7.0, 3.5), (16.0, 3.5)) is not None
    assert G.segment_hit(twin, (5.0, 1.0), (9.0, 6.0)) is None


def test_twin_rects_include_closed_border(twin):
    rects = twin.rects()
    assert [r["id"] for r in rects[:2]] == ["UNSEEN_0", "P1"]
    border = [r for r in rects if r["id"].startswith("BORDER_")]
    assert len(rects) == 6 and len(border) == 4 and all(r["cls"] == "wall" for r in border)
    assert rects[0]["yaw"] == 0.0                               # yaw defaults to 0 (axis-aligned)
    # the ring is exactly the border rects: re-painting rects() alone reproduces the grid
    again = G.StaticMap(None, rects=rects, extent=EXT)
    assert np.array_equal(again.grid, twin.grid)


def test_twin_bad_inputs():
    ok = {"id": "X", "cls": "wall", "cx": 1.0, "cz": 1.0, "w": 1.0, "d": 1.0}
    with pytest.raises(ValueError):
        G.StaticMap(None, rects=[ok], extent=(0.0, 0.0, 10.05, 5.0))        # not a whole number of cells
    with pytest.raises(ValueError):
        G.StaticMap(None, rects=[ok], extent=(0.0, 0.0, -10.0, 5.0))        # empty
    with pytest.raises(ValueError):
        G.StaticMap(None, rects=[dict(ok, cx=50.0, cz=50.0)], extent=EXT)   # entirely outside the extent
    with pytest.raises(ValueError):
        G.StaticMap(None, rects=[dict(ok, w=0.0)], extent=EXT)              # degenerate
    with pytest.raises(ValueError):
        G.StaticMap(None, rects=[{k: v for k, v in ok.items() if k != "id"}], extent=EXT)   # no id
    with pytest.raises(ValueError):
        G.StaticMap(None)                                                   # no scenario and no rects
    with pytest.raises(ValueError):
        W.build_world_files(rects=[ok], extent=EXT)                         # twin files need their own scenario


@pytest.fixture(scope="module")
def twin_files():
    return W.build_world_files(scenario_path=SCENARIO, rects=TWIN_RECTS, extent=EXT)


def _pgm(files) -> np.ndarray:
    parts = files["map.pgm"].split(b"\n", 3)
    w, h = (int(v) for v in parts[1].split())
    return np.frombuffer(parts[3], np.uint8).reshape(h, w)


def test_twin_world_files_follow_extent(twin_files):
    assert _pgm(twin_files).shape == (100, 200)
    y = twin_files["map.yaml"].decode()
    assert "origin: [-3.0, -1.5, 0.0]" in y and "resolution: 0.1" in y
    meta = json.loads(twin_files["world.json"])
    assert meta["extent"] == list(EXT) and meta["n_boxes"] == 6 and meta["n_tris"] == 72


def test_unseen_floor_is_wall_in_usd_and_map(twin_files):
    """At every cell centre: occupied in the map <=> inside a USD box.  Unseen floor and the ring outside the extent
    are walls in both, so Nav2 cannot plan into them and Isaac cannot walk into them."""
    img = _pgm(twin_files)
    boxes = json.loads(twin_files["boxes.json"])
    by_id = {b["id"]: b for b in boxes}
    assert by_id["UNSEEN_0"]["cls"] == "wall" and by_id["UNSEEN_0"]["h"] == W.HEIGHT["wall"]
    assert sum(b["id"].startswith("BORDER_") for b in boxes) == 4
    V, F = (np.load(io.BytesIO(twin_files["mesh.npz"]))[k] for k in ("vertices", "faces"))
    assert V.shape == (48, 3) and F.shape == (72, 3)

    def in_box(x, y):
        for b in boxes:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            u, v = (x - b["cx"]) * c + (y - b["cy"]) * s, -(x - b["cx"]) * s + (y - b["cy"]) * c
            if abs(u) <= b["w"] / 2 and abs(v) <= b["d"] / 2:
                return True
        return False

    for i in range(200):
        for j in range(100):
            x, y = EXT[0] + 0.05 + 0.1 * i, EXT[1] + 0.05 + 0.1 * j
            assert (img[99 - j, i] == 0) == in_box(x, y), (x, y)
