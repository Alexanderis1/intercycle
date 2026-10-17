"""Static world for the legged-robot sim: the L0 plant geometry as ONE static triangle mesh + the Nav2 occupancy map.

Source of truth: scenario/bodyshop.yaml via intercycle.sim.geometry.build_rects / StaticMap, or for the press-shop twin
the rects + extent from twin/compile.py (build_world_files(rects=, extent=)) (run locally, where the
repo's .venv has scipy + pydantic).  The output is a dict of files (name -> bytes) that the Modal functions write to a
temp dir, so the Isaac python never needs our dependencies:

  boxes.json        every static body as an upright box {id, cls, cx, cy, w, d, yaw, h}  (ROS/Isaac frame, metres)
  factory_l0.usda   one UsdGeom.Mesh "/factory/L0" with PhysicsCollisionAPI (static triangle-mesh collider)
  mesh.npz          the same mesh: vertices (V,3) float32, faces (F,3) int32  (for the Warp lidar ray-cast)
  map.pgm/map.yaml  Nav2 map_server occupancy map (0.1 m) = the robot's prior static map (CAD/SLAM), no dynamics

Frames: plan view (x along the line, z across the hall) maps to ROS/Isaac x = plan x, y = plan z, z up.  A plan yaw
(atan2(dx, dz), 0 = +z) maps to a ROS heading psi = pi/2 - yaw; a rect's plan yaw maps to box yaw = -yaw.
"""
from __future__ import annotations

import io
import json
import math

import numpy as np

# Box heights (m) per static class.  All exceed the lidar plane (~0.8 m) so a 2D scanner sees them; low clutter a real
# 2D scanner would miss is a stated limitation (README).
HEIGHT = {
    "wall": 4.0, "conveyor": 1.0, "curtain_post": 2.0, "lifter_post": 3.0, "robot_base": 1.0, "tip_dresser": 1.2,
    "fixture_table": 0.95, "column": 6.0, "rack_market": 2.5, "rack_dh": 2.2, "crah": 2.0,
}
DEFAULT_H = 2.0


def boxes_from_rects(rects: list[dict]) -> list[dict]:
    out = []
    for r in rects:
        out.append({"id": r["id"], "cls": r["cls"], "cx": r["cx"], "cy": r["cz"], "w": r["w"], "d": r["d"],
                    "yaw": -float(r.get("yaw", 0.0)), "h": HEIGHT.get(r["cls"], DEFAULT_H)})
    return out


def box_corners_xy(b: dict) -> np.ndarray:
    """(4, 2) footprint corners of an upright box, counter-clockwise."""
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    hw, hd = b["w"] / 2, b["d"] / 2
    loc = np.array([(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)])
    return np.stack((b["cx"] + loc[:, 0] * c - loc[:, 1] * s, b["cy"] + loc[:, 0] * s + loc[:, 1] * c), axis=1)


# 12 triangles per box, outward normals (counter-clockwise seen from outside); vertices 0-3 bottom, 4-7 top
_BOX_FACES = np.array([
    (0, 2, 1), (0, 3, 2),            # bottom (normal -z)
    (4, 5, 6), (4, 6, 7),            # top (+z)
    (0, 1, 5), (0, 5, 4),            # side y-
    (1, 2, 6), (1, 6, 5),            # side x+
    (2, 3, 7), (2, 7, 6),            # side y+
    (3, 0, 4), (3, 4, 7),            # side x-
], dtype=np.int32)


def boxes_to_mesh(boxes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Merge all boxes into one triangle mesh: vertices (8N, 3) float32, faces (12N, 3) int32."""
    if not boxes:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    V, F = [], []
    for k, b in enumerate(boxes):
        xy = box_corners_xy(b)
        v = np.concatenate((np.c_[xy, np.zeros(4)], np.c_[xy, np.full(4, b["h"])]))
        V.append(v)
        F.append(_BOX_FACES + 8 * k)
    return np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32)


def mesh_to_usda(V: np.ndarray, F: np.ndarray) -> str:
    """A self-contained USDA layer: Xform /factory (defaultPrim) with Mesh L0 as a static triangle-mesh collider."""
    pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in V.tolist())
    idx = ", ".join(str(i) for i in F.ravel().tolist())
    cnt = ", ".join("3" for _ in range(len(F)))
    if len(V):
        lo, hi = V.min(0), V.max(0)
        ext = f"({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}), ({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f})"
    else:
        ext = "(0, 0, 0), (0, 0, 0)"
    return (
        "#usda 1.0\n(\n    defaultPrim = \"factory\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\n"
        "def Xform \"factory\"\n{\n"
        "    def Mesh \"L0\" (\n        prepend apiSchemas = [\"PhysicsCollisionAPI\", \"PhysicsMeshCollisionAPI\"]\n    )\n    {\n"
        f"        float3[] extent = [{ext}]\n"
        f"        int[] faceVertexCounts = [{cnt}]\n"
        f"        int[] faceVertexIndices = [{idx}]\n"
        f"        point3f[] points = [{pts}]\n"
        "        uniform token subdivisionScheme = \"none\"\n"
        "        uniform token physics:approximation = \"none\"\n"
        "        bool physics:collisionEnabled = 1\n"
        "        color3f[] primvars:displayColor = [(0.55, 0.55, 0.58)]\n"
        "    }\n}\n"
    )


def occupancy_pgm(grid: np.ndarray) -> bytes:
    """StaticMap.grid [nx, nz] (True = occupied) -> binary PGM (row 0 = max y), 0 = occupied, 254 = free."""
    img = np.where(grid.T[::-1], 0, 254).astype(np.uint8)
    h, w = img.shape
    return f"P5\n{w} {h}\n255\n".encode() + img.tobytes()


def map_yaml(res: float, x0: float, y0: float) -> bytes:
    return (f"image: map.pgm\nmode: trinary\nresolution: {res}\norigin: [{x0}, {y0}, 0.0]\nnegate: 0\n"
            f"occupied_thresh: 0.65\nfree_thresh: 0.25\n").encode()


def build_world_files(empty: bool = False, scenario_path: str | None = None, rects: list[dict] | None = None,
                      extent: tuple[float, float, float, float] | None = None) -> dict[str, bytes]:
    """Build every world file.  empty=True: no factory (flat floor, closed only by the map border) for smoke tests.

    rects/extent (twin/compile.py): the static bodies and the plan-view extent (x0, z0, x1, z1) of a twin zone, used
    instead of the body-shop layout (see geometry.StaticMap).  The USD boxes and the map come from the same
    StaticMap.rects(), which in that mode includes the extent's border ring, so unseen floor (passed as wall rects) and
    everything outside the extent are walls in both.  rects needs the twin's own scenario_path (for the dock) and its
    extent (ValueError otherwise), so neither the body-shop scenario nor the body-shop extent is used for a twin.
    """
    from intercycle import scenario as scn_mod
    from intercycle.sim import geometry as G

    if rects is not None and not scenario_path:
        raise ValueError("build_world_files(rects=...) needs the twin's scenario_path")
    if rects is not None and extent is None:
        raise ValueError("build_world_files(rects=...) needs the twin's extent (x0, z0, x1, z1); there is no default")
    scn = scn_mod.load(scenario_path) if scenario_path else scn_mod.load()
    sm = G.StaticMap(scn, rects=rects, extent=extent)
    boxes = [] if empty else boxes_from_rects(sm.rects())
    V, F = boxes_to_mesh(boxes)
    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=V, faces=F)
    grid = sm.grid.copy()
    if empty:
        grid[:] = False
        grid[:2, :] = grid[-2:, :] = grid[:, :2] = grid[:, -2:] = True
    meta = {"empty": empty, "n_boxes": len(boxes), "n_tris": int(len(F)), "res": G.RES,
            "extent": list(sm.extent), "dock": list(scn.robot.get("dock", [4, 32])),
            "frames": "ROS x = plan x, ROS y = plan z, z up; psi = pi/2 - plan_yaw"}
    return {
        "boxes.json": json.dumps(boxes).encode(),
        "factory_l0.usda": mesh_to_usda(V, F).encode(),
        "mesh.npz": buf.getvalue(),
        "map.pgm": occupancy_pgm(grid),
        "map.yaml": map_yaml(G.RES, sm.x0, sm.z0),
        "world.json": json.dumps(meta).encode(),
    }


def write_world(files: dict[str, bytes], dest: str) -> str:
    import os

    os.makedirs(dest, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(dest, name), "wb") as f:
            f.write(data)
    return dest


def load_boxes(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["vertices"], d["faces"]
