"""R-W4 (slam_toolbox map building is CUT, PLAN_TWIN §6): the two localisation-prior fixes that are kept.

1. Start-pose noise: the robot does not start exactly where it believes it is. `perturb_start` draws the TRUE start
   pose uniformly within +-0.5 m (x and y) and +-10 deg (heading) of the nominal dock pose, seeded by the caller's key,
   collision-free against the static boxes. The episode puts the true pose in the simulator cfg ("start") and keeps the
   nominal pose as the robot's belief (timing_planner `init`, AMCL's initial pose).
2. Unmapped clutter: `add_unmapped_clutter` adds pallets, carts and bins to the simulated world (boxes.json, mesh.npz,
   factory_l0.usda: the lidar sees them, the camera is occluded by them, the robot collides with them) but NOT to the
   Nav2 map (map.pgm / map.yaml unchanged), seeded per (world, day). Placement keeps clear of the given points (dock,
   wait and inspection poses) and rectangles (hazard zones), and never cuts the free floor between those points for a
   robot disc of ROBOT_R (checked on the map grid), so a mission stays feasible.
Sizes and heights are ASSUMED (no clutter was measured); every item is taller than the 2D lidar plane (about 0.8 m):
low clutter a 2D scanner cannot see (bare pallets, forks) is excluded and disclosed (3D lidar is cut).
Simulator-side only (world truth); nothing here reaches ROS except through the sensors.
"""
from __future__ import annotations

import io
import json
import math
import re

import numpy as np

START_DXY = 0.5
START_DPSI = math.radians(10.0)
ROBOT_R = 0.55                    # body half-length 0.5 m + margin: the disc that must still pass between clutter
FOOT_HALF = (0.50, 0.32)          # robot footprint half-length / half-width (nav2.yaml), checked at the start heading
FOOT_CLEAR_M = 0.05               # ASSUMED clearance between the start footprint and any static box
CLEAR_R = 1.5                     # no clutter within this distance of a keep-clear point
# ASSUMED catalogue: (cls, length, width, height), all above the lidar plane
CATALOGUE = (("clutter_pallet_loaded", 1.2, 0.8, 1.2), ("clutter_cart", 1.0, 0.6, 1.0), ("clutter_bin", 0.6, 0.4, 0.9))


def _box_gap(x: float, y: float, boxes: list[dict]) -> float:
    """Distance from a point to the nearest static box footprint (0 inside)."""
    if not boxes:
        return float("inf")
    b = np.array([(q["cx"], q["cy"], q["w"] / 2, q["d"] / 2, q["yaw"]) for q in boxes], float)
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    dx, dy = x - b[:, 0], y - b[:, 1]
    lx, ly = np.abs(dx * c + dy * s), np.abs(-dx * s + dy * c)
    return float(np.hypot(np.maximum(lx - b[:, 2], 0), np.maximum(ly - b[:, 3], 0)).min())


def footprint_hits(x: float, y: float, psi: float, boxes: list[dict], clear_m: float = FOOT_CLEAR_M,
                   half: tuple[float, float] = FOOT_HALF) -> bool:
    """True if the robot's ORIENTED footprint (half-length x half-width at heading psi), grown by clear_m on every side,
    overlaps any static box footprint (exact separating-axis test; the corner radius hypot(0.50, 0.32) = 0.594 m is
    what a centre-distance check misses)."""
    if not boxes:
        return False
    from intercycle_sim.safety_scanner import _boxes_overlap, _rect_corners

    b = np.array([(q["cx"], q["cy"], q["w"] / 2, q["d"] / 2, q["yaw"]) for q in boxes], float)
    fa = _rect_corners(x, y, half[0] + clear_m, half[1] + clear_m, psi)
    fb = _rect_corners(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])
    return bool(_boxes_overlap(fa, fb).any())


def perturb_start(nominal: tuple[float, float, float], seed: int, boxes: list[dict] | None = None,
                  dxy: float = START_DXY, dpsi: float = START_DPSI, clear_m: float = FOOT_CLEAR_M,
                  max_tries: int = 200) -> dict:
    """{"true": [x, y, psi] (simulator start), "belief": nominal (the robot's initial pose), "error": [dx, dy, dpsi]}.
    The drawn pose is accepted only if the oriented footprint at the DRAWN heading, grown by clear_m, touches no static
    box (footprint_hits). Deterministic in seed. Raises if no collision-free pose is found (fail closed: never silently
    the exact pose)."""
    rng = np.random.default_rng(int(seed))
    x0, y0, p0 = (float(v) for v in nominal)
    for _ in range(max_tries):
        ex, ey = rng.uniform(-dxy, dxy, 2)
        ep = rng.uniform(-dpsi, dpsi)
        x, y = x0 + float(ex), y0 + float(ey)
        psi = math.atan2(math.sin(p0 + ep), math.cos(p0 + ep))
        if not footprint_hits(x, y, psi, boxes or [], clear_m):
            return {"true": [x, y, psi], "belief": [x0, y0, p0], "error": [float(ex), float(ey), float(ep)],
                    "seed": int(seed), "range": {"dxy_m": dxy, "dpsi_rad": dpsi}, "clear_m": clear_m}
    raise RuntimeError(f"perturb_start: no collision-free start within +-{dxy} m of {nominal}")


# ------------------------------------------------------------------------------------------------ map grid
def read_map(files: dict[str, bytes]) -> tuple[np.ndarray, float, float, float]:
    """(occupied [nx, ny] bool, res, x0, y0) from world.py's map.pgm / map.yaml (PGM row 0 = max y)."""
    pgm = files["map.pgm"]
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", pgm)
    if m is None:
        raise ValueError("map.pgm is not a binary PGM")
    w, h = int(m.group(1)), int(m.group(2))
    img = np.frombuffer(pgm[m.end():m.end() + w * h], np.uint8).reshape(h, w)
    occ = (img[::-1].T == 0)
    y = files["map.yaml"].decode()
    res = float(re.search(r"resolution:\s*([-\d.eE]+)", y).group(1))          # type: ignore[union-attr]
    org = re.search(r"origin:\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)", y)
    if org is None:
        raise ValueError("map.yaml has no origin")
    return occ, res, float(org.group(1)), float(org.group(2))


def _footprint_cells(cx, cy, hl, hw, yaw, res, x0, y0, shape) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = shape
    r = math.hypot(hl, hw)
    i0, i1 = max(0, int((cx - r - x0) / res)), min(nx, int((cx + r - x0) / res) + 1)
    j0, j1 = max(0, int((cy - r - y0) / res)), min(ny, int((cy + r - y0) / res) + 1)
    if i0 >= i1 or j0 >= j1:
        return np.zeros(0, int), np.zeros(0, int)
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    px, py = x0 + (ii + 0.5) * res - cx, y0 + (jj + 0.5) * res - cy
    c, s = math.cos(yaw), math.sin(yaw)
    inside = (np.abs(px * c + py * s) <= hl) & (np.abs(-px * s + py * c) <= hw)
    return ii[inside], jj[inside]


def _disc(res: float) -> np.ndarray:
    k = int(math.ceil(ROBOT_R / res))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx * xx + yy * yy) <= k * k


def _blocked(occ: np.ndarray, res: float) -> np.ndarray:
    """Cells a robot disc of ROBOT_R cannot be centred on."""
    from scipy import ndimage

    return ndimage.binary_dilation(occ, structure=_disc(res))


def _block_cells(blocked: np.ndarray, ii: np.ndarray, jj: np.ndarray, res: float) -> np.ndarray:
    """blocked plus the robot-disc dilation of the cells (ii, jj), computed in a local window."""
    from scipy import ndimage

    k = int(math.ceil(ROBOT_R / res))
    i0, i1 = max(0, int(ii.min()) - k), min(blocked.shape[0], int(ii.max()) + k + 1)
    j0, j1 = max(0, int(jj.min()) - k), min(blocked.shape[1], int(jj.max()) + k + 1)
    loc = np.zeros((i1 - i0, j1 - j0), bool)
    loc[ii - i0, jj - j0] = True
    out = blocked.copy()
    out[i0:i1, j0:j1] |= ndimage.binary_dilation(loc, structure=_disc(res))
    return out


def _connected_blocked(blocked: np.ndarray, res: float, x0: float, y0: float, pts: list[tuple[float, float]]) -> bool:
    from scipy import ndimage

    lab = np.asarray(ndimage.label(~blocked)[0])  # type: ignore[index]
    ids = set()
    for x, y in pts:
        i, j = int((x - x0) / res), int((y - y0) / res)
        if not (0 <= i < blocked.shape[0] and 0 <= j < blocked.shape[1]) or lab[i, j] == 0:
            return False
        ids.add(int(lab[i, j]))
    return len(ids) == 1


def _connected(occ: np.ndarray, res: float, x0: float, y0: float, pts: list[tuple[float, float]]) -> bool:
    """Every point reachable from the first for a robot disc of ROBOT_R on the grid."""
    return _connected_blocked(_blocked(occ, res), res, x0, y0, pts)


def add_unmapped_clutter(files: dict[str, bytes], seed: int, n: int, keep_clear: list[tuple[float, float]],
                         avoid_rects: list[tuple[float, float, float, float]] | None = None,
                         max_tries: int = 400) -> tuple[dict[str, bytes], list[dict]]:
    """New world files with n clutter boxes in boxes.json / mesh.npz / factory_l0.usda and the map files untouched.
    keep_clear: the dock, wait and inspection poses (they stay mutually reachable). Deterministic in seed."""
    from intercycle_sim import world as Wd

    if not keep_clear:
        raise ValueError("add_unmapped_clutter needs the keep-clear points (dock, wait and inspection poses)")
    occ, res, x0, y0 = read_map(files)
    blocked = _blocked(occ, res)
    if not _connected_blocked(blocked, res, x0, y0, keep_clear):
        raise ValueError("the keep-clear points are not mutually reachable on the map before any clutter")
    boxes = json.loads(files["boxes.json"].decode())
    free = np.argwhere(~occ)
    rng = np.random.default_rng(int(seed))
    grid = occ.copy()
    placed: list[dict] = []
    tries = 0
    while len(placed) < n and tries < max_tries:
        tries += 1
        cls, ln, wd, h = CATALOGUE[int(rng.integers(len(CATALOGUE)))]
        i, j = free[int(rng.integers(len(free)))]
        cx, cy = x0 + (i + 0.5) * res, y0 + (j + 0.5) * res
        yaw = float(rng.choice([0.0, math.pi / 2])) + float(rng.uniform(-0.15, 0.15))
        hl, hw = ln / 2, wd / 2
        r = math.hypot(hl, hw)
        if any(math.hypot(cx - px, cy - py) < CLEAR_R + r for px, py in keep_clear):
            continue
        if any(a - r <= cx <= c + r and b - r <= cy <= d + r for a, b, c, d in (avoid_rects or [])):
            continue
        ii, jj = _footprint_cells(cx, cy, hl, hw, yaw, res, x0, y0, grid.shape)
        if len(ii) == 0 or grid[ii, jj].any():
            continue
        trial = _block_cells(blocked, ii, jj, res)
        if not _connected_blocked(trial, res, x0, y0, keep_clear):
            continue
        blocked = trial
        grid[ii, jj] = True
        placed.append({"id": f"CL{len(placed):02d}", "cls": cls, "cx": round(cx, 3), "cy": round(cy, 3), "w": ln,
                       "d": wd, "yaw": round(yaw, 4), "h": h})
    if len(placed) < n:
        raise RuntimeError(f"add_unmapped_clutter: placed {len(placed)} of {n} items in {max_tries} tries")
    allb = boxes + placed
    V, F = Wd.boxes_to_mesh(allb)
    buf = io.BytesIO()
    np.savez_compressed(buf, vertices=V, faces=F)
    out = dict(files)
    out["boxes.json"] = json.dumps(allb).encode()
    out["mesh.npz"] = buf.getvalue()
    out["factory_l0.usda"] = Wd.mesh_to_usda(V, F).encode()
    meta = json.loads(files.get("world.json", b"{}").decode() or "{}")
    meta["unmapped_clutter"] = {"seed": int(seed), "n": len(placed), "items": placed, "in_map": False,
                                "source": "ASSUMED catalogue (R-W4)"}
    meta["n_boxes"], meta["n_tris"] = len(allb), int(len(F))
    out["world.json"] = json.dumps(meta).encode()
    return out, placed
