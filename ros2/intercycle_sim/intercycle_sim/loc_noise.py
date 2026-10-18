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

Wiring and keys (fix round 2): `episode.run_episode` calls `apply_loc_noise(cfg, world_files)`. The draws are keyed by
`start_seed(world_seed, day, slot)` and `clutter_seed(world_seed, day)`, derived like twin.rng.keyed_rng (SeedSequence
of world seed, day, blake2b64(entity), blake2b64(event)), so they are pure functions of (world, day, slot) and never of
the condition: every condition of a (world, day, slot) gets the same start error and the same clutter (common random
numbers for the paired primary). cfg contract:
  start        [x, y, psi] nominal dock pose = the robot's belief (planner init, AMCL initial pose)
  start_noise  {"world_seed": int >= 0, "day": int >= 0, "slot": int | str}; no other key is accepted (a condition, or
               anything run-specific, cannot enter the key)
  clutter      {"n": int >= 1, "keep_clear": [[x, y], ...], "avoid_rects": [[x0, y0, x1, y1], ...]}; needs
               start_noise (its world_seed and day key the placement); the dock (start) is always kept clear
The simulator gets the TRUE pose as cfg "start"; the world files get the clutter (map files unchanged); the record
(`loc_noise` in the episode result) holds the key, the seeds, the start error and the clutter items.
"""
from __future__ import annotations

import hashlib
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
START_ENTITY, CLUTTER_ENTITY = "robot_start_pose", "unmapped_clutter"
START_NOISE_KEYS = ("world_seed", "day", "slot")
CLUTTER_KEYS = ("n", "keep_clear", "avoid_rects")


# ------------------------------------------------------------------------------------------------ keyed seeds
def key64(s: str) -> int:
    """= twin.rng.key64 (stable 64-bit blake2b key of a string; asserted equal in tests/test_loc_noise.py)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _keyed_seed(world_seed: int, day: int, entity: str, event: str) -> int:
    if isinstance(world_seed, bool) or isinstance(day, bool) or int(world_seed) != world_seed or int(day) != day:
        raise ValueError("world_seed and day must be integers")
    if int(world_seed) < 0 or int(day) < 0:
        raise ValueError("world_seed and day must be non-negative")
    ss = np.random.SeedSequence([int(world_seed), int(day), key64(entity), key64(event)])
    return int(ss.generate_state(1, np.uint64)[0] >> np.uint64(1))       # 63-bit int for np.random.default_rng


def start_seed(world_seed: int, day: int, slot) -> int:
    """Seed of the start-pose draw: a pure function of (world, day, slot)."""
    return _keyed_seed(world_seed, day, START_ENTITY, f"slot:{slot}")


def clutter_seed(world_seed: int, day: int) -> int:
    """Seed of the clutter placement: a pure function of (world, day)."""
    return _keyed_seed(world_seed, day, CLUTTER_ENTITY, "place")


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


# numpy only (fix round 2): this now runs inside the episode container, whose Isaac python 3.11 has no scipy
# (intercycle_eval/social.py). tests/test_loc_noise.py checks both against scipy.ndimage on random grids.
def _dilate(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """= scipy.ndimage.binary_dilation(mask, structure) for a symmetric structure (border 0)."""
    k0, k1 = structure.shape[0] // 2, structure.shape[1] // 2
    nx, ny = mask.shape
    pad = np.zeros((nx + 2 * k0, ny + 2 * k1), bool)
    pad[k0:k0 + nx, k1:k1 + ny] = mask
    out = np.zeros_like(mask)
    for di, dj in np.argwhere(structure):
        out |= pad[di:di + nx, dj:dj + ny]
    return out


def _blocked(occ: np.ndarray, res: float) -> np.ndarray:
    """Cells a robot disc of ROBOT_R cannot be centred on."""
    return _dilate(occ, _disc(res))


def _block_cells(blocked: np.ndarray, ii: np.ndarray, jj: np.ndarray, res: float) -> np.ndarray:
    """blocked plus the robot-disc dilation of the cells (ii, jj), computed in a local window."""
    k = int(math.ceil(ROBOT_R / res))
    i0, i1 = max(0, int(ii.min()) - k), min(blocked.shape[0], int(ii.max()) + k + 1)
    j0, j1 = max(0, int(jj.min()) - k), min(blocked.shape[1], int(jj.max()) + k + 1)
    loc = np.zeros((i1 - i0, j1 - j0), bool)
    loc[ii - i0, jj - j0] = True
    out = blocked.copy()
    out[i0:i1, j0:j1] |= _dilate(loc, _disc(res))
    return out


def _spread_rows(free: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """Every run of free cells along axis 1 that holds a reached cell becomes reached (runs never cross rows)."""
    nx, ny = free.shape
    f = np.zeros((nx, ny + 1), bool)
    f[:, :ny] = free
    r = np.zeros((nx, ny + 1), bool)
    r[:, :ny] = reach & free
    f, r = f.ravel(), r.ravel()
    start = f & ~np.concatenate(([False], f[:-1]))
    ids = np.cumsum(start) * f
    hit = np.zeros(int(ids.max()) + 1, bool)
    hit[ids[r]] = True
    hit[0] = False
    return hit[ids].reshape(nx, ny + 1)[:, :ny]


def _reachable(free: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """4-connected component of free that holds seed (= the scipy.ndimage.label component), by alternating row and
    column run sweeps until nothing changes."""
    reach = np.zeros_like(free)
    if not free[seed]:
        return reach
    reach[seed] = True
    while True:
        new = _spread_rows(free, reach)
        new = _spread_rows(free.T, new.T).T
        if (new == reach).all():
            return reach
        reach = new


def _connected_blocked(blocked: np.ndarray, res: float, x0: float, y0: float, pts: list[tuple[float, float]]) -> bool:
    cells = []
    for x, y in pts:
        i, j = int((x - x0) / res), int((y - y0) / res)
        if not (0 <= i < blocked.shape[0] and 0 <= j < blocked.shape[1]) or blocked[i, j]:
            return False
        cells.append((i, j))
    reach = _reachable(~blocked, cells[0])
    return all(bool(reach[c]) for c in cells)


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


# ------------------------------------------------------------------------------------------------ episode wiring
def _check_keys(block: dict, allowed: tuple[str, ...], required: tuple[str, ...], what: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"cfg {what} must be a dict")
    extra = set(block) - set(allowed)
    if extra:
        raise ValueError(f"cfg {what} has keys {sorted(extra)}; only {list(allowed)} are allowed (the draw is keyed by "
                         "(world, day, slot) alone, never by the condition)")
    missing = [k for k in required if k not in block]
    if missing:
        raise ValueError(f"cfg {what} lacks {missing}")


def apply_loc_noise(cfg: dict, world: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    """(world files for the simulator, record). record: applied flags, the key, the seeds, "true" (the simulator's
    start), "belief" (= cfg start, the planner's init), the start error, the clutter items, and the AMCL spread the
    planner must use ("spread": START_DXY / 10 deg if start noise was applied, else 0). Raises on a malformed cfg (fail
    closed). With neither start_noise nor clutter in cfg: the files unchanged, true = belief = cfg start, spread 0."""
    nominal = [float(v) for v in cfg.get("start", [4.0, 32.0, 0.0])]
    if len(nominal) != 3:
        raise ValueError("cfg start must be [x, y, psi]")
    rec: dict = {"start_applied": False, "clutter_applied": False, "belief": nominal, "true": list(nominal),
                 "spread": {"xy_m": 0.0, "yaw_deg": 0.0}}
    sn, cl = cfg.get("start_noise"), cfg.get("clutter")
    if sn is None and cl is None:
        return world, rec
    if sn is None:
        raise ValueError("cfg clutter needs cfg start_noise (its world_seed and day key the placement)")
    _check_keys(sn, START_NOISE_KEYS, START_NOISE_KEYS, "start_noise")
    key = {"world_seed": int(sn["world_seed"]), "day": int(sn["day"]), "slot": sn["slot"]}
    rec["key"] = key
    out = world
    if cl is not None:
        _check_keys(cl, CLUTTER_KEYS, ("n", "keep_clear"), "clutter")
        n = cl["n"]
        if isinstance(n, bool) or int(n) != n or int(n) < 1:
            raise ValueError("cfg clutter n must be an integer >= 1")
        keep = [(float(p[0]), float(p[1])) for p in cl["keep_clear"]]
        keep = [(nominal[0], nominal[1])] + [p for p in keep if p != (nominal[0], nominal[1])]
        avoid = [tuple(float(v) for v in r) for r in (cl.get("avoid_rects") or [])]
        if any(len(r) != 4 for r in avoid):
            raise ValueError("cfg clutter avoid_rects must be [x0, y0, x1, y1] rectangles")
        cs = clutter_seed(key["world_seed"], key["day"])
        out, items = add_unmapped_clutter(world, cs, int(n), keep, avoid)      # type: ignore[arg-type]
        rec["clutter_applied"] = True
        rec["clutter"] = {"seed": cs, "n": len(items), "items": items, "in_map": False,
                          "inputs": {"keep_clear": [list(p) for p in keep], "avoid_rects": [list(r) for r in avoid]}}
    boxes = json.loads(out["boxes.json"].decode()) if "boxes.json" in out else []
    ss = start_seed(key["world_seed"], key["day"], key["slot"])
    st = perturb_start((nominal[0], nominal[1], nominal[2]), ss, boxes)
    rec.update(start_applied=True, true=st["true"], belief=st["belief"], start=st,
               spread={"xy_m": START_DXY, "yaw_deg": math.degrees(START_DPSI)})
    return out, rec
