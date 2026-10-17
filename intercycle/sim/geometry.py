"""2D collision geometry kernel: the plant's static obstacles, circle-set bodies for every mover, a lidar.

Conventions (match intercycle/truth/agents.py)
  * plan view: x along the line, z across the hall, metres; yaw = atan2(dx, dz), so forward = (sin yaw, cos yaw)
    and yaw 0 points along +z, yaw pi/2 along +x.
  * rect dicts: w = extent along x, d = extent along z (at yaw 0), (cx, cz) = centre.
  * circle sets are float arrays (k, 3) of (cx, cz, r).  Circles span the real width of each body (robot: r 0.42 for a
    0.8 m wide body, so the envelope is 0.84 x 1.08 m); rectangle corners are rounded off, the rest is slightly oversize.
  * footprint(cls, x, z, yaw) reference points: robot/agv/forklift = body centre, tugger = tractor centre (the three
    carts trail straight behind the heading), weld arm = arm base.  weld_arm_* are dynamic bodies that sit on a static
    robot base, so they are meant for overlaps() against movers, not for static_hit().

Static map
  The plant is rasterised at RES = 0.1 m over x -2..122, z -2..50 (border ring = solid, so the world is closed).
  Twin mode (twin/compile.py): StaticMap(None, rects=[...], extent=(x0, z0, x1, z1)) rasterises the caller's
  axis-aligned rects (press footprints, unseen floor as walls) over that extent instead of the body-shop layout; the
  module constants X0..Z1/NX/NZ stay the body-shop values, so code that needs the extent of a map reads sm.extent.
  A cell is occupied when its centre lies within 2.5 cm of a body (so even a 12 cm light-curtain post has >= 1 cell).
  `dist` is the Euclidean distance (scipy EDT) from a free cell centre to the nearest occupied cell *face*
  (EDT to cell centres minus RES/2), 0 inside.  `clearance()` bilinearly interpolates a signed version of it, so it
  is accurate to a couple of cm along flat faces and never reads less than 0.

Performance (CPython + numpy, no compiled extensions; measured on a busy 8-core laptop): clearance() of 3 points
~15 us, free() ~1 us, footprint() ~7 us, overlaps()/min_gap() ~8 us, lidar(180 rays, 40 circles) ~0.3 ms.
The static part of lidar() is vectorised ray marching (0.05 m steps, +-2.5 cm); lidar(..., exact=True) switches to sphere
tracing on the signed distance field (same answers to ~2 cm, ~3x slower).

Layout decisions that go beyond the YAML (all constants are at the top of the layout section):
  * Walls are inside the hall boundary (0.3 m); the x=96 wall is centred on x=96 and shared with the data hall.
    It has openings: 3 m door at z=24, and 2 m openings at z=8 (leads to the yard, there is no data hall at z<12)
    and z=32 (north aisle into the data hall).
  * Line stations carry two fixed robot bases at (x-1.8, 24-3.3) and (x+1.8, 24+3.3) (the diagonal pair used by
    sim/physics.py as arm bases) with a tip dresser outboard of each; the unload lifter has four frame posts instead.
    Closures cells have a 3.0 x 1.6 m fixture table at the machine centre and bases at (x-1.8, z-2.0), (x+1.8, z+2.0).
  * Light-curtain posts (0.12 m) stand just inside the four corners of each hazard zone (15 cm inset; 40 cm where
    the corner faces the walkway at x=52.5, so the 2 m walkway gap stays >= 1.6 m clear between posts).
  * Parts market: two rows of 5 m bays (1 m gaps) at z=1.2 and z=3.8, 1.2 m deep; the bays are cut back around the
    forklift cross aisles (x=2 is naturally clear; x 51.3..53.7 and 86.8..91.2 are cut) so the routes in
    truth/agents.py (forklifts through x=2, 52.5, 88, 90) never hit a rack.
  * Data hall: 3 rows x 12 racks (0.6 x 1.1 m) from x=101 at z=16/24/32 (7.2 m rows, ends at x=108.2),
    4 CRAH units 1.0 x 2.6 m at x=118.6, z=15/21/27/33.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

__all__ = [
    "RES", "X0", "X1", "Z0", "Z1", "StaticMap", "border_rects", "footprint", "overlaps", "min_gap", "static_hit",
    "static_gap", "lidar", "segment_hit", "CLASSES", "DIMS",
]

RES = 0.1
X0, X1, Z0, Z1 = -2.0, 122.0, -2.0, 50.0           # grid extent (m)
NX = int(round((X1 - X0) / RES))                    # 1240
NZ = int(round((Z1 - Z0) / RES))                    # 520
_INV = 1.0 / RES
_CX0 = X0 + RES / 2                                 # centre of cell (0, 0)
_CZ0 = Z0 + RES / 2
_INFL = 0.025                                       # rasterisation: centre within 2.5 cm of a body

# ----------------------------------------------------------------------------------------------- layout constants
WALL_T = 0.3
DOOR_W = 3.0
AISLE_OPEN_W = 2.0
AISLE_OPEN_Z = (8.0, 32.0)                          # openings in the x=96 wall (transport aisle, north aisle)
LINE_X = (6.0, 86.0)
CONV_W = 2.4
LINE_Z = 24.0
WALK_X = 52.5
WALK_GAP = 2.0
BASE_SIZE = 0.9
DRESSER_SIZE = 0.5
POST = 0.12
POST_INSET = 0.15
POST_INSET_WALK = 0.40
COL = 0.5
COL_PITCH = 12.0
COL_Z = (12.0, 36.0)
MARKET_Z = (1.2, 3.8)
MARKET_DEPTH = 1.2
MARKET_BAY = 5.0
MARKET_GAP = 1.0
MARKET_X = (3.6, 94.5)
MARKET_CROSS = ((51.3, 53.7), (86.8, 91.2))          # forklift cross aisles cut through the racks
RACK_W, RACK_D = 0.6, 1.1
RACK_X0_OFF = 5.0                                    # first rack starts 5 m inside the data hall wall (x=101)
ROW_PITCH = 8.0
CRAH_W, CRAH_D = 1.0, 2.6
CRAH_PITCH = 6.0
KEEP_CLEAR = {                                        # aisle strips columns must never block (x0, z0, x1, z1)
    "AISLE_N": (0.0, 30.0, 96.0, 34.0),
    "AISLE_S": (0.0, 14.0, 96.0, 18.0),
    "TRANSPORT": (0.0, 6.0, 96.0, 10.0),
}


def _rect(id_: str, cls: str, cx: float, cz: float, w: float, d: float, yaw: float = 0.0) -> dict:
    return {"id": id_, "cls": cls, "cx": float(cx), "cz": float(cz), "w": float(w), "d": float(d), "yaw": float(yaw)}


def _box(id_: str, cls: str, x0: float, z0: float, x1: float, z1: float) -> dict:
    return _rect(id_, cls, (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0)


def _overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _wall_x_segments(xc: float, z0: float, z1: float, openings: list[tuple[float, float]], prefix: str) -> list[dict]:
    """Vertical wall (thickness WALL_T, centred on x=xc) from z0..z1 with openings (zc, width)."""
    cuts = sorted((zc - w / 2, zc + w / 2) for zc, w in openings)
    out, z = [], z0
    for k, (a, b) in enumerate(cuts + [(z1, z1)]):
        if a > z + 1e-9:
            out.append(_box(f"{prefix}_{k}", "wall", xc - WALL_T / 2, z, xc + WALL_T / 2, a))
        z = max(z, b)
    return out


def build_rects(scn) -> tuple[list[dict], list[tuple[str, float, float]]]:
    """All static bodies of the plant and the fixed-arm base positions (machine id, x, z)."""
    hx0, hx1 = scn.hall["x"]
    hz0, hz1 = scn.hall["z"]
    dx0, dx1 = scn.data_hall["x"]
    dz0, dz1 = scn.data_hall["z"]
    door_z = scn.data_hall["door"][1]
    R: list[dict] = []
    arms: list[tuple[str, float, float]] = []

    # -- walls
    R.append(_box("WALL_W", "wall", hx0, hz0, hx0 + WALL_T, hz1))
    R.append(_box("WALL_S", "wall", hx0 + WALL_T, hz0, dx0, hz0 + WALL_T))
    R.append(_box("WALL_N", "wall", hx0 + WALL_T, hz1 - WALL_T, dx0, hz1))
    R += _wall_x_segments(dx0, hz0, hz1, [(door_z, DOOR_W)] + [(z, AISLE_OPEN_W) for z in AISLE_OPEN_Z], "WALL_E")
    R.append(_box("DH_WALL_S", "wall", dx0 + WALL_T / 2, dz0, dx1, dz0 + WALL_T))
    R.append(_box("DH_WALL_N", "wall", dx0 + WALL_T / 2, dz1 - WALL_T, dx1, dz1))
    R.append(_box("DH_WALL_E", "wall", dx1 - WALL_T, dz0 + WALL_T, dx1, dz1 - WALL_T))

    # -- roller-bed conveyor (car bodies ride on it) with the walkway gap
    cz0, cz1 = LINE_Z - CONV_W / 2, LINE_Z + CONV_W / 2
    R.append(_box("CONV_0", "conveyor", LINE_X[0], cz0, WALK_X - WALK_GAP / 2, cz1))
    R.append(_box("CONV_1", "conveyor", WALK_X + WALK_GAP / 2, cz0, LINE_X[1], cz1))

    # -- machines
    for m in scn.machines:
        x0, z0, x1, z1 = m.rect
        for k, (px, pz) in enumerate(((x0, z0), (x1, z0), (x0, z1), (x1, z1))):
            inset = POST_INSET_WALK if abs(px - WALK_X) < 1.0 else POST_INSET
            sx = 1 if px == x0 else -1
            sz = 1 if pz == z0 else -1
            R.append(_rect(f"{m.id}_LC{k}", "curtain_post", px + sx * inset, pz + sz * POST_INSET, POST, POST))
        if m.kind == "unload_lifter":
            for k, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
                R.append(_rect(f"{m.id}_FRAME{k}", "lifter_post", m.x + sx * 4.0, m.z + sz * 3.4, 0.4, 0.4))
        elif m.on_line:
            for k, (sx, sz) in enumerate(((-1, -1), (1, 1))):
                bx, bz = m.x + sx * 1.8, LINE_Z + sz * 3.3
                R.append(_rect(f"{m.id}_BASE{k}", "robot_base", bx, bz, BASE_SIZE, BASE_SIZE))
                R.append(_rect(f"{m.id}_TIP{k}", "tip_dresser", bx + sx * 1.15, bz, DRESSER_SIZE, DRESSER_SIZE))
                arms.append((m.id, bx, bz))
        else:
            R.append(_rect(f"{m.id}_TABLE", "fixture_table", m.x, m.z, 3.0, 1.6))
            for k, (sx, sz) in enumerate(((-1, -1), (1, 1))):
                bx, bz = m.x + sx * 1.8, m.z + sz * 2.0
                R.append(_rect(f"{m.id}_BASE{k}", "robot_base", bx, bz, BASE_SIZE, BASE_SIZE))
                arms.append((m.id, bx, bz))

    # -- building columns on a 12 x 12 m grid, skipping any that would block an aisle, the line or a hazard zone
    blocked = list(KEEP_CLEAR.values()) + [(LINE_X[0], cz0, LINE_X[1], cz1)] + [m.rect for m in scn.machines]
    x = hx0 + COL_PITCH
    while x < hx1 - 1.0:
        for z in COL_Z:
            box = (x - COL / 2, z - COL / 2, x + COL / 2, z + COL / 2)
            if not any(_overlap(box, b) for b in blocked):
                R.append(_rect(f"COL_{x:g}_{z:g}", "column", x, z, COL, COL))
        x += COL_PITCH

    # -- parts market racks (z 0..5)
    pitch = MARKET_BAY + MARKET_GAP
    for ri, zc in enumerate(MARKET_Z):
        k, xs = 0, MARKET_X[0]
        while xs < MARKET_X[1]:
            segs = [(xs, min(xs + MARKET_BAY, MARKET_X[1]))]
            for a, b in MARKET_CROSS:                    # cut the forklift cross aisles
                segs = [s for lo, hi in segs for s in ((lo, min(hi, a)), (max(lo, b), hi)) if s[1] - s[0] > 1e-9]
            for j, (lo, hi) in enumerate(segs):
                if hi - lo >= 1.0:
                    R.append(_box(f"MKT{ri}_{k}_{j}", "rack_market", lo, zc - MARKET_DEPTH / 2, hi, zc + MARKET_DEPTH / 2))
            xs += pitch
            k += 1

    # -- data hall racks and CRAH units
    rows = int(scn.data_hall_targets.get("rows", 3))
    per = int(scn.data_hall_targets.get("racks_per_row", 12))
    zmid = (dz0 + dz1) / 2
    for r in range(rows):
        zr = zmid + (r - (rows - 1) / 2) * ROW_PITCH
        for i in range(per):
            R.append(_rect(f"RACK_{r}_{i:02d}", "rack_dh", dx0 + RACK_X0_OFF + RACK_W * (i + 0.5), zr, RACK_W, RACK_D))
    nc = int(scn.data_hall_targets.get("crah_units", 4))
    for k in range(nc):
        R.append(_rect(f"CRAH{k}", "crah", dx1 - 1.4, zmid + (k - (nc - 1) / 2) * CRAH_PITCH, CRAH_W, CRAH_D))
    return R, arms


# ----------------------------------------------------------------------------------------------- static map
def _paint(grid: np.ndarray, r: dict, cx0: float = _CX0, cz0: float = _CZ0) -> None:
    """Mark the cells of `grid` whose centre lies within _INFL of rect r; (cx0, cz0) = centre of cell (0, 0)."""
    nx, nz = grid.shape
    cx, cz, hw, hd, yaw = r["cx"], r["cz"], r["w"] / 2 + _INFL, r["d"] / 2 + _INFL, r["yaw"]
    if yaw == 0.0:
        ex, ez = hw, hd
    else:
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ez = hw * c + hd * s, hw * s + hd * c
    i0 = max(int(math.ceil((cx - ex - cx0) * _INV - 1e-9)), 0)
    i1 = min(int(math.floor((cx + ex - cx0) * _INV + 1e-9)), nx - 1)
    j0 = max(int(math.ceil((cz - ez - cz0) * _INV - 1e-9)), 0)
    j1 = min(int(math.floor((cz + ez - cz0) * _INV + 1e-9)), nz - 1)
    if i1 < i0 or j1 < j0:
        return
    if yaw == 0.0:
        grid[i0:i1 + 1, j0:j1 + 1] = True
        return
    xs = cx0 + np.arange(i0, i1 + 1) * RES - cx
    zs = cz0 + np.arange(j0, j1 + 1) * RES - cz
    c, s = math.cos(yaw), math.sin(yaw)
    u = xs[:, None] * c - zs[None, :] * s
    v = xs[:, None] * s + zs[None, :] * c
    grid[i0:i1 + 1, j0:j1 + 1] |= (np.abs(u) <= hw) & (np.abs(v) <= hd)


def _check_extent(extent) -> tuple[float, float, float, float]:
    try:
        x0, z0, x1, z1 = (float(v) for v in extent)
    except (TypeError, ValueError):
        raise ValueError(f"extent must be (x0, z0, x1, z1) in metres, got {extent!r}") from None
    for lo, hi, ax in ((x0, x1, "x"), (z0, z1, "z")):
        n = (hi - lo) / RES
        if not (math.isfinite(lo) and math.isfinite(hi)) or n < 8 or abs(n - round(n)) > 1e-6:
            raise ValueError(f"extent {ax} {lo}..{hi} must be finite and a whole number (>= 8) of {RES} m cells")
    return x0, z0, x1, z1


def _check_rect(r: dict, ext: tuple[float, float, float, float]) -> dict:
    """A caller's rect as a normalised rect dict (yaw defaults to 0); refuses malformed or entirely-outside rects."""
    try:
        out = _rect(str(r["id"]), str(r["cls"]), r["cx"], r["cz"], r["w"], r["d"], r.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"rect needs id, cls, cx, cz, w, d (and optional yaw): {r!r} ({e})") from None
    vals = (out["cx"], out["cz"], out["w"], out["d"], out["yaw"])
    if not all(math.isfinite(v) for v in vals) or out["w"] <= 0.0 or out["d"] <= 0.0:
        raise ValueError(f"rect {out['id']!r} must have finite values and w, d > 0: {r!r}")
    c, s = abs(math.cos(out["yaw"])), abs(math.sin(out["yaw"]))
    ex, ez = out["w"] / 2 * c + out["d"] / 2 * s, out["w"] / 2 * s + out["d"] / 2 * c
    if not _overlap((out["cx"] - ex, out["cz"] - ez, out["cx"] + ex, out["cz"] + ez), ext):
        raise ValueError(f"rect {out['id']!r} lies entirely outside the extent {ext}")
    return out


def border_rects(extent: tuple[float, float, float, float]) -> list[dict]:
    """The 2-cell wall ring just inside the extent, as wall rects (ids BORDER_W/E/S/N): the raster's closing ring."""
    x0, z0, x1, z1 = extent
    t = 2 * RES
    return [_box("BORDER_W", "wall", x0, z0, x0 + t, z1), _box("BORDER_E", "wall", x1 - t, z0, x1, z1),
            _box("BORDER_S", "wall", x0 + t, z0, x1 - t, z0 + t), _box("BORDER_N", "wall", x0 + t, z1 - t, x1 - t, z1)]


class StaticMap:
    """Rasterised static obstacles of the plant.

    StaticMap(scn): the body-shop layout from build_rects(scn) over the module extent (X0, Z0, X1, Z1).
    StaticMap(scn, rects=[...], extent=(x0, z0, x1, z1)): the caller's static bodies (twin/compile.py: press footprints
    and the unseen floor as wall rects) instead of build_rects; scn is not read and may be None, arm_bases is [].
    The 2-cell ring just inside the extent is always solid; in rects mode it is also returned by rects() as the four
    BORDER_* wall rects, so the USD built from rects() is closed exactly like the map (unseen floor is a wall in both).
    extent (default: the module extent) must span a whole number of RES cells; the raster and every query follow it.
    """

    def __init__(self, scn, rects: list[dict] | None = None, extent=None):
        x0, z0, x1, z1 = (X0, Z0, X1, Z1) if extent is None else _check_extent(extent)
        nx, nz = int(round((x1 - x0) / RES)), int(round((z1 - z0) / RES))
        self.extent = (x0, z0, x1, z1)
        self.x0, self.z0, self.x1, self.z1, self.nx, self.nz = x0, z0, x1, z1, nx, nz
        self._cx0, self._cz0 = x0 + RES / 2, z0 + RES / 2
        self._c2 = np.array([[self._cx0], [self._cz0]])
        self._hi = np.array([[nx - 1.0001], [nz - 1.0001]])
        self._nzp = nz + 2 * _PAD
        if rects is None:
            if scn is None:
                raise ValueError("StaticMap needs a scenario (body-shop layout) or rects")
            self._rects, self.arm_bases = build_rects(scn)
        else:
            own = [_check_rect(r, self.extent) for r in rects]
            have = {r["id"]: r for r in own}
            for b in border_rects(self.extent):              # re-passing rects() is allowed: the ring is not doubled
                if b["id"] not in have:
                    own.append(b)
                elif have[b["id"]] != b:
                    raise ValueError(f"rect id {b['id']!r} is reserved for the extent's border ring")
            self._rects, self.arm_bases = own, []
        grid = np.zeros((nx, nz), dtype=bool)
        for r in self._rects:
            _paint(grid, r, self._cx0, self._cz0)
        grid[:2, :] = grid[-2:, :] = grid[:, :2] = grid[:, -2:] = True   # the world is closed (2-cell border ring)
        self.grid = grid
        d_free = ndimage.distance_transform_edt(~grid) * RES
        d_occ = ndimage.distance_transform_edt(grid) * RES
        sdf = np.where(grid, -(d_occ - RES / 2), d_free - RES / 2)       # signed: < 0 inside, faces at 0
        self.dist: np.ndarray = np.maximum(sdf, 0.0).astype(np.float32)   # [nx, nz]; 0 inside obstacles
        self._sdf = sdf
        # bilinear coefficients per cell: value = c0 + u*c1 + v*c2 + u*v*c3 (u, v = fractions toward +x, +z cell centres)
        a_, b_, c_, d_ = sdf[:-1, :-1], sdf[1:, :-1], sdf[:-1, 1:], sdf[1:, 1:]
        cf = np.zeros((nx, nz, 4))
        cf[:-1, :-1, 0], cf[:-1, :-1, 1], cf[:-1, :-1, 2], cf[:-1, :-1, 3] = a_, b_ - a_, c_ - a_, a_ - b_ - c_ + d_
        self._cf = cf.reshape(nx * nz, 4)
        occ = np.ones((nx + 2 * _PAD, nz + 2 * _PAD), dtype=np.uint8)      # padded so marching never indexes outside
        occ[_PAD:_PAD + nx, _PAD:_PAD + nz] = grid
        self._occ = occ.ravel()

    # ---- queries
    def rects(self) -> list[dict]:
        """Every static body as {"id","cls","cx","cz","w","d","yaw"}; w along x, d along z at yaw 0."""
        return [dict(r) for r in self._rects]

    def cell(self, x: float, z: float) -> tuple[int, int]:
        return int(math.floor((x - self.x0) * _INV)), int(math.floor((z - self.z0) * _INV))

    def _bilinear(self, P: np.ndarray) -> np.ndarray:
        """Signed distance at the points P (2, N) = (x row, z row), by bilinear interpolation of cell-centre values."""
        f = (P - self._c2) * _INV
        np.maximum(f, 0.0, out=f)
        np.minimum(f, self._hi, out=f)
        ij = f.astype(np.intp)
        f -= ij
        g = self._cf.take(ij[0] * self.nz + ij[1], axis=0)
        u, v = f[0], f[1]
        return g[:, 0] + u * (g[:, 1] + v * g[:, 3]) + v * g[:, 2]

    def _sdf1(self, x: float, z: float) -> float:
        """Signed distance at one point in pure Python (faster than numpy below ~10 points); -1 outside the world."""
        fx = (x - self._cx0) * _INV
        fz = (z - self._cz0) * _INV
        if not (0.0 <= fx <= self.nx - 1.0001 and 0.0 <= fz <= self.nz - 1.0001):
            return -1.0
        i = int(fx)
        j = int(fz)
        c0, c1, c2, c3 = self._cf[i * self.nz + j].tolist()
        u = fx - i
        v = fz - j
        return c0 + u * (c1 + v * c3) + v * c2

    def clearance(self, pts: np.ndarray) -> np.ndarray:
        """Distance (m) from each (x, z) point to the nearest static obstacle; 0 if inside (or outside the world)."""
        p = np.asarray(pts, dtype=np.float64)
        if p.ndim == 1:
            p = p[None, :]
        if p.shape[0] <= 10:
            f = self._sdf1
            return np.array([max(f(x, z), 0.0) for x, z in p[:, :2].tolist()])
        s = self._bilinear(p[:, :2].T)
        return np.maximum(s, 0.0, out=s)

    def free(self, x: float, z: float, r: float) -> bool:
        """True if a disc of radius r at (x, z) touches no static obstacle (scalar fast path)."""
        return self._sdf1(x, z) >= r

    # ---- lidar, static part
    def trace(self, x: float, z: float, ux: np.ndarray, uz: np.ndarray, rmax: float) -> np.ndarray:
        """Range along each unit ray (ux, uz) to the first static obstacle, rmax if none within rmax.

        Fixed-step marching (0.05 m) over the occupancy raster, fully vectorised in one pass (~4x faster than
        sphere tracing in numpy because there is no sequential iteration): the range is the middle of the step
        that first enters an occupied cell, so +-2.5 cm.  A ray that clips only the corner of a 12 cm post can slip
        between two samples; neighbouring rays (1 deg apart) still see it.  `trace_sphere` is the exact reference.
        """
        if not (self.x0 + RES < x < self.x1 - RES and self.z0 + RES < z < self.z1 - RES):
            return np.zeros(ux.shape[0])
        steps = _march_steps(min(rmax, _PAD * RES))
        bx = np.float32((x - self._cx0) * _INV + 0.5 + _PAD)
        bz = np.float32((z - self._cz0) * _INV + 0.5 + _PAD)
        fx = np.multiply.outer((ux * _INV).astype(np.float32), steps)
        fx += bx
        fz = np.multiply.outer((uz * _INV).astype(np.float32), steps)
        fz += bz
        k = fx.astype(np.intp)
        k *= self._nzp
        k += fz.astype(np.intp)
        hit = self._occ[k]
        first = hit.argmax(axis=1)
        got = hit[np.arange(ux.shape[0]), first] != 0
        return np.where(got, (first + 0.5) * _MARCH_H, rmax)

    def trace_sphere(self, x: float, z: float, ux: np.ndarray, uz: np.ndarray, rmax: float) -> np.ndarray:
        """Exact (to ~2 cm) reference: sphere tracing on the signed distance field, vectorised over the active rays."""
        n = ux.shape[0]
        res = np.full(n, rmax)
        if not self.free(x, z, 0.0):
            res[:] = 0.0
            return res
        ids = np.arange(n)
        t = np.zeros(n)
        bil = self._bilinear
        for _ in range(_MAX_ITER):
            s = bil(np.stack((x + ux * t, z + uz * t)))
            done = (s < _HIT_EPS) | (t >= rmax)
            if done.any():
                hit = done & (t < rmax)
                res[ids[hit]] = t[hit]
                keep = ~done
                if not keep.any():
                    return res
                ids, ux, uz, t, s = ids[keep], ux[keep], uz[keep], t[keep], s[keep]
            t = np.minimum(t + np.maximum(s * _SHRINK, _MIN_STEP), rmax)
        res[ids] = np.minimum(t, rmax)                  # (pathological) rays that never converged
        return res


_PAD = 250                                          # 25 m of solid padding around the raster (max lidar range)
_MARCH_H = 0.05
_MARCH: dict[float, np.ndarray] = {}
_HIT_EPS = 0.02
_SHRINK = 0.95
_MIN_STEP = 0.03
_MAX_ITER = 400


def _march_steps(rmax: float) -> np.ndarray:
    m = _MARCH.get(rmax)
    if m is None:
        K = max(int(math.ceil(rmax / _MARCH_H)), 1)
        m = _MARCH[rmax] = ((np.arange(K) + 1) * _MARCH_H).astype(np.float32)
    return m


# ----------------------------------------------------------------------------------------------- bodies
# (forward offset from the body reference point, lateral offset, radius).  Reference point:
# robot/agv/forklift = body centre, tugger = tractor centre (carts trail behind), weld arm = arm base.
_ROBOT = [(f, 0.0, 0.42) for f in (-0.12, 0.0, 0.12)]                       # 0.8 x 1.05 m (0.84 x 1.08 m envelope)
_AGV = [(-0.3, 0.0, 0.5), (0.3, 0.0, 0.5)]                                  # 1.0 x 1.6 m
_TRACTOR = [(-0.2, 0.0, 0.5), (0.2, 0.0, 0.5)]                              # 1.0 x 1.4 m
_CART_GAP = 0.2
_CARTS = [(-(0.7 + _CART_GAP + 0.8 + 1.8 * k) + dz, 0.0, 0.5) for k in range(3) for dz in (-0.3, 0.3)]   # 3 x (1.0 x 1.6 m)
_FORKLIFT = [(-0.4, 0.0, 0.55), (0.4, 0.0, 0.55), (1.2, 0.0, 0.45), (1.6, 0.0, 0.45)]   # 1.1 x 1.9 m + forks to +2.05 m

_BODIES: dict[str, np.ndarray] = {
    "robot": np.array(_ROBOT),
    "person": np.array([(0.0, 0.0, 0.3)]),
    "agv": np.array(_AGV),
    "tugger": np.array(_TRACTOR + _CARTS),
    "forklift": np.array(_FORKLIFT),
    "weld_arm_active": np.array([(0.0, 0.0, 2.2)]),
    "weld_arm_idle": np.array([(0.0, 0.0, 0.7)]),
}
_ALIASES = {"tugger_train": "tugger", "train": "tugger", "amr": "robot", "weld_arm": "weld_arm_idle"}
CLASSES = tuple(_BODIES)

# rectangles for renderers: cls -> list of (forward centre offset, length, width); weld arms are discs (length=width=2r)
DIMS: dict[str, list[tuple[float, float, float]]] = {
    "robot": [(0.0, 1.05, 0.8)],
    "person": [(0.0, 0.6, 0.6)],
    "agv": [(0.0, 1.6, 1.0)],
    "tugger": [(0.0, 1.4, 1.0)] + [(-(0.7 + _CART_GAP + 0.8 + 1.8 * k), 1.6, 1.0) for k in range(3)],
    "forklift": [(0.0, 1.9, 1.1), (1.5, 1.1, 0.9)],
    "weld_arm_active": [(0.0, 4.4, 4.4)],
    "weld_arm_idle": [(0.0, 1.4, 1.4)],
}
_FWD = {k: v[:, 0].copy() for k, v in _BODIES.items()}
_LAT = {k: v[:, 1].copy() for k, v in _BODIES.items()}
_RAD = {k: v[:, 2].copy() for k, v in _BODIES.items()}


def footprint(cls: str, x: float, z: float, yaw: float) -> np.ndarray:
    """(k, 3) circles (cx, cz, r) of a body of class `cls` at (x, z) heading `yaw` (forward = (sin yaw, cos yaw))."""
    cls = _ALIASES.get(cls, cls)
    try:
        f, l, r = _FWD[cls], _LAT[cls], _RAD[cls]
    except KeyError:
        raise KeyError(f"unknown body class {cls!r}; known: {CLASSES}") from None
    s, c = math.sin(yaw), math.cos(yaw)
    out = np.empty((f.shape[0], 3))
    out[:, 0] = x + s * f + c * l
    out[:, 1] = z + c * f - s * l
    out[:, 2] = r
    return out


def min_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Signed gap (m) between two circle sets: min over pairs of (centre distance - ra - rb); < 0 = penetration."""
    if len(a) == 0 or len(b) == 0:
        return math.inf
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) - a[:, None, 2] - b[None, :, 2]
    return float(d.min())


def overlaps(a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
    """True if any circle of `a` is closer than `margin` to any circle of `b` (margin 0 = touching/penetrating)."""
    return min_gap(a, b) < margin


def static_gap(sm: StaticMap, circles: np.ndarray) -> float:
    """Signed gap (m) between a circle set and the static map; negative = penetrating."""
    if len(circles) == 0:
        return math.inf
    return float((sm.clearance(circles[:, :2]) - circles[:, 2]).min())


def static_hit(sm: StaticMap, circles: np.ndarray) -> bool:
    """True if any circle penetrates static geometry (a circle centred inside an obstacle always hits)."""
    if len(circles) == 0:
        return False
    return bool((sm.clearance(circles[:, :2]) < circles[:, 2]).any())


def segment_hit(sm: StaticMap, a: tuple[float, float], b: tuple[float, float], cls: str = "robot",
                step: float = 0.25) -> tuple[float, float] | None:
    """Sweep a body of class `cls`, aligned with the segment a->b, along it every `step` m (vectorised).

    Returns the body-reference (x, z) of the first sample that penetrates static geometry, or None if the sweep is clear.
    """
    cls = _ALIASES.get(cls, cls)
    ax, az = a
    dx, dz = b[0] - ax, b[1] - az
    L = math.hypot(dx, dz)
    m = max(2, int(math.ceil(L / step)) + 1)
    f = np.linspace(0.0, 1.0, m)
    xs, zs = ax + dx * f, az + dz * f
    yaw = math.atan2(dx, dz) if L > 1e-9 else 0.0
    sy, cy = math.sin(yaw), math.cos(yaw)
    fw, lt, rd = _FWD[cls], _LAT[cls], _RAD[cls]
    cx = xs[:, None] + sy * fw[None, :] + cy * lt[None, :]
    cz = zs[:, None] + cy * fw[None, :] - sy * lt[None, :]
    bad = (sm.clearance(np.stack((cx.ravel(), cz.ravel()), axis=1)).reshape(m, -1) < rd[None, :]).any(axis=1)
    if not bad.any():
        return None
    i = int(bad.argmax())
    return float(xs[i]), float(zs[i])


# ----------------------------------------------------------------------------------------------- lidar
_RAY_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ray_table(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = _RAY_TABLES.get(n)
    if t is None:
        a = np.arange(n) * (2 * math.pi / n)
        t = _RAY_TABLES[n] = (np.cos(a), np.sin(a))
    return t


def lidar(sm: StaticMap, x: float, z: float, yaw: float, dyn: np.ndarray, n: int = 180, rmax: float = 12.0,
          exact: bool = False) -> np.ndarray:
    """360 degree range scan, ray i at heading yaw + i*2pi/n (ray 0 = straight ahead). Returns (n,) ranges in m, rmax = no return.

    Static geometry by vectorised ray marching on the raster (exact=True: sphere tracing on the distance field, ~4x
    slower); dynamic bodies by analytic ray-circle intersection with
    `dyn` (M, 3) circles (the sensor's own body must not be in `dyn`; a ray starting inside a circle returns 0).
    Occlusion is exact: per ray the nearest of static and dynamic returns wins. No ids leak out, only ranges.
    """
    ca, sa = _ray_table(n)
    s, c = math.sin(yaw), math.cos(yaw)
    ux = s * ca + c * sa                                # sin(yaw + a)
    uz = c * ca - s * sa                                # cos(yaw + a)
    r = sm.trace_sphere(x, z, ux, uz, rmax) if exact else sm.trace(x, z, ux, uz, rmax)
    if dyn is not None and len(dyn):
        dyn = np.asarray(dyn, dtype=np.float64)
        cx = dyn[:, 0] - x
        cz = dyn[:, 1] - z
        rr = dyn[:, 2]
        d2 = cx * cx + cz * cz
        near = d2 < (rmax + rr) ** 2
        if not near.all():
            cx, cz, rr, d2 = cx[near], cz[near], rr[near], d2[near]
        if cx.size:
            b = ux[:, None] * cx[None, :] + uz[:, None] * cz[None, :]          # projection of centre on ray
            disc = b * b - (d2 - rr * rr)[None, :]                             # r^2 - perpendicular distance^2
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = np.where((disc >= 0.0) & (b + sq > 0.0), np.maximum(b - sq, 0.0), np.inf)
            r = np.minimum(r, t.min(axis=1))
    return r
