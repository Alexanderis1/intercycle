"""T-W6: occlusion = MISSING (lane S4; PLAN_TWIN §2, WEAKNESSES T-W6). numpy only.

A machine-motion sample (Observation v2 `zone_motion`, flow / energy on a press ram ROI) is MISSING (NaN) while a
person or vehicle track overlaps the zone (floor) or the ram ROI (image). An occluded machine never reads as "no
motion" (the fail-dangerous direction: a person standing in front of the press = "idle, safe to enter").

Floor space (observer adapter, camera models):
  * an occluder is a detection circle (x, z, r) grown by `pad` (OCC_PAD_M = 0.5 m, the same value as
    twin/camera_v0.OCCLUSION_M; about 1.75 sigma of MEASURED_SHIM's 0.285 m floor noise);
  * a zone is an axis-aligned rect (x0, z0, x1, z1) grown square by `margin` (exactly the area
    `CameraShim.machine_of` counts machine motion in), or a polygon [(x, z), ...] grown by `margin` (Minkowski disc;
    the twin's press footprints / danger polygons);
  * the zone is occluded when any occluder circle touches the grown zone.
Image space (per-clip ingest of the real clips, pixel adapter):
  * a person / vehicle box (u0, v0, u1, v1) grown by `pad_frac` of its width / height on every side (ROI_PAD_FRAC =
    0.15, P8's face-box padding) against the ram ROI polygon in the same framing's pixels.
Tracks: a detector misses an occluder on some frames (MEASURED_SHIM: 32 % drops, 15 % gross errors). A hit is held
for `hold_s` after the last frame that saw it, so a missed detection inside an occlusion never lets a "no motion"
sample through. `hold_ticks(p_miss, eps)` sizes the hold for iid misses; a camera model with miss bursts (stat_v1)
passes a hold sized from its own burst-length distribution.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

OCC_PAD_M = 0.5          # occluder circle growth (m)
ROI_PAD_FRAC = 0.15      # image box growth per side, share of the box size
OCC_EPS = 1e-6           # target per-tick probability that a held occlusion leaks a sample (iid misses)


# ------------------------------------------------------------------------------------------------ track hold

def hold_ticks(p_miss: float, eps: float = OCC_EPS) -> int:
    """Smallest k >= 0 such that an occluder missed on k + 1 consecutive ticks has probability <= eps, for iid
    per-tick misses with probability p_miss. Holding a hit for k ticks then leaks with probability <= eps per tick."""
    if not 0.0 <= p_miss < 1.0:
        raise ValueError(f"p_miss must be in [0, 1), got {p_miss}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if p_miss == 0.0:
        return 0
    return max(0, math.ceil(math.log(eps) / math.log(p_miss) - 1e-12) - 1)


class HoldGate:
    """Per-channel hold: missing(t) = a hit at some tick in [t - hold_s, t]. `step` must see non-decreasing times.
    `value` (optional, per channel) is remembered at the last hit and returned while the hold lasts."""

    def __init__(self, n: int, hold_s: float):
        if hold_s < 0:
            raise ValueError("hold_s must be >= 0")
        self.n, self.hold_s = int(n), float(hold_s)
        self.reset()

    def reset(self) -> None:
        self.last = np.full(self.n, -math.inf)
        self.val = np.zeros(self.n)
        self.t = -math.inf

    def step(self, t: float, hit, value=None) -> tuple[np.ndarray, np.ndarray]:
        """(missing (n,) bool, held value (n,) float; 0 where not missing)."""
        if t < self.t:
            raise ValueError(f"HoldGate times must not go backwards ({t} < {self.t})")
        self.t = float(t)
        hit = np.asarray(hit, bool).reshape(self.n)
        self.last[hit] = t
        if value is not None:
            self.val[hit] = np.asarray(value, float).reshape(self.n)[hit]
        miss = (t - self.last) <= self.hold_s + 1e-9
        return miss, np.where(miss, self.val, 0.0)


def hold_series(t, hit, hold_s: float) -> np.ndarray:
    """Vectorised HoldGate over a whole clip: t (T,) non-decreasing, hit (T,) or (T, n) bool -> same-shape bool,
    True where a hit occurred within [t_i - hold_s, t_i]."""
    t = np.asarray(t, float)
    hit = np.asarray(hit, bool)
    if hit.shape[0] != t.shape[0]:
        raise ValueError(f"hit has {hit.shape[0]} rows, t has {t.shape[0]}")
    if t.size and (np.diff(t) < 0).any():
        raise ValueError("t must be non-decreasing")
    h2 = hit.reshape(len(t), -1)
    last = np.where(h2, t[:, None], -np.inf)
    last = np.maximum.accumulate(last, axis=0) if len(t) else last
    out = (t[:, None] - last) <= hold_s + 1e-9
    return out.reshape(hit.shape)


# ------------------------------------------------------------------------------------------------ floor space

def _as_zone(zone) -> tuple[str, np.ndarray]:
    a = np.asarray(zone, float)
    if a.ndim == 1 and a.shape == (4,):
        if not (a[0] <= a[2] and a[1] <= a[3]):
            raise ValueError(f"rect must be (x0, z0, x1, z1) with x0 <= x1, z0 <= z1, got {zone}")
        return "rect", a
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 3:
        return "poly", a
    raise ValueError(f"a zone is a rect (x0, z0, x1, z1) or a polygon [(x, z), ...] (>= 3 points), got shape {a.shape}")


def _point_in_poly(px: np.ndarray, pz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points."""
    inside = np.zeros(px.shape, bool)
    x0, z0 = poly[:, 0], poly[:, 1]
    x1, z1 = np.roll(x0, -1), np.roll(z0, -1)
    for a, b, c, d in zip(x0, z0, x1, z1):
        if b == d:
            continue
        cross = (b > pz) != (d > pz)
        xc = a + (pz - b) * (c - a) / (d - b)
        inside ^= cross & (px < xc)
    return inside


def _dist_to_poly(px: np.ndarray, pz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the polygon's area (0 inside)."""
    best = np.full(px.shape, np.inf)
    x0, z0 = poly[:, 0], poly[:, 1]
    x1, z1 = np.roll(x0, -1), np.roll(z0, -1)
    for a, b, c, d in zip(x0, z0, x1, z1):
        ex, ez = c - a, d - b
        L2 = ex * ex + ez * ez
        s = np.zeros(px.shape) if L2 == 0 else np.clip(((px - a) * ex + (pz - b) * ez) / L2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (a + s * ex), pz - (b + s * ez)))
    return np.where(_point_in_poly(px, pz, poly), 0.0, best)


def dist_to_zone(px, pz, zone, margin: float = 0.0) -> np.ndarray:
    """Distance from points to the zone grown by `margin` (0 inside). Rects grow square, polygons by a disc."""
    px, pz = np.asarray(px, float), np.asarray(pz, float)
    kind, a = _as_zone(zone)
    if kind == "rect":
        dx = np.maximum.reduce([a[0] - margin - px, np.zeros(px.shape), px - (a[2] + margin)])
        dz = np.maximum.reduce([a[1] - margin - pz, np.zeros(pz.shape), pz - (a[3] + margin)])
        return np.hypot(dx, dz)
    return np.maximum(_dist_to_poly(px, pz, a) - margin, 0.0)


def occluded_zones(x, z, r, zones: Sequence, margin: float = 1.0, pad: float = OCC_PAD_M) -> np.ndarray:
    """(M,) bool: some occluder circle (x, z, r + pad) touches zone m grown by `margin`."""
    x, z, r = (np.asarray(v, float).reshape(-1) for v in (x, z, r))
    out = np.zeros(len(zones), bool)
    if x.size == 0:
        return out
    ok = np.isfinite(x) & np.isfinite(z)
    rad = np.where(np.isfinite(r), np.maximum(r, 0.0), 0.0) + pad
    for m, zone in enumerate(zones):
        out[m] = bool((ok & (dist_to_zone(x, z, zone, margin) <= rad)).any())
    return out


class ZoneCells:
    """Cell centres (`res` grid) covering each zone grown by `margin`, to measure how much of a zone's floor the
    occluders hide (`hidden_share`)."""

    def __init__(self, zones: Sequence, margin: float = 1.0, res: float = 0.25):
        self.zones, self.margin, self.res = list(zones), float(margin), float(res)
        self.cells: list[np.ndarray] = []
        self.bbox = np.zeros((len(self.zones), 4))
        for m, zone in enumerate(self.zones):
            kind, a = _as_zone(zone)
            lo = (a[:2] if kind == "rect" else a.min(0)) - margin
            hi = (a[2:] if kind == "rect" else a.max(0)) + margin
            nx = max(1, int(math.ceil((hi[0] - lo[0]) / res)))
            nz = max(1, int(math.ceil((hi[1] - lo[1]) / res)))
            gx = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
            gz = lo[1] + (np.arange(nz) + 0.5) * (hi[1] - lo[1]) / nz
            px, pz = (v.reshape(-1) for v in np.meshgrid(gx, gz, indexing="ij"))
            keep = dist_to_zone(px, pz, zone, margin) <= 0.0
            self.cells.append(np.c_[px[keep], pz[keep]])
            self.bbox[m] = (lo[0], lo[1], hi[0], hi[1])

    def __len__(self) -> int:
        return len(self.zones)

    def hidden_share(self, x, z, r, pad: float = OCC_PAD_M) -> np.ndarray:
        """(M,) share in [0, 1] of each zone's cells within r + pad of some occluder centre."""
        x, z, r = (np.asarray(v, float).reshape(-1) for v in (x, z, r))
        out = np.zeros(len(self.zones))
        ok = np.isfinite(x) & np.isfinite(z)
        if not ok.any():
            return out
        x, z = x[ok], z[ok]
        rad = np.where(np.isfinite(r[ok]), np.maximum(r[ok], 0.0), 0.0) + pad
        for m, c in enumerate(self.cells):
            if not len(c):
                continue
            x0, z0, x1, z1 = self.bbox[m]
            near = (x + rad >= x0) & (x - rad <= x1) & (z + rad >= z0) & (z - rad <= z1)
            if not near.any():
                continue
            d = np.hypot(c[:, 0:1] - x[near][None, :], c[:, 1:2] - z[near][None, :])
            out[m] = float((d <= rad[near][None, :]).any(1).mean())
        return out


# ------------------------------------------------------------------------------------------------ image space

def _seg_cross(p, q, a, b) -> bool:
    """Closed segments pq and ab intersect (collinear overlap included)."""
    def orient(o, s, t):
        v = (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
        return 0 if abs(v) <= 1e-12 else (1 if v > 0 else -1)

    def on(o, s, t):
        return min(o[0], s[0]) - 1e-12 <= t[0] <= max(o[0], s[0]) + 1e-12 and \
            min(o[1], s[1]) - 1e-12 <= t[1] <= max(o[1], s[1]) + 1e-12

    d1, d2, d3, d4 = orient(a, b, p), orient(a, b, q), orient(p, q, a), orient(p, q, b)
    if d1 != d2 and d3 != d4:
        return True
    return (d1 == 0 and on(a, b, p)) or (d2 == 0 and on(a, b, q)) or (d3 == 0 and on(p, q, a)) or \
        (d4 == 0 and on(p, q, b))


def _as_roi(roi) -> np.ndarray:
    a = np.asarray(roi, float)
    if a.ndim == 1 and a.shape == (4,):
        u0, v0, u1, v1 = a
        if not (u0 <= u1 and v0 <= v1):
            raise ValueError(f"ROI box must be (u0, v0, u1, v1) with u0 <= u1, v0 <= v1, got {roi}")
        return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 3:
        return a
    raise ValueError(f"an ROI is a box (u0, v0, u1, v1) or a polygon [(u, v), ...] (>= 3 points), got shape {a.shape}")


def box_hits_roi(box, roi, pad_frac: float = ROI_PAD_FRAC) -> bool:
    """One box (u0, v0, u1, v1), grown by pad_frac of its size per side, overlaps (touches) the ROI polygon."""
    u0, v0, u1, v1 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (u0, v0, u1, v1)):
        return False
    if u1 < u0 or v1 < v0:
        raise ValueError(f"box must be (u0, v0, u1, v1) with u0 <= u1, v0 <= v1, got {tuple(box)}")
    pu, pv = pad_frac * (u1 - u0), pad_frac * (v1 - v0)
    u0, v0, u1, v1 = u0 - pu, v0 - pv, u1 + pu, v1 + pv
    poly = _as_roi(roi)
    if ((poly[:, 0] >= u0) & (poly[:, 0] <= u1) & (poly[:, 1] >= v0) & (poly[:, 1] <= v1)).any():
        return True                                                    # an ROI vertex inside the box
    corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    if _point_in_poly(corners[:, 0], corners[:, 1], poly).any():
        return True                                                    # a box corner inside the ROI
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        for j in range(len(poly)):
            if _seg_cross(p, q, poly[j], poly[(j + 1) % len(poly)]):
                return True
    return False


def boxes_hit_roi(boxes, roi, pad_frac: float = ROI_PAD_FRAC) -> bool:
    """Any of the (n, 4) person / vehicle boxes overlaps the ROI (see box_hits_roi). Empty -> False."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return any(box_hits_roi(bb, roi, pad_frac) for bb in b)


def roi_missing_series(t, boxes_per_frame: Sequence, roi, hold_s: float, pad_frac: float = ROI_PAD_FRAC) -> np.ndarray:
    """(T,) bool MISSING flag for one press ROI over a clip: frame i is MISSING when a person / vehicle box of frame i
    (boxes_per_frame[i], (n_i, 4) pixels, same framing as the ROI) overlaps the ROI, held for hold_s seconds of
    container-PTS time `t` (T,) after the last overlapping frame. Ingest sets flow / energy NaN and no stroke there."""
    t = np.asarray(t, float)
    if len(boxes_per_frame) != len(t):
        raise ValueError(f"{len(boxes_per_frame)} box lists for {len(t)} frames")
    hit = np.array([boxes_hit_roi(b, roi, pad_frac) for b in boxes_per_frame], bool)
    return hold_series(t, hit, hold_s)
