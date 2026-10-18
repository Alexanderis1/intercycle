"""Pure camera geometry for the twin (lane S3b; numpy only, no truth access): the division-model lens, the floor
homography and the pinhole camera rebuilt from (K, k1, H), plus the mover prisms. twin/camera.py (stat_v1) and
non-truth scripts (scripts/stat_v1_roi_occluders.py) share it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

IMG_W, IMG_H = 1920, 1080
HEIGHT_M = {"person": 1.75, "forklift": 2.2, "robot": 0.8}   # ASSUMED (plant PERSON_H, FORKLIFT_BOXES, ANYmal)
PRESS_H_M = 3.0                                              # ASSUMED press body height (occluder prism, ROI anchor)


def _undistort(pts: np.ndarray, k1: float, c: np.ndarray) -> np.ndarray:
    """Division model, distorted -> undistorted (as intercycle.framing.undistort; numpy only for the Isaac side)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    d = p - c
    return c + d / (1.0 + k1 * (d ** 2).sum(1, keepdims=True))


def _distort(pts: np.ndarray, k1: float, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted -> distorted (exact inverse of `_undistort` on its valid branch, as intercycle.framing.distort);
    returns (points, ok) with ok False outside the model's valid range."""
    p = np.asarray(pts, float).reshape(-1, 2)
    d = p - c
    r_u = np.sqrt((d ** 2).sum(1))
    out = p.copy()
    ok = np.ones(len(p), bool)
    if abs(k1) < 1e-18:
        return out, ok
    disc = 1.0 - 4.0 * k1 * r_u ** 2
    ok = disc >= 0
    nz = ok & (r_u > 1e-12)
    r_d = 2.0 * r_u[nz] / (1.0 + np.sqrt(disc[nz]))
    out[nz] = c + d[nz] * (r_d / r_u[nz])[:, None]
    out[~ok] = np.nan
    return out, ok


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, float).T
    out = np.full((len(p), 2), np.nan)
    ok = np.abs(q[:, 2]) > 1e-12
    out[ok] = q[ok, :2] / q[ok, 2:3]
    return out


@dataclass
class CamGeom:
    """One framing's pinhole camera, rebuilt from (K, k1, H): H maps undistorted px -> plan floor (x, z).
    K^-1 H^-1 = lambda [r1 r2 t] (the floor plane's two axes and the camera translation); the third axis is the
    floor normal, signed so that the camera centre is above the floor. Plan points are (x, z, height)."""
    fid: str
    K: np.ndarray
    k1: float
    H: np.ndarray
    M: np.ndarray = field(init=False)          # 3x3: camera coords = M @ (x, z, height) + t
    t: np.ndarray = field(init=False)
    nadir: np.ndarray = field(init=False)
    height: float = field(init=False)
    c: np.ndarray = field(init=False)

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        self.H = np.asarray(self.H, float) / self.H[2, 2]
        self.c = self.K[:2, 2].copy()
        A = np.linalg.inv(self.K) @ np.linalg.inv(self.H)
        lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
        if (lam * A[:, 2])[2] < 0:                 # the floor origin must lie in front of the camera
            lam = -lam
        r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
        r3 = np.cross(r1, r2)
        r3 /= np.linalg.norm(r3)
        M = np.column_stack([r1, r2, r3])
        centre = -np.linalg.solve(M, t)
        if centre[2] < 0:                          # the plan frame (x, z, up) is left-handed: flip the normal
            M = np.column_stack([r1, r2, -r3])
            centre = -np.linalg.solve(M, t)
        self.M, self.t = M, t
        self.nadir, self.height = centre[:2].copy(), float(centre[2])

    @classmethod
    def from_rec(cls, fid: str, K, r: dict) -> "CamGeom":
        return cls(fid, np.asarray(K, float), float(r["k1"]), np.asarray(r["H"], float))

    def project(self, xzy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) plan points (x, z, height) -> (raw px (n, 2), ok (n,)): ok = in front of the camera and inside the
        division model's valid range."""
        P = np.asarray(xzy, float).reshape(-1, 3)
        cam = P @ self.M.T + self.t
        ok = cam[:, 2] > 1e-6
        und = np.full((len(P), 2), np.nan)
        if ok.any():
            q = (self.K @ (cam[ok] / cam[ok, 2:3]).T).T
            und[ok] = q[:, :2]
        raw = np.full_like(und, np.nan)
        if ok.any():
            r, okd = _distort(und[ok], self.k1, self.c)
            raw[ok] = r
            ok[np.flatnonzero(ok)[~okd]] = False
        return raw, ok

    def floor_of(self, raw: np.ndarray) -> np.ndarray:
        """Raw px -> plan floor (x, z) through this camera's lens and H."""
        return _apply_h(self.H, _undistort(raw, self.k1, self.c))

    def rays(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw px (n, 2) -> (camera centre (3,), ray directions (n, 3)) in plan coords (x, z, height): the points
        centre + s * dir (s > 0) project to those pixels (inverse of `project`)."""
        und = _undistort(raw, self.k1, self.c)
        d_cam = np.linalg.solve(self.K, np.column_stack([und, np.ones(len(und))]).T).T
        Minv = np.linalg.inv(self.M)
        return -Minv @ self.t, d_cam @ Minv.T


def _pip(x: float, y: float, poly: np.ndarray) -> bool:
    """Point in polygon (even-odd ray casting)."""
    inside = False
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def ray_prism_entry(C: np.ndarray, d: np.ndarray, footprint, hgt: float) -> float | None:
    """Smallest s >= 0 at which the ray C + s d enters the vertical prism footprint x [0, hgt] (plan x, z, height),
    or None if it never does. Exact: the candidates are the crossings of the footprint's edges and of the two height
    planes; each interval between candidates is tested at its midpoint."""
    P = np.asarray(footprint, float).reshape(-1, 2)
    cand = [0.0]
    for i in range(len(P)):
        a, e = P[i], P[(i + 1) % len(P)] - P[i]
        A = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(float(np.linalg.det(A))) < 1e-12:
            continue
        s, u = np.linalg.solve(A, a - C[:2])
        if s > 0 and -1e-9 <= u <= 1.0 + 1e-9:
            cand.append(float(s))
    if abs(float(d[2])) > 1e-12:
        cand += [float(s) for s in ((0.0 - C[2]) / d[2], (hgt - C[2]) / d[2]) if s > 0]
    cand = sorted(set(cand))
    for s0, s1 in zip(cand, cand[1:]):
        X = C + 0.5 * (s0 + s1) * d
        if 0.0 <= X[2] <= hgt and _pip(float(X[0]), float(X[1]), P):
            return s0
    return None


def press_anchor(g_nom: CamGeom, poly_px, footprint, press_h: float) -> np.ndarray:
    """3-D anchor (n, 3) of an image ROI drawn on a press (lane S3b fix2). The ROI shows what the camera sees: the
    near surface of the press body. The NOMINAL camera's ray through the ROI centre enters the press prism
    (footprint x [0, press_h]) at X*; each ROI vertex ray meets the vertical plane through X* that faces the camera
    (normal = the horizontal direction camera -> X*). Projected through the nominal camera the anchor gives the ROI
    back; projected through a world's TRUE camera it gives the ROI where that world's image shows the press (the ROI
    moves with the press prism and the movers, all projected through the same true camera). The plane (the ram's
    depth = the visible surface at the ROI centre) is ASSUMED. Fails closed (ValueError) when the centre ray misses
    the prism or a vertex ray does not meet the plane in front of the camera."""
    poly = np.asarray(poly_px, float).reshape(-1, 2)
    C, D = g_nom.rays(np.vstack([poly.mean(0, keepdims=True), poly]))
    s_c = ray_prism_entry(C, D[0], footprint, press_h)
    if s_c is None:
        raise ValueError(f"{g_nom.fid}: the ROI centre ray misses the press prism (height {press_h} m)")
    Xc = C + s_c * D[0]
    h = Xc[:2] - C[:2]
    if float(np.hypot(*h)) < 1e-6:
        raise ValueError(f"{g_nom.fid}: the ROI centre is under the camera, no anchor plane")
    n = np.array([h[0], h[1], 0.0]) / float(np.hypot(*h))
    den = D[1:] @ n
    s = (Xc - C) @ n / np.where(np.abs(den) > 1e-12, den, np.nan)
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError(f"{g_nom.fid}: an ROI vertex ray does not meet the press plane in front of the camera")
    return C + s[:, None] * D[1:]


def roi_box(g: CamGeom, anchor: np.ndarray) -> tuple[float, float, float, float] | None:
    """Image bbox (u0, v0, u1, v1) of an ROI anchor through camera `g`, or None if a point is not projectable."""
    raw, ok = g.project(anchor)
    if not ok.all():
        return None
    return float(raw[:, 0].min()), float(raw[:, 1].min()), float(raw[:, 0].max()), float(raw[:, 1].max())


def box_share_in_image(box) -> float:
    """Share of an image bbox inside the IMG_W x IMG_H image (0 for None or an empty box)."""
    if box is None:
        return 0.0
    r0, s0, r1, s1 = box
    area = max(r1 - r0, 0.0) * max(s1 - s0, 0.0)
    inside = max(min(r1, IMG_W) - max(r0, 0.0), 0.0) * max(min(s1, IMG_H) - max(s0, 0.0), 0.0)
    return inside / area if area > 0 else 0.0


def circle_prism(x: float, z: float, r: float, hgt: float, n: int = 8) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([x + r * np.sin(a), z + r * np.cos(a)])
    return np.vstack([np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, hgt)])])


def bbox(poly, grow: float = 0.0) -> tuple[float, float, float, float]:
    """Axis-aligned box (x0, z0, x1, z1) of a floor polygon grown by `grow` m (no truth: site geometry only). Shared by
    twin/plant.py and the non-truth intercycle/observer/pixel_adapter.py."""
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) - grow, min(zs) - grow, max(xs) + grow, max(zs) + grow
