"""Pure camera geometry for the twin (lane S3b; numpy only, no truth access): the division-model lens, the floor
homography and the pinhole camera rebuilt from (K, k1, H), plus the mover prisms. twin/camera.py (stat_v1) and
non-truth scripts (scripts/stat_v1_roi_occluders.py) share it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

IMG_W, IMG_H = 1920, 1080
HEIGHT_M = {"person": 1.75, "forklift": 2.2, "robot": 0.8}   # ASSUMED (plant PERSON_H, FORKLIFT_BOXES, ANYmal)


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


def circle_prism(x: float, z: float, r: float, hgt: float, n: int = 8) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([x + r * np.sin(a), z + r * np.cos(a)])
    return np.vstack([np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, hgt)])])
