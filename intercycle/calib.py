"""CAL v1 (lane S2, PLAN_TWIN §2/§4): image -> floor homography per framing from agent-placed floor points.

Conventions (as ``intercycle.framing``): pixels in the native 1920x1080 frame; lens = one division-model k1 per
framing, distortion centre at the image centre; K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] with f = 1330 px (P6 prior).

The placed points carry NO typed lengths (none can be justified from the pictures), only floor relations: two
segments are parallel or perpendicular on the floor. So H is built from a ground-plane camera model:

* camera at height ``h`` above the floor, tilt ``theta`` down from horizontal, roll ``phi`` about the optical axis
  (yaw only rotates the floor frame and is fixed by the zone's axis choice);
* ``(theta, phi)`` are fitted by least squares so that the stated relations hold (residuals in degrees on the floor);
* ``h`` is the 4 m mount-height prior (P6), so ground coordinates scale linearly with ``h``;
* the zone frame puts the origin at one placed point and the +z axis along one floor direction.

Camera->world rotation ``M = R_tilt(theta) @ R_roll(phi)``; world axes X right, Y forward along the floor, Z up;
camera axes x right, y down, z forward. A ray ``M K^-1 (u, v, 1)`` meets the floor at
``(X, Y) = h (r_x, r_y) / (-r_z)``, so ``H_cam = diag(1, 1, -1/h) M K^-1`` (undistorted pixels -> floor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from intercycle import framing as fr

F_PRIOR = fr.F_PRIOR
CENTER = fr.CENTER
WALKER_H_M = fr.WALKER_H_M


def K_matrix(f: float = F_PRIOR, center: Sequence[float] = CENTER) -> np.ndarray:
    return np.array([[f, 0.0, center[0]], [0.0, f, center[1]], [0.0, 0.0, 1.0]])


def rotation(theta: float, phi: float) -> np.ndarray:
    """Camera -> world rotation for tilt ``theta`` (rad, down from horizontal) and roll ``phi`` (rad)."""
    c, s = math.cos(theta), math.sin(theta)
    r_tilt = np.array([[1.0, 0.0, 0.0], [0.0, -s, c], [0.0, -c, -s]])
    cr, sr = math.cos(phi), math.sin(phi)
    r_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return r_tilt @ r_roll


def h_cam(theta: float, phi: float, h: float, f: float = F_PRIOR, center: Sequence[float] = CENTER) -> np.ndarray:
    """Homography undistorted pixel -> camera-aligned floor (X, Y) in metres."""
    return np.diag([1.0, 1.0, -1.0 / h]) @ rotation(theta, phi) @ np.linalg.inv(K_matrix(f, center))


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (n, 2) points (plain division by w; NaN only where w ~ 0).

    The sign of w is NOT checked: a homography normalised to H[2,2] = 1 may carry either sign. Whether a pixel lies
    below the horizon is ``ground_ok``'s job.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, float).T
    out = np.full((len(p), 2), np.nan)
    ok = np.abs(q[:, 2]) > 1e-12
    out[ok] = q[ok, :2] / q[ok, 2:3]
    return out


def ground_ok(theta: float, phi: float, uv: np.ndarray, f: float = F_PRIOR,
              center: Sequence[float] = CENTER) -> np.ndarray:
    """True where the undistorted pixel's ray points below the horizon (meets the floor in front of the camera)."""
    p = np.asarray(uv, float).reshape(-1, 2)
    rays = (rotation(theta, phi) @ np.linalg.inv(K_matrix(f, center)) @ np.column_stack([p, np.ones(len(p))]).T).T
    return rays[:, 2] < -1e-9


def project_world(P: np.ndarray, theta: float, phi: float, h: float, f: float = F_PRIOR,
                  center: Sequence[float] = CENTER) -> np.ndarray:
    """World points (X, Y, Z) (camera at (0, 0, h)) -> undistorted pixels."""
    P = np.asarray(P, float).reshape(-1, 3)
    cam = (rotation(theta, phi).T @ (P - np.array([0.0, 0.0, h])).T).T
    return (K_matrix(f, center) @ (cam / cam[:, 2:3]).T).T[:, :2]


# --------------------------------------------------------------------------- lens


def undistort_cv2(pts: np.ndarray, k1: float, f: float = F_PRIOR, center: Sequence[float] = CENTER) -> np.ndarray:
    """Distorted -> undistorted pixels with ``cv2.undistortPoints(pts, K, dist, P=K)`` (division model via the
    rational coefficients of ``framing.opencv_coeffs``; agrees with ``framing.undistort`` to < 1e-4 px)."""
    import cv2

    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if abs(k1) < 1e-18:
        return p.reshape(-1, 2).copy()
    oc = fr.opencv_coeffs(k1, f=f, center=center)
    K = np.asarray(oc["K"], np.float64)
    dist = np.asarray(oc["dist"], np.float64)
    # OpenCV's default criteria (5 iterations) leave ~0.2 px at the frame corners; iterate to convergence
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    out = cv2.undistortPoints(p, K, dist, R=np.eye(3), P=K, criteria=crit)
    return np.asarray(out, float).reshape(-1, 2)


# --------------------------------------------------------------------------- relations and the fit


Kind = Literal["parallel", "perp", "vertical"]


@dataclass
class Segment:
    """A floor direction: either two placed points (ids) or a traced line (undistorted pixel chain)."""
    name: str
    pts: np.ndarray                  # (n >= 2, 2) undistorted pixels


@dataclass
class Relation:
    """``parallel`` / ``perp``: floor directions a and b; ``vertical``: image line a is a vertical world edge (b unused)."""
    kind: Kind
    a: str
    b: str = ""


def vertical_vp(theta: float, phi: float, f: float = F_PRIOR, center: Sequence[float] = CENTER) -> np.ndarray:
    """Homogeneous image point where vertical world lines meet (undistorted pixels)."""
    return K_matrix(f, center) @ rotation(theta, phi).T @ np.array([0.0, 0.0, 1.0])


def vertical_resid_deg(chain: np.ndarray, vp: np.ndarray) -> float:
    """Angle (deg) between an image chain's TLS direction and the direction from its midpoint to the vertical VP."""
    p = np.asarray(chain, float)
    q = p - p.mean(0)
    _, vec = np.linalg.eigh(q.T @ q)
    d = vec[:, -1]
    m = p.mean(0)
    e = vp[:2] - m * vp[2]
    n = np.linalg.norm(e)
    if n < 1e-12:
        return 90.0
    e = e / n
    return math.degrees(math.asin(min(1.0, abs(d[0] * e[1] - d[1] * e[0]))))


def floor_direction(Hc: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Unit floor direction of a segment/chain (TLS through its floor points), oriented along the pixel order."""
    g = apply_h(Hc, pts)
    g = g[np.isfinite(g).all(1)]
    if len(g) < 2:
        return np.array([np.nan, np.nan])
    if len(g) == 2:
        d = g[1] - g[0]
    else:
        q = g - g.mean(0)
        _, vec = np.linalg.eigh(q.T @ q)
        d = vec[:, -1]
        if d @ (g[-1] - g[0]) < 0:
            d = -d
    return d / np.linalg.norm(d)


def relation_residuals_deg(theta: float, phi: float, segs: dict[str, Segment], rels: Sequence[Relation],
                           f: float = F_PRIOR) -> np.ndarray:
    """Angle residual (deg) of every relation: parallel -> angle between the directions, perp -> deviation from 90."""
    Hc = h_cam(theta, phi, 1.0, f)
    floor_names = {r.a for r in rels if r.kind != "vertical"} | {r.b for r in rels if r.kind != "vertical"}
    dirs = {k: floor_direction(Hc, segs[k].pts) for k in floor_names}
    vp = vertical_vp(theta, phi, f)
    out = []
    for r in rels:
        if r.kind == "vertical":
            out.append(vertical_resid_deg(segs[r.a].pts, vp))
            continue
        da, db = dirs[r.a], dirs[r.b]
        if not (np.isfinite(da).all() and np.isfinite(db).all()):
            out.append(90.0)
            continue
        if r.kind == "parallel":
            x = abs(da[0] * db[1] - da[1] * db[0])
        else:
            x = abs(float(da @ db))
        out.append(math.degrees(math.asin(min(1.0, x))))
    return np.asarray(out)


def fit_orientation(segs: dict[str, Segment], rels: Sequence[Relation], f: float = F_PRIOR,
                    theta0_deg: Sequence[float] = (15.0, 25.0, 35.0, 50.0)) -> dict:
    """(theta, phi) that make the stated floor relations hold; residuals in degrees."""
    from scipy.optimize import least_squares

    vert = {r.a for r in rels if r.kind == "vertical"}
    allpts = np.vstack([s.pts for k, s in segs.items() if k not in vert])

    def resid(x):
        th, ph = math.radians(x[0]), math.radians(x[1])
        if not ground_ok(th, ph, allpts, f).all():
            return np.full(len(rels), 90.0)
        return relation_residuals_deg(th, ph, segs, rels, f)

    best = None
    for t0 in theta0_deg:
        r = least_squares(resid, x0=[t0, 0.0], bounds=([2.0, -30.0], [85.0, 30.0]), diff_step=1e-4)
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    return {"theta_deg": float(best.x[0]), "phi_deg": float(best.x[1]),
            "resid_deg": [float(v) for v in resid(best.x)], "rms_deg": float(np.sqrt(np.mean(resid(best.x) ** 2)))}


def zone_transform(Hc: np.ndarray, origin_uv: np.ndarray, axis: Segment, offset_xz: Sequence[float] = (0.0, 0.0),
                   flip_x: bool = False) -> np.ndarray:
    """3x3 similarity floor(X, Y) -> zone frame (x, z): origin at ``origin_uv``, +z along ``axis`` (pixel order).

    x is +90 deg clockwise from z seen from above (right-handed x/z/up would be the other way), mirrored if ``flip_x``.
    """
    o = apply_h(Hc, np.asarray(origin_uv, float).reshape(1, 2))[0]
    dz = floor_direction(Hc, axis.pts)
    dx = np.array([dz[1], -dz[0]])
    if flip_x:
        dx = -dx
    R = np.array([dx, dz])                        # rows: new x axis, new z axis
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = -R @ o + np.asarray(offset_xz, float)
    return T


def floor_h(theta: float, phi: float, h: float, origin_uv: np.ndarray, axis: Segment,
            offset_xz: Sequence[float] = (0.0, 0.0), f: float = F_PRIOR, flip_x: bool = False) -> np.ndarray:
    """Full homography undistorted pixel -> zone frame (x, z) in metres, normalised to H[2,2] = 1."""
    Hc = h_cam(theta, phi, h, f)
    H = zone_transform(Hc, origin_uv, axis, offset_xz, flip_x) @ Hc
    return H / H[2, 2]


def homography_from_points(uv_undist: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """``cv2.findHomography`` (plain DLT, all points) from undistorted pixels to floor points, H[2,2] = 1."""
    import cv2

    H, _ = cv2.findHomography(np.asarray(uv_undist, np.float64), np.asarray(xz, np.float64), 0)
    if H is None:
        raise ValueError("findHomography failed")
    return np.asarray(H, float) / H[2, 2]


# --------------------------------------------------------------------------- scale checks


def person_height_h(foot_uv: np.ndarray, head_uv: np.ndarray, theta: float, phi: float, f: float = F_PRIOR,
                    person_h: float = WALKER_H_M) -> np.ndarray:
    """Camera height (m) implied by each upright person (undistorted foot and head-top pixels), orientation fixed.

    The foot ray meets the floor at h * g1 (g1 = floor point at h = 1); the head is person_h above it, i.e. at
    h * (g1x, g1y, s - 1) from the camera with s = person_h / h. In camera coordinates that is h * (a + s b) with
    a = M^T (g1x, g1y, -1) and b = M^T (0, 0, 1), so the head row v gives s in closed form:
    (v - cy) / f = (a_y + s b_y) / (a_z + s b_z). Vectorised (the bootstrap re-solves every box per draw). NaN where
    the foot is above the horizon, the head point is behind the camera, or h falls outside [0.5, 30] m.
    """
    foot = np.asarray(foot_uv, float).reshape(-1, 2)
    head = np.asarray(head_uv, float).reshape(-1, 2)
    out = np.full(len(foot), np.nan)
    if not len(foot):
        return out
    M = rotation(theta, phi)
    g1 = apply_h(h_cam(theta, phi, 1.0, f), foot)
    ok = np.isfinite(g1).all(1) & ground_ok(theta, phi, foot, f)
    a = np.column_stack([g1, -np.ones(len(g1))]) @ M          # rows: M^T (g1x, g1y, -1)
    b = M.T @ np.array([0.0, 0.0, 1.0])
    t = (head[:, 1] - CENTER[1]) / f
    den = b[1] - t * b[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (t * a[:, 2] - a[:, 1]) / den
        hh = person_h / s
    cz = a[:, 2] + s * b[2]
    ok &= np.isfinite(hh) & (np.abs(den) > 1e-12) & (s > 0) & (cz > 1e-9) & (hh >= 0.5) & (hh <= 30.0)
    out[ok] = hh[ok]
    return out


def stripe_h(edge_a: np.ndarray, edge_b: np.ndarray, theta: float, phi: float, width_m: float = 0.10,
             f: float = F_PRIOR) -> float:
    """Camera height implied by a painted stripe of ``width_m`` (both edges, undistorted pixel chains)."""
    Hc = h_cam(theta, phi, 1.0, f)
    ga, gb = apply_h(Hc, edge_a), apply_h(Hc, edge_b)
    ga, gb = ga[np.isfinite(ga).all(1)], gb[np.isfinite(gb).all(1)]
    q = ga - ga.mean(0)
    _, vec = np.linalg.eigh(q.T @ q)
    w1 = float(np.median(np.abs((gb - ga.mean(0)) @ vec[:, 0])))
    return width_m / w1 if w1 > 0 else float("nan")


# --------------------------------------------------------------------------- bootstrap


@dataclass
class FramingCal:
    """Everything needed to rebuild H for one framing from (perturbed) inputs."""
    framing: str
    k1: float
    points_raw: dict[str, np.ndarray]          # id -> distorted pixel (as placed)
    sigma_px: dict[str, float]                  # id -> stated 1-sigma placement uncertainty (px, isotropic)
    lines_undist: dict[str, np.ndarray]         # traced line name -> undistorted chain
    segments: dict[str, tuple[str, str]]        # segment name -> (point id, point id)
    relations: list[Relation]
    origin: str                                 # point id of the zone origin
    axis: str                                   # segment or line name giving +z
    offset_xz: tuple[float, float] = (0.0, 0.0)
    flip_x: bool = False
    f: float = F_PRIOR
    extra: dict = field(default_factory=dict)
    lines_raw: dict[str, np.ndarray] | None = None   # raw (distorted) chains of lines_undist, to redraw k1
    k1_alt: float | None = None                       # if set, draws take k1 ~ U(k1, k1_alt) (a weak lens fit)

    def build(self, pts_raw: dict[str, np.ndarray] | None = None, lines: dict[str, np.ndarray] | None = None,
              h: float = fr.MOUNT_H_PRIOR_M, k1: float | None = None, orientation: dict | None = None) -> dict:
        """H at camera height ``h``; the orientation is refitted unless given (``k1`` defaults to ``self.k1``)."""
        k = self.k1 if k1 is None else float(k1)
        pr = self.points_raw if pts_raw is None else pts_raw
        ln = self.lines_undist if lines is None else lines
        ids = list(pr)
        und = dict(zip(ids, undistort_cv2(np.array([pr[i] for i in ids]), k, self.f)))
        segs = {n: Segment(n, np.array([und[a], und[b]])) for n, (a, b) in self.segments.items()}
        segs.update({n: Segment(n, c) for n, c in ln.items()})
        if orientation is None:
            used = {r.a for r in self.relations} | {r.b for r in self.relations if r.b}
            orientation = fit_orientation({k_: v for k_, v in segs.items() if k_ in used}, self.relations, self.f)
        ori = orientation
        th, ph = math.radians(ori["theta_deg"]), math.radians(ori["phi_deg"])
        Hc = h_cam(th, ph, h, self.f)
        T = zone_transform(Hc, und[self.origin], segs[self.axis], self.offset_xz, self.flip_x)
        H = T @ Hc
        # camera-aligned floor (0, 0) is the camera nadir
        return {"H": H / H[2, 2], "orientation": ori, "undist": und, "nadir_xz": (T @ [0.0, 0.0, 1.0])[:2],
                "k1": k, "h": float(h)}


def draw_orientations(cal: FramingCal, n: int = 200, seed: int = 20261018) -> list[dict]:
    """``n`` bootstrap draws of the lens and the orientation, built at the 4 m prior (``at_height`` rescales them).

    Per draw: every placed point jittered by N(0, sigma) (its stated 1-sigma), every traced line resampled with
    replacement, k1 ~ U(k1, k1_alt) when ``cal.k1_alt`` is set (lines then re-undistorted from ``cal.lines_raw``), and
    tilt / roll refitted. Each draw keeps its inputs so the same draw can be rebuilt at any camera height.
    """
    if cal.k1_alt is not None:
        missing = sorted(set(cal.lines_undist) - set(cal.lines_raw or {}))
        if missing:
            raise ValueError(f"{cal.framing}: k1 is drawn but raw chains are missing for {missing}")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k1 = cal.k1 if cal.k1_alt is None else float(rng.uniform(min(cal.k1, cal.k1_alt), max(cal.k1, cal.k1_alt)))
        pts = {k: v + rng.normal(0.0, cal.sigma_px[k], 2) for k, v in cal.points_raw.items()}
        lines = {}
        for k, c in cal.lines_undist.items():
            idx = np.sort(rng.integers(0, len(c), len(c)))
            lines[k] = c[idx] if cal.k1_alt is None else fr.undistort(np.asarray(cal.lines_raw[k])[idx], k1)  # type: ignore[index]
        b = cal.build(pts, lines, fr.MOUNT_H_PRIOR_M, k1=k1)
        b["pts_raw"], b["lines"] = pts, lines
        out.append(b)
    return out


def at_height(cal: FramingCal, draw: dict, h: float) -> dict:
    """The same bootstrap draw (points, lines, k1, orientation kept) rebuilt at camera height ``h``."""
    b = cal.build(draw["pts_raw"], draw["lines"], h, k1=draw["k1"], orientation=draw["orientation"])
    b["pts_raw"], b["lines"] = draw["pts_raw"], draw["lines"]
    return b


def bootstrap(cal: FramingCal, h_lo: float, h_hi: float, n: int = 200, seed: int = 20261018) -> list[dict]:
    """``draw_orientations`` with h drawn uniformly over [h_lo, h_hi], INDEPENDENTLY of the draw's tilt.

    Only for a known, tilt-independent height range (synthetic tests). CAL v1 draws h jointly with each draw's tilt
    (scripts/cal_v1.py ``joint_bootstrap``).
    """
    rng = np.random.default_rng(seed + 1)
    return [at_height(cal, d, float(rng.uniform(h_lo, h_hi))) for d in draw_orientations(cal, n, seed)]


def floor_of_raw(draw: dict, raw_px: np.ndarray) -> np.ndarray:
    """Zone-frame floor points (m) of RAW (distorted) pixels under one draw (its own k1 and H)."""
    return apply_h(draw["H"], undistort_cv2(np.asarray(raw_px, float).reshape(-1, 2), draw["k1"]))


def displacement_raw_m(nominal: dict, draws: Sequence[dict], raw_px: np.ndarray) -> np.ndarray:
    """(n_draws, n_pts) floor displacement (m) of RAW pixels between the nominal build and every draw (k1 per draw)."""
    x0 = floor_of_raw(nominal, raw_px)
    return np.array([np.linalg.norm(floor_of_raw(d, raw_px) - x0, axis=1) for d in draws])


def displacement_m(H0: np.ndarray, Hs: Sequence[np.ndarray], uv_undist: np.ndarray) -> np.ndarray:
    """(n_draws, n_pts) floor displacement (m) of each pixel between the nominal H0 and every draw."""
    x0 = apply_h(H0, uv_undist)
    return np.array([np.linalg.norm(apply_h(H, uv_undist) - x0, axis=1) for H in Hs])
