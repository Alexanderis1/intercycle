"""Probe P6 (PLAN_TWIN §3): framings, lens distortion (CAL v0 k1) and a scale cross-check for the press-shop cameras.

Pure numpy/scipy (cv2 only where a caller passes images). Conventions:

* Pixel coordinates are in the full 1920x1080 frame; the distortion centre is the image centre ``CENTER``.
* Lens model = one-parameter **division model** (Fitzgibbon 2001): ``x_u = c + (x_d - c) / (1 + k1 * r_d^2)``
  with ``r_d = |x_d - c|`` in pixels, so ``k1`` is in px^-2 (barrel distortion: k1 < 0). ``k1_norm = k1 * F_PRIOR^2``
  is the same number in focal-normalised units.
* Camera priors: UNV IPC2122CR3-PF40-A, 4.0 mm, 1/2.7", 1920x1080, mounted ~4 m high. The vendor FOV (86.5 H / 44.1 V)
  includes barrel distortion, so it is not used as a pinhole; f prior ~ 1330 px (from the VFOV), range 1020-1430 px.
* Ground-plane camera: zero roll, tilt ``theta`` down from horizontal, height ``h`` above the floor. World axes:
  X right, Y forward (along the floor), Z up; camera axes x right, y down, z forward.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

W, H_IMG = 1920, 1080
CENTER = ((W - 1) / 2.0, (H_IMG - 1) / 2.0)
F_PRIOR = 1330.0
F_RANGE = (1020.0, 1430.0)
MOUNT_H_PRIOR_M = 4.0
WALKER_H_M = 1.72
VENDOR_HFOV_DEG, VENDOR_VFOV_DEG = 86.5, 44.1


def focal_from_fov(fov_deg: float, extent_px: float) -> float:
    """Pinhole focal length (px) that maps a full field of view ``fov_deg`` onto ``extent_px`` pixels."""
    return (extent_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def vendor_k1_norm_prior(f: float = F_PRIOR) -> float:
    """Division-model k1 (focal-normalised) that would put the vendor's HFOV edge ray at the image edge, given ``f``.

    The edge ray (half-angle HFOV/2) lands at undistorted radius f*tan(HFOV/2); in the distorted image it is at W/2.
    Only a consistency check of the priors, not a measurement.
    """
    r_d = (W / 2.0) / f
    r_u = math.tan(math.radians(VENDOR_HFOV_DEG) / 2.0)
    return (r_d / r_u - 1.0) / r_d**2


# --------------------------------------------------------------------------- division model


def undistort(pts: np.ndarray, k1: float, center: Sequence[float] = CENTER) -> np.ndarray:
    """Distorted -> undistorted pixel coordinates under the division model (k1 in px^-2)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    c = np.asarray(center, float)
    d = p - c
    r2 = (d**2).sum(1, keepdims=True)
    return c + d / (1.0 + k1 * r2)


def distort(pts: np.ndarray, k1: float, center: Sequence[float] = CENTER) -> np.ndarray:
    """Undistorted -> distorted pixel coordinates (exact inverse of ``undistort`` on its valid branch)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    c = np.asarray(center, float)
    d = p - c
    r_u = np.sqrt((d**2).sum(1))
    out = p.copy()
    nz = r_u > 1e-12
    if abs(k1) < 1e-18:
        return out
    # r_u = r_d / (1 + k1 r_d^2)  ->  k1 r_u r_d^2 - r_d + r_u = 0, root that tends to r_u as k1 -> 0
    disc = 1.0 - 4.0 * k1 * r_u[nz] ** 2
    if np.any(disc < 0):
        raise ValueError("point outside the valid range of the division model")
    r_d = 2.0 * r_u[nz] / (1.0 + np.sqrt(disc))
    out[nz] = c + d[nz] * (r_d / r_u[nz])[:, None]
    return out


def line_rms(pts: np.ndarray) -> float:
    """RMS perpendicular distance (px) of points to their total-least-squares line."""
    p = np.asarray(pts, float).reshape(-1, 2)
    q = p - p.mean(0)
    ev = np.linalg.eigvalsh(q.T @ q / len(q))
    return float(math.sqrt(max(ev[0], 0.0)))


def chain_cost(chains: Iterable[np.ndarray], k1: float, center: Sequence[float] = CENTER) -> float:
    """Sum over chains of the mean squared distance to the chain's TLS line after undistortion."""
    return float(sum(line_rms(undistort(c, k1, center)) ** 2 for c in chains))


def fit_k1(chains: Sequence[np.ndarray], center: Sequence[float] = CENTER,
           k1_norm_bounds: tuple[float, float] = (-1.2, 0.4), f_norm: float = F_PRIOR) -> dict:
    """Plumb-line fit of one division-model k1 to point chains that are straight lines in the world.

    Returns k1 (px^-2), k1_norm, the RMS line residual before and after, and per-chain residuals after.
    """
    from scipy.optimize import minimize_scalar

    if not chains:
        raise ValueError("no chains")
    s = f_norm**2
    res = minimize_scalar(lambda kn: chain_cost(chains, kn / s, center), bounds=k1_norm_bounds, method="bounded",
                          options={"xatol": 1e-5})
    kn_best = float(getattr(res, "x"))
    k1 = kn_best / s
    after = [line_rms(undistort(c, k1, center)) for c in chains]
    before = [line_rms(c) for c in chains]
    return {"k1": k1, "k1_norm": kn_best, "rms_before_px": before, "rms_after_px": after,
            "at_bound": bool(min(abs(kn_best - b) for b in k1_norm_bounds) < 1e-3)}


def fit_k1_robust(chains: Sequence[np.ndarray], max_after_px: float = 1.5, n_top: int = 3,
                  center: Sequence[float] = CENTER) -> dict:
    """Joint fit, drop chains whose residual after undistortion exceeds ``max_after_px``, refit (twice).

    Then the plan's literal estimate: the fit on the ``n_top`` longest consistent chains (2-3 stripes/edges), plus
    the leave-one-out spread over the consistent chains.
    """
    keep = list(range(len(chains)))
    for _ in range(3):
        fit = fit_k1([chains[i] for i in keep], center=center)
        new = [i for i, r in zip(keep, fit["rms_after_px"]) if r <= max_after_px]
        if len(new) < 2 or new == keep:
            break
        keep = new
    lengths = [chain_length(chains[i]) for i in keep]
    top = [keep[j] for j in np.argsort(lengths)[::-1][:n_top]]
    top_fit = fit_k1([chains[i] for i in top], center=center)
    all_fit = fit_k1([chains[i] for i in keep], center=center)
    loo = []
    if len(keep) >= 3:
        for i in keep:
            loo.append(fit_k1([chains[j] for j in keep if j != i], center=center)["k1_norm"])
    return {"consistent": keep, "top": top, "top_fit": top_fit, "all_fit": all_fit, "loo_k1_norm": loo}


def chain_length(pts: np.ndarray) -> float:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = p - p.mean(0)
    _, vec = np.linalg.eigh(q.T @ q)
    proj = q @ vec[:, -1]
    return float(proj.max() - proj.min())


def opencv_coeffs(k1: float, f: float = F_PRIOR, center: Sequence[float] = CENTER, n: int = 60) -> dict:
    """OpenCV rational distortion vector [k1, k2, 0, 0, k3, k4, k5, k6] with K(f, center) fitted to the division model.

    With it, ``cv2.undistortPoints(pts, K, dist, P=K)`` reproduces ``undistort(pts, k1)``; ``max_err_px`` is the
    worst disagreement over the 1920x1080 frame.
    """
    us, vs = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H_IMG - 1, n))
    xd = np.stack([us.ravel(), vs.ravel()], 1)
    xu = undistort(xd, k1, center)
    c = np.asarray(center, float)
    nd, nu = (xd - c) / f, (xu - c) / f
    ru2 = (nu**2).sum(1)
    ok = ru2 > 1e-8
    # OpenCV rational model: x_d = x_u (1 + a1 r^2 + a2 r^4 + a3 r^6) / (1 + b1 r^2 + b2 r^4 + b3 r^6), r = r_u.
    # Linearised: s - 1 = a.R - s * b.R with s = r_d / r_u.
    s = np.sqrt((nd[ok] ** 2).sum(1) / ru2[ok])
    R = np.stack([ru2[ok], ru2[ok] ** 2, ru2[ok] ** 3], 1)
    A = np.hstack([R, -s[:, None] * R])
    coef, *_ = np.linalg.lstsq(A, s - 1.0, rcond=None)
    a, b = coef[:3], coef[3:]
    pred = nu[ok] * ((1.0 + R @ a) / (1.0 + R @ b))[:, None]
    err = np.sqrt(((pred - nd[ok]) ** 2).sum(1)).max() * f
    K = [[f, 0.0, c[0]], [0.0, f, c[1]], [0.0, 0.0, 1.0]]
    dist = [float(a[0]), float(a[1]), 0.0, 0.0, float(a[2]), float(b[0]), float(b[1]), float(b[2])]
    return {"K": K, "dist": dist, "max_err_px": float(err)}


# --------------------------------------------------------------------------- edge chains (plumb-line candidates)


def edge_chains(gray: np.ndarray, min_extent_px: float = 250.0, max_quad_rms_px: float = 1.0,
                max_sagitta_px: float = 40.0, canny: tuple[int, int] = (40, 120)) -> list[np.ndarray]:
    """Long, smooth, branch-free Canny edge chains (candidate straight world lines), in the image's pixels.

    Branch points (>2 edge neighbours) are removed so T-junctions split chains. A chain is kept if its extent along
    its main axis is >= ``min_extent_px``, a quadratic in the along-axis coordinate explains it to
    ``max_quad_rms_px`` (a smooth curve, not texture), and its sagitta is <= ``max_sagitta_px``.
    """
    import cv2

    g = cv2.GaussianBlur(gray, (5, 5), 1.2)
    e = (cv2.Canny(g, canny[0], canny[1]) > 0).astype(np.uint8)
    nb = cv2.filter2D(e, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - e
    e[(e > 0) & (nb > 2)] = 0
    n, lab = cv2.connectedComponents(e, connectivity=8)
    out = []
    ys, xs = np.nonzero(lab)
    labels = lab[ys, xs]
    order = np.argsort(labels, kind="stable")
    ys, xs, labels = ys[order], xs[order], labels[order]
    cuts = np.flatnonzero(np.diff(labels)) + 1
    for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        if len(xx) < min_extent_px * 0.8:
            continue
        p = np.stack([xx, yy], 1).astype(float)
        q = p - p.mean(0)
        _, vec = np.linalg.eigh(q.T @ q)
        a, b = q @ vec[:, -1], q @ vec[:, 0]
        if a.max() - a.min() < min_extent_px:
            continue
        if len(p) > 1.6 * (a.max() - a.min()):  # thick or doubled structure, not a 1-px curve
            continue
        coef = np.polyfit(a, b, 2)
        if math.sqrt(np.mean((np.polyval(coef, a) - b) ** 2)) > max_quad_rms_px:
            continue
        half = (a.max() - a.min()) / 2.0
        if abs(coef[0]) * half**2 > max_sagitta_px:
            continue
        out.append(p)
    return out


def trace_edge(gray: np.ndarray, p0: Sequence[float], p1: Sequence[float], half_width: float = 40.0,
               step: float = 6.0, min_grad: float = 8.0, lock_px: float = 5.0,
               polarity: float | None = None) -> np.ndarray:
    """Sub-pixel points of one straight-in-the-world edge (stripe border, wall edge) near the hint segment p0-p1.

    Pass 1: along the hint, every ``step`` px, the strongest gradient along the normal within +-``half_width`` px
    (its sign picks the polarity of the edge). Pass 2: a quadratic is fitted to pass-1 points (robustly), and the
    search is repeated within +-``lock_px`` of that curve with the same polarity, so the trace stays on one edge.
    Returns an (n, 2) array of pixel points.
    """
    from scipy.ndimage import map_coordinates

    g = np.asarray(gray, float)
    a0, a1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.hypot(*(a1 - a0)))
    t_dir = (a1 - a0) / L
    n_dir = np.array([-t_dir[1], t_dir[0]])
    ts = np.arange(0.0, L, step)

    def search(offsets_center: np.ndarray, hw: float, polarity: float | None):
        offs = np.arange(-hw, hw + 0.5, 0.5)
        base = a0[None, :] + ts[:, None] * t_dir[None, :] + offsets_center[:, None] * n_dir[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n_dir[None, None, :]
        prof = map_coordinates(g, [pts[..., 1].ravel(), pts[..., 0].ravel()], order=1, mode="nearest")
        prof = prof.reshape(len(ts), len(offs))
        prof = (prof[:, :-4] + prof[:, 1:-3] + prof[:, 2:-2] + prof[:, 3:-1] + prof[:, 4:]) / 5.0
        der = (prof[:, 2:] - prof[:, :-2]) / 1.0  # per 1 px (offset step 0.5 px)
        sgn = np.sign(der[np.arange(len(ts)), np.abs(der).argmax(1)]) if polarity is None else np.full(len(ts), polarity)
        sd = der * sgn[:, None]
        k = sd.argmax(1)
        val = sd[np.arange(len(ts)), k]
        # parabolic sub-sample peak
        km, kp = np.clip(k - 1, 0, der.shape[1] - 1), np.clip(k + 1, 0, der.shape[1] - 1)
        a, b, c = sd[np.arange(len(ts)), km], val, sd[np.arange(len(ts)), kp]
        den = a - 2 * b + c
        safe = np.where(np.abs(den) > 1e-9, den, 1.0)
        frac = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / safe, 0.0)
        off = offs[0] + (k + 1 + 2 + np.clip(frac, -1, 1)) * 0.5  # +2 from the 5-tap smoothing, +1 from the derivative
        return offsets_center + off, val, sgn

    off1, val1, sgn1 = search(np.zeros(len(ts)), half_width, polarity)
    ok = val1 >= min_grad
    if ok.sum() < 6:
        return np.empty((0, 2))
    pol = float(np.sign(np.median(sgn1[ok]))) if polarity is None else float(polarity)
    ok &= sgn1 == pol
    coef = np.polyfit(ts[ok], off1[ok], 2)
    for _ in range(3):  # trimmed refit
        r = np.abs(np.polyval(coef, ts) - off1)
        keep = ok & (r <= max(2.0, 2.5 * np.median(r[ok])))
        if keep.sum() < 6:
            break
        coef = np.polyfit(ts[keep], off1[keep], 2)
    off2, val2, _ = search(np.polyval(coef, ts), lock_px, pol)
    good = val2 >= min_grad
    pts = a0[None, :] + ts[good, None] * t_dir[None, :] + off2[good, None] * n_dir[None, :]
    return pts


# --------------------------------------------------------------------------- ground-plane camera and scale


def _rot(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return c, s


def foot_to_ground(uv: np.ndarray, f: float, h: float, theta: float,
                   center: Sequence[float] = CENTER) -> np.ndarray:
    """Undistorted foot pixels -> ground points (X, Y) in metres (rows below the horizon only; NaN otherwise)."""
    p = np.asarray(uv, float).reshape(-1, 2)
    c, s = _rot(theta)
    xc, yc = (p[:, 0] - center[0]) / f, (p[:, 1] - center[1]) / f
    Yd = c - yc * s  # zc = 1
    Zd = -yc * c - s
    below = Zd < -1e-9
    t = np.full(len(p), np.nan)
    t[below] = h / -Zd[below]
    return np.stack([t * xc, t * Yd], 1)


def project(P: np.ndarray, f: float, h: float, theta: float, center: Sequence[float] = CENTER) -> np.ndarray:
    """World points (X, Y, Z) -> undistorted pixels."""
    P = np.asarray(P, float).reshape(-1, 3)
    c, s = _rot(theta)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2] - h
    zc = Y * c - Z * s
    yc = -Y * s - Z * c
    return np.stack([center[0] + f * X / zc, center[1] + f * yc / zc], 1)


def head_rows(foot_uv: np.ndarray, f: float, h: float, theta: float, person_h: float = WALKER_H_M,
              center: Sequence[float] = CENTER) -> np.ndarray:
    """Predicted image row of the head top for a person of ``person_h`` standing at each foot pixel."""
    g = foot_to_ground(foot_uv, f, h, theta, center)
    P = np.column_stack([g, np.full(len(g), person_h)])
    return project(P, f, h, theta, center)[:, 1]


def fit_height_tilt(foot_uv: np.ndarray, head_v: np.ndarray, f: float, person_h: float = WALKER_H_M,
                    center: Sequence[float] = CENTER, loss_scale_px: float = 6.0) -> dict:
    """Robust LS fit of camera height h (m) and tilt (deg) from person foot pixels and head-top rows (undistorted)."""
    from scipy.optimize import least_squares

    foot_uv = np.asarray(foot_uv, float).reshape(-1, 2)
    head_v = np.asarray(head_v, float)

    def resid(x):
        r = head_rows(foot_uv, f, x[0], math.radians(x[1]), person_h, center) - head_v
        return np.nan_to_num(r, nan=1e3)

    best = None
    for th0 in (5.0, 20.0, 35.0, 50.0):
        r = least_squares(resid, x0=[MOUNT_H_PRIOR_M, th0], bounds=([1.0, 0.0], [15.0, 80.0]), loss="soft_l1",
                          f_scale=loss_scale_px)
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    rr = resid(best.x)
    return {"h_m": float(best.x[0]), "tilt_deg": float(best.x[1]), "median_abs_resid_px": float(np.median(np.abs(rr))),
            "n": int(len(head_v))}


def fit_height_linear(foot_v: np.ndarray, height_px: np.ndarray, person_h: float = WALKER_H_M) -> dict:
    """Horizon method (zero-tilt approximation): height_px = (person_h / h) * (v_foot - v0), Theil-Sen fit."""
    from scipy.stats import theilslopes

    ts_res = theilslopes(np.asarray(height_px, float), np.asarray(foot_v, float))
    slope, icpt, lo, hi = np.asarray(tuple(ts_res)[:4], float).tolist()
    v0 = -icpt / slope if slope != 0 else float("nan")
    return {"h_m": float(person_h / slope), "h_m_lo": float(person_h / hi), "h_m_hi": float(person_h / lo),
            "horizon_v": float(v0), "n": int(len(foot_v))}


def stripe_height(edge_a: np.ndarray, edge_b: np.ndarray, k1: float, f: float, theta: float,
                  width_m: float = 0.10, center: Sequence[float] = CENTER) -> dict:
    """Camera height implied by a floor stripe of known width (scale cross-check).

    Both edges are undistorted and projected to the floor with h = 1 m; ground coordinates scale linearly with h, so
    h = width_m / (stripe width measured at h = 1). Returns h and the stripe width in px.
    """
    ga = foot_to_ground(undistort(edge_a, k1, center), f, 1.0, theta, center)
    gb = foot_to_ground(undistort(edge_b, k1, center), f, 1.0, theta, center)
    ga, gb = ga[np.isfinite(ga).all(1)], gb[np.isfinite(gb).all(1)]
    q = ga - ga.mean(0)
    _, vec = np.linalg.eigh(q.T @ q)
    nrm = vec[:, 0]
    w1 = float(np.median(np.abs((gb - ga.mean(0)) @ nrm)))
    qa = edge_a - edge_a.mean(0)
    _, va = np.linalg.eigh(qa.T @ qa)
    w_px = float(np.median(np.abs((edge_b - edge_a.mean(0)) @ va[:, 0])))
    return {"h_m": width_m / w1 if w1 > 0 else float("nan"), "width_px": w_px, "n_a": int(len(ga)), "n_b": int(len(gb))}


def ground_m_per_px(v: float, f: float, h: float, theta: float, center: Sequence[float] = CENTER) -> float:
    """Metres on the floor per horizontal pixel at image row ``v`` (image centre column), for stripe/pallet checks."""
    g = foot_to_ground(np.array([[center[0], v], [center[0] + 1.0, v]]), f, h, theta, center)
    return float(np.hypot(*(g[1] - g[0])))


# --------------------------------------------------------------------------- framings


def corner_displacement(Hm: np.ndarray, w: float, h: float) -> float:
    """Largest displacement (px) of the 4 image corners under homography ``Hm``."""
    c = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], float)
    m = (np.asarray(Hm, float) @ c.T).T
    m = m[:, :2] / m[:, 2:3]
    return float(np.sqrt(((m - c[:, :2]) ** 2).sum(1)).max())


def assign_framings(ids: Sequence[str], camera: Callable[[str], int], can_seed: Callable[[str], bool],
                    match: Callable[[str, str], tuple[int, float]], min_inliers: int, max_disp_px: float) -> dict:
    """Greedy framing assignment per camera.

    Each clip is matched (``match(clip, ref) -> (inliers, corner displacement px)``) to the references of its camera in
    creation order and joins the first one with >= ``min_inliers`` and displacement <= ``max_disp_px``. Otherwise it
    founds a new framing if ``can_seed(clip)`` (train clips), else it is marked ``c<cam>_unmatched``.
    Returns {clip: {"framing_id", "ref", "inliers", "disp_px"}} and the reference list per camera.
    """
    refs: dict[int, list[str]] = {}
    out: dict[str, dict] = {}
    for cid in ids:
        cam = camera(cid)
        rl = refs.setdefault(cam, [])
        best = None
        for k, ref in enumerate(rl):
            inl, disp = match(cid, ref)
            if inl >= min_inliers and disp <= max_disp_px:
                best = (k, ref, inl, disp)
                break
        if best is not None:
            k, ref, inl, disp = best
            out[cid] = {"framing_id": f"c{cam}_f{k}", "ref": ref, "inliers": int(inl), "disp_px": float(disp)}
        elif can_seed(cid):
            rl.append(cid)
            out[cid] = {"framing_id": f"c{cam}_f{len(rl) - 1}", "ref": cid, "inliers": -1, "disp_px": 0.0}
        else:
            out[cid] = {"framing_id": f"c{cam}_unmatched", "ref": None, "inliers": 0, "disp_px": float("nan")}
    return {"clips": out, "refs": {str(k): v for k, v in refs.items()}}


def merge_framings(inv, framing: dict[str, str]):
    """Return a copy of inventory DataFrame ``inv`` with framing_id set for the clips in ``framing``.

    Only TRAIN and DEV rows may be set; a TEST clip id in ``framing`` raises.
    """
    out = inv.copy()
    split = dict(zip(out["clip_id"], out["split"]))
    bad = [c for c in framing if split.get(c) not in ("train", "dev")]
    if bad:
        raise ValueError(f"framing ids only for train/dev clips present in the inventory; offending: {bad[:5]}")
    m = out["clip_id"].map(framing)
    out["framing_id"] = m.where(m.notna(), out["framing_id"])
    return out
