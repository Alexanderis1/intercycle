"""The robot's own camera on the twin plant (lane S5a): the no-motion gate and the inspection capture, geometric only.

numpy + the standard library (Isaac-deps rule, tests/test_isaac_deps.py): a port of the body-shop model
(ros2/intercycle_eval/intercycle_eval/camera.py, which the twin may not import) with the same constants, so the gate
and capture behave as on the body-shop plant. It opens no file: the plant passes the press-ram parts, the static
caster (the runner's `attach(caster=...)`) and the mover primitives in memory.

No-motion gate (published on /intercycle/camera/observations, R2's contract): every GATE_DT, per press whose danger
rect is within GATE_RANGE_M of the robot, the press-ram parts are the targets; visible = >= VIS_FRAC of them inside the
wide-angle frustum (aimed at a FIXED site-map point, +- AIM_SIGMA) and unoccluded, AND a Bernoulli P_DETECT detection;
motion_seen = visible AND a visible part moves faster than MOTION_MPS. Published LATENCY_S later.
R-W3 (OPEN, disclosed): this is the body-shop near-perfect motion detector (true part speed, no pixel threshold, no
false motion, no platform sway). Lane S2 delivers the R-W3 pixel-motion gate as a module twin/plant.py imports in
`make_gate`; until then `GATE_SOURCE` tags every observation with this model.

Inspection capture: in_image, unoccluded (static mesh + movers), incidence < 60 deg, pixel density, blur < 1 px, taken
after the body has been still for HOLD_S (else at dwell_end, settled = false). The checkpoint's zoom multiplies the
focal length for pixel density and blur only (the aim tolerance is not narrowed): a PLACEHOLDER until lane S2's TASKS
inspection model (20x zoom, thermal, detectability formula, E-W3) replaces `capture`. pass = every check and
px/m >= DORI_INSPECT (1000 px/m, Amendment 1).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

W_PX, H_PX = 1920, 1080
HFOV, VFOV = math.radians(70.0), math.radians(45.0)
FX = (W_PX / 2) / math.tan(HFOV / 2)
FY = (H_PX / 2) / math.tan(VFOV / 2)
MOUNT = (0.45, 0.0, 0.35)
P_DETECT = 0.95
LATENCY_S = 0.15
MOTION_MPS = 0.05
GATE_DT = 0.2
GATE_HFOV, GATE_VFOV = math.radians(110.0), math.radians(80.0)
GATE_FX = (W_PX / 2) / math.tan(GATE_HFOV / 2)
GATE_FY = (H_PX / 2) / math.tan(GATE_VFOV / 2)
GATE_RANGE_M = 6.0
VIS_FRAC = 0.5
AIM_SIGMA = math.radians(2.5)
EXPOSURE_S = 0.004
DORI_IDENTIFY, DORI_INSPECT = 250.0, 1000.0
MAX_INCIDENCE_DEG = 60.0
HOLD_S, HOLD_V, HOLD_W = 1.0, 0.05, 0.05
GATE_SOURCE = "twin gate v0 (body-shop R3 gate ported; R-W3 open: true part speed, no pixel model)"
CAPTURE_SOURCE = "twin capture v0 (geometric; zoom on px/m and blur only; PLACEHOLDER until S2 TASKS)"


def cam_pose(x: float, y: float, z: float, psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array((x + c * MOUNT[0] - s * MOUNT[1], y + s * MOUNT[0] + c * MOUNT[1], z + MOUNT[2]))


def aim(cam: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    v = p - cam
    return math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))


def project(cam: np.ndarray, yaw: float, pitch: float, pts: np.ndarray, fx: float = FX, fy: float = FY):
    """(u, v) pixel coordinates (centre = 0) and forward depth of world points in a camera aimed (yaw, pitch)."""
    d = pts - cam[None, :]
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = np.array((cp * cy, cp * sy, sp))
    left = np.array((-sy, cy, 0.0))
    up = np.cross(fwd, left)
    zf = d @ fwd
    u = -(d @ left) / np.maximum(zf, 1e-9) * fx
    v = (d @ up) / np.maximum(zf, 1e-9) * fy
    return u, v, zf


def in_image(u, v, zf) -> np.ndarray:
    return (zf > 0.05) & (np.abs(u) <= W_PX / 2) & (np.abs(v) <= H_PX / 2)


def ray_circles(ox: float, oy: float, ux: np.ndarray, uy: np.ndarray, circles: np.ndarray | None, z: float,
                rmax: float) -> np.ndarray:
    """Range along each unit ray to the nearest vertical cylinder (cx, cy, r, zmax) whose top is above z."""
    r = np.full(ux.shape[0], rmax)
    if circles is None or len(circles) == 0:
        return r
    cc = circles[circles[:, 3] >= z]
    if len(cc) == 0:
        return r
    cx, cy, rr = cc[:, 0] - ox, cc[:, 1] - oy, cc[:, 2]
    bproj = ux[:, None] * cx[None, :] + uy[:, None] * cy[None, :]
    disc = bproj * bproj - (cx * cx + cy * cy - rr * rr)[None, :]
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = np.where((disc >= 0.0) & (bproj + sq > 0.0), np.maximum(bproj - sq, 0.0), np.inf)
    return np.minimum(r, t.min(axis=1))


def ray_boxes(ox: float, oy: float, ux: np.ndarray, uy: np.ndarray, boxes: np.ndarray | None, z: float,
              rmax: float) -> np.ndarray:
    """Range along each unit ray to the nearest upright box (cx, cy, hw, hd, yaw, zmax) whose top is above z."""
    r = np.full(ux.shape[0], rmax)
    if boxes is None or len(boxes) == 0:
        return r
    b = boxes[boxes[:, 5] >= z]
    if len(b) == 0:
        return r
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    dx, dy = ox - b[:, 0], oy - b[:, 1]
    lox, loy = dx * c + dy * s, -dx * s + dy * c
    ldx = ux[:, None] * c[None, :] + uy[:, None] * s[None, :]
    ldy = -ux[:, None] * s[None, :] + uy[:, None] * c[None, :]
    ldx = np.where(np.abs(ldx) < 1e-12, 1e-12, ldx)
    ldy = np.where(np.abs(ldy) < 1e-12, 1e-12, ldy)
    t1x, t2x = (-b[:, 2][None, :] - lox[None, :]) / ldx, (b[:, 2][None, :] - lox[None, :]) / ldx
    t1y, t2y = (-b[:, 3][None, :] - loy[None, :]) / ldy, (b[:, 3][None, :] - loy[None, :]) / ldy
    tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
    tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
    t = np.where(tmax >= np.maximum(tmin, 0.0), np.maximum(tmin, 0.0), np.inf)
    return np.minimum(r, t.min(axis=1))


def occluded(static, cam: np.ndarray, pts: np.ndarray, prims: dict | None) -> np.ndarray:
    """True per point if the line from the camera is blocked before reaching it (static caster in 3D; movers as upright
    primitives that occlude if they reach the lower end of the line of sight)."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, bool)
    v = pts - cam[None, :]
    rng = np.linalg.norm(v, axis=1)
    u = v / np.maximum(rng, 1e-9)[:, None]
    lim = rng - 0.05
    blocked = np.zeros(len(pts), bool) if static is None else static.cast(cam, u, float(rng.max()) + 0.1) < lim
    if prims:
        h = np.hypot(u[:, 0], u[:, 1])
        ok = h > 1e-6
        if ok.any():
            ux, uy = u[ok, 0] / h[ok], u[ok, 1] / h[ok]
            hr = rng[ok] * h[ok]
            zmin = float(min(cam[2], pts[:, 2].min()))
            rr = np.minimum(ray_circles(cam[0], cam[1], ux, uy, prims.get("circles"), zmin, float(hr.max()) + 0.1),
                            ray_boxes(cam[0], cam[1], ux, uy, prims.get("boxes"), zmin, float(hr.max()) + 0.1))
            blocked[ok] |= rr / h[ok] < lim[ok]
    return blocked & (lim > 0)


class Gate:
    """No-motion observations per press, with detection noise and latency (interface: due / observe / ready)."""

    source = GATE_SOURCE

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.prev: dict[str, tuple[float, np.ndarray]] = {}
        self.buf: deque = deque()
        self.t_next = 0.0
        self.n_obs = 0

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.t_next

    def observe(self, t: float, cam: np.ndarray, machines: list[tuple[str, np.ndarray, np.ndarray]], static,
                prims: dict | None) -> None:
        """machines: [(press id, ram parts (k, 3) xyz world, fixed aim point (3,))]; prims = mover primitives."""
        self.t_next = t + GATE_DT
        targets = []
        for mid, parts, aim_pt in machines:
            yaw, pitch = aim(cam, aim_pt)
            yaw += self.rng.normal(0.0, AIM_SIGMA)
            pitch += self.rng.normal(0.0, AIM_SIGMA)
            prev = self.prev.get(mid)
            spd = np.zeros(len(parts))
            if prev is not None and prev[1].shape == parts.shape and t - prev[0] > 1e-6:
                spd = np.linalg.norm(parts - prev[1], axis=1) / (t - prev[0])
            self.prev[mid] = (t, parts.copy())
            u, v, zf = project(cam, yaw, pitch, parts, GATE_FX, GATE_FY)
            inimg = in_image(u, v, zf)
            occ = np.ones(len(parts), bool)
            if inimg.any():
                occ[inimg] = occluded(static, cam, parts[inimg], prims)
            vis = inimg & ~occ
            frac = float(vis.mean()) if len(parts) else 0.0
            det = frac >= VIS_FRAC and self.rng.random() < P_DETECT
            motion = bool(det and (spd[vis] > MOTION_MPS).any())
            targets.append({"id": mid, "visible": bool(det), "motion_seen": motion, "vis_frac": round(frac, 2)})
        self.buf.append((t + LATENCY_S, targets))
        self.n_obs += 1

    def ready(self, t: float) -> list[dict]:
        out = []
        while self.buf and self.buf[0][0] <= t + 1e-9:
            _, tg = self.buf.popleft()
            if tg:
                out.append({"stamp": round(t, 4), "source": self.source, "targets": tg})
        return out


def capture(t: float, cam: np.ndarray, target_xyz: np.ndarray, normal: np.ndarray, v_world: np.ndarray, wz: float,
            static, prims: dict | None, rng: np.random.Generator, zoom: float = 1.0) -> dict:
    yaw, pitch = aim(cam, target_xyz)
    yaw_e, pitch_e = yaw + rng.normal(0.0, AIM_SIGMA), pitch + rng.normal(0.0, AIM_SIGMA)
    u, v, zf = project(cam, yaw_e, pitch_e, target_xyz[None, :])
    inimg = bool(in_image(u, v, zf)[0])
    occ = bool(occluded(static, cam, target_xyz[None, :], prims)[0])
    los = target_xyz - cam
    rng_m = float(np.linalg.norm(los))
    ulos = los / max(rng_m, 1e-9)
    n = normal / max(float(np.linalg.norm(normal)), 1e-9)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(-ulos @ n)))))
    f = FX * float(zoom)
    px_per_m = f / max(rng_m, 1e-6) * max(math.cos(math.radians(inc)), 0.0)
    vw = np.asarray(v_world, float)
    v_perp = vw - (vw @ ulos) * ulos
    blur = (float(np.linalg.norm(v_perp)) / max(rng_m, 1e-6) + abs(float(wz))) * f * EXPOSURE_S
    checks = {"in_image": inimg, "unoccluded": not occ, "incidence_ok": inc < MAX_INCIDENCE_DEG,
              "dori_identify": px_per_m >= DORI_IDENTIFY, "blur_ok": blur < 1.0}
    ok = all(checks.values())
    return {"t": round(t, 3), "range_m": round(rng_m, 3), "incidence_deg": round(inc, 2), "zoom": float(zoom),
            "px_per_m": round(px_per_m, 1), "blur_px": round(blur, 3), "checks": checks,
            "pass": bool(ok and px_per_m >= DORI_INSPECT), "source": CAPTURE_SOURCE}
