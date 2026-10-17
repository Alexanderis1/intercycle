"""Sensor models for the legged robot (no RTX): lidar, IMU, leg/IMU odometry, a geometric camera model.

All of them take the simulator's TRUE state as input and return only noisy sensor outputs; the true state never
leaves the simulator process (it is logged to disk for evaluation only).

Lidar (`/scan`): 2D, 360 deg, on a gravity-levelled frame (IMU-stabilised mount), mounted on base_link.
  ranges = min( static hit, moving-body hit ) per ray, then range limits, Gaussian noise, random dropouts.
  * static hit: Warp ray-cast against the ONE static L0 triangle mesh (`WarpMeshCaster`, Isaac Lab's Warp); the numpy
    fallback (`BoxCaster`) intersects the same upright boxes analytically (identical for horizontal rays).
  * moving bodies: analytic ray-circle / ray-oriented-box intersection with the movers' primitives (movers.py), i.e.
    the same shapes that are kinematic colliders in Isaac.  Isaac Lab 2.3's RayCaster only sees static meshes, hence
    the merge.  Occlusion is exact per ray (nearest return wins).  A body below the lidar plane is not seen.
IMU (`/imu`): gyro = true body rate + bias random walk + white noise; accelerometer = specific force (finite-difference
  of the true world velocity minus gravity, in body axes) + bias random walk + white noise.  No orientation is
  published (orientation_covariance[0] = -1), like a raw IMU.
Odometry (`/odom`): LABELLED leg/IMU-odometry stand-in = the true planar body velocity with a scale error, a bias random
  walk and white noise, integrated in the odom frame (so it drifts).  Not ground truth; AMCL corrects it.
Camera model (stub for R3): pinhole frustum + occlusion ray-cast + pixel density + incidence + detection noise/latency.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


# ------------------------------------------------------------------------------------------- analytic ray casts (2D)
def ray_boxes(ox: float, oy: float, ux: np.ndarray, uy: np.ndarray, boxes: np.ndarray, z: float,
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
    lox = dx * c + dy * s                               # ray origin in each box frame (M,)
    loy = -dx * s + dy * c
    ldx = ux[:, None] * c[None, :] + uy[:, None] * s[None, :]   # ray directions in each box frame (N, M)
    ldy = -ux[:, None] * s[None, :] + uy[:, None] * c[None, :]
    eps = 1e-12
    ldx = np.where(np.abs(ldx) < eps, eps, ldx)
    ldy = np.where(np.abs(ldy) < eps, eps, ldy)
    t1x = (-b[:, 2][None, :] - lox[None, :]) / ldx
    t2x = (b[:, 2][None, :] - lox[None, :]) / ldx
    t1y = (-b[:, 3][None, :] - loy[None, :]) / ldy
    t2y = (b[:, 3][None, :] - loy[None, :]) / ldy
    tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
    tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
    hit = (tmax >= np.maximum(tmin, 0.0))
    t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    return np.minimum(r, t.min(axis=1))


def ray_circles(ox: float, oy: float, ux: np.ndarray, uy: np.ndarray, circles: np.ndarray, z: float,
                rmax: float) -> np.ndarray:
    """Range along each unit ray to the nearest vertical cylinder (cx, cy, r, zmax) whose top is above z."""
    r = np.full(ux.shape[0], rmax)
    if circles is None or len(circles) == 0:
        return r
    cc = circles[circles[:, 3] >= z]
    if len(cc) == 0:
        return r
    cx, cy, rr = cc[:, 0] - ox, cc[:, 1] - oy, cc[:, 2]
    d2 = cx * cx + cy * cy
    bproj = ux[:, None] * cx[None, :] + uy[:, None] * cy[None, :]
    disc = bproj * bproj - (d2 - rr * rr)[None, :]
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = np.where((disc >= 0.0) & (bproj + sq > 0.0), np.maximum(bproj - sq, 0.0), np.inf)
    return np.minimum(r, t.min(axis=1))


# ------------------------------------------------------------------------------------------- static casters
class BoxCaster:
    """numpy fallback for the static mesh: analytic rays against the same upright boxes (exact for level rays)."""

    name = "numpy-boxes"

    def __init__(self, boxes: list[dict]):
        self.b = np.array([(b["cx"], b["cy"], b["w"] / 2, b["d"] / 2, b["yaw"], b["h"]) for b in boxes],
                          dtype=float).reshape(-1, 6)

    def cast(self, origin: np.ndarray, dirs: np.ndarray, rmax: float) -> np.ndarray:
        return ray_boxes(float(origin[0]), float(origin[1]), dirs[:, 0], dirs[:, 1], self.b, float(origin[2]), rmax)


try:  # Warp ships with Isaac Lab (python 3.11); the ROS-side python does not need it
    import warp as _wp

    @_wp.kernel
    def _warp_cast(mesh: _wp.uint64, origin: _wp.vec3, dirs: _wp.array(dtype=_wp.vec3), rmax: float,
                   out: _wp.array(dtype=float)):
        i = _wp.tid()
        q = _wp.mesh_query_ray(mesh, origin, dirs[i], rmax)
        if q.result:
            out[i] = q.t
        else:
            out[i] = rmax
except Exception:  # noqa: BLE001
    _wp = None
    _warp_cast = None


def _warp_kernel():
    if _warp_cast is None:
        raise ImportError("warp not available")
    return _warp_cast


class WarpMeshCaster:
    """Warp ray-cast (wp.mesh_query_ray, BVH) against the ONE static L0 triangle mesh; GPU or CPU."""

    name = "warp-mesh"

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str = "cuda:0"):
        import warp as wp

        wp.init()
        self.wp = wp
        self.device = device
        self._pts = wp.array(np.asarray(vertices, np.float32), dtype=wp.vec3, device=device)
        self._idx = wp.array(np.asarray(faces, np.int32).ravel(), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self._pts, indices=self._idx)
        self.kernel = _warp_kernel()
        self.name = f"warp-mesh ({device}, {len(faces)} tris)"

    def cast(self, origin: np.ndarray, dirs: np.ndarray, rmax: float) -> np.ndarray:
        wp = self.wp
        n = dirs.shape[0]
        d = wp.array(np.asarray(dirs, np.float32), dtype=wp.vec3, device=self.device)
        out = wp.zeros(n, dtype=float, device=self.device)
        o = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        wp.launch(self.kernel, dim=n, inputs=[self.mesh.id, o, d, float(rmax), out], device=self.device)
        return np.minimum(out.numpy().astype(float), rmax)


def make_static_caster(boxes: list[dict], vertices=None, faces=None, prefer_warp: bool = True, device: str = "cuda:0"):
    if prefer_warp and vertices is not None and len(vertices):
        try:
            return WarpMeshCaster(vertices, faces, device)
        except Exception as e:  # noqa: BLE001 - fall back, the caller logs caster.name
            print(f"[sensors] Warp caster unavailable ({e!r}); using the numpy box caster", flush=True)
    return BoxCaster(boxes)


# ------------------------------------------------------------------------------------------- lidar
class LidarModel:
    def __init__(self, static, n: int = 360, rmin: float = 0.10, rmax: float = 25.0, sigma: float = 0.015,
                 dropout: float = 0.005, mount: tuple[float, float, float] = (0.30, 0.0, 0.20), rate_hz: float = 10.0,
                 rng: np.random.Generator | None = None):
        self.static, self.n, self.rmin, self.rmax = static, n, rmin, rmax
        self.sigma, self.dropout, self.mount, self.rate_hz = sigma, dropout, mount, rate_hz
        self.rng = rng or np.random.default_rng(0)
        self.inc = 2 * math.pi / n
        self.angles = -math.pi + np.arange(n) * self.inc          # laser frame, counter-clockwise from -pi

    def true_ranges(self, x: float, y: float, z: float, psi: float, prims: dict | None) -> np.ndarray:
        """Noise-free merged ranges (levelled frame at the mount point), rmax = no return."""
        c, s = math.cos(psi), math.sin(psi)
        mx, my, mz = self.mount
        ox, oy, oz = x + c * mx - s * my, y + s * mx + c * my, z + mz
        a = psi + self.angles
        dirs = np.stack((np.cos(a), np.sin(a), np.zeros_like(a)), axis=1)
        r = self.static.cast(np.array((ox, oy, oz)), dirs, self.rmax)
        if prims:
            r = np.minimum(r, ray_circles(ox, oy, dirs[:, 0], dirs[:, 1], prims.get("circles"), oz, self.rmax))
            r = np.minimum(r, ray_boxes(ox, oy, dirs[:, 0], dirs[:, 1], prims.get("boxes"), oz, self.rmax))
        return r

    def scan(self, x: float, y: float, z: float, psi: float, prims: dict | None) -> list[float]:
        r = self.true_ranges(x, y, z, psi, prims)
        hit = r < self.rmax
        noisy = r + self.rng.normal(0.0, self.sigma, r.shape)
        drop = self.rng.random(r.shape) < self.dropout
        out = np.where(hit & ~drop & (noisy >= self.rmin), np.clip(noisy, self.rmin, self.rmax), np.inf)
        return [float(v) for v in out]

    def msg_meta(self) -> dict:
        return {"angle_min": -math.pi, "angle_max": -math.pi + (self.n - 1) * self.inc, "angle_increment": self.inc,
                "range_min": self.rmin, "range_max": self.rmax, "scan_time": 1.0 / self.rate_hz,
                "mount": list(self.mount)}


# ------------------------------------------------------------------------------------------- IMU
def quat_to_R(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(q_wxyz) -> float:
    w, x, y, z = (float(v) for v in q_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ImuModel:
    """Consumer-grade MEMS IMU (values in the ballpark of a BMI088 / ADIS-class unit)."""

    def __init__(self, gyro_sigma: float = 0.003, gyro_bias_rw: float = 2e-4, acc_sigma: float = 0.03,
                 acc_bias_rw: float = 1e-3, rng: np.random.Generator | None = None):
        self.gs, self.gb_rw, self.as_, self.ab_rw = gyro_sigma, gyro_bias_rw, acc_sigma, acc_bias_rw
        self.rng = rng or np.random.default_rng(1)
        self.gb = self.rng.normal(0.0, 0.002, 3)
        self.ab = self.rng.normal(0.0, 0.02, 3)
        self._v_prev = None

    def step(self, dt: float, R_wb: np.ndarray, omega_b: np.ndarray, v_w: np.ndarray) -> dict:
        v_w = np.asarray(v_w, float)
        a_w = np.zeros(3) if self._v_prev is None else (v_w - self._v_prev) / dt
        self._v_prev = v_w.copy()
        f_b = R_wb.T @ (a_w + np.array((0.0, 0.0, 9.81)))
        sq = math.sqrt(dt)
        self.gb += self.rng.normal(0.0, self.gb_rw * sq, 3)
        self.ab += self.rng.normal(0.0, self.ab_rw * sq, 3)
        g = np.asarray(omega_b, float) + self.gb + self.rng.normal(0.0, self.gs, 3)
        a = f_b + self.ab + self.rng.normal(0.0, self.as_, 3)
        return {"gyro": g.tolist(), "acc": a.tolist(), "gyro_var": self.gs ** 2, "acc_var": self.as_ ** 2}


# ------------------------------------------------------------------------------------------- odometry
class OdomModel:
    """LABELLED leg/IMU odometry stand-in: true planar body velocity * (1 + scale) + bias random walk + noise, integrated."""

    def __init__(self, x0: float = 0.0, y0: float = 0.0, psi0: float = 0.0, scale_sigma: float = 0.02,
                 v_sigma: float = 0.02, w_sigma: float = 0.005, v_bias_rw: float = 0.003, w_bias_rw: float = 0.0005,
                 rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(2)
        self.scale = 1.0 + self.rng.normal(0.0, scale_sigma, 3)
        self.bias = np.zeros(3)
        self.vs, self.ws, self.vrw, self.wrw = v_sigma, w_sigma, v_bias_rw, w_bias_rw
        self.x, self.y, self.psi = 0.0, 0.0, 0.0           # odom frame starts at the robot's start pose
        self.start = (x0, y0, psi0)

    def step(self, dt: float, vx_b: float, vy_b: float, wz: float) -> dict:
        sq = math.sqrt(dt)
        self.bias += self.rng.normal(0.0, 1.0, 3) * np.array((self.vrw, self.vrw, self.wrw)) * sq
        m = np.array((vx_b, vy_b, wz)) * self.scale + self.bias + \
            self.rng.normal(0.0, 1.0, 3) * np.array((self.vs, self.vs, self.ws))
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.x += (c * m[0] - s * m[1]) * dt
        self.y += (s * m[0] + c * m[1]) * dt
        self.psi = math.atan2(math.sin(self.psi + m[2] * dt), math.cos(self.psi + m[2] * dt))
        return {"x": self.x, "y": self.y, "psi": self.psi, "vx": float(m[0]), "vy": float(m[1]), "wz": float(m[2]),
                "v_var": self.vs ** 2, "w_var": self.ws ** 2}


# ------------------------------------------------------------------------------------------- camera (stub for R3)
class LatencyBuffer:
    """Delivers values `latency_s` after they were produced (sim time)."""

    def __init__(self, latency_s: float):
        self.lat = latency_s
        self.q: deque = deque()

    def push(self, t: float, value) -> None:
        self.q.append((t, value))

    def pop_ready(self, t: float) -> list:
        out = []
        while self.q and self.q[0][0] + self.lat <= t + 1e-9:
            out.append(self.q.popleft()[1])
        return out


class CameraModel:
    """Geometric robot camera (no rendering): pinhole frustum, occlusion ray-cast, pixel density, incidence angle,
    a detection probability with noise and a latency.  R3 builds the inspection checks / no-motion check on it.

    observe(...) -> dict per target: in_frustum, occluded, range_m, incidence_deg, px_per_m (at the target),
    detected (Bernoulli(p_detect) when visible), motion_seen (for moving targets: visible and speed above a threshold).
    """

    def __init__(self, static, hfov_deg: float = 70.0, vfov_deg: float = 45.0, width_px: int = 1920,
                 height_px: int = 1080, mount: tuple[float, float, float] = (0.45, 0.0, 0.35), p_detect: float = 0.95,
                 latency_s: float = 0.15, motion_thresh_mps: float = 0.05, rng: np.random.Generator | None = None):
        self.static, self.hfov, self.vfov = static, math.radians(hfov_deg), math.radians(vfov_deg)
        self.w, self.h, self.mount, self.p = width_px, height_px, mount, p_detect
        self.fx = (width_px / 2) / math.tan(self.hfov / 2)
        self.motion_thresh = motion_thresh_mps
        self.buf = LatencyBuffer(latency_s)
        self.rng = rng or np.random.default_rng(3)

    def pose(self, x: float, y: float, z: float, psi: float, pan: float = 0.0, tilt: float = 0.0):
        c, s = math.cos(psi), math.sin(psi)
        mx, my, mz = self.mount
        return np.array((x + c * mx - s * my, y + s * mx + c * my, z + mz)), psi + pan, tilt

    def observe(self, t: float, cam_xyz: np.ndarray, cam_yaw: float, cam_pitch: float, targets: list[dict],
                prims: dict | None = None) -> list[dict]:
        """targets: [{"id", "xyz": (3,), "normal": (3,) or None, "speed": m/s (optional)}]; pushes into the latency
        buffer and returns the (delayed) observations that are ready at time t."""
        fwd = np.array((math.cos(cam_pitch) * math.cos(cam_yaw), math.cos(cam_pitch) * math.sin(cam_yaw),
                        math.sin(cam_pitch)))
        res = []
        for tg in targets:
            p = np.asarray(tg["xyz"], float)
            v = p - cam_xyz
            rng_m = float(np.linalg.norm(v))
            u = v / max(rng_m, 1e-9)
            az = math.atan2(u[1], u[0]) - cam_yaw
            az = math.atan2(math.sin(az), math.cos(az))
            el = math.asin(max(-1.0, min(1.0, u[2]))) - cam_pitch
            in_fr = abs(az) <= self.hfov / 2 and abs(el) <= self.vfov / 2 and float(u @ fwd) > 0
            occl = False
            if in_fr:
                d = u[None, :]
                r_st = float(self.static.cast(cam_xyz, d, rng_m)[0])
                r_dyn = rng_m
                if prims:
                    r_dyn = min(float(ray_circles(cam_xyz[0], cam_xyz[1], d[:, 0], d[:, 1], prims.get("circles"),
                                                  cam_xyz[2], rng_m)[0]),
                                float(ray_boxes(cam_xyz[0], cam_xyz[1], d[:, 0], d[:, 1], prims.get("boxes"),
                                                cam_xyz[2], rng_m)[0]))
                occl = min(r_st, r_dyn) < rng_m - 0.05
            n = tg.get("normal")
            inc = 0.0 if n is None else math.degrees(math.acos(max(-1.0, min(1.0, float(-u @ np.asarray(n, float))))))
            px_per_m = self.fx / max(rng_m, 1e-6) * max(math.cos(math.radians(inc)), 0.0)
            vis = in_fr and not occl
            det = vis and (self.rng.random() < self.p)
            res.append({"id": tg["id"], "t": t, "in_frustum": in_fr, "occluded": occl, "range_m": rng_m,
                        "incidence_deg": inc, "px_per_m": px_per_m, "detected": bool(det),
                        "motion_seen": bool(det and tg.get("speed", 0.0) > self.motion_thresh)})
        self.buf.push(t, res)
        out = []
        for r in self.buf.pop_ready(t):
            out.extend(r)
        return out


# ------------------------------------------------------------------------------------------- suite
class SensorSuite:
    """Turns the simulator's true state into one bridge frame (protocol.py).  dt = policy step (s)."""

    def __init__(self, static, dt: float, seed: int = 0, start=(0.0, 0.0, 0.0), lidar_kw: dict | None = None):
        rs = np.random.SeedSequence(seed).spawn(4)
        self.dt = dt
        self.lidar = LidarModel(static, rng=np.random.default_rng(rs[0]), **(lidar_kw or {}))
        self.imu = ImuModel(rng=np.random.default_rng(rs[1]))
        self.odom = OdomModel(*start, rng=np.random.default_rng(rs[2]))
        self.camera = CameraModel(static, rng=np.random.default_rng(rs[3]))
        self.scan_every = max(1, int(round(1.0 / (self.lidar.rate_hz * dt))))
        self.k = 0

    def frame(self, t: float, pos_w, quat_wxyz, v_w, v_b, omega_b, prims: dict | None) -> dict:
        R = quat_to_R(quat_wxyz)
        fr = {"t": t,
              "imu": self.imu.step(self.dt, R, np.asarray(omega_b, float), np.asarray(v_w, float)),
              "odom": self.odom.step(self.dt, float(v_b[0]), float(v_b[1]), float(omega_b[2]))}
        if self.k % self.scan_every == 0:
            psi = yaw_of(quat_wxyz)
            fr["scan"] = dict(self.lidar.msg_meta(), ranges=self.lidar.scan(float(pos_w[0]), float(pos_w[1]),
                                                                            float(pos_w[2]), psi, prims))
        self.k += 1
        return fr
