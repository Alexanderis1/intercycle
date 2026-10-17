"""Robot camera checks on R1's geometric camera model (sim side; no rendering): the no-motion gate and inspection capture.

Camera: the robot's pan-tilt inspection head (ANYmal-class payload), 1920 x 1080, 70 x 45 deg FOV
(intercycle_sim.sensors.CameraModel defaults), mounted 0.45 m ahead / 0.35 m above base_link. The head aims at the
site-map location of what it looks at (zone centre / inspection target); aiming uses the robot's true pose inside the
simulator plus a Gaussian aim error (stand-in for pose-estimate + head error, 2.5 deg).

No-motion gate (published on /intercycle/camera/observations, R2's contract). Wide-angle view 110 x 80 deg, aimed at a
FIXED site-map point per machine (zone centre at part height; the lifter at its carriage-travel midpoint), +- 2.5 deg:
  per machine whose hazard zone is within GATE_RANGE_M of the robot, every 0.2 s:
    the machine's MOVING parts (arm links and tools; lifter carriage; static arm bases excluded) are the targets.
    visible     = >= VIS_FRAC of the parts are inside the frustum and unoccluded (ray-cast against the static L0 mesh and
                  every other moving body), AND the detector fires (Bernoulli p_detect 0.95)
    motion_seen = visible AND a visible part moves faster than 0.05 m/s (finite difference over 0.2 s)
  then delayed by the camera latency (0.15 s) before it is published, stamped when it becomes available.
  A weld arm that dwells at a weld point is momentarily still: the camera cannot tell that from idle. That is the
  real weakness of a camera-only check and is left in.

Inspection capture (DIRECTION 5.5), checked at the moment of capture:
  in_image      target centre projects inside the image (after the aim error)
  unoccluded    line of sight free of static mesh and moving bodies
  incidence     angle between the line of sight and the target surface normal < 60 deg
  pixel density px/m at the target >= 250 (IEC 62676-4 DORI "Identify"); we also report the 1000 px/m "Inspect" level
  motion blur   (|v_perp| / range + |omega_yaw|) * f_px * exposure < 1 px, exposure 4 ms
Capture happens when the true body speed stays < 0.05 m/s and |yaw rate| < 0.05 rad/s for HOLD_S = 1.0 s after the
planner's dwell_start; if that never happens before dwell_end, the frame is taken at dwell_end (settled = false).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from intercycle_sim.sensors import ray_boxes, ray_circles

W_PX, H_PX = 1920, 1080
HFOV, VFOV = math.radians(70.0), math.radians(45.0)
FX = (W_PX / 2) / math.tan(HFOV / 2)
FY = (H_PX / 2) / math.tan(VFOV / 2)
MOUNT = (0.45, 0.0, 0.35)
P_DETECT = 0.95
LATENCY_S = 0.15
MOTION_MPS = 0.05
GATE_DT = 0.2
# the no-motion gate uses the robot's wide-angle view (same 1920 x 1080 sensor, 110 x 80 deg lens), aimed at a FIXED
# per-machine site-map point (never at the moving parts: that would be truth)
GATE_HFOV, GATE_VFOV = math.radians(110.0), math.radians(80.0)
GATE_FX = (W_PX / 2) / math.tan(GATE_HFOV / 2)
GATE_FY = (H_PX / 2) / math.tan(GATE_VFOV / 2)
LIFTER_AIM_Z = 3.2               # midpoint of the lifter carriage travel (0.35 .. 6 m), site-map value
GATE_RANGE_M = 6.0
VIS_FRAC = 0.5
AIM_SIGMA = math.radians(2.5)       # head aim error: the head is aimed from the TRUE pose, so this stands in for
                                    # the pose-estimate + head error (AUDIT: 2-3 deg)
EXPOSURE_S = 0.004
DORI_IDENTIFY, DORI_INSPECT = 250.0, 1000.0
MAX_INCIDENCE_DEG = 60.0
HOLD_S, HOLD_V, HOLD_W = 1.0, 0.05, 0.05
TARGET_Z = 1.0
PART_Z = 1.2
SOURCE = "camera_model (R1 geometric, R3 gate)"


def cam_pose(x: float, y: float, z: float, psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array((x + c * MOUNT[0] - s * MOUNT[1], y + s * MOUNT[0] + c * MOUNT[1], z + MOUNT[2]))


def aim(cam: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    v = p - cam
    return math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))


def project(cam: np.ndarray, yaw: float, pitch: float, pts: np.ndarray, fx: float | None = None, fy: float | None = None):
    """(u, v) pixel coordinates (centre = 0) and forward depth of world points in a camera aimed (yaw, pitch)."""
    fx = FX if fx is None else fx
    fy = FY if fy is None else fy
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


def occluded(static, cam: np.ndarray, pts: np.ndarray, prims: dict | None) -> np.ndarray:
    """True per point if the straight line from the camera is blocked before reaching it (static mesh at full 3D, moving
    bodies as upright primitives that occlude if they reach the lower end of the line of sight). Vectorised over points."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, bool)
    v = pts - cam[None, :]
    rng = np.linalg.norm(v, axis=1)
    u = v / np.maximum(rng, 1e-9)[:, None]
    lim = rng - 0.05
    r = static.cast(cam, u, float(rng.max()) + 0.1)
    blocked = r < lim
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
    """No-motion observations per machine, with detection noise and latency."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.prev: dict[str, tuple[float, np.ndarray]] = {}
        self.buf: deque = deque()
        self.t_next = 0.0
        self.n_obs = 0

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.t_next

    def observe(self, t: float, cam: np.ndarray, machines: list[tuple[str, np.ndarray, np.ndarray]], static,
                prims_other) -> None:
        """machines: [(id, parts (k, 3) xyz world, aim point (3,))]; prims_other(id) -> sensor prims without that machine."""
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
            elif prev is not None and prev[1].shape != parts.shape:
                spd = np.full(len(parts), 1.0)          # parts appeared / disappeared (lifter, transfer): motion
            self.prev[mid] = (t, parts.copy())
            if len(parts) == 0:
                targets.append({"id": mid, "visible": False, "motion_seen": False})
                continue
            u, v, zf = project(cam, yaw, pitch, parts, GATE_FX, GATE_FY)
            inimg = in_image(u, v, zf)
            occ = np.ones(len(parts), bool)
            if inimg.any():
                occ[inimg] = occluded(static, cam, parts[inimg], prims_other(mid))
            vis_parts = inimg & ~occ
            frac = float(vis_parts.mean())
            det = frac >= VIS_FRAC and self.rng.random() < P_DETECT
            motion = bool(det and (spd[vis_parts] > MOTION_MPS).any())
            targets.append({"id": mid, "visible": bool(det), "motion_seen": motion, "vis_frac": round(frac, 2)})
        self.buf.append((t + LATENCY_S, targets))
        self.n_obs += 1

    def ready(self, t: float) -> list[dict]:
        out = []
        while self.buf and self.buf[0][0] <= t + 1e-9:
            ta, tg = self.buf.popleft()
            if tg:
                out.append({"stamp": round(t, 4), "source": SOURCE, "targets": tg})
        return out


def capture(t: float, cam: np.ndarray, target_xyz: np.ndarray, normal: np.ndarray, v_world: np.ndarray, wz: float,
            static, prims: dict | None, rng: np.random.Generator) -> dict:
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
    px_per_m = FX / max(rng_m, 1e-6) * max(math.cos(math.radians(inc)), 0.0)
    v_perp = np.asarray(v_world, float) - (np.asarray(v_world, float) @ ulos) * ulos
    blur = (float(np.linalg.norm(v_perp)) / max(rng_m, 1e-6) + abs(float(wz))) * FX * EXPOSURE_S
    checks = {"in_image": inimg, "unoccluded": not occ, "incidence_ok": inc < MAX_INCIDENCE_DEG,
              "dori_identify": px_per_m >= DORI_IDENTIFY, "blur_ok": blur < 1.0}
    return {"t": round(t, 3), "cam_xyz": [round(float(c), 4) for c in cam], "cam_yaw": round(yaw_e, 5),
            "cam_pitch": round(pitch_e, 5), "hfov_deg": 70.0, "res": [W_PX, H_PX],
            "target_xyz": [round(float(c), 4) for c in target_xyz], "range_m": round(rng_m, 3),
            "incidence_deg": round(inc, 2), "px_per_m": round(px_per_m, 1), "blur_px": round(blur, 3),
            "dori_inspect": px_per_m >= DORI_INSPECT, "checks": checks, "pass": all(checks.values()),
            "pass_inspect": all(checks.values()) and px_per_m >= DORI_INSPECT}
