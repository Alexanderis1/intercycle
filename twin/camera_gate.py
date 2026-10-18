"""R-W3 camera gate: the robot's no-motion gate as a pixel-motion detector (lane S2; reviewed by S5a).

Replaces the body-shop near-perfect gate of twin/robot_camera.py (`Gate`: motion = true part speed > 0.05 m/s) with
the same interface (due / observe / ready / n_obs / source) and the same inputs, so twin/plant.py swaps it in
`make_gate` by one import. numpy + the standard library only (Isaac-deps rule).

Model, per GATE_DT and per press in range (all constants ASSUMED priors, disclosed in `GATE_SOURCE`):
* Frame difference over N_DIFF frames at FRAME_HZ (dt_d = N_DIFF / FRAME_HZ). A visible ram part moving at true speed
  v (from the plant's ram-part positions) shifts in the image by d_px = |v_perp| * f / R * dt_d (v_perp = the part
  velocity component perpendicular to the line of sight, f = the wide gate focal length, R = range). Motion is seen
  iff max d_px over the visible parts > K_PX (not on true speed: a slow ram near TDC / BDC, or a far one, is missed).
* Ego-motion: the camera's own translation between observations (from the camera positions the plant passes) plus a
  body-sway angular rate omega ~ N(0, SWAY_STILL) when the body is still and N(0, SWAY_WALK) when it moves, give a
  global image shift g_px = (|omega| + |v_cam_perp| / R) * f * dt_d. Global-motion compensation removes all but
  EGO_RESIDUAL of it; a residual > K_PX is false motion on that press (fail-safe direction: the robot waits).
* Detection probability (range / occlusion dependent): visible iff the unoccluded in-frame share of the parts is
  >= VIS_FRAC, and a Bernoulli p = P_MAX * share * min(1, size_px / SIZE_FULL_PX), size_px = the ram's apparent size
  (RAM_SIZE_M * f / R). Not visible = no motion claim either way (the robot treats it as unknown).
* Fail closed: a press with no previous observation, or one older than STALE_DT = 1.5 * GATE_DT (first sight, or
  back in range after a gap), has no frame difference: it is reported visible = False (unknown), never
  visible with no motion.
* Latency: LATENCY_S (pipeline) + dt_d (the frame-difference window), the observation is published that much later.
Not done (listed): Isaac body rates instead of the sway prior; validation on R4 renders.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from twin import robot_camera as RC

FRAME_HZ = 30.0
N_DIFF = 3
DT_DIFF = N_DIFF / FRAME_HZ
K_PX = 2.0                       # motion threshold over the frame-difference window (px)
SWAY_STILL = math.radians(0.3)   # body-sway angular rate sigma, standing (rad/s), ASSUMED
SWAY_WALK = math.radians(4.0)    # body-sway angular rate sigma, walking (rad/s), ASSUMED
CAM_STILL_MPS = 0.05             # the camera is "still" below this translation speed
EGO_RESIDUAL = 0.2               # share of the global image shift left after compensation, ASSUMED
P_MAX = 0.95
SIZE_FULL_PX = 40.0
RAM_SIZE_M = 0.8                 # 2 * plant RAM_HALF_M
LATENCY_S = RC.LATENCY_S + DT_DIFF
STALE_DT = 1.5 * RC.GATE_DT          # previous observation older than this: no frame difference, visible = False
GATE_SOURCE = ("twin gate v1 (R-W3 pixel-motion model: frame difference 3 @ 30 Hz, K 2 px, sway + ego residual false "
               "motion, range/occlusion detection p, latency 0.25 s; constants ASSUMED priors)")


def pixel_shift(spd_perp: np.ndarray, rng: np.ndarray, f: float = RC.GATE_FX) -> np.ndarray:
    """Image shift (px) over the frame-difference window of points moving at spd_perp (m/s) at range rng (m)."""
    return np.asarray(spd_perp, float) * f / np.maximum(np.asarray(rng, float), 1e-6) * DT_DIFF


def p_detect(share: float, rng_m: float, f: float = RC.GATE_FX) -> float:
    """Detection probability of a ram whose unoccluded in-frame share is `share` at range rng_m."""
    if share < RC.VIS_FRAC:
        return 0.0
    size_px = RAM_SIZE_M * f / max(rng_m, 1e-6)
    return P_MAX * share * min(1.0, size_px / SIZE_FULL_PX)


class Gate:
    """No-motion observations per press from a pixel-motion model (interface: due / observe / ready / n_obs)."""

    source = GATE_SOURCE

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.prev: dict[str, tuple[float, np.ndarray]] = {}
        self.cam_prev: tuple[float, np.ndarray] | None = None
        self.buf: deque = deque()
        self.t_next = 0.0
        self.n_obs = 0
        self.n_false = 0                 # observations whose motion claim came from ego-motion only (diagnostic)

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.t_next

    def observe(self, t: float, cam: np.ndarray, machines: list[tuple[str, np.ndarray, np.ndarray]], static,
                prims: dict | None) -> None:
        """machines: [(press id, ram parts (k, 3) xyz world, fixed aim point (3,))]; prims = mover primitives."""
        self.t_next = t + RC.GATE_DT
        cam = np.asarray(cam, float)
        v_cam = np.zeros(3)
        if self.cam_prev is not None and t - self.cam_prev[0] > 1e-6:
            v_cam = (cam - self.cam_prev[1]) / (t - self.cam_prev[0])
        self.cam_prev = (t, cam.copy())
        walking = float(np.linalg.norm(v_cam[:2])) > CAM_STILL_MPS
        omega = abs(float(self.rng.normal(0.0, SWAY_WALK if walking else SWAY_STILL)))
        targets = []
        for mid, parts, aim_pt in machines:
            parts = np.asarray(parts, float).reshape(-1, 3)
            yaw, pitch = RC.aim(cam, aim_pt)
            yaw += self.rng.normal(0.0, RC.AIM_SIGMA)
            pitch += self.rng.normal(0.0, RC.AIM_SIGMA)
            prev = self.prev.get(mid)
            vel = np.zeros_like(parts)
            # a frame difference needs a recent previous frame: none, or older than STALE_DT, is "unknown" (fail closed)
            fresh = prev is not None and prev[1].shape == parts.shape and 1e-6 < t - prev[0] <= STALE_DT
            if fresh and prev is not None:
                vel = (parts - prev[1]) / (t - prev[0])
            self.prev[mid] = (t, parts.copy())
            u, v, zf = RC.project(cam, yaw, pitch, parts, RC.GATE_FX, RC.GATE_FY)
            inimg = RC.in_image(u, v, zf)
            occ = np.ones(len(parts), bool)
            if inimg.any():
                occ[inimg] = RC.occluded(static, cam, parts[inimg], prims)
            vis = inimg & ~occ
            share = float(vis.mean()) if len(parts) else 0.0
            los = parts - cam[None, :]
            rngs = np.linalg.norm(los, axis=1)
            ulos = los / np.maximum(rngs, 1e-9)[:, None]
            rng_m = float(rngs[vis].mean()) if vis.any() else float(rngs.mean()) if len(rngs) else math.inf
            det = bool(self.rng.random() < p_detect(share, rng_m)) and fresh
            v_perp = vel - (np.sum(vel * ulos, axis=1))[:, None] * ulos
            d_px = pixel_shift(np.linalg.norm(v_perp, axis=1), rngs)
            vc_perp = v_cam[None, :] - (ulos @ v_cam)[:, None] * ulos
            g_px = (omega + np.linalg.norm(vc_perp, axis=1) / np.maximum(rngs, 1e-6)) * RC.GATE_FX * DT_DIFF
            true_motion = bool(vis.any() and float(d_px[vis].max()) > K_PX)
            false_motion = bool(vis.any() and float((EGO_RESIDUAL * g_px[vis]).max()) > K_PX)
            motion = bool(det and (true_motion or false_motion))
            self.n_false += int(det and false_motion and not true_motion)
            targets.append({"id": mid, "visible": det, "motion_seen": motion, "vis_frac": round(share, 2)})
        self.buf.append((t + LATENCY_S, targets))
        self.n_obs += 1

    def ready(self, t: float) -> list[dict]:
        out = []
        while self.buf and self.buf[0][0] <= t + 1e-9:
            _, tg = self.buf.popleft()
            if tg:
                out.append({"stamp": round(t, 4), "source": self.source, "targets": tg})
        return out
