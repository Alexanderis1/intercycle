"""Social force model for the simulated people, numpy only (runs inside Isaac's python 3.11: no numba, no scipy).

The force terms and default parameters are those of PySocialForce (Gao & contributors, MIT licence,
github.com/yuxiang-gao/PySocialForce), re-implemented here because PySocialForce needs numba, which Isaac's python lacks:
  * desired force   (v0 * e_goal - v) / tau                                   relaxation_time 0.5 s
  * social force    Moussaid et al. 2009 ("Experimental study of the behavioural mechanisms underlying
                    self-organization in human crowds"), factor 5.1, lambda 2.0, gamma 0.35, n 2, n' 3
  * obstacle force  exp(-d / sigma) away from the nearest static surface, sigma 0.2, factor 10, threshold 3 m
Speeds are capped at 1.3 x the preferred speed (PySocialForce max_speed_multiplier).

The ROBOT is one more agent in the social force (it moves with its true velocity), and for people who react its
1.0 x 0.64 m body is also an obstacle (the obstacle term, plus a small keep-right tangential share so a head-on approach
steps around it; the Moussaid term alone lets a walker creep into a STANDING agent). `reacts` flags who reacts to the robot: ~20 % of people are impassive / distracted and
ignore it (they still avoid each other and walls). The "invisible robot" stress condition sets every flag False.
Nobody yields on command: the only coupling is the force.
"""
from __future__ import annotations

import numpy as np

TAU = 0.5
SOC_FACTOR, LAMBDA, GAMMA, N_ANG, N_VEL = 5.1, 2.0, 0.35, 2.0, 3.0
OBS_SIGMA, OBS_FACTOR, OBS_THRESH = 0.2, 10.0, 3.0
MAX_SPEED_MULT = 1.3
PERSON_R = 0.3
ROBOT_HL, ROBOT_HW = 0.5, 0.32          # robot footprint half-length / half-width (obstacle term)
KEEP_RIGHT = 0.3                        # tangential share of the robot-obstacle force (step around, keep right)


def _norm(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1)
    return v / np.maximum(n, 1e-9)[..., None], n


def _angle(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[..., 1], v[..., 0])


def social_force(pos: np.ndarray, vel: np.ndarray, src_pos: np.ndarray, src_vel: np.ndarray,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Moussaid social force on each agent i (pos (N,2)) from each source j (src_pos (M,2)); mask (N,M) False = ignore.
    Same algebra as PySocialForce.forces.SocialForce (diff = i - j, velocity term v_j - v_i)."""
    if len(pos) == 0 or len(src_pos) == 0:
        return np.zeros_like(pos)
    diff = pos[:, None, :] - src_pos[None, :, :]                        # (N, M, 2)
    ddir, dlen = _norm(diff)
    vdiff = src_vel[None, :, :] - vel[:, None, :]
    inter = LAMBDA * vdiff + ddir
    idir, ilen = _norm(inter)
    theta = _angle(idir) - _angle(ddir)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    B = GAMMA * np.maximum(ilen, 1e-9)
    f_v = np.exp(-dlen / B - np.square(N_VEL * B * theta))
    f_a = -np.sign(theta) * np.exp(-dlen / B - np.square(N_ANG * B * theta))
    left = np.stack((-idir[..., 1], idir[..., 0]), axis=-1)
    f = f_v[..., None] * idir + f_a[..., None] * left
    valid = dlen > 1e-6
    if mask is not None:
        valid &= mask
    f = np.where(valid[..., None], f, 0.0)
    return SOC_FACTOR * f.sum(axis=1)


def box_closest(px: np.ndarray, py: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point on each upright box (cx, cy, hw, hd, yaw) to each point. Returns (N, B, 2) points and (N, B) dist
    (0 inside)."""
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    dx, dy = px[:, None] - boxes[None, :, 0], py[:, None] - boxes[None, :, 1]
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    qx = np.clip(lx, -boxes[:, 2], boxes[:, 2])
    qy = np.clip(ly, -boxes[:, 3], boxes[:, 3])
    wx = boxes[None, :, 0] + qx * c - qy * s
    wy = boxes[None, :, 1] + qx * s + qy * c
    pts = np.stack((wx, wy), axis=-1)
    d = np.hypot(px[:, None] - wx, py[:, None] - wy)
    return pts, d


def obstacle_force(pos: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """PySocialForce ObstacleForce: exp(-d/sigma) pushed away from the nearest obstacle point, within the threshold."""
    if len(pos) == 0 or len(boxes) == 0:
        return np.zeros_like(pos)
    pts, d = box_closest(pos[:, 0], pos[:, 1], boxes)
    k = np.argmin(d, axis=1)
    near = pts[np.arange(len(pos)), k]
    dist = d[np.arange(len(pos)), k] - PERSON_R
    away, _ = _norm(pos - near)
    mag = np.where(dist < OBS_THRESH, np.exp(-np.maximum(dist, 0.0) / OBS_SIGMA), 0.0)
    return OBS_FACTOR * mag[:, None] * away


class Crowd:
    """People integrated with the social force at the plant tick (semi-implicit Euler)."""

    def __init__(self, n: int, static_boxes: np.ndarray, reacts: np.ndarray):
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.active = np.zeros(n, bool)
        self.reacts = np.asarray(reacts, bool)
        self.boxes = np.asarray(static_boxes, float).reshape(-1, 5)

    def place(self, i: int, xy) -> None:
        self.pos[i] = xy
        self.vel[i] = 0.0
        self.active[i] = True

    def step(self, dt: float, goals: np.ndarray, v0: np.ndarray, robot_xy, robot_v, robot_psi: float = 0.0) -> None:
        a = np.flatnonzero(self.active)
        if len(a) == 0:
            return
        p, v = self.pos[a], self.vel[a]
        gdir, gdist = _norm(goals[a] - p)
        want = np.where(gdist[:, None] > 0.2, gdir * v0[a, None], 0.0)
        f = (want - v) / TAU
        m = ~np.eye(len(a), dtype=bool)
        f += social_force(p, v, p, v, m)
        if robot_xy is not None:
            rm = self.reacts[a][:, None]
            f += social_force(p, v, np.asarray(robot_xy, float)[None, :], np.asarray(robot_v, float)[None, :], rm)
            # a reacting person also treats the robot's body as an obstacle (PySocialForce obstacle term on the 1.0 x
            # 0.64 m footprint box), with a small keep-right tangential component so a head-on approach steps around
            # instead of stalling on the symmetry axis
            box = np.array([[robot_xy[0], robot_xy[1], ROBOT_HL, ROBOT_HW, robot_psi]])
            fo = obstacle_force(p, box) * self.reacts[a][:, None]
            heading, _ = _norm(np.where(np.linalg.norm(v, axis=1)[:, None] > 0.05, v, gdir))
            right = np.stack((heading[:, 1], -heading[:, 0]), axis=1)
            f += fo + KEEP_RIGHT * np.linalg.norm(fo, axis=1)[:, None] * right
        near = np.abs(self.boxes[:, 0][None, :] - p[:, 0:1]).min(axis=0) < 12.0 if len(self.boxes) else None
        if near is not None and near.any():
            f += obstacle_force(p, self.boxes[near])
        v = v + f * dt
        vmax = MAX_SPEED_MULT * np.maximum(v0[a], 0.05)
        sp = np.linalg.norm(v, axis=1)
        v = np.where((sp > vmax)[:, None], v * (vmax / np.maximum(sp, 1e-9))[:, None], v)
        self.vel[a] = v
        self.pos[a] = p + v * dt
