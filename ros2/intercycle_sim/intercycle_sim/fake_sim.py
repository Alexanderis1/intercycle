"""TEST DOUBLE ONLY (never a result): a kinematic omnidirectional base with first-order velocity tracking.

It exists so the bridge, the sensor models and the Nav2 bringup can be checked on a cheap Modal CPU container before
any GPU time is spent.  Every robot result comes from the Isaac Lab ANYmal-D backend (isaac_backend.py).
"""
from __future__ import annotations

import math

import numpy as np


class FakeBackend:
    name = "fake-kinematic-TEST-DOUBLE"

    def __init__(self, cfg: dict, boxes: list[dict]):
        self.dt = 0.02
        sx, sy, spsi = cfg.get("start", [4.0, 32.0, 0.0])
        self.x, self.y, self.psi, self.z = float(sx), float(sy), float(spsi), 0.6
        self.v = np.zeros(3)                 # body-frame vx, vy, wz
        self.tau = 0.25
        self.boxes = np.array([(b["cx"], b["cy"], b["w"] / 2, b["d"] / 2, b["yaw"]) for b in boxes]).reshape(-1, 5)
        self.policy_info = {"policy": "none (kinematic test double)", "observes_true_velocity": False}

    def set_movers(self, part_xyp: np.ndarray) -> None:
        pass

    def step(self, cmd: tuple[float, float, float]) -> dict:
        a = min(1.0, self.dt / self.tau)
        v_prev_w = self._v_world()
        self.v += a * (np.clip(np.asarray(cmd, float), [-1, -1, -1], [1, 1, 1]) - self.v)
        vw = self._v_world()
        nx, ny = self.x + vw[0] * self.dt, self.y + vw[1] * self.dt
        if not self._collides(nx, ny):
            self.x, self.y = nx, ny
        else:
            self.v[:2] = 0.0
            vw = np.zeros(3)
        self.psi = math.atan2(math.sin(self.psi + self.v[2] * self.dt), math.cos(self.psi + self.v[2] * self.dt))
        del v_prev_w
        return self.state(vw)

    def _v_world(self) -> np.ndarray:
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array((c * self.v[0] - s * self.v[1], s * self.v[0] + c * self.v[1], 0.0))

    def _collides(self, x: float, y: float, r: float = 0.45) -> bool:
        if not len(self.boxes):
            return False
        b = self.boxes
        c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
        dx, dy = x - b[:, 0], y - b[:, 1]
        lx, ly = np.abs(dx * c + dy * s), np.abs(-dx * s + dy * c)
        qx, qy = np.maximum(lx - b[:, 2], 0), np.maximum(ly - b[:, 3], 0)
        return bool((np.hypot(qx, qy) < r).any())

    def state(self, vw=None) -> dict:
        vw = self._v_world() if vw is None else vw
        return {"pos_w": [self.x, self.y, self.z], "quat_wxyz": [math.cos(self.psi / 2), 0.0, 0.0, math.sin(self.psi / 2)],
                "v_w": list(vw), "v_b": [self.v[0], self.v[1], 0.0], "omega_b": [0.0, 0.0, self.v[2]]}

    def contacts(self) -> list[dict]:
        return []

    def close(self) -> None:
        pass
