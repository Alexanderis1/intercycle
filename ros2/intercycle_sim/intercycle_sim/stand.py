"""STAND MODE of the locomotion layer (R-W6): hold the stance pose under a zero /cmd_vel against the ROBOT'S OWN pose
estimate, never the simulator's true pose.

Under a zero command the pretrained velocity policy creeps (about 5 cm/s and 1.5 rad/min in the dev runs), which a real
legged robot's stand state does not do. After `settle_s` of zero command a P-loop holds the pose the estimate had when
the hold started. The estimate is the robot's leg/IMU odometry stand-in (sensors.OdomModel: scale error, bias random walk,
noise), so odometry drift now shows up as real station-keeping drift (the base follows the drifting estimate). Without
an estimate the hold does nothing (zero command): it never falls back to truth. Any non-zero command releases the hold.
"""
from __future__ import annotations

import math

K_P = 1.0
V_MAX, W_MAX = 0.15, 0.2
DEADBAND_M, DEADBAND_RAD = 0.02, 0.02


class StandHold:
    def __init__(self, dt: float, settle_s: float = 0.2):
        self.dt, self.settle_s = float(dt), float(settle_s)
        self.zero_n = 0
        self.hold: tuple[float, float, float] | None = None
        self.n_holds = 0

    def step(self, cmd: tuple[float, float, float], est: tuple[float, float, float] | None) -> tuple[float, float, float]:
        vx, vy, wz = (float(v) for v in cmd)
        if max(abs(vx), abs(vy), abs(wz)) > 0.01:
            self.zero_n, self.hold = 0, None
            return vx, vy, wz
        self.zero_n += 1
        if self.zero_n * self.dt < self.settle_s or est is None:
            return vx, vy, wz
        x, y, psi = (float(v) for v in est)
        if self.hold is None:
            self.hold = (x, y, psi)
            self.n_holds += 1
        hx, hy, hpsi = self.hold
        c, s = math.cos(psi), math.sin(psi)
        ex, ey = c * (hx - x) + s * (hy - y), -s * (hx - x) + c * (hy - y)
        epsi = math.atan2(math.sin(hpsi - psi), math.cos(hpsi - psi))
        if math.hypot(ex, ey) < DEADBAND_M and abs(epsi) < DEADBAND_RAD:
            return 0.0, 0.0, 0.0

        def cl(v: float, m: float) -> float:
            return max(-m, min(m, v))

        return cl(K_P * ex, V_MAX), cl(K_P * ey, V_MAX), cl(K_P * epsi, W_MAX)
