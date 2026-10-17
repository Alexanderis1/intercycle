"""Ground truth for a machine installed after the map was learned (workstream M, new-machine experiment).

Only the truth knows about it: the observer is never told (the base scenario's `machines` list, which the observer
reads for its hazard zones, does not contain it). The camera sees its moving parts like any other machine part.

`installs` entries live in `scenario/bodyshop_newmachine.yaml` (an overlay on scenario/bodyshop.yaml):
    {id, kind: pick_place, x, z, t_install_s, period_s, busy, jitter, commission_s, link_m, pick: [dx, dz], place: [dx, dz]}

Model: a single 2-link pick-and-place arm on a pedestal at (x, z), links `link_m` each, drawn as 5 circles (pedestal,
upper-arm mid, elbow, forearm mid, tool) like the cell robots in truth/kinematics.py. Steady cycle: `busy` of the
period active (home -> pick -> place -> home, short dwells), the rest stowed. During the first `commission_s` after
installation the cell is being commissioned: cycles of random length (gaps 0-90 s, active 0.6-1.4x) - the observer must
not mistake that for a converged period. Afterwards cycles follow the period with `jitter` (sigma, fraction).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

R5 = (0.55, 0.40, 0.40, 0.40, 0.40)


def load_installs(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get("installs", []))


class NewMachine:
    def __init__(self, cfg: dict, seed: int = 0, t_end: float | None = None):
        self.cfg = dict(cfg)
        self.id = cfg["id"]
        self.x, self.z = float(cfg["x"]), float(cfg["z"])
        self.t_install = float(cfg["t_install_s"])
        self.P = float(cfg.get("period_s", 50.0))
        self.busy = float(cfg.get("busy", 0.7))
        self.jit = float(cfg.get("jitter", 0.01))
        self.comm = float(cfg.get("commission_s", 1800.0))
        self.L = float(cfg.get("link_m", 1.6))
        self.pick = np.array(cfg.get("pick", [-2.6, 0.6]), float)
        self.place = np.array(cfg.get("place", [2.6, -0.6]), float)
        self.home = np.array([0.0, 0.5], float)
        rng = np.random.default_rng([seed, 4242, int(self.t_install)])
        t_end = t_end if t_end is not None else self.t_install + 30 * 86_400.0
        starts, acts = [], []
        t = self.t_install
        while t < t_end:
            if t < self.t_install + self.comm:
                a = self.busy * self.P * rng.uniform(0.6, 1.4)
                gap = rng.uniform(0.0, 90.0)
                starts.append(t)
                acts.append(a)
                t += a + gap
            else:
                p = self.P * (1 + self.jit * rng.standard_normal())
                starts.append(t)
                acts.append(self.busy * p)
                t += p
        self.starts, self.acts = np.array(starts), np.array(acts)

    # ------------------------------------------------------------------ truth API
    def rect(self, pad: float = 0.0) -> tuple[float, float, float, float]:
        """Axis-aligned box around everything the arm can sweep (the true hazard zone)."""
        r = 2 * self.L + max(R5) + pad
        return (self.x - r, self.z - r, self.x + r, self.z + r)

    def state(self, t: float) -> tuple[str, float]:
        """("absent" | "commissioning" | "active" | "idle", fraction within the state)."""
        if t < self.t_install:
            return "absent", 0.0
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        if i < 0:
            return "idle", 0.0
        f = (t - self.starts[i]) / self.acts[i]
        tag = "commissioning" if t < self.t_install + self.comm else "active"
        return (tag, float(f)) if f < 1.0 else ("idle", 0.0)

    def cycle_phase(self, t: float) -> float:
        """Fraction of the current cycle (start to next start), NaN before installation."""
        if t < self.t_install:
            return math.nan
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        if i < 0 or i + 1 >= len(self.starts):
            return math.nan
        return float((t - self.starts[i]) / (self.starts[i + 1] - self.starts[i]))

    def _tool(self, f: float) -> np.ndarray:
        """Tool offset from the pedestal at active fraction f: home -> pick (dwell) -> place (dwell) -> home."""
        legs = [(0.00, 0.25, self.home, self.pick), (0.25, 0.35, self.pick, self.pick), (0.35, 0.70, self.pick, self.place),
                (0.70, 0.80, self.place, self.place), (0.80, 1.00, self.place, self.home)]
        for a, b, p0, p1 in legs:
            if f < b:
                u = (f - a) / (b - a)
                u = u * u * (3 - 2 * u)
                if p0 is self.pick and p1 is self.place:          # swing round the far side (arc, not a straight line)
                    a0, a1 = math.atan2(p0[0], p0[1]), math.atan2(p1[0], p1[1])
                    d = (a1 - a0) % (2 * math.pi)
                    ang = a0 + d * u
                    rr = np.hypot(*p0) * (1 - u) + np.hypot(*p1) * u
                    return np.array([rr * math.sin(ang), rr * math.cos(ang)])
                return p0 + (p1 - p0) * u
        return self.home.copy()

    def parts(self, t: float) -> np.ndarray:
        """(5, 3) circles (cx, cz, r) of the arm at t; empty before installation."""
        s, f = self.state(t)
        if s == "absent":
            return np.zeros((0, 3))
        tool = self._tool(f) if s in ("active", "commissioning") else self.home
        d = float(np.hypot(*tool))
        d = min(max(d, 0.3), 2 * self.L * 0.97)
        u = tool / max(np.hypot(*tool), 1e-9) * d
        half = d / 2
        h = math.sqrt(max(self.L * self.L - half * half, 0.0))
        nrm = np.array([u[1], -u[0]]) / max(d, 1e-9)
        elbow = u / 2 + nrm * h
        b = np.array([self.x, self.z])
        pts = [b, b + elbow / 2, b + elbow, b + (elbow + u) / 2, b + u]
        return np.array([[p[0], p[1], r] for p, r in zip(pts, R5)])
