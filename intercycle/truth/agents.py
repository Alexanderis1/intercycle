"""Ground-truth movers: AGVs, tugger trains, forklifts, people. Deterministic per seed, queryable at any t.

Each mover follows a polyline (loop or back-and-forth) at its own speed, with pauses at stops, and is
only present while its schedule says so. People flood the north aisle around break starts/ends
(walking to and from the break area at the west end), which is when aisle traffic peaks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.scenario import Scenario
from intercycle.types import MoverClass

DAY = 86_400.0


@dataclass
class PathAgent:
    id: str
    cls: MoverClass
    pts: list[tuple[float, float]]
    speed: float
    phase_s: float = 0.0
    loop: bool = True                  # False = back and forth
    pause_s: float = 0.0               # dwell at every vertex
    windows: list[tuple[float, float]] = field(default_factory=list)   # time-of-day presence; empty = always
    _seg: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        pts = self.pts if self.loop else self.pts + self.pts[-2::-1]
        self._pts = pts + ([pts[0]] if self.loop else [])
        self._seg = [math.dist(a, b) for a, b in zip(self._pts, self._pts[1:])]
        self._cycle = sum(s / self.speed + self.pause_s for s in self._seg)

    def present(self, t: float) -> bool:
        if not self.windows:
            return True
        tod = t % DAY
        return any(a <= tod < b for a, b in self.windows)

    def state(self, t: float) -> tuple[float, float, float, float] | None:
        """(x, z, heading, speed) at t if it never had to stop, or None if absent."""
        if not self.present(t):
            return None
        return self.at(t)

    def at(self, tau: float) -> tuple[float, float, float, float]:
        """Pose after `tau` seconds of unobstructed travel on its own clock (ignores presence)."""
        u = (tau + self.phase_s) % self._cycle
        for (a, b), s in zip(zip(self._pts, self._pts[1:]), self._seg):
            if u < self.pause_s:
                return a[0], a[1], math.atan2(b[0] - a[0], b[1] - a[1]), 0.0
            u -= self.pause_s
            d = s / self.speed
            if u < d:
                f = u / d
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, math.atan2(b[0] - a[0], b[1] - a[1]), self.speed
            u -= d
        a = self._pts[-1]
        return a[0], a[1], 0.0, 0.0


def agents(scn: Scenario, seed: int) -> list[PathAgent]:
    rng = np.random.default_rng([seed, 11])
    out: list[PathAgent] = []
    # AGVs: line-side delivery loop through the south aisle and the transport aisle
    loop = [(2, 16), (90, 16), (90, 8), (2, 8)]
    L = sum(math.dist(a, b) for a, b in zip(loop, loop[1:] + loop[:1]))
    for k in range(4):
        out.append(PathAgent(f"AGV{k}", MoverClass.AGV, loop, 1.2, phase_s=k * L / 4 / 1.2 + rng.uniform(0, 10), pause_s=4.0))
    # tugger trains on the transport aisle
    for k in range(2):
        out.append(PathAgent(f"TUG{k}", MoverClass.TUGGER, [(3, 8), (93, 8)], 1.8, phase_s=rng.uniform(0, 100), loop=False, pause_s=40.0))
    # forklifts: market -> line ends -> closures cells along the north aisle (crosses the robot's main route)
    for k, x in enumerate((2.0, 90.0, 52.5)):
        route = [(x, 3.0), (x, 32.0), (21.0 if k != 1 else 46.0, 32.0), (21.0 if k != 1 else 46.0, 35.5)]
        if k == 2:
            route = [(52.5, 3.0), (52.5, 10.0), (88.0, 10.0), (88.0, 3.0)]
        out.append(PathAgent(f"FL{k}", MoverClass.FORKLIFT, route, 1.6, phase_s=rng.uniform(0, 200), loop=False, pause_s=25.0))
    # people: walkers along the north and south aisles (shift-dependent presence)
    day = [(6 * 3600, 22 * 3600)]
    for k in range(10):
        z = 32.0 + rng.uniform(-1.2, 1.2) if k % 3 else 16.0 + rng.uniform(-1.2, 1.2)
        a, b = sorted(rng.uniform(3, 92, 2))
        if b - a < 15:
            b = min(94, a + 25)
        out.append(PathAgent(f"P{k}", MoverClass.PERSON, [(a, z), (b, z)], rng.uniform(0.9, 1.4), phase_s=rng.uniform(0, 300),
                             loop=False, pause_s=rng.uniform(5, 60), windows=day if k < 7 else []))
    # break crowds: from every station to the break area at the west end, then back
    for b in scn.breaks:
        for k in range(6):
            x0 = 12 + 9 * k
            dur = (x0 - 3) / 1.2
            out.append(PathAgent(f"BRK{int(b.t0)}_{k}", MoverClass.PERSON, [(x0, 31.0 + 0.4 * k % 2), (3.0, 31.5)], 1.2,
                                 phase_s=-b.t0, loop=False, pause_s=b.minutes * 60 - dur,
                                 windows=[(b.t0 - 5, b.t1 + dur + 5)]))
    # data-hall technicians
    for k in range(2):
        out.append(PathAgent(f"TECH{k}", MoverClass.PERSON, [(100.0, 20.0 + 8 * k), (116.0, 20.0 + 8 * k)], 0.8,
                             phase_s=rng.uniform(0, 100), loop=False, pause_s=90.0, windows=[(7 * 3600, 19 * 3600)]))
    return out


def snapshot(ag: list[PathAgent], t: float) -> list[tuple[str, MoverClass, float, float, float, float]]:
    out = []
    for a in ag:
        s = a.state(t)
        if s is not None:
            out.append((a.id, a.cls, *s))
    return out
