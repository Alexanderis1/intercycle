"""The floor as it actually behaves once a robot is in it.

Truth windows are written in *machine time*. Each clock group (the whole line, or one cell) can be
frozen by the robot: a HOLD (robot inside an idle zone when the next start is due) or a TRIP
(robot crosses into an active zone: protective stop, restart after `recovery_after_trip_s`).
While frozen, every zone in the group is idle and the group's machine time does not advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from intercycle.scenario import Machine, Scenario
from intercycle.truth.machines import all_windows, state_at
from intercycle.types import Window


@dataclass
class Freeze:
    t0: float
    t1: float
    kind: str           # "hold" | "trip"
    machine: str


@dataclass
class Group:
    name: str
    freezes: list[Freeze] = field(default_factory=list)

    def frozen_before(self, t: float) -> float:
        return sum(max(0.0, min(t, f.t1) - f.t0) for f in self.freezes if f.t0 < t)

    def is_frozen(self, t: float) -> bool:
        return any(f.t0 <= t < f.t1 for f in self.freezes)

    def mt(self, t: float) -> float:
        return t - self.frozen_before(t)

    def frozen_until(self, t: float) -> float | None:
        return max((f.t1 for f in self.freezes if f.t0 <= t < f.t1), default=None)


class World:
    def __init__(self, scn: Scenario, seed: int, t0: float, t1: float):
        self.scn = scn
        self.seed = seed
        self.W: dict[str, list[Window]] = all_windows(scn, seed, t0 - 3600, t1 + 3600)
        self.groups: dict[str, Group] = {}
        self.group_of: dict[str, str] = {}
        for m in scn.machines:
            g = "line" if m.on_line else m.id
            self.group_of[m.id] = g
            self.groups.setdefault(g, Group(g))
        self.events: list[dict] = []

    # ---- state queries (actual time) ----
    def group(self, mid: str) -> Group:
        return self.groups[self.group_of[mid]]

    def window(self, mid: str, t: float) -> Window | None:
        return state_at(self.W[mid], self.group(mid).mt(t))[1]

    def active(self, mid: str, t: float) -> bool:
        g = self.group(mid)
        if g.is_frozen(t):
            return False
        _, w = state_at(self.W[mid], g.mt(t))
        return w is not None and g.mt(t) < w.is_

    def active_many(self, mid: str, ts: np.ndarray) -> np.ndarray:
        """`active` for an array of actual times (what a ceiling camera's motion detector sees, before noise)."""
        ts = np.asarray(ts, float)
        g, ws = self.group(mid), self.W[mid]
        mt, frozen = ts.copy(), np.zeros(len(ts), bool)
        for f in g.freezes:
            mt -= np.clip(ts, f.t0, f.t1) - f.t0
            frozen |= (f.t0 <= ts) & (ts < f.t1)
        if not ws:
            return np.zeros(len(ts), bool)
        bs = np.array([w.bs for w in ws])
        is_ = np.array([w.is_ for w in ws])
        ie = np.array([w.ie for w in ws])
        k = np.searchsorted(bs, mt, side="right") - 1
        kk = np.maximum(k, 0)
        return (k >= 0) & (mt < ie[kk]) & (mt < is_[kk]) & ~frozen

    def true_idle_windows(self, mid: str, t: float, horizon: float = 600.0) -> list[tuple[float, float]]:
        """Idle windows from t on, in actual time, assuming no further freezes (what a PLC oracle knows).

        The first window starts at t if the zone is idle now.
        """
        g = self.group(mid)
        mt0 = g.mt(t)
        fu = g.frozen_until(t)
        shift = (fu - t) if fu is not None else 0.0
        conv = lambda m: t + shift + (m - mt0)
        out: list[tuple[float, float]] = []
        i, w = state_at(self.W[mid], mt0)
        if w is None:
            return out
        if fu is not None:
            if mt0 >= w.is_:
                out.append((t, conv(w.ie)))
            else:
                out.append((t, fu))
                out.append((conv(w.is_), conv(w.ie)))
        elif mt0 >= w.is_:
            out.append((t, conv(w.ie)))
        else:
            out.append((conv(w.is_), conv(w.ie)))
        for w2 in self.W[mid][i + 1:]:
            if w2.bs - mt0 > horizon:
                break
            out.append((conv(w2.is_), conv(w2.ie)))
        return out

    def idle_remaining(self, mid: str, t: float) -> float:
        ws = self.true_idle_windows(mid, t, 120)
        return ws[0][1] - t if ws and ws[0][0] <= t else 0.0

    def next_start_after(self, mid: str, t: float) -> float:
        return t + self.idle_remaining(mid, t)

    # ---- robot effects ----
    def trip(self, mid: str, t: float, who: str, where: str | None = None) -> dict:
        g = self.group(mid)
        rec = self.scn.line.recovery_after_trip_s
        g.freezes.append(Freeze(t, t + rec, "trip", mid))
        ev = {"t": t, "type": "trip", "machine": where or mid, "who": who, "lost_s": rec, "group": g.name}
        self.events.append(ev)
        return ev

    def cross(self, mid: str, t: float, inside_s: float, who: str) -> dict:
        """Robot crosses into zone `mid` at t and stays inside for inside_s. Returns what happened."""
        g = self.group(mid)
        if self.active(mid, t):
            return self.trip(mid, t, who)
        t_exit = t + inside_s
        start = self.next_start_after(mid, t)
        hold = 0.0
        if start < t_exit:
            hold = t_exit - start
            g.freezes.append(Freeze(start, t_exit, "hold", mid))
        ev = {"t": t, "type": "inspect", "machine": mid, "who": who, "hold_s": hold, "lost_s": hold, "group": g.name}
        self.events.append(ev)
        return ev

    def line_time_lost(self) -> float:
        return sum(e.get("lost_s", 0.0) for e in self.events)


def planned_stop_at(scn: Scenario, t: float) -> tuple[float, float] | None:
    tod, base = t % 86400, t - t % 86400
    for a, b, _ in scn.line_stops():
        if a <= tod < b:
            return base + a, base + b
    return None


def zone_contains(m: Machine, x: float, z: float) -> bool:
    x0, z0, x1, z1 = m.rect
    return x0 <= x <= x1 and z0 <= z <= z1
