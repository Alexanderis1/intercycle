"""Line-side extras: rendered workers that are NOT in the trace (build lead's rules, DIRECTION.md §7 X).

  * Only outside the guarding: line-side operators at the light-curtain openings (load points, with a walk to the
    part cart and back), an inspector at the S80 unload, a maintenance tech at a cell that is STOPPED / HELD /
    TRIPPED for the whole clip. Never inside a cell.
  * Per view: at most `cap` extras (2) and never more than the trace persons in that view (the minimum over the
    clip, so it holds on every frame).
  * GT: class person, entity_id "extra_<n>", extra=true (scene_real.assign_ids / render/cctv/gt.py).
  * Mostly standing / working at one spot, with a short walk and back.

Pure python (no bpy): scene_real passes the camera test and the trace, test_extras.py checks the rules.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

GUARD_GAP = 0.15        # m, guard panels stand this far outside the cell rectangle (scene_real._guards)
STAND_OFF = 0.75        # m, guard / light curtain -> the worker's centre
CART_STAND = 0.9        # m, part cart centre -> the operator picking from it (cart is 0.9 m deep)
WALK_V = 1.1            # m/s
TURN_S = 0.6            # s, turn on the spot after a walk
CLEAR_M = 1.2           # m, minimum distance to any trace mover during the clip
APART_M = 1.5           # m, minimum distance between two extras during the clip
DEPTH_M = (4.0, 30.0)   # camera distance of the work spot: closer is cut off, farther is a few pixels
HEAD_Z = 1.8
DOWN = frozenset({"STOPPED", "HELD", "TRIPPED"})
LOADED = ("weld_station", "framing", "respot")   # cells with an operator load point at each opening
MOVERS = ("person", "forklift", "agv", "tugger")


@dataclass(frozen=True)
class Extra:
    """Dwell at `a` (clip_a, facing face_a), walk to `b`, dwell (clip_b, facing face_b), walk back; loops."""
    role: str
    cell: str
    a: tuple[float, float]
    face_a: float
    b: tuple[float, float]
    face_b: float
    clip_a: str
    clip_b: str
    dwell_a: float
    dwell_b: float
    phase: float

    @property
    def walk_s(self) -> float:
        return math.dist(self.a, self.b) / WALK_V

    @property
    def period(self) -> float:
        return self.dwell_a + self.dwell_b + 2.0 * self.walk_s

    def state(self, t: float) -> tuple[float, float, float, str, float]:
        """(x, y, yaw rad, clip, speed m/s) at sim time t."""
        (ax, ay), (bx, by), w = self.a, self.b, self.walk_s
        go, back = math.atan2(by - ay, bx - ax), math.atan2(ay - by, ax - bx)
        u = (t + self.phase) % self.period
        if u < self.dwell_a:
            return ax, ay, _turn(back, self.face_a, u), self.clip_a, 0.0
        u -= self.dwell_a
        if u < w:
            k = u / w
            return ax + k * (bx - ax), ay + k * (by - ay), go, "walk", WALK_V
        u -= w
        if u < self.dwell_b:
            return bx, by, _turn(go, self.face_b, u), self.clip_b, 0.0
        k = (u - self.dwell_b) / w
        return bx + k * (ax - bx), by + k * (ay - by), back, "walk", WALK_V


def _turn(y0: float, y1: float, u: float) -> float:
    d = (y1 - y0 + math.pi) % (2.0 * math.pi) - math.pi
    return y0 + d * min(1.0, u / TURN_S)


def _phase(seed: int, *key) -> float:
    return random.Random(":".join(map(str, (seed, *key)))).uniform(0.0, 60.0)


def candidates(scn: dict, state_at: Callable[[str, float], str], times: Iterable[float], cart_y: float,
               seed: int = 0) -> list[Extra]:
    """Every spot an extra may use in [times]: operators at each opening of the loaded on-line cells, the S80
    inspector (south side), a maintenance tech at the north opening of each cell that is down on every sample."""
    times = list(times)
    out: list[Extra] = []
    for m in scn["machines"]:
        if not m.get("on_line"):
            continue
        mid, cx, cy, d = m["id"], m["x"], m["z"], m["size"][1]
        edge = {sg: cy + sg * (d / 2 + GUARD_GAP + STAND_OFF) for sg in (-1, 1)}
        face = {sg: -sg * math.pi / 2 for sg in (-1, 1)}            # towards the cell
        kind = m.get("kind")
        if kind in LOADED:
            out.append(Extra("operator", mid, (cx - 0.7, edge[-1]), face[-1], (cx, cart_y + CART_STAND), -math.pi / 2,
                             "work", "work", 16.0, 4.0, _phase(seed, mid, "S")))
            out.append(Extra("operator", mid, (cx + 0.7, edge[1]), face[1], (cx + 2.8, edge[1]), face[1],
                             "work", "idle", 16.0, 4.0, _phase(seed, mid, "N")))
            if times and all(state_at(mid, t) in DOWN for t in times):
                out.append(Extra("maintenance", mid, (cx + 0.3, edge[1]), face[1], (cx - 1.2, edge[1]), face[1],
                                 "work", "work", 12.0, 6.0, _phase(seed, mid, "M")))
        elif kind == "unload_lifter":
            out.append(Extra("inspector", mid, (cx + 1.0, edge[-1]), face[-1], (cx - 2.0, edge[-1]), face[-1],
                             "idle", "idle", 7.0, 7.0, _phase(seed, mid, "I")))
    return out


def select(cands: list[Extra], times: list[float], in_view: Callable[[tuple[float, float, float]], Optional[float]],
           persons_in_view: Callable[[float], int], movers_at: Callable[[float], list[tuple[float, float]]],
           cap: int = 2) -> list[Extra]:
    """The extras for one view: n = min(cap, fewest trace persons in view over `times`); a candidate qualifies when
    its work spot is in view (feet and head, DEPTH_M) and it keeps CLEAR_M from every trace mover on every sample.
    Maintenance first, then the nearest; one extra per cell, APART_M between extras (the caller names them extra_<n>).
    in_view(p) returns the camera distance of world point p, or None when p is outside the frame."""
    if not times or cap <= 0:
        return []
    n = min(cap, min(persons_in_view(t) for t in times))
    if n <= 0:
        return []
    movers = {t: movers_at(t) for t in times}
    ok = []
    for e in cands:
        feet, head = in_view((*e.a, 0.0)), in_view((*e.a, HEAD_Z))
        if feet is None or head is None or not DEPTH_M[0] <= feet <= DEPTH_M[1]:
            continue
        path = {t: e.state(t)[:2] for t in times}
        if any(math.dist(path[t], p) < CLEAR_M for t in times for p in movers[t]):
            continue
        ok.append((e.role != "maintenance", feet, e, path))
    ok.sort(key=lambda r: (r[0], r[1]))
    chosen: list[tuple[Extra, dict]] = []
    for _, _, e, path in ok:
        if len(chosen) == n:
            break
        if any(c.cell == e.cell or min(math.dist(path[t], cp[t]) for t in times) < APART_M for c, cp in chosen):
            continue
        chosen.append((e, path))
    return [e for e, _ in chosen]
