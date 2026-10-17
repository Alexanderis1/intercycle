"""Export a run as a replayable trace for the photoreal renderer and the dashboard.

Contract (trace.parquet), one row per entity per sample:
  t (sim s) · entity_id · cls · x · z · yaw (rad, 0 = +z) · state
cls: robot | machine | line | person | agv | tugger | forklift
state: robot -> move/wait/inside · machine -> active/idle/hold/trip · line -> seconds since last index
Background traffic (people, vehicles) is SYNTHETIC decoration for the render, generated per seed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from intercycle.sim.run import Runner

TRANSFER_S = 8.0


def _robot_track(r: Runner) -> list[tuple[float, float, float, str]]:
    """Robot key points, with the 1.5 m entry/exit into hazard zones expanded."""
    ins = r.scn.inspection
    pts = []
    by_node = {t.node: t for t in r.g.targets}
    for t, x, z, st in sorted(r.res.trace, key=lambda p: p[0]):
        pts.append((t, x, z, "wait" if st in ("depart", "dock") else "move"))
        if st == "enter":
            node = next((n for n, p in r.g.pos.items() if p == (x, z) and n in by_node), None)
            m = r.scn.machine(by_node[node].machine) if node else None
            if m is not None:
                dz = math.copysign(1.0, m.z - z)
                walk = ins.entry_m / ins.robot_speed_mps
                ix, iz = x, z + dz * ins.entry_m
                pts += [(t + walk, ix, iz, "inside"), (t + walk + ins.dwell_s, ix, iz, "inside"),
                        (t + 2 * walk + ins.dwell_s, x, z, "move")]
    pts.sort(key=lambda p: p[0])
    return pts


def _interp(pts, ts):
    tt = np.array([p[0] for p in pts])
    xs, zs = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
    x, z = np.interp(ts, tt, xs), np.interp(ts, tt, zs)
    i = np.clip(np.searchsorted(tt, ts, side="right") - 1, 0, len(pts) - 1)
    states = [pts[k][3] for k in i]
    moving = np.hypot(np.gradient(x), np.gradient(z)) > 1e-4
    states = [("move" if mv and s != "inside" else s if s == "inside" else "wait") for s, mv in zip(states, moving)]
    yaw = np.unwrap(np.arctan2(np.gradient(x), np.gradient(z)))
    for k in range(1, len(yaw)):           # hold heading while standing still
        if not moving[k]:
            yaw[k] = yaw[k - 1]
    return x, z, yaw, states


def _machine_state(r: Runner, mid: str, t: float) -> str:
    g = r.world.group(mid)
    for f in g.freezes:
        if f.t0 <= t < f.t1:
            return f.kind
    return "active" if r.world.active(mid, t) else "idle"


def _traffic(r: Runner, ts: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng([seed, 7])
    rows = []
    T = len(ts)
    # AGV loop on the south aisle and around the line ends
    loop = [(2, 16), (90, 16), (90, 8), (2, 8)]
    seg = [math.dist(a, b) for a, b in zip(loop, loop[1:] + loop[:1])]
    L = sum(seg)
    for k in range(4):
        s0 = k * L / 4
        for t in ts:
            d = (s0 + (t - ts[0]) * 1.2) % L
            for (a, b), sl in zip(zip(loop, loop[1:] + loop[:1]), seg):
                if d <= sl:
                    f = d / sl
                    rows.append({"t": t, "entity_id": f"AGV{k}", "cls": "agv", "x": a[0] + (b[0] - a[0]) * f,
                                 "z": a[1] + (b[1] - a[1]) * f, "yaw": math.atan2(b[0] - a[0], b[1] - a[1]), "state": "move"})
                    break
                d -= sl
    # tugger trains and forklifts on the transport aisle / market
    for k, (cls, z, v) in enumerate([("tugger", 8.0, 1.8), ("tugger", 8.0, 1.8), ("forklift", 3.0, 1.5), ("forklift", 2.0, 1.5), ("forklift", 4.0, 1.5)]):
        ph = rng.uniform(0, 180)
        for t in ts:
            u = ((t + ph) * v) % 180
            x = u if u < 90 else 180 - u
            rows.append({"t": t, "entity_id": f"{cls.upper()}{k}", "cls": cls, "x": 3 + x * 0.95, "z": z,
                         "yaw": math.pi / 2 if u < 90 else -math.pi / 2, "state": "move"})
    # operators at closures cells and walkers in the aisles; techs in the data hall
    people = [(21 + dx, 38.5) for dx in (-3, 3)] + [(46 + dx, 38.5) for dx in (-3, 3)]
    for k, (px, pz) in enumerate(people):
        for t in ts:
            rows.append({"t": t, "entity_id": f"OP{k}", "cls": "person", "x": px + 0.4 * math.sin(t / 7 + k),
                         "z": pz, "yaw": math.pi, "state": "work"})
    for k in range(8):
        a, b = rng.uniform(4, 90, 2)
        z = [14.8, 17.2, 4.2, 6.0][k % 4] if k % 2 else [30.6, 33.4][k % 2]
        sp = rng.uniform(0.9, 1.4)
        for t in ts:
            span = abs(b - a)
            u = ((t * sp) % (2 * span))
            x = min(a, b) + (u if u < span else 2 * span - u)
            rows.append({"t": t, "entity_id": f"P{k}", "cls": "person", "x": x, "z": z,
                         "yaw": math.pi / 2 if u < span else -math.pi / 2, "state": "walk"})
    for k in range(2):
        for t in ts:
            rows.append({"t": t, "entity_id": f"TECH{k}", "cls": "person", "x": 104 + 8 * k + math.sin(t / 15 + k),
                         "z": 20 + 8 * math.sin(t / 40 + 2 * k), "yaw": 0.0, "state": "work"})
    return rows


def export(r: Runner, ta: float, tb: float, hz: float = 5.0, traffic: bool = True) -> pd.DataFrame:
    ts = np.arange(ta, tb, 1.0 / hz)
    rows = []
    x, z, yaw, st = _interp(_robot_track(r), ts)
    rows += [{"t": t, "entity_id": "ROBOT", "cls": "robot", "x": a, "z": b, "yaw": c, "state": s}
             for t, a, b, c, s in zip(ts, x, z, yaw, st)]
    for m in r.scn.machines:
        rows += [{"t": t, "entity_id": m.id, "cls": "machine", "x": m.x, "z": m.z, "yaw": 0.0,
                  "state": _machine_state(r, m.id, t)} for t in ts]
    for t in ts:
        g = r.world.group("S10")
        w = r.world.window("S10", t)
        phase = (g.mt(t) - w.bs) if (w is not None and not g.is_frozen(t)) else -1.0
        rows.append({"t": t, "entity_id": "LINE", "cls": "line", "x": 0.0, "z": 24.0, "yaw": 0.0, "state": f"{phase:.2f}"})
    if traffic:
        rows += _traffic(r, ts, r.seed)
    df = pd.DataFrame(rows)
    df["t"] = df["t"].astype("float64")
    return df


def interesting_window(r: Runner, length_s: float = 180.0) -> tuple[float, float]:
    """A clip that shows the point: robot B waiting at a boundary, then entering as the window opens."""
    for i in r.res.inspections:
        if i["kind"] == "machine" and 8.0 < i["wait_s"] < 40.0 and i["hold_s"] == 0:
            return i["t"] - i["wait_s"] - 30.0, i["t"] - i["wait_s"] - 30.0 + length_s
    return r.t0 + 600, r.t0 + 600 + length_s
