"""Timing layer (pure logic, no ROS): WHEN to depart / wait and WHICH inspection target is next. Used by `timing_planner`.

Method: Safe Interval Path Planning (SIPP; Phillips & Likhachev, ICRA 2011), reused from `intercycle.robot.sipp`, on
the robot's provisioned topological site graph (`intercycle.robot.sitegraph`, public site map) extended with one
"inside" node per machine target (the inspection pose inside the hazard zone). Only inside nodes carry safe intervals:
the observer's predicted idle windows (from the verified broadcast, via `observer_bridge`). Edge durations are graph
distance / nominal speed; Nav2 computes and executes the real paths, MPPI + the collision monitor handle residual
dynamics. SIPP is the standard method; nothing here is claimed as novel.

Entry rule at a hazard boundary (DIRECTION 5.4, BRIEF): enter only if (a) a predicted idle window is open now with at
least `need_s` left (inside travel + dwell + exit + margin) and (b) the robot's own camera confirms "no motion" for the
target machine, `confirm_n` consecutive fresh observations spanning at least `local_check_latency_s`.
Per condition:  learned / oracle -> (a) and (b);  none -> (b) only (reactive, no map);  static -> never enter, inspect
from the wait pose at the boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from intercycle.robot import sipp
from intercycle.robot.sitegraph import SiteGraph, Target


@dataclass
class TargetPlan:
    target: Target
    wait_xy: tuple[float, float]           # plan / ROS xy (identical: x = plan x, y = plan z)
    inside_xy: tuple[float, float] | None
    face_psi: float                        # ROS heading facing the target
    t_arrive_wait: float
    t_depart_wait: float                   # SIPP: when to leave the wait node (>= arrival: wait there)
    t_goal: float                          # SIPP: entry time at the goal node
    expanded: int


@dataclass
class TimingCore:
    graph: SiteGraph
    speed: float = 0.5                     # nominal walking speed for edge durations (m/s)
    dwell_s: float = 4.0
    margin_s: float = 1.0
    horizon_s: float = 900.0
    wait_offset_m: float = 0.95            # wait pose: robot centre this far outside the boundary (1.0 m long body)
    enter_s: float | None = 4.5            # measured wait -> inside (Nav2, test double r2_fake1: 3.9 s for 1.7 m)
    exit_s: float | None = 6.5             # measured inside -> wait (Nav2 BackUp at 0.3 m/s); None = distance / speed
    inside: dict = field(default_factory=dict)

    @classmethod
    def from_scenario(cls, scn, **kw) -> TimingCore:
        core = cls(SiteGraph(scn), **kw)
        core._build()
        return core

    def _build(self) -> None:
        g = self.graph
        for tg in g.targets:
            if tg.kind != "machine":
                continue
            ix, iz = g.inspect_pose(tg)
            wx, wz = g.pos[tg.node]
            m = g.scn.machine(tg.machine)
            if abs(iz - wz) > 1e-6:                      # push the wait pose out along the entry line
                edge = m.rect[1] if iz > wz else m.rect[3]
                g.pos[tg.node] = (wx, edge - math.copysign(self.wait_offset_m, iz - wz))
            n = f"IN_{tg.machine}"
            g.pos[n] = (ix, iz)
            g.adj.setdefault(n, {})
            wx, wz = g.pos[tg.node]
            d = math.hypot(ix - wx, iz - wz) / self.speed
            g.adj[tg.node][n] = d
            g.adj[n][tg.node] = d
            self.inside[tg.machine] = n
        # re-time every edge at the nominal speed (the site graph is built at the scenario's 1 m/s); the wait -> inside
        # edge takes the measured entry time
        for a, nb in g.adj.items():
            for b in nb:
                (xa, za), (xb, zb) = g.pos[a], g.pos[b]
                nb[b] = math.hypot(xa - xb, za - zb) / self.speed
                if b.startswith("IN_") and self.enter_s is not None:
                    nb[b] = self.enter_s

    def inside_travel_s(self, tg: Target) -> float:
        n = self.inside.get(tg.machine)
        return 0.0 if n is None else self.graph.adj[tg.node][n]

    def enter_time(self, tg: Target) -> float:
        return self.enter_s if self.enter_s is not None else self.inside_travel_s(tg)

    def exit_time(self, tg: Target) -> float:
        return self.exit_s if self.exit_s is not None else self.inside_travel_s(tg)

    def need_s(self, tg: Target) -> float:
        """Idle time needed from the moment the robot leaves the wait pose: enter + dwell + exit + margin."""
        return self.enter_time(tg) + self.dwell_s + self.exit_time(tg) + self.margin_s

    def nearest_node(self, x: float, y: float) -> str:
        return min((n for n in self.graph.pos if not n.startswith("IN_")),
                   key=lambda n: math.hypot(self.graph.pos[n][0] - x, self.graph.pos[n][1] - y))

    def face_psi(self, tg: Target) -> float:
        wx, wz = self.graph.pos[tg.node]
        if tg.machine is not None:
            m = self.graph.scn.machine(tg.machine)
            return math.atan2(m.z - wz, m.x - wx)
        return 0.0

    def plan_target(self, tg: Target, t0: float, xy: tuple[float, float], windows: dict, condition: str) -> TargetPlan | None:
        """SIPP from the robot's nearest node to the target's goal node. Machine goal = inside node with the idle
        windows as safe intervals (learned/oracle; unknown windows -> no plan); none: inside node always "safe" (the
        camera decides at the boundary); static: the wait node (never enters)."""
        g = self.graph
        start = self.nearest_node(*xy)
        t_start = t0 + math.hypot(g.pos[start][0] - xy[0], g.pos[start][1] - xy[1]) / self.speed
        hazard = tg.kind == "machine" and condition != "static"
        goal = self.inside[tg.machine] if hazard else tg.node
        need = self.need_s(tg) - self.enter_time(tg) if hazard else self.dwell_s

        def safe(n: str):
            if not n.startswith("IN_"):
                return None
            if condition == "none":
                return None
            w = windows.get(n[3:])
            return [] if w is None else [(float(a), float(b)) for a, b in w]

        p = sipp.plan(g.adj, start, t_start, goal, safe, need_s=need, horizon_s=self.horizon_s)
        if p is None:
            return None
        if hazard and len(p.nodes) >= 2:
            k = len(p.nodes) - 2                         # the wait node just before the inside node
            t_arr, t_dep = p.arrive[k], p.depart[k]
        else:
            t_arr = t_dep = p.goal_time
        ins = g.pos[goal] if hazard else None
        return TargetPlan(tg, g.pos[tg.node], ins, self.face_psi(tg), t_arr, t_dep, p.goal_time, p.expanded)

    def choose(self, remaining: list[Target], t0: float, xy, windows: dict, condition: str) -> TargetPlan | None:
        """Next target = the one whose SIPP plan finishes (goal entry + dwell) earliest."""
        best = None
        for tg in remaining:
            tp = self.plan_target(tg, t0, xy, windows, condition)
            if tp is not None and (best is None or tp.t_goal < best.t_goal):
                best = tp
        return best


def window_open(windows, t: float, need_s: float) -> tuple[float, float] | None:
    """The predicted idle window containing t with at least need_s left, else None."""
    for a, b in windows or []:
        if a <= t and b - t >= need_s:
            return (a, b)
    return None


class CameraGate:
    """Robot camera 'no motion' confirmation per machine from `/intercycle/camera/observations` messages."""

    def __init__(self, confirm_n: int = 2, min_span_s: float = 0.3, fresh_s: float = 1.0):
        self.n, self.span, self.fresh = confirm_n, min_span_s, fresh_s
        self.hist: dict[str, list[tuple[float, bool]]] = {}

    def observe(self, t: float, target_id: str, visible: bool, motion_seen: bool) -> None:
        h = self.hist.setdefault(target_id, [])
        h.append((t, bool(visible) and not bool(motion_seen)))
        del h[:-32]

    def clear(self, target_id: str, t: float) -> bool:
        h = [e for e in self.hist.get(target_id, []) if t - e[0] <= self.fresh + self.span]
        if len(h) < self.n:
            return False
        tail = h[-self.n:]
        k = len(h)
        while k > 0 and h[k - 1][1]:                     # length of the trailing run of "clear" observations
            k -= 1
        run = h[k:]
        return all(ok for _, ok in tail) and t - tail[-1][0] <= self.fresh and len(run) >= self.n \
            and run[-1][0] - run[0][0] >= self.span - 1e-9
