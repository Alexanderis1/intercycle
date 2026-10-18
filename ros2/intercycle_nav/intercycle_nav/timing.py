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
Per condition:  learned / oracle -> (a) and (b);  none -> (b) only (no map);  static -> never enter, inspect
from the wait pose at the boundary;  reactive (E-W4 baseline) -> (a) and (b) exactly as learned, with the windows from its
own camera (`reactive.py`); a machine with no window now (unseen, stale, moving or too little history) is planned as
reachable, so it goes to the boundary to wait and look (its windows never cover a future still run).
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
    wait_offset_m: float = 1.05            # wait pose: robot centre this far outside the boundary (1.0 m body, turns in place)
    enter_s: float | None = 5.0            # measured wait -> inside, 2.05 m (test double: 3.7-5.5 s, one 7.6 s)
    exit_s: float | None = 6.5             # inside -> wait with Nav2 BackUp at 0.35 m/s (7.0 s measured at 0.3); None = dist/speed
    inside: dict = field(default_factory=dict)
    _tg_of: dict = field(default_factory=dict)

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
            self._tg_of[n] = tg
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
            if condition == "reactive" and not w:
                # reactive: its windows only ever cover the CURRENT still run, never a future one, so "no window now"
                # (never seen, stale, moving, or too little history: None or []) is planned like an unseen machine:
                # the inside node is reachable, the robot goes to the wait pose, and the open-window check (a) and the
                # camera check (b) at the boundary (wait_for_entry) and the dwell cut decide whether it enters
                return None
            # the robot crosses the boundary right after leaving the wait node, so the zone must already be idle at
            # departure: arrival inside is allowed from window start + entry time
            return [] if w is None else [(float(a) + self.enter_time(self._tg_of[n]), float(b)) for a, b in w]

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


INIT_SPREAD_XY_M = 0.5                 # = intercycle_sim.loc_noise.START_DXY (R-W4 start-pose noise, uniform +-0.5 m)
INIT_SPREAD_YAW_DEG = 10.0             # = loc_noise.START_DPSI (uniform +-10 deg)


def initial_pose_covariance(spread_xy_m: float = INIT_SPREAD_XY_M,
                            spread_yaw_deg: float = INIT_SPREAD_YAW_DEG) -> list[float]:
    """Row-major 6x6 covariance (x, y, z, roll, pitch, yaw) for AMCL's /initialpose that matches the start-pose draw:
    a uniform error on [-a, a] has variance a^2 / 3 -> var_xy = 0.5^2/3 = 0.0833 m^2, var_yaw = (10 deg)^2/3 = 0.0102
    rad^2. nav2_simple_commander's setInitialPose sends a zero covariance, which lets AMCL start with every particle on
    the believed pose (and random injection is off in nav2.yaml), so it may never cover the true start (audit fix 4)."""
    if spread_xy_m < 0 or spread_yaw_deg < 0:
        raise ValueError("initial pose spread must be >= 0")
    c = [0.0] * 36
    c[0] = c[7] = float(spread_xy_m) ** 2 / 3.0
    c[35] = math.radians(float(spread_yaw_deg)) ** 2 / 3.0
    return c


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


# ------------------------------------------------------------------------------------------------ result records
def camera_stub_used(sources) -> bool:
    """True if any observation that fed the camera gate came from a test double (the cheat guard, DIRECTION 5.1)."""
    return any("stub" in s.lower() or "test double" in s.lower() for s in sources)


def interrupted_result(events: list[dict], camera_sources: dict, require_camera: bool, condition: str,
                       n_targets: int | None = None) -> dict:
    """goal.json when the planner is interrupted at the end of a fixed-length episode: the raw events plus the same
    cheat-guard fields and counts as a completed run (R3's runner checks every evaluation episode from these)."""
    done = [e for e in events if e.get("event") == "target_done"]
    n_insp = sum(1 for e in done if str(e.get("outcome", "")).startswith("inspected"))
    return {"result": "INTERRUPTED", "targets": events, "condition": condition, "n_inspected": n_insp,
            "n_done": len(done), "n_targets": n_targets, "camera_sources": dict(camera_sources),
            "camera_stub_used": camera_stub_used(camera_sources), "require_camera": bool(require_camera)}
