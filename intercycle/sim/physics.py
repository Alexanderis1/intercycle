"""Time-stepped simulation (10 Hz) in which every entity has a collision body.

- Movers (people, AGVs, tugger trains, forklifts) follow their routes on their own clocks and stop
  for anything in their forward field, including the robot.
- The robot drives with a diff-drive model and a safety-scanner style protective field (stop) and
  warning field (slow down). It senses only through its scanner, its own camera (machine motion at the
  boundary it faces) and the BLE broadcasts. It never reads ground truth.
- Hazard zones are geometric: any robot circle inside an ACTIVE zone trips the safety system; a robot
  inside an IDLE zone inhibits the next start (hold). The line walkway is a hazard zone during transfers.
- Every overlap between bodies is logged as a collision incident. Nothing is silently resolved.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.codec.layout import quantize_fast, quantize_model
from intercycle.robot import policies as P
from intercycle.robot.predict import idle_windows
from intercycle.robot.sipp import plan as sipp_plan
from intercycle.robot.sitegraph import CROSS_X, TRANSFER_S, SiteGraph
from intercycle.scenario import Scenario, load
from intercycle.sim import geometry as G
from intercycle.sim.movers import MoverSystem
from intercycle.sim.observer_shim import ShimObserver
from intercycle.sim.world import Freeze, World, planned_stop_at
from intercycle.truth.kinematics import Kinematics
from intercycle.truth.agents import agents as make_agents
from intercycle.types import DAY0_UNIX, MoverClass, State

DT = 0.1
CLS = {MoverClass.PERSON: "person", MoverClass.AGV: "agv", MoverClass.TUGGER: "tugger", MoverClass.FORKLIFT: "forklift"}
# forward fields: (front offset from body centre, stop distance, slow distance, half width)
FIELD = {"robot": (0.53, 0.6, 2.0, 0.5), "person": (0.3, 0.5, 1.0, 0.35), "agv": (0.8, 1.2, 2.5, 0.6),
         "tugger": (0.7, 1.5, 3.0, 0.6), "forklift": (2.05, 1.5, 3.0, 0.65)}
VMAX, WMAX = 1.0, 1.2
WALK = "WALK"


def field_circles(cls: str, x: float, z: float, yaw: float, which: str) -> np.ndarray:
    front, stop, slow, hw = FIELD[cls]
    reach = stop if which == "stop" else slow
    ds = np.arange(front + hw * 0.5, front + reach + 1e-6, max(0.2, hw * 0.8))
    s, c = math.sin(yaw), math.cos(yaw)
    return np.stack([x + s * ds, z + c * ds, np.full_like(ds, hw)], axis=1)


def circles_gap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.inf
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) - a[:, None, 2] - b[None, :, 2]
    return float(d.min())


def circles_in_rect(c: np.ndarray, rect) -> bool:
    x0, z0, x1, z1 = rect
    nx = np.clip(c[:, 0], x0, x1)
    nz = np.clip(c[:, 1], z0, z1)
    return bool((np.hypot(c[:, 0] - nx, c[:, 1] - nz) < c[:, 2]).any())


@dataclass
class MoverSim:
    agent: object
    cls: str
    tau: float
    pose: tuple = (0.0, 0.0, 0.0, 0.0)
    present: bool = False
    blocked_s: float = 0.0
    body: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def update_body(self):
        x, z, yaw, _ = self.pose
        self.body = G.footprint(self.cls, x, z, yaw) if self.present else np.zeros((0, 3))


@dataclass
class RobotSim:
    x: float
    z: float
    yaw: float = math.pi / 2
    v: float = 0.0
    wps: list = field(default_factory=list)
    stop_since: float | None = None
    stops: int = 0
    stop_s: float = 0.0
    block_t: float | None = None          # when progress stalled (reset only after moving 0.5 m)
    block_xy: tuple = (0.0, 0.0)
    body: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def update_body(self):
        self.body = G.footprint("robot", self.x, self.z, self.yaw)


class PhysicsRunner:
    def __init__(self, scn: Scenario, strategy: str, seed: int, t0: float, t1: float, fault: str | None = None,
                 record_hz: float = 0.0):
        self.scn, self.strategy, self.seed, self.t0, self.t1, self.fault = scn, strategy, seed, t0, t1, fault
        self.world = World(scn, seed, t0, t1)
        self.sm = G.StaticMap(scn)
        self.g = SiteGraph(scn)
        self.obs = ShimObserver(scn, self.world, seed, fault=fault if fault in ("phase", "flip") else None)
        self.slow_model = [quantize_model(m) for m in self.obs.models]
        self.idx = {m.id: i for i, m in enumerate(scn.machines)}
        self.ins = scn.inspection
        self.need = self.ins.inside_s
        self.rng = np.random.default_rng([seed, 31])
        self.movers = MoverSystem(make_agents(scn, seed), self.sm, t0)
        self._act_cache: tuple[float, dict] = (-1.0, {})
        self.kin = Kinematics(scn, self.world)
        self.allowed_zone: str | None = None        # the one hazard zone the robot may enter (its inspection target)
        self.robot = RobotSim(*scn.robot["dock"])
        self.zones = {m.id: m.rect for m in scn.machines}
        self.zones[WALK] = (CROSS_X - 1.0, 22.8, CROSS_X + 1.0, 25.2)
        self.inside: dict[str, bool] = {k: False for k in self.zones}
        from intercycle.truth.kinematics import DOCK_DX, DOCK_Z
        self.bays = [(m.x + DOCK_DX - 0.8, DOCK_Z - 0.5, m.x + DOCK_DX + 0.8, DOCK_Z + 0.5) for m in scn.machines if m.on_line]
        self.entered_idle: dict[str, bool] = {k: False for k in self.zones}
        self.arm_bases = [(m.id, m.x + dx, (24 + dz * 3.3) if m.on_line else (m.z + dz * 2.0))
                          for m in scn.machines if m.kind != "unload_lifter" for dx, dz in ((-1.8, -1), (1.8, 1))]
        self.events: list[dict] = []
        self.inspections: list[dict] = []
        self.collisions: list[dict] = []
        self.mover_collisions = 0
        self.record_hz = record_hz
        self.trace: list[tuple] = []
        self._fast_cache: dict[float, object] = {}
        self._live_hist: list[tuple[float, list]] = []
        self.t = t0

    # ------------------------------------------------------------------ truth-side physics
    def _active(self, zid: str, t: float) -> bool:
        if zid == WALK:
            g = self.world.group("S10")
            if g.is_frozen(t):
                return False
            w = self.world.window("S10", t)
            return w is not None and g.mt(t) - w.bs < TRANSFER_S
        return self.world.active(zid, t)

    def _machine_active(self, t: float) -> dict[str, bool]:
        if self._act_cache[0] != t:
            self._act_cache = (t, {m.id: self.world.active(m.id, t) for m in self.scn.machines})
        return self._act_cache[1]

    def _arm_circles(self, t: float) -> np.ndarray:
        return self.kin.all_circles(t)

    def _step_movers(self, t: float, dt: float):
        self.movers.step(t, dt, [self.robot.body], self._arm_circles(t))

    def _dyn_circles_near(self, x: float, z: float, r: float = 12.0) -> np.ndarray:
        mv = self.movers.all_bodies()
        arms = self._arm_circles(self.t)
        allc = np.vstack([mv, arms]) if len(mv) else arms
        return allc[(np.abs(allc[:, 0] - x) < r) & (np.abs(allc[:, 1] - z) < r)]

    def _zone_physics(self, t: float):
        rb = self.robot.body
        for zid, rect in self.zones.items():
            x0, z0, x1, z1 = rect
            if not (x0 - 2 < self.robot.x < x1 + 2 and z0 - 2 < self.robot.z < z1 + 2):
                if self.inside[zid]:
                    self.inside[zid] = False
                continue
            ins = circles_in_rect(rb, rect)
            gid = "S10" if zid == WALK else zid
            if ins and not self.inside[zid]:
                if self._active(zid, t):
                    ev = self.world.trip(gid, t, self.strategy, where=zid)
                    self.events.append(ev)
            if ins and self._active(zid, t):          # machine wants to run with the robot inside -> hold
                g = self.world.group(gid)
                last = g.freezes[-1] if g.freezes else None
                if last and last.kind == "hold" and last.t1 >= t - DT - 1e-6:
                    last.t1 = t + DT
                else:
                    g.freezes.append(Freeze(t, t + DT, "hold", zid))
                    self.events.append({"t": t, "type": "hold", "machine": zid, "who": self.strategy, "group": g.name})
            self.inside[zid] = ins

    def _collision_check(self, t: float):
        rb = self.robot.body
        if G.static_hit(self.sm, rb):
            self.collisions.append({"t": t, "with": "static", "x": self.robot.x, "z": self.robot.z})
        dyn = self._dyn_circles_near(self.robot.x, self.robot.z, 4.0)
        if len(dyn) and circles_gap(rb, dyn) < -0.02:
            self.collisions.append({"t": t, "with": "dynamic", "x": self.robot.x, "z": self.robot.z})

    # ------------------------------------------------------------------ robot sensing (no truth)
    def keepout_hit(self, circles: np.ndarray) -> bool:
        """Hazard zones (and the walkway) are keep-out areas in the robot's map, like a Nav2 keepout filter."""
        for zid, rect in self.zones.items():
            if zid == self.allowed_zone or (zid == WALK and self.allowed_zone == WALK):
                continue
            if circles_in_rect(circles, rect):
                return True
        return any(circles_in_rect(circles, b) for b in self.bays)

    def scanner_gap(self, which: str) -> float:
        r = self.robot
        fc = field_circles("robot", r.x, r.z, r.yaw, which)
        gs = float((self.sm.clearance(fc[:, :2]) - fc[:, 2]).min())
        gd = circles_gap(fc, self._dyn_circles_near(r.x, r.z, 5.0))
        return min(gs, gd)

    def sees_motion(self, zid: str):
        lat = self.ins.local_check_latency_s
        return lambda t: self._active(zid, t - lat)

    def _issue_time(self, t: float) -> float:
        ti = math.floor((t - 0.08) / 2.0) * 2.0
        if self.fault == "freeze":
            ti = min(ti, self.t0 + 1800.0)
        if self.fault == "replay":
            ti -= 600.0
        return ti

    def knowledge(self, mid: str):
        i = self.idx[mid]

        def k(t: float):
            if self.fault == "forge":
                return None, None, None
            ti = self._issue_time(t)
            if ti not in self._fast_cache:
                self._fast_cache[ti] = quantize_fast(self.obs.fast(ti))
                if len(self._fast_cache) > 64:
                    self._fast_cache.pop(next(iter(self._fast_cache)))
            f = self._fast_cache[ti]
            return self.slow_model[i], f.machines[i], f.issue_time
        return k

    def live_movers(self, t: float) -> list[tuple[float, float, float, float]]:
        """Live layer as the robot decodes it: tracked movers 0.3 s old, with tracking noise."""
        if self.fault == "forge" or not self._live_hist:
            return []
        lag = [h for h in self._live_hist if h[0] <= t - 0.3]
        return lag[-1][1] if lag else []

    # ------------------------------------------------------------------ planning
    def _mover_intervals(self, t: float, horizon: float = 15.0):
        movers = self.live_movers(t)
        if not movers:
            return lambda n: None
        mv = np.array(movers)
        taus = np.arange(0.0, horizon, 0.5)
        px = mv[:, 0, None] + np.sin(mv[:, 2, None]) * mv[:, 3, None] * taus[None, :]
        pz = mv[:, 1, None] + np.cos(mv[:, 2, None]) * mv[:, 3, None] * taus[None, :]
        cache = {}

        def unsafe(n: str):
            if n in cache:
                return cache[n]
            x, z = self.g.pos[n]
            hit = (np.hypot(px - x, pz - z) < 1.8).any(axis=0)
            if not hit.any():
                cache[n] = None
                return None
            ivs, a = [], -math.inf
            k = 0
            while k < len(taus):
                if hit[k]:
                    b0 = k
                    while k < len(taus) and hit[k]:
                        k += 1
                    ivs.append((a, t + taus[b0] - 0.5))
                    a = t + (taus[k - 1] if k - 1 < len(taus) else horizon) + 1.0
                else:
                    k += 1
            ivs.append((a, math.inf))
            ivs = [(s, e) for s, e in ivs if e > s]
            cache[n] = ivs
            return ivs
        return unsafe

    def _windows(self, mid: str, t: float):
        if self.strategy == "C":
            return self.world.true_idle_windows(mid, t, 900)
        model, now, issue = self.knowledge(mid)(t)
        return idle_windows(model, now, issue, t, 900) if (model and now and issue) else None

    def plan_to(self, t: float, start: str, target) -> tuple[list[str], float]:
        """Returns (node path, planned entry/arrival time)."""
        if self.strategy in ("A0", "A") or target is None:
            return self._dijkstra(start, target.node), math.nan
        movers = self._mover_intervals(t)
        need = self.need + (self.ins.safety_margin_s if self.strategy == "B" else 0.0)
        wins = None
        if target.kind == "machine":
            w = self._windows(target.machine, t)
            wins = [(a, b) for a, b in w if b - a >= need] if w else None

        def safe(n):
            if n == target.node and wins:
                return wins
            return movers(n)
        p = sipp_plan(self.g.adj, start, t, target.node, safe, need_s=need if wins else 0.0, horizon_s=900)
        if p is None:
            return self._dijkstra(start, target.node), math.nan
        return p.nodes, p.goal_time

    def _dijkstra(self, a: str, b: str) -> list[str]:
        adj = self.g.adj
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            for v, w in adj[u].items():
                if d + w < dist.get(v, math.inf):
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(pq, (d + w, v))
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1]

    # ------------------------------------------------------------------ main loop
    _straight_mode = False
    _V = np.array([-0.3, 0.0, 0.3, 0.6, 1.0])
    _W = np.array([-1.2, -0.6, -0.25, 0.0, 0.25, 0.6, 1.2])

    def _drive(self, t: float, dt: float) -> bool:
        """Sampling local planner (DWB-like): roll out (v, w) pairs for 1.5 s, keep the ones that stay clear
        of walls, bodies and keep-out zones, pick the best by progress, heading and clearance.
        Protective stop if nothing is safe. Returns True when the waypoint list is empty."""
        r = self.robot
        if not r.wps:
            r.v = 0.0
            return True
        tx, tz = r.wps[0]
        d = math.hypot(tx - r.x, tz - r.z)
        final = len(r.wps) == 1
        if not final and d < 2.5:                       # intermediate waypoint occupied: skip it if the shortcut is clear
            dyn = self._dyn_circles_near(tx, tz, 1.5)
            if (len(dyn) and (np.hypot(dyn[:, 0] - tx, dyn[:, 1] - tz) - dyn[:, 2]).min() < 0.6
                    and self._segment_clear((r.x, r.z), r.wps[1])):
                r.wps.pop(0)
                return False
        if d < (0.12 if final else 0.35) or (not final and d < 0.8 and self._passed(r, tx, tz)):
            r.wps.pop(0)
            return not r.wps
        straight = self._straight_mode
        if d < 0.7 or straight:                         # precise approach / inspection move: face the point, then go
            err = (math.atan2(tx - r.x, tz - r.z) - r.yaw + math.pi) % (2 * math.pi) - math.pi
            reverse = straight and abs(err) > 2.4             # target behind: back out, keep facing the machine
            if reverse:
                err = err - math.copysign(math.pi, err)
            w = max(-WMAX, min(WMAX, 2.5 * err))
            v = 0.0 if abs(err) > 0.35 else (min(VMAX, 0.15 + 1.6 * d) if straight else min(0.3, d))
            if reverse:
                v = -min(0.8, v)
            nyaw = (r.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
            nx, nz = r.x + v * math.sin(nyaw) * dt, r.z + v * math.cos(nyaw) * dt
            fp = G.footprint("robot", nx, nz, nyaw)
            dyn = self._dyn_circles_near(r.x, r.z, 3.0)
            ok = (G.static_gap(self.sm, fp) > 0.03 and (not len(dyn) or circles_gap(fp, dyn) > 0.1)
                  and not (self.keepout_hit(fp) and not self.keepout_hit(r.body)))
            if ok:
                r.yaw, r.x, r.z, r.v = nyaw, nx, nz, v
                r.stop_since = None
            else:
                self.movers.ask_yield(field_circles("robot", r.x, r.z, r.yaw, "slow"), r.x, r.z, r.yaw, t)
                if r.stop_since is None:
                    r.stop_since, r.stops = t, r.stops + 1
                r.stop_s += dt
                r.v = 0.0
            return False
        V, W = np.meshgrid(self._V, self._W)
        V, W = V.ravel(), W.ravel()
        steps = np.arange(1, 6) * 0.3
        yaw = r.yaw + W[:, None] * steps[None, :]
        xs = r.x + np.cumsum(V[:, None] * np.sin(yaw) * 0.3, axis=1)
        zs = r.z + np.cumsum(V[:, None] * np.cos(yaw) * 0.3, axis=1)
        pts = np.stack([xs.ravel(), zs.ravel()], axis=1)
        stat = self.sm.clearance(pts).reshape(xs.shape) - 0.52
        dyn = self._dyn_circles_near(r.x, r.z, 4.5)
        if len(dyn):
            dd = (np.hypot(pts[:, None, 0] - dyn[None, :, 0], pts[:, None, 1] - dyn[None, :, 1]) - dyn[None, :, 2]).min(axis=1)
            dynm = dd.reshape(xs.shape) - 0.52
        else:
            dynm = np.full(xs.shape, 9.0)
        clear = np.minimum(stat, dynm)
        ko = np.zeros(len(V), bool)
        rects = [rc for zid, rc in self.zones.items() if zid != self.allowed_zone] + self.bays
        for (x0, z0, x1, z1) in rects:
            if x0 - 3 > r.x or r.x > x1 + 3 or z0 - 3 > r.z or r.z > z1 + 3:
                continue
            dist = np.hypot(np.maximum(np.maximum(x0 - xs, xs - x1), 0), np.maximum(np.maximum(z0 - zs, zs - z1), 0))
            now = math.hypot(max(x0 - r.x, r.x - x1, 0), max(z0 - r.z, r.z - z1, 0))
            ko |= ((dist < 0.5) & (dist < now - 0.01)).any(axis=1)      # never get deeper into a keep-out margin
        safe = (clear.min(axis=1) > 0.08) & ~ko
        dist_end = np.hypot(tx - xs[:, -1], tz - zs[:, -1])
        head_err = np.abs((np.arctan2(tx - xs[:, -1], tz - zs[:, -1]) - yaw[:, -1] + np.pi) % (2 * np.pi) - np.pi)
        score = (-dist_end * 2.0 - head_err * 0.6 + np.minimum(clear.min(axis=1), 1.0) * 0.8 - (V == 0) * 0.5
                 - (V < 0) * 1.5)                          # reverse only to get out of a dead end
        if d < 0.8:
            score -= np.abs(V - min(0.3, d)) * 2.0
        if not safe.any() or clear[:, 0].max() < 0.6:
            self.movers.ask_yield(field_circles("robot", r.x, r.z, r.yaw, "slow"), r.x, r.z, r.yaw, t)
        if not safe.any():
            v, w = 0.0, 0.0
        else:
            k = int(np.argmax(np.where(safe, score, -1e9)))
            v, w = float(V[k]), float(W[k])
        if v == 0.0 and (not safe.any() or w == 0.0):
            if r.stop_since is None:
                r.stop_since, r.stops = t, r.stops + 1
            r.stop_s += dt
            if t - r.stop_since > 4.0:                   # blocked: overtake through the free lane if there is one
                self._sidestep()
                r.stop_since = t
        else:
            r.stop_since = None
        nyaw = (r.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        nx, nz = r.x + v * math.sin(nyaw) * dt, r.z + v * math.cos(nyaw) * dt
        if self.keepout_hit(G.footprint("robot", nx, nz, nyaw)) and not self.keepout_hit(r.body):
            nx, nz, v = r.x, r.z, 0.0
        r.yaw, r.x, r.z, r.v = nyaw, nx, nz, v
        return False

    def _segment_clear(self, a, b) -> bool:
        """Straight drive from a to b stays off walls and out of keep-out areas (robot radius 0.5)."""
        n = max(2, int(math.dist(a, b) / 0.25))
        xs = np.linspace(a[0], b[0], n)
        zs = np.linspace(a[1], b[1], n)
        if (self.sm.clearance(np.stack([xs, zs], axis=1)) < 0.55).any():
            return False
        rects = [rc for zid, rc in self.zones.items() if zid != self.allowed_zone] + self.bays
        for (x0, z0, x1, z1) in rects:
            dist = np.hypot(np.maximum(np.maximum(x0 - xs, xs - x1), 0), np.maximum(np.maximum(z0 - zs, zs - z1), 0))
            if (dist < 0.5).any():
                return False
        return True

    def _replan_from_here(self, goal_node: str) -> None:
        """Recovery: route again from the nearest reachable graph node."""
        r = self.robot
        cand = sorted(self.g.pos, key=lambda n: math.dist(self.g.pos[n], (r.x, r.z)))
        for n in cand[:6]:
            if self._segment_clear((r.x, r.z), self.g.pos[n]):
                path = self._dijkstra(n, goal_node)
                r.wps = [self.g.pos[n]] + self._lane_waypoints(path)
                return

    @staticmethod
    def _passed(r, tx, tz) -> bool:
        return (tx - r.x) * math.sin(r.yaw) + (tz - r.z) * math.cos(r.yaw) < 0

    def _sidestep(self):
        r = self.robot
        s, c = math.sin(r.yaw), math.cos(r.yaw)
        for side in (1.0, -1.0):
            px, pz = r.x + side * c * 1.3, r.z - side * s * 1.3
            ahead = G.footprint("robot", px + s * 2.5, pz + c * 2.5, r.yaw)
            dyn = self._dyn_circles_near(px, pz, 4.0)
            spot = G.footprint("robot", px, pz, r.yaw)
            spot_m = spot + np.array([0.0, 0.0, 0.3])          # keep a margin from keep-out areas
            if (self.sm.free(px, pz, 0.6) and not self.keepout_hit(spot_m) and not self.keepout_hit(ahead)
                    and (not len(dyn) or (circles_gap(spot, dyn) > 0.3 and circles_gap(ahead, dyn) > 0.3))):
                r.wps = [(px, pz), (px + s * 2.5, pz + c * 2.5)] + r.wps
                return

    def run(self) -> dict:
        r, t = self.robot, self.t0
        node, mission = "DOCK", self.g.mission()
        tour_done: set[str] = set()
        deferred: list = []
        k_fixed = 0
        phase, target, path, entry_t, ctx, unknown_since = "choose", None, [], math.nan, None, None
        t_arrive, hold0 = 0.0, 0
        next_rec = t
        while t < self.t1:
            self._step_movers(t, DT)
            if math.floor(t * 2) != math.floor((t - DT) * 2):       # live layer issue at 2 Hz
                snap = [(x + self.rng.normal(0, 0.15), z + self.rng.normal(0, 0.15), yaw, sp)
                        for _, _, x, z, yaw, sp in self.movers.snapshot() if self.rng.random() > 0.02]
                self._live_hist.append((t, snap))
                if len(self._live_hist) > 8:
                    self._live_hist.pop(0)
            r.update_body()
            # ---------------- behaviour state machine
            if phase == "choose":
                in_break = planned_stop_at(self.scn, t) is not None
                if deferred and in_break:
                    target = deferred.pop(0)
                elif self.strategy in ("B", "C"):
                    todo = [x for x in mission if x.id not in tour_done]
                    if not todo:
                        tour_done.clear()
                        todo = list(mission)
                    best = None
                    for cand in todo:
                        if cand.kind == "machine" and not in_break:
                            mdl = self.slow_model[self.idx[cand.machine]] if self.strategy == "B" else self.obs.models[self.idx[cand.machine]]
                            if not P.fits_ever(mdl, self.need + (self.ins.safety_margin_s if self.strategy == "B" else 0.0)):
                                if cand not in deferred:
                                    deferred.append(cand)
                                tour_done.add(cand.id)
                                continue
                        pth, gt = self.plan_to(t, node, cand)
                        travel = sum(self.g.adj[a][b] for a, b in zip(pth, pth[1:]))
                        score = (gt if not math.isnan(gt) else t + travel) + (self.need if cand.kind == "machine" else cand.dwell_s)
                        if best is None or score < best[0]:
                            best = (score, cand, pth, gt)
                    if best is None:
                        t += DT
                        continue
                    _, target, path, entry_t = best
                else:
                    target = mission[k_fixed % len(mission)]
                    k_fixed += 1
                    path, entry_t = self.plan_to(t, node, target)
                if self.strategy in ("A0", "A"):
                    path, entry_t = self.plan_to(t, node, target)
                r.wps = self._lane_waypoints(path)
                phase, t_arrive = "travel", None
            elif phase == "travel":
                nxt = self._next_node(path, r)
                if self.allowed_zone == WALK and nxt != "X" and not self.inside.get(WALK):
                    self.allowed_zone = None
                if nxt == "X" and not self._walkway_ok(t):
                    r.v = 0.0
                    self.allowed_zone = None
                elif nxt == "X" or self.inside.get(WALK):
                    self.allowed_zone = WALK
                    self._drive(t, DT)
                if r.block_t is None or math.dist((r.x, r.z), r.block_xy) > 0.5:
                    r.block_t, r.block_xy = t, (r.x, r.z)
                elif t - r.block_t > 20.0:              # no progress for 20 s: re-route (recovery behaviour)
                    self._replan_from_here(target.node)
                    r.block_t = t
                elif self._drive(t, DT):
                    node, t_arrive = target.node, t
                    if target.kind == "rack":
                        phase, dwell_end = "rack", t + target.dwell_s
                    else:
                        phase, ctx, unknown_since, next_check = "boundary", self._ctx(target.machine), None, t
            elif phase == "rack":
                if t >= dwell_end:
                    self.inspections.append({"t": t, "target": target.id, "kind": "rack", "wait_s": 0.0, "hold_s": 0.0})
                    tour_done.add(target.id)
                    phase = "choose"
            elif phase == "boundary":
                ix, iz = self.g.inspect_pose(target)
                face = math.atan2(ix - r.x, iz - r.z)
                err = (face - r.yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(err) > 0.02:
                    r.yaw = (r.yaw + max(-WMAX * DT, min(WMAX * DT, err)) + math.pi) % (2 * math.pi) - math.pi
                if t >= next_check:
                    if self.strategy == "B":
                        unknown_since = unknown_since if unknown_since is not None else t
                        tc, note, nxt_t = P.b_intercycle(ctx, t, waited_unknown=t - unknown_since)
                        if note != "prediction unavailable":
                            unknown_since = None
                    else:
                        fn = {"A0": P.a0_shutdown_only, "A": P.a_reactive, "C": P.c_oracle}[self.strategy]
                        tc, note, nxt_t = fn(ctx, t)
                    if tc is not None:
                        ix, iz = self.g.inspect_pose(target)
                        r.wps = [(ix, iz)]
                        self.allowed_zone = target.machine
                        self._straight_mode = True
                        phase, note_in, t_enter = "enter", note, t
                        hold0 = sum(f.t1 - f.t0 for f in self.world.group(target.machine).freezes if f.kind == "hold")
                        trips0 = sum(1 for e in self.events if e["type"] == "trip")
                    else:
                        next_check = max(nxt_t, t + DT)
                    if t - t_arrive > 3600:
                        phase = "choose"
            elif phase == "enter":
                if self._drive(t, DT):
                    phase, dwell_end = "dwell", t + self.ins.dwell_s
            elif phase == "dwell":
                if t >= dwell_end:
                    r.wps = [self.g.pos[target.node]]
                    phase = "exit"
            elif phase == "exit":
                if self._drive(t, DT) and not any(self.inside.values()):
                    hold = sum(f.t1 - f.t0 for f in self.world.group(target.machine).freezes if f.kind == "hold") - hold0
                    trips = sum(1 for e in self.events if e["type"] == "trip") - trips0
                    self.inspections.append({"t": t_enter, "target": target.id, "kind": "machine", "wait_s": t_enter - t_arrive,
                                             "hold_s": hold, "trips": trips, "note": note_in})
                    tour_done.add(target.id)
                    self.allowed_zone = None
                    self._straight_mode = False
                    phase = "choose"
            # ---------------- physics consequences
            r.update_body()
            self._zone_physics(t)
            self._collision_check(t)
            if self.record_hz and t >= next_rec:
                self.trace.append((t, r.x, r.z, r.yaw, phase, [(i, c, x, z, y) for i, c, x, z, y, _ in self.movers.snapshot()]))
                next_rec = t + 1.0 / self.record_hz
            t += DT
            self.t = t
        return self.metrics()

    def _lane_waypoints(self, path: list[str]) -> list[tuple[float, float]]:
        """Drive on the right of long aisle segments (0.8 m), like the vehicles; nodes elsewhere as-is."""
        aisle = lambda n: (n[0] in "NS" and n[1:2].isdigit())
        out = []
        for k, (a, b) in enumerate(zip(path, path[1:])):
            (xa, za), (xb, zb) = self.g.pos[a], self.g.pos[b]
            if aisle(a) and aisle(b) and abs(za - zb) < 1e-6 and abs(xb - xa) > 2.5:
                yaw = math.atan2(xb - xa, zb - za)
                ox, oz = 0.35 * math.cos(yaw), -0.35 * math.sin(yaw)   # keep right, clear of the AGV bays at the aisle edge
                turning_off = k + 2 < len(path) and not aisle(path[k + 2])
                ex = xb - 2.0 * math.sin(yaw) if turning_off else xb       # cut the corner before leaving the aisle
                if turning_off and not self._segment_clear((ex + ox, zb + oz), self.g.pos[path[k + 2]]):
                    ex = xb                                                  # the shortcut would clip a keep-out area
                out += [(xa + ox, za + oz), (ex + ox, zb + oz)]
            else:
                out.append((xb, zb))
        return out

    def _next_node(self, path, r):
        for n in path[1:]:
            if math.dist(self.g.pos[n], (r.x, r.z)) > 0.1 and r.wps and self.g.pos[n] == r.wps[0]:
                return n
        return None

    def _walkway_ok(self, t: float) -> bool:
        """Step onto the walkway only after seeing a transfer end (all robots look before crossing)."""
        lat = self.ins.local_check_latency_s
        act_now = self._active(WALK, t - lat)
        st = getattr(self, "_wk", None)
        if st is None or t - st[0] > 0.2 + DT:
            st = [t, False]
        st[0] = t
        if act_now:
            st[1] = True
        self._wk = st
        if act_now:
            return False
        if self.strategy in ("A0", "A"):
            return True
        ps = planned_stop_at(self.scn, t)
        return st[1] or bool(ps and ps[1] - t > 30 and t - ps[0] > TRANSFER_S + 2.0)

    def _ctx(self, mid: str) -> P.Ctx:
        return P.Ctx(self.need, self.ins.safety_margin_s, self.sees_motion(mid), lambda t: planned_stop_at(self.scn, t),
                     self.knowledge(mid), lambda t: self.world.true_idle_windows(mid, t, 600))

    def metrics(self) -> dict:
        stops = sum(max(0.0, min(b, self.t1 % 86400) - max(a, self.t0 % 86400)) for a, b, _ in self.scn.line_stops())
        prod_h = (self.t1 - self.t0 - stops) / 3600
        mach = [i for i in self.inspections if i["kind"] == "machine"]
        trips = [e for e in self.events if e["type"] == "trip"]
        holds = [e for e in self.events if e["type"] == "hold"]
        hold_s = sum(f.t1 - f.t0 for g in self.world.groups.values() for f in g.freezes if f.kind == "hold")
        lost = hold_s + len(trips) * self.scn.line.recovery_after_trip_s
        return {"strategy": self.strategy, "seed": self.seed, "production_h": round(prod_h, 3),
                "machine_inspections": len(mach), "rack_inspections": sum(i["kind"] == "rack" for i in self.inspections),
                "machine_insp_per_prod_h": len(mach) / prod_h if prod_h else 0.0,
                "trips": len(trips), "holds": len(holds), "hold_s": hold_s, "line_time_lost_s": lost,
                "line_cost_usd": lost * 2.3e6 / 3600,
                "zero_loss_share": (sum(1 for i in mach if i["hold_s"] < 0.05 and i["trips"] == 0) / len(mach)) if mach else 0.0,
                "mean_wait_s": (sum(i["wait_s"] for i in mach) / len(mach)) if mach else 0.0,
                "robot_collisions": len(self.collisions), "mover_collisions": self.movers.incidents,
                "protective_stops": self.robot.stops, "protective_stop_s": round(self.robot.stop_s, 1),
                "movers_blocked_s": round(float(self.movers.blocked_s.sum()), 1)}


def run(strategy: str, seed: int, t0: float = 7 * 3600, t1: float = 13 * 3600, scn: Scenario | None = None,
        fault: str | None = None, record_hz: float = 0.0) -> PhysicsRunner:
    r = PhysicsRunner(scn or load(), strategy, seed, t0, t1, fault, record_hz)
    r.run()
    return r
