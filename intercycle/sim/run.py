"""Run one inspection shift for one robot strategy on one seed; return metrics, events and a trace."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.codec.layout import quantize_fast, quantize_model
from intercycle.observer import signals
from intercycle.robot import policies as P
from intercycle.robot.predict import transfer_windows_from_index
from intercycle.robot.sipp import plan as sipp_plan
from intercycle.robot.sitegraph import CROSS_X, TRANSFER_S, SiteGraph
from intercycle.scenario import Scenario, load
from intercycle.sim.observer_learned import LearnedObserver, true_models
from intercycle.sim.world import World, planned_stop_at
from intercycle.types import DAY0_UNIX, State

STRATEGIES = ("A0", "A", "B", "C")
ISSUE_EVERY_S = 2.0
BLE_LATENCY_S = 0.08
CAM_HZ = 5.0               # robot's own camera at the boundary (same gate as ros2/intercycle_nav CameraGate)
CAM_P_OCCLUDE, CAM_BURST_S = 0.05, 1.5   # occluded = no confirmation (reads as "not clear")
CAM_P_MISS = 0.05          # visible, moving, but no motion detected (the dangerous error)
CAM_P_FALSE = 0.02         # motion reported on a still machine
CAM_CLEAR_N = 3            # consecutive clear samples needed (>= 0.4 s at 5 Hz)
JITTER = 0.05              # per-seed robot speed / dwell spread (±)


class OwnCamera:
    """The robot's own camera looking at a zone (SYNTHETIC noise model; the event sim's stand-in for the
    geometric camera model of the Isaac runs). `moving(key, t)` is True unless the last CAM_CLEAR_N samples
    before t - latency were all clear (visible and no motion seen); occlusion never confirms."""

    def __init__(self, seed: int, t0: float, t1: float, latency_s: float):
        self.seed, self.t0, self.lat = seed, t0 - 120.0, latency_s
        self.n = int((t1 + 120.0 - self.t0) * CAM_HZ) + 1
        self.noise: dict[str, tuple] = {}

    def _noise(self, key: str):
        if key not in self.noise:
            rng = np.random.default_rng([self.seed, 11, sum(map(ord, key))])
            occ = signals._occlusion(self.n, rng, CAM_P_OCCLUDE, CAM_BURST_S * CAM_HZ)
            self.noise[key] = (occ, rng.random(self.n) < CAM_P_MISS, rng.random(self.n) < CAM_P_FALSE)
        return self.noise[key]

    def moving(self, key: str, truth, t: float) -> bool:
        occ, miss, false = self._noise(key)
        k1 = int(math.floor((t - self.lat - self.t0) * CAM_HZ))
        for k in range(max(k1 - CAM_CLEAR_N + 1, 0), k1 + 1):
            if occ[k]:
                return True
            act = truth(self.t0 + k / CAM_HZ)
            if (act and not miss[k]) or (not act and false[k]):
                return True
        return False


@dataclass
class RunResult:
    strategy: str
    seed: int
    t0: float
    t1: float
    inspections: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    trace: list[tuple] = field(default_factory=list)       # (t, x, z, state)
    metrics: dict = field(default_factory=dict)


def _dijkstra(adj, a, b):
    dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj[u].items():
            if d + w < dist.get(v, math.inf):
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


class Runner:
    def __init__(self, scn: Scenario, strategy: str, seed: int, t0: float, t1: float, fault: str | None = None):
        self.scn, self.strategy, self.seed, self.t0, self.t1, self.fault = scn, strategy, seed, t0, t1, fault
        self.world = World(scn, seed, t0, t1)
        self.g = SiteGraph(scn)
        self.obs = LearnedObserver(scn, self.world, seed, t0, t1, fault=fault if fault in ("phase", "flip", "long") else None)
        rng = np.random.default_rng([seed, 13])
        self.speed_f = 1.0 + float(rng.uniform(-JITTER, JITTER))     # actual travel time factor (robot doesn't know it)
        self.dwell_f = 1.0 + float(rng.uniform(-JITTER, JITTER))     # actual time inside a zone
        self.cam = OwnCamera(seed, t0, t1, float(rng.uniform(scn.inspection.local_check_latency_s, 1.0)))
        self.true_models = true_models(scn)                    # oracle C only
        self.res = RunResult(strategy, seed, t0, t1)
        self.ins = scn.inspection
        self.need = self.ins.inside_s
        self.idx = {m.id: i for i, m in enumerate(scn.machines)}
        self._fast_cache: dict[float, object] = {}
        self.slow_model = [quantize_model(m) for m in self.obs.models]
        self.trips_seen = 0

    # ---------- robot knowledge (B) ----------
    def _issue_time(self, t: float) -> float:
        ti = math.floor((t - BLE_LATENCY_S) / ISSUE_EVERY_S) * ISSUE_EVERY_S
        if self.fault == "freeze":                              # broadcaster stuck since t0+1800
            ti = min(ti, self.t0 + 1800.0)
        if self.fault == "replay":
            ti -= 600.0
        return ti

    def _fast(self, ti: float, i: int):
        key = (ti, i if i in self.obs.cells else -1)          # the line is nowcast together, cells one by one
        if key not in self._fast_cache:
            self._fast_cache[key] = quantize_fast(self.obs.fast(ti, want=i))
            if len(self._fast_cache) > 512:
                self._fast_cache.pop(next(iter(self._fast_cache)))
        return self._fast_cache[key]

    def knowledge(self, mid: str):
        i = self.idx[mid]

        def k(t: float):
            if self.fault == "forge":                           # receiver rejects every frame
                return None, None, None
            ti = self._issue_time(t)
            f = self._fast(ti, i)
            return self.slow_model[i], f.machines[i], f.issue_time
        return k

    # ---------- local sensing / truth hooks ----------
    def sees_motion(self, mid: str):
        return lambda t: self.cam.moving(mid, lambda tt: self.world.active(mid, tt), t)

    def sees_transfer(self, t: float) -> bool:
        return self.cam.moving("WALKWAY", self.transfer_active, t)

    def ctx(self, mid: str) -> P.Ctx:
        return P.Ctx(self.need, self.ins.safety_margin_s, self.sees_motion(mid),
                     lambda t: planned_stop_at(self.scn, t), self.knowledge(mid),
                     lambda t: self.world.true_idle_windows(mid, t, 600))

    # ---------- motion ----------
    def _log(self, t, node, state):
        x, z = self.g.pos[node]
        self.res.trace.append((t, x, z, state))

    def transfer_active(self, t: float) -> bool:
        if self.world.group("S10").is_frozen(t):
            return False
        w = self.world.window("S10", t)
        return w is not None and self.world.group("S10").mt(t) - w.bs < TRANSFER_S

    def _walkway_clear(self, t: float, dur: float) -> bool:
        """True if no transfer is active during [t, t+dur] (sampled)."""
        return not any(self.transfer_active(t + k * 0.5) for k in range(int(dur / 0.5) + 1))

    def travel(self, t: float, start: str, goal: str, goal_machine: str | None) -> tuple[float, str]:
        adj = self.g.adj
        path = None
        if self.strategy == "B" and goal_machine is not None:
            path = self._sipp_path(t, start, goal, goal_machine)
        if path is None:
            path = _dijkstra(adj, start, goal)
        for a, b in zip(path, path[1:]):
            if b == "X":                                        # about to step onto the walkway
                t = self._cross_walkway(t)
            self._log(t, a, "depart")
            t += adj[a][b] * self.speed_f
            self._log(t, b, "move")
        return t, goal

    def _cross_walkway(self, t: float) -> float:
        """Step onto the line-crossing walkway. SIPP decides when B *arrives* here; every robot (A0, A, B) uses the
        same local rule on its own camera: wait to see a transfer end (a full gap follows), or a planned stop in which
        the line has finished its last cycle. C (oracle) knows the transfer schedule."""
        dur = 2 * self.g.adj["X"][f"N{CROSS_X:g}"]
        if self.strategy == "C":
            while not self._walkway_clear(t, dur):
                t += 0.5
        else:
            seen_active, t_start = False, t
            while t - t_start < 600.0:                          # a stopped line can restart at any moment
                ps = planned_stop_at(self.scn, t)
                act = self.sees_transfer(t)
                if ps and ps[1] - t > dur + 5.0 and t - ps[0] > TRANSFER_S + 2.0 and not act:
                    break                                       # calendar + own camera: line has finished its last cycle
                if seen_active and not act:
                    break                                       # observed a transfer end: full gap ahead
                seen_active |= act
                t += 0.25
        if not self._walkway_clear(t, dur):                     # a transfer starts while on the walkway
            self.world.trip("S10", t, self.strategy, where="WALKWAY")
        return t

    def _sipp_path(self, t: float, start: str, goal: str, mid: str):
        model, now, issue = self.knowledge(mid)(t)
        from intercycle.robot.predict import idle_windows
        wins = idle_windows(model, now, issue, t, horizon_s=900) if (model and now and issue) else None
        need = self.need + self.ins.safety_margin_s
        if wins is not None:
            wins = [(a, b) for a, b in wins if b - a >= need]
        dur = 2 * self.g.adj["X"][f"N{CROSS_X:g}"]
        m10, n10, i10 = self.knowledge("S10")(t)
        xw = None
        if m10 and n10 and n10.state in (State.ACTIVE, State.IDLE):
            tis = i10 - DAY0_UNIX
            idx_t = tis + (n10.next_idle_len_s if n10.state == State.IDLE
                           else n10.next_idle_in_s - m10.busy_fraction * m10.period_s)
            xw = transfer_windows_from_index(idx_t, m10.period_s, t, 900, TRANSFER_S + 1, dur / 2 + 1)

        def safe(n):
            if n == goal:
                return wins if wins else None
            if n == "X":
                return xw
            return None
        p = sipp_plan(self.g.adj, start, t, goal, safe, need_s=need if wins else 0.0, horizon_s=900)
        return p.nodes if p else None

    # ---------- one inspection ----------
    def inspect_machine(self, t: float, node: str, target) -> tuple[float, str]:
        mid = target.machine
        t, node = self.travel(t, node, target.node, mid)
        t_arrive = t
        ctx = self.ctx(mid)
        policy = {"A0": P.a0_shutdown_only, "A": P.a_reactive, "B": None, "C": P.c_oracle}[self.strategy]
        unknown_since, trips_here = None, 0
        while t < self.t1:
            if self.strategy == "B":
                if unknown_since is None:
                    unknown_since = t
                tc, note, nxt = P.b_intercycle(ctx, t, waited_unknown=t - unknown_since)
                if note != "prediction unavailable":
                    unknown_since = None
            else:
                tc, note, nxt = policy(ctx, t)
            if tc is None:
                t = max(nxt, t + 0.05)
                continue
            inside = self.need * self.dwell_f                     # actual time inside (planning uses the nominal need)
            ev = self.world.cross(mid, tc, inside, self.strategy)
            self._log(tc, node, "enter")
            if ev["type"] == "trip":
                trips_here += 1
                t = tc + self.scn.line.recovery_after_trip_s + 1.0
                if trips_here >= 3:
                    break
                continue
            t = tc + inside
            self._log(t, node, "exit")
            self.res.inspections.append({"t": tc, "target": target.id, "kind": "machine", "wait_s": tc - t_arrive,
                                         "hold_s": ev["hold_s"], "note": note, "trips": trips_here})
            return t, node
        self.res.inspections.append({"t": t, "target": target.id, "kind": "machine", "wait_s": t - t_arrive,
                                     "hold_s": 0.0, "note": "not done before end of shift", "trips": trips_here, "skipped": True})
        return t, node

    def run(self) -> RunResult:
        t, node = self.t0, "DOCK"
        self._log(t, node, "dock")
        mission = self.g.mission()
        deferred: list = []
        k = 0
        while t < self.t1:
            in_break = planned_stop_at(self.scn, t) is not None
            if deferred and in_break:
                target = deferred.pop(0)
            else:
                target = mission[k % len(mission)]
                k += 1
            if target.kind == "rack":
                t, node = self.travel(t, node, target.node, None)
                t += target.dwell_s
                self.res.inspections.append({"t": t, "target": target.id, "kind": "rack", "wait_s": 0.0, "hold_s": 0.0})
                continue
            if self.strategy in ("B", "C") and not in_break:
                i = self.idx[target.machine]
                model = self.slow_model[i] if self.strategy == "B" else self.true_models[i]
                if not P.fits_ever(model, self.need + (self.ins.safety_margin_s if self.strategy == "B" else 0.0)):
                    if target not in deferred:
                        deferred.append(target)     # never fits during production: do it in the next break
                    continue
            t, node = self.inspect_machine(t, node, target)
        self.res.events = self.world.events
        self.res.metrics = metrics(self)
        return self.res


def metrics(r: Runner) -> dict:
    scn, res = r.scn, r.res
    stops = sum(max(0.0, min(b, r.t1 % 86400) - max(a, r.t0 % 86400)) for a, b, _ in scn.line_stops())
    prod_h = (r.t1 - r.t0 - stops) / 3600
    mach = [i for i in res.inspections if i["kind"] == "machine" and not i.get("skipped")]
    trips = [e for e in res.events if e["type"] == "trip"]
    holds = [e for e in res.events if e["type"] == "inspect" and e["hold_s"] > 0.05]
    lost = sum(e.get("lost_s", 0.0) for e in res.events)
    return {
        "strategy": r.strategy, "seed": r.seed, "production_h": round(prod_h, 3),
        "machine_inspections": len(mach), "rack_inspections": sum(i["kind"] == "rack" for i in res.inspections),
        "machine_insp_per_prod_h": len(mach) / prod_h if prod_h else 0.0,
        "trips": len(trips), "holds": len(holds), "hold_s": sum(e["hold_s"] for e in holds),
        "line_time_lost_s": lost, "line_cost_usd": lost * 2.3e6 / 3600,
        "zero_loss_share": (sum(1 for i in mach if i["hold_s"] < 0.05 and i["trips"] == 0) / len(mach)) if mach else 0.0,
        "mean_wait_s": (sum(i["wait_s"] for i in mach) / len(mach)) if mach else 0.0,
        "skipped": sum(1 for i in res.inspections if i.get("skipped")),
    }


def run(strategy: str, seed: int, t0: float = 7 * 3600, t1: float = 13 * 3600, scn: Scenario | None = None,
        fault: str | None = None) -> RunResult:
    return Runner(scn or load(), strategy, seed, t0, t1, fault).run()


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for s in STRATEGIES:
        m = run(s, seed).metrics
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})
