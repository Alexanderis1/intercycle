"""The A-lite CPU double (PLAN_TWIN §7 item 5; SIMULATED body shop, descriptive only; truth side = the harness).

One episode = one generated world (KeyedWorld seed, W-W1) x one condition, two generated days:
- day 0 (production day 0, 06:05 -> 06:05): the observer learns. The robot is off. The A-lite observer
  (observer.AliteObserver: noisy camera MEASURED_SHIM -> learn_cycles / nowcast, LayeredMap, M-W4 detectors,
  T-W2 phase origin) watches the plant (plant.AlitePlant: ProgramKinematics, W-W2) at 2 Hz. At 06:05 of day 1 it
  learns its slow layer (cycle models, line offsets, phase origin) on day 0.
- day 1: the observer keeps running; the robot runs MISSIONS missions of MISSION_S (fixed start times), each one round
  of the body-shop inspection tour (SiteGraph.mission order, machine targets only; the robot idles after its round
  or stops when the mission time is up). At every target it does, in this order:
    1. a FENCE CAPTURE from the wait node on the hazard boundary (outside the zone): it passes when the machine's arms
       are parked (machine idle or held: arms stowed at home) during the whole CAPTURE_S exposure;
    2. a TIMED ENTRY into the zone (inside_s = 2 x entry_m / speed + dwell_s): World.cross gives a trip (zone active
       when crossed: 90 s protective stop of its clock group) or a hold (the machine's next start falls while the robot
       is inside: the start waits until it leaves); clean otherwise.
  Conditions (same world, same camera noise, same robot speed / dwell factors: common random numbers):
    static    captures on arrival, never enters (the static keep-out of the whole zone);
    reactive  best reactive, from its own camera only: captures once its camera has seen the machine still
              (OwnCamera rule: 3 clear samples at 5 Hz); enters at a still onset when the median of the last 5
              complete still runs its camera saw at that machine (persisted within the world) is >= inside_s +
              margin; before it has seen one it watches (up to WAIT_MAX_S), after 3 runs below that it declines;
    learned   InterCycle on the A-lite observer: captures at the start of a predicted idle window
              (robot.predict.idle_windows on the fast layer) long enough for the exposure, when the map's L1 envelope
              at the predicted phase bins over the exposure (T-W2 origin: own_phase + lag / P) stays inside the
              machine's parked footprint (its smallest learned bin envelope + 1 cell), and its own camera sees no
              motion; enters at the start of a predicted idle
              window >= inside_s + margin with its own camera clear, never when an UNKNOWN box of the map touches the
              zone (M-W4; the robot receives the boxes at its mission start, C-W7 is cut) or the slow layer says the idle never fits. No prediction: it waits (up to WAIT_MAX_S), then
              does the reactive capture and declines the entry.
  Parameters are fixed here (no tuning for any condition; disclosed).
Per mission: captures passed / attempted, entries clean / attempted, trips, holds, line time lost; Y = passed
captures + clean entries, 0 if the mission had any trip or hold (the PLAN §5 primary rule, adapted). Descriptive table
only (docs/ALITE_PREREG.md). The robot never appears in the observer's camera (W-W10, as in the body-shop event sim).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.alite.observer import AliteObserver
from intercycle.alite.phase_origin import robot_phase, to_map_phase
from intercycle.alite.plant import AlitePlant
from intercycle.alite.world import DAY, day_bounds
from intercycle.observer import mapping as M
from intercycle.observer import signals
from intercycle.robot.policies import fits_ever
from intercycle.robot.predict import idle_windows
from intercycle.robot.sitegraph import CROSS_X, TRANSFER_S, SiteGraph
from intercycle.scenario import Scenario
from intercycle.sim.world import planned_stop_at
from intercycle.types import DAY0_UNIX, MachineNow, State
from twin.rng import keyed_rng

CONDITIONS = ("learned", "reactive", "static")
MISSION_S = 720.0
MISSION_STARTS_H = (7.0, 8.0, 10.0, 11.0, 13.0, 15.0, 16.0, 18.0)       # day 1, local time
CAPTURE_S = 1.0
GUARD_S = 0.3
WAIT_MAX_S = 120.0
ISSUE_EVERY_S = 2.0
BLE_LATENCY_S = 0.08
STEP_S = 0.2
# the robot's own camera at the boundary (the body-shop event sim's noise model, intercycle/sim/run.py OwnCamera)
CAM_HZ, CAM_P_OCCLUDE, CAM_BURST_S, CAM_P_MISS, CAM_P_FALSE, CAM_CLEAR_N = 5.0, 0.05, 1.5, 0.05, 0.02, 3
JITTER = 0.05
PARKED_MIN_VISITS = 8      # = LayeredMap's own min_visits rule for a bin's envelope
REACTIVE_RUNS = 5          # reactive idle estimate = median of the last 5 complete still runs its camera saw there
REACTIVE_MIN_RUNS = 3      # ... and it declines only after 3 runs say the idle is too short


def _idle_estimate(runs: list[float]) -> float:
    return float(np.median(runs[-REACTIVE_RUNS:]))


class OwnCamera:
    """The robot's own camera on one zone (same model as intercycle/sim/run.py OwnCamera, keyed noise): moving(t) is
    True unless the last CAM_CLEAR_N samples before t - latency were all clear; occlusion never confirms."""

    def __init__(self, world_seed: int, key: str, t0: float, t1: float, latency_s: float):
        self.t0, self.lat = t0 - 120.0, latency_s
        n = int((t1 + 120.0 - self.t0) * CAM_HZ) + 1
        rng = keyed_rng(world_seed, 2, f"own_camera/{key}", "noise")
        self.occ = signals._occlusion(n, rng, CAM_P_OCCLUDE, CAM_BURST_S * CAM_HZ)
        self.miss = rng.random(n) < CAM_P_MISS
        self.false = rng.random(n) < CAM_P_FALSE

    def moving(self, truth, t: float) -> bool:
        k1 = int(math.floor((t - self.lat - self.t0) * CAM_HZ))
        for k in range(max(k1 - CAM_CLEAR_N + 1, 0), k1 + 1):
            if self.occ[k]:
                return True
            act = truth(self.t0 + k / CAM_HZ)
            if (act and not self.miss[k]) or (not act and self.false[k]):
                return True
        return False

    def saw_motion(self, truth, t: float) -> bool:
        """Positive evidence of motion: each of the last CAM_CLEAR_N samples was visible and reported motion. Unlike
        `moving` (fail closed: occluded = not clear), occlusion is never evidence that something moved (E-W6)."""
        k1 = int(math.floor((t - self.lat - self.t0) * CAM_HZ))
        if k1 - CAM_CLEAR_N + 1 < 0:
            return False
        for k in range(k1 - CAM_CLEAR_N + 1, k1 + 1):
            if self.occ[k]:
                return False
            act = truth(self.t0 + k / CAM_HZ)
            if not ((act and not self.miss[k]) or (not act and self.false[k])):
                return False
        return True


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


@dataclass
class MissionResult:
    start: float
    targets: list[dict] = field(default_factory=list)
    trips: int = 0
    holds: int = 0
    lost_s: float = 0.0

    def summary(self) -> dict:
        cap = [x for x in self.targets if x.get("capture") is not None]
        ent = [x for x in self.targets if x.get("entry") not in (None, "declined")]
        passed = sum(x["capture"] == "passed" for x in cap)
        clean = sum(x["entry"] == "clean" for x in ent)
        y = 0 if (self.trips or self.holds) else passed + clean
        return {"start": self.start, "targets_reached": len(self.targets), "captures_attempted": len(cap),
                "captures_passed": passed, "entries_attempted": len(ent), "entries_clean": clean,
                "entries_declined": sum(x.get("entry") == "declined" for x in self.targets),
                "trips": self.trips, "holds": self.holds, "lost_s": round(self.lost_s, 2), "Y": y}


class Episode:
    def __init__(self, scn: Scenario, world_seed: int, condition: str, learn_s: float | None = None,
                 mission_starts_h: tuple[float, ...] = MISSION_STARTS_H, mission_s: float = MISSION_S,
                 mission_day: int = 1):
        """learn_s: length of the learning window from 06:05 of day 0 (default: the whole production day 0).
        mission_starts_h: mission start times, hours of the day `mission_day` (default day 1); a dry run may use a
        short learning window and day 0."""
        if condition not in CONDITIONS:
            raise ValueError(f"condition {condition!r} not in {CONDITIONS}")
        self.scn, self.seed, self.cond = scn, int(world_seed), condition
        a1 = day_bounds(scn)[1]
        self.t_learn0 = a1
        self.t_learn1 = a1 + (DAY if learn_s is None else float(learn_s))
        self.starts = [mission_day * DAY + h * 3600.0 for h in mission_starts_h]
        if self.starts and self.starts[0] < self.t_learn1:
            raise ValueError("the first mission starts before the learning window ends")
        self.mission_s = float(mission_s)
        t_end = (self.starts[-1] if self.starts else self.t_learn1) + mission_s + 60.0
        self.plant = AlitePlant(scn, world_seed, self.t_learn0, t_end)
        self.world, self.kin = self.plant.world, self.plant.kin
        self.obs = AliteObserver(scn)
        self.g = SiteGraph(scn)
        self.ids = [m.id for m in scn.machines]
        self.idx = {m: i for i, m in enumerate(self.ids)}
        rng = keyed_rng(world_seed, 2, "robot", "factors")
        self.speed_f = 1.0 + float(rng.uniform(-JITTER, JITTER))
        self.dwell_f = 1.0 + float(rng.uniform(-JITTER, JITTER))
        self.cam_lat = float(rng.uniform(scn.inspection.local_check_latency_s, 1.0))
        self.need = scn.inspection.inside_s
        self.margin = scn.inspection.safety_margin_s
        self.cams: dict[str, OwnCamera] = {}
        self._k = 0                               # next observer tick index
        self._fast: dict[float, list[MachineNow]] = {}
        self.still_runs: dict[str, list[float]] = {}          # reactive: complete still runs seen per machine
        self.missions: list[MissionResult] = []
        self.unknown_at_start: list[tuple[float, float, float, float]] = []   # the map's UNKNOWN boxes at mission start

    # ------------------------------------------------------------------ observer co-simulation
    def advance(self, t: float) -> None:
        """Step the observer through every camera tick delivered at or before t (the robot's time never runs behind
        it, so a freeze the robot causes is always in the observer's future)."""
        hz = M.FRAME_HZ
        while self.t_learn0 + self._k / hz <= t:
            tk = self.t_learn0 + self._k / hz
            self.obs.step(self.plant.observe(tk))
            self._k += 1
            if self.obs.models is None and tk >= self.t_learn1:
                self.obs.learn_slow(self.t_learn0, self.t_learn1)
                self.obs.trim(3600.0)
            elif self._k % int(3600 * hz) == 0 and self.obs.models is not None:
                self.obs.trim(3600.0)

    def fast_at(self, t: float) -> tuple[list[MachineNow], float]:
        """The fast layer the robot holds at t: the last issue (every ISSUE_EVERY_S) received BLE_LATENCY_S ago."""
        ti = math.floor((t - BLE_LATENCY_S) / ISSUE_EVERY_S) * ISSUE_EVERY_S
        if ti not in self._fast:
            self.advance(ti)
            self._fast[ti] = self.obs.fast(ti)
            if len(self._fast) > 64:
                self._fast.pop(next(iter(self._fast)))
        return self._fast[ti], ti

    # ------------------------------------------------------------------ robot sensing / motion
    def cam(self, mid: str) -> OwnCamera:
        if mid not in self.cams:
            t1 = (self.starts[-1] if self.starts else self.t_learn1) + self.mission_s + 60.0
            self.cams[mid] = OwnCamera(self.seed, mid, self.starts[0] if self.starts else self.t_learn1, t1,
                                       self.cam_lat)
        return self.cams[mid]

    def sees_motion(self, mid: str, t: float) -> bool:
        return self.cam(mid).moving(lambda tt: self.world.active(mid, tt), t)

    def transfer_active(self, t: float) -> bool:
        g = self.world.group("S10")
        if g.is_frozen(t):
            return False
        w = self.world.window("S10", t)
        return w is not None and g.mt(t) - w.bs < TRANSFER_S

    def _walkway_clear(self, t: float, dur: float) -> bool:
        return not any(self.transfer_active(t + k * 0.5) for k in range(int(dur / 0.5) + 1))

    def _cross_walkway(self, t: float) -> float:
        """The body-shop camera rule (intercycle/sim/run.py Runner._cross_walkway, every condition): wait to see a
        transfer end, or a planned stop in which the line has finished its last cycle. One change (E-W6): a transfer
        counts as seen only on positive evidence (`OwnCamera.saw_motion`); in run.py any fail-closed 'moving' reading
        (occlusion, a false motion) counts, so the robot steps on at a random point of the transfer cycle."""
        dur = 2 * self.g.adj["X"][f"N{CROSS_X:g}"]
        cam = self.cam("WALKWAY")
        seen_active, t_start = False, t
        while t - t_start < 600.0:
            ps = planned_stop_at(self.scn, t)
            act = cam.moving(self.transfer_active, t)
            if ps and ps[1] - t > dur + 5.0 and t - ps[0] > TRANSFER_S + 2.0 and not act:
                break
            if seen_active and not act:
                break
            seen_active |= cam.saw_motion(self.transfer_active, t)
            t += 0.25
        if not self._walkway_clear(t, dur):
            self.world.trip("S10", t, self.cond, where="WALKWAY")
        return t

    def travel(self, t: float, start: str, goal: str) -> float:
        path = _dijkstra(self.g.adj, start, goal)
        for a, b in zip(path, path[1:]):
            if b == "X":
                t = self._cross_walkway(t)
            t += self.g.adj[a][b] * self.speed_f
        return t

    def parked(self, mid: str, t0: float, t1: float) -> bool:
        """Truth (scoring only): the machine's arms are stowed (idle or held) over [t0, t1]."""
        return all(self.kin.phase(mid, float(t))[0] in ("idle", "hold") for t in np.arange(t0, t1 + 1e-9, 0.1))

    # ------------------------------------------------------------------ learned: map + timing
    def _windows(self, mid: str, t: float):
        f, ti = self.fast_at(t)
        mods = self.obs.models
        i = self.idx[mid]
        if mods is None:
            return None, None, f[i], ti
        mod = mods[i]
        wins = idle_windows(mod, f[i], DAY0_UNIX + ti, t, horizon_s=300.0)
        return wins, mod, f[i], ti

    def parked_envelope(self, mid: str) -> np.ndarray | None:
        """The machine's parked footprint on the map: the smallest L1 envelope among its phase bins that have at least
        PARKED_MIN_VISITS visits, dilated by 1 cell (None: no bin learned enough)."""
        mp = self.obs.map
        i = mp.machine_ids().index(mid)
        vis = np.asarray(mp.l1_visits[i])
        ok = [b for b in range(M.N_PHASE) if float(np.max(vis[b])) >= PARKED_MIN_VISITS]
        if not ok:
            return None
        b0 = min(ok, key=lambda b: int(mp.envelope(mid, b).sum()))
        return M.dilate(mp.envelope(mid, b0), 1)

    def _map_parked(self, mid: str, t: float, ti: float, now: MachineNow) -> bool:
        """The map says the arms are parked over [t, t + CAPTURE_S]: the L1 envelope at the predicted phase bins (phase
        in the map's origin, T-W2: own_phase + lag / P; MapView.predicted_bins widening) stays inside the parked
        footprint (`parked_envelope`). Unknown phase, lag or footprint: not parked (fail closed)."""
        mods, org = self.obs.models, self.obs.origin
        i = self.idx[mid]
        if mods is None or org is None or mid not in self.obs.map.machine_ids():
            return False
        rp = robot_phase(mods[i], now, ti, t)
        ph = to_map_phase(None if rp is None else rp % 1.0, org.lag(i), mods[i].period_s)
        park = self.parked_envelope(mid)
        if ph is None or park is None:
            return False
        mp = self.obs.map
        for dt in (0.0, CAPTURE_S):
            env = np.zeros_like(park)
            for b in mp.predicted_bins(i, ph, M.RUNNING, dt):
                env |= mp.envelope(mid, b)
            if (env & ~park).any():
                return False
        return True

    def _unknown_in_zone(self, mid: str) -> bool:
        """An UNKNOWN box (M-W4) the robot holds touches the zone. The robot receives the map's UNKNOWN boxes at its
        mission start only (PLAN_TWIN §7 item 4: C-W7, the live push, is cut)."""
        x0, z0, x1, z1 = self.scn.machine(mid).rect
        return any(a < x1 and c > x0 and b < z1 and d > z0 for a, b, c, d in self.unknown_at_start)

    # ------------------------------------------------------------------ actions
    def capture(self, mid: str, t: float, t_end: float) -> tuple[float, str | None]:
        """Returns (time after the action, 'passed' | 'failed' | None = not attempted)."""
        tc: float | None = None
        if self.cond == "static":
            tc = t
        elif self.cond == "reactive":
            tt = t
            while tt < min(t + WAIT_MAX_S, t_end):
                if not self.sees_motion(mid, tt):
                    tc = tt
                    break
                tt += STEP_S
        else:
            tt = t
            while tt < min(t + WAIT_MAX_S, t_end):
                wins, mod, now, ti = self._windows(mid, tt)
                if wins:
                    a, b = next(((a, b) for a, b in wins if b - max(a, tt) >= CAPTURE_S + 2 * GUARD_S),
                                (None, None))
                    if a is not None and b is not None:
                        start = max(a + GUARD_S, tt)
                        if start > tt + STEP_S:
                            tt = min(start, tt + ISSUE_EVERY_S)
                            continue
                        if self._map_parked(mid, tt, ti, now) and not self.sees_motion(mid, tt):
                            tc = tt
                            break
                tt += STEP_S
            if tc is None and tt < t_end:                    # no usable prediction: the reactive capture
                t2 = tt
                while t2 < min(tt + WAIT_MAX_S, t_end):
                    if not self.sees_motion(mid, t2):
                        tc = t2
                        break
                    t2 += STEP_S
        if tc is None or tc + CAPTURE_S > t_end:
            return (tc if tc is not None else min(t + WAIT_MAX_S, t_end)), None
        self.advance(tc)
        return tc + CAPTURE_S, ("passed" if self.parked(mid, tc, tc + CAPTURE_S) else "failed")

    def entry(self, mid: str, t: float, t_end: float, res: MissionResult) -> tuple[float, str | None]:
        """Returns (time after the action, 'clean' | 'hold' | 'trip' | 'declined' | None = no time left)."""
        if self.cond == "static":
            return t, "declined"
        need = self.need + self.margin
        tc: float | None = None
        if self.cond == "reactive":
            runs = self.still_runs.setdefault(mid, [])
            tt, still_since, was_moving = t, None, None
            while tt < min(t + WAIT_MAX_S, t_end):
                mv = self.sees_motion(mid, tt)
                if was_moving is not None and not mv and was_moving:
                    still_since = tt
                    if runs and _idle_estimate(runs) >= need:
                        tc = tt
                        break
                if mv and was_moving is False and still_since is not None:
                    runs.append(tt - still_since)
                    if len(runs) >= REACTIVE_MIN_RUNS and _idle_estimate(runs) < need:
                        return tt, "declined"
                was_moving = mv
                tt += STEP_S
            if tc is None:
                return tt, ("declined" if tt < t_end else None)
        else:
            mods = self.obs.models
            if mods is None or not fits_ever(mods[self.idx[mid]], need):
                return t, "declined"
            if self._unknown_in_zone(mid):
                return t, "declined"
            tt = t
            while tt < min(t + WAIT_MAX_S, t_end):
                wins, mod, now, ti = self._windows(mid, tt)
                if wins:
                    fit = next(((a, b) for a, b in wins if b - max(a, tt) >= need), None)
                    if fit is not None:
                        start = max(fit[0], tt)
                        if start > tt + 1e-6:
                            tt = min(start, tt + ISSUE_EVERY_S)
                            continue
                        if not self.sees_motion(mid, tt):
                            tc = tt
                            break
                tt += STEP_S
            if tc is None:
                return tt, ("declined" if tt < t_end else None)
        if tc + self.need * self.dwell_f > t_end:
            return tc, None
        self.advance(tc)
        inside = self.need * self.dwell_f
        ev = self.world.cross(mid, tc, inside, self.cond)
        if ev["type"] == "trip":
            res.trips += 1
            res.lost_s += float(ev["lost_s"])
            return tc + self.scn.line.recovery_after_trip_s + 1.0, "trip"
        if ev["hold_s"] > 0.05:
            res.holds += 1
            res.lost_s += float(ev["hold_s"])
            return tc + inside, "hold"
        return tc + inside, "clean"

    def mission(self, t0: float) -> MissionResult:
        res = MissionResult(t0)
        t_end = t0 + self.mission_s
        self.advance(t0)
        self.unknown_at_start = [(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                                 for r in self.obs.map.unknown_rects]
        t, node = t0, "DOCK"
        for tg in self.g.mission():
            if tg.kind != "machine" or tg.machine is None:
                continue
            t = self.travel(t, node, tg.node)
            node = tg.node
            if t >= t_end:
                break
            mid = tg.machine
            rec: dict = {"target": mid, "t_arrive": round(t - t0, 2)}
            t, rec["capture"] = self.capture(mid, t, t_end)
            if t < t_end:
                t, rec["entry"] = self.entry(mid, t, t_end, res)
            res.targets.append(rec)
            if t >= t_end:
                break
        n0 = len([e for e in self.world.events if e.get("machine") == "WALKWAY" and t0 <= e["t"] < t_end])
        res.trips += n0
        res.lost_s += n0 * self.scn.line.recovery_after_trip_s
        return res

    def run(self) -> dict:
        for s in self.starts:
            self.missions.append(self.mission(s))
        self.advance(self.starts[-1] + self.mission_s if self.starts else self.t_learn1)
        ms = [m.summary() for m in self.missions]
        tot = {k: sum(m[k] for m in ms) for k in ("captures_attempted", "captures_passed", "entries_attempted",
                                                  "entries_clean", "entries_declined", "trips", "holds", "lost_s", "Y")}
        org = self.obs.origin
        return {"kind": "alite_episode", "simulated": True, "world_seed": self.seed, "condition": self.cond,
                "learn_h": (self.t_learn1 - self.t_learn0) / 3600.0, "mission_s": self.mission_s,
                "n_missions": len(ms), "totals": tot, "missions": ms,
                "targets": [m.targets for m in self.missions],
                "observer": {"models": [None if self.obs.models is None else
                                        {"i": m.index, "P": round(m.period_s, 3), "busy": round(m.busy_fraction, 3),
                                         "conf": m.confidence} for m in (self.obs.models or [])],
                             "phase_origin_lag_s": None if org is None else [None if not math.isfinite(x) else
                                                                              round(x, 3) for x in org.lag_s],
                             "events": [(round(e["t"], 1), e["kind"]) for e in self.obs.events][:200],
                             "n_unknown_rects_end": len(self.obs.map.unknown_rects)}}


def run(scn: Scenario, world_seed: int, condition: str, **kw) -> dict:
    return Episode(scn, world_seed, condition, **kw).run()
