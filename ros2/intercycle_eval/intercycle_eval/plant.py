"""The reactive plant (R3, DIRECTION 5.4): everything around the robot, stepped inside the simulator process.

    cfg["mover_source"] = "intercycle_eval.plant:make"      (runs in Isaac's python 3.11; numpy + pydantic + yaml)

The robot perceives none of this directly: the plant drives Isaac kinematic colliders (`poses`), the analytic
primitives the lidar / camera models hit (`sensor_prims`), and the robot-camera no-motion observations
(`frame_extras` -> /intercycle/camera/observations). Truth is logged for evaluation only.

What is simulated (plant time = plant_t0 + sim time):
  * Machines: arm envelopes, skids, lifter carriage and docked AGVs replayed from intercycle.truth.kinematics on the
    episode's World (cycle jitter, drift, andon stops, cell micro-stops, planned stops).
  * Light curtains (BRIEF): the robot footprint crossing into an ACTIVE hazard zone = TRIP (World.trip: protective
    stop, line stop, restart after recovery_after_trip_s = 90 s; a line station stops the whole line). Robot inside an
    IDLE zone when the next start is due = HOLD until it leaves (line stations hold the whole line). The line walkway
    (x = 52.5 +- 1 across the conveyor) is guarded during skid transfers, as in the event sim.
    The freezes feed back into the kinematics (arms freeze mid-motion) and into the observer's cameras.
  * People: social force (social.py; PySocialForce terms) along their truth routes, the robot is a repulsive agent,
    20 % impassive; `invisible_robot` = nobody reacts (stress condition).
  * Forklifts follow their routes and do NOT yield: the driver brakes only after a 1.2 s reaction once the robot or a
    person is in the 10 m corridor ahead, with a bounded deceleration of 1.5 m/s^2.
  * AGVs / tugger trains: ISO 3691-4-style scanner fields: protective field (stop, 1.0 m/s^2) and warning field
    (slow to 0.3 m/s); automatic restart 1 s after the protective field clears.
  * Isaac colliders: every mover part is a kinematic body; machine parts (about 120 circles) are mapped onto a pool of
    kinematic cylinders assigned to the parts nearest the robot (sensors always see ALL parts analytically).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from intercycle_sim import movers as MV

from intercycle_eval import camera as CAM
from intercycle_eval.social import Crowd

TICK = 0.1                       # plant dynamics / zone physics / log rate (s)
FOOT = ((-0.34, 0.32), (0.0, 0.32), (0.34, 0.32))   # ANYmal-D footprint ~1.0 x 0.64 m as 3 circles (fwd offset, r)
ROBOT_DISC_R = 0.5
PERSONAL_SPACE_M = 0.45          # Hall's intimate distance, person centre to robot footprint edge
PERSON_R = 0.3
IMPASSIVE_FRAC = 0.2
FL_REACTION_S, FL_DECEL, FL_CORRIDOR, FL_ACCEL = 1.2, 1.5, 10.0, 0.5
AGV_DECEL, AGV_TR, AGV_MARGIN, AGV_WARN, AGV_SLOW, AGV_RESTART_S, AGV_ACCEL = 1.0, 0.1, 0.3, 1.5, 0.3, 1.0, 0.5
POOL = {"arm65": (0.65, 2.0, 6), "arm50": (0.50, 2.0, 22), "skid": (1.18, 1.9, 6), "lift": (1.509, 2.2, 3),
        "dock": (0.50, 1.2, 4)}
POOL_RANGE = 8.0
CLS_OF = {0: "person", 1: "agv", 2: "tugger", 3: "forklift"}
FRONT = {"agv": 0.8, "tugger": 0.7, "forklift": 2.05}
WIDTH = {"agv": 1.0, "tugger": 1.0, "forklift": 1.1}


def _yaw_to_psi(yaw: float) -> float:
    return math.pi / 2 - yaw


def circ_gap(px: float, py: float, r: float, circles: np.ndarray) -> np.ndarray:
    if circles is None or len(circles) == 0:
        return np.zeros(0)
    return np.hypot(circles[:, 0] - px, circles[:, 1] - py) - circles[:, 2] - r


def box_gap(px: float, py: float, r: float, boxes: np.ndarray) -> np.ndarray:
    """boxes (cx, cy, hw, hd, yaw[, ...])."""
    if boxes is None or len(boxes) == 0:
        return np.zeros(0)
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    dx, dy = px - boxes[:, 0], py - boxes[:, 1]
    lx, ly = np.abs(dx * c + dy * s), np.abs(-dx * s + dy * c)
    out = np.hypot(np.maximum(lx - boxes[:, 2], 0), np.maximum(ly - boxes[:, 3], 0))
    inside = (lx <= boxes[:, 2]) & (ly <= boxes[:, 3])
    out = np.where(inside, -np.minimum(boxes[:, 2] - lx, boxes[:, 3] - ly), out)
    return out - r


def circles_in_rect(fc: list[tuple[float, float, float]], rect) -> bool:
    x0, y0, x1, y1 = rect
    for x, y, r in fc:
        qx, qy = min(max(x, x0), x1), min(max(y, y0), y1)
        if (x - qx) ** 2 + (y - qy) ** 2 < r * r:
            return True
    return False


def ttc(rel_p: np.ndarray, rel_v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Time to first contact of discs moving at constant velocity (rel = other - robot); inf if never; 0 if touching."""
    a = (rel_v ** 2).sum(-1)
    b = 2 * (rel_p * rel_v).sum(-1)
    c = (rel_p ** 2).sum(-1) - R ** 2
    disc = b * b - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    out = np.where((disc >= 0) & (a > 1e-9) & (t1 >= 0), t1, np.inf)
    return np.where(c <= 0, 0.0, out)


class _Vehicle:
    def __init__(self, ag, cls: str):
        self.ag, self.cls = ag, cls
        self.tau = None
        self.r = 1.0             # progress rate = speed / nominal speed
        self.state = "drive"     # drive | slow | stop | react | brake
        self.t_seen = None
        self.t_clear = None
        self.pose = None
        self.vel = np.zeros(2)
        self.n_stops = 0


class Plant:
    def __init__(self, cfg: dict):
        from intercycle import scenario as S
        from intercycle.robot.sitegraph import CROSS_X, TRANSFER_S, SiteGraph
        from intercycle.sim.world import Freeze, World
        from intercycle.truth.agents import agents as truth_agents
        from intercycle.truth.kinematics import Kinematics

        self._Freeze, self.TRANSFER_S = Freeze, TRANSFER_S
        p = dict(cfg.get("plant") or {})
        self.cfg, self.p = cfg, p
        self.scn = scn = S.load()
        self.seed = int(p.get("seed", cfg.get("seed", 0)))
        self.t0 = float(p.get("plant_t0", 36000.0))
        self.T = float(cfg.get("seconds", 600.0))
        self.world = World(scn, self.seed, self.t0, self.t0 + self.T + 120.0)
        self.kin = Kinematics(scn, self.world)
        self.rng = np.random.default_rng([self.seed, 31])
        self.invisible = bool(p.get("invisible_robot", False))
        self.work = os.path.dirname(cfg.get("world_dir", "/tmp/x/world"))
        self.out_dir = cfg.get("out_dir", self.work)

        # ---- hazard zones (+ the guarded line walkway)
        self.zones = {m.id: tuple(float(v) for v in m.rect) for m in scn.machines}
        self.zones["WALK"] = (CROSS_X - 1.0, 24.0 - 1.2, CROSS_X + 1.0, 24.0 + 1.2)
        self.zone_gid = {m.id: m.id for m in scn.machines}
        self.zone_gid["WALK"] = "S10"
        self.inside = {z: False for z in self.zones}
        self.inside_since = {z: None for z in self.zones}

        # ---- inspection targets (sim-side geometry of what the camera inspects)
        g = SiteGraph(scn)
        self.targets = {}
        for tg in g.targets:
            if tg.kind != "machine":
                continue
            ix, iz = g.inspect_pose(tg)
            m = scn.machine(tg.machine)
            bases = [(bx, bz) for mid, bx, bz in self.kin.arm_bases if mid == m.id]
            if bases:
                bx, bz = min(bases, key=lambda b: math.hypot(b[0] - ix, b[1] - iz))
                r0 = 0.65
            else:
                bx, bz, r0 = m.x, m.z + math.copysign(min(2.0, abs(iz - m.z) - 1.0), iz - m.z), 0.0
            n = np.array((ix - bx, iz - bz, 0.0))
            n /= max(np.linalg.norm(n), 1e-9)
            self.targets[tg.id] = {"machine": m.id, "xyz": np.array((bx, bz, CAM.TARGET_Z)) + n * r0, "normal": n}

        # ---- movers from the truth routes (only those present at some point of the episode)
        ts = self.t0 + np.arange(0.0, self.T + 30.0, 10.0)
        self.people, self.vehicles = [], []
        for ag in truth_agents(scn, self.seed):
            if not any(ag.present(t) for t in ts):
                continue
            cls = CLS_OF.get(int(ag.cls), "person")
            (self.people if cls == "person" else self.vehicles).append(ag if cls == "person" else _Vehicle(ag, cls))
        self.np_ = len(self.people)
        reacts = self.rng.random(self.np_) >= float(p.get("impassive_frac", IMPASSIVE_FRAC))
        if self.invisible:
            reacts[:] = False
        self.reacts = reacts
        boxes = []
        try:
            from intercycle_sim.world import load_boxes

            boxes = load_boxes(os.path.join(cfg["world_dir"], "boxes.json"))
        except Exception:  # noqa: BLE001 - local tests without a world dir
            pass
        self.static_boxes = np.array([(b["cx"], b["cy"], b["w"] / 2, b["d"] / 2, b["yaw"]) for b in boxes],
                                     float).reshape(-1, 5)
        self.crowd = Crowd(self.np_, self.static_boxes, reacts)
        self.p_tau = np.zeros(self.np_)
        self.p_present = np.zeros(self.np_, bool)
        self.p_psi = np.zeros(self.np_)

        # fixed mover spec (Isaac creates one kinematic body per part) + the machine collider pool
        spec = [{"id": a.id, "cls": "person"} for a in self.people] + [{"id": v.ag.id, "cls": v.cls} for v in self.vehicles]
        self.mover_parts = MV.parts_for(spec)
        self.pool_ids = []
        for key, (r, h, n) in POOL.items():
            for k in range(n):
                sid = f"MC_{key}_{k:02d}"
                spec.append({"id": sid, "cls": f"cyl:{r}:{h}"})
                self.pool_ids.append(sid)
        self._spec = spec
        self.pool_assign: dict[str, dict] = {key: {} for key in POOL}   # pool key -> {slot k: (body key, idx)}

        # ---- runtime state
        self.caster = None
        self.robot = None            # (x, y, psi, vx_w, vy_w, wz, z)
        self.t_tick = -1e9
        self.t = 0.0
        self.mpose: dict[str, tuple[float, float, float]] = {}      # mover id -> (x, y, psi) at the last tick
        self.mvel: dict[str, tuple[float, float]] = {}
        self.t_mpose = 0.0
        self.gate = CAM.Gate(np.random.default_rng([self.seed, 77]))
        self.cap_rng = np.random.default_rng([self.seed, 78])
        self.pending = []            # inspection captures in progress
        self.captures = []
        self.outbox_cap = []
        self.events = []
        self.rows = []
        self.contacts = []
        self.ps_in = False
        self.ps_events = 0
        self.ps_time = 0.0
        self.mins = {"person": math.inf, "vehicle": math.inf, "machine": math.inf, "static": math.inf,
                     "ttc_person": math.inf, "ttc_vehicle": math.inf}
        self.overlap = {"person": 0, "vehicle": 0, "machine": 0}
        self._overlap_prev = {"person": False, "vehicle": False, "machine": False}
        self.wall_plant = 0.0
        self.n_ticks = 0
        self._last_state_write = -1e9
        self._prim_cache = (None, None)
        self._mach_cache = (None, None)
        self._mp_cache = (None, None)
        self._mc_cache = (None, None)
        self.feed = bool(p.get("feed", True))
        self._write_state(0.0)

    # ------------------------------------------------------------------ hooks called by intercycle_sim.runner
    def spec(self) -> list[dict]:
        return list(self._spec)

    def attach(self, caster=None, dt: float = 0.02) -> None:
        self.caster = caster

    def _machine_circles(self, t: float) -> dict[str, np.ndarray]:
        if self._mach_cache[0] != t:
            self._mach_cache = (t, self.kin.bodies(self.t0 + t))
        return self._mach_cache[1]

    def _pool_key(self, body: str, r: float) -> str:
        if body == "SKIDS":
            return "skid"
        if body == "AGVDOCK":
            return "dock"
        if body == "S80":
            return "lift"
        return "arm65" if r > 0.6 else "arm50"

    def poses(self, t: float) -> dict[str, tuple[float, float, float]]:
        out = {}
        dt = t - self.t_mpose
        for mid, (x, y, psi) in self.mpose.items():
            vx, vy = self.mvel.get(mid, (0.0, 0.0))
            out[mid] = (x + vx * dt, y + vy * dt, psi)
        bodies = self._machine_circles(t)
        for key, asg in self.pool_assign.items():
            for k, (body, i) in asg.items():
                c = bodies.get(body)
                if c is not None and i < len(c):
                    out[f"MC_{key}_{k:02d}"] = (float(c[i, 0]), float(c[i, 1]), 0.0)
        return out

    def _mover_prims(self, t: float) -> dict:
        if self._mp_cache[0] == t:
            return self._mp_cache[1]
        dt = t - self.t_mpose
        cur = {}
        for mid, (x, y, psi) in self.mpose.items():
            vx, vy = self.mvel.get(mid, (0.0, 0.0))
            cur[mid] = (x + vx * dt, y + vy * dt, psi)
        pp = MV.part_poses(self.mover_parts, cur) if self.mover_parts else np.zeros((0, 3))
        out = MV.primitives(self.mover_parts, pp) if self.mover_parts else {"circles": np.zeros((0, 4)),
                                                                             "boxes": np.zeros((0, 6))}
        self._mp_cache = (t, out)
        return out

    def _machine_prims(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(circles (K, 4) cx, cy, r, zmax, body label per row)."""
        if self._mc_cache[0] == t:
            return self._mc_cache[1]
        bodies = self._machine_circles(t)
        out, lab = [], []
        for body, c in bodies.items():
            if len(c) == 0:
                continue
            h = POOL[self._pool_key(body, float(c[0, 2]))][1]
            out.append(np.column_stack((c[:, :3], np.full(len(c), h))))
            lab += [body] * len(c)
        res = (np.concatenate(out) if out else np.zeros((0, 4)), np.array(lab))
        self._mc_cache = (t, res)
        return res

    def sensor_prims(self, t: float, skip_machine: str | None = None) -> dict:
        """ALL moving bodies as analytic primitives (what the lidar and camera models hit): mover circles (people) first,
        then machine-part circles; boxes = vehicle parts."""
        mp = self._mover_prims(t)
        mc, lab = self._machine_prims(t)
        if skip_machine is not None and len(mc):
            mc = mc[lab != skip_machine]
        if skip_machine is None and self._prim_cache[0] == t:
            return self._prim_cache[1]
        out = {"circles": np.concatenate((mp["circles"], mc)), "boxes": mp["boxes"]}
        if skip_machine is None:
            self._prim_cache = (t, out)
        return out

    def after_step(self, t: float, st: dict, contacts: list[dict]) -> None:
        w = time.time()
        self.t = t
        x, y, z = (float(v) for v in st["pos_w"][:3])
        from intercycle_sim.sensors import yaw_of

        psi = yaw_of(st["quat_wxyz"])
        vw = st["v_w"]
        self.robot = (x, y, psi, float(vw[0]), float(vw[1]), float(st["omega_b"][2]), z)
        if contacts:
            self._classify_contacts(t, contacts)
        self._captures_step(t, st)
        if t - self.t_tick >= TICK - 1e-9:
            dt = min(t - self.t_tick, 0.5) if self.t_tick > -1e8 else TICK
            self.t_tick = t
            self._tick(t, dt)
        if self.gate.due(t):
            self._gate(t)
        self.wall_plant += time.time() - w

    def frame_extras(self, t: float) -> dict:
        out = {}
        obs = self.gate.ready(t)
        if obs:
            out["camera_obs"] = obs
        if self.outbox_cap:
            out["inspection_capture"] = self.outbox_cap
            self.outbox_cap = []
        return out

    def on_ros(self, t: float, events: list[dict]) -> None:
        """Messages from ROS (the timing planner's /intercycle/inspection dwell events)."""
        for e in events:
            kind = e.get("event")
            if kind == "dwell_start":
                self.pending.append({"target": e.get("target"), "machine": e.get("machine"),
                                     "inside": bool(e.get("inside")), "t_start": t, "hold_since": None})
            elif kind == "dwell_end":
                for pc in [q for q in self.pending if q["target"] == e.get("target")]:
                    self._capture(t, pc, settled=False)

    # ------------------------------------------------------------------ dynamics
    def _tick(self, t: float, dt: float) -> None:
        T = self.t0 + t
        rx, ry, rpsi, rvx, rvy, rwz, _ = self.robot
        fc = [(rx + o * math.cos(rpsi), ry + o * math.sin(rpsi), r) for o, r in FOOT]
        self._zones(t, T, fc)
        self._people(T, dt, (rx, ry), (rvx, rvy))
        self._vehicles(T, dt, (rx, ry))
        self.t_mpose = t
        self._pool(t, rx, ry)
        self._metrics(t, T, fc)
        if self.feed and t - self._last_state_write >= 0.1 - 1e-9:
            self._write_state(t)
        self.n_ticks += 1

    def _zone_active(self, zid: str, T: float) -> bool:
        if zid == "WALK":
            g = self.world.group("S10")
            if g.is_frozen(T):
                return False
            w = self.world.window("S10", T)
            return w is not None and g.mt(T) - w.bs < self.TRANSFER_S
        return self.world.active(zid, T)

    def _zones(self, t: float, T: float, fc) -> None:
        rx, ry = fc[1][0], fc[1][1]
        for zid, rect in self.zones.items():
            x0, y0, x1, y1 = rect
            if not (x0 - 2 < rx < x1 + 2 and y0 - 2 < ry < y1 + 2):
                if self.inside[zid]:
                    self._exit(zid, t)
                continue
            ins = circles_in_rect(fc, rect)
            gid = self.zone_gid[zid]
            if ins and not self.inside[zid]:
                act = self._zone_active(zid, T)
                ev = {"t": round(t, 2), "type": "enter", "zone": zid, "active": act}
                self.events.append(ev)
                self.inside_since[zid] = t
                if act:
                    tr = self.world.trip(gid, T, "robot", where=zid)
                    self.events.append({"t": round(t, 2), "type": "trip", "zone": zid, "group": tr["group"],
                                        "lost_s": tr["lost_s"]})
            if ins and self._zone_active(zid, T):          # the machine wants to start with the robot inside -> hold
                g = self.world.group(gid)
                last = g.freezes[-1] if g.freezes else None
                if last and last.kind == "hold" and last.machine == zid and last.t1 >= T - TICK - 0.05:
                    last.t1 = T + TICK
                else:
                    g.freezes.append(self._Freeze(T, T + TICK, "hold", zid))
                    self.events.append({"t": round(t, 2), "type": "hold", "zone": zid, "group": g.name})
            if not ins and self.inside[zid]:
                self._exit(zid, t)
            self.inside[zid] = ins

    def _exit(self, zid: str, t: float) -> None:
        self.inside[zid] = False
        t_in = self.inside_since.get(zid)
        self.events.append({"t": round(t, 2), "type": "exit", "zone": zid,
                            "inside_s": None if t_in is None else round(t - t_in, 2)})

    def _people(self, T: float, dt: float, rxy, rv) -> None:
        if not self.np_:
            return
        goals = np.zeros((self.np_, 2))
        v0 = np.zeros(self.np_)
        for i, ag in enumerate(self.people):
            pres = ag.present(T)
            if pres and not self.p_present[i]:
                self.p_tau[i] = T
                x, zz, yaw, _ = ag.at(T)
                self.crowd.place(i, (x, zz))
                self.p_psi[i] = _yaw_to_psi(yaw)
            elif not pres and self.p_present[i]:
                self.crowd.active[i] = False
            self.p_present[i] = pres
            if not pres:
                continue
            rx_, rz_, _, _ = ag.at(self.p_tau[i])
            lag = math.hypot(rx_ - self.crowd.pos[i, 0], rz_ - self.crowd.pos[i, 1])
            self.p_tau[i] += dt * (1.0 if lag < 1.0 else 0.25)
            gx, gz, _, _ = ag.at(self.p_tau[i] + 1.0)
            goals[i] = (gx, gz)
            v0[i] = ag.speed
        self.crowd.step(dt, goals, v0, None if self.invisible else rxy, rv)
        for i, ag in enumerate(self.people):
            if not self.p_present[i]:
                self.mpose.pop(ag.id, None)
                self.mvel.pop(ag.id, None)
                continue
            vx, vy = self.crowd.vel[i]
            if vx * vx + vy * vy > 0.01:
                self.p_psi[i] = math.atan2(vy, vx)
            self.mpose[ag.id] = (float(self.crowd.pos[i, 0]), float(self.crowd.pos[i, 1]), float(self.p_psi[i]))
            self.mvel[ag.id] = (float(vx), float(vy))

    def _hazards(self, rxy) -> list[tuple[float, float, float, str]]:
        hz = [(rxy[0], rxy[1], ROBOT_DISC_R, "robot")]
        for i in np.flatnonzero(self.crowd.active):
            hz.append((float(self.crowd.pos[i, 0]), float(self.crowd.pos[i, 1]), PERSON_R, "person"))
        return hz

    def _vehicles(self, T: float, dt: float, rxy) -> None:
        hz = None
        for v in self.vehicles:
            ag = v.ag
            if not ag.present(T):
                v.tau = None
                self.mpose.pop(ag.id, None)
                self.mvel.pop(ag.id, None)
                continue
            if v.tau is None:
                v.tau, v.r, v.state = T, 1.0, "drive"
            x, zz, yaw, spd = ag.at(v.tau)
            psi = _yaw_to_psi(yaw)
            c, s = math.cos(psi), math.sin(psi)
            hz = hz if hz is not None else self._hazards(rxy)
            vcur = v.r * ag.speed if spd > 0 else 0.0
            front, half_w = FRONT[v.cls], WIDTH[v.cls] / 2

            def in_field(length: float, extra_w: float) -> str | None:
                for hx, hy, hr, who in hz:
                    lx, ly = (hx - x) * c + (hy - zz) * s, -(hx - x) * s + (hy - zz) * c
                    if front - hr <= lx <= front + length + hr and abs(ly) <= half_w + extra_w + hr:
                        return who
                return None

            target_r, decel = 1.0, AGV_ACCEL
            if v.cls == "forklift":
                who = in_field(FL_CORRIDOR, 0.5) if spd > 0 else None
                if who:
                    v.t_seen = v.t_seen if v.t_seen is not None else T
                    v.t_clear = None
                    if T - v.t_seen >= FL_REACTION_S:          # driver reacts after 1.2 s, then brakes (bounded)
                        if v.state != "brake":
                            v.n_stops += 1
                            self.events.append({"t": round(T - self.t0, 2), "type": "forklift_brake", "id": ag.id,
                                                "who": who, "v": round(vcur, 2)})
                        v.state, target_r, decel = "brake", 0.0, FL_DECEL
                    else:
                        v.state = "react"
                else:
                    v.t_seen = None
                    if v.state == "brake":
                        v.t_clear = v.t_clear if v.t_clear is not None else T
                        if T - v.t_clear < 1.0:
                            target_r, decel = 0.0, FL_DECEL
                        else:
                            v.state = "drive"
                    else:
                        v.state = "drive"
            else:                                               # AGV / tugger: ISO 3691-4 protective + warning field
                lp = vcur * AGV_TR + vcur * vcur / (2 * AGV_DECEL) + AGV_MARGIN
                who = in_field(lp, 0.1) if spd > 0 or v.state == "stop" else None
                if who:
                    if v.state != "stop":
                        v.n_stops += 1
                        self.events.append({"t": round(T - self.t0, 2), "type": "agv_protective_stop", "id": ag.id,
                                            "who": who, "v": round(vcur, 2)})
                    v.state, v.t_clear, target_r, decel = "stop", None, 0.0, AGV_DECEL
                elif v.state == "stop":
                    v.t_clear = v.t_clear if v.t_clear is not None else T
                    if T - v.t_clear < AGV_RESTART_S:
                        target_r, decel = 0.0, AGV_DECEL
                    else:
                        v.state = "drive"
                if v.state != "stop" and spd > 0 and in_field(lp + AGV_WARN, 0.3):
                    target_r = min(target_r, AGV_SLOW / ag.speed)
                    v.state = "slow" if target_r < 1.0 else v.state
            step = (AGV_ACCEL if target_r > v.r else decel) * dt / max(ag.speed, 1e-6)
            v.r = min(target_r, v.r + step) if target_r > v.r else max(target_r, v.r - step)
            v.tau += dt * v.r
            x, zz, yaw, spd = ag.at(v.tau)
            psi = _yaw_to_psi(yaw)
            sp = v.r * spd
            self.mpose[ag.id] = (float(x), float(zz), float(psi))
            self.mvel[ag.id] = (sp * math.cos(psi), sp * math.sin(psi))

    def _pool(self, t: float, rx: float, ry: float) -> None:
        """Assign the Isaac collider pool to the machine parts nearest the robot (sticky assignment)."""
        bodies = self._machine_circles(t)
        cand: dict[str, list[tuple[float, tuple[str, int]]]] = {k: [] for k in POOL}
        for body, c in bodies.items():
            if len(c) == 0:
                continue
            d = np.hypot(c[:, 0] - rx, c[:, 1] - ry) - c[:, 2]
            for i in np.flatnonzero(d < POOL_RANGE):
                cand[self._pool_key(body, float(c[i, 2]))].append((float(d[i]), (body, int(i))))
        for key, lst in cand.items():
            n = POOL[key][2]
            want = {bi for _, bi in sorted(lst)[:n]}
            asg = self.pool_assign[key]
            for k in [k for k, bi in asg.items() if bi not in want]:
                del asg[k]
            have = set(asg.values())
            free = [k for k in range(n) if k not in asg]
            for bi in want - have:
                if free:
                    asg[free.pop(0)] = bi

    # ------------------------------------------------------------------ metrics (truth, evaluation only)
    def _metrics(self, t: float, T: float, fc) -> None:
        rx, ry, rpsi, rvx, rvy, rwz, _ = self.robot
        pr = self.sensor_prims(t)
        circ, boxes = pr["circles"], pr["boxes"]
        n_mc = len(self.mover_parts)
        # categories: mover parts are listed first in sensor_prims (circles: people only; boxes: vehicles)
        pc = circ[:sum(1 for p in self.mover_parts if p.shape == "cyl")]
        mc = circ[len(pc):]
        gp = min((float(circ_gap(x, y, r, pc).min()) for x, y, r in fc), default=math.inf) if len(pc) else math.inf
        gv = min((float(box_gap(x, y, r, boxes).min()) for x, y, r in fc), default=math.inf) if len(boxes) else math.inf
        gm = min((float(circ_gap(x, y, r, mc).min()) for x, y, r in fc), default=math.inf) if len(mc) else math.inf
        near_static = self.static_boxes[(np.abs(self.static_boxes[:, 0] - rx) < 8) & (np.abs(self.static_boxes[:, 1] - ry) < 8)] \
            if len(self.static_boxes) else self.static_boxes
        gs = min((float(box_gap(x, y, r, near_static).min()) for x, y, r in fc), default=math.inf) if len(near_static) else math.inf
        # TTC (constant velocity, discs): people (r 0.3) and vehicle parts (bounding discs)
        rp = np.array((rx, ry))
        rv = np.array((rvx, rvy))
        tp = tv = math.inf
        pid = [a.id for a in self.people if a.id in self.mpose]
        if pid:
            P = np.array([self.mpose[i][:2] for i in pid])
            V = np.array([self.mvel[i] for i in pid])
            tp = float(ttc(P - rp, V - rv, np.full(len(P), ROBOT_DISC_R + PERSON_R)).min())
            dmin_c = np.hypot(P[:, 0] - rx, P[:, 1] - ry)
            # personal space: person centre within 0.45 m of the robot footprint edge
            edge = min(float((np.hypot(P[:, 0] - x, P[:, 1] - y) - r).min()) for x, y, r in fc)
            ins = edge < PERSONAL_SPACE_M
            if ins and not self.ps_in:
                self.ps_events += 1
            if ins:
                self.ps_time += TICK
            self.ps_in = ins
            del dmin_c
        vid = [v.ag.id for v in self.vehicles if v.ag.id in self.mpose]
        if vid and len(boxes):
            Vp, Vv, Vr = [], [], []
            for v in self.vehicles:
                if v.ag.id not in self.mpose:
                    continue
                x, y, psi = self.mpose[v.ag.id]
                for prt in [q for q in self.mover_parts if q.mover_id == v.ag.id]:
                    Vp.append((x + prt.fwd * math.cos(psi), y + prt.fwd * math.sin(psi)))
                    Vv.append(self.mvel[v.ag.id])
                    Vr.append(math.hypot(prt.length / 2, prt.width / 2))
            tv = float(ttc(np.array(Vp) - rp, np.array(Vv) - rv, np.array(Vr) + ROBOT_DISC_R).min())
        for k, gg in (("person", gp), ("vehicle", gv), ("machine", gm), ("static", gs)):
            self.mins[k] = min(self.mins[k], gg)
        self.mins["ttc_person"] = min(self.mins["ttc_person"], tp)
        self.mins["ttc_vehicle"] = min(self.mins["ttc_vehicle"], tv)
        for k, gg in (("person", gp), ("vehicle", gv), ("machine", gm)):
            ov = gg < -0.02
            if ov and not self._overlap_prev[k]:
                self.overlap[k] += 1
                self.events.append({"t": round(t, 2), "type": "overlap", "with": k, "gap_m": round(gg, 3)})
            self._overlap_prev[k] = ov
        zin = next((z for z, v in self.inside.items() if v), None)
        self.rows.append({"t": round(t, 2), "x": round(rx, 3), "y": round(ry, 3), "psi": round(rpsi, 4),
                          "v": round(math.hypot(rvx, rvy), 3), "wz": round(rwz, 3),
                          "gap_person": None if gp == math.inf else round(gp, 3),
                          "gap_vehicle": None if gv == math.inf else round(gv, 3),
                          "gap_machine": None if gm == math.inf else round(gm, 3),
                          "gap_static": None if gs == math.inf else round(gs, 3),
                          "ttc_person": None if tp == math.inf else round(tp, 2),
                          "ttc_vehicle": None if tv == math.inf else round(tv, 2), "zone": zin,
                          "n_people": int(self.crowd.active.sum())})
        del n_mc

    def _classify_contacts(self, t: float, contacts: list[dict]) -> None:
        rx, ry, rpsi = self.robot[:3]
        fc = [(rx + o * math.cos(rpsi), ry + o * math.sin(rpsi), r) for o, r in FOOT]
        pr = self.sensor_prims(t)
        npc = sum(1 for p in self.mover_parts if p.shape == "cyl")
        cands = {"person": pr["circles"][:npc], "machine": pr["circles"][npc:]}
        best, bg = "static", 0.15
        for k, c in cands.items():
            if len(c):
                g = min(float(circ_gap(x, y, r, c).min()) for x, y, r in fc)
                if g < bg:
                    best, bg = k, g
        if len(pr["boxes"]):
            g = min(float(box_gap(x, y, r, pr["boxes"]).min()) for x, y, r in fc)
            if g < bg:
                best, bg = "vehicle", g
        if self.contacts and t - self.contacts[-1]["t_last"] < 0.5 and self.contacts[-1]["with"] == best:
            self.contacts[-1]["t_last"] = round(t, 3)
            self.contacts[-1]["max_force_n"] = max(self.contacts[-1]["max_force_n"],
                                                  max(c.get("force_n", 0.0) for c in contacts))
            return
        self.contacts.append({"t": round(t, 3), "t_last": round(t, 3), "with": best, "gap_m": round(bg, 3),
                              "bodies": sorted({c.get("body") for c in contacts}),
                              "max_force_n": max(c.get("force_n", 0.0) for c in contacts)})

    # ------------------------------------------------------------------ robot camera
    def _gate(self, t: float) -> None:
        if self.caster is None or self.robot is None:
            self.gate.t_next = t + CAM.GATE_DT
            return
        rx, ry, rpsi, *_, rz = self.robot
        cam = CAM.cam_pose(rx, ry, rz, rpsi)
        bodies = self._machine_circles(t)
        ms = []
        for m in self.scn.machines:
            x0, y0, x1, y1 = m.rect
            d = math.hypot(max(x0 - rx, 0, rx - x1), max(y0 - ry, 0, ry - y1))
            if d > CAM.GATE_RANGE_M:
                continue
            c = bodies.get(m.id)
            if m.kind == "unload_lifter":
                h = self.kin._compute(self.t0 + t).lift_h or 6.0
                parts = np.array([(m.x + dx, m.z, min(h, 6.0)) for dx in (-1.533, 0.0, 1.533)])
            elif c is not None and len(c):
                keep = [i for i in range(len(c)) if i % 5 != 0]       # drop the static arm bases
                parts = np.column_stack((c[keep, 0], c[keep, 1], np.full(len(keep), CAM.PART_Z)))
            else:
                parts = np.zeros((0, 3))
            ms.append((m.id, parts, np.array((m.x, m.z, CAM.PART_Z))))
        if not ms:
            self.gate.t_next = t + CAM.GATE_DT
            return
        self.gate.observe(t, cam, ms, self.caster, lambda mid: self.sensor_prims(t, skip_machine=mid))

    def _captures_step(self, t: float, st: dict) -> None:
        if not self.pending:
            return
        v = math.hypot(st["v_w"][0], st["v_w"][1])
        w = abs(float(st["omega_b"][2]))
        for pc in list(self.pending):
            if v < CAM.HOLD_V and w < CAM.HOLD_W:
                pc["hold_since"] = pc["hold_since"] if pc["hold_since"] is not None else t
                if t - pc["hold_since"] >= CAM.HOLD_S:
                    self._capture(t, pc, settled=True, st=st)
            else:
                pc["hold_since"] = None

    def _capture(self, t: float, pc: dict, settled: bool, st: dict | None = None) -> None:
        if pc in self.pending:
            self.pending.remove(pc)
        tg = self.targets.get(pc["target"])
        rx, ry, rpsi, vx, vy, wz, rz = self.robot
        rec = {"target": pc["target"], "machine": pc["machine"], "inside": pc["inside"], "settled": settled,
               "t_dwell_start": round(pc["t_start"], 3), "robot_xy": [round(rx, 3), round(ry, 3)],
               "robot_psi": round(rpsi, 4)}
        if tg is None or self.caster is None:
            rec.update({"pass": False, "error": "unknown target" if tg is None else "no caster"})
        else:
            cam = CAM.cam_pose(rx, ry, rz, rpsi)
            rec.update(CAM.capture(t, cam, tg["xyz"], tg["normal"], np.array((vx, vy, 0.0)), wz, self.caster,
                                   self.sensor_prims(t), self.cap_rng))
        rec["source"] = CAM.SOURCE
        self.captures.append(rec)
        self.outbox_cap.append({k: v for k, v in rec.items()})

    # ------------------------------------------------------------------ observer feed + outputs
    def _write_state(self, t: float) -> None:
        """Plant clock + freezes for the episode's observer process (it rebuilds the same World)."""
        self._last_state_write = t
        st = {"t": round(t, 3), "plant_t0": self.t0, "seed": self.seed,
              "freezes": {g: [[f.t0, f.t1, f.kind, f.machine] for f in grp.freezes] for g, grp in self.world.groups.items()}}
        try:
            tmp = os.path.join(self.work, "plant_state.json.tmp")
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, os.path.join(self.work, "plant_state.json"))
        except OSError:
            pass

    def lost_time(self) -> dict:
        """Union of freeze time inside the episode per group (trips: protective stop + 90 s recovery; holds)."""
        a, b = self.t0, self.t0 + self.t
        out = {}
        for gname, g in self.world.groups.items():
            iv = sorted((max(a, f.t0), min(b, f.t1), f.kind) for f in g.freezes if f.t1 > a and f.t0 < b)
            tot, cur0, cur1 = 0.0, None, None
            for s, e, _ in iv:
                if cur1 is None or s > cur1:
                    if cur1 is not None:
                        tot += cur1 - cur0
                    cur0, cur1 = s, e
                else:
                    cur1 = max(cur1, e)
            if cur1 is not None:
                tot += cur1 - cur0
            out[gname] = {"lost_s": round(tot, 2),
                          "trip_s": round(sum(min(b, f.t1) - max(a, f.t0) for f in g.freezes if f.kind == "trip" and f.t1 > a and f.t0 < b), 2),
                          "hold_s": round(sum(min(b, f.t1) - max(a, f.t0) for f in g.freezes if f.kind == "hold" and f.t1 > a and f.t0 < b), 2),
                          "trips": sum(1 for f in g.freezes if f.kind == "trip" and a <= f.t0 < b),
                          "holds": sum(1 for f in g.freezes if f.kind == "hold" and a <= f.t0 < b)}
        return out

    def finish(self, out_dir: str, sim_summary: dict | None = None) -> dict:
        lt = self.lost_time()
        line = lt.get("line", {})
        cells = {g: v for g, v in lt.items() if g != "line"}
        fin = lambda v: None if v == math.inf else round(v, 3)  # noqa: E731
        summ = {
            "seed": self.seed, "plant_t0": self.t0, "sim_s": round(self.t, 2), "invisible_robot": self.invisible,
            "n_people": self.np_, "n_impassive": int((~self.reacts).sum()), "n_vehicles": len(self.vehicles),
            "trips": sum(1 for e in self.events if e["type"] == "trip"),
            "trips_by_zone": {z: sum(1 for e in self.events if e["type"] == "trip" and e["zone"] == z) for z in self.zones
                              if any(e["type"] == "trip" and e["zone"] == z for e in self.events)},
            "hold_events": sum(1 for e in self.events if e["type"] == "hold"),
            "zone_entries": sum(1 for e in self.events if e["type"] == "enter"),
            "line_time_lost_s": line.get("lost_s", 0.0), "cell_time_lost_s": round(sum(v["lost_s"] for v in cells.values()), 2),
            "lost_by_group": lt,
            "contacts": len(self.contacts),
            "contacts_by": {k: sum(1 for c in self.contacts if c["with"] == k) for k in ("person", "vehicle", "machine", "static")},
            "overlaps_by": dict(self.overlap),
            "min_gap_m": {k: fin(self.mins[k]) for k in ("person", "vehicle", "machine", "static")},
            "min_ttc_s": {"person": fin(self.mins["ttc_person"]), "vehicle": fin(self.mins["ttc_vehicle"])},
            "personal_space_events": self.ps_events, "personal_space_s": round(self.ps_time, 1),
            "forklift_brakes": sum(1 for e in self.events if e["type"] == "forklift_brake"),
            "agv_protective_stops": sum(1 for e in self.events if e["type"] == "agv_protective_stop"),
            "captures": len(self.captures), "captures_pass": sum(1 for c in self.captures if c.get("pass")),
            "camera_source": CAM.SOURCE, "camera_gate_obs": self.gate.n_obs,
            "plant_wall_s": round(self.wall_plant, 2), "plant_ticks": self.n_ticks,
            "freezes": {g: [[round(f.t0, 3), round(f.t1, 3), f.kind, f.machine] for f in grp.freezes]
                        for g, grp in self.world.groups.items() if grp.freezes},
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plant_log.json"), "w") as f:
            json.dump(self.rows, f)
        with open(os.path.join(out_dir, "plant_events.json"), "w") as f:
            json.dump({"events": self.events, "contacts": self.contacts, "captures": self.captures}, f, default=float)
        with open(os.path.join(out_dir, "plant_summary.json"), "w") as f:
            json.dump(summ, f, indent=1, default=float)
        return summ


def make(cfg: dict) -> Plant:
    return Plant(cfg)
