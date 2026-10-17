"""Vectorised mover engine: people and vehicles with collision bodies, lanes and yielding.

Rules (realistic plant traffic, all deterministic):
- everyone keeps right of their route (vehicles 0.9 m, people 0.5 m), so opposing flows pass;
- each mover has a forward protective field; vehicles stop (or slow) for any body in it, including the robot;
- people additionally step sideways (up to 1.2 m, at 0.6 m/s) toward the freer side to walk around;
- every body overlap is an incident, counted once per pair per contact.
"""
from __future__ import annotations

import math

import numpy as np

from intercycle.sim import geometry as G
from intercycle.truth.agents import PathAgent
from intercycle.types import MoverClass

CLS = {MoverClass.PERSON: "person", MoverClass.AGV: "agv", MoverClass.TUGGER: "tugger", MoverClass.FORKLIFT: "forklift"}
LANE = {"person": 0.5, "agv": 0.9, "tugger": 0.9, "forklift": 0.9}
# forward field: (front offset, stop distance, slow distance, half width)
FIELD = {"robot": (0.53, 0.6, 2.0, 0.5), "person": (0.3, 0.45, 1.0, 0.32), "agv": (0.8, 1.2, 2.5, 0.6),
         "tugger": (0.7, 1.5, 3.0, 0.6), "forklift": (2.05, 1.5, 3.0, 0.65)}
DAY = 86_400.0


def field_template(cls: str, which: str) -> np.ndarray:
    front, stop, slow, hw = FIELD[cls]
    reach = stop if which == "stop" else slow
    ds = np.arange(front + hw * 0.5, front + reach + 1e-6, max(0.2, hw * 0.8))
    return np.stack([np.zeros_like(ds), ds, np.full_like(ds, hw)], axis=1)        # local: x right, z forward


def place(tpl: np.ndarray, x: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(tpl)
    out[:, 0] = x + tpl[:, 0] * c + tpl[:, 1] * s
    out[:, 1] = z - tpl[:, 0] * s + tpl[:, 1] * c
    out[:, 2] = tpl[:, 2]
    return out


def gap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.inf
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) - a[:, None, 2] - b[None, :, 2]
    return float(d.min())


class MoverSystem:
    def __init__(self, ag: list[PathAgent], sm: G.StaticMap, t0: float):
        self.ag, self.sm = ag, sm
        self.n = len(ag)
        self.cls = [CLS.get(a.cls, "person") for a in ag]
        self.tau = np.full(self.n, t0)
        self.lat = np.zeros(self.n)
        self.lat_target = np.zeros(self.n)
        self.blocked_s = np.zeros(self.n)
        self.blocked_s_run = np.zeros(self.n)
        self.yield_until = np.zeros(self.n)
        self.present = np.zeros(self.n, bool)
        self.pose = np.zeros((self.n, 4))                     # x, z, yaw, speed (after lanes)
        self.body_tpl = {c: G.footprint(c, 0.0, 0.0, 0.0) for c in set(self.cls)}
        self.stop_tpl = {c: field_template(c, "stop") for c in set(self.cls)}
        self.slow_tpl = {c: field_template(c, "slow") for c in set(self.cls)}
        self.bodies: list[np.ndarray] = [np.zeros((0, 3))] * self.n
        self.contacts: set[tuple[int, int]] = set()
        self.incidents = 0
        # knot tables (time on own clock -> position), pauses as repeated knots
        self.knots = []
        for a in ag:
            pts = a._pts
            tk, xk, zk = [0.0], [pts[0][0]], [pts[0][1]]
            t = 0.0
            for (p, q), sl in zip(zip(pts, pts[1:]), a._seg):
                if a.pause_s > 0:
                    t += a.pause_s
                    tk.append(t), xk.append(p[0]), zk.append(p[1])
                t += sl / a.speed
                tk.append(t), xk.append(q[0]), zk.append(q[1])
            self.knots.append((np.array(tk), np.array(xk), np.array(zk), a._cycle, a.phase_s, a.speed))

    def _route_pose(self, i: int) -> tuple[float, float, float, float]:
        tk, xk, zk, cyc, ph, sp = self.knots[i]
        u = (self.tau[i] + ph) % cyc
        j = min(int(np.searchsorted(tk, u, side="right")) - 1, len(tk) - 2)
        dt = tk[j + 1] - tk[j]
        f = (u - tk[j]) / dt if dt > 0 else 0.0
        x = xk[j] + (xk[j + 1] - xk[j]) * f
        z = zk[j] + (zk[j + 1] - zk[j]) * f
        moving = (xk[j + 1] != xk[j]) or (zk[j + 1] != zk[j])
        k = j if moving else min(j + 1, len(tk) - 2)
        yaw = math.atan2(xk[k + 1] - xk[k], zk[k + 1] - zk[k]) if (xk[k + 1] != xk[k] or zk[k + 1] != zk[k]) else 0.0
        return x, z, yaw, (sp if moving else 0.0)

    def step(self, t: float, dt: float, extra_bodies: list[np.ndarray], vehicle_only: np.ndarray | None = None):
        """extra_bodies: robot footprint etc. (seen by everyone); vehicle_only: machine parts (vehicles yield to them)."""
        tod = t % DAY
        for i, a in enumerate(self.ag):
            self.present[i] = (not a.windows) or any(w0 <= tod < w1 for w0, w1 in a.windows)
        idx = np.flatnonzero(self.present)
        for i in idx:
            x, z, yaw, sp = self._route_pose(i)
            base = self.ag[i].lane if self.ag[i].lane is not None else LANE[self.cls[i]]
            off = base + self.lat[i]
            c, s = math.cos(yaw), math.sin(yaw)
            x, z = x + off * c, z - off * s                # keep right
            self.pose[i] = (x, z, yaw, sp)
            self.bodies[i] = place(self.body_tpl[self.cls[i]], x, z, yaw)
        for i in np.flatnonzero(~self.present):
            self.bodies[i] = np.zeros((0, 3))
        P = self.pose[idx, :2]
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) if len(idx) else np.zeros((0, 0))
        extra = np.vstack([b for b in extra_bodies if len(b)]) if any(len(b) for b in extra_bodies) else np.zeros((0, 3))
        for a_i, i in enumerate(idx):
            x, z, yaw, sp = self.pose[i]
            cls = self.cls[i]
            rate = 1.0
            if sp > 0:
                near = [self.bodies[idx[b]] for b in np.flatnonzero((D[a_i] < 7.0) & (np.arange(len(idx)) != a_i))]
                if len(extra):
                    ne = extra[np.hypot(extra[:, 0] - x, extra[:, 1] - z) < 7.0]
                    if len(ne):
                        near.append(ne)
                if vehicle_only is not None and cls != "person" and len(vehicle_only):
                    nv = vehicle_only[np.hypot(vehicle_only[:, 0] - x, vehicle_only[:, 1] - z) < 7.0]
                    if len(nv):
                        near.append(nv)
                if near:
                    ob = np.vstack(near)
                    if gap(place(self.stop_tpl[cls], x, z, yaw), ob) < 0:
                        rate = 0.0
                    elif gap(place(self.slow_tpl[cls], x, z, yaw), ob) < 0:
                        rate = 0.5
                    if rate < 1.0 and cls != "person":          # vehicles hold their lane; ask people in the way to step aside
                        f = place(self.slow_tpl[cls], x, z, yaw)
                        for b in np.flatnonzero((D[a_i] < 7.0) & (np.arange(len(idx)) != a_i)):
                            j = idx[b]
                            if self.cls[j] == "person" and gap(f, self.bodies[j]) < 0:
                                self._step_aside(j, x, z, yaw, t)
                    if cls == "person" and rate < 1.0:          # walk around: pick the freer side
                        c, s = math.cos(yaw), math.sin(yaw)
                        best, side_room = 0.0, -1.0
                        for side in (1.0, -1.0):
                            px, pz = x + side * 1.0 * c + s * 0.8, z - side * 1.0 * s + c * 0.8
                            room = min(float(self.sm.clearance(np.array([[px, pz]]))[0]),
                                       float(np.hypot(ob[:, 0] - px, ob[:, 1] - pz).min() - ob[:, 2].max()))
                            if room > side_room:
                                best, side_room = side, room
                        if side_room > 0.35:
                            lim = 1.2 if cls == "person" else 1.0
                            self.lat_target[i] = float(np.clip(self.lat[i] + best * lim, -lim, lim))
                            if cls == "person":
                                rate = max(rate, 0.5)
                    elif rate == 1.0 and t > self.yield_until[i]:
                        self.lat_target[i] *= 0.98
            dl = self.lat_target[i] - self.lat[i]
            lat_v = 0.8 if cls == "person" else 0.0
            new_lat = self.lat[i] + float(np.clip(dl, -lat_v * dt, lat_v * dt))
            c, s = math.cos(yaw), math.sin(yaw)
            nb = place(self.body_tpl[cls], x + (new_lat - self.lat[i]) * c, z - (new_lat - self.lat[i]) * s, yaw)
            others = [self.bodies[idx[b]] for b in np.flatnonzero((D[a_i] < 4.0) & (np.arange(len(idx)) != a_i))]
            if others:
                ob_all = np.vstack(others)
                g_new, g_old = gap(nb, ob_all), gap(self.bodies[i], ob_all)
                free_dyn = g_new > 0.05 or g_new > g_old + 1e-3          # free, or at least moving apart
            else:
                free_dyn = True
            if float((self.sm.clearance(nb[:, :2]) - nb[:, 2]).min()) > 0.05 and free_dyn:
                self.lat[i] = new_lat                   # lateral move only into free space
            self.blocked_s_run[i] = self.blocked_s_run[i] + dt if rate == 0.0 else 0.0
            if rate < 1.0:
                self.blocked_s[i] += dt * (1 - rate)
            self.tau[i] += dt * rate
        # incidents: overlapping bodies, counted once per contact
        cur = set()
        for a_i in range(len(idx)):
            for b_i in np.flatnonzero(D[a_i, a_i + 1:] < 4.0) + a_i + 1:
                i, j = idx[a_i], idx[b_i]
                if gap(self.bodies[i], self.bodies[j]) < -0.05:
                    cur.add((i, j))
        self.incidents += len(cur - self.contacts)
        self.contacts = cur

    def ask_yield(self, field: np.ndarray, x: float, z: float, yaw: float, t: float) -> None:
        """A vehicle or the robot whose field touches a person asks them to step aside (people do, in plants)."""
        for j in np.flatnonzero(self.present):
            if self.cls[j] == "person" and abs(self.pose[j][0] - x) < 6 and gap(field, self.bodies[j]) < 0:
                self._step_aside(j, x, z, yaw, t)

    def _step_aside(self, j: int, x: float, z: float, yaw: float, t: float) -> None:
        """Person j moves away from the line of travel of a body at (x, z) heading `yaw`, in j's own lateral frame."""
        pj = self.pose[j]
        side = 1.0 if ((pj[0] - x) * math.cos(yaw) - (pj[1] - z) * math.sin(yaw)) >= 0 else -1.0
        away = (side * math.cos(yaw), -side * math.sin(yaw))               # world direction away from the path
        own = (math.cos(pj[2]), -math.sin(pj[2]))                          # person's own lateral axis
        dot = away[0] * own[0] + away[1] * own[1]
        if abs(dot) < 0.3:                                                 # walking across the path: just stop
            return
        self.lat_target[j] = float(np.clip(self.lat[j] + math.copysign(1.4, dot), -1.4, 1.4))
        self.yield_until[j] = t + 8.0

    def incident_pairs(self) -> list[tuple[str, str]]:
        return [(self.cls[i], self.cls[j]) for i, j in self.contacts]

    def all_bodies(self) -> np.ndarray:
        b = [x for x in self.bodies if len(x)]
        return np.vstack(b) if b else np.zeros((0, 3))

    def snapshot(self) -> list[tuple]:
        return [(self.ag[i].id, self.cls[i], *self.pose[i]) for i in np.flatnonzero(self.present)]
