"""A-lite M-W4 scenario (PLAN_TWIN §7 item 4; truth side: the harness that injects the changes and scores the map).

Changes injected on the simulated body shop after the observer has learned the floor:
- a program change: station S60 switches to a new car-model program (programs.ProgramKinematics model C: model B's
  weld points in reverse order, shifted 0.4 m along the line) on every cycle from t_change on;
- new arms placed 0.5, 1.0, 1.5 and 2.0 m from an existing cell (the sweep line of the new arm's tool runs that far
  outside the edge of HEM1 or SEAL1): a 2-link arm (2.6 m links, the body-shop arm radii) whose tool sweeps +-1.5 m
  along the edge for half of its own cycle (NEW_ARM_PERIODS) and is stowed over its base for the other half.
  Inside zone + 1 m (the 0.5 m arm) only the in-zone detector (S7a, change_detect_l1) can see it; further out the
  static-space detector (observer/change.py) does. The new arm is handed to the camera as a part the plant does not
  know (AlitePlant.extra_parts).

Measured per change, every second after it appears: the share of the 0.25 m cells the change truly occupies that the
map does NOT broadcast FREE (inside the attributed machine's L1 envelope for its current phase bin, inside a box of
map.unknown_rects, or in an offline camera's region) and the share inside UNKNOWN boxes alone. Reported:
- t_cover_s: the first time after which every occupied cell is not FREE for the next 60 s (inf if never);
- t_unknown_s: the same with UNKNOWN boxes alone (the robot receives them at its next mission start: C-W7 is cut);
- the mean shares over 0-1, 1-5 and 5-30 min.
False alarms: in-zone (inzone_change) and static-space (change_detected) events BEFORE t_change, over the hours the
detectors ran. Everything is SIMULATED; the camera is the measured-error shim; descriptive only.
"""
from __future__ import annotations

import math

import numpy as np

from intercycle.alite.observer import AliteObserver
from intercycle.alite.plant import AlitePlant
from intercycle.observer import mapping as M
from intercycle.scenario import Scenario
from intercycle.truth import kinematics as K

PROGRAM_CHANGE_STATION = "S60"
NEW_ARM_PERIODS = (50.0, 55.0, 37.0, 42.0)
NEW_ARMS = (                       # (name, near machine, distance m, edge, period s)
    ("new_arm_HEM1_east_0.5m", "HEM1", 0.5, "east", NEW_ARM_PERIODS[0]),
    ("new_arm_SEAL1_north_1.0m", "SEAL1", 1.0, "north", NEW_ARM_PERIODS[1]),
    ("new_arm_HEM1_west_1.5m", "HEM1", 1.5, "west", NEW_ARM_PERIODS[2]),
    ("new_arm_SEAL1_east_2.0m", "SEAL1", 2.0, "east", NEW_ARM_PERIODS[3]),
)
REACH_BACK_M = 3.0                 # the new arm's base sits this far further out than its tool's sweep line (less
                                   # where the hall wall is closer: base >= 0.7 m inside the wall)
SWEEP_M = 1.5
HOLD_S = 60.0


class NewArm:
    """A 2-link arm the plant does not know: base outside the cell edge, tool sweeping along the edge."""

    def __init__(self, scn: Scenario, near: str, dist: float, edge: str, period: float, t_start: float):
        x0, z0, x1, z1 = scn.machine(near).rect
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        out = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}[edge]
        hx, hz = scn.hall["x"], scn.hall["z"]
        if out[0]:
            line = (x1 + dist) if out[0] > 0 else (x0 - dist)
            back = min(REACH_BACK_M, ((hx[1] - line) if out[0] > 0 else (line - hx[0])) - 0.7)
            self.tool0 = np.array([line, cz])
            self.axis = np.array([0.0, 1.0])
            self.base = np.array([line + out[0] * back, cz])
        else:
            line = (z1 + dist) if out[1] > 0 else (z0 - dist)
            back = min(REACH_BACK_M, ((hz[1] - line) if out[1] > 0 else (line - hz[0])) - 0.7)
            self.tool0 = np.array([cx, line])
            self.axis = np.array([1.0, 0.0])
            self.base = np.array([cx, line + out[1] * back])
        self.period, self.t_start = float(period), float(t_start)
        self.radii = np.array(K.ARM_R, float)[None, None, :]

    def circles(self, t: float) -> np.ndarray:
        u = ((t - self.t_start) % self.period) / self.period
        if u < 0.5:
            tool = self.tool0 + self.axis * SWEEP_M * math.sin(2 * math.pi * u / 0.5)
        else:
            tool = self.base.copy()                          # stowed over its base
        c = K._chain_circles(self.base[None, None, :], tool[None, None, :], self.radii, np.array([[-1.0]]))
        return c.reshape(-1, 3)


def occupied_cells(circles: np.ndarray, res: float = 0.25) -> np.ndarray:
    """Centres of the res-m cells whose centre lies inside any circle."""
    pts = []
    for x, z, r in np.asarray(circles, float).reshape(-1, 3):
        i0, i1 = int(math.floor((x - r) / res)), int(math.floor((x + r) / res))
        j0, j1 = int(math.floor((z - r) / res)), int(math.floor((z + r) / res))
        gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        cx, cz = (gi + 0.5) * res, (gj + 0.5) * res
        k = np.hypot(cx - x, cz - z) <= r
        pts.append(np.stack([cx[k], cz[k]], 1))
    return np.unique(np.vstack(pts), axis=0) if pts else np.zeros((0, 2))


def in_rects(pts: np.ndarray, rects) -> np.ndarray:
    if not len(rects) or not len(pts):
        return np.zeros(len(pts), bool)
    R = np.array(rects, float)
    return ((pts[:, :1] >= R[None, :, 0]) & (pts[:, :1] <= R[None, :, 2]) &
            (pts[:, 1:] >= R[None, :, 1]) & (pts[:, 1:] <= R[None, :, 3])).any(1)


def in_envelope(mp: M.LayeredMap, pts: np.ndarray, i: int, b: int) -> np.ndarray:
    out = np.zeros(len(pts), bool)
    if b < 0 or not len(pts):
        return out
    env = mp.envelope(mp.spec["machines"][i]["id"], b)
    res = mp.spec["env_res"]
    ix = np.floor((pts[:, 0] - mp.l1_x0[i]) / res).astype(int)
    iz = np.floor((pts[:, 1] - mp.l1_z0[i]) / res).astype(int)
    ok = (ix >= 0) & (ix < env.shape[0]) & (iz >= 0) & (iz < env.shape[1])
    out[ok] = env[ix[ok], iz[ok]]
    return out


def cam_offline(mp: M.LayeredMap, pts: np.ndarray) -> np.ndarray:
    if not mp.cam_offline or not len(pts):
        return np.zeros(len(pts), bool)
    s = mp.spec["site"]
    ix = np.clip(np.floor((pts[:, 0] - s["x0"]) / s["res"]).astype(int), 0, mp.nx - 1)
    iz = np.clip(np.floor((pts[:, 1] - s["z0"]) / s["res"]).astype(int), 0, mp.nz - 1)
    return np.isin(mp.cam_of_cell[ix, iz], list(mp.cam_offline))


def first_full(series: list[tuple[float, float]], hold_s: float = HOLD_S) -> float:
    """First time after which every sample of the next hold_s seconds has share 1 (inf if never)."""
    ts = np.array([dt for dt, _ in series])
    ok = np.array([s >= 1.0 for _, s in series])
    if not len(ts):
        return math.inf
    bad = np.flatnonzero(~ok)
    for j in range(len(ts)):
        if not ok[j] or ts[j] + hold_s > ts[-1]:
            continue
        nxt = bad[bad > j]
        if not len(nxt) or ts[nxt[0]] >= ts[j] + hold_s:
            return float(ts[j])
    return math.inf


def _win(series, a, b) -> float:
    v = [s for dt, s in series if a <= dt < b]
    return float(np.mean(v)) if v else float("nan")


def run(scn: Scenario, world_seed: int, t_learn0: float, learn_s: float, after_s: float,
        progress: bool = False) -> dict:
    """Learn the floor for learn_s from t_learn0, inject every change at t_change = t_learn0 + learn_s, run after_s."""
    t_ch = t_learn0 + learn_s
    t_end = t_ch + after_s
    arms = {name: NewArm(scn, near, d, edge, P, t_ch) for name, near, d, edge, P in NEW_ARMS}
    near_of = {name: near for name, near, *_ in NEW_ARMS}

    def extra(t: float) -> dict[str, np.ndarray]:
        return {name: a.circles(t) for name, a in arms.items()} if t >= t_ch else {}

    plant = AlitePlant(scn, world_seed, t_learn0, t_end, program_change={PROGRAM_CHANGE_STATION: t_ch},
                       extra_parts=extra)
    ob = AliteObserver(scn)
    ids = ob.ids
    cover = {n: [] for n in [PROGRAM_CHANGE_STATION + "_program_change", *arms]}
    unk = {n: [] for n in cover}
    hz = M.FRAME_HZ
    n_ticks = int(round((t_end - t_learn0) * hz))
    pre_events: list[dict] = []
    post_events: list[dict] = []
    for k in range(n_ticks):
        t = t_learn0 + k / hz
        n_ev = len(ob.events)
        cond = ob.step(plant.observe(t))
        new = [e for e in ob.events[n_ev:] if e.get("kind") in ("inzone_change", "change_detected")]
        (pre_events if t < t_ch else post_events).extend(new)
        if progress and k % int(3600 * hz) == 0:
            print(f"  world {world_seed} t={t / 3600:.2f} h", flush=True)
        if t < t_ch or k % int(hz):
            continue
        mp = ob.map
        rects = list(mp.unknown_rects)
        tc = float(cond.t)
        for name in cover:
            if name in arms:
                circ = arms[name].circles(tc)
                i = ids.index(near_of[name])
            else:
                circ = np.asarray(plant.kin.bodies(tc)[PROGRAM_CHANGE_STATION])
                i = ids.index(PROGRAM_CHANGE_STATION)
            occ = occupied_cells(circ)
            if not len(occ):
                continue
            box = in_rects(occ, rects) | cam_offline(mp, occ)
            nf = box | in_envelope(mp, occ, i, int(cond.phase_bin[i]))
            cover[name].append((tc - t_ch, float(nf.mean())))
            unk[name].append((tc - t_ch, float(box.mean())))
    out = {"world_seed": world_seed, "t_change": t_ch, "learn_h": learn_s / 3600, "after_min": after_s / 60,
           "false_alarms_before_change": len(pre_events), "false_alarm_kinds": sorted({e["kind"] for e in pre_events}),
           "detector_hours_before_change": learn_s / 3600, "changes": {}}
    for name in cover:
        c, u = cover[name], unk[name]
        out["changes"][name] = {
            "t_cover_s": first_full(c), "t_unknown_s": first_full(u),
            "cover_0_1min": _win(c, 0, 60), "cover_1_5min": _win(c, 60, 300), "cover_5_30min": _win(c, 300, 1800),
            "unknown_0_1min": _win(u, 0, 60), "unknown_1_5min": _win(u, 60, 300), "unknown_5_30min": _win(u, 300, 1800),
            "n_samples": len(c)}
    out["events_after_change"] = [(round(e["t"] - t_ch, 1), e["kind"]) for e in post_events][:40]
    return out
