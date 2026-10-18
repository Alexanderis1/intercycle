"""The twin plant (lane S5a; PLAN_TWIN §2 "runs/<id>/...", §4 S5a items 3-4; WEAKNESSES W-W4, R-W2, R-W3, R-W10).

    cfg["mover_source"] = "twin.plant:make"          (runs inside intercycle_sim.runner, Isaac or the CPU double)

Everything around the robot in one press-shop zone mission, on `twin/engine.py`. There is no WALK zone, no line
walkway, no body-shop kinematics and no body-shop truth agents: presses, people and forklifts come from the day event
script only, and the press ram is the only moving machine part.

cfg["plant"] (fail closed: the EVAL identity fields have no defaults): site, safety, day_script, run_id, zone, world,
world_split, condition, code{head, ...}; optional tasks (twin/tasks.yaml: checked against the site), dynamics,
plant_t0, run_dir, obs_presses / obs_zones, reset_override_s (the U4 descriptive reset sweep, see below),
allow_short_mission (dev only, see "Mission length").

Truth and files (the plant's output reaches the observer side ONLY through files):
  * `runs/<id>/realised_state.jsonl` (TRUTH, for EVAL): the engine's 2 Hz stream over the mission, the robot added as
    a mover of class `robot`, written through `twin.engine.RealisedWriter` (fail closed: published on `finish` only).
  * `runs/<id>/obs_v2.jsonl`: the camera model runs IN the plant process and is the only consumer of the realised
    ticks (twin/camera.py stat_v1, `make_camera`; twin/camera_v0.py MEASURED_SHIM stays as the G0 reference). Each
    tick becomes one Observation v2 line (T-W6 hold = the camera's fitted `hold_s`) (`intercycle.observer.mapping.obs_to_dict`), written to the file only once the sim clock has
    reached its delivery time `t` (= t_cap + latency), so a consumer tailing the file live never sees an observation
    early. Lines still undelivered when the mission ends are written by `finish`. The broadcaster and observer_live
    read this file only; the plant writes no plant_state.json (the body-shop feed) and sends ROS nothing but the
    robot camera's no-motion gate observations and inspection outcomes (`frame_extras`).
  * `runs/<id>/log.parquet` (`twin.schemas.tables.LogRow`; `log.jsonl` with the same rows where pyarrow is missing,
    i.e. in the Isaac python) and `runs/<id>/summary.json` (`twin.schemas.runs.Summary`), plus `plant_summary.json`
    (diagnostics that the Summary contract has no field for). `finish` is fail closed: any exception while it closes
    the mission still writes summary.json with status "error" (status_reason = the exception) and the log rows
    gathered so far, so an errored mission is scored as a failure (PLAN_TWIN §5 Rules), never as missing logs.

Checkpoints (tasks.yaml): only the mission zone's `mission_order` counts in attempted / passed / declined; any other
target in a dwell_start / decline event is an `error` row and sets status "error".

Safety (`twin/safety.yaml`, U4, W-W4 / R-W2), checked every SAFETY_DT on the robot footprint (3 circles) against each
press's danger polygon dilated by its `dilation_m`:
  * entry: the press's entry rule for its displayed state decides. `trip` = protective stop: the press is frozen
    `tripped` while the robot is inside, then `resetting` for a manual reset drawn from the press's lognormal reset
    distribution (U4: median 240 s, p90 900 s for every ASSUMED press), starting when the robot has left (the reset is
    done from outside the guarding). Draws: `twin.rng.keyed_rng(world_seed, day, "reset:<press>", "t0=<plant_t0>:trip:<n>")`,
    so paired conditions share the n-th reset (common random numbers).
  * hold: while the robot is inside, a restart that falls due (a scheduled running bout begins, or a reset ends) is
    delayed: the press is frozen `held` while the robot stays inside, at most to the end of that running bout. One
    `press_hold` row per delayed restart, valued at the measure of its held freezes.
  * latched restart interlock (W-W4, U4 "light curtain plus manual reset", R-W2(b) "manual reset on ANY entry"): a
    light curtain with manual reset cannot sense the robot inside, so EVERY entry latches the restart interlock,
    whatever the press state: after any visit that did not trip it, the press restarts only after a manual reset from
    outside, drawn like the trip reset (key "t0=<plant_t0>:entry_reset:<n>"), state `resetting`, starting when the
    robot has left. A reset that delays a scheduled restart is a hold (one press_hold row, value = the delayed
    running seconds, unless that restart was already held during the visit). So leaving a stopped press just before
    its restart costs about as much as leaving just after it (no cost cliff at the restart). Not latched: an entry
    the rules `allow` into a running press (a muted curtain: an incident instead). Other guards (auto reset) restart
    as soon as the robot has left.
  * incident: the robot inside the danger zone of a press that is running unfrozen (an entry the rules allow), or a
    geometric overlap with a person or a forklift while the robot moves (> MOVING_MPS). Stationary overlaps (someone
    walking into a parked robot) are logged as contacts, not incidents.
  Robot-caused stops (trips), holds and incidents go to log.parquet and summary.json; any of them zeroes the mission's
  Y (PLAN_TWIN §5). Their rows carry the checkpoint the robot was working on (`_checkpoint_for`). press_time_lost_s =
  the time the robot's freezes (tripped + reset + held) overlap the scheduled running bouts, reset tails past the
  mission end included (the production is lost either way).
  * reset sweep (U4, descriptive only): cfg["plant"]["reset_override_s"] replaces every manual-reset draw by that fixed
    duration (it must lie in each press's `reset.sweep_s`); the run is tagged a sweep run in summary.json
    (`variant` "reset_sweep" + `reset_override_s`), plant_summary.json, the mission_start row and every reset row.

Mission length: Summary.T_s is the fixed 720 s denominator, so cfg["seconds"] != 720 is refused at init unless
cfg["plant"]["allow_short_mission"] is True on a dev world (tests, the slice); such a run is `variant` "short_dev".
Only `variant` "primary" runs may enter the primary metric (`twin.schemas.runs.require_primary`).

Fail closed during the mission: an exception inside a runner hook (poses, after_step, frame_extras, on_ros) is
recorded, the plant stops its work for the rest of the mission, and `finish` writes summary.json with status "error"
(the runner's loop never sees the exception, so it always reaches `finish`).

World interface (`TwinWorld`, the shape of intercycle.sim.world.World the robot stack uses): active, active_many,
trip, group(...).freezes, groups, true_idle_windows, idle_remaining, next_start_after; one group per press (presses
are off-line machines). Times are absolute engine seconds (the realised stream's clock). `trip` goes through the
plant (logged, counted, and seen by the hold logic); a TwinWorld without a plant refuses it.

Press-ram mover: the ram plate of each press (ASSUMED geometry, RAM_*) moves vertically with the engine's ram position
at any sim time; its points are the targets of the robot's no-motion gate (twin/robot_camera.py). It is not an Isaac
collider (intercycle_sim movers move in the floor plane only; the press body is the static box in boxes.json).

R-W3 hook: `make_gate` builds the robot's no-motion gate. Lane S2 delivers the R-W3 camera-gate module; it replaces the
default (the body-shop near-perfect gate, disclosed) by one import in `make_gate`, with the same interface
(due / observe / ready / n_obs / source).
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_args

import numpy as np

from intercycle.observer import mapping as OM
from twin import robot_camera as RC
from twin.camera import CameraStatV1, Detections
from twin.engine import Engine, RealisedWriter, load_day_script
from twin.rng import check_world_seed, keyed_rng
from twin.schemas import load
from twin.schemas.common import MISSION_T_S, TICK_HZ, Condition
from twin.schemas.dynamics import Dynamics
from twin.schemas.runs import CodeHashes, RealisedMover, Summary
from twin.schemas.safety import PressSafety, Safety
from twin.schemas.site import Site, point_in_polygon
from twin.schemas.tables import LogRow
from twin.schemas.tasks import Tasks

SAFETY_DT = 0.1                   # safety / zone check period (s)
FOOT = ((-0.34, 0.32), (0.0, 0.32), (0.34, 0.32))   # ANYmal-D footprint ~1.0 x 0.64 m as 3 circles (fwd offset, r)
ROBOT_ID, ROBOT_R = "robot", 0.5
MOVING_MPS = 0.1                  # robot speed above which an overlap is robot-caused (incident)
OVERLAP_M = 0.02                  # geometric overlap = gap below -OVERLAP_M
PERSON_R, PERSON_H = 0.3, 1.75
FORKLIFT_BOXES = ((0.0, 1.9, 1.1, 2.2), (1.5, 1.1, 0.9, 1.0))   # (fwd, length, width, height): intercycle_sim PARTS
# Press ram plate (ASSUMED: not observable from 4 m; only the stroke timing is learned): top dead centre height,
# stroke, half side of the plate square around the footprint centre.
RAM_TDC_Z, RAM_STROKE_M, RAM_HALF_M = 2.4, 0.5, 0.4
RAM_GEOMETRY_SOURCE = "assumed"
OBS_OTHER = 4                     # Observation class code of the robot ("other"; not PERSON / VEHICLE / PART / SKID)
OBS_CLS = {"person": OM.PERSON, "vehicle": OM.VEHICLE, "other": OBS_OTHER}
EPS_T = 1e-6
FREEZE_KIND = {"tripped": "trip", "resetting": "trip", "held": "hold"}
WORLD_SPLITS = ("dev", "test")
CONDITIONS = get_args(Condition)
REQUIRED_IDENTITY = ("world", "world_split", "condition", "code")


# ------------------------------------------------------------------------------------------------ geometry


def _seg_dist(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    s = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / L2))
    return math.hypot(px - ax - s * dx, pz - az - s * dz)


def poly_dist(x: float, z: float, poly) -> float:
    """Distance from (x, z) to the polygon (0 inside)."""
    if point_in_polygon(x, z, poly):
        return 0.0
    n = len(poly)
    return min(_seg_dist(x, z, *poly[i], *poly[(i + 1) % n]) for i in range(n))


def circles_hit_poly(circles, poly, dilation: float) -> bool:
    """Any footprint circle (x, z, r) touches the polygon dilated by `dilation` (exact Minkowski test)."""
    return any(poly_dist(x, z, poly) < r + dilation for x, z, r in circles)


def _bbox(poly, grow: float = 0.0) -> tuple[float, float, float, float]:
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) - grow, min(zs) - grow, max(xs) + grow, max(zs) + grow


def box_gap(px: float, py: float, r: float, boxes: np.ndarray) -> np.ndarray:
    """Signed gaps from a disc to boxes (cx, cy, hw, hd, yaw[, ...]); negative = overlap."""
    if len(boxes) == 0:
        return np.zeros(0)
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    dx, dy = px - boxes[:, 0], py - boxes[:, 1]
    lx, ly = np.abs(dx * c + dy * s), np.abs(-dx * s + dy * c)
    out = np.hypot(np.maximum(lx - boxes[:, 2], 0), np.maximum(ly - boxes[:, 3], 0))
    inside = (lx <= boxes[:, 2]) & (ly <= boxes[:, 3])
    return np.where(inside, -np.minimum(boxes[:, 2] - lx, boxes[:, 3] - ly), out) - r


def yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def plan_heading(psi: float) -> float:
    """ROS yaw (from +x, counter-clockwise) -> plan heading (0 = +z, clockwise towards +x), wrapped to (-pi, pi]."""
    h = math.pi / 2 - psi
    return math.atan2(math.sin(h), math.cos(h))


def floor_share(rects, floors, margin: float = OM.ZONE_MARGIN_M, res: float = 0.25) -> np.ndarray:
    """(M, n_cam) share of each machine zone's floor (rect grown by `margin`, sampled on a `res` grid of cell centres)
    inside each camera's visible-floor polygon: the twin's `zone_camera_share` (mapping.py §v2). A cell seen by several
    cameras counts for the first one, so rows sum to <= 1 (a row < 1 = part of the zone is never seen)."""
    W = np.zeros((len(rects), len(floors)))
    for i, (x0, z0, x1, z1) in enumerate(rects):
        nx = max(1, int(math.ceil((x1 - x0 + 2 * margin) / res)))
        nz = max(1, int(math.ceil((z1 - z0 + 2 * margin) / res)))
        gx = x0 - margin + (np.arange(nx) + 0.5) * (x1 - x0 + 2 * margin) / nx
        gz = z0 - margin + (np.arange(nz) + 0.5) * (z1 - z0 + 2 * margin) / nz
        for px in gx:
            for pz in gz:
                k = next((c for c, poly in enumerate(floors) if point_in_polygon(float(px), float(pz), poly)), None)
                if k is not None:
                    W[i, k] += 1.0
        W[i] /= nx * nz
    return W


# ------------------------------------------------------------------------------------------------ world interface


@dataclass(frozen=True)
class WorldFreeze:
    t0: float
    t1: float
    kind: str           # "trip" (tripped / resetting) | "hold" (held), as intercycle.sim.world.Freeze
    machine: str
    state: str          # the engine's displayed state: tripped / resetting / held


class TwinGroup:
    """One press (an off-line machine is its own clock group). Freezes are read from the engine (truth)."""

    def __init__(self, engine: Engine, pid: str):
        self.name, self._e, self._pid = pid, engine, pid

    @property
    def freezes(self) -> list[WorldFreeze]:
        return [WorldFreeze(f.t0, f.t1, FREEZE_KIND[f.state], f.press, f.state)
                for f in self._e.presses[self._pid].freezes]

    def is_frozen(self, t: float) -> bool:
        return self._e.presses[self._pid].frozen_component(t) is not None

    def frozen_until(self, t: float) -> float | None:
        c = self._e.presses[self._pid].frozen_component(t)
        return None if c is None else c[1]


class TwinWorld:
    """The robot stack's World interface on the twin engine (truth: plant, oracle feed and EVAL only)."""

    def __init__(self, engine: Engine, on_trip: Callable[..., dict] | None = None):
        self.engine = engine
        self.groups = {pid: TwinGroup(engine, pid) for pid in engine.presses}
        self.group_of = {pid: pid for pid in engine.presses}
        self.events: list[dict] = []
        self._on_trip = on_trip           # the Plant's trip handler (logs, counts and freezes the trip)

    def group(self, mid: str) -> TwinGroup:
        return self.groups[self.group_of[mid]]

    @property
    def freezes(self) -> list[WorldFreeze]:
        return [f for g in self.groups.values() for f in g.freezes]

    def scheduled_running(self, mid: str, t: float) -> bool:
        p = self.engine.presses[mid]
        return p.events[p.event_index(t)].kind == "running"

    def active(self, mid: str, t: float) -> bool:
        """The press runs at t (a scheduled running bout, not frozen by a trip, reset or hold)."""
        return self.scheduled_running(mid, t) and not self.group(mid).is_frozen(t)

    def active_many(self, mid: str, ts) -> np.ndarray:
        return np.array([self.active(mid, float(t)) for t in np.asarray(ts, float).reshape(-1)], bool)

    def _active_intervals(self, mid: str, a: float, b: float) -> list[tuple[float, float]]:
        p = self.engine.presses[mid]
        out: list[tuple[float, float]] = []
        for e, e0, e1 in zip(p.events, p.e0, p.e1):
            if e.kind != "running" or e1 <= a or e0 >= b:
                continue
            cur = max(e0, a)
            end = min(e1, b)
            for f0, f1 in p.union:
                if f1 <= cur or f0 >= end:
                    continue
                if f0 > cur:
                    out.append((cur, f0))
                cur = max(cur, f1)
            if cur < end:
                out.append((cur, end))
        return out

    def true_idle_windows(self, mid: str, t: float, horizon: float = 600.0) -> list[tuple[float, float]]:
        """Idle windows from t on (absolute time), assuming no further freezes (what a PLC oracle knows): every window
        whose start is within `horizon` of t, with its full end (a window running past the scripted day ends there).
        The first window starts at t if the press is idle now."""
        end = self.engine.t_end
        act = self._active_intervals(mid, t, end)
        out, cur = [], t
        for a, b in act:
            if a > cur + EPS_T:
                out.append((cur, a))
            cur = max(cur, b)
        if cur < end - EPS_T:
            out.append((cur, end))
        return [w for w in out if w[0] <= t + horizon]

    def idle_remaining(self, mid: str, t: float) -> float:
        ws = self.true_idle_windows(mid, t, 0.0)
        return ws[0][1] - t if ws and ws[0][0] <= t + EPS_T else 0.0

    def next_start_after(self, mid: str, t: float) -> float:
        return t + self.idle_remaining(mid, t)

    def trip(self, mid: str, t: float, who: str, where: str | None = None, reset_s: float | None = None) -> dict:
        """A protective stop of press `mid` at absolute time t, done by the plant (`Plant.world_trip`): counted in
        robot_caused_press_stops, logged (press_trip / press_restart) and seen by the hold logic. A TwinWorld without
        a plant refuses it (fail closed: a trip nobody logs would be lost time without a robot-caused stop)."""
        if self._on_trip is None:
            raise RuntimeError("TwinWorld.trip needs the plant (Plant.world_trip): a trip must be logged and counted")
        ev = self._on_trip(mid, t, who, where, reset_s)
        self.events.append(ev)
        return ev


# ------------------------------------------------------------------------------------------------ safety


class _PressGuard:
    """Per-press safety state for one mission."""

    def __init__(self, ps: PressSafety, poly, dilation: float):
        self.id = ps.press
        self.cfg = ps
        self.rule = {r.press_state: r.action for r in ps.entry_rules}
        self.poly, self.dilation = poly, float(dilation)
        self.bbox = _bbox(poly, self.dilation)
        self.inside = False
        self.entry_state: str | None = None
        self.entry_action: str | None = None
        self.tripped = False
        self.trip_t0 = 0.0
        self.trip_cp: str | None = None
        self.n_trips = 0
        self.holding = False              # a delayed restart is being held now
        self.hold_t0 = 0.0
        self.hold_t1 = 0.0                # end of the latest held freeze of the open hold
        self.hold_s = 0.0                 # measure of the open hold's held freezes
        self.hold_cp: str | None = None
        self.visit_held = False           # a restart was held during the current visit
        self.n_entry_resets = 0           # latched-interlock manual resets drawn so far (key entry_reset:<n>)
        self.incident_entry = False
        self.chunk_end = -math.inf        # end of the latest freeze this plant applied to the press

    @property
    def latching(self) -> bool:
        """A light curtain with manual reset: a curtain break latches the restart interlock (W-W4)."""
        return self.cfg.guard.device == "light_curtain" and self.cfg.guard.reset == "manual"


def reset_draw(ps: PressSafety, world_seed: int, day: int, key: str, override_s: float | None = None) -> float:
    """One manual-reset duration (s) from the press's lognormal reset distribution, keyed (CRN). `override_s` (the U4
    descriptive sweep) replaces the draw by that fixed duration."""
    if override_s is not None:
        return float(override_s)
    g = keyed_rng(world_seed, day, f"reset:{ps.press}", key)
    return float(math.exp(ps.reset.mu + ps.reset.sigma * g.standard_normal()))


# ------------------------------------------------------------------------------------------------ camera -> obs v2


def make_camera(site: Site, world_seed: int, day: int):
    """The twin camera model run in the plant process: stat_v1 (twin/camera.py, lane S3b). The G0 stand-in
    twin/camera_v0.py (MEASURED_SHIM) is kept as the noise reference of tests/test_stat_v1_noise.py."""
    return CameraStatV1(site, world_seed, day)


def make_gate(world_seed: int, day: int, key: str):
    """The robot's no-motion gate. R-W3 hook: lane S2's camera-gate module replaces this default here."""
    return RC.Gate(keyed_rng(world_seed, day, "robot_gate", key))


class ObsAdapter:
    """camera_v0 Detections -> Observation v2 (zones = `press_ids` in order, machine zone = the danger rect; cameras =
    `zone_ids` in order, each camera's visible floor = its zone's floor polygon), per the mapping.py §v2 contract:
    zone_vis = the share of the zone's floor (rect + ZONE_MARGIN_M) that a camera with cam_floor_ok sees (`floor_share`)
    minus the share hidden by person / vehicle detections, and zone_motion NaN where zone_vis < ZONE_VIS_MIN or an
    occluder touches the zone (held, T-W6), through `mapping.ObsV2Converter`. The camera's own NaN (occlusion seen in
    truth, T-W6) is kept. The robot is class OBS_OTHER (never PERSON: it must not enter L3). Stateful (the T-W6 hold):
    call it on every tick in time order."""

    def __init__(self, site: Site, press_ids: list[str] | None = None, zone_ids: list[str] | None = None,
                 hold_s: float = OM.OCC_HOLD_S):
        self.press_ids = list(press_ids) if press_ids else [p.id for p in site.presses]
        self.zone_ids = list(zone_ids) if zone_ids else [z.id for z in site.zones]
        zones = {z.id: z for z in site.zones}
        presses = {p.id: p for p in site.presses}
        self.cams = [zones[z].camera for z in self.zone_ids]
        self.rects = [_bbox(presses[p].danger_polygon.value, presses[p].dilation_m.value) for p in self.press_ids]
        self.share = floor_share(self.rects, [zones[z].floor_polygon.value for z in self.zone_ids])
        self.conv = OM.ObsV2Converter(self.share, self.rects, hold_s=hold_s)

    def __call__(self, d: Detections) -> OM.Observation:
        n = len(d)
        mach = np.full(n, -1, np.int16)
        for i in range(n):
            for j, (x0, z0, x1, z1) in enumerate(self.rects):
                if x0 <= d.x[i] <= x1 and z0 <= d.z[i] <= z1:
                    mach[i] = j
                    break
        cam_ok = np.array([bool(d.cam_ok.get(c, False)) for c in self.cams], bool)
        vis = np.clip(self.share @ cam_ok.astype(float), 0.0, 1.0)
        vis = np.where(vis > 1.0 - 1e-9, 1.0, vis)
        zm = np.array([d.zone_motion.get(p, math.nan) for p in self.press_ids], np.float64)
        zm[vis < OM.ZONE_VIS_MIN] = np.nan
        obs = OM.Observation(t=float(d.t), t_cap=float(d.t_cap), cls=np.array([OBS_CLS[c] for c in d.cls], np.int8),
                             x=np.asarray(d.x, np.float64), z=np.asarray(d.z, np.float64),
                             r=np.asarray(d.r, np.float64), heading=np.asarray(d.heading, np.float64),
                             moving=np.asarray(d.moving, bool), machine=mach, cam_ok=cam_ok,
                             zone_motion=zm.astype(np.float32), zone_vis=vis.astype(np.float32),
                             cam_floor_ok=cam_ok.copy())
        return self.conv(obs)


# ------------------------------------------------------------------------------------------------ the plant


def _identity(p: dict) -> tuple[int, str, str, CodeHashes]:
    """The EVAL identity fields of cfg["plant"]; fail closed (ValueError) when one is missing or invalid."""
    missing = [k for k in REQUIRED_IDENTITY if p.get(k) is None]
    if missing:
        raise ValueError(f"cfg['plant'] is missing {missing} (EVAL identity fields have no defaults: fail closed)")
    world, split, cond, code = p["world"], p["world_split"], p["condition"], p["code"]
    if isinstance(world, bool) or not isinstance(world, int) or world < 0:
        raise ValueError(f"cfg['plant']['world'] must be an int >= 0, got {world!r}")
    if split not in WORLD_SPLITS:
        raise ValueError(f"cfg['plant']['world_split'] must be one of {WORLD_SPLITS}, got {split!r}")
    if cond not in CONDITIONS:
        raise ValueError(f"cfg['plant']['condition'] must be one of {CONDITIONS}, got {cond!r}")
    if not isinstance(code, dict) or not code.get("head"):
        raise ValueError("cfg['plant']['code'] must carry a non-empty 'head'")
    return world, split, cond, CodeHashes(**code)


class Plant:
    """The runner's mover source and plant hooks (spec, attach, poses, after_step, frame_extras, on_ros, finish)."""

    def __init__(self, cfg: dict):
        p = dict(cfg.get("plant") or {})
        self.cfg, self.p = cfg, p
        self.world_idx, self.world_split, self.condition, self.code = _identity(p)
        self.site: Site = load(Site, p["site"])
        self.safety: Safety = load(Safety, p["safety"])
        dyn = load(Dynamics, p["dynamics"]) if p.get("dynamics") else None
        self.script = load_day_script(p["day_script"])
        check_world_seed(self.script.world_seed)
        self.world_seed, self.day = int(self.script.world_seed), int(self.script.day)
        if self.safety.site_id != self.site.site_id:
            raise ValueError(f"safety.yaml site {self.safety.site_id!r} != site {self.site.site_id!r}")
        guards = {ps.press: ps for ps in self.safety.presses}
        missing = sorted({x.id for x in self.site.presses} - set(guards))
        if missing:
            raise ValueError(f"safety.yaml has no entry for presses {missing} (fail closed: every press is guarded)")
        self.reset_override_s = self._reset_override(p.get("reset_override_s"))
        self.engine = Engine(self.script, self.site, dyn)
        self.t0 = float(p.get("plant_t0", self.script.t0_s))
        self.world = TwinWorld(self.engine, on_trip=self.world_trip)
        if abs(self.t0 * TICK_HZ - round(self.t0 * TICK_HZ)) > 1e-6:
            raise ValueError(f"plant_t0={self.t0} is not on the {TICK_HZ} Hz grid")
        self.T = float(cfg.get("seconds", MISSION_T_S))
        self.short_mission = self._check_mission_length(p.get("allow_short_mission"))
        if self.t0 <self.engine.t0 - EPS_T or self.t0 + self.T > self.engine.t_end + EPS_T:
            raise ValueError(f"mission [{self.t0}, {self.t0 + self.T}) is outside the scripted day "
                             f"[{self.engine.t0}, {self.engine.t_end})")
        self.run_id = str(p["run_id"])
        self.zone = str(p["zone"])
        if self.zone not in {z.id for z in self.site.zones}:
            raise ValueError(f"unknown zone {self.zone!r}")
        presses = {x.id: x for x in self.site.presses}
        # checkpoints (twin/tasks.yaml, lane S2): planned count and capture targets, checked against the site
        self.tasks: Tasks | None = load(Tasks, p["tasks"]) if p.get("tasks") else None
        self.checkpoints = {c.id: c for c in self.tasks.checkpoints} if self.tasks else {}
        self.planned: list[str] = self._check_tasks(presses) if self.tasks else []
        self.run_dir = Path(p.get("run_dir") or cfg.get("out_dir") or ".")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.guards = {pid: _PressGuard(guards[pid], presses[pid].danger_polygon.value, presses[pid].dilation_m.value)
                       for pid in self.engine.presses}
        self.ram_centre = {pid: tuple(np.mean(np.asarray(presses[pid].footprint.value, float), axis=0))
                           for pid in self.engine.presses}
        self.gate_aim = {pid: np.array((c[0], c[1], RAM_TDC_Z - RAM_STROKE_M / 2)) for pid, c in self.ram_centre.items()}
        self.zone_polys = [(z.id, z.floor_polygon.value) for z in self.site.zones]
        # movers present during the mission (Isaac colliders + lidar), from the script
        a, b = self.t0, self.t0 + self.T
        self.movers = [m for m in self.engine.movers if m.end >= a and m.start <= b]
        self._spec = [{"id": m.id, "cls": m.cls} for m in self.movers]
        # outputs
        self.writer = RealisedWriter(self.engine, self.run_dir / "realised_state.jsonl", self.run_id,
                                     self.world_idx, t_from=self.t0)
        self.writer.__enter__()
        self.k_next = round(self.t0 * TICK_HZ)
        self.k_end = round((self.t0 + self.T) * TICK_HZ)          # ticks k / 2 in [t0, t0 + T)
        self.written = -math.inf
        self.camera = make_camera(self.site, self.world_seed, self.day)
        self.adapter = ObsAdapter(self.site, p.get("obs_presses"), p.get("obs_zones"),
                                  hold_s=float(getattr(self.camera, "hold_s", OM.OCC_HOLD_S)))
        self.obs_path = self.run_dir / "obs_v2.jsonl"
        self._obs_f = open(self.obs_path, "w", encoding="utf-8", newline="\n")
        self._obs_buf: deque[OM.Observation] = deque()            # computed, not yet delivered (t > sim time)
        self.n_obs = 0
        key = f"t0={self.t0:.1f}"
        self.gate = make_gate(self.world_seed, self.day, key)
        self.cap_rng = keyed_rng(self.world_seed, self.day, "robot_capture", key)
        # runtime state
        self.caster = None
        self.robot: tuple[float, float, float, float, float, float, float] | None = None
        self.t = 0.0
        self.t_safety = -1e9
        self.path_len = 0.0
        self._prev_xy: tuple[float, float] | None = None
        self.log: list[dict] = []
        self.events: list[dict] = []
        self.pending: list[dict] = []
        self.captures: list[dict] = []
        self.outbox_cap: list[dict] = []
        self.attempted: set[str] = set()
        self.passed: set[str] = set()
        self.declined: set[str] = set()
        self.trips = self.holds = self.incidents = 0
        self.errors: list[str] = []
        self._overlap_prev: dict[str, bool] = {}
        self.min_gap = {"person": math.inf, "forklift": math.inf}
        self.contacts: list[dict] = []
        self.wall_plant = 0.0
        self.finished: dict | None = None
        self._ended = False
        self._broken = False                                       # a runner hook raised: no further plant work
        self._last_poses: dict[str, tuple[float, float, float]] = {}
        self._log_row(0.0, "mission_start", ROBOT_ID, zone=self.zone,
                      detail={"plant_t0": self.t0, "T_s": self.T, "condition": self.condition,
                              "world": self.world_idx, "world_split": self.world_split,
                              "reset_override_s": self.reset_override_s, "variant": self.variant})

    @property
    def variant(self) -> str:
        """Summary.variant: "primary" only for a full 720 s mission on the U4 lognormal reset; a short dev/test
        mission is "short_dev", a U4 reset-sweep run "reset_sweep" (both never pooled into the primary metric)."""
        if self.short_mission:
            return "short_dev"
        return "reset_sweep" if self.reset_override_s is not None else "primary"

    def _check_mission_length(self, allow) -> bool:
        """Summary.T_s is the fixed 720 s denominator (PLAN_TWIN §5): a mission of another length is refused (fail
        closed), unless cfg["plant"]["allow_short_mission"] is True on a dev world (tests, the slice); such a run is
        tagged variant "short_dev". Returns True for an allowed short mission."""
        if abs(self.T - MISSION_T_S) <= EPS_T:
            return False
        if allow is not True:
            raise ValueError(f"cfg seconds={self.T} != the fixed mission length T={MISSION_T_S} s; set "
                             "cfg['plant']['allow_short_mission']=True for a dev/test run (tagged short_dev)")
        if self.world_split != "dev":
            raise ValueError("allow_short_mission is refused on a test world (fail closed)")
        return True

    def _reset_override(self, v) -> float | None:
        """The U4 descriptive reset sweep: a fixed reset duration inside every press's sweep interval, or None."""
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"reset_override_s must be a number of seconds, got {v!r}")
        for ps in self.safety.presses:
            lo, hi = ps.reset.sweep_s.lo, ps.reset.sweep_s.hi
            if not lo <= float(v) <= hi:
                raise ValueError(f"reset_override_s={v} is outside press {ps.press}'s reset sweep [{lo}, {hi}] s")
        return float(v)

    def _check_tasks(self, presses: dict) -> list[str]:
        """tasks.yaml against the site (fail closed): same site_id, known zones and presses, a press in its
        checkpoint's zone, and an entry for the mission zone. Returns the mission zone's mission_order."""
        assert self.tasks is not None
        if self.tasks.site_id != self.site.site_id:
            raise ValueError(f"tasks.yaml site {self.tasks.site_id!r} != site {self.site.site_id!r}")
        zone_ids = {z.id for z in self.site.zones}
        for z in self.tasks.zones:
            if z.zone not in zone_ids:
                raise ValueError(f"tasks.yaml zone {z.zone!r} is not a zone of site {self.site.site_id!r}")
        for c in self.tasks.checkpoints:
            if c.zone not in zone_ids:
                raise ValueError(f"checkpoint {c.id}: unknown zone {c.zone!r}")
            if c.press is not None:
                if c.press not in presses:
                    raise ValueError(f"checkpoint {c.id}: unknown press {c.press!r}")
                if presses[c.press].zone != c.zone:
                    raise ValueError(f"checkpoint {c.id}: press {c.press} is in zone {presses[c.press].zone}, "
                                     f"not {c.zone}")
        zt = next((z for z in self.tasks.zones if z.zone == self.zone), None)
        if zt is None:
            raise ValueError(f"tasks.yaml has no mission order for zone {self.zone!r}")
        return list(zt.mission_order)

    # ------------------------------------------------------------------ runner hooks
    def spec(self) -> list[dict]:
        return list(self._spec)

    def attach(self, caster=None, dt: float = 0.02) -> None:
        self.caster = caster

    def _mover_states(self, T: float) -> list[tuple[Any, tuple[float, float, float, float]]]:
        out = []
        for m in self.movers:
            at = m.at(T)
            if at is not None:
                out.append((m, at))
        return out

    def poses(self, t: float) -> dict[str, tuple[float, float, float]]:
        """Mover poses in the ROS frame (x = plan x, y = plan z, psi from +x counter-clockwise). A broken plant
        returns the last poses it computed (the mission is already an error)."""
        if self._broken:
            return dict(self._last_poses)
        try:
            self._last_poses = {m.id: (x, z, math.pi / 2 - h) for m, (x, z, h, _) in self._mover_states(self.t0 + t)}
        except Exception as e:  # noqa: BLE001 - fail closed: recorded, the mission ends with status "error"
            self._hook_failed("poses", t, e)
        return dict(self._last_poses)

    def _hook_failed(self, hook: str, t: float, exc: Exception) -> None:
        """A runner hook raised during the mission: record it, stop all further plant work (`_broken`) and let
        `finish` write summary.json with status "error" (PLAN_TWIN §5: a failure, never missing logs)."""
        self._broken = True
        reason = f"plant {hook} failed at sim t={t:.2f} s: {exc!r}"[:500]
        self.errors.append(reason)
        try:
            self._log_row(t, "error", ROBOT_ID, zone=self.zone, detail={"reason": reason})
        except Exception:  # noqa: BLE001 - the reason is in self.errors either way
            pass

    def mover_prims(self, T: float) -> dict[str, np.ndarray]:
        """Analytic mover primitives (ROS frame): circles (cx, cy, r, zmax) = people, boxes (cx, cy, hw, hd, yaw, zmax)
        = forklift body + load (intercycle_sim.movers PARTS dimensions)."""
        circ, box = [], []
        for m, (x, z, h, _) in self._mover_states(T):
            if m.cls == "person":
                circ.append((x, z, PERSON_R, PERSON_H))
            else:
                psi = math.pi / 2 - h
                for fwd, ln, wd, ht in FORKLIFT_BOXES:
                    box.append((x + fwd * math.cos(psi), z + fwd * math.sin(psi), ln / 2, wd / 2, psi, ht))
        return {"circles": np.array(circ, float).reshape(-1, 4), "boxes": np.array(box, float).reshape(-1, 6)}

    def ram_parts(self, pid: str, T: float) -> np.ndarray:
        """The press-ram mover: plate points (x, y, z) at the engine's ram position (0 = top dead centre)."""
        cx, cz = self.ram_centre[pid]
        zr = RAM_TDC_Z - RAM_STROKE_M * float(self.engine.presses[pid].ram(T))
        a = RAM_HALF_M
        return np.array([(cx, cz, zr), (cx - a, cz - a, zr), (cx + a, cz - a, zr), (cx - a, cz + a, zr),
                         (cx + a, cz + a, zr)])

    def after_step(self, t: float, st: dict, contacts: list[dict]) -> None:
        """Runner hook after every physics step. Fail closed: an exception is recorded (`_hook_failed`) and the
        plant does no further work; `finish` then writes status "error"."""
        self.t = t
        if self._broken:
            return
        w = time.time()
        try:
            self._after_step(t, st, contacts)
        except Exception as e:  # noqa: BLE001 - recorded as the mission's error (scored as a failure)
            self._hook_failed("after_step", t, e)
        self.wall_plant += time.time() - w

    def _after_step(self, t: float, st: dict, contacts: list[dict]) -> None:
        x, y, z = (float(v) for v in st["pos_w"][:3])
        psi = yaw_of(st["quat_wxyz"])
        vw = st["v_w"]
        self.robot = (x, y, psi, float(vw[0]), float(vw[1]), float(st["omega_b"][2]), z)
        if self._prev_xy is not None:
            self.path_len += math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        self._prev_xy = (x, y)
        if contacts:
            self._contacts(t, contacts)
        self._captures_step(t, st)
        if t - self.t_safety >= SAFETY_DT - 1e-9:
            self.t_safety = t
            self._safety(t)
        self._write_ticks(t)
        if self.gate.due(t):
            self._gate(t)

    def frame_extras(self, t: float) -> dict:
        """What ROS gets from the plant: the robot camera's gate observations and inspection outcomes, nothing else.
        Fail closed like `after_step` (a broken plant sends nothing)."""
        if self._broken:
            return {}
        try:
            return self._frame_extras(t)
        except Exception as e:  # noqa: BLE001 - recorded as the mission's error (scored as a failure)
            self._hook_failed("frame_extras", t, e)
            return {}

    def _frame_extras(self, t: float) -> dict:
        out: dict = {}
        obs = self.gate.ready(t)
        if obs:
            out["camera_obs"] = obs
        if self.outbox_cap:
            out["inspection_capture"] = self.outbox_cap
            self.outbox_cap = []
        return out

    def on_ros(self, t: float, events: list[dict]) -> None:
        """The timing planner's inspection events: dwell_start / dwell_end (capture), decline (checkpoint declined).
        With tasks.yaml, a target outside the mission zone's mission_order is an error (not counted). Fail closed like
        `after_step`."""
        if self._broken:
            return
        try:
            self._on_ros(t, events)
        except Exception as e:  # noqa: BLE001 - recorded as the mission's error (scored as a failure)
            self._hook_failed("on_ros", t, e)

    def _on_ros(self, t: float, events: list[dict]) -> None:
        for e in events:
            kind, cp = e.get("event"), e.get("target")
            if kind not in ("dwell_start", "dwell_end", "decline"):
                continue
            cp = str(cp)
            if self.tasks is not None and cp not in self.planned and kind != "dwell_end":
                self._off_plan(t, cp, str(kind))
                continue
            if kind == "dwell_start":
                self.attempted.add(cp)
                self._log_row(t, "checkpoint_attempt", ROBOT_ID, checkpoint=cp, zone=self.zone,
                              detail={"inside": bool(e.get("inside"))})
                self.pending.append({"target": cp, "inside": bool(e.get("inside")), "t_start": t, "hold_since": None})
            elif kind == "dwell_end":
                for pc in [q for q in self.pending if q["target"] == cp]:
                    self._capture(t, pc, settled=False)
            else:
                self.declined.add(cp)
                self._log_row(t, "checkpoint_decline", ROBOT_ID, checkpoint=cp, zone=self.zone,
                              detail={"reason": e.get("reason")})

    def _off_plan(self, t: float, cp: str, kind: str) -> None:
        where = (f"checkpoint of zone {self.checkpoints[cp].zone}" if cp in self.checkpoints
                 else "no tasks.yaml entry")
        reason = f"{kind} for {cp!r}, not in zone {self.zone}'s mission_order ({where})"
        self.errors.append(reason)
        self._log_row(t, "error", ROBOT_ID, checkpoint=cp, zone=self.zone, detail={"reason": reason})

    # ------------------------------------------------------------------ realised stream + camera
    def _robot_mover(self) -> RealisedMover:
        assert self.robot is not None
        x, y, psi, vx, vy = self.robot[:5]
        zone = next((zid for zid, poly in self.zone_polys if point_in_polygon(x, y, poly)), None)
        return RealisedMover(id=ROBOT_ID, cls="robot", zone=zone, x=round(x, 6), z=round(y, 6),
                             heading=round(plan_heading(psi), 6), speed=round(math.hypot(vx, vy), 6), r=ROBOT_R)

    def _write_ticks(self, t: float) -> None:
        T = self.t0 + t
        while self.k_next < self.k_end and self.k_next / TICK_HZ <= T + 1e-9:
            tk = self.engine.tick(self.k_next / TICK_HZ, extra=[self._robot_mover()])
            self.writer.write(tk)
            self.written = tk.t
            self._obs_buf.append(self.adapter(self.camera.observe(tk)))
            self.k_next += 1
        self._deliver(T)

    def _deliver(self, T: float | None) -> None:
        """Write the buffered observations whose delivery time t has been reached (T None: all of them)."""
        while self._obs_buf and (T is None or self._obs_buf[0].t <= T + 1e-9):
            obs = self._obs_buf.popleft()
            self._obs_f.write(json.dumps(OM.obs_to_dict(obs), separators=(",", ":"), allow_nan=False) + "\n")
            self.n_obs += 1
        self._obs_f.flush()

    # ------------------------------------------------------------------ safety
    def _freeze(self, g: _PressGuard, a: float, b: float, state: str) -> float:
        """Apply a freeze from max(a, the written horizon) to b; returns the applied length."""
        a = max(a, self.written + EPS_T)
        if b <= a + 1e-9:
            return 0.0
        self.engine.apply_freeze(g.id, a, b, state)
        g.chunk_end = max(g.chunk_end, b)
        return b - a

    def _next_run(self, pid: str, a: float, b: float) -> tuple[float, float] | None:
        """(start, end of its running bout) of the earliest time in [a, b) at which the press is scheduled to run."""
        p = self.engine.presses[pid]
        for e, e0, e1 in zip(p.events, p.e0, p.e1):
            if e.kind == "running" and e1 > a and e0 < b:
                return max(a, e0), e1
        return None

    def _reset_s(self, g: _PressGuard, key: str) -> float:
        return reset_draw(g.cfg, self.world_seed, self.day, f"t0={self.t0:.1f}:{key}", self.reset_override_s)

    def _reset_tag(self) -> dict:
        return ({"reset_source": "sweep_override", "reset_override_s": self.reset_override_s}
                if self.reset_override_s is not None else {"reset_source": "lognormal_u4"})

    def _checkpoint_for(self, g: _PressGuard) -> str | None:
        """The checkpoint a robot-caused stop / hold at press g belongs to: the latest pending dwell target gated by
        this press, else the mission's first not-yet-passed checkpoint gated by it, else the latest pending target."""
        for pc in reversed(self.pending):
            c = self.checkpoints.get(str(pc["target"]))
            if c is not None and c.press == g.id:
                return c.id
        mine = [cid for cid in self.planned if self.checkpoints[cid].press == g.id]
        todo = [cid for cid in mine if cid not in self.passed]
        if todo or mine:
            return (todo or mine)[0]
        return str(self.pending[-1]["target"]) if self.pending else None

    def _footprint(self) -> list[tuple[float, float, float]]:
        assert self.robot is not None
        x, y, psi = self.robot[:3]
        return [(x + o * math.cos(psi), y + o * math.sin(psi), r) for o, r in FOOT]

    def _safety(self, t: float) -> None:
        T = self.t0 + t
        assert self.robot is not None
        fc = self._footprint()
        x, y = self.robot[0], self.robot[1]
        for g in self.guards.values():
            x0, z0, x1, z1 = g.bbox
            near = x0 - 1.0 < x < x1 + 1.0 and z0 - 1.0 < y < z1 + 1.0
            ins = near and circles_hit_poly(fc, g.poly, g.dilation)
            if ins and not g.inside:
                self._enter(g, t, T)
            if ins:
                self._inside(g, t, T)
            elif g.inside:
                self._exit(g, t, T)
            g.inside = ins
        self._overlaps(t, T, fc)

    def _start_trip(self, g: _PressGuard, T: float) -> None:
        g.tripped, g.trip_t0 = True, max(T, self.written + EPS_T)
        g.trip_cp = self._checkpoint_for(g)
        g.n_trips += 1
        self.trips += 1
        self._freeze(g, T, T + SAFETY_DT, "tripped")

    def _enter(self, g: _PressGuard, t: float, T: float) -> None:
        st, _ = self.engine.presses[g.id].state(T)
        action = g.rule[st]
        g.entry_state, g.entry_action, g.incident_entry = st, action, False
        g.visit_held = False
        self.events.append({"t": round(t, 3), "type": "enter", "press": g.id, "state": st, "action": action})
        if action == "trip":
            self._start_trip(g, T)
            self.events.append({"t": round(t, 3), "type": "trip", "press": g.id, "state": st})

    def _inside(self, g: _PressGuard, t: float, T: float) -> None:
        if g.tripped:                                     # continuous from the trip: no gap between chunks
            self._freeze(g, g.chunk_end, T + SAFETY_DT, "tripped")
            return
        allow_running = g.entry_action == "allow" and g.entry_state == "running"
        if not allow_running:
            # hold: a restart (scheduled bout start or reset end) that falls due while the robot is inside is delayed
            # from the end of this plant's latest freeze (or the written horizon): a restart due between two
            # safety steps is held from the moment it falls due, never later than the next unwritten tick. A held
            # freeze never runs past its running bout; one hold = one delayed restart.
            t_free = max(g.chunk_end, self.written + EPS_T)
            if t_free < T + SAFETY_DT:
                nxt = self._next_run(g.id, t_free, T + SAFETY_DT)
                if nxt is None:
                    self._close_hold(g, t)                # no restart pending in this chunk
                else:
                    a, bout_end = nxt
                    if not g.holding:
                        g.holding, g.hold_t0, g.hold_s = True, max(a, self.written + EPS_T), 0.0
                        g.hold_cp = self._checkpoint_for(g)
                        self.holds += 1
                        self.events.append({"t": round(t, 3), "type": "hold", "press": g.id})
                    b = min(T + SAFETY_DT, bout_end)
                    held = self._freeze(g, a, b, "held")
                    if held > 0:
                        g.hold_s += held
                        g.hold_t1 = b
                        g.visit_held = True
                    if bout_end <= T + SAFETY_DT:
                        self._close_hold(g, t)            # the scheduled running bout ended while held
        st, frozen = self.engine.presses[g.id].state(T)
        if st == "running" and not frozen and not g.incident_entry:
            g.incident_entry = True
            self.incidents += 1
            self._log_row(t, "incident", g.id, zone=self.site.zone_of(g.id).id,
                          checkpoint=self._checkpoint_for(g),
                          detail={"reason": "robot inside the danger zone of a running press (entry allowed)",
                                  "entry_state": g.entry_state, "entry_action": g.entry_action})

    def _close_hold(self, g: _PressGuard, t: float) -> None:
        if g.holding:
            self._log_row(t, "press_hold", g.id, zone=self.site.zone_of(g.id).id, checkpoint=g.hold_cp,
                          value=round(g.hold_s, 3),
                          detail={"t_abs0": g.hold_t0, "t_abs1": g.hold_t1,
                                  "cause": "restart delayed by robot presence"})
            g.holding, g.hold_s = False, 0.0

    def _exit(self, g: _PressGuard, t: float, T: float) -> None:
        self.events.append({"t": round(t, 3), "type": "exit", "press": g.id})
        zone = self.site.zone_of(g.id).id
        if g.tripped:
            reset = self._reset_s(g, f"trip:{g.n_trips - 1}")
            r0 = g.chunk_end
            self._freeze(g, r0, r0 + reset, "resetting")
            stopped = r0 - g.trip_t0
            self._log_row(t, "press_trip", g.id, zone=zone, checkpoint=g.trip_cp, value=round(stopped + reset, 3),
                          detail={"t_abs0": g.trip_t0, "tripped_s": round(stopped, 3), "reset_s": round(reset, 3),
                                  "entry_state": g.entry_state, "guard": g.cfg.guard.device,
                                  "guard_type": g.cfg.guard.type, **self._reset_tag()})
            self._log_row(r0 + reset - self.t0, "press_restart", g.id, zone=zone, checkpoint=g.trip_cp,
                          value=round(reset, 3), detail={"t_abs": r0 + reset, "after": "manual reset"})
            g.tripped = False
        else:
            held = g.visit_held
            cp = g.hold_cp if held else self._checkpoint_for(g)
            self._close_hold(g, t)
            muted = g.entry_action == "allow" and g.entry_state == "running"
            if g.latching and not muted:
                self._entry_reset(g, t, T, zone, cp, held)
        g.visit_held = False
        g.entry_state = g.entry_action = None

    def _running_measure(self, pid: str, a: float, b: float) -> float:
        """Seconds of [a, b) inside the press's scheduled running bouts."""
        p = self.engine.presses[pid]
        return sum(max(0.0, min(b, e1) - max(a, e0)) for e, e0, e1 in zip(p.events, p.e0, p.e1)
                   if e.kind == "running")

    def _entry_reset(self, g: _PressGuard, t: float, T: float, zone: str, cp: str | None, held: bool) -> None:
        """W-W4 / U4: the robot broke a light curtain with manual reset. The curtain cannot sense the robot inside, so
        the break latched the restart interlock on entry, whatever the press state: the press restarts only after a
        manual reset from outside, drawn from the press's lognormal (key entry_reset:<n>), state `resetting` from the
        exit. A reset that delays a scheduled restart is a hold (one hold per delayed restart: a visit that already
        held that restart is not counted again)."""
        reset = self._reset_s(g, f"entry_reset:{g.n_entry_resets}")
        g.n_entry_resets += 1
        r0 = max(T, self.written + EPS_T)
        self._freeze(g, r0, r0 + reset, "resetting")
        delayed = self._running_measure(g.id, r0, r0 + reset)
        counted = not held and delayed > 0.0
        if counted:
            self.holds += 1
            self.events.append({"t": round(t, 3), "type": "hold", "press": g.id, "via": "entry_reset"})
            self._log_row(t, "press_hold", g.id, zone=zone, checkpoint=cp, value=round(delayed, 3),
                          detail={"t_abs0": r0, "t_abs1": r0 + reset, "held_s": 0.0, "delayed_s": round(delayed, 3),
                                  "cause": "restart delayed by the manual reset after the robot's curtain break "
                                           "(latched restart interlock)"})
        self._log_row(r0 + reset - self.t0, "press_restart", g.id, zone=zone, checkpoint=cp, value=round(reset, 3),
                      detail={"t_abs0": r0, "t_abs": r0 + reset, "reset_s": round(reset, 3),
                              "after": "manual reset after a curtain break (light-curtain restart interlock)",
                              "held_restart": held, "delays_restart": delayed > 0.0, "entry_state": g.entry_state,
                              "guard": g.cfg.guard.device, **self._reset_tag()})

    def world_trip(self, mid: str, T: float, who: str, where: str | None = None, reset_s: float | None = None) -> dict:
        """TwinWorld.trip: a protective stop of press `mid` at absolute time T, counted and logged like a robot entry.
        With the robot inside the press, it stays tripped until the robot leaves (the reset follows `_exit`);
        otherwise tripped for one safety step, then the manual reset (`reset_s`, else the keyed draw)."""
        g = self.guards[mid]
        if T <= self.written + EPS_T:
            raise ValueError(f"trip at {T} is at or before the latest written tick {self.written}")
        t = T - self.t0
        zone = self.site.zone_of(mid).id
        if g.inside:
            if g.tripped:
                return {"t": T, "type": "trip", "machine": where or mid, "who": who, "lost_s": None, "group": mid,
                        "note": "already tripped"}
            self._close_hold(g, t)
            self._start_trip(g, T)
            self.events.append({"t": round(t, 3), "type": "trip", "press": mid, "who": who})
            return {"t": T, "type": "trip", "machine": where or mid, "who": who, "lost_s": None, "group": mid}
        cp = self._checkpoint_for(g)
        n = g.n_trips
        g.n_trips += 1
        self.trips += 1
        reset = float(reset_s) if reset_s is not None else self._reset_s(g, f"trip:{n}")
        self._freeze(g, T, T + SAFETY_DT, "tripped")
        r0 = T + SAFETY_DT
        self._freeze(g, r0, r0 + reset, "resetting")
        self.events.append({"t": round(t, 3), "type": "trip", "press": mid, "who": who})
        self._log_row(t, "press_trip", mid, zone=zone, checkpoint=cp, value=round(SAFETY_DT + reset, 3),
                      detail={"t_abs0": T, "tripped_s": SAFETY_DT, "reset_s": round(reset, 3), "who": who,
                              "via": "TwinWorld.trip", "guard": g.cfg.guard.device, "guard_type": g.cfg.guard.type,
                              **({"reset_source": "caller"} if reset_s is not None else self._reset_tag())})
        self._log_row(r0 + reset - self.t0, "press_restart", mid, zone=zone, checkpoint=cp, value=round(reset, 3),
                      detail={"t_abs": r0 + reset, "after": "manual reset"})
        return {"t": T, "type": "trip", "machine": where or mid, "who": who, "lost_s": SAFETY_DT + reset, "group": mid}

    def _overlaps(self, t: float, T: float, fc) -> None:
        assert self.robot is not None
        speed = math.hypot(self.robot[3], self.robot[4])
        for m, (x, z, h, _) in self._mover_states(T):
            if m.cls == "person":
                gap = min(math.hypot(x - cx, z - cz) - r - PERSON_R for cx, cz, r in fc)
            else:
                psi = math.pi / 2 - h
                bx = np.array([(x + f * math.cos(psi), z + f * math.sin(psi), ln / 2, wd / 2, psi)
                               for f, ln, wd, _ in FORKLIFT_BOXES])
                gap = min(float(box_gap(cx, cz, r, bx).min()) for cx, cz, r in fc)
            self.min_gap[m.cls] = min(self.min_gap[m.cls], gap)
            ov = gap < -OVERLAP_M
            if ov and not self._overlap_prev.get(m.id, False):
                moving = speed > MOVING_MPS
                self._log_row(t, "contact", m.id, zone=self.zone, value=round(gap, 3),
                              detail={"with": m.cls, "robot_v": round(speed, 3), "robot_moving": moving,
                                      "kind": "geometric_overlap"})
                if moving:
                    self.incidents += 1
                    self._log_row(t, "incident", ROBOT_ID, zone=self.zone, value=round(gap, 3),
                                  detail={"reason": f"robot overlapped a {m.cls} while moving", "mover": m.id})
            self._overlap_prev[m.id] = ov

    def _contacts(self, t: float, contacts: list[dict]) -> None:
        """Physical contacts reported by the backend (Isaac): logged; robot-person / robot-forklift classification is
        done on the geometric overlaps (`_overlaps`), which the double and Isaac share."""
        if self.contacts and t - self.contacts[-1]["t_last"] < 0.5:
            self.contacts[-1]["t_last"] = round(t, 3)
            return
        self.contacts.append({"t": round(t, 3), "t_last": round(t, 3),
                              "bodies": sorted({str(c.get("body")) for c in contacts}),
                              "max_force_n": max(float(c.get("force_n", 0.0)) for c in contacts)})
        self._log_row(t, "contact", ROBOT_ID, zone=self.zone, value=None,
                      detail={"kind": "physics_contact", "bodies": self.contacts[-1]["bodies"]})

    # ------------------------------------------------------------------ robot camera
    def _gate(self, t: float) -> None:
        if self.caster is None or self.robot is None:
            self.gate.t_next = t + RC.GATE_DT
            return
        T = self.t0 + t
        rx, ry, rpsi, *_, rz = self.robot
        ms = []
        for g in self.guards.values():
            x0, z0, x1, z1 = g.bbox
            if math.hypot(max(x0 - rx, 0, rx - x1), max(z0 - ry, 0, ry - z1)) > RC.GATE_RANGE_M:
                continue
            ms.append((g.id, self.ram_parts(g.id, T), self.gate_aim[g.id]))
        if not ms:
            self.gate.t_next = t + RC.GATE_DT
            return
        self.gate.observe(t, RC.cam_pose(rx, ry, rz, rpsi), ms, self.caster, self.mover_prims(T))

    def _captures_step(self, t: float, st: dict) -> None:
        if not self.pending:
            return
        v = math.hypot(st["v_w"][0], st["v_w"][1])
        w = abs(float(st["omega_b"][2]))
        for pc in list(self.pending):
            if v < RC.HOLD_V and w < RC.HOLD_W:
                pc["hold_since"] = pc["hold_since"] if pc["hold_since"] is not None else t
                if t - pc["hold_since"] >= RC.HOLD_S:
                    self._capture(t, pc, settled=True)
            else:
                pc["hold_since"] = None

    def _capture(self, t: float, pc: dict, settled: bool) -> None:
        if pc in self.pending:
            self.pending.remove(pc)
        cp = self.checkpoints.get(str(pc["target"]))
        assert self.robot is not None
        rx, ry, rpsi, vx, vy, wz, rz = self.robot
        rec: dict = {"target": pc["target"], "inside": pc["inside"], "settled": settled, "t_dwell_start": pc["t_start"]}
        if cp is None:
            rec.update({"pass": False, "error": "unknown checkpoint (no tasks.yaml entry)"})
            self.errors.append(f"capture of unknown checkpoint {pc['target']!r}")
            self._log_row(t, "error", ROBOT_ID, checkpoint=str(pc["target"]), zone=self.zone,
                          detail={"reason": rec["error"]})
        else:
            xyz = np.array(cp.target_xzh, float)
            n = np.array((cp.normal_xz[0], cp.normal_xz[1], 0.0))
            rec.update(RC.capture(t, RC.cam_pose(rx, ry, rz, rpsi), xyz, n, np.array((vx, vy, 0.0)), wz,
                                  self.caster, self.mover_prims(self.t0 + t), self.cap_rng, zoom=cp.zoom))
        ok = bool(rec.get("pass"))
        if ok:
            self.passed.add(str(pc["target"]))
        self._log_row(t, "checkpoint_pass" if ok else "checkpoint_fail", ROBOT_ID, checkpoint=str(pc["target"]),
                      zone=self.zone, detail={k: rec.get(k) for k in ("settled", "range_m", "incidence_deg",
                                                                        "px_per_m", "blur_px", "checks", "zoom")})
        self.captures.append(rec)
        self.outbox_cap.append({"t": rec.get("t"), "target": rec["target"], "pass": ok, "checks": rec.get("checks"),
                                "settled": settled, "source": RC.CAPTURE_SOURCE})

    # ------------------------------------------------------------------ outputs
    def _log_row(self, t: float, kind: str, entity: str, checkpoint: str | None = None, zone: str | None = None,
                 value: float | None = None, detail: dict | None = None) -> None:
        row = LogRow(run_id=self.run_id, t=max(0.0, round(float(t), 3)), kind=kind, entity=entity,  # type: ignore[arg-type]
                     checkpoint=checkpoint, zone=zone, value=value,
                     detail=None if detail is None else json.dumps(detail, sort_keys=True, default=float))
        self.log.append(row.model_dump(by_alias=True))

    def lost_time(self) -> dict[str, float]:
        """Per press: seconds the plant's freezes (tripped + reset + held) overlap scheduled running bouts."""
        out = {}
        for pid, p in self.engine.presses.items():
            tot = 0.0
            for e, e0, e1 in zip(p.events, p.e0, p.e1):
                if e.kind == "running":
                    tot += p.frozen_measure(e0, e1)
            out[pid] = round(tot, 3)
        return out

    def _counts(self) -> dict[str, int]:
        planned = len(self.planned) if self.tasks else len(self.attempted | self.declined)
        return {"checkpoints_planned": planned, "attempted": len(self.attempted),
                "passed": len(self.passed & self.attempted), "declined": len(self.declined - self.attempted)}

    def _summary(self, status: str, reason: str | None, counts: dict[str, int], lost_s: float,
                 sim_summary: dict | None) -> Summary:
        return Summary(contract="twin.summary/1", run_id=self.run_id,
                       engine="isaac" if self.cfg.get("sim") == "isaac" else "double",
                       world=self.world_idx, world_split=self.world_split, day=self.day,  # type: ignore[arg-type]
                       zone=self.zone, condition=self.condition, T_s=MISSION_T_S, status=status,  # type: ignore[arg-type]
                       status_reason=reason, robot_caused_press_stops=self.trips, holds=self.holds,
                       incidents=self.incidents, press_time_lost_s=round(lost_s, 3),
                       path_length_m=round(self.path_len, 3), code=self.code, variant=self.variant,  # type: ignore[arg-type]
                       reset_override_s=self.reset_override_s,
                       wall_s=None if sim_summary is None else sim_summary.get("wall_s"), **counts)

    def finish(self, out_dir: str | None = None, sim_summary: dict | None = None) -> dict:
        """Close the mission: publish the realised stream, write summary.json, log.parquet and plant_summary.json.
        Fail closed: any exception still writes summary.json with status "error" plus the log rows so far."""
        if self.finished is not None:
            return self.finished
        try:
            self.finished = self._finish(sim_summary)
        except Exception as e:  # noqa: BLE001 - recorded as the mission's error (scored as a failure, never dropped)
            self.finished = self._finish_failed(e, sim_summary)
        return self.finished

    def _finish(self, sim_summary: dict | None) -> dict:
        t_end = self.t
        for g in self.guards.values():                     # a robot still inside at the end: reset / hold close
            if g.inside:
                self._exit(g, t_end, self.t0 + t_end)
                g.inside = False
        status, reason = "ok", None
        try:
            self.writer.__exit__(None, None, None)         # publishes realised_state.jsonl (fail closed: >= 1 tick)
        except Exception as e:  # noqa: BLE001 - recorded as the mission's error, never swallowed silently
            self.errors.append(f"realised stream not published: {e!r}")
        self._deliver(None)                                # observations still in flight at the mission end
        self._obs_f.close()
        if self.errors:                                    # in order: a mission hook's failure comes first
            status, reason = "error", "; ".join(self.errors)[:500]
        if self.k_next < self.k_end and status == "ok":
            status, reason = "error", f"mission ended at sim t={t_end:.2f} s before T={self.T} s"
        self._log_row(t_end, "mission_end", ROBOT_ID, zone=self.zone, detail={"status": status})
        self._ended = True
        lost = self.lost_time()
        summ = self._summary(status, reason, self._counts(), sum(lost.values()), sim_summary)
        log_file = write_log(self.log, self.run_dir)
        diag = {"run_id": self.run_id, "plant_t0": self.t0, "sim_s": round(t_end, 3), "ticks": self.n_obs,
                "lost_by_press_s": lost, "trips": self.trips, "holds": self.holds, "incidents": self.incidents,
                "checkpoints_planned_source": "tasks.yaml" if self.tasks else "observed (no tasks.yaml given)",
                "min_gap_m": {k: (None if v == math.inf else round(v, 3)) for k, v in self.min_gap.items()},
                "freezes": [[f.press, round(f.t0, 3), round(f.t1, 3), f.state] for f in self.engine.freezes()],
                "gate_source": getattr(self.gate, "source", None), "gate_obs": getattr(self.gate, "n_obs", None),
                "capture_source": RC.CAPTURE_SOURCE, "captures": self.captures, "events": self.events,
                "ram_geometry": {"source": RAM_GEOMETRY_SOURCE, "tdc_z_m": RAM_TDC_Z, "stroke_m": RAM_STROKE_M,
                                 "half_m": RAM_HALF_M},
                "camera": type(self.camera).__name__, "log_file": log_file.name, "plant_wall_s": round(self.wall_plant, 2),
                "reset_sweep": self._sweep_tag(),
                "safety": {ps.press: {"guard": ps.guard.model_dump(), "reset_median_s": ps.reset.median_s,
                                      "reset_p90_s": ps.reset.p90_s} for ps in self.safety.presses}}
        (self.run_dir / "plant_summary.json").write_text(json.dumps(diag, indent=1, default=float) + "\n",
                                                        encoding="utf-8", newline="\n")
        out = summ.model_dump(mode="json")
        _write_json(self.run_dir / "summary.json", out)    # last: summary.json marks a completely closed mission
        return out

    def _sweep_tag(self) -> dict:
        return {"sweep_run": self.reset_override_s is not None, "reset_override_s": self.reset_override_s,
                "note": "U4 descriptive reset sweep: fixed reset duration, not the lognormal" if
                self.reset_override_s is not None else "U4 lognormal reset (median 240 s, p90 900 s)"}

    def _finish_failed(self, exc: Exception, sim_summary: dict | None) -> dict:
        """The fail-closed end of `finish`: status "error" with the exception as status_reason, the log rows gathered
        so far (plus an error row), summary.json always written."""
        reason = f"plant finish failed: {exc!r}"[:500]
        self.errors.append(reason)
        try:
            self.writer.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"realised stream not published: {e!r}")
        try:
            if not self._obs_f.closed:
                self._deliver(None)
                self._obs_f.close()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"obs_v2 not closed: {e!r}")
        self._log_row(self.t, "error", ROBOT_ID, zone=self.zone, detail={"reason": reason})
        if not self._ended:
            self._log_row(self.t, "mission_end", ROBOT_ID, zone=self.zone, detail={"status": "error"})
        try:
            log_name = write_log(self.log, self.run_dir).name
        except Exception as e:  # noqa: BLE001 - e.g. pyarrow failed: the same rows as log.jsonl
            log_name = write_log_jsonl(self.log, self.run_dir).name
            self.errors.append(f"log.parquet not written: {e!r}")
        try:
            lost_s = sum(self.lost_time().values())
        except Exception:  # noqa: BLE001
            lost_s = 0.0
        c = self._counts()
        c["passed"] = min(c["passed"], c["attempted"])
        c["checkpoints_planned"] = max(c["checkpoints_planned"], c["attempted"] + c["declined"])
        try:
            out = self._summary("error", reason, c, lost_s, sim_summary).model_dump(mode="json")
        except Exception as e:  # noqa: BLE001 - last resort: still a summary.json with status error
            out = {"contract": "twin.summary/1", "run_id": self.run_id, "status": "error", "variant": self.variant,
                   "status_reason": f"{reason}; summary invalid: {e!r}"[:1000]}
        _write_json(self.run_dir / "summary.json", out)
        try:
            _write_json(self.run_dir / "plant_summary.json",
                        {"run_id": self.run_id, "status": "error", "errors": self.errors, "log_file": log_name,
                         "trips": self.trips, "holds": self.holds, "incidents": self.incidents,
                         "reset_sweep": self._sweep_tag(), "events": self.events})
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        return out


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")


def write_log_jsonl(rows: list[dict], run_dir: Path) -> Path:
    """log.jsonl: the LogRow rows in the contract's column order, one JSON object per line."""
    from twin.schemas.tables import columns

    cols = columns(LogRow)
    p = run_dir / "log.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({c: r[c] for c, _ in cols}) + "\n")
    return p


def write_log(rows: list[dict], run_dir: Path) -> Path:
    """log.parquet in the contract's column order and types (twin.schemas.tables.columns(LogRow)); log.jsonl with the
    same rows where pyarrow is missing (the Isaac python)."""
    from twin.schemas.tables import columns

    cols = columns(LogRow)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return write_log_jsonl(rows, run_dir)
    types = {"string": pa.string(), "float64": pa.float64(), "int64": pa.int64(), "bool": pa.bool_()}
    schema = pa.schema([(c, types[a]) for c, a in cols])
    table = pa.Table.from_pylist([{c: r[c] for c, _ in cols} for r in rows], schema=schema)
    p = run_dir / "log.parquet"
    pq.write_table(table, p)
    return p


def make(cfg: dict) -> Plant:
    return Plant(cfg)
