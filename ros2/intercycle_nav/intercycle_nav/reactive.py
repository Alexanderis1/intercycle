"""Reactive baseline (E-W4; PLAN_TWIN §5 "Baseline fairness"): idle windows from the robot's OWN camera only.

Pure logic, no ROS, no truth. Used by `observer_bridge` (condition "reactive") and read by `timing_planner` exactly like the
learned condition's windows, so the reactive arm runs the SAME window planner (SIPP, `timing.TimingCore`) and the SAME
entry logic: (a) an idle window open now with need_s left AND (b) the camera gate's "no motion" confirmation. It has the
same sensors, zoom head, rules and checkpoints. The only difference is where the windows come from:

  learned   the observer's verified broadcast (ceiling cameras, days of history);
  reactive  the robot's own no-motion gate stream (`/intercycle/camera/observations`, the same messages that feed the
            camera gate), turned into STILL / MOVING runs per machine, persisted within the world across its missions.
            No camera history before the first mission of a world (by design; the claim states it).

Estimator, per machine (`MachineTrack`):
  * every gate sample is MOVING (visible and motion seen), STILL (visible, no motion) or UNSEEN (not visible);
  * a change of state is accepted after it persists `debounce_s` (the run starts at the first sample of the streak);
  * an UNSEEN gap longer than `max_gap_s` ends the run as right-censored at the last visible sample, state UNKNOWN;
  * every finished STILL run is kept as (duration, censored); a run whose start was not seen (the robot arrived while
    the machine was already still) is a lower bound, so it is kept as censored;
  * robot presence (audit fix 2; pad fixed in fix round 2): while the robot's estimated centre is inside or within
    PRESENCE_PAD_M = 0.81 m (body reach 0.66 m + 0.15 m localisation margin) of ANY hazard zone, the plant may hold the
    machine (or a trip may freeze its line), so what the camera then sees is not natural. On entry every machine's open
    run is right-censored at the entry time (the last tick the robot was known outside: natural only up to there), and
    the machine is "tainted" until its next accepted MOVING with the robot outside: runs that start while tainted are
    never recorded and give no window. All machines, not only the zone's own, because a hold or a trip can stop a whole
    line and the robot does not know the line groups. The prediction of a natural run in progress at entry continues
    (the dwell cut keeps working from it); only what is stored is cut.
Window now (`windows`): only while the machine is fresh (a visible sample within `fresh_s`) and STILL for at least
`min_still_s`, from the Kaplan-Meier estimate of the STILL-run length conditional on the elapsed time e:
    end = onset + Q,  Q = the smallest observed length d > e with P(D <= d | D > e) >= quantile,
          capped at the longest length known to be reached (nothing is known beyond it: fail closed),
    minus `extra_margin_s`; it needs at least `min_at_risk` past runs longer than e.
Unknown / stale machine -> None, observed but no usable window -> []; for reactive the planner treats both as "go to
the boundary and look" (timing.TimingCore.plan_target).
Persistence within the world fails closed: an empty state path is the only way to say "first mission"; a named state
file that is missing is an error; each state records the sha256 of the state it was loaded from (`parent_sha256`), so the
EVAL runner can check the mission chain with `verify_chain`.

Tunable parameter space: `PARAM_SPACE` (frozen in PREREG v1, checked by the G2 auditor), `ReactiveConfig` is the frozen,
hashable config; `candidate_configs()` is the fixed 8-config tuning set (default first, then 7 seeded draws). Nothing in
here is tuned: the defaults are a declared starting point, the separate baseline-tuning agent picks among the 8.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass, field, fields

import numpy as np

CONDITION = "reactive"
SCHEMA = "intercycle.reactive.v1"
STATE_SCHEMA = "intercycle.reactive_state.v2"
MOVING, STILL, UNKNOWN = "MOVING", "STILL", "UNKNOWN"
FRESH_S = 1.0                 # = CameraGate.fresh_s (timing.py): an observation older than this is not "now"
MAX_RUNS = 400                # STILL runs kept per machine (most recent)
# Robot presence pad (audit fix round 2): the plant decides holds and trips with the robot's BODY circles
# (intercycle_eval.plant.FOOT: forward offsets +-0.34 m, r 0.32 m), so the body reaches max|o| + r = 0.66 m from the
# centre. Presence uses that reach plus a localisation margin for the AMCL error, so every pose at which the plant can
# hold or trip counts as "present" (fail closed). FIXED, not tunable; part of space_json, so PREREG v1 freezes it. The
# held-zone mask rule keeps its own 0.5 m pad (masks.snapshot held_pad_m).
BODY_REACH_M = 0.66           # = max(|o|) + r over intercycle_eval.plant.FOOT (asserted in test_reactive.py)
LOC_MARGIN_M = 0.15           # ASSUMED AMCL position error allowance
PRESENCE_PAD_M = BODY_REACH_M + LOC_MARGIN_M     # 0.81 m


@dataclass(frozen=True)
class ReactiveConfig:
    """The reactive arm's tunable parameters (all of them). Frozen + canonical JSON -> sha256."""
    debounce_s: float = 0.4          # a new state must persist this long before it is accepted
    min_still_s: float = 2.0         # stillness threshold: no window before the machine has been still this long
    quantile: float = 0.10           # accepted risk that the machine restarts before the predicted window end
    min_at_risk: int = 3             # past STILL runs longer than the elapsed time needed for a prediction
    extra_margin_s: float = 2.0      # subtracted from the predicted window end
    max_gap_s: float = 3.0           # unseen gap that ends (censors) the current run
    use_left_censored: bool = False  # True: a STILL run whose start was not seen is timed from the first sample

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, **asdict(self)}, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> ReactiveConfig:
        d = {k: v for k, v in d.items() if k != "schema"}
        names = {f.name for f in fields(cls)}
        if set(d) - names:
            raise ValueError(f"unknown reactive config keys {sorted(set(d) - names)}")
        cfg = cls(**d)
        validate(cfg)
        return cfg


# The declared search space (PREREG v1). Every config the reactive arm may run with has each field in these values.
PARAM_SPACE: dict[str, tuple] = {
    "debounce_s": (0.0, 0.4, 1.0),
    "min_still_s": (0.5, 2.0, 5.0, 10.0),
    "quantile": (0.05, 0.10, 0.20, 0.35),
    "min_at_risk": (1, 3, 5),
    "extra_margin_s": (0.0, 2.0, 5.0),
    "max_gap_s": (1.0, 3.0),
    "use_left_censored": (False, True),
}
CANDIDATE_SEED = 13855        # fixed before any run; candidate_configs() is deterministic
N_CANDIDATES = 8              # PLAN_TWIN §5: 8 configs per arm


def validate(cfg: ReactiveConfig) -> None:
    for k, allowed in PARAM_SPACE.items():
        v = getattr(cfg, k)
        if not any(v == a and type(v) is type(a) for a in allowed):
            raise ValueError(f"reactive config {k}={v!r} is outside the frozen space {allowed}")


def space_json() -> str:
    return json.dumps({"schema": SCHEMA, "space": {k: list(v) for k, v in PARAM_SPACE.items()},
                       "fresh_s": FRESH_S, "max_runs": MAX_RUNS, "candidate_seed": CANDIDATE_SEED,
                       "presence_pad_m": PRESENCE_PAD_M, "body_reach_m": BODY_REACH_M, "loc_margin_m": LOC_MARGIN_M,
                       "n_candidates": N_CANDIDATES, "default": json.loads(ReactiveConfig().to_json()),
                       "candidates": [c.sha256() for c in candidate_configs()]},
                      sort_keys=True, separators=(",", ":"))


def space_sha256() -> str:
    return hashlib.sha256(space_json().encode()).hexdigest()


def candidate_configs(n: int = N_CANDIDATES, seed: int = CANDIDATE_SEED) -> list[ReactiveConfig]:
    """The fixed tuning set: the default config, then n-1 distinct grid points drawn with a seeded RNG (no result
    informs this list)."""
    keys = list(PARAM_SPACE)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*(PARAM_SPACE[k] for k in keys))]
    default = ReactiveConfig()
    rest = [g for g in grid if ReactiveConfig(**g) != default]
    picks = random.Random(seed).sample(range(len(rest)), n - 1)
    out = [default] + [ReactiveConfig(**rest[i]) for i in picks]
    for c in out:
        validate(c)
    return out


# ------------------------------------------------------------------------------------------------ estimator
def km_window_length(runs: list[tuple[float, bool]], elapsed: float, quantile: float, min_at_risk: int) -> float | None:
    """Predicted total STILL length given the run has lasted `elapsed` (Kaplan-Meier, right-censored runs), or None.
    runs: (length, censored). Returns the smallest observed event length d > elapsed with
    1 - S(d)/S(elapsed) >= quantile, capped at the longest length known to be reached (max length > elapsed)."""
    at = sorted((d, c) for d, c in runs if d > elapsed)
    if len(at) < max(1, min_at_risk):
        return None
    cap = at[-1][0]
    n = len(at)
    surv = 1.0
    i = 0
    while i < len(at):
        d = at[i][0]
        j = i
        events = 0
        while j < len(at) and at[j][0] == d:
            events += 0 if at[j][1] else 1
            j += 1
        if events:
            surv *= 1.0 - events / n
            if 1.0 - surv >= quantile - 1e-12:
                return min(d, cap)
        n -= j - i
        i = j
    return cap


@dataclass
class MachineTrack:
    cfg: ReactiveConfig
    cur: str = UNKNOWN
    onset: float | None = None
    onset_seen: bool = False
    cand: tuple[str, float] | None = None
    last_vis: float | None = None
    still_runs: list = field(default_factory=list)     # [(length_s, censored)]
    busy_runs: list = field(default_factory=list)      # [(length_s, censored)] (logged, not used for windows)
    n_obs: int = 0
    run_open: bool = False        # the current run is still to be recorded (False: recorded already, or never natural)
    run_natural: bool = True      # the current run started with no robot presence since the last accepted MOVING
    tainted: bool = False         # robot presence since this machine's last accepted MOVING with the robot outside
    present: bool = False         # the robot is in (or at) a hazard zone now
    n_robot_censored: int = 0     # STILL runs right-censored at a robot entry
    n_robot_dropped: int = 0      # STILL runs never recorded because they started while tainted

    def _close(self, t_end: float, censored: bool) -> None:
        if self.onset is None or not self.run_open:
            return
        self.run_open = False
        d = round(max(0.0, t_end - self.onset), 3)
        rec = (d, bool(censored or not self.onset_seen))
        if self.cur == STILL:
            self.still_runs.append(rec)
            del self.still_runs[:-MAX_RUNS]
        elif self.cur == MOVING:
            self.busy_runs.append(rec)
            del self.busy_runs[:-MAX_RUNS]

    def _gap_check(self, t: float) -> None:
        if self.cur != UNKNOWN and self.last_vis is not None and t - self.last_vis > self.cfg.max_gap_s:
            self._close(self.last_vis, censored=True)
            self.cur, self.onset, self.onset_seen, self.cand, self.run_open = UNKNOWN, None, False, None, False

    def robot_presence(self, t_entry: float, present: bool) -> None:
        """The robot entered (present goes True; t_entry = last time it was known outside) or left a hazard zone."""
        if present and not self.present:
            self.tainted = True
            if self.cur != UNKNOWN and self.onset is not None and self.run_open:
                if self.onset < t_entry:
                    if self.cur == STILL:
                        self.n_robot_censored += 1
                    self._close(t_entry, censored=True)      # natural only up to the entry
                else:                                       # started after the last tick known outside: not natural
                    if self.cur == STILL:
                        self.n_robot_dropped += 1
                    self.run_open = self.run_natural = False
        self.present = bool(present)

    def observe(self, t: float, visible: bool, motion_seen: bool) -> None:
        self.n_obs += 1
        self._gap_check(t)
        if not visible:
            return
        s = MOVING if motion_seen else STILL
        self.last_vis = t
        if s == self.cur:
            self.cand = None
            return
        if self.cand is None or self.cand[0] != s:
            self.cand = (s, t)
        if t - self.cand[1] + 1e-9 >= self.cfg.debounce_s:
            t0 = self.cand[1]
            seen = self.cur != UNKNOWN
            self._close(t0, censored=False)
            if self.tainted and s == MOVING and not self.present:
                self.tainted = False                        # the machine runs again with the robot outside
            natural = not self.tainted
            if not natural and s == STILL:
                self.n_robot_dropped += 1
            self.cur, self.onset, self.onset_seen, self.cand = s, t0, seen, None
            self.run_open = self.run_natural = natural

    def finish(self, t: float) -> None:
        """End of a mission: the open run is right-censored at the last visible sample."""
        self._gap_check(t)
        if self.cur != UNKNOWN and self.last_vis is not None:
            self._close(self.last_vis, censored=True)
        self.cur, self.onset, self.onset_seen, self.cand, self.last_vis = UNKNOWN, None, False, None, None
        self.run_open, self.run_natural, self.tainted, self.present = False, True, False, False

    def windows(self, t: float) -> list | None:
        if self.cur == UNKNOWN or self.last_vis is None or t - self.last_vis > FRESH_S:
            return None
        if self.cur == MOVING or not self.run_natural:
            return []                                       # moving, or a stop the robot may have caused
        assert self.onset is not None
        if not self.onset_seen and not self.cfg.use_left_censored:
            return []
        e = t - self.onset
        if e < self.cfg.min_still_s:
            return []
        q = km_window_length(self.still_runs, e, self.cfg.quantile, self.cfg.min_at_risk)
        if q is None:
            return []
        end = self.onset + q - self.cfg.extra_margin_s
        return [(self.onset, end)] if end > t else []


class OwnCameraEstimator:
    """The reactive arm's timing and idle knowledge: built only from its own camera, persisted within one world."""

    def __init__(self, cfg: ReactiveConfig, machine_ids: list[str], world_id: str):
        validate(cfg)
        if not world_id:
            raise ValueError("the reactive estimator needs the world id (its history is persisted within one world)")
        self.cfg, self.world_id = cfg, str(world_id)
        self.tracks = {m: MachineTrack(cfg) for m in machine_ids}
        self.missions: list[str] = []
        self.sources: dict[str, int] = {}
        self.parent_sha256: str | None = None       # sha256 of the state file this one was loaded from (None: first)
        self.n_entries = 0
        self._inside = False
        self._last_out_t: float | None = None

    def observe(self, t: float, machine_id: str, visible: bool, motion_seen: bool, source: str = "") -> None:
        tr = self.tracks.get(machine_id)
        if tr is None:
            return
        tr.observe(float(t), bool(visible), bool(motion_seen))
        if source:
            self.sources[source] = self.sources.get(source, 0) + 1

    def presence(self, t: float, inside: bool) -> None:
        """Robot presence tick (observer_bridge, from the localisation estimate): inside = robot_present(...), i.e. in
        or within PRESENCE_PAD_M of any hazard zone (or pose lost). Entry time = the last tick the robot was known
        outside (fail closed)."""
        t, inside = float(t), bool(inside)
        if inside and not self._inside:
            t_entry = self._last_out_t if self._last_out_t is not None else t
            self.n_entries += 1
            for tr in self.tracks.values():
                tr.robot_presence(t_entry, True)
        elif not inside and self._inside:
            for tr in self.tracks.values():
                tr.robot_presence(t, False)
        if not inside:
            self._last_out_t = t
        self._inside = inside

    def windows(self, t: float) -> dict[str, list | None]:
        return {m: tr.windows(t) for m, tr in self.tracks.items()}

    def states(self) -> dict[str, str]:
        return {m: tr.cur for m, tr in self.tracks.items()}

    def finish_mission(self, t: float, mission_tag: str) -> None:
        if not mission_tag:
            raise ValueError("finish_mission needs the mission tag (the EVAL runner checks the mission chain with it)")
        for tr in self.tracks.values():
            tr.finish(t)
        self.missions.append(str(mission_tag))
        self._inside, self._last_out_t = False, None

    # ---- persistence within the world
    def to_state(self) -> dict:
        return {"schema": STATE_SCHEMA, "world_id": self.world_id, "config": json.loads(self.cfg.to_json()),
                "config_sha256": self.cfg.sha256(), "space_sha256": space_sha256(), "missions": list(self.missions),
                "parent_sha256": self.parent_sha256, "camera_sources": dict(self.sources), "n_entries": self.n_entries,
                "machines": {m: {"still": [list(r) for r in tr.still_runs], "busy": [list(r) for r in tr.busy_runs],
                                 "n_obs": tr.n_obs, "robot_censored": tr.n_robot_censored,
                                 "robot_dropped": tr.n_robot_dropped} for m, tr in self.tracks.items()}}

    @classmethod
    def from_state(cls, state: dict | None, cfg: ReactiveConfig, machine_ids: list[str], world_id: str,
                   state_sha256: str | None = None) -> OwnCameraEstimator:
        """state None = the world's first mission (no camera history). A state from another world, another config or
        another parameter space is refused (fail closed: histories never cross worlds or configs). state_sha256 = the
        sha256 of the loaded file (recorded as the new state's parent_sha256)."""
        est = cls(cfg, machine_ids, world_id)
        if state is None:
            return est
        if not state_sha256:
            raise ValueError("a loaded reactive state needs its file sha256 (the mission chain is checked from it)")
        est.parent_sha256 = str(state_sha256)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"reactive state schema {state.get('schema')!r} != {STATE_SCHEMA}")
        if str(state.get("world_id")) != str(world_id):
            raise ValueError(f"reactive state belongs to world {state.get('world_id')!r}, not {world_id!r}")
        if state.get("config_sha256") != cfg.sha256() or state.get("space_sha256") != space_sha256():
            raise ValueError("reactive state was written with another config or parameter space")
        est.missions = [str(m) for m in state.get("missions", [])]
        est.sources = {str(k): int(v) for k, v in state.get("camera_sources", {}).items()}
        for m, d in state.get("machines", {}).items():
            tr = est.tracks.get(m)
            if tr is None:
                continue
            tr.still_runs = [(float(a), bool(b)) for a, b in d.get("still", [])][-MAX_RUNS:]
            tr.busy_runs = [(float(a), bool(b)) for a, b in d.get("busy", [])][-MAX_RUNS:]
            tr.n_obs = int(d.get("n_obs", 0))
            tr.n_robot_censored = int(d.get("robot_censored", 0))
            tr.n_robot_dropped = int(d.get("robot_dropped", 0))
        est.n_entries = int(state.get("n_entries", 0))
        return est


FAILED_SUFFIX = ".save_failed"          # marker observer_bridge writes next to the state when the final save fails


def load_state_file(path: str) -> tuple[dict | None, str | None]:
    """(state, sha256 of the file bytes). An EMPTY path is the only way to say "first mission of the world" -> (None,
    None). A named file that does not exist is an error (fail closed: a lost or misnamed state must never silently
    start the arm with no history)."""
    if not path:
        return None, None
    with open(path, "rb") as f:                     # FileNotFoundError if the named state is missing
        raw = f.read()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def load_state(path: str) -> dict | None:
    """The state only (see load_state_file)."""
    return load_state_file(path)[0]


def verify_chain(state_out_path: str, mission_tag: str, prev_state_path: str | None, world_id: str) -> dict:
    """The EVAL runner's check of one reactive mission (fail closed: raises ValueError / OSError).
    The saved state exists, has no save-failure marker, belongs to the world, ends with this mission's tag, was loaded
    from exactly the previous mission's saved state (parent_sha256 = its sha256; None for the world's first mission)
    and holds that state's missions plus this one. Returns {"sha256", "missions"}."""
    import os

    if os.path.exists(state_out_path + FAILED_SUFFIX):
        raise ValueError(f"reactive state save failed: {state_out_path + FAILED_SUFFIX}")
    st, sha = load_state_file(state_out_path)
    if st is None or sha is None:
        raise ValueError("verify_chain needs the saved state path")
    if st.get("schema") != STATE_SCHEMA or str(st.get("world_id")) != str(world_id):
        raise ValueError(f"reactive state has schema {st.get('schema')!r} / world {st.get('world_id')!r}")
    missions = [str(m) for m in st.get("missions", [])]
    if not missions or missions[-1] != str(mission_tag):
        raise ValueError(f"reactive state does not end with mission {mission_tag!r}: {missions[-3:]}")
    if prev_state_path:
        prev, prev_sha = load_state_file(prev_state_path)
        if prev is None or st.get("parent_sha256") != prev_sha:
            raise ValueError("reactive state was not loaded from the previous mission's state (parent_sha256)")
        if missions[:-1] != [str(m) for m in prev.get("missions", [])]:
            raise ValueError("reactive mission list is not the previous one plus this mission")
    elif st.get("parent_sha256") is not None or len(missions) != 1:
        raise ValueError("the world's first reactive mission must start from no state (no parent, one mission)")
    return {"sha256": sha, "missions": len(missions)}


def save_state(path: str, est: OwnCameraEstimator) -> None:
    import os

    if not path:
        raise ValueError("save_state needs a path")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(est.to_state(), f)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ masks
def robot_present(site, xy, had_pose: bool) -> bool:
    """The estimator's presence input (observer_bridge): the localisation estimate xy is in or within PRESENCE_PAD_M
    of any hazard zone (axis-aligned pad, which over-covers the plant's Euclidean body test at the corners), or the
    pose is lost after the first fix (fail closed; before the first fix the robot is at its dock, outside every zone)."""
    if xy is None:
        return bool(had_pose)
    return site.zone_of(float(xy[0]), float(xy[1]), pad=PRESENCE_PAD_M) is not None


def zone_open(windows, t: float, horizon_s: float) -> bool:
    """The learned rule (masks.idle_through) on the reactive windows: one window covers all of [t, t + H]."""
    return any(a <= t and b >= t + horizon_s for a, b in windows or [])


def reactive_masks(grid, site, windows: dict, t: float, horizon_s: float,
                   held_id: str | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(keep-out, floor, info) for the reactive arm, with the learned arm's light-curtain rule (masks.keepout_mask /
    keepout_floor): a hazard zone's BARE rectangle is keep-out unless its own-camera window covers [t, t + H] or the
    robot is inside it (held); the site's hard keep-outs always. No map: no L1 envelopes, no UNKNOWN cells, no speed
    mask (those are the observer's product). Keep-out and floor are the same here (the floor is what nothing may
    free)."""
    from intercycle_nav.masks import KEEP_OUT

    g = grid.zeros()
    grid.fill_boxes(g, [r[:4] for r in site.hard_keepout], KEEP_OUT)
    per = {}
    for mid, r in site.rects.items():
        op = mid == held_id or zone_open(windows.get(mid), t, horizon_s)
        per[mid] = "open" if op else "closed"
        if not op:
            grid.fill_boxes(g, [r], KEEP_OUT)               # the bare rect, as learned (no static margin)
    return g, g.copy(), {"condition": CONDITION, "source": "own_camera", "zones": per,
                         "keepout_cells": int((g > 0).sum())}


def snapshot_json(t: float, est: OwnCameraEstimator, extra: dict | None = None) -> dict:
    """The /intercycle/observer/state message for the reactive arm (same keys timing_planner reads as for learned)."""
    w = est.windows(t)
    st = est.states()
    return {"t": t, "condition": CONDITION, "map_ok": False, "map_version": None, "slow_ok": False, "fast_age_s": None,
            "machines": {m: {"state": st[m], "phase": None, "held": False,
                             "windows": None if wm is None else [[round(a, 3), round(b, 3)] for a, b in wm[:8]]}
                         for m, wm in w.items()},
            "reactive": {"config_sha256": est.cfg.sha256(), "world_id": est.world_id, "missions": len(est.missions),
                         "parent_sha256": est.parent_sha256, "n_entries": est.n_entries,
                         "still_runs": {m: len(tr.still_runs) for m, tr in est.tracks.items()},
                         "tainted": {m: tr.tainted for m, tr in est.tracks.items()}},
            **(extra or {})}
