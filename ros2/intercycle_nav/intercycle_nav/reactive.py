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
    the machine was already still) is a lower bound, so it is kept as censored.
Window now (`windows`): only while the machine is fresh (a visible sample within `fresh_s`) and STILL for at least
`min_still_s`, from the Kaplan-Meier estimate of the STILL-run length conditional on the elapsed time e:
    end = onset + Q,  Q = the smallest observed length d > e with P(D <= d | D > e) >= quantile,
          capped at the longest length known to be reached (nothing is known beyond it: fail closed),
    minus `extra_margin_s`; it needs at least `min_at_risk` past runs longer than e.
Unknown / stale machine -> None (the planner treats it as "unknown: go and look"); observed but no usable window -> [].

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
STATE_SCHEMA = "intercycle.reactive_state.v1"
MOVING, STILL, UNKNOWN = "MOVING", "STILL", "UNKNOWN"
FRESH_S = 1.0                 # = CameraGate.fresh_s (timing.py): an observation older than this is not "now"
MAX_RUNS = 400                # STILL runs kept per machine (most recent)


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

    def _close(self, t_end: float, censored: bool) -> None:
        if self.onset is None:
            return
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
            self.cur, self.onset, self.onset_seen, self.cand = UNKNOWN, None, False, None

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
            self.cur, self.onset, self.onset_seen, self.cand = s, t0, seen, None

    def finish(self, t: float) -> None:
        """End of a mission: the open run is right-censored at the last visible sample."""
        self._gap_check(t)
        if self.cur != UNKNOWN and self.last_vis is not None:
            self._close(self.last_vis, censored=True)
        self.cur, self.onset, self.onset_seen, self.cand, self.last_vis = UNKNOWN, None, False, None, None

    def windows(self, t: float) -> list | None:
        if self.cur == UNKNOWN or self.last_vis is None or t - self.last_vis > FRESH_S:
            return None
        if self.cur == MOVING:
            return []
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

    def observe(self, t: float, machine_id: str, visible: bool, motion_seen: bool, source: str = "") -> None:
        tr = self.tracks.get(machine_id)
        if tr is None:
            return
        tr.observe(float(t), bool(visible), bool(motion_seen))
        if source:
            self.sources[source] = self.sources.get(source, 0) + 1

    def windows(self, t: float) -> dict[str, list | None]:
        return {m: tr.windows(t) for m, tr in self.tracks.items()}

    def states(self) -> dict[str, str]:
        return {m: tr.cur for m, tr in self.tracks.items()}

    def finish_mission(self, t: float, mission_tag: str) -> None:
        for tr in self.tracks.values():
            tr.finish(t)
        self.missions.append(str(mission_tag))

    # ---- persistence within the world
    def to_state(self) -> dict:
        return {"schema": STATE_SCHEMA, "world_id": self.world_id, "config": json.loads(self.cfg.to_json()),
                "config_sha256": self.cfg.sha256(), "space_sha256": space_sha256(), "missions": list(self.missions),
                "camera_sources": dict(self.sources),
                "machines": {m: {"still": [list(r) for r in tr.still_runs], "busy": [list(r) for r in tr.busy_runs],
                                 "n_obs": tr.n_obs} for m, tr in self.tracks.items()}}

    @classmethod
    def from_state(cls, state: dict | None, cfg: ReactiveConfig, machine_ids: list[str], world_id: str) -> OwnCameraEstimator:
        """state None = the world's first mission (no camera history). A state from another world, another config or
        another parameter space is refused (fail closed: histories never cross worlds or configs)."""
        est = cls(cfg, machine_ids, world_id)
        if state is None:
            return est
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
        return est


def load_state(path: str) -> dict | None:
    """The world's persisted reactive state, or None when the file does not exist (first mission of the world)."""
    import os

    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(path: str, est: OwnCameraEstimator) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(est.to_state(), f)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ masks
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
                         "still_runs": {m: len(tr.still_runs) for m, tr in est.tracks.items()}},
            **(extra or {})}
