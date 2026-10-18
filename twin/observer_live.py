"""The twin's live observer (PLAN_TWIN §2 / §4 lane S6; WEAKNESSES R-W1, T-W1, T-W6, M-W4, M-W7, C-W5).

What it reads: ONLY `runs/<id>/obs_v2.jsonl`, one `intercycle.observer.mapping.obs_to_dict` line per 2 Hz tick of an
Observation v2 (written by twin/camera.py stat_v1). `ObsV2Reader` refuses any other file name, so the run's truth
files can never be read through it, and every line is checked with `mapping.check_obs_v2` (zones = the spec's machines
in order, cameras = the spec's cameras). Nothing here imports the body-shop simulator, its truth-fed observer or any
truth module (tests/test_observer_live.py walks the import graph).

R-W1 / T-W1: the fast layer is `learn_cycles` + `nowcast` on Observation v2 `zone_motion`, the same stream the map
learns from (the measured-error camera), never a truth "zone active" flag. Per zone m and tick: motion = zone_motion[m]
> motion_thr; NaN (MISSING: occluded zone, invalid camera) stays NaN (T-W6: MISSING is never "no motion").
learn_cycles constants come from the machine kind (`learn_cycles.KINDS`; press = PRESS_PLACEHOLDER until probe P4), and
a run marked sealed (cfg.sealed or INTERCYCLE_SEALED_RUN=1) refuses any constant still at its placeholder, the motion
threshold included (`PlaceholderConstantError`). The motion threshold of the twin's camera (stat_v1: zone_motion = ram-
ROI energy / the press's median idle energy, idle median 1.0) is `config_for_stat_v1(...)`, read from
twin/stat_v1_params.json `energy.motion_thr` (`stat_v1_motion_thr`; the JSON is read, twin.camera is never imported)
with its source recorded; the 0.5 default stays a PLACEHOLDER for 0/1 streams (the v1 converter's moving-PART count).
A twin run on stat_v1 output must use `config_for_stat_v1` (at 0.5 every idle press would read as motion).
  * slow layer: the MachineModel per machine, re-learned every learn_every_s on the last learn_window_s; a machine
    without a confident model has confidence 0 (the robot ignores < MIN_CONF). SlowLayer.traffic is empty, as in the
    body-shop SyntheticSchedule: the robot reads traffic from the map layer's L2/L3.
  * fast layer: `learn_cycles.nowcast` per machine at t; UNKNOWN (confidence 0) while the newest zone sample is MISSING,
    while the stream is stale (no tick for stale_s) and while no confident model exists.

Map layer (M-W4, M-W7): a `mapping.LayeredMap` on the twin spec (`twin_spec`) learns from the same observations. Its
conditioning is computed here from the stream (`_cond`): per press, phase bin 0 while running, STOP_BIN once no
motion has been OBSERVED, without a gap, for the learned idle + 3 sigma (a MISSING tick or ticks absent from the stream
restart the count: a stroke may hide inside any gap; the same evidence `learn_cycles.forecast` needs for STOPPED), -1
(unknown) while MISSING or without a confident model. A press period of 1.2-3 s is 2.4-6 ticks, so the body-shop's 16 phase
bins carry no information: the twin uses bins (0, STOP_BIN) only. The M-W4 change detector is
`change_detect_l1.L1ChangeDetector` (ChangeLayer per press grid; observed = floor_ok_of(obs) and zone_vis >= vis_min)
with the LEAD DECISION: fail closed, i.e. every cell of a press grid not learned yet in bin 0 or STOP_BIN is UNKNOWN
(`availability()` reports the cost). The M-W7 margin is read ONLY from the frozen JSON through
`intercycle.margin_inflate.load_margin` and applied with `margin_inflate.inflate`, PER BIN (LEAD DECISION, audit S6
#3, option B): the broadcast copy of the map (`map_snapshot`, a `TwinMapSnapshot`) overrides `envelope()` so that
`pack_map` sends, for every running phase bin 0..N_PHASE-1, the M-W7-inflated envelope of bin 0 (the twin learns only
bin 0 and STOP_BIN, `twin_bin`; the robot never meets an unlearned bin and never falls back to the union rule), and for
STOP_BIN the inflated STOP_BIN envelope. unknown_rects carry only the live map's own UNKNOWN rects plus the detector's
UNKNOWN cells (CUSUM alarms and the fail-closed unlearned cells), inflated by the same margin (the frozen rule inflates
"occupied OR UNKNOWN"). The robot (intercycle_nav.masks, Site.own_stop) looks up, over its keep-out horizon
[t, t + H] (H = 6 s), STOP_BIN for a press whose own fast entry is a confident STOPPED (confidence >= 4) that stays
inside its TTL over the WHOLE horizon, bin 0 while it cycles, and the union of all bins for anything else (UNKNOWN,
missing, low confidence, a stop expiring inside the horizon, a fast layer older than 30 s: fail closed).
`robot_bins` / `calibration_pred` give S7b's dev recorder exactly that lookup, un-inflated, from what the robot HOLDS
at its lookup time: the slow and fast layers last broadcast to it (quantised as the wire does: the TTL in 2 s steps
rounded down, the issue time to the ms) and the map version it completed (`HeldMap`: the un-inflated per-twin-bin
envelopes, the detector's UNKNOWN cells and the map period, frozen when that map was packed; never this observer's
current map, detector or models, which keep changing after the broadcast). Then inflate(calibration_pred) == the held
map's broadcast per-bin envelope OR its broadcast UNKNOWN cells, i.e. what the robot gets, so the certified margin
applies to it. The margin's cell_m must
equal the L1 grid cell (env_res) and margin_m must fit in the grid's margin around the zone, else ValueError (fail
closed: re-calibrate on this grid). `tags["m7_application"]` carries this into every run's diagnostics.

Time of day (P2: the real clips carry no timing information): any time-of-day / shift structure the map's L2/L3
slices learn comes from the generated days, i.e. from the generator's ASSUMED structure. `tags` says so and is part of
`diagnostics()`.

Signing (C-W5): `sign_layers(key, t, keyring_now)` encodes slow / fast / map under key_id_of(key) and refuses to sign
when that key is not in the keyring valid now (the robot would reject it: never broadcast unverifiable layers);
`verify_blob` is the receiver's check against an explicit keyring and fails closed. sign_layers also returns `held`:
the decoded slow and fast layers and the `HeldMap` of that bundle (what a robot holds once it has them). The robot's
MapReceiver keeps the first map it completes for a version and ignores a later bundle of the same version with another
sha, so the recorder must keep the HeldMap whose (version, sha) the robot completed (MapReceiver.manifest).
"""
from __future__ import annotations

import json
import math
import os
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from nacl.signing import SigningKey

from intercycle.codec import maplayer as ML
from intercycle.codec.layout import (FAST, SLOW, decode_fast, decode_slow, encode_fast, encode_slow, peek_header,
                                     quantize_fast, quantize_model, quantize_now, quantize_slow)
from intercycle.codec.sign import Keyring, sign, verify
from intercycle.codec.trust import TrustError, key_id_of
from intercycle.margin_inflate import Margin, inflate, load_margin
from intercycle.observer import learn_cycles as LC
from intercycle.observer import mapping as M
from intercycle.observer import change_detect as CD
from intercycle.observer.change_detect_l1 import L1_CFG, L1ChangeDetector
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State

OBS_FILE = "obs_v2.jsonl"
STAT_V1_PARAMS = Path(__file__).resolve().parent / "stat_v1_params.json"
PRESS_CONSTS = Path(__file__).resolve().parent / "observer_press_consts.json"
TWIN_BINS = (0, M.STOP_BIN)
SEALED_ENV = "INTERCYCLE_SEALED_RUN"
TOD_TAG = ("ASSUMED: no hour-of-day structure is observed in the real clips (probe P2: no timing information); "
           "any time-of-day or shift structure in the map's slices comes from the generated days")
# the robot's lookup parameters (intercycle_nav.masks MaskParams.horizon_s / step_s and ObserverState.fast_max_age_s
# defaults, = observer_bridge's horizon_s / step_s defaults; tests/test_observer_live.py pins the equality)
ROBOT_HORIZON_S = 6.0
ROBOT_STEP_S = 0.5
ROBOT_FAST_MAX_AGE_S = 30.0
M7_TAG = ("LEAD DECISION (audit S6 #3, option B): per-bin M-W7 inflation. Every running phase bin is broadcast as the "
          "inflated bin-0 envelope and STOP_BIN as the inflated STOP_BIN envelope; only the change detector's UNKNOWN "
          "and fail-closed unlearned cells (inflated) are broadcast as UNKNOWN boxes. The robot opens a press's STOP_BIN "
          "envelope only on its own confident STOPPED (confidence >= 4) that stays inside its TTL over the whole "
          "keep-out horizon [t, t + 6 s] (audit S6-optB fix 1: a stop expiring inside the horizon reads the union); "
          "UNKNOWN, missing or low confidence reads the union of both bins (fail closed). The robot decodes the TTL in "
          "2 s steps rounded down, so the STOP_BIN envelope opens only when the encoded TTL >= horizon + lag (6 s + "
          "the lookup lag), i.e. learner TTL >= 8 s (the learner's STOPPED TTL is min(ttl_s, 10 s), floor 2 s); no "
          "bounded stop validity is in use (none decided). A running press reads bin 0 only while the held map has "
          "a learned period, else the union. S7b's calibration pred = calibration_pred(slow, fast, map_held, t), the "
          "same lookup un-inflated, from the layers the robot holds at its lookup time t (quantised as decoded) and "
          "the map version it holds (HeldMap), never the observer's live map")
STOP_ENTRY_TAG = ("LEAD DECISION (lane S5b-entry): a twin press's own confident STOPPED (fast and slow confidence >= 4, "
                  "inside its TTL) is one idle window [issue, issue + ttl] for the robot, for off-line Site.own_stop "
                  "presses only (predict.idle_windows stop_window; the body shop keeps None). The zone rect opens only "
                  "while that window covers [t, t + H] (masks.idle_through) and the planner enters only while it still "
                  "leaves need_s (timing.window_open); the reactive arm uses the same planner and zone rule on its own-"
                  "camera window. A held press (robot inside / reaching into its zone) reads STOP_BIN in robot_bins and "
                  "calibration_pred, as masks.machine_bins. DISCLOSED (pending lead decision): the learner caps a "
                  "STOPPED TTL at min(ttl_s, 10 s), ttl_s <= 10 / stops_per_h, so with need_s 16.5 s the learned arm "
                  "never enters a stopped press, while the reactive arm can. DISCLOSED (observer-fix probe): on the "
                  "twin the learned stops_per_h is inflated (long inter-stroke gaps from camera misses count as stops, "
                  "20-170 / h against ~3-5 / h true), so the STOPPED TTL is pinned at the 2 s floor and the robot's "
                  "STOP_BIN lookup (TTL >= 8 s) never opens at all: the learned arm reads the union for a stopped press. "
                  "Even at the true rate 10 / stops_per_h < 8 s. The TTL bounds stop STARTS, not stop ENDS: a restart "
                  "inside the 6 s horizon is not guarded by it (pending lead decision: a stop-duration model)")
MISFIT_TAG = ("observer-fix: after an observed quiet run longer than the model's stop threshold ends in motion (a "
              "STOPPED the press contradicted: a restart, or a cycle longer than the model), the press's STOPPED is "
              "suppressed (UNKNOWN, confidence 0) and its map conditioning is unknown (-1) for misfit_hold_s; a re-fit "
              "on a window less than half observed never replaces a confident model by a non-confident one; a stream gap "
              "longer than learn_every_s restarts the re-learn schedule. The map layer is packed once per mission "
              "(broadcast_live): M-W4 alarms during a mission reach the robot only through the fast / slow layers")


def twin_bin(b: int) -> int:
    """The twin bin whose envelope serves map phase bin b: STOP_BIN for STOP_BIN, 0 for every running bin 0..N_PHASE-1
    (a 1.2-3 s press period is 2.4-6 ticks at 2 Hz, so the twin learns bins (0, STOP_BIN) only). Anything else raises."""
    b = int(b)
    if b == M.STOP_BIN:
        return M.STOP_BIN
    if 0 <= b < M.N_PHASE:
        return 0
    raise ValueError(f"phase bin {b} outside 0..{M.STOP_BIN}")


def robot_bins(model: MachineModel | None, now: MachineNow | None, issue_t: float, t: float, map_period: float,
               horizon_s: float = ROBOT_HORIZON_S, step_s: float = ROBOT_STEP_S,
               fast_max_age_s: float = ROBOT_FAST_MAX_AGE_S, held: bool = False) -> tuple[int, ...]:
    """The twin bins the robot looks up for one press (intercycle_nav.masks.machine_bins for an off-line press with
    Site.own_stop) at plant time t over its keep-out horizon [t, t + horizon_s], from the slow model and the
    fast entry it holds, the fast layer issued at plant time issue_t, and the press's period in the map it holds
    (`map_period` = HeldMap.period[i], NaN when the map had none). model, now and issue_t are quantised as the wire
    does first (layout.quantize_model / quantize_now: the TTL in 2 s steps rounded down; the issue time to the ms), so
    an un-encoded entry gives the lookup the robot makes after decoding it (a 6.51 s TTL is 6 s on the robot):
      * held (the robot is inside / reaching into the press zone, masks.MachineView.held: it holds the press) ->
        (STOP_BIN,) whatever the layers say, as machine_bins (audit minor, lane S5b-entry);
      * missing, UNKNOWN, low confidence (fast or slow < MIN_CONF), a fast layer older than fast_max_age_s, an entry
        past its TTL at t -> TWIN_BINS, the union (fail closed);
      * a confident STOPPED -> (STOP_BIN,) only while the entry stays inside its TTL at every step of the horizon grid
        (np.arange(0, horizon_s, step_s), as machine_bins), else TWIN_BINS (the press may restart once it expires);
      * a confident ACTIVE / IDLE with a usable slow period (own_phase) AND a learned map period (predicted_bins
        cycling: every running phase bin reads bin 0) -> (0,); without a map period the robot reads the union."""
    if held:
        return (M.STOP_BIN,)
    if model is None or now is None:
        return TWIN_BINS
    model, now = quantize_model(model), quantize_now(now)
    sec, ms = divmod(round((DAY0_UNIX + float(issue_t)) * 1000), 1000)      # layout._split_ms: the decoded issue time
    issue_unix = sec + ms / 1000
    if now.confidence < LC.MIN_CONF or model.confidence < LC.MIN_CONF:
        return TWIN_BINS
    if DAY0_UNIX + t - issue_unix > fast_max_age_s:                          # the robot drops the whole fast layer
        return TWIN_BINS
    left = issue_unix - DAY0_UNIX + now.ttl_s - t                            # masks.ObserverState.snapshot stop_left
    if left < 0:
        return TWIN_BINS
    if now.state == State.STOPPED:
        last = float(np.arange(0.0, horizon_s + 1e-9, step_s)[-1])
        return (M.STOP_BIN,) if last <= left else TWIN_BINS
    if now.state in (State.ACTIVE, State.IDLE) and math.isfinite(model.period_s) and model.period_s > 0 \
            and math.isfinite(map_period) and map_period > 0:
        return (0,)
    return TWIN_BINS


@dataclass(frozen=True, eq=False)
class HeldMap:
    """The map part of what the robot holds: one packed map (version, sha of its compressed payload), frozen at packing
    time. Per press (zone order): `env[i][b]` the UN-inflated envelope of twin bin b (0, STOP_BIN) on the L1 grid, as
    TwinMapSnapshot inflated and packed it; `unknown[i]` the detector's UNKNOWN cells (CUSUM + fail-closed unlearned),
    un-inflated, as broadcast (inflated) in unknown_rects; `period[i]` the map period the robot decodes (NaN: none)."""
    version: int
    sha: bytes
    ids: tuple
    env: tuple
    unknown: tuple
    period: np.ndarray

    def pred(self, i: int, bins) -> np.ndarray:
        """Press i's M-W7 `pred` for a lookup of map bins `bins`: the held envelope union over their twin bins OR the
        held UNKNOWN cells (before inflation)."""
        tb = sorted({twin_bin(b) for b in bins})
        if not tb:
            raise ValueError("empty bin lookup")
        out = self.unknown[i].copy()
        for b in tb:
            out |= self.env[i][b]
        return out


class TwinMapSnapshot(M.LayeredMap):
    """The broadcast copy of the twin's live map (built by `ObserverLive.map_snapshot`, packed by `pack_map`):
    envelope(machine, b) is the M-W7-inflated envelope of twin_bin(b). It shares the live map's count arrays, so pack
    it right away (as sign_layers does); its box-cover cache is its own."""

    _margin: Margin
    _infl_cache: dict

    @classmethod
    def of(cls, live: M.LayeredMap, margin: Margin, unknown_rects: list) -> TwinMapSnapshot:
        m2 = cls.__new__(cls)
        m2.__dict__.update(live.__dict__)
        m2.__dict__.pop("_hot_cache", None)
        m2._cover_cache = {}
        m2._infl_cache = {}
        m2._margin = margin
        m2.unknown_rects = list(unknown_rects)
        return m2

    def envelope(self, machine_id: str, phase_bin: int, p_min: float = 0.05) -> np.ndarray:
        key = (machine_id, twin_bin(phase_bin), float(p_min))
        if key not in self._infl_cache:
            env = super().envelope(machine_id, key[1], p_min)
            self._infl_cache[key] = inflate(env, self._margin, cell_m=float(self.spec["env_res"]))
        return self._infl_cache[key].copy()


# ------------------------------------------------------------------------------------------------ spec

def twin_spec(machines: list[dict], extent: tuple[float, float, float, float], cameras: list[tuple[float, float]],
              site_id: int, key_id: int, env_res: float = 0.25, margin: float = 1.0, site_res: float = 1.0,
              name: str = "twin", version: int = 1) -> dict:
    """A LayeredMap spec for the twin (the shape of mapping.build_spec, without the body-shop hall fields).
    machines: [{"id", "kind", "rect": (x0, z0, x1, z1)}] in zone order (= Observation.machine / zone_motion order);
    extent = (x0, z0, x1, z1) of the site grid; cameras = one floor point per camera polygon, in cam_ok order."""
    x0, z0, x1, z1 = map(float, extent)
    if not (x1 > x0 and z1 > z0):
        raise ValueError(f"empty extent {extent}")
    mm = []
    for m in machines:
        a, b, c, d = map(float, m["rect"])
        gx0, gz0 = math.floor((a - margin) / env_res) * env_res, math.floor((b - margin) / env_res) * env_res
        mm.append(dict(id=str(m["id"]), kind=str(m["kind"]), rect=[a, b, c, d], x0=gx0, z0=gz0,
                       nx=int(math.ceil((c + margin - gx0) / env_res)), nz=int(math.ceil((d + margin - gz0) / env_res)),
                       on_line=False, period_nominal=float(m.get("period_nominal", float("nan")))))
    site = dict(x0=x0, z0=z0, res=float(site_res), nx=int(math.ceil((x1 - x0) / site_res)),
                nz=int(math.ceil((z1 - z0) / site_res)))
    return dict(name=name, version=int(version), site_id=int(site_id), key_id=int(key_id), env_res=float(env_res),
                margin=float(margin), machines=mm, site=site, cameras=[[float(c[0]), float(c[1])] for c in cameras],
                n_phase=M.N_PHASE, n_tod=M.N_TOD, n_prod=M.N_PROD)


# ------------------------------------------------------------------------------------------------ input

class ObsV2Reader:
    """Incremental reader of runs/<id>/obs_v2.jsonl (the camera process appends while the observer reads). Only
    complete lines are parsed; a partial last line waits for the next poll. Refuses any file not named obs_v2.jsonl."""

    def __init__(self, path: str | os.PathLike, n_zone: int, n_cam: int):
        p = Path(path)
        if p.name != OBS_FILE:
            raise ValueError(f"observer_live reads only {OBS_FILE}, not {p.name!r}")
        self.path, self.n_zone, self.n_cam = p, int(n_zone), int(n_cam)
        self._pos, self._buf, self.n_lines = 0, b"", 0

    def poll(self) -> list[M.Observation]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        self._pos += len(data)
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            self.n_lines += 1
            d = json.loads(ln)
            if d.get("v") != 2:
                raise ValueError(f"{self.path}:{self.n_lines}: not an Observation v2 line (v={d.get('v')!r})")
            o = M.obs_from_dict(d)
            M.check_obs_v2(o, self.n_zone, self.n_cam)
            out.append(o)
        return out


# ------------------------------------------------------------------------------------------------ config

@dataclass(frozen=True)
class ObserverLiveConfig:
    margin_path: str                                    # frozen M-W7 JSON (twin/m7_margin.json), required
    consts: dict = field(default_factory=lambda: dict(LC.KINDS))   # machine kind -> learn_cycles.CycleConsts
    motion_thr: float = 0.5                             # zone_motion > thr = motion
    motion_thr_source: str = (LC.PLACEHOLDER + ": zone_motion units are set by the camera model (stat_v1, lane S3b); "
                              "0.5 is right for the v1 converter's moving-PART count")
    learn_first_s: float = 300.0                        # first model after this much stream
    learn_every_s: float = 300.0
    learn_window_s: float = 1200.0                      # the slice's 20 min of learning
    stale_s: float = 1.5                                # fast layer UNKNOWN when the newest tick is older than this
    misfit_hold_s: float = 120.0                        # STOPPED suppressed this long after a contradicted stop
    latency_s: float = 0.0                              # zone_motion is stamped at capture (t_cap): no extra shift
    sealed: bool = False
    hz: float = M.FRAME_HZ


def stat_v1_motion_thr(path: str | os.PathLike = STAT_V1_PARAMS) -> tuple[float, str]:
    """(value, source) of the stat_v1 motion threshold from the fitted parameter file (scripts/stat_v1_fit.py:
    Youden J of the normalised TRAIN ROI energy). Fails closed (ValueError) on a file that is not a stat_v1 parameter
    file or has no finite positive threshold."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("contract") != "twin.stat_v1_params/1":
        raise ValueError(f"{p}: not a stat_v1 parameter file")
    mt = d["energy"]["motion_thr"]
    v = float(mt["value"])
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{p}: energy.motion_thr {v!r} is not a finite positive threshold")
    return v, (f"stat_v1 (twin/{p.name} energy.motion_thr = {v}, youden_j {mt.get('youden_j')}, "
               f"tpr {mt.get('tpr_moving')}, fpr {mt.get('fpr_idle')}): {mt.get('rule')}")


def press_consts(path: str | os.PathLike = PRESS_CONSTS) -> LC.CycleConsts:
    """The press CycleConsts measured from probe P4 (twin/observer_press_consts.json, `learn_cycles.consts_from_config`).
    Fails closed (ValueError) on a file that is not that contract or still leaves a placeholder."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.pop("contract", None) != "twin.observer_press_consts/1":
        raise ValueError(f"{p}: not an observer press-constants file")
    c = LC.consts_from_config(d)
    if c.kind != "press" or c.placeholder:
        raise ValueError(f"{p}: press constants {c.placeholder} still placeholders")
    return c


def config_for_stat_v1(margin_path: str, params_path: str | os.PathLike = STAT_V1_PARAMS,
                       consts_path: str | os.PathLike = PRESS_CONSTS, **kw) -> ObserverLiveConfig:
    """The config for a twin run on stat_v1 output: motion_thr and its source from twin/stat_v1_params.json, and the
    press cycle constants measured by probe P4 (`press_consts`), so a sealed run (INTERCYCLE_SEALED_RUN=1) passes the
    placeholder check."""
    thr, src = stat_v1_motion_thr(params_path)
    if "consts" not in kw:
        kw["consts"] = dict(LC.KINDS, press=press_consts(consts_path))
    return ObserverLiveConfig(margin_path=margin_path, motion_thr=thr, motion_thr_source=src, **kw)


def config_for_camera(camera_model: str, margin_path: str, **kw) -> ObserverLiveConfig:
    """The observer config of a twin run, chosen by the camera model that wrote its obs_v2.jsonl (the plant's
    `camera.provenance()["model"]`, recorded in summary.json `camera.model` and in the broadcaster's
    observer_spec.json). Only "stat_v1" is a twin-run camera (`config_for_stat_v1`); any other name (camera_v0, the
    G0 reference, an empty or unknown model) is refused, so a twin run never falls back to the placeholder
    threshold (fail closed)."""
    if camera_model != "stat_v1":
        raise ValueError(f"camera model {camera_model!r}: a twin run's observer needs the stat_v1 camera "
                         "(config_for_stat_v1); no other camera model has a twin-run config")
    return config_for_stat_v1(margin_path, **kw)


def is_sealed(cfg: ObserverLiveConfig) -> bool:
    return bool(cfg.sealed) or os.environ.get(SEALED_ENV) == "1"


# ------------------------------------------------------------------------------------------------ observer

class ObserverLive:
    def __init__(self, spec: dict, cfg: ObserverLiveConfig, obs_path: str | os.PathLike | None = None):
        self.spec, self.cfg = spec, cfg
        self.ids = [m["id"] for m in spec["machines"]]
        self.M = len(self.ids)
        sealed = is_sealed(cfg)
        self.consts = []
        for m in spec["machines"]:
            if m["kind"] not in cfg.consts:
                raise ValueError(f"machine {m['id']}: no learn_cycles constants for kind {m['kind']!r}")
            self.consts.append(LC.require_measured(cfg.consts[m["kind"]], sealed))
        self._thr_placeholder = str(cfg.motion_thr_source).upper().startswith(LC.PLACEHOLDER)
        if sealed and self._thr_placeholder:
            raise LC.PlaceholderConstantError(f"motion_thr {cfg.motion_thr} is a placeholder ({cfg.motion_thr_source})")
        self.margin: Margin = load_margin(cfg.margin_path)       # raises unless frozen and valid (fail closed)
        if not math.isclose(self.margin.cell_m, float(spec["env_res"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"M-W7 margin cell_m {self.margin.cell_m} != L1 grid cell {spec['env_res']}: re-calibrate")
        if self.margin.margin_m > float(spec["margin"]) + 1e-9:
            raise ValueError(f"M-W7 margin {self.margin.margin_m} m does not fit in the {spec['margin']} m L1 grid margin")
        self.map = M.LayeredMap(spec=spec)
        # half-life: the L1 adapter's default (24 h / 17 bins of ticks per bin) assumes 17 evenly visited bins; the twin
        # visits bin 0 nearly always, so a bin-0 cell would forget in ~1.4 h of wall clock. 24 h of ticks per bin
        # keeps M-W4's 24 h half-life for the running bin (forgetting runs on observed coverage, never on wall time).
        self.det = L1ChangeDetector(self.map, cfg=replace(L1_CFG, half_life_ticks=CD.DAY_TICKS),
                                    fail_closed_unlearned=True, bins=TWIN_BINS)
        self.reader = None if obs_path is None else ObsV2Reader(obs_path, self.M, len(spec["cameras"]))
        self._pending: list[M.Observation] = []
        self._t: list[float] = []
        self._y: list[np.ndarray] = []
        self.models = [MachineModel(i, 0.0, 0.0, 0.5, 0.0, 0.0, 0) for i in range(self.M)]
        self.learned: list[LC.Learned | None] = [None] * self.M
        self._quiet = np.zeros(self.M)                  # observed seconds without motion since the last motion or gap
        self._misfit_t = np.full(self.M, -np.inf)       # last time an observed quiet run > stop threshold ended in motion
        self.n_misfit = np.zeros(self.M, int)
        self.n_dup = 0                                  # duplicated / reordered ticks dropped by advance()
        self.t_first: float | None = None
        self.t_last: float | None = None
        self._next_learn: float | None = None
        self.seq = 0
        self.events: list[dict] = []
        self.n_ticks = 0
        self.tags = dict(time_of_day=TOD_TAG, m7_application=M7_TAG, stop_entry=STOP_ENTRY_TAG,
                         cycle_consts={k: dict(placeholder=list(c.placeholder), source=c.source)
                                       for k, c in cfg.consts.items()},
                         motion_thr=dict(value=cfg.motion_thr, source=cfg.motion_thr_source),
                         misfit=MISFIT_TAG, sealed=sealed)

    # ---------------------------------------------------------------- ingest
    def motion(self, obs: M.Observation) -> np.ndarray:
        """(M,) float: 1 motion, 0 no motion, NaN MISSING (never thresholded). With the PLACEHOLDER threshold (sized for
        0/1 moving-part counts) a non-integer zone_motion (a continuous energy, e.g. stat_v1's ROI energy ratio, idle
        median 1.0) is refused (PlaceholderConstantError, fail closed, S3b fix2): such a stream needs its own threshold
        (`config_for_stat_v1`), never 0.5."""
        zm = np.asarray(obs.zone_motion, float)
        if self._thr_placeholder:
            f = zm[np.isfinite(zm)]
            if (f != np.round(f)).any():
                raise LC.PlaceholderConstantError(
                    f"zone_motion {f.tolist()} is a continuous energy but motion_thr {self.cfg.motion_thr} is a "
                    f"placeholder ({self.cfg.motion_thr_source}): use the camera model's threshold "
                    "(config_for_stat_v1 for stat_v1)")
        # T-W6 / M-W10 (fail closed, observer side): a zone not fully seen (zone_vis < ZONE_VIS_MIN) is MISSING whatever
        # the producer wrote, and so is every zone while any camera's floor view is invalid (cam_ok & cam_floor_ok): the
        # spec does not say which camera sees which zone, so a dead camera's zero energy is never read as idle
        seen = np.asarray(obs.zone_vis, float) >= M.ZONE_VIS_MIN
        if not M.floor_ok_of(obs).all():
            seen[:] = False
        return np.where(np.isfinite(zm) & seen, (zm > self.cfg.motion_thr).astype(float), np.nan)

    @staticmethod
    def _stop_after(md: MachineModel) -> float:
        """Observed no-motion seconds after which a press reads STOPPED (learn_cycles.forecast's idle + 3 sigma)."""
        P, b = md.period_s, md.busy_fraction
        return (1 - b) * P + 3 * max(1.4 * md.jitter * P, 0.5)

    def misfit(self, i: int, t: float) -> bool:
        """Press i's model was contradicted within misfit_hold_s of t: a quiet run past its stop threshold ended in
        motion (a restart, or a cycle longer than the model: the stale model would read every new idle as STOPPED)."""
        return t - float(self._misfit_t[i]) < self.cfg.misfit_hold_s

    def _cond(self, t: float, y: np.ndarray) -> M.Cond:
        bins = np.full(self.M, -1, np.int16)
        period, busy, jit = (np.full(self.M, np.nan) for _ in range(3))
        for i, md in enumerate(self.models):
            if md.confidence < LC.MIN_CONF or not np.isfinite(y[i]) or self.misfit(i, t):
                continue
            P, b = md.period_s, md.busy_fraction
            bins[i] = M.STOP_BIN if self._quiet[i] > self._stop_after(md) else 0
            period[i], busy[i], jit[i] = P, b, md.jitter
        return M.Cond(t, M.tod_bin(t), M.RUNNING, bins, np.full(self.M, np.nan), period, busy, jit)

    def ingest(self, obs: M.Observation) -> M.Cond:
        if not obs.is_v2:
            raise ValueError("observer_live takes Observation v2 only")
        M.check_obs_v2(obs, self.M, len(self.spec["cameras"]))
        t = float(obs.t_cap)
        if self.t_last is not None and t <= self.t_last:
            raise ValueError(f"tick at {t} not after {self.t_last}: the stream must be in time order")
        y = self.motion(obs)
        dt = 1.0 / self.cfg.hz
        # observed seconds without motion since the last motion OR gap (MISSING tick, or ticks absent from the stream):
        # the same gap-free evidence learn_cycles.forecast needs for STOPPED (a stroke may hide inside any gap)
        gap = self.t_last is not None and t - self.t_last > 1.5 * dt
        for i, md in enumerate(self.models):           # misfit: an observed would-be STOPPED that ended in motion
            if y[i] == 1.0 and md.confidence >= LC.MIN_CONF and self._quiet[i] > self._stop_after(md):
                self._misfit_t[i] = t
                self.n_misfit[i] += 1
        q = np.where(y == 0.0, self._quiet + dt, 0.0)
        self._quiet = np.where(y == 0.0, dt, q) if gap else q
        self._t.append(t)
        self._y.append(y)
        keep = int(max(self.cfg.learn_window_s, 400.0) * self.cfg.hz) + 10
        if len(self._t) > 2 * keep:
            del self._t[:-keep], self._y[:-keep]
        long_gap = self.t_last is not None and t - self.t_last > self.cfg.learn_every_s
        if self.t_first is None:
            self.t_first = t
            self._next_learn = t + self.cfg.learn_first_s
        elif long_gap:
            # a stream gap longer than the re-learn period (e.g. learning period -> mission): the window would hold
            # (almost) only post-gap ticks, so restart the schedule instead of re-fitting on them now
            self._next_learn = t + self.cfg.learn_every_s
        self.t_last = t
        if self._next_learn is not None and t >= self._next_learn:
            self.relearn()
            self._next_learn = t + self.cfg.learn_every_s
        cond = self._cond(t, y)
        self.map.update(obs, t, cond)
        self.events += self.det.step(obs, cond)
        self.n_ticks += 1
        return cond

    def relearn(self) -> None:
        T, Y = np.array(self._t), np.array(self._y, float).reshape(len(self._t), self.M)
        k = T >= T[-1] - self.cfg.learn_window_s
        for i in range(self.M):
            L = LC.learn(T[k], Y[k, i], index=i, latency_s=self.cfg.latency_s, consts=self.consts[i])
            observed_s = float(np.isfinite(Y[k, i]).sum()) / self.cfg.hz
            if (L.model.confidence < LC.MIN_CONF and self.models[i].confidence >= LC.MIN_CONF
                    and observed_s < 0.5 * self.cfg.learn_window_s):
                # a non-confident re-fit on a window less than half OBSERVED (MISSING bursts, a stream gap) is no
                # evidence against the confident model: keep it.
                # MISSING ticks stay UNKNOWN in the fast layer, and a contradicted model is caught by `misfit`
                continue
            self.learned[i] = L
            self.models[i] = L.model

    def advance(self, t: float) -> int:
        """Ingest every tick of obs_v2.jsonl that has ARRIVED by t (obs.t <= t: capture + latency). Returns the count."""
        if self.reader is None:
            raise RuntimeError("no obs_v2.jsonl path given")
        self._pending += self.reader.poll()
        n = 0
        while self._pending and self._pending[0].t <= t:
            o = self._pending.pop(0)
            if self.t_last is not None and float(o.t_cap) <= self.t_last:    # duplicated / reordered line: drop it
                self.n_dup += 1
                continue
            self.ingest(o)
            n += 1
        return n

    # ---------------------------------------------------------------- layers
    def slow(self, t: float) -> SlowLayer:
        return SlowLayer(self.spec["site_id"], self.spec["key_id"], DAY0_UNIX + t, [], list(self.models))

    def fast(self, t: float) -> FastLayer:
        self.seq = (self.seq + 1) & 0xFFFF
        out = []
        stale = self.t_last is None or t - self.t_last > self.cfg.stale_s
        T = np.array(self._t)
        Y = np.array(self._y, float).reshape(len(self._t), self.M) if self._t else np.zeros((0, self.M))
        for i, md in enumerate(self.models):
            if stale or md.confidence < LC.MIN_CONF:
                out.append(MachineNow(i, State.UNKNOWN, 0.0, 0.0, 0.0, 0, self.seq))
                continue
            k = T >= t - max(6 * md.period_s, 200.0) - 20.0
            now = LC.nowcast(md, T[k], Y[k, i], t, seq=self.seq, latency_s=self.cfg.latency_s, consts=self.consts[i])
            if now.state == State.STOPPED and self.misfit(i, t):        # a contradicted model: never a STOPPED
                now = MachineNow(i, State.UNKNOWN, 0.0, 0.0, 0.0, 0, self.seq)
            out.append(now)
        return FastLayer(self.spec["site_id"], self.spec["key_id"], DAY0_UNIX + t, self.seq, out)

    def pred_l1(self, i: int, bins=TWIN_BINS) -> np.ndarray:
        """Press i's L1 grid [nx, nz] for a lookup of map bins `bins` on the LIVE map: what it marks occupied (envelope
        union over the twin bins of `bins`) OR UNKNOWN (the change detector, fail-closed unlearned cells included), on
        the cell_m = env_res grid, before inflation. Default: the union of both twin bins (the fail-closed lookup).
        For diagnostics (availability) only: the M-W7 calibration pred is `calibration_pred`, from the HELD map."""
        tb = sorted({twin_bin(b) for b in bins})
        if not tb:
            raise ValueError("empty bin lookup")
        env = self.map.envelope_union(self.ids[i], tb, self.det.cfg.p_min)
        return np.asarray(env, bool) | self.det.unknown_mask(self.ids[i])

    def inflated_l1(self, i: int, bins=TWIN_BINS) -> np.ndarray:
        """What the robot gets for that lookup: pred_l1 inflated by the frozen margin (= the broadcast per-bin envelopes
        of `bins` OR the broadcast UNKNOWN cells of the press grid)."""
        return inflate(self.pred_l1(i, bins), self.margin, cell_m=float(self.spec["env_res"]))

    def unknown_l1(self, i: int) -> np.ndarray:
        """Press i's broadcast UNKNOWN cells: the detector's UNKNOWN (CUSUM + fail-closed unlearned), inflated."""
        return inflate(self.det.unknown_mask(self.ids[i]), self.margin, cell_m=float(self.spec["env_res"]))

    def calibration_pred(self, slow: SlowLayer, fast: FastLayer, map_held: HeldMap, t: float,
                         horizon_s: float = ROBOT_HORIZON_S, step_s: float = ROBOT_STEP_S,
                         fast_max_age_s: float = ROBOT_FAST_MAX_AGE_S,
                         held: Sequence[bool] | None = None) -> list[np.ndarray]:
        """S7b's dev recorder: per press, the un-inflated pred of the lookup the robot makes at plant time t over its
        keep-out horizon (`robot_bins`), computed ONLY from what the robot holds at t: the slow and fast layers last
        broadcast to it (quantised here as the robot decodes them) and the map version it completed (`map_held`, from
        sign_layers()["held"]["map"] / `hold`, whose (version, sha) is the robot's MapReceiver.manifest). Never this
        observer's current map, detector or models. inflate(pred) is then the held map's broadcast per-bin envelope
        OR its broadcast UNKNOWN cells for that lookup, i.e. what the robot gets. held: per press (zone order) whether
        the robot holds it at t (inside / reaching into its zone, masks.MachineView.held) -> its STOP_BIN lookup; None =
        the robot holds no press."""
        if tuple(map_held.ids) != tuple(self.ids):
            raise ValueError(f"held map presses {map_held.ids} != observer presses {self.ids}")
        hold = [False] * self.M if held is None else [bool(h) for h in held]
        if len(hold) != self.M:
            raise ValueError(f"held flags {len(hold)} != presses {self.M}")
        slow, fast = quantize_slow(slow), quantize_fast(fast)
        models = {m.index: m for m in slow.machines}
        nows = {m.index: m for m in fast.machines}
        issue_t = float(fast.issue_time) - DAY0_UNIX
        return [map_held.pred(i, robot_bins(models.get(i), nows.get(i), issue_t, t, float(map_held.period[i]),
                                            horizon_s, step_s, fast_max_age_s, held=hold[i]))
                for i in range(self.M)]

    def hold(self, bundle: ML.MapBundle) -> HeldMap:
        """The HeldMap of a bundle packed from this observer's map NOW (call it right after pack_map, as sign_layers
        does: the envelopes and UNKNOWN cells are copied from the live state, the period is decoded from the bundle)."""
        man = bundle.manifest
        decoded = ML.decode_payload(zlib.decompress(b"".join(bundle.chunks[:man.n_chunks])))
        if [d["id"] for d in decoded.spec["machines"]] != self.ids:
            raise ValueError("bundle presses differ from this observer's")
        p_min = self.det.cfg.p_min
        env = tuple({b: np.asarray(self.map.envelope(pid, b, p_min), bool).copy() for b in TWIN_BINS} for pid in self.ids)
        unknown = tuple(np.asarray(self.det.unknown_mask(pid), bool).copy() for pid in self.ids)
        return HeldMap(int(man.version), bytes(man.sha), tuple(self.ids), env, unknown, np.array(decoded.period, float))

    def map_snapshot(self) -> TwinMapSnapshot:
        """The broadcast copy of the live map: per-bin inflated envelopes (`TwinMapSnapshot.envelope`), and
        unknown_rects = the live map's own plus a safe box cover of every press grid's inflated detector UNKNOWN."""
        boxes = list(self.map.unknown_rects)
        res = float(self.spec["env_res"])
        for i in range(self.M):
            U = self.unknown_l1(i)
            if not U.any():
                continue
            x0, z0 = float(self.map.l1_x0[i]), float(self.map.l1_z0[i])
            for i0, j0, i1, j1 in M.safe_box_cover(U, 12):
                boxes.append((x0 + i0 * res, z0 + j0 * res, x0 + i1 * res, z0 + j1 * res))
        return TwinMapSnapshot.of(self.map, self.margin, boxes)

    def availability(self) -> dict:
        """What failing closed costs: share of press-grid cells UNKNOWN only because a twin bin is not learned yet
        (M-W4 lead decision); share of press-grid cells broadcast UNKNOWN after the M-W7 inflation; and per lookup
        (running = bin 0, stopped = STOP_BIN, union = fail-closed) the share of press-grid cells the robot gets closed."""
        def share(grids: list[np.ndarray]) -> float:
            n = sum(int(a.size) for a in grids)
            return float(sum(int(a.sum()) for a in grids) / n) if n else 0.0

        return dict(unlearned=self.det.availability_cost(),
                    inflated_unknown_share=share([self.unknown_l1(i) for i in range(self.M)]),
                    closed_share={k: share([self.inflated_l1(i, b) for i in range(self.M)])
                                  for k, b in (("running", (0,)), ("stopped", (M.STOP_BIN,)), ("union", TWIN_BINS))},
                    margin_m=self.margin.margin_m)

    def diagnostics(self) -> dict:
        def miss(L: LC.Learned | None):
            return None if L is None else L.diag.get("missing_share")

        return dict(n_ticks=self.n_ticks, t_first=self.t_first, t_last=self.t_last, tags=self.tags,
                    n_misfit=self.n_misfit.tolist(), n_dup=self.n_dup,
                    models=[dict(id=self.ids[i], period_s=m.period_s, confidence=m.confidence,
                                 missing_share=miss(self.learned[i])) for i, m in enumerate(self.models)],
                    availability=self.availability(), events=len(self.events))

    # ---------------------------------------------------------------- signing (C-W5)
    def sign_layers(self, key: SigningKey, t: float, keyring_now: Keyring, map_version: int = 1) -> dict:
        """Signed slow and fast blobs and the map bundle, all under key_id_of(key), and `held`: the decoded slow and
        fast layers and the bundle's HeldMap (`hold`), what a robot holds once it has them (calibration_pred's
        inputs). Refuses (TrustError) when that key is not the one the keyring valid now holds under that id: the
        robot would drop everything (fail closed)."""
        kid = key_id_of(key.verify_key)
        vk = keyring_now.get(kid)
        if vk is None or vk.encode() != key.verify_key.encode():
            raise TrustError(f"signing key {kid} is not in the keyring valid now: refusing to broadcast")
        s = self.slow(t)
        f = self.fast(t)
        s_pay = encode_slow(SlowLayer(s.site_id, kid, s.issue_time, s.traffic, s.machines))
        f_pay = encode_fast(FastLayer(f.site_id, kid, f.issue_time, f.seq, f.machines))
        m2 = self.map_snapshot()
        m2.spec = dict(m2.spec, key_id=kid)
        bundle = ML.pack_map(m2, key, map_version, issue_time=DAY0_UNIX + t, p_min=self.det.cfg.p_min)
        held = dict(slow=decode_slow(s_pay), fast=decode_fast(f_pay), map=self.hold(bundle))
        return dict(slow=sign(s_pay, key), fast=sign(f_pay, key), map=bundle, key_id=kid, held=held)


def verify_blob(blob: bytes, keyring: Keyring):
    """Receiver check of a slow / fast blob against an explicit keyring: the decoded layer, or TrustError (malformed,
    unknown key, bad signature, or a header key_id that is not the verifying key's)."""
    ok, payload, kid, reason = verify(blob, keyring)
    if not ok:
        raise TrustError(f"layer rejected: {reason}")
    if key_id_of(keyring[kid]) != kid:
        raise TrustError(f"key_id {kid} is not derived from its verify key")
    layer = peek_header(payload).layer
    if layer == SLOW:
        return decode_slow(payload)
    if layer == FAST:
        return decode_fast(payload)
    raise TrustError(f"unexpected layer {layer}")
