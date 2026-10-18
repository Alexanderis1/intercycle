"""T-W2 for A-lite: one phase origin for the map's L1 bins and the robot's own_phase (observer + robot side; never
imports ground truth).

The defect (WEAKNESSES T-W2). The map's L1 bins of a line station start at the skid index: the ConditionTracker
(observer/mapping.py) takes a line station's phase as (t - last line index) / P, where the line index is the rise of
"a skid is moving". The robot's `own_phase` (ros2/intercycle_nav masks.py) starts at the zone's own motion start: it
is built from the fast layer, whose next_idle_in / next_idle_len come from learn_cycles' forecast of the zone's own
motion signal (ACTIVE: phase = busy - next_idle_in / P; IDLE: phase = 1 - next_idle_len / P). The two origins differ
by the station's lag behind the index (0-10.5 s on the body shop = up to 2.8 bins of 60 s / 16), plus whatever bias
the learned busy fraction puts into the robot's formula, while the robot pads only +-1 bin, so it can look up the
wrong envelope. In the press shop this is superseded by the topology (PLAN_TWIN §6); A-lite needs the fix, so it
lives here, in an A-lite-only module, and changes neither mapping.py nor masks.py.

The fix (WEAKNESSES T-W2 "real fix": the observer learns each station's lag from the skid index, puts it in the slow
layer, own_phase adds lag / P):
1. `LagCalibrator` (observer side). On its own learning window, after the slow layer is learned, the observer replays
   its fast layer at past issue times (samples up to each time only) and pairs the phase the robot would compute from
   it (`robot_phase`, the same formula as masks.own_phase) with the map's phase at that time (the tracker's phase from
   the skid index; ticks in STOP / unknown bins skipped). lag_i = the circular median of (map phase - robot phase)
   x P over the pairs (>= MIN_PAIRS, and >= MIN_AGREE of them within the robot's 1-bin pad of it, else unknown). Learned from the observer's own data, for every machine (cells
   too: their map origin is their own tracker rise, which may sit differently from learn_cycles' edges).
2. `PhaseOrigin` (slow layer): per machine index, the lag in seconds (NaN = unknown). It travels next to the slow
   layer's MachineModels (the body-shop codec's MachineModel has no lag field and the codec is not changed: A-lite
   hands this record to its robot in-process).
3. `to_map_phase` (robot side): map phase = own_phase + lag / P (mod 1), with P the slow layer's period. An unknown
   lag (or an unknown own_phase) gives None: the robot then uses the union of all bins (masks.py rule), fail closed.

tests/test_alite_phase_origin.py checks, through the noisy camera (MEASURED_SHIM) on fresh data after the learning
window, that the corrected phase and the map's phase agree within 0.5 bin, and that the uncorrected own_phase does not
for the late stations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.observer import mapping as M
from intercycle.types import DAY0_UNIX, MachineModel, MachineNow, State

MIN_PAIRS = 20
ROBOT_MIN_CONF = 4                      # masks.own_phase ignores entries / models below this
# A lag is trusted only when its own calibration pairs agree with it: at least MIN_AGREE of them within PAD_BINS of the
# learned lag (PAD_BINS = the robot's +-1 bin pad, masks.py). A machine whose robot-side phase is not a consistent
# estimate of the map's phase (S50 framing: dwell pauses read as idle, pairs spread over +-4 bins) has no single origin:
# its circular median is not a systematic offset but a point between modes that moves with any change of the nowcast
# (lane S8-alitefix: the T-W6 MISSING rules in learn_cycles moved it from -0.1 to -0.7 bin). Its lag is unknown (NaN)
# and the robot uses the union of all bins there (fail closed). Development worlds 801 / 802: S50 30 % within 1 bin,
# every other machine 82-100 %.
MIN_AGREE = 0.5
PAD_BINS = 1.0


@dataclass(frozen=True)
class PhaseOrigin:
    """Per machine (scenario order): lag of the robot's phase origin behind the map's, seconds (NaN = unknown)."""
    lag_s: tuple[float, ...]
    n: tuple[int, ...] = ()
    agree: tuple[float, ...] = ()       # share of the calibration pairs within PAD_BINS of the lag (NaN: < MIN_PAIRS)

    def lag(self, i: int) -> float:
        return float(self.lag_s[i]) if 0 <= i < len(self.lag_s) else math.nan


def robot_phase(model: MachineModel | None, now: MachineNow | None, t_issue: float, t: float) -> float | None:
    """The robot's own_phase (ros2/intercycle_nav masks.own_phase, same formula; plant time t_issue instead of the
    unix issue time): fraction of the cycle since the zone's motion started, not wrapped. None when not trusted."""
    if model is None or now is None:
        return None
    if now.confidence < ROBOT_MIN_CONF or model.confidence < ROBOT_MIN_CONF or \
            now.state in (State.STOPPED, State.UNKNOWN) or t > t_issue + now.ttl_s:
        return None
    T, b = float(model.period_s), float(model.busy_fraction)
    if not (T > 0 and np.isfinite(T)):
        return None
    if now.state == State.ACTIVE:
        ph = min(max(b - now.next_idle_in_s / T, 0.0), b)
    else:
        ph = min(max(1.0 - now.next_idle_len_s / T, b), 1.0)
    return ph + (t - t_issue) / T


def issue_unix(t_issue: float) -> float:
    return DAY0_UNIX + t_issue


def _wrap(x):
    return (np.asarray(x, float) + 0.5) % 1.0 - 0.5


def circular_median(d: np.ndarray) -> float:
    """Median of phase differences (cycles) around their circular mean, wrapped to [-0.5, 0.5)."""
    d = np.asarray(d, float)
    c = math.atan2(np.sin(2 * np.pi * d).mean(), np.cos(2 * np.pi * d).mean()) / (2 * np.pi)
    return float(_wrap(np.median(_wrap(d - c)) + c))


@dataclass
class LagCalibrator:
    n_machines: int
    pairs: list[list[float]] = field(default_factory=list)       # per machine: map - robot phase, cycles
    period: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.pairs = [[] for _ in range(self.n_machines)]
        self.period = [math.nan] * self.n_machines

    def add(self, i: int, map_phase: float, robot_ph: float, period_s: float) -> None:
        if np.isfinite(map_phase) and np.isfinite(robot_ph) and np.isfinite(period_s):
            self.pairs[i].append(float(_wrap(map_phase - robot_ph)))
            self.period[i] = float(period_s)

    def origin(self) -> PhaseOrigin:
        """Per machine: lag = circular median of its pairs x P; NaN (unknown) with fewer than MIN_PAIRS pairs or when
        fewer than MIN_AGREE of them lie within PAD_BINS of that median (no single origin, see MIN_AGREE)."""
        lag, n, agree = [], [], []
        for i in range(self.n_machines):
            p = np.array(self.pairs[i], float)
            n.append(len(p))
            if len(p) < MIN_PAIRS:
                lag.append(math.nan)
                agree.append(math.nan)
                continue
            c = circular_median(p)
            a = float(np.mean(np.abs(_wrap(p - c)) * M.N_PHASE <= PAD_BINS))
            agree.append(a)
            lag.append(c * self.period[i] if a >= MIN_AGREE else math.nan)
        return PhaseOrigin(tuple(lag), tuple(n), tuple(agree))


def to_map_phase(own_phase: float | None, lag_s: float, period_s: float) -> float | None:
    """The robot's phase in the map's convention: own_phase + lag / P, mod 1. None (fail closed: union of all bins)
    when own_phase, the lag or the period is unknown."""
    if own_phase is None or not np.isfinite(own_phase) or not np.isfinite(lag_s):
        return None
    if not (np.isfinite(period_s) and period_s > 0):
        return None
    return float((own_phase + lag_s / period_s) % 1.0)


def bin_error(a: float, b: float, n_bins: int = M.N_PHASE) -> float:
    """Signed circular difference a - b of two phases (fractions of a cycle), in bins."""
    return float(_wrap(a - b) * n_bins)
