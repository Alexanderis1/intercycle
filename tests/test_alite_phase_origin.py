"""A-lite T-W2 (PLAN_TWIN §7 item 3): the robot's phase and the map's phase share one origin, within 0.5 bin.

The A-lite plant (KeyedWorld + ProgramKinematics) is watched through the measured-error camera (MEASURED_SHIM); the
A-lite observer learns its slow layer and the phase origin on the first 30 min, then for 15 min of fresh data the
robot's own_phase (ros2/intercycle_nav masks.own_phase, from the fast layer) is compared with the map's phase (the
tracker's, from the skid index) at the same instants. The systematic offset (circular median of the difference) must
be <= 0.5 bin per machine after the correction, and is > 0.5 bin without it for the stations that lag the index by
>= 3 s (T-W2: up to 2.8 bins on the body shop).

Disclosed, not T-W2: the framing station S50 (long dwell pauses inside its active time) has a bimodal nowcast under the
measured noise (mid-cycle pauses read as idle), so its per-sample error has large tails whatever the origin; only its
systematic offset is pinned here.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from intercycle.alite import phase_origin as PO
from intercycle.alite.observer import AliteObserver
from intercycle.alite.plant import AlitePlant
from intercycle.observer import mapping as M
from intercycle.scenario import load
from intercycle.types import DAY0_UNIX, MachineModel, MachineNow, State

NAV = Path(__file__).resolve().parents[1] / "ros2" / "intercycle_nav"
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))
from intercycle_nav.masks import own_phase  # noqa: E402  # pyright: ignore[reportMissingImports]

SCN = load()
T0 = 7 * 3600.0
LEARN_S, EVAL_S = 30 * 60.0, 15 * 60.0


@pytest.fixture(scope="module")
def run():
    pl = AlitePlant(SCN, 801, T0, T0 + LEARN_S + EVAL_S)
    ob = AliteObserver(SCN, change_detection=False)
    for k in range(int(LEARN_S * M.FRAME_HZ)):
        ob.step(pl.observe(T0 + k / M.FRAME_HZ))
    ob.learn_slow(T0, T0 + LEARN_S)
    assert ob.models is not None and ob.origin is not None
    corr = {i: [] for i in range(len(SCN.machines))}
    raw = {i: [] for i in range(len(SCN.machines))}
    for k in range(int(EVAL_S * M.FRAME_HZ)):
        t = T0 + LEARN_S + k / M.FRAME_HZ
        cond = ob.step(pl.observe(t))
        if k % 10:
            continue
        f = ob.fast(t)
        for i, mod in enumerate(ob.models):
            ph = own_phase(mod, f[i], DAY0_UNIX + t, t)
            if ph is None or cond.phase_bin[i] in (-1, M.STOP_BIN) or not np.isfinite(cond.phase[i]):
                continue
            c = PO.to_map_phase(ph % 1.0, ob.origin.lag(i), mod.period_s)
            raw[i].append(PO.bin_error(ph % 1.0, float(cond.phase[i])))
            if c is not None:
                corr[i].append(PO.bin_error(c, float(cond.phase[i])))
    return ob, corr, raw


def _offset_bins(errs):
    return PO.circular_median(np.asarray(errs) / M.N_PHASE) * M.N_PHASE


def test_robot_phase_formula_is_masks_own_phase():
    mod = MachineModel(0, 60.0, 0.7, 0.01, 0.0, 1.0, 12)
    for st, nin, nlen in ((State.ACTIVE, 12.0, 18.0), (State.IDLE, 0.0, 9.0), (State.ACTIVE, 50.0, 18.0)):
        now = MachineNow(0, st, nin, nlen, 5.0, 12, 1)
        for dt in (0.0, 1.5, 4.0, 6.0):
            a = own_phase(mod, now, DAY0_UNIX + 1000.0, 1000.0 + dt)
            b = PO.robot_phase(mod, now, 1000.0, 1000.0 + dt)
            assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12)


def test_lags_are_learned_from_the_skid_index(run):
    ob, _, _ = run
    lag = ob.origin.lag_s
    for i, m in enumerate(SCN.machines):
        if m.on_line and m.kind != "unload_lifter":
            assert np.isfinite(lag[i]), m.id
            print(f"{m.id}: learned lag {lag[i]:.2f} s, station stagger {m.offset_s} s")
            if m.id == "S50":       # its robot-side phase is bimodal (module docstring): the lag absorbs that bias
                continue
            assert abs(lag[i] - m.offset_s) < 1.0, (m.id, lag[i], m.offset_s)     # station stagger, observer-learned


def test_one_origin_within_half_a_bin(run):
    ob, corr, raw = run
    checked = 0
    for i, m in enumerate(SCN.machines):
        if not math.isfinite(ob.origin.lag(i)) or len(corr[i]) < 40:
            continue
        off = _offset_bins(corr[i])
        e = np.abs(corr[i])
        print(f"{m.id}: n {len(e)} corrected offset {off:+.2f} bins, |err| <= 0.5 bin {np.mean(e <= 0.5):.2f}; "
              f"uncorrected offset {_offset_bins(raw[i]):+.2f} bins")
        assert abs(off) <= 0.5, (m.id, off)
        checked += 1
    assert checked >= 8


def test_without_the_correction_late_stations_are_off_by_more_than_half_a_bin(run):
    _, _, raw = run
    for i, m in enumerate(SCN.machines):
        if m.on_line and m.offset_s >= 3.0 and m.kind != "unload_lifter":
            assert abs(_offset_bins(raw[i])) > 0.5, m.id


def test_unknown_lag_or_phase_fails_closed():
    assert PO.to_map_phase(0.3, math.nan, 60.0) is None
    assert PO.to_map_phase(None, 2.0, 60.0) is None
    assert PO.to_map_phase(0.3, 2.0, math.nan) is None
    assert PO.to_map_phase(0.95, 6.0, 60.0) == pytest.approx(0.05)
