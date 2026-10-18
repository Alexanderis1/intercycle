"""Entry into a STOPPED twin press (lane S5b-entry, lead decision), pure logic, no ROS.

Learned: a twin press's (masks.Site.own_stop) own confident STOPPED fast entry inside its TTL is one idle window
[issue, issue + ttl] (predict.idle_windows stop_window), so the zone rect opens while it covers [t, t + H]
(masks.idle_through) and the SAME window planner (timing.TimingCore.plan_target + window_open) plans an entry while it
leaves need_s. UNKNOWN, missing, low confidence, expired, or a TTL that no longer covers the horizon / need_s keep the
zone closed and give no entry (fail closed). Reactive: its own-camera STILL-run window feeds the same planner and the
same zone rule. Body shop: a STOPPED still gives no window (unchanged).

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test/test_stop_entry.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))

from intercycle import scenario as S  # noqa: E402
from intercycle.codec.layout import encode_fast, encode_slow  # noqa: E402
from intercycle.codec.sign import demo_key, sign  # noqa: E402
from intercycle.codec.trust import key_id_of  # noqa: E402
from intercycle.observer import learn_cycles as LC  # noqa: E402
from intercycle.observer import mapping as M  # noqa: E402
from intercycle.robot.predict import idle_windows  # noqa: E402
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State  # noqa: E402

from intercycle_nav import feed as F  # noqa: E402
from intercycle_nav import masks as K  # noqa: E402
from intercycle_nav import reactive as R  # noqa: E402
from intercycle_nav.timing import TimingCore, window_open  # noqa: E402

SEED = 7
T0 = 6 * 3600.0
TWIN_Z9 = REPO / "twin" / "z9" / "scenario.yaml"
H = K.MaskParams().horizon_s

needs_twin = pytest.mark.skipif(not TWIN_Z9.exists(), reason="twin/z9/scenario.yaml not built")


def _state(site: K.Site, nows: list[tuple[State, int, float, float]], model_conf: int = 10,
           t: float = T0, models: list[MachineModel] | None = None) -> K.ObserverState:
    """Signed slow + fast layers for every machine of `site`: nows[i] = (state, confidence, ttl_s, next_idle_len_s).
    models: the slow layer's models (default: a hand-set 2 s press model per machine with `model_conf`)."""
    key = demo_key(SEED)
    kid = key_id_of(key.verify_key)
    st = K.ObserverState(site, K.demo_keyring(SEED))
    if models is None:
        models = [MachineModel(i, 2.0, 0.4, 0.01, 0.0, 1.0, model_conf) for i in range(len(site.ids))]
    fast = [MachineNow(i, s, 0.0, ln, ttl, c, 1) for i, (s, c, ttl, ln) in enumerate(nows)]
    st.ingest(t, F.BLOB + sign(encode_slow(SlowLayer(site.site_id, kid, DAY0_UNIX + t, [], models)), key))
    st.ingest(t, F.BLOB + sign(encode_fast(FastLayer(site.site_id, kid, DAY0_UNIX + t, 1, fast)), key))
    assert st.stats["ok"] == 2, st.all_stats()
    return st


def _twin():
    scn = S.load(str(TWIN_Z9))
    core = TimingCore.from_scenario(scn, scenario_path=str(TWIN_Z9), enter_s=5.0, exit_s=6.5,
                                    dwell_s=scn.inspection.dwell_s, margin_s=scn.inspection.safety_margin_s)
    tg = next(t for t in core.graph.mission() if t.kind == "machine")
    return K.Site.from_scenario(str(TWIN_Z9)), core, tg


def _planner_windows(snap: K.Snapshot) -> dict:
    """What timing_planner reads: the /intercycle/observer/state JSON's per-machine windows."""
    return {mid: m["windows"] for mid, m in K.snapshot_json(snap, "learned")["machines"].items()}


def _plans_entry(core: TimingCore, tg, windows: dict, t: float, condition: str) -> bool:
    """The planner's whole entry path at the boundary: a SIPP plan to the inside node, then (a) window_open with need_s
    (the camera check (b) is the same for both arms and is not modelled here)."""
    xy = core.graph.pos[tg.node]                         # standing at the wait pose
    tp = core.plan_target(tg, t, xy, windows, condition)
    if tp is None or tp.inside_xy is None:
        return False
    return window_open(windows.get(tg.machine), t, core.need_s(tg)) is not None


# ---------------------------------------------------------------- predict.idle_windows
def test_idle_windows_stop_window_rules():
    mod = MachineModel(0, 2.0, 0.4, 0.01, 0.0, 1.0, 10)
    issue = DAY0_UNIX + T0
    stop = MachineNow(0, State.STOPPED, 0.0, 0.0, 30.0, 8, 1)
    assert idle_windows(mod, stop, issue, T0 + 1.0) is None                     # default: unchanged (None)
    assert idle_windows(mod, stop, issue, T0 + 1.0, stop_window=True) == [(T0, T0 + 30.0)]
    assert idle_windows(mod, stop, issue, T0 + 30.5, stop_window=True) is None  # past its TTL
    low = MachineNow(0, State.STOPPED, 0.0, 0.0, 30.0, 3, 1)
    assert idle_windows(mod, low, issue, T0 + 1.0, stop_window=True) is None
    assert idle_windows(MachineModel(0, 2.0, 0.4, 0.01, 0.0, 1.0, 3), stop, issue, T0 + 1.0, stop_window=True) is None
    unk = MachineNow(0, State.UNKNOWN, 0.0, 0.0, 30.0, 8, 1)
    assert idle_windows(mod, unk, issue, T0 + 1.0, stop_window=True) is None
    act = MachineNow(0, State.ACTIVE, 0.5, 0.8, 10.0, 8, 1)                    # running: unchanged by the flag
    assert idle_windows(mod, act, issue, T0 + 0.1, stop_window=True) == idle_windows(mod, act, issue, T0 + 0.1)


# ---------------------------------------------------------------- learned arm, twin press
@needs_twin
def test_learned_plans_an_entry_into_a_confidently_stopped_press():
    site, core, tg = _twin()
    assert core.need_s(tg) == pytest.approx(16.5)
    st = _state(site, [(State.STOPPED, 8, 30.0, 0.0)])
    t = T0 + 0.5
    snap = st.snapshot(t)
    mv = snap.machines[0]
    assert mv.stop_ok and mv.windows == [(T0, T0 + 30.0)]
    assert K.idle_through(mv.windows, t, t + H)                                 # zone rect open (light curtain rule)
    assert _plans_entry(core, tg, _planner_windows(snap), t, "learned")
    # the same stop late in its TTL: no longer covers the horizon -> zone closed, and need_s not left -> no entry
    late = st.snapshot(T0 + 26.0)
    assert not K.idle_through(late.machines[0].windows, T0 + 26.0, T0 + 26.0 + H)
    assert not _plans_entry(core, tg, _planner_windows(late), T0 + 26.0, "learned")
    # past its TTL: no window at all
    assert st.snapshot(T0 + 30.5).machines[0].windows is None


@needs_twin
@pytest.mark.parametrize("state, conf, ttl, idle_len", [
    (State.ACTIVE, 8, 10.0, 0.8),          # RUNNING: strokes 2 s apart, no window covers H or need_s
    (State.IDLE, 8, 10.0, 0.8),
    (State.UNKNOWN, 0, 10.0, 0.0),
    (State.STOPPED, 3, 30.0, 0.0),         # low confidence
    (State.STOPPED, 8, 10.0, 0.0),         # the learner's STOPPED TTL cap (10 s) < need_s 16.5 s: no entry
])
def test_learned_never_enters_running_unknown_or_unconfident(state, conf, ttl, idle_len):
    site, core, tg = _twin()
    st = _state(site, [(state, conf, ttl, idle_len)])
    t = T0 + 0.1
    snap = st.snapshot(t)
    w = snap.machines[0].windows
    assert not _plans_entry(core, tg, _planner_windows(snap), t, "learned")
    if state != State.STOPPED or conf < 4:
        assert not K.idle_through(w, t, t + H)


@needs_twin
def test_learned_missing_fast_layer_gives_no_entry():
    site, core, tg = _twin()
    st = K.ObserverState(site, K.demo_keyring(SEED))                            # nothing received
    snap = st.snapshot(T0)
    assert snap.machines[0].windows is None and snap.machines[0].state == "UNKNOWN"
    assert not _plans_entry(core, tg, _planner_windows(snap), T0, "learned")


def test_stop_window_only_for_own_stop_off_line_machines():
    """Body-shop semantics unchanged: an on-line machine, or an off-line one without own_stop, gets no stop window."""
    ids = ["a", "b", "c"]
    rects = {m: (4.0 * i, 0.0, 4.0 * i + 2.0, 2.0) for i, m in enumerate(ids)}
    site = K.Site(ids, rects, [], 1, SEED, (-2.0, -2.0, 14.0, 4.0), on_line={"a": True, "b": False, "c": False},
                  own_stop={"a": True, "b": False, "c": True})
    st = _state(site, [(State.STOPPED, 8, 30.0, 0.0)] * 3)
    w = [m.windows for m in st.snapshot(T0 + 0.5).machines]
    assert w == [None, None, [(T0, T0 + 30.0)]]


def test_body_shop_stopped_machine_still_has_no_window():
    site = K.Site.from_scenario()
    assert not any(site.own_stop.values())
    st = _state(site, [(State.STOPPED, 8, 30.0, 0.0)] * len(site.ids))
    assert all(m.windows is None for m in st.snapshot(T0 + 0.5).machines)


# ---------------------------------------------------------------- reactive arm, same planner and zone rule
def _reactive_still_history(est: R.OwnCameraEstimator, mid: str, t: float, n_stops: int, stop_s: float,
                            run_s: float = 10.0, dt: float = 0.5) -> float:
    """Feed n_stops cycles of (moving run_s, still stop_s) at 2 Hz; ends moving. Returns the time after the last one."""
    for _ in range(n_stops):
        for k in range(int(run_s / dt)):
            est.observe(t + k * dt, mid, True, True)
        t += run_s
        for k in range(int(stop_s / dt)):
            est.observe(t + k * dt, mid, True, False)
        t += stop_s
    for k in range(int(run_s / dt)):
        est.observe(t + k * dt, mid, True, True)
    return t + run_s


@needs_twin
def test_reactive_plans_an_entry_into_a_stopped_press_from_its_own_camera():
    site, core, tg = _twin()
    mid = tg.machine
    assert mid is not None
    est = R.OwnCameraEstimator(R.ReactiveConfig(), [mid], "dev_w1")
    t = _reactive_still_history(est, mid, T0, n_stops=5, stop_s=60.0)
    for k in range(int(4.0 / 0.5)):                                             # the press stops now: still for 4 s
        est.observe(t + k * 0.5, mid, True, False)
    tn = t + 3.5
    w = est.windows(tn)
    assert w[mid] and R.zone_open(w[mid], tn, H)                                # the same light-curtain rule as learned
    planner = {m: d["windows"] for m, d in R.snapshot_json(tn, est)["machines"].items()}
    assert _plans_entry(core, tg, planner, tn, "reactive")


@needs_twin
def test_reactive_never_enters_running_or_unknown():
    site, core, tg = _twin()
    mid = tg.machine
    assert mid is not None
    est = R.OwnCameraEstimator(R.ReactiveConfig(), [mid], "dev_w1")
    t = _reactive_still_history(est, mid, T0, n_stops=5, stop_s=60.0)
    planner = lambda tt: {m: d["windows"] for m, d in R.snapshot_json(tt, est)["machines"].items()}   # noqa: E731
    # RUNNING: the last samples are motion
    assert est.windows(t - 0.5)[mid] == [] and not _plans_entry(core, tg, planner(t - 0.5), t - 0.5, "reactive")
    # UNKNOWN: nothing seen for longer than fresh_s
    assert est.windows(t + 5.0)[mid] is None and not _plans_entry(core, tg, planner(t + 5.0), t + 5.0, "reactive")
    # stroke gaps only (still runs of 1 s): no history of a still run long enough -> no window
    est2 = R.OwnCameraEstimator(R.ReactiveConfig(), [mid], "dev_w2")
    t2 = _reactive_still_history(est2, mid, T0, n_stops=20, stop_s=1.0, run_s=1.0)
    for k in range(6):
        est2.observe(t2 + k * 0.5, mid, True, False)
    w2 = est2.windows(t2 + 2.5)[mid]
    assert not R.zone_open(w2, t2 + 2.5, H)
    assert not _plans_entry(core, tg, {mid: w2}, t2 + 2.5, "reactive")


def test_window_planner_is_shared():
    """Both arms read windows with the same rules: idle_through (masks) == zone_open (reactive) on the same windows."""
    for w, t in (([(0.0, 30.0)], 1.0), ([(0.0, 30.0)], 25.0), ([], 1.0), (None, 1.0), ([(2.0, 30.0)], 1.0)):
        assert K.idle_through(w, t, t + H) == R.zone_open(w, t, H)
    assert not math.isnan(H)



# ---------------------------------------------------------------- the real learner's STOPPED TTL (fix round 1, major)
def _press_signal(t0: float, t1: float, stops: list[tuple[float, float]], period_s: float = 2.4,
                  busy: float = 0.4, dt: float = 0.5):
    """A 2 Hz zone-motion signal of a press stroking every period_s (busy share `busy`), still during `stops`."""
    t = np.arange(t0, t1, dt)
    y = (((t - t0) % period_s) < busy * period_s).astype(float)
    for a, b in stops:
        y[(t >= a) & (t < b)] = 0.0
    return t, y


@needs_twin
def test_pinned_asymmetry_real_learner_stop_never_enters_while_reactive_does():
    """PINS A DISCLOSED ASYMMETRY (fix round 1 of lane S5b-entry, major; needs a lead decision in prereg/AMENDMENTS.md
    before FREEZE). The learned arm fed by the REAL learner (learn_cycles.learn + nowcast, press constants) on a long
    stop: forecast caps a STOPPED TTL at min(ttl_s, 10 s), and ttl_s itself is capped at 10 / stops_per_h (here ~4
    stops / h -> ~2.6 s), so the stop window [issue, issue + ttl] never leaves need_s (16.5 s): no entry, all stop long.
    The reactive arm, from its own camera, has seen >= 3 long still runs and plans an entry at the stop onset. When the
    lead changes the learner's STOPPED TTL, this test must be replaced by the entry test the decision calls for."""
    site, core, tg = _twin()
    mid = tg.machine
    assert mid is not None and site.ids.index(mid) == 0
    need = core.need_s(tg)
    stop0 = T0 + 1500.0                                                         # the long stop: 200 s
    stops = [(T0 + 200.0 * k, T0 + 200.0 * k + 60.0) for k in range(1, 7)] + [(stop0, stop0 + 200.0)]
    t, y = _press_signal(T0, stop0 + 200.0, stops)
    press = LC.KINDS["press"]
    learned = LC.learn(t[t < stop0 - 100.0], y[t < stop0 - 100.0], consts=press)
    md = learned.model
    assert md.confidence >= LC.MIN_CONF
    for el in (10.0, 30.0, 60.0, 120.0, 190.0):                                 # the whole long stop
        tn = stop0 + el
        k = (t <= tn) & (t >= tn - max(6 * md.period_s, 200.0) - 20.0)          # observer_live.fast's window
        now = LC.nowcast(md, t[k], y[k], tn, consts=press)
        assert now.state == State.STOPPED and now.confidence >= 4
        assert now.ttl_s <= 10.0 < need                                         # the learner's cap: the reason
        st = _state(site, [(now.state, now.confidence, now.ttl_s, 0.0)], t=tn, models=[md])
        snap = st.snapshot(tn + 0.5)
        w = snap.machines[0].windows
        assert w is not None and len(w) == 1 and w[0][0] == pytest.approx(tn, abs=1e-3)   # a stop window, short
        assert not _plans_entry(core, tg, _planner_windows(snap), tn + 0.5, "learned")
    # reactive, its own camera on the same press (seen continuously here): enters the same long stop at its onset
    est = R.OwnCameraEstimator(R.ReactiveConfig(), [mid], "dev_w1")
    tr = stop0 + 4.0
    for tt, yy in zip(t, y):
        if tt > tr:
            break
        est.observe(float(tt), mid, True, bool(yy > 0.5))
    wr = est.windows(tr)[mid]
    assert wr and R.zone_open(wr, tr, H)
    planner = {m: d["windows"] for m, d in R.snapshot_json(tr, est)["machines"].items()}
    assert _plans_entry(core, tg, planner, tr, "reactive")

# ---------------------------------------------------------------- held -> STOP bin
def test_machine_bins_held_reads_the_stop_bin():
    """masks.machine_bins: the robot inside a zone holds the press -> (STOP_BIN,), the rule robot_bins mirrors."""
    mv = K.MachineView("p", 0, "UNKNOWN", None, None, held=True)
    assert K.machine_bins(None, 0, mv, M.RUNNING, K.MaskParams(), on_line=False, own_stop=True) == (M.STOP_BIN,)  # type: ignore[arg-type]
