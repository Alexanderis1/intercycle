"""T-W6 occlusion = MISSING (lane S4, PLAN_TWIN §2/§4; WEAKNESSES T-W6).

A zone_motion sample is MISSING (NaN) while a person or vehicle track overlaps the zone (floor) or the ram ROI (image).
Stress cases: 30-300 s occlusions, an operator standing in the zone, a parked vehicle, measured missed detections.
The camera here is CameraShim(MEASURED_SHIM) with physical occlusion added by the test: while the occluder stands in
front of the machine the real camera cannot see the machine's moving parts, so without the T-W6 rule the zone reads 0
("no motion" = idle, the fail-dangerous direction).

What the rule can and cannot do (causal, per tick): from the first tick the occluder is DETECTED until it leaves, every
sample is MISSING (the hold bridges missed detections). Ticks before its first detection cannot be told from an idle
machine; the tests pin that this onset gap is the only leak (expected p/(1-p) = 0.73 ticks per occlusion for an
occluder that appears in front of the machine at once under MEASURED_SHIM's per-tick miss p = 0.42).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer import occlusion as O  # noqa: E402
from intercycle.scenario import load  # noqa: E402

SCN = load()
NCAM = len(SCN.cameras)
NZ = len(SCN.machines)
ZONES = [m.rect for m in SCN.machines]
SHARE = M.zone_camera_share(ZONES, SCN.cameras)
K = [m.id for m in SCN.machines].index("S30")            # the machine under test, rect (26, 18.5, 34, 29.5)
T0 = 3600.0
SHIM: dict[str, Any] = dict(M.MEASURED_SHIM)
CLEAN: dict[str, Any] = dict(SHIM, gross_p=0.0, fp_ratio=0.0)    # measured drops and noise, no gross errors / FPs


@dataclass
class Run:
    s: np.ndarray          # (T,) seconds from start
    zm: np.ndarray         # (T,) zone_motion of S30
    occ: np.ndarray        # (T,) truth: an occluder hides S30
    det: np.ndarray        # (T,) an occluder detection touches S30's zone this tick
    obs: list              # the v2 observations


def _run(dur_s: float, spans=(), seed: int = 0, operator=None, vehicle_span=None, physical: bool = True,
         converter: str = "hold", shim: dict[str, Any] | None = None) -> Run:
    """Simulate `dur_s` seconds at 2 Hz of machine S30 running (a part swinging, period 12 s). During each (a, b) in
    `spans` (seconds from start) a person stands in front of the machine at (30, 30.2); during `vehicle_span` a
    forklift (2 circles) is parked over the zone; `operator` = (x, z) of a person standing there the whole run.
    physical=True removes the machine's parts from the camera's view while an occluder is in front (a real camera).
    Returns a Run."""
    cam = M.CameraShim(SCN, seed=seed, blackouts_per_h=0.0, **(SHIM if shim is None else shim))
    conv = M.ObsV2Converter(SHARE, ZONES) if converter == "hold" else None
    s_, zm, occ, det, out = [], [], [], [], []
    for i in range(int(dur_s * M.FRAME_HZ)):
        s = i / M.FRAME_HZ
        t = T0 + s
        ph = 2 * math.pi * t / 12.0
        circ, cls = [], []
        front = any(a <= s < b for a, b in spans)
        if front:
            circ.append([30.0, 30.2, 0.3])
            cls.append(M.PERSON)
        if operator is not None:
            circ.append([operator[0], operator[1], 0.3])
            cls.append(M.PERSON)
        parked = vehicle_span is not None and vehicle_span[0] <= s < vehicle_span[1]
        if parked:
            circ += [[29.0, 30.0, 0.6], [30.2, 30.0, 0.6]]
            cls += [M.VEHICLE, M.VEHICLE]
        op_in = operator is not None and bool(O.occluded_zones([operator[0]], [operator[1]], [0.3], [ZONES[K]],
                                                                margin=1.0, pad=0.0)[0])
        hidden = physical and (front or parked or op_in)
        parts = {} if hidden else {"S30": np.array([[30 + 2.0 * math.sin(ph), 24 + math.cos(ph), 0.5]])}
        n = len(circ)
        st = M.SimState(t=t, mover_circles=np.array(circ, float).reshape(-1, 3), mover_cls=np.array(cls, np.int8),
                        mover_heading=np.zeros(n), mover_speed=np.zeros(n), parts=parts)
        o = cam.observe(st)
        if conv is not None:
            v = conv(o)
        elif converter == "tick":
            v = M.obs_v1_to_v2(o, SHARE, zones=ZONES)
        else:                                                   # "v1": camera health only (zones nobody can touch)
            v = M.obs_v1_to_v2(o, SHARE, zones=[(1e6, 1e6, 1e6, 1e6)] * NZ)
        M.check_obs_v2(v, NZ, NCAM)
        assert v.zone_motion is not None
        oc = np.isin(o.cls, M.OCC_CLASSES)
        s_.append(s)
        zm.append(float(v.zone_motion[K]))
        occ.append(front or parked or op_in)
        det.append(bool(O.occluded_zones(o.x[oc], o.z[oc], o.r[oc], [ZONES[K]], margin=1.0)[0]))
        out.append(v)
    return Run(s=np.array(s_), zm=np.array(zm), occ=np.array(occ, bool), det=np.array(det, bool), obs=out)


def _first_det(r: Run, mask: np.ndarray) -> int:
    """Index of the first tick inside `mask` with an occluder detection on the zone."""
    i = np.flatnonzero(mask & r.det)
    assert i.size, "the occluder was never detected"
    return int(i[0])


def _assert_missing_from_first_detection(r: Run, during: np.ndarray) -> int:
    """Every occluded tick from the occluder's first detection on is MISSING; earlier occluded ticks (the onset gap,
    occluder not yet detected) are the only finite ones. Returns the onset gap in ticks."""
    idx = np.flatnonzero(during)
    f = _first_det(r, during)
    zm = r.zm
    assert np.isnan(zm[f:idx[-1] + 1]).all(), f"{np.isfinite(zm[f:idx[-1] + 1]).sum()} leaks after the first detection"
    gap = f - idx[0]
    assert not r.det[idx[0]:f].any()
    assert (zm[idx[0]:f] == 0).all(), "onset ticks read 'no motion' (the occluder is not yet detected)"
    return gap


# ------------------------------------------------------------------------------------------------ the defect

def test_without_the_rule_an_occluded_running_machine_reads_no_motion():
    """The defect T-W6 fixes: with camera health only, a person in front of a running press gives zone_motion 0."""
    r = _run(60.0, spans=[(20.0, 50.0)], converter="v1")
    assert (r.zm[r.occ] == 0).all(), "occluded ticks read 'no motion' without the rule"
    assert np.isfinite(r.zm).all()


# ------------------------------------------------------------------------------------------------ stress: 30-300 s

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dur", [30.0, 120.0, 300.0])
def test_occlusion_of_30_to_300_s_is_missing_from_its_first_detection(dur, seed):
    a = 20.0
    r = _run(a + dur + 40.0, spans=[(a, a + dur)], seed=seed)
    s, zm = r.s, r.zm
    during = (s >= a) & (s < a + dur)
    assert during.sum() == int(dur * M.FRAME_HZ)
    gap = _assert_missing_from_first_detection(r, during)
    assert gap <= 4                                     # p^5 = 0.013 per occlusion under MEASURED_SHIM
    # it ends: after the hold the zone is observed again (false-positive people / gross errors may blank some ticks)
    after = s >= a + dur + M.OCC_HOLD_S + 0.5
    assert np.isfinite(zm[after]).any() and np.nanmax(zm[after]) > 0, "the running machine is seen again"


def test_hold_tail_ends_hold_s_after_the_last_detection():
    a, dur = 10.0, 60.0
    r = _run(a + dur + 30.0, spans=[(a, a + dur)], seed=1, shim=CLEAN)
    s, zm, det = r.s, r.zm, r.det
    last = int(np.flatnonzero(det)[-1])
    tail = (s > s[last]) & (s <= s[last] + M.OCC_HOLD_S)
    assert np.isnan(zm[tail]).all()
    assert np.isfinite(zm[s > s[last] + M.OCC_HOLD_S + 1e-6]).all()


def test_hold_covers_the_measured_missed_detections():
    """Without the hold (per-tick rule) MEASURED_SHIM's 32 % drops + 15 % gross errors leak 'no motion' samples
    during a 300 s occlusion; with the hold none leak after the first detection."""
    a, dur = 10.0, 300.0
    rt = _run(a + dur + 10.0, spans=[(a, a + dur)], seed=4, converter="tick")
    during = (rt.s >= a) & (rt.s < a + dur)
    f = _first_det(rt, during)
    leaks = np.isfinite(rt.zm[f:]) & during[f:]
    assert leaks.sum() > 50 and (rt.zm[f:][leaks] == 0).all(), "per-tick rule leaks zeros on missed detections"
    rh = _run(a + dur + 10.0, spans=[(a, a + dur)], seed=4)
    _assert_missing_from_first_detection(rh, during)


def test_hold_size_follows_the_measured_miss_rate():
    p = 1 - (1 - SHIM["p_drop"]) * (1 - SHIM["gross_p"])
    assert M.OCC_P_MISS == pytest.approx(p)
    k = O.hold_ticks(p, O.OCC_EPS)
    assert M.OCC_HOLD_S == k / M.FRAME_HZ == 8.0
    assert p ** (k + 1) <= O.OCC_EPS < p ** k


# ------------------------------------------------------------------------------------------------ stress: operator / vehicle

@pytest.mark.parametrize("seed", [0, 1])
def test_operator_standing_in_the_zone_keeps_it_missing_for_the_whole_run(seed):
    r = _run(600.0, operator=(30.0, 24.0), seed=seed)
    _assert_missing_from_first_detection(r, np.ones(len(r.s), bool))
    zv = np.array([o.zone_vis[K] for o in r.obs])
    f = _first_det(r, np.ones(len(r.s), bool))
    assert (zv[f:] < 1).all() and (zv[f:] > 0).all(), "zone_vis drops by the hidden floor share, the camera works"


def test_operator_just_outside_the_grown_zone():
    """Geometry: a person 4 m beyond the zone + 1 m margin never blanks it when the camera makes no gross errors /
    false positives. Under MEASURED_SHIM the person's gross errors (15 % of ticks, thrown 5-20 m) land on the zone now
    and then, and the fail-closed hold keeps each such hit for 8 s: the zone loses about a third of its samples
    (measured 0.65 observed with the hold vs 0.97 per tick). That availability cost is the price of T-W6."""
    clean = _run(600.0, operator=(30.0, 34.5), shim=CLEAN)
    assert np.isfinite(clean.zm).all()
    held = np.isfinite(_run(600.0, operator=(30.0, 34.5)).zm).mean()
    tick = np.isfinite(_run(600.0, operator=(30.0, 34.5), converter="tick").zm).mean()
    assert 0.5 <= held < tick and tick >= 0.95, (held, tick)


def test_parked_vehicle_over_the_zone_is_missing():
    r = _run(200.0, vehicle_span=(30.0, 150.0))
    s = r.s
    during = (s >= 30.0) & (s < 150.0)
    _assert_missing_from_first_detection(r, during)
    assert np.isfinite(r.zm[s >= 150.0 + M.OCC_HOLD_S + 1.0]).mean() >= 0.9


def test_other_zones_far_from_the_occluder_stay_observed():
    r = _run(120.0, spans=[(10.0, 110.0)], shim=CLEAN)
    zm = np.stack([o.zone_motion for o in r.obs])
    far = [m for m, z in enumerate(ZONES) if O.dist_to_zone([30.0], [30.2], z, 1.0)[0] > 3.0 + 1.0]
    assert far and np.isfinite(zm[:, far]).all()


# ------------------------------------------------------------------------------------------------ converter contract

def _obs(xs=(), zs=(), cls: Any = M.PERSON, rs: Any = 0.3, t: float = 10.0, machine: Any = -1, moving: Any = False):
    n = len(xs)
    cl = np.full(n, cls, np.int8) if np.isscalar(cls) else np.asarray(cls, np.int8)
    rr = np.full(n, rs, float) if np.isscalar(rs) else np.asarray(rs, float)
    mc = np.full(n, machine, np.int16) if np.isscalar(machine) else np.asarray(machine, np.int16)
    mv = np.full(n, moving, bool) if np.isscalar(moving) else np.asarray(moving, bool)
    return M.Observation(t + 0.2, t, cl, np.array(xs, float), np.array(zs, float), rr, np.full(n, np.nan), mv, mc,
                         np.ones(NCAM, bool))


def test_zones_are_required_by_the_converter():
    with pytest.raises(TypeError):
        M.obs_v1_to_v2(_obs(), SHARE)                                  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="zones"):
        M.obs_v1_to_v2(_obs(), SHARE, zones=ZONES[:-1])


@pytest.mark.parametrize("cls, hit", [(M.PERSON, True), (M.VEHICLE, True), (M.PART, False), (M.SKID, False)])
def test_only_people_and_vehicles_occlude(cls, hit):
    v = M.obs_v1_to_v2(_obs([30.0], [24.0], cls=cls, machine=K if cls in (M.PART, M.SKID) else -1), SHARE, zones=ZONES)
    assert v.zone_motion is not None
    assert bool(np.isnan(v.zone_motion[K])) == hit


def test_edge_geometry_margin_and_pad():
    # rect S30 x 26..34 grown by 1 m -> 25..35; circle r 0.3 + pad 0.5 = 0.8
    x_in = 35.0 + 0.8 - 1e-6
    x_out = 35.0 + 0.8 + 1e-3
    for x, want in ((x_in, True), (x_out, False)):
        v = M.obs_v1_to_v2(_obs([x], [24.0]), SHARE, zones=ZONES)
        assert v.zone_motion is not None
        assert bool(np.isnan(v.zone_motion[K])) == want, x


def test_converter_applies_occlusion_to_a_v2_camera_output():
    """stat_v1 (S3b) writes v2 directly; the same converter re-applies the observer-side rule on its detections."""
    v = M.obs_v1_to_v2(_obs(), SHARE, zones=ZONES)
    from dataclasses import replace
    energy = np.full(NZ, 0.7, np.float32)
    cam_out = replace(_obs([30.0], [24.0]), zone_motion=energy, zone_vis=v.zone_vis, cam_floor_ok=v.cam_floor_ok)
    conv = M.ObsV2Converter(SHARE, ZONES)
    w = conv(cam_out)
    M.check_obs_v2(w, NZ, NCAM)
    assert w.zone_motion is not None and np.isnan(w.zone_motion[K])
    others = np.isfinite(w.zone_motion)
    assert (w.zone_motion[others] == np.float32(0.7)).all()
    assert cam_out.zone_motion is energy and np.isfinite(energy).all(), "input untouched"


def test_converter_hold_and_time_order():
    conv = M.ObsV2Converter(SHARE, ZONES, hold_s=2.0)
    a = conv(_obs([30.0], [24.0], t=10.0))
    b = conv(_obs(t=11.5))                 # occluder missed: held
    c = conv(_obs(t=12.5))                 # past the hold
    for o, want in ((a, True), (b, True), (c, False)):
        assert o.zone_motion is not None and bool(np.isnan(o.zone_motion[K])) == want
    assert b.zone_vis is not None and a.zone_vis is not None and b.zone_vis[K] == a.zone_vis[K] < 1
    with pytest.raises(ValueError, match="backwards"):
        conv(_obs(t=5.0))
    conv.reset()
    conv(_obs(t=5.0))


# ------------------------------------------------------------------------------------------------ occlusion.py units

def test_hold_series_matches_the_gate():
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.3, 0.7, 400))
    hit = rng.random((400, 3)) < 0.05
    g = O.HoldGate(3, 4.0)
    want = np.stack([g.step(ti, h)[0] for ti, h in zip(t, hit)])
    np.testing.assert_array_equal(O.hold_series(t, hit, 4.0), want)
    np.testing.assert_array_equal(O.hold_series(t, hit[:, 0], 4.0), want[:, 0])
    with pytest.raises(ValueError):
        O.hold_series(t[::-1], hit, 1.0)


def test_hold_ticks():
    assert O.hold_ticks(0.0) == 0
    assert O.hold_ticks(0.5, 0.25) == 1                 # 0.5^2 = 0.25 <= eps
    assert O.hold_ticks(0.5, 0.2) == 2
    with pytest.raises(ValueError):
        O.hold_ticks(1.0)


def test_polygon_zones_grow_by_a_disc():
    tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    np.testing.assert_allclose(O.dist_to_zone([1.0, 6.0, -1.0, 3.0], [1.0, 0.0, -1.0, 3.0], tri),
                               [0.0, 2.0, math.sqrt(2), math.sqrt(2)], atol=1e-12)
    hit = O.occluded_zones([6.0, 6.0], [0.0, 0.0], [0.2, 0.2], [tri], margin=1.0, pad=0.79)
    assert not hit[0]
    assert O.occluded_zones([6.0], [0.0], [0.2], [tri], margin=1.0, pad=0.8)[0]
    cells = O.ZoneCells([tri], margin=0.0, res=0.1)
    assert cells.hidden_share([0.0], [0.0], [0.0], pad=100.0)[0] == pytest.approx(1.0)
    assert cells.hidden_share([50.0], [50.0], [0.0])[0] == 0.0


def test_rect_and_square_rect_polygon_agree_inside_the_margin_box():
    r = (26.0, 18.5, 34.0, 29.5)
    poly = [(26.0, 18.5), (34.0, 18.5), (34.0, 29.5), (26.0, 29.5)]
    px = np.linspace(20, 40, 41)
    pz = np.full_like(px, 24.0)
    np.testing.assert_allclose(O.dist_to_zone(px, pz, r, 1.0), O.dist_to_zone(px, pz, poly, 1.0), atol=1e-12)


ROI = [(800.0, 200.0), (1000.0, 200.0), (1000.0, 420.0), (800.0, 420.0)]


@pytest.mark.parametrize("box, want", [
    ((850, 300, 900, 600), True),                  # person box overlaps the ram ROI
    ((1010, 300, 1100, 600), True),                # 10 px right of it: the 15 % pad (13.5 px) reaches it
    ((1020, 300, 1100, 600), False),               # 20 px right: no
    ((700, 100, 1100, 500), True),                 # box contains the whole ROI
    ((880, 280, 900, 300), True),                  # box inside the ROI
    ((500, 600, 600, 900), False),
    ((np.nan, 0, 1, 1), False),
])
def test_box_vs_ram_roi(box, want):
    assert O.box_hits_roi(box, ROI) == want


def test_box_vs_concave_roi_edges_cross():
    L = [(0, 0), (100, 0), (100, 20), (20, 20), (20, 100), (0, 100)]          # L-shaped ROI
    assert not O.box_hits_roi((40, 40, 90, 90), L, pad_frac=0.0)               # in the notch
    assert O.box_hits_roi((10, -10, 15, 200), L, pad_frac=0.0)                 # crosses edges, no vertex inside
    assert O.boxes_hit_roi(np.zeros((0, 4)), L) is False


@pytest.mark.parametrize("dur", [30.0, 300.0])
def test_roi_missing_series_holds_through_detector_gaps(dur):
    """Real-clip ingest (S3a): a person at the press for `dur` s at 12 fps, detected on ~60 % of frames with gaps up
    to 2 s; MISSING covers the whole stay (hold 2.5 s) and ends after it."""
    fps, a = 12.0, 5.0
    t = np.arange(int((a + dur + 10) * fps)) / fps
    rng = np.random.default_rng(7)
    seen = rng.random(len(t)) < 0.6
    for g0 in rng.choice(len(t), 5, replace=False):
        seen[g0:g0 + int(2 * fps)] = False
    present = (t >= a) & (t < a + dur)
    seen[present & (np.abs(t - a) < 0.1)] = True                    # detected on arrival
    boxes = [[(850, 300, 900, 600)] if (p and s) else [] for p, s in zip(present, seen)]
    miss = O.roi_missing_series(t, boxes, ROI, hold_s=2.5)
    first = int(np.flatnonzero(present & seen)[0])
    last = int(np.flatnonzero(present & seen)[-1])
    assert miss[first:last + 1].all()
    assert not miss[t > t[last] + 2.5 + 1e-6].any()
    assert not miss[t < a].any()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roi_occlusion_streaming_equals_the_ingest_series(seed):
    """ONE image ROI rule (user decision Sun ~08:25 "Same ROI rule in twin"): the twin camera's tick-by-tick
    `RoiOcclusion` gives exactly the real-clip ingest's `roi_missing_series` on the same boxes and times."""
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(600) / 2.0, 6)
    boxes = []
    for _ in t:
        b = []
        for _ in range(int(rng.integers(0, 3))):
            u, v = rng.uniform(600, 1100), rng.uniform(100, 700)
            b.append((u, v, u + rng.uniform(10, 120), v + rng.uniform(20, 250)))
        boxes.append(b)
    hold = [0.0, 2.5, 7.5][seed]
    want = O.roi_missing_series(t, boxes, ROI, hold_s=hold)
    occ = O.RoiOcclusion(ROI, hold)
    got = np.array([occ.step_boxes(tt, b) for tt, b in zip(t, boxes)])
    assert np.array_equal(got, want)
    occ.reset()
    got2 = np.array([occ.step(tt, any(occ.hits(bb) for bb in b)) for tt, b in zip(t, boxes)])
    assert np.array_equal(got2, want)
    assert want.any() and not want.all()
