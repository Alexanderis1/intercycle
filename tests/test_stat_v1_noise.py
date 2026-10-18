"""twin/camera.py stat_v1 (lane S3b; PLAN_TWIN §2 twin/camera.py row; WEAKNESSES M-W2, T-W6).

The plan's acceptance test: the median position error of stat_v1 is >= MEASURED_SHIM's. Plus the model's stated
properties: sigma >= 0.285 m per axis, one CAL draw per world, common random numbers, classes (robot = other, no
vehicle class on camera 9), the moving flag from noisy positions, T-W6 ROI MISSING, and the fitted hold.
"""
from __future__ import annotations

import json
import math
from typing import cast

import numpy as np
import pytest

from intercycle.observer import mapping as OM
from twin import camera as CAM
from twin.camera_v0 import CameraV0
from twin.schemas import load
from twin.schemas.runs import RealisedTick
from twin.schemas.site import Site, point_in_polygon

SITE = load(Site, CAM.REPO / "twin" / "site.yaml")
WORLD = 7


def shim_median_m() -> float:
    """MEASURED_SHIM's median radial error: 85 % Rayleigh(0.285) + 15 % U(5, 20) m (all > the median)."""
    s, g = float(cast(float, OM.MEASURED_SHIM["sigma_m"])), float(cast(float, OM.MEASURED_SHIM["gross_p"]))
    return s * math.sqrt(-2.0 * math.log(1.0 - 0.5 / (1.0 - g)))


def visible_points(cam: CAM.CameraStatV1, zone: str, n: int, seed: int = 0) -> list[tuple[float, float]]:
    z = cam.zones[zone]
    poly = z.floor_polygon.value
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    g = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x, y = float(g.uniform(min(xs), max(xs))), float(g.uniform(min(zs), max(zs)))
        if point_in_polygon(x, y, poly) and cam._framing_of(zone, x, y) is not None:
            out.append((x, y))
    return out


def tick(t: float, movers: list[dict], presses: list[dict] | None = None) -> RealisedTick:
    if presses is None:
        presses = [dict(id=p.id, state="idle", ram=0.0, stroke=False, freeze=False) for p in SITE.presses]
    return RealisedTick.model_validate(dict(kind="tick", t=t, presses=presses, movers=movers))


def person(i: int, x: float, z: float, zone: str, speed: float = 0.0, cls: str = "person") -> dict:
    return dict(id=f"{cls}{i}", cls=cls, zone=zone, x=x, z=z, heading=0.0, speed=speed, r=0.3)


def run_static(cam, pts, zone, n_ticks: int, t0: float = 100.0, cls: str = "person"):
    for k in range(n_ticks):
        cam.observe(tick(t0 + k / 2.0, [person(i, x, z, zone, cls=cls) for i, (x, z) in enumerate(pts)]))


# ------------------------------------------------------------------------------------------------ the plan's test

def test_median_position_error_at_least_measured_shim():
    errs = []
    for zone in ("z9", "z14"):
        cam = CAM.CameraStatV1(SITE, WORLD, 1)
        cam.err_log = []
        pts = visible_points(cam, zone, 12, seed=1)
        run_static(cam, pts, zone, 200)
        errs += [math.hypot(dx, dz) for _, dx, dz in cam.err_log]
    assert len(errs) > 1000
    med = float(np.median(errs))
    assert med >= shim_median_m(), (med, shim_median_m())
    # the shim itself on the same truth (camera_v0, no false positives): its median matches the formula
    v0 = CameraV0(SITE, WORLD, 1, shim=dict(OM.MEASURED_SHIM, fp_ratio=0.0))
    e0 = []
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    pts = visible_points(cam, "z9", 1, seed=2)
    for k in range(4000):
        d = v0.observe(tick(100.0 + k / 2.0, [person(0, *pts[0], "z9")]))
        if len(d):
            e0.append(math.hypot(d.x[0] - pts[0][0], d.z[0] - pts[0][1]))
    assert abs(float(np.median(e0)) - shim_median_m()) < 0.03
    assert med >= float(np.median(e0))


def test_per_axis_sigma_at_least_shim_without_gross():
    """Non-gross errors, demeaned per mover (the world's calibration bias is a constant per point): the per-axis
    sigma of the fluctuating part is >= 0.285 m up to sampling error."""
    cam = CAM.CameraStatV1(SITE, WORLD, 1, h_mode="fixed_scale")
    cam.err_log = []
    pts = visible_points(cam, "z9", 8, seed=3)
    run_static(cam, pts, "z9", 1500)
    by: dict[str, list[tuple[float, float]]] = {}
    for mid, dx, dz in cam.err_log:
        if math.hypot(dx, dz) < 4.0:                       # gross offsets are >= 5 m
            by.setdefault(mid, []).append((dx, dz))
    for mid, v in by.items():
        a = np.asarray(v)
        assert len(a) > 300
        sd = float(np.sqrt(((a - a.mean(0)) ** 2).mean(0).mean()))
        assert sd >= 0.285 * 0.85, (mid, sd)               # AR(1) samples: allow the sampling error of ~300 ticks


# ------------------------------------------------------------------------------------------------ geometry / CAL

def test_projection_round_trip_and_one_draw_per_world():
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    for fid, g in cam.true.items():
        nom = cam.nominal[fid]
        for geom in (g, nom):
            raw = np.array([[900.0, 800.0], [300.0, 1000.0], [1500.0, 700.0]])
            fl = geom.floor_of(raw)
            back, ok = geom.project(np.column_stack([fl, np.zeros(len(fl))]))
            assert ok.all() and np.allclose(back, raw, atol=1e-3)
    # same world -> same draw on every day; another world -> (generally) another draw; both framings of camera 9
    cam2 = CAM.CameraStatV1(SITE, WORLD, 5)
    assert cam2.draw_index == cam.draw_index
    idx = {CAM.CameraStatV1(SITE, w, 1).draw_index[9] for w in range(12)}
    assert len(idx) > 6
    # the camera rebuilt from (K, k1, H) sits at the draw's height (one mount per camera); nominal = the 4 m prior
    hd = json.loads(CAM.HDRAWS_PATH.read_text(encoding="utf-8"))
    for fid in ("c9_f0", "c9_f1", "c14_f0"):
        cam_id = 9 if fid.startswith("c9") else 14
        want = hd["framings"][fid]["draws"][cam.draw_index[cam_id]]["h_m"]
        assert abs(cam.true[fid].height - want) < 1e-6 * want
        assert abs(cam.nominal[fid].height - 4.0) < 1e-6
    assert cam.cal_source == "cal_v1 draws"


def test_stub_site_has_no_calibration_error_and_sealed_refuses_it(monkeypatch):
    v0 = load(Site, CAM.REPO / "twin" / "site_v0.yaml")
    cam = CAM.CameraStatV1(v0, WORLD, 1)
    assert cam.cal_source.startswith("site framings") and not cam.draw_index
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(ValueError):
        CAM.CameraStatV1(v0, WORLD, 1)


def test_calibration_bias_is_constant_per_world():
    """The world's true camera differs from the nominal one: a standing person's mean error is a fixed offset."""
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    cam.err_log = []
    pts = visible_points(cam, "z14", 1, seed=4)
    run_static(cam, pts, "z14", 800)
    a = np.asarray([(dx, dz) for _, dx, dz in cam.err_log if math.hypot(dx, dz) < 4.0])
    raw = cam._framing_of("z14", *pts[0])
    assert raw is not None
    want = cam.nominal[raw[0]].floor_of(raw[1][None, :] + cam.bias_px)[0] - np.asarray(pts[0])
    assert np.allclose(a.mean(0), want, atol=0.12)


# ------------------------------------------------------------------------------------------------ CRN, classes

def test_common_random_numbers_robot_far_away():
    cam_a = CAM.CameraStatV1(SITE, WORLD, 2)
    cam_b = CAM.CameraStatV1(SITE, WORLD, 2)
    pts = visible_points(cam_a, "z9", 3, seed=5)
    far = visible_points(cam_a, "z14", 1, seed=6)[0]
    for k in range(60):
        t = 50.0 + k / 2.0
        ppl = [person(i, x, z, "z9", speed=0.0) for i, (x, z) in enumerate(pts)]
        da = cam_a.observe(tick(t, ppl))
        db = cam_b.observe(tick(t, ppl + [dict(id="robot", cls="robot", zone="z14", x=far[0], z=far[1],
                                                 heading=0.0, speed=0.5, r=0.5)]))
        sel = db.zone == "z9"
        assert np.array_equal(da.x, db.x[sel]) and np.array_equal(da.z, db.z[sel])


def test_robot_is_other_and_forklift_classes():
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    p14 = visible_points(cam, "z14", 3, seed=7)
    p9 = visible_points(cam, "z9", 3, seed=8)
    seen = {"z9": set(), "z14": set(), "robot": set()}
    for k in range(200):
        t = 10.0 + k / 2.0
        mv = [person(i, x, z, "z14", cls="forklift") for i, (x, z) in enumerate(p14)]
        mv += [person(10 + i, x, z, "z9", cls="forklift") for i, (x, z) in enumerate(p9)]
        mv += [dict(id="robot", cls="robot", zone="z9", x=p9[0][0] + 1.0, z=p9[0][1], heading=0.0, speed=0.0, r=0.5)]
        d = cam.observe(tick(t, mv))
        for c, z in zip(d.cls, d.zone):
            seen[str(z)].add(str(c))
    assert "vehicle" in seen["z14"]
    assert "vehicle" not in seen["z9"]                       # no vehicle class on camera 9
    assert "other" in seen["z9"]                             # the robot
    assert cam.stats["forklift_as_person_c9"] > 0


def test_moving_flag_from_noisy_positions():
    cam = CAM.CameraStatV1(SITE, WORLD, 3)
    pts = visible_points(cam, "z9", 6, seed=9)
    mv_static = []
    for k in range(400):
        d = cam.observe(tick(200.0 + k / 2.0, [person(i, x, z, "z9") for i, (x, z) in enumerate(pts)]))
        mv_static += list(d.moving[d.cls == "person"])
    share = float(np.mean(mv_static))
    assert 0.02 < share < 0.95                              # standing people are flagged moving by noise sometimes
    cam = CAM.CameraStatV1(SITE, WORLD, 3)
    x0, z0 = pts[0]
    mv_walk = []
    for k in range(40):
        z = z0 + 0.6 * k / 2.0 * (1 if z0 < -5 else -1)     # 1.2 m/s ... stays on the floor for 20 s
        x = x0
        if not point_in_polygon(x, z, cam.zones["z9"].floor_polygon.value):
            break
        d = cam.observe(tick(300.0 + k / 2.0, [person(0, x, z, "z9", speed=1.2)]))
        mv_walk += list(d.moving[:1])
    assert len(mv_walk) > 5 and float(np.mean(mv_walk)) > share


# ------------------------------------------------------------------------------------------------ T-W6 / hold

def test_roi_missing_when_detected_occluder_covers_the_ram():
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    fid, (r0, s0, r1, s1) = cam.roi["P14b"]
    nom = cam.true[fid]
    # a person standing on the floor below the ROI in the image, in front of the press (first such pixel row)
    foot = None
    for dv in range(40, 600, 20):
        f = nom.floor_of(np.array([[(r0 + r1) / 2, s1 + dv]]))[0]
        if (point_in_polygon(float(f[0]), float(f[1]), cam.zones["z14"].floor_polygon.value)
                and cam._depth(fid, f) < cam._depth(fid, cam.press_centre["P14b"])):
            foot = f
            break
    assert foot is not None
    pr = [dict(id=p.id, state="running", ram=0.5, stroke=False, freeze=False) for p in SITE.presses]
    vals, nan = [], 0
    for k in range(200):
        d = cam.observe(tick(10.0 + k / 2.0, [person(0, float(foot[0]), float(foot[1]), "z14")], pr))
        v = d.zone_motion["P14b"]
        nan += math.isnan(v)
        vals.append(v)
    assert cam.stats["roi_missing"] + cam.stats["roi_blocked_unseen"] == 200   # the person covers the ROI
    assert nan == cam.stats["roi_missing"] > 100                                # detected -> MISSING (T-W6)
    finite = np.asarray([v for v in vals if np.isfinite(v)])                     # missed: the blocked ROI reads idle
    assert len(finite) == cam.stats["roi_blocked_unseen"]
    # unblocked running press: finite energies from the moving pool, mostly above MOTION_THR
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    e = [cam.observe(tick(10.0 + k / 2.0, [], pr)).zone_motion["P9a"] for k in range(300)]
    assert np.isfinite(e).all() and float(np.mean(np.asarray(e) > CAM.MOTION_THR)) > 0.75
    idle = [cam.observe(tick(500.0 + k / 2.0, [])).zone_motion["P9a"] for k in range(300)]
    assert float(np.mean(np.asarray(idle) > CAM.MOTION_THR)) < 0.3


def test_hold_sized_from_fitted_bursts():
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    assert cam.hold_s >= OM.OCC_HOLD_S          # the occluded regime misses more often than the shim's iid misses
    for cn in cam.noise.values():
        k = CAM.hold_ticks_markov(cn, 1.0 - cn.occ_ratio)
        assert k / cam.hz <= cam.hold_s
        # simulated leak check of the chain alone (no occluded regime): runs of >= k + 1 misses are rare
        g = np.random.default_rng(0)
        m, run, worst = False, 0, 0
        for _ in range(20000):
            m = g.random() < (cn.p_stay_miss if m else cn.p_enter_miss)
            run = run + 1 if m else 0
            worst = max(worst, run)
        assert worst <= k + 1


def test_plant_uses_stat_v1_and_its_hold():
    from twin import plant
    cam = plant.make_camera(SITE, WORLD, 1)
    assert isinstance(cam, CAM.CameraStatV1)
    ad = plant.ObsAdapter(SITE, hold_s=cam.hold_s)
    assert ad.conv.hold_s == cam.hold_s
