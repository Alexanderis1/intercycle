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
from twin.cam_geom import PRESS_H_M
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


def test_floor_covariance_min_eigen_at_least_shim():
    """The model's floor covariance (pixel noise through the nominal map + the floor top-up) has its SMALLEST
    eigenvalue >= 0.285^2 at every visible point of both zones: sigma >= 0.285 m in every direction, so per axis
    (x and z) too. Analytic, 2000 points per zone."""
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    for zone in ("z9", "z14"):
        for x, z in visible_points(cam, zone, 2000, seed=11):
            fo = cam._framing_of(zone, x, z)
            assert fo is not None
            fid, raw = fo
            S = cam._floor_cov(fid, raw)
            T = CAM.floor_topup(S, CAM.SIGMA_MIN_M)
            tot = S + T @ T.T
            lmin = float(np.linalg.eigvalsh(tot)[0])
            assert lmin >= CAM.SIGMA_MIN_M ** 2 * (1 - 1e-9), (zone, x, z, math.sqrt(max(lmin, 0.0)))
            assert min(tot[0, 0], tot[1, 1]) >= CAM.SIGMA_MIN_M ** 2 * (1 - 1e-9)


def test_per_axis_and_minor_axis_sigma_at_least_shim_without_gross():
    """Empirical: non-gross errors, demeaned per mover (the world's calibration bias is a constant per point). The
    sd along x, along z and along the minor eigen-direction of each point's error covariance is >= 0.285 m up to
    sampling error (AR(1) samples, 4000 ticks per point: about 4 sd of the estimate below 0.285 m allowed)."""
    for zone in ("z9", "z14"):
        cam = CAM.CameraStatV1(SITE, WORLD, 1, h_mode="fixed_scale")
        cam.err_log = []
        pts = visible_points(cam, zone, 6, seed=3)
        run_static(cam, pts, zone, 4000)
        by: dict[str, list[tuple[float, float]]] = {}
        for mid, dx, dz in cam.err_log:
            if math.hypot(dx, dz) < 4.0:                   # gross offsets are >= 5 m
                by.setdefault(mid, []).append((dx, dz))
        assert len(by) == len(pts)
        for mid, v in by.items():
            a = np.asarray(v)
            assert len(a) > 600
            c = np.cov((a - a.mean(0)).T)
            sd_x, sd_z = math.sqrt(c[0, 0]), math.sqrt(c[1, 1])
            sd_minor = math.sqrt(float(np.linalg.eigvalsh(c)[0]))
            for name, sd in (("x", sd_x), ("z", sd_z), ("minor", sd_minor)):
                assert sd >= 0.285 * 0.88, (zone, mid, name, sd)


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


def _hits(b, roi, pad: float = 0.15) -> bool:
    x0, y0, x1, y1 = b
    r0, s0, r1, s1 = roi
    pu, pv = pad * (x1 - x0), pad * (y1 - y0)
    return x0 - pu <= r1 and x1 + pu >= r0 and y0 - pv <= s1 and y1 + pv >= s0


def _overlap_depth(b, roi, pad: float = 0.15) -> float:
    """How deep (px) the padded box overlaps the ROI (> 0: overlapping; the smallest of the four overlaps)."""
    x0, y0, x1, y1 = b
    r0, s0, r1, s1 = roi
    pu, pv = pad * (x1 - x0), pad * (y1 - y0)
    return min(x1 + pu - r0, r1 - (x0 - pu), y1 + pv - s0, s1 - (y0 - pv))


def _at_roi_occluder(gn, roi, centre, cls: str = "person"):
    """(prism, overlap depth px) of a standing mover in front of the press whose padded box (nominal camera)
    overlaps the drawn ROI deepest: feet on a grid along the line camera nadir -> press centre (+-3 m sideways)."""
    from twin.cam_geom import circle_prism, roi_box
    ax = np.asarray(centre, float) - gn.nadir
    perp = np.array([-ax[1], ax[0]]) / np.linalg.norm(ax)
    best = None
    for f in np.linspace(0.3, 0.98, 120):
        for dl in np.linspace(-3.0, 3.0, 41):
            q = gn.nadir + f * ax + dl * perp
            prism = circle_prism(float(q[0]), float(q[1]), 0.3 if cls == "person" else 1.0, CAM.HEIGHT_M[cls])
            b = roi_box(gn, prism)
            if b is None:
                continue
            m = _overlap_depth(b, roi)
            if best is None or m > best[1]:
                best = (prism, m)
    assert best is not None and best[1] > 0
    return best


def test_ram_roi_follows_the_press_through_every_cal_draw():
    """Audit fix2 (major): the ram ROI is anchored on the press (`cam_geom.press_anchor`) and projected through the
    world's TRUE camera, like the press prism and the movers. Through the nominal camera the anchor gives the drawn ROI
    back; a fixed at-ROI occluder (a person in front of the press whose padded box overlaps the drawn ROI by >= 20 px
    in the nominal image; a forklift for P14d, whose ROI is 2-3 m up) hits the anchored ROI in every one of the 200
    CAL v1 joint draws, for every press. With the ROI left at its drawn
    pixels (the fix1 camera), the same occluder misses P9a's ROI in many draws (the press moved, the ROI did not)."""
    from twin.cam_geom import CamGeom, press_anchor, roi_box
    hd = json.loads(CAM.HDRAWS_PATH.read_text(encoding="utf-8"))
    fr = {f.id: f for f in SITE.framings}
    fixed_hits = {}
    for p in SITE.presses:
        if not p.ram_roi.value:
            continue
        r = p.ram_roi.value[0]
        fid, poly = r.framing, np.asarray(r.polygon_px, float)
        K = np.asarray(fr[fid].K.value, float)
        gn = CamGeom(fid, K, float(fr[fid].k1.value), np.asarray(fr[fid].H_img_to_floor.value, float))
        centre = np.asarray(p.footprint.value, float).mean(0)
        anc = press_anchor(gn, poly, p.footprint.value, PRESS_H_M)
        # the anchor lies on the press's visible surface: inside the footprint's depth range, 0-3 m high
        assert (anc[:, 2] > -1.0).all() and (anc[:, 2] < PRESS_H_M + 1.0).all(), (p.id, anc)
        back, ok = gn.project(anc)
        assert ok.all() and np.allclose(back, poly, atol=1e-6), p.id
        drawn = roi_box(gn, anc)
        prism, depth = _at_roi_occluder(gn, drawn, centre)
        if depth < 20.0:                                   # P14d's ROI is 2-3 m up: only a taller mover covers it
            prism, depth = _at_roi_occluder(gn, drawn, centre, cls="forklift")
        assert depth >= 20.0, (p.id, depth)
        n_anch, n_fixed = 0, 0
        for rec in hd["framings"][fid]["draws"]:
            gt = CamGeom.from_rec(fid, K, rec)
            tb, pb = roi_box(gt, anc), roi_box(gt, prism)
            assert pb is not None
            n_anch += tb is not None and _hits(pb, tb)
            n_fixed += _hits(pb, drawn)
        assert n_anch == len(hd["framings"][fid]["draws"]), (p.id, depth, n_anch)
        fixed_hits[p.id] = n_fixed
    assert fixed_hits["P9a"] < 150, fixed_hits          # the fix1 camera's fixed ROI lost the press in many draws
    assert fixed_hits["P9a"] < 150, fixed_hits          # the fix1 camera's fixed ROI lost the press in many draws


def test_library_replay_roi_hits_stable_across_cal_draws():
    """Audit fix2 (major): the DAYGEN v1 path library replayed through the camera geometry (scripts/
    stat_v1_roi_occluders.py `twin_replay`, every 5th CAL v1 draw) hits the P9a and P14c ROIs in every draw, near the
    nominal-camera share (P9a: within 10 %; P14c: the parallax of the near-camera floor, q5 >= half of it). On the fix1
    camera (ROI fixed at its drawn pixels) P9a had 0 hits in 123 of the 200 draws."""
    import importlib.util
    from intercycle import esk
    try:
        lib = esk.data_root() / "dyn" / "path_library_v1.npz"
    except Exception as e:                                   # no data root configured
        pytest.skip(f"no data root: {e}")
    if not lib.exists():
        pytest.skip("DAYGEN v1 path library not present")
    spec = importlib.util.spec_from_file_location("stat_v1_roi_occluders",
                                                  CAM.REPO / "scripts" / "stat_v1_roi_occluders.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = mod.twin_replay(SITE, presses=["P9a", "P14c"], draw_step=5)["presses"]
    for pid, rel in (("P9a", 0.9), ("P14c", 0.5)):
        r = rep[pid]
        q5, q50, _q95 = r["twin_hit_share_unheld_draws_q5_50_95"]
        nom = r["twin_hit_share_unheld_nominal"]
        assert r["draws_with_zero_hits"] == 0, (pid, r)
        assert nom > 0.1 and q5 >= rel * nom and abs(q50 - nom) <= 0.1 * nom + 0.02, (pid, r)
    assert rep["P9a"]["fixed_roi_draws_with_zero_hits"] > 10           # the fix1 error, for contrast


def test_camera_roi_is_the_anchored_roi_of_the_world_draw():
    """CameraStatV1.roi = the anchored ROI through the world's true camera (one CAL draw per world), roi_nominal =
    the drawn box; roi_vis is the true box's share inside the image; worlds with different draws move the ROI."""
    from twin.cam_geom import box_share_in_image, press_anchor, roi_box
    boxes = {}
    for w in range(6):
        cam = CAM.CameraStatV1(SITE, w, 1)
        for pid, (fid, tb) in cam.roi.items():
            p = cam.presses[pid]
            poly = np.asarray(p.ram_roi.value[0].polygon_px, float)
            want = roi_box(cam.true[fid], press_anchor(cam.nominal[fid], poly, p.footprint.value, PRESS_H_M))
            assert want is not None and np.allclose(tb, want)
            assert cam.roi_nominal[pid][1] == pytest.approx((poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(),
                                                             poly[:, 1].max()))
            assert cam.roi_vis[pid] == {cam.cam_of_framing[fid]: box_share_in_image(tb)}
            boxes.setdefault(pid, set()).add(tuple(round(v, 3) for v in tb))
    assert len(boxes["P9a"]) > 1


def test_at_roi_occluder_makes_p9a_missing_in_every_world():
    """End to end through the camera: a person standing in front of P9a over its ROI (zone-9 floor, detected) makes
    the ROI MISSING (held) on most ticks in every world, whatever its CAL draw."""
    from twin.cam_geom import CamGeom
    fr = {f.id: f for f in SITE.framings}
    p = next(pp for pp in SITE.presses if pp.id == "P9a")
    fid = p.ram_roi.value[0].framing
    gn = CamGeom(fid, np.asarray(fr[fid].K.value, float), float(fr[fid].k1.value),
                 np.asarray(fr[fid].H_img_to_floor.value, float))
    poly = np.asarray(p.ram_roi.value[0].polygon_px, float)
    roi = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
    prism, _ = _at_roi_occluder(gn, roi, np.asarray(p.footprint.value, float).mean(0))
    x, z = prism[:8, 0].mean(), prism[:8, 1].mean()
    pr = [dict(id=q.id, state="running", ram=0.5, stroke=False, freeze=False) for q in SITE.presses]
    zone = next((zz.id for zz in SITE.zones if point_in_polygon(x, z, zz.floor_polygon.value)), None)
    for w in range(8):
        cam = CAM.CameraStatV1(SITE, w, 1)
        vals = [cam.observe(tick(10.0 + k / 2.0, [dict(id="op", cls="person", zone=zone, x=float(x), z=float(z),
                                                       heading=0.0, speed=0.0, r=0.3)], pr)).zone_motion["P9a"]
                for k in range(200)]
        assert float(np.isnan(vals).mean()) > 0.9, (w, cam.draw_index, float(np.isnan(vals).mean()))


def test_stub_site_has_no_calibration_error_and_sealed_refuses_it(monkeypatch):
    v0 = load(Site, CAM.REPO / "twin" / "site_v0.yaml")
    cam = CAM.CameraStatV1(v0, WORLD, 1)
    assert cam.cal_source.startswith("site framings") and not cam.draw_index
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(ValueError):
        CAM.CameraStatV1(v0, WORLD, 1)


def test_partial_or_missing_cal_draws_are_refused_on_a_calibrated_site():
    """No silent true = nominal fallback outside sealed runs: a site with cal_sources needs CAL v1 draws for every
    framing, and a partial match (e.g. a renamed framing id) is refused on any site."""
    hd = json.loads(CAM.HDRAWS_PATH.read_text(encoding="utf-8"))
    part = dict(hd, framings={k: v for k, v in hd["framings"].items() if k != "c9_f1"})
    with pytest.raises(ValueError, match="framings have CAL v1 draws"):
        CAM.CameraStatV1(SITE, WORLD, 1, hdraws=part)
    assert SITE.cal_sources
    with pytest.raises(ValueError, match="cal_sources"):
        CAM.CameraStatV1(SITE, WORLD, 1, hdraws=dict(hd, framings={}))
    prov = CAM.CameraStatV1(SITE, WORLD, 1).provenance()
    assert prov["cal_source"] == "cal_v1 draws" and set(prov["draw_index"]) == {"9", "14"}
    assert prov["hold_s"] > 0 and len(prov["params_sha256"]) == 64


def test_calibration_bias_is_constant_per_world():
    """The world's true camera differs from the nominal one: a standing person's mean error is a fixed offset."""
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    cam.err_log = []
    pts = visible_points(cam, "z14", 1, seed=4)
    run_static(cam, pts, "z14", 800)
    a = np.asarray([(dx, dz) for _, dx, dz in cam.err_log if math.hypot(dx, dz) < 4.0])
    raw = cam._framing_of("z14", *pts[0])
    assert raw is not None
    want = cam.nominal[raw[0]].floor_of(raw[1][None, :] + cam.bias_px[14])[0] - np.asarray(pts[0])
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
    st = cam.stats
    assert st["roi_missing"] + st["roi_held"] + st["roi_blocked_unseen"] == 200  # the person covers the ROI
    assert st["roi_missing"] > 100                                              # detected -> MISSING (T-W6)
    assert nan == st["roi_missing"] + st["roi_held"]                            # missed after a detection: held
    finite = np.asarray([v for v in vals if np.isfinite(v)])                     # missed before any detection only
    assert len(finite) == st["roi_blocked_unseen"]
    first_nan = next(i for i, v in enumerate(vals) if math.isnan(v))
    assert np.isfinite(vals[:first_nan]).all() and np.isnan(vals[first_nan:]).all()
    # unblocked running press: finite energies from the moving pool, mostly above MOTION_THR
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    e = [cam.observe(tick(10.0 + k / 2.0, [], pr)).zone_motion["P9a"] for k in range(300)]
    assert np.isfinite(e).all() and float(np.mean(np.asarray(e) > CAM.MOTION_THR)) > 0.75
    idle = [cam.observe(tick(500.0 + k / 2.0, [])).zone_motion["P9a"] for k in range(300)]
    assert float(np.mean(np.asarray(idle) > CAM.MOTION_THR)) < 0.3


def _roi_occluder_far_from_zone(cam, pid: str, min_gap: float):
    """A floor point whose standing person covers press `pid`'s ram ROI in the image (in front of the press) while
    its floor position is more than `min_gap` from the press's danger rect (+ dilation): the observer's floor rule
    (rect + 1 m margin, circle r + 0.5 m pad) never arms on it."""
    from twin import plant
    from intercycle.observer.occlusion import ROI_PAD_FRAC
    p = next(pp for pp in SITE.presses if pp.id == pid)
    fid, (r0, s0, r1, s1) = cam.roi[pid]
    rect = plant._bbox(p.danger_polygon.value, p.dilation_m.value)
    poly = cam.zones[p.zone].floor_polygon.value
    xs, zs = [q[0] for q in poly], [q[1] for q in poly]
    g = np.random.default_rng(0)
    for _ in range(50000):
        x, z = float(g.uniform(min(xs), max(xs))), float(g.uniform(min(zs), max(zs)))
        if not point_in_polygon(x, z, poly):
            continue
        fo = cam._framing_of(p.zone, x, z)
        if fo is None or fo[0] != fid or cam._depth(fid, (x, z)) >= cam._depth(fid, cam.press_centre[pid]):
            continue
        b = cam._box(fid, CAM._circle_prism(x, z, 0.3, CAM.HEIGHT_M["person"]))
        if b is None:
            continue
        pu, pv = ROI_PAD_FRAC * (b[2] - b[0]), ROI_PAD_FRAC * (b[3] - b[1])
        gap = math.hypot(max(rect[0] - x, 0.0, x - rect[2]), max(rect[1] - z, 0.0, z - rect[3]))
        if b[0] - pu <= r1 and b[2] + pu >= r0 and b[1] - pv <= s1 and b[3] + pv >= s0 and gap > min_gap:
            return x, z, gap
    raise AssertionError(f"no such floor point for {pid}")


def test_image_space_hold_covers_occluder_outside_the_floor_zone():
    """T-W6 inside the camera: a person standing in front of running P14a covers its ROI in the image, but stands
    > 1.8 m (margin 1 m + pad 0.5 m + r 0.3 m) from the rect, so the observer's floor hold never arms. After the first
    tick the person is detected, the camera keeps the ROI MISSING for hold_s after every detection: no finite sample
    (no 'no motion' leak) until the person leaves. On bf91cdd's camera (and params) the same scene leaks 47-79 finite
    samples in 1200 ticks after the first detection (worlds 3 / 7 / 17), 40-68 of them <= MOTION_THR.
    Since fix2 the ROI follows the press through the world's true camera, so such a far occluder exists only in some
    worlds (world 7 has none for P14a): the test uses the first three worlds that have one."""
    from intercycle.observer.occlusion import occluded_zones
    from twin import plant
    pr = [dict(id=p.id, state="running", ram=0.0, stroke=False, freeze=False) for p in SITE.presses]
    found: list[tuple[CAM.CameraStatV1, float, float, float]] = []
    for world in range(40):
        cam = CAM.CameraStatV1(SITE, world, 1)
        try:
            found.append((cam, *_roi_occluder_far_from_zone(cam, "P14a", 1.8 + 0.05)))
        except AssertionError:
            continue
        if len(found) == 3:
            break
    assert len(found) == 3
    for cam, x, z, gap in found:
        world = cam.world_seed
        p = next(pp for pp in SITE.presses if pp.id == "P14a")
        rect = plant._bbox(p.danger_polygon.value, p.dilation_m.value)
        assert not occluded_zones(np.array([x]), np.array([z]), np.array([0.3]), [rect], margin=1.0, pad=0.5).any()
        first, leaks = None, 0
        for k in range(1200):
            d = cam.observe(tick(10.0 + k / 2.0, [person(0, x, z, "z14")], pr))
            v = d.zone_motion["P14a"]
            if first is None and math.isnan(v):
                first = k
            elif first is not None and np.isfinite(v):
                leaks += 1
        assert first is not None and first < 20, (world, first)
        assert leaks == 0, (world, leaks, gap)
        assert cam.stats["roi_held"] > 0                     # missed ticks were bridged by the image-space hold
    # the hold ends: once the occluder leaves, the ROI reads energies again after hold_s
    cam = CAM.CameraStatV1(SITE, found[0][0].world_seed, 1)
    x, z, _ = _roi_occluder_far_from_zone(cam, "P14a", 1.85)
    for k in range(20):
        cam.observe(tick(10.0 + k / 2.0, [person(0, x, z, "z14")], pr))
    last = max(k for k in [cam.roi_last_hit["P14a"]])
    h14 = cam.hold_ticks_by_camera[14]                    # P14a is seen by camera 14: camera 14's own hold
    assert h14 < cam.hold_ticks
    vals = [cam.observe(tick(20.0 + k / 2.0, [], pr)).zone_motion["P14a"] for k in range(cam.hold_ticks + 10)]
    k_free = 40 + np.arange(len(vals))
    assert all(math.isnan(v) for v, kk in zip(vals, k_free) if kk - last <= h14)
    assert all(np.isfinite(v) for v, kk in zip(vals, k_free) if kk - last > h14)


def test_off_floor_person_at_the_press_occludes_the_zone9_roi():
    """Where the real zone-9 ROI occluders stand (twin/stat_v1_roi_occluders.json: TRAIN foot points of the boxes over
    the P9a / P9b ROIs lie in or at the press footprint, off the visible floor): a twin person standing there (zone
    None, off every floor polygon) is an image-only occluder. It makes the ROI MISSING while detected (held after),
    and gives no floor detection."""
    pr = [dict(id=p.id, state="running", ram=0.5, stroke=False, freeze=False) for p in SITE.presses]
    for pid in ("P9a", "P9b"):
        cam = CAM.CameraStatV1(SITE, WORLD, 1)
        p = cam.presses[pid]
        fid, (r0, s0, r1, s1) = cam.roi[pid]
        fp = np.asarray(p.footprint.value, float)
        g = np.random.default_rng(1)
        spot = None
        for _ in range(20000):
            x = float(g.uniform(fp[:, 0].min() - 2, fp[:, 0].max() + 2))
            z = float(g.uniform(fp[:, 1].min() - 2, fp[:, 1].max() + 2))
            if any(point_in_polygon(x, z, zz.floor_polygon.value) for zz in SITE.zones):
                continue
            if not point_in_polygon(x, z, p.footprint.value):    # at the press (70 % of the real occluders' feet)
                continue
            b = cam._box(fid, CAM._circle_prism(x, z, 0.3, CAM.HEIGHT_M["person"]))
            if b is None:
                continue
            pu, pv = 0.15 * (b[2] - b[0]), 0.15 * (b[3] - b[1])
            if b[0] - pu <= r1 and b[2] + pu >= r0 and b[1] - pv <= s1 and b[3] + pv >= s0:
                spot = (x, z)
                break
        assert spot is not None, pid
        vals, ndet = [], 0
        for k in range(300):
            mv = [dict(id="op", cls="person", zone=None, x=spot[0], z=spot[1], heading=0.0, speed=0.0, r=0.3)]
            d = cam.observe(tick(10.0 + k / 2.0, mv, pr))
            ndet += len(d)
            vals.append(d.zone_motion[pid])
        assert ndet == 0                                     # off the visible floor: no floor detection
        assert cam.stats["image_only"] == 300
        nan = np.isnan(vals)
        assert nan.mean() > 0.9, (pid, nan.mean())         # MISSING while detected, held across misses
        first = int(np.argmax(nan))
        assert nan[first:].all()


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


def test_per_camera_rates_not_pooled():
    """Recall, occluded / clean ratio and pixel sigma are per camera (camera 9 occluded person recall 15/55 vs camera
    14 17/22): the hold covers camera 9's own occluded recall (>= 21 s), not the pooled one."""
    P = CAM._P
    c9, c14 = CAM.class_noise(P, 9, "person"), CAM.class_noise(P, 14, "person")
    rj = P["labels"]["recall_jeffreys"]
    assert c9.recall_clean * c9.occ_ratio == pytest.approx(rj["c9_person_occluded"]["mean"])
    assert c14.recall_clean * c14.occ_ratio == pytest.approx(rj["c14_person_occluded"]["mean"])
    assert c9.occ_ratio < 0.4 < 0.8 < c14.occ_ratio
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    assert not np.allclose(cam.sig_px[9], cam.sig_px[14])
    assert cam.hold_s >= CAM.hold_ticks_markov(c9, 1.0 - c9.occ_ratio) / cam.hz >= 21.0
    assert set(cam.hold_detail) == {f"c{c}_{k}" for c in (9, 14) for k in ("person", "forklift", "robot")}
    assert all(v["hold_s_at_lo95"] >= v["hold_s"] for v in cam.hold_detail.values())


def test_hold_is_per_camera():
    """Audit fix2 (minor): one hold per camera (max over that camera's classes), not camera 9's forklift-driver hold
    everywhere: camera 14's presses are held for camera 14's own fitted bursts."""
    cam = CAM.CameraStatV1(SITE, WORLD, 1)
    for c in (9, 14):
        want = max(v["hold_s"] for k, v in cam.hold_detail.items() if k.startswith(f"c{c}_"))
        assert cam.hold_s_by_camera[c] == want
    assert cam.hold_s_by_camera[14] < cam.hold_s_by_camera[9] == cam.hold_s
    for p in SITE.presses:
        assert cam.hold_s_by_press[p.id] == cam.hold_s_by_camera[cam.cam_of_zone[p.zone]]
    assert cam.provenance()["hold_s_by_camera"] == {"9": cam.hold_s_by_camera[9], "14": cam.hold_s_by_camera[14]}


def test_plant_uses_stat_v1_and_its_hold():
    from twin import plant
    cam = plant.make_camera(SITE, WORLD, 1)
    assert isinstance(cam, CAM.CameraStatV1)
    ad = plant.ObsAdapter(SITE, hold_s=cam.hold_s_by_press)
    assert np.array_equal(ad.conv.hold_s, [cam.hold_s_by_press[p] for p in ad.press_ids])
    assert plant.ObsAdapter(SITE, hold_s=cam.hold_s).conv.hold_s == cam.hold_s     # one hold for every press
    with pytest.raises(TypeError):
        plant.ObsAdapter(SITE)                         # type: ignore[call-arg]  # no default hold (fail closed)
    with pytest.raises(ValueError, match="no hold for presses"):
        plant.ObsAdapter(SITE, hold_s={"P9a": 1.0})


def test_hold_gate_per_channel():
    from intercycle.observer.occlusion import HoldGate
    g = HoldGate(2, np.array([1.0, 3.0]))
    g.step(0.0, [True, True])
    miss, _ = g.step(2.0, [False, False])
    assert miss.tolist() == [False, True]
    with pytest.raises(ValueError):
        HoldGate(2, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        HoldGate(2, np.array([1.0, -1.0]))


def test_summary_without_camera_provenance_is_refused():
    """Audit fix2 (minor): Summary.camera is required, so EVAL fails closed on a run whose calibration draw is
    unrecorded."""
    from pydantic import ValidationError
    from twin.schemas.runs import Summary
    s = json.loads((CAM.REPO / "tests" / "fixtures" / "twin_summary.json").read_text(encoding="utf-8"))
    Summary.model_validate(s)
    with pytest.raises(ValidationError):
        Summary.model_validate({k: v for k, v in s.items() if k != "camera"})
    with pytest.raises(ValidationError):
        Summary.model_validate(dict(s, camera=dict(s["camera"], hold_s=1.0)))   # hold_s = the largest camera hold


def test_observer_live_reads_the_stat_v1_motion_threshold(tmp_path):
    """observer_live's threshold for stat_v1 zone_motion comes from twin/stat_v1_params.json (not the 0.5
    placeholder), with its source recorded, and observer_live does not import twin.camera (a truth reader)."""
    import ast

    from twin import observer_live as OL
    thr, src = OL.stat_v1_motion_thr()
    assert thr == CAM.MOTION_THR and thr > 1.0                 # idle median 1.0: 0.5 would read every press as motion
    assert src.startswith("stat_v1") and "energy.motion_thr" in src
    cfg = OL.config_for_stat_v1("unused_margin.json")
    assert cfg.motion_thr == thr and cfg.motion_thr_source == src
    assert not str(cfg.motion_thr_source).upper().startswith("PLACEHOLDER")
    tree = ast.parse((CAM.REPO / "twin" / "observer_live.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.startswith("twin") for m in mods), mods
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"contract": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        OL.stat_v1_motion_thr(bad)


def _site_v1_plant(tmp_path, world: int, seconds: float, intents=(), segments=()):
    """A twin plant on twin/site.yaml (the real site: 6 presses, all running), with a safety file derived from
    twin/safety.yaml for the site's id and presses (twin/safety.yaml covers site_v0 only)."""
    import yaml

    from twin import daygen_stub
    from twin import plant
    from twin.schemas.dayscript import DayScript, PressEvent

    pids = [p.id for p in SITE.presses]
    sd = yaml.safe_load((CAM.REPO / "twin" / "safety.yaml").read_text(encoding="utf-8"))
    sd["site_id"] = SITE.site_id
    sd["presses"] = [dict(sd["presses"][0], press=pid) for pid in pids]
    sp = tmp_path / "safety_v1.yaml"
    sp.write_text(yaml.safe_dump(sd), encoding="utf-8")
    t0 = 86_400.0
    ev = [PressEvent(press=pid, kind="running", t0=0.0, t1=600.0, period_s=2.0, phase_s=0.3, cause="production",
                     source="prior") for pid in pids]
    ds = DayScript(contract="twin.day_script/1", generator="test", generator_version="0",
                   rng_scheme=daygen_stub.RNG_SCHEME, world_seed=world, day=9, t0_s=t0, duration_s=600.0,
                   zones=["z9", "z14"], presses=pids, press_events=ev, mover_intents=list(intents),
                   segments=list(segments))
    dp = tmp_path / f"w{world}.day_script.json"
    dp.write_text(json.dumps(ds.model_dump(mode="json")), encoding="utf-8")
    cfg = {"seconds": seconds, "sim": "fake",
           "plant": {"site": str(CAM.REPO / "twin" / "site.yaml"), "safety": str(sp), "day_script": str(dp),
                     "run_id": f"w{world}", "zone": "z9", "world": world, "world_split": "dev",
                     "condition": "learned", "code": {"head": "test"}, "plant_t0": t0,
                     "run_dir": str(tmp_path / "runs" / f"w{world}"), "allow_short_mission": True}}
    pl = plant.make(cfg)
    st = {"pos_w": [-50.0, -50.0, 0.6], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "v_w": [0.0, 0.0, 0.0],
          "omega_b": [0.0, 0.0, 0.0]}                  # the robot parked far off both zones
    k = 0
    while round((k + 1) * 0.1, 6) <= seconds + 1e-9:
        t = round((k + 1) * 0.1, 6)
        pl.poses(t)
        pl.after_step(t, st, [])
        pl.frame_extras(t)
        k += 1
    pl.finish()
    obs = [OM.obs_from_dict(json.loads(ln)) for ln in (pl.run_dir / "obs_v2.jsonl").open(encoding="utf-8")]
    return pl, obs


@pytest.mark.parametrize("world", [3, 7])
def test_site_v1_plant_empty_floor_running_presses_give_finite_zone_motion(tmp_path, monkeypatch, world):
    """End to end on the real site: camera stat_v1 + twin/site.yaml + ObsAdapter. With an empty floor and every press
    running, zone_motion is finite on most ticks for every press (the ROI's visibility, not the walkable-floor share
    of rect + margin (0-0.40 here), sets zone_vis), and the timing signal reaches obs_v2.jsonl. Since fix2 the ROI
    follows the press through the world's true camera: a press whose ROI the world's camera puts partly outside the
    image (world 3: P14d) is MISSING on every tick (zone_vis < 1, fail closed), never read as a value."""
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))
    pl, obs = _site_v1_plant(tmp_path, world, 60.0)
    assert len(obs) >= 100
    zm = np.array([np.asarray(o.zone_motion, float) for o in obs])
    zv = np.array([np.asarray(o.zone_vis, float) for o in obs])
    fin = np.isfinite(zm).mean(0)
    cut = np.array([min(pl.camera.roi_vis[p].values()) < 1.0 for p in pl.adapter.press_ids])
    assert world != 3 or cut.tolist() == [p == "P14d" for p in pl.adapter.press_ids]
    assert (fin[~cut] > 0.9).all() and (fin[cut] == 0).all(), dict(zip(pl.adapter.press_ids, fin))
    assert (zv[:, ~cut] == 1.0).all() and (zv[:, cut] < 1.0).all()
    # the walkable-floor share alone would have forced NaN on every press (ZONE_VIS_MIN = 1)
    from twin import plant
    fl = plant.floor_share(pl.adapter.rects, [z.floor_polygon.value for z in SITE.zones])
    assert (fl.sum(1) < OM.ZONE_VIS_MIN).all()
    # a running press reads above the fitted motion threshold far more often than an idle press does (fpr_idle 0.12)
    fpr = float(CAM._P["energy"]["motion_thr"]["fpr_idle"])
    for j, pid in enumerate(pl.adapter.press_ids):
        if cut[j]:
            continue
        v = zm[np.isfinite(zm[:, j]), j]
        assert float(np.mean(v > CAM.MOTION_THR)) > 2.5 * fpr, pid
    # the world's calibration draw is recorded in summary.json and plant_summary.json
    from twin.schemas.runs import Summary
    summ = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert summ.camera is not None and summ.camera.cal_source == "cal_v1 draws"
    assert summ.camera.draw_index == {str(c): i for c, i in pl.camera.draw_index.items()}
    assert summ.camera.hold_s == pl.camera.hold_s
    assert summ.camera.hold_s_by_camera == {str(c): h for c, h in pl.camera.hold_s_by_camera.items()}
    assert np.array_equal(pl.adapter.conv.hold_s, [pl.camera.hold_s_by_press[p] for p in pl.adapter.press_ids])
    diag = json.loads((pl.run_dir / "plant_summary.json").read_text(encoding="utf-8"))
    assert diag["camera_provenance"] == summ.camera.model_dump(mode="json")
