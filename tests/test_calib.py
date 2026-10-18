"""CAL v1 math (intercycle/calib.py) on a synthetic camera with known tilt, roll, height and k1."""
from __future__ import annotations

import math

import numpy as np
import pytest

from intercycle import calib as C
from intercycle import framing as fr

TH, PH, H, K1 = math.radians(28.0), math.radians(2.5), 4.0, -1.6e-7


def _pix(P):
    """World floor/3D points -> distorted pixels."""
    P = np.asarray(P, float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    return fr.distort(C.project_world(P, TH, PH, H), K1)


def _cal(sigma=0.0):
    # an L on the floor: corner (1, 6), long side along +Y to (1.3, 11) rotated slightly, arm along +X
    yaw = math.radians(7.0)
    d_long = np.array([math.sin(yaw), math.cos(yaw)])
    d_arm = np.array([math.cos(yaw), -math.sin(yaw)])
    o = np.array([1.0, 6.0])
    pts = {"q1": o, "q2": o + 1.2 * d_arm, "q3": o + 5.0 * d_long, "q4": o + 5.0 * d_long + 1.0 * d_arm}
    raw = {k: _pix(v[None])[0] for k, v in pts.items()}
    # a long traced line parallel to the long side, 3 m to the left
    line_w = np.array([o - 3.0 * d_arm + t * d_long for t in np.linspace(-1, 8, 40)])
    line = fr.undistort(_pix(line_w), K1)
    cal = C.FramingCal(
        framing="syn", k1=K1, points_raw=raw, sigma_px={k: sigma for k in raw}, lines_undist={"L": line},
        segments={"long": ("q1", "q3"), "arm1": ("q1", "q2"), "arm2": ("q3", "q4")},
        relations=[C.Relation("perp", "long", "arm1"), C.Relation("perp", "long", "arm2"),
                   C.Relation("parallel", "long", "L")],
        origin="q1", axis="long")
    return cal, pts, d_long, d_arm


def test_undistort_cv2_matches_division_model():
    p = np.array([[100.0, 80.0], [1800.0, 1000.0], [960.0, 540.0], [1500.0, 300.0]])
    assert np.allclose(C.undistort_cv2(p, K1), fr.undistort(p, K1), atol=1e-3)


def test_h_cam_matches_framing_zero_roll():
    uv = np.array([[900.0, 800.0], [1300.0, 1000.0]])
    a = C.apply_h(C.h_cam(TH, 0.0, H), uv)
    b = fr.foot_to_ground(uv, fr.F_PRIOR, H, TH)
    assert np.allclose(a, b, atol=1e-9)


def test_fit_recovers_orientation_and_metric_frame():
    cal, pts, d_long, d_arm = _cal()
    b = cal.build()
    assert b["orientation"]["theta_deg"] == pytest.approx(28.0, abs=0.05)
    assert b["orientation"]["phi_deg"] == pytest.approx(2.5, abs=0.05)
    Hm = b["H"]
    # q3 is 5 m along +z from the origin; q2 1.2 m along the arm (x, up to its sign)
    q3 = C.apply_h(Hm, b["undist"]["q3"][None])[0]
    q2 = C.apply_h(Hm, b["undist"]["q2"][None])[0]
    assert q3 == pytest.approx([0.0, 5.0], abs=1e-3)
    assert abs(q2[0]) == pytest.approx(1.2, abs=1e-3) and q2[1] == pytest.approx(0.0, abs=1e-3)


def test_find_homography_reproduces_model_h():
    cal, *_ = _cal()
    b = cal.build()
    und = np.array([b["undist"][k] for k in ("q1", "q2", "q3", "q4")])
    xz = C.apply_h(b["H"], und)
    H2 = C.homography_from_points(und, xz)
    assert np.allclose(H2, b["H"], atol=1e-6)


def test_person_and_stripe_height_recover_h():
    foot_w = np.array([[0.5, 7.0, 0.0], [2.0, 9.0, 0.0], [-1.0, 12.0, 0.0]])
    head_w = foot_w + np.array([0.0, 0.0, C.WALKER_H_M])
    foot = C.project_world(foot_w, TH, PH, H)
    head = C.project_world(head_w, TH, PH, H)
    hs = C.person_height_h(foot, head, TH, PH)
    assert np.allclose(hs, H, atol=1e-3)
    ea = C.project_world(np.array([[0.0, y, 0.0] for y in np.linspace(6, 10, 20)]), TH, PH, H)
    eb = C.project_world(np.array([[0.1, y, 0.0] for y in np.linspace(6, 10, 20)]), TH, PH, H)
    assert C.stripe_h(ea, eb, TH, PH, 0.10) == pytest.approx(H, rel=1e-6)


def test_vertical_edges_fix_roll():
    # vertical world edges at several places: zero residual at the true orientation, and with one floor-parallel
    # pair they pin tilt and roll
    edges = {}
    for i, (x, y) in enumerate([(-4.0, 9.0), (3.0, 7.0), (6.0, 14.0), (-2.0, 15.0)]):
        P = np.array([[x, y, z] for z in np.linspace(0.2, 2.5, 25)])
        edges[f"v{i}"] = C.Segment(f"v{i}", C.project_world(P, TH, PH, H))
    vp = C.vertical_vp(TH, PH)
    for s in edges.values():
        assert C.vertical_resid_deg(s.pts, vp) < 1e-6
    fl = {"a": C.Segment("a", C.project_world(np.array([[0.0, y, 0.0] for y in np.linspace(5, 12, 20)]), TH, PH, H)),
          "b": C.Segment("b", C.project_world(np.array([[2.0, y, 0.0] for y in np.linspace(5, 12, 20)]), TH, PH, H))}
    segs = {**edges, **fl}
    rels = [C.Relation("vertical", k) for k in edges] + [C.Relation("parallel", "a", "b")]
    ori = C.fit_orientation(segs, rels)
    assert ori["theta_deg"] == pytest.approx(28.0, abs=0.05)
    assert ori["phi_deg"] == pytest.approx(2.5, abs=0.05)


def test_negative_w_normalisation_keeps_points():
    cal, *_ = _cal()
    b = cal.build()
    H = b["H"]
    assert np.isfinite(C.apply_h(H, b["undist"]["q3"][None])).all()
    assert np.isfinite(C.apply_h(-H, b["undist"]["q3"][None])).all()


def test_bootstrap_zero_noise_spread_is_scale_only():
    cal, *_ = _cal(sigma=0.0)
    b0 = cal.build()
    draws = C.bootstrap(cal, 4.0, 4.0, n=5)
    d = C.displacement_m(b0["H"], [x["H"] for x in draws], np.array([b0["undist"]["q3"]]))
    assert d.max() < 1e-6
    draws = C.bootstrap(cal, 3.6, 4.4, n=40)
    d = C.displacement_m(b0["H"], [x["H"] for x in draws], np.array([b0["undist"]["q3"]]))
    assert 0.0 < d.max() <= 0.5 + 1e-6            # q3 is 5 m from the origin: at most 10 % of 5 m


def test_bootstrap_noise_widens():
    cal, *_ = _cal(sigma=3.0)
    b0 = cal.build()
    draws = C.bootstrap(cal, 4.0, 4.0, n=30)
    d = C.displacement_m(b0["H"], [x["H"] for x in draws], np.array([b0["undist"]["q3"]]))
    assert np.percentile(d, 95) > 1e-3


def test_person_height_closed_form_rejects_bad_boxes():
    foot_w = np.array([[0.5, 7.0, 0.0]])
    foot = C.project_world(foot_w, TH, PH, H)
    head = C.project_world(foot_w + np.array([0.0, 0.0, C.WALKER_H_M]), TH, PH, H)
    above = np.array([[960.0, -400.0]])                    # above the horizon (row -167 at tilt 28 deg, f 1330)
    hs = C.person_height_h(np.vstack([foot, above]), np.vstack([head, above - [0.0, 50.0]]), TH, PH)
    assert hs[0] == pytest.approx(H, abs=1e-6) and np.isnan(hs[1])
    # the head drawn BELOW the foot is no upright person
    assert np.isnan(C.person_height_h(foot, foot + [0.0, 40.0], TH, PH)).all()


def test_at_height_rescales_one_draw_about_the_origin():
    cal, *_ = _cal(sigma=2.0)
    d = C.draw_orientations(cal, n=3, seed=1)
    for x in d:
        assert x["h"] == 4.0
        a = C.floor_of_raw(x, cal.points_raw["q3"])
        b = C.floor_of_raw(C.at_height(cal, x, 5.0), cal.points_raw["q3"])
        assert np.allclose(b, a * 5.0 / 4.0, atol=1e-9)       # origin at a placed point, no offset: pure scaling
        assert C.at_height(cal, x, 5.0)["orientation"] is x["orientation"]


def test_k1_is_drawn_between_fit_and_alternative():
    cal, *_ = _cal(sigma=0.0)
    cal.lines_raw = {k: fr.distort(v, cal.k1) for k, v in cal.lines_undist.items()}
    cal.k1_alt = 2.0 * cal.k1
    d = C.draw_orientations(cal, n=20, seed=2)
    k = np.array([x["k1"] for x in d])
    assert (k <= cal.k1 + 1e-18).all() and (k >= cal.k1_alt - 1e-18).all() and k.std() > 0
    # a wrong k1 bends the lines: the refitted orientation moves, the zero-noise nominal does not
    b0 = cal.build()
    disp = C.displacement_raw_m(b0, d, np.array([cal.points_raw["q4"]]))
    assert disp.max() > 1e-3
    cal.lines_raw = None
    with pytest.raises(ValueError):
        C.draw_orientations(cal, n=1)


def test_person_height_solves_every_visible_upright_person():
    # regression: the old per-box bisection bracketed h in [0.5, 30] m; at 0.5 m the 1.72 m head is above the camera
    # and the projected row wraps, so the bracket had no sign change and ~38 % of c9_f0's boxes were silently dropped
    th, ph = math.radians(32.7), math.radians(2.4)
    xs, ys = np.meshgrid(np.linspace(-6, 6, 9), np.linspace(3, 20, 9))
    foot_w = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    foot = C.project_world(foot_w, th, ph, H)
    head = C.project_world(foot_w + np.array([0.0, 0.0, C.WALKER_H_M]), th, ph, H)
    vis = (foot[:, 0] > 0) & (foot[:, 0] < fr.W) & (foot[:, 1] < fr.H_IMG) & (head[:, 1] > 0)
    assert vis.sum() > 20
    hs = C.person_height_h(foot[vis], head[vis], th, ph)
    assert np.allclose(hs, H, atol=1e-6)
