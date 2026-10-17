"""Synthetic checks for intercycle.framing (probe P6 / CAL v0 k1)."""
import math

import numpy as np
import pandas as pd
import pytest

from intercycle import framing as fr


def test_focal_priors_match_plan():
    assert fr.focal_from_fov(fr.VENDOR_VFOV_DEG, 1080) == pytest.approx(1330, abs=10)
    assert fr.focal_from_fov(fr.VENDOR_HFOV_DEG, 1920) == pytest.approx(1020, abs=5)
    assert fr.vendor_k1_norm_prior() < 0  # vendor FOV wider than the pinhole one => barrel


def test_division_roundtrip():
    k1 = -0.4 / fr.F_PRIOR**2
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [1919, 1079], (200, 2))
    assert np.allclose(fr.undistort(fr.distort(pts, k1), k1), pts, atol=1e-6)


def _lines(k1, n_lines=3, seed=1):
    rng = np.random.default_rng(seed)
    chains = []
    for _ in range(n_lines):
        a = rng.uniform([100, 100], [1800, 980])
        ang = rng.uniform(0, math.pi)
        t = np.linspace(-700, 700, 400)
        p = a + np.outer(t, [math.cos(ang), math.sin(ang)])
        p = p[(p[:, 0] > -300) & (p[:, 0] < 2200) & (p[:, 1] > -300) & (p[:, 1] < 1400)]
        d = fr.distort(p, k1)
        d = d[(d[:, 0] >= 0) & (d[:, 0] < 1920) & (d[:, 1] >= 0) & (d[:, 1] < 1080)]
        chains.append(d + rng.normal(0, 0.3, d.shape))
    return chains


def test_fit_k1_recovers_truth():
    k_true = -0.35
    chains = _lines(k_true / fr.F_PRIOR**2, n_lines=3)
    fit = fr.fit_k1(chains)
    assert fit["k1_norm"] == pytest.approx(k_true, abs=0.03)
    assert max(fit["rms_after_px"]) < 0.6


def test_fit_k1_robust_drops_curved_chain():
    k_true = -0.3
    chains = _lines(k_true / fr.F_PRIOR**2, n_lines=4, seed=3)
    t = np.linspace(0, math.pi / 2, 300)
    chains.append(np.stack([960 + 400 * np.cos(t), 540 + 400 * np.sin(t)], 1))  # a real circle
    r = fr.fit_k1_robust(chains)
    assert 4 not in r["consistent"]
    assert r["top_fit"]["k1_norm"] == pytest.approx(k_true, abs=0.04)


def test_opencv_coeffs_reproduce_division_model():
    cv2 = pytest.importorskip("cv2")
    k1 = -0.35 / fr.F_PRIOR**2
    oc = fr.opencv_coeffs(k1)
    assert oc["max_err_px"] < 1.0
    pts = np.array([[100.0, 100.0], [1800.0, 950.0], [960.0, 1000.0]])
    K = np.array(oc["K"])
    u = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.array(oc["dist"]), P=K).reshape(-1, 2)
    assert np.allclose(u, fr.undistort(pts, k1), atol=1.5)


def test_ground_camera_roundtrip_and_height_fit():
    f, h, th = 1330.0, 4.0, math.radians(28)
    rng = np.random.default_rng(2)
    X = rng.uniform(-4, 4, 80)
    Y = rng.uniform(3, 15, 80)
    feet = fr.project(np.column_stack([X, Y, np.zeros(80)]), f, h, th)
    back = fr.foot_to_ground(feet, f, h, th)
    assert np.allclose(back, np.column_stack([X, Y]), atol=1e-6)
    heads = fr.project(np.column_stack([X, Y, np.full(80, 1.72)]), f, h, th)[:, 1]
    fit = fr.fit_height_tilt(feet, heads + rng.normal(0, 1.0, 80), f)
    assert fit["h_m"] == pytest.approx(4.0, rel=0.03)
    assert fit["tilt_deg"] == pytest.approx(28, abs=1.5)
    # a scaled scene (people 10 % smaller in pixels at the same feet) reads as a 10 % higher camera
    fit2 = fr.fit_height_tilt(feet, feet[:, 1] - 0.9 * (feet[:, 1] - heads), f)
    assert fit2["h_m"] > 4.2


def test_height_linear_zero_tilt():
    f, h = 1330.0, 4.0
    Y = np.linspace(4, 20, 30)
    feet = fr.project(np.column_stack([np.zeros(30), Y, np.zeros(30)]), f, h, 0.0)
    heads = fr.project(np.column_stack([np.zeros(30), Y, np.full(30, 1.72)]), f, h, 0.0)
    r = fr.fit_height_linear(feet[:, 1], feet[:, 1] - heads[:, 1])
    assert r["h_m"] == pytest.approx(4.0, rel=1e-6)
    assert r["horizon_v"] == pytest.approx(fr.CENTER[1], abs=1e-6)


def test_ground_m_per_px_scales_with_height():
    a = fr.ground_m_per_px(900, 1330, 4.0, math.radians(25))
    b = fr.ground_m_per_px(900, 1330, 4.4, math.radians(25))
    assert b / a == pytest.approx(1.1, rel=1e-9)


def test_corner_displacement():
    Hm = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], float)
    assert fr.corner_displacement(Hm, 960, 540) == pytest.approx(math.hypot(5, 3))


def test_assign_framings_greedy():
    pos = {"a": 0, "b": 1, "c": 50, "d": 52, "e": 200}
    split = {"a": "train", "b": "train", "c": "train", "d": "dev", "e": "dev"}

    def match(x, ref):
        return 100, abs(pos[x] - pos[ref])

    r = fr.assign_framings(list(pos), lambda c: 9, lambda c: split[c] == "train", match, 50, 10)
    ids = {c: v["framing_id"] for c, v in r["clips"].items()}
    assert ids == {"a": "c9_f0", "b": "c9_f0", "c": "c9_f1", "d": "c9_f1", "e": "c9_unmatched"}
    assert r["refs"] == {"9": ["a", "c"]}


def test_merge_framings_refuses_test():
    inv = pd.DataFrame({"clip_id": ["x", "y", "z"], "split": ["train", "dev", "test"],
                        "framing_id": ["c9_f0"] * 3})
    out = fr.merge_framings(inv, {"x": "c9_f1", "y": "c9_f0"})
    assert list(out["framing_id"]) == ["c9_f1", "c9_f0", "c9_f0"]
    with pytest.raises(ValueError):
        fr.merge_framings(inv, {"z": "c9_f1"})


def test_edge_chains_finds_drawn_lines():
    cv2 = pytest.importorskip("cv2")
    img = np.full((1080, 1920), 90, np.uint8)
    cv2.rectangle(img, (300, 400), (1600, 460), 220, -1)  # a bright stripe: two long horizontal edges
    chains = fr.edge_chains(img)
    assert len(chains) >= 2
    assert all(fr.line_rms(c) < 1.0 for c in chains)


def test_trace_edge_follows_distorted_stripe_and_k1_fit():
    cv2 = pytest.importorskip("cv2")
    k_true = -0.3
    k1 = k_true / fr.F_PRIOR**2
    img = np.full((1080, 1920), 80, np.uint8)
    hints = []
    for (a, b) in [((1500, 0), (1900, 1079)), ((100, 1000), (1800, 950)), ((50, 100), (600, 1050))]:
        t = np.linspace(-0.2, 1.2, 800)[:, None]
        line = np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float))
        d = fr.distort(line, k1)
        cv2.polylines(img, [np.round(d * 4).astype(np.int32)], False, 220, 24, cv2.LINE_AA, shift=2)
        hints.append((fr.distort(np.array([a], float), k1)[0], fr.distort(np.array([b], float), k1)[0]))
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    chains = [fr.trace_edge(img, p, q) for p, q in hints]
    assert all(len(c) > 50 for c in chains)
    fit = fr.fit_k1(chains)
    assert fit["k1_norm"] == pytest.approx(k_true, abs=0.03)


def test_stripe_height_recovers_camera_height():
    f, h, th, k1 = 1330.0, 4.0, math.radians(27), -0.29 / fr.F_PRIOR**2
    Y = np.linspace(3, 12, 60)
    a = fr.project(np.column_stack([np.full(60, 1.0), Y, np.zeros(60)]), f, h, th)
    b = fr.project(np.column_stack([np.full(60, 1.1), Y, np.zeros(60)]), f, h, th)
    r = fr.stripe_height(fr.distort(a, k1), fr.distort(b, k1), k1, f, th, width_m=0.10)
    assert r["h_m"] == pytest.approx(4.0, rel=0.02)
