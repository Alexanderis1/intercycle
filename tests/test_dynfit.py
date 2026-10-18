"""DYN v1 building blocks (twin/dynfit.py): KM, stroke intervals, track pieces and speed, groups, camera, KS."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from twin import dynfit as DF
from twin.schemas import Site, load

SITE = Path(__file__).resolve().parents[1] / "twin" / "site.yaml"

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_kaplan_meier_matches_hand_computation():
    # events 1, 2, 4; censored 3 -> S(1)=3/4, S(2)=3/4*2/3=1/2, S(4)=1/2*(1-1/1)=0
    t, s = DF.kaplan_meier(np.array([1.0, 2.0, 4.0]), np.array([3.0]))
    assert t.tolist() == [1.0, 2.0, 4.0]
    assert np.allclose(s, [0.75, 0.5, 0.0])
    assert DF.km_survival_at(t, s, 0.5) == 1.0 and DF.km_survival_at(t, s, 3.0) == 0.5


def test_kaplan_meier_without_censoring_is_the_empirical_survival():
    d = np.array([1.0, 1.0, 2.0, 3.0])
    t, s = DF.kaplan_meier(d, np.zeros(0))
    assert np.allclose(s, [0.5, 0.25, 0.0])


def test_km_tail_sample_stays_on_observed_support():
    t, s = DF.kaplan_meier(np.array([1.0, 2.0, 5.0, 6.0]), np.array([7.0, 9.0]))
    rng = np.random.default_rng(0)
    x = [DF.km_sample_tail(rng, t, s, 3.0) for _ in range(200)]
    assert set(x) <= {5.0, 6.0}


def test_stroke_intervals_complete_and_censored():
    t = np.arange(0, 10.0, 0.5)
    stroke = np.zeros(len(t), bool)
    stroke[[0, 4, 8, 16]] = True              # 0, 2, 4, 8 s
    missing = np.zeros(len(t), bool)
    missing[11] = True                        # 5.5 s: interrupts 4 -> 8
    comp, cens = DF.stroke_intervals(t, stroke, missing)
    assert comp.tolist() == [2.0, 2.0]
    assert sorted(cens.tolist()) == [1.5, 1.5]       # 4 -> 5.5 (MISSING) and 8 -> 9.5 (clip end)


def test_observed_pieces_cut_at_floor_exit_and_jump():
    t = np.arange(0, 6.0, 1 / 12)
    x = np.where(t < 3.0, 1.0 + 0.5 * t, 6.0 + 0.5 * (t - 3.0))   # 3.5 m jump at 3 s (an ID switch)
    z = np.full(len(t), 5.0)
    ps = DF.observed_pieces(t, x, z, SQUARE, "person", 1.0)
    assert len(ps) == 2
    assert ps[0][0][-1] < 3.0 and ps[1][0][0] >= 2.9
    exit_z = 5.0 + 1.0 * t                                          # leaves the floor at z = 10 (t = 5 s)
    ps = DF.observed_pieces(t, np.full(len(t), 5.0), exit_z, SQUARE, "person", 1.0)
    assert len(ps) == 1 and ps[0][0][-1] < 5.0
    assert all(DF.inside(p[1], p[2], SQUARE).all() for p in ps)


def test_track_speed_of_a_constant_walk():
    t = np.arange(0, 5.0, 1 / 12)
    assert DF.track_speed(t, 1.2 * t, np.zeros(len(t))) == pytest.approx(1.2, rel=1e-6)
    assert math.isnan(DF.track_speed(np.array([0.0, 0.5]), np.zeros(2), np.zeros(2)))


def test_link_groups_joins_close_overlapping_walkers_only():
    t = np.arange(0, 4.0, 0.25)
    a = (t, t, np.zeros(len(t)))
    b = (t, t, np.full(len(t), 0.8))          # 0.8 m beside a
    c = (t, t, np.full(len(t), 5.0))          # far away
    lab = DF.link_groups([a, b, c])
    assert lab[0] == lab[1] != lab[2]


def test_decimate_keeps_ends_and_step():
    t = np.arange(0, 2.0, 1 / 12)
    k = DF.decimate(t)
    assert k[0] == 0 and k[-1] == len(t) - 1
    assert np.all(np.diff(t[k])[:-1] >= DF.DECIM_S - 1e-9)


def test_pixel_camera_round_trip_without_noise():
    site = load(Site, SITE)
    f = next(fr for fr in site.framings if fr.id == "c9_f0")
    cam = DF.PixelCamera(H=np.asarray(f.H_img_to_floor.value), k1=float(f.k1.value), sigma_u=0.0, sigma_v=0.0,
                         p_miss_start=0.0, p_miss_end=1.0)
    z9 = next(z for z in site.zones if z.id == "z9").floor_polygon.value
    x0 = float(np.mean([p[0] for p in z9]))
    z0 = float(np.mean([p[1] for p in z9]))
    t = np.arange(0, 2.0, 1 / 12)
    x, z = np.full(len(t), x0), z0 + 0.1 * t
    to, xo, zo = cam.observe(np.random.default_rng(1), t, x, z)
    assert len(to) > 0
    m = np.isin(t, to)
    assert np.allclose(xo, x[m], atol=1e-6) and np.allclose(zo, z[m], atol=1e-6)


def test_pixel_camera_miss_chain_rate():
    site = load(Site, SITE)
    f = next(fr for fr in site.framings if fr.id == "c9_f0")
    cam = DF.PixelCamera(H=np.asarray(f.H_img_to_floor.value), k1=0.0, sigma_u=0.0, sigma_v=0.0,
                         p_miss_start=0.1, p_miss_end=0.4)
    z9 = next(z for z in site.zones if z.id == "z9").floor_polygon.value
    x0, z0 = float(np.mean([p[0] for p in z9])), float(np.mean([p[1] for p in z9]))
    n = 20000
    to, _, _ = cam.observe(np.random.default_rng(2), np.arange(n) / 12, np.full(n, x0), np.full(n, z0))
    assert len(to) / n == pytest.approx(0.4 / 0.5, abs=0.03)


def test_lens_model_matches_intercycle_framing():
    from intercycle import framing as FR

    rng = np.random.default_rng(4)
    pts = rng.uniform([0, 0], [1920, 1080], (200, 2))
    for k1 in (-1.66e-07, -6.3e-08, 0.0):
        assert np.allclose(DF.undistort(pts, k1), FR.undistort(pts, k1))
        assert np.allclose(DF.distort(pts, k1), FR.distort(pts, k1))
    assert DF.CENTER == FR.CENTER


def test_ks_matches_scipy():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 300), rng.normal(0.3, 1.2, 200)
    ref = ks_2samp(a, b).statistic  # type: ignore[attr-defined]
    assert DF.ks_2samp(a, b) == pytest.approx(float(ref), abs=1e-12)


def test_beta_posterior_and_fwhm():
    assert DF.beta_posterior(3, 10) == (4.0, 8.0)
    prof = np.array([0, 0, 1, 2, 1, 0, 0], float)
    assert DF.fwhm_s(prof, 0.1) == pytest.approx(0.3)
