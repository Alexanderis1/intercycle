"""M-W7 conformal margin (lane S7b): calibration at the real fast-layer latency with a Learn-then-Test bound.

(a) zero-latency calibration is optimistic: on a synthetic moving-hazard stream it picks a smaller margin than the
    calibration at the real fast-layer age, and that margin fails alpha on fresh streams that have the real latency;
(b) the LTT bound holds at the stated alpha / delta in simulation (iid and clustered units), where the old
    conformal-risk-control rule (mean-only guarantee) does not hold with probability 1 - delta.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from intercycle.eval import conformal_margin as CM

CELL = 0.5
NY = NX = 24
DT = 0.5  # 2 Hz


# ------------------------------------------------------------------------------------------------ synthetic stream

def _stream(seed: int, minutes: float = 20.0, n_movers: int = 4, slice_s: float = 300.0) -> dict[str, Any]:
    """Movers bounce in a 12 x 12 m room at 0.8-1.6 m/s; truth = disc of radius 1 cell around each mover; the map's
    prediction from data at tick k = the truth at tick k (a perfect nowcast); the real fast-layer age is lognormal
    (median 0.8 s, clipped 0.3-2.0 s)."""
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 / DT)
    t = np.arange(T) * DT
    pos = rng.uniform(1.0, NX * CELL - 1.0, (n_movers, 2))
    ang = rng.uniform(0, 2 * np.pi, n_movers)
    spd = rng.uniform(0.8, 1.6, n_movers)
    yy, xx = np.mgrid[0:NY, 0:NX]
    cy, cx = (yy + 0.5) * CELL, (xx + 0.5) * CELL
    truth = np.zeros((T, NY, NX), bool)
    lim = NX * CELL
    for k in range(T):
        for i in range(n_movers):
            truth[k] |= (cy - pos[i, 0]) ** 2 + (cx - pos[i, 1]) ** 2 <= (1.0 * CELL) ** 2
        if rng.random() < 0.02:
            ang = ang + rng.normal(0, 1.0, n_movers)
        vel = np.stack([np.sin(ang), np.cos(ang)], 1) * spd[:, None]
        pos = pos + vel * DT
        for d in range(2):
            lo, hi = pos[:, d] < 0.3, pos[:, d] > lim - 0.3
            pos[lo, d] = 0.6 - pos[lo, d]
            pos[hi, d] = 2 * (lim - 0.3) - pos[hi, d]
            if d == 0:
                ang[lo | hi] = np.pi - ang[lo | hi]
            else:
                ang[lo | hi] = -ang[lo | hi]
    age = np.clip(rng.lognormal(math.log(0.8), 0.4, T), 0.3, 2.0)
    slice_id = (t // slice_s).astype(np.int64)
    return dict(t=t, truth=truth, t_pred=t.copy(), pred=truth.copy(), age_s=age, slice_id=slice_id)


MARGINS = tuple(round(0.25 * k, 2) for k in range(0, 17))  # 0 .. 4 m


def _units(streams, zero_latency: bool, episode_s: float = 30.0):
    parts = []
    for (w, d), s in streams.items():
        age = np.zeros_like(s["age_s"]) if zero_latency else s["age_s"]
        parts.append(CM.units_from_stream(s["t"], s["truth"], s["t_pred"], s["pred"], age, s["slice_id"], world=w,
                                          day=d, cell_m=CELL, margins_m=MARGINS, episode_s=episode_s))
    return CM.UnitCounts.concat(parts)


@pytest.fixture(scope="module")
def dev_and_fresh():
    dev = {(w, d): _stream(1000 * w + d) for w in range(4) for d in range(3)}
    fresh = {(w, d): _stream(777_000 + 1000 * w + d) for w in range(10, 14) for d in range(3)}
    return dev, fresh


def test_a_zero_latency_calibration_is_optimistic(dev_and_fresh):
    dev, fresh = dev_and_fresh
    alpha, delta = 0.10, 0.10
    real = CM.calibrate(_units(dev, False), alpha, delta, required_slices=range(4), zero_latency=_units(dev, True),
                        cell_m=CELL, episode_s=30.0)
    assert real["status"] == "ok"
    m_real, m_zero = real["margin_m"], real["margin_m_zero_latency"]
    assert m_zero is not None and m_real is not None
    assert m_zero < m_real, (m_zero, m_real)
    # on fresh streams that have the real latency, the zero-latency margin misses far more than alpha ...
    held = _units(fresh, False)
    assert CM.unit_mean_miss_share(held, m_zero) > 2 * alpha
    # ... while the margin calibrated at the real latency holds alpha, in every held-out world
    assert CM.unit_mean_miss_share(held, m_real) <= alpha
    assert all(v <= alpha for v in CM.per_world_miss_share(held, m_real).values())
    # the per-slice and leave-one-world-out diagnostics are present
    assert set(real["risk_per_slice_at_margin"]) == {0, 1, 2, 3}
    assert len(real["leave_one_world_out"]) == 4


# ------------------------------------------------------------------------------------------------ (b) LTT in simulation

M_GRID = np.round(np.arange(0.0, 6.0001, 0.05), 4)


def _true_risk(draw, n_mc: int = 400_000, seed: int = 12345) -> np.ndarray:
    S = draw(np.random.default_rng(seed), n_mc)
    return np.array([np.exp(-m / S).mean() for m in M_GRID])


def _crc_margin(L: np.ndarray, alpha: float) -> int | None:
    """The old map_metrics.conformal_margin rule (mean-only guarantee)."""
    n = len(L)
    for k in range(L.shape[1]):
        if n / (n + 1) * L[:, k].mean() + 1 / (n + 1) <= alpha:
            return k
    return None


def _violation_rates(draw_units, true_risk, alpha, delta, trials, mode):
    rng = np.random.default_rng(2024)
    v_ltt = v_crc = 0
    none = 0
    for _ in range(trials):
        S, clusters = draw_units(rng)
        L = np.exp(-M_GRID[None, :] / S[:, None])
        r = CM.ltt_fixed_sequence(L, M_GRID, alpha, delta, clusters, mode)
        if r["chosen_index"] is None:
            none += 1
        elif true_risk[r["chosen_index"]] > alpha:
            v_ltt += 1
        k = _crc_margin(L, alpha)
        if k is not None and true_risk[k] > alpha:
            v_crc += 1
    return v_ltt / trials, v_crc / trials, none / trials


def test_b_ltt_bound_holds_iid():
    alpha, delta, n, trials = 0.10, 0.10, 400, 300
    draw = lambda rng, k: rng.gamma(2.0, 0.5, k)  # noqa: E731
    R = _true_risk(draw)
    assert R[0] > alpha > R[-1]  # the grid brackets alpha
    v_ltt, v_crc, none = _violation_rates(lambda rng: (draw(rng, n), np.arange(n)), R, alpha, delta, trials, "kish")
    tol = 3 * math.sqrt(delta * (1 - delta) / trials)
    assert none == 0
    assert v_ltt <= delta + tol, v_ltt
    # the expectation-only rule is NOT a (1 - delta) bound: it violates alpha far more often than delta
    assert v_crc > delta + tol, v_crc


def test_b_ltt_bound_holds_clustered_with_kish():
    """Units correlated within (world, day) clusters: 30 clusters x 15 units with a shared cluster scale."""
    alpha, delta, trials = 0.10, 0.10, 300
    C, m = 40, 20

    def draw_units(rng):
        Sc = rng.gamma(2.0, 0.5, C)
        S = np.repeat(Sc, m) * np.exp(0.25 * rng.standard_normal(C * m))
        return S, np.repeat(np.arange(C), m)

    def draw(rng, k):
        return rng.gamma(2.0, 0.5, k) * np.exp(0.25 * rng.standard_normal(k))

    R = _true_risk(draw)
    v_kish, _, none = _violation_rates(draw_units, R, alpha, delta, trials, "kish")
    v_iid, _, _ = _violation_rates(draw_units, R, alpha, delta, trials, "none")
    tol = 3 * math.sqrt(delta * (1 - delta) / trials)
    assert none < 0.25  # the price of the correction: sometimes no margin is certified (then: do not freeze)
    assert v_kish <= delta + tol, v_kish
    # demonstration (fixed seed): treating the 800 correlated units as iid violates alpha more often than delta
    assert v_iid > delta > v_kish, (v_iid, v_kish)


# ------------------------------------------------------------------------------------------------ building blocks

def test_hb_pvalue_basic():
    # zero observed loss: Bentkus term e * (1 - alpha)^n
    assert CM.hb_pvalue(0.0, 100, 0.05) == pytest.approx(min(math.e * 0.95 ** 100, math.exp(-100 * math.log(1 / 0.95))))
    assert CM.hb_pvalue(0.2, 100, 0.1) == 1.0  # rhat above alpha: never rejected
    assert CM.hb_pvalue(0.0, 0, 0.1) == 1.0
    ps = [CM.hb_pvalue(r, 500, 0.1) for r in (0.0, 0.02, 0.05, 0.08)]
    assert ps == sorted(ps)  # monotone in rhat
    assert CM.hb_pvalue(0.02, 2000, 0.1) < CM.hb_pvalue(0.02, 200, 0.1)  # more units, stronger evidence


def test_fixed_sequence_stops_at_first_non_rejection():
    L = np.array([[0.9, 0.0, 0.5, 0.0]] * 300)  # non-monotone on purpose: margin 1 would pass alone
    r = CM.ltt_fixed_sequence(L, [0, 1, 2, 3], 0.1, 0.1, np.arange(300), "none")
    assert r["margin_m"] == 3.0  # testing from the largest margin down stops at margin 2


def test_icc_and_kish():
    g = np.repeat(np.arange(10), 20)
    y_ind = np.random.default_rng(0).random(200)
    assert CM.kish_n_eff(y_ind, g) > 150
    y_cl = np.repeat(np.random.default_rng(1).random(10), 20)  # identical within cluster
    assert CM.kish_n_eff(y_cl, g) == pytest.approx(10.0, rel=1e-6)
    assert CM.icc_oneway(np.zeros(50), np.repeat(np.arange(5), 10)) == 0.0
    assert CM.icc_oneway(np.random.default_rng(2).random(10), np.zeros(10)) == 1.0


def test_align_by_age_uses_the_prediction_the_robot_holds():
    t_pred = np.array([0.0, 0.5, 1.0, 1.5])
    idx = CM.align_by_age(np.array([0.0, 1.0, 1.5, 1.5]), np.array([0.1, 0.0, 0.8, 0.5]), t_pred)
    assert idx.tolist() == [-1, 2, 1, 2]
    with pytest.raises(ValueError):
        CM.align_by_age(np.array([1.0]), np.array([-0.1]), t_pred)


def test_tick_counts_match_brute_force_disc_dilation():
    rng = np.random.default_rng(3)
    truth = rng.random((6, 9, 11)) < 0.15
    pred = rng.random((4, 9, 11)) < 0.08
    pred[2] = False  # an empty prediction covers nothing at any margin
    idx = np.array([0, 1, 2, 3, -1, 1])
    margins = (0.0, 0.25, 0.5, 0.75, 1.0)
    occ, miss = CM.tick_counts(truth, pred, idx, 0.25, margins)
    yy, xx = np.mgrid[0:9, 0:11]
    for j in range(6):
        if idx[j] < 0:
            assert occ[j] == 0 and not miss[j].any()
            continue
        assert occ[j] == truth[j].sum()
        py, px = np.nonzero(pred[idx[j]])
        for k, m in enumerate(margins):
            cov = np.zeros((9, 11), bool)
            for a, b in zip(py, px):
                cov |= ((yy - a) ** 2 + (xx - b) ** 2) * 0.25 ** 2 <= m ** 2 + 1e-12
            assert miss[j, k] == (truth[j] & ~cov).sum(), (j, m)
    assert np.all(np.diff(miss, axis=1) <= 0)


def test_units_never_cross_slices_and_drop_invalid_ticks():
    t = np.arange(20) * 0.5
    sl = np.array([0] * 7 + [1] * 13)
    occ = np.ones(20, np.int64)
    miss = np.zeros((20, 2), np.int64)
    valid = np.ones(20, bool)
    valid[:2] = False
    u = CM.stream_units(t, sl, occ, miss, valid, world=1, day=2, margins_m=(0.0, 1.0), episode_s=3.0)
    # valid ticks 2..6 are slice 0 (blocks 0, 1); ticks 7..19 are slice 1 (blocks 1, 2, 3)
    assert u.dropped_ticks == 2
    assert u.slice_id.tolist() == [0, 0, 1, 1, 1]
    assert u.n_ticks.tolist() == [4, 1, 5, 6, 2]
    assert u.occ.tolist() == [4, 1, 5, 6, 2]


def test_calibrate_refuses_missing_slices_and_unfrozen_results(tmp_path):
    n = 15  # e * 0.9^15 = 0.56 > delta: too few units to certify alpha 0.1 even with zero observed loss
    L_occ = np.full(n, 10, np.int64)
    u = CM.UnitCounts(world=np.zeros(n, np.int64), day=np.arange(n) % 5, slice_id=np.zeros(n, np.int64),
                      t0=np.arange(float(n)), n_ticks=np.full(n, 10), occ=L_occ,
                      miss=np.stack([np.full(n, 10), np.zeros(n, np.int64)], 1), margins_m=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="time slice"):
        CM.calibrate(u, 0.1, 0.1, required_slices=[0, 1])
    res = CM.calibrate(u, 0.1, 0.1, required_slices=[0])
    assert res["status"] == "no_valid_margin"  # too few units
    with pytest.raises(ValueError, match="no valid margin"):
        CM.write_margin_file(res, tmp_path / "m.json")


def test_write_and_freeze(tmp_path):
    n = 400
    u = CM.UnitCounts(world=np.repeat(np.arange(4), 100), day=np.tile(np.repeat(np.arange(4), 25), 4),
                      slice_id=np.tile(np.arange(4), 100), t0=np.arange(float(n)), n_ticks=np.full(n, 60),
                      occ=np.full(n, 20, np.int64), miss=np.stack([np.full(n, 20), np.zeros(n, np.int64)], 1),
                      margins_m=np.array([0.0, 0.5]))
    res = CM.calibrate(u, 0.1, 0.1, required_slices=range(4), cell_m=0.25, episode_s=720.0)
    assert res["status"] == "ok" and res["margin_m"] == 0.5 and res["margin_cells"] == 2.0
    p = CM.write_margin_file(res, tmp_path / "m7.json")
    body = json.loads(p.read_text(encoding="utf-8"))
    assert body["frozen"] is True and body["margin_m"] == 0.5 and body["alpha"] == 0.1 and body["delta"] == 0.1
    assert body["unit"]["episode_s"] == 720.0 and len(body["input_sha256"]) == 64
    with pytest.raises(FileExistsError):
        CM.write_margin_file(res, p)


def test_dev_only_loader(tmp_path, monkeypatch):
    s = _stream(5, minutes=1.0)
    log = tmp_path / "truth_access.log"

    def save(name, **over):
        d: dict[str, Any] = dict(s, world_seed=3, day=2, cell_m=CELL, split="dev")
        d.update(over)
        f = tmp_path / name
        np.savez(f, **d)
        return f

    good = save("dev.npz")
    assert int(CM.load_dev_stream(good, log)["world_seed"]) == 3
    assert "conformal_margin" in log.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="dev worlds only"):
        CM.load_dev_stream(save("test.npz", split="test"), log)
    with pytest.raises(ValueError, match="not a dev seed"):
        CM.load_dev_stream(save("seed.npz", world_seed=123456), log)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(RuntimeError):
        CM.load_dev_stream(good, log)


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))
    files = []
    for w in range(3):
        for d in range(2):
            s = _stream(50 + 10 * w + d, minutes=10.0)
            f = tmp_path / f"w{w}d{d}.npz"
            np.savez(f, **s, world_seed=w, day=d, cell_m=CELL, split="dev")
            files.append(str(f))
    out = tmp_path / "m7.json"
    rc = CM.main([*files, "--alpha", "0.2", "--delta", "0.1", "--episode-s", "30", "--margins-m", "0:4:0.25",
                  "--slices", "0,1", "--out", str(out)])
    assert rc == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["margin_m"] > body["margin_m_zero_latency"]
    assert body["fast_layer_age_s"]["p50"] == pytest.approx(0.8, abs=0.1)
