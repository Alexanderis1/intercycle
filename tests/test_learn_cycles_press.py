"""Lane S6 (PLAN_TWIN §4): learn_cycles on a press-shop zone_motion stream.

* constants per machine kind (press: Schmitt w=1, FLIP_S 0.25 s; DEBOUNCE_S and P_MIN placeholders until probe P4,
  refused in a run marked sealed);
* T-W6: MISSING (NaN) is masked, never thresholded to "no motion";
* a synthetic press stream with realistic stroke periods (1.2-3 s) at the 2 Hz Observation v2 rate and 10 % NaN bursts
  is learned with confidence >= MIN_CONF and a period within 5 %.
"""
from __future__ import annotations

import numpy as np
import pytest

from intercycle.observer import learn_cycles as LC
from intercycle.types import State

PRESS = LC.PRESS_PLACEHOLDER


def press_stream(P: float, n: int = 2400, dt: float = 0.5, duty: float = 0.4, jit: float = 0.01,
                 nan_share: float = 0.10, burst: tuple[int, int] = (2, 8), seed: int = 0, t0: float = 1000.0,
                 stop: tuple[float, float] | None = None):
    """2 Hz binary motion of a press stroking every P s (cycle-to-cycle jitter `jit`, ram moving for duty x P), sampled
    at the capture instants, with MISSING bursts of 2-8 ticks until `nan_share` of the ticks are NaN. `stop` = (t0, t1):
    no strokes start inside it."""
    rng = np.random.default_rng(seed)
    T = n * dt
    iv = P * (1 + jit * rng.standard_normal(int(T / P) + 10))
    s = t0 - rng.uniform(0, P) + np.cumsum(iv)
    if stop is not None:
        s = np.r_[s[s < stop[0]], s[s >= stop[0]] + (stop[1] - stop[0])]
    t = t0 + dt * np.arange(n) + rng.uniform(0, dt)
    k = np.searchsorted(s, t, side="right") - 1
    ph = t - s[np.clip(k, 0, len(s) - 1)]
    y = ((k >= 0) & (ph < duty * P)).astype(float)
    m = np.zeros(n, bool)
    while m.mean() < nan_share:
        a = int(rng.integers(0, n))
        m[a:a + int(rng.integers(burst[0], burst[1] + 1))] = True
    y[m] = np.nan
    return t, y


@pytest.mark.parametrize("P", [1.2, 1.6, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_press_stream_with_nan_bursts_is_learned(P, seed):
    t, y = press_stream(P, seed=seed)
    assert np.isnan(y).mean() >= 0.10
    L = LC.learn(t, y, consts=PRESS, latency_s=0.0)
    assert L.model.confidence >= LC.MIN_CONF, L.diag
    assert abs(L.model.period_s / P - 1) < 0.05, (L.model.period_s, P)


def test_body_shop_constants_do_not_learn_a_fast_press():
    """The body-shop kind cannot see a 1.6 s press (P_MIN 5 s, 4 s debounce): the per-kind constants are needed."""
    t, y = press_stream(1.6, seed=0)
    L = LC.learn(t, y, latency_s=0.0)
    assert not (L.model.confidence >= LC.MIN_CONF and abs(L.model.period_s / 1.6 - 1) < 0.05)


def test_prep_keeps_missing():
    t = np.arange(10) * 0.5
    y = np.array([0, 1, np.nan, 1, 0, np.nan, np.nan, 0.7, 0.2, 1])
    _, yb, dt = LC._prep(t, y)
    assert dt == 0.5
    assert np.array_equal(np.isnan(yb), np.isnan(y))
    assert yb[np.isfinite(yb)].tolist() == [0, 1, 1, 0, 1, 0, 1]


def test_schmitt_holds_through_missing():
    y = np.array([1, 1, 1, np.nan, np.nan, np.nan, 1, 0, 0, np.nan, 0])
    assert LC._schmitt(y, 1).tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert LC._schmitt(np.full(5, np.nan), 3).tolist() == [0] * 5


def test_edge_next_to_missing_is_not_timed():
    """A stop that starts inside a MISSING gap: the fall at the end of the gap is not reported as an edge time."""
    t = np.arange(400) * 0.5
    y = ((t % 20.0) < 8.0).astype(float)                # 20 s cycle, 8 s active (body-shop kind)
    gap = (t >= 105.0) & (t < 112.0)                    # the fall at t = 108 happens inside the gap
    y[gap] = np.nan
    e = LC.detect_edges(t, y, latency_s=0.0)
    assert len(e.fall) > 5
    assert not np.any(np.abs(e.fall - 112.0) < 1.0)     # zero-filled data would put a fall at 105, held data at 112
    assert not np.any(np.abs(e.fall - 105.0) < 1.0)
    assert np.any(np.abs(e.fall - 128.0) < 1.0)         # the next, fully observed fall is there


def test_nowcast_is_unknown_while_missing():
    t, y = press_stream(2.0, seed=3)
    L = LC.learn(t, y, consts=PRESS, latency_s=0.0)
    assert L.model.confidence >= LC.MIN_CONF
    i = int(np.flatnonzero(np.isnan(y))[-1])
    n = LC.nowcast(L.model, t, y, t[i], consts=PRESS, latency_s=0.0)
    assert n.state == State.UNKNOWN and n.confidence == 0
    z = np.nan_to_num(y)                                 # the old behaviour: MISSING read as "no motion"
    n0 = LC.nowcast(L.model, t, z, t[i], consts=PRESS, latency_s=0.0)
    assert n0.state != State.UNKNOWN


def test_press_stop_is_reported_stopped():
    t, y = press_stream(2.0, seed=4, nan_share=0.0, stop=(1600.0, 1700.0))
    win = t < 1590.0
    L = LC.learn(t[win], y[win], consts=PRESS, latency_s=0.0)
    assert L.model.confidence >= LC.MIN_CONF
    n = LC.nowcast(L.model, t, y, 1620.0, consts=PRESS, latency_s=0.0)
    assert n.state == State.STOPPED
    run = LC.nowcast(L.model, t, y, 1500.0, consts=PRESS, latency_s=0.0)
    assert run.state in (State.ACTIVE, State.IDLE)


def test_placeholder_constants_fail_loudly_in_a_sealed_run():
    assert set(PRESS.placeholder) == {"debounce_s", "p_min"}
    assert PRESS.schmitt_w == 1 and PRESS.flip_s == 0.25
    LC.require_measured(PRESS, sealed=False)            # dev runs may use the placeholder (disclosed)
    with pytest.raises(LC.PlaceholderConstantError, match="debounce_s, p_min"):
        LC.require_measured(PRESS, sealed=True)
    LC.require_measured(LC.BODY_SHOP, sealed=True)


def test_consts_from_config_clears_placeholders_only_with_a_source():
    half = LC.consts_from_config({"kind": "press", "p_min": {"value": 0.9, "source": "P4: 0.8 x p10 period (n=12)"}})
    assert half.p_min == 0.9 and half.placeholder == ("debounce_s",)
    with pytest.raises(LC.PlaceholderConstantError):
        LC.require_measured(half, sealed=True)
    still = LC.consts_from_config({"kind": "press", "p_min": {"value": 0.9, "source": "PLACEHOLDER"}})
    assert "p_min" in still.placeholder
    full = LC.consts_from_config({"kind": "press", "p_min": {"value": 0.9, "source": "P4 p10"},
                                  "debounce_s": {"value": 0.25, "source": "P4 shortest gap"}})
    assert LC.require_measured(full, sealed=True) is full
    with pytest.raises(ValueError):
        LC.consts_from_config({"kind": "press", "p_min": 0.9})
    with pytest.raises(ValueError):
        LC.consts_from_config({"kind": "press", "nope": {"value": 1, "source": "x"}})
    with pytest.raises(ValueError):
        LC.consts_from_config({"kind": "lathe"})


def test_body_shop_kind_is_the_module_constants():
    b = LC.BODY_SHOP
    assert (b.schmitt_w, b.flip_s, b.debounce_s, b.p_min, b.p_max, b.min_run) == (3, LC.FLIP_S, LC.DEBOUNCE_S, LC.P_MIN,
                                                                                  LC.P_MAX, 2)
    assert b.period_from == "acf" and b.cycle_tol_frac is None and not b.placeholder
