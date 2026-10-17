"""Fault injection. A broken or hostile broadcast may cost robot B its advantage, never make it
less safe than a robot without the code (robot A)."""
import pytest

from intercycle.sim.run import run

SEEDS = range(10, 18)
T0, T1 = 7 * 3600, 10 * 3600


@pytest.fixture(scope="module")
def a_trips():
    return sum(run("A", s, T0, T1).metrics["trips"] for s in SEEDS)


@pytest.mark.parametrize("fault", ["phase", "flip", "replay", "forge", "freeze"])
def test_faulty_code_never_less_safe_than_no_code(fault, a_trips):
    b_trips = sum(run("B", s, T0, T1, fault=fault).metrics["trips"] for s in SEEDS)
    assert b_trips <= a_trips, (fault, b_trips, a_trips)


def test_clean_broadcast_gives_zero_loss():
    for seed in (5, 6):
        m = run("B", seed).metrics
        assert m["trips"] == 0 and m["holds"] == 0, m
