"""Fault injection. A broken or hostile broadcast may cost robot B its advantage, never make it
less safe than a robot without the code (robot A)."""
from concurrent.futures import ProcessPoolExecutor

import pytest

from intercycle.scenario import load
from intercycle.sim.observer_learned import learned_models
from intercycle.sim.run import run

SEEDS = range(10, 18)
T0, T1 = 7 * 3600, 10 * 3600
FAULTS = ["phase", "flip", "replay", "forge", "freeze"]


def _trips(job):
    strategy, seed, fault = job
    return job, run(strategy, seed, T0, T1, fault=fault).metrics["trips"]


@pytest.fixture(scope="module")
def trips():
    """Every (strategy, seed, fault) run once, in parallel (B runs the learned observer: ~10 s each)."""
    learned_models(load())                          # train once (disk-cached) before fanning out
    jobs = [("A", s, None) for s in SEEDS] + [("B", s, f) for f in FAULTS for s in SEEDS]
    with ProcessPoolExecutor(4) as ex:
        return dict(ex.map(_trips, jobs))


@pytest.mark.parametrize("fault", FAULTS)
def test_faulty_code_never_less_safe_than_no_code(fault, trips):
    a_trips = sum(trips[("A", s, None)] for s in SEEDS)
    b_trips = sum(trips[("B", s, fault)] for s in SEEDS)
    assert b_trips <= a_trips, (fault, b_trips, a_trips)


def test_clean_broadcast_gives_zero_loss():
    for seed in (5, 6):
        m = run("B", seed).metrics
        assert m["trips"] == 0 and m["holds"] == 0, m
