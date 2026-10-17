"""Fault injection. A broken or hostile broadcast may cost robot B its advantage, never make it
less safe than a robot without the code (robot A)."""
from concurrent.futures import ProcessPoolExecutor

import pytest

from intercycle.scenario import load
from intercycle.sim.observer_learned import learned_models
from intercycle.sim.run import run

SEEDS = range(10, 18)
T0, T1 = 7 * 3600, 10 * 3600
FAULTS = ["phase", "flip", "long", "replay", "forge", "freeze"]


def _metrics(job):
    strategy, seed, fault = job
    m = run(strategy, seed, T0, T1, fault=fault).metrics
    return job, (m["trips"], m["line_time_lost_s"])


@pytest.fixture(scope="module")
def runs():
    """Every (strategy, seed, fault) run once, in parallel (B runs the learned observer: ~10 s each)."""
    learned_models(load())                          # train once (disk-cached) before fanning out
    jobs = [("A", s, None) for s in SEEDS] + [("B", s, f) for f in FAULTS for s in SEEDS]
    with ProcessPoolExecutor(4) as ex:
        return dict(ex.map(_metrics, jobs))


@pytest.mark.parametrize("fault", FAULTS)
def test_faulty_code_never_less_safe_than_no_code(fault, runs):
    """Trips and line time lost (trips + holds): B with a broken broadcast <= A without any code."""
    a_trips, a_lost = (sum(runs[("A", s, None)][i] for s in SEEDS) for i in (0, 1))
    b_trips, b_lost = (sum(runs[("B", s, fault)][i] for s in SEEDS) for i in (0, 1))
    assert b_trips <= a_trips, (fault, b_trips, a_trips)
    assert b_lost <= a_lost, (fault, b_lost, a_lost)


@pytest.mark.xfail(reason="E-W6, pre-existing since 4660155; real fix in lane S8 (A-lite)", strict=False)
def test_clean_broadcast_gives_zero_loss():
    for seed in (5, 6):
        m = run("B", seed).metrics
        assert m["trips"] == 0 and m["holds"] == 0, m
