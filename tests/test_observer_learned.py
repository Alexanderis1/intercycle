"""Robot B's broadcast comes from the learned observer (trained on a separate day), never from past truth."""
import numpy as np
import pytest

from intercycle.scenario import load
from intercycle.sim.observer_learned import TRAIN_SEED, LearnedObserver, learned_models, true_models
from intercycle.sim.world import Freeze, World
from intercycle.types import State

T0, T1 = 7 * 3600.0, 8 * 3600.0


@pytest.fixture(scope="module")
def scn():
    return load()


def test_training_day_models_match_the_plant(scn):
    models, _ = learned_models(scn)
    for m, t in zip(models, true_models(scn)):
        assert m.confidence >= 4
        assert abs(m.period_s / t.period_s - 1) < 0.01, (m, t)
        assert abs(m.busy_fraction - t.busy_fraction) < 0.02, (m, t)


def test_evaluation_seed_must_not_be_the_training_day(scn):
    with pytest.raises(ValueError):
        LearnedObserver(scn, World(scn, TRAIN_SEED, T0, T1), TRAIN_SEED, T0, T1)


def test_camera_never_samples_ahead_and_sees_robot_freezes(scn):
    w = World(scn, 3, T0, T1)
    obs = LearnedObserver(scn, w, 3, T0, T1)
    cam = obs.cams[0]
    t, y = cam.upto(T0 + 600.0, 600.0)
    assert t[-1] <= T0 + 600.0 and cam.t[cam.n_done - 1] <= T0 + 600.0
    w.groups["line"].freezes.append(Freeze(T0 + 610.0, T0 + 700.0, "trip", "S10"))     # robot trips the line
    t, y = cam.upto(T0 + 700.0, 80.0)
    assert y[(t > T0 + 615.0) & (t < T0 + 695.0)].mean() < 0.1                          # frozen line reads idle (flips only)


def test_nowcast_tracks_the_episode(scn):
    """Predicted next idle start for ACTIVE line stations vs the episode's truth, at 2 s issues over 30 min."""
    w = World(scn, 4, T0, T1)
    obs = LearnedObserver(scn, w, 4, T0, T1)
    err = []
    for ti in np.arange(T0 + 600.0, T0 + 2400.0, 2.0):
        f = obs.fast(float(ti), want=0)
        for i in obs.line:
            mn, mid = f.machines[i], scn.machines[i].id
            if mn.state == State.ACTIVE and mn.confidence >= 4 and w.active(mid, float(ti)):
                wins = w.true_idle_windows(mid, float(ti), 300)
                if wins:
                    err.append(abs(mn.next_idle_in_s - (wins[0][0] - ti)))
    err = np.array(err)
    assert len(err) > 1000
    assert np.median(err) < 1.0 and np.mean(err < 3.0) > 0.9, (np.median(err), np.mean(err < 3.0))
