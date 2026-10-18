"""A-lite M-W4 scenario (PLAN_TWIN §7 item 4): a program change and new arms 0.5-2 m from an existing cell become
UNKNOWN on the observer's map (S7a's in-zone detector + the static-space detector), short smoke version of
scripts/alite_mw4.py (the descriptive run learns 24 h)."""
import math

import numpy as np
import pytest

from intercycle.alite import mw4
from intercycle.alite.world import day_bounds
from intercycle.scenario import load

SCN = load()


def test_new_arms_sit_inside_the_hall_at_their_distances():
    t = 1000.0
    for name, near, d, edge, P in mw4.NEW_ARMS:
        arm = mw4.NewArm(SCN, near, d, edge, P, t)
        x0, z0, x1, z1 = SCN.machine(near).rect
        line = {"east": x1 + d, "west": x0 - d, "north": z1 + d, "south": z0 - d}[edge]
        assert abs(float(arm.tool0[0 if edge in ("east", "west") else 1]) - line) < 1e-9
        hx, hz = SCN.hall["x"], SCN.hall["z"]
        assert hx[0] < arm.base[0] < hx[1] and hz[0] < arm.base[1] < hz[1], name
        for dt in np.linspace(0, P, 9):
            c = arm.circles(t + dt)
            assert c.shape == (5, 3)
        assert np.allclose(arm.circles(t + 0.75 * P)[:, :2], arm.base)            # stowed half of the cycle


def test_first_full_and_occupied_cells():
    s = [(float(k), 1.0 if k >= 10 else 0.5) for k in range(200)]
    assert mw4.first_full(s, 60.0) == 10.0
    assert math.isinf(mw4.first_full([(float(k), 0.9) for k in range(200)], 60.0))
    occ = mw4.occupied_cells(np.array([[10.0, 10.0, 0.5]]))
    assert 10 <= len(occ) <= 16 and np.all(np.hypot(occ[:, 0] - 10, occ[:, 1] - 10) <= 0.5)


@pytest.fixture(scope="module")
def short():
    return mw4.run(SCN, 801, day_bounds(SCN)[1] + 3600.0, 20 * 60.0, 5 * 60.0)


def test_changes_become_unknown_within_5_min_and_no_false_alarm(short):
    print({k: (round(v["t_unknown_s"], 1), round(v["unknown_1_5min"], 2)) for k, v in short["changes"].items()})
    assert short["false_alarms_before_change"] == 0
    for name, c in short["changes"].items():
        assert c["n_samples"] > 200, name
        assert c["t_unknown_s"] <= 240.0, (name, c)


def test_learned_envelopes_are_scored_against_truth(short):
    e = short["envelopes"]
    print({k: e[k] for k in ("coverage", "iou", "bins_scored", "bins_seen")}, e["per_machine"])
    assert e["bins_scored"] >= 64
    assert 0.0 <= e["iou"] <= e["coverage"] <= 1.0
    assert e["coverage"] > 0.5


def test_envelope_score_counts_cells_outside_the_grid_as_missed():
    from intercycle.alite.observer import AliteObserver
    mp = AliteObserver(SCN).map
    sc = mw4.EnvelopeScore(mp)
    i = mp.machine_ids().index("HEM1")
    far = np.array([[mp.l1_x0[i] - 50.0, mp.l1_z0[i] - 50.0, 0.3]])
    for _ in range(10):
        sc.add(mp, i, 3, far)
    mp.l1_visits[i, 3] = mp.min_visits
    r = sc.score(mp)
    assert r["bins_scored"] == 1 and r["coverage"] == 0.0 and r["iou"] == 0.0
