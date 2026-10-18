"""A-lite W-W1 check (PLAN_TWIN §7 item 1): the body-shop World does not use the keyed day-script RNG; KeyedWorld does."""
import numpy as np
import pytest

from intercycle.alite import world as AW
from intercycle.scenario import load
from intercycle.sim.world import World
from intercycle.truth.kinematics import Kinematics

SCN = load()
DAY = 86400.0
SEED = 801


def _rel(ws, t0, n=60):
    return np.array([w.bs - t0 for w in ws if w.bs >= t0][:n])


def test_body_shop_world_replays_one_stream_per_window():
    """The check's finding (W-W1): the body-shop World built for 07:00 of day 0 and of day 6 gives the same cycle
    starts relative to the window start (one RNG stream restarted per query window), up to the takt drift."""
    a = World(SCN, 5, 7 * 3600.0, 8 * 3600.0)
    b = World(SCN, 5, 6 * DAY + 7 * 3600.0, 6 * DAY + 8 * 3600.0)
    ra, rb = _rel(a.W["S10"], 7 * 3600.0), _rel(b.W["S10"], 6 * DAY + 7 * 3600.0)
    assert np.allclose(ra, rb, atol=1e-6)


def test_keyed_world_does_not_depend_on_the_query_window():
    t0, t1 = 7 * 3600.0, 9 * 3600.0
    a = AW.KeyedWorld(SCN, SEED, t0, t1)
    b = AW.KeyedWorld(SCN, SEED, t0 - 5 * 3600.0, t1 + 3 * 3600.0)
    for m in SCN.machines:
        wa = [w for w in a.W[m.id] if t0 <= w.bs < t1]
        wb = [w for w in b.W[m.id] if t0 <= w.bs < t1]
        assert wa and wa == wb, m.id


def test_keyed_world_days_differ_and_reproduce():
    d0 = AW.KeyedWorld(SCN, SEED, 7 * 3600.0, 8 * 3600.0)
    d1 = AW.KeyedWorld(SCN, SEED, DAY + 7 * 3600.0, DAY + 8 * 3600.0)
    r0, r1 = _rel(d0.W["S10"], 7 * 3600.0), _rel(d1.W["S10"], DAY + 7 * 3600.0)
    assert not np.allclose(r0[:len(r1)], r1[:len(r0)], atol=0.05)
    again = AW.KeyedWorld(SCN, SEED, 7 * 3600.0, 8 * 3600.0)
    assert again.W == d0.W
    other = AW.KeyedWorld(SCN, SEED + 1, 7 * 3600.0, 8 * 3600.0)
    assert other.W["HEM1"] != d0.W["HEM1"]


def test_day_boundary_is_the_shift_change_and_streams_are_independent():
    """Day 1 does not depend on day 0: the windows of day 1 built alone equal those of a two-day build."""
    a0, a1 = AW.day_bounds(SCN)
    assert (a0, a1) == (6 * 3600.0, 6 * 3600.0 + 300.0)
    one = AW.keyed_windows(SCN, SEED, DAY + a1 + 10, DAY + a1 + 20)
    two = AW.keyed_windows(SCN, SEED, a1 + 10, DAY + a1 + 20)
    for m in SCN.machines:
        assert [w for w in two[m.id] if w.bs >= DAY + a1] == [w for w in one[m.id] if w.bs >= DAY + a1], m.id
    line = AW.line_day(SCN, SEED, 0)
    assert line[0][0] == a1 and abs(line[-1][0] + line[-1][1] - (DAY + a1)) < 1e-9


def test_takt_drift_is_bounded_over_a_week():
    """The body-shop World over 7 days drifts takt 60 -> 69 s (W-W1); the keyed one stays within a few percent."""
    w = AW.KeyedWorld(SCN, SEED, 7 * 3600.0, 7 * DAY)
    tk = AW.takt_series(w, "S10")
    tk = tk[tk < 1.5 * SCN.line.takt_s]
    days = np.array_split(tk, 7)
    means = np.array([d.mean() for d in days])
    assert np.all(np.abs(means / SCN.line.takt_s - 1) < 0.03), means
    levels = [AW.drift_level(SCN, SEED, p) for p in range(20)]
    assert max(map(abs, levels)) < 0.04 and np.std(levels) > 0


def test_keyed_world_drives_the_body_shop_kinematics_and_robot_effects():
    w = AW.KeyedWorld(SCN, SEED, 7 * 3600.0, 7.5 * 3600.0)
    kin = Kinematics(SCN, w)
    t = 7 * 3600.0 + 100.0
    assert set(kin.bodies(t)) >= {"S10", "HEM1", "SKIDS"}
    ev = w.cross("S80", t, 7.0, "test")
    assert ev["type"] in ("trip", "inspect")


def test_test_world_seeds_are_refused(monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(PermissionError):
        AW.KeyedWorld(SCN, 1000, 7 * 3600.0, 8 * 3600.0)
