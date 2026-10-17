"""Rules of the line-side extras (render/blender/extras.py): outside the guarding, cap per view, schedule.
    python -m pytest render/blender/test_extras.py -q"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extras  # noqa: E402

SCN = yaml.safe_load((Path(__file__).resolve().parents[2] / "scenario" / "bodyshop.yaml").read_text(encoding="utf-8"))
TIMES = [36000.0 + 0.5 * k for k in range(20)]
CART_Y = 13.2


def _cands(state="ACTIVE"):
    return extras.candidates(SCN, lambda mid, t: state, TIMES, CART_Y, seed=1)


def _inside_cell(x, y):
    for m in SCN["machines"]:
        w, d = m["size"]
        if abs(x - m["x"]) <= w / 2 + extras.GUARD_GAP and abs(y - m["z"]) <= d / 2 + extras.GUARD_GAP:
            return m["id"]
    return None


def test_every_spot_and_walk_stays_outside_the_guarding():
    for e in _cands("STOPPED"):
        for k in range(200):
            x, y, *_ = e.state(36000.0 + k * e.period / 200)
            assert _inside_cell(x, y) is None, (e.role, e.cell, x, y)


def test_maintenance_only_at_a_cell_down_on_every_sample():
    assert not [e for e in _cands("ACTIVE") if e.role == "maintenance"]
    assert {e.cell for e in _cands("TRIPPED") if e.role == "maintenance"} == {"S10", "S20", "S30", "S40", "S50", "S60", "S70"}
    flip = extras.candidates(SCN, lambda mid, t: "ACTIVE" if t > TIMES[-2] else "STOPPED", TIMES, CART_Y)
    assert not [e for e in flip if e.role == "maintenance"]


def test_schedule_dwells_walks_and_loops():
    e = next(c for c in _cands() if c.role == "operator" and c.cell == "S20" and c.b[1] < c.a[1])
    t0 = -e.phase                                   # u = 0: start of the dwell at a
    x, y, _, clip, v = e.state(t0 + 1.0)
    assert (x, y, clip, v) == (*e.a, "work", 0.0)
    x, y, yaw, clip, v = e.state(t0 + e.dwell_a + e.walk_s / 2)
    assert clip == "walk" and v == extras.WALK_V and math.isclose(y, (e.a[1] + e.b[1]) / 2)
    assert math.isclose(yaw, math.atan2(e.b[1] - e.a[1], e.b[0] - e.a[0]))
    assert e.state(t0 + e.dwell_a + e.walk_s + 1.0)[:2] == e.b
    assert e.state(t0 + 1.0) == e.state(t0 + 1.0 + e.period)
    _, _, yaw, _, _ = e.state(t0 + extras.TURN_S + 0.01)
    assert math.isclose(yaw, e.face_a)
    go = math.atan2(e.b[1] - e.a[1], e.b[0] - e.a[0])                  # turned to the walk before leaving
    _, _, yaw, _, _ = e.state(t0 + e.dwell_a - 0.001)
    assert abs(math.sin((yaw - go) / 2)) < 0.01


def test_heading_never_jumps_between_frames():
    for e in _cands("STOPPED"):
        prev = None
        for k in range(int(e.period * 25) + 1):
            yaw = e.state(36000.0 + k / 25)[2]
            if prev is not None:                            # wrapped difference: at most a turn step per frame
                assert abs((yaw - prev + math.pi) % (2 * math.pi) - math.pi) < math.pi / (extras.TURN_S * 25) + 1e-6, (e.role, e.cell, k)
            prev = yaw


def _all_in_view(p):
    return 10.0


def test_cap_is_two_and_never_more_than_trace_persons_in_view():
    cands = _cands()
    no_movers = lambda t: []  # noqa: E731
    assert len(extras.select(cands, TIMES, _all_in_view, lambda t: 9, no_movers)) == 2
    assert len(extras.select(cands, TIMES, _all_in_view, lambda t: 9 if t < TIMES[-1] else 1, no_movers)) == 1
    assert extras.select(cands, TIMES, _all_in_view, lambda t: 0, no_movers) == []
    assert len(extras.select(cands, TIMES, _all_in_view, lambda t: 9, no_movers, cap=5)) == 5


def test_selection_keeps_clear_of_movers_and_out_of_view_spots():
    cands = _cands()
    first = extras.select(cands, TIMES, _all_in_view, lambda t: 9, lambda t: [], cap=1)[0]
    blocked = extras.select(cands, TIMES, _all_in_view, lambda t: 9, lambda t: [first.state(t)[:2]], cap=1)[0]
    assert blocked != first
    assert extras.select(cands, TIMES, lambda p: None, lambda t: 9, lambda t: []) == []
    assert extras.select(cands, TIMES, lambda p: 50.0, lambda t: 9, lambda t: []) == []


def test_one_extra_per_cell_and_maintenance_first():
    picked = extras.select(_cands("HELD"), TIMES, _all_in_view, lambda t: 9, lambda t: [], cap=7)
    assert len({e.cell for e in picked}) == len(picked)
    assert all(e.role == "maintenance" for e in picked)
