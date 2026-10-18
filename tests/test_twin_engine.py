"""twin/engine.py (lane S1b): a day event script -> the realised stream at 2 Hz (PLAN_TWIN §2), on the G0 stub day
(twin/daygen_stub.py: 30-min day, 2 presses, Poisson walkers, 1 forklift). W-W1 (keyed draws, no replay), W-W3 (the
engine adds no stops), W-W4 (apply_freeze executes the plant's trip / reset / hold)."""
from __future__ import annotations

import json
import math
import random

import pytest

from twin import daygen_stub
from twin.engine import DT, FREEZE_STATES, MOVER_R, Engine, RealisedWriter, load_day_script, write_realised
from twin.schemas import load
from twin.schemas.dayscript import DayScript, MoverIntent, PressEvent, Segment
from twin.schemas.dynamics import Dynamics
from twin.schemas.runs import RealisedMover, parse_realised
from twin.schemas.site import Site

SITE = load(Site, daygen_stub.SITE_V0)
DYN = load(Dynamics, daygen_stub.DYNAMICS_V0)


@pytest.fixture(autouse=True)
def _truth_log(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))


def _day(world=3, day=2, **kw) -> DayScript:
    return daygen_stub.generate_day(world, day, site=SITE, dynamics=DYN, **kw)


def _engine(script: DayScript | None = None) -> Engine:
    return Engine(script or _day(), SITE, DYN)


def _lines(engine: Engine, tmp_path, name="a") -> bytes:
    p = write_realised(engine, tmp_path / name / "realised_state.jsonl", run_id="r", world=0)
    return p.read_bytes()


def _script(events: list[PressEvent], intents=(), segments=(), t0_s=86_400.0, duration=120.0) -> DayScript:
    return DayScript(contract="twin.day_script/1", generator="test", generator_version="0", rng_scheme=daygen_stub.RNG_SCHEME,
                     world_seed=0, day=2, t0_s=t0_s, duration_s=duration, zones=["z9", "z14"],
                     presses=sorted({e.press for e in events}, key=[e.press for e in events].index),
                     press_events=events, mover_intents=list(intents), segments=list(segments))


def _run(press="P9a", t0=0.0, t1=120.0, period=4.0, phase=1.0) -> PressEvent:
    return PressEvent(press=press, kind="running", t0=t0, t1=t1, period_s=period, phase_s=phase, cause="production",
                      source="prior")


def _stop(press="P9a", t0=0.0, t1=120.0) -> PressEvent:
    return PressEvent(press=press, kind="stopped", t0=t0, t1=t1, cause="short_stop", source="assumed")


# ------------------------------------------------------------------------------------------------- the stream


def test_stub_day_stream_validates_against_the_frozen_schema(tmp_path):
    script = _day()
    p = write_realised(_engine(script), tmp_path / "runs" / "x" / "realised_state.jsonl", run_id="x", world=0)
    with open(p, encoding="utf-8") as f:
        header, ticks = parse_realised(f)
        ts = list(ticks)                                 # parse_realised checks the 2 Hz grid and the press set
    assert header.hz == 2.0 and header.day == 2 and header.t0_s == script.t0_s
    assert header.presses == ["P9a", "P14a"] and header.zones == ["z9", "z14"]
    assert len(ts) == int(script.duration_s * 2) and ts[0].t == script.t0_s
    classes = {m.cls for tk in ts for m in tk.movers}
    assert classes == {"person", "forklift"}
    assert sum(1 for tk in ts for pr in tk.presses if pr.stroke) > 0
    assert all(0.0 <= pr.ram <= 1.0 for tk in ts for pr in tk.presses)
    assert {m.id for tk in ts for m in tk.movers if m.cls == "forklift"} == {"forklift:0"}
    assert all(m.r == MOVER_R[m.cls] for tk in ts for m in tk.movers)
    raw = p.read_bytes()
    assert b"\r\n" not in raw and b"NaN" not in raw


def test_camera_v0_reads_the_engine_stream(tmp_path):
    """Slice step 3 -> 4: the plant-side camera stand-in consumes the engine's stream unchanged."""
    from twin import camera_v0
    script = _day()
    p = write_realised(_engine(script), tmp_path / "realised_state.jsonl", run_id="x", world=0)
    dets = list(camera_v0.stream(p, SITE, world_seed=script.world_seed))
    assert len(dets) == int(script.duration_s * 2)
    assert set().union(*(d.zone_motion for d in dets)) == {"P9a", "P14a"}
    assert sum(len(d) for d in dets) > 0


def test_engine_writes_and_reads_truth_through_the_guard(tmp_path):
    script = _day()
    sp = daygen_stub.write_day(script, tmp_path / "w3_d2.day_script.json")
    assert load_day_script(sp) == script
    write_realised(Engine(load_day_script(sp), SITE, DYN), tmp_path / "realised_state.jsonl", run_id="x", world=0)
    log = (tmp_path / "truth_access.log").read_text(encoding="utf-8").splitlines()
    rows = [r.split("\t") for r in log[1:]]
    assert [(r[1], r[2]) for r in rows] == [("twin/daygen_stub.py", "w"), ("twin/engine.py", "r"),
                                             ("twin/engine.py", "r"), ("twin/engine.py", "w")]
    with pytest.raises(ValueError, match="not a truth path"):
        write_realised(_engine(script), tmp_path / "out.jsonl", run_id="x", world=0)


def test_deterministic_byte_identical(tmp_path):
    assert _lines(_engine(), tmp_path, "a") == _lines(_engine(), tmp_path, "b")


def test_ticks_are_order_independent():
    eng = _engine()
    times = list(eng.tick_times())
    fwd = {t: eng.tick(t) for t in times}
    shuffled = times[:]
    random.Random(7).shuffle(shuffled)
    eng2 = _engine()
    for t in shuffled:
        assert eng2.tick(t) == fwd[t]
    assert eng.tick(times[100]) == fwd[times[100]]      # re-evaluating a tick changes nothing


def test_generation_order_does_not_change_the_stream(tmp_path):
    a = _day()
    b = _day(entity_order=["forklifts", "z14", "P14a", "z9", "P9a"])
    assert _lines(_engine(a), tmp_path, "a") == _lines(_engine(b), tmp_path, "b")


def test_w_w1_keyed_entities_and_no_replay():
    """Dropping zone z14's walkers leaves every z9 walker's realised track unchanged (keyed draws, W-W1); another
    day of the same world is a different day, not a replay."""
    full = _engine(_day())
    part = _engine(_day(entity_order=["P9a", "P14a", "z9", "forklifts"]))

    def z9(eng):
        return [[m for m in eng.tick(t).movers if m.id.startswith("walker:z9:")] for t in eng.tick_times()]

    assert z9(full) == z9(part)
    assert any(m.id.startswith("walker:z14:") for t in full.tick_times() for m in full.tick(t).movers)
    other = _engine(_day(day=3))
    a = [(p.state, p.stroke) for t in full.tick_times() for p in full.tick(t).presses]
    b = [(p.state, p.stroke) for t in other.tick_times() for p in other.tick(t).presses]
    assert a != b


def test_engine_adds_no_stops_w_w3():
    """Without freezes the press state follows the script's events exactly (the engine draws nothing)."""
    script = _day()
    eng = _engine(script)
    for t in eng.tick_times():
        rel = t - script.t0_s
        for pr in eng.tick(t).presses:
            ev = next(e for e in script.press_events if e.press == pr.id and e.t0 <= rel < e.t1)
            assert pr.state == ev.kind and not pr.freeze


# ------------------------------------------------------------------------------------------------- presses


def test_strokes_follow_phase_and_period():
    s = _script([_run(phase=1.0, period=4.0), _stop("P14a")])
    eng = Engine(s, SITE, stroke_impulse_s=0.6)
    st = eng.stroke_times("P9a")
    assert st == pytest.approx([s.t0_s + 1.0 + 4.0 * k for k in range(30)])
    flagged = [t for t in eng.tick_times() if eng.tick(t).presses[0].stroke]
    assert flagged == pytest.approx([math.ceil(x * 2) / 2 for x in st])
    # ram: 0 at top dead centre, the impulse 0.5 (1 - cos(2 pi s / d)) for s < d
    t = s.t0_s + 1.0 + 0.5
    assert eng.tick(t).presses[0].ram == pytest.approx(0.5 * (1 - math.cos(2 * math.pi * 0.5 / 0.6)), abs=1e-6)
    assert eng.tick(s.t0_s + 3.0).presses[0].ram == 0.0
    p14 = [eng.tick(t).presses[1] for t in eng.tick_times()]
    assert all(p.state == "stopped" and p.ram == 0.0 and not p.stroke for p in p14)


def test_stroke_impulse_comes_from_dynamics():
    assert Engine(_day(), SITE, DYN).presses["P9a"].impulse == DYN.get("press.stroke_impulse_s").value
    assert Engine(_day(), SITE, stroke_impulse_s=0.3).presses["P9a"].impulse == 0.3


def test_apply_freeze_stops_strokes_holds_ram_and_delays_the_bout():
    s = _script([_run(phase=1.0, period=4.0), _stop("P14a")])
    free = Engine(s, SITE, stroke_impulse_s=0.6)
    eng = Engine(s, SITE, stroke_impulse_s=0.6)
    a, b = s.t0_s + 21.2, s.t0_s + 31.2                # freeze mid-stroke (stroke at 21.0) for 10 s
    eng.apply_freeze("P9a", a, b, "tripped")
    for t in eng.tick_times():
        p, q = eng.tick(t).presses[0], free.tick(t).presses[0]
        if t < a:
            assert p == q
        elif t < b:
            assert p.state == "tripped" and p.freeze and not p.stroke
            assert p.ram == pytest.approx(free.presses["P9a"].ram_free(a), abs=1e-6) and p.ram > 0
        else:
            assert p.state == "running" and not p.freeze
    before = [x for x in free.stroke_times("P9a") if x < a]
    after = [x + 10.0 for x in free.stroke_times("P9a") if x >= a and x + 10.0 < s.t0_s + 120.0]
    assert eng.stroke_times("P9a") == pytest.approx(before + after)


def test_freeze_does_not_move_scheduled_boundaries_and_drops_overrun_strokes():
    s = _script([_run(t1=60.0, phase=0.0, period=5.0), _stop(t0=60.0, t1=120.0), _stop("P14a")])
    eng = Engine(s, SITE)
    eng.apply_freeze("P9a", s.t0_s + 50.0, s.t0_s + 70.0, "resetting")
    st = eng.stroke_times("P9a")
    assert st == pytest.approx([s.t0_s + 5.0 * k for k in range(10)])     # 50, 55 lost: the stop starts at 60
    assert eng.tick(s.t0_s + 75.0).presses[0].state == "stopped"


def test_a_stroke_due_at_the_freeze_start_starts_when_it_ends():
    s = _script([_run(phase=0.0, period=10.0), _stop("P14a")])
    eng = Engine(s, SITE)
    eng.apply_freeze("P9a", s.t0_s + 10.0, s.t0_s + 13.0, "held")
    st = eng.stroke_times("P9a")
    assert st[:3] == pytest.approx([s.t0_s, s.t0_s + 13.0, s.t0_s + 23.0])
    assert not eng.tick(s.t0_s + 10.0).presses[0].stroke and eng.tick(s.t0_s + 13.0).presses[0].stroke


def test_hold_across_a_restart_delays_the_first_stroke():
    s = _script([_stop(t1=30.0), _run(t0=30.0, phase=0.0, period=4.0), _stop("P14a")])
    eng = Engine(s, SITE)
    eng.apply_freeze("P9a", s.t0_s + 25.0, s.t0_s + 37.0, "held")
    assert eng.tick(s.t0_s + 30.0).presses[0].state == "held"
    assert eng.stroke_times("P9a")[0] == pytest.approx(s.t0_s + 37.0)
    assert eng.tick(s.t0_s + 36.5).presses[0].ram == 0.0


def test_freeze_calls_commute_and_state_precedence_is_fixed(tmp_path):
    s = _day()
    fz = [("P9a", s.t0_s + 100.0, s.t0_s + 160.0, "tripped"), ("P9a", s.t0_s + 150.0, s.t0_s + 400.0, "resetting"),
          ("P14a", s.t0_s + 300.0, s.t0_s + 320.0, "held"), ("P9a", s.t0_s + 390.0, s.t0_s + 420.0, "held"),
          ("P14a", s.t0_s + 310.0, s.t0_s + 330.0, "tripped")]
    out = []
    for order in (fz, fz[::-1], [fz[2], fz[0], fz[4], fz[1], fz[3]]):
        eng = _engine(s)
        for f in order:
            eng.apply_freeze(*f)
        out.append(_lines(eng, tmp_path, f"o{len(out)}"))
    assert out[0] == out[1] == out[2]
    eng = _engine(s)
    for f in fz:
        eng.apply_freeze(*f)
    assert eng.tick(s.t0_s + 155.0).presses[0].state == "tripped"      # tripped > resetting
    assert eng.tick(s.t0_s + 395.0).presses[0].state == "resetting"    # resetting > held
    assert eng.tick(s.t0_s + 315.0).presses[1].state == "tripped"      # tripped > held
    assert FREEZE_STATES == ("tripped", "resetting", "held")


def test_apply_freeze_fails_closed():
    s = _day()
    eng = _engine(s)
    with pytest.raises(KeyError):
        eng.apply_freeze("P99", s.t0_s, s.t0_s + 1)
    with pytest.raises(ValueError):
        eng.apply_freeze("P9a", s.t0_s + 5, s.t0_s + 5)
    with pytest.raises(ValueError):
        eng.apply_freeze("P9a", s.t0_s, s.t0_s + 5, "stopped")
    with pytest.raises(ValueError):
        eng.apply_freeze("P9a", s.t0_s, math.inf)
    eng.tick(s.t0_s + 10.0)
    eng.apply_freeze("P9a", s.t0_s + 10.0, s.t0_s + 12.0)           # at the latest emitted tick: allowed
    with pytest.raises(ValueError, match="rewrite emitted truth"):
        eng.apply_freeze("P9a", s.t0_s + 9.5, s.t0_s + 12.0)


def test_tick_refuses_off_grid_and_out_of_day_times():
    s = _day()
    eng = _engine(s)
    with pytest.raises(ValueError, match="grid"):
        eng.tick(s.t0_s + 0.25)
    with pytest.raises(ValueError, match="outside"):
        eng.tick(s.t0_s + s.duration_s)
    with pytest.raises(ValueError, match="outside"):
        eng.tick(s.t0_s - DT)


# ------------------------------------------------------------------------------------------------- movers


def _walker_script(t_start=10.0, scale=1.0, pts=((0.0, 2.0, 3.0), (5.0, 7.0, 3.0), (8.0, 7.0, 3.0), (10.0, 7.0, 5.0))):
    seg = Segment(id="seg:w", cls="person", source="synthetic_stub", points=list(pts))
    it = MoverIntent(entity="walker:z9:0000", cls="person", zone="z9", t_start=t_start, segment="seg:w",
                     speed_scale=scale)
    return _script([_stop(), _stop("P14a")], [it], [seg])


def test_mover_follows_its_segment():
    s = _walker_script()
    eng = Engine(s, SITE)
    assert eng.tick(s.t0_s + 9.5).movers == []                     # before t_start
    m = eng.tick(s.t0_s + 12.5).movers[0]                          # local 2.5 s: halfway along +x at 1 m/s
    assert (m.x, m.z, m.speed, m.zone) == pytest.approx((4.5, 3.0, 1.0, "z9"))
    assert m.heading == pytest.approx(math.pi / 2)                 # +x is clockwise from +z
    p = eng.tick(s.t0_s + 16.5).movers[0]                          # pause leg: speed 0, heading carried over
    assert p.speed == 0.0 and p.heading == pytest.approx(math.pi / 2) and (p.x, p.z) == (7.0, 3.0)
    q = eng.tick(s.t0_s + 19.0).movers[0]
    assert q.heading == pytest.approx(0.0) and q.z == pytest.approx(4.0)
    assert eng.tick(s.t0_s + 20.0).movers and not eng.tick(s.t0_s + 20.5).movers   # end inclusive


def test_mover_speed_scale_and_zone_none_outside_the_floor():
    s = _walker_script(t_start=0.0, scale=2.0)
    m = Engine(s, SITE).tick(s.t0_s + 1.0).movers[0]               # local 2.0 s
    assert (m.x, m.speed) == pytest.approx((4.0, 2.0))
    s = _walker_script(t_start=0.0, pts=((0.0, 20.0, 3.0), (5.0, 25.0, 3.0)))   # between the zones: unseen floor
    assert Engine(s, SITE).tick(s.t0_s + 1.0).movers[0].zone is None


def test_extra_movers_are_appended_and_validated(tmp_path):
    s = _day()
    eng = _engine(s)
    robot = RealisedMover(id="robot", cls="robot", zone="z14", x=31.0, z=1.0, heading=0.0, speed=0.0, r=0.45)
    tk = eng.tick(s.t0_s, [robot])
    assert tk.movers[-1] == robot
    p = write_realised(eng, tmp_path / "realised_state.jsonl", "r", 0, s.t0_s, s.t0_s + 5.0, extra=lambda t: [robot])
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[1:]]
    assert len(rows) == 10 and all(r["movers"][-1]["id"] == "robot" for r in rows)
    with pytest.raises(ValueError):
        eng.tick(s.t0_s, [robot, robot])


def test_writer_refuses_gaps(tmp_path):
    s = _day()
    eng = _engine(s)
    with RealisedWriter(eng, tmp_path / "realised_state.jsonl", "r", 0) as w:
        w.write(eng.tick(s.t0_s))
        with pytest.raises(ValueError, match="not 0.5 s after"):
            w.write(eng.tick(s.t0_s + 1.0))
