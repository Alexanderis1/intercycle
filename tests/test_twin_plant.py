"""twin/plant.py (lane S5a items 3-4; W-W4, R-W2, R-W3): trip, hold, the lognormal manual reset, truth isolation
(the plant's output reaches the observer side only through files) and the body-shop plant left unchanged.

The robot is driven by hand through the runner's hooks (poses / after_step / frame_extras / on_ros / finish), the way
intercycle_sim.runner calls them, at 10 Hz."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from twin import daygen_stub
from twin import plant as PL
from twin import robot_camera as RC
from twin.schemas import load
from twin.schemas.dayscript import DayScript, MoverIntent, PressEvent, Segment
from twin.schemas.runs import Summary, parse_realised, require_primary
from twin.schemas.safety import Safety
from twin.schemas.site import Site
from twin.schemas.tables import LogRow, validate_table
from intercycle.observer import mapping as OM

REPO = Path(__file__).resolve().parents[1]
SITE_PATH = daygen_stub.SITE_V0
SAFETY_PATH = REPO / "twin" / "safety.yaml"
T0 = 86_400.0
OUT = (6.25, 3.0)            # outside P9a's danger polygon (z 5.0-6.0, dilation 0.3)
IN = (6.25, 4.6)             # footprint inside the dilated polygon


@pytest.fixture(autouse=True)
def _truth_log(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))


def _run(press="P9a", t0=0.0, t1=600.0, period=4.0, phase=1.0) -> PressEvent:
    return PressEvent(press=press, kind="running", t0=t0, t1=t1, period_s=period, phase_s=phase, cause="production",
                      source="prior")


def _stop(press="P9a", t0=0.0, t1=600.0) -> PressEvent:
    return PressEvent(press=press, kind="stopped", t0=t0, t1=t1, cause="short_stop", source="assumed")


def _script(p9a: list[PressEvent], intents=(), segments=(), duration=600.0, world_seed=5) -> DayScript:
    events = p9a + [_run("P14a", 0.0, duration, 3.0, 0.5)]
    return DayScript(contract="twin.day_script/1", generator="test", generator_version="0",
                     rng_scheme=daygen_stub.RNG_SCHEME, world_seed=world_seed, day=9, t0_s=T0, duration_s=duration,
                     zones=["z9", "z14"], presses=["P9a", "P14a"], press_events=events,
                     mover_intents=list(intents), segments=list(segments))


def _cfg(tmp_path, script: DayScript, seconds=120.0, safety=SAFETY_PATH, name="r1", **plant) -> dict:
    """The tests' cfg: the EVAL identity fields (world, world_split, condition, code) are set here only; the plant
    itself has no defaults for them. Test missions are shorter than T = 720 s, so allow_short_mission is set."""
    sp = tmp_path / f"{name}.day_script.json"
    sp.write_text(json.dumps(script.model_dump(mode="json")), encoding="utf-8")
    return {"seconds": seconds, "sim": "fake",
            "plant": {"site": str(SITE_PATH), "safety": str(safety), "day_script": str(sp), "run_id": name,
                      "zone": "z9", "world": 0, "world_split": "dev", "condition": "learned",
                      "code": {"head": "test"}, "plant_t0": T0, "run_dir": str(tmp_path / "runs" / name),
                      "allow_short_mission": True, **plant}}


def _plant(tmp_path, script: DayScript, seconds=120.0, safety=SAFETY_PATH, name="r1", **plant) -> PL.Plant:
    return PL.make(_cfg(tmp_path, script, seconds, safety, name, **plant))


def _tasks(tmp_path, **over) -> Path:
    """The tasks fixture on the site's id (tests/fixtures/twin_tasks.yaml: cp1 at P9a and cp2 in z9, cp3 at P14a in
    z14)."""
    d = yaml.safe_load((REPO / "tests" / "fixtures" / "twin_tasks.yaml").read_text(encoding="utf-8"))
    d["site_id"] = load(Site, SITE_PATH).site_id
    d.update(over)
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return p


def _st(x, y, vx=0.0, vy=0.0, psi=0.0) -> dict:
    return {"pos_w": [x, y, 0.6], "quat_wxyz": [math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)],
            "v_w": [vx, vy, 0.0], "omega_b": [0.0, 0.0, 0.0]}


def _drive(plant: PL.Plant, where, seconds: float, dt: float = 0.1, extras: list | None = None) -> None:
    """where(t) -> (x, y[, vx, vy]); steps the hooks like intercycle_sim.runner."""
    k = 0
    while True:
        t = round((k + 1) * dt, 6)
        if t > seconds + 1e-9:
            break
        plant.poses(t)
        plant.after_step(t, _st(*where(t)), [])
        fx = plant.frame_extras(t)
        if extras is not None:
            extras.append(fx)
        k += 1


def _ticks(plant: PL.Plant):
    with open(plant.run_dir / "realised_state.jsonl", encoding="utf-8") as f:
        header, ticks = parse_realised(f)
        return header, list(ticks)


def _p9(tk):
    return next(p for p in tk.presses if p.id == "P9a")


def _log(plant: PL.Plant) -> list[dict]:
    import pyarrow.parquet as pq

    p = plant.run_dir / "log.parquet"
    assert validate_table(p, LogRow) > 0
    return pq.read_table(p).to_pylist()


# ------------------------------------------------------------------------------------------------ trip + reset


def test_robot_entering_a_running_press_trips_it_and_the_reset_starts_after_exit(tmp_path):
    pl = _plant(tmp_path, _script([_run()]))
    _drive(pl, lambda t: IN if 10.0 <= t < 20.0 else OUT, 120.0)
    summ = pl.finish()
    s = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert s.model_dump(mode="json") == summ
    assert s.robot_caused_press_stops == 1 and s.holds == 0 and s.incidents == 0 and s.status == "ok"
    reset = PL.reset_draw(pl.safety.presses[0], 5, 9, f"t0={T0:.1f}:trip:0")
    fz = [f for f in pl.engine.freezes() if f.press == "P9a"]
    tripped = [f for f in fz if f.state == "tripped"]
    resetting = [f for f in fz if f.state == "resetting"]
    assert min(f.t0 for f in tripped) == pytest.approx(T0 + 10.0, abs=0.11)
    t_exit = max(f.t1 for f in tripped)
    assert t_exit == pytest.approx(T0 + 20.1, abs=0.11)
    assert len(resetting) == 1 and resetting[0].t0 == pytest.approx(t_exit) and \
        resetting[0].t1 - resetting[0].t0 == pytest.approx(reset)
    _, ts = _ticks(pl)
    for tk in ts:
        p = _p9(tk)
        rel = tk.t - T0
        if 10.5 <= rel < 20.0:
            assert p.state == "tripped" and p.freeze and not p.stroke
        elif 20.5 <= rel < 20.0 + reset:
            assert p.state == "resetting" and p.freeze and not p.stroke
        elif rel < 9.5:
            assert p.state == "running" and not p.freeze
    rows = _log(pl)
    trip = [r for r in rows if r["kind"] == "press_trip"]
    assert len(trip) == 1 and trip[0]["entity"] == "P9a" and trip[0]["zone"] == "z9"
    assert trip[0]["value"] == pytest.approx(t_exit - (T0 + 10.0) + reset, abs=0.11)
    assert json.loads(trip[0]["detail"])["guard"] == "light_curtain"
    assert any(r["kind"] == "press_restart" for r in rows)
    assert s.press_time_lost_s == pytest.approx(min(t_exit + reset, T0 + 600.0) - (T0 + 10.0), abs=0.11)
    # the robot-caused stop reaches the observer's stream (zone_motion of P9a, index 0) within a tick + latency
    zm = [(d["t_cap"] - T0, d["zone_motion"][0]) for d in map(json.loads, (pl.run_dir / "obs_v2.jsonl").open())]
    run = [v for t, v in zm if t < 9.5 and v is not None]
    stop = [v for t, v in zm if 10.5 + 0.5 <= t < 20.0 and v is not None]
    # stat_v1 units (twin/camera.py item 11): normalised ROI energy, idle pool while stopped; MOTION_THR is the
    # matching observer threshold (camera_v0's ram-speed units gave max(stop) < 0.3 < max(run))
    from twin.camera import MOTION_THR
    assert np.mean(np.asarray(stop) > MOTION_THR) < 0.35 < np.mean(np.asarray(run) > MOTION_THR)


def test_trip_while_the_robot_stays_inside_keeps_the_press_tripped(tmp_path):
    pl = _plant(tmp_path, _script([_run()]), seconds=60.0)
    _drive(pl, lambda t: IN if t >= 10.0 else OUT, 60.0)
    pl.finish()
    _, ts = _ticks(pl)
    assert all(_p9(tk).state == "tripped" for tk in ts if tk.t - T0 >= 10.5)
    fz = [f for f in pl.engine.freezes() if f.press == "P9a" and f.state == "resetting"]
    assert len(fz) == 1 and fz[0].t0 == pytest.approx(T0 + 60.1, abs=0.11)   # reset only once the mission ends


def test_reset_draws_follow_the_u4_lognormal_and_are_common_random_numbers():
    sfty = load(Safety, SAFETY_PATH)
    ps = sfty.presses[0]
    assert ps.guard.type == "ASSUMED" and ps.guard.device == "light_curtain" and ps.guard.reset == "manual"
    d = np.array([PL.reset_draw(ps, 7, 9, f"k{i}") for i in range(6000)])
    assert np.median(d) == pytest.approx(240.0, rel=0.05)
    assert np.quantile(d, 0.9) == pytest.approx(900.0, rel=0.08)
    assert (d > 0).all()
    assert PL.reset_draw(ps, 7, 9, "k1") == d[1]                    # keyed: the same key gives the same reset
    assert PL.reset_draw(sfty.presses[1], 7, 9, "k1") != d[1]       # another press, another stream
    assert PL.reset_draw(ps, 8, 9, "k1") != d[1]                    # another world


def test_paired_conditions_share_the_reset(tmp_path):
    a = _plant(tmp_path, _script([_run()]), seconds=40.0, name="a", condition="learned")
    b = _plant(tmp_path, _script([_run()]), seconds=40.0, name="b", condition="reactive")
    _drive(a, lambda t: IN if 5.0 <= t < 8.0 else OUT, 40.0)
    _drive(b, lambda t: IN if 15.0 <= t < 30.0 else OUT, 40.0)
    a.finish()
    b.finish()
    ra = [f.t1 - f.t0 for f in a.engine.freezes() if f.state == "resetting"]
    rb = [f.t1 - f.t0 for f in b.engine.freezes() if f.state == "resetting"]
    assert ra == pytest.approx(rb) and len(ra) == 1


# ------------------------------------------------------------------------------------------------ hold


def test_restart_due_while_the_robot_is_inside_is_held_then_needs_a_manual_reset_after_exit(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0)]), seconds=60.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    pl.finish()
    s = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert s.robot_caused_press_stops == 0 and s.holds == 1 and s.incidents == 0
    held = [f for f in pl.engine.freezes() if f.state == "held"]
    assert min(f.t0 for f in held) == pytest.approx(T0 + 30.0, abs=1e-6)   # held from the moment the restart is due
    t_exit = max(f.t1 for f in held)
    assert t_exit == pytest.approx(T0 + 40.1, abs=0.11)
    # light curtain + manual reset (W-W4): the held restart needs a manual reset from outside once the robot has left
    reset = PL.reset_draw(pl.safety.presses[0], 5, 9, f"t0={T0:.1f}:entry_reset:0")
    rs = [f for f in pl.engine.freezes() if f.state == "resetting"]
    assert len(rs) == 1 and rs[0].t0 == pytest.approx(t_exit) and rs[0].t1 - rs[0].t0 == pytest.approx(reset)
    _, ts = _ticks(pl)
    for tk in ts:
        rel, p = tk.t - T0, _p9(tk)
        if rel < 30.0:
            assert p.state == "stopped" and not p.freeze
        elif rel < 40.0:
            assert p.state == "held" and p.freeze and not p.stroke
        elif 40.5 <= rel < t_exit - T0 + reset:
            assert p.state == "resetting" and p.freeze and not p.stroke
    rows = _log(pl)
    hold = [r for r in rows if r["kind"] == "press_hold"]
    assert len(hold) == 1 and hold[0]["value"] == pytest.approx(10.1, abs=0.11)
    assert "restart delayed" in json.loads(hold[0]["detail"])["cause"]
    rst = [r for r in rows if r["kind"] == "press_restart"]
    assert len(rst) == 1 and rst[0]["value"] == pytest.approx(reset, abs=1e-3)
    det = json.loads(rst[0]["detail"])
    assert "curtain break" in det["after"] and det["held_restart"] is True and det["entry_state"] == "stopped"
    assert s.press_time_lost_s == pytest.approx(min(t_exit + reset, T0 + 600.0) - (T0 + 30.0), abs=0.11)


def test_an_auto_reset_guard_restarts_when_the_robot_leaves(tmp_path):
    d = yaml.safe_load(SAFETY_PATH.read_text(encoding="utf-8"))
    for ps in d["presses"]:
        ps["guard"] = {"type": "OBSERVED", "device": "interlocked_gate", "reset": "auto", "evidence_clips": ["c9_x"]}
    sp = tmp_path / "safety_auto.yaml"
    sp.write_text(yaml.safe_dump(d), encoding="utf-8")
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0)]), seconds=60.0, safety=sp)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    s = pl.finish()
    assert s["holds"] == 1 and s["press_time_lost_s"] == pytest.approx(10.1, abs=0.11)
    assert [f for f in pl.engine.freezes() if f.state == "resetting"] == []


def test_each_delayed_restart_is_its_own_hold_valued_at_its_held_time(tmp_path):
    """Audit probe: stop 0-30, run 30-50, stop 50-80, run 80+, robot inside 10-90: two holds (20 s and 10.1 s), no
    stopped time counted in a hold."""
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0, t1=50.0), _stop(t0=50.0, t1=80.0),
                                   _run(t0=80.0)]), seconds=120.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 90.0 else OUT, 120.0)
    s = pl.finish()
    rows = [r for r in _log(pl) if r["kind"] == "press_hold"]
    assert s["holds"] == 2 and len(rows) == 2
    assert [r["value"] for r in rows] == pytest.approx([20.0, 10.1], abs=0.11)
    held = sum(f.t1 - f.t0 for f in pl.engine.freezes() if f.state == "held")
    assert sum(r["value"] for r in rows) == pytest.approx(held, abs=1e-3)
    assert all(f.t1 <= T0 + 50.0 + 1e-9 or f.t0 >= T0 + 80.0 - 1e-9 for f in pl.engine.freezes()
               if f.state == "held")


def test_entering_a_stopped_press_latches_the_curtain_and_needs_a_manual_reset(tmp_path):
    """W-W4 / U4: a light-curtain break latches the restart interlock on ANY entry. Stopped 0-100 s, robot inside
    10-40 s: a manual reset from the exit (40 s); it outlasts the stop, so the restart at 100 s is delayed (a hold,
    lost time > 0). With the stop lasting to 300 s the same reset ends before the restart: no hold, no lost time."""
    pl = _plant(tmp_path, _script([_stop(t1=100.0), _run(t0=100.0)]), seconds=60.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    s = pl.finish()
    reset = PL.reset_draw(pl.safety.presses[0], 5, 9, f"t0={T0:.1f}:entry_reset:0")
    assert 60.0 < reset < 260.0                                     # this world's draw outlasts the 100 s stop only
    rs = [f for f in pl.engine.freezes() if f.press == "P9a"]
    assert len(rs) == 1 and rs[0].state == "resetting" and rs[0].t0 == pytest.approx(T0 + 40.0, abs=0.11)
    assert rs[0].t1 - rs[0].t0 == pytest.approx(reset)
    assert (s["robot_caused_press_stops"], s["holds"], s["incidents"]) == (0, 1, 0)
    assert s["press_time_lost_s"] > 0
    assert s["press_time_lost_s"] == pytest.approx(rs[0].t1 - (T0 + 100.0), abs=1e-3)
    rows = _log(pl)
    hold = [r for r in rows if r["kind"] == "press_hold"]
    assert len(hold) == 1 and hold[0]["value"] == pytest.approx(s["press_time_lost_s"], abs=1e-3)
    assert "curtain break" in json.loads(hold[0]["detail"])["cause"]
    rst = [r for r in rows if r["kind"] == "press_restart"]
    assert len(rst) == 1 and rst[0]["value"] == pytest.approx(reset, abs=1e-3)
    assert json.loads(rst[0]["detail"])["delays_restart"] is True
    long = _plant(tmp_path, _script([_stop(t1=300.0), _run(t0=300.0)]), seconds=60.0, name="long")
    _drive(long, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    s2 = long.finish()
    assert [f.state for f in long.engine.freezes()] == ["resetting"]
    assert (s2["holds"], s2["press_time_lost_s"]) == (0, 0.0)
    rst2 = [r for r in _log(long) if r["kind"] == "press_restart"]
    assert len(rst2) == 1 and json.loads(rst2[0]["detail"])["delays_restart"] is False


def test_no_cost_cliff_at_the_scheduled_restart(tmp_path):
    """Audit repro: stopped 0-30 s, running from 30 s, robot inside from 10 s. Leaving just before the restart (29.5 s)
    and just after it (30.5 s) both give a manual reset, a hold and lost time; the lost times differ by the extra
    second of presence, not by a whole reset."""
    out = {}
    for name, leave in (("before", 29.5), ("after", 30.5)):
        pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0)]), seconds=60.0, name=name)
        _drive(pl, lambda t, leave=leave: IN if 10.0 <= t < leave else OUT, 60.0)
        out[name] = pl.finish()
        rs = [f for f in pl.engine.freezes() if f.state == "resetting"]
        assert len(rs) == 1 and rs[0].t0 == pytest.approx(T0 + leave, abs=0.11)
    reset = PL.reset_draw(load(Safety, SAFETY_PATH).presses[0], 5, 9, f"t0={T0:.1f}:entry_reset:0")
    a, b = out["before"], out["after"]
    assert a["holds"] == b["holds"] == 1
    assert a["press_time_lost_s"] == pytest.approx(reset - 0.5, abs=0.11)
    assert b["press_time_lost_s"] == pytest.approx(reset + 0.5, abs=0.11)
    assert b["press_time_lost_s"] - a["press_time_lost_s"] == pytest.approx(1.0, abs=0.25)


def test_an_allowed_entry_into_a_running_press_is_an_incident(tmp_path):
    d = yaml.safe_load(SAFETY_PATH.read_text(encoding="utf-8"))
    for ps in d["presses"]:
        ps["entry_rules"] = [dict(r, action="allow") if r["press_state"] == "running" else r for r in ps["entry_rules"]]
    sp = tmp_path / "safety_allow.yaml"
    sp.write_text(yaml.safe_dump(d), encoding="utf-8")
    pl = _plant(tmp_path, _script([_run()]), seconds=30.0, safety=sp)
    _drive(pl, lambda t: IN if 10.0 <= t < 20.0 else OUT, 30.0)
    s = pl.finish()
    assert s["incidents"] == 1 and s["robot_caused_press_stops"] == 0 and s["holds"] == 0
    assert [r["kind"] for r in _log(pl)].count("incident") == 1


def test_a_safety_file_missing_a_press_fails_closed(tmp_path):
    d = yaml.safe_load(SAFETY_PATH.read_text(encoding="utf-8"))
    d["presses"] = d["presses"][:1]
    sp = tmp_path / "safety_one.yaml"
    sp.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError, match="no entry for presses"):
        _plant(tmp_path, _script([_run()]), safety=sp)


def test_trip_and_hold_rows_name_the_checkpoint(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=20.0), _run(t0=20.0)]), seconds=120.0, tasks=str(_tasks(tmp_path)),
                reset_override_s=60.0)
    pl.attach(caster=_OpenCaster())

    def where(t):                       # hold 20-25.1, manual reset 25.1-85.1, then a trip on entry at 90
        if t == pytest.approx(5.0):
            pl.on_ros(t, [{"event": "dwell_start", "target": "cp1", "inside": True}])
        return IN if 10.0 <= t < 25.0 or 90.0 <= t < 92.0 else OUT
    _drive(pl, where, 120.0)
    pl.finish()
    rows = {r["kind"]: r for r in _log(pl) if r["kind"] in ("press_trip", "press_hold", "press_restart")}
    assert set(rows) == {"press_trip", "press_hold", "press_restart"}
    assert all(r["checkpoint"] == "cp1" for r in rows.values())


# ------------------------------------------------------------------------------------------------ fail closed


def test_an_off_zone_dwell_start_is_an_error_and_summary_json_is_still_written(tmp_path):
    """Audit repro: zone z9, dwell_start for cp1, cp2 and cp3 (cp3 is a z14 checkpoint)."""
    pl = _plant(tmp_path, _script([_run()]), seconds=30.0, tasks=str(_tasks(tmp_path)))
    pl.attach(caster=_OpenCaster())

    def where(t):
        if t == pytest.approx(2.0):
            pl.on_ros(t, [{"event": "dwell_start", "target": c, "inside": False} for c in ("cp1", "cp2", "cp3")])
            pl.on_ros(t, [{"event": "decline", "target": "cpX", "reason": "test"}])
        return OUT
    _drive(pl, where, 30.0)
    s = pl.finish()
    disk = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert disk.model_dump(mode="json") == s
    assert s["status"] == "error" and "cp3" in s["status_reason"]
    assert (s["checkpoints_planned"], s["attempted"], s["declined"]) == (2, 2, 0)
    errs = [r for r in _log(pl) if r["kind"] == "error"]
    assert {r["checkpoint"] for r in errs} == {"cp3", "cpX"}
    assert (pl.run_dir / "realised_state.jsonl").exists() and (pl.run_dir / "plant_summary.json").exists()


def test_an_exception_inside_finish_still_writes_an_error_summary_and_the_log(tmp_path, monkeypatch):
    pl = _plant(tmp_path, _script([_run()]), seconds=20.0)
    _drive(pl, lambda t: IN if 5.0 <= t < 8.0 else OUT, 20.0)

    def boom(rows, run_dir):
        raise OSError("disk full (test)")
    monkeypatch.setattr(PL, "write_log", boom)
    s = pl.finish()
    disk = json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert disk == s
    Summary.model_validate(disk)
    assert s["status"] == "error" and "disk full (test)" in s["status_reason"]
    assert s["robot_caused_press_stops"] == 1                       # what happened is kept
    rows = [json.loads(ln) for ln in (pl.run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "mission_start" and "press_trip" in kinds and "error" in kinds
    assert kinds.count("mission_end") == 1
    for r in rows:
        LogRow.model_validate(r)
    assert (pl.run_dir / "realised_state.jsonl").exists()
    assert pl.finish() is s                                         # idempotent


def test_an_exception_inside_a_mission_hook_still_writes_an_error_summary(tmp_path, monkeypatch):
    """A plant hook that raises during the mission (here the camera model) must not leave the runner without
    summary.json / log.parquet / plant_summary.json: the plant records the error, stops its work and finish() writes
    status "error"."""
    pl = _plant(tmp_path, _script([_run()]), seconds=30.0)
    calls = {"n": 0}
    real = pl.camera.observe

    def flaky(tk):
        calls["n"] += 1
        if calls["n"] == 10:
            raise RuntimeError("camera model failed (test)")
        return real(tk)
    monkeypatch.setattr(pl.camera, "observe", flaky)
    _drive(pl, lambda t: OUT, 30.0)                                 # the runner loop never sees the exception
    pl.on_ros(20.0, [{"event": "dwell_start", "target": "cp1", "inside": False}])
    assert pl._broken and calls["n"] == 10                          # no plant work after the failure
    s = pl.finish()
    disk = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert disk.model_dump(mode="json") == s
    assert s["status"] == "error" and "after_step" in s["status_reason"] and "camera model failed" in s["status_reason"]
    assert s["attempted"] == 0
    kinds = [r["kind"] for r in _log(pl)]
    assert kinds[0] == "mission_start" and "error" in kinds and kinds[-1] == "mission_end"
    assert (pl.run_dir / "plant_summary.json").exists()

    def boom(*a, **k):
        raise RuntimeError("hook failed (test)")
    for hook, target, call in (("poses", "_mover_states", lambda q: q.poses(1.0)),
                               ("frame_extras", "_frame_extras", lambda q: q.frame_extras(1.0)),
                               ("on_ros", "_on_ros", lambda q: q.on_ros(1.0, [{"event": "decline", "target": "x"}]))):
        q = _plant(tmp_path, _script([_run()]), seconds=30.0, name=f"h_{hook}")
        monkeypatch.setattr(q, target, boom)
        out = call(q)
        assert q._broken and out in ({}, None)
        r = q.finish()
        assert r["status"] == "error" and f"plant {hook} failed" in r["status_reason"]
        assert (q.run_dir / "summary.json").exists() and (q.run_dir / "log.parquet").exists()


def test_a_mission_length_other_than_t_is_refused_unless_allowed_on_dev(tmp_path):
    with pytest.raises(ValueError, match="fixed mission length"):
        _plant(tmp_path, _script([_run()]), seconds=300.0, name="m1", allow_short_mission=None)
    with pytest.raises(ValueError, match="test world"):
        _plant(tmp_path, _script([_run()]), seconds=300.0, name="m2", world_split="test")
    full = _plant(tmp_path, _script([_run(t1=800.0)], duration=800.0), seconds=720.0, name="m3",
                  allow_short_mission=None)
    _drive(full, lambda t: OUT, 720.0, dt=0.5)
    s = full.finish()
    assert s["status"] == "ok" and s["T_s"] == 720.0 and s["variant"] == "primary" and s["reset_override_s"] is None
    assert require_primary(Summary.model_validate(s)).run_id == "m3"
    short = _plant(tmp_path, _script([_run()]), seconds=30.0, name="m4")
    _drive(short, lambda t: OUT, 30.0)
    s4 = short.finish()
    assert s4["variant"] == "short_dev" and s4["T_s"] == 720.0
    with pytest.raises(ValueError, match="not a primary run"):
        require_primary(Summary.model_validate(s4))


def test_eval_identity_fields_have_no_defaults(tmp_path):
    for k in ("world", "world_split", "condition", "code"):
        cfg = _cfg(tmp_path, _script([_run()]), name=f"id_{k}")
        del cfg["plant"][k]
        with pytest.raises(ValueError, match=k):
            PL.make(cfg)
    for k, bad in (("world", -1), ("world", "3"), ("world_split", "train"), ("condition", "best"),
                   ("code", {"head": ""})):
        cfg = _cfg(tmp_path, _script([_run()]), name=f"bad_{k}")
        cfg["plant"][k] = bad
        with pytest.raises(ValueError, match=k):
            PL.make(cfg)


def test_tasks_must_match_the_site(tmp_path):
    with pytest.raises(ValueError, match="tasks.yaml site"):
        _plant(tmp_path, _script([_run()]), tasks=str(REPO / "tests" / "fixtures" / "twin_tasks.yaml"))
    d = yaml.safe_load(_tasks(tmp_path).read_text(encoding="utf-8"))
    d["checkpoints"][0]["press"] = "P99"
    bad = tmp_path / "tasks_bad_press.yaml"
    bad.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown press"):
        _plant(tmp_path, _script([_run()]), tasks=str(bad))
    d["checkpoints"][0]["press"] = "P14a"                          # a z14 press on a z9 checkpoint
    bad.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError, match="is in zone z14"):
        _plant(tmp_path, _script([_run()]), tasks=str(bad))


def test_reset_override_runs_the_u4_sweep_and_is_tagged(tmp_path):
    pl = _plant(tmp_path, _script([_run()]), seconds=40.0, reset_override_s=60.0)
    _drive(pl, lambda t: IN if 5.0 <= t < 8.0 else OUT, 40.0)
    pl.finish()
    rs = [f for f in pl.engine.freezes() if f.state == "resetting"]
    assert len(rs) == 1 and rs[0].t1 - rs[0].t0 == pytest.approx(60.0)
    diag = json.loads((pl.run_dir / "plant_summary.json").read_text(encoding="utf-8"))
    assert diag["reset_sweep"]["sweep_run"] is True and diag["reset_sweep"]["reset_override_s"] == 60.0
    rows = _log(pl)
    assert json.loads(rows[0]["detail"])["reset_override_s"] == 60.0
    trip = next(r for r in rows if r["kind"] == "press_trip")
    assert json.loads(trip["detail"])["reset_source"] == "sweep_override"
    # summary.json itself says it is not a primary run (an EVAL aggregator reading only summary.json refuses it)
    summ = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert summ.variant == "short_dev" and summ.reset_override_s == 60.0       # a short test mission carrying it
    full = _plant(tmp_path, _script([_run(t1=800.0)], duration=800.0), seconds=720.0, name="rf", reset_override_s=60.0,
                  allow_short_mission=None)
    _drive(full, lambda t: OUT, 720.0, dt=0.5)
    sf = Summary.model_validate(full.finish())
    assert sf.variant == "reset_sweep" and sf.reset_override_s == 60.0 and sf.status == "ok"
    with pytest.raises(ValueError, match="not a primary run"):
        require_primary(sf)
    for bad in ({"variant": "primary", "reset_override_s": 60.0}, {"variant": "reset_sweep", "reset_override_s": None}):
        with pytest.raises(ValueError, match="reset_override_s"):
            Summary.model_validate({**sf.model_dump(mode="json"), **bad})
    with pytest.raises(ValueError, match="outside press"):
        _plant(tmp_path, _script([_run()]), name="r2", reset_override_s=30.0)    # below the 60-900 s sweep
    plain = _plant(tmp_path, _script([_run()]), name="r3", seconds=10.0)
    _drive(plain, lambda t: OUT, 10.0)
    plain.finish()
    assert json.loads((plain.run_dir / "plant_summary.json").read_text(encoding="utf-8"))["reset_sweep"]["sweep_run"] \
        is False


# ------------------------------------------------------------------------------------------------ obs v2 adapter


def test_zone_vis_drops_below_one_with_a_person_on_the_press_rect(tmp_path):
    seg = Segment(id="s0", cls="person", source="synthetic_stub", points=[(0.0, 6.0, 5.5), (60.0, 6.5, 5.5)])
    intents = [MoverIntent(entity="person:0", cls="person", zone="z9", t_start=0.0, segment="s0", speed_scale=1.0)]
    pl = _plant(tmp_path, _script([_run()], intents=intents, segments=[seg]), seconds=20.0)
    _drive(pl, lambda t: OUT, 20.0)
    pl.finish()
    obs = [OM.obs_from_dict(json.loads(ln)) for ln in (pl.run_dir / "obs_v2.jsonl").open(encoding="utf-8")]
    v2 = [(np.asarray(o.zone_vis), np.asarray(o.zone_motion)) for o in obs]
    vis = np.array([v[0] for v, _ in v2])
    zm = np.array([m[0] for _, m in v2])
    assert (vis < 1.0).mean() > 0.8                                # the person hides part of P9a's zone floor
    assert np.isnan(zm[vis < 1.0]).all() and np.isnan(zm).all()   # MISSING (detections held across misses, T-W6)
    assert all(v[1] == 1.0 for v, _ in v2)                         # P14a's zone is untouched


def test_floor_share_counts_the_unseen_part_of_a_zone():
    floors = [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0),
                                                                     (10.0, 10.0)]]
    w = PL.floor_share([(2.0, 2.0, 4.0, 4.0), (9.0, 2.0, 11.0, 4.0), (-4.0, 2.0, 0.0, 4.0)], floors, margin=1.0)
    assert w[0] == pytest.approx([1.0, 0.0])
    assert w[1] == pytest.approx([0.5, 0.5])
    assert w[2].sum() == pytest.approx(1.0 / 6.0) and w[2][1] == 0.0   # 1 m of the 6 m wide zone is on the floor


def test_obs_lines_are_written_only_once_their_delivery_time_is_reached(tmp_path):
    pl = _plant(tmp_path, _script([_run()]), seconds=20.0)
    k = 0
    while k < 200:
        t = round((k + 1) * 0.1, 6)
        pl.after_step(t, _st(*OUT), [])
        pl._obs_f.flush()
        lines = (pl.run_dir / "obs_v2.jsonl").read_text(encoding="utf-8").splitlines()
        assert all(json.loads(ln)["t"] <= T0 + t + 1e-9 for ln in lines)
        assert len(lines) == int((t - OM.LATENCY_S) * 2 + 1e-9) + 1 if t >= OM.LATENCY_S else not lines
        k += 1
    pl.finish()
    assert len((pl.run_dir / "obs_v2.jsonl").read_text(encoding="utf-8").splitlines()) == 40


# ------------------------------------------------------------------------------------------------ world interface


def test_world_interface_follows_the_freezes(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0, t1=200.0), _stop(t0=200.0)]), seconds=60.0)
    w = pl.world
    assert not w.active("P9a", T0 + 10) and w.active("P9a", T0 + 35)
    assert w.true_idle_windows("P9a", T0 + 10, 600) == [(T0 + 10, T0 + 30), (T0 + 200, T0 + 600)]
    assert w.true_idle_windows("P9a", T0 + 10, 60) == [(T0 + 10, T0 + 30)]
    assert w.idle_remaining("P9a", T0 + 10) == pytest.approx(20.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    reset = PL.reset_draw(pl.safety.presses[0], 5, 9, f"t0={T0:.1f}:entry_reset:0")
    t_go = T0 + 40.1 + reset                                       # held, then the manual reset after exit
    assert not w.active("P9a", T0 + 35) and not w.active("P9a", T0 + 45) and w.active("P9a", t_go + 1.0)
    assert w.group("P9a").is_frozen(T0 + 35) and w.group("P9a").frozen_until(T0 + 35) == pytest.approx(t_go, abs=0.11)
    assert {(f.kind, f.state) for f in w.group("P9a").freezes} == {("hold", "held"), ("trip", "resetting")}
    assert w.group("P14a").freezes == []
    assert list(w.active_many("P9a", [T0 + 10, T0 + 35, t_go + 1.0])) == [False, False, True]
    ev = w.trip("P14a", T0 + 70.0, "test", reset_s=30.0)          # after the written horizon (t0 + 59.5)
    assert ev["group"] == "P14a" and not w.active("P14a", T0 + 80) and w.active("P14a", T0 + 101)
    s = pl.finish()
    # a trip through the World interface is a logged, counted robot-caused stop
    assert s["robot_caused_press_stops"] == 1
    trip = [r for r in _log(pl) if r["kind"] == "press_trip"]
    assert len(trip) == 1 and trip[0]["entity"] == "P14a" and trip[0]["value"] == pytest.approx(30.1)
    assert json.loads(trip[0]["detail"])["via"] == "TwinWorld.trip"


def test_world_trip_without_a_plant_and_in_the_past_fail_closed(tmp_path):
    pl = _plant(tmp_path, _script([_run()]), seconds=20.0)
    with pytest.raises(RuntimeError, match="needs the plant"):
        PL.TwinWorld(pl.engine).trip("P9a", T0 + 5.0, "test", reset_s=10.0)
    _drive(pl, lambda t: OUT, 10.0)
    with pytest.raises(ValueError, match="written tick"):
        pl.world.trip("P9a", T0 + 5.0, "test", reset_s=10.0)
    ev = pl.world.trip("P9a", T0 + 12.0, "test")                   # keyed draw, the press's first trip
    want = PL.SAFETY_DT + PL.reset_draw(pl.safety.presses[0], 5, 9, f"t0={T0:.1f}:trip:0")
    assert ev["lost_s"] == pytest.approx(want)
    s = pl.finish()
    assert s["robot_caused_press_stops"] == 1 and s["status"] == "error"   # mission cut at 10 s of 20 s


# ------------------------------------------------------------------------------------------------ truth isolation


class _OpenCaster:
    """Static caster with nothing in the way (the runner passes the real one)."""
    name = "open"

    def cast(self, origin, dirs, rmax):
        return np.full(len(dirs), rmax)


def test_plant_output_reaches_the_observer_side_only_through_files(tmp_path):
    seg = Segment(id="s0", cls="person", source="synthetic_stub", points=[(0.0, 2.0, 2.0), (60.0, 12.0, 2.0)])
    intents = [MoverIntent(entity="person:0", cls="person", zone="z9", t_start=0.0, segment="s0", speed_scale=1.0)]
    pl = _plant(tmp_path, _script([_run()], intents=intents, segments=[seg]), seconds=40.0)
    pl.attach(caster=_OpenCaster())
    assert pl.spec() == [{"id": "person:0", "cls": "person"}]
    extras: list[dict] = []
    _drive(pl, lambda t: (6.25, 2.5), 40.0, extras=extras)
    assert not (pl.run_dir / "realised_state.jsonl").exists()       # truth is published on finish only
    pl.finish()
    # ROS gets the robot camera's gate observations and inspection outcomes, nothing else
    keys = {k for fx in extras for k in fx}
    assert keys == {"camera_obs"}
    obs = [o for fx in extras for o in fx.get("camera_obs", [])]
    assert obs and all(set(o) == {"stamp", "source", "targets"} for o in obs)
    assert all(set(tg) <= {"id", "visible", "motion_seen", "vis_frac"} for o in obs for tg in o["targets"])
    assert any(tg["motion_seen"] for o in obs for tg in o["targets"])        # the press-ram mover moves
    assert obs[0]["source"] == RC.GATE_SOURCE
    # the observer side's only input: obs_v2.jsonl, Observation v2 lines, the robot as class OTHER (never PERSON)
    lines = (pl.run_dir / "obs_v2.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 80
    fields = {"v", "t", "t_cap"} | set(OM.OBS_V1_FIELDS[2:]) | set(OM.OBS_V2_FIELDS)
    robot_seen = False
    for ln in lines:
        d = json.loads(ln)
        assert set(d) == fields and d["v"] == 2
        o = OM.obs_from_dict(d)
        OM.check_obs_v2(o, n_zone=2, n_cam=2)
        robot_seen |= bool((o.cls == PL.OBS_OTHER).any())
        assert not (np.hypot(o.x - 6.25, o.z - 2.5)[o.cls == OM.PERSON] < 1e-9).any()
    assert robot_seen
    assert not (pl.run_dir / "plant_state.json").exists()          # no body-shop feed
    # the realised stream carries the robot, and only the engine opened truth (read the script, wrote the stream)
    _, ts = _ticks(pl)
    assert len(ts) == 80 and all(any(m.cls == "robot" for m in tk.movers) for tk in ts)
    log = (tmp_path / "truth_access.log").read_text(encoding="utf-8").splitlines()[1:]
    callers = {(ln.split("\t")[1], ln.split("\t")[2]) for ln in log}
    assert callers == {("twin/engine.py", "r"), ("twin/engine.py", "w")}


def _code_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {a.name for a in n.names} | ({n.module} if isinstance(n, ast.ImportFrom) and n.module else set())
    return out


def test_twin_plant_has_no_body_shop_world():
    names = _code_names(REPO / "twin" / "plant.py")
    for banned in ("CROSS_X", "Kinematics", "truth_agents", "WALK", "TRANSFER_S", "intercycle.sim.world",
                   "intercycle.truth.kinematics", "intercycle.truth.agents", "intercycle_eval", "intercycle_sim"):
        assert banned not in names, banned
    src = (REPO / "twin" / "plant.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "plant_state.json" not in lits and "WALK" not in lits


# ------------------------------------------------------------------------------------------------ body shop


BODY_SHOP_PLANT_SHA256 = "ac065a0dda7f0917061574c9b9f222fcb48e7d823154c3d29d67cae6b3aa4e5e"   # e06ac35, LF


def test_body_shop_plant_unchanged():
    b = (REPO / "ros2" / "intercycle_eval" / "intercycle_eval" / "plant.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(b).hexdigest() == BODY_SHOP_PLANT_SHA256


def test_site_and_safety_files_validate():
    site = load(Site, SITE_PATH)
    sfty = load(Safety, SAFETY_PATH)
    assert sfty.site_id == site.site_id
    assert {p.press for p in sfty.presses} == {p.id for p in site.presses}
    assert all(p.guard.type == "ASSUMED" for p in sfty.presses)
