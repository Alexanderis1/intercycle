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
from twin.schemas.runs import Summary, parse_realised
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


def _plant(tmp_path, script: DayScript, seconds=120.0, safety=SAFETY_PATH, name="r1", **plant) -> PL.Plant:
    sp = tmp_path / f"{name}.day_script.json"
    sp.write_text(json.dumps(script.model_dump(mode="json")), encoding="utf-8")
    cfg = {"seconds": seconds, "sim": "fake",
           "plant": {"site": str(SITE_PATH), "safety": str(safety), "day_script": str(sp), "run_id": name,
                     "zone": "z9", "world": 0, "world_split": "dev", "condition": "learned", "plant_t0": T0,
                     "run_dir": str(tmp_path / "runs" / name), **plant}}
    return PL.make(cfg)


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
    assert max(stop) < 0.3 < max(run)


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


def test_restart_due_while_the_robot_is_inside_is_held_and_logged(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0)]), seconds=60.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    pl.finish()
    s = Summary.model_validate(json.loads((pl.run_dir / "summary.json").read_text(encoding="utf-8")))
    assert s.robot_caused_press_stops == 0 and s.holds == 1 and s.incidents == 0
    held = [f for f in pl.engine.freezes() if f.state == "held"]
    assert min(f.t0 for f in held) == pytest.approx(T0 + 30.0, abs=1e-6)   # held from the moment the restart is due
    assert max(f.t1 for f in held) == pytest.approx(T0 + 40.1, abs=0.11)
    _, ts = _ticks(pl)
    for tk in ts:
        rel, p = tk.t - T0, _p9(tk)
        if rel < 30.0:
            assert p.state == "stopped" and not p.freeze
        elif rel < 40.0:
            assert p.state == "held" and p.freeze and not p.stroke
        elif rel >= 40.5:
            assert p.state == "running" and not p.freeze
    rows = [r for r in _log(pl) if r["kind"] == "press_hold"]
    assert len(rows) == 1 and rows[0]["value"] == pytest.approx(10.1, abs=0.11)
    assert "restart delayed" in json.loads(rows[0]["detail"])["cause"]
    assert s.press_time_lost_s == pytest.approx(10.1, abs=0.11)


def test_entering_and_leaving_a_stopped_press_costs_nothing(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=100.0), _run(t0=100.0)]), seconds=60.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    s = pl.finish()
    assert (s["robot_caused_press_stops"], s["holds"], s["incidents"], s["press_time_lost_s"]) == (0, 0, 0, 0.0)
    assert pl.engine.freezes() == []


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


# ------------------------------------------------------------------------------------------------ world interface


def test_world_interface_follows_the_freezes(tmp_path):
    pl = _plant(tmp_path, _script([_stop(t1=30.0), _run(t0=30.0, t1=200.0), _stop(t0=200.0)]), seconds=60.0)
    w = pl.world
    assert not w.active("P9a", T0 + 10) and w.active("P9a", T0 + 35)
    assert w.true_idle_windows("P9a", T0 + 10, 600) == [(T0 + 10, T0 + 30), (T0 + 200, T0 + 600)]
    assert w.true_idle_windows("P9a", T0 + 10, 60) == [(T0 + 10, T0 + 30)]
    assert w.idle_remaining("P9a", T0 + 10) == pytest.approx(20.0)
    _drive(pl, lambda t: IN if 10.0 <= t < 40.0 else OUT, 60.0)
    assert not w.active("P9a", T0 + 35) and w.active("P9a", T0 + 45)
    assert w.group("P9a").is_frozen(T0 + 35) and w.group("P9a").frozen_until(T0 + 35) == pytest.approx(T0 + 40.1, abs=0.11)
    assert {f.kind for f in w.group("P9a").freezes} == {"hold"} and w.group("P14a").freezes == []
    assert list(w.active_many("P9a", [T0 + 10, T0 + 35, T0 + 45])) == [False, False, True]
    ev = w.trip("P14a", T0 + 70.0, "test", reset_s=30.0)          # after the written horizon (t0 + 59.5)
    assert ev["group"] == "P14a" and not w.active("P14a", T0 + 80) and w.active("P14a", T0 + 101)
    pl.finish()


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
