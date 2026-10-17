"""Twin contracts (PLAN_TWIN §2): JSON export frozen, golden fixtures and v0 stubs validate, keyed RNG order
independence (W-W1), the MEASURED_SHIM camera adapter and the runtime truth guard."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from intercycle.observer.mapping import MEASURED_SHIM as _SHIM
from twin import camera_v0, daygen_stub, rng as R
from twin import schemas as S
from twin.schemas import runs as RU
from twin.schemas import tables as TB
from twin.truth_guard import LOG_HEADER, open_truth

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
TWIN = REPO / "twin"
SHIM: dict[str, Any] = dict(_SHIM)


@pytest.fixture(autouse=True)
def _truth_log(tmp_path, monkeypatch):
    """Keep test runs out of the tracked results/truth_access.log."""
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)


def _raw(path: Path):
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix == ".yaml" else json.loads(text)


# ------------------------------------------------------------------------------------------------ JSON export


@pytest.mark.parametrize("name", list(S.MODELS))
def test_json_export_is_current(name):
    p = TWIN / "schemas" / "json" / f"{name}.schema.json"
    assert p.exists(), f"missing {p}; run python -m twin.schemas"
    assert json.loads(p.read_text(encoding="utf-8")) == S.json_schema(name), \
        f"{p.name} differs from the model: run python -m twin.schemas and commit (a contract change after G0 needs the lead)"


def test_table_columns_follow_the_plan():
    assert [c for c, _ in TB.columns(TB.InventoryRow)] == \
        ["clip_id", "camera", "framing_id", "group_id", "class", "duration", "sha256", "split"]
    assert [c for c, _ in TB.columns(TB.TrackRow)] == \
        ["split", "t", "track_id", "class", "u", "v", "x", "y", "conf", "framing"]


# ------------------------------------------------------------------------------------------------ golden fixtures

DOC_FIXTURES = [("twin_site.yaml", S.Site), ("twin_tasks.yaml", S.Tasks), ("twin_safety.yaml", S.Safety),
                ("twin_dynamics.json", S.Dynamics), ("twin_day_script.json", S.DayScript),
                ("twin_summary.json", S.Summary), ("twin_primary_metric.json", S.PrimaryMetric)]
TABLE_FIXTURES = [("twin_inventory.parquet", TB.InventoryRow), ("twin_tracks.parquet", TB.TrackRow),
                  ("twin_machines.parquet", TB.MachineRow), ("twin_log.parquet", TB.LogRow)]


@pytest.mark.parametrize("name,model", DOC_FIXTURES)
def test_doc_fixture_validates(name, model):
    S.load(model, FIX / name)


@pytest.mark.parametrize("name,model", TABLE_FIXTURES)
def test_table_fixture_validates(name, model):
    assert S.validate_table(FIX / name, model) > 0


def test_realised_fixture_validates():
    with open(FIX / "twin_realised_state.jsonl", encoding="utf-8") as f:
        header, ticks = S.parse_realised(f)
        n = sum(1 for _ in ticks)
    assert header.hz == 2.0 and n == 8


def test_realised_rejects_off_grid_tick():
    lines = (FIX / "twin_realised_state.jsonl").read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[3])
    bad["t"] += 0.1
    lines[3] = json.dumps(bad)
    _, ticks = S.parse_realised(lines)
    with pytest.raises(ValueError, match="not 0.5 s after"):
        list(ticks)


# ------------------------------------------------------------------------------------------------ negative cases


def _expect_invalid(model, data, match):
    with pytest.raises(ValidationError, match=match):
        model.model_validate(data)


def test_site_rejects_untagged_value_and_stray_clip():
    d = _raw(FIX / "twin_site.yaml")
    bad = copy.deepcopy(d)
    bad["presses"][0]["dilation_m"] = 0.3                                   # no source tag
    _expect_invalid(S.Site, bad, "dilation_m")
    bad = copy.deepcopy(d)
    bad["zones"][1]["floor_polygon"]["clips"] = ["c14_test_9999"]           # not in cal_sources
    _expect_invalid(S.Site, bad, "cal_sources")
    bad = copy.deepcopy(d)
    bad["zones"] = bad["zones"][:1]                                          # 2 zones
    _expect_invalid(S.Site, bad, "zones")


def test_safety_enforces_u4_for_assumed_guards():
    d = _raw(FIX / "twin_safety.yaml")
    bad = copy.deepcopy(d)
    bad["presses"][0]["reset"]["median_s"] = 120.0
    _expect_invalid(S.Safety, bad, "U4")
    bad = copy.deepcopy(d)
    bad["presses"][1]["guard"]["evidence_clips"] = []
    _expect_invalid(S.Safety, bad, "evidence")
    bad = copy.deepcopy(d)
    bad["presses"][0]["reset"]["sweep_s"] = {"lo": 60.0, "hi": 1800.0}       # U4 sweep is 1-15 min
    _expect_invalid(S.Safety, bad, "sweep")
    for state in ("tripped", "held", "resetting"):
        bad = copy.deepcopy(d)
        bad["presses"][0]["entry_rules"] = [r for r in bad["presses"][0]["entry_rules"] if r["press_state"] != state]
        _expect_invalid(S.Safety, bad, "entry_rules")
        bad["presses"][0]["entry_rules"].append({"press_state": "running", "action": "allow"})   # duplicate state
        _expect_invalid(S.Safety, bad, "duplicate")
    ps = S.Safety.model_validate(d).presses[0].reset
    assert ps.sigma == pytest.approx(math.log(900 / 240) / 1.2815515655446004)


def test_entry_rules_cover_every_realised_press_state():
    from typing import get_args

    from twin.schemas import safety as SF
    assert set(SF.PRESS_STATES) == set(get_args(RU.RealisedPress.model_fields["state"].annotation))


def test_dynamics_rules():
    d = _raw(FIX / "twin_dynamics.json")
    bad = copy.deepcopy(d)
    bad["observed"]["person.speed_mps"]["n"] = 0                             # fitted with n = 0
    _expect_invalid(S.Dynamics, bad, "n > 0")
    bad = copy.deepcopy(d)
    bad["assumed"]["staff.staffing_factor"]["source"] = "fitted"
    bad["assumed"]["staff.staffing_factor"]["n"] = 3
    bad["assumed"]["staff.staffing_factor"]["clips"] = ["c9_train_0001"]
    _expect_invalid(S.Dynamics, bad, "ASSUMED")
    bad = copy.deepcopy(d)
    bad["assumed"]["staff.staffing_factor"]["value"] = 1.5                   # outside its sweep
    _expect_invalid(S.Dynamics, bad, "sweep")
    bad = copy.deepcopy(d)
    del bad["assumed"]["staff.staffing_factor"]["sweep"]
    _expect_invalid(S.Dynamics, bad, "sweep")


def test_primary_metric_rules():
    d = _raw(FIX / "twin_primary_metric.json")
    bad = copy.deepcopy(d)
    bad["missions"][2]["Y"] = 1                                              # tripped mission must score 0
    _expect_invalid(S.PrimaryMetric, bad, "Y = 1")
    bad = copy.deepcopy(d)
    bad["missions"][0]["T_s"] = 600.0
    _expect_invalid(S.PrimaryMetric, bad, "T_s")
    bad = copy.deepcopy(d)
    bad["per_world"][0]["sum_Y"] += 1
    _expect_invalid(S.PrimaryMetric, bad, "per_world")
    bad = copy.deepcopy(d)
    bad["missions"][0]["checkpoints"][0]["outcome"] = "tripped"              # tripped without a press stop
    _expect_invalid(S.PrimaryMetric, bad, "robot_caused_press_stop")
    bad = copy.deepcopy(d)
    bad["per_world"][0]["per_h"] *= 1 + 1e-6                                # outside the 1e-9 tolerance
    _expect_invalid(S.PrimaryMetric, bad, "per_world")
    bad = copy.deepcopy(d)
    bad["per_world"] = bad["per_world"][:-1]                                 # a (world, condition) missing
    _expect_invalid(S.PrimaryMetric, bad, "per_world")
    pm = S.PrimaryMetric.model_validate(d)
    w0 = {(w.world, w.condition): w for w in pm.per_world}
    assert w0[(0, "learned")].per_h == pytest.approx(2 / 1440 * 3600)
    assert w0[(0, "reactive")].sum_Y == 0


def test_primary_metric_accepts_an_equivalent_float_formula():
    """y * 3600 / t and y / t * 3600 differ in the last bits for some (Y, T); both must validate (audit major 3)."""
    differ = [(y, n) for y in range(1, 41) for n in range(1, 9) if y * 3600.0 / (n * 720.0) != y / (n * 720.0) * 3600.0]
    assert differ, "the case the tolerance exists for must occur"
    y, n = differ[0]
    missions = [{"run_id": f"r{k}", "world": 0, "day": 9 + k % 2, "zone": "z9", "condition": "learned", "T_s": 720.0,
                 "Y": (y // n) + (1 if k < y % n else 0), "robot_caused_press_stop": False, "hold": False,
                 "incident": False, "error": False, "press_time_lost_s": 0.0,
                 "checkpoints": [{"checkpoint": f"cp{j}", "outcome": "passed"} for j in range((y // n) + (1 if k < y % n else 0))]
                 or [{"checkpoint": "cp0", "outcome": "declined"}]}
                for k in range(n)]
    assert sum(m["Y"] for m in missions) == y
    pw = [{"world": 0, "condition": "learned", "sum_Y": y, "sum_T_s": n * 720.0, "per_h": y * 3600.0 / (n * 720.0)}]
    S.PrimaryMetric.model_validate({"contract": "twin.primary_metric/1", "engine": "double", "split": "dev",
                                    "missions": missions, "per_world": pw})


def test_tasks_mission_order_must_cover_zone():
    d = _raw(FIX / "twin_tasks.yaml")
    d["zones"][0]["mission_order"] = ["cp1"]
    _expect_invalid(S.Tasks, d, "mission_order")


def test_day_script_events_must_tile_the_day():
    d = _raw(FIX / "twin_day_script.json")
    ev = [e for e in d["press_events"] if e["press"] == "P9a"]
    ev[-1]["t1"] -= 10.0
    _expect_invalid(S.DayScript, d, "cover")


def test_tables_reject_bad_rows_and_types(tmp_path):
    t = pq.read_table(FIX / "twin_inventory.parquet")
    with pytest.raises(ValidationError, match="implies camera"):
        S.validate_table(t.set_column(1, "camera", pa.array([14, 9, 14, 14], pa.int64())), TB.InventoryRow)
    with pytest.raises(ValueError, match="arrow type"):
        S.validate_table(t.set_column(5, "duration", pa.array([1, 2, 3, 4], pa.int64())), TB.InventoryRow)
    with pytest.raises(ValueError, match="columns"):
        S.validate_table(t.select([1, 0, 2, 3, 4, 5, 6, 7]), TB.InventoryRow)
    tests_only = t.set_column(7, "split", pa.array(["test"] * 4))
    with pytest.raises(ValueError, match="test split"):
        S.validate_table(tests_only, TB.InventoryRow)                         # the default refuses test rows
    assert S.validate_table(tests_only, TB.InventoryRow, allow_test=True) == 4
    m = pq.read_table(FIX / "twin_machines.parquet")
    with pytest.raises(ValidationError, match="missing frame"):
        S.validate_table(m.set_column(8, "missing", pa.array([True] * 4)), TB.MachineRow)


@pytest.mark.parametrize("row,match", [
    ({"missing": False, "flow_v": float("nan"), "roi_energy": float("nan"), "stroke": True}, "not missing"),
    ({"missing": False, "flow_v": 1.0, "roi_energy": None, "stroke": False}, "not missing"),
    ({"missing": False, "flow_v": float("inf"), "roi_energy": 0.5, "stroke": False}, "not missing"),
    ({"missing": True, "flow_v": float("nan"), "roi_energy": float("nan"), "stroke": True}, "missing frame"),
])
def test_machine_row_nan_means_missing(row, match):
    base = {"split": "train", "clip_id": "c9_train_0001", "press": "P9a", "framing": "cam9_f0", "t": 0.0}
    with pytest.raises(ValidationError, match=match):
        TB.MachineRow.model_validate({**base, **row})
    TB.MachineRow.model_validate({**base, "missing": False, "flow_v": 3.0, "roi_energy": 0.2, "stroke": True})
    TB.MachineRow.model_validate({**base, "missing": True, "flow_v": None, "roi_energy": None, "stroke": False})


# ------------------------------------------------------------------------------------------------ v0 stubs


def test_site_v0_is_all_prior():
    site = S.load(S.Site, TWIN / "site_v0.yaml")
    assert len(site.zones) == 2 and len(site.presses) == 2 and site.cal_sources == []
    raw = yaml.safe_load((TWIN / "site_v0.yaml").read_text(encoding="utf-8"))

    def sources(o):
        if isinstance(o, dict):
            if "source" in o:
                yield o["source"]
            for v in o.values():
                yield from sources(v)
        elif isinstance(o, list):
            for v in o:
                yield from sources(v)
    assert set(sources(raw)) == {"prior"}


def test_dynamics_v0_is_priors_only():
    dyn = S.load(S.Dynamics, TWIN / "dynamics_v0.json")
    params = [*dyn.observed.values(), *dyn.assumed.values()]
    assert params and all(p.source in ("prior", "assumed") and p.n == 0 and not p.clips for p in params)
    assert all(p.source == "prior" for p in dyn.observed.values())
    tod = {k: p for k, p in dyn.assumed.items() if p.time_of_day}
    assert "site.night_share" in tod and all(p.source == "assumed" and "ASSUMED" in (p.note or "") for p in tod.values())
    assert not any(p.time_of_day for p in dyn.observed.values())


def test_time_of_day_fields_are_assumed():
    """P2: no timing signal, so time-of-day / shift structure is ASSUMED and tagged (user decision)."""
    d = _raw(TWIN / "dynamics_v0.json")
    bad = copy.deepcopy(d)
    bad["observed"]["site.night_share"] = bad["assumed"].pop("site.night_share")      # moved to OBSERVED
    _expect_invalid(S.Dynamics, bad, "must be ASSUMED")
    bad = copy.deepcopy(d)
    bad["assumed"]["site.night_share"]["time_of_day"] = False                          # tag removed
    _expect_invalid(S.Dynamics, bad, "tag it time_of_day")
    bad = copy.deepcopy(d)
    bad["assumed"]["site.night_share"]["source"] = "prior"
    _expect_invalid(S.Dynamics, bad, "source assumed")
    bad = copy.deepcopy(d)
    bad["assumed"]["staff.shift_start_h"] = {"value": 6.0, "unit": "h", "source": "assumed", "n": 0,
                                             "sweep": {"lo": 5.0, "hi": 7.0}}          # untagged shift field
    _expect_invalid(S.Dynamics, bad, "tag it time_of_day")
    from twin.schemas.dayscript import PressEvent
    ev = {"press": "P9a", "kind": "stopped", "t0": 0.0, "t1": 60.0, "cause": "break", "source": "prior"}
    with pytest.raises(ValidationError, match="break"):
        PressEvent.model_validate(ev)
    PressEvent.model_validate({**ev, "source": "assumed"})


def test_daygen_stub_day_validates_and_is_deterministic():
    a = daygen_stub.generate_day(3, 1)
    b = daygen_stub.generate_day(3, 1)
    assert a.model_dump() == b.model_dump()
    assert a.duration_s == 1800.0 and a.presses == ["P9a", "P14a"]
    assert sum(m.cls == "forklift" for m in a.mover_intents) == 1
    assert sum(m.cls == "person" for m in a.mover_intents) >= 1
    S.DayScript.model_validate(json.loads(a.model_dump_json()))
    assert daygen_stub.generate_day(3, 2).model_dump() != a.model_dump()


def test_daygen_stub_seed_guard(monkeypatch):
    with pytest.raises(PermissionError):
        daygen_stub.generate_day(1000, 1, duration_s=60.0)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    daygen_stub.generate_day(1000, 1, duration_s=60.0)


# ------------------------------------------------------------------------------------------------ keyed RNG (W-W1)


def test_keyed_rng_is_order_independent():
    keys = [(5, 1, "press:P9a", "schedule"), (5, 1, "walker:z9:0001", "path"), (5, 2, "press:P9a", "schedule"),
            (6, 1, "press:P9a", "schedule")]
    fwd = {k: R.keyed_rng(*k).random(16) for k in keys}
    junk = R.keyed_rng(5, 1, "other", "event").random(1000)                  # unrelated draws in between
    rev = {k: R.keyed_rng(*k).random(16) for k in reversed(keys)}
    assert junk.size == 1000
    for k in keys:
        np.testing.assert_array_equal(fwd[k], rev[k])
    vals = [tuple(v) for v in fwd.values()]
    assert len(set(vals)) == len(vals), "different keys must give different streams (day, world, entity)"


def test_daygen_events_do_not_depend_on_generation_order_or_other_entities():
    full = daygen_stub.generate_day(11, 4)
    rev = daygen_stub.generate_day(11, 4, entity_order=["forklifts", "z14", "P14a", "z9", "P9a"])
    assert full.model_dump() == rev.model_dump()
    only = daygen_stub.generate_day(11, 4, entity_order=["P14a", "z14"])
    assert [e for e in full.press_events if e.press == "P14a"] == only.press_events
    assert [m for m in full.mover_intents if m.zone == "z14" and m.cls == "person"] == only.mover_intents


# ------------------------------------------------------------------------------------------------ camera_v0


def _site():
    return S.load(S.Site, TWIN / "site_v0.yaml")


def test_camera_v0_on_fixture_stream(tmp_path, monkeypatch):
    site = _site()
    a = list(camera_v0.stream(FIX / "twin_realised_state.jsonl", site, world_seed=1))
    b = list(camera_v0.stream(FIX / "twin_realised_state.jsonl", site, world_seed=1))
    assert len(a) == 8
    assert [camera_v0.detections_to_json(d) for d in a] == [camera_v0.detections_to_json(d) for d in b]
    assert all(math.isnan(d.zone_motion["P9a"]) for d in a), "a person at the P9a ROI makes its motion MISSING"
    assert all(math.isfinite(d.zone_motion["P14a"]) for d in a)
    assert all(d.t == pytest.approx(d.t_cap + 0.2) for d in a)
    assert set(np.concatenate([d.cls for d in a])) <= {"person", "vehicle", "other"}
    log = Path(str(tmp_path / "truth_access.log")).read_text(encoding="utf-8")
    assert log.startswith(LOG_HEADER) and "twin/camera_v0.py\tr\t" in log


def test_camera_v0_noise_matches_measured_shim():
    site = _site()
    cam = camera_v0.CameraV0(site, world_seed=2, day=1)
    n_ticks, hits, errs = 4000, 0, []
    for i in range(n_ticks):
        tk = RU.RealisedTick(kind="tick", t=0.5 * i, presses=[], movers=[
            RU.RealisedMover(id="p", cls="person", zone="z9", x=3.0, z=3.0, heading=0.0, speed=1.0, r=0.3)])
        d = cam.observe(tk)
        real = [j for j in range(len(d)) if d.moving[j]]                    # false positives are never moving
        hits += len(real)
        errs += [math.hypot(d.x[j] - 3.0, d.z[j] - 3.0) for j in real]
    assert hits / n_ticks == pytest.approx(1 - SHIM["p_drop"], abs=0.03)
    errs = np.array(errs)
    assert np.mean(errs > 5.0) == pytest.approx(SHIM["gross_p"], abs=0.03)
    rayleigh_median = SHIM["sigma_m"] * math.sqrt(2 * math.log(2))
    assert np.median(errs[errs < 5.0]) == pytest.approx(rayleigh_median, rel=0.08)


def _mover(mid, cls, x, z, zone="z9", speed=1.0, r=0.3):
    return RU.RealisedMover(id=mid, cls=cls, zone=zone, x=x, z=z, heading=0.3, speed=speed, r=r)


def _people_and_presses(d):
    keep = [j for j in range(len(d)) if d.cls[j] != "other"]
    return ([(str(d.cls[j]), float(d.x[j]), float(d.z[j]), float(d.r[j]), str(d.zone[j])) for j in keep],
            {k: v for k, v in d.zone_motion.items() if not math.isnan(v)})


def test_camera_v0_noise_is_common_across_conditions():
    """CRN: adding the robot (it differs between paired conditions) leaves every other detection and the press
    motion noise unchanged; noise is keyed by (world_seed, day, entity, tick), so tick order does not matter."""
    site = _site()
    people = [_mover(f"walker:z9:{i:04d}", "person", 2.0 + i, 3.0) for i in range(4)] +              [_mover("forklift:0", "forklift", 10.0, 2.0, speed=1.8, r=1.0),
              _mover("walker:z14:0000", "person", 33.0, 2.0, zone="z14")]
    robot = _mover("robot", "robot", 31.0, 1.0, zone="z14", speed=0.5, r=0.45)
    press = [RU.RealisedPress(id=p.id, state="running", ram=0.3, stroke=False, freeze=False) for p in site.presses]
    with_r, without = [], []
    for i in range(200):
        t = 86400.0 + 0.5 * i
        with_r.append(RU.RealisedTick(kind="tick", t=t, presses=press, movers=[robot, *people]))
        without.append(RU.RealisedTick(kind="tick", t=t, presses=press, movers=people))
    a = [_people_and_presses(d) for d in camera_v0.CameraV0(site, 5, 9).run(with_r)]
    b = [_people_and_presses(d) for d in camera_v0.CameraV0(site, 5, 9).run(without)]
    assert a == b
    # order independence: the same tick observed alone gives the same detections
    lone = camera_v0.CameraV0(site, 5, 9)
    lone._prev_ram = {p.id: 0.3 for p in site.presses}
    assert _people_and_presses(lone.observe(without[150])) == b[150]
    # and other keys give other noise
    c = [_people_and_presses(d) for d in camera_v0.CameraV0(site, 5, 10).run(without)]
    assert c != b


@pytest.mark.parametrize("cls", ["person", "forklift", "robot"])
def test_camera_v0_occluders_include_vehicles_and_the_robot(cls):
    """S4 patch (T-W6): a person, a forklift or the robot at the press makes its zone_motion MISSING."""
    site = _site()
    p9 = next(p for p in site.presses if p.id == "P9a")
    xs, zs = [q[0] for q in p9.footprint.value], [q[1] for q in p9.footprint.value]
    occ = _mover("occ", cls, max(xs) + 0.2, (min(zs) + max(zs)) / 2, zone=p9.zone, speed=0.0, r=0.5)
    press = [RU.RealisedPress(id=p.id, state="running", ram=0.3, stroke=False, freeze=False) for p in site.presses]
    d = camera_v0.CameraV0(site, 1, 1).observe(RU.RealisedTick(kind="tick", t=0.0, presses=press, movers=[occ]))
    assert math.isnan(d.zone_motion["P9a"]) and math.isfinite(d.zone_motion["P14a"])


# ------------------------------------------------------------------------------------------------ runtime truth guard


def test_open_truth_refuses_callers_outside_the_allowlist():
    with pytest.raises(PermissionError, match="may not open truth"):
        open_truth(FIX / "twin_realised_state.jsonl")
    with pytest.raises(ValueError, match="not a truth path"):
        open_truth(FIX / "twin_site.yaml")
