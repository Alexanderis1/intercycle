"""Thin vertical slice end to end (PLAN_TWIN §4; scripts/slice.py), locally on the G0 stubs.

Runs steps 1-8 with the local scripted double (the Modal fake-backend + Nav2 mission is `scripts/slice.py --modal`,
run at the gates) and checks the plan's assertions: truth allowlist + Isaac-deps tests pass, signatures verify with
an explicit keyring and fail closed, >= 1 inspection attempt, the metric JSON validates against its schema, a press
stop caused by the robot appears in the Observation within 1 tick + latency. The checkers themselves are shown to
fail on a broken input (a late observation, a mission without an attempt, an errored mission)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("slice_script", REPO / "scripts" / "slice.py")
assert _spec and _spec.loader
SL = importlib.util.module_from_spec(_spec)
sys.modules["slice_script"] = SL
_spec.loader.exec_module(SL)


@pytest.fixture(autouse=True)
def _truth_log(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    import os

    out = tmp_path_factory.mktemp("slice")
    old = os.environ.get("INTERCYCLE_TRUTH_LOG")
    os.environ["INTERCYCLE_TRUTH_LOG"] = str(out / "truth_access.log")
    try:
        rep = SL.run(out, world=0, day=9, zone="z14", guard_tests=True)
    finally:
        if old is None:
            os.environ.pop("INTERCYCLE_TRUTH_LOG", None)
        else:
            os.environ["INTERCYCLE_TRUTH_LOG"] = old
    return rep, out


def test_slice_is_green_on_stubs(report):
    rep, out = report
    bad = {k: v for part in ("steps", "assertions") for k, v in rep[part].items() if not v["ok"]}
    assert not bad, bad
    assert rep["green"]
    assert json.loads((out / "slice_report.json").read_text(encoding="utf-8"))["green"]


def test_step1_world_files_and_stub_inputs(report):
    rep, out = report
    s = rep["steps"]["1_build"]
    for z in ("z9", "z14"):
        for f in SL.FILES_STEP1:
            assert (out / "worlds" / z / f).stat().st_size > 0
    assert s["stubs"]["tasks"] is True or (SL.C.TWIN / "tasks.yaml").exists()


def test_step2_day_has_two_presses_walkers_and_one_forklift(report):
    s = report[0]["steps"]["2_day"]
    assert len(s["presses"]) == 2 and s["forklifts"] == 1 and s["walkers"] >= 1
    assert s["duration_s"] == SL.DAY_S == 1800.0


def test_steps3_5_observation_at_2hz_and_observer_learned(report):
    rep, _ = report
    e = rep["steps"]["3_4_engine_adapter"]
    assert e["n_obs"] == int(SL.LEARN_S * 2) and e["hz"] == 2.0
    assert e["t_cap1"] - e["t_cap0"] == SL.LEARN_S - 0.5
    o = rep["steps"]["5_observer"]
    assert o["ticks"] == int(SL.LEARN_S * 2)
    assert any(m["confidence"] > 0 for m in o["models"]), "no press learned in 20 min"


def test_signatures_verify_and_fail_closed(report):
    sig = report[0]["assertions"]["signatures"]
    assert sig["ok"] and all(sig["fail_closed"].values())
    assert set(sig["fail_closed"]) >= {"empty_keyring", "foreign_key", "tampered", "signer_outside_keyring",
                                        "expired_keyring"}


def test_inspection_attempt_and_metric(report):
    rep, out = report
    assert rep["assertions"]["inspection_attempt"]["attempted"] >= 1
    pm = SL.PrimaryMetric.model_validate_json((out / "primary_metric.json").read_text(encoding="utf-8"))
    m = pm.missions[0]
    assert m.T_s == 720.0 and m.condition == "learned" and pm.split == "dev"
    # the trip probe causes a press stop on purpose: the mission is zeroed and its press checkpoint is "tripped"
    assert m.robot_caused_press_stop and m.Y == 0
    assert dict((c.checkpoint, c.outcome) for c in m.checkpoints)["cp_P14a_ram"] == "tripped"


def test_press_stop_appears_within_one_tick_plus_latency(report):
    st = report[0]["assertions"]["stop_in_observation"]["stops"]
    assert st and all(x["ok"] for x in st)
    for x in st:
        assert x["obs_t"] <= x["first_frozen_tick"] + 0.5 + SL.OM.LATENCY_S + 1e-6


def test_truth_never_opened_by_the_slice(report):
    """The slice is an orchestrator: every truth opening in the access log is by an allowlisted module."""
    _, out = report
    lines = [ln.split("\t") for ln in (out / "truth_access.log").read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    from twin import truth_guard as TG

    callers = {ln[1] for ln in lines}
    assert callers and callers <= set(TG.TRUTH_READERS) | set(TG.TRUTH_PRODUCERS)
    assert "scripts/slice.py" not in callers


# ---------------------------------------------------------------- the checkers fail on broken inputs

def test_stop_checker_rejects_a_late_observation(report, tmp_path):
    _, out = report
    rd = out / "runs" / "mission_local"
    adapter = SL.PL.ObsAdapter(SL.load(SL.Site, SL.inputs()["site"]))
    st = report[0]["assertions"]["stop_in_observation"]["stops"][0]
    lines = (rd / "obs_v2.jsonl").read_text(encoding="utf-8").splitlines()
    late = tmp_path / "late"
    late.mkdir()
    out_lines = []
    for ln in lines:
        d = json.loads(ln)
        if abs(d["t_cap"] - st["obs_t_cap"]) < 1e-6:
            d["t"] = d["t_cap"] + 1.0                      # delivered after 1 tick + latency
        out_lines.append(json.dumps(d))
    (late / "obs_v2.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    with pytest.raises(SL.SliceError, match="within 1 tick"):
        SL.check_stop_propagation(rd, late / "obs_v2.jsonl", adapter, 0.5)
    moving = tmp_path / "moving"
    moving.mkdir()
    i = adapter.press_ids.index(st["press"])
    out_lines = []
    for ln in lines:
        d = json.loads(ln)
        if abs(d["t_cap"] - st["obs_t_cap"]) < 1e-6:
            d["zone_motion"][i] = 2.0                      # the press still reported moving after the stop
        out_lines.append(json.dumps(d))
    (moving / "obs_v2.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    with pytest.raises(SL.SliceError, match="within 1 tick"):
        SL.check_stop_propagation(rd, moving / "obs_v2.jsonl", adapter, 0.5)


def test_metric_zeroes_an_errored_mission_and_attempt_check_fails_on_zero(report, tmp_path):
    _, out = report
    rd = out / "runs" / "mission_local"
    bad = tmp_path / "err"
    bad.mkdir()
    s = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    s.update(status="error", status_reason="test", robot_caused_press_stops=0, holds=0, incidents=0)
    (bad / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    rows = SL.read_log(rd)
    (bad / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["kind"] != "press_trip") + "\n",
                                   encoding="utf-8")
    pm, info = SL.mission_metric(bad, ["cp_P14a_ram", "cp_z14_panel"])
    assert pm.missions[0].error and pm.missions[0].Y == 0
    with pytest.raises(SL.SliceError, match="no inspection attempt"):
        SL._attempts({"summary": dict(info["summary"], attempted=0)})
    with pytest.raises(SL.SliceError, match="checkpoints"):
        SL.mission_metric(bad, ["cp_P14a_ram"])


def test_guard_tests_failure_is_reported(monkeypatch):
    monkeypatch.setattr(SL, "GUARD_TESTS", ("tests/does_not_exist_test.py",))
    with pytest.raises(SL.SliceError, match="guard tests failed"):
        SL.run_guard_tests()


def test_slice_names_no_truth_path():
    """The orchestrator rule (twin/truth_guard.py): no truth token in the slice's string literals."""
    from _truth_scan import parse, string_literals
    from twin.truth_guard import TRUTH_TOKENS

    lits = string_literals(parse(REPO / "scripts" / "slice.py"))
    assert not [s for s in lits if any(t in s for t in TRUTH_TOKENS)]


def test_modal_cfgs_carry_the_twin_scenario_and_plant(report):
    rep, out = report
    paths = SL.inputs()
    cz = SL.C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])["z14"]
    c = SL.modal_cfg("t1", paths, "x" + SL.FILE_SUFFIX, cz, 100.0, "z14", 0, "timing", 2.0)
    assert c["mover_source"] == "twin.plant:make" and c["sim"] == "fake" and c["max_rtf"] == 2.0 and c["twin"]
    assert "-p scenario:=$IC_WORLD/scenario.yaml" in c["goal_cmd"] and c["scenario"] == "scenario.yaml"
    assert c["plant"]["day_script"] == "/tmp/ic_t1/world/x" + SL.FILE_SUFFIX
    p = SL.modal_cfg("t2", paths, "x" + SL.FILE_SUFFIX, cz, 100.0, "z14", 0, "patrol", 2.0)
    assert p["scenario"] == "" and isinstance(p["goal"][0], list) and p["goal"][-1][:2] == list(cz.graph.pos["DOCK"])
    wb = SL.world_bytes(out / "worlds" / "z14", out / ("w0_d9" + SL.FILE_SUFFIX))
    assert {"boxes.json", "mesh.npz", "map.yaml", "map.pgm", "world.json", "factory_l0.usda", "scenario.yaml"} <= set(wb)
