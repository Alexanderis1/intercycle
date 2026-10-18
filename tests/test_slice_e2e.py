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


def _cz():
    paths = SL.inputs()
    return paths, SL.C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])["z14"]


def test_modal_learned_cfg_runs_the_learned_stack():
    """--modal timing: the plant in condition learned, the twin broadcaster (obs_v2 only), the bridge with the episode
    keyring and the zone's mask grid, timing_planner on the zone's sitegraph.json; the day script is plant-only."""
    import modal_isaac_robot as R

    paths, cz = _cz()
    c = SL.modal_cfg("t1", paths, "x" + SL.FILE_SUFFIX, cz, 100.0, "z14", 0, "timing", 2.0)
    assert c["mover_source"] == "twin.plant:make" and c["sim"] == "fake" and c["max_rtf"] == 2.0 and c["twin"]
    assert c["plant"]["condition"] == "learned" and c["seconds"] == SL.MISSION_S == 600.0
    assert "-p scenario:=$IC_WORLD/scenario.yaml" in c["goal_cmd"] and c["scenario"] == "scenario.yaml"
    assert "-p sitegraph:=$IC_WORLD/sitegraph.json" in c["goal_cmd"] and "-p condition:=learned" in c["goal_cmd"]
    bc, br = c["extra_launch"]
    assert "-m twin.broadcast_live" in bc and "--obs $IC_OUT/obs_v2.jsonl" in bc and "run_cfg" not in bc
    assert "-p condition:=learned" in br and "keyring:=" in br and "observer_vk.txt" in br
    x0, z0, x1, z1 = cz.extent
    assert f"-p mask_origin:=[{x0},{z0}]" in br and f"-p mask_size:=[{x1 - x0},{z1 - z0}]" in br
    # the day script lives outside the world dir the robot side is given ($IC_WORLD = /tmp/ic_<tag>/world)
    assert SL.REMOTE_PLANT_DIR == R.PLANT_DIR
    assert c["plant"]["day_script"] == f"{R.PLANT_DIR}/t1/x" + SL.FILE_SUFFIX
    assert not c["plant"]["day_script"].startswith("/tmp/ic_")


def test_modal_probe_and_patrol_cfgs_are_labelled_for_what_they_are():
    paths, cz = _cz()
    p = SL.modal_cfg("t2", paths, "x" + SL.FILE_SUFFIX, cz, 100.0, "z14", 0, "probe", 2.0)
    assert p["plant"]["condition"] == "no_map" and len(p["extra_launch"]) == 1
    assert "-p condition:=none" in p["extra_launch"][0] and "keyring" not in p["extra_launch"][0]
    assert "-p condition:=none" in p["goal_cmd"]
    w = SL.modal_cfg("t3", paths, "x" + SL.FILE_SUFFIX, cz, 100.0, "z14", 0, "patrol", 2.0)
    assert w["wiring_only"] and w["plant"]["condition"] == "no_map" and w["scenario"] == ""
    assert isinstance(w["goal"][0], list) and w["goal"][-1][:2] == list(cz.graph.pos["DOCK"])
    with pytest.raises(SL.SliceError, match="unknown Modal planner"):
        SL.modal_cfg("t4", paths, "x", cz, 100.0, "z14", 0, "nope", 2.0)


def test_modal_world_files_carry_the_learned_observer_inputs_and_no_truth(report):
    rep, out = report
    _, cz = _cz()
    site = SL.load(SL.Site, SL.inputs()["site"])
    adapter = SL.PL.ObsAdapter(site)
    spec = SL.observer_spec(site, {"z14": cz}, adapter)
    mp, _ = SL.margin_path(out)
    learn = out / "runs" / "learn" / "obs_v2.jsonl"
    wb = SL.world_bytes(out / "worlds" / "z14", learn, spec, [m.id for m in cz.scenario.machines], mp, "z14")
    assert {"boxes.json", "mesh.npz", "map.yaml", "map.pgm", "world.json", "factory_l0.usda", "scenario.yaml",
            "sitegraph.json", "observer_spec.json", "m7_margin.json", "learn_obs.jsonl"} <= set(wb)
    from twin import truth_guard as TG

    assert not [n for n in wb if TG.is_truth_path(n)]
    assert wb["learn_obs.jsonl"] == learn.read_bytes()
    rec = json.loads(wb["observer_spec.json"])
    assert rec["zone_machines"] == [m.id for m in cz.scenario.machines] and rec["spec"]["site_id"] == spec["site_id"]


def _record_truth_reads(monkeypatch) -> list[tuple[str, str]]:
    """Wrap builtins.open / io.open / Path.open: every opening of a truth path is recorded with its first non-stdlib
    caller (repo-relative)."""
    import builtins
    import inspect
    import io
    import sysconfig

    from twin import truth_guard as TG

    std = Path(sysconfig.get_paths()["stdlib"]).resolve()
    seen: list[tuple[str, str]] = []
    real_open, real_popen = builtins.open, Path.open

    def caller() -> str:
        for fr in inspect.stack()[2:]:
            f = Path(fr.filename).resolve()
            if f == Path(__file__).resolve() and fr.function in ("fake_open", "fake_popen"):
                continue
            if std in f.parents:
                continue
            try:
                return f.relative_to(REPO).as_posix()
            except ValueError:
                return f.as_posix()
        return "?"

    def fake_open(file, *a, **kw):
        if isinstance(file, (str, Path)) and TG.is_truth_path(file):
            seen.append((caller(), str(file)))
        return real_open(file, *a, **kw)

    def fake_popen(self, *a, **kw):
        if TG.is_truth_path(self):
            seen.append((caller(), str(self)))
        return real_popen(self, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(Path, "open", fake_popen)
    return seen


def test_shipping_the_day_script_reads_truth_only_through_the_guard(report, tmp_path, monkeypatch):
    """Audit fix: the Modal payload (world files + plant files) reads truth ONLY through twin.truth_guard.open_truth,
    called by an allowlisted reader (twin/plant.py), and the read is in the access log; the orchestrator reads none."""
    rep, out = report
    log = tmp_path / "truth.log"
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(log))
    seen = _record_truth_reads(monkeypatch)
    _, cz = _cz()
    site = SL.load(SL.Site, SL.inputs()["site"])
    spec = SL.observer_spec(site, {"z14": cz}, SL.PL.ObsAdapter(site))
    mp, _ = SL.margin_path(out)
    SL.world_bytes(out / "worlds" / "z14", out / "runs" / "learn" / "obs_v2.jsonl", spec,
                   [m.id for m in cz.scenario.machines], mp, "z14")
    assert seen == [], f"world files read truth: {seen}"
    script = out / ("w0_d9" + SL.FILE_SUFFIX)
    name, data = SL.PL.day_script_payload(script)
    assert name == script.name and data
    assert [c for c, _ in seen] == ["twin/truth_guard.py"], seen          # the only opening: inside open_truth
    from twin import truth_guard as TG

    lines = [ln.split("\t") for ln in log.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    assert [(ln[1], Path(ln[3]).name) for ln in lines] == [("twin/plant.py", script.name)]
    assert lines[0][1] in TG.TRUTH_READERS and lines[0][2] == "rb"


def test_the_recorder_sees_a_direct_read(report, monkeypatch):
    """Self-test: a direct Path.read_bytes of the script (the old world_bytes) is caught as this file's read."""
    _, out = report
    seen = _record_truth_reads(monkeypatch)
    (out / ("w0_d9" + SL.FILE_SUFFIX)).read_bytes()
    assert seen and seen[0][0] == "tests/test_slice_e2e.py"


def test_feed_check_requires_verified_layers(tmp_path):
    rows = [{"t": 1.0, "map_ok": True, "slow_ok": False, "fast_age_s": None, "stats": {}},
            {"t": 2.0, "map_ok": True, "slow_ok": True, "fast_age_s": 0.4, "stats": {"bridge": {"ok": 3}}}]
    (tmp_path / "observer_bridge.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:1]) + "\n", encoding="utf-8")
    with pytest.raises(SL.SliceError, match="never verified"):
        SL.check_feed_verified(tmp_path)
    (tmp_path / "observer_bridge.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert SL.check_feed_verified(tmp_path)["first_verified_t"] == 2.0


def test_metric_schema_check_uses_the_json_schema_file(report, tmp_path, monkeypatch):
    """The metric file is validated with jsonschema against the schema FILE: a schema file that rejects the
    instance fails the check even though the pydantic model accepts it."""
    rep, out = report
    pm = SL.PrimaryMetric.model_validate_json((out / "primary_metric.json").read_text(encoding="utf-8"))
    assert "jsonschema" in SL.check_metric_schema(pm, tmp_path / "pm.json")["validated"]
    fake = tmp_path / "twin"
    (fake / "schemas" / "json").mkdir(parents=True)
    sch = json.loads((SL.C.TWIN / "schemas" / "json" / "primary_metric.schema.json").read_text(encoding="utf-8"))
    sch["properties"]["engine"] = {"const": "isaac"}
    (fake / "schemas" / "json" / "primary_metric.schema.json").write_text(json.dumps(sch), encoding="utf-8")
    monkeypatch.setattr(SL.C, "TWIN", fake)
    with pytest.raises(SL.SliceError, match="does not validate"):
        SL.check_metric_schema(pm, tmp_path / "pm2.json")


# ---------------------------------------------------------------- the twin broadcaster -> the robot's bridge state

class _Sink:
    def __init__(self):
        self.records: list[bytes] = []

    def frames(self, frs):
        from intercycle_nav.feed import FRAME

        self.records += [FRAME + bytes(f) for f in frs]
        return len(frs)


def _broadcast(report, tmp_path, zone="z14"):
    """twin/broadcast_live's pipeline (world files -> learned observer -> signed map / slow / fast) into an in-memory
    sink; returns (vk line, [(t, record)], zone world dir, t_learn_end)."""
    from intercycle_nav.broadcast import Broadcaster, pack_map_for_key
    from twin import broadcast_live as BL

    _, out = report
    _, cz = _cz()
    site = SL.load(SL.Site, SL.inputs()["site"])
    spec = SL.observer_spec(site, {zone: cz}, SL.PL.ObsAdapter(site))
    mp, _ = SL.margin_path(out)
    world, work = tmp_path / "world", tmp_path / "work"
    world.mkdir()
    work.mkdir()
    for n, b in SL.world_bytes(out / "worlds" / zone, out / "runs" / "learn" / "obs_v2.jsonl", spec,
                               [m.id for m in cz.scenario.machines], mp, zone).items():
        (world / n).write_bytes(b)
    key, vk = BL.episode_key()
    BL.write_vk(work, vk)
    spec2, zm, margin, learn = BL.load_inputs(world, work)
    assert learn == work / "learn" / "obs_v2.jsonl" and (work / BL.VK_FILE).read_text(encoding="utf-8") == vk
    ob, t_now, n = BL.learned_observer(spec2, margin, learn)
    assert n == int(SL.LEARN_S * 2)
    bundle = pack_map_for_key(ob.map_snapshot(), key, version=1, issue_time=SL.DAY0_UNIX + t_now)
    zv = BL.ZoneView(ob, zm)
    sink = _Sink()
    bc = Broadcaster(sink, key, bundle, zv.slow, zv.fast)
    recs: list[tuple[float, bytes]] = []
    for k in range(20):                                    # 10 s of 2 Hz ticks: the map carousel completes
        t = t_now + 0.5 * k
        bc.tick(t)
        recs += [(t, r) for r in sink.records]
        sink.records.clear()
    return vk, recs, world, t_now


def test_twin_broadcast_verifies_in_the_robot_bridge_state(report, tmp_path):
    """The twin broadcaster's records, verified by masks.ObserverState with the episode's keyring (the bridge's
    -p keyring:=<observer_vk.txt>) on the zone's scenario: map, slow and fast layers all verify; the machines are
    the zone's, in its scenario order; masks.Site takes the twin zone's extent."""
    from intercycle_nav import masks as MK

    vk, recs, world, _ = _broadcast(report, tmp_path)
    site = MK.Site.from_scenario(str(world / "scenario.yaml"))
    st = MK.ObserverState(site, MK.keyring_from_hex(vk))
    for t, r in recs:
        st.ingest(t, r)
    snap = st.snapshot(recs[-1][0])
    assert snap.map_ok and snap.slow_ok and snap.fast_age_s is not None, st.all_stats()
    assert [m.id for m in snap.machines] == site.ids == ["P14a"]
    assert st.stats["map_complete"] == 1 and st.stats["ok"] >= 2, st.all_stats()
    assert not [k for k in st.stats if k.startswith("rej")], st.all_stats()
    assert site.extent == tuple(float(v) for v in _cz()[1].extent)


def test_twin_broadcast_fails_closed_on_a_foreign_keyring(report, tmp_path):
    from intercycle_nav import masks as MK
    from twin import broadcast_live as BL

    _, recs, world, _ = _broadcast(report, tmp_path)
    _, other = BL.episode_key()
    st = MK.ObserverState(MK.Site.from_scenario(str(world / "scenario.yaml")), MK.keyring_from_hex(other))
    for t, r in recs:
        st.ingest(t, r)
    snap = st.snapshot(recs[-1][0])
    assert not snap.map_ok and not snap.slow_ok and snap.fast_age_s is None
    assert all(m.state == "UNKNOWN" and m.windows is None for m in snap.machines)


def test_twin_broadcast_refuses_bad_inputs(report, tmp_path):
    from twin import broadcast_live as BL

    mp, _ = SL.margin_path(report[1])
    (tmp_path / BL.SPEC_FILE).write_text(json.dumps({"contract": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        BL.load_inputs(tmp_path, tmp_path / "work")
    with pytest.raises(ValueError, match="not all in the observer spec"):
        BL.spec_record({"machines": [{"id": "P14a"}]}, "z14", ["P9a"])
    empty = tmp_path / "learn" / "obs_v2.jsonl"
    empty.parent.mkdir()
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no learning-period observations"):
        BL.learned_observer(SL.OL.twin_spec([dict(id="P14a", kind="press", rect=(0, 0, 1, 1))], (0, 0, 5, 5),
                                            [(1.0, 1.0)], site_id=1, key_id=1), str(mp), empty)
