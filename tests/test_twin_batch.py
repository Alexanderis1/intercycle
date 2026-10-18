"""intercycle.eval.twin_batch (the cut-design CPU-double batch plan) and scripts/run_double.py end to end on the stub
world with the local wiring backend (no Modal), scored by twin_eval."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from intercycle.eval import twin_batch as TB
from intercycle.eval import twin_eval as E
from twin import truth_guard as TG

REPO = Path(__file__).resolve().parents[1]
WORLDS = [TB.World(i, i, "test") for i in range(16)]


def test_plan_is_the_192_cell_cut_design_crn_paired():
    ms = TB.plan(WORLDS, ["z9", "z14"])
    assert len(ms) == 192 and {m.condition for m in ms} == {"static", "learned", "reactive"}
    by = {}
    for m in ms:
        by.setdefault(m.cell, set()).add(m.t0)
    assert len(by) == 64 and all(len(v) == 1 for v in by.values())        # one mission start per cell (CRN)
    for m in ms:
        lo = (m.day - 1) * TB.DAY_S + TB.SLOT_PAD_S
        assert lo <= m.t0 <= m.day * TB.DAY_S - 720 - TB.SLOT_PAD_S and (m.t0 * 2) == int(m.t0 * 2)
    named = E.parse_run_id(ms[0].run_id)
    assert len({m.run_id for m in ms}) == 192 and named is not None and named[0] == (0, 9, "z9", "static")
    for bad in (("static", "learned"), ("static", "learned", "reactive", "oracle"), ("no_map", "learned", "reactive")):
        with pytest.raises(TB.BatchError, match="cut design"):
            TB.plan(WORLDS, ["z9"], conditions=bad)


def test_rounds_follow_the_reactive_chain_and_keep_cells_together():
    ms = TB.plan(WORLDS, ["z9", "z14"])
    rs = TB.rounds(ms, wave_size=48)
    assert [len(r) for r in rs] == [48, 48, 48, 48]
    for w, chain in TB.reactive_chains(ms).items():
        assert [m.t0 for m in chain] == sorted(m.t0 for m in chain)
        for r, m in enumerate(chain):
            assert m in rs[r]
            assert {x.condition for x in rs[r] if x.cell == m.cell} == {"static", "learned", "reactive"}
    with pytest.raises(TB.BatchError, match="wave size"):
        TB.rounds(ms, wave_size=40)


def test_reruns_repeat_all_conditions_of_an_invalid_cell_and_stop():
    ms = TB.plan(WORLDS[:1], ["z9"])
    rnd = [m for m in ms if m.day == 9]
    again = TB.reruns(rnd, [rnd[1].run_id])
    assert sorted(m.run_id for m in again) == sorted(f"w0_d9_z9_{c}__a1" for c in ("static", "learned", "reactive"))
    again2 = TB.reruns(again, [again[0].run_id])
    assert all(m.attempt == 2 for m in again2)
    assert TB.reruns(again2, [again2[0].run_id]) == []                   # MAX_ATTEMPTS: the cell stays unresolved
    sa = TB.scored_attempt(rnd + again, [rnd[1].run_id])
    assert sa[(0, 9, "z9")] == 1
    inv = TB.merge_invalid([{"invalid": [rnd[1].run_id]}], TB.mission_records(rnd + again))
    assert inv["invalid"][0]["condition"] == rnd[1].condition and inv["invalid"][0]["split"] == "test"
    with pytest.raises(TB.BatchError, match="never planned"):
        TB.merge_invalid([{"invalid": ["w9_d9_z9_static"]}], TB.mission_records(rnd))


def test_projection_uses_the_h1_rates():
    p = TB.projection(16, 192, observer_wall_s=2400, mission_wall_s=400, n_rounds=4)
    assert p["usd_missions"] == pytest.approx(192 * 400 * (4 * 0.0000131 + 8 * 0.00000222), rel=1e-3)
    assert p["wall_s"] == 2400 + 4 * 400


def test_sealed_batch_and_cli_refuse_without_the_sealed_flag(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_double as RD  # pyright: ignore[reportMissingImports]  (scripts/ on sys.path)

    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(RD.DoubleError, match="sealed_run"):
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)])
    assert RD.main(["--sealed", "--out", str(tmp_path)]) == 2
    assert RD.main(["--out", str(tmp_path)]) == 2                         # neither --dev nor sealed_run
    with pytest.raises(RD.DoubleError, match="dev worlds"):
        RD.dev_worlds([4])
    with pytest.raises(RD.DoubleError, match="unknown"):
        RD.run_batch(tmp_path, RD.dev_worlds([0]), backend="isaac")
    with pytest.raises(RD.DoubleError, match="dev worlds only"):
        RD.run_batch(tmp_path, [RD.TB.World(0, 1000, "test")], max_cells=1)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(RD.DoubleError, match="sealed batch runs only"):         # never the wiring double
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)], backend="local")
    monkeypatch.setattr(RD, "SEALED_BACKEND_READY", False)
    with pytest.raises(RD.DoubleError, match="sealed batch runs only"):         # modal refused while not ready
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)], backend="modal")


def _modal_ctx(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_double as RD  # pyright: ignore[reportMissingImports]  (scripts/ on sys.path)

    ctx = RD.setup(tmp_path)
    RD.SL.BW.build(ctx["paths"]["site"], ctx["paths"]["dynamics"], ctx["paths"]["tasks"], tmp_path / "worlds")
    return RD, ctx


def test_modal_cfg_runs_the_real_policy_of_each_condition(tmp_path):
    """learned: the snapshot broadcaster (--snapshot-t0 = the mission start) + the bridge with the episode keyring;
    static: no feed, static bridge and planner; reactive: own camera, frozen default config, world id from the design
    index (never the seed), the mission tag, and the input state only when the chain has a previous step. Every cfg is
    the full T, the plant condition = the arm, and the robot side carries the zone scenario (check_robot_scenario)."""
    RD, ctx = _modal_ctx(tmp_path)
    w = RD.TB.World(0, 0, "dev")
    t0 = 8 * RD.TB.DAY_S + 1234.5
    cfgs = {c: RD.modal_cfg(ctx, RD.TB.Mission(w, 9, "z9", c, t0), "d9.json")
            for c in ("learned", "static", "reactive")}
    for c, cfg in cfgs.items():
        assert cfg["seconds"] == 720.0 and cfg["timeout_s"] == 720.0 and cfg["max_rtf"] == 2.0
        assert cfg["plant"]["condition"] == c and cfg["plant"]["plant_t0"] == t0
        assert cfg["plant"]["day_script"] == f"/root/plant/w0_d9_z9_{c}/d9.json"
        assert "allow_short_mission" not in cfg["plant"]
        assert f"-p condition:={c}" in cfg["goal_cmd"]
        bridge = [x for x in cfg["extra_launch"] if "observer_bridge" in x]
        assert len(bridge) == 1 and f"-p condition:={c}" in bridge[0]
    lb = cfgs["learned"]["extra_launch"]
    assert len(lb) == 2 and f"--snapshot-t0 {t0}" in lb[0] and "twin.broadcast_live" in lb[0]
    assert "keyring:=" in lb[1] and "feed:=unix://" in lb[1]
    assert len(cfgs["static"]["extra_launch"]) == 1 and "feed:=none" in cfgs["static"]["extra_launch"][0]
    rb = cfgs["reactive"]["extra_launch"][0]
    assert "world_id:=dev0" in rb and "mission_tag:=w0_d9_z9_reactive" in rb and "reactive_state:=" not in rb
    assert "reactive_config.json" in rb
    rb2 = RD.modal_cfg(ctx, RD.TB.Mission(w, 10, "z9", "reactive", t0 + 86400.0), "d10.json",
                       reactive_in=True)["extra_launch"][0]
    assert f"reactive_state:=$IC_WORLD/{RD.REACTIVE_IN}" in rb2
    with pytest.raises(RD.DoubleError, match="cut design"):
        RD.modal_cfg(ctx, RD.TB.Mission(w, 9, "z9", "oracle", t0), "d9.json")


def test_modal_world_files_have_no_truth_and_the_frozen_default_reactive_config(tmp_path):
    RD, ctx = _modal_ctx(tmp_path)
    from intercycle_nav import reactive as RX  # pyright: ignore[reportMissingImports]
    from twin import broadcast_live as BL

    wb = RD.zone_world_bytes(ctx, tmp_path, "z9")
    assert {BL.SPEC_FILE, BL.MARGIN_FILE, RD.REACTIVE_CFG, "map.yaml", "scenario.yaml"} <= set(wb)
    assert not any(TG.is_truth_path(n) for n in wb)
    assert RX.ReactiveConfig.from_dict(json.loads(wb[RD.REACTIVE_CFG])).sha256() == RX.ReactiveConfig().sha256()


def test_snapshot_broadcaster_refuses_a_snapshot_from_after_the_mission_start(tmp_path):
    import pickle

    from twin import broadcast_live as BL

    RD, ctx = _modal_ctx(tmp_path)
    snaps, _ = RD.observer_phase(ctx, RD.TB.World(0, 0, "dev"), [(1, "z9", 1800.0)], day_s=3600.0)
    (tmp_path / BL.SNAPSHOT_FILE).write_bytes(snaps[(1, "z9")])
    ob, t_last, n = BL.snapshot_observer(tmp_path, 1800.0)
    assert t_last < 1800.0 and n > 0 and ob.n_ticks == n
    with pytest.raises(ValueError, match="predate"):
        BL.snapshot_observer(tmp_path, t_last)
    (tmp_path / BL.SNAPSHOT_FILE).write_bytes(pickle.dumps({"not": "an observer"}))
    with pytest.raises(ValueError, match="ObserverLive"):
        BL.snapshot_observer(tmp_path, 1800.0)


def test_sealed_robot_stage_runs_the_double_with_the_revealed_seeds_only_under_the_flag(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import sealed_robot as SRB  # pyright: ignore[reportMissingImports]

    seen = {}

    def fake(out, seeds):
        seen.update(out=out, seeds=seeds)
        return {"runs_root": str(out / "runs"), "n_missions": 192}

    rec = {"attempt": 2, "seeds": {"seed_ints": [1000 + i for i in range(16)]}}
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(SRB.RobotStageError, match="sealed flag"):
        SRB.run_stage("double", rec, repo=tmp_path, double_runner=fake)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    out = SRB.run_stage("double", rec, repo=tmp_path, double_runner=fake)
    assert seen["seeds"] == rec["seeds"]["seed_ints"] and seen["out"] == tmp_path / SRB.DOUBLE_OUT / "attempt_002"
    assert out["n_missions"] == 192
    for st, r, msg in (("isaac", rec, "descriptive"), ("nope", rec, "unknown"),
                       ("double", {"attempt": 3, "seeds": {"seed_ints": [1]}}, "16 revealed")):
        with pytest.raises(SRB.RobotStageError, match=msg):
            SRB.run_stage(st, r, repo=tmp_path, double_runner=fake)


def test_local_batch_end_to_end_into_twin_eval(tmp_path, monkeypatch):
    """Stub world, dev world 0, both zones, the 3 conditions on days 9-10 (12 missions, 720 s each), the observer phase
    on SHORT 1 h days (dev-only test setting), classify_invalid per round, scored by twin_eval on the dev split."""
    sys.path.insert(0, str(REPO / "scripts"))
    import run_double as RD  # pyright: ignore[reportMissingImports]  (scripts/ on sys.path)

    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))
    rec = RD.run_batch(tmp_path / "out", RD.dev_worlds([0]), day_s=3600.0)
    assert rec["n_missions"] == 12 and rec["n_rounds"] == 4
    st = rec["observer_phase"][0]
    assert st["ticks"] > 0 and len(st["snapshot_bytes"]) == 4
    runs = Path(rec["runs_root"])
    inv = json.loads((runs / TB.INVALID_ALL).read_text(encoding="utf-8"))
    assert inv["invalid"] == []
    tasks = REPO / "twin" / "tasks_v0.yaml"
    rep = E.run_eval(runs, tasks, split="dev", invalid=runs / TB.INVALID_ALL)
    assert rep["design"]["conditions"] == ["static", "learned", "reactive"] and rep["unresolved_cells"] == []
    assert len(rep["primary_metric"]["missions"]) == 12
    assert {m["condition"] for m in rep["primary_metric"]["missions"]} == {"static", "learned", "reactive"}
    t0s = {c["run_id"]: c["t0"] for b in runs.glob("round*/batch.json")
           for c in json.loads(b.read_text(encoding="utf-8"))["cfgs"]}
    learned = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in runs.glob("round*/w0_*_learned.json")}
    assert len(learned) == 4 and set(learned) <= set(t0s)
    for rid, r in learned.items():                                       # every snapshot predates its mission start
        assert r["observer"]["t_last"] < t0s[rid]
        assert r["observer"]["t_last"] > t0s[rid] - 5.0                  # ... and is the state AT the start
    assert os.path.exists(tmp_path / "out" / "costs.json")


def test_slots_of_a_world_day_never_overlap_and_first_draws_are_kept():
    """The reactive chain crosses zones: no two slots of a world-day closer than one mission (twin_batch doc)."""
    n_moved = 0
    for seed in range(400):
        for d in (9, 10):
            sl = TB.day_slots(seed, d, ["z9", "z14"])
            assert abs(sl["z9"] - sl["z14"]) >= 720
            assert sl["z9"] == TB.slot_t0(seed, d, "z9")                  # the first zone is the first draw
            n_moved += sl["z14"] != TB.slot_t0(seed, d, "z14")
    assert 0 < n_moved < 40                                              # only the overlapping world-days redraw
    ms = TB.plan([TB.World(i, s, "test") for i, s in enumerate(range(400))], ["z9", "z14"])
    for chain in TB.reactive_chains(ms).values():
        assert all(b.t0 >= a.t0 + 720 for a, b in zip(chain, chain[1:]))


def test_rounds_refuse_an_overlapping_chain():
    w = TB.World(0, 0, "test")
    ms = [TB.Mission(w, 9, z, c, 9 * 86400.0 + (0 if z == "z9" else 100)) for z in ("z9", "z14")
          for c in ("static", "learned", "reactive")]
    with pytest.raises(TB.BatchError, match="overlap"):
        TB.rounds(ms)


def test_the_sealed_plan_matches_twin_eval_s_sealed_test_design():
    """The launcher's 192 test-split run ids are exactly the cells of twin_eval's sealed design (fixed, K = 16)."""
    from intercycle.eval import twin_design as D

    ms = TB.plan(WORLDS, ["z9", "z14"])
    des = D.make_design([], {}, split="test", engine="double", worlds=list(range(16)))
    assert des.fixed and des.conditions == ("static", "learned", "reactive") and des.worlds == tuple(range(16))
    cells = {D.parse_run_id(m.run_id) for m in ms}
    want = {((w, d, z, c), 0) for w in des.worlds for d in des.days for z in ("z9", "z14") for c in des.conditions}
    assert len(want) == 192 and cells == want
    with pytest.raises(D.EvalError, match="K = 16"):
        D.make_design([], {}, split="test", engine="double", worlds=list(range(15)))
