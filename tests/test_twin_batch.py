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
    assert len({m.run_id for m in ms}) == 192 and E.parse_run_id(ms[0].run_id)[0] == (0, 9, "z9", "static")
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
    import run_double as RD

    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(RD.DoubleError, match="sealed_run"):
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)])
    assert RD.main(["--sealed", "--out", str(tmp_path)]) == 2
    assert RD.main(["--out", str(tmp_path)]) == 2                         # neither --dev nor sealed_run
    with pytest.raises(RD.DoubleError, match="dev worlds"):
        RD.dev_worlds([4])
    with pytest.raises(RD.DoubleError, match="not wired"):
        RD.run_batch(tmp_path, RD.dev_worlds([0]), backend="modal")


@pytest.mark.slow
def test_local_batch_end_to_end_into_twin_eval(tmp_path, monkeypatch):
    """Stub world, dev world 0, both zones, the 3 conditions on days 9-10 (12 missions, 720 s each), the observer phase
    on SHORT 1 h days (dev-only test setting), classify_invalid per round, scored by twin_eval on the dev split."""
    sys.path.insert(0, str(REPO / "scripts"))
    import run_double as RD

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
    learned = [json.loads(p.read_text(encoding="utf-8")) for p in runs.glob("round*/w0_*_learned.json")]
    assert len(learned) == 4 and all(r["observer"]["t_last"] < 9 * 86400 + 1e9 for r in learned)
    assert os.path.exists(tmp_path / "out" / "costs.json")
