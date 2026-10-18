"""S1b item 4: the EVAL runner (intercycle/eval/twin_eval.py) and its truth slot (twin/eval_metrics.py) on synthetic
run dirs: scoring, errors as failures, re-runs (incl. crashed runs without summary.json), the explicit design, the
exact sign-flip, gatekeeping, outside-gate reports, gate 3 (split / day filters, frozen margin, the stream builder) and
gate 4."""
from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

from intercycle.eval import conformal_margin as CM
from intercycle.eval import twin_eval as E
from intercycle.margin_inflate import SCHEMA, Margin, inflation_spec
from twin import eval_metrics as EM
from twin.schemas.tables import LogRow, columns

ORDERS = {"z9": ["a1", "a2", "a3"], "z14": ["b1", "b2"]}
SHA = "0" * 64
REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "twin" / "site_v0.yaml"


def write_tasks(p: Path) -> Path:
    zones = [{"zone": z, "dock": {"x": 0, "z": 0, "yaw": 0}, "mission_order": cps} for z, cps in ORDERS.items()]
    p.write_text(json.dumps({"contract": "twin.tasks/1", "site_id": "s", "zones": zones, "checkpoints": []}),
                 encoding="utf-8")
    return p


def write_run(root: Path, world: int, day: int, zone: str, cond: str, *, passed: tuple[str, ...] = (),
              declined: tuple[str, ...] = (), failed: tuple[str, ...] = (), trip: str | None = None,
              hold: str | None = None, incident: bool = False, contacts: int = 0, lost: float = 0.0,
              status: str = "ok", error_t: float | None = None, attempt: int = 0, split: str = "test",
              engine: str = "double", variant: str = "primary", jsonl: bool = False, counts_override=None,
              summary_override: dict | None = None, rid: str | None = None) -> Path:
    rid = rid or f"w{world}_d{day}_{zone}_{cond}" + (f"__a{attempt}" if attempt else "")
    d = root / rid
    d.mkdir(parents=True)
    rows = [dict(t=0.0, kind="mission_start", entity="robot", checkpoint=None, zone=zone, value=None, detail=None)]
    att = set(passed) | set(failed) | ({trip} if trip else set()) | ({hold} if hold else set())
    for cp in ORDERS[zone]:
        if cp in att:
            rows.append(dict(t=10.0, kind="checkpoint_attempt", entity="robot", checkpoint=cp, zone=zone, value=None,
                             detail=None))
        if cp in passed:
            rows.append(dict(t=20.0, kind="checkpoint_pass", entity="robot", checkpoint=cp, zone=zone, value=None,
                             detail=None))
        if cp in failed:
            rows.append(dict(t=20.0, kind="checkpoint_fail", entity="robot", checkpoint=cp, zone=zone, value=None,
                             detail=None))
        if cp in declined:
            rows.append(dict(t=20.0, kind="checkpoint_decline", entity="robot", checkpoint=cp, zone=zone, value=None,
                             detail=None))
    if trip:
        rows.append(dict(t=30.0, kind="press_trip", entity="P", checkpoint=trip, zone=zone, value=lost, detail=None))
    if hold:
        rows.append(dict(t=30.0, kind="press_hold", entity="P", checkpoint=hold, zone=zone, value=lost, detail=None))
    if incident:
        rows.append(dict(t=40.0, kind="incident", entity="robot", checkpoint=None, zone=zone, value=0.1, detail=None))
    for _ in range(contacts):
        rows.append(dict(t=41.0, kind="contact", entity="m1", checkpoint=None, zone=zone, value=0.2, detail=None))
    if error_t is not None:
        rows.append(dict(t=error_t, kind="error", entity="robot", checkpoint=None, zone=zone, value=None, detail="{}"))
    rows.append(dict(t=720.0 if error_t is None else error_t, kind="mission_end", entity="robot", checkpoint=None,
                     zone=zone, value=None, detail=None))
    for r in rows:
        r["run_id"] = rid
    cols = columns(LogRow)
    if jsonl:
        (d / "log.jsonl").write_text("".join(json.dumps({c: r[c] for c, _ in cols}) + "\n" for r in rows),
                                     encoding="utf-8")
    else:
        types = {"string": pa.string(), "float64": pa.float64()}
        schema = pa.schema([(c, types[a]) for c, a in cols])
        pq.write_table(pa.Table.from_pylist([{c: r[c] for c, _ in cols} for r in rows], schema=schema),
                       d / "log.parquet")
    trips, holds, incs = counts_override or (int(trip is not None), int(hold is not None), int(incident))
    s = {"contract": "twin.summary/1", "run_id": rid, "engine": engine, "world": world, "world_split": split,
         "day": day, "zone": zone, "condition": cond, "T_s": 720.0, "status": status,
         "status_reason": None if status == "ok" else "boom", "checkpoints_planned": len(ORDERS[zone]),
         "attempted": len(att), "passed": len(passed), "declined": len(set(declined) - att),
         "robot_caused_press_stops": trips, "holds": holds, "incidents": incs, "press_time_lost_s": lost,
         "path_length_m": 10.0, "code": {"head": "abc"}, "variant": variant}
    if variant == "reset_sweep":
        s["reset_override_s"] = 60.0
    s.update(summary_override or {})
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    return d


def design(root: Path, worlds: Iterable[int] = range(6), conds=("static", "learned", "reactive"), learned_extra: int = 1,
           split: str = "test") -> None:
    """Complete design: per (world, day, zone): static passes 1, reactive 1, learned 1 + learned_extra (z9 only)."""
    for w in worlds:
        for day in (9, 10):
            for z, cps in ORDERS.items():
                for c in conds:
                    n = 1 + (learned_extra if c == "learned" and z == "z9" else 0)
                    write_run(root, w, day, z, c, passed=tuple(cps[:n]), split=split)


def drop(root: Path, rid: str) -> None:
    shutil.rmtree(root / rid)


def evaluate(root: Path, tmp: Path, **kw) -> dict:
    """run_eval on the synthetic 6-world test design (partial: the sealed design has 16 worlds and 5 conditions)."""
    kw.setdefault("worlds", list(range(6)))
    kw.setdefault("conditions", ("static", "learned", "reactive"))
    kw.setdefault("allow_partial_design", True)
    return E.run_eval(root, write_tasks(tmp / "tasks.json"), **kw)


def write_margin(p: Path, margin_m: float = 0.25, cell_m: float = 0.25, episode_s: float | None = 720.0) -> Path:
    m = Margin(margin_m, cell_m)
    body = {"schema": SCHEMA, "status": "ok", "frozen": True, "margin_m": margin_m, "cell_m": cell_m,
            "inflation": inflation_spec(m), "unit": {"episode_s": episode_s, "kind": "episode"}}
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ sign-flip


def test_signflip_exact_known_cases():
    assert E.signflip_p([1.0, 2.0, 3.0], "greater") == pytest.approx(1 / 8)
    assert E.signflip_p([3.0, -1.0, 2.0], "greater") == pytest.approx(2 / 8)    # sums >= 4: 6 and 4
    assert E.signflip_p([3.0, -1.0, 2.0], "less") == pytest.approx(7 / 8)       # sums <= 4: all but 6
    assert E.signflip_p([0.5] * 16, "greater") == pytest.approx(1 / 2 ** 16)
    assert E.signflip_p([-0.5] * 16, "less") == pytest.approx(1 / 2 ** 16)
    assert E.signflip_p([0.0, 0.0], "greater") == 1.0                           # ties count against H1
    # brute force against an independent enumeration
    rng = np.random.default_rng(3)
    d = rng.normal(0.3, 1.0, 10)
    obs = d.sum()
    import itertools

    sums = [sum(s * x for s, x in zip(signs, d)) for signs in itertools.product((1, -1), repeat=10)]
    assert E.signflip_p(d, "greater") == pytest.approx(np.mean([v >= obs - 1e-9 for v in sums]))
    with pytest.raises(E.EvalError):
        E.signflip_p([1.0] * 21, "greater")
    with pytest.raises(E.EvalError):
        E.signflip_p([math.nan], "greater")


def test_poisson_and_clopper_pearson_known_values():
    r = E.poisson_exact(0, 1.0)
    assert r["lo"] == 0.0 and r["hi"] == pytest.approx(3.6889, abs=1e-4)
    r = E.poisson_exact(1, 2.0)
    assert r["lo"] == pytest.approx(0.0253 / 2, abs=1e-4) and r["hi"] == pytest.approx(5.5716 / 2, abs=1e-4)
    lo, hi = E.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)


# ------------------------------------------------------------------------------------------------ scoring


def test_complete_design_primary_and_gates(tmp_path):
    root = tmp_path / "runs"
    design(root)
    rep = evaluate(root, tmp_path)
    g = rep["gatekeeping"]
    # learned +1 inspection per z9 mission: per world (2 days x 2 zones x 720 s = 0.8 h) +2 / 0.8 h = 2.5 / h
    assert g[0]["d_w"] == pytest.approx([2.5] * 6) and g[0]["p"] == pytest.approx(1 / 64) and g[0]["claimed"]
    # no press time lost anywhere: e_w = 0 - 30 < 0 for every world -> p = 1/64
    assert g[1]["p"] == pytest.approx(1 / 64) and g[1]["claimed"]
    assert g[2]["status"] == "no input" and not g[2]["claimed"] and not g[3]["claimed"]
    pm = rep["primary_metric"]
    assert len(pm["missions"]) == 6 * 2 * 2 * 3
    out = rep["reported_outside_gate"]["learned_minus_static"]
    assert out["mean_per_h"] == pytest.approx(2.5) and not out["static_matched_or_beat_learned"]
    tab = rep["checkpoint_outcomes"]["learned"]
    assert tab["a1"]["passed"] == 12 and tab["a2"]["passed"] == 12 and tab["a3"]["unattempted"] == 12
    assert rep["reruns_per_condition"] == {"static": 0, "learned": 0, "reactive": 0}
    assert rep["design"]["worlds"] == list(range(6)) and not rep["design"]["fixed"]


def test_trip_hold_incident_zero_y_and_feed_h2(tmp_path):
    root = tmp_path / "runs"
    design(root)
    for w in range(6):            # replace world w's day-9 z9 learned run
        drop(root, f"w{w}_d9_z9_learned")
    write_run(root, 0, 9, "z9", "learned", passed=("a1",), trip="a2", lost=300.0)
    write_run(root, 1, 9, "z9", "learned", passed=("a1", "a2"), hold="a3", lost=90.0)
    write_run(root, 2, 9, "z9", "learned", passed=("a1", "a2"), incident=True)
    for w in range(3, 6):
        write_run(root, w, 9, "z9", "learned", passed=("a1", "a2"))
    rep = evaluate(root, tmp_path)
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w0_d9_z9_learned"]["Y"] == 0 and ms["w0_d9_z9_learned"]["robot_caused_press_stop"]
    assert ms["w1_d9_z9_learned"]["Y"] == 0 and ms["w1_d9_z9_learned"]["hold"]
    assert ms["w2_d9_z9_learned"]["Y"] == 0 and ms["w2_d9_z9_learned"]["incident"]
    assert not ms["w0_d9_z9_learned"]["error"]                     # summary and log agree
    outs = {c["checkpoint"]: c["outcome"] for c in ms["w0_d9_z9_learned"]["checkpoints"]}
    assert outs == {"a1": "passed", "a2": "tripped", "a3": "unattempted"}
    d2 = rep["gatekeeping"][1]["d_w"]
    assert d2[0] == pytest.approx(300.0 / 0.8) and d2[1] == pytest.approx(90.0 / 0.8) and d2[2] == 0.0
    saf = rep["reported_outside_gate"]["safety_poisson"]["learned"]
    assert saf["trips"]["count"] == 1 and saf["holds"]["count"] == 1 and saf["incidents"]["count"] == 1
    assert saf["trips"]["exposure_h"] == pytest.approx(6 * 4 * 0.2)


def test_errors_and_missing_runs_are_failures_with_remaining_time_lost(tmp_path):
    root = tmp_path / "runs"
    design(root)
    drop(root, "w0_d9_z9_learned")
    write_run(root, 0, 9, "z9", "learned", passed=("a1", "a2"), status="error", error_t=500.0, lost=20.0)
    drop(root, "w1_d10_z14_learned")                               # missing run
    drop(root, "w2_d9_z9_learned")                                 # summary says 0 trips, log has one: error
    write_run(root, 2, 9, "z9", "learned", passed=("a1",), trip="a2", counts_override=(0, 0, 0))
    rep = evaluate(root, tmp_path)
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    e = ms["w0_d9_z9_learned"]
    assert e["error"] and e["Y"] == 0 and e["T_s"] == 720.0 and e["press_time_lost_s"] == pytest.approx(20.0 + 220.0)
    m = ms["missing:w1_d10_z14_learned"]
    assert m["error"] and m["Y"] == 0 and m["press_time_lost_s"] == 720.0
    assert ms["w2_d9_z9_learned"]["error"]
    reasons = {x["run_id"]: x["reason"] for x in rep["errored_missions"]}
    assert "status error" in reasons["w0_d9_z9_learned"] and reasons["missing:w1_d10_z14_learned"] == "missing run"
    assert "!= log rows" in reasons["w2_d9_z9_learned"]
    assert len(rep["primary_metric"]["missions"]) == 6 * 2 * 2 * 3          # never dropped


@pytest.mark.parametrize("override", [{"passed": 0}, {"attempted": 3}, {"declined": 1}])
def test_summary_checkpoint_counts_must_match_the_log(tmp_path, override):
    root = tmp_path / "runs"
    design(root)
    drop(root, "w0_d9_z9_learned")
    write_run(root, 0, 9, "z9", "learned", passed=("a1", "a2"), summary_override=override)
    rep = evaluate(root, tmp_path)
    reasons = {x["run_id"]: x["reason"] for x in rep["errored_missions"]}
    assert "(attempted, passed, declined)" in reasons["w0_d9_z9_learned"]
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w0_d9_z9_learned"]["Y"] == 0 and ms["w0_d9_z9_learned"]["press_time_lost_s"] == 720.0


def test_declined_and_failed_checkpoints_are_consistent(tmp_path):
    root = tmp_path / "runs"
    design(root)
    drop(root, "w0_d9_z9_learned")
    write_run(root, 0, 9, "z9", "learned", passed=("a1",), failed=("a2",), declined=("a3",))
    rep = evaluate(root, tmp_path)
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    m = ms["w0_d9_z9_learned"]
    assert not m["error"] and m["Y"] == 1
    assert {c["checkpoint"]: c["outcome"] for c in m["checkpoints"]} == {"a1": "passed", "a2": "failed",
                                                                         "a3": "declined"}


def test_run_id_naming_another_cell_is_an_error(tmp_path):
    root = tmp_path / "runs"
    design(root)
    drop(root, "w0_d9_z9_learned")
    write_run(root, 0, 9, "z9", "learned", passed=("a1", "a2"), rid="w9_d9_z9_learned")   # summary: world 0
    rep = evaluate(root, tmp_path)
    reasons = {x["run_id"]: x["reason"] for x in rep["errored_missions"]}
    assert "names cell" in reasons["w9_d9_z9_learned"]


def test_infra_invalid_reruns_repeat_all_conditions(tmp_path):
    root = tmp_path / "runs"
    design(root)
    for c in ("static", "learned", "reactive"):                   # attempt 1 of (w0, d9, z9): all conditions re-run
        write_run(root, 0, 9, "z9", c, passed=("a1", "a2", "a3"), attempt=1)
    inv = tmp_path / "invalid.json"
    inv.write_text(json.dumps({"invalid": ["w0_d9_z9_static", "w3_d10_z14_reactive"]}), encoding="utf-8")
    rep = evaluate(root, tmp_path, invalid=inv)
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w0_d9_z9_learned__a1"]["Y"] == 3 and "w0_d9_z9_learned" not in ms   # the whole cell is re-run
    assert ms["w0_d9_z9_static__a1"]["Y"] == 3
    assert rep["reruns_per_condition"] == {"static": 1, "learned": 1, "reactive": 1}
    assert ms["w3_d10_z14_reactive"]["error"]                     # no re-run: scored as a failure, flagged
    assert rep["unresolved_cells"] == ["w3_d10_z14"]
    first = {(w["world"], w["condition"]): w for w in rep["first_attempt_per_world"]}
    assert first[(0, "learned")]["sum_Y"] < next(w for w in rep["primary_metric"]["per_world"]
                                                  if (w["world"], w["condition"]) == (0, "learned"))["sum_Y"]


def test_crashed_attempt_without_summary_moves_to_the_clean_rerun(tmp_path):
    """Audit S1b-eval #1: a container that crashed before summary.json is found, placed by its run_id and the
    invalid list, and the clean __a1 re-run of all conditions is scored."""
    root = tmp_path / "runs"
    design(root)
    (root / "w0_d9_z9_learned" / "summary.json").unlink()
    for c in ("static", "learned", "reactive"):
        write_run(root, 0, 9, "z9", c, passed=("a1", "a2", "a3"), attempt=1)
    inv = tmp_path / "invalid.json"
    inv.write_text(json.dumps({"invalid": ["w0_d9_z9_learned"]}), encoding="utf-8")
    rep = evaluate(root, tmp_path, invalid=inv)
    assert rep["reruns_per_condition"] == {"static": 1, "learned": 1, "reactive": 1}
    assert rep["errored_missions"] == [] and rep["unresolved_cells"] == []
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w0_d9_z9_learned__a1"]["Y"] == 3 and ms["w0_d9_z9_reactive__a1"]["Y"] == 3
    first = {m["run_id"] for m in rep["primary_metric"]["missions"]}
    assert "w0_d9_z9_learned" not in first and "missing:w0_d9_z9_learned" not in first


def test_invalid_id_without_any_directory_is_placed_by_its_record(tmp_path):
    root = tmp_path / "runs"
    design(root)
    drop(root, "w1_d10_z14_static")                                # the container never wrote anything
    for c in ("static", "learned", "reactive"):
        write_run(root, 1, 10, "z14", c, passed=("b1", "b2"), attempt=1)
    inv = tmp_path / "invalid.json"
    rec = {"run_id": "tag-0017", "world": 1, "day": 10, "zone": "z14", "condition": "static", "split": "test",
           "engine": "double"}
    inv.write_text(json.dumps({"invalid": [rec]}), encoding="utf-8")
    rep = evaluate(root, tmp_path, invalid=inv)
    # tag-0017 has no __a suffix: it is attempt 0 of the cell, which is therefore re-run as a whole
    assert rep["reruns_per_condition"] == {"static": 1, "learned": 1, "reactive": 1}
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w1_d10_z14_static__a1"]["Y"] == 2 and not any(r.startswith("missing:") for r in ms)
    with pytest.raises(E.EvalError, match="cell keys"):
        inv.write_text(json.dumps([{"run_id": "x", "world": 1}]), encoding="utf-8")
        E.load_invalid(inv)


def test_crashed_run_not_listed_invalid_is_an_error_failure(tmp_path):
    root = tmp_path / "runs"
    design(root)
    (root / "w2_d9_z14_reactive" / "summary.json").unlink()        # crashed after the first command: valid
    rep = evaluate(root, tmp_path)
    reasons = {x["run_id"]: x["reason"] for x in rep["errored_missions"]}
    assert "summary.json unreadable" in reasons["w2_d9_z14_reactive"]
    ms = {m["run_id"]: m for m in rep["primary_metric"]["missions"]}
    assert ms["w2_d9_z14_reactive"]["Y"] == 0 and ms["w2_d9_z14_reactive"]["press_time_lost_s"] == 720.0


def test_a_missing_world_is_scored_as_failures_not_dropped(tmp_path):
    """Audit S1b-eval #4: every w5_* run is absent; world 5 stays in the design and scores missing failures."""
    root = tmp_path / "runs"
    design(root)
    for d in sorted(root.glob("w5_*")):
        shutil.rmtree(d)
    rep = evaluate(root, tmp_path)
    g1 = rep["gatekeeping"][0]
    assert g1["worlds"] == list(range(6)) and g1["d_w"][5] == 0.0
    missing = [x["run_id"] for x in rep["errored_missions"]]
    assert len(missing) == 2 * 2 * 3 and all(r.startswith("missing:w5_") for r in missing)
    assert len(rep["primary_metric"]["missions"]) == 6 * 2 * 2 * 3
    assert g1["p"] == pytest.approx(2 / 64)                        # K = 6 kept: world 5's d = 0 ties


def test_test_design_is_explicit_and_sealed(tmp_path):
    root = tmp_path / "runs"
    design(root)
    tasks = write_tasks(tmp_path / "tasks.json")
    with pytest.raises(E.EvalError, match="--worlds"):
        E.run_eval(root, tasks)                                    # worlds are never taken from the runs
    with pytest.raises(E.EvalError, match="K = 16"):
        E.run_eval(root, tasks, worlds=range(6))
    with pytest.raises(E.EvalError, match="days"):
        E.run_eval(root, tasks, worlds=range(16), days=[9])
    with pytest.raises(E.EvalError, match="all conditions"):
        E.run_eval(root, tasks, worlds=range(16), conditions=("static", "learned", "reactive"))
    write_run(root, 7, 9, "z9", "learned")                         # a test run outside the design: refused
    with pytest.raises(E.EvalError, match="outside the test design"):
        evaluate(root, tmp_path)
    shutil.rmtree(root / "w7_d9_z9_learned")
    write_run(root, 0, 3, "z9", "learned")                         # day 3 has no robot run in a test world
    with pytest.raises(E.EvalError, match="outside the test design"):
        evaluate(root, tmp_path)


def test_sealed_design_scores_all_320_cells(tmp_path):
    """The fixed sealed design: 16 worlds x 2 days x 2 zones x 5 conditions; absent conditions are missing failures."""
    root = tmp_path / "runs"
    design(root, worlds=range(16), conds=("static", "learned", "reactive"))
    rep = E.run_eval(root, write_tasks(tmp_path / "tasks.json"), worlds=range(16))
    assert rep["design"]["fixed"] and rep["design"]["conditions"] == list(E.CONDITIONS)
    assert len(rep["primary_metric"]["missions"]) == 320
    miss = {x["run_id"].split("_")[-1] for x in rep["errored_missions"]}
    assert miss == {"map", "oracle"} and len(rep["errored_missions"]) == 128
    assert rep["gatekeeping"][0]["p"] == pytest.approx(1 / 2 ** 16)


def test_non_primary_runs_and_other_splits_are_not_pooled(tmp_path):
    root = tmp_path / "runs"
    design(root)
    write_run(root / "sweep", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), variant="reset_sweep")
    write_run(root / "dev", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), split="dev")
    write_run(root / "isaac", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), engine="isaac", jsonl=True)
    rep = evaluate(root, tmp_path)
    assert rep["excluded_non_primary_runs"] == ["w0_d9_z9_learned"]
    assert len(rep["primary_metric"]["missions"]) == 72
    rep_i = evaluate(root / "isaac", tmp_path, engine="isaac", worlds=[0])   # an Isaac run with log.jsonl
    mi = {m["run_id"]: m for m in rep_i["primary_metric"]["missions"]}
    assert mi["w0_d9_z9_learned"]["Y"] == 3 and rep_i["engine"] == "isaac"
    assert mi["missing:w0_d9_z9_reactive"]["error"]


def test_static_beats_learned_flag_and_failed_primary_stops_the_chain(tmp_path):
    root = tmp_path / "runs"
    design(root, learned_extra=0)                                  # learned = reactive = static
    rep = evaluate(root, tmp_path)
    g = rep["gatekeeping"]
    assert g[0]["p"] == 1.0 and not g[0]["claimed"]
    assert g[1]["p"] == pytest.approx(1 / 64) and g[1]["reject"] and not g[1]["claimed"]   # not reached
    assert rep["reported_outside_gate"]["learned_minus_static"]["static_matched_or_beat_learned"]


def test_oracle_share_reported_only_when_the_oracle_gain_is_credibly_positive(tmp_path):
    root = tmp_path / "runs"
    design(root, conds=("static", "learned", "reactive"))
    for w in range(6):
        for day in (9, 10):
            write_run(root, w, day, "z9", "oracle", passed=("a1", "a2", "a3"))
            write_run(root, w, day, "z14", "oracle", passed=("b1",))
    four = ("static", "learned", "reactive", "oracle")
    rep = evaluate(root, tmp_path, conditions=four)
    o = rep["reported_outside_gate"]["oracle_share"]
    assert o["reported"] and o["share"] == pytest.approx(0.5)       # learned +1 of the oracle's +2 per z9 mission
    root2 = tmp_path / "runs2"
    design(root2, conds=four, learned_extra=1)                     # oracle = reactive
    rep2 = evaluate(root2, tmp_path, conditions=four)
    o2 = rep2["reported_outside_gate"]["oracle_share"]
    assert not o2["reported"] and "learned_minus_reactive" in o2


# ------------------------------------------------------------------------------------------------ gate 3 / truth slot


def _stream(path: Path, world: int, day: int, zone: str, rng, T: int = 40, miss_p: float = 0.0,
            split: str = "test") -> None:
    truth = rng.random((T, 8, 8)) < 0.15
    pred = truth.copy()
    pred &= rng.random(pred.shape) >= miss_p                       # drop some occupied cells from the prediction
    # prediction k is built from data observed at t_k - 0.5 s = what the robot holds at t_k (age 0.5 s)
    np.savez(path, t=np.arange(T) * 0.5, truth=truth, t_pred=np.arange(T) * 0.5 - 0.5, pred=pred,
             age_s=np.full(T, 0.5), slice_id=np.zeros(T, np.int64), world=world, day=day, zone=zone,
             cell_m=0.25, split=split)


def _streams(tmp_path: Path, worlds=range(6), miss_p: float = 0.0) -> list[Path]:
    rng = np.random.default_rng(5)
    out = []
    for w in worlds:
        for day in (9, 10):
            for z in ORDERS:
                p = tmp_path / "streams" / f"w{w}_d{day}_{z}.realised_state.npz"
                p.parent.mkdir(exist_ok=True)
                _stream(p, w, day, z, rng, miss_p=miss_p)            # the map covers every occupied cell
                out.append(p)
    return out


def test_eval_metrics_counts_equal_the_calibration_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    rng = np.random.default_rng(1)
    p = tmp_path / "w0_d9_z9.realised_state.npz"
    _stream(p, 0, 9, "z9", rng, miss_p=0.5)
    d = EM.load_stream(p)
    margin = Margin(0.25, 0.25)
    idx = EM.align_by_age(d["t"], d["age_s"], d["t_pred"])
    assert np.array_equal(idx, CM.align_by_age(d["t"], d["age_s"], d["t_pred"]))
    occ, miss = EM.tick_counts(d["truth"], d["pred"], idx, margin, 0.25)
    occ2, miss2 = CM.tick_counts(d["truth"], d["pred"], idx, 0.25, (0.0, 0.25))
    assert np.array_equal(occ, occ2) and np.array_equal(miss, miss2[:, 1])
    log = (tmp_path / "access.log").read_text(encoding="utf-8")
    assert "twin/eval_metrics.py" in log and "realised_state" in log


def test_truth_stays_closed_until_every_run_is_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    root = tmp_path / "runs"
    write_run(root, 0, 9, "z9", "learned")
    p = tmp_path / "w0_d9_z9.realised_state.npz"
    _stream(p, 0, 9, "z9", np.random.default_rng(2))
    open_run = root / "w0_d9_z9_static"
    open_run.mkdir()
    (open_run / "obs_v2.jsonl").write_text("", encoding="utf-8")   # a mission still running: no summary yet
    kw: dict[str, Any] = dict(split="test", days=(9, 10))
    with pytest.raises(EM.NotClosed):
        EM.h3_tick_counts([p], root, Margin(0.25, 0.25), **kw)
    assert not (tmp_path / "access.log").exists()                 # nothing was opened
    EM.h3_tick_counts([p], root, Margin(0.25, 0.25), invalid=["w0_d9_z9_static"], **kw)   # crashed and listed: ok
    torn = root / "w0_d9_z9_learned" / "log.parquet"
    torn.write_bytes(torn.read_bytes()[:-8])                       # a log still being written
    with pytest.raises(EM.NotClosed, match="PAR1"):
        EM.h3_tick_counts([p], root, Margin(0.25, 0.25), invalid=["w0_d9_z9_static"], **kw)
    with pytest.raises(ValueError, match="realised_state"):
        EM.load_stream(tmp_path / "x.npz")


def test_eval_metrics_refuses_dev_and_observer_only_day_streams(tmp_path, monkeypatch):
    """Audit S1b-eval #2 (truth side): only test-split streams of the scored days are counted."""
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    root = tmp_path / "runs"
    write_run(root, 0, 9, "z9", "learned")
    rng = np.random.default_rng(4)
    dev = tmp_path / "w0_d9_z9_dev.realised_state.npz"
    _stream(dev, 0, 9, "z9", rng, split="dev")
    day3 = tmp_path / "w0_d3_z9.realised_state.npz"
    _stream(day3, 0, 3, "z9", rng)
    with pytest.raises(ValueError, match="evaluated split"):
        EM.h3_tick_counts([dev], root, Margin(0.25, 0.25), split="test", days=(9, 10))
    with pytest.raises(ValueError, match="not a scored day"):
        EM.h3_tick_counts([day3], root, Margin(0.25, 0.25), split="test", days=(9, 10))


def test_gate3_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    root = tmp_path / "runs"
    design(root)
    body = EM.h3_tick_counts(_streams(tmp_path), root, Margin(0.25, 0.25), split="test", days=(9, 10))
    out = tmp_path / "h3.json"
    out.write_text(json.dumps(body), encoding="utf-8")
    mf = write_margin(tmp_path / "m7.json")
    rep = evaluate(root, tmp_path, h3_counts=out, margin_file=mf)
    g3 = rep["gatekeeping"][2]
    assert g3["shares"] == pytest.approx([0.0] * 6) and g3["p"] == pytest.approx(1 / 64) and g3["claimed"]
    assert g3["episode_s"] == 720.0 and g3["margin_m"] == 0.25
    with pytest.raises(E.EvalError, match="margin-file"):
        evaluate(root, tmp_path, h3_counts=out)                    # the frozen margin is required
    with pytest.raises(E.EvalError, match="frozen margin"):
        evaluate(root, tmp_path, h3_counts=out, margin_file=write_margin(tmp_path / "m2.json", margin_m=0.5))
    with pytest.raises(E.EvalError, match="episode_s"):
        evaluate(root, tmp_path, h3_counts=out, margin_file=write_margin(tmp_path / "m3.json", episode_s=None))
    body["streams"] = [s for s in body["streams"] if s["world"] != 5]  # a world without counts: not evaluable
    out.write_text(json.dumps(body), encoding="utf-8")
    g3 = evaluate(root, tmp_path, h3_counts=out, margin_file=mf)["gatekeeping"][2]
    assert g3["status"] == "not evaluable" and not g3["reject"] and "w5_d9_z9" in g3["missing_streams"]
    body["streams"] = [s for s in body["streams"] if (s["world"], s["day"], s["zone"]) != (4, 10, "z14")]
    out.write_text(json.dumps(body), encoding="utf-8")             # one (day, zone) cell missing: not evaluable
    g3 = evaluate(root, tmp_path, h3_counts=out, margin_file=mf)["gatekeeping"][2]
    assert math.isnan(g3["shares"][4]) and "w4_d10_z14" in g3["missing_streams"]


def _count(world: int, day: int, zone: str, miss: int, split: str = "test", T: int = 10) -> dict:
    return {"stream": f"w{world}_d{day}_{zone}", "world": world, "day": day, "zone": zone, "split": split,
            "cell_m": 0.25, "margin_m": 0.25, "t": (np.arange(T) * 0.5).tolist(), "slice_id": [0] * T,
            "occ": [4] * T, "miss": [miss] * T, "valid": [True] * T}


def test_h3_refuses_dev_streams_and_observer_only_days():
    """Audit S1b-eval #2 (EVAL side): the probe's day-3 / dev-split streams with full misses are refused, never
    pooled into the test-day share."""
    kw: dict[str, Any] = dict(split="test", worlds=[0], days=(9, 10), zones=["z9"], episode_s=720.0, cell_m=0.25)
    good = [_count(0, 9, "z9", 0), _count(0, 10, "z9", 0)]
    assert E.h3_world_shares(good, 0.25, **kw)["shares"] == {0: 0.0}
    with pytest.raises(E.EvalError, match="not a scored day"):
        E.h3_world_shares(good + [_count(0, 3, "z9", 4)], 0.25, **kw)
    with pytest.raises(E.EvalError, match="evaluated split"):
        E.h3_world_shares(good + [_count(0, 9, "z9", 4, split="dev")], 0.25, **kw)
    with pytest.raises(E.EvalError, match="design world"):
        E.h3_world_shares(good + [_count(3, 9, "z9", 4)], 0.25, **kw)
    with pytest.raises(E.EvalError, match="two H3 streams"):
        E.h3_world_shares(good + [_count(0, 9, "z9", 4)], 0.25, **kw)


def test_h3_share_matches_unit_mean_miss_share(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    rng = np.random.default_rng(7)
    p = tmp_path / "w3_d9_z9.realised_state.npz"
    _stream(p, 3, 9, "z9", rng, T=3000, miss_p=0.3)               # 1500 s: 3 units of 720 s
    c = EM.stream_counts(p, Margin(0.0, 0.25))
    d = EM.load_stream(p)
    units = CM.units_from_stream(d["t"], d["truth"], d["t_pred"], d["pred"], d["age_s"], d["slice_id"], world=3,
                                 day=9, cell_m=0.25, margins_m=(0.0,))
    want = CM.unit_mean_miss_share(units, 0.0)
    kw: dict[str, Any] = dict(split="test", worlds=[3], days=[9], zones=["z9"], cell_m=0.25)
    got = E.h3_world_shares([dict(c, margin_m=0.0)], 0.0, episode_s=720.0, **kw)
    assert got["shares"] == {3: pytest.approx(want)} and len(units.occ) == 3
    # the unit length comes from the margin file: 360 s units cut the same stream into 5 units
    units5 = CM.units_from_stream(d["t"], d["truth"], d["t_pred"], d["pred"], d["age_s"], d["slice_id"], world=3,
                                  day=9, cell_m=0.25, margins_m=(0.0,), episode_s=360.0)
    got5 = E.h3_world_shares([dict(c, margin_m=0.0)], 0.0, episode_s=360.0, **kw)
    assert got5["shares"] == {3: pytest.approx(CM.unit_mean_miss_share(units5, 0.0))} and len(units5.occ) == 5


# ------------------------------------------------------------------------------------------------ stream builder

T0 = 3600.0
PLATE = (slice(3, 7), slice(3, 7))   # P9a (footprint centre (6.25, 7.25)) +- 0.4 m on the grid x0 5.0, z0 6.0, 0.25


def _realised(d: Path, rid: str, world: int, day: int, n: int) -> None:
    lines = [json.dumps({"kind": "header", "contract": "twin.realised_state/1", "run_id": rid, "world": world,
                         "day": day, "hz": 2.0, "t0_s": T0, "zones": ["z9", "z14"], "presses": ["P9a", "P14a"]})]
    for k in range(n):
        running = k < n - 4                                         # the last 4 ticks: stopped at TDC (no truth)
        lines.append(json.dumps({"kind": "tick", "t": T0 + 0.5 * k, "movers": [], "presses": [
            {"id": "P9a", "state": "running" if running else "stopped", "ram": 0.3 if running else 0.0,
             "stroke": False, "freeze": False},
            {"id": "P14a", "state": "running", "ram": 0.5, "stroke": True, "freeze": False}]}))
    (d / EM.REALISED).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _held(d: Path, n: int) -> None:
    t_use = T0 + 0.5 * np.arange(n)
    pred0 = np.zeros((n, 10, 10), bool)
    pred0[::2][(slice(None),) + PLATE] = True                      # even predictions cover the plate, odd ones miss it
    np.savez(d / EM.HELD_MAP, press_ids=np.array(["P9a", "P14a"]), res=0.25, x0=np.array([5.0, 32.0]),
             z0=np.array([6.0, 4.0]), nx=np.array([10, 8]), nz=np.array([10, 8]), t_pred=t_use - 0.5,
             pred_0=pred0, pred_1=np.zeros((n, 8, 8), bool), t_use=t_use, age_s=np.full(n, 0.5),
             slice_id=np.zeros(n, np.int64))


def test_truth_grid_rule():
    from twin.schemas.runs import RealisedTick

    tk = RealisedTick.model_validate({"kind": "tick", "t": 0.0, "movers": [], "presses": [
        {"id": "P9a", "state": "stopped", "ram": 0.0, "stroke": False, "freeze": False}]})
    assert not EM.truth_grid(tk, "P9a", (6.25, 7.25), 5.0, 6.0, 10, 10, 0.25).any()     # stopped at TDC
    tk.presses[0].ram = 0.2                                        # stopped below TDC: occupied
    g = EM.truth_grid(tk, "P9a", (6.25, 7.25), 5.0, 6.0, 10, 10, 0.25)
    want = np.zeros((10, 10), bool)
    want[PLATE] = True
    assert np.array_equal(g, want)


def test_stream_builder_end_to_end_into_h3(tmp_path, monkeypatch):
    """Audit S1b-eval #3: the gate-3 streams are built from closed run dirs (realised_state.jsonl through open_truth
    + the observer's held map) and counted at the frozen margin, then scored by twin_eval."""
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    root = tmp_path / "runs"
    design(root, worlds=[0])
    n = 20
    for z in ("z9", "z14"):
        for day in (9, 10):
            d = root / f"w0_d{day}_{z}_learned"
            _realised(d, d.name, 0, day, n)
            _held(d, n)
    tasks = write_tasks(tmp_path / "tasks.json")
    kw: dict[str, Any] = dict(worlds=[0], allow_partial_design=True)
    open_run = root / "w0_d9_z9_static" / "summary.json"
    saved = open_run.read_text(encoding="utf-8")
    open_run.unlink()                                              # one run still open: no truth is read
    with pytest.raises(EM.NotClosed):
        EM.h3_from_runs(root, tasks, SITE, Margin(0.0, 0.25), **kw)
    assert not (tmp_path / "access.log").exists()
    open_run.write_text(saved, encoding="utf-8")
    body = EM.h3_from_runs(root, tasks, SITE, Margin(0.0, 0.25), **kw)
    assert body["skipped"] == [] and len(body["streams"]) == 4
    s = next(c for c in body["streams"] if (c["day"], c["zone"]) == (9, "z9"))
    occ, miss = np.array(s["occ"]), np.array(s["miss"])
    assert np.array_equal(occ, [16] * (n - 4) + [0] * 4)           # only the zone's press (P9a) is scored
    assert np.array_equal(miss, [16 if k % 2 else 0 for k in range(n - 4)] + [0] * 4)
    z14 = next(c for c in body["streams"] if (c["day"], c["zone"]) == (9, "z14"))
    assert sum(z14["occ"]) > 0 and z14["miss"] == z14["occ"]       # P14a runs, its held grid is empty
    log = (tmp_path / "access.log").read_text(encoding="utf-8")
    assert log.count("twin/eval_metrics.py") == 4 and "realised_state.jsonl" in log
    out = tmp_path / "h3.json"
    out.write_text(json.dumps(body), encoding="utf-8")
    rep = E.run_eval(root, tasks, conditions=("static", "learned", "reactive"), h3_counts=out,
                     margin_file=write_margin(tmp_path / "m7.json", margin_m=0.0), **kw)
    g3 = rep["gatekeeping"][2]
    assert g3["shares"] == pytest.approx([(0.5 + 0.5 + 1.0 + 1.0) / 4])
    # a cell without a held map gets no stream (skipped), so its world is not evaluable downstream
    (root / "w0_d10_z14_learned" / EM.HELD_MAP).unlink()
    body2 = EM.h3_from_runs(root, tasks, SITE, Margin(0.0, 0.25), **kw)
    assert [x["cell"] for x in body2["skipped"]] == ["w0_d10_z14"] and len(body2["streams"]) == 3


# ------------------------------------------------------------------------------------------------ gate 4


def test_h4_small_group_rule_and_bootstrap():
    frames = [{"camera": 9, "group": g, "tp": tp, "n_gt": 10} for g, tp in (("a", 9), ("b", 8), ("c", 9), ("d", 7))]
    frames += [{"camera": 14, "group": f"g{k}", "tp": 1, "n_gt": 1} for k in range(12)]
    r = E.h4_recall(frames, 0.7)
    assert r["small_group_rule"] and r["groups_per_camera"] == {"9": 4, "14": 12}
    assert r["lowest_group_recall"] == 0.7 and r["reject"]
    assert r["ci_label"] == "ignores clustering (only 16 groups)"
    lo, hi = r["clopper_pearson_95"]
    assert lo == pytest.approx(stats.beta.ppf(0.025, 45, 52 - 45 + 1))
    assert not E.h4_recall(frames, 0.75)["reject"]                # group d (0.7) is below r0
    big = [{"camera": c, "group": f"{c}{k}", "tp": 9, "n_gt": 10} for c in (9, 14) for k in range(6)]
    rb = E.h4_recall(big, 0.85)
    assert not rb["small_group_rule"] and rb["bootstrap_lower_95"] == pytest.approx(0.9) and rb["reject"]
    with pytest.raises(E.EvalError):
        E.h4_recall([{"camera": 9, "group": "a", "tp": 3, "n_gt": 2}], 0.5)


def test_h4_gate_in_report(tmp_path):
    root = tmp_path / "runs"
    design(root)
    fr = tmp_path / "frames.json"
    fr.write_text(json.dumps([{"camera": 9, "group": "a", "tp": 9, "n_gt": 10}]), encoding="utf-8")
    rep = evaluate(root, tmp_path, recall_frames=fr, r0=0.8)
    g4 = rep["gatekeeping"][3]
    assert g4["reject"] and not g4["claimed"]                      # gate 3 had no input: chain stops there
    with pytest.raises(E.EvalError, match="r0"):
        evaluate(root, tmp_path, recall_frames=fr)


def test_eval_module_names_no_truth_file():
    from twin import truth_guard as TG

    src = (Path(E.__file__)).read_text(encoding="utf-8")
    import ast

    lits = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    body = [s for s in lits if not s.startswith(("EVAL runner",))]
    assert not [s for s in body if any(t in s for t in TG.TRUTH_TOKENS)]


def test_cli_writes_the_report(tmp_path, capsys):
    root = tmp_path / "runs"
    design(root, split="dev")
    out = tmp_path / "out" / "eval.json"
    tasks = str(write_tasks(tmp_path / "t.json"))
    assert E.main(["--runs", str(root), "--tasks", tasks, "--split", "dev", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["contract"] == E.CONTRACT and rep["gatekeeping"][0]["claimed"]
    assert "H1 primary" in capsys.readouterr().out
    with pytest.raises(E.EvalError, match="--worlds"):              # the test split needs the explicit design
        E.main(["--runs", str(root), "--tasks", tasks, "--out", str(out)])
