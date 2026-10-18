"""S1b item 4: the EVAL runner (intercycle/eval/twin_eval.py) and its truth slot (twin/eval_metrics.py) on synthetic
run dirs: scoring, errors as failures, re-runs, the exact sign-flip, gatekeeping, outside-gate reports, gate 3 / 4."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

from intercycle.eval import conformal_margin as CM
from intercycle.eval import twin_eval as E
from intercycle.margin_inflate import Margin
from twin import eval_metrics as EM
from twin.schemas.tables import LogRow, columns

ORDERS = {"z9": ["a1", "a2", "a3"], "z14": ["b1", "b2"]}
SHA = "0" * 64


def write_tasks(p: Path) -> Path:
    zones = [{"zone": z, "dock": {"x": 0, "z": 0, "yaw": 0}, "mission_order": cps} for z, cps in ORDERS.items()]
    p.write_text(json.dumps({"contract": "twin.tasks/1", "site_id": "s", "zones": zones, "checkpoints": []}),
                 encoding="utf-8")
    return p


def write_run(root: Path, world: int, day: int, zone: str, cond: str, *, passed: tuple[str, ...] = (),
              declined: tuple[str, ...] = (), failed: tuple[str, ...] = (), trip: str | None = None,
              hold: str | None = None, incident: bool = False, contacts: int = 0, lost: float = 0.0,
              status: str = "ok", error_t: float | None = None, attempt: int = 0, split: str = "test",
              engine: str = "double", variant: str = "primary", jsonl: bool = False, counts_override=None) -> Path:
    rid = f"w{world}_d{day}_{zone}_{cond}" + (f"__a{attempt}" if attempt else "")
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
         "attempted": len(att), "passed": len(passed), "declined": len(declined), "robot_caused_press_stops": trips,
         "holds": holds, "incidents": incs, "press_time_lost_s": lost, "path_length_m": 10.0,
         "code": {"head": "abc"}, "variant": variant}
    if variant == "reset_sweep":
        s["reset_override_s"] = 60.0
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    return d


def design(root: Path, worlds=range(6), conds=("static", "learned", "reactive"), learned_extra: int = 1) -> None:
    """Complete design: per (world, day, zone): static passes 1, reactive 1, learned 1 + learned_extra (z9 only)."""
    for w in worlds:
        for day in (9, 10):
            for z, cps in ORDERS.items():
                for c in conds:
                    n = 1 + (learned_extra if c == "learned" and z == "z9" else 0)
                    write_run(root, w, day, z, c, passed=tuple(cps[:n]))


def evaluate(root: Path, tmp: Path, **kw) -> dict:
    return E.run_eval(root, write_tasks(tmp / "tasks.json"), **kw)


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


def test_trip_hold_incident_zero_y_and_feed_h2(tmp_path):
    root = tmp_path / "runs"
    design(root)
    for w in range(6):            # replace world w's day-9 z9 learned run
        d = root / f"w{w}_d9_z9_learned"
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
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
    d = root / "w0_d9_z9_learned"
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    write_run(root, 0, 9, "z9", "learned", passed=("a1", "a2"), status="error", error_t=500.0, lost=20.0)
    d = root / "w1_d10_z14_learned"                                # missing run
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    d = root / "w2_d9_z9_learned"                                  # summary says 0 trips, log has one: error
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
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


def test_non_primary_runs_and_other_splits_are_not_pooled(tmp_path):
    root = tmp_path / "runs"
    design(root)
    write_run(root / "sweep", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), variant="reset_sweep")
    write_run(root / "dev", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), split="dev")
    write_run(root / "isaac", 0, 9, "z9", "learned", passed=("a1", "a2", "a3"), engine="isaac", jsonl=True)
    rep = evaluate(root, tmp_path)
    assert rep["excluded_non_primary_runs"] == ["w0_d9_z9_learned"]
    assert len(rep["primary_metric"]["missions"]) == 72
    rep_i = evaluate(root / "isaac", tmp_path, engine="isaac")      # an Isaac run with log.jsonl
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
    rep = evaluate(root, tmp_path)
    o = rep["reported_outside_gate"]["oracle_share"]
    assert o["reported"] and o["share"] == pytest.approx(0.5)       # learned +1 of the oracle's +2 per z9 mission
    root2 = tmp_path / "runs2"
    design(root2, conds=("static", "learned", "reactive", "oracle"), learned_extra=1)   # oracle = reactive
    rep2 = E.run_eval(root2, write_tasks(tmp_path / "t2.json"))
    o2 = rep2["reported_outside_gate"]["oracle_share"]
    assert not o2["reported"] and "learned_minus_reactive" in o2


# ------------------------------------------------------------------------------------------------ gate 3 / truth slot


def _stream(path: Path, world: int, day: int, zone: str, rng, T: int = 40, miss_p: float = 0.0) -> None:
    truth = rng.random((T, 8, 8)) < 0.15
    pred = truth.copy()
    pred &= rng.random(pred.shape) >= miss_p                       # drop some occupied cells from the prediction
    # prediction k is built from data observed at t_k - 0.5 s = what the robot holds at t_k (age 0.5 s)
    np.savez(path, t=np.arange(T) * 0.5, truth=truth, t_pred=np.arange(T) * 0.5 - 0.5, pred=pred,
             age_s=np.full(T, 0.5), slice_id=np.zeros(T, np.int64), world=world, day=day, zone=zone,
             cell_m=0.25, split="test")


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
    with pytest.raises(EM.NotClosed):
        EM.h3_tick_counts([p], root, Margin(0.25, 0.25))
    assert not (tmp_path / "access.log").exists()                 # nothing was opened
    EM.h3_tick_counts([p], root, Margin(0.25, 0.25), invalid=["w0_d9_z9_static"])   # crashed and listed: ok
    torn = root / "w0_d9_z9_learned" / "log.parquet"
    torn.write_bytes(torn.read_bytes()[:-8])                       # a log still being written
    with pytest.raises(EM.NotClosed, match="PAR1"):
        EM.h3_tick_counts([p], root, Margin(0.25, 0.25), invalid=["w0_d9_z9_static"])
    with pytest.raises(ValueError, match="realised_state"):
        EM.load_stream(tmp_path / "x.npz")


def test_gate3_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "access.log"))
    root = tmp_path / "runs"
    design(root)
    rng = np.random.default_rng(5)
    streams = []
    for w in range(6):
        for day in (9, 10):
            for z in ORDERS:
                p = tmp_path / "streams" / f"w{w}_d{day}_{z}.realised_state.npz"
                p.parent.mkdir(exist_ok=True)
                _stream(p, w, day, z, rng, miss_p=0.0)            # the map covers every occupied cell
                streams.append(p)
    body = EM.h3_tick_counts(streams, root, Margin(0.25, 0.25))
    out = tmp_path / "h3.json"
    out.write_text(json.dumps(body), encoding="utf-8")
    rep = evaluate(root, tmp_path, h3_counts=out)
    g3 = rep["gatekeeping"][2]
    assert g3["shares"] == pytest.approx([0.0] * 6) and g3["p"] == pytest.approx(1 / 64) and g3["claimed"]
    shares = E.h3_world_shares(E.load_h3_counts(out), 0.25)
    assert shares == {w: 0.0 for w in range(6)}
    body["streams"] = [s for s in body["streams"] if s["world"] != 5]  # a world without counts: not evaluable
    out.write_text(json.dumps(body), encoding="utf-8")
    g3 = evaluate(root, tmp_path, h3_counts=out)["gatekeeping"][2]
    assert g3["status"] == "not evaluable" and not g3["reject"]


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
    got = E.h3_world_shares([dict(c, margin_m=0.0)], 0.0)
    assert got == {3: pytest.approx(want)} and len(units.occ) == 3


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
    design(root)
    out = tmp_path / "out" / "eval.json"
    assert E.main(["--runs", str(root), "--tasks", str(write_tasks(tmp_path / "t.json")), "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["contract"] == E.CONTRACT and rep["gatekeeping"][0]["claimed"]
    assert "H1 primary" in capsys.readouterr().out
