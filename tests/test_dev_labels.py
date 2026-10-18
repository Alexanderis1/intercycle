"""The machine-readable scored DEV label set (twin/dev_labels.py, labels/dev/scored.json; lane S4-labels fix round 1)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from twin import dev_labels as DL
from twin.schemas.labels import SampleItem, sample_items_sha256

DEV = Path(__file__).resolve().parents[1] / "labels" / "dev"
SHA = "a" * 64


def test_committed_scored_set_loads_and_holds_20_distinct_windows_and_40_frames():
    s = DL.load_scored(DEV)
    assert len(s.frames) == 40
    assert len(s.windows) == 20
    rows = [r for r in s.doc["stroke_windows"] if r["status"] == "scored"]
    for press in ("P9a", "P9b"):
        shas = [r["clip_sha256"] for r in rows if r["press"] == press]
        assert len(shas) == 10 and len(set(shas)) == 10
    assert all(w.strokes for w in s.windows)
    dup = [r for r in s.doc["frames"] if r["status"] == "duplicate"]
    assert [(r["sample_id"], r["dup_of"]) for r in dup] == [("dev-f-c9-14", "dev-f-c9-03")]


def test_committed_statuses_are_derived_from_the_labels_not_typed():
    doc = json.loads((DEV / "scored.json").read_bytes())
    sha_of = {r["clip_id"]: r["clip_sha256"] for r in doc["frames"] + doc["stroke_windows"]}
    parts = [DL.Part("labels.json", (DEV / "labels.json").read_bytes(), "sample.json",
                     (DEV / "sample.json").read_bytes()),
             DL.Part("labels_s1.json", (DEV / "labels_s1.json").read_bytes(), "sample_s1.json",
                     (DEV / "sample_s1.json").read_bytes(), supplement=True)]
    assert DL.build_scored(parts, sha_of) == doc
    labs = {w.sample_id: w for p in parts for w in p.labels.stroke_windows}
    for r in doc["stroke_windows"]:
        if r["status"] == "ambiguous":
            assert (labs[r["sample_id"]].note or "").startswith("NOT SCORABLE:") and not labs[r["sample_id"]].strokes
        if r["status"] == "not_running":
            assert not labs[r["sample_id"]].strokes


def test_load_scored_fails_closed_on_a_changed_label_file(tmp_path):
    for f in DEV.iterdir():
        shutil.copy(f, tmp_path / f.name)
    p = tmp_path / "labels.json"
    p.write_bytes(p.read_bytes().replace(b'"dev-labeller-A"', b'"dev-labeller-B"', 1))
    with pytest.raises(ValueError, match="labels.json changed"):
        DL.load_scored(tmp_path)
    shutil.copy(DEV / "labels.json", p)
    (tmp_path / "scored.sha256").write_text("0" * 64 + "  scored.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scored.sha256"):
        DL.load_scored(tmp_path)


def test_hidden_spans_parse():
    assert DL.hidden_spans("x hidden 0.16-1.30 s and hidden 4.45-5.60 s (covered)") == [[0.16, 1.3], [4.45, 5.6]]
    assert DL.hidden_spans(None) == []


# ---- synthetic label sets -------------------------------------------------------------------------------------

def _win(sid, clip, t1=8.0):
    return SampleItem(sample_id=sid, kind="stroke_window", clip_id=clip, group_id="g", camera=9, split="dev",
                      press="P9b", t0=0.0, t1=t1)


def _part(items, labels, supplement=False, name="labels.json"):
    sl = dict(contract="twin.label_samples/1", split="dev", seed=1, rule="r",
              items=[i.model_dump(mode="json") for i in items], items_sha256=sample_items_sha256(items))
    ls = dict(contract="twin.labels/1", split="dev", labeller_id="t", labeller_kind="claude_agent", blind=True,
              proposals_shown=False, rules_sha256=SHA, sample_list_sha256=sl["items_sha256"], frames=[],
              stroke_windows=[dict(sample_id=i.sample_id, clip_id=i.clip_id, press="P9b", t0=0.0, t1=i.t1,
                                   strokes=labels[i.sample_id][0], note=labels[i.sample_id][1])
                              for i in items if i.sample_id in labels])
    return DL.Part(name, json.dumps(ls).encode(), name.replace("labels", "sample"), json.dumps(sl).encode(),
                   supplement=supplement)


def test_status_rules_and_fail_closed(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 2)
    items = [_win("dev-sw-P9b-01", "c1"), _win("dev-sw-P9b-02", "c2"), _win("dev-sw-P9b-03", "c3"),
             _win("dev-sw-P9b-04", "c4"), _win("dev-sw-P9b-05", "c5")]
    sha = {"c1": "s1", "c2": "s2", "c3": "s1", "c4": "s4", "c5": "s5"}
    labels = {"dev-sw-P9b-01": ([1.0], "ok"), "dev-sw-P9b-02": ([], "press not running: static"),
              "dev-sw-P9b-03": ([1.0], "copy"), "dev-sw-P9b-04": ([], "NOT SCORABLE: ram not seen"),
              "dev-sw-P9b-05": ([2.0], "hidden 3.00-4.00 s")}
    doc = DL.build_scored([_part(items, labels)], sha)
    st = {r["sample_id"]: (r["status"], r["dup_of"]) for r in doc["stroke_windows"]}
    assert st == {"dev-sw-P9b-01": ("scored", None), "dev-sw-P9b-02": ("not_running", None),
                  "dev-sw-P9b-03": ("duplicate", "dev-sw-P9b-01"), "dev-sw-P9b-04": ("ambiguous", None),
                  "dev-sw-P9b-05": ("scored", None)}
    assert doc["stroke_windows"][4]["hidden"] == [[3.0, 4.0]]
    bad = dict(labels, **{"dev-sw-P9b-02": ([], "nothing here")})
    with pytest.raises(ValueError, match="fail closed"):
        DL.build_scored([_part(items, bad)], sha)
    bad = dict(labels, **{"dev-sw-P9b-04": ([1.0], "NOT SCORABLE: but clicked")})
    with pytest.raises(ValueError, match="NOT SCORABLE"):
        DL.build_scored([_part(items, bad)], sha)


def test_supplement_stopping_rule(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 2)
    base = _part([_win("dev-sw-P9b-01", "c1")], {"dev-sw-P9b-01": ([1.0], "ok")})
    sup_items = [_win("dev-sws1-P9b-01", "c2"), _win("dev-sws1-P9b-r1", "c3"), _win("dev-sws1-P9b-r2", "c4")]
    sha = {"c1": "s1", "c2": "s2", "c3": "s3", "c4": "s4"}
    ok = {"dev-sws1-P9b-01": ([], "press not running"), "dev-sws1-P9b-r1": ([1.0], "ok")}
    doc = DL.build_scored([base, _part(sup_items, ok, True, "labels_s1.json")], sha)
    assert [r["status"] for r in doc["stroke_windows"]] == ["scored", "not_running", "scored", "not_needed"]
    early = {"dev-sws1-P9b-01": ([], "press not running")}            # r1 skipped before the target is reached
    with pytest.raises(ValueError):
        DL.build_scored([base, _part(sup_items, early, True, "labels_s1.json")], sha)
    with pytest.raises(ValueError, match="not labelled"):                    # a primary must be labelled
        DL.build_scored([base, _part(sup_items, {"dev-sws1-P9b-r1": ([1.0], "ok")}, True, "labels_s1.json")], sha)


# ---- fix round 2: edge spans, spans that reach the window edge, exact stopping rule, load_scored re-check ----------

def test_edge_spans_parse_and_are_returned_with_the_hidden_spans(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 1)
    assert DL.edge_spans("possible BDC near 7.6 s: edge, not clicked; edge 7.00-8.00 s") == [[7.0, 8.0]]
    items = [_win("dev-sw-P9b-01", "c1")]
    labels = {"dev-sw-P9b-01": ([1.0], "hidden 2.00-3.00 s. possible BDC: edge, not clicked; edge 7.00-8.00 s")}
    doc = DL.build_scored([_part(items, labels)], {"c1": "s1"})
    row = doc["stroke_windows"][0]
    assert row["hidden"] == [[2.0, 3.0]] and row["edge"] == [[7.0, 8.0]]
    assert doc["counts"]["scored_edge_seconds"] == 1.0


def test_an_unclicked_edge_bdc_without_an_edge_span_fails_closed(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 1)
    items = [_win("dev-sw-P9b-01", "c1")]
    labels = {"dev-sw-P9b-01": ([1.0], "possible BDC near 7.9 s at the window end: edge, not clicked")}
    with pytest.raises(ValueError, match="edge span"):
        DL.build_scored([_part(items, labels)], {"c1": "s1"})


@pytest.mark.parametrize("note,fixed", [("hidden 5.00-7.97 s", "hidden 5.00-7.98 s"),
                                        ("edge 7.00-7.95 s", "edge 7.00-7.98 s"),
                                        ("hidden 0.05-1.00 s", "hidden 0.00-1.00 s")])
def test_a_span_that_stops_just_short_of_the_window_edge_fails_closed(monkeypatch, note, fixed):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 1)
    items = [_win("dev-sw-P9b-01", "c1", t1=7.9742)]
    with pytest.raises(ValueError, match="short of the window"):
        DL.build_scored([_part(items, {"dev-sw-P9b-01": ([1.0], note)})], {"c1": "s1"})
    DL.build_scored([_part(items, {"dev-sw-P9b-01": ([1.0], fixed)})], {"c1": "s1"})


def test_a_span_outside_the_window_fails_closed(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 1)
    items = [_win("dev-sw-P9b-01", "c1")]
    with pytest.raises(ValueError, match="outside the window"):
        DL.build_scored([_part(items, {"dev-sw-P9b-01": ([1.0], "hidden 9.00-9.50 s")})], {"c1": "s1"})


def test_a_press_must_end_with_exactly_the_target_of_scored_windows(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 2)
    items = [_win("dev-sw-P9b-01", "c1"), _win("dev-sw-P9b-02", "c2"), _win("dev-sw-P9b-03", "c3")]
    sha = {"c1": "s1", "c2": "s2", "c3": "s3"}
    under = {"dev-sw-P9b-01": ([1.0], "ok"), "dev-sw-P9b-02": ([], "press not running"),
             "dev-sw-P9b-03": ([], "press not running")}
    with pytest.raises(ValueError, match="ends with 1 scored windows"):
        DL.build_scored([_part(items, under)], sha)
    over = {k: ([1.0], "ok") for k in under}
    with pytest.raises(ValueError, match="ends with 3 scored windows"):
        DL.build_scored([_part(items, over)], sha)


def test_a_supplementary_reserve_labelled_after_the_target_fails_closed(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 2)
    base = _part([_win("dev-sw-P9b-01", "c1")], {"dev-sw-P9b-01": ([1.0], "ok")})
    sup_items = [_win("dev-sws1-P9b-01", "c2"), _win("dev-sws1-P9b-r1", "c3"), _win("dev-sws1-P9b-r2", "c4")]
    sha = {"c1": "s1", "c2": "s2", "c3": "s3", "c4": "s4"}
    # the target (2) is reached at the primary 01; r1 is then labelled although it is not needed
    late = {"dev-sws1-P9b-01": ([1.0], "ok"), "dev-sws1-P9b-r1": ([], "press not running")}
    with pytest.raises(ValueError, match="labelled after"):
        DL.build_scored([base, _part(sup_items, late, True, "labels_s1.json")], sha)


def test_load_scored_rebuilds_the_statuses_and_fails_closed_on_a_rehashed_edit(tmp_path):
    for f in DEV.iterdir():
        shutil.copy(f, tmp_path / f.name)
    doc = json.loads((tmp_path / "scored.json").read_bytes())
    row = next(r for r in doc["stroke_windows"] if r["status"] == "not_running")
    row["status"] = "scored"                                   # an 11th scored window, scored.sha256 updated to match
    raw = DL.dump(doc)
    (tmp_path / "scored.json").write_bytes(raw)
    (tmp_path / "scored.sha256").write_text(DL.sha256_bytes(raw) + "  scored.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not what the label files give"):
        DL.load_scored(tmp_path)


def test_committed_edge_spans_and_spans_reach_the_window_edges():
    s = DL.load_scored(DEV)
    rows = {r["sample_id"]: r for r in s.doc["stroke_windows"]}
    for sid in ("dev-sw-P9a-04", "dev-sw-P9a-09"):
        assert rows[sid]["edge"] and rows[sid]["edge"][-1][1] >= rows[sid]["t1"]
        assert s.ignore[sid] == sorted(rows[sid]["hidden"] + rows[sid]["edge"])
    for sid in ("dev-sws1-P9b-r3", "dev-sws1-P9b-r6"):
        assert rows[sid]["hidden"][-1][1] >= rows[sid]["t1"]


def test_committed_c9_f1_frames_box_the_left_border_operator():
    s = DL.load_scored(DEV)
    fr = {f.sample_id: f for f in s.frames}
    for sid in ("dev-f-c9-03", "dev-f-c9-06", "dev-f-c9-09", "dev-f-c9-12", "dev-f-c9-17", "dev-f-c9-19",
                "dev-fs1-c9-01", "dev-f-c9-01", "dev-f-c9-07"):
        left = [b for b in fr[sid].boxes if b.cls == "person" and b.x0 == 0.0 and 330 <= b.y0 <= 420
                and b.y1 <= 480]
        assert len(left) == 1 and left[0].truncated and left[0].occluded, sid


# ---- RC-freeze fix1: the sealed TEST labels (load_scored_test, scripts/sealed_run.py score) -----------------------

def _test_sets(win_labels, extra_frame_in_windows=False):
    items = [SampleItem(sample_id="test-f-c9-01", kind="frame", clip_id="c0", group_id="g", camera=9, split="test",
                        t_pts=1.0)]
    items += [SampleItem(sample_id=sid, kind="stroke_window", clip_id=c, group_id="g", camera=9, split="test",
                         press="P9b", t0=0.0, t1=8.0)
              for sid, c in (("test-sw-P9b-01", "c1"), ("test-sw-P9b-02", "c2"), ("test-sw-P9b-r1", "c3"))]
    sl = dict(contract="twin.label_samples/1", split="test", seed=1, rule="r",
              items=[i.model_dump(mode="json") for i in items], items_sha256=sample_items_sha256(items))
    head = dict(contract="twin.labels/1", split="test", labeller_id="agent:t", labeller_kind="claude_agent",
                blind=True, proposals_shown=False, rules_sha256=SHA, sample_list_sha256=sl["items_sha256"])
    fr = dict(sample_id="test-f-c9-01", clip_id="c0", t_pts=1.0, boxes=[], note="nobody in view")
    frames = dict(head, frames=[fr], stroke_windows=[])
    wins = dict(head, frames=[fr] if extra_frame_in_windows else [],
                stroke_windows=[dict(sample_id=i.sample_id, clip_id=i.clip_id, press="P9b", t0=0.0, t1=8.0,
                                     strokes=win_labels[i.sample_id][0], note=win_labels[i.sample_id][1])
                                for i in items if i.sample_id in win_labels])
    blobs = {"labels/sample.json": json.dumps(sl).encode(), "labels/labels_windows.json": json.dumps(wins).encode(),
             "labels/labels_frames.json": json.dumps(frames).encode()}
    return blobs, {"c0": "s0", "c1": "s1", "c2": "s2", "c3": "s3"}


def test_load_scored_test_uses_list_order_for_reserves(monkeypatch):
    monkeypatch.setattr(DL, "TARGET_PER_PRESS", 2)
    full = {"test-sw-P9b-01": ([1.0], "ok"), "test-sw-P9b-02": ([2.0], "hidden 3.00-4.00 s"),
            "test-sw-P9b-r1": ([1.5], "ok")}
    blobs, sha = _test_sets(full)
    sd = DL.load_scored_test(blobs, "labels/sample.json", sha)
    assert [f.sample_id for f in sd.frames] == ["test-f-c9-01"]
    assert [w.sample_id for w in sd.windows] == ["test-sw-P9b-01", "test-sw-P9b-02"]       # r1 not needed
    assert sd.hidden["test-sw-P9b-02"] == [[3.0, 4.0]]
    assert sd.doc["counts"]["short_presses"] == {}
    stopped = dict(full, **{"test-sw-P9b-02": ([], "press not running")})                  # r1 fills the gap
    sd = DL.load_scored_test(_test_sets(stopped)[0], "labels/sample.json", sha)
    assert [w.sample_id for w in sd.windows] == ["test-sw-P9b-01", "test-sw-P9b-r1"]
    short = dict(stopped, **{"test-sw-P9b-r1": ([], "NOT SCORABLE: ram not seen")})       # reported, not refused
    sd = DL.load_scored_test(_test_sets(short)[0], "labels/sample.json", sha)
    assert sd.doc["counts"]["short_presses"] == {"P9b": 1}


def test_load_scored_test_fails_closed():
    full = {"test-sw-P9b-01": ([1.0], "ok"), "test-sw-P9b-02": ([2.0], "ok"), "test-sw-P9b-r1": ([1.5], "ok")}
    blobs, sha = _test_sets({k: v for k, v in full.items() if k != "test-sw-P9b-r1"})
    with pytest.raises(ValueError, match="unlabelled"):                     # every item is labelled, reserves too
        DL.load_scored_test(blobs, "labels/sample.json", sha)
    blobs, sha = _test_sets(full, extra_frame_in_windows=True)
    with pytest.raises(ValueError, match="more than one label file"):
        DL.load_scored_test(blobs, "labels/sample.json", sha)
    blobs, sha = _test_sets(full)
    dev = blobs["labels/labels_windows.json"].replace(b'"split": "test"', b'"split": "dev"')
    with pytest.raises(ValueError):                                          # a dev set is refused
        DL.load_scored_test(dict(blobs, **{"labels/labels_windows.json": dev}), "labels/sample.json", sha)


def test_load_scored_test_split_dev_on_the_real_dev_base_pair():
    """RC-final (sealed_run dev-score): the test loader with split dev on the committed dev base pair (labels.json +
    sample.json) gives the base part's scored frames and a subset of load_scored's scored windows."""
    d = Path(__file__).resolve().parents[1] / "labels" / "dev"
    blobs = {f"labels/{n}": (d / n).read_bytes() for n in ("labels.json", "sample.json")}
    doc = json.loads((d / "scored.json").read_text(encoding="utf-8"))
    sha_of = {r["clip_id"]: r["clip_sha256"] for r in doc["frames"] + doc["stroke_windows"]}
    sd = DL.load_scored_test(blobs, "labels/sample.json", sha_of, split="dev")
    ref = DL.load_scored(d)
    base_frames = [r["sample_id"] for r in doc["frames"] if r["status"] == "scored" and r["label_set"] == "labels.json"]
    assert [f.sample_id for f in sd.frames] == base_frames
    assert {w.sample_id for w in sd.windows} <= {w.sample_id for w in ref.windows}
    with pytest.raises(ValueError, match="test .* or dev"):
        DL.load_scored_test(blobs, "labels/sample.json", sha_of, split="train")
