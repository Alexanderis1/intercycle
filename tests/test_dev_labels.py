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
