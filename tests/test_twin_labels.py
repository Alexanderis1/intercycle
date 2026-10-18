"""Labels contract, sealed hash formats and Gemma window edits (PLAN_TWIN §2, §5; audit major 4 and minor)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from twin import schemas as S
from twin.schemas import labels as LB
from twin.schemas import prereg as PR

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"


def _j(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ labels


def test_label_fixtures_validate_and_are_complete():
    samples = S.load(S.SampleList, FIX / "twin_label_samples.json")
    labels = S.load(S.LabelSet, FIX / "twin_labels.json")
    S.check_complete(labels, samples)
    assert any(not f.boxes for f in labels.frames), "the fixture keeps an empty frame (empty frames are labelled)"


def test_sample_list_hash_detects_a_change():
    d = _j("twin_label_samples.json")
    d["items"][0]["t_pts"] = 2.5
    with pytest.raises(ValidationError, match="items_sha256"):
        S.SampleList.model_validate(d)


def test_empty_frame_must_still_be_labelled():
    samples = S.load(S.SampleList, FIX / "twin_label_samples.json")
    d = _j("twin_labels.json")
    d["frames"] = [f for f in d["frames"] if f["boxes"]]                       # drop the empty frame
    with pytest.raises(ValueError, match="not labelled"):
        S.check_complete(S.LabelSet.model_validate(d), samples)
    S.check_complete(S.LabelSet.model_validate(d), samples, require_all=False)  # a blind subset relabel is fine


def test_labels_must_match_the_sample_list():
    samples = S.load(S.SampleList, FIX / "twin_label_samples.json")
    d = _j("twin_labels.json")
    d["frames"][0]["t_pts"] = 2.5
    with pytest.raises(ValueError, match="does not match"):
        S.check_complete(S.LabelSet.model_validate(d), samples)
    d = _j("twin_labels.json")
    d["sample_list_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        S.check_complete(S.LabelSet.model_validate(d), samples)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["frames"][0]["boxes"][0].pop("foot_u"), "foot point"),
    (lambda d: d["frames"][0]["boxes"][0].update(foot_u=10.0), "inside its box"),
    (lambda d: d["frames"][0]["boxes"][1].update(foot_u=1400.0, foot_v=400.0), "no foot point"),
    (lambda d: d["frames"][0]["boxes"][0].update(x1=700.0), "empty"),
    (lambda d: d["frames"][0]["boxes"][0].update(cls="vehicle"), "cls"),
    (lambda d: d["stroke_windows"][0].update(strokes=[2.0, 1.0]), "increase"),
    (lambda d: d["stroke_windows"][0].update(strokes=[5.5]), "outside"),
    (lambda d: d["stroke_windows"][0].update(t1=12.0), "window length"),
    (lambda d: d.update(blind=True, proposals_shown=True), "blind"),
    (lambda d: d.update(split="test", blind=False), "test labels"),
    (lambda d: d["frames"][0]["boxes"][0].update(src="proposal_corrected"), "proposals"),
    (lambda d: (d.update(blind=False, proposals_shown=True), d["frames"][0]["boxes"][0].update(src="proposal_accepted")),
     "never accept"),
    (lambda d: d.pop("labeller_id"), "labeller_id"),
])
def test_label_set_rules(edit, match):
    d = copy.deepcopy(_j("twin_labels.json"))
    edit(d)
    with pytest.raises(ValidationError, match=match):
        S.LabelSet.model_validate(d)


# ------------------------------------------------------------------------------------------------ sealed hashes


@pytest.mark.parametrize("name,kind", [("twin_MANIFEST.sha256", "MANIFEST"), ("twin_LABELS.sha256", "LABELS")])
def test_sealed_fixture_roundtrips(name, kind):
    text = (FIX / name).read_text(encoding="utf-8")
    s = S.parse_sealed(text)
    assert s.kind == kind and S.render_sealed(s) == text.replace("\r\n", "\n")
    assert [(c.camera, c.cls, c.n) for c in s.counts] == [(9, 0, 1), (14, 3, 2)]


def _lines(name: str) -> list[str]:
    return (FIX / name).read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()


@pytest.mark.parametrize("mutate,match", [
    (lambda L: [L[0], L[2], L[1], *L[3:]], "sorted"),                                   # file lines out of order
    (lambda L: [ln.replace("n=2", "n=3") for ln in L], "count"),                        # counts do not match ids
    (lambda L: [ln.replace("#ids 0_fx1,", "#ids 0_fx9,") for ln in L], "differ"),       # ids != clip files
    (lambda L: [L[0], L[1].replace(L[1][:2], "ff"), *L[2:]], "aggregate"),             # a file hash changed
    (lambda L: [ln for ln in L if not ln.startswith("#aggregate")], "aggregate"),
    (lambda L: ["# twin.sealed/2 MANIFEST", *L[1:]], "first line"),
    (lambda L: [*L, "stray line"], "unrecognised"),
])
def test_manifest_rejects_tampering(mutate, match):
    text = "\n".join(mutate(_lines("twin_MANIFEST.sha256"))) + "\n"
    with pytest.raises(ValueError, match=match):
        S.parse_sealed(text)


def test_labels_hash_file_needs_the_sample_list_hash():
    lines = [ln for ln in _lines("twin_LABELS.sha256") if not ln.startswith("#sample_list")]
    with pytest.raises(ValueError, match="sample_list"):
        S.parse_sealed("\n".join(lines) + "\n")


def test_sealed_clip_ids_need_a_class_prefix():
    with pytest.raises(ValueError, match="class prefix"):
        PR.make_sealed("MANIFEST", [("ab" * 32, "clips/x_1.mp4")], ["x_1"])


# ------------------------------------------------------------------------------------------------ Gemma window edits


def _edit(**kw):
    d = {"clip_id": "c9_dev_0003", "press": "P9a", "window_id": "w1", "t0": 10.0, "t1": 20.0, "t0_new": 12.0,
         "t1_new": 18.0, "mode": "replay", "model": "gemma"}
    d.update(kw)
    return d


def test_window_edit_may_only_shorten():
    S.WindowEdit.model_validate(_edit())
    S.WindowEdit.model_validate(_edit(t0_new=10.0, t1_new=20.0))
    for kw in ({"t0_new": 9.0}, {"t1_new": 21.0}, {"t0_new": 15.0, "t1_new": 15.0}, {"t0_new": 22.0, "t1_new": 25.0}):
        with pytest.raises(ValidationError, match="shorten only"):
            S.WindowEdit.model_validate(_edit(**kw))


def test_sample_item_kind_rules():
    with pytest.raises(ValidationError, match="t_pts only"):
        LB.SampleItem.model_validate({"sample_id": "a", "kind": "frame", "clip_id": "c", "group_id": "g", "camera": 9,
                                      "split": "dev", "t_pts": 1.0, "press": "P9a"})
    with pytest.raises(ValidationError, match="press, t0 and t1"):
        LB.SampleItem.model_validate({"sample_id": "a", "kind": "stroke_window", "clip_id": "c", "group_id": "g",
                                      "camera": 9, "split": "dev", "t0": 1.0, "t1": 3.0})
