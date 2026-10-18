"""labels/dev/labels_blind_B.json (lane S4-labels-blind, U14): the second, independent DEV labeller's blind set answers
exactly the first 20 frame items of the hashed sample list, under the committed labelling rule, with no proposals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from twin.schemas.labels import LabelSet, SampleList, check_complete

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "labels" / "dev"


def _load():
    labels = LabelSet.model_validate(json.loads((DEV / "labels_blind_B.json").read_text(encoding="utf-8")))
    samples = SampleList.model_validate(json.loads((DEV / "sample.json").read_text(encoding="utf-8")))
    return labels, samples


def test_blind_set_answers_the_first_20_frames():
    labels, samples = _load()
    check_complete(labels, samples, require_all=False)
    first20 = [i.sample_id for i in samples.items if i.kind == "frame"][:20]
    assert [f.sample_id for f in labels.frames] == first20
    assert labels.stroke_windows == []


def test_blind_set_is_honest_and_under_the_committed_rule():
    labels, _ = _load()
    assert labels.labeller_id == "agent:dev-labeller-b"
    assert labels.labeller_kind == "claude_agent"
    assert labels.blind is True and labels.proposals_shown is False
    assert {b.src for f in labels.frames for b in f.boxes} <= {"labeller"}
    assert labels.rules_sha256 == hashlib.sha256((ROOT / "labels" / "RULES.md").read_bytes()).hexdigest()


def test_boxes_follow_the_size_and_class_rules():
    labels, _ = _load()
    for f in labels.frames:
        for b in f.boxes:
            assert b.cls == "person"                  # camera 9: no vehicle class is labelled (RULES.md)
            assert b.y1 - b.y0 >= 20                  # minimum person box height
            assert 0 <= b.x0 and b.x1 <= 1920 and b.y1 <= 1080
