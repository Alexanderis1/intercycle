"""scripts/label_extract.py fails closed (lane S4-labels fix round 1): it extracts DEV clips only."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from intercycle import esk

_SPEC = importlib.util.spec_from_file_location(
    "label_extract", Path(__file__).resolve().parents[1] / "scripts" / "label_extract.py")
assert _SPEC and _SPEC.loader
LE = importlib.util.module_from_spec(_SPEC)
sys.modules["label_extract"] = LE
_SPEC.loader.exec_module(LE)


def _fake_load(split, root=None):
    if split != "dev":
        raise AssertionError("label_extract must only ever ask for the dev split")
    return pd.DataFrame({"clip_id": ["0_te1", "0_tr2"], "split": ["dev", "dev"], "framing_id": ["c9_f1", "c9_f1"]})


def _sample(tmp_path: Path, clip_ids: list[str], split: str = "dev", item_split: str = "dev") -> Path:
    items = [dict(sample_id=f"dev-f-c9-{k:02d}", kind="frame", clip_id=c, group_id="g", camera=9, split=item_split,
                  t_pts=1.0) for k, c in enumerate(clip_ids, 1)]
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"contract": "twin.label_samples/1", "split": split, "items": items}), encoding="utf-8")
    return p


def test_items_accepts_dev_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(esk, "load", _fake_load)
    got = LE.items(_sample(tmp_path, ["0_te1", "0_tr2"]))
    assert [i["clip_id"] for i in got] == ["0_te1", "0_tr2"]


def test_items_refuses_a_clip_not_in_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(esk, "load", _fake_load)
    with pytest.raises(SystemExit, match="not a dev clip"):
        LE.items(_sample(tmp_path, ["0_te1", "0_te99"]))


def test_items_refuses_non_dev_items_and_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(esk, "load", _fake_load)
    with pytest.raises(SystemExit, match="not a dev clip"):
        LE.items(_sample(tmp_path, ["0_te1"], item_split="test"))
    with pytest.raises(SystemExit, match="dev only"):
        LE.items(_sample(tmp_path, ["0_te1"], split="test"))
