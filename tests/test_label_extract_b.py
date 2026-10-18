"""scripts/label_extract_b.py (lane S4-labels-blind, U14 second labeller): the first N frame items of the hashed DEV
sample list, in list order, extracted to labeller B's own store, dev clips only (fails closed)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from intercycle import esk

_SPEC = importlib.util.spec_from_file_location(
    "label_extract_b", Path(__file__).resolve().parents[1] / "scripts" / "label_extract_b.py")
assert _SPEC and _SPEC.loader
LB = importlib.util.module_from_spec(_SPEC)
sys.modules["label_extract_b"] = LB
_SPEC.loader.exec_module(LB)


def _fake_load(split, root=None):
    if split != "dev":
        raise AssertionError("label_extract_b must only ever ask for the dev split")
    return pd.DataFrame({"clip_id": [f"0_tr{k}" for k in range(30)], "split": ["dev"] * 30,
                         "framing_id": ["c9_f1"] * 30})


def _sample(tmp_path: Path, n_frames: int, split: str = "dev", clip: str | None = None) -> Path:
    items = []
    for k in range(n_frames):
        items.append(dict(sample_id=f"dev-f-c9-{k + 1:02d}", kind="frame", clip_id=clip or f"0_tr{k}", group_id="g",
                          camera=9, split="dev", t_pts=1.0))
        if k == 3:   # a stroke window in between must be skipped, not counted
            items.append(dict(sample_id="dev-s-P9a-01", kind="stroke_window", clip_id="0_tr1", group_id="g",
                              camera=9, split="dev", press="P9a", t0=0.0, t1=5.0))
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"contract": "twin.label_samples/1", "split": split, "items": items}), encoding="utf-8")
    return p


def test_first_n_frames_in_list_order(tmp_path, monkeypatch):
    monkeypatch.setattr(esk, "load", _fake_load)
    got = LB.blind_items(_sample(tmp_path, 25), 20)
    assert [i["sample_id"] for i in got] == [f"dev-f-c9-{k:02d}" for k in range(1, 21)]
    assert all(i["kind"] == "frame" for i in got)


def test_refuses_short_list_and_non_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(esk, "load", _fake_load)
    with pytest.raises(SystemExit, match="only 5 frame items"):
        LB.blind_items(_sample(tmp_path, 5), 20)
    with pytest.raises(SystemExit, match="dev only"):
        LB.blind_items(_sample(tmp_path, 25, split="test"), 20)
    with pytest.raises(SystemExit, match="not a dev clip"):
        LB.blind_items(_sample(tmp_path, 25, clip="0_te99"), 20)


def test_store_is_labeller_b_own(tmp_path, monkeypatch):
    monkeypatch.setenv(esk.DATA_ENV, str(tmp_path))
    s = LB.store()
    assert s == tmp_path / "labelling_b"
    assert s.is_dir()
