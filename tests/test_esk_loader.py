"""intercycle.esk: the sealed test split, column contract and the split.csv check."""
from __future__ import annotations

import pandas as pd
import pytest

from intercycle import esk


def _inv(tmp_path):
    rows = [
        {"clip_id": "0_tr1", "camera": 9, "framing_id": "c9_f0", "group_id": "c9_g000", "class": 0, "duration": 14.0,
         "sha256": "a", "split": "train"},
        {"clip_id": "3_tr2", "camera": 14, "framing_id": "c14_f0", "group_id": "c14_g000", "class": 3,
         "duration": 5.0, "sha256": "b", "split": "dev"},
        {"clip_id": "7_te1", "camera": 14, "framing_id": "c14_f0", "group_id": "c14_g001", "class": 7,
         "duration": 6.0, "sha256": "c", "split": "test"},
    ]
    pd.DataFrame(rows, columns=esk.COLUMNS).to_parquet(tmp_path / "inventory.parquet")
    (tmp_path / "split.csv").write_text(
        "clip_id,group_id,split,sha256\n0_tr1,c9_g000,train,a\n3_tr2,c14_g000,dev,b\n7_te1,c14_g001,test,c\n",
        encoding="utf-8")
    return tmp_path


def test_train_dev_load(tmp_path, monkeypatch):
    monkeypatch.delenv(esk.SEALED_ENV, raising=False)
    root = _inv(tmp_path)
    assert esk.load("train", root=root)["clip_id"].tolist() == ["0_tr1"]
    assert sorted(esk.load(["train", "dev"], root=root)["clip_id"]) == ["0_tr1", "3_tr2"]


def test_test_split_is_sealed(tmp_path, monkeypatch):
    root = _inv(tmp_path)
    monkeypatch.delenv(esk.SEALED_ENV, raising=False)
    with pytest.raises(esk.SealedSplitError):
        esk.load("test", root=root)
    with pytest.raises(esk.SealedSplitError):
        esk.load(["train", "test"], root=root)
    monkeypatch.setenv(esk.SEALED_ENV, "0")
    with pytest.raises(esk.SealedSplitError):
        esk.load("test", root=root)
    monkeypatch.setenv(esk.SEALED_ENV, "1")
    assert esk.load("test", root=root)["clip_id"].tolist() == ["7_te1"]


def test_unknown_split_and_env_root(tmp_path, monkeypatch):
    root = _inv(tmp_path)
    with pytest.raises(ValueError):
        esk.load("val", root=root)
    monkeypatch.setenv(esk.DATA_ENV, str(root))
    assert len(esk.load("dev")) == 1


def test_camera_of_prefix():
    assert [esk.camera_of(f"{p}_tr1") for p in range(8)] == [9, 9, 9, 14, 9, 9, 9, 14]


def test_check_against_split(tmp_path, monkeypatch):
    root = _inv(tmp_path)
    inv = pd.read_parquet(root / "inventory.parquet")
    esk.check_against_split(inv, root / "split.csv")
    inv.loc[0, "split"] = "dev"
    with pytest.raises(AssertionError):
        esk.check_against_split(inv, root / "split.csv")
