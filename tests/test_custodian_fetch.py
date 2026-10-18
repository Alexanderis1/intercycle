"""scripts/custodian_fetch.py: test-row selection and the MANIFEST.sha256 format (no download)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "custodian_fetch", Path(__file__).resolve().parents[1] / "scripts" / "custodian_fetch.py")
assert _SPEC and _SPEC.loader
C = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(C)


def test_rows_and_manifest(tmp_path):
    p = tmp_path / "split_a1.csv"
    p.write_text("clip_id,group_id,split,sha256,framing_id\n7_te1,g1,test,bb,c14_f0\n0_tr1,g0,train,aa,c9_f0\n"
                 "1_tr2,g2,test,cc,c9_f1\n3_tr4,g3,dev,dd,c14_f0\n", encoding="utf-8")
    rows = C.test_rows(p)
    assert rows == [("1_tr2", "cc"), ("7_te1", "bb")]
    text = C.manifest_text(rows, "ff")
    assert text.splitlines() == ["cc  clips/1_tr2.mp4", "bb  clips/7_te1.mp4", "# ids 2: 1_tr2 7_te1",
                                 "# camera 9: 1", "# camera 14: 1", "# class 1: 1", "# class 7: 1",
                                 "# split_a1.csv sha256: ff"]


def test_repo_test_rows_match_split():
    rows = C.test_rows(Path(__file__).resolve().parents[1] / "prereg" / "split_a1.csv")
    assert len(rows) == 126 and rows == sorted(rows)
