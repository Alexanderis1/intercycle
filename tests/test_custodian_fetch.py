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
    a, b = "c" * 64, "b" * 64
    text = C.manifest_text([("1_tr2", a), ("7_te1", b)])
    lines = text.splitlines()
    assert lines[:4] == ["# twin.sealed/1 MANIFEST", f"{a}  clips/1_tr2.mp4", f"{b}  clips/7_te1.mp4", "#ids 1_tr2,7_te1"]
    assert lines[4:6] == ["#count camera=9 class=1 n=1", "#count camera=14 class=7 n=1"]
    assert lines[6].startswith("#aggregate ") and len(lines) == 7


def test_manifest_is_accepted_by_the_contract_parser_and_sealed_run(tmp_path):
    """The written MANIFEST is twin.sealed/1: parse_sealed accepts it, and sealed_run's loader reads it as MANIFEST."""
    import sys

    from twin.schemas.prereg import aggregate_sha256, parse_sealed

    rows = [("0_te1", "a" * 64), ("3_te2", "d" * 64), ("0_te0", "e" * 64)]
    text = C.manifest_text(sorted(rows))
    s = parse_sealed(text)
    assert s.kind == "MANIFEST" and s.ids == ["0_te0", "0_te1", "3_te2"]
    assert [(f.sha256, f.path) for f in s.files] == [(h, f"clips/{c}.mp4") for c, h in sorted(rows)]
    assert s.aggregate == aggregate_sha256(s.files)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import sealed_run as SR  # pyright: ignore[reportMissingImports]

    man = tmp_path / SR.INPUTS["manifest"]
    man.parent.mkdir(parents=True)
    man.write_text(text, encoding="utf-8", newline="\n")
    assert SR.load_sealed(tmp_path, "manifest", "MANIFEST") == s
    man.write_bytes(text.replace("\n", "\r\n").encode())                 # a CRLF checkout loads the same
    assert SR.load_sealed(tmp_path, "manifest", "MANIFEST") == s


def test_repo_test_rows_match_split():
    rows = C.test_rows(Path(__file__).resolve().parents[1] / "prereg" / "split_a1.csv")
    assert len(rows) == 126 and rows == sorted(rows)
