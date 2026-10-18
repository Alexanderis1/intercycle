"""scripts/dyn_fit.py (lane S3a DYN v1) fix round 1: fail-closed ordering (provenance before any write), the
unchanged-npz rule that keeps the pinned library sha256, and the pinned DEV-labels hash."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "dyn_fit.py"


@pytest.fixture(scope="module")
def fit():
    spec = importlib.util.spec_from_file_location("dyn_fit_script", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_provenance_check_runs_before_every_write_and_covers_every_id_set():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]

    def name(c: ast.Call) -> str:
        f = c.func
        return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""

    check = [c for c in calls if name(c) == "assert_train_only"]
    assert len(check) == 1
    writes = [c for c in calls if name(c) in ("write_text", "write_bytes", "put_npz", "savez", "mkdir", "open")]
    assert writes and all(w.lineno > check[0].lineno for w in writes), [(name(w), w.lineno) for w in writes]
    src = ast.get_source_segment(SCRIPT.read_text(encoding="utf-8"), main) or ""
    for key in ('"library"', '"press_masks"', '"layout_states"', '"valcam_jitter_gaps"', ':clips"', ':judged"',
                ':running"', '"dynamics"'):
        assert key in src, key


def test_unchanged_npz_is_not_rewritten(fit, tmp_path):
    p = tmp_path / "lib.npz"
    arrays = {"a": np.arange(5.0), "s": np.array(["x", "y"], dtype=object)}
    raw = fit.npz_bytes(arrays)
    sha = fit.put_npz(p, arrays, raw)
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()
    before = p.read_bytes()
    # the same arrays again (np.savez stamps the time: new bytes) leave the file and its sha256 as they are
    assert fit.npz_sha(p, arrays, fit.npz_bytes(arrays)) == sha
    assert fit.put_npz(p, arrays, fit.npz_bytes(arrays)) == sha and p.read_bytes() == before
    changed = {"a": np.arange(6.0), "s": arrays["s"]}
    assert fit.put_npz(p, changed, fit.npz_bytes(changed)) != sha
    assert not fit.same_npz(p, arrays) and fit.same_npz(p, changed)


def test_labels_hash_mismatch_fails_closed(fit, tmp_path, monkeypatch):
    d = tmp_path / "labels" / "dev"
    d.mkdir(parents=True)
    (d / "scored.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
    monkeypatch.setattr(fit, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        fit.load_labels(None)
    sha = hashlib.sha256((d / "scored.json").read_bytes()).hexdigest()
    assert fit.load_labels(None, sha) == ([], sha)
    assert fit.labels_source(None) == "labels/dev (tree)"


def test_dynamics_records_the_full_labels_hash_and_ref(fit):
    d = json.loads((REPO / "twin" / "dynamics.json").read_text(encoding="utf-8"))
    for z in ("z9", "z14"):
        note = d["assumed"][f"zone.{z}.walker_rate_per_h"]["note"]
        assert fit.SCORED_SHA256 in note and fit.LABELS_REF in note
