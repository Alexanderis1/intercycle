"""scripts/prereg_r0.py: the one-sided Clopper-Pearson rule and the committed r0 value (PREREG v1 §4 H4)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prereg_r0", ROOT / "scripts" / "prereg_r0.py")
assert _spec is not None and _spec.loader is not None
R = importlib.util.module_from_spec(_spec)
sys.modules["prereg_r0"] = R
_spec.loader.exec_module(R)


def test_cp_lower_one_sided():
    from scipy.stats import beta

    assert R.cp_lower_one_sided(54, 87) == pytest.approx(beta.ppf(0.05, 54, 34))
    assert R.cp_lower_one_sided(0, 10) == 0.0
    # n successes of n: lower bound = 0.05 ** (1 / n)
    assert R.cp_lower_one_sided(20, 20) == pytest.approx(0.05 ** (1 / 20))
    with pytest.raises(ValueError):
        R.cp_lower_one_sided(0, 0)


def test_committed_r0_matches_its_counts_and_inputs():
    import hashlib

    d = json.loads((ROOT / "prereg" / "r0_v1.json").read_text(encoding="utf-8"))
    s = d["scored"]
    assert s["frames"] == 19 and d["det_conf"] == 0.4 and d["level"] == 0.95
    assert d["r0"] == round(R.cp_lower_one_sided(s["tp"], s["n_gt"]), 4)
    assert sum(p["tp"] for p in d["per_frame"] if p["status"] == "scored") == s["tp"]
    assert sum(p["n_gt"] for p in d["per_frame"] if p["status"] == "scored") == s["n_gt"]
    b = (ROOT / "labels" / "dev" / "labels_blind_B.json").read_bytes()
    assert hashlib.sha256(b).hexdigest() == d["labels_blind_B_sha256"]
