"""scripts/label_agreement.py (lane S4-labels-blind, U14): inter-labeller agreement of the blind DEV frames.
IoU >= 0.5 greedy one-to-one matching, recall / precision of B against A, foot-point pixel error on matched pairs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from twin.schemas.labels import Box, FrameLabel

_SPEC = importlib.util.spec_from_file_location(
    "label_agreement", Path(__file__).resolve().parents[1] / "scripts" / "label_agreement.py")
assert _SPEC and _SPEC.loader
LA = importlib.util.module_from_spec(_SPEC)
sys.modules["label_agreement"] = LA
_SPEC.loader.exec_module(LA)


def _p(x0, y0, x1, y1, fu=None, fv=None, **kw) -> Box:
    return Box(cls="person", x0=x0, y0=y0, x1=x1, y1=y1, foot_u=(x0 + x1) / 2 if fu is None else fu,
               foot_v=y1 if fv is None else fv, **kw)


def _f(sid, boxes) -> FrameLabel:
    return FrameLabel(sample_id=sid, clip_id="c", t_pts=1.0, boxes=boxes)


def test_iou():
    assert LA.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert LA.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert LA.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_greedy_one_to_one_takes_the_best_pair_first():
    a = [_p(0, 0, 10, 10), _p(2, 0, 12, 10)]
    b = [_p(2, 0, 12, 10)]
    pairs = LA.match(a, b, 0.5)
    assert pairs == [(1, 0, pytest.approx(1.0))]      # B's one box goes to A's identical box, not to both


def test_below_threshold_is_not_a_match():
    a = [_p(0, 0, 10, 10)]
    b = [_p(6, 0, 16, 10)]                             # IoU 4/16 = 0.25
    assert LA.match(a, b, 0.5) == []


def test_agreement_counts_recall_precision_and_foot_error():
    a = [_f("f1", [_p(0, 0, 10, 100), _p(100, 0, 110, 100)]), _f("f2", []), _f("f3", [_p(0, 0, 50, 50)])]
    b = [_f("f1", [_p(0, 0, 10, 100, fu=8, fv=96)]), _f("f2", [_p(0, 0, 10, 10)]), _f("f3", [_p(0, 0, 50, 50)])]
    r = LA.agreement(a, b)
    assert (r["n_a"], r["n_b"], r["tp"]) == (3, 3, 2)
    assert r["recall_b_vs_a"] == pytest.approx(2 / 3)
    assert r["precision_b_vs_a"] == pytest.approx(2 / 3)       # f2's extra box is a B-only box
    assert sorted(r["foot_px"]) == pytest.approx([0.0, 5.0])   # (8,96) vs (5,100) -> 5 px
    assert r["per_frame"]["f1"] == dict(n_a=2, n_b=1, tp=1)


def test_agreement_fails_closed_on_frame_mismatch():
    with pytest.raises(ValueError, match="same frames"):
        LA.agreement([_f("f1", [])], [_f("f2", [])])


def test_clopper_pearson_bounds():
    lo, hi = LA.clopper_pearson(8, 10)
    assert 0.44 < lo < 0.45 and 0.97 < hi < 0.98
    assert LA.clopper_pearson(0, 5)[0] == 0.0 and LA.clopper_pearson(5, 5)[1] == 1.0
