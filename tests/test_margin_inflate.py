"""M-W7 (lane S7b): the ONE inflation rule shared by the offline calibration and the robot side.

The robot reads only the frozen JSON (twin/m7_margin.json) through intercycle.margin_inflate.load_margin and inflates
with intercycle.margin_inflate.inflate; the calibration counts coverage with the same rule. If the two disagreed (for
example the robot flooring a fractional radius in cells), the certified risk would not apply to what the robot does.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from intercycle import margin_inflate as MI
from intercycle.eval import conformal_margin as CM

REPO = Path(__file__).resolve().parents[1]


def _brute(pred2d: np.ndarray, margin_m: float, cell_m: float) -> np.ndarray:
    ny, nx = pred2d.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cov = np.zeros_like(pred2d)
    for a, b in zip(*np.nonzero(pred2d)):
        cov |= ((yy - a) ** 2 + (xx - b) ** 2) * cell_m ** 2 <= margin_m ** 2 + 1e-9 * cell_m ** 2
    return cov


def test_radius2_rounding_rule():
    assert MI.radius2_cells(0.0, 0.25) == 0
    assert MI.radius2_cells(0.25, 0.25) == 1
    assert MI.radius2_cells(0.35, 0.25) == 1  # 1.4 cells: the diagonal neighbour (sqrt 2) is NOT covered
    assert MI.radius2_cells(math.sqrt(2) * 0.25, 0.25) == 2  # float noise below 2.0 still covers the diagonal
    assert MI.radius2_cells(0.375, 0.25) == 2  # 1.5 cells covers the diagonal; floor(1.5) = 1 would not
    with pytest.raises(ValueError):
        MI.radius2_cells(-0.1, 0.25)
    with pytest.raises(ValueError):
        MI.radius2_cells(0.5, 0.0)


def test_inflate_is_the_euclidean_disc_on_cell_centres():
    rng = np.random.default_rng(7)
    for trial in range(4):
        pred = rng.random((3, 13, 17)) < 0.05
        for m in (0.0, 0.25, 0.3, 0.375, 0.5, 0.6, 1.0, 1.25, 2.2):
            got = MI.inflate(pred, MI.Margin(m, 0.25), cell_m=0.25)
            assert got.dtype == bool and got.shape == pred.shape
            for k in range(3):
                assert np.array_equal(got[k], _brute(pred[k], m, 0.25)), (trial, m, k)
    single = np.zeros((7, 7), bool)
    single[3, 3] = True
    assert MI.inflate(single, MI.Margin(0.375, 0.25), cell_m=0.25)[2, 2]  # 1.5 cells reaches the diagonal
    assert not MI.inflate(single, MI.Margin(0.3, 0.25), cell_m=0.25)[2, 2]


def test_inflate_refuses_a_different_grid():
    with pytest.raises(ValueError, match="cell"):
        MI.inflate(np.zeros((4, 4), bool), MI.Margin(0.5, 0.25), cell_m=0.5)


def test_calibration_coverage_equals_inflate():
    """tick_counts (the calibration's loss) misses exactly the truth cells outside inflate(pred, margin)."""
    rng = np.random.default_rng(11)
    truth = rng.random((5, 15, 19)) < 0.2
    pred = rng.random((5, 15, 19)) < 0.06
    margins = tuple(round(0.05 * k, 2) for k in range(0, 41))  # 0 .. 2 m, many fractional radii in cells
    idx = np.arange(5)
    for cell in (0.25, 0.2):
        occ, miss = CM.tick_counts(truth, pred, idx, cell, margins)
        for k, m in enumerate(margins):
            cov = MI.inflate(pred, MI.Margin(m, cell), cell_m=cell)
            assert np.array_equal(miss[:, k], (truth & ~cov).sum((1, 2))), (cell, m)


def test_load_margin_reads_only_a_valid_frozen_file(tmp_path):
    good: dict[str, Any] = dict(schema=MI.SCHEMA, status="ok", frozen=True, margin_m=0.5, cell_m=0.25,
                                inflation=MI.inflation_spec(MI.Margin(0.5, 0.25)))
    p = tmp_path / "m7.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    assert MI.load_margin(p) == MI.Margin(0.5, 0.25)
    for bad in (dict(good, schema="x"), dict(good, frozen=False), dict(good, status="no_valid_margin"),
                dict(good, margin_m=None), {k: v for k, v in good.items() if k != "inflation"},
                dict(good, inflation=dict(good["inflation"], radius2_cells=3)),
                dict(good, inflation=dict(good["inflation"], kernel="square")),
                dict(good, inflation=dict(good["inflation"], radius_m=0.75))):
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            MI.load_margin(p)


def test_helper_is_robot_safe():
    """numpy + standard library only (robot-side dependency rule), and it never names the calibration module."""
    src = (REPO / "intercycle" / "margin_inflate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            tops.add((node.module or "").split(".")[0])
    assert tops <= {"__future__", "numpy", "json", "math", "dataclasses", "pathlib", "os", "typing"}, tops
    assert "conformal_margin" not in src
