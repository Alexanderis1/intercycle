"""scripts/cal_v1.py fail-closed checks (no images needed)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("cal_v1", Path(__file__).resolve().parents[1] / "scripts" / "cal_v1.py")
assert _spec is not None and _spec.loader is not None
S = importlib.util.module_from_spec(_spec)
sys.modules["cal_v1"] = S
_spec.loader.exec_module(S)

HINTS = {"_doc": "x", "A": [[0, 0], [1, 1]], "B#1": [[0, 0], [1, 1]], "B#2": [[0, 0], [1, 1]]}


def test_hinted_lines_join_pieces():
    assert S.hinted_lines(HINTS) == {"A", "B"}


def test_require_all_passes_only_with_every_hinted_line():
    S.require_all("f", "k1", HINTS, {"A": object(), "B": object()}, {})
    with pytest.raises(SystemExit):
        S.require_all("f", "k1", HINTS, {"A": object()}, {})                       # B missing
    with pytest.raises(SystemExit):
        S.require_all("f", "k1", HINTS, {"A": object(), "B": object()}, {"B": "B#2: trace failed"})
    with pytest.raises(SystemExit):
        S.require_all("f", "k1", {"A": [[0, 0], [1, 1]]}, {"A": object()}, {})     # one line is not enough
