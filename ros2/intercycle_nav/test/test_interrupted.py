"""goal.json shape when the timing planner is interrupted at the end of a fixed-length episode (R3's cheat guard)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from intercycle_nav.timing import camera_stub_used, interrupted_result  # noqa: E402


def test_interrupted_result_carries_the_cheat_guard_fields():
    ev = [{"event": "chosen", "t": 1.0, "target": "S10"},
          {"event": "target_done", "t": 9.0, "target": "S10", "outcome": "inspected"},
          {"event": "chosen", "t": 9.0, "target": "S20"},
          {"event": "target_done", "t": 20.0, "target": "S20", "outcome": "missed_no_entry"},
          {"event": "chosen", "t": 20.0, "target": "S30"}]
    out = interrupted_result(ev, {"camera_model (R1 geometric, R3 gate)": 40}, True, "learned", 10)
    assert out["result"] == "INTERRUPTED" and out["condition"] == "learned"
    assert out["n_inspected"] == 1 and out["n_done"] == 2 and out["n_targets"] == 10
    assert out["camera_stub_used"] is False and out["require_camera"] is True
    assert out["camera_sources"] == {"camera_model (R1 geometric, R3 gate)": 40}
    assert out["targets"] == ev


def test_stub_detected():
    assert camera_stub_used({"camera_stub (TEST DOUBLE)": 3})
    assert not camera_stub_used({"camera_model (R1 geometric, R3 gate)": 3})
    out = interrupted_result([], {"camera_stub (TEST DOUBLE)": 3}, False, "none")
    assert out["camera_stub_used"] and out["require_camera"] is False
