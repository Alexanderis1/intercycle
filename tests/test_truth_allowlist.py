"""Truth allowlist (PLAN_TWIN §2 "Truth guards"): only the listed modules may open or import truth.

Truth = realised_state*, plant_state, the day event script, per-world dynamics draws, world seeds and defect state.
Static checks over every .py file in twin/, intercycle/, scripts/, cloud/, ros2/ and render/ (tests excluded):
  1. a string literal naming a truth path (docstrings excepted) only in TRUTH_READERS, TRUTH_PRODUCERS or the
     contract files;
  2. imports of truth modules (twin.engine, twin.plant, twin.camera, ...) only in readers, producers or the listed
     orchestrators;
  3. calls of open_truth only in readers or producers.
The runtime half (open_truth refuses other callers and logs every opening) is tested in test_twin_contracts.py.
"""
from __future__ import annotations

import ast

import pytest

from _truth_scan import SCAN_ROOTS, imports, module_name, parse, py_files, rel, string_literals
from twin import truth_guard as TG

# Exactly the plan's list (§2 "Only these may open it"): engine, plant, camera, oracle_feed, the EVAL metrics module
# and S7b's offline M-W7 calibration. Changing any of these pins needs the lead.
PLAN_READERS = ("twin/engine.py", "twin/plant.py", "twin/camera.py", "twin/oracle_feed.py", "twin/eval_metrics.py",
                "twin/margin_calib.py")
# camera_v0 fills the twin/camera.py slot (G0 stand-in), it is not an extra reader.
STAND_INS = {"twin/camera_v0.py": "twin/camera.py"}
# Writers of truth (day generators, engine, plant; the body-shop ros2 plant writes plant_state.json), separate list.
PRODUCERS = ("twin/daygen.py", "twin/daygen_stub.py", "twin/engine.py", "twin/plant.py",
             "ros2/intercycle_eval/intercycle_eval/plant.py")
# Launchers that may import truth modules to start the plant / generator; they never open truth.
ORCHESTRATORS = ("scripts/slice.py", "scripts/sealed_run.py")

# Pre-existing violations, each with its owner; the strict xfail below turns red once fixed, so the entry is removed.
KNOWN = {
    "ros2/intercycle_eval/intercycle_eval/broadcaster.py":
        "S1b: the body-shop broadcaster reads plant_state.json; the twin broadcaster reads obs_v2.jsonl only",
}


def _contract(r: str) -> bool:
    return any(r == c or (c.endswith("/") and r.startswith(c)) for c in TG.CONTRACT_FILES)


def _literal_violations() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in py_files(*SCAN_ROOTS):
        r = rel(f)
        if r in TG.TRUTH_READERS or r in TG.TRUTH_PRODUCERS or _contract(r):
            continue
        hits = sorted({s for s in string_literals(parse(f)) if any(t in s for t in TG.TRUTH_TOKENS)})
        if hits:
            out[r] = hits
    return out


def test_allowlist_is_the_plan_list():
    assert TG.PLAN_READERS == PLAN_READERS
    assert TG.STAND_INS == STAND_INS and set(STAND_INS.values()) <= set(PLAN_READERS)
    assert TG.TRUTH_READERS == PLAN_READERS + tuple(STAND_INS)
    assert TG.TRUTH_PRODUCERS == PRODUCERS
    assert TG.ORCHESTRATORS == ORCHESTRATORS


def test_only_producers_may_write_truth(tmp_path, monkeypatch):
    """open_truth refuses a write from a reader that is not a producer (camera_v0 reads realised state only)."""
    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "log"))
    src = (f"from twin.truth_guard import open_truth\n"
           f"with open_truth({str(tmp_path / 'x.realised_state.jsonl')!r}, 'w') as f:\n    f.write('')\n")
    code = compile(src, str(TG.REPO / "twin" / "camera_v0.py"), "exec")
    with pytest.raises(PermissionError, match="may not open truth"):
        exec(code, {"__name__": "fake_camera_v0"})
    code = compile(src, str(TG.REPO / "twin" / "daygen_stub.py"), "exec")
    exec(code, {"__name__": "fake_daygen"})                                  # a producer may write
    assert (tmp_path / "x.realised_state.jsonl").exists()


def test_no_unlisted_file_names_a_truth_path():
    bad = {k: v for k, v in _literal_violations().items() if k not in KNOWN}
    assert not bad, f"truth paths named outside the allowlist (twin/truth_guard.py): {bad}"


@pytest.mark.xfail(strict=True, reason="known pre-existing violations, see KNOWN (owners named there)")
def test_known_literal_violations_are_fixed():
    v = _literal_violations()
    assert not [k for k in KNOWN if k in v]


def test_truth_modules_are_imported_only_by_the_allowlist():
    ok = set(TG.TRUTH_READERS) | set(TG.TRUTH_PRODUCERS) | set(TG.ORCHESTRATORS)
    bad = {}
    for f in py_files(*SCAN_ROOTS):
        r = rel(f)
        if r in ok:
            continue
        mods = [m for m in imports(parse(f), module_name(f), f.name == "__init__.py") if TG.is_truth_module(m)]
        if mods:
            bad[r] = sorted(set(mods))
    assert not bad, f"truth modules imported outside the allowlist: {bad}"


def test_open_truth_is_called_only_by_the_allowlist():
    ok = set(TG.TRUTH_READERS) | set(TG.TRUTH_PRODUCERS)
    bad = []
    for f in py_files(*SCAN_ROOTS):
        r = rel(f)
        if r in ok or r == "twin/truth_guard.py":
            continue
        for n in ast.walk(parse(f)):
            if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == "open_truth"
                                            or getattr(n.func, "attr", None) == "open_truth"):
                bad.append(r)
    assert not bad, f"open_truth called outside the allowlist: {sorted(set(bad))}"


def test_scanner_sees_a_violation(tmp_path):
    """The literal and import rules really fire (guards against a scanner that silently finds nothing)."""
    src = 'import twin.engine\nfrom twin import camera\nP = "runs/x/realised_state.jsonl"\n'
    tree = ast.parse(src)
    assert any(TG.is_truth_module(m) for m in imports(tree, "intercycle.robot.x"))
    assert "twin.camera" in imports(tree, "intercycle.robot.x")
    assert any("realised_state" in s for s in string_literals(tree))
