"""The learner and the robot must never see ground truth.

Twin rules (PLAN_TWIN §2 "Truth guards"):
* `twin.` is a forbidden prefix for the robot side: the learner / codec / robot modules of `intercycle` and the
  robot process (ros2/intercycle_nav) never import any twin.* module (masks.assert_no_truth enforces the same at
  run time; owner S3b).
* the import graph of observer_live, the policy and the broadcaster never reaches truth (intercycle.truth,
  intercycle.sim, the Isaac world intercycle_sim, the plant, and every twin truth module in twin.truth_guard).
* ros2/intercycle_eval and ros2/intercycle_nav contain no "realised_state" / "plant_state" string literal, except in
  the plant that writes them.
"""
import ast
from pathlib import Path

import pytest

from _truth_scan import chain, import_closure, parse, py_files, rel, resolve, string_literals
from twin import truth_guard as TG

PKG = Path(__file__).resolve().parents[1] / "intercycle"
FORBIDDEN_IN = ["observer/learn_cycles.py", "observer/learn_traffic.py", "observer/mapping.py", "observer/change.py",
                "observer/phases.py", "codec", "robot"]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _forbidden_files() -> list[Path]:
    out = []
    for rel_ in FORBIDDEN_IN:
        p = PKG / rel_
        out += [p] if p.suffix == ".py" else list(p.rglob("*.py")) if p.exists() else []
    return [f for f in out if f.exists()]


def test_learner_and_robot_do_not_import_truth():
    for f in _forbidden_files():
        bad = [m for m in _imports(f) if m.startswith("intercycle.truth")]
        assert not bad, f"{f} imports ground truth: {bad}"


# ------------------------------------------------------------------------------------------------ twin rules

ROBOT_SIDE_ROOTS = ("ros2/intercycle_nav/intercycle_nav",)


def _is_twin(m: str) -> bool:
    return m == "twin" or m.startswith("twin.")


def test_robot_side_never_imports_twin():
    files = _forbidden_files() + py_files(*ROBOT_SIDE_ROOTS)
    assert files
    bad = {rel(f): sorted({m for m in _imports(f) if _is_twin(m)}) for f in files}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"robot-side modules import twin.* (forbidden prefix): {bad}"


# Entry points: observer_live (S6), the policies (intercycle.robot.policies; the S5b reactive baseline when it lands)
# and the robot process modules, plus the broadcaster (S1b). Entries not on disk yet are skipped.
ENTRIES = ("twin.observer_live", "intercycle.robot.policies", "twin.policy_reactive", "intercycle_nav.timing_planner",
           "intercycle_nav.observer_bridge", "intercycle_nav.masks", "intercycle_eval.broadcaster")
TRUTH_PREFIXES = ("intercycle.truth", "intercycle.sim", "intercycle_sim", "intercycle_eval.plant")
KNOWN_GRAPH = {
    "intercycle_eval.broadcaster": "S1b (R-W1): the body-shop broadcaster imports intercycle.sim (observer_learned, "
                                   "world); the twin broadcaster must read obs_v2.jsonl only",
}


def _is_truth(m: str) -> bool:
    return TG.is_truth_module(m) or any(m == p or m.startswith(p + ".") for p in TRUTH_PREFIXES)


def _graph_hits(entry: str) -> list[str]:
    seen = import_closure(entry)
    return [chain(seen, m) for m in sorted(seen) if _is_truth(m)]


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e not in KNOWN_GRAPH])
def test_import_graph_never_reaches_truth(entry):
    if resolve(entry) is None:
        pytest.skip(f"{entry} not on disk yet")
    hits = _graph_hits(entry)
    assert not hits, f"{entry} reaches truth: {hits}"


@pytest.mark.parametrize("entry", list(KNOWN_GRAPH))
@pytest.mark.xfail(strict=True, reason="known pre-existing violation, owner named in KNOWN_GRAPH")
def test_known_graph_violations_are_fixed(entry):
    assert not _graph_hits(entry)


def test_graph_scan_finds_existing_entries():
    assert sum(resolve(e) is not None for e in ENTRIES) >= 4


ROS_SCAN = ("ros2/intercycle_eval", "ros2/intercycle_nav")
ROS_TOKENS = ("realised_state", "plant_state")
KNOWN_LITERALS = {"ros2/intercycle_eval/intercycle_eval/broadcaster.py": "S1b: reads plant_state.json"}


def _ros_literal_hits() -> dict[str, list[str]]:
    out = {}
    for f in py_files(*ROS_SCAN):
        r = rel(f)
        if r in TG.TRUTH_PRODUCERS or r in TG.TRUTH_READERS:
            continue
        hits = sorted({s for s in string_literals(parse(f)) if any(t in s for t in ROS_TOKENS)})
        if hits:
            out[r] = hits
    return out


def test_ros_packages_name_no_truth_files():
    bad = {k: v for k, v in _ros_literal_hits().items() if k not in KNOWN_LITERALS}
    assert not bad, f"realised_state / plant_state named in the ROS packages: {bad}"


@pytest.mark.xfail(strict=True, reason="known pre-existing violation, owner named in KNOWN_LITERALS")
def test_known_ros_literals_are_fixed():
    assert not [k for k in KNOWN_LITERALS if k in _ros_literal_hits()]
