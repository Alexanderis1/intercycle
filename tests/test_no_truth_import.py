"""The learner and the robot must never see ground truth."""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "intercycle"
FORBIDDEN_IN = ["observer/learn_cycles.py", "observer/learn_traffic.py", "observer/mapping.py", "observer/change.py",
                "observer/phases.py", "observer/change_detect.py", "observer/change_detect_l1.py", "codec", "robot"]
# PLAN_TWIN §2 Truth guards: the robot / episode-side ROS packages never import the twin (plant, engine, camera, truth)
TWIN_FREE = [REPO / "ros2" / "intercycle_nav", REPO / "ros2" / "intercycle_eval"]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _is_twin(module: str) -> bool:
    """`twin` itself (`import twin`, `from twin import engine`) or any `twin.*` module."""
    return module == "twin" or module.startswith("twin.")


def test_learner_and_robot_do_not_import_truth():
    for rel in FORBIDDEN_IN:
        p = PKG / rel
        files = [p] if p.suffix == ".py" else list(p.rglob("*.py")) if p.exists() else []
        for f in files:
            if f.exists():
                bad = [m for m in _imports(f) if m.startswith("intercycle.truth")]
                assert not bad, f"{f} imports ground truth: {bad}"


def test_ros2_nav_and_eval_do_not_import_twin():
    files = [f for d in TWIN_FREE for f in d.rglob("*.py")]
    assert len(files) > 10, "the ROS 2 packages were not found"
    for f in files:
        bad = [m for m in _imports(f) if _is_twin(m)]
        assert not bad, f"{f} imports the twin: {bad}"


def test_twin_scan_catches_every_import_form(tmp_path):
    for src in ("import twin.engine", "import twin", "from twin import engine", "from twin.camera import stat_v1",
                "def f():\n    import twin.plant as P\n"):
        p = tmp_path / "m.py"
        p.write_text(src + "\n", encoding="utf-8")
        assert any(_is_twin(m) for m in _imports(p)), src
    p = tmp_path / "ok.py"
    p.write_text("import twinkle\nfrom intercycle import twins_doc\n", encoding="utf-8")
    assert not any(_is_twin(m) for m in _imports(p))
