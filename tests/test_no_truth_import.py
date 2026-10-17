"""The learner and the robot must never see ground truth."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "intercycle"
FORBIDDEN_IN = ["observer/learn_cycles.py", "observer/learn_traffic.py", "observer/mapping.py", "observer/change.py",
                "observer/phases.py", "codec", "robot"]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_learner_and_robot_do_not_import_truth():
    for rel in FORBIDDEN_IN:
        p = PKG / rel
        files = [p] if p.suffix == ".py" else list(p.rglob("*.py")) if p.exists() else []
        for f in files:
            if f.exists():
                bad = [m for m in _imports(f) if m.startswith("intercycle.truth")]
                assert not bad, f"{f} imports ground truth: {bad}"
