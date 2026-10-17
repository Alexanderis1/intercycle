"""PLAN_TWIN §2 truth guards: the offline M-W7 calibration (intercycle.eval.conformal_margin) reads truth, so it must
never be reachable from the live chain: twin/observer_live.py, the robot policy or the broadcaster. Those read only the
frozen JSON margin it writes.

The check walks the static import graph (every `import` / `from ... import`, including ones nested in functions, plus
the parent packages' __init__ files) from each live entry point through repo-local modules, and also rejects a string
literal naming the module inside an import-capable call (a dynamic `importlib.import_module(...)`)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TARGET = "intercycle.eval.conformal_margin"
# top-level package -> directory that contains it
ROOTS = {"intercycle": REPO, "twin": REPO, "intercycle_nav": REPO / "ros2" / "intercycle_nav",
         "intercycle_eval": REPO / "ros2" / "intercycle_eval", "intercycle_sim": REPO / "ros2" / "intercycle_sim"}


def _live_entry_points() -> dict[str, list[Path]]:
    g = lambda pat: sorted(p for p in REPO.glob(pat) if p.is_file())  # noqa: E731
    return {
        "observer_live": g("twin/observer_live.py") + g("twin/observer_live/**/*.py"),
        "policy": g("intercycle/robot/**/*.py") + g("ros2/intercycle_nav/intercycle_nav/**/*.py")
        + g("twin/**/*policy*.py"),
        "broadcaster": g("ros2/intercycle_eval/intercycle_eval/broadcaster.py") + g("twin/**/*broadcast*.py")
        + g("intercycle/codec/**/*.py"),
        # the truth-free inflation helper the robot side imports to apply the frozen JSON margin
        "margin_helper": g("intercycle/margin_inflate.py"),
    }


def _file_of(mod: str) -> Path | None:
    top = mod.split(".")[0]
    if top not in ROOTS:
        return None
    base = ROOTS[top].joinpath(*mod.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _module_of(path: Path) -> str:
    for top, root in ROOTS.items():
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == top:
            parts = list(rel.with_suffix("").parts)
            return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    raise ValueError(f"{path} is not under a known package root")


def _with_parents(mod: str) -> list[str]:
    parts = mod.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


_DYNAMIC_IMPORTERS = {"import_module", "__import__", "find_spec", "spec_from_file_location", "run_module",
                      "import_string", "load_module", "exec", "eval", "getattr"}


def _call_name(func: ast.expr) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _edges(path: Path) -> tuple[set[str], list[str]]:
    """(imported module names, string literals) of one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    me = _module_of(path)
    pkg = me if path.name == "__init__.py" else me.rpartition(".")[0]
    mods, strings = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                src = ".".join(base + ([node.module] if node.module else []))
            else:
                src = node.module or ""
            mods.add(src)
            mods.update(f"{src}.{a.name}" for a in node.names)  # `from pkg import submodule`
        elif isinstance(node, ast.Call) and _call_name(node.func) in _DYNAMIC_IMPORTERS:
            # a dynamic import names its module in a string argument; a deny-list of module names (as in
            # masks.assert_no_truth) is not an import, so only arguments of import-capable calls count
            strings.extend(a.value for a in ast.walk(node) if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return mods, strings


def _closure(files: list[Path]) -> dict[str, Path]:
    seen: dict[str, Path] = {}
    todo = [_module_of(f) for f in files]
    while todo:
        mod = todo.pop()
        for m in _with_parents(mod):
            if m in seen:
                continue
            f = _file_of(m)
            if f is None:
                continue
            seen[m] = f
            todo.extend(_edges(f)[0])
    return seen


@pytest.mark.parametrize("name", ["observer_live", "policy", "broadcaster", "margin_helper"])
def test_live_chain_never_reaches_the_margin_calibration(name):
    files = _live_entry_points()[name]
    if name != "observer_live":
        assert files, f"no {name} entry point found: the guard would be vacuous"
    reach = _closure(files)
    assert TARGET not in reach, f"{name} reaches {TARGET} through its import graph"
    for mod, f in reach.items():
        lits = [s for s in _edges(f)[1] if s.strip() == "conformal_margin" or s.strip().endswith(".conformal_margin")]
        assert not lits, f"{name}: {mod} names the margin calibration in a string literal {lits} (dynamic import?)"


def test_graph_walker_is_not_vacuous(tmp_path, monkeypatch):
    """Positive controls: the walker sees a real known edge, and flags a planted import of the target."""
    bc = REPO / "ros2" / "intercycle_eval" / "intercycle_eval" / "broadcaster.py"
    assert "intercycle.codec.maplayer" in _closure([bc])  # imported inside a function in broadcaster.py
    pkg = tmp_path / "twin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "observer_live.py").write_text("def f():\n    from intercycle.eval import conformal_margin\n",
                                          encoding="utf-8")
    monkeypatch.setitem(ROOTS, "twin", tmp_path)
    assert TARGET in _closure([pkg / "observer_live.py"])
