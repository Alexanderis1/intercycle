"""Static scanning helpers shared by the truth-guard tests (AST, no imports of the scanned code)."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("twin", "intercycle", "scripts", "cloud", "ros2", "render")
# Where top-level packages live: twin / intercycle at the repo root, the ROS packages under ros2/<pkg>/.
MODULE_ROOTS = (REPO, REPO / "ros2" / "intercycle_eval", REPO / "ros2" / "intercycle_nav", REPO / "ros2" / "intercycle_sim")


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO).as_posix()


def py_files(*roots: str) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        base = REPO / r
        if base.exists():
            out += [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def parse(p: Path) -> ast.Module:
    return ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))


def _docstring_ids(tree: ast.Module) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def string_literals(tree: ast.Module) -> list[str]:
    """Every str constant except docstrings (f-string parts included)."""
    skip = _docstring_ids(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def module_name(p: Path) -> str | None:
    for root in MODULE_ROOTS:
        try:
            parts = list(p.resolve().relative_to(root).with_suffix("").parts)
        except ValueError:
            continue
        if parts and parts[0] in ("twin", "intercycle", "intercycle_eval", "intercycle_nav", "intercycle_sim"):
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
    return None


def imports(tree: ast.Module, modname: str | None = None, is_pkg: bool = False) -> list[str]:
    """Absolute names of every import anywhere in the file (function-level too). `from a import b` yields both a and
    a.b (b may be a submodule). Relative imports are resolved against `modname`."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if not modname:
                    continue
                pkg = modname.split(".") if is_pkg else modname.split(".")[:-1]
                pkg = pkg[: len(pkg) - (node.level - 1)]
                base = ".".join([*pkg, base] if base else pkg)
            if base:
                out.append(base)
                out += [f"{base}.{a.name}" for a in node.names if a.name != "*"]
    return out


def resolve(mod: str) -> Path | None:
    parts = mod.split(".")
    for root in MODULE_ROOTS:
        f = root.joinpath(*parts).with_suffix(".py")
        if f.exists():
            return f
        f = root.joinpath(*parts, "__init__.py")
        if f.exists():
            return f
    return None


def import_closure(entry: str) -> dict[str, str]:
    """Module names reachable from `entry` by static imports: {module: the module that imported it}. Every imported
    name is recorded (also ones not on disk yet, e.g. a future twin.engine); only repo modules are expanded."""
    seen: dict[str, str] = {entry: ""}
    todo = [entry]
    while todo:
        m = todo.pop()
        f = resolve(m)
        if f is None:
            continue
        for dep in imports(parse(f), m, f.name == "__init__.py"):
            # a dotted import also loads its parent packages
            parts = dep.split(".")
            for k in range(1, len(parts) + 1):
                d = ".".join(parts[:k])
                if d not in seen:
                    seen[d] = m
                    todo.append(d)
    return seen


def chain(seen: dict[str, str], m: str) -> str:
    out = [m]
    while seen.get(out[-1]):
        out.append(seen[out[-1]])
    return " <- ".join(out)
