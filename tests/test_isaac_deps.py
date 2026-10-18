"""Isaac-side dependencies (PLAN_TWIN §2 "Truth guards"): the twin modules the Isaac container imports may only pull
numpy, astropy, pydantic, nacl, yaml and the standard library (plus repo code that obeys the same rule).

Every module under twin/ is treated as Isaac-side unless listed in CPU_ONLY (twin.build_worlds). Two checks, both fail closed:

1. **Static (AST) scan** of every Isaac-side twin file and of every repo module they reach by static imports
   (`_truth_scan.import_closure`). It collects every import anywhere in the file, including imports inside functions
   and `if TYPE_CHECKING:` blocks, plus `importlib.import_module(...)`, `import_module(...)` and `__import__(...)`
   calls. A call whose module name is not a string literal is an error (it cannot be checked). The only exceptions
   are listed in LAZY_EXCEPTIONS, per (file, top-level package), and only for imports inside a function body (a
   CPU-side helper that the Isaac container never calls).
2. **Runtime hook** in a fresh interpreter: `builtins.__import__` and `importlib.import_module` are wrapped, and any
   import made BY twin.* or intercycle.* code (the importer is taken from the caller's globals, or from the calling
   frame when `__import__` is called without globals) of a package outside the allowlist fails and is reported.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from _truth_scan import REPO, import_closure, module_name, parse, py_files, rel, resolve

ALLOWED = ("numpy", "astropy", "pydantic", "nacl", "yaml", "twin", "intercycle")
REPO_PACKAGES = ("twin", "intercycle")
# ROS-free modules of the ros2/ tree that Isaac-side twin code may import BY EXACT NAME (twin.broadcast_live signs and
# frames through intercycle_nav.broadcast and sends through intercycle_nav.feed). They are scanned by the same rule;
# anything else under intercycle_nav (rclpy nodes) stays forbidden.
ISAAC_SAFE_ROS = ("intercycle_nav", "intercycle_nav.broadcast", "intercycle_nav.feed")
ROS_NAV_ROOT = REPO / "ros2" / "intercycle_nav"
# twin modules that never run in the Isaac container: build_worlds writes the world files the container receives
# (it imports scipy through intercycle.sim.geometry); twin.compile itself stays Isaac-safe.
CPU_ONLY: tuple[str, ...] = ("twin.build_worlds",)
# (repo-relative file, top-level package) -> why the lazy (function-level) import is not Isaac-side.
LAZY_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("twin/schemas/tables.py", "pyarrow"): "validate_table() checks parquet files on the CPU side only",
    ("twin/plant.py", "pyarrow"): "write_log() writes log.parquet on the CPU double only; in the Isaac python the "
                                  "import fails and the plant writes the same rows to log.jsonl (S5a)",
}
IMPORT_CALLS = ("import_module", "__import__")

PROBE = r"""
import builtins, importlib, json, sys
ALLOWED = set(json.loads(sys.argv[1]))
MODULES = json.loads(sys.argv[2])
SAFE = set(json.loads(sys.argv[3])) if len(sys.argv) > 3 else set()
REPO_PKGS = ("twin", "intercycle", "intercycle_nav")
orig_import = builtins.__import__
orig_import_module = importlib.import_module
bad = []
def check(importer, name):
    if importer.split(".")[0] in REPO_PKGS:
        if name in SAFE:
            return
        top = name.split(".")[0]
        if top and top not in ALLOWED and top not in sys.stdlib_module_names:
            bad.append([importer, name])
            raise ImportError(f"{importer} imports {name}: not allowed on the Isaac side")
def hook(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        g = globals if globals is not None else sys._getframe(1).f_globals
        check((g or {}).get("__name__") or "", name)
    return orig_import(name, globals, locals, fromlist, level)
def import_module_hook(name, package=None):
    if not name.startswith("."):
        check(sys._getframe(1).f_globals.get("__name__") or "", name)
    return orig_import_module(name, package)
builtins.__import__ = hook
importlib.import_module = import_module_hook
errors = []
for m in MODULES:
    try:
        orig_import_module(m)
    except Exception as e:  # report every failure, keep going
        errors.append([m, repr(e)])
print(json.dumps({"bad": bad, "errors": errors}))
"""


# ------------------------------------------------------------------------------------------------ static scan


def import_sites(tree: ast.Module, modname: str | None = None, is_pkg: bool = False) -> list[tuple[str | None, bool, int]]:
    """[(absolute module name or None if not a literal, inside a function, line)] for every import statement and
    every import_module / __import__ call. Relative imports resolve against `modname` (they are repo code)."""
    out: list[tuple[str | None, bool, int]] = []

    def visit(node: ast.AST, lazy: bool) -> None:
        if isinstance(node, ast.Import):
            out.extend((a.name, lazy, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if modname:
                    pkg = modname.split(".") if is_pkg else modname.split(".")[:-1]
                    pkg = pkg[: len(pkg) - (node.level - 1)]
                    out.append((".".join([*pkg, node.module] if node.module else pkg), lazy, node.lineno))
            elif node.module:
                out.append((node.module, lazy, node.lineno))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in IMPORT_CALLS:
                arg = node.args[0] if node.args else None
                lit = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
                if lit is not None and lit.startswith("."):
                    lit = REPO_PACKAGES[0]          # relative import_module(".x", package): repo code
                out.append((lit, lazy, node.lineno))
        inner = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    visit(tree, False)
    return out


def static_violations(files: list[Path], exceptions: dict[tuple[str, str], str] | None = None,
                      root: Path = REPO) -> list[str]:
    exc = LAZY_EXCEPTIONS if exceptions is None else exceptions
    bad: list[str] = []
    for f in files:
        r = f.resolve().relative_to(root.resolve()).as_posix()
        m = module_name(f) if root == REPO else ".".join(Path(r).with_suffix("").parts)
        for name, lazy, line in import_sites(parse(f), m, f.name == "__init__.py"):
            if name is None:
                bad.append(f"{r}:{line}: dynamic import with a non-literal module name")
                continue
            top = name.split(".")[0]
            if top in ALLOWED or top in sys.stdlib_module_names or name in ISAAC_SAFE_ROS:
                continue
            if lazy and (r, top) in exc:
                continue
            bad.append(f"{r}:{line}: imports {name}" + ("" if lazy else " at module level"))
    return bad


def _twin_modules() -> list[str]:
    mods = []
    for f in py_files("twin"):
        m = module_name(f)
        if m and not m.endswith("__main__") and m not in CPU_ONLY:
            mods.append(m)
    return sorted(set(mods))


def isaac_side_files() -> list[Path]:
    """Every Isaac-side twin file plus every repo file reachable from them by static imports."""
    files: set[Path] = set()
    for m in _twin_modules():
        for dep in import_closure(m):
            if (dep.split(".")[0] in REPO_PACKAGES or dep in ISAAC_SAFE_ROS) and dep not in CPU_ONLY:
                f = resolve(dep)
                if f is not None:
                    files.add(f.resolve())
    return sorted(files)


def test_static_scan_covers_twin_and_the_intercycle_modules_it_reaches():
    files = {rel(f) for f in isaac_side_files()}
    assert "twin/camera_v0.py" in files and "twin/schemas/tables.py" in files
    assert "intercycle/observer/mapping.py" in files, "the closure must follow twin -> intercycle imports"


def test_isaac_side_files_import_only_isaac_deps_statically():
    bad = static_violations(isaac_side_files())
    assert not bad, "non-Isaac imports (function-level and import_module calls included):\n" + "\n".join(bad)


def test_lazy_exceptions_are_still_needed():
    """An exception whose import is gone must be removed (the map only shrinks)."""
    stale = []
    for (r, top), _why in LAZY_EXCEPTIONS.items():
        f = REPO / r
        names = [n for n, lazy, _ in import_sites(parse(f), module_name(f)) if n and lazy] if f.exists() else []
        if not any(n.split(".")[0] == top for n in names):
            stale.append((r, top))
    assert not stale, f"stale LAZY_EXCEPTIONS entries: {stale}"


# ------------------------------------------------------------------------------------------------ runtime hook


def _probe(mods: list[str], cwd: Path, pythonpath: Path, extra_path: tuple[Path, ...] = ()) -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in (pythonpath, *extra_path)), "PYTHONUTF8": "1"}
    res = subprocess.run([sys.executable, "-c", PROBE, json.dumps(ALLOWED), json.dumps(mods),
                          json.dumps(ISAAC_SAFE_ROS)], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_isaac_safe_ros_modules_are_scanned():
    files = {rel(f) for f in isaac_side_files()}
    assert {"ros2/intercycle_nav/intercycle_nav/broadcast.py", "ros2/intercycle_nav/intercycle_nav/feed.py"} <= files


def test_twin_modules_use_only_isaac_deps_at_runtime():
    mods = _twin_modules()
    assert "twin.schemas" in mods and "twin.camera_v0" in mods
    out = _probe(mods, REPO, REPO, (ROS_NAV_ROOT,))
    assert not out["bad"], f"non-Isaac imports: {out['bad']}"
    assert not out["errors"], f"import failures: {out['errors']}"


# ------------------------------------------------------------------------------------------------ self-tests


BAD_SOURCES = {
    "top_level": "import pandas\n",
    "in_function": "def f():\n    import pandas\n    return pandas\n",
    "from_in_method": "class C:\n    def m(self):\n        from pandas import DataFrame\n        return DataFrame\n",
    "import_module": "import importlib\nP = importlib.import_module('pandas')\n",
    "import_module_from": "from importlib import import_module\nP = import_module('pandas.io')\n",
    "dunder_import": "P = __import__('pandas')\n",
    "dynamic_name": "import importlib\nNAME = 'pan' + 'das'\nP = importlib.import_module(NAME)\n",
    "type_checking": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas\n",
    "ros_node": "from intercycle_nav.observer_bridge import ObserverBridge\n",   # only ISAAC_SAFE_ROS by exact name
}


def _fake_pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "twin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for name, src in BAD_SOURCES.items():
        (pkg / f"bad_{name}.py").write_text(src, encoding="utf-8")
    (pkg / "lazy_ok.py").write_text("import numpy\ndef f():\n    import pyarrow\n    return pyarrow\n", encoding="utf-8")
    return pkg


def test_static_scan_catches_every_import_form(tmp_path):
    pkg = _fake_pkg(tmp_path)
    for name in BAD_SOURCES:
        bad = static_violations([pkg / f"bad_{name}.py"], exceptions={}, root=tmp_path)
        assert bad, f"static scan missed the {name} form"
    exc = {("twin/lazy_ok.py", "pyarrow"): "test"}
    assert static_violations([pkg / "lazy_ok.py"], exceptions=exc, root=tmp_path) == []
    assert static_violations([pkg / "lazy_ok.py"], exceptions={}, root=tmp_path), "no exception -> flagged"
    # an exception never covers a module-level import of the same package
    (pkg / "eager.py").write_text("import pyarrow\n", encoding="utf-8")
    assert static_violations([pkg / "eager.py"], exceptions={("twin/eager.py", "pyarrow"): "test"}, root=tmp_path)


def test_runtime_hook_catches_import_forms(tmp_path):
    """The runtime hook sees module-level imports, import_module and __import__ (function-level imports are the
    static scan's job: they never run at import time)."""
    _fake_pkg(tmp_path)
    for name in ("top_level", "import_module", "import_module_from", "dunder_import"):
        out = _probe([f"twin.bad_{name}"], tmp_path, tmp_path)
        assert out["bad"] and out["bad"][0][0] == f"twin.bad_{name}", f"runtime hook missed {name}: {out}"
        assert out["bad"][0][1].split(".")[0] == "pandas"
