"""Isaac-side dependencies (PLAN_TWIN §2 "Truth guards"): the twin modules the Isaac container imports may only pull
numpy, astropy, pydantic, nacl, yaml and the standard library (plus repo code that obeys the same rule).

Every module under twin/ is treated as Isaac-side unless listed in CPU_ONLY (none at G0). The check runs in a fresh
interpreter with an import hook on builtins.__import__: any import made BY twin.* or intercycle.* code of another
top-level package fails and is reported. Imports made inside third-party packages (pydantic's own deps) are not
checked; lazy imports inside functions (e.g. pyarrow in twin.schemas.tables.validate_table) are not executed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from _truth_scan import REPO, module_name, py_files

ALLOWED = ("numpy", "astropy", "pydantic", "nacl", "yaml", "twin", "intercycle")
CPU_ONLY: tuple[str, ...] = ()          # twin modules that never run in the Isaac container (none yet)

PROBE = r"""
import builtins, importlib, json, sys
ALLOWED = set(json.loads(sys.argv[1]))
MODULES = json.loads(sys.argv[2])
orig = builtins.__import__
bad = []
def hook(name, globals=None, locals=None, fromlist=(), level=0):
    importer = (globals or {}).get("__name__") or ""
    if level == 0 and importer.split(".")[0] in ("twin", "intercycle"):
        top = name.split(".")[0]
        if top not in ALLOWED and top not in sys.stdlib_module_names:
            bad.append([importer, name])
            raise ImportError(f"{importer} imports {name}: not allowed on the Isaac side")
    return orig(name, globals, locals, fromlist, level)
builtins.__import__ = hook
errors = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:  # report every failure, keep going
        errors.append([m, repr(e)])
print(json.dumps({"bad": bad, "errors": errors}))
"""


def _twin_modules() -> list[str]:
    mods = []
    for f in py_files("twin"):
        m = module_name(f)
        if m and not m.endswith("__main__") and m not in CPU_ONLY:
            mods.append(m)
    return sorted(set(mods))


def test_twin_modules_use_only_isaac_deps():
    mods = _twin_modules()
    assert "twin.schemas" in mods and "twin.camera_v0" in mods
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1"}
    res = subprocess.run([sys.executable, "-c", PROBE, json.dumps(ALLOWED), json.dumps(mods)], cwd=REPO,
                         env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert not out["bad"], f"non-Isaac imports: {out['bad']}"
    assert not out["errors"], f"import failures: {out['errors']}"


def test_hook_catches_a_forbidden_import(tmp_path):
    """Self-test: a twin module importing pandas is reported."""
    pkg = tmp_path / "twin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "bad_mod.py").write_text("import pandas\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONUTF8": "1"}
    res = subprocess.run([sys.executable, "-c", PROBE, json.dumps(ALLOWED), json.dumps(["twin.bad_mod"])],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["bad"] == [["twin.bad_mod", "pandas"]]
