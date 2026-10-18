"""Twin mode carries no body-shop layout (PLAN_TWIN §2 `twin/compile.py`, §4 S1a): a grep over code tokens bans
CROSS_X, S80 and data_hall in the twin-mode code paths of compile, plant, broadcaster, timing, masks, mapping and
sitegraph.

"Code tokens" = identifiers, attribute names, imported names and aliases, keyword and parameter names, def / class
names and string literals (f-string parts included), read from the AST: comments and docstrings may explain the rule.
CROSS_X and data_hall match as substrings (data_hall_targets too); S80 matches as a station id (not S800 / XS80).

Code paths in twin mode:
* every module under twin/ (compile and the TwinSiteGraph, build_worlds, plant, engine, camera, observer_live, ...);
* the shared modules twin mode runs through (SHARED: broadcaster, timing / timing_planner, masks, mapping). Their
  pre-existing body-shop reads are listed in KNOWN with the owner and the patch; a strict xfail turns red once fixed;
* the body-shop plant and the body-shop SiteGraph are replaced in twin mode (twin/plant.py, twin.compile's
  TwinSiteGraph): twin modules never import the body-shop plant and use nothing from intercycle.robot.sitegraph but
  Target, and never name the body-shop builders SiteGraph / build_rects.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from _truth_scan import REPO, imports, module_name, parse, py_files, rel, string_literals

BANNED = ("CROSS_X", "S80", "data_hall")
_PATTERNS = {"CROSS_X": re.compile("CROSS_X"), "S80": re.compile(r"(?<![A-Za-z0-9])S80(?![0-9])"),
             "data_hall": re.compile("data_hall")}
TWIN_ROOTS = ("twin",)
SHARED = {
    "broadcaster": "ros2/intercycle_eval/intercycle_eval/broadcaster.py",
    "timing": "ros2/intercycle_nav/intercycle_nav/timing.py",
    "timing_planner": "ros2/intercycle_nav/intercycle_nav/timing_planner.py",
    "masks": "ros2/intercycle_nav/intercycle_nav/masks.py",
    "mapping": "intercycle/observer/mapping.py",
}
REPLACED = {"plant": "intercycle_eval.plant", "sitegraph": "intercycle.robot.sitegraph"}
SITEGRAPH_ALLOWED = {"Target"}
BODY_SHOP_BUILDERS = {"SiteGraph", "build_rects"}
# (file, token) -> owner and patch. Scenario.site_extent (intercycle/scenario.py) is the replacement for the reads.
KNOWN = {
    ("intercycle/observer/mapping.py", "data_hall"):
        "S4: _Camera floor (l.605) and build_spec (l.1182) read scn.data_hall['x'][1]; use scn.site_extent[2]",
    ("ros2/intercycle_nav/intercycle_nav/masks.py", "data_hall"):
        "S3b: Masks.from_scenario (l.196) builds ext from scn.data_hall; use ext = scn.site_extent",
}


def code_tokens(tree: ast.Module) -> list[str]:
    out: list[str] = list(string_literals(tree))
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
        elif isinstance(n, ast.alias):
            out += [n.name] + ([n.asname] if n.asname else [])
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.append(n.module)
        elif isinstance(n, ast.keyword) and n.arg:
            out.append(n.arg)
        elif isinstance(n, ast.arg):
            out.append(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(n.name)
    return out


def banned_hits(tree: ast.Module) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for tok in code_tokens(tree):
        for b, pat in _PATTERNS.items():
            if pat.search(tok):
                hits.setdefault(b, []).append(tok)
    return {b: sorted(set(v)) for b, v in hits.items()}


def twin_files() -> list[Path]:
    return py_files(*TWIN_ROOTS)


def _shared_hits() -> dict[tuple[str, str], list[str]]:
    out = {}
    for r in SHARED.values():
        f = REPO / r
        assert f.exists(), f"shared twin-mode module {r} moved: update SHARED"
        for b, toks in banned_hits(parse(f)).items():
            out[(r, b)] = toks
    return out


# ------------------------------------------------------------------------------------------------ the rule


def test_twin_modules_name_no_body_shop_layout():
    files = twin_files()
    assert any(rel(f) == "twin/compile.py" for f in files)
    bad = {rel(f): h for f in files if (h := banned_hits(parse(f)))}
    assert not bad, f"body-shop layout tokens {BANNED} in twin-mode code: {bad}"


def test_shared_twin_mode_modules_name_no_body_shop_layout():
    bad = {k: v for k, v in _shared_hits().items() if k not in KNOWN}
    assert not bad, f"body-shop layout tokens in shared twin-mode modules: {bad}"


@pytest.mark.parametrize("key", list(KNOWN), ids=[f"{Path(f).name}:{t}" for f, t in KNOWN])
@pytest.mark.xfail(strict=True, reason="known pre-existing body-shop read, owner and patch in KNOWN")
def test_known_shared_hits_are_fixed(key):
    assert key not in _shared_hits()


def test_twin_modules_do_not_use_the_replaced_body_shop_modules():
    bad: dict[str, list[str]] = {}
    for f in twin_files():
        tree = parse(f)
        r = rel(f)
        mods = imports(tree, module_name(f), f.name == "__init__.py")
        for m in mods:
            if m == REPLACED["plant"] or m.startswith(REPLACED["plant"] + "."):
                bad.setdefault(r, []).append(f"imports {m}")
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == REPLACED["sitegraph"]:
                extra = sorted({a.name for a in n.names} - SITEGRAPH_ALLOWED)
                if extra:
                    bad.setdefault(r, []).append(f"from {n.module} import {extra}")
            elif isinstance(n, ast.Import) and any(a.name == REPLACED["sitegraph"] for a in n.names):
                bad.setdefault(r, []).append(f"import {REPLACED['sitegraph']} (use Target via from-import only)")
        used = {t for t in code_tokens(tree) if t in BODY_SHOP_BUILDERS}
        if used:
            bad.setdefault(r, []).append(f"names the body-shop builders {sorted(used)}")
    assert not bad, f"twin modules reach the body-shop plant / site graph: {bad}"


def test_shared_list_matches_the_plan_names():
    # PLAN §4 S1a: compile, plant, broadcaster, timing, masks, mapping and sitegraph; compile lives under twin/
    assert set(SHARED) | set(REPLACED) | {"compile"} >= {"compile", "plant", "broadcaster", "timing", "masks",
                                                          "mapping", "sitegraph"}


# ------------------------------------------------------------------------------------------------ self-tests


@pytest.mark.parametrize("src, token", [
    ("x = scn.data_hall['x']\n", "data_hall"),
    ("x = scn.data_hall_targets\n", "data_hall"),
    ("from intercycle.robot.sitegraph import CROSS_X\n", "CROSS_X"),
    ("from intercycle.robot import sitegraph as S\nx = S.CROSS_X\n", "CROSS_X"),
    ("d = {'data_hall': 1}\n", "data_hall"),
    ("f(data_hall=1)\n", "data_hall"),
    ("def f(data_hall):\n    return 0\n", "data_hall"),
    ("ids = ['S10', 'S80']\n", "S80"),
    ("if body == 'S80':\n    pass\n", "S80"),
    ("x = f'{a}S80'\n", "S80"),
    ("def f():\n    '''doc'''\n    k = 'CROSS_X'\n", "CROSS_X"),
])
def test_scanner_catches_every_form(src, token):
    assert token in banned_hits(ast.parse(src))


@pytest.mark.parametrize("src", [
    "# CROSS_X, S80 and data_hall are banned here\nx = 1\n",
    '"""No CROSS_X, S80 or data_hall in twin mode."""\nx = 1\n',
    "def f():\n    '''Not the data_hall.'''\n    return 0\n",
    "ids = ['S800', 'XS80', 'S8']\n",
    "cross_x = 1\nDATA_HALLWAY = 2\n",
])
def test_scanner_ignores_comments_docstrings_and_lookalikes(src):
    assert banned_hits(ast.parse(src)) == {}


def test_site_extent_replaces_the_data_hall_read():
    from intercycle import scenario as S

    body = S.load()
    assert body.site_extent == (float(body.hall["x"][0]), float(body.hall["z"][0]),
                                float(body.model_dump()["data_" + "hall"]["x"][1]), float(body.hall["z"][1]))
    twin = body.model_copy(update={"data_" + "hall": {}, "hall": {"x": [29.8, 38.2], "z": [-0.2, 8.2]}})
    assert twin.site_extent == (29.8, -0.2, 38.2, 8.2)
