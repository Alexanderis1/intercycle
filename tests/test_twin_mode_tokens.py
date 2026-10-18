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
  TwinSiteGraph): neither twin modules nor the SHARED modules import the body-shop plant, use anything from
  intercycle.robot.sitegraph but Target, name the body-shop builders SiteGraph / build_rects, or call a SHARED
  function / method that calls one of those builders (one level: e.g. TimingCore.from_scenario builds SiteGraph(scn),
  so a module calling TimingCore.from_scenario reaches the body-shop graph). SHARED hits of this rule are KNOWN
  entries too (strict xfail, owner and patch).
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
# (file, body-shop reach) -> owner and patch: the SiteGraph / build_rects rule applied to the SHARED modules.
KNOWN_REACH = {
    ("ros2/intercycle_nav/intercycle_nav/timing.py", "SiteGraph"):
        "S1b: TimingCore.from_scenario (l.51-52) builds the body-shop SiteGraph(scn) (CROSS_X nodes); twin mode: "
        "from_scenario -> TimingCore(compiled_zone.graph)._build(); type TimingCore.graph (l.39) with a SiteGraph-"
        "interface Protocol (scn, pos, adj, targets, inspect_pose, mission) and import only Target (l.22)",
    ("ros2/intercycle_nav/intercycle_nav/timing_planner.py", "TimingCore.from_scenario"):
        "S1b: TimingPlanner (l.55) builds its core with TimingCore.from_scenario(scn); twin mode: build it as "
        "TimingCore(<the zone's compiled site graph>, ...)._build(), the graph handed over as data via run_cfg "
        "(intercycle_nav never imports twin, tests/test_no_truth_import.py TWIN_FREE)",
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


def _is_builder_call(n: ast.AST) -> bool:
    if not isinstance(n, ast.Call):
        return False
    f = n.func
    return (isinstance(f, ast.Name) and f.id in BODY_SHOP_BUILDERS) or \
        (isinstance(f, ast.Attribute) and f.attr in BODY_SHOP_BUILDERS)


def builder_callables(trees: list[ast.Module]) -> set[str]:
    """Names of the functions ("f") and methods ("Class.m") in these modules whose body calls a body-shop builder."""
    out: set[str] = set()
    for tree in trees:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(_is_builder_call(n) for n in ast.walk(node)):
                    out.add(node.name)
            elif isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                            any(_is_builder_call(n) for n in ast.walk(m)):
                        out.add(f"{node.name}.{m.name}")
    return out


def body_shop_reach(tree: ast.Module, modname: str | None = None, is_pkg: bool = False,
                    via: set[str] | frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """{reach key: what was found}. Keys: a body-shop builder name (named anywhere, incl. an import of it), any other
    name imported from intercycle.robot.sitegraph but Target, `import intercycle.robot.sitegraph`, an import of the
    body-shop plant, or a call of a `via` callable ("Class.m" / "f") that itself calls a builder."""
    out: dict[str, list[str]] = {}
    for m in imports(tree, modname, is_pkg):
        if m == REPLACED["plant"] or m.startswith(REPLACED["plant"] + "."):
            out.setdefault(REPLACED["plant"], []).append(f"imports {m}")
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == REPLACED["sitegraph"]:
            for a in n.names:
                if a.name not in SITEGRAPH_ALLOWED:
                    out.setdefault(a.name, []).append(f"from {n.module} import {a.name}")
        elif isinstance(n, ast.Import) and any(a.name == REPLACED["sitegraph"] for a in n.names):
            out.setdefault(f"import {REPLACED['sitegraph']}", []).append("use Target via from-import only")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            key = f"{n.func.value.id}.{n.func.attr}"
            if key in via:
                out.setdefault(key, []).append(f"calls {key} (l.{n.lineno}), which calls a body-shop builder")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in via:
            out.setdefault(n.func.id, []).append(f"calls {n.func.id} (l.{n.lineno}), which calls a body-shop builder")
    for t in code_tokens(tree):
        if t in BODY_SHOP_BUILDERS:
            out.setdefault(t, []).append(f"names the body-shop builder {t}")
    return {k: sorted(set(v)) for k, v in out.items()}


def _shared_via() -> set[str]:
    return builder_callables([parse(REPO / r) for r in SHARED.values()])


def _shared_reach() -> dict[tuple[str, str], list[str]]:
    via = _shared_via()
    out = {}
    for r in SHARED.values():
        f = REPO / r
        for k, v in body_shop_reach(parse(f), module_name(f), f.name == "__init__.py", via).items():
            out[(r, k)] = v
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
    via = _shared_via()
    bad = {rel(f): h for f in twin_files() if (h := body_shop_reach(parse(f), module_name(f), f.name == "__init__.py",
                                                                    via))}
    assert not bad, f"twin modules reach the body-shop plant / site graph: {bad}"


def test_shared_twin_mode_modules_do_not_reach_the_body_shop_builders():
    bad = {k: v for k, v in _shared_reach().items() if k not in KNOWN_REACH}
    assert not bad, f"shared twin-mode modules reach the body-shop plant / site graph: {bad}"


@pytest.mark.parametrize("key", list(KNOWN_REACH), ids=[f"{Path(f).name}:{t}" for f, t in KNOWN_REACH])
@pytest.mark.xfail(strict=True, reason="known pre-existing body-shop graph build, owner and patch in KNOWN_REACH")
def test_known_shared_reach_is_fixed(key):
    assert key not in _shared_reach()


def test_from_scenario_is_found_as_a_builder_callable():
    # the one-level reach is computed from the code, not listed by hand
    assert "TimingCore.from_scenario" in _shared_via()


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


@pytest.mark.parametrize("src, key", [
    ("from intercycle.robot.sitegraph import SiteGraph\ng = SiteGraph(scn)\n", "SiteGraph"),
    ("from intercycle.robot import sitegraph\ng = sitegraph.SiteGraph(scn)\n", "SiteGraph"),
    ("import intercycle.robot.sitegraph\n", "import intercycle.robot.sitegraph"),
    ("from intercycle.robot.sitegraph import CROSS_X, Target\n", "CROSS_X"),
    ("from intercycle.sim.geometry import build_rects\nr = build_rects(scn)\n", "build_rects"),
    ("from intercycle_eval import plant\n", "intercycle_eval.plant"),
    ("core = TimingCore.from_scenario(scn)\n", "TimingCore.from_scenario"),
])
def test_reach_scanner_flags_a_shared_module_building_the_body_shop_graph(src, key):
    via = builder_callables([ast.parse("class TimingCore:\n    @classmethod\n    def from_scenario(cls, scn):\n"
                                       "        return cls(SiteGraph(scn))\n")])
    assert via == {"TimingCore.from_scenario"}
    assert key in body_shop_reach(ast.parse(src), "intercycle_nav.x", False, via)


def test_reach_scanner_passes_target_only_and_twin_graphs():
    src = ("from intercycle.robot.sitegraph import Target\n"
           "core = TimingCore(compiled_zone.graph)\ncore._build()\n")
    assert body_shop_reach(ast.parse(src), "intercycle_nav.x", False, {"TimingCore.from_scenario"}) == {}


def test_site_extent_replaces_the_data_hall_read():
    from intercycle import scenario as S

    body = S.load()
    assert body.site_extent == (float(body.hall["x"][0]), float(body.hall["z"][0]),
                                float(body.model_dump()["data_" + "hall"]["x"][1]), float(body.hall["z"][1]))
    twin = body.model_copy(update={"data_" + "hall": {}, "hall": {"x": [29.8, 38.2], "z": [-0.2, 8.2]}})
    assert twin.site_extent == (29.8, -0.2, 38.2, 8.2)
