"""A-lite isolation (PLAN_TWIN §7, lane S8): the A-lite modules stay outside the sealed path, and their observer side
never sees truth (static AST scan, nothing imported).

* no module outside the A-lite files imports intercycle.alite (so scripts/sealed_run.py, when it lands, cannot reach
  it), and the sealed runner's import closure never contains it;
* the A-lite observer side (observer.py, phase_origin.py) never reaches truth or the twin: intercycle.truth,
  intercycle.sim, every twin truth module, any twin.* module;
* nothing in A-lite reaches intercycle.sim.observer_learned (the A-lite observer is the observer_live-style one).
"""
from _truth_scan import REPO, chain, import_closure, imports, module_name, parse, py_files, rel, resolve
from twin import truth_guard as TG

ALITE_FILES = {"scripts/alite_run.py", "scripts/alite_mw4.py", "cloud/modal_alite.py"}
OBSERVER_SIDE = ("intercycle.alite.observer", "intercycle.alite.phase_origin")
TRUTH_PREFIXES = ("intercycle.truth", "intercycle.sim", "intercycle_sim", "intercycle_eval.plant")


def _is_alite(m: str) -> bool:
    return m == "intercycle.alite" or m.startswith("intercycle.alite.")


def _alite_modules() -> list[str]:
    return sorted(m for m in (module_name(p) for p in py_files("intercycle/alite")) if m)


def test_nothing_outside_alite_imports_it():
    bad = {}
    for p in py_files("twin", "intercycle", "scripts", "cloud", "ros2", "render"):
        r = rel(p)
        if r.startswith("intercycle/alite/") or r in ALITE_FILES:
            continue
        m = module_name(p)
        hits = sorted({d for d in imports(parse(p), m, p.name == "__init__.py") if _is_alite(d)})
        if hits:
            bad[r] = hits
    assert not bad, f"non-A-lite modules import intercycle.alite: {bad}"


def test_sealed_runner_closure_has_no_alite():
    p = REPO / "scripts" / "sealed_run.py"
    if not p.exists():
        return                                         # not landed in this branch; the scan above still holds
    seen: dict[str, str] = {}
    for dep in imports(parse(p)):
        for m, src in import_closure(dep).items():
            seen.setdefault(m, src)
    bad = [chain(seen, m) for m in seen if _is_alite(m)]
    assert not bad, bad


def test_alite_observer_side_never_reaches_truth_or_twin():
    for entry in OBSERVER_SIDE:
        assert resolve(entry) is not None, entry
        seen = import_closure(entry)
        bad = [chain(seen, m) for m in seen
               if TG.is_truth_module(m) or m == "twin" or m.startswith("twin.")
               or any(m == p or m.startswith(p + ".") for p in TRUTH_PREFIXES)]
        assert not bad, f"{entry} reaches truth: {bad}"


def test_alite_never_reaches_observer_learned():
    mods = _alite_modules()
    assert "intercycle.alite.episode" in mods and "intercycle.alite.observer" in mods
    for entry in mods:
        seen = import_closure(entry)
        assert "intercycle.sim.observer_learned" not in seen, chain(seen, "intercycle.sim.observer_learned")
    for f in ALITE_FILES:
        p = REPO / f
        if p.exists():
            assert not [d for d in imports(parse(p)) if d.endswith("observer_learned")], f
