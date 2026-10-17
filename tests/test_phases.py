"""Unsupervised phase discovery (observer/phases.py): learned from camera observations only, scored against truth here.

A short SYNTHETIC floor window (scripts/phase_discovery.simulate, ~40 min at 2 Hz incl. the 09:00 break) trains the model;
the numpy motif engine is used so the test does not pay stumpy's numba JIT.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase_discovery as PD  # noqa: E402  (truth allowed in scripts/tests)

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer import phases as P  # noqa: E402
from intercycle.scenario import load  # noqa: E402

H = 3600.0


@pytest.fixture(scope="module")
def fitted():
    P.MOTIF_ENGINE = "numpy"
    d = PD.simulate(0, 8.4 * H, 9.1 * H, progress=False)
    lay = P.Layout.from_scenario(load())
    pm = P.PhaseModel.fit_features(d["T"], d["X"], d["F"], lay, 2.0)
    return d, pm


def test_class_codes_match_mapping():
    assert (P.PERSON, P.VEHICLE, P.PART, P.SKID) == (M.PERSON, M.VEHICLE, M.PART, M.SKID)


def test_no_truth_or_calendar_in_phases():
    src = (ROOT / "intercycle" / "observer" / "phases.py").read_text(encoding="utf-8")
    mods = [n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module]
    mods += [a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if m.startswith(("intercycle.truth", "intercycle.sim"))]
    assert "line_stops" not in src and "calendar_state" not in src and "breaks" not in src


def test_periods_and_bins(fitted):
    d, pm = fitted
    for mid in ("S30", "S60", "HEM1", "SEAL1"):
        mp = pm.machines[mid]
        assert mp is not None
        nominal = 45.0 if mid == "HEM1" else 60.0
        assert abs(mp.period_s / nominal - 1) < 0.02, (mid, mp.period_s)
        assert P.K_MIN + 1 <= pm.n_bins(mid) <= P.K_MAX + 1
        assert pm.bin_names(mid)[-1] == "blocked" and "idle" in pm.bin_names(mid)


def test_offline_segmentation_matches_truth(fitted):
    d, pm = fitted
    i = list(d["mids"]).index("S30")
    pred = pm.machines["S30"].label_offline(d["X"][:, i])
    truth = PD.coarse("weld", d["lab_names"][d["lab"][:, i]])
    mp = PD.majority_map(pred, truth)
    named = np.vectorize(mp.get)(pred)
    assert (named == truth).mean() > 0.85
    assert {"weld", "idle"} <= set(mp.values())


def test_online_interface(fitted):
    d, pm = fitted
    scn = load()
    pm.reset()
    assert pm.phase_bin("S30", 0.0) == -1 and pm.line_state(0.0) == "unknown"
    # feed raw observations through the camera shim (the path workstream M uses)
    from intercycle.sim.world import World
    from intercycle.truth.kinematics import Kinematics
    world = World(scn, 0, 10 * H, 10.2 * H)
    kin = Kinematics(scn, world)
    shim = M.CameraShim(scn, seed=5)
    mids = [m.id for m in scn.machines]
    seen = set()
    for k in range(240):
        t = 10 * H + k / 2.0
        bd = kin.bodies(t)
        obs = shim.observe(M.SimState(t, parts={m: bd[m] for m in mids}, skids=bd["SKIDS"]))
        pm.update(obs)
        b = pm.phase_bin("S30", obs.t_cap)
        assert -1 <= b < pm.n_bins("S30")
        seen.add(b)
        assert pm.line_state(obs.t_cap) in P.LINE_STATES
    assert len(seen) >= 4                                  # a 2 min stretch walks through several sub-phases


def test_save_load_roundtrip(fitted, tmp_path):
    d, pm = fitted
    pm.save(tmp_path / "pm.pkl")
    pm2 = P.PhaseModel.load(tmp_path / "pm.pkl", load())
    sl = slice(0, 600)
    b1, s1, _, _ = pm.decode_features(d["T"][sl], d["X"][sl], d["F"][sl])
    b2, s2, _, _ = pm2.decode_features(d["T"][sl], d["X"][sl], d["F"][sl])
    assert (b1 == b2).all() and (s1 == s2).all()
