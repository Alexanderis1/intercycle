"""R3: plant (light-curtain trip/hold, reactive movers), camera checks, stats, cfg builder, cheat guard."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "ros2" / "intercycle_sim", ROOT / "ros2" / "intercycle_eval", ROOT / "ros2" / "intercycle_nav"):
    sys.path.insert(0, str(p))

from intercycle_eval import camera as CAM  # noqa: E402
from intercycle_eval import social, stats  # noqa: E402
from intercycle_sim import movers as MV  # noqa: E402
from intercycle_sim.sensors import BoxCaster  # noqa: E402


@pytest.fixture(scope="module")
def world_dir(tmp_path_factory):
    from intercycle_sim import world as W

    d = tmp_path_factory.mktemp("w") / "world"
    d.mkdir()
    W.write_world(W.build_world_files(), str(d))
    return d


def _plant(world_dir, seed=21, t0=10 * 3600 + 1234.0, **kw):
    from intercycle_eval.plant import Plant

    cfg = {"world_dir": str(world_dir), "out_dir": str(world_dir.parent / "out"), "seconds": 120,
           "plant": {"seed": seed, "plant_t0": t0, **kw}}
    P = Plant(cfg)
    from intercycle_sim.world import load_boxes

    P.attach(caster=BoxCaster(load_boxes(str(world_dir / "boxes.json"))))
    return P


def _st(x, y, psi=0.0, vx=0.0, vy=0.0):
    return {"pos_w": [x, y, 0.6], "quat_wxyz": [math.cos(psi / 2), 0, 0, math.sin(psi / 2)], "v_w": [vx, vy, 0],
            "v_b": [vx, vy, 0], "omega_b": [0, 0, 0]}


def test_parts_explicit_shapes():
    ps = MV.parts_for([{"id": "a", "cls": "cyl:0.5:2.0"}, {"id": "b", "cls": "box:1.0:0.5:1.2"}])
    assert ps[0].shape == "cyl" and ps[0].length == 1.0 and ps[0].height == 2.0
    assert ps[1].shape == "box" and ps[1].width == 0.5


def test_social_force_pushes_away():
    f = social.social_force(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert f[0, 0] < 0                      # an agent ahead pushes back


def test_crowd_impassive_ignores_robot():
    c = social.Crowd(2, np.zeros((0, 5)), reacts=np.array([True, False]))
    c.place(0, (0.0, 0.0))
    c.place(1, (0.0, 10.0))
    goals = np.array([[5.0, 0.0], [5.0, 10.0]])
    for _ in range(10):
        c.step(0.1, goals, np.array([1.2, 1.2]), robot_xy=None, robot_v=(0, 0))
    a = c.pos.copy()
    c2 = social.Crowd(2, np.zeros((0, 5)), reacts=np.array([True, False]))
    c2.place(0, (0.0, 0.0))
    c2.place(1, (0.0, 10.0))
    for _ in range(10):
        c2.step(0.1, goals, np.array([1.2, 1.2]), robot_xy=(0.8, 0.0), robot_v=(0, 0))
        c2.step(0.0, goals, np.array([1.2, 1.2]), robot_xy=(0.8, 10.0), robot_v=(0, 0))
    assert np.linalg.norm(c2.pos[0] - a[0]) > 0.05     # reacting person deflected
    assert np.allclose(c2.pos[1], a[1], atol=1e-9)        # impassive person unaffected


def test_trip_on_active_entry_and_hold_on_idle(world_dir):
    P = _plant(world_dir)
    T = P.t0
    # find a time when S10 is active, then put the robot's footprint inside its zone
    t = 0.0
    while not P.world.active("S10", T + t):
        t += 0.1
    P.after_step(t, _st(12.0, 31.0), [])
    P.after_step(t + 0.1, _st(12.0, 28.5), [])
    assert any(e["type"] == "trip" and e["zone"] == "S10" for e in P.events)
    lt = P.lost_time()
    assert P.world.group("S10").freezes[0].kind == "trip"
    del lt
    # hold: a fresh plant, robot enters during an idle window and stays past the next start
    P2 = _plant(world_dir)
    t = 0.0
    while P2.world.active("S10", P2.t0 + t):
        t += 0.1
    t += 0.2
    P2.after_step(t, _st(12.0, 31.0), [])
    for k in range(1, 900):
        P2.after_step(t + 0.1 * k, _st(12.0, 28.5), [])
    assert not any(e["type"] == "trip" for e in P2.events)
    assert any(e["type"] == "hold" for e in P2.events)
    P2.t = t + 90.0
    assert P2.lost_time()["line"]["hold_s"] > 10.0


def test_vehicles_do_not_yield_but_react(world_dir):
    P = _plant(world_dir)
    for v in P.vehicles:
        if v.cls == "forklift":
            assert v.ag.speed > 0
    # robot standing still for 60 s in the north aisle: everything keeps running, poses exist
    for k in range(600):
        P.after_step(0.1 * (k + 1), _st(30.0, 32.0), [])
    assert P.mpose
    assert all(np.isfinite(v).all() for v in map(np.array, P.mpose.values()))


def test_gate_publishes_camera_model_observations(world_dir):
    P = _plant(world_dir)
    obs = []
    for k in range(50):
        t = 0.02 * (k + 1)
        P.after_step(t, _st(12.0, 30.6, psi=-math.pi / 2), [])
        obs += P.frame_extras(t).get("camera_obs", [])
    assert obs and all(o["source"].startswith("camera_model") for o in obs)
    ids = {tg["id"] for o in obs for tg in o["targets"]}
    assert "S10" in ids
    assert all(set(tg) >= {"id", "visible", "motion_seen"} for o in obs for tg in o["targets"])


def test_capture_checks():
    caster = BoxCaster([])
    cam = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    ok = CAM.capture(0.0, cam, np.array([1.2, 0.0, 1.0]), np.array([-1.0, 0, 0]), np.zeros(3), 0.0, caster, None, rng)
    assert ok["pass"] and ok["px_per_m"] > 250
    far = CAM.capture(0.0, cam, np.array([8.0, 0.0, 1.0]), np.array([-1.0, 0, 0]), np.zeros(3), 0.0, caster, None, rng)
    assert not far["checks"]["dori_identify"]
    slant = CAM.capture(0.0, cam, np.array([1.2, 0.0, 1.0]), np.array([0.0, 1.0, 0]), np.zeros(3), 0.0, caster, None, rng)
    assert not slant["checks"]["incidence_ok"]
    blur = CAM.capture(0.0, cam, np.array([1.2, 0.0, 1.0]), np.array([-1.0, 0, 0]), np.array([0, 0.5, 0]), 0.0, caster,
                       None, rng)
    assert not blur["checks"]["blur_ok"]
    occ = CAM.capture(0.0, cam, np.array([3.0, 0.0, 1.0]), np.array([-1.0, 0, 0]), np.zeros(3), 0.0, caster,
                      {"circles": np.array([[1.5, 0.0, 0.3, 2.0]]), "boxes": np.zeros((0, 6))}, rng)
    assert not occ["checks"]["unoccluded"]


def test_stats_helpers():
    lo, hi = stats.clopper_pearson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32           # rule of three ~ 0.3
    m = stats.mcnemar_exact(np.array([1, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 0]))
    assert m["a_only"] == 3 and m["b_only"] == 0
    assert stats.wilcoxon_p(np.zeros(5)) is None
    a, b = stats.bootstrap_ci(np.ones(5))
    assert a == b == 1.0


def test_cfg_and_cheat_guard():
    from intercycle_eval.batch import episode_cfg, plant_start
    from intercycle_sim.episode import eval_check
    from intercycle import scenario as S

    c = episode_cfg("x", 20, "learned")
    assert c["eval"] and c["mover_source"] == "intercycle_eval.plant:make"
    assert any("intercycle_eval.broadcaster" in e for e in c["extra_launch"])
    assert "camera_stub" not in " ".join(c["extra_launch"])
    t0 = plant_start(1000, 660.0)
    assert all(not (a < t0 + 720 and b > t0 - 60) for a, b, _ in S.load().line_stops())
    assert c["plant"]["plant_t0"] == episode_cfg("x", 20, "none")["plant"]["plant_t0"]      # paired
    good = {"goal": {"camera_sources": {CAM.SOURCE: 100}, "camera_stub_used": False, "require_camera": True}}
    assert eval_check({"eval": True, "sim": "isaac"}, good)["valid"]
    bad = {"goal": {"camera_sources": {"camera_stub (TEST DOUBLE)": 100}, "camera_stub_used": True, "require_camera": True}}
    assert not eval_check({"eval": True, "sim": "isaac"}, bad)["valid"]
    assert not eval_check({"eval": True, "sim": "fake"}, good)["valid"]


def test_signflip_and_holm():
    p = stats.signflip_p(np.ones(10))
    assert abs(p - 2 / 1024) < 1e-12                  # only all-plus and all-minus reach |mean| = 1
    assert stats.signflip_p(np.zeros(5)) is None
    h = stats.holm({"a": 0.01, "b": 0.04, "c": None, "d": 0.03})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["d"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12 and h["c"] is None
