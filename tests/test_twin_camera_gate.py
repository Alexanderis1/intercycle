"""R-W3 camera gate (twin/camera_gate.py): pixel-motion threshold, sway / ego false motion, range-dependent
detection, latency, and the plant hook."""
from __future__ import annotations

import math

import numpy as np

from twin import camera_gate as CG
from twin import plant as PL
from twin import robot_camera as RC


class _Open:
    name = "open"

    def cast(self, origin, dirs, rmax):
        return np.full(len(dirs), rmax)


def _ram(x: float, z_up: float) -> np.ndarray:
    """Four ram-plate corner points (x, y, z-up) at distance x ahead of the origin."""
    return np.array([(x, -0.4, z_up), (x, 0.4, z_up), (x, -0.4, z_up - 0.1), (x, 0.4, z_up - 0.1)])


def _run(speed_mps: float, dist_m: float, cam_v=(0.0, 0.0, 0.0), n: int = 60, seed: int = 1):
    """A ram fixed in the world (moving vertically at speed_mps) watched by a camera moving at cam_v."""
    g = CG.Gate(np.random.default_rng(seed))
    cam0 = np.array((0.0, 0.0, 0.8))
    out = []
    for k in range(n):
        t = k * RC.GATE_DT
        cam = cam0 + np.array(cam_v) * (t % 2.0)          # walks back and forth, ram stays in view
        parts = _ram(dist_m, 1.6 + speed_mps * t)
        g.observe(t, cam, [("P", parts, np.array((dist_m, 0.0, 1.55)))], _Open(), None)
        out += g.ready(t + 1.0)
    return [tg for o in out for tg in o["targets"]], g


def test_interface_matches_the_v0_gate():
    for name in ("due", "observe", "ready"):
        assert callable(getattr(CG.Gate, name))
    g = CG.Gate(np.random.default_rng(0))
    assert g.n_obs == 0 and g.source == CG.GATE_SOURCE and g.due(0.0)


def test_plant_builds_the_r_w3_gate():
    g = PL.make_gate(1, 0, "k")
    assert isinstance(g, CG.Gate) and g.source == CG.GATE_SOURCE


def test_pixel_shift_formula():
    assert math.isclose(float(CG.pixel_shift(np.array([0.5]), np.array([4.0]))[0]),
                        0.5 * RC.GATE_FX / 4.0 * CG.N_DIFF / CG.FRAME_HZ)


def test_fast_near_ram_is_seen_moving():
    tg, _ = _run(0.3, 3.0)
    vis = [t for t in tg if t["visible"]]
    assert len(vis) > 40
    assert all(t["motion_seen"] for t in vis[1:])


def test_slow_ram_below_the_pixel_threshold_is_missed():
    # 0.06 m/s at 5.5 m: above the v0 gate's 0.05 m/s, but below K_PX over the frame-difference window
    speed, dist = 0.06, 5.5
    assert float(CG.pixel_shift(np.array([speed]), np.array([dist]))[0]) < CG.K_PX
    tg, _ = _run(speed, dist)
    vis = [t for t in tg if t["visible"]]
    assert vis and not any(t["motion_seen"] for t in vis)


def test_still_ram_and_still_robot_has_almost_no_false_motion():
    tg, g = _run(0.0, 3.0, n=200)
    vis = [t for t in tg if t["visible"]]
    assert vis and sum(t["motion_seen"] for t in vis) <= 2


def test_walking_body_sway_gives_false_motion_on_a_still_ram():
    tg, g = _run(0.0, 3.0, cam_v=(0.0, 0.6, 0.0), n=200)
    vis = [t for t in tg if t["visible"]]
    assert sum(t["motion_seen"] for t in vis) > 20 and g.n_false > 20


def test_detection_probability_drops_with_range_and_occlusion():
    assert CG.p_detect(1.0, 2.0) > CG.p_detect(1.0, 20.0) > 0.0
    assert CG.p_detect(0.4, 2.0) == 0.0
    assert CG.p_detect(0.75, 2.0) < CG.p_detect(1.0, 2.0)


def test_latency_includes_the_frame_difference_window():
    g = CG.Gate(np.random.default_rng(0))
    g.observe(0.0, np.array((0.0, 0.0, 0.8)), [("P", _ram(3.0, 1.6), np.array((3.0, 0.0, 1.55)))], _Open(), None)
    assert g.ready(CG.LATENCY_S - 0.01) == []
    assert len(g.ready(CG.LATENCY_S + 1e-6)) == 1
    assert math.isclose(CG.LATENCY_S, RC.LATENCY_S + CG.N_DIFF / CG.FRAME_HZ)


class _Always:
    """Detection always succeeds, no sway (isolates the stale-frame rule)."""

    def normal(self, *a, **k):
        return 0.0

    def random(self):
        return 0.0


def test_no_recent_previous_frame_is_unknown_not_still():
    """Fail closed (R-W3): first sight of a press, or a previous frame older than STALE_DT (out of range, back in),
    gives visible = False, never visible with motion_seen = False for a ram that may be moving."""
    g = CG.Gate(np.random.default_rng(0))
    g.rng = _Always()                                   # type: ignore[assignment]
    cam, aim = np.array((0.0, 0.0, 0.8)), np.array((3.0, 0.0, 1.55))

    def obs(t: float, z: float) -> dict:
        g.observe(t, cam, [("P", _ram(3.0, z), aim)], _Open(), None)
        return g.ready(t + 1.0)[-1]["targets"][0]

    assert obs(0.0, 1.6)["visible"] is False                      # first sight
    second = obs(RC.GATE_DT, 1.6)
    assert second["visible"] is True and second["motion_seen"] is False
    # a cycling ram that is back at the same height after a long gap: averaged speed ~ 0 -> must be unknown
    late = obs(RC.GATE_DT + 5.0, 1.6)
    assert late["visible"] is False and late["motion_seen"] is False
    assert obs(2 * RC.GATE_DT + 5.0, 1.6)["visible"] is True
    assert CG.STALE_DT == 1.5 * RC.GATE_DT
