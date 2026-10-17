"""Pure-logic tests for intercycle_nav (no ROS): verification, TTL, masks, SIPP timing, camera gate, no-truth imports.

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))

from intercycle import scenario as S  # noqa: E402
from intercycle.codec import ble  # noqa: E402
from intercycle.codec import maplayer as ML  # noqa: E402
from intercycle.codec.layout import FAST, encode_fast  # noqa: E402
from intercycle.codec.sign import demo_key, sign  # noqa: E402
from intercycle.observer import mapping as M  # noqa: E402
from intercycle.types import DAY0_UNIX, State  # noqa: E402

from intercycle_nav import feed as F  # noqa: E402
from intercycle_nav import masks as K  # noqa: E402
from intercycle_nav import timing as T  # noqa: E402
from intercycle_nav.broadcast import Broadcaster, SyntheticSchedule  # noqa: E402

NPZ = REPO / "out" / "layered_map.npz"
SITE = K.Site.from_scenario()
GRID = K.Grid(-2.0, -2.0, 0.25, 496, 208)
T0 = 10 * 3600.0


class _MemWriter:
    def __init__(self, tr):
        self.tr = tr

    def frames(self, frs):
        for f in frs:
            self.tr.push(F.FRAME + f)


def _sched():
    scn = S.load()
    return SyntheticSchedule(scn.site_id, scn.key_id, [60.0] * 8 + [45.0, 60.0], [m.busy for m in scn.machines])


def _cell(x, y):
    return int((x - GRID.x0) / GRID.res), int((y - GRID.y0) / GRID.res)


@pytest.fixture(scope="module")
def learned():
    if not NPZ.exists():
        pytest.skip("out/layered_map.npz not built (scripts/build_map.py)")
    lm = M.LayeredMap.load(NPZ)
    key = demo_key(SITE.key_id)
    return ML.pack_map(lm, key, version=2, issue_time=DAY0_UNIX + T0 - 60.0)


def _run(state, bc, tr, t0, t1, dt=0.05):
    t = t0
    while t < t1:
        bc.tick(t)
        for r in tr.poll():
            state.ingest(t, r)
        t += dt
    return t


def test_forged_fast_layer_is_rejected():
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    sch = _sched()
    f = sch.fast(T0)
    good = ble.frame(sign(encode_fast(f), demo_key(SITE.key_id)), layer=FAST)
    forged_layer = replace(f, seq=f.seq + 1, machines=[replace(m, state=State.IDLE, next_idle_len_s=200.0, ttl_s=300.0)
                                                        for m in f.machines])
    forged = ble.frame(sign(encode_fast(forged_layer), demo_key(4242)), layer=FAST)
    for fr in good:
        st.ingest(T0, F.FRAME + fr)
    for fr in forged:
        st.ingest(T0 + 0.1, F.FRAME + fr)
    st.ingest(T0 + 0.2, F.BLOB + b"\x00" * 90)               # junk blob
    s = st.all_stats()
    assert s["layers"].get("bad_signature", 0) == 1
    assert s["bridge"]["ok"] == 1
    snap = st.snapshot(T0 + 0.3)
    assert snap.machines[0].state == State(f.machines[0].state).name   # the forged "IDLE for 200 s" never took effect


def test_ttl_expiry_makes_machines_unknown_and_closes_zones():
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    sch = _sched()
    tr = F.MemoryTransport()
    bc = Broadcaster(_MemWriter(tr), demo_key(SITE.key_id), None, sch.slow, sch.fast)
    _run(st, bc, tr, T0, T0 + 2.0)
    snap = st.snapshot(T0 + 2.0)
    assert all(m.phase is not None for m in snap.machines)
    late = st.snapshot(T0 + 2.0 + 10.0)                      # ttl 4 s, nothing new heard
    assert all(m.state == "UNKNOWN" and m.phase is None and m.windows is None for m in late.machines)


def test_static_and_none_conditions():
    g, _ = K.keepout_mask(GRID, "none", SITE, None, None, K.MaskParams())
    assert not g.any()
    g, info = K.keepout_mask(GRID, "static", SITE, None, None, K.MaskParams())
    for mid, (x0, y0, x1, y1) in SITE.rects.items():
        assert g[_cell((x0 + x1) / 2, (y0 + y1) / 2)] == K.KEEP_OUT, mid
    assert g[_cell(4.0, 32.0)] == 0                           # dock / north aisle stays free
    # learned condition without any map -> conservative static fallback
    g2, info2 = K.keepout_mask(GRID, "learned", SITE, None, None, K.MaskParams())
    assert info2["source"] == "static_fallback_no_map" and (g2 == g).all()


def test_learned_mask_follows_the_predicted_idle_window(learned):
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    sch = _sched()
    tr = F.MemoryTransport()
    bc = Broadcaster(_MemWriter(tr), demo_key(SITE.key_id), learned, sch.slow, sch.fast)
    t = _run(st, bc, tr, T0, T0 + 12.0)
    assert st.map_view(t) is not None, st.all_stats()
    p = K.MaskParams()
    core = T.TimingCore.from_scenario(S.load())
    ix, iy = core.graph.pos[core.inside["HEM1"]]
    seen = {"open_idle": 0, "closed_active": 0, "open_active": 0}
    for k in range(90):                                    # two HEM1 cycles (45 s)
        tq = t + k * 1.0
        _run(st, bc, tr, tq, tq + 1.0)
        snap = st.snapshot(tq + 1.0)
        g, info = K.keepout_mask(GRID, "learned", SITE, snap, st.map_view(tq + 1.0), p)
        st_now, _, _ = sch.state_at(8, tq + 1.0)
        st_soon = {sch.state_at(8, tq + 1.0 + d)[0] for d in np.arange(0, p.horizon_s + 0.01, 0.5)}
        blocked = g[_cell(ix, iy)] > 0
        if st_soon == {State.IDLE}:
            seen["open_idle"] += not blocked
        if State.ACTIVE in st_soon:
            seen["closed_active"] += blocked
            seen["open_active"] += not blocked
    assert seen["open_active"] == 0, seen                  # never open while active within the horizon
    assert seen["closed_active"] > 0 and seen["open_idle"] > 0, seen


def test_held_zone_and_self_clear(learned):
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    sch = _sched()
    tr = F.MemoryTransport()
    bc = Broadcaster(_MemWriter(tr), demo_key(SITE.key_id), learned, sch.slow, sch.fast)
    t = _run(st, bc, tr, T0, T0 + 12.0)
    core = T.TimingCore.from_scenario(S.load())
    xy = core.graph.pos[core.inside["HEM1"]]
    snap = st.snapshot(t, robot_xy=xy)
    hem = [m for m in snap.machines if m.id == "HEM1"][0]
    assert hem.held
    g, info = K.keepout_mask(GRID, "learned", SITE, snap, st.map_view(t), K.MaskParams(), robot_xy=xy)
    assert g[_cell(*xy)] == 0 and info["bins"]["HEM1"]["bins"] == "held"


def test_speed_mask_has_limits(learned):
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    sch = _sched()
    tr = F.MemoryTransport()
    bc = Broadcaster(_MemWriter(tr), demo_key(SITE.key_id), learned, sch.slow, sch.fast)
    t = _run(st, bc, tr, T0, T0 + 12.0)
    g, info = K.speed_mask(GRID, "learned", st.snapshot(t), st.map_view(t), K.MaskParams())
    vals = set(np.unique(g).tolist())
    assert vals <= {0, 40, 60, 80} and len(vals) > 1


def test_file_transport_roundtrip(tmp_path):
    path = tmp_path / "feed.bin"
    w = F.FeedWriter(f"file://{path}")
    w.frames([b"abc", b"de"])
    w.blob(b"xyz")
    tr = F.open_transport(f"file://{path}")
    assert tr.poll() == [F.FRAME + b"abc", F.FRAME + b"de", F.BLOB + b"xyz"]
    w.frames([b"q"])
    assert tr.poll() == [F.FRAME + b"q"]


def test_sipp_waits_for_the_window_and_picks_the_earliest_target():
    core = T.TimingCore.from_scenario(S.load())
    tgs = {t.id: t for t in core.graph.targets}
    t0 = 1000.0
    xy = (4.0, 32.0)
    late = {"HEM1": [(t0 + 300.0, t0 + 330.0)], "SEAL1": [(t0 + 100.0, t0 + 130.0)]}
    tp = core.choose([tgs["HEM1"], tgs["SEAL1"]], t0, xy, late, "learned")
    assert tp.target.id == "SEAL1"
    assert tp.t_depart_wait >= t0 + 100.0 - 1e-6           # never cross the boundary before the window opens
    assert tp.t_goal >= t0 + 100.0 + core.enter_time(tp.target) - 1e-6
    assert core.need_s(tp.target) == pytest.approx(core.enter_s + core.dwell_s + core.exit_s + core.margin_s)
    short = {"HEM1": [(t0 + 50.0, t0 + 52.0)]}             # shorter than need_s: no plan
    assert core.plan_target(tgs["HEM1"], t0, xy, short, "learned") is None
    assert core.plan_target(tgs["HEM1"], t0, xy, {}, "learned") is None     # unknown windows: never enter
    st = core.plan_target(tgs["HEM1"], t0, xy, {}, "static")
    assert st is not None and st.inside_xy is None                        # static: inspect from the boundary
    assert core.plan_target(tgs["HEM1"], t0, xy, {}, "none").inside_xy is not None
    assert T.window_open([(10.0, 30.0)], 15.0, 10.0) and not T.window_open([(10.0, 30.0)], 25.0, 10.0)


def test_camera_gate():
    cg = T.CameraGate(confirm_n=2, min_span_s=0.3, fresh_s=1.0)
    assert not cg.clear("S10", 0.0)
    cg.observe(0.0, "S10", True, False)
    cg.observe(0.2, "S10", True, False)
    assert not cg.clear("S10", 0.2)                         # span 0.2 < 0.3
    cg.observe(0.4, "S10", True, False)
    assert cg.clear("S10", 0.4)
    assert not cg.clear("S10", 2.0)                         # stale
    cg.observe(0.6, "S10", True, True)                      # motion seen
    assert not cg.clear("S10", 0.6)
    cg.observe(0.8, "S10", False, False)                    # not visible: not a confirmation
    assert not cg.clear("S10", 0.8)


def test_robot_modules_never_import_truth_or_sim():
    code = ("import sys; sys.path.insert(0, %r); import intercycle_nav.masks, intercycle_nav.timing, intercycle_nav.feed;"
            "bad=[m for m in sys.modules if m.startswith(('intercycle.truth','intercycle.sim'))]; print(bad); "
            "sys.exit(1 if bad else 0)") % str(PKG)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert r.returncode == 0, r.stdout + r.stderr
