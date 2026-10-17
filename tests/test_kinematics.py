"""Ground-truth machine kinematics: moving-part bodies as a function of cycle phase."""
import time

import numpy as np
import pytest

from intercycle.robot.sitegraph import TRANSFER_S
from intercycle.scenario import load
from intercycle.sim.world import Freeze, World
from intercycle.truth import kinematics as K
from intercycle.truth.kinematics import Kinematics

T0 = 7 * 3600.0
T1 = T0 + 3600.0
WELDERS = ["S10", "S20", "S30", "S40", "S50", "S60", "S70"]


@pytest.fixture(scope="module")
def scn():
    return load()


@pytest.fixture(scope="module")
def base(scn):
    w = World(scn, 0, T0, T1)
    return w, Kinematics(scn, w)


@pytest.fixture()
def fresh(scn):
    w = World(scn, 0, T0, T1)
    return w, Kinematics(scn, w)


def rect_margin(c: np.ndarray, rect) -> float:
    """Smallest distance from any circle's edge to the rectangle boundary (negative = pokes out)."""
    x0, z0, x1, z1 = rect
    return float(min((c[:, 0] - c[:, 2] - x0).min(), (x1 - c[:, 0] - c[:, 2]).min(),
                     (c[:, 1] - c[:, 2] - z0).min(), (z1 - c[:, 1] - c[:, 2]).min()))


def overlaps_rect(c: np.ndarray, rect) -> bool:
    x0, z0, x1, z1 = rect
    nx, nz = np.clip(c[:, 0], x0, x1), np.clip(c[:, 1], z0, z1)
    return bool((np.hypot(c[:, 0] - nx, c[:, 1] - nz) < c[:, 2]).any())


def window_time(ws, i, f):
    """Time at active fraction f of window i."""
    return ws[i].bs + f * (ws[i].is_ - ws[i].bs)


# ---------------------------------------------------------------------------------------------------- structure
def test_api_shapes_and_read_only(base, scn):
    w, kin = base
    t = T0 + 600.3
    b = kin.bodies(t)
    assert set(b) == {m.id for m in scn.machines} | {"SKIDS", "AGVDOCK"}
    for mid in WELDERS:
        assert b[mid].shape == (2 * K.N_ARM_C, 3)
    assert b["HEM1"].shape == b["SEAL1"].shape == (2 * K.N_ARM_C, 3)
    allc = kin.all_circles(t)
    assert allc.ndim == 2 and allc.shape[1] == 3
    assert len(allc) == sum(len(v) for v in b.values())
    assert np.isfinite(allc).all() and (allc[:, 2] > 0).all()
    assert not allc.flags.writeable and not b["S10"].flags.writeable


def test_deterministic(scn):
    w1, w2 = World(scn, 3, T0, T1), World(scn, 3, T0, T1)
    k1, k2 = Kinematics(scn, w1), Kinematics(scn, w2)
    for t in (T0 + 100.0, T0 + 777.7, T0 + 1500.1):
        assert np.array_equal(k1.all_circles(t), k2.all_circles(t))


def test_phase_matches_world(base, scn):
    w, kin = base
    rng = np.random.default_rng(1)
    for t in T0 + rng.uniform(60, 3000, 300):
        for m in scn.machines:
            st, fr = kin.phase(m.id, float(t))
            assert st in ("active", "idle")
            assert 0.0 <= fr <= 1.0
            assert (st == "active") == w.active(m.id, float(t))


def test_poses_render_friendly(fresh, scn):
    w, kin = fresh
    ws = w.W["S30"]
    t = window_time(ws, 5, 0.5)
    for frozen in (None, "trip", "hold"):
        if frozen:
            w.groups["line"].freezes.append(Freeze(t - 1.0, t + 30.0, frozen, "S30"))
        p = kin.poses(t)
        assert set(p) == {m.id for m in scn.machines} | {"SKIDS", "AGVDOCK"}
        arms = p["S30"]["arms"]
        assert len(arms) == 2 and p["S30"]["state"] == (frozen or "active")
        circ = kin.bodies(t)["S30"].reshape(2, K.N_ARM_C, 3)
        for a, c in zip(arms, circ):
            assert np.allclose(a["tool_xy"], c[4, :2]) and np.allclose(a["elbow_xy"], c[2, :2])
            assert np.isfinite([a["shoulder_yaw"], a["elbow"]]).all() and a["stowed"] == (frozen == "hold")
            assert np.hypot(*(np.array(a["tool_xy"]) - a["base"])) <= 2 * K.LINK
        assert p["SKIDS"]["z"] == 24.0 and len(p["SKIDS"]["x"]) in (7, 8)
        assert p["S80"]["lifter_h"] > 0 and isinstance(p["AGVDOCK"]["docked"], list)
        w.groups["line"].freezes.clear()


# ---------------------------------------------------------------------------------------------------- envelopes
def test_bodies_inside_zone_rect(base, scn):
    w, kin = base
    worst = {m.id: np.inf for m in scn.machines}
    for t in np.arange(T0 + 120, T0 + 120 + 600, 0.25):
        b = kin.bodies(float(t))
        for m in scn.machines:
            c = b[m.id]
            if len(c):
                worst[m.id] = min(worst[m.id], rect_margin(c, m.rect))
    for mid, mg in worst.items():
        assert mg >= -0.2, (mid, mg)


def test_tip_dress_reaches_boundary_weld_does_not(base, scn):
    w, kin = base
    for mid in WELDERS:
        m = scn.machine(mid)
        ws = w.W[mid]
        weld, dress, posts = [], [], []
        for i in (4, 5):
            for f in np.linspace(0.0, 0.8, 81, endpoint=False):
                weld.append(rect_margin(kin.bodies(window_time(ws, i, f))[mid], m.rect))
            for f in np.linspace(0.86, 0.92, 31):
                dress.append(rect_margin(kin.bodies(window_time(ws, i, f))[mid], m.rect))
        assert min(weld) > 1.3, (mid, min(weld))
        assert min(dress) < 1.3, (mid, min(dress))
        assert min(dress) > 0.0, (mid, min(dress))     # still inside the zone


def test_weld_points_over_body_and_dresser_at_aisle(base, scn):
    w, kin = base
    for a, spec in enumerate(kin.arm_specs):
        m = scn.machine(spec.mid)
        if not m.on_line:
            continue
        pts, dress = spec.q[1:-2], spec.q[-2]
        assert (np.abs(pts[:, 1] - 24.0) <= 1.2 + 1e-9).all()
        assert abs(abs(dress[1] - 24.0) - K.DRESS_DZ) < 1e-9
        assert abs(pts[:, 0] - m.x).max() <= 2.2             # sweep stays in x within the station
        tool = kin._tab[a, :K.N_TAB, 4, :2]                  # tool circle through the active fraction
        for p in pts:                                        # every weld point is visited
            assert np.hypot(*(tool - p).T).min() < 0.02
        assert np.hypot(*(tool - dress).T).min() < 0.02      # and so is the dresser


def test_home_pose_is_stowed_over_base(base, scn):
    w, kin = base
    t = window_time(w.W["S30"], 5, 0.0) + 0.5 * 12.0 + 48.0      # idle part of the cycle
    assert kin.phase("S30", t)[0] == "idle"
    c = kin.bodies(t)["S30"].reshape(2, K.N_ARM_C, 3)
    for arm, (mid, bx, bz) in zip(c, [b for b in kin.arm_bases if b[0] == "S30"]):
        assert np.allclose(arm[:, :2], [bx, bz])                # every circle sits over the base
        assert arm[:, 2].max() <= 0.65 + 1e-9
    # robot bases at (x -+ 1.8, 24 -+ 3.3)
    xs = sorted((round(bx, 3), round(bz, 3)) for mid, bx, bz in kin.arm_bases if mid == "S30")
    assert xs == [(28.2, 20.7), (31.8, 27.3)]


def test_arm_motion_is_continuous(base, scn):
    w, kin = base
    dt = 0.05
    for mid in ("S20", "S60", "HEM1", "SEAL1"):
        ws = w.W[mid]
        ts = np.arange(ws[6].bs - 1.0, ws[7].bs + 1.0, dt)
        P = np.stack([kin.bodies(float(t))[mid] for t in ts])
        v = np.hypot(*np.diff(P[:, :, :2], axis=0).transpose(2, 0, 1)) / dt
        assert v.max() < 5.0, (mid, v.max())


def test_cells_move_around_fixture(base, scn):
    w, kin = base
    reach = {"HEM1": (1.5, 0.9, 1.8, 1.1, 0.15), "SEAL1": (1.0, 0.4, 1.4, 0.7, 0.30)}   # |dx|, |dz| bounds; f from
    for mid, (rx, rz, mx, mz, f0) in reach.items():
        m = scn.machine(mid)
        ws = w.W[mid]
        ts = np.array([window_time(ws, 3, f) for f in np.linspace(f0, 0.88, 400)])
        tool = np.stack([kin.bodies(float(t))[mid].reshape(2, K.N_ARM_C, 3)[:, 4, :2] for t in ts])   # (T, 2, 2)
        d = np.abs(tool - [m.x, m.z])
        assert d[..., 0].max() > rx and d[..., 1].max() > rz       # works over the 3.0 x 1.6 fixture
        assert d[..., 0].max() < mx and d[..., 1].max() < mz       # and stays within its clearance
        assert not np.allclose(tool[0], tool[-1])
    # roller path: the loop has four distinct corners
    spec = [s for s in kin.arm_specs if s.mid == "HEM1"][0]
    assert len(np.unique(np.round(spec.q[1:-1], 3), axis=0)) == 4


# ---------------------------------------------------------------------------------------------------- skids
def _transfer_flag(w, t):
    g = w.group("S10")
    win = w.window("S10", t)
    return win is not None and 0.0 <= g.mt(t) - win.bs < TRANSFER_S


def test_skids_overlap_walkway_only_during_transfer(base):
    w, kin = base
    hit_in, hit_out, n_tr = 0, 0, 0
    for t in np.arange(T0 + 100, T0 + 100 + 360, 0.1):
        t = float(t)
        sk = kin.bodies(t)["SKIDS"]
        tr = _transfer_flag(w, t)
        ov = overlaps_rect(sk, K.WALK_RECT)
        n_tr += tr
        if ov and tr:
            hit_in += 1
        if ov and not tr:
            hit_out += 1
    assert hit_out == 0
    assert hit_in > 0.5 * 10 * 6 * 0.5     # overlaps for a good part of every transfer (6 cycles x ~8 s x 10 Hz)
    assert n_tr > 0


def test_skids_stay_on_conveyor_and_move_one_pitch(base, scn):
    w, kin = base
    x0, x1 = K.LINE_X
    for t in np.arange(T0 + 100, T0 + 100 + 180, 0.1):
        sk = kin.bodies(float(t))["SKIDS"]
        assert (sk[:, 1] - sk[:, 2] >= 24 - 1.2 - 1e-9).all() and (sk[:, 1] + sk[:, 2] <= 24 + 1.2 + 1e-9).all()
        assert (sk[:, 0] - sk[:, 2] >= x0 - 1e-6).all() and (sk[:, 0] + sk[:, 2] <= x1 + 1e-6).all()
    ws = w.W["S10"]
    a = np.array(kin.poses(ws[5].bs + 0.0)["SKIDS"]["x"])
    b = np.array(kin.poses(ws[5].bs + TRANSFER_S + 0.01)["SKIDS"]["x"])
    assert len(a) == len(b) == 8
    pitch = b - a
    assert np.allclose(pitch[1:7], 9.0)                     # every body moves +9 m (S70 -> S80 hop is 12 m)
    assert np.isclose(pitch[7], 12.0)
    # body speed bounded (cubic ease: peak 1.5 x mean)
    ts = np.arange(ws[5].bs, ws[5].bs + TRANSFER_S, 0.05)
    xs = np.stack([kin.bodies(float(t))["SKIDS"][1::3, 0] for t in ts])
    assert (np.abs(np.diff(xs, axis=0)).max() / 0.05) < 1.5 * 12 / TRANSFER_S * 1.1       # cubic ease, 12 m hop
    # at rest no body touches the walkway gap
    for t in (ws[5].bs + 9.0, ws[5].bs + 30.0, ws[5].bs + 55.0):
        assert not overlaps_rect(kin.bodies(float(t))["SKIDS"], K.WALK_RECT)


# ---------------------------------------------------------------------------------------------------- lifter / AGV
def test_lifter_carriage_descends_during_active(base, scn):
    w, kin = base
    m = scn.machine("S80")
    ws = w.W["S80"]
    prof = []
    for f in np.linspace(0, 1, 101, endpoint=False):
        t = window_time(ws, 6, f)
        c = kin.bodies(t)["S80"]
        prof.append(len(c))
        if len(c):
            assert rect_margin(c, m.rect) >= 0.0
            assert abs(c[:, 0].mean() - m.x) < 0.01 and abs(c[:, 1].mean() - m.z) < 0.01      # centred on the station
    prof = np.array(prof)
    assert prof[:15].sum() == 0 and prof[85:].sum() == 0 and prof[40] == 3        # overhead at both ends, down at 0.4
    h = [kin.poses(window_time(ws, 6, f))["S80"]["lifter_h"] for f in (0.0, 0.2, 0.4, 0.7, 0.95)]
    assert h[0] > 5 and h[2] < 0.5 and h[1] < h[0] and h[3] > h[2]
    # idle: no carriage, and the picked body is gone from the conveyor until the next transfer
    t_idle = ws[6].is_ + 1.0
    assert len(kin.bodies(t_idle)["S80"]) == 0
    assert len(kin.bodies(window_time(ws, 6, 0.2))["SKIDS"]) == 24
    assert len(kin.bodies(window_time(ws, 6, 0.6))["SKIDS"]) == 21


def test_agv_dock_every_sixth_cycle_for_20_s(base, scn):
    w, kin = base
    t_all = np.arange(T0 + 3.0, T0 + 3.0 + 1500.0, 0.5)
    for j, mid in enumerate(WELDERS + ["S80"]):
        m = scn.machine(mid)
        occ = []
        for t in t_all:
            c = kin.bodies(float(t))["AGVDOCK"]
            sel = c[np.abs(c[:, 0] - m.x) < 1.0]
            occ.append(len(sel) > 0)
            if len(sel):
                assert np.allclose(sel[:, 1], K.DOCK_Z)
                assert (sel[:, 1] - sel[:, 2] > m.rect[3]).all()                         # outside the hazard zone
        occ = np.array(occ)
        edges = np.flatnonzero(np.diff(occ.astype(int)) == 1) + 1
        assert occ.sum() * 0.5 / max(len(edges), 1) == pytest.approx(20.0, abs=1.0)
        bs = [wd.bs for wd in w.W[mid]]
        idx = [int(np.searchsorted(bs, t_all[e], side="right")) - 1 for e in edges]
        assert len(idx) >= 3
        assert np.all(np.diff(idx) == 6), (mid, idx)                                      # every 6th cycle
        assert np.allclose([t_all[e] - bs[i] for e, i in zip(edges, idx)], K.DOCK_T0, atol=0.6)   # same phase each time
    # stations are staggered: not all docks are busy at once
    n = [len(kin.bodies(float(t))["AGVDOCK"]) // 2 for t in t_all]
    assert 0 < max(n) <= 3


# ---------------------------------------------------------------------------------------------------- freezes
def test_trip_freezes_everything_mid_motion(fresh, scn):
    w, kin = fresh
    ws = w.W["S30"]
    ta = window_time(ws, 5, 0.4)                       # S30 mid-weld; line index 3 s earlier -> skids at rest
    ts = ws[5].bs - 3.0 + 4.0                          # a time 4 s into the transfer window: skids mid-motion
    poses = {}
    for t_trip in (ta, ts):
        w2 = World(scn, 0, T0, T1)
        k2 = Kinematics(scn, w2)
        before = k2.bodies(t_trip)
        poses[t_trip] = before["S30"].copy()
        w2.groups["line"].freezes.append(Freeze(t_trip, t_trip + 90.0, "trip", "S30"))
        for dt in (0.0, 0.1, 5.0, 45.0, 89.9):
            now = k2.bodies(t_trip + dt)
            for key in before:
                if key not in ("HEM1", "SEAL1"):                     # the cells run their own clocks
                    assert np.array_equal(before[key], now[key]), (key, dt)
            assert k2.phase("S30", t_trip + dt)[0] == "trip"
        assert k2.phase("S30", t_trip + 45.0)[1] == pytest.approx(0.5, abs=0.01)
        # it resumes exactly where it stopped
        after = k2.bodies(t_trip + 90.0 + 2.0)
        ref = World(scn, 0, T0, T1)
        kref = Kinematics(scn, ref).bodies(t_trip + 2.0)
        for key in kref:
            if key not in ("HEM1", "SEAL1"):
                assert np.allclose(after[key], kref[key]), key
    # a tripped S30 is mid-motion, not at home
    home = np.repeat(np.array([[28.2, 20.7], [31.8, 27.3]]), K.N_ARM_C, axis=0)
    assert np.abs(poses[ta][:, :2] - home).max() > 0.5
    assert len(poses[ts]) == 2 * K.N_ARM_C


def test_trip_in_cell_group_only_stops_that_cell(fresh, scn):
    w, kin = fresh
    ws = w.W["HEM1"]
    t = window_time(ws, 4, 0.5)
    w.groups["HEM1"].freezes.append(Freeze(t, t + 90.0, "trip", "HEM1"))
    a, b = kin.bodies(t + 10.0), kin.bodies(t + 60.0)
    assert np.array_equal(a["HEM1"], b["HEM1"])
    assert not np.array_equal(a["S30"], b["S30"]) or not np.array_equal(a["S20"], b["S20"])   # the line keeps running
    assert kin.phase("S30", t + 10.0)[0] in ("active", "idle")


def test_hold_stays_at_home(fresh, scn):
    w, kin = fresh
    ws = w.W["S30"]
    t = window_time(ws, 5, 0.4)
    idle_pose = kin.bodies(ws[5].is_ + 5.0)["S30"].copy()
    w.groups["line"].freezes.append(Freeze(t - 1.0, t + 20.0, "hold", "S30"))
    for dt in (0.0, 3.0, 19.0):
        c = kin.bodies(t + dt)
        assert np.array_equal(c["S30"], idle_pose)
        assert kin.phase("S30", t + dt)[0] == "hold"
        assert len(c["S80"]) == 0
    assert kin.phase("S30", t + 19.0)[1] == pytest.approx(20.0 / 21.0, abs=1e-6)
    # a hold at the start of the cycle (machine about to start) keeps the line as it was, nothing jumps
    t_next = ws[6].bs
    w.groups["line"].freezes.append(Freeze(t_next + 5.0, t_next + 15.0, "hold", "S30"))   # note: freezes shift later times
    a, b = kin.bodies(t_next + 6.0), kin.bodies(t_next + 14.0)
    assert np.array_equal(a["SKIDS"], b["SKIDS"])


def test_machine_time_matches_group(fresh):
    w, kin = fresh
    rng = np.random.default_rng(5)
    t = T0 + 50.0
    for _ in range(40):
        t += rng.uniform(5, 40)
        d = float(rng.choice([0.1, 2.0, 15.0, 90.0]))
        w.groups["line"].freezes.append(Freeze(t, t + d, "trip" if d > 50 else "hold", "S30"))
        t += d * rng.uniform(0.3, 1.5)
    g = w.group("S30")
    grid = np.arange(T0, t + 200, 1.7)
    for tt in grid:                                      # monotone
        assert kin.machine_time("S30", float(tt)) == pytest.approx(g.mt(float(tt)), abs=1e-9)
    for tt in rng.permutation(grid):                     # arbitrary order: the cache must never lie
        assert kin.machine_time("S30", float(tt)) == pytest.approx(g.mt(float(tt)), abs=1e-9)
    # overlapping and out-of-order freezes
    g.freezes.append(Freeze(T0 + 400.0, T0 + 500.0, "trip", "S30"))
    g.freezes.append(Freeze(T0 + 450.0, T0 + 470.0, "hold", "S30"))
    for tt in rng.permutation(grid):
        assert kin.machine_time("S30", float(tt)) == pytest.approx(g.mt(float(tt)), abs=1e-9)
    assert kin.phase("S30", T0 + 460.0)[0] == "trip"       # a trip wins over a hold


def test_cache_follows_freezes(fresh):
    w, kin = fresh
    ws = w.W["S30"]
    t = window_time(ws, 5, 0.4)
    a = kin.all_circles(t).copy()
    assert np.array_equal(kin.all_circles(t), a)
    w.groups["line"].freezes.append(Freeze(t - 5.0, t + 20.0, "hold", "S30"))
    b = kin.all_circles(t)
    assert not np.array_equal(a, b)                       # same t, new freeze: recomputed (S30 now at home)


# ---------------------------------------------------------------------------------------------------- budget
def test_timing_budget(scn):
    w = World(scn, 0, T0, T0 + 6 * 3600)
    kin = Kinematics(scn, w)
    t = T0 + 100.0
    for i in range(30):                                   # a lived-in run: many trips and holds so far
        w.groups["line"].freezes.append(Freeze(t, t + (90.0 if i % 3 == 0 else 3.0), "trip" if i % 3 == 0 else "hold", "S10"))
        t += 600.0
    ts = (T0 + 17000.0 + np.arange(3000) * 0.1).tolist()
    res = {}
    for name, fn in (("all_circles", kin.all_circles), ("bodies", kin.bodies)):
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            for tt in ts:
                fn(tt)
            best = min(best, (time.perf_counter() - t0) / len(ts))
        res[name] = best
    print({k: f"{v * 1e6:.1f} us" for k, v in res.items()})
    assert res["all_circles"] < 100e-6 and res["bodies"] < 100e-6, res
