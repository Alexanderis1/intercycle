"""M-W10 camera watchdog (lane S4, PLAN_TWIN §4; WEAKNESSES M-W10): image health -> cam_floor_ok -> UNKNOWN per camera.

Image side (intercycle/observer/watchdog.py): a frozen, black, obstructed or blurred camera is flagged; a healthy one
is not (including under a lighting change); the flag is lifted once the image is healthy again for recover_s.
Map side (mapping.py): cam_floor_ok False -> every cell of that camera's floor region is UNKNOWN (not p = 0), its
detections are not evidence and it gets no visits; a camera with no visits in a slice is UNKNOWN in that slice;
UNKNOWN is lifted when health returns. Fast; no floor simulation, no video.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from intercycle.codec import maplayer as ML
from intercycle.observer import mapping as M
from intercycle.observer import watchdog as W
from intercycle.scenario import load

SCN = load()
NCAM = len(SCN.cameras)
H, WID = 72, 128
DT = 0.5                                      # 2 Hz


def _scene(seed: int = 0) -> np.ndarray:
    """A textured grey scene (floor stripes, boxes, fine texture), uint8."""
    rng = np.random.default_rng(seed)
    img = 90 + 40 * rng.random((H, WID))
    img[:, ::16] += 60                                        # stripes
    img[20:40, 30:60] = 200                                   # a bright box
    img[45:65, 80:120] = 30                                   # a dark box
    return np.clip(img, 0, 255)


def _frame(base: np.ndarray, rng, noise: float = 2.0) -> np.ndarray:
    """One live frame: the scene plus sensor noise (so consecutive frames never hash equal)."""
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def _box_blur(img: np.ndarray, k: int = 7) -> np.ndarray:
    pad = k // 2
    p = np.pad(img.astype(float), pad, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    out = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    return out[:img.shape[0], :img.shape[1]]


def _run(wd: W.CameraWatchdog, frames, t0: float = 0.0):
    out = []
    for i, f in enumerate(frames):
        out.append(wd.update(t0 + i * DT, f))
    return out


def _healthy(n: int, seed: int = 1, base=None):
    rng = np.random.default_rng(seed)
    base = _scene() if base is None else base
    return [_frame(base, rng) for _ in range(n)]


def _warm(params: W.WatchdogParams | None = None) -> tuple[W.CameraWatchdog, float]:
    """A watchdog past warm-up and recovery on healthy frames; returns it and the next tick time."""
    wd = W.CameraWatchdog(params or W.WatchdogParams())
    n = 30
    hs = _run(wd, _healthy(n))
    assert hs[-1].ok and hs[-1].reason == W.OK
    return wd, n * DT


# ------------------------------------------------------------------------------------------------ image side

def test_healthy_camera_is_ok_after_warmup_and_fails_closed_before():
    wd = W.CameraWatchdog()
    hs = _run(wd, _healthy(30))
    assert not hs[0].ok and hs[0].reason == W.WARMUP, "no reference yet: fail closed"
    assert hs[-1].ok
    first_ok = next(i for i, h in enumerate(hs) if h.ok)
    p = W.WatchdogParams()
    assert first_ok * DT >= p.recover_s - 1e-9
    assert all(h.ok for h in hs[first_ok:]), "a healthy camera never flaps"


def test_given_reference_skips_warmup():
    ref = W.reference_from_frames(_healthy(8, seed=7))
    wd = W.CameraWatchdog(ref=ref)
    hs = _run(wd, _healthy(10))
    assert hs[0].reason != W.WARMUP
    assert hs[-1].ok


def test_frozen_camera():
    wd, t = _warm()
    still = _healthy(1, seed=3)[0]
    hs = _run(wd, [still] * 12, t0=t)
    p = W.WatchdogParams()
    k = int(round(p.frozen_s / DT))
    assert all(h.ok for h in hs[:k]), "a repeat shorter than frozen_s is not a fault yet"
    assert all(not h.ok and h.reason == W.FROZEN for h in hs[k:])


def test_black_camera():
    wd, t = _warm()
    hs = _run(wd, [np.zeros((H, WID), np.uint8) + np.uint8(i % 2) for i in range(4)], t0=t)
    assert all(not h.ok and h.reason == W.BLACK for h in hs), [h.reason for h in hs]


def test_uniform_grey_or_covered_lens_is_obstructed():
    wd, t = _warm()
    rng = np.random.default_rng(4)
    grey = [np.clip(140 + rng.normal(0, 1.0, (H, WID)), 0, 255).astype(np.uint8) for _ in range(3)]
    hs = _run(wd, grey, t0=t)
    assert all(not h.ok and h.reason == W.OBSTRUCTED for h in hs)


def test_partly_covered_lens_is_obstructed():
    wd, t = _warm()
    rng = np.random.default_rng(5)
    fr = []
    for f in _healthy(3, seed=6):
        g = f.astype(float)
        g[:, : WID // 2] = 60 + rng.normal(0, 1.0, (H, WID // 2))     # left half covered (a sheet / box at the lens)
        fr.append(np.clip(g, 0, 255).astype(np.uint8))
    hs = _run(wd, fr, t0=t)
    assert all(not h.ok and h.reason == W.OBSTRUCTED for h in hs), [h.reason for h in hs]


def test_blurred_camera():
    wd, t = _warm()
    rng = np.random.default_rng(8)
    blurred = [np.clip(_box_blur(_scene(), 7) + rng.normal(0, 0.3, (H, WID)), 0, 255).astype(np.uint8) for _ in range(4)]
    hs = _run(wd, blurred, t0=t)
    assert all(not h.ok and h.reason == W.BLURRED for h in hs), [(h.reason, round(h.sharp_ratio, 3)) for h in hs]


def test_lighting_change_is_not_a_fault():
    wd, t = _warm()
    rng = np.random.default_rng(9)
    dim = [np.clip(_scene() * 0.4 + rng.normal(0, 1.5, (H, WID)), 0, 255).astype(np.uint8) for _ in range(10)]
    hs = _run(wd, dim, t0=t)
    assert all(h.ok for h in hs), [(h.reason, round(h.sharp_ratio, 3)) for h in hs]


@pytest.mark.parametrize("fault", ["frozen", "black", "blurred"])
def test_recovered_camera_is_lifted_after_recover_s(fault):
    wd, t = _warm()
    p = W.WatchdogParams()
    if fault == "frozen":
        bad = [_healthy(1, seed=3)[0]] * 8
    elif fault == "black":
        bad = [np.zeros((H, WID), np.uint8)] * 4
    else:
        bad = [np.clip(_box_blur(_scene(), 7), 0, 255).astype(np.uint8) + np.uint8(i % 2) for i in range(4)]
    hs = _run(wd, bad, t0=t)
    assert not hs[-1].ok
    t += len(bad) * DT
    hs = _run(wd, _healthy(12, seed=11), t0=t)
    k = int(round(p.recover_s / DT))
    assert all(not h.ok and h.reason == W.RECOVERING for h in hs[:k]), [h.reason for h in hs]
    assert all(h.ok for h in hs[k:]), "UNKNOWN is lifted once the image is healthy again"


def test_missing_frame_is_not_ok_and_restarts_recovery():
    wd, t = _warm()
    h = wd.update(t, None)
    assert not h.ok and h.reason == W.NO_FRAME
    hs = _run(wd, _healthy(10, seed=12), t0=t + DT)
    assert not hs[0].ok and hs[-1].ok


def test_bank_of_watchdogs():
    bank = W.ImageWatchdog(3)
    rng = np.random.default_rng(13)
    base = _scene()
    ok = None
    for i in range(30):
        f = _frame(base, rng)
        ok = bank.step(i * DT, [f, f.copy() + np.uint8(i % 2), np.zeros((H, WID), np.uint8) if i >= 20 else f])
    assert ok is not None and ok.dtype == bool and ok.tolist() == [True, True, False]
    st = bank.status[2]
    assert st is not None and st.reason == W.BLACK
    with pytest.raises(ValueError):
        bank.step(99.0, [None, None])


# ------------------------------------------------------------------------------------------------ map side

def _obs(t, xs=(), zs=(), cls=M.PERSON, cam_ok=None, floor_ok=None, machine=-1, moving=False):
    n = len(xs)
    o = M.Observation(t + 0.2, t, np.full(n, cls, np.int8), np.array(xs, float), np.array(zs, float), np.full(n, 0.4),
                      np.full(n, np.nan), np.full(n, moving, bool), np.full(n, machine, np.int16),
                      np.ones(NCAM, bool) if cam_ok is None else np.asarray(cam_ok, bool))
    if floor_ok is not None:
        M_ = len(SCN.machines)
        o = replace(o, zone_motion=np.zeros(M_, np.float32), zone_vis=np.ones(M_, np.float32),
                      cam_floor_ok=np.asarray(floor_ok, bool) & o.cam_ok)
    return o


def _cond(m: M.LayeredMap, t: float, tod: int | None = None) -> M.Cond:
    Mn = m.M
    return M.Cond(t, M.tod_bin(t) if tod is None else tod, M.RUNNING, np.zeros(Mn, np.int16), np.zeros(Mn),
                  np.full(Mn, np.nan), np.full(Mn, np.nan), np.full(Mn, np.nan))


def _known_map() -> M.LayeredMap:
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    return m


def _cell_xy(m: M.LayeredMap, cam: int) -> tuple[float, float]:
    ix, iz = np.argwhere(m.cam_of_cell == cam)[len(np.argwhere(m.cam_of_cell == cam)) // 2]
    s = m.spec["site"]
    return s["x0"] + (ix + 0.5) * s["res"], s["z0"] + (iz + 0.5) * s["res"]


def test_unhealthy_camera_cells_are_unknown_not_free_and_lifted_on_recovery():
    m = _known_map()
    cam = 4
    x, z = _cell_xy(m, cam)
    t = 10 * 3600.0
    S = M.tod_bin(t) * M.N_PROD + M.RUNNING
    hits0, vis0 = m.hits[M.PERSON][S].sum(), m.visits_cam[S].copy()
    bad = np.ones(NCAM, bool)
    bad[cam] = False
    o = _obs(t, [x], [z], floor_ok=bad)
    m.update(o, t, _cond(m, t))
    U = m.unknown_mask()
    assert (U == (m.cam_of_cell == cam)).all(), "exactly the unhealthy camera's cells are UNKNOWN"
    assert (m.unknown_mask(M.tod_bin(t), M.RUNNING) == U).all()
    assert m.hits[M.PERSON][S].sum() == hits0, "a stale / frozen image's detections are not evidence"
    assert m.visits_cam[S][cam] == vis0[cam] and (np.delete(m.visits_cam[S] - vis0, cam) == 1).all(), \
        "an unhealthy camera gets no visit (its cells are not pushed towards p = 0)"
    # the robot gets it as occupied boxes
    steps = m.predict_obstacles(t, 0.0, {}, M.RUNNING, with_traffic=False)
    unk = steps[0].boxes[[b == M.UNKNOWN_ID for b in steps[0].box_machine]]
    assert len(unk) and any(x0 <= x <= x1 and z0 <= z <= z1 for x0, z0, x1, z1 in unk)
    # health returns -> lifted
    m.update(_obs(t + 0.5, floor_ok=np.ones(NCAM, bool)), t + 0.5, _cond(m, t + 0.5))
    assert not m.unknown_mask().any()


def test_heartbeat_blackout_is_unknown_too_on_v1():
    m = _known_map()
    ok = np.ones(NCAM, bool)
    ok[2] = False
    m.update(_obs(100.0, cam_ok=ok), 100.0, _cond(m, 100.0))
    assert (m.unknown_mask() == (m.cam_of_cell == 2)).all()
    m.update(_obs(100.5), 100.5, _cond(m, 100.5))
    assert not m.unknown_mask().any()


def test_camera_with_no_visits_in_a_slice_is_unknown_not_zero():
    m = M.LayeredMap(SCN)
    m.l1_visits[:] = 100
    tod = 20
    S = tod * M.N_PROD + M.RUNNING
    m.visits_cam[S] = 1000
    m.visits_cam[S, 7] = 0                                   # camera 7 never healthy during this slice
    m.visits_cam[(tod + 1) * M.N_PROD + M.RUNNING] = 1000    # another slice, all cameras seen (static part stays known)
    p, _ = m.traffic("person", tod, M.RUNNING)
    assert (p[m.cam_of_cell == 7] == 0).all()               # counts say 0 ...
    U = m.unknown_mask(tod, M.RUNNING)
    assert (U == (m.cam_of_cell == 7)).all(), "... but the cells are UNKNOWN in that slice, not free"
    assert not m.unknown_mask(tod + 1, M.RUNNING).any(), "only in the slice where the camera had no visits"
    m.visits_cam[S, 7] = m.min_ticks - 1                     # under-observed (below the slice trust rule) is UNKNOWN too
    assert (m.unknown_mask(tod, M.RUNNING) == (m.cam_of_cell == 7)).all()


def test_condition_tracker_does_not_read_a_frozen_camera_as_a_stopped_machine():
    tr = M.ConditionTracker(SCN, use_learner=False)
    mi = 0
    cam = int(tr.cam_of_machine[mi])
    x0, z0, x1, z1 = SCN.machines[mi].rect
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    t = 0.0
    for _ in range(10):                                      # machine running
        tr.step(_obs(t, [xc], [zc], cls=M.PART, machine=mi, moving=True, floor_ok=np.ones(NCAM, bool)))
        t += DT
    assert tr.cur[mi] == 1
    bad = np.ones(NCAM, bool)
    bad[cam] = False
    for _ in range(20):                                      # camera frozen: the image shows no motion
        tr.step(_obs(t, floor_ok=bad))
        t += DT
    assert tr.cur[mi] == 1, "no motion seen through a frozen camera is MISSING, not a stop"


def test_converter_applies_the_watchdog_to_zone_motion():
    rects = [m_.rect for m_ in SCN.machines]
    share = M.zone_camera_share(rects, SCN.cameras)
    conv = M.ObsV2Converter(share, rects)
    o = _obs(50.0)
    v = conv(o)
    zm, zv, cf = v.zone_motion, v.zone_vis, v.cam_floor_ok
    assert zm is not None and zv is not None and cf is not None and np.isfinite(zm).all() and cf.all()
    cam = int(np.argmax(share[0]))
    bad = np.ones(NCAM, bool)
    bad[cam] = False
    v = conv(_obs(50.5), floor_ok=bad)
    M.check_obs_v2(v, len(rects), NCAM)
    zm, zv, cf = v.zone_motion, v.zone_vis, v.cam_floor_ok
    assert zm is not None and zv is not None and cf is not None
    hit = share[:, cam] > 0
    assert hit.any() and np.isnan(zm[hit]).all(), "a zone seen by an unhealthy camera is MISSING"
    assert np.allclose(zv[hit], np.clip(1.0 - share[hit, cam], 0, 1), atol=1e-6)
    assert np.isfinite(zm[~hit]).all() and not cf[cam] and cf[np.arange(NCAM) != cam].all()
    # a v2 observation from a camera model is masked the same way
    v2 = conv(_obs(51.0, floor_ok=np.ones(NCAM, bool)), floor_ok=bad)
    assert v2.zone_motion is not None and np.isnan(v2.zone_motion[hit]).all()
    with pytest.raises(ValueError):
        conv(_obs(51.5), floor_ok=np.ones(NCAM - 1, bool))


def test_end_to_end_frozen_camera_to_unknown_and_back():
    """Frames -> watchdog -> converter -> map: a frozen camera's cells become UNKNOWN and are lifted after recovery."""
    rects = [m_.rect for m_ in SCN.machines]
    conv = M.ObsV2Converter(M.zone_camera_share(rects, SCN.cameras), rects)
    m = _known_map()
    bank = W.ImageWatchdog(NCAM)
    rng = np.random.default_rng(21)
    base = _scene()
    cam = 5
    frozen = _frame(base, rng)
    states = []
    t = 0.0
    for k in range(80):
        frames = [_frame(base, rng) for _ in range(NCAM)]
        if 30 <= k < 50:
            frames[cam] = frozen
        ok = bank.step(t, frames)
        v = conv(_obs(t), floor_ok=ok)
        m.update(v, t, _cond(m, t))
        states.append(bool(m.unknown_mask()[m.cam_of_cell == cam].all()) if k >= 20 else None)
        t += DT
    p = W.WatchdogParams()
    k_frz = 30 + int(round(p.frozen_s / DT))
    assert states[25] is False and all(states[k_frz:50]), "UNKNOWN while frozen"
    k_back = 50 + int(round(p.recover_s / DT))
    assert all(states[50:k_back]) and not any(states[k_back:]), "lifted after recover_s of healthy frames"


def test_slice_unknown_reaches_robot_through_the_codec():
    """The robot's decoded map marks the under-observed camera's cells UNKNOWN in that slice, as the observer does."""
    m = M.LayeredMap(SCN)
    m.l1_visits[:] = 100
    m.visits_cam[:] = 1000
    tod = 20
    m.visits_cam[tod * M.N_PROD + M.RUNNING, 7] = 0
    payload, sizes = ML.encode_payload(m)
    assert payload[4] == ML.PAYLOAD_VERSION == 3 and sizes["SLICE_UNKNOWN"] > 1
    dm = ML.decode_payload(payload)
    for tb in (tod, tod + 1):
        assert np.array_equal(dm.unknown_mask(tb, M.RUNNING), m.unknown_mask(tb, M.RUNNING))
    assert (dm.unknown_mask(tod, M.RUNNING) == (m.cam_of_cell == 7)).all()
    # a v2 payload (no SLICE_UNKNOWN section) still decodes; the view then cannot tell (None), as before the patch
    v2 = payload[:4] + bytes([2]) + payload[5:len(payload) - sizes["SLICE_UNKNOWN"]]
    dm2 = ML.decode_payload(v2)
    assert dm2._slice_unknown(tod, M.RUNNING) is None and np.array_equal(dm2.unknown_mask(), dm.unknown_mask())


def test_slice_unknown_reaches_the_robot_keepout_mask():
    """Robot side (ros2/intercycle_nav masks.keepout_mask applies view.unknown_mask(tod, prod)): the cells of a camera
    with no visits in the current slice are keep-out on the robot's costmap, from the decoded payload alone."""
    import sys
    from pathlib import Path

    nav = Path(__file__).resolve().parents[1] / "ros2" / "intercycle_nav"
    if str(nav) not in sys.path:
        sys.path.insert(0, str(nav))
    from intercycle_nav import masks as K  # pyright: ignore[reportMissingImports]  (ros2 package dir, added above)

    m = M.LayeredMap(SCN)
    m.l1_visits[:] = 100
    m.visits_cam[:] = 1000
    tod, cam = 20, 7
    m.visits_cam[tod * M.N_PROD + M.RUNNING, cam] = 0
    dm = ML.decode_payload(ML.encode_payload(m)[0])
    site = K.Site.from_scenario()
    grid = K.Grid(-2.0, -2.0, 0.25, 496, 208)
    p = K.MaskParams()
    s = m.spec["site"]
    ij = np.argwhere(m.cam_of_cell == cam)
    cx = s["x0"] + (ij[:, 0] + 0.5) * s["res"]
    cz = s["z0"] + (ij[:, 1] + 0.5) * s["res"]
    gi = np.floor((cx - grid.x0) / grid.res).astype(int)
    gj = np.floor((cz - grid.y0) / grid.res).astype(int)
    inside = (gi >= 0) & (gi < grid.nx) & (gj >= 0) & (gj < grid.ny)
    assert inside.sum() > 10
    for tb, want in ((tod, True), (tod + 1, False)):
        snap = K.Snapshot(tb * M.TOD_BIN_S + 100.0, True, 2, True, None, M.RUNNING, [])
        g, info = K.keepout_mask(grid, "learned", site, snap, dm, p)
        unk = g[gi[inside], gj[inside]] == K.KEEP_OUT
        if want:
            assert unk.all(), "every cell of the camera unseen in this slice is keep-out on the robot"
            assert info["unknown_cells_1m"] == int((m.cam_of_cell == cam).sum())
        else:
            assert info["unknown_cells_1m"] == 0, "only in the slice where the camera had no visits"


# ------------------------------------------------------------------------------------------------ audit fixes (S4-fix1)

def _straddle() -> tuple[int, int, int]:
    """(machine index, own camera, a neighbour camera whose cells meet the machine's L1 grid): S10 and camera 4."""
    m = M.LayeredMap(SCN)
    i = [d["id"] for d in m.spec["machines"]].index("S10")
    own = int(m.cam_of_machine[i])
    return i, own, 4


def _l1_cover_obs(m: M.LayeredMap, i: int, t: float, floor_ok) -> M.Observation:
    """Machine i's parts over its whole L1 grid (one 0.5 m disc every 0.5 m), as a v2 observation."""
    res = m.spec["env_res"]
    xs = m.l1_x0[i] + np.arange(0.25, m.l1_nx[i] * res, 0.5)
    zs = m.l1_z0[i] + np.arange(0.25, m.l1_nz[i] * res, 0.5)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    o = _obs(t, X.ravel(), Z.ravel(), cls=M.PART, machine=i, moving=True, floor_ok=floor_ok)
    return replace(o, r=np.full(len(o.x), 0.6))


def test_l1_cameras_of_a_machine_include_its_neighbours():
    m = M.LayeredMap(SCN)
    i, own, nb = _straddle()
    assert m.l1_cams[i, own] and m.l1_cams[i, nb], "S10's L1 grid lies in cameras 0 and 4"
    fok = np.ones(NCAM, bool)
    assert m.l1_cams_ok(fok)[i]
    fok[nb] = False
    assert not m.l1_cams_ok(fok)[i], "a neighbour camera's outage makes the machine's L1 unobserved"
    m.cam_offline = {nb}
    assert not m.l1_cams_ok(np.ones(NCAM, bool))[i], "an offline (bumped) neighbour too"


def test_straddling_machine_envelope_does_not_decay_under_a_neighbour_camera_outage():
    """Audit blocker: while a neighbour camera is unhealthy its detections are dropped, so the machine's L1 must get no
    visit either; otherwise its envelope cells in that camera's region are pushed towards p = 0 and stay low after the
    UNKNOWN is lifted (fail-dangerous)."""
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    i, own, nb = _straddle()
    ok = np.ones(NCAM, bool)
    bad = ok.copy()
    bad[nb] = False
    t = 0.0
    for _ in range(40):                                       # learn: the arm sweeps the whole grid in bin 0
        m.update(_l1_cover_obs(m, i, t, ok), t, _cond(m, t))
        t += DT
    p0 = m.envelope_prob("S10", 0).copy()
    v0 = float(m.l1_visits[i, 0])
    res = m.spec["env_res"]
    s = m.spec["site"]
    gx = m.l1_x0[i] + (np.arange(m.l1_nx[i]) + 0.5) * res
    gz = m.l1_z0[i] + (np.arange(m.l1_nz[i]) + 0.5) * res
    six = np.clip(np.floor((gx - s["x0"]) / s["res"]).astype(int), 0, m.nx - 1)
    siz = np.clip(np.floor((gz - s["z0"]) / s["res"]).astype(int), 0, m.nz - 1)
    in_nb = m.cam_of_cell[six][:, siz] == nb
    assert in_nb.any() and (p0[in_nb] > 0.99).all()
    for _ in range(200):                                      # neighbour camera frozen for 100 s, the arm keeps moving
        m.update(_l1_cover_obs(m, i, t, bad), t, _cond(m, t))
        t += DT
    assert float(m.l1_visits[i, 0]) == v0, "no L1 visit while a camera covering the machine's grid is unhealthy"
    m.update(_l1_cover_obs(m, i, t, ok), t, _cond(m, t))        # recovery: UNKNOWN lifted
    assert not m.cam_unhealthy
    p1 = m.envelope_prob("S10", 0)
    assert (p1[in_nb] > 0.99).all(), f"envelope in the neighbour's cells decayed to {p1[in_nb].min():.3f}"
    assert (p1[~in_nb] > 0.99).all()


def _drive_tracker(tr: M.ConditionTracker, t: float, n: int, mi: int, P: float, line_P: float, floor_ok,
                   moving_parts: bool = True) -> tuple[float, M.Cond]:
    """n ticks: machine mi's part moves for the first 25 % of each period P, a skid moves for the first 20 % of each
    line period."""
    x0, z0, x1, z1 = SCN.machines[mi].rect
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    sx, sz = SCN.machines[0].x, SCN.machines[0].z
    c = None
    for _ in range(n):
        xs, zs, cl, mach, mv = [], [], [], [], []
        if moving_parts:
            xs += [xc, sx]
            zs += [zc, sz]
            cl += [M.PART, M.SKID]
            mach += [mi, 0]
            mv += [(t % P) < 0.25 * P, (t % line_P) < 0.2 * line_P]
        o = M.Observation(t + 0.2, t, np.array(cl, np.int8), np.array(xs, float), np.array(zs, float),
                          np.full(len(xs), 0.4), np.full(len(xs), np.nan), np.array(mv, bool), np.array(mach, np.int16),
                          np.ones(NCAM, bool))
        Mn = len(SCN.machines)
        o = replace(o, zone_motion=np.zeros(Mn, np.float32), zone_vis=np.ones(Mn, np.float32),
                    cam_floor_ok=np.asarray(floor_ok, bool))
        c = tr.step(o)
        t += DT
    assert c is not None
    return t, c


def test_cond_gives_unknown_not_stop_behind_an_unhealthy_camera():
    """Audit minor: an off-line machine behind a frozen camera must not go overdue (STOP_BIN) in Cond, and the stations
    behind frozen cameras must not vote the line stopped (ANDON)."""
    tr = M.ConditionTracker(SCN, use_learner=False)
    mi = [m_.id for m_ in SCN.machines].index("HEM1")
    P, LP = 45.0, 30.0
    ok = np.ones(NCAM, bool)
    t, c = _drive_tracker(tr, 0.0, int(8 * P / DT), mi, P, LP, ok)
    assert abs(tr.period[mi] - P) < 1.0 and abs(tr.period[tr.L] - LP) < 1.0
    assert 0 <= c.phase_bin[mi] < M.N_PHASE and not c.line_stopped
    bad = ok.copy()
    bad[[0, 1, 2, 3, int(tr.cam_of_machine[mi])]] = False     # every line station's camera and HEM1's camera frozen
    t, c = _drive_tracker(tr, t, int(2.5 * P / DT), mi, P, LP, bad, moving_parts=False)
    assert c.phase_bin[mi] == -1 and np.isnan(c.phase[mi]), f"HEM1 bin {c.phase_bin[mi]}: unknown, never STOP_BIN"
    assert (c.phase_bin[tr.on_line] == -1).all()
    assert not c.line_stopped and c.prod_state == M.RUNNING, "stations nobody can see do not vote the line stopped"
    t, c = _drive_tracker(tr, t, int(2 * P / DT), mi, P, LP, ok)    # health back: known again
    assert 0 <= c.phase_bin[mi] < M.N_PHASE


def test_mapbuilder_discovered_bins_are_unknown_behind_an_unhealthy_camera():
    class _PM:                                                 # a phase model that always claims sub-phase 2
        def update(self, obs):
            pass

        def phase_bin(self, mid, t):
            return 2

        def n_bins(self, mid):
            return 6

        def line_state(self, t):
            return "running"

    mb = M.MapBuilder(SCN, phase_model=_PM(), use_bins=True)
    mi = [m_.id for m_ in SCN.machines].index("HEM1")
    bad = np.ones(NCAM, bool)
    bad[int(mb.tracker.cam_of_machine[mi])] = False
    c = mb.condition(_obs(10.0, floor_ok=bad))
    assert c.phase_bin[mi] == -1
    assert (c.phase_bin[np.arange(len(c.phase_bin)) != mi] == 2).sum() >= 1


def test_change_detector_cusum_ignores_a_frozen_camera():
    from intercycle.observer import change as CH

    m = _known_map()
    det = CH.ChangeDetector(m)
    st = det.static.reshape(-1)
    cam = 4
    c_bad = int(np.flatnonzero(st & (det.cam_of_cell == cam))[0])
    c_ok = int(np.flatnonzero(st & (det.cam_of_cell == 0))[0])
    det.S[[c_bad, c_ok]] = 5.0
    bad = np.ones(NCAM, bool)
    bad[cam] = False
    det.step(_obs(10.0, floor_ok=bad))
    assert det.S[c_bad] == 5.0, "a frozen camera's 'no motion' is not evidence for the CUSUM"
    assert det.S[c_ok] < 5.0


def test_change_detector_spawned_machine_learns_only_when_all_its_cameras_are_healthy():
    from intercycle.observer import change as CH

    m = _known_map()
    det = CH.ChangeDetector(m)
    s = m.spec["site"]
    # a free area across the camera 4 / camera 5 border (rect as the detector stores it: evidence box + margin)
    xb = zb = None
    for ix in range(m.nx - 1):
        a, b = m.cam_of_cell[ix, :], m.cam_of_cell[ix + 1, :]
        js = np.flatnonzero(((a == 4) & (b == 5)) & det.static[ix, :] & det.static[ix + 1, :])
        if len(js):
            xb, zb = s["x0"] + (ix + 1) * s["res"], s["z0"] + (js[len(js) // 2] + 0.5) * s["res"]
            break
    assert xb is not None and zb is not None
    cd = CH.Candidate(1, 0.0, rect=(xb - 3.0, zb - 2.0, xb + 3.0, zb + 2.0))
    ev: list = []
    det._spawn(cd, 0.0, 20.0, ev)
    i = cd.machine_index
    cd.last_rise = 0.0
    assert m.l1_cams[i, 4] and m.l1_cams[i, 5]
    own = int(m.cam_of_machine[i])
    other = 9 - own                                           # 4 <-> 5
    ok = np.ones(NCAM, bool)
    for cam_down in (own, other):
        bad = ok.copy()
        bad[cam_down] = False
        v = m.l1_visits[i].sum()
        det._learn(cd, _obs(1.0, [xb], [zb], cls=M.PART, moving=True, floor_ok=bad), 1.0, ev)
        assert m.l1_visits[i].sum() == v, f"camera {cam_down} unhealthy: no L1 visit for the spawned machine"
    v = m.l1_visits[i].sum()
    det._learn(cd, _obs(1.0, [xb], [zb], cls=M.PART, moving=True, floor_ok=ok), 1.0, ev)
    assert m.l1_visits[i].sum() == v + 1


def test_phase_features_are_missing_behind_an_unhealthy_camera():
    from intercycle.observer import phases as P

    lay = P.Layout.from_scenario(SCN)
    fe = P.Features(lay)
    mi = lay.mids.index("HEM1")
    bad = np.ones(NCAM, bool)
    bad[int(lay.cam_of_machine[mi])] = False
    x0, z0, x1, z1 = SCN.machines[mi].rect
    o = _obs(5.0, [(x0 + x1) / 2], [(z0 + z1) / 2], cls=M.PART, machine=mi, moving=False, floor_ok=bad)
    _, X, _ = fe.extract(o)
    assert np.isnan(X[mi]).all(), "a frozen camera's zero motion is MISSING, not data"
    assert np.isfinite(X[np.arange(len(X)) != mi]).all()
