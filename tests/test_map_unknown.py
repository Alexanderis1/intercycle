"""UNKNOWN cell state, conservative box covers, inflation margin, codec v2 and change detection (fast; no floor sim)."""
from __future__ import annotations

import zlib

import numpy as np

from intercycle.codec import maplayer as ML
from intercycle.codec.sign import demo_key
from intercycle.observer import mapping as M
from intercycle.observer.change import ChangeDetector
from intercycle.scenario import load

SCN = load()
NCAM = len(SCN.cameras)


def _obs(t, xs=(), zs=(), rs=(), cls=M.PART, moving=True, machine=-1, cam_ok=None):
    n = len(xs)
    return M.Observation(t + 0.2, t, np.full(n, cls, np.int8), np.array(xs, float), np.array(zs, float), np.array(rs, float),
                         np.full(n, np.nan), np.full(n, moving, bool), np.full(n, machine, np.int16),
                         np.ones(NCAM, bool) if cam_ok is None else cam_ok)


def test_safe_box_cover_contains_every_cell():
    rng = np.random.default_rng(0)
    for k in range(30):
        g = rng.random((40, 36)) < 0.08
        g |= M.mask_from_circles(rng.uniform(0, 9, (4, 3)) + [0, 0, 0.5], 0, 0, 40, 36, 0.25)
        for mb in (4, 12):
            boxes = M.safe_box_cover(g, mb)
            cov = np.zeros_like(g)
            for i0, j0, i1, j1 in boxes:
                cov[i0:i1, j0:j1] = True
            assert len(boxes) <= mb
            assert (cov | ~g).all(), "a mask cell is outside every box (would be broadcast as free)"
    assert M.safe_box_cover(np.zeros((5, 5), bool)).shape == (0, 4)


def test_dilate_and_margin():
    g = np.zeros((11, 11), bool)
    g[5, 5] = True
    d = M.dilate(g, 2)
    assert d[5, 3] and d[3, 5] and d[7, 5] and not d[5, 2] and d.sum() >= 13
    assert np.array_equal(M.dilate(g, 0), g)
    assert abs(M.iso13855_margin(2.0, 0.2, 0.3) - 0.7) < 1e-12


def test_unknown_state_of_an_empty_and_partial_map():
    m = M.LayeredMap(SCN)
    U = m.unknown_mask()
    assert U.all(), "a map that has seen nothing is all UNKNOWN"
    # unseen machine envelope = the whole zone grid (conservative), never empty
    assert m.envelope("S30", 3).all()
    assert not m.l1_known("S30")
    # a map that saw every camera: only explicit unknowns remain
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    m.l1_hits[:] = 0
    assert not m.unknown_mask().any()
    m.cam_offline = {6}
    assert (m.unknown_mask() == (m.cam_of_cell == 6)).all()
    m.cam_offline = set()
    m.unknown_rects = [(70.0, 38.0, 75.0, 44.0)]
    U = m.unknown_mask()
    s = m.spec["site"]
    assert U[71 - int(s["x0"]), 40 - int(s["z0"])] and U.sum() == 5 * 6
    # unresolvable traffic slice -> everything UNKNOWN
    m2 = M.LayeredMap(SCN)
    m2.visits_cam[:] = 0
    m2.l1_visits[:] = 100
    m2.visits_cam[20 * M.N_PROD + M.RUNNING] = 1000
    assert not m2.unknown_mask(20, M.RUNNING).any()
    assert m2.unknown_mask(20, M.BREAK).sum() == 0       # falls back to the same tod while running
    assert m2.unknown_mask(5, M.ANDON).all()


def test_unknown_reaches_robot_through_the_codec():
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    m.unknown_rects = [(70.0, 38.0, 75.0, 44.0)]
    m.cam_offline = {9}
    payload, sizes = ML.encode_payload(m)
    assert payload[4] == ML.PAYLOAD_VERSION == 3 and sizes["UNKNOWN"] > 0
    dm = ML.decode_payload(payload)
    assert np.array_equal(dm.unknown_mask(), m.unknown_mask())
    steps = dm.predict_obstacles(10 * 3600.0, 1.0, {}, M.RUNNING, step_s=1.0, with_traffic=False)
    unk = steps[0].boxes[[o == M.UNKNOWN_ID for o in steps[0].box_machine]]
    assert len(unk)
    cov = np.zeros_like(m.unknown_mask())
    s = m.spec["site"]
    for x0, z0, x1, z1 in unk:
        cov[int(x0 - s["x0"]):int(x1 - s["x0"]), int(z0 - s["z0"]):int(z1 - s["z0"])] = True
    assert (cov | ~m.unknown_mask()).all(), "every UNKNOWN cell is inside an UNKNOWN box the robot treats as occupied"
    # a v1 payload (no UNKNOWN section) still decodes, with no unknown cells
    v1 = payload[:4] + bytes([1]) + payload[5:len(payload) - sizes["UNKNOWN"] - sizes["SLICE_UNKNOWN"]]
    assert not ML.decode_payload(v1).unknown_mask().any()
    # signed bundle round trip carries it too
    b = ML.pack_map(m, demo_key(SCN.key_id), version=2, issue_time=1.8e9)
    raw = zlib.decompress(b"".join(b.chunks[:b.manifest.n_chunks])[:b.manifest.payload_len])
    assert np.array_equal(ML.decode_payload(raw).unknown_mask(), m.unknown_mask())


def test_inflate_margin_grows_machine_boxes():
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    i = m._mi("HEM1")
    m._l1(i)[:, 10:20, 10:20] = 100.0
    m.period[i] = 45.0
    m.jitter[i] = 0.01
    a = m.predict_obstacles(1000.0, 0.0, {"HEM1": 0.3}, M.RUNNING, with_traffic=False)[0]
    b = m.predict_obstacles(1000.0, 0.0, {"HEM1": 0.3}, M.RUNNING, with_traffic=False, inflate_m=0.5)[0]
    ha = a.boxes[[o == "HEM1" for o in a.box_machine]]
    hb = b.boxes[[o == "HEM1" for o in b.box_machine]]
    assert np.allclose(hb - ha, [-0.5, -0.5, 0.5, 0.5])


def test_add_machine_then_update_and_codec():
    m = M.LayeredMap(SCN)
    k = m.add_machine("NEW1", (68.0, 37.0, 76.0, 45.0))
    assert k == len(SCN.machines) and m.machine_ids()[-1] == "NEW1"
    m.l1_view(k)[3, 5:9, 5:9] = 20.0
    m.l1_visits[k, :] = 20.0
    m.period[k] = 50.0
    assert m.envelope("NEW1", 3).sum() == 16
    tr = M.ConditionTracker(SCN, use_learner=False)
    o = _obs(100.0, [30.0, 72.0], [24.0, 41.0], [0.5, 0.5])
    m.update(o, o.t_cap, tr.step(o))                     # tracker knows 10 machines, map 11: must not crash
    dm = ML.decode_payload(ML.encode_payload(m)[0])
    assert "NEW1" in dm.machine_ids() and len(dm.boxes_for("NEW1", (3,)))


def _run_detector(det, t0, n_ticks, gen):
    evs = []
    for k in range(n_ticks):
        t = t0 + k * 0.5
        evs += det.step(gen(t))
    return evs


def _known_map():
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    return m


def test_change_detector_finds_a_periodic_machine_and_spawns_it():
    m = _known_map()
    det = ChangeDetector(m)
    P, busy = 40.0, 0.6
    rng = np.random.default_rng(1)

    def gen(t):
        u = (t % P) / P
        if u < busy:
            a = 2 * np.pi * u / busy
            return _obs(t, [72 + 2 * np.cos(a), 72 + np.cos(a)], [41 + 2 * np.sin(a), 41 + np.sin(a)], [0.5, 0.5])
        return _obs(t)
    evs = _run_detector(det, 30000.0, int(20 * 60 * 2), gen)
    kinds = [e["kind"] for e in evs]
    assert "change_detected" in kinds and "machine_spawned" in kinds, kinds
    det_t = next(e["t"] for e in evs if e["kind"] == "change_detected")
    assert det_t - 30000.0 < 120
    sp = next(e for e in evs if e["kind"] == "machine_spawned")
    assert abs(sp["period_s"] - P) < 1.0
    assert m.unknown_rects or "machine_converged" in kinds
    assert "machine_converged" in kinds
    assert not m.unknown_rects, "UNKNOWN is lifted once the spawned envelope has converged"
    assert m.l1_known(sp["machine"])


def test_change_detector_ignores_transient_clutter_and_flags_aperiodic_motion_without_spawning():
    m = _known_map()
    det = ChangeDetector(m)
    rng = np.random.default_rng(2)
    blobs = []

    def gen(t):
        nonlocal blobs
        if rng.random() < 60 / 3600 * 0.5:
            blobs.append((rng.uniform(0, 120), rng.uniform(0, 48), t + rng.uniform(0.5, 3.0)))
        blobs = [b for b in blobs if b[2] > t]
        return _obs(t, [b[0] for b in blobs], [b[1] for b in blobs], [0.5] * len(blobs))
    evs = _run_detector(det, 40000.0, 3600 * 2, gen)
    assert not [e for e in evs if e["kind"] == "change_detected"], "transient clutter must not raise a change"
    # an aperiodic persistent flicker: detected (UNKNOWN, conservative) but never spawned
    det2 = ChangeDetector(_known_map())

    def flick(t):
        if 40000 <= t < 41800 and rng.random() < 0.3:
            return _obs(t, [88 + rng.normal(0, 0.3)], [44 + rng.normal(0, 0.3)], [0.5])
        return _obs(t)
    evs = _run_detector(det2, 40000.0, 3600 * 2, flick)
    kinds = [e["kind"] for e in evs]
    assert "change_detected" in kinds and "machine_spawned" not in kinds and "change_cleared" in kinds


def test_displaced_image_of_a_known_machine_is_not_a_new_machine():
    """Motion just outside SEAL1's zone that is phase-locked to SEAL1's own cycle (what a bumped camera produces)."""
    m = _known_map()
    det = ChangeDetector(m)
    tr = M.ConditionTracker(SCN, use_learner=False)
    si = [x.id for x in SCN.machines].index("SEAL1")
    x0, z0, x1, z1 = SCN.machine("SEAL1").rect
    P = 60.0
    evs = []
    for k in range(int(25 * 60 * 2)):
        t = 50000.0 + k * 0.5
        u = (t % P) / P
        mv = u < 0.6
        xs, zs, ms = [46.0], [41.0], [si]
        if mv:
            xs.append(x1 + 1.6 + 0.8 * np.sin(2 * np.pi * u))   # its displaced copy, outside zone + margin
            zs.append(41.0)
            ms.append(-1)
        o = M.Observation(t + 0.2, t, np.full(len(xs), M.PART, np.int8), np.array(xs), np.array(zs), np.full(len(xs), 0.5),
                          np.full(len(xs), np.nan), np.full(len(xs), mv), np.array(ms, np.int16), np.ones(NCAM, bool))
        evs += det.step(o, tr.step(o))
    kinds = [e["kind"] for e in evs]
    assert "change_detected" in kinds
    assert "machine_spawned" not in kinds and "change_explained" in kinds, kinds


def test_spawn_waits_for_steady_cycles_and_survives_dwell_pauses():
    """Irregular commissioning cycles first, then a 50 s cycle whose active part has two 3.5 s dwells (motion pauses)."""
    m = _known_map()
    det = ChangeDetector(m)
    rng = np.random.default_rng(5)
    t0 = 60000.0
    starts, t = [], t0
    while t < t0 + 1800:                                  # commissioning: random active length and gaps
        starts.append((t, 35 * rng.uniform(0.6, 1.4)))
        t += starts[-1][1] + rng.uniform(0, 90)
    while t < t0 + 4800:
        starts.append((t, 35.0))
        t += 50.0 * (1 + 0.01 * rng.standard_normal())
    S = np.array([s for s, _ in starts])
    A = np.array([a for _, a in starts])

    def gen(t):
        i = int(np.searchsorted(S, t, side="right")) - 1
        f = (t - S[i]) / A[i] if i >= 0 else 2.0
        if f >= 1.0 or 0.25 <= f < 0.35 or 0.70 <= f < 0.80:      # idle or dwelling: nothing moves
            return _obs(t)
        a = 2 * np.pi * f
        return _obs(t, [72 + 2 * np.cos(a)], [41 + 2 * np.sin(a)], [0.5])
    evs = _run_detector(det, t0, int(4800 * 2), gen)
    sp = [e for e in evs if e["kind"] == "machine_spawned"]
    assert len(sp) == 1, [e["kind"] for e in evs]
    assert sp[0]["t"] > t0 + 1800, "no model while the cycles are still irregular"
    assert abs(sp[0]["period_s"] - 50.0) < 1.5
    cd = next(c for c in det.cands if c.spawned)
    # phase locked to the true starts: the spawned model's last start is a true cycle start
    assert np.min(np.abs(S - cd.last_rise)) < 2.0


def test_motion_flag_is_per_body_group():
    """A body group that appears (S80 carriage) is moving; the other groups keep their own frame differencing."""
    sh = M.CameraShim(SCN, seed=1, p_drop=0.0, sigma_m=0.0, blackouts_per_h=0)
    sh.observe(M.SimState(0.0, parts={"S10": np.array([[12.0, 24.0, 0.5]]), "S80": np.zeros((0, 3))}))
    o = sh.observe(M.SimState(0.5, parts={"S10": np.array([[12.0, 24.0, 0.5]]), "S80": np.array([[78.0, 24.0, 0.5]] * 2)}))
    assert o.moving.tolist() == [False, True, True]
