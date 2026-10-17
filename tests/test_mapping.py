"""Layered map of dynamics: camera shim, conditioning, learned layers, map codec (SYNTHETIC floor, truth used only for scoring).

The module-scoped fixture simulates ~4 h of floor at 2 Hz in four windows (two breaks, a shift change, ordinary running
and a traced window with andon stops), builds the map from camera observations only, and the tests score it against
the kinematics ground truth.
"""
from __future__ import annotations

import ast
import math
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_map as B  # noqa: E402  (floor-simulation harness; truth is allowed in scripts and tests)

from intercycle.codec import ble  # noqa: E402
from intercycle.codec import maplayer as ML  # noqa: E402
from intercycle.codec.sign import demo_key, keyring_from, sign  # noqa: E402
from intercycle.observer import mapping as M  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.types import DAY0_UNIX  # noqa: E402

SCN = load()
H = 3600.0
WINDOWS = [(0, 8.33 * H, 9.75 * H),      # break 09:00-09:15
           (0, 11.25 * H, 12.25 * H),    # break 11:30-12:00
           (0, 13.6 * H, 14.4 * H)]      # shift change 14:00-14:05
TRACE = (0, 16.0 * H, 17.33 * H)         # traced: andon stops at 16:04, 16:42, 16:50 and the break 17:00-17:15
KEY = demo_key(SCN.key_id)
KEYRING = keyring_from({SCN.key_id: KEY})
SITE = SCN.site_id
MIDS = [m.id for m in SCN.machines]


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("maps")
    with ProcessPoolExecutor(3) as ex:
        futs = [ex.submit(B._worker, (s, a, b, str(tmp / f"w{i}.npz"))) for i, (s, a, b) in enumerate(WINDOWS)]
        m, _, info = B.run_window(*TRACE, record=True)
        for f in futs:
            f.result()
    for i in range(len(WINDOWS)):
        m.merge(M.LayeredMap.load(tmp / f"w{i}.npz"))
    return SimpleNamespace(m=m, info=info, kin=info["kin"], world=info["world"])


@pytest.fixture(scope="module")
def bundle(sim):
    return ML.pack_map(sim.m, KEY, version=7, parity_group=8)


@pytest.fixture(scope="module")
def truth_union(sim):
    """True swept cells per machine (union over 5 minutes of fine-sampled kinematics, skids assigned by zone)."""
    return truth_cells(sim)


def truth_cells(sim, t_lo: float = 600.0, t_hi: float = 900.0, dt: float = 0.1, binned: bool = False, mids=None):
    sp = sim.m.spec
    shim = M.CameraShim(SCN)
    t0 = sim.info["t0"]
    mids = mids or MIDS
    res = sp["env_res"]
    out = {mid: np.zeros((d["nx"], d["nz"]), bool) for d in sp["machines"] for mid in [d["id"]] if mid in mids}
    per_bin = {mid: np.zeros((M.N_PHASE, d["nx"], d["nz"]), bool) for d in sp["machines"] for mid in [d["id"]] if mid in mids}
    for t in np.arange(t0 + t_lo, t0 + t_hi, dt):
        bd = sim.kin.bodies(t)
        sk = bd["SKIDS"]
        lab = shim.machine_of(sk[:, 0], sk[:, 1]) if len(sk) else np.zeros(0, int)
        for k, d in enumerate(sp["machines"]):
            if d["id"] not in out:
                continue
            c = bd[d["id"]]
            if len(sk) and (lab == k).any():
                c = np.vstack([c, sk[lab == k]]) if len(c) else sk[lab == k]
            if not len(c):
                continue
            g = M.mask_from_circles(c, d["x0"], d["z0"], d["nx"], d["nz"], res)
            out[d["id"]] |= g
            if binned:
                ph = sim.kin.cycle_phase("S10" if SCN.machine(d["id"]).on_line else d["id"], t)
                per_bin[d["id"]][min(int(ph * M.N_PHASE), M.N_PHASE - 1)] |= g
    return (out, per_bin) if binned else out


# ------------------------------------------------------------------------------------------------ unit pieces

def test_box_cover_is_exact_and_greedy():
    g = M.mask_from_circles(np.array([[5.0, 5.0, 2.2], [8.0, 6.0, 1.5]]), 0, 0, 48, 40, 0.25)
    for mb in (6, 12, 24):
        boxes = M.box_cover(g, mb, 0.999)
        cov = np.zeros_like(g)
        for i0, j0, i1, j1 in boxes:
            assert g[i0:i1, j0:j1].all(), "a box contains a cell that is not in the mask (false area)"
            cov[i0:i1, j0:j1] = True
        assert len(boxes) <= mb
        if mb >= 12:
            assert cov.sum() == g.sum()
    assert M.box_cover(np.zeros((6, 6), bool)).shape == (0, 4)
    assert M.box_cover(np.ones((6, 6), bool)).tolist() == [[0, 0, 6, 6]]


def test_raster_matches_brute_force():
    rng = np.random.default_rng(0)
    cx, cz, r = rng.uniform(0, 10, 20), rng.uniform(0, 10, 20), rng.uniform(0.3, 2.0, 20)
    for centre in (True, False):
        ci, ix, iz = M.raster(cx, cz, r, 0.0, 0.0, 40, 40, 0.25, centre=centre)
        got = {(int(a), int(b), int(c)) for a, b, c in zip(ci, ix, iz)}
        want = set()
        for k in range(20):
            for i in range(40):
                for j in range(40):
                    if centre:
                        d = math.hypot((i + 0.5) * 0.25 - cx[k], (j + 0.5) * 0.25 - cz[k])
                    else:
                        d = math.hypot(max(i * 0.25 - cx[k], cx[k] - (i + 1) * 0.25, 0), max(j * 0.25 - cz[k], cz[k] - (j + 1) * 0.25, 0))
                    if d <= r[k]:
                        want.add((k, i, j))
        assert got == want


def test_camera_shim_statistics():
    shim = M.CameraShim(SCN, seed=5)
    circ = np.array([[30.0 + 3 * k, 20.0, 0.5] for k in range(10)])
    err, n_seen, n_all, ok_hist = [], 0, 0, []
    for k in range(8000):
        st = M.SimState(1000.0 + k * 0.5, mover_circles=circ, mover_cls=np.zeros(10, np.int8), mover_heading=np.zeros(10),
                        mover_speed=np.ones(10))
        o = shim.observe(st)
        assert abs(o.t - o.t_cap - 0.2) < 1e-9                         # 0.2 s latency
        ok_hist.append(o.cam_ok.copy())
        n_all += 10
        n_seen += len(o)
        if len(o) == 10:
            err.append(np.c_[o.x - circ[:, 0], o.z - circ[:, 1]])
    err = np.concatenate(err)
    assert abs(err.std() - 0.15) < 0.01                                 # floor-projection noise sigma 0.15 m
    ok_hist = np.array(ok_hist)
    black = 1 - ok_hist.mean()
    assert 0.004 < black < 0.025                                        # ~4 blackouts / h / camera, 10 s each (~1.1 %)
    runs = []                                                           # blackout lengths are exactly 10 s
    for c in range(ok_hist.shape[1]):
        x = np.r_[1, ok_hist[:, c], 1].astype(int)
        d = np.diff(x)
        runs += (np.flatnonzero(d == 1) - np.flatnonzero(d == -1)).tolist()
    assert runs and set(runs) <= {20, 21}                               # 10 s at 2 Hz
    assert 0.92 < n_seen / n_all < 0.96                                 # 5 % dropouts (+ ~1 % blackouts)
    # a still body is not 'moving', a displaced one is
    s0 = M.SimState(0.0, parts={"S10": np.array([[10.0, 10.0, 0.5]])})
    shim2 = M.CameraShim(SCN, seed=1, p_drop=0.0, sigma_m=0.0)
    shim2.observe(s0)
    assert not shim2.observe(s0).moving.any()
    s1 = M.SimState(0.5, parts={"S10": np.array([[10.5, 10.0, 0.5]])})
    assert shim2.observe(s1).moving.all()


def test_mapping_and_codec_never_import_truth():
    for rel in ("intercycle/observer/mapping.py", "intercycle/codec/maplayer.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in mods if m.startswith("intercycle.truth") or m.startswith("intercycle.sim")], rel


# ------------------------------------------------------------------------------------------------ learned layers

def test_envelopes_cover_truth_with_little_false_area(sim, truth_union):
    m = sim.m
    cov, fa, nf, ne = [], [], 0, 0
    for mid in MIDS:
        e = m.envelope_union(mid, range(M.N_BINS), 0.05)
        t = truth_union[mid]
        cov.append((e & t).sum() / t.sum())
        fa.append((e & ~t).sum() / e.sum())
        nf += (e & ~t).sum()
        ne += e.sum()
    print("envelope coverage", np.round(cov, 3), "false", np.round(fa, 3), "aggregate false", round(nf / ne, 3))
    assert min(cov) >= 0.95
    assert max(fa) <= 0.20 and nf / ne <= 0.20
    # the broadcast box covers (what the robot actually receives) keep the guarantee
    bcov, bfa, bf, bn = [], [], 0, 0
    for mid in MIDS:
        d = sim.m.spec["machines"][MIDS.index(mid)]
        cells = np.zeros((d["nx"], d["nz"]), bool)
        for i0, j0, i1, j1 in m.box_cover(mid, tuple(range(M.N_BINS)), 0.05, max_boxes=24, cover_frac=0.995, world=False):
            cells[i0:i1, j0:j1] = True
        t = truth_union[mid]
        bcov.append((cells & t).sum() / t.sum())
        bf += (cells & ~t).sum()
        bn += cells.sum()
    print("box-cover coverage", np.round(bcov, 3), "false", round(bf / bn, 3))
    assert min(bcov) >= 0.95 and bf / bn <= 0.20


def test_envelope_depends_on_cycle_phase(sim):
    ids = ["S30", "S80", "HEM1", "SEAL1"]
    truth, per_bin = truth_cells(sim, binned=True, mids=ids, t_lo=600, t_hi=1000)
    m = sim.m
    right, wrong, area_ratio = [], [], []
    for mid in ids:
        union = m.envelope_union(mid, range(M.N_BINS), 0.05)
        for b in range(M.N_PHASE):
            t = per_bin[mid][b]
            if not t.any():
                continue
            e = m.envelope(mid, b, 0.05)
            right.append((e & t).sum() / t.sum())
            wrong.append((m.envelope(mid, (b + 8) % M.N_PHASE, 0.05) & t).sum() / t.sum())
            area_ratio.append(e.sum() / union.sum())
    print("per-bin coverage right-phase", round(float(np.mean(right)), 3), "opposite phase", round(float(np.mean(wrong)), 3),
          "envelope area / union", round(float(np.mean(area_ratio)), 3))
    assert np.mean(right) >= 0.90                      # the phase-binned envelope covers what really moves in that phase
    assert np.mean(wrong) <= np.mean(right) - 0.2      # ... and it is not just the union (opposite phase covers much less)
    assert np.mean(area_ratio) <= 0.8


def test_phase_and_production_state_tracking(sim):
    info, world, kin = sim.info, sim.world, sim.kin
    t, prod, bins = info["t"], info["prod"], info["bins"]
    t0 = info["t0"]
    P = SCN.line.takt_s
    # ---- truth: planned stops from the calendar, andon stops from the line's windows
    cal_break = np.zeros(len(t), bool)
    for a, b, kind in SCN.line_stops():
        cal_break |= (t % 86400 >= a) & (t % 86400 < b)
    andon = np.zeros(len(t), bool)       # ticks inside a true andon stop (after the missed index, before the restart)
    near = np.zeros(len(t), bool)        # ticks whose state is ambiguous (just around a stop / calendar edge)
    for w in world.W["S10"]:
        if w.kind == "andon" and t0 < w.bs < t[-1]:
            andon |= (t > w.bs + 1.06 * P + 4.0) & (t < w.ie - 2.0)
            near |= (t > w.bs + P * 0.9) & (t < w.ie + 6.0)
    for a, b, _ in SCN.line_stops():
        near |= (t % 86400 > a - 90) & (t % 86400 < b + 90)
    warm = t < t0 + 400
    assert andon.sum() > 150, "the traced window should contain several andon stops"
    brk = cal_break & ~warm
    print("break ticks", brk.sum(), "labelled break", (prod[brk] == M.BREAK).mean())
    assert (prod[brk & (t % 86400 > 17 * H + 5)] == M.BREAK).mean() >= 0.99
    rec = (prod[andon & ~warm] == M.ANDON).mean()
    fp = (prod[~near & ~warm & ~cal_break] != M.RUNNING).mean()
    print("andon recall", round(float(rec), 3), "false andon/other while running", round(float(fp), 4))
    assert rec >= 0.9
    assert fp <= 0.01
    # ---- phase bins vs the kinematics' own cycle phase (normal cycles only)
    ok = ~near & ~warm
    errs = {}
    for mid in ("S30", "S80", "HEM1", "SEAL1"):
        i = MIDS.index(mid)
        ref = "S10" if SCN.machine(mid).on_line else mid
        truth_bin = np.minimum((np.array([kin.cycle_phase(ref, x) for x in t[ok]]) * M.N_PHASE).astype(int), M.N_PHASE - 1)
        got = bins[ok, i]
        d = np.abs(truth_bin - got)
        d = np.minimum(d, M.N_PHASE - d)
        valid = got >= 0
        errs[mid] = (float((d[valid] <= 1).mean()), float(valid.mean()))
    print("phase-bin within +-1 of truth (share, share known):", errs)
    for mid, (share, known) in errs.items():
        assert known > 0.9 and share >= 0.9, (mid, share, known)


def test_break_people_crowd_on_north_aisle_west(sim):
    m = sim.m
    pb, hb = B.aggregate_traffic(m, M.PERSON, M.BREAK)
    pr, hr = B.aggregate_traffic(m, M.PERSON, M.RUNNING)
    s = m.spec["site"]
    ix = slice(int(2 // s["res"]), int(60 // s["res"]))
    iz = slice(int(30 // s["res"]), int(34 // s["res"]))
    mass_b, mass_r = pb[ix, iz].sum(), pr[ix, iz].sum()            # expected number of people in the aisle, x 2..60 m
    peak_b, peak_r = pb[ix, iz].max(), pr[ix, iz].max()
    # share of all break-time people mass that is in the north aisle west of x=60
    print(f"north aisle x<60: expected people break {mass_b:.1f} vs running {mass_r:.1f}; peak cell {peak_b:.2f} vs {peak_r:.2f}")
    assert mass_b >= 2.0 * mass_r
    assert peak_b >= 2.0 * peak_r and peak_b >= 0.5
    # the same through the public API at a break time bin (09:00) vs a running bin
    p_api, _ = m.traffic("person", int(9 * 2), M.BREAK)
    p_run, _ = m.traffic("person", int(9 * 2), M.RUNNING)
    assert p_api[ix, iz].sum() > 1.5 * p_run[ix, iz].sum()


def test_vehicle_layer_heading(sim):
    m = sim.m
    p, h = B.aggregate_traffic(m, M.VEHICLE, M.RUNNING)
    s = m.spec["site"]
    iz = int(16.5 // s["res"])                                      # south aisle: AGV loop runs eastbound (+x = heading bin 2) here
    xs = np.arange(10, 85)
    sel = (p[xs, iz] > 0.02) & (h[xs, iz] >= 0)
    assert sel.sum() >= 20
    assert (h[xs, iz][sel] == 2).mean() >= 0.8
    # parked AGVs at the line docks (z = 30.6) are vehicles too
    assert p[10:80, int(30.6 // s["res"])].max() > 0.02


def test_map_save_load_roundtrip(sim, tmp_path):
    f = tmp_path / "map.npz"
    sim.m.save(f)
    m2 = M.LayeredMap.load(f)
    assert np.array_equal(m2.l1_hits, sim.m.l1_hits) and np.array_equal(m2.hits[M.PERSON], sim.m.hits[M.PERSON])
    assert np.array_equal(m2.envelope("S30", 3), sim.m.envelope("S30", 3))
    assert np.array_equal(m2.traffic("person", 18, M.BREAK)[0], sim.m.traffic("person", 18, M.BREAK)[0])


def test_predict_obstacles(sim, bundle):
    m = sim.m
    dm = ML.decode_payload(zlib.decompress(b"".join(bundle.chunks[:bundle.manifest.n_chunks])[:bundle.manifest.payload_len]))
    phase0 = {mid: 0.10 for mid in MIDS}
    for mp in (m, dm):
        steps = mp.predict_obstacles(9 * H + 600, 20.0, phase0, M.RUNNING, step_s=0.5)
        assert len(steps) == 41 and steps[0].t == 9 * H + 600 and steps[-1].t == 9 * H + 620
        assert all(s.boxes.shape[1] == 4 and len(s.box_machine) == len(s.boxes) for s in steps)
        area = lambda s: float(((s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1])).sum())
        assert len({round(area(s), 1) for s in steps}) > 3          # the envelopes move with the predicted phase
        stopped = mp.predict_obstacles(9 * H + 600, 3.0, phase0, M.BREAK, step_s=1.0)
        assert area(stopped[0]) < area(steps[0])                    # stopped machines: only the parked pose
        unknown = mp.predict_obstacles(9 * H + 600, 3.0, {}, M.RUNNING, step_s=1.0)
        assert area(unknown[0]) > area(steps[0])                     # unknown phase = union of everything (conservative)
        assert len(steps[0].circles) == len(steps[0].circle_p)


# ------------------------------------------------------------------------------------------------ codec

def _carousel_receive(b, rx, frames, t0=60.0, fps=20.0):
    t, out = t0, None
    for f in frames:
        t += 1 / fps
        r = rx.ingest(b.manifest.issue_time - DAY0_UNIX + t, f)
        out = r or out
    return out


def test_codec_round_trip(sim, bundle):
    b, m = bundle, sim.m
    assert all(len(f) <= ble.ADV_MAX for f in b.frames + b.manifest_frames)
    assert max(len(f) for f in b.frames) == ble.ADV_MAX              # chunks fill the 251 B frame
    rx = ML.MapReceiver(KEYRING, SITE)
    dm = _carousel_receive(b, rx, b.carousel())
    assert dm is not None and rx.stats["map_ok"] == 1
    assert rx.cache.get(SITE, 7) is dm
    # L1: identical box covers
    for mid in MIDS:
        for bin_ in range(M.N_BINS):
            want = m.box_cover(mid, bin_, 0.05, max_boxes=12)
            got = dm.box_cover(mid, bin_)
            assert np.allclose(np.sort(want, 0), np.sort(got, 0)), (mid, bin_)
    # machine models
    assert np.allclose(np.nan_to_num(dm.period), np.round(np.nan_to_num(m.period) * 100) / 100)
    # L2 / L3: probabilities within the 4-bit quantisation (+ one hysteresis step), headings agree
    bad_p, n_p, bad_h, n_h, n_slices = 0, 0, 0, 0, 0
    for c in (M.PERSON, M.VEHICLE):
        for prod in range(M.N_PROD):
            for tod in range(M.N_TOD):
                if not m._observed(tod, prod, c):
                    assert not dm._observed(tod, prod, c)
                    continue
                n_slices += 1
                p0, h0 = m.traffic(c, tod, prod, fallback=False)
                p1, h1 = dm.traffic(c, tod, prod, fallback=False)
                big = p0 >= 0.03
                n_p += big.sum()
                bad_p += (np.abs(np.log(np.maximum(p1[big], 1e-6) / p0[big])) > 2 * math.log(50) / 14 + 0.05).sum()
                hh = (p0 >= 0.1) & (h0 >= 0)
                n_h += hh.sum()
                bad_h += (h1[hh] != h0[hh]).sum()
    print(f"slices {n_slices}; probability cells off by > 2 codes {bad_p}/{n_p}; heading mismatches {bad_h}/{n_h}")
    assert n_slices >= 20 and bad_p / n_p < 0.02 and bad_h / n_h < 0.12
    # same fallback behaviour as the learned map
    for c in ("person", "vehicle"):
        a, _ = m.traffic(c, 40, M.ANDON)
        z, _ = dm.traffic(c, 40, M.ANDON)
        assert (z > 0.02).sum() >= 0.8 * (a > 0.02).sum()
    st = b.stats
    print({k: st[k] for k in ("L1", "L2_vehicle", "L3_person", "raw_bytes", "compressed_bytes", "n_chunks", "n_parity", "total_bytes")})


def test_tampering_is_rejected(sim, bundle):
    b = bundle
    man = b.manifest
    # 1. manifest bit flipped (frame re-built, so the CRC is fine): signature check fails, nothing is accepted
    bad_blob = bytearray(b.manifest_blob)
    bad_blob[40] ^= 0x01
    rx = ML.MapReceiver(KEYRING, SITE)
    for f in ble.frame(bytes(bad_blob), msg_id=man.map_id, layer=ML.LAYER_MAP):
        rx.ingest(100.0, f)
    assert rx.stats["bad_signature"] == 1 and rx.manifest is None
    # 1b. manifest under a key id the robot does not know
    import dataclasses
    rx = ML.MapReceiver(KEYRING, SITE)
    for f in ble.frame(sign(ML.encode_manifest(dataclasses.replace(man, key_id=42)), KEY), msg_id=man.map_id, layer=ML.LAYER_MAP):
        rx.ingest(100.0, f)
    assert rx.stats["unknown_key"] == 1 and rx.manifest is None
    # 2. a forged map signed with another key under the same key id, and one under an unknown key id
    other = demo_key(99)
    forged = ML.pack_map(sim.m, other, version=9)
    rx = ML.MapReceiver(KEYRING, SITE)
    assert _carousel_receive(forged, rx, forged.carousel()) is None
    assert rx.stats["bad_signature"] >= 1 and rx.stats["map_ok"] == 0 and rx.cache.newest(SITE) is None
    # 3. a chunk altered in flight with a valid frame CRC: rejected by the signed hash table, the good chunk still completes the map
    frames = list(b.frames)
    idx = 5
    data = bytearray(b.chunks[idx])
    data[10] ^= 0xFF
    frames[idx] = ML.chunk_frame(man.map_id, idx, bytes(data))
    rx = ML.MapReceiver(KEYRING, SITE)
    assert _carousel_receive(b, rx, b.manifest_frames + frames[:man.n_chunks]) is None
    assert rx.stats["bad_chunk"] == 1 and 5 not in rx._chunks and rx.progress < 1.0
    assert rx.ingest(500.0, b.frames[idx]) is not None               # the genuine chunk 5 arrives later
    # 4. flipped bit in a chunk frame: CRC8 drops it before it is even considered
    corrupt = bytearray(b.frames[3])
    corrupt[50] ^= 0x04
    assert ML.parse_chunk(bytes(corrupt)) is None
    # 5. replay: an expired manifest, and an older version after a newer one is cached
    rx = ML.MapReceiver(KEYRING, SITE)
    late = man.issue_time - DAY0_UNIX + man.valid_s + 10
    for f in b.manifest_frames:
        rx.ingest(late, f)
    assert rx.stats["expired"] == 1 and rx.manifest is None
    rx = ML.MapReceiver(KEYRING, SITE)
    _carousel_receive(b, rx, b.carousel())
    old = ML.pack_map(sim.m, KEY, version=3)
    assert _carousel_receive(old, rx, old.carousel()) is None
    assert rx.stats["stale_version"] >= 1
    # 6. wrong site
    rx = ML.MapReceiver(KEYRING, SITE + 1)
    _carousel_receive(b, rx, b.carousel())
    assert rx.stats["wrong_site"] >= 1 and rx.cache.newest(SITE + 1) is None


def test_chunk_loss_assembly_completes_after_retries(sim, bundle):
    b = bundle
    L = b.carousel()
    rng = np.random.default_rng(3)
    # heavy iid loss: 40 %, a few loops of the carousel
    rx = ML.MapReceiver(KEYRING, SITE)
    loops, done = 0, None
    while done is None and loops < 12:
        keep = [f for f in L if rng.random() > 0.4]
        done = _carousel_receive(b, rx, keep)
        loops += 1
    assert done is not None and loops > 1 and loops <= 8
    # the first whole loop is lost (robot just entered range), then clean loops: complete within one loop
    rx = ML.MapReceiver(KEYRING, SITE)
    assert _carousel_receive(b, rx, L[: len(L) // 2]) is None and 0 < rx.progress < 1
    assert _carousel_receive(b, rx, L) is not None
    # chunks first, manifest last: pending chunks are verified once the manifest is in
    rx = ML.MapReceiver(KEYRING, SITE)
    assert _carousel_receive(b, rx, b.frames[:b.manifest.n_chunks]) is None
    assert rx.stats["chunk_before_manifest"] == b.manifest.n_chunks
    assert _carousel_receive(b, rx, b.manifest_frames) is not None
    # XOR parity: one chunk lost in every group of 8 is rebuilt without ever being received
    rx = ML.MapReceiver(KEYRING, SITE)
    n = b.manifest.n_chunks
    keep = [f for i, f in enumerate(b.frames) if not (i < n and i % 8 == 3)]
    assert _carousel_receive(b, rx, b.manifest_frames + keep) is not None
    assert rx.stats["repaired"] == len([i for i in range(n) if i % 8 == 3])
    # Monte Carlo at the spec point: 20 frames/s, 10 % loss
    sim_res = ML.simulate_assembly(b, KEYRING, SITE, loss=0.10, fps=20.0, trials=40, seed=2)
    nb = ML.pack_map(sim.m, KEY, version=7, parity_group=0)
    sim_np = ML.simulate_assembly(nb, KEYRING, SITE, loss=0.10, fps=20.0, trials=40, seed=2)
    print("assembly at 20 fps, 10 % loss: parity", {k: round(v, 2) for k, v in sim_res.items() if k in ("mean_s", "p95_s", "max_s", "loop_s")},
          "no parity", {k: round(v, 2) for k, v in sim_np.items() if k in ("mean_s", "p95_s", "max_s", "loop_s")})
    assert sim_res["completed"] == 40 and sim_np["completed"] == 40
    assert sim_res["mean_s"] <= sim_np["mean_s"] + 0.5                # parity does not hurt
    assert sim_res["p95_s"] < 60


def test_robot_caches_map_per_version(sim, bundle):
    b = bundle
    cache = ML.MapCache()
    rx = ML.MapReceiver(KEYRING, SITE, cache)
    first = _carousel_receive(b, rx, b.carousel())
    assert first is not None
    # same version again (another visit, a new receiver object sharing the cache): nothing is downloaded
    rx2 = ML.MapReceiver(KEYRING, SITE, cache)
    assert _carousel_receive(b, rx2, b.carousel()) is None
    assert rx2.stats["cached"] >= 1 and rx2.stats["chunk_ok"] == 0
    assert rx2.latest(b.manifest.issue_time - DAY0_UNIX + 100) is first
    # a new version downloads and is cached next to the old one
    import copy
    m2 = copy.copy(sim.m)                                            # shallow copy: same counts, other machine models
    m2.period = sim.m.period + 1.0
    b2 = ML.pack_map(m2, KEY, version=8, parity_group=8)
    second = _carousel_receive(b2, rx2, b2.carousel())
    assert second is not None and cache.get(SITE, 7) is first and cache.get(SITE, 8) is second
    assert cache.newest(SITE)[0] == 8
    # a half-finished download resumes in a later pass over the same version
    cache3 = ML.MapCache()
    r3 = ML.MapReceiver(KEYRING, SITE, cache3)
    half = b.manifest_frames + b.frames[: b.manifest.n_chunks // 2]
    assert _carousel_receive(b, r3, half) is None
    got = r3.stats["chunk_ok"]
    r4 = ML.MapReceiver(KEYRING, SITE, cache3)
    assert _carousel_receive(b, r4, b.carousel()) is not None
    assert r4.stats["chunk_ok"] <= b.manifest.n_total - got
    # the map layer is separable from the other three layers by its layer byte
    assert all(ML.is_map_frame(f) for f in b.frames + b.manifest_frames)
    assert not ML.is_map_frame(ble.frame(b"x" * 100, msg_id=1, layer=0)[0])


# ------------------------------------------------------------------------------------------------ tracker on a synthetic signal

def test_condition_tracker_on_synthetic_motion():
    """An off-line cell with a 45 s cycle (30 s active), one camera blackout over a start, then a 2-minute stop."""
    tr = M.ConditionTracker(SCN, use_learner=False)
    mi = MIDS.index("HEM1")
    P, t0 = 45.0, 7 * H
    got, want, stopped_bins = [], [], []
    for k in range(int(1500 * 2)):
        t = t0 + k * 0.5
        u = (t - t0) % P
        stop = 900 <= t - t0 < 1020                                   # no motion for 2 minutes
        moving = (u < 30) and not stop
        ok = np.ones(10, bool)
        if 495 <= t - t0 < 505:                                       # blackout covering the cycle start at 495 s
            ok[tr.cam_of_machine[mi]] = False
        obs = M.Observation(t + 0.2, t, np.array([M.PART], np.int8), np.array([21.0]), np.array([41.0]), np.array([0.5]),
                            np.array([np.nan]), np.array([moving]), np.array([mi], np.int16), ok)
        c = tr.step(obs)
        if t - t0 > 300 and not stop and not (1020 <= t - t0 < 1080):
            got.append(c.phase_bin[mi])
            want.append(min(int(u / P * M.N_PHASE), M.N_PHASE - 1))
        if 960 < t - t0 < 1020:
            stopped_bins.append(c.phase_bin[mi])
    got, want = np.array(got), np.array(want)
    d = np.abs(got - want)
    d = np.minimum(d, M.N_PHASE - d)
    assert abs(tr.period[mi] - P) < 0.6
    assert (got >= 0).all() and (d <= 1).mean() >= 0.95
    assert set(stopped_bins) == {M.STOP_BIN}                          # overdue -> stopped bin
    first = M.ConditionTracker(SCN, use_learner=False)
    assert (first._cond(t0).phase_bin == -1).all()                    # nothing known yet -> unknown, never guessed


def test_vectorised_mover_tracks_match_the_agents():
    """The harness's vectorised trajectories must equal PathAgent.at / present (guards against drift in truth/agents.py)."""
    from intercycle.truth.agents import agents as make_agents
    t = np.random.default_rng(1).uniform(0, 3 * 86400, 400)
    for a in make_agents(SCN, 0):
        x, z, hd, sp, pres = B.agent_track(a, t)
        for k in range(0, 400, 7):
            ex, ez, eh, es = a.at(t[k])
            assert (abs(x[k] - ex) < 1e-6 and abs(z[k] - ez) < 1e-6 and abs(sp[k] - es) < 1e-9), (a.id, k)
            if es > 0:
                assert abs((hd[k] - eh + math.pi) % (2 * math.pi) - math.pi) < 1e-6
            assert bool(pres[k]) == a.present(t[k])
