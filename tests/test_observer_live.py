"""Lane S6: twin/observer_live.py (R-W1, T-W6, M-W4 lead decision, M-W7, C-W5)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from nacl.signing import SigningKey

from intercycle.codec import maplayer as ML
from intercycle.codec.trust import TrustError, TrustStore, issue_cert
from intercycle.margin_inflate import SCHEMA, Margin, inflation_spec
from intercycle.observer import learn_cycles as LC
from intercycle.observer import mapping as M
from intercycle.codec.layout import encode_fast, encode_slow
from intercycle.codec.sign import sign
from intercycle.codec.trust import key_id_of
from intercycle.types import DAY0_UNIX, FastLayer, MachineNow, SlowLayer, State
from twin import observer_live as OL

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2" / "intercycle_nav"))
from intercycle_nav import feed as F  # noqa: E402  # pyright: ignore[reportMissingImports]  (ros2 package dir, added above)
from intercycle_nav import masks as K  # noqa: E402  # pyright: ignore[reportMissingImports]
PRESSES = [dict(id="P1", kind="press", rect=(2.0, 2.0, 4.0, 4.0)), dict(id="P2", kind="press", rect=(12.0, 2.0, 14.0, 4.0))]
PERIODS = (1.6, 2.5)
SPEC = OL.twin_spec(PRESSES, (0.0, 0.0, 20.0, 8.0), [(4.0, 4.0), (14.0, 4.0)], site_id=7, key_id=1)
T0 = 6 * 3600.0


def margin_file(tmp_path: Path, margin_m: float = 0.5, cell_m: float = 0.25, frozen: bool = True) -> str:
    mg = Margin(margin_m, cell_m)
    p = tmp_path / "m7_margin.json"
    p.write_text(json.dumps(dict(schema=SCHEMA, frozen=frozen, status="ok", margin_m=margin_m, cell_m=cell_m,
                                 inflation=inflation_spec(mg))), encoding="utf-8")
    return str(p)


def stream(minutes: float = 20.0, nan_share: float = 0.10, seed: int = 0, stop=None, part_dx: float = 0.0,
           still_part: bool = True):
    """Synthetic 2 Hz Observation v2 ticks: two presses stroking every PERIODS s (1 % jitter, ram moving 40 % of the
    cycle), a PART detection on each ram (moving while it strokes), one walker, MISSING bursts on zone_motion.
    `stop` = (press index, t0, t1): that press does not stroke inside [t0, t1). part_dx shifts the ram detections along
    x; still_part=False emits a ram's PART detection only while it moves (a still ram is not a moving part)."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * 2)
    t = T0 + 0.5 * np.arange(n) + 0.13
    act = np.zeros((n, 2), bool)
    for i, P in enumerate(PERIODS):
        s = T0 - rng.uniform(0, P) + np.cumsum(P * (1 + 0.01 * rng.standard_normal(int(n * 0.5 / P) + 10)))
        if stop is not None and stop[0] == i:
            s = np.r_[s[s < stop[1]], s[s >= stop[1]] + (stop[2] - stop[1])]
        k = np.searchsorted(s, t, side="right") - 1
        act[:, i] = (k >= 0) & (t - s[np.clip(k, 0, None)] < 0.4 * P)
    miss = np.zeros((n, 2), bool)
    for i in range(2):
        while miss[:, i].mean() < nan_share:
            a = int(rng.integers(0, n))
            miss[a:a + int(rng.integers(2, 9)), i] = True
    out = []
    for j in range(n):
        zm = np.where(miss[j], np.nan, act[j].astype(float)).astype(np.float32)
        zv = np.where(miss[j], 0.5, 1.0).astype(np.float32)
        wx = 8.0 + 3.0 * np.sin(j / 50.0)
        keep = np.array([still_part or act[j, 0], still_part or act[j, 1], True])
        cls = np.array([M.PART, M.PART, M.PERSON], np.int8)
        out.append(M.Observation(
            t=float(t[j] + 0.2), t_cap=float(t[j]), cls=cls[keep],
            x=np.array([3.0 + part_dx, 13.0 + part_dx, wx])[keep], z=np.array([3.0, 3.0, 6.5])[keep],
            r=np.array([0.4, 0.4, 0.3])[keep], heading=np.array([np.nan, np.nan, 0.0])[keep],
            moving=np.array([act[j, 0], act[j, 1], True])[keep],
            machine=np.array([0, 1, -1], np.int16)[keep], cam_ok=np.ones(2, bool), zone_motion=zm, zone_vis=zv,
            cam_floor_ok=np.ones(2, bool)))
    return out


def write_jsonl(path: Path, obs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in obs:
            f.write(json.dumps(M.obs_to_dict(o)) + "\n")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    obs = stream(stop=(0, T0 + 600.0, T0 + 660.0))
    p = d / "obs_v2.jsonl"
    write_jsonl(p, obs)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(d)), obs_path=p)
    ob.advance(obs[-1].t)
    return ob, obs, d


# ---------------------------------------------------------------- input
@pytest.mark.parametrize("name", ["realised_state.jsonl", "plant_state.json", "obs_v2.json", "log.parquet"])
def test_reader_reads_only_obs_v2(tmp_path, name):
    with pytest.raises(ValueError, match="obs_v2.jsonl"):
        OL.ObsV2Reader(tmp_path / name, 2, 2)


def test_reader_is_incremental_and_waits_for_whole_lines(tmp_path):
    p = tmp_path / "obs_v2.jsonl"
    obs = stream(minutes=0.1, nan_share=0.0)
    lines = [json.dumps(M.obs_to_dict(o)) for o in obs[:3]]
    p.write_text(lines[0] + "\n" + lines[1][:20], encoding="utf-8")
    r = OL.ObsV2Reader(p, 2, 2)
    assert len(r.poll()) == 1
    with open(p, "a", encoding="utf-8") as f:
        f.write(lines[1][20:] + "\n" + lines[2] + "\n")
    got = r.poll()
    assert [o.t_cap for o in got] == [obs[1].t_cap, obs[2].t_cap]


def test_reader_refuses_v1_lines_and_wrong_shapes(tmp_path):
    p = tmp_path / "obs_v2.jsonl"
    o = stream(minutes=0.01, nan_share=0.0)[0]
    d = M.obs_to_dict(o)
    p.write_text(json.dumps({k: v for k, v in d.items() if k not in M.OBS_V2_FIELDS} | {"v": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="v2"):
        OL.ObsV2Reader(p, 2, 2).poll()
    p.write_text(json.dumps(d) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        OL.ObsV2Reader(p, 3, 2).poll()


# ---------------------------------------------------------------- fast / slow layers (R-W1, T-W6)
def test_presses_learned_from_zone_motion_with_nan_bursts(run):
    ob, obs, _ = run
    assert ob.n_ticks == len(obs)
    for i, P in enumerate(PERIODS):
        md = ob.models[i]
        assert md.confidence >= LC.MIN_CONF, ob.learned[i].diag
        assert abs(md.period_s / P - 1) < 0.05
        assert ob.learned[i].diag["missing_share"] >= 0.08


def test_missing_is_never_no_motion(run):
    ob, obs, _ = run
    j = max(k for k, o in enumerate(obs) if np.isnan(o.zone_motion[1]))
    assert np.isnan(ob.motion(obs[j])[1])
    T, Y = np.array(ob._t), np.array(ob._y)
    assert np.isnan(Y[:, 1]).mean() >= 0.08                # stored as MISSING, not as 0
    # a fast layer issued at a MISSING tick reports that press UNKNOWN with confidence 0
    sub = OL.ObserverLive(SPEC, ob.cfg)
    for o in obs[:j + 1]:
        sub.ingest(o)
    f = sub.fast(obs[j].t_cap)
    assert f.machines[1].state == State.UNKNOWN and f.machines[1].confidence == 0


def test_fast_layer_states_and_staleness(run):
    ob, obs, _ = run
    t = obs[-1].t_cap
    f = ob.fast(t)
    for i, mn in enumerate(f.machines):
        if np.isfinite(obs[-1].zone_motion[i]):
            assert mn.state in (State.ACTIVE, State.IDLE) and mn.confidence >= LC.MIN_CONF
    stale = ob.fast(t + 5.0)
    assert all(m.state == State.UNKNOWN and m.confidence == 0 for m in stale.machines)


def test_running_presses_are_never_read_stopped_through_nan_bursts(tmp_path):
    """Audit S6 #1 (probe on this stream, seed 1: P1 STOPPED with conf >= 4 on 23 ticks, 8 while the ram moved). Fast
    layer at every tick after the first model, 2 running presses, 10 % NaN bursts: no STOPPED with conf >= MIN_CONF,
    no IDLE / STOPPED while the ram moves (beyond one tick of quantisation), and the map never conditions on STOP_BIN."""
    obs = stream(minutes=12.0, seed=1)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    bad, useful, n = [], 0, 0
    for j, o in enumerate(obs):
        c = ob.ingest(o)
        assert (c.phase_bin != M.STOP_BIN).all(), (o.t_cap - T0, c.phase_bin)
        if o.t_cap < T0 + 360.0:
            continue
        f = ob.fast(o.t_cap)
        for i, mn in enumerate(f.machines):
            n += 1
            ok = mn.confidence >= LC.MIN_CONF
            useful += ok and mn.state in (State.ACTIVE, State.IDLE)
            moving = bool(obs[j].moving[i] and obs[j - 1].moving[i])
            if ok and (mn.state == State.STOPPED or (mn.state == State.IDLE and moving)):
                bad.append((round(o.t_cap - T0, 1), i, State(mn.state).name))
    assert min(m.confidence for m in ob.models) >= LC.MIN_CONF
    assert not bad, bad[:5]
    assert useful >= 0.5 * n, useful / n


def test_ticks_absent_from_the_stream_are_missing_not_held(tmp_path):
    """Audit S6 #2: 30 s of ticks absent from obs_v2.jsonl (dropped lines) are a gap, not held 'no motion': no STOPPED
    and no STOP_BIN right after the hole, and the learned model counts the hole as MISSING."""
    obs = stream(minutes=15.5, nan_share=0.0, seed=2)
    hole = (T0 + 700.0, T0 + 730.0)
    obs = [o for o in obs if not hole[0] <= o.t_cap < hole[1]]
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    after = 0
    for o in obs:
        c = ob.ingest(o)
        if hole[1] <= o.t_cap < hole[1] + 2.0:
            assert min(m.confidence for m in ob.models) >= LC.MIN_CONF          # learned before the hole
            assert (c.phase_bin != M.STOP_BIN).all()
            f = ob.fast(o.t_cap)
            assert all(m.state != State.STOPPED for m in f.machines)
            after += 1
    assert after >= 3
    assert ob.t_last is not None and ob.t_last > T0 + 900.0                     # relearned on a window holding the hole
    assert all(L is not None and L.diag["missing_share"] > 0.02 for L in ob.learned)


def test_press_stop_reads_stopped_bin(run):
    ob, obs, _ = run
    sub = OL.ObserverLive(SPEC, ob.cfg)
    bins = []
    for o in obs:
        c = sub.ingest(o)
        if T0 + 630.0 <= o.t_cap < T0 + 655.0:
            bins.append(int(c.phase_bin[0]))
    assert bins.count(M.STOP_BIN) > 0.6 * len(bins)


# ---------------------------------------------------------------- M-W4 lead decision: fail closed in unlearned bins
def test_press_grids_are_unknown_until_both_twin_bins_are_learned(tmp_path, run):
    fresh = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))
    assert all(L.half_life_ticks == 24 * 3600 * 2 for L in fresh.det.layers)     # 24 h for the running bin (M-W4)
    for i, pid in enumerate(fresh.ids):
        assert fresh.det.unknown_mask(pid).all()
    assert fresh.availability()["unlearned"]["all"] == 1.0
    ob, _, _ = run
    a = ob.availability()
    assert a["unlearned"]["per_machine"]["P1"] == 0.0        # P1 stopped 60 s: bin 0 and STOP_BIN learned
    assert a["unlearned"]["per_machine"]["P2"] == 1.0        # P2 never stopped: its STOP_BIN is not learned
    assert ob.det.unknown_mask("P2").all()
    m = ob.map_snapshot()
    g = M.LayeredMap(spec=SPEC)
    i = 1
    res = SPEC["env_res"]
    xs = g.l1_x0[i] + (np.arange(g.l1_nx[i]) + 0.5) * res
    zs = g.l1_z0[i] + (np.arange(g.l1_nz[i]) + 0.5) * res
    for x in xs:
        for z in zs:
            assert any(r[0] <= x <= r[2] and r[1] <= z <= r[3] for r in m.unknown_rects), (x, z)


# ---------------------------------------------------------------- M-W7 margin from JSON only
def test_margin_is_read_from_the_frozen_json_and_applied(run, tmp_path):
    ob, _, _ = run
    assert ob.margin == Margin(0.5, 0.25)
    for i in range(ob.M):
        p, q = ob.pred_l1(i), ob.inflated_l1(i)
        assert (q | ~p).all() and q.sum() >= p.sum()
    with pytest.raises(ValueError, match="frozen"):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path, frozen=False)))
    with pytest.raises(ValueError, match="cell_m"):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path, cell_m=0.5)))
    with pytest.raises(ValueError, match="does not fit"):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path, margin_m=1.5)))
    with pytest.raises(FileNotFoundError):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=str(tmp_path / "missing.json")))


def _centres(ob, i: int, mask: np.ndarray):
    res = SPEC["env_res"]
    ix, iz = np.nonzero(mask)
    return zip(ob.map.l1_x0[i] + (ix + 0.5) * res, ob.map.l1_z0[i] + (iz + 0.5) * res)


def _covered(boxes, x: float, z: float) -> bool:
    return any(r[0] <= x <= r[2] and r[1] <= z <= r[3] for r in np.asarray(boxes, float).reshape(-1, 4))


def test_only_detector_unknown_cells_are_broadcast_unknown(run):
    """Option B: unknown_rects = the live map's own + the detector's UNKNOWN cells (CUSUM, fail-closed unlearned),
    inflated by the margin. The learned running envelope of P1 is NOT broadcast as UNKNOWN."""
    ob, _, _ = run
    m = ob.map_snapshot()
    assert isinstance(m, OL.TwinMapSnapshot)
    for i in range(ob.M):
        U = ob.unknown_l1(i)
        assert (U | ~ob.det.unknown_mask(ob.ids[i])).all()          # inflation only grows the detector's set
        for x, z in _centres(ob, i, U):
            assert _covered(m.unknown_rects, x, z)
    assert ob.map.unknown_rects is not m.unknown_rects             # the live map is not modified
    env0 = np.asarray(ob.map.envelope("P1", 0, ob.det.cfg.p_min), bool) & ~ob.unknown_l1(0)
    assert env0.any()
    assert not all(_covered(m.unknown_rects, x, z) for x, z in _centres(ob, 0, env0))


def test_every_running_bin_is_broadcast_as_the_inflated_bin0_envelope(run):
    ob, _, _ = run
    m = ob.map_snapshot()
    for i, pid in enumerate(ob.ids):
        b0 = m.box_cover(pid, 0)
        for b in range(M.N_PHASE):
            assert np.array_equal(m.envelope(pid, b), m.envelope(pid, 0))
            assert np.array_equal(m.box_cover(pid, b), b0)
        for x, z in _centres(ob, i, inflate_env(ob, pid, 0)):
            assert _covered(b0, x, z)
        for x, z in _centres(ob, i, inflate_env(ob, pid, M.STOP_BIN)):
            assert _covered(m.box_cover(pid, M.STOP_BIN), x, z)
    with pytest.raises(ValueError):
        OL.twin_bin(17)
    assert ob.map._cover_cache is not m._cover_cache               # the live map's covers stay un-inflated


def inflate_env(ob, pid: str, b: int) -> np.ndarray:
    from intercycle.margin_inflate import inflate
    return inflate(np.asarray(ob.map.envelope(pid, b, ob.det.cfg.p_min), bool), ob.margin, cell_m=SPEC["env_res"])


# ---------------------------------------------------------------- placeholders in a sealed run
def test_placeholders_fail_loudly_in_a_sealed_run(tmp_path, monkeypatch):
    mp = margin_file(tmp_path)
    with pytest.raises(LC.PlaceholderConstantError):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=mp, sealed=True))
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(LC.PlaceholderConstantError):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=mp))
    measured = LC.consts_from_config({"kind": "press", "p_min": {"value": 1.0, "source": "P4"},
                                      "debounce_s": {"value": 0.25, "source": "P4"}})
    with pytest.raises(LC.PlaceholderConstantError, match="motion_thr"):
        OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=mp, consts={"press": measured}))
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=mp, consts={"press": measured},
                                                     motion_thr_source="stat_v1 energy p99 of idle ROI (dev)"))
    assert ob.tags["sealed"] is True


def test_unknown_machine_kind_is_refused(tmp_path):
    spec = OL.twin_spec([dict(id="X", kind="lathe", rect=(1, 1, 2, 2))], (0, 0, 5, 5), [(1, 1)], 7, 1)
    with pytest.raises(ValueError, match="lathe"):
        OL.ObserverLive(spec, OL.ObserverLiveConfig(margin_path=margin_file(tmp_path)))


def test_time_of_day_structure_is_tagged_assumed(run):
    assert run[0].diagnostics()["tags"]["time_of_day"].startswith("ASSUMED")


def test_m7_application_tag_states_option_b(run):
    ob, _, _ = run
    tag = ob.diagnostics()["tags"]["m7_application"]
    assert tag.startswith("LEAD DECISION") and "option B" in tag and "calibration_pred" in tag
    a = ob.availability()
    assert set(a["closed_share"]) == {"running", "stopped", "union"}
    assert a["closed_share"]["union"] >= max(a["closed_share"]["running"], a["closed_share"]["stopped"])


# ---------------------------------------------------------------- option B on the robot side (end to end)
@pytest.fixture(scope="module")
def still_run(tmp_path_factory):
    """P1's ram sits at the edge of its zone rect (part_dx 0.9: its envelope reaches outside the rect) and is a PART
    only while it moves; P1 stops for 60 s (STOP_BIN learned, empty envelope); P2 never stops (fail closed)."""
    d = tmp_path_factory.mktemp("still")
    obs = stream(stop=(0, T0 + 600.0, T0 + 660.0), part_dx=0.9, still_part=False)
    p = d / "obs_v2.jsonl"
    write_jsonl(p, obs)
    ob = OL.ObserverLive(SPEC, OL.ObserverLiveConfig(margin_path=margin_file(d)), obs_path=p)
    ob.advance(obs[-1].t)
    return ob, obs


def _robot(ob, t: float, nows: list[tuple[State, int]]):
    """The robot's verified view after one signed broadcast of ob's slow + map and a crafted fast layer."""
    now_u = DAY0_UNIX + t
    root, key = SigningKey.generate(), SigningKey.generate()
    ring = _store(key, root, now_u).keyring_at(now_u)
    kid = key_id_of(key.verify_key)
    site = K.Site([d["id"] for d in PRESSES], {d["id"]: tuple(map(float, d["rect"])) for d in PRESSES}, [],
                  SPEC["site_id"], kid, (0.0, 0.0, 20.0, 8.0), on_line={d["id"]: False for d in PRESSES},
                  own_stop={d["id"]: True for d in PRESSES})
    st = K.ObserverState(site, dict(ring))
    out = ob.sign_layers(key, t, ring)
    for k, fr in enumerate(out["map"].carousel()):
        st.ingest(t + k * 1e-4, F.FRAME + fr)
    s = ob.slow(t)
    st.ingest(t, F.BLOB + sign(encode_slow(SlowLayer(s.site_id, kid, s.issue_time, [], s.machines)), key))
    fast = [MachineNow(i, sv, 0.3 if sv == State.ACTIVE else 0.0, 0.5, 10.0, c, 1) for i, (sv, c) in enumerate(nows)]
    st.ingest(t, F.BLOB + sign(encode_fast(FastLayer(s.site_id, kid, DAY0_UNIX + t, 1, fast)), key))
    view = st.map_view(t + 0.5)
    assert view is not None, st.all_stats()
    snap = st.snapshot(t + 0.5)
    return site, view, snap


GRID = K.Grid(0.0, 0.0, 0.25, 80, 32)


def _outside_cells(ob, site) -> np.ndarray:
    """Keep-out grid cells of P1's inflated bin-0 envelope that lie outside its zone rect grown by the mask's
    inflate_m (the rect itself is the light-curtain rule's, closed while no idle window covers the horizon)."""
    p = K.MaskParams()
    x0, z0, x1, z1 = site.rects["P1"]
    g = p.inflate_m + GRID.res
    sel = np.zeros((GRID.nx, GRID.ny), bool)
    for x, z in _centres(ob, 0, inflate_env(ob, "P1", 0)):
        if not (x0 - g <= x <= x1 + g and z0 - g <= z <= z1 + g):
            sel[int(x / GRID.res), int(z / GRID.res)] = True
    return sel


def test_confident_stop_opens_the_danger_cells_running_keeps_them_closed(still_run):
    ob, obs = still_run
    t = obs[-1].t_cap
    assert ob.availability()["unlearned"]["per_machine"]["P1"] == 0.0
    assert not np.asarray(ob.map.envelope("P1", M.STOP_BIN, ob.det.cfg.p_min), bool).any()   # still ram: no part
    site, view, snap = _robot(ob, t, [(State.STOPPED, 8), (State.ACTIVE, 8)])
    assert snap.prod == M.RUNNING
    p = K.MaskParams()
    sel = _outside_cells(ob, site)
    assert sel.any()
    g_stop, info = K.keepout_mask(GRID, "learned", site, snap, view, p)
    assert info["bins"]["P1"]["bins"] == 1                                   # (STOP_BIN,)
    assert (g_stop[sel] == K.FREE).all()
    site, view, snap = _robot(ob, t, [(State.ACTIVE, 8), (State.ACTIVE, 8)])
    g_run, info = K.keepout_mask(GRID, "learned", site, snap, view, p)
    assert info["bins"]["P1"]["bins"] != "all"                               # cycling: running bins, bin-0 envelope
    assert (g_run[sel] == K.KEEP_OUT).all()


@pytest.mark.parametrize("now", [(State.UNKNOWN, 0), (State.STOPPED, 3), (State.IDLE, 2), None])
def test_unknown_missing_or_low_confidence_keeps_the_danger_cells_closed(still_run, now):
    ob, obs = still_run
    t = obs[-1].t_cap
    nows = [(State.UNKNOWN, 0) if now is None else now, (State.ACTIVE, 8)]
    site, view, snap = _robot(ob, t, nows)
    if now is None:                                                          # no fast entry for P1 at all
        snap.machines[0] = K.MachineView("P1", 0, "UNKNOWN", None, None)
    g, info = K.keepout_mask(GRID, "learned", site, snap, view, K.MaskParams())
    assert info["bins"]["P1"]["bins"] == "all"
    assert (g[_outside_cells(ob, site)] == K.KEEP_OUT).all()


def test_fail_closed_press_stays_closed_even_when_stopped(still_run):
    """P2 never stopped: its STOP_BIN is unlearned, so its whole grid is broadcast UNKNOWN (M-W4 lead decision) and a
    confident STOPPED opens nothing."""
    ob, obs = still_run
    t = obs[-1].t_cap
    site, view, snap = _robot(ob, t, [(State.ACTIVE, 8), (State.STOPPED, 8)])
    g, info = K.keepout_mask(GRID, "learned", site, snap, view, K.MaskParams())
    assert info["bins"]["P2"]["bins"] == 1
    for x, z in _centres(ob, 1, np.ones((ob.map.l1_nx[1], ob.map.l1_nz[1]), bool)):
        if 0 <= x < 20.0 and 0 <= z < 8.0:
            assert g[int(x / GRID.res), int(z / GRID.res)] == K.KEEP_OUT, (x, z)


def test_calibration_pred_is_the_robot_lookup_uninflated(still_run):
    """S7b alignment: inflate(calibration_pred) == what the robot gets for its lookup, per press, for every fast state."""
    ob, obs = still_run
    t = obs[-1].t_cap
    s = ob.slow(t)
    for sv, c, want in ((State.STOPPED, 8, (M.STOP_BIN,)), (State.ACTIVE, 8, (0,)), (State.IDLE, 8, (0,)),
                        (State.UNKNOWN, 0, OL.TWIN_BINS), (State.STOPPED, 3, OL.TWIN_BINS)):
        fast = FastLayer(s.site_id, 1, DAY0_UNIX + t, 1, [MachineNow(i, sv, 0.0, 0.5, 10.0, c, 1) for i in range(ob.M)])
        assert OL.robot_bins(s.machines[0], fast.machines[0]) == want
        preds = ob.calibration_pred(fast)
        m = ob.map_snapshot()
        for i, pid in enumerate(ob.ids):
            assert np.array_equal(preds[i], ob.pred_l1(i, want))
            got = np.zeros_like(preds[i])
            for b in want:
                got |= m.envelope(pid, b, ob.det.cfg.p_min)
            got |= ob.unknown_l1(i)
            assert np.array_equal(OL.inflate(preds[i], ob.margin, cell_m=SPEC["env_res"]), got)
    assert OL.robot_bins(None, None) == OL.TWIN_BINS


# ---------------------------------------------------------------- C-W5 signing, explicit keyring, fail closed
def _store(key: SigningKey, root: SigningKey, now: float) -> TrustStore:
    st = TrustStore(root.verify_key, SPEC["site_id"])
    st.add_cert(issue_cert(root, key.verify_key, SPEC["site_id"], int(now - 3600), int(now + 3600)))
    return st


def test_layers_are_signed_and_verify_only_with_the_keyring(run):
    ob, obs, _ = run
    t = obs[-1].t_cap
    now = DAY0_UNIX + t
    root, key, other = SigningKey.generate(), SigningKey.generate(), SigningKey.generate()
    ring = _store(key, root, now).keyring_at(now)
    out = ob.sign_layers(key, t, ring)
    s, f = OL.verify_blob(out["slow"], ring), OL.verify_blob(out["fast"], ring)
    assert [m.confidence >= LC.MIN_CONF for m in s.machines] == [True, True]
    assert len(f.machines) == 2
    rx = ML.MapReceiver(ring, SPEC["site_id"])
    got = None
    for k, fr in enumerate(out["map"].carousel()):
        got = rx.ingest(t + k / 20.0, fr) or got
    assert got is not None
    for bad in ({}, _store(other, root, now).keyring_at(now)):           # empty / wrong keyring: rejected
        with pytest.raises(TrustError):
            OL.verify_blob(out["slow"], bad)
        rx2 = ML.MapReceiver(bad, SPEC["site_id"])
        assert all(rx2.ingest(t + k / 20.0, fr) is None for k, fr in enumerate(out["map"].carousel()))
    with pytest.raises(TrustError):                                      # a key outside the keyring is never used
        ob.sign_layers(other, t, ring)
    expired = _store(key, root, now).keyring_at(now + 7200)
    with pytest.raises(TrustError):
        ob.sign_layers(key, t, expired)
    tampered = bytearray(out["fast"])
    tampered[20] ^= 1
    with pytest.raises(TrustError):
        OL.verify_blob(bytes(tampered), ring)


# ---------------------------------------------------------------- the twin path never reaches truth (R-W1)
FORBIDDEN = ("intercycle.sim", "intercycle.truth", "intercycle.eval.conformal_margin", "intercycle.observer.signals",
             "twin.engine", "twin.plant", "twin.camera", "twin.camera_v0", "twin.oracle_feed", "twin.truth_guard",
             "twin.daygen", "twin.daygen_stub")
FORBIDDEN_STRINGS = ("realised_state", "plant_state", "observer_learned", "LearnedObserver", "true_models")


def _file_of(mod: str) -> Path | None:
    base = REPO.joinpath(*mod.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _imports(path: Path, mod: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            src = node.module or ""
            if node.level:
                base = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                src = ".".join(base + ([node.module] if node.module else []))
            out.add(src)
            out.update(f"{src}.{a.name}" for a in node.names)
    return out


def _closure(start: str) -> set[str]:
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        parts = m.split(".")
        for k in range(1, len(parts) + 1):
            mm = ".".join(parts[:k])
            if mm in seen:
                continue
            seen.add(mm)
            f = _file_of(mm)
            if f is not None and mm.split(".")[0] in ("intercycle", "twin"):
                todo += [x for x in _imports(f, mm) if x.split(".")[0] in ("intercycle", "twin")]
    return seen


def test_observer_live_import_graph_never_reaches_truth_or_the_body_shop_observer():
    mods = _closure("twin.observer_live")
    assert "intercycle.observer.learn_cycles" in mods and "intercycle.observer.change_detect" in mods
    assert "intercycle.margin_inflate" in mods
    bad = sorted(m for m in mods if any(m == f or m.startswith(f + ".") for f in FORBIDDEN))
    assert not bad, bad


def test_observer_live_code_names_no_truth_file():
    tree = ast.parse((REPO / "twin" / "observer_live.py").read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    lits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + \
            [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    for s in FORBIDDEN_STRINGS:
        assert not any(s in x for x in lits + names), s


def test_import_walker_catches_a_forbidden_import(tmp_path, monkeypatch):
    assert "intercycle.sim.observer_learned" in _closure("intercycle.sim.observer_learned")
    assert any(m.startswith("intercycle.sim") for m in _closure("intercycle.eval.run_seeds"))
