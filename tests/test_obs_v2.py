"""Observation v2 (lane S4, PLAN_TWIN §2): zone_motion / zone_vis / cam_floor_ok, the v1 -> v2 converter, the
npz / jsonl formats and the golden fixtures tests/fixtures/obs_v2_*.npz (WEAKNESSES T-W6, M-W10, M-W2).

Regenerate the golden fixtures (only after an intended contract change):
    python tests/test_obs_v2.py --regen
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.scenario import load  # noqa: E402

SCN = load()
NCAM = len(SCN.cameras)
NZ = len(SCN.machines)
RECTS = [m.rect for m in SCN.machines]
SHARE = M.zone_camera_share(RECTS, SCN.cameras)
FIX = ROOT / "tests" / "fixtures"


def _obs(t: float = 10.0, xs: Any = (), zs: Any = (), cls: Any = M.PART, moving: Any = True, machine: Any = -1, cam_ok: Any = None):
    n = len(xs)
    cl = np.full(n, cls, np.int8) if np.isscalar(cls) else np.asarray(cls, np.int8)
    mv = np.full(n, moving, bool) if np.isscalar(moving) else np.asarray(moving, bool)
    mc = np.full(n, machine, np.int16) if np.isscalar(machine) else np.asarray(machine, np.int16)
    return M.Observation(t + 0.2, t, cl, np.array(xs, float), np.array(zs, float), np.full(n, 0.4), np.full(n, np.nan),
                         mv, mc, np.ones(NCAM, bool) if cam_ok is None else np.asarray(cam_ok, bool))


def _v2(o):
    """(zone_motion, zone_vis, cam_floor_ok) of a v2 observation."""
    assert o.zone_motion is not None and o.zone_vis is not None and o.cam_floor_ok is not None
    return o.zone_motion, o.zone_vis, o.cam_floor_ok


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    np.testing.assert_array_equal(a, b)          # NaN == NaN here


def _same_obs(o1, o2):
    for f in M.OBS_V1_FIELDS + (M.OBS_V2_FIELDS if o1.is_v2 or o2.is_v2 else ()):
        _same(getattr(o1, f), getattr(o2, f))


# ------------------------------------------------------------------------------------------------ contract

def test_v1_observation_still_builds_positionally_and_is_not_v2():
    o = _obs()
    assert not o.is_v2 and o.zone_motion is None and o.zone_vis is None and o.cam_floor_ok is None
    with pytest.raises(ValueError, match="not a v2"):
        M.check_obs_v2(o)


def test_zone_camera_share_rows_sum_to_one_and_follow_the_camera_cells():
    assert SHARE.shape == (NZ, NCAM)
    np.testing.assert_allclose(SHARE.sum(1), 1.0)
    # the camera covering each zone centre always holds part of that zone
    cen = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2] for r in RECTS])
    cams = np.array(SCN.cameras, float)
    c0 = np.hypot(cen[:, None, 0] - cams[None, :, 0], cen[:, None, 1] - cams[None, :, 1]).argmin(1)
    assert (SHARE[np.arange(NZ), c0] > 0).all()
    # a single camera sees everything
    np.testing.assert_allclose(M.zone_camera_share(RECTS, [(0.0, 0.0)]), 1.0)


def test_convert_counts_moving_parts_per_zone_and_keeps_v1_fields():
    o = _obs(xs=[10, 11, 12, 30, 60, 21], zs=[20, 21, 22, 20, 33, 41],
             cls=[M.PART, M.PART, M.PART, M.PART, M.PERSON, M.SKID],
             moving=[True, True, False, True, True, True], machine=[0, 0, 0, 2, -1, 8])
    v = M.obs_v1_to_v2(o, SHARE, zones=RECTS)
    M.check_obs_v2(v, NZ, NCAM)
    want = np.zeros(NZ, np.float32)
    want[0], want[2] = 2, 1                     # still parts, people (here away from every zone) and skids
    #                                           are not machine motion
    _same(v.zone_motion, want)
    _same(v.zone_vis, np.ones(NZ, np.float32))
    _same(v.cam_floor_ok, o.cam_ok)
    assert all(getattr(v, f) is getattr(o, f) for f in M.OBS_V1_FIELDS), "v1 fields must pass through untouched"
    assert not o.is_v2, "the converter must not modify its input"


def test_dead_camera_makes_its_zones_missing_never_idle():
    """T-W6 / M-W10 safety direction: a zone whose floor is not fully seen is MISSING (NaN), not 'no motion' (0)."""
    k = int(np.flatnonzero(SHARE[0] > 0)[-1])     # a camera that sees only part of zone 0
    assert 0 < SHARE[0, k] < 1
    ok = np.ones(NCAM, bool)
    ok[k] = False
    v = M.obs_v1_to_v2(_obs(cam_ok=ok), SHARE, zones=RECTS)
    M.check_obs_v2(v, NZ, NCAM)
    hit = SHARE[:, k] > 0
    zm, zv, cf = _v2(v)
    assert np.isnan(zm[hit]).all()
    assert (zm[~hit] == 0).all()
    np.testing.assert_allclose(zv[hit], 1 - SHARE[hit, k], rtol=1e-6)
    assert not cf[k] and cf.sum() == NCAM - 1
    # all cameras down: every zone MISSING with zone_vis 0
    v0 = M.obs_v1_to_v2(_obs(cam_ok=np.zeros(NCAM, bool)), SHARE, zones=RECTS)
    M.check_obs_v2(v0)
    zm0, zv0, _ = _v2(v0)
    assert np.isnan(zm0).all() and (zv0 == 0).all()
    # an explicit lower vis_min keeps a partly seen zone observed
    vl = M.obs_v1_to_v2(_obs(cam_ok=ok), SHARE, vis_min=0.4, zones=RECTS)
    assert np.isfinite(_v2(vl)[0][hit & (zv >= 0.4)]).all()


@pytest.mark.parametrize("bad, msg", [
    (dict(zone_motion=np.zeros(NZ, np.float64)), "zone_motion"),                    # wrong dtype
    (dict(zone_vis=np.full(NZ, 1.5, np.float32)), "zone_vis"),
    (dict(zone_vis=np.full(NZ, np.nan, np.float32)), "zone_vis"),
    (dict(zone_motion=np.full(NZ, -1.0, np.float32)), ">= 0"),
    (dict(zone_motion=np.zeros(NZ - 1, np.float32)), "differ"),
    (dict(zone_vis=np.zeros(NZ, np.float32)), "MISSING"),                           # unseen zone reported as idle
    (dict(cam_floor_ok=np.zeros(NCAM - 1, bool)), "differ"),
])
def test_check_rejects_malformed(bad, msg):
    v = M.obs_v1_to_v2(_obs(), SHARE, zones=RECTS)
    from dataclasses import replace
    with pytest.raises(ValueError, match=msg):
        M.check_obs_v2(replace(v, **bad))


def test_check_rejects_floor_ok_on_dead_heartbeat_and_wrong_counts():
    v = M.obs_v1_to_v2(_obs(), SHARE, zones=RECTS)
    from dataclasses import replace
    ok = np.ones(NCAM, bool)
    ok[0] = False
    with pytest.raises(ValueError, match="heartbeat"):
        M.check_obs_v2(replace(v, cam_ok=ok))
    with pytest.raises(ValueError, match="zones"):
        M.check_obs_v2(v, n_zone=NZ + 1)
    with pytest.raises(ValueError, match="cameras"):
        M.check_obs_v2(v, n_cam=NCAM + 1)


def test_jsonl_and_npz_round_trip(tmp_path):
    ok = np.ones(NCAM, bool)
    ok[3] = False
    seq = [M.obs_v1_to_v2(_obs(t=1.0, xs=[10, 30], zs=[20, 20], machine=[0, 2]), SHARE, zones=RECTS),
           M.obs_v1_to_v2(_obs(t=1.5, cam_ok=ok), SHARE, zones=RECTS),
           M.obs_v1_to_v2(_obs(t=2.0, xs=[50], zs=[40], cls=M.PERSON, moving=False), SHARE, zones=RECTS)]
    for o in seq:
        line = json.dumps(M.obs_to_dict(o), allow_nan=False)        # strict JSON: NaN written as null
        o2 = M.obs_from_dict(json.loads(line))
        assert o2.is_v2
        _same_obs(o, o2)
    p = tmp_path / "seq.npz"
    M.save_obs_npz(p, seq, zone_camera_share=SHARE)
    back, extra = M.load_obs_npz(p)
    assert len(back) == 3
    for a, b in zip(seq, back):
        _same_obs(a, b)
    np.testing.assert_array_equal(extra["zone_camera_share"], SHARE)
    # v1 sequences keep working and stay v1
    v1 = [_obs(t=1.0, xs=[1.0], zs=[2.0])]
    M.save_obs_npz(tmp_path / "v1.npz", v1)
    b1, _ = M.load_obs_npz(tmp_path / "v1.npz")
    assert not b1[0].is_v2
    _same_obs(v1[0], b1[0])
    d1 = M.obs_to_dict(v1[0])
    assert d1["v"] == 1 and "zone_motion" not in d1 and not M.obs_from_dict(d1).is_v2
    with pytest.raises(ValueError, match="mixed"):
        M.save_obs_npz(tmp_path / "mixed.npz", [seq[0], v1[0]])


def test_layered_map_accepts_v2_and_drop_offline_keeps_v2_fields():
    m = M.LayeredMap(SCN)
    m.cam_offline = {1}
    v = M.obs_v1_to_v2(_obs(xs=[10, 30], zs=[20, 20], machine=[0, 2]), SHARE, zones=RECTS)
    d = m._drop_offline(v)
    assert d.is_v2
    _same(d.zone_motion, v.zone_motion)
    cf = _v2(d)[2]
    assert not d.cam_ok[1] and not cf[1] and cf.sum() == NCAM - 1
    tr = M.ConditionTracker(SCN)
    m.update(v, v.t, tr.step(v))                 # v2 is a drop-in for every v1 consumer


# ------------------------------------------------------------------------------------------------ golden fixtures

def _shim_sequence(n_ticks: int = 60):
    """A measured-noise CameraShim over a hand-made floor (no truth simulator): parts swinging in zones 0, 2 and 8,
    one standing in zone 4, a person walking through zone 2, frequent short blackouts. Returns v1 observations."""
    shim: dict[str, Any] = dict(M.MEASURED_SHIM)
    cam = M.CameraShim(SCN, seed=11, blackout_s=4.0, blackouts_per_h=240.0, **shim)
    obs = []
    for i in range(n_ticks):
        t = 3600.0 + i / M.FRAME_HZ
        ph = 2 * math.pi * t / 12.0
        parts = {
            "S10": np.array([[12 + 1.5 * math.sin(ph), 24.0, 0.4], [11.0, 22.0, 0.3]]),
            "S30": np.array([[30 + 2.0 * math.sin(ph + 1), 24 + math.cos(ph), 0.5]]),
            "S50": np.array([[47.5, 24.0, 0.4]]),
            "HEM1": np.array([[21 + math.sin(2 * ph), 41.0, 0.4]]) if i % 20 < 12 else np.array([[21.0, 41.0, 0.4]]),
        }
        st = M.SimState(t=t, mover_circles=np.array([[25 + 0.6 * i * 0.5, 24.0, 0.3]]), mover_cls=np.array([M.PERSON], np.int8),
                        mover_heading=np.array([math.pi / 2]), mover_speed=np.array([1.2]), parts=parts)
        obs.append(cam.observe(st))
    return obs


def _cases_sequence():
    """Hand-built edge cases: all cameras up with motion; one seam camera down; everything down; empty frame."""
    ok_seam = np.ones(NCAM, bool)
    ok_seam[int(np.flatnonzero(SHARE[0] > 0)[-1])] = False
    return [_obs(t=0.0, xs=[10, 11, 30], zs=[20, 21, 20], machine=[0, 0, 2]),
            _obs(t=0.5, xs=[10, 30], zs=[20, 20], machine=[0, 2], cam_ok=ok_seam),
            _obs(t=1.0, cam_ok=np.zeros(NCAM, bool)),
            _obs(t=1.5)]


def make_fixtures(out: Path = FIX) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, seq in (("obs_v2_shim.npz", _shim_sequence()), ("obs_v2_cases.npz", _cases_sequence())):
        conv = M.ObsV2Converter(SHARE, RECTS)
        v2 = [conv(o) for o in seq]
        M.save_obs_npz(out / name, v2, zone_camera_share=SHARE, zones=np.asarray(RECTS, float))


@pytest.mark.parametrize("name", ["obs_v2_shim.npz", "obs_v2_cases.npz"])
def test_golden_fixture_reconverts_identically(name):
    obs, extra = M.load_obs_npz(FIX / name)
    W = extra["zone_camera_share"]
    np.testing.assert_allclose(W, SHARE, atol=1e-12)
    np.testing.assert_array_equal(extra["zones"], np.asarray(RECTS, float))
    assert obs and all(o.is_v2 for o in obs)
    conv = M.ObsV2Converter(W, [tuple(z) for z in extra["zones"]])
    for o in obs:
        M.check_obs_v2(o, NZ, NCAM)
        v1 = M.Observation(**{f: getattr(o, f) for f in M.OBS_V1_FIELDS})
        _same_obs(conv(v1), o)
    zm = np.stack([_v2(o)[0] for o in obs])
    assert np.isnan(zm).any() and np.nanmax(zm) > 0, "fixture must exercise both MISSING and motion"


def test_golden_shim_fixture_matches_the_camera_shim():
    """Pins CameraShim + converter end to end: a change to either must regenerate the fixture on purpose."""
    obs, _ = M.load_obs_npz(FIX / "obs_v2_shim.npz")
    conv = M.ObsV2Converter(SHARE, RECTS)
    fresh = [conv(o) for o in _shim_sequence(len(obs))]
    for a, b in zip(obs, fresh):
        _same_obs(a, b)


if __name__ == "__main__":
    if "--regen" in sys.argv:
        make_fixtures()
        print("wrote", sorted(p.name for p in FIX.glob("obs_v2_*.npz")))
