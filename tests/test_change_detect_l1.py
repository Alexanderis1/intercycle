"""M-W4 on the body-shop code (observer/change_detect_l1.py, the adapter lane S8 / A-lite uses).

Two existing machines (HEM1, SEAL1) are learned for 40 min through the measured-error camera shim (32 % miss,
sigma 0.285 m, 15 % gross errors, blackouts), with the existing LayeredMap and the existing static-space detector
(observer/change.py) running alongside. Then the floor changes and runs 15 min more. The observer's phase of the two
existing machines is given (the ConditionTracker's job, not under test here).

Scenario "arms": new moving arms whose sweep line is 0.5, 1.0, 1.5 and 2.0 m from a zone edge (footprint d +- 0.4 m).
Scenario "inzone": changes that stay inside zone + 1 m: HEM1's program changes (its rollers run on new tracks inside the
zone) and a part is parked for good 0.5 m outside SEAL1's edge (a newly always-occupied cell).

Measured per change: the share of its true 0.25 m footprint cells that the map no longer broadcasts FREE (inside an
UNKNOWN box of map.unknown_rects, or inside the existing machine's envelope learned before the change), over time,
with today's code (change.py only) and with the in-zone detector added.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from intercycle.observer import mapping as M
from intercycle.observer.change import ChangeDetector
from intercycle.observer.change_detect_l1 import L1ChangeDetector
from intercycle.scenario import load

SCN = load()
IDS = [m.id for m in SCN.machines]
HI, SI = IDS.index("HEM1"), IDS.index("SEAL1")
P_HEM, P_SEAL = 45.0, 60.0
T0 = 30000.0
LEARN_S, AFTER_S = 40 * 60.0, 15 * 60.0
T_CH = T0 + LEARN_S
R_ARM = 0.4


def _arm(d_name, near, P, c, off, ax):
    """A new arm: two circles (offset `off` along the sweep axis) sweeping +-1.5 m (z) / +-2 m (x) for half of its
    cycle, parked at the rest pose for the other half."""
    def circles(t):
        u = ((t - T_CH) % P) / P
        f = math.sin(2 * math.pi * u / 0.5) if u < 0.5 else 0.0
        a = 1.5 * f if ax == "z" else 2.0 * f
        dx, dz = (0.0, a) if ax == "z" else (a, 0.0)
        return np.array([[c[0] + dx, c[1] + dz, R_ARM], [c[0] + off[0] + dx, c[1] + off[1] + dz, R_ARM]])
    return d_name, near, circles, P


ARMS = [
    _arm("east_HEM1_0.5m", HI, 50.0, (26.0 + 0.5, 41.0), (0.0, 0.8), "z"),
    _arm("north_SEAL1_1.0m", SI, 55.0, (46.0, 45.0 + 1.0), (0.8, 0.0), "x"),
    _arm("west_HEM1_1.5m", HI, 37.0, (16.0 - 1.5, 41.0), (0.0, 0.8), "z"),
    _arm("east_SEAL1_2.0m", SI, 42.0, (51.0 + 2.0, 41.0), (0.0, 0.8), "z"),
]


def _hem_rollers(t, reverse=False):
    """HEM1: two rollers travelling 7 m along the zone in 70 % of the cycle, parked for the rest. The new program runs
    the same tracks in reverse: the same footprint (so the envelope union over phases and change.py's camera-bump check
    see nothing), but every phase bin now has the rollers where the map learned free floor."""
    u = ((t - T0) % P_HEM) / P_HEM
    f = min(u / 0.7, 1.0)
    xr = 24.5 - 7.0 * f if reverse else 17.5 + 7.0 * f
    return np.array([[xr, 40.0, 0.5], [xr, 42.0, 0.5]])


INZONE = [
    ("HEM1_program_change", HI, lambda t: _hem_rollers(t, reverse=True), P_HEM),
    ("SEAL1_parked_part_0.5m", SI, lambda t: np.array([[51.5, 40.6, R_ARM], [51.5, 41.4, R_ARM]]), P_SEAL),
]


def _state_fn(changes, program_change):
    def state(t):
        parts = {}
        parts["HEM1"] = _hem_rollers(t, reverse=program_change and t >= T_CH)
        v = ((t - T0) % P_SEAL) / P_SEAL                  # SEAL1: an arm sweeping inside its zone, 60 % active
        a = 2 * math.pi * min(v / 0.6, 1.0)
        parts["SEAL1"] = np.array([[46.0 + 3.0 * math.cos(a), 41.0 + 2.5 * math.sin(a), 0.5],
                                   [46.0 + 1.5 * math.cos(a), 41.0 + 1.2 * math.sin(a), 0.5]])
        if t >= T_CH:
            for name, _, circles, _ in changes:
                if name != "HEM1_program_change":
                    parts[name] = circles(t)
        return M.SimState(t=t, parts=parts)
    return state


def _cond(t):
    pb = np.full(len(IDS), -1, np.int16)
    ph = np.full(len(IDS), np.nan)
    for i, P in ((HI, P_HEM), (SI, P_SEAL)):
        ph[i] = ((t - T0) % P) / P
        pb[i] = int(ph[i] * M.N_PHASE)
    nan = np.full(len(IDS), np.nan)
    return M.Cond(t=t, tod_bin=int(M.tod_bin(t)), prod_state=0, phase_bin=pb, phase=ph, period=nan, busy=nan, jitter=nan)


def _occupied(circles, t, res=0.25):
    """Centres of the 0.25 m cells the change truly occupies at time t."""
    pts = []
    for x, z, r in circles(t):
        for i in range(int((x - r) / res) - 1, int((x + r) / res) + 2):
            for j in range(int((z - r) / res) - 1, int((z + r) / res) + 2):
                if math.hypot((i + 0.5) * res - x, (j + 0.5) * res - z) <= r:
                    pts.append(((i + 0.5) * res, (j + 0.5) * res))
    return np.unique(np.array(pts).reshape(-1, 2), axis=0)


def _in_rects(pts, rects):
    if not len(rects):
        return np.zeros(len(pts), bool)
    R = np.array(rects, float)
    return ((pts[:, :1] >= R[None, :, 0]) & (pts[:, :1] <= R[None, :, 2]) &
            (pts[:, 1:] >= R[None, :, 1]) & (pts[:, 1:] <= R[None, :, 3])).any(1)


def _in_envelope(mp, pts, i, b):
    """Cells inside machine i's broadcast envelope for phase bin b (not FREE)."""
    env = mp.envelope(IDS[i], b)
    res = mp.spec["env_res"]
    ix = np.floor((pts[:, 0] - mp.l1_x0[i]) / res).astype(int)
    iz = np.floor((pts[:, 1] - mp.l1_z0[i]) / res).astype(int)
    ok = (ix >= 0) & (ix < env.shape[0]) & (iz >= 0) & (iz < env.shape[1])
    out = np.zeros(len(pts), bool)
    out[ok] = env[ix[ok], iz[ok]]
    return out


WEEK_FACTOR = 7 * 24 * 3600.0 / LEARN_S                  # 40 min of learning stands in for a week of history


def _age_history(mp, det):
    """Scale the learned counts to a week of history (p, and the Markov null's ratios, unchanged): the map's L1 counts x
    WEEK_FACTOR (no forgetting), the in-zone layers' decayed counts x WEEK_FACTOR capped at their saturation
    1 / (1 - lam). Simulating the week tick by tick is out of reach for a unit test."""
    mp.l1_hits *= WEEK_FACTOR
    mp.l1_visits *= WEEK_FACTOR
    if det is None:
        return
    for L in det.layers:
        sat = 1.0 / (1.0 - L.lam)
        f = np.where(L.n > 0, np.minimum(WEEK_FACTOR, sat / np.maximum(L.n, 1e-9)), 1.0)
        for a in (L.n, L.k, L.kx, L.e):
            a *= f


def _cam_offline(mp, pts):
    """Cells in the region of a camera the map holds offline / bumped (UNKNOWN in MapView.unknown_mask)."""
    if not mp.cam_offline:
        return np.zeros(len(pts), bool)
    s = mp.spec["site"]
    ix = np.clip(np.floor((pts[:, 0] - s["x0"]) / s["res"]).astype(int), 0, mp.nx - 1)
    iz = np.clip(np.floor((pts[:, 1] - s["z0"]) / s["res"]).astype(int), 0, mp.nz - 1)
    return np.isin(mp.cam_of_cell[ix, iz], list(mp.cam_offline))


LEGACY_EVENTS: dict = {}


def _simulate(changes, program_change: bool, with_inzone: bool):
    cam = M.CameraShim(SCN, seed=7, **M.MEASURED_SHIM)  # pyright: ignore[reportArgumentType]  (mixed-type dict literal)
    mp = M.LayeredMap(SCN)
    legacy = ChangeDetector(mp)
    det = L1ChangeDetector(mp) if with_inzone else None
    state = _state_fn(changes, program_change)
    cover = {name: [] for name, *_ in changes}            # (t - T_CH, share of truly occupied cells not FREE)
    early = []
    aged = False
    lev = LEGACY_EVENTS.setdefault((program_change, with_inzone), [])
    for k in range(int((LEARN_S + AFTER_S) * M.FRAME_HZ)):
        t = T0 + k / M.FRAME_HZ
        obs = cam.observe(state, t)
        cond = _cond(obs.t_cap)
        if obs.t_cap >= T_CH and not aged:
            _age_history(mp, det)
            aged = True
        mp.update(obs, obs.t_cap, cond)
        lev += [(round(e["t"] - T_CH), e["kind"]) for e in legacy.step(obs)]
        if det is not None:
            ev = det.step(obs, cond)
            if obs.t_cap < T_CH:
                early += [e for e in ev if e["kind"] == "inzone_change"]
            det.sync()
        if obs.t_cap >= T_CH and k % 2 == 0:
            for name, i, circles, _ in changes:
                occ = _occupied(circles, obs.t_cap)
                nf = (_in_envelope(mp, occ, i, int(cond.phase_bin[i])) | _in_rects(occ, mp.unknown_rects)
                      | _cam_offline(mp, occ))
                cover[name].append((obs.t_cap - T_CH, float(nf.mean())))
    return cover, early


WINDOWS = ((0, 60), (60, 180), (180, 300), (300, AFTER_S + 1))


def _win(series, a, b):
    v = [s for dt, s in series if a <= dt < b]
    return float(np.mean(v)) if v else float("nan")


def _first_full(series, hold_s=60.0):
    """First time after which every sample of the next `hold_s` seconds has every occupied cell not FREE."""
    ts = np.array([dt for dt, _ in series])
    ok = np.array([s >= 1.0 for _, s in series])
    for j in range(len(ts)):
        w = (ts >= ts[j]) & (ts < ts[j] + hold_s)
        if ok[w].all() and ts[j] + hold_s <= ts[-1]:
            return float(ts[j])
    return math.inf


@pytest.fixture(scope="module")
def runs():
    return {(scn, w): _simulate(ARMS if scn == "arms" else INZONE, scn == "inzone", w)
            for scn in ("arms", "inzone") for w in (False, True)}


def _report(runs, scn, changes):
    print("\nshare of the cells the change truly occupies that are NOT broadcast FREE, mean over 0-1 / 1-3 / 3-5 / "
          "5-15 min after it appears (1 - unsafe-free share):")
    for name, *_ in changes:
        tt, tn = runs[(scn, False)][0][name], runs[(scn, True)][0][name]
        print(f"  {name}: today {[round(_win(tt, a, b), 3) for a, b in WINDOWS]} | with the in-zone detector "
              f"{[round(_win(tn, a, b), 3) for a, b in WINDOWS]}, fully covered (60 s hold) from {_first_full(tn):.0f} s "
              f"(today {_first_full(tt):.0f} s)")
    print(f"  in-zone false alarms in the 40 min before the change: {len(runs[(scn, True)][1])}")
    for w in (False, True):
        ev = LEGACY_EVENTS[(scn == "inzone", w)]
        print(f"  change.py events ({'with' if w else 'without'} the in-zone detector), (s after the change, kind): "
              f"{[e for e in ev if e[1] != 'change_cleared'][:8]}")


def test_new_arm_0p5_to_2m_from_a_zone_becomes_unknown_within_5_min(runs):
    _report(runs, "arms", ARMS)
    cover, early = runs[("arms", True)]
    assert not early, f"in-zone alarms before any change: {early}"
    for name, *_ in ARMS:
        assert _first_full(cover[name]) <= 300, f"{name}: occupied cells still broadcast FREE after 5 min"
        assert _win(cover[name], 300, AFTER_S + 1) >= 0.99, f"{name}: occupied cells broadcast FREE after 5 min"


def test_program_change_and_newly_occupied_cell_inside_zone_margin_become_unknown(runs):
    """Fails on today's code: change.py skips zone + 1 m and only counts MOVING parts, and against a week of counts the
    map's own envelope takes hours to take the change in."""
    _report(runs, "inzone", INZONE)
    cover, early = runs[("inzone", True)]
    today = runs[("inzone", False)][0]
    assert not early, f"in-zone alarms before any change: {early}"
    for name, *_ in INZONE:
        assert _first_full(cover[name]) <= 300, f"{name}: occupied cells still broadcast FREE after 5 min"
        assert _win(cover[name], 300, AFTER_S + 1) >= 0.99, f"{name}: occupied cells broadcast FREE after 5 min"
        assert _win(today[name], 300, AFTER_S + 1) < 0.5, f"{name}: today's code was expected to leave it FREE (M-W4)"
