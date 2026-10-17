"""M-W4 on the body-shop code (observer/change_detect_l1.py, the adapter lane S8 / A-lite uses).

Two existing machines (HEM1, SEAL1) are learned for 40 min through the measured-error camera shim (32 % miss,
sigma 0.285 m, 15 % gross errors, blackouts), turned into Observation v2 by the S4 converter, with the existing
LayeredMap and the existing static-space detector (observer/change.py) running alongside. Then the floor changes and runs
15 min more. The observer's phase of the two existing machines is given (the ConditionTracker's job, not under test).

Scenario "arms_in" (M-W4, fails on today's code): new moving arms that sweep free floor INSIDE an existing zone, 0.5 and
1.0 m in from its edge, so that their detections stay inside zone + 1 m even with the measured noise (> 3 sigma from the
margin line). change.py never looks there and the map's week of counts absorbs them only after hours.
Scenario "arms_out" (regression, passes on today's code too): new arms whose sweep line is 0.5, 1.0, 1.5 and 2.0 m
OUTSIDE a zone edge (footprint d +- 0.4 m). At the measured sigma their detections spill past zone + 1 m and change.py
already catches them; the in-zone detector must not get in its way.
Scenario "inzone" (M-W4, fails on today's code): HEM1's program changes (its rollers run their tracks in reverse) and a
part is parked for good 0.5 m outside SEAL1's edge (a newly always-occupied cell).
Scenario "lift": "inzone" with a short half-life so that the forgetting absorbs the changes and UNKNOWN is lifted inside
the run: the changed cells must stay not FREE after the lift (audit S7a #1).

Measured per change: the share of its true 0.25 m footprint cells that the map no longer broadcasts FREE (inside the
machine's envelope for the current phase bin, inside a box of map.unknown_rects, or in an offline camera's region),
over time, with today's code (change.py only) and with the in-zone detector added. In every run with the detector the
broadcast invariant is checked: a cell the plain map broadcasts FREE is FREE in the detector's state and monitored.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from intercycle.observer import mapping as M
from intercycle.observer.change import ChangeDetector
from intercycle.observer.change_detect import ChangeConfig
from intercycle.observer.change_detect_l1 import L1_CFG, L1ChangeDetector
from intercycle.scenario import load

SCN = load()
IDS = [m.id for m in SCN.machines]
RECTS = [m.rect for m in SCN.machines]
SHARE = M.zone_camera_share(RECTS, SCN.cameras)
HI, SI = IDS.index("HEM1"), IDS.index("SEAL1")
P_HEM, P_SEAL = 45.0, 60.0
T0 = 30000.0
LEARN_S, AFTER_S = 40 * 60.0, 15 * 60.0
T_CH = T0 + LEARN_S
R_ARM = 0.4


def _arm(d_name, near, P, c, off, ax, amp=None):
    """A new arm: two circles (offset `off`) sweeping +-amp along `ax` (default 1.5 m on z, 2 m on x) for half of its
    cycle, parked at the rest pose for the other half."""
    amp = amp if amp is not None else (1.5 if ax == "z" else 2.0)

    def circles(t):
        u = ((t - T_CH) % P) / P
        f = math.sin(2 * math.pi * u / 0.5) if u < 0.5 else 0.0
        a = amp * f
        dx, dz = (0.0, a) if ax == "z" else (a, 0.0)
        return np.array([[c[0] + dx, c[1] + dz, R_ARM], [c[0] + off[0] + dx, c[1] + off[1] + dz, R_ARM]])
    return d_name, near, circles, P


# HEM1 rect (16, 37, 26, 45): rollers on z = 40 and 42 (r 0.5); SEAL1 rect (41, 37, 51, 45): an arm on an ellipse
# 3 x 2.5 m around (46, 41) (r 0.5, reaches x 49.5). The "inside" arms sweep floor those machines never use.
ARMS_IN = [
    _arm("HEM1_south_0.5m_inside", HI, 50.0, (20.6, 37.0 + 0.5), (0.8, 0.0), "x"),
    _arm("HEM1_north_1.0m_inside", HI, 55.0, (20.6, 45.0 - 1.0), (0.8, 0.0), "x"),
    _arm("SEAL1_east_0.5m_inside", SI, 42.0, (51.0 - 0.5, 40.6), (0.0, 0.8), "z"),
]

ARMS_OUT = [
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


def _state_fn(changes, program_change, until=None):
    """`until`: {change name: time it goes away} (default: it stays)."""
    until = until or {}

    def state(t):
        parts = {}
        parts["HEM1"] = _hem_rollers(t, reverse=program_change and t >= T_CH)
        v = ((t - T0) % P_SEAL) / P_SEAL                  # SEAL1: an arm sweeping inside its zone, 60 % active
        a = 2 * math.pi * min(v / 0.6, 1.0)
        parts["SEAL1"] = np.array([[46.0 + 3.0 * math.cos(a), 41.0 + 2.5 * math.sin(a), 0.5],
                                   [46.0 + 1.5 * math.cos(a), 41.0 + 1.2 * math.sin(a), 0.5]])
        if t >= T_CH:
            for name, _, circles, _ in changes:
                if name != "HEM1_program_change" and t < until.get(name, math.inf):
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
    """Scale the learned counts to a week of history (p, and the Markov null's ratios, unchanged): the map's L1 counts
    and the layers' undecayed counts x WEEK_FACTOR (no forgetting), the layers' decayed counts x WEEK_FACTOR capped at
    their saturation 1 / (1 - lam). Simulating the week tick by tick is out of reach for a unit test."""
    mp.l1_hits *= WEEK_FACTOR
    mp.l1_visits *= WEEK_FACTOR
    if det is None:
        return
    for L in det.layers:
        sat = 1.0 / (1.0 - L.lam)
        f = np.where(L.n > 0, np.minimum(WEEK_FACTOR, sat / np.maximum(L.n, 1e-9)), 1.0)
        for a in (L.n, L.k, L.kx, L.e):
            a *= f
        L.N *= WEEK_FACTOR
        L.K *= WEEK_FACTOR


def _cam_offline(mp, pts):
    """Cells in the region of a camera the map holds offline / bumped (UNKNOWN in MapView.unknown_mask)."""
    if not mp.cam_offline:
        return np.zeros(len(pts), bool)
    s = mp.spec["site"]
    ix = np.clip(np.floor((pts[:, 0] - s["x0"]) / s["res"]).astype(int), 0, mp.nx - 1)
    iz = np.clip(np.floor((pts[:, 1] - s["z0"]) / s["res"]).astype(int), 0, mp.nz - 1)
    return np.isin(mp.cam_of_cell[ix, iz], list(mp.cam_offline))


def _grid_centres(mp, i):
    res = mp.spec["env_res"]
    gx = mp.l1_x0[i] + (np.arange(mp.l1_nx[i]) + 0.5) * res
    gz = mp.l1_z0[i] + (np.arange(mp.l1_nz[i]) + 0.5) * res
    px, pz = np.meshgrid(gx, gz, indexing="ij")
    return np.stack([px.reshape(-1), pz.reshape(-1)], 1)


def _check_broadcast(mp, det):
    """The broadcast invariant: every UNKNOWN cell (dilated) of the detector is inside a box of map.unknown_rects, and in
    every phase bin the layer has learned, every cell the map broadcasts FREE (outside the bin's envelope and every
    UNKNOWN box) is monitored by the CUSUM. Returns the number of violating cells."""
    bad = 0
    for i, L in enumerate(det.layers):
        inbox = _in_rects(_grid_centres(mp, i), mp.unknown_rects)
        bad += int((L.unknown_mask() & ~inbox).sum())
        for b in range(M.N_BINS):
            if not L.trusted(b).all():
                continue
            free = ~mp.envelope(IDS[i], b).reshape(-1) & ~inbox
            bad += int((free & ~det.monitored(IDS[i], b).reshape(-1)).sum())
    return bad


LEGACY_EVENTS: dict = {}


def _simulate(changes, program_change: bool, with_inzone: bool, cfg: ChangeConfig | None = None,
              after_s: float = AFTER_S, key=None, until=None):
    cam = M.CameraShim(SCN, seed=7, **M.MEASURED_SHIM)  # pyright: ignore[reportArgumentType]  (mixed-type dict literal)
    conv = M.ObsV2Converter(SHARE, RECTS)
    mp = M.LayeredMap(SCN)
    legacy = ChangeDetector(mp)
    det = L1ChangeDetector(mp, cfg) if with_inzone else None
    until = until or {}
    state = _state_fn(changes, program_change, until)
    cover = {name: [] for name, *_ in changes}            # (t - T_CH, share of truly occupied cells not FREE)
    absorbed = {name: [] for name, *_ in changes}         # (t - T_CH, share of them with decayed p >= p_min)
    early, lifted = [], []
    aged = False
    bad = 0
    lev = LEGACY_EVENTS.setdefault(key or (program_change, with_inzone), [])
    for k in range(int((LEARN_S + after_s) * M.FRAME_HZ)):
        t = T0 + k / M.FRAME_HZ
        obs = conv(cam.observe(state, t))
        cond = _cond(obs.t_cap)
        if obs.t_cap >= T_CH and not aged:
            _age_history(mp, det)
            aged = True
        mp.update(obs, obs.t_cap, cond)
        lev += [(round(e["t"] - T_CH), e["kind"]) for e in legacy.step(obs)]
        if det is not None:
            ev = det.step(obs, cond)                      # after mp.update; pushes its boxes into mp.unknown_rects
            if obs.t_cap < T_CH:
                early += [e for e in ev if e["kind"] == "inzone_change"]
            else:
                lifted += [(round(e["t"] - T_CH), e["machine"]) for e in ev if e["kind"] == "inzone_cleared"]
            if obs.t_cap >= T_CH and k % 40 == 0:
                bad += _check_broadcast(mp, det)
        if obs.t_cap >= T_CH and k % 2 == 0:
            for name, i, circles, _ in changes:
                if obs.t_cap >= until.get(name, math.inf):
                    continue
                occ = _occupied(circles, obs.t_cap)
                b = int(cond.phase_bin[i])
                nf = (_in_envelope(mp, occ, i, b) | _in_rects(occ, mp.unknown_rects) | _cam_offline(mp, occ))
                cover[name].append((obs.t_cap - T_CH, float(nf.mean())))
                if det is not None:
                    res = mp.spec["env_res"]
                    pd = det.envelope_prob(IDS[i], b)
                    ix = np.clip(np.floor((occ[:, 0] - mp.l1_x0[i]) / res).astype(int), 0, pd.shape[0] - 1)
                    iz = np.clip(np.floor((occ[:, 1] - mp.l1_z0[i]) / res).astype(int), 0, pd.shape[1] - 1)
                    absorbed[name].append((obs.t_cap - T_CH, float((pd[ix, iz] >= det.cfg.p_min).mean())))
    return dict(cover=cover, absorbed=absorbed, early=early, lifted=lifted, bad=bad)


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


SCENARIOS = {"arms_in": (ARMS_IN, False), "arms_out": (ARMS_OUT, False), "inzone": (INZONE, True)}
LIFT_CFG = ChangeConfig(dilate=2, half_life_ticks=300.0, clear_ticks=600)
LIFT_AFTER_S = 25 * 60.0
PARK_GONE_S = 10 * 60.0                                  # the parked part is taken away 10 min after it appeared


@pytest.fixture(scope="module")
def runs():
    out = {(scn, w): _simulate(ch, prog, w, key=(scn, w)) for scn, (ch, prog) in SCENARIOS.items() for w in (False, True)}
    out[("lift", True)] = _simulate(INZONE, True, True, LIFT_CFG, LIFT_AFTER_S, key=("lift", True),
                                    until={"SEAL1_parked_part_0.5m": T_CH + PARK_GONE_S})
    return out


def _report(runs, scn, changes, today=True):
    print(f"\n[{scn}] share of the cells the change truly occupies that are NOT broadcast FREE, mean over 0-1 / 1-3 / "
          "3-5 / 5-15 min after it appears (1 - unsafe-free share):")
    for name, *_ in changes:
        tn = runs[(scn, True)]["cover"][name]
        line = (f"  {name}: with the in-zone detector {[round(_win(tn, a, b), 3) for a, b in WINDOWS]}, fully covered "
                f"(60 s hold) from {_first_full(tn):.0f} s")
        if today:
            tt = runs[(scn, False)]["cover"][name]
            line += f" | today {[round(_win(tt, a, b), 3) for a, b in WINDOWS]}, from {_first_full(tt):.0f} s"
        print(line)
    r = runs[(scn, True)]
    print(f"  in-zone false alarms in the 40 min before the change: {len(r['early'])}; broadcast-invariant violations "
          f"{r['bad']}; UNKNOWN lifted (s after the change, machine): {r['lifted'][:6]}")
    for w in ((False, True) if today else (True,)):
        ev = LEGACY_EVENTS[(scn, w)]
        print(f"  change.py events ({'with' if w else 'without'} the in-zone detector), (s after the change, kind): "
              f"{[e for e in ev if e[1] != 'change_cleared'][:8]}")


def test_new_arm_inside_a_zone_is_missed_today_and_unknown_within_5_min(runs):
    """Fails on today's code (M-W4): an arm sweeping free floor 0.5-1 m inside an existing zone stays FREE (change.py
    skips zone + 1 m; the map's week of counts absorbs it only after hours)."""
    _report(runs, "arms_in", ARMS_IN)
    r = runs[("arms_in", True)]
    today = runs[("arms_in", False)]["cover"]
    assert not r["early"], f"in-zone alarms before any change: {r['early']}"
    assert r["bad"] == 0, "a cell the map broadcasts FREE is held by the detector"
    for name, *_ in ARMS_IN:
        assert _first_full(today[name]) > 300 and _win(today[name], 300, AFTER_S + 1) < 0.5, \
            f"{name}: today's code was expected to leave it FREE (M-W4)"
        assert _first_full(r["cover"][name]) <= 300, f"{name}: occupied cells still broadcast FREE after 5 min"
        assert _win(r["cover"][name], 300, AFTER_S + 1) >= 0.99, f"{name}: occupied cells broadcast FREE after 5 min"


def test_new_arm_0p5_to_2m_outside_a_zone_is_caught_today_and_with_the_detector(runs):
    """Regression, passes on today's code too: at the measured sigma (0.285 m) these arms' detections spill past
    zone + 1 m and change.py catches them. The in-zone detector must not delay that."""
    _report(runs, "arms_out", ARMS_OUT)
    r = runs[("arms_out", True)]
    today = runs[("arms_out", False)]["cover"]
    assert not r["early"], f"in-zone alarms before any change: {r['early']}"
    assert r["bad"] == 0
    for name, *_ in ARMS_OUT:
        for c in (today[name], r["cover"][name]):
            assert _first_full(c) <= 300, f"{name}: occupied cells still broadcast FREE after 5 min"
            assert _win(c, 300, AFTER_S + 1) >= 0.99, f"{name}: occupied cells broadcast FREE after 5 min"
        assert _first_full(r["cover"][name]) <= _first_full(today[name]) + 10, f"{name}: the detector delayed change.py"


def test_program_change_and_newly_occupied_cell_inside_zone_margin_become_unknown(runs):
    """Fails on today's code: change.py skips zone + 1 m and only counts MOVING parts, and against a week of counts the
    map's own envelope takes hours to take the change in."""
    _report(runs, "inzone", INZONE)
    r = runs[("inzone", True)]
    today = runs[("inzone", False)]["cover"]
    assert not r["early"], f"in-zone alarms before any change: {r['early']}"
    assert r["bad"] == 0
    for name, *_ in INZONE:
        assert _first_full(r["cover"][name]) <= 300, f"{name}: occupied cells still broadcast FREE after 5 min"
        assert _win(r["cover"][name], 300, AFTER_S + 1) >= 0.99, f"{name}: occupied cells broadcast FREE after 5 min"
        assert _win(today[name], 300, AFTER_S + 1) < 0.5, f"{name}: today's code was expected to leave it FREE (M-W4)"


def test_decayed_counts_never_lift_a_change_the_map_still_broadcasts_free(runs):
    """Audit S7a #1 on the body-shop map. With a short half-life (300 observed ticks per bin) the decayed counts absorb
    HEM1's program change within minutes, while the map's undecayed week of counts still broadcasts those cells FREE.
    The old rule lifted UNKNOWN on the decayed counts (then the cells were broadcast FREE for hours); now the lift is
    judged on the map's broadcast, so the change stays covered to the end. The parked part is taken away after 10 min:
    then its UNKNOWN is lifted (availability comes back) once its cells show no hit for clear_ticks."""
    _report(runs, "lift", INZONE, today=False)
    r = runs[("lift", True)]
    assert not r["early"] and r["bad"] == 0
    name = "HEM1_program_change"
    ab = r["absorbed"][name]
    print(f"  {name}: share of its occupied cells with decayed p >= p_min, mean over the last 5 min: "
          f"{_win(ab, LIFT_AFTER_S - 300, LIFT_AFTER_S + 1):.2f}")
    assert _win(ab, LIFT_AFTER_S - 300, LIFT_AFTER_S + 1) >= 0.9, "scenario check: the decayed counts absorbed it"
    c = r["cover"][name]
    t_full = _first_full(c)
    # The 300-tick half-life is far shorter than the default (172800 / 17 per bin): the null learns the new program
    # within minutes and slows the CUSUM, so full cover comes later than in the "inzone" run (229 s). What this test
    # pins is that, once covered, the change is never exposed again although its decayed p is above p_min.
    assert t_full <= LIFT_AFTER_S - 600, f"{name}: never fully covered"
    assert min(s for dt, s in c if dt >= t_full) >= 0.99, f"{name}: occupied cells broadcast FREE after the decay"
    park = "SEAL1_parked_part_0.5m"
    assert _first_full(r["cover"][park], hold_s=30.0) <= 300
    gone = [dt for dt, m in r["lifted"] if m == "SEAL1" and dt >= PARK_GONE_S]
    assert gone and gone[0] <= PARK_GONE_S + LIFT_CFG.clear_ticks / M.FRAME_HZ + 120, \
        f"the parked part's UNKNOWN was not lifted after it went away (lifts {r['lifted'][:8]})"


# ---------------------------------------------------------------------------------------------- adapter contract
def _part_obs(t, pts, machine, cam_ok, cam_floor_ok=None, zone_vis=None):
    n = len(pts)
    cf = np.asarray(cam_ok, bool).copy() if cam_floor_ok is None else np.asarray(cam_floor_ok, bool)
    zv = np.ones(len(IDS), np.float32) if zone_vis is None else np.asarray(zone_vis, np.float32)
    return M.Observation(t=t, t_cap=t, cls=np.full(n, M.PART, np.int8), x=np.array([p[0] for p in pts], float),
                         z=np.array([p[1] for p in pts], float), r=np.full(n, 0.4), heading=np.full(n, np.nan),
                         moving=np.ones(n, bool), machine=np.full(n, machine, np.int16), cam_ok=np.asarray(cam_ok, bool),
                         zone_motion=np.zeros(len(IDS), np.float32), zone_vis=zv, cam_floor_ok=cf)


def _cond_bins(b):
    pb = np.full(len(IDS), -1, np.int16)
    pb[HI] = pb[SI] = b
    nan = np.full(len(IDS), np.nan)
    return M.Cond(t=0.0, tod_bin=0, prod_state=0, phase_bin=pb, phase=nan, period=nan, busy=nan, jitter=nan)


def _cam_of(mp, x, z):
    s = mp.spec["site"]
    return int(mp.cam_of_cell[int((x - s["x0"]) // s["res"]), int((z - s["z0"]) // s["res"])])


def test_adapter_hit_rule_matches_the_map_l1_update():
    """Audit S7a #4: detections located in an offline (bumped) or unhealthy camera's cells are dropped by
    LayeredMap.update (_drop_offline) and must be dropped by the adapter too, and the rest rasterised the same way.
    The oracle is the map's own L1 raster of what `_drop_offline` keeps (fed to a fresh map with every camera healthy):
    whether the map then also skips the whole L1 tick of a machine whose grid a lost camera touches (lane S4's
    per-grid rule, lane/S4-fix1) is its visit accounting, not the hit rule. The adapter instead marks only the lost
    camera's cells unobserved (test_adapter_observed_comes_from_observation_v2), so it keeps learning and monitoring
    the healthy part of the grid."""
    mp = M.LayeredMap(SCN)
    det = L1ChangeDetector(mp)
    pts = [(16.5, 38.0), (19.0, 44.0), (24.0, 38.5), (25.6, 43.2)]          # HEM1 grid, both sides of a camera seam
    cams = {_cam_of(mp, x, z) for x, z in pts}
    assert len(cams) >= 2, "scenario check: the points must span two cameras"
    home = int(mp.cam_of_machine[HI])
    other = next(c for c in cams if c != home)
    ok = np.ones(len(SCN.cameras), bool)
    for case in ("plain", "bumped", "unhealthy"):
        mp.cam_offline = {other} if case == "bumped" else set()
        cf = ok.copy()
        if case == "unhealthy":
            cf[other] = False
        obs = _part_obs(1.0, pts, HI, ok, cf)
        mp.update(obs, 1.0, _cond_bins(3))               # sets the map's live cam_unhealthy for _drop_offline
        kept = mp._drop_offline(obs)                     # pyright: ignore[reportPrivateUsage]  (the rule under test)
        ref = M.LayeredMap(SCN)
        ref.update(_part_obs(1.0, list(zip(kept.x, kept.z)), HI, ok), 1.0, _cond_bins(3))
        want = ref.l1_view(HI)[3].reshape(-1) > 0
        got = det.evidence_hits(obs, HI)
        assert want.any()
        assert np.array_equal(got, want), f"{case}: adapter hits differ from the map's L1 increments"
        if case != "plain":
            assert want.sum() < np.asarray(det.evidence_hits(_part_obs(1.0, pts, HI, ok), HI, gone=()), bool).sum()


def test_adapter_observed_comes_from_observation_v2():
    """Audit S7a #5: the adapter never assumes the whole grid is observed. A v1 observation is refused (no visibility
    information: fail closed), a zone that is not fully visible (zone_vis < 1: occluded, camera unhealthy) is not
    observed at all (no learning, no decay, no lift), and the cells of a bumped camera are not observed."""
    mp = M.LayeredMap(SCN)
    det = L1ChangeDetector(mp)
    ok = np.ones(len(SCN.cameras), bool)
    v1 = M.Observation(t=0.0, t_cap=0.0, cls=np.zeros(0, np.int8), x=np.zeros(0), z=np.zeros(0), r=np.zeros(0),
                       heading=np.zeros(0), moving=np.zeros(0, bool), machine=np.zeros(0, np.int16), cam_ok=ok)
    mp.update(v1, 0.0, _cond_bins(0))
    with pytest.raises(ValueError):
        det.step(v1, _cond_bins(0))
    L = det.layers[HI]
    zv = np.ones(len(IDS), np.float32)
    zv[HI] = 0.97
    obs = _part_obs(1.0, [], HI, ok, zone_vis=zv)
    mp.update(obs, 1.0, _cond_bins(0))
    det.step(obs, _cond_bins(0))
    assert L.n.sum() == 0 and L.N.sum() == 0, "a partly hidden zone was counted as observed"
    assert det.layers[SI].n[0].min() == 1.0, "a fully visible zone is observed"
    pts_other = (16.5, 38.0)
    other = _cam_of(mp, *pts_other)
    assert other != int(mp.cam_of_machine[HI])
    mp.cam_offline = {other}
    obs = _part_obs(2.0, [], HI, ok)
    mp.update(obs, 2.0, _cond_bins(0))
    det.step(obs, _cond_bins(0))
    seen = L.n[0] > 0
    cam_cell = det.cam_of_l1_cell[HI]
    assert not seen[cam_cell == other].any(), "cells of a bumped camera were counted as observed"
    assert seen[cam_cell == int(mp.cam_of_machine[HI])].all()


def test_adapter_step_requires_the_map_update_of_the_same_tick():
    """sync() compares with the map's envelope, so it must see this tick's counts: step before update is refused."""
    mp = M.LayeredMap(SCN)
    det = L1ChangeDetector(mp, L1_CFG)
    obs = _part_obs(1.0, [], HI, np.ones(len(SCN.cameras), bool))
    with pytest.raises(RuntimeError):
        det.step(obs, _cond_bins(0))
    mp.update(obs, 1.0, _cond_bins(0))
    det.step(obs, _cond_bins(0))
    with pytest.raises(RuntimeError):
        det.step(obs, _cond_bins(0))


def test_sync_keeps_other_owners_boxes_even_when_identical():
    mp = M.LayeredMap(SCN)
    det = L1ChangeDetector(mp)
    other = (1.0, 2.0, 3.0, 4.0)
    mp.unknown_rects = [other, other]          # one is the detector's own box from last tick, one another owner's
    det._owned = [other]
    det.sync()
    assert mp.unknown_rects.count(other) == 1, "another owner's identical box was removed"
    assert all(r in mp.unknown_rects for r in det._owned)
