"""A-lite minimal W-W2 (PLAN_TWIN §7 item 2): 2 programs per station from a seeded model mix, tip-dress every
N in [8, 30] cycles, +-3 cm path noise."""
import numpy as np
import pytest

from intercycle.alite import programs as AP
from intercycle.alite.world import KeyedWorld
from intercycle.scenario import load
from intercycle.sim.world import World
from intercycle.truth import kinematics as K

SCN = load()
T0, T1 = 7 * 3600.0, 9 * 3600.0
SEED = 801


@pytest.fixture(scope="module")
def kin():
    return AP.ProgramKinematics(SCN, KeyedWorld(SCN, SEED, T0, T1))


def _cycles(kin, mid, t0=T0, t1=T1):
    """(window, program, dress) of the windows starting in [t0, t1)."""
    r = kin._byid[mid]
    pg = next(p for p in kin._progs if p.r is r)
    return [(i, int(pg.prog[i]), bool(pg.dress[i])) for i, b in enumerate(r.bs) if t0 <= b < t1], r


def _tool(kin, mid, t):
    return np.asarray(kin.bodies(t)[mid]).reshape(-1, K.N_ARM_C, 3)[:, 4, :2]


def test_needs_a_keyed_world():
    with pytest.raises(TypeError):
        AP.ProgramKinematics(SCN, World(SCN, 3, T0, T1))  # pyright: ignore[reportArgumentType]  (the refusal under test)


def test_two_programs_follow_the_model_mix(kin):
    for mid in ("S10", "S60", "HEM1"):
        cyc, _ = _cycles(kin, mid)
        progs = np.array([p for _, p, _ in cyc])
        assert set(progs) == {0, 1}, mid
        assert 0.15 <= progs.mean() <= 0.45, (mid, progs.mean())


def test_the_body_carries_its_model_down_the_line(kin):
    """The body at station j in line cycle k was fed at cycle k - j: S20 runs S10's program one cycle later."""
    a, _ = _cycles(kin, "S10")
    b, _ = _cycles(kin, "S20")
    pa = {i: p for i, p, _ in a}
    pb = {i: p for i, p, _ in b}
    common = [i for i in pb if i - 1 in pa and i > 10]
    assert len(common) > 50
    assert all(pb[i] == pa[i - 1] for i in common)


def test_tip_dress_every_n_cycles(kin):
    for mid, N in kin.dress_every.items():
        assert AP.DRESS_EVERY[0] <= N <= AP.DRESS_EVERY[1]
        cyc, _ = _cycles(kin, mid)
        d = np.array([dr for _, _, dr in cyc])
        idx = np.flatnonzero(d)
        assert len(idx) >= 2 and set(np.diff(idx)) <= {N}, mid
    assert len(set(kin.dress_every.values())) > 1          # stations differ


def test_tool_reaches_the_dresser_only_on_dress_cycles(kin):
    mid = "S30"
    dress_pt = np.array([[m for m in kin.dressers if m[0] == mid][k][1:] for k in range(2)])
    cyc, r = _cycles(kin, mid)
    seen = {True: [], False: []}
    for i, _, dr in cyc[5:60]:
        b, s = r.bs[i], r.is_[i]
        t = b + 0.89 * (s - b)                              # middle of the dress phase of the active time
        if kin.cycle_program(mid, t) is None or kin.phase(mid, t)[0] != "active":
            continue
        d = np.hypot(*(_tool(kin, mid, t) - dress_pt).T).min()
        seen[dr].append(d)
    assert seen[True] and seen[False]
    assert max(seen[True]) < 0.1 and min(seen[False]) > 1.0, seen


def test_path_noise_is_within_3_cm_and_varies_per_cycle(kin):
    mid = "S10"
    cyc, r = _cycles(kin, mid)
    same = [(i, p, dr) for i, p, dr in cyc if p == 0 and not dr][:20]
    tools = []
    for i, _, _ in same:
        b, s = r.bs[i], r.is_[i]
        tools.append(_tool(kin, mid, b + 0.3 * (s - b)))
    tools = np.array(tools)
    spread = np.abs(tools - tools.mean(0)).max()
    assert 0.005 < spread <= 2 * AP.PATH_NOISE_M + 1e-9
    bases = np.asarray(kin.bodies(r.bs[same[0][0]] + 1.0)[mid]).reshape(-1, K.N_ARM_C, 3)[:, 0, :2]
    x = SCN.machine(mid).x
    assert np.allclose(bases, [[x - K.BASE_DX, K.LINE_Z - K.BASE_DZ], [x + K.BASE_DX, K.LINE_Z + K.BASE_DZ]])


def test_paths_depend_on_the_world_seed():
    a = AP.ProgramKinematics(SCN, KeyedWorld(SCN, SEED, T0, T0 + 600))
    b = AP.ProgramKinematics(SCN, KeyedWorld(SCN, SEED + 1, T0, T0 + 600))
    ta = a._progs[0].tabs[0][0]
    tb = b._progs[0].tabs[0][0]
    assert not np.allclose(ta, tb)
    c = AP.ProgramKinematics(SCN, KeyedWorld(SCN, SEED, T0, T0 + 600))
    assert np.array_equal(c._progs[0].tabs[0][0], ta)


def test_program_change_takes_over_from_t(kin):
    t_ch = T0 + 1800.0
    k2 = AP.ProgramKinematics(SCN, KeyedWorld(SCN, SEED, T0, T1), program_change={"S60": t_ch})
    cyc, r = _cycles(k2, "S60")
    after = [p for i, p, _ in cyc if r.bs[i] >= t_ch]
    before = [p for i, p, _ in cyc if r.bs[i] < t_ch]
    assert set(after) == {2} and 2 not in before
    t = next(r.bs[i] + 0.4 * (r.is_[i] - r.bs[i]) for i, _, _ in cyc if r.bs[i] >= t_ch + 120)
    assert not np.allclose(_tool(k2, "S60", t), _tool(kin, "S60", t))


def test_stowed_pose_is_the_same_for_every_program(kin):
    mid = "S10"
    cyc, r = _cycles(kin, mid)
    poses = []
    for i, _, _ in cyc[3:30]:
        t = r.is_[i] + 0.5 * (r.ie[i] - r.is_[i])         # idle part of the window
        if kin.phase(mid, t)[0] == "idle":
            poses.append(np.asarray(kin.bodies(t)[mid]).copy())
    assert len(poses) > 5 and all(np.array_equal(p, poses[0]) for p in poses)
