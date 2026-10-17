"""Ground-truth machine kinematics: a collision body for every moving part, as a function of cycle phase.

The learner must never import this module (it lives in `truth/`). It replaces the crude `_arm_circles`
placeholder in sim/physics.py and is what the observer later has to learn as phase-indexed envelopes (layer L1).

Everything is driven by `World`: the cycle windows (machine time) and the freezes. Machine time of a clock
group (the whole line, or one cell) does not advance while the group is frozen, so a TRIP leaves every part
frozen mid-motion by construction. A HOLD means the machine was about to start: arms are shown at home.

Model (plan view, metres; (x, z) with the line along x at z = 24)
  * Arms ("weld robots", also hemming / sealing robots): 2-link planar chain base -> elbow -> tool, drawn as 5
    circles (base, upper-arm mid, elbow, forearm mid, tool) so the chain has no gaps. IDLE = stowed over the base.
    ACTIVE fraction f (0..1 of the active time) for weld / respot / framing stations:
        0.00-0.80  home -> weld point 1 -> ... -> weld point n (move, then dwell at each point), over the
                   car body on the conveyor (|z-24| <= 1.2), sweeping in x within the station
        0.80-0.86  swing out (past the base) to the tip dresser at the aisle side of the base (|z-24| = 4.2)
        0.86-0.92  dress (dwell)           -> the envelope reaches within ~0.8 m of the zone boundary
        0.92-1.00  return home
    HEM1 (roller hemming): both robots take the roller head round the fixture perimeter (3.0 x 1.6 m table at the
    machine centre, 0.25 m clearance), 180 degrees apart. SEAL1: each robot traces a serpentine bead over its half
    of the fixture. Both: approach 0-8 %, path 8-92 %, return 92-100 %.
    Poses are looked up in tables precomputed at construction (`N_TAB` samples of f), so a call is a gather.
  * SKIDS: car bodies 4.2 x 1.9 m (3 covering circles each) on the conveyor. At every line index (first
    `TRANSFER_S` = 8 s of the cycle of the line's first station) every body moves to the next station slot (cubic
    ease; +9 m, the last hop S70 -> S80 is 12 m so the body ends up centred under the lifter). A new body is fed in
    at the conveyor head (it pops in at x = 8.6 at the start of the transfer). The S80 body disappears when the
    lifter picks it (f = 0.45). At rest no body is inside the walkway gap x = 52.5 +- 1.
  * Lifter (S80): carriage 4.6 x 2.6 m centred on the station. Height 6 m (overhead) -> 0.35 m during f 0.10-0.30,
    stays down 0.30-0.55 (picks the body at 0.45), back up 0.55-0.80. It is a floor hazard (circles) only below
    `H_CLEAR` = 2.2 m.
  * AGVDOCK: every line station has a docking slot at (x, 30.6), north of its zone. An AGV (1.2 x 0.8 m, 2 circles)
    parks there for 20 s starting `DOCK_T0` = 14 s after the station's window start, on every 6th cycle
    ((cycle index + station index) % 6 == 0). It is driven by machine time (a frozen line freezes the schedule).

Performance (budget 100 us for all machines): `all_circles` ~ 55 us, `bodies` ~ 40 us at distinct t, `phase` ~ 6 us;
the result for a repeated t
(physics calls it several times per step) is cached. Returned arrays are read-only views: copy before mutating.
"""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from intercycle.robot.sitegraph import CROSS_X, TRANSFER_S
from intercycle.scenario import Machine, Scenario
from intercycle.sim.world import World

LINE_Z = 24.0
CONV_HALF_W = 1.2                       # conveyor footprint |z - 24| <= 1.2 (geometry.CONV_W = 2.4)
LINE_X = (6.0, 86.0)                    # conveyor extent in x (geometry.LINE_X)
WALK_RECT = (CROSS_X - 1.0, LINE_Z - CONV_HALF_W, CROSS_X + 1.0, LINE_Z + CONV_HALF_W)   # same as sim/physics.py

# ---- arms
BASE_DX, BASE_DZ = 1.8, 3.3             # line station robot bases at (x -+ 1.8, 24 -+ 3.3)
CELL_DX, CELL_DZ = 1.8, 2.0             # cell robot bases at (x -+ 1.8, z -+ 2.0) (as sim/geometry.py)
DRESS_DZ = 4.2                          # tip dresser at |z - 24| = 4.2, i.e. 0.9 m aisle-side of the base
LINK = 2.6                              # both links; reach 5.2
REACH_MARGIN = 0.97                     # every commanded tool point must be within 97 % of the reach
ARM_R = (0.65, 0.50, 0.50, 0.50, 0.50)  # base, upper-arm mid, elbow, forearm mid, tool (links 1.3 m apart)
N_ARM_C = 5
N_TAB = 4096
T_WELD_END, T_DRESS_END = 0.80, 0.92    # sub-phase boundaries of the active fraction (weld stations)
FOLD_D0, FOLD_D1 = 0.4, 2.4             # tool-base distances over which the elbow unfolds from the midpoint
KIND = {                                # n weld points, tool speed (relative), dwell per point (relative)
    "weld_station": (6, 2.5, 0.5),
    "respot": (8, 2.5, 0.3),
    "framing": (4, 2.0, 1.2),
}
FIXTURE = (3.0, 1.6)                    # cell fixture table
LOOP_PAD = 0.25                         # hemming roller path clearance around the table

# ---- skids
SKID_L, SKID_W = 4.2, 1.9
SKID_REL_X = (-1.4, 0.0, 1.4)           # 3 circles covering the 4.2 x 1.9 body
SKID_R = math.hypot(SKID_W / 2, SKID_L / 6)
X_FEED = LINE_X[0] + 2.6                # new bodies are fed in at the conveyor head
NS = 2048

# ---- lifter
LIFT_L, LIFT_W = 4.6, 2.6
LIFT_REL_X = (-1.533, 0.0, 1.533)
LIFT_R = math.hypot(LIFT_W / 2, LIFT_L / 6)
LIFT_H_TOP, LIFT_H_LOW, H_CLEAR = 6.0, 0.35, 2.2
F_PICK = 0.45

# ---- AGV dock
DOCK_Z, DOCK_EVERY, DOCK_S, DOCK_T0 = 30.6, 6, 20.0, 14.0
DOCK_DX = 2.6          # dock sits east of the station centre, clear of the robot's wait point at (x, 30.2)
AGV_REL_X, AGV_R = (-0.3, 0.3), 0.5

_EMPTY = np.zeros((0, 3))
_EMPTY.flags.writeable = False


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ss1(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def lift_height(f: float) -> float:
    """Lifter carriage height (m) at active fraction f."""
    if f < 0.10:
        return LIFT_H_TOP
    if f < 0.30:
        return LIFT_H_TOP + (LIFT_H_LOW - LIFT_H_TOP) * _ss1((f - 0.10) / 0.20)
    if f < 0.55:
        return LIFT_H_LOW
    if f < 0.80:
        return LIFT_H_LOW + (LIFT_H_TOP - LIFT_H_LOW) * _ss1((f - 0.55) / 0.25)
    return LIFT_H_TOP


# ------------------------------------------------------------------------------------------ freeze-aware clock
class _Clock:
    """Machine time of one group + the active freeze, in O(#recent freezes) instead of O(#freezes).

    Equivalent to `Group.mt(t)` / `Group.is_frozen(t)`. Freezes that ended more than 2 s before the query are summed
    once and cached; if time goes backwards (or the freeze list shrinks) the cache is bypassed / reset. While the
    freeze list is sorted by start time (it is, in a run) the scan stops at the first freeze that starts after t.
    """
    __slots__ = ("g", "n", "base", "hi", "seen", "ok", "sig", "_ak", "_av")

    def __init__(self, g):
        self.g, self.n, self.base, self.hi = g, 0, 0.0, -math.inf
        self.seen, self.ok, self.sig = 0, True, (0,)
        self._ak, self._av = None, 0.0

    def changed(self) -> bool:
        """Has the freeze list changed since the last query? (count and the last freeze, which physics extends)."""
        fz = self.g.freezes
        if not fz:
            return self.sig != (0,)
        f = fz[-1]
        return self.sig != (len(fz), f.t0, f.t1, f.kind)

    def query(self, t: float):
        """(machine time, freeze kind or None, fraction elapsed in that freeze)."""
        fz = self.g.freezes
        n = len(fz)
        if not n:
            self.sig = (0,)
            return t, None, 0.0
        f = fz[-1]
        self.sig = (n, f.t0, f.t1, f.kind)
        if n < self.n or n < self.seen:
            self.n, self.base, self.hi, self.seen, self.ok = 0, 0.0, -math.inf, 0, True
        if self.ok and self.seen < n:
            j = self.seen
            last = fz[j - 1].t0 if j else -math.inf
            while j < n:
                a = fz[j].t0
                if a < last:
                    self.ok = False
                    break
                last = a
                j += 1
            self.seen = j
        if t < self.hi:                       # time went back past an absorbed freeze: no cache
            start, fro = 0, 0.0
        else:
            k, cut = self.n, t - 2.0
            while k < n:
                f = fz[k]
                if f.t1 < cut:
                    d = f.t1 - f.t0
                    if d > 0.0:
                        self.base += d
                    if f.t1 > self.hi:
                        self.hi = f.t1
                    k += 1
                else:
                    break
            self.n = k
            start, fro = k, self.base
        kind, k0, k1 = None, 0.0, 0.0
        srt = self.ok
        for i in range(start, n):
            f = fz[i]
            a = f.t0
            if a <= t:
                b = f.t1
                if t < b:
                    if kind is None or (f.kind == "trip" and kind != "trip"):
                        kind, k0, k1 = f.kind, a, b
                    d = t - a
                else:
                    d = b - a
                if d > 0.0:
                    fro += d
            elif srt:
                break
        fr = (t - k0) / (k1 - k0) if kind is not None and k1 > k0 else 0.0
        return t - fro + self._before_anchor(t), kind, fr

    def _before_anchor(self, t: float) -> float:
        """Frozen time before the latest planned-stop start <= t: not carried past it (`Group.mt`)."""
        anc = getattr(self.g, "anchors", None)
        if not anc:
            return 0.0
        a = self.g.anchor(t)
        if a == -math.inf:
            return 0.0
        key = (a, self.sig)
        if getattr(self, "_ak", None) != key:
            self._ak, self._av = key, self.g.frozen_before(a)
        return self._av


# ------------------------------------------------------------------------------------------ arm tables
class _ArmSpec:
    __slots__ = ("mid", "base", "q", "brk", "mf", "r_tool", "sgn")

    def __init__(self, mid, base, q, brk, mf, r_tool, sgn=-1.0):
        self.mid, self.base, self.q, self.brk, self.mf, self.r_tool, self.sgn = mid, base, q, brk, mf, r_tool, sgn


def _timeline_weld(base, pts, dress, v, dwell):
    """Home -> weld points (move, dwell) -> dresser (move, dress) -> home."""
    q = np.vstack([base, pts, dress, base])
    d = np.hypot(*np.diff(q, axis=0).T)
    n = len(pts)
    tm = d[:n] / v
    w = (tm + dwell)
    w = w / w.sum() * T_WELD_END
    mf = np.concatenate([tm / (tm + dwell), [0.5, 0.75]])
    w = np.concatenate([w, [T_DRESS_END - T_WELD_END, 1.0 - T_DRESS_END]])
    return q, np.concatenate([[0.0], np.cumsum(w)]), mf


def _timeline_path(base, pts):
    q = np.vstack([base, pts, base])
    d = np.hypot(*np.diff(q, axis=0).T)
    n = len(pts)
    w = np.maximum(d[:n], 0.3)
    w = w / w.sum() * 0.92
    mf = np.concatenate([np.ones(n), [0.75]])
    w = np.concatenate([w, [0.08]])
    return q, np.concatenate([[0.0], np.cumsum(w)]), mf


def _weld_arms(m: Machine) -> list[_ArmSpec]:
    n, v, dwell = KIND[m.kind]
    out = []
    for r, side in enumerate((-1.0, 1.0)):
        rng = np.random.default_rng([7, sum(map(ord, m.id)), r])
        base = np.array([m.x + side * BASE_DX, LINE_Z + side * BASE_DZ])
        sk = np.arange(n) / (n - 1)
        xs = (-1.8 + 3.6 * (1.0 - np.abs(2.0 * sk - 1.0))) * (-side) + rng.uniform(-0.15, 0.15, n)   # out and back
        dz = np.clip(np.where(sk < 0.5, 0.35, 0.9) + rng.uniform(-0.12, 0.12, n), 0.2, 1.1)         # two rows
        pts = np.stack([m.x + xs, LINE_Z + side * dz], axis=1)
        dress = np.array([base[0], LINE_Z + side * DRESS_DZ])
        q, brk, mf = _timeline_weld(base, pts, dress, v, dwell)
        out.append(_ArmSpec(m.id, base, q, brk, mf, ARM_R[4]))
    return out


def _cell_arms(m: Machine) -> list[_ArmSpec]:
    out = []
    fw, fd = FIXTURE
    for r, side in enumerate((-1.0, 1.0)):
        base = np.array([m.x + side * CELL_DX, m.z + side * CELL_DZ])
        if m.kind == "roller_hemming":
            hx, hz = fw / 2 + LOOP_PAD, fd / 2 + LOOP_PAD
            corners = np.array([(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]) * (-side)
            pts = np.vstack([corners, corners[:1]]) + [m.x, m.z]
            r_tool = 0.50
        else:                                                    # sealer: serpentine bead over this robot's half
            xa, xb = (-1.3, -0.15) if side < 0 else (1.3, 0.15)
            zs = (-0.55, 0.0, 0.55) if side < 0 else (0.55, 0.0, -0.55)
            pts = np.array([(xa, zs[0]), (xb, zs[0]), (xb, zs[1]), (xa, zs[1]), (xa, zs[2]), (xb, zs[2])]) + [m.x, m.z]
            r_tool = 0.45
        q, brk, mf = _timeline_path(base, pts)
        out.append(_ArmSpec(m.id, base, q, brk, mf, r_tool, sgn=1.0))
    return out


def _chain_circles(base, tool, radii, sgn):
    """2-link planar IK, vectorised. base (A,1,2), tool (A,M,2), radii (A,1,5) -> circles (A,M,5,3).

    Handedness `sgn` (A,1) is fixed per arm: line robots bend their elbow toward +x (low-x base) when the tool is across
    the conveyor; cell robots bend the other way so the elbow stays over the fixture. The second robot of a station
    is the first rotated by 180 degrees, so it has the same sgn. For a short tool-base distance the elbow folds onto
    the midpoint (a stowed arm is a single stack of circles over the base) so the chain moves continuously.
    """
    L1 = L2 = LINK
    dv = tool - base
    d = np.hypot(dv[..., 0], dv[..., 1])
    assert (d <= (L1 + L2) * REACH_MARGIN + 1e-9).all(), "tool point out of reach"
    dc = np.maximum(d, 1e-9)
    a = dc / 2.0                                              # equal links
    h = np.sqrt(np.maximum(L1 * L1 - a * a, 0.0))
    ux, uz = dv[..., 0] / dc, dv[..., 1] / dc
    mid = base + dv / 2.0
    ik = np.stack([base[..., 0] + a * ux - sgn * h * uz, base[..., 1] + a * uz + sgn * h * ux], axis=-1)
    w = _ss((d - FOLD_D0) / (FOLD_D1 - FOLD_D0))[..., None]
    elbow = mid + w * (ik - mid)
    b = np.broadcast_to(base, tool.shape)
    pts = np.stack([b, (b + elbow) / 2, elbow, (elbow + tool) / 2, tool], axis=-2)        # (A,M,5,2)
    rr = np.broadcast_to(radii, pts.shape[:-1])[..., None]
    return np.concatenate([pts, rr], axis=-1)


def _build_arm_table(specs: list[_ArmSpec]) -> np.ndarray:
    """(A, N_TAB + 1, 5, 3) circles by active fraction; the last sample is the stowed (home) pose."""
    A = len(specs)
    S = max(len(s.q) - 1 for s in specs)
    Q = np.empty((A, S + 1, 2))
    brk = np.ones((A, S + 1))
    mf = np.ones((A, S))
    base = np.empty((A, 1, 2))
    radii = np.empty((A, 1, N_ARM_C))
    sgn = np.array([s.sgn for s in specs])[:, None]
    for a, s in enumerate(specs):
        k = len(s.q)
        Q[a, :k] = s.q
        Q[a, k:] = s.base
        brk[a, :k] = s.brk
        brk[a, k - 1] = 1.0
        mf[a, :k - 1] = s.mf
        base[a, 0] = s.base
        radii[a, 0] = (*ARM_R[:4], s.r_tool)
    f = np.arange(N_TAB) / N_TAB
    seg = (f[None, :, None] >= brk[:, None, 1:]).sum(-1)                      # (A,N)
    ar = np.arange(A)[:, None]
    b0, b1 = brk[ar, seg], brk[ar, seg + 1]
    u = (f[None, :] - b0) / (b1 - b0)
    s_ = _ss(u / mf[ar, seg])
    p0, p1 = Q[ar, seg], Q[ar, seg + 1]
    tool = p0 + (p1 - p0) * s_[..., None]
    tool = np.concatenate([tool, base], axis=1)                              # last sample: home
    tab = _chain_circles(base, tool, radii, sgn)
    tab.flags.writeable = False
    return tab


# ------------------------------------------------------------------------------------------ the class
class _Mach:
    """Per-machine lookup record (window times as plain lists for bisect)."""
    __slots__ = ("m", "k", "gi", "bs", "is_", "ie", "tab", "n_arm")


class _State:
    __slots__ = ("t", "skids", "agv", "views", "mt", "kind", "mstate", "mfrac", "midx", "lift_h", "transfer",
                 "docked", "flat", "bodies")


class Kinematics:
    CIRCLES_PER_ARM = N_ARM_C

    def __init__(self, scn: Scenario, world: World):
        self.scn, self.world = scn, world
        self.machine_ids = [m.id for m in scn.machines]
        gnames = list(world.groups)
        self._groups = [world.groups[g] for g in gnames]
        self._clocks = [_Clock(g) for g in self._groups]
        gidx = {g: i for i, g in enumerate(gnames)}

        # ---- arms (tables are per arm; regrouped per machine as (N_TAB + 1, 5 * arms, 3) for cheap gathers)
        specs: list[_ArmSpec] = []
        for m in scn.machines:
            if m.kind in KIND and m.on_line:
                specs += _weld_arms(m)
            elif m.kind in ("roller_hemming", "sealer") and not m.on_line:
                specs += _cell_arms(m)
        self.arm_specs = specs
        self.arm_machine = [s.mid for s in specs]
        self.A = len(specs)
        self._tab = _build_arm_table(specs) if specs else np.zeros((0, N_TAB + 1, N_ARM_C, 3))
        self.arm_bases = [(s.mid, float(s.base[0]), float(s.base[1])) for s in specs]
        wk = {m.id for m in scn.machines if m.kind in KIND}
        self.dressers = [(s.mid, float(s.q[-2][0]), float(s.q[-2][1])) for s in specs if s.mid in wk]

        # ---- machine records
        line = [m for m in scn.machines if m.on_line]
        self._line_ms = sorted(line, key=lambda m: m.x)
        self._mach: list[_Mach] = []
        self._byid: dict[str, _Mach] = {}
        for k, m in enumerate(scn.machines):
            r = _Mach()
            ws = world.W[m.id]
            r.m, r.k, r.gi = m, k, gidx[world.group_of[m.id]]
            r.bs, r.is_, r.ie = [w.bs for w in ws], [w.is_ for w in ws], [w.ie for w in ws]
            idx = [i for i, a in enumerate(self.arm_machine) if a == m.id]
            r.n_arm = len(idx)
            if idx:
                t = self._tab[idx[0]:idx[-1] + 1]                                   # (n, N+1, 5, 3)
                r.tab = np.ascontiguousarray(t.transpose(1, 0, 2, 3)).reshape(N_TAB + 1, -1, 3)
                r.tab.flags.writeable = False
            else:
                r.tab = None
            self._mach.append(r)
            self._byid[m.id] = r
        if len({len(r.bs) for r in self._mach if r.m.on_line}) > 1:
            raise ValueError("line machines must have aligned cycle windows")
        self._index = self._byid[min(line, key=lambda m: m.offset_s).id] if line else None   # carries the line index
        self._lifter = next((self._byid[m.id] for m in line if m.kind == "unload_lifter"), None)

        # ---- skids: body j travels slot[j] -> slot[j+1]
        xs = [m.x for m in self._line_ms]
        start = np.array([X_FEED] + xs[:-1])
        end = np.array(xs)
        u = np.linspace(0.0, 1.0, NS + 1)
        xc = start[None, :] + (end - start)[None, :] * _ss(u)[:, None]                 # (NS+1, nb)
        nb = len(xs)
        T = np.empty((NS + 1, nb, 3, 3))
        T[..., 0] = xc[:, :, None] + np.array(SKID_REL_X)[None, None, :]
        T[..., 1] = LINE_Z
        T[..., 2] = SKID_R
        self._skid_tab = T.reshape(NS + 1, nb * 3, 3)
        self._skid_tab.flags.writeable = False
        self._skid_rest = self._skid_tab[NS]
        self._skid_rest_picked = self._skid_tab[NS][:-3]

        # ---- lifter carriage (constant circles when present)
        if self._lifter is not None:
            lm = self._lifter.m
            c = np.array([(lm.x + dx, lm.z, LIFT_R) for dx in LIFT_REL_X])
            c.flags.writeable = False
            self._lift_c = c
        else:
            self._lift_c = _EMPTY

        # ---- AGV docks (one per line station): (machine record, station index, circles)
        self._docks = []
        for j, m in enumerate(self._line_ms):
            c = np.array([(m.x + DOCK_DX + dx, DOCK_Z, AGV_R) for dx in AGV_REL_X])
            c.flags.writeable = False
            self._docks.append((self._byid[m.id], j, c))

        self._t: float | None = None
        self._st: _State | None = None

    # ------------------------------------------------------------------ core
    def _compute(self, t: float) -> _State:
        st = self._st
        if st is not None and t == self._t and not any(c.changed() for c in self._clocks):
            return st
        mts, kinds = [], []
        for c in self._clocks:
            mt, kd, _ = c.query(t)
            mts.append(mt)
            kinds.append(kd)
        M = len(self._mach)
        mstate = [0] * M            # 0 idle, 1 active
        mfrac = [0.0] * M
        mwin = [-1] * M
        mrel = [0.0] * M
        midx = [N_TAB] * M          # arm-table sample per machine (N_TAB = stowed at home)
        views = [_EMPTY] * M
        for r in self._mach:
            k = r.k
            mt = mts[r.gi]
            bsl = r.bs
            i = bisect_right(bsl, mt) - 1
            idx = N_TAB
            if i >= 0:
                e = r.ie[i]
                if mt < e:
                    b = bsl[i]
                    mwin[k] = i
                    mrel[k] = mt - b
                    s = r.is_[i]
                    if mt < s:
                        mstate[k] = 1
                        f = (mt - b) / (s - b)
                        mfrac[k] = f
                        if kinds[r.gi] != "hold":
                            idx = int(f * N_TAB)
                    elif e > s:
                        mfrac[k] = (mt - s) / (e - s)
            if r.tab is not None:
                midx[k] = idx
                views[k] = r.tab[idx]
        # skids / lifter
        skids, transfer, lift_h = _EMPTY, None, None
        rl = self._lifter
        ri = self._index
        if ri is not None:
            i = mwin[ri.k]
            s = mrel[ri.k] if i >= 0 else math.inf
            if s < TRANSFER_S:
                transfer = s / TRANSFER_S
                skids = self._skid_tab[int(transfer * NS)]
            else:
                pick_abs = math.inf
                if rl is not None and 0 <= i < len(rl.bs):
                    b = rl.bs[i]
                    pick_abs = b + F_PICK * (rl.is_[i] - b)
                skids = self._skid_rest if mts[ri.gi] < pick_abs else self._skid_rest_picked
        if rl is not None:
            k = rl.k
            if mstate[k] and kinds[rl.gi] != "hold":
                lift_h = lift_height(mfrac[k])
                if lift_h < H_CLEAR:
                    views[k] = self._lift_c
            else:
                lift_h = LIFT_H_TOP
        # AGV docks
        parts, docked = [], []
        for r, j, c in self._docks:
            i = mwin[r.k]
            if i >= 0 and (i + j) % DOCK_EVERY == 0 and DOCK_T0 <= mrel[r.k] < DOCK_T0 + DOCK_S:
                parts.append(c)
                docked.append(r.m.id)
        st = _State()
        st.t, st.skids, st.views = t, skids, views
        st.agv = (np.concatenate(parts) if len(parts) > 1 else parts[0]) if parts else _EMPTY
        st.mt, st.kind, st.mstate, st.mfrac, st.midx = mts, kinds, mstate, mfrac, midx
        st.lift_h, st.transfer, st.docked = lift_h, transfer, docked
        st.flat = None
        st.bodies = None
        self._t, self._st = t, st
        return st

    # ------------------------------------------------------------------ API
    def bodies(self, t: float) -> dict[str, np.ndarray]:
        """Machine id -> (k,3) circles (cx, cz, r) of its moving parts at t, plus "SKIDS" and "AGVDOCK".

        Arms: 5 circles each (base, upper-arm mid, elbow, forearm mid, tool), 2 arms per station. The S80 entry is the
        lifter carriage (3 circles) while it is below `H_CLEAR`, otherwise empty. Read-only views.
        """
        st = self._compute(t)
        if st.bodies is None:
            out = dict(zip(self.machine_ids, st.views))
            out["SKIDS"] = st.skids
            out["AGVDOCK"] = st.agv
            st.bodies = out
        return dict(st.bodies)

    def all_circles(self, t: float) -> np.ndarray:
        """Every moving part stacked, (K,3). Read-only; cached per (t, freeze state)."""
        st = self._compute(t)
        if st.flat is None:
            a = np.concatenate(st.views + [st.skids, st.agv])
            a.flags.writeable = False
            st.flat = a
        return st.flat

    def phase(self, mid: str, t: float) -> tuple[str, float]:
        """Ground truth for tests / visualisation: ("active"|"idle"|"hold"|"trip", fraction within that state)."""
        r = self._byid[mid]
        mt, kind, fr = self._clocks[r.gi].query(t)
        if kind is not None:
            return kind, fr
        i = bisect_right(r.bs, mt) - 1
        if i < 0 or mt >= r.ie[i]:
            return "idle", 0.0
        b, s, e = r.bs[i], r.is_[i], r.ie[i]
        if mt < s:
            return "active", (mt - b) / (s - b)
        return "idle", (mt - s) / (e - s) if e > s else 0.0

    def cycle_phase(self, mid: str, t: float) -> float:
        """Position in the machine's cycle window, 0..1 ((mt - bs) / (ie - bs)); 0 outside any window."""
        r = self._byid[mid]
        mt = self._clocks[r.gi].query(t)[0]
        i = bisect_right(r.bs, mt) - 1
        if i < 0 or mt >= r.ie[i]:
            return 0.0
        return (mt - r.bs[i]) / (r.ie[i] - r.bs[i])

    def machine_time(self, mid: str, t: float) -> float:
        return self._clocks[self._byid[mid].gi].query(t)[0]

    def poses(self, t: float) -> dict[str, dict]:
        """Render-friendly joint state. Angles in the project convention (0 = +z, clockwise toward +x)."""
        st = self._compute(t)
        out: dict[str, dict] = {}
        for r in self._mach:
            ph = self.phase(r.m.id, t)
            e: dict = {"state": ph[0], "frac": ph[1]}
            if r.tab is not None:
                arms = []
                circ = r.tab[st.midx[r.k]].reshape(r.n_arm, N_ARM_C, 3)
                for c in circ:
                    b, el, tl = c[0, :2], c[2, :2], c[4, :2]
                    yu = math.atan2(el[0] - b[0], el[1] - b[1]) if np.hypot(*(el - b)) > 1e-6 else 0.0
                    yf = math.atan2(tl[0] - el[0], tl[1] - el[1]) if np.hypot(*(tl - el)) > 1e-6 else yu
                    arms.append({"base": (float(b[0]), float(b[1])), "shoulder_yaw": yu,
                                 "elbow": (yf - yu + math.pi) % (2 * math.pi) - math.pi,
                                 "elbow_xy": (float(el[0]), float(el[1])), "tool_xy": (float(tl[0]), float(tl[1])),
                                 "stowed": bool(st.midx[r.k] == N_TAB)})
                e["arms"] = arms
            if r is self._lifter:
                e["lifter_h"] = st.lift_h
                e["carriage"] = {"xy": (r.m.x, r.m.z), "size": (LIFT_L, LIFT_W), "hazard": len(st.views[r.k]) > 0}
                e["carrying"] = bool(st.mstate[r.k] and st.mfrac[r.k] >= F_PICK and st.kind[r.gi] != "hold")
            out[r.m.id] = e
        sk = st.skids
        out["SKIDS"] = {"x": [float(v) for v in sk[1::3, 0]], "z": LINE_Z, "size": (SKID_L, SKID_W),
                        "transfer_progress": st.transfer}
        out["AGVDOCK"] = {"docked": [{"station": m, "xy": (self._byid[m].m.x, DOCK_Z)} for m in st.docked]}
        return out
