"""Minimal W-W2 for A-lite: robot programs that vary by car model, tip-dress every N cycles, path noise (SIMULATED
plant; truth side).

The defect (WEAKNESSES W-W2): body-shop arm paths are seed-independent (`rng([7, id, r])`, kinematics.py), identical
every cycle, with a tip-dress every cycle, so phases and L1 envelopes are trivially periodic. `ProgramKinematics` is
the body-shop `Kinematics` (same skids, lifter, AGV docks, freezes, holds) with, per PLAN_TWIN §7 item 2:

- 2 robot programs per station. Line stations (weld / respot / framing): program 0 = car model A (the body-shop weld
  pattern), program 1 = car model B (a longer, wider weld pattern: 2 more weld points, +-2.0 m along the body instead
  of +-1.8 m, rows 0.45 / 1.05 m off the line instead of 0.35 / 0.9 m). Weld points get a per-world jitter (the
  plant's own teach points), keyed (world_seed, 0, "<station>/arm<r>", "program<k>"), so paths differ between
  worlds. Cells: program 1 runs the program-0 path in reverse (hemming roller the other way round, the sealer bead
  from the other end).
- a seeded model-mix sequence: per production day p the bodies fed into the line are model B with probability
  MODEL_MIX[1] = 0.3, keyed (world_seed, p + 1, "line", "model_mix"); the body at station j in line cycle k of that
  day is the one fed at cycle k - j (the first j cycles of a day take theirs from the end of the same day's sequence:
  the line restarts after the shift change). Cells draw their own model per cycle, keyed (world_seed, p + 1, cell,
  "model_mix").
- tip-dress every N cycles: N per station in [8, 30], keyed (world_seed, 0, station, "tipdress"); a station dresses
  on the first cycle of the production day and every N cycles after. A cycle without a dress returns home straight
  from the last weld point (weld 0-0.80 of the active time, return 0.80-1.0); a dress cycle is the body-shop timeline
  (swing out 0.80-0.86, dress 0.86-0.92, return 0.92-1.0).
- +-3 cm path noise: per cycle and arm a planar offset U(-0.03, 0.03) m on x and z of every circle but the base,
  keyed (world_seed, p + 1, station, "path_noise").
- program change (M-W4 scenario): `program_change={station: t_from}` gives that station a third program (model C:
  program 1's weld points in reverse order, shifted 0.4 m along the line) on every cycle that starts at or after
  t_from (machine time of the window start).

Arm speed is fixed per program (the body-shop timeline fractions); cycle jitter stays in the waits (the window
model), as W-W2 asks. Stowed poses (idle, hold) are the same for every program. Needs a `KeyedWorld` (its windows
start at a production-day start, so the per-day cycle counter is absolute).
"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np

from intercycle.alite.world import KeyedWorld, production_day
from intercycle.scenario import Machine, Scenario
from intercycle.truth import kinematics as K
from twin.rng import keyed_rng

MODEL_MIX = (0.7, 0.3)                 # model A, model B
DRESS_EVERY = (8, 30)                  # tip-dress every N cycles, N uniform on these integers (inclusive)
PATH_NOISE_M = 0.03
K_MAX = 4096                           # per-day sequence length (a day has ~1400 line cycles)
PROGRAM_C_SHIFT_M = 0.4


def _weld_points(m: Machine, side: float, r: int, world_seed: int, prog: int) -> np.ndarray:
    n, _, _ = K.KIND[m.kind]
    rng = K_rng(world_seed, f"{m.id}/arm{r}", f"program{min(prog, 1)}")
    if prog == 0:
        amp, rows, nn = 1.8, (0.35, 0.9), n
    else:
        amp, rows, nn = 2.0, (0.45, 1.05), n + 2
    sk = np.arange(nn) / (nn - 1)
    xs = (-amp + 2 * amp * (1.0 - np.abs(2.0 * sk - 1.0))) * (-side) + rng.uniform(-0.15, 0.15, nn)
    dz = np.clip(np.where(sk < 0.5, rows[0], rows[1]) + rng.uniform(-0.12, 0.12, nn), 0.2, 1.1)
    pts = np.stack([m.x + xs, K.LINE_Z + side * dz], axis=1)
    if prog == 2:                                           # model C: model B's points reversed, shifted along the line
        pts = pts[::-1] + np.array([PROGRAM_C_SHIFT_M * side, 0.0])
    return pts


def K_rng(world_seed: int, entity: str, event: str) -> np.random.Generator:
    """Plant constants (teach points, dress interval): keyed on day 0, they do not change from day to day."""
    return keyed_rng(world_seed, 0, entity, event)


def _timeline_nodress(base, pts, v, dwell):
    """Home -> weld points (move, dwell) over 0-0.80 -> straight home over 0.80-1.0 (no tip-dress)."""
    q = np.vstack([base, pts, base])
    d = np.hypot(*np.diff(q, axis=0).T)
    n = len(pts)
    tm = d[:n] / v
    w = tm + dwell
    w = w / w.sum() * K.T_WELD_END
    mf = np.concatenate([tm / (tm + dwell), [0.75]])
    w = np.concatenate([w, [1.0 - K.T_WELD_END]])
    return q, np.concatenate([[0.0], np.cumsum(w)]), mf


def _line_specs(m: Machine, world_seed: int, prog: int, dress: bool) -> list:
    _, v, dwell = K.KIND[m.kind]
    out = []
    for r, side in enumerate((-1.0, 1.0)):
        base = np.array([m.x + side * K.BASE_DX, K.LINE_Z + side * K.BASE_DZ])
        pts = _weld_points(m, side, r, world_seed, prog)
        if dress:
            q, brk, mf = K._timeline_weld(base, pts, np.array([base[0], K.LINE_Z + side * K.DRESS_DZ]), v, dwell)
        else:
            q, brk, mf = _timeline_nodress(base, pts, v, dwell)
        out.append(K._ArmSpec(m.id, base, q, brk, mf, K.ARM_R[4]))
    return out


def _cell_specs(m: Machine, world_seed: int, prog: int) -> list:
    specs = K._cell_arms(m)
    out = []
    for r, s in enumerate(specs):
        rng = K_rng(world_seed, f"{m.id}/arm{r}", "program0")
        q = np.array(s.q, float)
        inner = q[1:-1] + rng.uniform(-0.1, 0.1, q[1:-1].shape)            # the plant's own teach points
        if prog == 1:
            inner = inner[::-1]
        q2 = np.vstack([q[:1], inner, q[-1:]])
        _, brk, mf = K._timeline_path(q2[0], q2[1:-1])
        out.append(K._ArmSpec(m.id, s.base, q2, brk, mf, s.r_tool, sgn=s.sgn))
    return out


def _machine_table(specs: list) -> np.ndarray:
    t = K._build_arm_table(specs)                                         # (A, N+1, 5, 3)
    out = np.ascontiguousarray(t.transpose(1, 0, 2, 3)).reshape(K.N_TAB + 1, -1, 3)
    out.flags.writeable = False
    return out


class _Prog:
    """Per-machine program tables and per-window (program, dress, offset)."""
    __slots__ = ("r", "tabs", "prog", "dress", "off")


class ProgramKinematics(K.Kinematics):
    def __init__(self, scn: Scenario, world: KeyedWorld, program_change: dict[str, float] | None = None):
        if not isinstance(world, KeyedWorld):
            raise TypeError("ProgramKinematics needs a KeyedWorld (absolute per-day cycle counters)")
        super().__init__(scn, world)
        seed = world.seed
        self.world_seed = seed
        self.program_change = dict(program_change or {})
        jpos = {m.id: j for j, m in enumerate(self._line_ms)}
        self._progs: list[_Prog] = []
        self.dress_every: dict[str, int] = {}
        line_idx = self._index
        # line cycle -> (production day, counter within the day), from the index-carrying station's windows
        lday, lk = self._day_counter(line_idx.bs) if line_idx is not None else ([], [])
        mix_cache: dict[tuple[str, int], np.ndarray] = {}

        def mix(entity: str, p: int) -> np.ndarray:
            if (entity, p) not in mix_cache:
                mix_cache[(entity, p)] = keyed_rng(seed, p + 1, entity, "model_mix").random(K_MAX) < MODEL_MIX[1]
            return mix_cache[(entity, p)]

        for r in self._mach:
            m = r.m
            if r.tab is None or m.kind == "unload_lifter":
                continue
            pg = _Prog()
            pg.r = r
            n = len(r.bs)
            line = m.kind in K.KIND and m.on_line
            if line:
                n_prog = 3 if m.id in self.program_change else 2
                pg.tabs = [[_machine_table(_line_specs(m, seed, k, d)) for d in (False, True)] for k in range(n_prog)]
                N = int(K_rng(seed, m.id, "tipdress").integers(DRESS_EVERY[0], DRESS_EVERY[1] + 1))
                self.dress_every[m.id] = N
                days, ks = lday, lk
                j = jpos[m.id]
                prog = np.array([int(mix("line", p)[(k - j) % K_MAX]) for p, k in zip(days, ks)], np.int8)
                dress = np.array([k % N == 0 for k in ks], bool)
            else:
                n_prog = 2
                pg.tabs = [[_machine_table(_cell_specs(m, seed, k))] * 2 for k in range(n_prog)]
                days, ks = self._day_counter(r.bs)
                prog = np.array([int(mix(m.id, p)[k % K_MAX]) for p, k in zip(days, ks)], np.int8)
                dress = np.zeros(n, bool)
            if m.id in self.program_change:
                t_from = float(self.program_change[m.id])
                prog = np.where(np.asarray(r.bs) >= t_from, 2, prog).astype(np.int8)
            off = np.zeros((n, r.n_arm * K.N_ARM_C, 3))
            for p in sorted(set(days)):
                sel = np.flatnonzero(np.asarray(days) == p)
                nz = keyed_rng(seed, p + 1, m.id, "path_noise").uniform(-PATH_NOISE_M, PATH_NOISE_M,
                                                                          (K_MAX, r.n_arm, 2))
                kk = np.asarray(ks)[sel] % K_MAX
                o = np.repeat(nz[kk], K.N_ARM_C, axis=1)                    # (len(sel), n_arm * 5, 2)
                o[:, ::K.N_ARM_C, :] = 0.0                                  # bases do not move
                off[sel, :, :2] = o
            pg.prog, pg.dress, pg.off = prog, dress, off
            self._progs.append(pg)
        self._pst = None

    def _day_counter(self, bs: list[float]) -> tuple[list[int], list[int]]:
        days, ks = [], []
        last, k = None, 0
        for b in bs:
            p = production_day(self.scn, b)
            k = k + 1 if p == last else 0
            last = p
            days.append(p)
            ks.append(k)
        return days, ks

    def cycle_program(self, mid: str, t: float) -> tuple[int, bool] | None:
        """(program, tip-dress) of the window the machine is in at t (truth; tests / scoring), None outside."""
        for pg in self._progs:
            if pg.r.m.id == mid:
                i = bisect_right(pg.r.bs, self.machine_time(mid, t)) - 1
                return (int(pg.prog[i]), bool(pg.dress[i])) if i >= 0 else None
        return None

    def _compute(self, t: float):
        st = super()._compute(t)
        if st is self._pst:
            return st
        for pg in self._progs:
            r = pg.r
            idx = st.midx[r.k]
            if idx == K.N_TAB:                       # stowed (idle / hold): the same pose in every program
                continue
            i = bisect_right(r.bs, st.mt[r.gi]) - 1
            if i < 0:
                continue
            v = pg.tabs[pg.prog[i]][int(pg.dress[i])][idx] + pg.off[i]
            v.flags.writeable = False
            st.views[r.k] = v
        st.flat = None
        st.bodies = None
        self._pst = st
        return st
