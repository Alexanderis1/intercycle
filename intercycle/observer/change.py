"""Change detection on the floor: a new moving machine in space the map says is static (never imports ground truth).

Pipeline (all from camera observations; workstream M, DIRECTION section 3 item 5):

1. Residual. Per tick, on the 1 m site grid: x_c = 1 if a moving detection that is neither a tracked person nor a
   vehicle nor a skid ("unexplained motion": class PART outside every known hazard zone + margin) touches cell c. Cells
   inside a known zone (+ margin) are not static space and are skipped; cells of a camera that is offline this tick are
   not updated (no evidence either way).
2. CUSUM per cell (one-sided, Bernoulli): S_c = min(2h, max(0, S_c + x_c - k)). With k = 0.05 a 3 s glare blob reaches 5.7,
   a moving arm (x ~ 0.6 per tick) passes h = 10 in ~10 s. A cell with S_c >= h is ALARMED.
3. Candidate. Alarmed cells join the nearest candidate within `join_m` (else start one). A candidate's region is the
   bounding box of every evidence cell seen near it, grown by `margin_m`, and is UNKNOWN (= occupied) from the first
   alarm. It is reported (`change_detected`) once it has had alarmed cells for `confirm_s`. It is cleared (UNKNOWN
   lifted, `change_cleared`) after `clear_s` without any evidence unless it has been spawned.
4. Spawn. A candidate's motion signal (any evidence in its region) is buffered (last `learn_window_s`) and every
   `learn_every_s` the observer's own cycle learner (`observer.learn_cycles.learn`: ACF + Lomb-Scargle period, debounced
   cycle chain, confidence 0..7) is run on it. Once it reports a confident period (>= MIN_CONF) over >= K clean cycles
   and two consecutive estimates agree within 2 %, an L1 model is spawned (`machine_spawned`), unless the cycle is
   phase-locked to a known neighbour (see below): `LayeredMap.add_machine` gets a zone, and every tick this module
   rasterises the region's detections into that machine's local grid binned by its own phase (phase = time since the
   last cycle start / period, 16 bins + STOP when overdue; a debounced rise counts as a cycle start only >= 0.75 period
   after the previous one, so pauses inside a cycle cannot reset the phase). UNKNOWN stays until every phase bin has
   `min_visits` observed ticks and `settle_cycles` more cycles have passed (`machine_converged`).
5. Camera bump. Every `bump_every_s`, for each camera that sees known machines, the last `bump_window_s` of moving
   part detections inside those machines' local grids are matched against the learned envelopes at integer shifts
   of +-`bump_max_m`. If a non-zero shift explains them clearly better than no shift (twice in a row) the camera is
   declared bumped (`camera_bump`): its region becomes UNKNOWN (map.cam_offline) and change candidates centred in it
   are dropped and suppressed - a shifted camera must not look like a new machine.

Events are returned by `step()` as dicts {kind, t, ...} (the caller forwards them to observer/stream.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from intercycle.observer import mapping as M


@dataclass
class Candidate:
    cid: int
    t_first: float
    cells: set = field(default_factory=set)          # evidence cells (flat site index)
    t_last_ev: float = 0.0
    t_alarm_first: float = math.nan
    reported: bool = False
    t_reported: float = math.nan
    spawned: bool = False
    t_spawned: float = math.nan
    converged: bool = False
    t_converged: float = math.nan
    machine_index: int = -1
    machine_id: str = ""
    # motion signal / cycle
    cur: int = 0
    opp: int = 0
    opp_t0: float = math.nan
    rises: list = field(default_factory=list)
    period: float = math.nan
    last_rise: float = math.nan
    spawn_rise_n: int = 0
    rect: tuple = ()
    rise_phases: list = field(default_factory=list)  # known machines' phases (observer's tracker) at this candidate's rises
    explained_by: str = ""                           # phase-locked to a known machine next to it: displaced, not new
    yt: list = field(default_factory=list)           # motion signal buffer for the cycle learner
    yv: list = field(default_factory=list)
    yph: list = field(default_factory=list)          # known machines' phases at each buffered tick
    next_learn: float = math.nan
    estimates: list = field(default_factory=list)    # (t, period, confidence, n_cycles)


class ChangeDetector:
    def __init__(self, mp: M.LayeredMap, k: float = 0.05, h: float = 10.0, join_m: float = 4.0, margin_m: float = 1.0,
                 confirm_s: float = 30.0, clear_s: float = 600.0, K: int = 6, tol: float = 0.05, min_period_s: float = 5.0,
                 max_period_s: float = 600.0, settle_cycles: int = 4, zone_margin_m: float = 1.0, bump_every_s: float = 30.0,
                 bump_window_s: float = 180.0, bump_max_m: float = 2.0, bump_gain: float = 1.25, rise_confirm: int = 2,
                 lock_near_m: float = 2.0, lock_tol: float = 0.06, learn_window_s: float = 900.0, learn_every_s: float = 60.0,
                 fall_confirm: int = 4, hz: float = M.FRAME_HZ):
        self.map = mp
        self.k, self.h, self.join_m, self.margin_m = k, h, join_m, margin_m
        self.confirm_s, self.clear_s, self.K, self.tol = confirm_s, clear_s, K, tol
        self.min_P, self.max_P, self.settle_cycles = min_period_s, max_period_s, settle_cycles
        self.rise_confirm, self.fall_confirm, self.hz = rise_confirm, fall_confirm, hz
        self.lock_near_m, self.lock_tol = lock_near_m, lock_tol
        self.learn_window_s, self.learn_every_s = learn_window_s, learn_every_s
        self._t, self._cond = 0.0, None
        s = mp.spec["site"]
        self.site = s
        self.nx, self.nz = s["nx"], s["nz"]
        known = np.zeros((self.nx, self.nz), bool)
        g = zone_margin_m
        for d in mp.spec["machines"]:
            x0, z0, x1, z1 = d["rect"]
            known |= mp.rect_cells((x0 - g, z0 - g, x1 + g, z1 + g))
        self.static = ~known                                  # L0 static space (no known moving machine)
        self.S = np.zeros(self.nx * self.nz)
        self.cands: list[Candidate] = []
        self._next_id = 1
        self.n_machines_known = len(mp.spec["machines"])
        # camera bump
        self.bump_every_s, self.bump_window_s, self.bump_max = bump_every_s, bump_window_s, bump_max_m
        self.bump_gain = bump_gain
        self.bumped: dict[int, tuple[float, float, float]] = {}
        self._bump_strikes: dict[int, int] = {}
        self._next_bump: float | None = None
        self._bump_hist: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []   # (t, x, z, r) of moving parts
        self.cam_of_cell = mp.cam_of_cell.reshape(-1)
        self.bump_baseline: dict[int, float] = {}

    # ------------------------------------------------------------------ helpers
    def _cell_xy(self, c: int) -> tuple[float, float]:
        i, j = divmod(int(c), self.nz)
        r = self.site["res"]
        return self.site["x0"] + (i + 0.5) * r, self.site["z0"] + (j + 0.5) * r

    def _rect_of(self, cells) -> tuple[float, float, float, float]:
        ij = np.array([divmod(int(c), self.nz) for c in cells])
        r, g = self.site["res"], self.margin_m
        return (self.site["x0"] + ij[:, 0].min() * r - g, self.site["z0"] + ij[:, 1].min() * r - g,
                self.site["x0"] + (ij[:, 0].max() + 1) * r + g, self.site["z0"] + (ij[:, 1].max() + 1) * r + g)

    def _centre(self, c: Candidate) -> tuple[float, float]:
        x0, z0, x1, z1 = c.rect
        return (x0 + x1) / 2, (z0 + z1) / 2

    def _cam_of_xy(self, x: float, z: float) -> int:
        cams = np.array(self.map.spec["cameras"], float)
        return int(np.hypot(cams[:, 0] - x, cams[:, 1] - z).argmin())

    def unknown_rects(self) -> list[tuple]:
        return [c.rect for c in self.cands if c.rect and not c.converged]

    def _sync_unknown(self) -> None:
        base = [r for r in self.map.unknown_rects if getattr(self, "_owned", None) is None or r not in self._owned]
        self._owned = self.unknown_rects()
        self.map.unknown_rects = base + self._owned

    # ------------------------------------------------------------------ step
    def step(self, obs: M.Observation, cond: M.Cond | None = None) -> list[dict]:
        """One tick. `cond` (the ConditionTracker's output for the same tick) enables the phase-lock check."""
        t = obs.t_cap
        self._t, self._cond = t, cond
        ev: list[dict] = []
        sel = (obs.cls == M.PART) & obs.moving & (obs.machine < 0)
        x = np.zeros(self.nx * self.nz)
        if sel.any():
            s = self.site
            _, ix, iz = M.raster(obs.x[sel], obs.z[sel], obs.r[sel], s["x0"], s["z0"], self.nx, self.nz, s["res"], centre=False)
            x[np.unique(ix * self.nz + iz)] = 1.0
        valid = obs.cam_ok[self.cam_of_cell] & self.static.reshape(-1)
        for cam in self.bumped:
            valid &= self.cam_of_cell != cam
        self.S = np.where(valid, np.clip(self.S + x - self.k, 0.0, 2.0 * self.h), self.S)   # capped: re-arms after motion stops
        alarmed = np.flatnonzero(self.S >= self.h)
        evid = np.flatnonzero(valid & (x > 0))
        # ---- candidates: assign alarmed cells, grow with nearby evidence
        for c in alarmed:
            cx, cz = self._cell_xy(c)
            best, bd = None, self.join_m
            for cd in self.cands:
                d = min(math.hypot(cx - a, cz - b) for a, b in (self._cell_xy(q) for q in list(cd.cells)[:64])) if cd.cells else math.inf
                if cd.rect:
                    x0, z0, x1, z1 = cd.rect
                    d = min(d, math.hypot(max(x0 - cx, 0, cx - x1), max(z0 - cz, 0, cz - z1)))
                if d <= bd:
                    best, bd = cd, d
            if best is None:
                if self._suppressed(cx, cz):
                    continue
                best = Candidate(self._next_id, t)
                self._next_id += 1
                self.cands.append(best)
            best.cells.add(int(c))
            if not np.isfinite(best.t_alarm_first):
                best.t_alarm_first = t
                best.t_last_ev = t
            if x[c] > 0:
                best.t_last_ev = t
            best.rect = self._rect_of(best.cells)
        for cd in self.cands:
            if not cd.rect:
                continue
            x0, z0, x1, z1 = cd.rect
            near = [int(c) for c in evid if self._in_rect(c, (x0 - 2, z0 - 2, x1 + 2, z1 + 2))]
            inside = [c for c in near if self._in_rect(c, cd.rect)]
            if near:
                cd.cells.update(near)
                cd.rect = self._rect_of(cd.cells)
                cd.t_last_ev = t
            self._cycle(cd, t, bool(inside), ev)
            if not cd.spawned and not cd.explained_by:
                cd.yt.append(t)
                cd.yv.append(1.0 if inside else 0.0)
                cd.yph.append(np.asarray(cond.phase[:self.n_machines_known], float) if cond is not None
                              else np.full(self.n_machines_known, np.nan))
                while cd.yt and cd.yt[0] < t - self.learn_window_s:
                    cd.yt.pop(0)
                    cd.yv.pop(0)
                    cd.yph.pop(0)
                if not np.isfinite(cd.next_learn):
                    cd.next_learn = t + self.learn_every_s
                elif t >= cd.next_learn:
                    cd.next_learn = t + self.learn_every_s
                    self._try_learn(cd, t, ev)
            if not cd.reported and t - cd.t_alarm_first >= self.confirm_s and t - cd.t_last_ev < self.confirm_s:
                cd.reported, cd.t_reported = True, t
                cx, cz = self._centre(cd)
                ev.append(dict(kind="change_detected", t=t, candidate=cd.cid, x=cx, z=cz, rect=list(cd.rect), cells=len(cd.cells)))
        # ---- clear stale candidates (never spawned)
        keep = []
        for cd in self.cands:
            if not cd.spawned and t - cd.t_last_ev > self.clear_s:
                if cd.reported:
                    ev.append(dict(kind="change_cleared", t=t, candidate=cd.cid, rect=list(cd.rect)))
                for c in cd.cells:
                    self.S[c] = 0.0
                continue
            keep.append(cd)
        self.cands = keep
        # ---- spawned models learn their envelope
        for cd in self.cands:
            if cd.spawned:
                self._learn(cd, obs, t, ev)
        # ---- camera bump check
        self._bump_record(obs, t)
        if self._next_bump is None:
            self._next_bump = t + self.bump_every_s
        elif t >= self._next_bump:
            self._next_bump = t + self.bump_every_s
            ev += self._bump_check(t)
        self._sync_unknown()
        return ev

    def _in_rect(self, c: int, rect) -> bool:
        x, z = self._cell_xy(c)
        return rect[0] <= x <= rect[2] and rect[1] <= z <= rect[3]

    def _suppressed(self, x: float, z: float) -> bool:
        return self._cam_of_xy(x, z) in self.bumped

    # ------------------------------------------------------------------ cycle of a candidate
    def _cycle(self, cd: Candidate, t: float, y: bool, ev: list) -> None:
        if (cd.cur == 1) == y:
            cd.opp = 0
            return
        if cd.opp == 0:
            cd.opp_t0 = t
        cd.opp += 1
        thr = self.rise_confirm if cd.cur == 0 else self.fall_confirm
        if cd.opp < thr:
            return
        rising = cd.cur == 0
        cd.cur, cd.opp = (1 if rising else 0), 0
        if not rising:
            return
        te = cd.opp_t0 - 0.5 / self.hz
        if not cd.spawned:
            return
        # spawned: a rise is a new cycle start only well after the previous start (pauses inside a cycle are not starts)
        if np.isfinite(cd.last_rise) and te - cd.last_rise < 0.75 * cd.period:
            return
        iv = te - cd.last_rise if np.isfinite(cd.last_rise) else math.nan
        cd.rises.append(te)
        del cd.rises[:-32]
        cd.last_rise = te
        if np.isfinite(iv) and 0.88 * cd.period < iv < 1.12 * cd.period:
            cd.period = 0.9 * cd.period + 0.1 * float(iv)

    def _try_learn(self, cd: Candidate, t: float, ev: list) -> None:
        """Run the cycle learner on the candidate's buffered motion signal; spawn once the period is confident and
        stable and not phase-locked to a known neighbour."""
        from intercycle.observer import learn_cycles as LC
        T = np.array(cd.yt)
        if len(T) < 60 or T[-1] - T[0] < 0.5 * self.learn_window_s:
            return
        Y = np.array(cd.yv, float)
        try:
            L = LC.learn(T, Y, latency_s=0.0)
        except Exception:
            return
        mod = L.model
        n = int(L.diag.get("n_cycles", 0))
        P = float(mod.period_s)
        if not (mod.confidence >= LC.MIN_CONF and n >= self.K and self.min_P <= P <= self.max_P):
            cd.estimates.clear()
            return
        cd.estimates.append((t, P, int(mod.confidence), n))
        if len(cd.estimates) < 2 or abs(cd.estimates[-2][1] / P - 1.0) >= 0.02:
            return
        rises = np.asarray(L.edges.rise, float)
        rises = rises[rises <= T[-1]]
        if not len(rises):
            return
        ph = np.array(cd.yph)
        k = np.clip(np.searchsorted(T, rises[-self.K:]), 0, len(T) - 1)
        cd.rise_phases = [ph[j] for j in k]
        lock = self._locked_to(cd)
        if lock == "?":
            return
        if lock:
            if not cd.explained_by:
                cd.explained_by = lock
                ev.append(dict(kind="change_explained", t=t, candidate=cd.cid, machine=lock, period_s=P,
                               note="phase-locked to a known machine next to it: displaced motion (camera bump?), "
                                    "not a new machine; region stays UNKNOWN"))
            return
        cd.rises = [float(r) for r in rises[-8:]]
        cd.last_rise = float(rises[-1])
        self._spawn(cd, t, P, ev, conf=int(mod.confidence), n_cycles=n, jitter=float(mod.jitter))

    def _locked_to(self, cd: Candidate) -> str:
        """Id of a known machine within `lock_near_m` of the candidate whose observed cycle phase is the same at every
        one of the candidate's last K rises (circular spread < lock_tol), else ''. A new machine is not phase-locked to
        its neighbour; the displaced image of the neighbour (a bumped camera) is. "?" = a neighbour's phase is not known
        at enough of those rises yet (the spawn waits)."""
        if not cd.rect:
            return ""
        if len(cd.rise_phases) < self.K:
            cd.rise_phases = [np.full(self.n_machines_known, np.nan)] * (self.K - len(cd.rise_phases)) + cd.rise_phases
        ph = np.array(cd.rise_phases[-self.K:])
        x0, z0, x1, z1 = cd.rect
        g = self.lock_near_m
        for i, d in enumerate(self.map.spec["machines"][:self.n_machines_known]):
            a0, b0, a1, b1 = d["rect"]
            if a1 < x0 - g or a0 > x1 + g or b1 < z0 - g or b0 > z1 + g:
                continue
            col = ph[:, i]
            col = col[np.isfinite(col)]
            if len(col) < 3:                                  # neighbour's phase not known yet: undecided, do not spawn yet
                return "?"
            ang = 2 * np.pi * (col % 1.0)
            R = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
            spread = math.sqrt(max(-2 * math.log(max(R, 1e-12)), 0.0)) / (2 * np.pi)
            if spread < self.lock_tol:
                return d["id"]
        return ""

    def _spawn(self, cd: Candidate, t: float, P: float, ev: list, conf: int = 0, n_cycles: int = 0, jitter: float = 0.02) -> None:
        mp = self.map
        k = sum(1 for c in self.cands if c.spawned) + 1
        cd.machine_id = f"NEW{k}"
        x0, z0, x1, z1 = cd.rect
        g = self.margin_m
        rect = (x0 + g, z0 + g, x1 - g, z1 - g)              # the zone is the evidence box; the L1 grid adds the margin
        cd.machine_index = mp.add_machine(cd.machine_id, rect)
        cd.period, cd.spawned, cd.t_spawned = P, True, t
        cd.spawn_rise_n = len(cd.rises)
        mp.period[cd.machine_index] = P
        mp.jitter[cd.machine_index] = max(jitter, 0.005)
        ev.append(dict(kind="machine_spawned", t=t, candidate=cd.cid, machine=cd.machine_id, period_s=P, rect=list(rect),
                       confidence=conf, n_cycles=n_cycles))

    def phase_bin(self, cd: Candidate, t: float) -> int:
        if not (cd.spawned and np.isfinite(cd.last_rise)):
            return -1
        ph = (t - cd.last_rise) / cd.period
        if ph > 1.0 + max(0.06, 3.0 / cd.period):
            return M.STOP_BIN
        return int(min(max(math.floor(ph * M.N_PHASE), 0), M.N_PHASE - 1))

    def phase_now(self) -> dict[str, float]:
        """{machine id: phase fraction} for spawned machines (what the fast layer would carry)."""
        out = {}
        for cd in self.cands:
            if cd.spawned and np.isfinite(cd.last_rise):
                out[cd.machine_id] = float((self._t - cd.last_rise) / cd.period)
        return out

    def _learn(self, cd: Candidate, obs: M.Observation, t: float, ev: list) -> None:
        mp, i = self.map, cd.machine_index
        b = self.phase_bin(cd, t)
        if b < 0 or not obs.cam_ok[mp.cam_of_machine[i]]:
            return
        mp.l1_visits[i, b] += 1.0
        sel = (obs.cls == M.PART) & (obs.machine < 0)
        x0, z0 = mp.l1_x0[i], mp.l1_z0[i]
        nx, nz = mp.l1_nx[i], mp.l1_nz[i]
        res = mp.spec["env_res"]
        if sel.any():
            _, ix, iz = M.raster(obs.x[sel], obs.z[sel], np.maximum(obs.r[sel] - mp.shrink_m, 0.05), x0, z0, nx, nz, res, centre=True)
            if len(ix):
                cells = np.unique(ix * nz + iz)
                mp.l1_view(i)[b].reshape(-1)[cells] += 1.0
        mp._cover_cache.clear()
        if not cd.converged and (mp.l1_visits[i, :M.N_PHASE] >= mp.min_visits).all() and \
                len(cd.rises) - cd.spawn_rise_n >= self.settle_cycles:
            cd.converged, cd.t_converged = True, t
            ev.append(dict(kind="machine_converged", t=t, machine=cd.machine_id, period_s=cd.period))

    # ------------------------------------------------------------------ camera bump
    def _bump_record(self, obs: M.Observation, t: float) -> None:
        sel = (obs.cls == M.PART) & obs.moving
        self._bump_hist.append((t, obs.x[sel].copy(), obs.z[sel].copy(), obs.r[sel].copy()))
        while self._bump_hist and self._bump_hist[0][0] < t - self.bump_window_s:
            self._bump_hist.pop(0)

    def _bump_check(self, t: float) -> list[dict]:
        mp = self.map
        if not self._bump_hist:
            return []
        X = np.concatenate([h[1] for h in self._bump_hist])
        Z = np.concatenate([h[2] for h in self._bump_hist])
        R = np.concatenate([h[3] for h in self._bump_hist])
        ev = []
        res = mp.spec["env_res"]
        smax = int(round(self.bump_max / res))
        for cam in sorted(set(int(c) for c in mp.cam_of_machine[:self.n_machines_known])):
            if cam in self.bumped:
                continue
            num = np.zeros((2 * smax + 1, 2 * smax + 1))
            den = 0.0
            for i in np.flatnonzero(mp.cam_of_machine[:self.n_machines_known] == cam):
                mid = mp.spec["machines"][i]["id"]
                if not mp.l1_known(mid):
                    continue
                ref = np.zeros((mp.l1_nx[i], mp.l1_nz[i]))
                for b in range(M.N_BINS):
                    if mp.l1_visits[i, b] >= mp.min_visits:
                        ref = np.maximum(ref, mp.envelope_prob(mid, b))
                x0, z0, nx, nz = mp.l1_x0[i], mp.l1_z0[i], mp.l1_nx[i], mp.l1_nz[i]
                inb = (X >= x0) & (X < x0 + nx * res) & (Z >= z0) & (Z < z0 + nz * res)
                if inb.sum() < 50:
                    continue
                _, ix, iz = M.raster(X[inb], Z[inb], np.maximum(R[inb] - mp.shrink_m, 0.05), x0, z0, nx, nz, res, centre=True)
                H = np.zeros((nx, nz))
                np.add.at(H, (ix, iz), 1.0)
                den += H.sum()
                P = np.pad(ref, smax)
                for a in range(2 * smax + 1):
                    for b_ in range(2 * smax + 1):
                        # detections shifted back by (a - smax, b - smax) cells: H[i, j] compared with ref[i - da, j - db]
                        num[a, b_] += (H * P[2 * smax - a:2 * smax - a + nx, 2 * smax - b_:2 * smax - b_ + nz]).sum()
            if den <= 0:
                continue
            sc = num / den
            s0 = sc[smax, smax]
            a, b_ = np.unravel_index(int(sc.argmax()), sc.shape)
            if (a, b_) != (smax, smax) and sc[a, b_] > self.bump_gain * max(s0, 1e-6) and max(abs(a - smax), abs(b_ - smax)) >= 2:
                self._bump_strikes[cam] = self._bump_strikes.get(cam, 0) + 1
            else:
                self._bump_strikes[cam] = 0
                self.bump_baseline[cam] = float(s0)
            if self._bump_strikes[cam] >= 2:
                dx, dz = (a - smax) * res, (b_ - smax) * res
                self.bumped[cam] = (t, dx, dz)
                mp.cam_offline.add(cam)
                dropped = []
                for cd in list(self.cands):
                    if not cd.spawned and cd.rect and self._cam_of_xy(*self._centre(cd)) == cam:
                        dropped.append(cd.cid)
                        self.cands.remove(cd)
                ev.append(dict(kind="camera_bump", t=t, camera=cam, dx=dx, dz=dz, score_shift=float(sc[a, b_]),
                               score_zero=float(s0), dropped_candidates=dropped))
        return ev
