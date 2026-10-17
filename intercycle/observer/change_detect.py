"""M-W4: in-zone change detection and count forgetting per OBSERVED tick (never imports ground truth; numpy only).

The defect (docs/WEAKNESSES.md M-W4, safety direction). The map's occupancy is p = hits / visits per cell and slice, and
a cell with p < p_min is broadcast FREE. `observer/change.py` only watches static space outside every known zone + 1 m,
so a new arm whose detections stay inside zone + 1 m (with the measured sigma 0.285 m: an arm inside the zone or hugging
its edge; an arm 0.5 m or more outside the edge spills past the margin and change.py catches it, see
tests/test_change_detect_l1.py), a robot program change or a newly always-occupied cell inside zone + 1 m is absorbed
into the counts. The counts never decay, so against a week of visits such a cell needs about 9 h (a 30 %-swept cell
about 30 h) to reach p_min, and until then it is broadcast FREE.

The fix, as one small object per grid (`ChangeLayer`):

1. Counts with forgetting per observed tick. Per (slice, cell) the layer keeps decayed visits n, hits k and entries e
   (a hit whose previous observed tick of the same cell was not a hit). On a tick where the cell is OBSERVED, n, k and e
   of the current slice are multiplied by lam = 2 ** (-1 / half_life_ticks) and then incremented. Cells that are not
   observed (camera down, occluded, outside the frustum) and the other slices are not touched: forgetting runs on
   coverage, never on wall time, so a camera outage or an unvisited slice does not erase what the map knows.
   The layer also keeps the undecayed visits N and hits K of the same ticks (the map's counts without forgetting).
   p = k / n is the occupancy with forgetting (`p()`), P = K / N the undecayed one (`p_lifetime()`).
   Forgetting may only ADD occupancy: a cell is OCCUPIED in a slice when p >= p_min OR P >= p_min. So a cell that a
   robot uses only now and then (a weekly or maintenance program variant, a rare reach) and whose decayed p has fallen
   below p_min during its days of non-use is still held by its undecayed P; forgetting never broadcasts it FREE.
   (A cell whose undecayed P is below p_min too is FREE, exactly as in the map without this module: not a new mode.)

2. CUSUM on in-zone hits in FREE cells. A cell is MONITORED in the current slice when n >= min_visits and it is not
   OCCUPIED there (p < p_min AND P < p_min: it is broadcast FREE). Per monitored, observed cell a one-sided CUSUM
   S = clip(S + llr, 0, cap) runs on its hit sequence. Camera hits come in runs (one arm pass covers a cell for several
   ticks, a rare excursion is one burst), so the log-likelihood ratio is the one of a two-state Markov chain, not of iid
   Bernoulli draws: llr depends on the cell's previous observed state (miss/hit) and its current one,
       miss->hit  log(b1 / b0)        miss->miss  log((1 - b1) / (1 - b0))
       hit->hit   log(a1 / a0)        hit->miss   log((1 - a1) / (1 - a0))
   with the null entry and stay probabilities b0 = P(hit | miss before), a0 = P(hit | hit before) estimated from the
   cell's own decayed counts in this slice (weak priors, b0 clipped to >= b_floor), and the change alternative "entries
   at least m times as frequent (and at least b_alt per tick), stays m times as long": b1 = max(m * b0, b_alt),
   1 - a1 = (1 - a0) / m. A short burst that the cell already shows at its learned rate (a rare tool change, a gross
   detection error) adds one entry term and is drained by the miss->miss terms; a new arm (frequent entries, long stays)
   or a newly always-occupied cell (long stays) climbs at a bounded rate.
   The run state is gap-filled: x = 1 while the cell had a hit in its last `hold` + 1 observed ticks, so the camera's
   iid misses inside one run (32 % per frame) do not read as many short runs.
   Two CUSUMs share these increments: S[c] pooled over all slices (a change not locked to the slices, e.g. a new arm
   with its own cycle) and Ss[s, c] per slice (a change in some slices only, e.g. a program change that puts the arm in
   cells at phases where they used to be free). max(S, Ss) >= h  ->  the cell is UNKNOWN at once (that very tick).

3. Lifting UNKNOWN. An unexplained hit is a hit on a cell that the broadcast shows FREE (point 4); it counts for the
   cell itself and for every cell within `dilate` cells of it (its UNKNOWN dilation is what protects such a neighbour,
   e.g. a cell the change occupies too rarely for its own CUSUM). "Went away" is judged in the slice where the change
   was seen (re-audit S7a fix1: with time-of-day slices visited in long blocks, a lift counted over all slices cleared a
   change seen only in slice B during a block of slice A, where it never occurs, and B was then broadcast FREE while
   occupied). Per (slice, cell) the layer keeps, since the cell's last unexplained hit IN THAT SLICE: its observed ticks
   in that slice (own), its observed ticks in any slice (all), and the separate visits of that slice in which it was
   observed (a visit = a run of consecutive steps with the same slice index). A slice is clear for the cell when
       own >= clear_ticks                                                     (long blocks: 15 min inside the slice), or
       all >= clear_ticks and visits >= clear_visits and own >= clear_min_ticks       (fast-cycling slices, e.g. phase
                                                  bins: the slice was sampled across the whole window, several times),
   and a slice with no unexplained hit at all is clear. An UNKNOWN cell is lifted only when EVERY slice is clear, so a
   change seen only in slice B is never cleared by ticks of slice A. The second rule keeps the single-slice behaviour
   (clear_ticks of the cell's observed ticks) for fast-cycling slices, instead of clear_ticks per slice (17 x longer on
   the body-shop phase bins). Its two guards: clear_visits = 3 separate visits (a change in a slice visited in long
   blocks must be absent from whole visits), clear_min_ticks = 12 own ticks (a persistent change at the measured 0.68
   detection rate is missed 12 ticks in a row with probability 0.32 ** 12 ~ 1e-6). This covers both ends in each slice:
   the change went away there (no hits), or the broadcast took it in there (its hits are no longer on FREE cells). An
   unobserved cell is never lifted, and its S is frozen.
   Availability cost: an unexplained hit in a rarely visited slice (the STOP bin, a weekend time-of-day bin), also a
   stray detection near the change, keeps the cell UNKNOWN until that slice is clear again by one of the two rules,
   however long that takes in wall clock.
   Availability cost: detection noise around a persistent change (sigma 0.285 m) keeps hitting FREE fringe cells, so
   the dilated UNKNOWN blob around it usually stays as long as the change does.

4. The lift is decided against what is BROADCAST, never against counts the broadcast does not use (audit S7a #1).
   Standalone (a twin grid, lane S6), the broadcast is the layer's own `state(slice)`: per cell FREE / OCCUPIED /
   UNKNOWN; UNKNOWN where the cell is UNKNOWN (dilated) or its slice is not trusted yet (n < min_visits: fail closed,
   costs the first min_visits observed ticks of each (slice, cell) after a start, since nothing is persisted), OCCUPIED
   by the rule of point 1, FREE otherwise. The caller broadcasts `state()` (`not_free()`), never its own undecayed
   counts: a cell lifted because the decayed counts absorbed the change is OCCUPIED in `state()` from that tick on.
   When the broadcast is someone else's (the body-shop LayeredMap's undecayed envelope), the caller passes it to
   `step(..., free=)`: the CUSUM monitors and the lift is judged on those FREE cells, so a lift happens only once that
   broadcast covers the change (`change_detect_l1.L1ChangeDetector` does this itself; forgetting then serves the
   CUSUM's null only, and the UNKNOWN box stays until the map's undecayed counts take the change in, e.g. ~13 h for a
   newly always-occupied cell against a week of history).

Sensitivity floor (disclosed, M-W4): a change that occupies a cell only ~10 % of the observed ticks, not locked to the
slices (e.g. 10 % of a new arm's own cycle), is below the CUSUM's floor: the pooled CUSUM drifts down at that rate, and
in the synthetic test 10 % sweeps are not (all) detected within 2 h (tests/test_change_detect.py::
test_sensitivity_floor_10pct_sweep_is_not_detected_in_2h). Such cells stay FREE until the forgetting absorbs them:
the decayed p tends to 0.68 x the occupied share, so a 10 % cell reaches p_min after ~1.9 half-lives (~2 days at the
default 24 h), and a cell occupied less than p_min / 0.68 = 7.4 % of the time never does. On the body-shop map (where
the broadcast is the undecayed envelope, point 4) only the map's own counts take it in. In the same synthetic test 30 %
and 50 % sweeps are detected within 5 min (max 294 s and 239 s).

What the caller decides (and is documented by the caller, e.g. twin/observer_live.py):
- the grid (a machine's 0.25 m L1 envelope grid, the twin's floor grid) and the slices (phase bins, time-of-day bins;
  1 slice if unsliced); `half_life_ticks` counts the observed ticks of ONE (slice, cell), so with S slices visited
  evenly and full coverage the wall-clock half-life is S * half_life_ticks / hz. The default is 24 h of wall clock at
  2 Hz spread over the slices: 172800 / n_slices observed ticks (the 24 h half-life of WEAKNESSES M-W4);
- which detections are hits ("in-zone hits": e.g. machine parts inside the zone + margin; people are not machine
  changes);
- which cells are observed on a tick: from Observation v2 (cam_floor_ok via mapping.floor_ok_of, zone_vis), never
  "all cells" by default (an occluded cell that counts as observed would decay towards FREE and could be lifted).

API (all arrays flat over the grid, length n_cells):
    layer = ChangeLayer(n_cells, n_slices=1, cfg=ChangeConfig(), shape=None)
    res = layer.step(slice_idx, hits, observed, free=None)   # StepResult(alarmed=idx, lifted=idx); free: point 4
    layer.state(slice_idx)                            # int8 FREE / OCCUPIED / UNKNOWN: what to broadcast
    layer.not_free(slice_idx)                         # state != FREE
    layer.unknown_mask()                              # bool[n_cells], dilated by cfg.dilate cells when shape is given
    layer.p(slice_idx), layer.p_lifetime(slice_idx)   # decayed / undecayed occupancy (NaN where never observed)
    layer.occupied(slice_idx), layer.trusted(slice_idx), layer.monitored(slice_idx)
    layer.visits(slice_idx), layer.S, layer.Ss, layer.unknown
    layer.slices_clear(cells)                         # bool[n_slices, len(cells)]: the per-slice lift rule (point 3)
    CD.half_life_for(wall_hours, n_slices)            # half_life_ticks for a wall-clock half-life
    CD.drift_per_tick(cfg, a0, b0, q, r)              # expected llr per tick (analytic detection delay ~ h / drift)
State is in-process only (numpy arrays on the object); nothing is persisted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HZ = 2.0
DAY_TICKS = 24 * 3600 * HZ          # 172800 observed ticks = 24 h at 2 Hz
FREE, OCCUPIED, UNKNOWN = 0, 1, 2   # `ChangeLayer.state` codes


@dataclass(frozen=True)
class ChangeConfig:
    """Constants of the in-zone detector. Defaults are set from the model, not tuned on the test streams (see
    tests/test_change_detect.py for the numbers they give)."""
    p_min: float = 0.05             # the map's FREE threshold (L1 envelope / hot-cell threshold): monitored below it
    min_visits: float = 60.0        # (decayed) observed ticks of a (slice, cell) before its null is trusted
    h: float = 20.0                 # alarm threshold (log-likelihood units); cap = cap_mult * h
    cap_mult: float = 1.5
    m: float = 4.0                  # change alternative: entries m x as frequent, stays m x as long
    b_alt: float = 0.05             # ... and at least one entry per 1 / b_alt observed ticks
    b_floor: float = 1e-3           # lower clip of the null entry probability (bounds one entry's evidence)
    b_prior: tuple = (0.5, 50.0)    # (pseudo entries, pseudo empty ticks) for b0
    a_prior: tuple = (1.0, 2.0)     # (pseudo stays, pseudo hits) for a0: prior mean 0.5
    hold: int = 2                   # run state = a hit within the last hold + 1 observed ticks (bridges detection gaps)
    clear_ticks: int = 1800         # observed ticks without an unexplained hit before UNKNOWN is lifted (15 min at 2 Hz)
    clear_visits: int = 3           # ... and, when counted over all slices, separate visits of each slice that had one
    clear_min_ticks: int = 12       # ... and observed ticks of that slice itself (module docstring, point 3)
    half_life_ticks: float | None = None   # None: DAY_TICKS / n_slices (24 h wall clock with even slice coverage)
    dilate: int = 0                 # cells added around UNKNOWN cells by unknown_mask() (needs `shape`)


@dataclass
class StepResult:
    alarmed: np.ndarray             # flat cell indices that became UNKNOWN this tick
    lifted: np.ndarray              # flat cell indices whose UNKNOWN was lifted this tick


class ChangeLayer:
    """Decayed occupancy counts + in-zone Markov CUSUM for one grid. See the module docstring."""

    def __init__(self, n_cells: int, n_slices: int = 1, cfg: ChangeConfig | None = None, shape: tuple | None = None):
        self.cfg = cfg = cfg or ChangeConfig()
        if shape is not None and int(np.prod(shape)) != n_cells:
            raise ValueError(f"shape {shape} does not hold {n_cells} cells")
        self.n_cells, self.n_slices, self.shape = int(n_cells), int(n_slices), shape
        hl = cfg.half_life_ticks if cfg.half_life_ticks is not None else DAY_TICKS / self.n_slices
        if not hl > 0:
            raise ValueError("half_life_ticks must be > 0")
        self.half_life_ticks = float(hl)
        self.lam = 2.0 ** (-1.0 / self.half_life_ticks)
        if 1.0 / (1.0 - self.lam) < cfg.min_visits:
            raise ValueError(f"half_life_ticks {hl:g}: decayed visits saturate at {1.0 / (1.0 - self.lam):.0f} < "
                             f"min_visits {cfg.min_visits:g}, no cell would ever be trusted")
        z = (self.n_slices, self.n_cells)
        self.n = np.zeros(z)            # decayed observed ticks
        self.k = np.zeros(z)            # decayed hit ticks (occupancy with forgetting p = k / n)
        self.N = np.zeros(z)            # undecayed observed ticks
        self.K = np.zeros(z)            # undecayed hit ticks (occupancy without forgetting P = K / N)
        self.kx = np.zeros(z)           # decayed run-state ticks (x = 1)
        self.e = np.zeros(z)            # decayed run entries (x: 0 -> 1)
        self.prev = np.zeros(self.n_cells, bool)       # run state x at the cell's last observed tick
        self.since = np.full(self.n_cells, 1 << 30, np.int64)      # observed ticks since the cell's last hit
        self.S = np.zeros(self.n_cells)                 # pooled CUSUM per cell (changes not locked to the slices)
        self.Ss = np.zeros(z)                           # CUSUM per (slice, cell) (a change in some slices only)
        self.unknown = np.zeros(self.n_cells, bool)
        # lift bookkeeping (module docstring, point 3), per (slice, cell) since its last unexplained hit in that slice
        never = -(1 << 40)
        self.obs_ticks = np.zeros(z, np.int64)          # observed ticks per (slice, cell)
        self.obs_all = np.zeros(self.n_cells, np.int64)            # observed ticks per cell, any slice
        self.last_unexpl = np.full(z, never, np.int64)  # obs_ticks of (slice, cell) at its last unexplained hit there
        self.last_unexpl_all = np.full(z, never, np.int64)         # obs_all of the cell at that same hit
        self.visits_since = np.zeros(z, np.int64)       # visits of the slice with the cell observed, since that hit
        self._visit = 0                                 # id of the current visit (a run of steps with one slice index)
        self._visit_slice = -1
        self._visit_seen = np.full(self.n_cells, -1, np.int64)     # last visit id in which the cell was observed
        self.cap = cfg.cap_mult * cfg.h
        self.n_alarms = 0

    # ------------------------------------------------------------------ queries
    def p(self, slice_idx: int = 0) -> np.ndarray:
        n = self.n[slice_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, self.k[slice_idx] / np.where(n > 0, n, 1.0), np.nan)

    def p_lifetime(self, slice_idx: int = 0) -> np.ndarray:
        N = self.N[slice_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(N > 0, self.K[slice_idx] / np.where(N > 0, N, 1.0), np.nan)

    def visits(self, slice_idx: int = 0) -> np.ndarray:
        return self.n[slice_idx].copy()

    def trusted(self, slice_idx: int = 0) -> np.ndarray:
        """Cells whose decayed visits in this slice are enough to trust their counts (and the CUSUM's null)."""
        return self.n[slice_idx] >= self.cfg.min_visits

    def occupied(self, slice_idx: int = 0) -> np.ndarray:
        """Occupied by the counts: decayed p >= p_min OR undecayed P >= p_min (forgetting only adds occupancy)."""
        pm = self.cfg.p_min
        return ((self.n[slice_idx] > 0) & (self.k[slice_idx] >= pm * self.n[slice_idx])) | \
               ((self.N[slice_idx] > 0) & (self.K[slice_idx] >= pm * self.N[slice_idx]))

    def monitored(self, slice_idx: int = 0, free: np.ndarray | None = None) -> np.ndarray:
        """Cells broadcast FREE in this slice (by the counts, or the given `free` broadcast) whose null is trusted: the
        CUSUM runs on them."""
        fr = ~self.occupied(slice_idx) if free is None else np.asarray(free, bool).reshape(-1)
        return self.trusted(slice_idx) & fr

    def state(self, slice_idx: int = 0, unknown: np.ndarray | None = None) -> np.ndarray:
        """int8[n_cells] FREE / OCCUPIED / UNKNOWN in this slice: what is broadcast (module docstring, point 4).
        `unknown` = a precomputed unknown_mask() (to share one dilation across slices)."""
        U = self.unknown_mask() if unknown is None else unknown
        out = np.where(self.occupied(slice_idx), OCCUPIED, FREE).astype(np.int8)
        out[U | ~self.trusted(slice_idx)] = UNKNOWN
        return out

    def not_free(self, slice_idx: int = 0, unknown: np.ndarray | None = None) -> np.ndarray:
        return self.state(slice_idx, unknown) != FREE

    def slices_clear(self, cells: np.ndarray | None = None) -> np.ndarray:
        """bool[n_slices, len(cells)]: the slice holds no recent unexplained hit for the cell (module docstring, point 3).
        The lift needs every slice clear."""
        c = self.cfg
        sel = slice(None) if cells is None else cells
        own = self.obs_ticks[:, sel] - self.last_unexpl[:, sel]
        spread = self.obs_all[sel][None, :] - self.last_unexpl_all[:, sel]
        return (own >= c.clear_ticks) | ((spread >= c.clear_ticks) & (self.visits_since[:, sel] >= c.clear_visits) &
                                         (own >= c.clear_min_ticks))

    def unknown_mask(self) -> np.ndarray:
        U = self.unknown.copy()
        d = self.cfg.dilate
        if d > 0 and self.shape is not None and U.any():
            U = _dilate(U.reshape(self.shape), d).reshape(-1)
        return U

    # ------------------------------------------------------------------ step
    def null_params(self, slice_idx: int, cells: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(a0, b0): null stay and entry probabilities in this slice (all cells, or the given flat indices)."""
        c = self.cfg
        sel = slice(None) if cells is None else cells
        n, k, e = self.n[slice_idx, sel], self.kx[slice_idx, sel], self.e[slice_idx, sel]
        b0 = (e + c.b_prior[0]) / (np.maximum(n - k, 0.0) + c.b_prior[1])
        a0 = (np.maximum(k - e, 0.0) + c.a_prior[0]) / (k + c.a_prior[1])
        return np.clip(a0, 0.02, 0.995), np.clip(b0, c.b_floor, 0.5)

    def llr(self, slice_idx: int, cells: np.ndarray, prev: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """Markov log-likelihood ratio (change vs null) of the transition prev -> hit on the given cells."""
        c = self.cfg
        a0, b0 = self.null_params(slice_idx, cells)
        b1 = np.maximum(np.minimum(np.maximum(c.m * b0, c.b_alt), 0.95), b0)
        a1 = 1.0 - (1.0 - a0) / c.m
        return np.where(prev,
                        np.where(hit, np.log(a1 / a0), np.log((1.0 - a1) / (1.0 - a0))),
                        np.where(hit, np.log(b1 / b0), np.log((1.0 - b1) / (1.0 - b0))))

    def step(self, slice_idx: int, hits: np.ndarray, observed: np.ndarray, free: np.ndarray | None = None) -> StepResult:
        """One tick. hits / observed: bool[n_cells]. A cell that is not observed is left exactly as it was.
        `free` (bool[n_cells], optional): the cells the BROADCAST shows FREE in this slice right now, when the broadcast
        is not this layer's `state()` (the body-shop LayeredMap's undecayed envelope, see change_detect_l1). Default:
        FREE by this layer's counts (not `occupied`). The CUSUM runs on trusted FREE cells, and a hit on a FREE cell is
        an unexplained hit (it blocks the lift of every UNKNOWN cell within `dilate` cells of it)."""
        if not 0 <= slice_idx < self.n_slices:
            raise IndexError(f"slice {slice_idx} outside 0..{self.n_slices - 1}")
        hits = np.asarray(hits, bool).reshape(-1)
        obs = np.asarray(observed, bool).reshape(-1)
        if hits.shape[0] != self.n_cells or obs.shape[0] != self.n_cells:
            raise ValueError("hits / observed must have n_cells entries")
        s = slice_idx
        fr = ~self.occupied(s) if free is None else np.asarray(free, bool).reshape(-1)
        if fr.shape[0] != self.n_cells:
            raise ValueError("free must have n_cells entries")
        hits = hits & obs
        idx = np.flatnonzero(obs)
        empty = np.zeros(0, np.int64)
        if not len(idx):
            return StepResult(empty, empty)
        h = hits[idx]
        pv = self.prev[idx]
        self.since[idx] = np.where(h, 0, self.since[idx] + 1)
        x = self.since[idx] <= self.cfg.hold                        # run state (gap-filled hits)
        # ---- CUSUM on monitored cells (null from the counts BEFORE this tick)
        mon = (self.trusted(s) & fr)[idx]
        alarmed, lifted = empty, empty
        if s != self._visit_slice:                       # a new visit of slice s starts
            self._visit += 1
            self._visit_slice = s
        first = idx[self._visit_seen[idx] != self._visit]
        self.visits_since[s, first] += 1
        self._visit_seen[idx] = self._visit
        self.obs_ticks[s, idx] += 1
        self.obs_all[idx] += 1
        unexpl = idx[h & fr[idx]]
        if len(unexpl):                                  # also blocks the lift of UNKNOWN cells whose dilation covers it
            u = np.zeros(self.n_cells, bool)
            u[unexpl] = True
            if self.cfg.dilate > 0 and self.shape is not None:
                u = _dilate(u.reshape(self.shape), self.cfg.dilate).reshape(-1)
            self.last_unexpl[s, u] = self.obs_ticks[s, u]          # in THIS slice only (point 3)
            self.last_unexpl_all[s, u] = self.obs_all[u]
            self.visits_since[s, u] = 0
            self._visit_seen[u] = self._visit               # the visit holding the hit is never counted
        if mon.any():
            im = idx[mon]
            r = self.llr(s, im, pv[mon], x[mon])
            self.S[im] = np.clip(self.S[im] + r, 0.0, self.cap)
            self.Ss[s, im] = np.clip(self.Ss[s, im] + r, 0.0, self.cap)
            new = im[((self.S[im] >= self.cfg.h) | (self.Ss[s, im] >= self.cfg.h)) & ~self.unknown[im]]
            if len(new):
                self.unknown[new] = True
                self.n_alarms += len(new)
                alarmed = new
        # ---- lift UNKNOWN once EVERY slice is clear (point 3)
        iu = idx[self.unknown[idx]]
        if len(iu):
            done = iu[self.slices_clear(iu).all(0)]
            if len(done):
                self.unknown[done] = False
                self.S[done] = 0.0
                self.Ss[:, done] = 0.0
                lifted = done
        # ---- counts with forgetting on observed cells of this slice only
        lam = self.lam
        self.n[s, idx] = self.n[s, idx] * lam + 1.0
        self.k[s, idx] = self.k[s, idx] * lam + h
        self.N[s, idx] += 1.0
        self.K[s, idx] += h
        self.kx[s, idx] = self.kx[s, idx] * lam + x
        self.e[s, idx] = self.e[s, idx] * lam + (x & ~pv)
        self.prev[idx] = x
        return StepResult(alarmed, lifted)


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Chebyshev dilation of a 2-D bool grid by `cells` (square structuring element)."""
    out = mask.copy()
    nx, nz = mask.shape
    for dx in range(-cells, cells + 1):
        for dz in range(-cells, cells + 1):
            if dx == 0 and dz == 0:
                continue
            src = mask[max(0, -dx):nx - max(0, dx), max(0, -dz):nz - max(0, dz)]
            out[max(0, dx):max(0, dx) + src.shape[0], max(0, dz):max(0, dz) + src.shape[1]] |= src
    return out


def half_life_for(wall_hours: float, n_slices: int, hz: float = HZ) -> float:
    """Observed ticks per (slice, cell) that give a wall-clock half-life of `wall_hours` when the slices are visited
    evenly and the cell is always observed."""
    return wall_hours * 3600.0 * hz / max(int(n_slices), 1)


def drift_per_tick(cfg: ChangeConfig, a0: float, b0: float, p_hit_after_miss: float, p_hit_after_hit: float) -> float:
    """Expected llr per observed tick of a monitored cell whose null is (a0, b0) when its hits actually follow a Markov
    chain with entry probability p_hit_after_miss and stay probability p_hit_after_hit (positive = the CUSUM climbs).
    Used to state detection delays analytically next to the simulated ones."""
    b1 = max(min(max(cfg.m * b0, cfg.b_alt), 0.95), b0)
    a1 = 1.0 - (1.0 - a0) / cfg.m
    q, r = p_hit_after_miss, p_hit_after_hit
    pi = q / (1.0 - r + q) if (1.0 - r + q) > 0 else 1.0           # stationary hit share
    return (pi * (r * math.log(a1 / a0) + (1 - r) * math.log((1 - a1) / (1 - a0)))
            + (1 - pi) * (q * math.log(b1 / b0) + (1 - q) * math.log((1 - b1) / (1 - b0))))
