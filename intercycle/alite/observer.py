"""The A-lite observer (observer side; never imports ground truth). twin/observer_live.py-style (PLAN_TWIN §2, R-W1):
the timing comes from learn_cycles + nowcast on the zone motion of a NOISY camera, never from a truth activity flag
(intercycle.sim.observer_learned has that hook and is not used by A-lite).

Input: one camera tick at a time (an Observation from mapping.CameraShim, the measured error model MEASURED_SHIM:
sigma 0.285 m, 32 % misses, 15 % gross errors, false positives, blackouts), converted to Observation v2 by S4's
ObsV2Converter (occlusion and dead cameras = MISSING, T-W6 / M-W10).

What it keeps:
- the map: the body-shop LayeredMap (L1 envelopes per phase bin, L2 / L3), fed by MapBuilder with the
  ConditionTracker (its own phase, production state);
- M-W4: S7a's in-zone change detector `change_detect_l1.L1ChangeDetector` (CUSUM on hits in FREE cells -> UNKNOWN,
  forgetting per observed tick) next to the map; its UNKNOWN boxes go into map.unknown_rects;
- T-W2: per tick the map's phase (tracker), for `phase_origin.LagCalibrator` in `learn_slow`;
- the zone motion signal per machine (t_cap, motion > 0), MISSING ticks dropped, for learn_cycles.

Slow layer (`learn_slow`): learn_cycles.learn_line on the line stations and learn on the cells, over the buffered
signal of a learning window; plus the PhaseOrigin. Fast layer (`fast`): learn_cycles.line_nowcast / nowcast on the
samples up to the issue time. Fail closed: a machine whose zone was MISSING in any of the last `missing_veto_s`
seconds gets an UNKNOWN entry (confidence 0), as does every machine before the slow layer exists.
Timestamps: the signal is stamped with the capture time, so learn_cycles runs with latency_s = 0.
"""
from __future__ import annotations

import math

import numpy as np

from intercycle.alite.phase_origin import LagCalibrator, PhaseOrigin, robot_phase
from intercycle.observer import learn_cycles as LC
from intercycle.observer import mapping as M
from intercycle.observer.change_detect_l1 import L1ChangeDetector
from intercycle.scenario import Scenario
from intercycle.types import MachineModel, MachineNow, State


class AliteObserver:
    def __init__(self, scn: Scenario, missing_veto_s: float = 2.0, change_detection: bool = True):
        self.scn = scn
        self.ids = [m.id for m in scn.machines]
        rects = [m.rect for m in scn.machines]
        self.conv = M.ObsV2Converter(M.zone_camera_share(rects, scn.cameras), rects)
        self.builder = M.MapBuilder(scn)
        self.map = self.builder.map
        self.tracker = self.builder.tracker
        self.det = L1ChangeDetector(self.map) if change_detection else None
        self.line = [i for i, m in enumerate(scn.machines) if m.on_line]
        self.cells = [i for i, m in enumerate(scn.machines) if not m.on_line]
        self.missing_veto_s = float(missing_veto_s)
        n = len(self.ids)
        self._tb = np.empty(1 << 14)                # capture times
        self._yb = np.empty((1 << 14, n))           # per tick: motion (0 / 1), NaN = MISSING
        self._pb = np.empty((1 << 14, n))           # per tick: the map's phase (tracker), NaN = unknown / STOP bin
        self._n = 0
        self.models: list[MachineModel] | None = None
        self.offsets: list[float] = [0.0] * n
        self.origin: PhaseOrigin | None = None
        self.events: list[dict] = []
        self.last_cond: M.Cond | None = None
        self.seq = 0

    # ------------------------------------------------------------------ input
    def step(self, obs: M.Observation, learn_map: bool = True) -> M.Cond:
        v2 = self.conv(obs)
        cond = self.builder.feed(v2) if learn_map else self.builder.condition(v2)
        if self.det is not None and learn_map:
            self.events += self.det.step(v2, cond)
        zm = np.asarray(v2.zone_motion, float)
        y = np.where(np.isfinite(zm), (zm > 0).astype(float), np.nan)
        ph = np.where((cond.phase_bin >= 0) & (cond.phase_bin != M.STOP_BIN), cond.phase, np.nan)
        if self._n == len(self._tb):
            self._tb = np.concatenate([self._tb, np.empty(len(self._tb))])
            self._yb = np.concatenate([self._yb, np.empty_like(self._yb)])
            self._pb = np.concatenate([self._pb, np.empty_like(self._pb)])
        self._tb[self._n] = float(v2.t_cap)
        self._yb[self._n] = y
        self._pb[self._n] = ph
        self._n += 1
        self.last_cond = cond
        return cond

    def signal(self, i: int, t0: float = -math.inf, t1: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        """(t, y) of machine i's zone motion with MISSING ticks dropped, t in [t0, t1]."""
        t = self._tb[:self._n]
        a, b = int(np.searchsorted(t, t0, side="left")), int(np.searchsorted(t, t1, side="right"))
        t, y = t[a:b], self._yb[a:b, i]
        k = np.isfinite(y)
        return t[k], y[k].astype(np.int8)

    def trim(self, keep_s: float = 3600.0) -> None:
        """Drop buffered samples older than keep_s before the newest (the fast layer looks back <= ~6 periods)."""
        if not self._n:
            return
        t = self._tb[:self._n]
        k = int(np.searchsorted(t, t[-1] - keep_s))
        self._tb[:self._n - k] = self._tb[k:self._n]
        self._yb[:self._n - k] = self._yb[k:self._n]
        self._pb[:self._n - k] = self._pb[k:self._n]
        self._n -= k

    def map_phase_at(self, t: float) -> np.ndarray:
        """The map's phase per machine at the last tick captured at or before t (NaN unknown / STOP)."""
        k = int(np.searchsorted(self._tb[:self._n], t, side="right")) - 1
        return self._pb[k].copy() if k >= 0 else np.full(len(self.ids), np.nan)

    def missing_since(self, t: float, span_s: float) -> np.ndarray:
        """bool per machine: its zone was MISSING on a tick captured in [t - span_s, t]."""
        tb = self._tb[:self._n]
        a, b = int(np.searchsorted(tb, t - span_s, side="left")), int(np.searchsorted(tb, t, side="right"))
        return np.isnan(self._yb[a:b]).any(0) if b > a else np.zeros(len(self.ids), bool)

    # ------------------------------------------------------------------ slow layer
    def learn_slow(self, t0: float, t1: float, calib_s: float = 1200.0, calib_every_s: float = 10.0) -> list[MachineModel]:
        """Slow layer from the buffered signal of [t0, t1], then the T-W2 phase origin calibrated on the last calib_s
        of the window (phase_origin.LagCalibrator: the fast layer replayed at past issue times vs the map's phase)."""
        sig = {i: self.signal(i, t0, t1) for i in range(len(self.ids))}
        learned = dict(zip(self.line, LC.learn_line([sig[i] for i in self.line], (t0, t1), self.line, latency_s=0.0)))
        for i in self.cells:
            learned[i] = LC.learn(*sig[i], window=(t0, t1), index=i, latency_s=0.0)
        self.models = [learned[i].model for i in range(len(self.ids))]
        self.offsets = [float(learned[i].diag.get("offset_s", 0.0)) for i in range(len(self.ids))]
        cal = LagCalibrator(len(self.ids))
        for ti in np.arange(max(t0, t1 - calib_s), t1, calib_every_s):
            f = self.fast(float(ti), veto=False)
            mp = self.map_phase_at(float(ti))
            for i, mod in enumerate(self.models):
                rp = robot_phase(mod, f[i], float(ti), float(ti))
                if rp is not None:
                    cal.add(i, float(mp[i]), rp % 1.0, mod.period_s)
        self.origin = cal.origin()
        return self.models

    # ------------------------------------------------------------------ fast layer
    def _span(self, i: int) -> float:
        assert self.models is not None
        return max(6 * self.models[i].period_s, 200.0) + 20.0

    def fast(self, t_issue: float, veto: bool = True) -> list[MachineNow]:
        """Fast entries (MachineNow per machine) from the samples captured up to t_issue. veto=False only for the
        T-W2 calibration replay (the robot never gets an entry without the MISSING veto)."""
        self.seq = (self.seq + 1) & 0xFFFF
        n = len(self.ids)
        if self.models is None:
            return [MachineNow(i, State.UNKNOWN, 0.0, 0.0, 0.0, 0, self.seq) for i in range(n)]
        now: dict[int, MachineNow] = {}
        sig = [self.signal(i, t_issue - self._span(i), t_issue) for i in self.line]
        for i, mn in zip(self.line, LC.line_nowcast([self.models[i] for i in self.line], sig,
                                                    [self.offsets[i] for i in self.line], t_issue, self.seq,
                                                    latency_s=0.0)):
            now[i] = mn
        for i in self.cells:
            now[i] = LC.nowcast(self.models[i], *self.signal(i, t_issue - self._span(i), t_issue), t_issue,
                                seq=self.seq, latency_s=0.0)
        miss = self.missing_since(t_issue, self.missing_veto_s) if veto else np.zeros(n, bool)
        out = []
        for i in range(n):
            if miss[i]:                                                          # fail closed (T-W6)
                out.append(MachineNow(i, State.UNKNOWN, 0.0, 0.0, 0.0, 0, self.seq))
            else:
                out.append(now[i])
        return out
