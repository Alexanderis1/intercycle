"""Stand-in observer until the camera learner is wired in: past-only, model-based nowcasts.

It reads the floor's *past* state (what cameras would have seen up to `latency` ago) and predicts
forward with a slightly-wrong model. It never reads future truth. Swappable with the learner.
"""
from __future__ import annotations

import numpy as np

from intercycle.scenario import Scenario
from intercycle.sim.world import World
from intercycle.truth.machines import state_at
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State


def ttl_rule(period: float, jitter: float, stops_per_h: float) -> float:
    """Half the time until a ±10%-of-period error at 95% (random-walk phase), capped by stop rate."""
    if jitter <= 0:
        base = 300.0
    else:
        n95 = (0.1 / (1.96 * jitter)) ** 2          # cycles until 1.96·σ·√n = 10% of period
        base = 0.5 * n95 * period
    if stops_per_h > 1.0:
        base = min(base, 10.0)
    return float(np.clip(base, 2.0, 300.0))


class ShimObserver:
    def __init__(self, scn: Scenario, world: World, seed: int, latency_s: float = 0.2, fault: str | None = None):
        self.scn, self.world, self.lat, self.fault = scn, world, latency_s, fault
        rng = np.random.default_rng([seed, 99])
        self.models: list[MachineModel] = []
        for i, m in enumerate(scn.machines):
            T = scn.line.takt_s if m.on_line else float(m.period_s)
            j = scn.line.jitter if m.on_line else float(m.jitter or 0)
            sp = scn.line.andon_stops_per_h if m.on_line else float(m.stops_per_h or 0)
            self.models.append(MachineModel(i, T * (1 + 7e-4 * rng.standard_normal()), m.busy + 0.01 * rng.standard_normal(),
                                            j, scn.line.drift_per_h if m.on_line else 0.0, sp, 3 if j >= 0.05 else 12))
        self.rng = rng
        self.seq = 0

    def slow(self, t: float) -> SlowLayer:
        return SlowLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t, [[0] * 24 for _ in self.scn.zones], self.models)

    def _now(self, i: int, t_issue: float) -> MachineNow:
        m, model, w = self.scn.machines[i], self.models[i], self.world
        to = t_issue - self.lat
        g = w.group(m.id)
        mt = g.mt(to)
        _, win = state_at(w.W[m.id], mt)
        T, busy = model.period_s, model.busy_fraction
        ttl = ttl_rule(T, model.jitter, model.stops_per_h)
        if win is None:
            return MachineNow(i, State.UNKNOWN, 0, 0, ttl, 0, self.seq)
        frozen = g.frozen_until(to) is not None
        if not frozen and mt < win.is_:                       # active now (observed)
            act_start = to - (mt - win.bs)
            nin = act_start + busy * T - t_issue
            st = State.ACTIVE
            if nin < -1.0:                                    # active longer than expected
                return MachineNow(i, State.UNKNOWN, 0, 0, ttl, 2, self.seq)
            nin, nlen = max(0.0, nin), (1 - busy) * T
        else:                                                 # idle now
            f0 = max((f.t0 for f in g.freezes if f.t0 <= to < f.t1), default=None)
            idle_start = f0 if frozen else to - (mt - win.is_)
            end = idle_start + (1 - busy) * T
            if to > end + 3 * model.jitter * T + 0.5:
                return MachineNow(i, State.STOPPED, 0, 0, ttl, 12, self.seq)
            st, nin, nlen = State.IDLE, 0.0, max(0.0, end - t_issue)
        if self.fault == "phase":                             # observer bug: signed but wrong
            nin = max(0.0, nin + float(self.rng.uniform(-T / 2, T / 2)))
        if self.fault == "flip":
            st = State.IDLE if st == State.ACTIVE else State.ACTIVE
        return MachineNow(i, st, nin, nlen, ttl, model.confidence, self.seq)

    def fast(self, t_issue: float) -> FastLayer:
        self.seq = (self.seq + 1) & 0xFFFF
        return FastLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t_issue, self.seq,
                         [self._now(i, t_issue) for i in range(len(self.scn.machines))])
