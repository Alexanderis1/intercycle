"""Robot B's broadcast from the LEARNED observer (replaces ShimObserver for every result).

Two halves, as on a real site:
- Slow layer: cycle models learned by `observer/learn_cycles.py` from the camera signal of a separate
  TRAINING day (seed `TRAIN_SEED`, never an evaluation seed). The learner only ever sees (t, y).
- Fast layer: `learn_cycles.line_nowcast` (line stations, pooled) and `nowcast` (cells) on the episode's own
  ceiling-camera signal, using only samples up to the issue time.

The camera (this module's sim half) samples the floor as the robot changes it (holds and trips freeze the line)
at 2 Hz with the same noise model as `observer/signals.py`: 0.2 s latency, occlusion bursts, false flips.
It reads the episode's world state, as a camera would; nothing here looks into the future.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from intercycle.observer import learn_cycles as LC
from intercycle.observer import signals
from intercycle.scenario import Scenario
from intercycle.sim.world import World
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State

TRAIN_SEED = 99                      # training day: inside the train range [0, 99], disjoint from dev seeds 0-19 and held-out 1000+
TRAIN_WINDOW = (6 * 3600 + 600.0, 10 * 3600.0)   # 06:10-10:00 of the training day (includes the 09:00 break)
CACHE = Path(__file__).resolve().parents[2] / "out" / "cache"


@lru_cache(maxsize=16)
def _train(scn_json: str, seed: int, t0: float, t1: float) -> tuple[tuple[MachineModel, ...], tuple[float, ...]]:
    key = hashlib.sha1(f"{scn_json}|{seed}|{t0}|{t1}|{Path(LC.__file__).stat().st_mtime_ns}".encode()).hexdigest()[:16]
    f = CACHE / f"observer_models_{key}.json"
    if f.exists():                                   # trained once per scenario; sweeps reuse it across processes
        d = json.loads(f.read_text())
        return tuple(MachineModel(**m) for m in d["models"]), tuple(d["offsets"])
    models, offs = _fit(Scenario.model_validate_json(scn_json), seed, t0, t1)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp")
    tmp.write_text(json.dumps({"models": [asdict(m) for m in models], "offsets": list(offs)}))
    tmp.replace(f)
    return models, offs


def _fit(scn: Scenario, seed: int, t0: float, t1: float) -> tuple[tuple[MachineModel, ...], tuple[float, ...]]:
    ids = [m.id for m in scn.machines]
    line = [m.id for m in scn.machines if m.on_line]
    sig = {i: signals.machine_signal(scn, i, seed, t0, t1) for i in ids}
    learned = dict(zip(line, LC.learn_line([sig[i] for i in line], (t0, t1), [ids.index(i) for i in line])))
    for i in ids:
        if i not in learned:
            learned[i] = LC.learn(*sig[i], window=(t0, t1), index=ids.index(i))
    return (tuple(learned[i].model for i in ids), tuple(float(learned[i].diag.get("offset_s", 0.0)) for i in ids))


def learned_models(scn: Scenario, seed: int = TRAIN_SEED, window: tuple[float, float] = TRAIN_WINDOW):
    """(models, line offsets) learned from the training day's camera signals. Cached per scenario."""
    return _train(scn.model_dump_json(), seed, *window)


class _Camera:
    """Motion-detector output for one zone of the episode, extended lazily and re-read after the robot freezes the floor."""

    def __init__(self, world: World, mid: str, seed: int, t_start: float, t_end: float,
                 hz: float = signals.HZ, latency_s: float = signals.LATENCY_S):
        self.world, self.mid, self.hz, self.lat = world, mid, hz, latency_s
        n = int(round((t_end - t_start) * hz)) + 1
        self.t = t_start + np.arange(n) / hz
        rng = np.random.default_rng([seed, 4, sum(map(ord, mid))])
        self.noise_occ = signals._occlusion(n, rng, signals.P_OCCLUDE, signals.BURST_SAMPLES)
        self.noise_flip = rng.random(n) < signals.P_FLIP
        self.y = np.zeros(n, np.int8)
        self.n_done = 0
        self.n_freeze = 0

    def _invalidate(self) -> None:
        fr = self.world.group(self.mid).freezes
        if len(fr) != self.n_freeze:
            t_new = min(f.t0 for f in fr[self.n_freeze:])
            self.n_done = min(self.n_done, int(np.searchsorted(self.t, t_new + self.lat, side="left")))
            self.n_freeze = len(fr)

    def upto(self, t_now: float, span_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Samples with t in [t_now - span_s, t_now]."""
        self._invalidate()
        hi = int(np.searchsorted(self.t, t_now, side="right"))
        if self.n_done < hi:                                  # never sample ahead of t_now: the robot may still freeze the floor
            a = self.n_done
            act = self.world.active_many(self.mid, self.t[a:hi] - self.lat)
            self.y[a:hi] = ((act & ~self.noise_occ[a:hi]) ^ self.noise_flip[a:hi]).astype(np.int8)
            self.n_done = hi
        lo = int(np.searchsorted(self.t, t_now - span_s, side="left"))
        return self.t[lo:hi], self.y[lo:hi]


class LearnedObserver:
    """Same interface as the old ShimObserver: `.models`, `.slow(t)`, `.fast(t_issue)`."""

    def __init__(self, scn: Scenario, world: World, seed: int, t0: float, t1: float, fault: str | None = None,
                 train_seed: int = TRAIN_SEED, train_window: tuple[float, float] = TRAIN_WINDOW):
        if seed == train_seed:
            raise ValueError(f"evaluation seed {seed} is the observer's training day")
        self.scn, self.world, self.fault = scn, world, fault
        models, offs = learned_models(scn, train_seed, train_window)
        self.models: list[MachineModel] = list(models)
        self.offsets = list(offs)
        self.line = [i for i, m in enumerate(scn.machines) if m.on_line]
        self.cells = [i for i, m in enumerate(scn.machines) if not m.on_line]
        self.cams = [_Camera(world, m.id, seed, t0 - 1800.0, t1 + 60.0) for m in scn.machines]
        self.rng = np.random.default_rng([seed, 99])
        self.seq = 0

    def slow(self, t: float) -> SlowLayer:
        return SlowLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t, [[0] * 24 for _ in self.scn.zones], self.models)

    def _span(self, i: int) -> float:
        return max(6 * self.models[i].period_s, 200.0) + 20.0     # what forecast() looks at, plus margin

    def fast(self, t_issue: float, want: int | None = None) -> FastLayer:
        """Fast layer issued at t_issue. `want` = the one machine the caller reads: the line is always nowcast
        together (pooled), a cell only when asked; entries not computed are UNKNOWN (conf 0)."""
        self.seq = (self.seq + 1) & 0xFFFF
        now: dict[int, MachineNow] = {}
        if want is None or want in self.line:
            sig = [self.cams[i].upto(t_issue, self._span(i)) for i in self.line]
            for i, mn in zip(self.line, LC.line_nowcast([self.models[i] for i in self.line], sig,
                                                        [self.offsets[i] for i in self.line], t_issue, self.seq)):
                now[i] = mn
        for i in self.cells:
            if want is None or want == i:
                now[i] = LC.nowcast(self.models[i], *self.cams[i].upto(t_issue, self._span(i)), t_issue, seq=self.seq)
        out = [self._fault(now[i]) if i in now else MachineNow(i, State.UNKNOWN, 0.0, 0.0, 0.0, 0, self.seq)
               for i in range(len(self.models))]
        return FastLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t_issue, self.seq, out)

    def _fault(self, mn: MachineNow) -> MachineNow:
        if self.fault == "phase":                             # observer bug: signed but wrong
            T = self.models[mn.index].period_s
            return MachineNow(mn.index, mn.state, max(0.0, mn.next_idle_in_s + float(self.rng.uniform(-T / 2, T / 2))),
                              mn.next_idle_len_s, mn.ttl_s, mn.confidence, mn.seq)
        if self.fault == "flip" and mn.state in (State.ACTIVE, State.IDLE):
            st = State.IDLE if mn.state == State.ACTIVE else State.ACTIVE
            return MachineNow(mn.index, st, mn.next_idle_in_s, mn.next_idle_len_s, mn.ttl_s, mn.confidence, mn.seq)
        return mn


def true_models(scn: Scenario) -> list[MachineModel]:
    """The scenario's own cycle parameters: what a PLC integration would know (oracle C only)."""
    out = []
    for i, m in enumerate(scn.machines):
        if m.on_line:
            T, j, d, sp = scn.line.takt_s, scn.line.jitter, scn.line.drift_per_h, scn.line.andon_stops_per_h
        else:
            T, j, d, sp = float(m.period_s), float(m.jitter or 0.0), 0.0, float(m.stops_per_h or 0.0)
        out.append(MachineModel(i, T, m.busy, j, d, sp, 15))
    return out
