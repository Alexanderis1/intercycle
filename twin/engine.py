"""Twin engine (lane S1b; PLAN_TWIN §2 "Day event script" and "runs/<id>/realised_state.jsonl").

Executes one day event script (`twin.schemas.dayscript.DayScript`, TRUTH) and produces the realised state at 2 Hz
(`twin.schemas.runs.RealisedHeader` + `RealisedTick`), which the plant (S5a) writes to `runs/<id>/realised_state.jsonl`
and EVAL scores against. numpy, pydantic (via twin.schemas) and the standard library only (Isaac-deps rule,
tests/test_isaac_deps.py). The engine draws NO random numbers: every draw was made by the day generator from
`twin.rng.keyed_rng(world_seed, day, entity, event)` (W-W1), so the stream is a pure function of
(script, site, stroke impulse, freezes).

Determinism and order independence:
* `tick(t)` is a pure function of t and the freezes applied so far: ticks can be evaluated in any order (and twice)
  and give the same values; two engines on the same inputs give byte-identical streams.
* `apply_freeze` calls commute: the frozen time is the union of the freeze intervals and the displayed state is
  chosen by a fixed precedence (tripped > resetting > held), never by call order.
* movers are emitted sorted by id; an entity's path depends only on its own intent and segment.

Press model (W-W3: the engine adds no stops of its own; every stop, short or long, comes from the script):
* the script's press events are pinned to the wall clock (absolute time = script.t0_s + event time). A `running`
  event strokes at production time tau = phase_s + k * period_s, where tau(t) = (t - event start) minus the frozen
  time inside the event up to t. A freeze therefore stops the strokes and delays every later stroke of the same
  running bout by the frozen time (time lost), but never moves a scheduled event boundary.
* ram (0 = top dead centre, 1 = bottom): one stroke is the impulse 0.5 * (1 - cos(2 pi s / d)) for s in [0, d),
  d = `press.stroke_impulse_s` (dynamics; 0.6 s prior), capped at the period; 0 between strokes and while stopped.
  During a freeze the ram holds the position it had when the freeze began (a protective stop halts the ram
  mid-motion).
* `stroke` is True on the tick whose interval (t - 0.5, t] contains a stroke start. A stroke due while the press is
  frozen (including exactly at the freeze's start) starts when the freeze ends; strokes due at or after the running
  event's end are dropped.
* state: `running` / `stopped` from the script event (the engine never emits `idle`); during a freeze the freeze's
  state (`tripped`, `resetting` or `held`) with `freeze` True.

apply_freeze(press, t0, t1, state) (W-W4 / R-W2): the plant decides WHEN a robot-caused stop, a manual reset or a
hold happens and how long it lasts (safety.yaml: trip -> lognormal manual reset, U4); the engine only executes it.
Times are absolute sim seconds (the realised stream's clock), interval [t0, t1). A freeze may start at the latest
tick already emitted (the plant re-emits that tick after deciding) but never earlier: rewriting emitted truth fails
closed.

Movers: intent i is present for absolute t in [t0_s + t_start, t0_s + t_start + duration / speed_scale]; its position
is the segment polyline at local time (t - start) * speed_scale; speed = leg length / leg time * speed_scale;
heading 0 = +z, clockwise towards +x (atan2(dx, dz)), carried over zero-length legs (pauses); zone = the first
site zone whose visible floor polygon contains it, else None. Radii: person 0.3 m, forklift 1.0 m (priors, the
G0 fixture's values).
"""
from __future__ import annotations

import bisect
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import numpy as np

from twin.schemas.common import TICK_HZ
from twin.schemas.dayscript import DayScript, PressEvent
from twin.schemas.dynamics import Dynamics
from twin.schemas.runs import RealisedHeader, RealisedMover, RealisedPress, RealisedTick
from twin.schemas.site import Site, point_in_polygon
from twin.truth_guard import open_truth

DT = 1.0 / TICK_HZ
DEFAULT_IMPULSE_S = 0.6                       # dynamics_v0 `press.stroke_impulse_s` prior
MOVER_R = {"person": 0.3, "forklift": 1.0}    # m, priors (G0 fixture values)
FREEZE_STATES = ("tripped", "resetting", "held")   # precedence order, highest first
ROUND = 6                                     # decimals written to the stream
EPS = 1e-9


@dataclass(frozen=True)
class Freeze:
    press: str
    t0: float
    t1: float
    state: str


class _Press:
    """One press: its script events in absolute time and its freezes."""

    def __init__(self, pid: str, events: list[PressEvent], t0_s: float, impulse_s: float):
        self.id = pid
        self.events = events
        self.e0 = [t0_s + e.t0 for e in events]
        self.e1 = [t0_s + e.t1 for e in events]
        self.impulse = impulse_s
        self.freezes: list[Freeze] = []
        self.union: list[tuple[float, float]] = []
        self._strokes: list[float] | None = None

    def add(self, f: Freeze) -> None:
        self.freezes.append(f)
        self._strokes = None
        iv = sorted((x.t0, x.t1) for x in self.freezes)
        out: list[list[float]] = []
        for a, b in iv:
            if out and a <= out[-1][1] + EPS:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        self.union = [(a, b) for a, b in out]

    # ---------------------------------------------------------------------------------------------- freezes

    def frozen_component(self, t: float) -> tuple[float, float] | None:
        for a, b in self.union:
            if a <= t < b:
                return a, b
        return None

    def freeze_state(self, t: float) -> str | None:
        here = {f.state for f in self.freezes if f.t0 <= t < f.t1}
        return next((s for s in FREEZE_STATES if s in here), None)

    def frozen_measure(self, a: float, b: float) -> float:
        """Length of the frozen union inside [a, b]."""
        return sum(max(0.0, min(b, y) - max(a, x)) for x, y in self.union if y > a and x < b)

    # ---------------------------------------------------------------------------------------------- events

    def event_index(self, t: float) -> int:
        i = bisect.bisect_right(self.e0, t) - 1
        return min(max(i, 0), len(self.events) - 1)

    def tau(self, i: int, t: float) -> float:
        return (t - self.e0[i]) - self.frozen_measure(self.e0[i], t)

    def _period_impulse(self, e: PressEvent) -> tuple[float, float]:
        p = float(e.period_s or 0.0)
        return p, min(self.impulse, p)

    def ram_free(self, t: float) -> float:
        i = self.event_index(t)
        e = self.events[i]
        if e.kind != "running":
            return 0.0
        p, d = self._period_impulse(e)
        u = self.tau(i, t) - float(e.phase_s or 0.0)
        if u < 0:
            return 0.0
        s = u - math.floor(u / p) * p
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * s / d)) if s < d else 0.0

    def ram(self, t: float) -> float:
        c = self.frozen_component(t)
        return self.ram_free(c[0] if c else t)

    def wall_of(self, i: int, tau: float) -> float:
        """Earliest NON-frozen wall time in running event i at which the production time reaches tau (a stroke due
        at a freeze's start, or while frozen, starts when the freeze ends)."""
        cur, rem = self.e0[i], tau
        for a, b in self.union:
            if b <= cur:
                continue
            gap = max(a, cur) - cur
            if rem < gap:
                break
            rem -= gap
            cur = b
        return cur + rem

    def stroke_times(self) -> list[float]:
        """Absolute stroke start times, sorted; strokes due at or after a running event's end are dropped (events
        are [t0, t1)). Cached until the next freeze."""
        if self._strokes is None:
            out: list[float] = []
            for i, e in enumerate(self.events):
                if e.kind != "running":
                    continue
                p, phase = self._period_impulse(e)[0], float(e.phase_s or 0.0)
                k = 0
                while True:
                    w = self.wall_of(i, phase + k * p)
                    if w >= self.e1[i] - EPS:
                        break
                    out.append(w)
                    k += 1
            self._strokes = sorted(out)
        return self._strokes

    def stroke_in(self, ta: float, tb: float) -> bool:
        """A stroke starts in the wall interval (ta, tb]."""
        st = self.stroke_times()
        j = bisect.bisect_right(st, ta + EPS)
        return j < len(st) and st[j] <= tb + EPS

    def state(self, t: float) -> tuple[str, bool]:
        fs = self.freeze_state(t)
        if fs is not None:
            return fs, True
        return ("running" if self.events[self.event_index(t)].kind == "running" else "stopped"), False


class _Mover:
    def __init__(self, entity: str, cls: str, start: float, scale: float, pts: list[tuple[float, float, float]]):
        a = np.asarray(pts, dtype=float)
        self.id, self.cls, self.start, self.scale = entity, cls, start, scale
        self.dt, self.x, self.z = a[:, 0], a[:, 1], a[:, 2]
        self.end = start + float(self.dt[-1]) / scale
        dx, dz, dtt = np.diff(self.x), np.diff(self.z), np.diff(self.dt)
        length = np.hypot(dx, dz)
        self.speed = length / dtt * scale
        head = np.where(length > EPS, np.arctan2(dx, dz), np.nan)
        last = next((float(h) for h in head if not math.isnan(h)), 0.0)
        filled = []
        for h in head:                                    # carry the heading over pauses (zero-length legs)
            last = last if math.isnan(h) else float(h)
            filled.append(last)
        self.heading = np.asarray(filled)

    def at(self, t: float) -> tuple[float, float, float, float] | None:
        if t < self.start - EPS or t > self.end + EPS:
            return None
        s = min(max((t - self.start) * self.scale, 0.0), float(self.dt[-1]))
        j = int(np.searchsorted(self.dt, s, side="right")) - 1
        j = min(max(j, 0), len(self.dt) - 2)
        f = (s - self.dt[j]) / (self.dt[j + 1] - self.dt[j])
        x = float(self.x[j] + f * (self.x[j + 1] - self.x[j]))
        z = float(self.z[j] + f * (self.z[j + 1] - self.z[j]))
        return x, z, float(self.heading[j]), float(self.speed[j])


ExtraMovers = Callable[[float], Iterable[RealisedMover]]


class Engine:
    """Executes one DayScript; see the module docstring for the semantics."""

    def __init__(self, script: DayScript, site: Site, dynamics: Dynamics | None = None,
                 stroke_impulse_s: float | None = None):
        if stroke_impulse_s is None:
            stroke_impulse_s = (float(dynamics.get("press.stroke_impulse_s").value) if dynamics is not None
                                else DEFAULT_IMPULSE_S)
        if not stroke_impulse_s > 0:
            raise ValueError("stroke_impulse_s must be > 0")
        zones = {z.id for z in site.zones}
        if set(script.zones) - zones:
            raise ValueError(f"script zones {sorted(set(script.zones) - zones)} are not in the site")
        if set(script.presses) - {p.id for p in site.presses}:
            raise ValueError("script presses are not in the site")
        if abs(script.t0_s * TICK_HZ - round(script.t0_s * TICK_HZ)) > 1e-6:
            raise ValueError(f"script t0_s={script.t0_s} is not on the {TICK_HZ} Hz grid (the stream's first tick "
                             "would not be t0_s)")
        self.script, self.site = script, site
        self.t0 = float(script.t0_s)
        self.t_end = self.t0 + float(script.duration_s)
        self.presses = {pid: _Press(pid, [e for e in script.press_events if e.press == pid], self.t0,
                                    float(stroke_impulse_s)) for pid in script.presses}
        seg = {s.id: s for s in script.segments}
        self.movers = sorted((_Mover(m.entity, m.cls, self.t0 + m.t_start, m.speed_scale, seg[m.segment].points)
                              for m in script.mover_intents), key=lambda m: m.id)
        self._floors = [(z.id, z.floor_polygon.value) for z in site.zones]
        self._horizon = -math.inf                         # latest tick time emitted by tick()

    # ---------------------------------------------------------------------------------------------- freezes

    def apply_freeze(self, press: str, t0: float, t1: float, state: str = "tripped") -> None:
        """Freeze `press` over absolute [t0, t1) with the displayed `state` (tripped / resetting / held)."""
        if press not in self.presses:
            raise KeyError(f"unknown press {press!r}")
        if state not in FREEZE_STATES:
            raise ValueError(f"freeze state {state!r} not in {FREEZE_STATES}")
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
            raise ValueError(f"empty or invalid freeze [{t0}, {t1})")
        if t0 < self._horizon - EPS:
            raise ValueError(f"freeze at {t0} starts before the latest emitted tick {self._horizon}: "
                             "it would rewrite emitted truth")
        self.presses[press].add(Freeze(press, float(t0), float(t1), state))

    def freezes(self) -> list[Freeze]:
        return sorted((f for p in self.presses.values() for f in p.freezes),
                      key=lambda f: (f.press, f.t0, f.t1, f.state))

    # ---------------------------------------------------------------------------------------------- stream

    def tick_times(self, t_from: float | None = None, t_to: float | None = None) -> np.ndarray:
        """The 2 Hz grid in [t_from, t_to) (defaults: the scripted day), absolute k / 2 seconds."""
        a = self.t0 if t_from is None else max(self.t0, t_from)
        b = self.t_end if t_to is None else min(self.t_end, t_to)
        k0, k1 = math.ceil(a * TICK_HZ - EPS), math.ceil(b * TICK_HZ - EPS)
        return np.arange(k0, k1, dtype=np.int64) / TICK_HZ

    def _zone(self, x: float, z: float) -> str | None:
        return next((zid for zid, poly in self._floors if point_in_polygon(x, z, poly)), None)

    def tick(self, t: float, extra: Iterable[RealisedMover] = ()) -> RealisedTick:
        """The realised state at absolute time t (on the 2 Hz grid, inside the scripted day)."""
        if abs(t * TICK_HZ - round(t * TICK_HZ)) > 1e-6:
            raise ValueError(f"t={t} is not on the {TICK_HZ} Hz grid")
        if not self.t0 - EPS <= t < self.t_end - EPS:
            raise ValueError(f"t={t} is outside the scripted day [{self.t0}, {self.t_end})")
        presses = []
        for pid in self.script.presses:
            p = self.presses[pid]
            st, fr = p.state(t)
            presses.append(RealisedPress(id=pid, state=st, ram=round(p.ram(t), ROUND),   # type: ignore[arg-type]
                                         stroke=p.stroke_in(t - DT, t), freeze=fr))
        movers = []
        for m in self.movers:
            at = m.at(t)
            if at is None:
                continue
            x, z, h, v = at
            movers.append(RealisedMover(id=m.id, cls=m.cls, zone=self._zone(x, z), x=round(x, ROUND),  # type: ignore[arg-type]
                                        z=round(z, ROUND), heading=round(h, ROUND), speed=round(v, ROUND),
                                        r=MOVER_R[m.cls]))
        movers += list(extra)
        self._horizon = max(self._horizon, t)
        return RealisedTick(kind="tick", t=float(t), presses=presses, movers=movers)

    def ticks(self, t_from: float | None = None, t_to: float | None = None,
              extra: ExtraMovers | None = None) -> Iterator[RealisedTick]:
        for t in self.tick_times(t_from, t_to):
            yield self.tick(float(t), extra(float(t)) if extra else ())

    def header(self, run_id: str, world: int, t_from: float | None = None) -> RealisedHeader:
        """The stream header; t0_s = the time of the stream's FIRST tick: the first grid tick at or after t_from
        (default: the scripted day's start, which __init__ keeps on the grid). An empty range fails closed."""
        ts = self.tick_times(t_from)
        if len(ts) == 0:
            raise ValueError(f"no tick of the scripted day [{self.t0}, {self.t_end}) at or after t_from={t_from}")
        return RealisedHeader(kind="header", contract="twin.realised_state/1", run_id=run_id, world=world,
                              day=self.script.day, hz=TICK_HZ, t0_s=float(ts[0]), zones=list(self.script.zones),
                              presses=list(self.script.presses))

    def stroke_times(self, press: str) -> list[float]:
        """Absolute stroke start times of `press` under the freezes applied so far (truth; for EVAL / tests)."""
        return self.presses[press].stroke_times()


# -------------------------------------------------------------------------------------------------- files


def load_day_script(path: str | os.PathLike) -> DayScript:
    """Read a day event script (truth: the opening is checked and logged by twin.truth_guard)."""
    with open_truth(path, "r") as f:
        return DayScript.model_validate(json.load(f))


def _line(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


class RealisedWriter:
    """Writes runs/<id>/realised_state.jsonl: the header, then ticks on the contiguous 2 Hz grid (checked), the first
    tick at exactly header.t0_s (the first grid tick at or after t_from; default the scripted day's start).

    Fail closed for EVAL, which scores against this file: the stream is written to `<path>.partial` and moved onto
    `path` (os.replace) only when the with-block exits without an exception. A crash deletes the partial file, and a
    stale stream already at `path` (an earlier attempt) is removed when the writer opens, so a cut-off truth stream
    is never at the final path. (A hard kill leaves only the .partial, never the final file.)"""

    def __init__(self, engine: Engine, path: str | os.PathLike, run_id: str, world: int, t_from: float | None = None):
        self.engine, self.path = engine, Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.header = engine.header(run_id, world, t_from)
        self._f: IO[Any] | None = None
        self._prev: float | None = None

    def __enter__(self) -> RealisedWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._f = open_truth(self.partial, "w", newline="\n")
        self._f.write(_line(self.header))
        return self

    def write(self, tick: RealisedTick) -> None:
        if self._f is None:
            raise RuntimeError("RealisedWriter is not open")
        if self._prev is None and abs(tick.t - self.header.t0_s) > 1e-6:
            raise ValueError(f"first tick t={tick.t} is not the header's t0_s={self.header.t0_s}")
        if self._prev is not None and abs(tick.t - self._prev - DT) > 1e-6:
            raise ValueError(f"tick t={tick.t} is not {DT} s after {self._prev}")
        if [p.id for p in tick.presses] != self.header.presses:
            raise ValueError("tick presses differ from the header")
        self._f.write(_line(tick))
        self._prev = tick.t

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        if self._f is None:
            return
        self._f.close()
        self._f = None
        if et is None:
            os.replace(self.partial, self.path)
        else:
            self.partial.unlink(missing_ok=True)


def write_realised(engine: Engine, path: str | os.PathLike, run_id: str, world: int, t_from: float | None = None,
                   t_to: float | None = None, extra: ExtraMovers | None = None) -> Path:
    """Run the engine over [t_from, t_to) (default: the whole scripted day) and write the realised stream."""
    with RealisedWriter(engine, path, run_id, world, t_from) as w:
        for tk in engine.ticks(t_from, t_to, extra):
            w.write(tk)
    return Path(path)


if __name__ == "__main__":
    import argparse

    from twin.schemas import load

    ap = argparse.ArgumentParser(description="Execute a day event script into a realised_state.jsonl (truth).")
    ap.add_argument("--script", required=True, help="day event script json (name contains day_script)")
    ap.add_argument("--site", required=True, help="site yaml")
    ap.add_argument("--dynamics", help="dynamics json (stroke impulse); default 0.6 s")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--world", type=int, required=True, help="world index (not the seed)")
    ap.add_argument("--out", required=True, help="runs/<id>/realised_state.jsonl")
    a = ap.parse_args()
    eng = Engine(load_day_script(a.script), load(Site, a.site), load(Dynamics, a.dynamics) if a.dynamics else None)
    print(write_realised(eng, a.out, a.run_id, a.world))
