"""The day event script (owner S3a; executed by `twin/engine.py` with `apply_freeze(press, t0, t1)`).

One file per (world, day). It holds press events, mover intents and the resampled (or, in the stub, synthetic)
segments the intents follow. Every random draw that built it came from `twin.rng.keyed_rng(world_seed, day, entity,
event)`, so an entity's events do not depend on which other entities exist or the order they are generated in (W-W1).
Times are seconds from the start of the scripted day; `t0_s` is that start in absolute sim time. This file is TRUTH.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Strict, unique

MoverClass = Literal["person", "forklift"]
RNG_SCHEME = "numpy SeedSequence([world_seed, day, blake2b64(entity), blake2b64(event)]) -> PCG64"
# A day script file is named <stem> + FILE_SUFFIX (the token marks it as truth for twin/truth_guard.py); launchers
# (scripts/slice.py, scripts/sealed_run.py) take the name from here and never spell the token themselves.
FILE_SUFFIX = ".day_script.json"


class PressEvent(Strict):
    press: str
    kind: Literal["running", "stopped"]
    t0: float = Field(ge=0)
    t1: float
    period_s: float | None = Field(default=None, gt=0)    # stroke period while running
    phase_s: float | None = Field(default=None, ge=0)     # first stroke at t0 + phase_s
    cause: Literal["production", "short_stop", "long_stop", "break"]
    source: Literal["observed", "assumed", "prior"]

    @model_validator(mode="after")
    def _shape(self):
        if self.t1 <= self.t0:
            raise ValueError(f"press event {self.press} [{self.t0}, {self.t1}] is empty")
        if self.kind == "running" and (self.period_s is None or self.phase_s is None):
            raise ValueError("a running event needs period_s and phase_s")
        if self.kind == "stopped" and self.period_s is not None:
            raise ValueError("a stopped event has no period_s")
        if self.cause == "break" and self.source != "assumed":
            raise ValueError("a break is time-of-day / shift structure: its source is assumed (P2: no timing signal)")
        return self


class Segment(Strict):
    """A path: rows (dt, x, z), dt from the segment start, strictly increasing from 0."""
    id: str
    cls: MoverClass
    source: Literal["resampled", "synthetic_stub"]
    clip_id: str | None = None
    track_id: int | None = None
    points: list[tuple[float, float, float]] = Field(min_length=2)

    @model_validator(mode="after")
    def _shape(self):
        if self.source == "resampled" and (not self.clip_id or self.track_id is None):
            raise ValueError(f"resampled segment {self.id} needs clip_id and track_id")
        dts = [p[0] for p in self.points]
        if dts[0] != 0 or any(b <= a for a, b in zip(dts, dts[1:])):
            raise ValueError(f"segment {self.id}: dt must start at 0 and increase strictly")
        return self

    @property
    def duration_s(self) -> float:
        return self.points[-1][0]


class MoverIntent(Strict):
    entity: str
    cls: MoverClass
    zone: str
    t_start: float = Field(ge=0)
    segment: str
    speed_scale: float = Field(gt=0)


class DayScript(Strict):
    contract: Literal["twin.day_script/1"]
    generator: str
    generator_version: str
    rng_scheme: Literal["numpy SeedSequence([world_seed, day, blake2b64(entity), blake2b64(event)]) -> PCG64"]
    world_seed: int = Field(ge=0)
    day: int = Field(ge=1)
    t0_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    zones: list[str] = Field(min_length=1)
    presses: list[str] = Field(min_length=1)
    press_events: list[PressEvent]
    mover_intents: list[MoverIntent]
    segments: list[Segment]

    @model_validator(mode="after")
    def _refs(self):
        unique(self.presses, "press")
        unique([s.id for s in self.segments], "segment")
        unique([m.entity for m in self.mover_intents], "mover entity")
        seg = {s.id: s for s in self.segments}
        for p in self.presses:
            ev = [e for e in self.press_events if e.press == p]
            if not ev:
                raise ValueError(f"press {p} has no events")
            if ev[0].t0 != 0 or abs(ev[-1].t1 - self.duration_s) > 1e-6:
                raise ValueError(f"press {p}: events must cover [0, duration_s]")
            for a, b in zip(ev, ev[1:]):
                if abs(b.t0 - a.t1) > 1e-6:
                    raise ValueError(f"press {p}: events must be sorted and contiguous ({a.t1} -> {b.t0})")
        for e in self.press_events:
            if e.press not in self.presses:
                raise ValueError(f"press event for unknown press {e.press}")
        for m in self.mover_intents:
            if m.segment not in seg:
                raise ValueError(f"intent {m.entity}: unknown segment {m.segment}")
            if seg[m.segment].cls != m.cls:
                raise ValueError(f"intent {m.entity}: class {m.cls} != segment class {seg[m.segment].cls}")
            if m.zone not in self.zones:
                raise ValueError(f"intent {m.entity}: unknown zone {m.zone}")
            if m.t_start >= self.duration_s:
                raise ValueError(f"intent {m.entity} starts after the day ends")
        return self
