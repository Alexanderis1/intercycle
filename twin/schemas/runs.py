"""Per-run files under `runs/<id>/` (owner S5a) and the primary-metric JSON (PLAN_TWIN §5, E-W12).

* `realised_state.jsonl` (TRUTH): line 1 is a `RealisedHeader`, then one `RealisedTick` per 0.5 s (2 Hz). Only the
  modules in `twin.truth_guard.TRUTH_READERS` may open it; the observer side reads `obs_v2.jsonl` only.
* `log.parquet`: the plant's event log, rows per `twin.schemas.tables.LogRow`.
* `summary.json`: one mission's outcome counts.
* the primary-metric JSON: per mission Y with the fixed denominator T = 720 s, a per-checkpoint outcome table
  (declined / passed / held / tripped, plus failed and unattempted so every checkpoint has a row), and the per-world
  value sum(Y) / sum(T) per hour pooled over both test days and both zones.

Observation v2 is NOT defined here: it is owned by lane S4 in `intercycle/observer/mapping.py`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import MISSION_T_S, TICK_HZ, Condition, Engine, Strict, unique

# ------------------------------------------------------------------------------------------- realised state (truth)


class RealisedHeader(Strict):
    kind: Literal["header"]
    contract: Literal["twin.realised_state/1"]
    run_id: str
    world: int = Field(ge=0)                 # world index (not the seed)
    day: int = Field(ge=1)
    hz: float = Field(ge=TICK_HZ, le=TICK_HZ)        # always 2 Hz
    t0_s: float
    zones: list[str] = Field(min_length=1)
    presses: list[str] = Field(min_length=1)


class RealisedPress(Strict):
    id: str
    state: Literal["running", "idle", "stopped", "tripped", "held", "resetting"]
    ram: float = Field(ge=0, le=1)           # ram position, 0 = top dead centre, 1 = bottom
    stroke: bool                             # a stroke started within this tick
    freeze: bool                             # apply_freeze is active (robot-caused stop or hold)


class RealisedMover(Strict):
    id: str
    cls: Literal["person", "forklift", "robot"]
    zone: str | None                         # zone whose floor contains it; None = outside every visible floor
    x: float
    z: float
    heading: float                           # rad, 0 = +z, clockwise towards +x
    speed: float = Field(ge=0)
    r: float = Field(gt=0)


class RealisedTick(Strict):
    kind: Literal["tick"]
    t: float
    presses: list[RealisedPress]
    movers: list[RealisedMover]

    @model_validator(mode="after")
    def _unique(self):
        unique([p.id for p in self.presses], "press")
        unique([m.id for m in self.movers], "mover")
        return self


def parse_realised(lines: Iterable[str]) -> tuple[RealisedHeader, Iterator[RealisedTick]]:
    """Parse a realised_state.jsonl stream lazily; the ticks iterator checks the 2 Hz grid and the press set."""
    it = iter(lines)
    header = RealisedHeader.model_validate(json.loads(next(it)))
    want = set(header.presses)
    dt = 1.0 / TICK_HZ

    def ticks() -> Iterator[RealisedTick]:
        prev = None
        for line in it:
            if not line.strip():
                continue
            tk = RealisedTick.model_validate(json.loads(line))
            if {p.id for p in tk.presses} != want:
                raise ValueError(f"tick t={tk.t}: presses {[p.id for p in tk.presses]} != header {sorted(want)}")
            if prev is not None and abs(tk.t - prev - dt) > 1e-6:
                raise ValueError(f"tick t={tk.t}: not {dt} s after the previous tick {prev}")
            prev = tk.t
            yield tk

    return header, ticks()


# ------------------------------------------------------------------------------------------------ summary.json


class CodeHashes(Strict):
    head: str
    generator_sha256: str | None = None
    dynamics_sha256: str | None = None


class Summary(Strict):
    contract: Literal["twin.summary/1"]
    run_id: str
    engine: Engine
    world: int = Field(ge=0)
    world_split: Literal["dev", "test"]
    day: int = Field(ge=1)
    zone: str
    condition: Condition
    T_s: float = Field(ge=MISSION_T_S, le=MISSION_T_S)   # fixed denominator, 720 s
    status: Literal["ok", "error", "infra_invalid"]
    status_reason: str | None = None
    checkpoints_planned: int = Field(ge=0)
    attempted: int = Field(ge=0)
    passed: int = Field(ge=0)
    declined: int = Field(ge=0)
    robot_caused_press_stops: int = Field(ge=0)
    holds: int = Field(ge=0)
    incidents: int = Field(ge=0)
    press_time_lost_s: float = Field(ge=0)
    path_length_m: float = Field(ge=0)
    code: CodeHashes
    wall_s: float | None = None

    @model_validator(mode="after")
    def _counts(self):
        if self.passed > self.attempted or self.attempted + self.declined > self.checkpoints_planned:
            raise ValueError("need passed <= attempted and attempted + declined <= checkpoints_planned")
        if self.status != "ok" and not self.status_reason:
            raise ValueError("a non-ok status needs status_reason")
        return self


# ------------------------------------------------------------------------------------------- primary metric JSON

Outcome = Literal["passed", "declined", "held", "tripped", "failed", "unattempted"]


class CheckpointOutcome(Strict):
    checkpoint: str
    outcome: Outcome


class MissionMetric(Strict):
    run_id: str
    world: int = Field(ge=0)
    day: int = Field(ge=1)
    zone: str
    condition: Condition
    T_s: float = Field(ge=MISSION_T_S, le=MISSION_T_S)   # fixed denominator, 720 s
    Y: int = Field(ge=0)
    robot_caused_press_stop: bool
    hold: bool
    incident: bool
    error: bool                              # any non-infrastructure error: Y = 0 with full T (§5 Rules)
    press_time_lost_s: float = Field(ge=0)   # stops + holds incl. reset time (H2)
    checkpoints: list[CheckpointOutcome] = Field(min_length=1)

    @property
    def zeroed(self) -> bool:
        return self.robot_caused_press_stop or self.hold or self.incident or self.error

    @model_validator(mode="after")
    def _y(self):
        unique([c.checkpoint for c in self.checkpoints], f"checkpoint of {self.run_id}")
        outs = [c.outcome for c in self.checkpoints]
        if "tripped" in outs and not self.robot_caused_press_stop:
            raise ValueError(f"{self.run_id}: a tripped checkpoint implies robot_caused_press_stop")
        if "held" in outs and not self.hold:
            raise ValueError(f"{self.run_id}: a held checkpoint implies hold")
        want = 0 if self.zeroed else outs.count("passed")
        if self.Y != want:
            raise ValueError(f"{self.run_id}: Y = {self.Y}, expected {want}")
        return self


class WorldValue(Strict):
    world: int
    condition: Condition
    sum_Y: int = Field(ge=0)
    sum_T_s: float = Field(gt=0)
    per_h: float = Field(ge=0)


class PrimaryMetric(Strict):
    contract: Literal["twin.primary_metric/1"]
    engine: Engine
    split: Literal["dev", "test"]
    missions: list[MissionMetric] = Field(min_length=1)
    per_world: list[WorldValue]

    @model_validator(mode="after")
    def _pooled(self):
        unique([m.run_id for m in self.missions], "run")
        if self.per_world != world_values(self.missions):
            raise ValueError("per_world does not equal the pooled sum(Y)/sum(T) of the missions")
        return self


def world_values(missions: list[MissionMetric]) -> list[WorldValue]:
    """Per (world, condition): sum(Y) / sum(T) per hour, pooled over days and zones; sorted by (world, condition)."""
    acc: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0, 0.0])
    for m in missions:
        a = acc[(m.world, m.condition)]
        a[0] += m.Y
        a[1] += m.T_s
    return [WorldValue(world=w, condition=c, sum_Y=int(y), sum_T_s=t, per_h=y / t * 3600.0)  # type: ignore[arg-type]
            for (w, c), (y, t) in sorted(acc.items())]

