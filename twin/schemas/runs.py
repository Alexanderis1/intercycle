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
import math
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


SummaryVariant = Literal["primary", "reset_sweep", "short_dev"]


class CodeHashes(Strict):
    head: str
    generator_sha256: str | None = None
    dynamics_sha256: str | None = None


class CameraProvenance(Strict):
    """The twin camera model's provenance in a run (twin/camera.py `CameraStatV1.provenance`)."""
    model: str
    cal_source: str                          # "cal_v1 draws" / "cal_v1 draws_fixed_scale" / "site framings (...)"
    h_mode: str
    draw_index: dict[str, int]               # camera id -> CAL v1 draw index (the world's true camera); {} = no draws
    hold_s: float = Field(ge=0)              # the largest per-camera hold
    hold_s_by_camera: dict[str, float]       # camera id -> the hold of the presses that camera sees (S3b fix2)
    params_sha256: str

    @model_validator(mode="after")
    def _holds(self):
        if not self.hold_s_by_camera:
            raise ValueError("hold_s_by_camera must name the hold of every camera")
        if any(not (math.isfinite(h) and h >= 0) for h in self.hold_s_by_camera.values()):
            raise ValueError("every per-camera hold must be finite and >= 0")
        if abs(max(self.hold_s_by_camera.values()) - self.hold_s) > 1e-9:
            raise ValueError("hold_s must be the largest per-camera hold")
        return self


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
    # "primary" = a full 720 s mission on the U4 lognormal reset; "reset_sweep" = the U4 descriptive reset sweep
    # (reset_override_s replaced every reset draw); "short_dev" = a dev mission shorter than T (tests, the slice).
    # Only "primary" runs may enter the primary metric (`require_primary`).
    variant: SummaryVariant = "primary"
    reset_override_s: float | None = Field(default=None, gt=0)
    # required (S3b fix2): a summary without the world's calibration draw and holds does not validate, so EVAL fails
    # closed on a run whose camera is unrecorded
    camera: CameraProvenance

    @model_validator(mode="after")
    def _counts(self):
        if self.passed > self.attempted or self.attempted + self.declined > self.checkpoints_planned:
            raise ValueError("need passed <= attempted and attempted + declined <= checkpoints_planned")
        if self.status != "ok" and not self.status_reason:
            raise ValueError("a non-ok status needs status_reason")
        if (self.variant == "reset_sweep") != (self.reset_override_s is not None) and self.variant != "short_dev":
            raise ValueError("reset_override_s is set exactly on a reset_sweep run (a short_dev run may carry it)")
        if self.variant == "short_dev" and self.world_split != "dev":
            raise ValueError("a short_dev run must be on a dev world")
        return self


def require_primary(s: Summary) -> Summary:
    """The EVAL primary path's gate (PLAN_TWIN §5): refuse a summary.json that is not a primary run (a U4 reset-sweep
    run or a short dev mission), so descriptive runs are never pooled into the primary metric."""
    if s.variant != "primary" or s.reset_override_s is not None:
        raise ValueError(f"run {s.run_id} is a {s.variant!r} run (reset_override_s={s.reset_override_s}), "
                         "not a primary run: it may not enter the primary metric")
    return s


# ------------------------------------------------------------------------------------------- primary metric JSON

Outcome = Literal["passed", "declined", "held", "tripped", "failed", "unattempted"]
# per_world floats are checked with this relative tolerance (y*3600/t and y/t*3600 differ in the last bits).
PER_WORLD_REL_TOL = 1e-9


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
        want = world_values(self.missions)
        same = len(self.per_world) == len(want) and all(
            (g.world, g.condition, g.sum_Y) == (w.world, w.condition, w.sum_Y)
            and math.isclose(g.sum_T_s, w.sum_T_s, rel_tol=PER_WORLD_REL_TOL)
            and math.isclose(g.per_h, w.per_h, rel_tol=PER_WORLD_REL_TOL, abs_tol=0.0)
            for g, w in zip(self.per_world, want))
        if not same:
            raise ValueError("per_world does not equal the pooled sum(Y)/sum(T) of the missions "
                             f"(sorted by world, condition; per_h within rel {PER_WORLD_REL_TOL})")
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

