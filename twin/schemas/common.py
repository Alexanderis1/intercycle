"""Shared building blocks of the twin contracts (PLAN_TWIN §2)."""
from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")

# Where a geometric / site value comes from. `prior` = placeholder or vendor prior (v0), `cal_v0` / `cal_v1` = CAL
# fits from the user's clicks (v1 = bootstrap), `traced_v0` / `traced_v1` = hand-traced on train keyframes,
# `measured` = measured from train clips by code, `assumed` = a stated assumption.
SiteSource = Literal["prior", "cal_v0", "cal_v1", "traced_v0", "traced_v1", "measured", "assumed"]

Point = tuple[float, float]                    # plan frame (x, z) in metres, or (u, v) in pixels where stated
Polygon = Annotated[list[Point], Field(min_length=3)]
Mat3 = Annotated[list[Annotated[list[float], Field(min_length=3, max_length=3)]], Field(min_length=3, max_length=3)]

Camera = Literal[9, 14]
Split = Literal["train", "dev", "test"]
Condition = Literal["no_map", "static", "learned", "reactive", "oracle"]
Engine = Literal["double", "isaac"]

MISSION_T_S = 720.0                            # fixed denominator per mission (PLAN_TWIN §5)
TICK_HZ = 2.0                                  # realised stream / observation rate


class Strict(BaseModel):
    """Base of every contract model: unknown keys are errors."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Tagged(Strict, Generic[T]):
    """A value with its source tag. `clips` lists the train clip ids it was derived from (must be in cal_sources)."""
    value: T
    source: SiteSource
    clips: list[str] = Field(default_factory=list)
    note: str | None = None


class Interval(Strict):
    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self):
        if self.hi < self.lo:
            raise ValueError(f"interval hi {self.hi} < lo {self.lo}")
        return self


def unique(ids: list[str], what: str) -> None:
    seen: set[str] = set()
    dup = [i for i in ids if i in seen or seen.add(i)]
    if dup:
        raise ValueError(f"duplicate {what} ids: {sorted(set(dup))}")
