"""`twin/safety.yaml`: guard type per press, reset distribution, entry rules (owner S5a + user; U4, W-W4, R-W2).

U4: guard types cannot be seen from 4 m, so every ASSUMED press gets a light curtain plus manual reset, with the
reset time lognormal (median 4 min, p90 15 min). The reset sweep is descriptive only.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Interval, Strict, unique

Z90 = 1.2815515655446004                     # standard normal 0.9 quantile
U4_MEDIAN_S, U4_P90_S = 240.0, 900.0
U4_SWEEP_S = (60.0, 900.0)                   # U4: the reset-time sweep is 1-15 min (descriptive)
# Every press state of the realised stream (twin/schemas/runs.py RealisedPress.state) needs exactly one entry rule.
PRESS_STATES = ("running", "idle", "stopped", "tripped", "held", "resetting")


class Guard(Strict):
    type: Literal["OBSERVED", "ASSUMED"]
    device: Literal["light_curtain", "fixed_fence", "interlocked_gate", "two_hand", "none"]
    reset: Literal["manual", "auto"]
    evidence_clips: list[str] = Field(default_factory=list)   # train clip ids showing the guard (OBSERVED only)


class ResetDist(Strict):
    family: Literal["lognormal"]
    median_s: float = Field(gt=0)
    p90_s: float = Field(gt=0)
    sweep_s: Interval                        # descriptive sweep (U4: 1-15 min)

    @model_validator(mode="after")
    def _shape(self):
        if self.p90_s <= self.median_s:
            raise ValueError(f"reset p90 {self.p90_s} must exceed the median {self.median_s}")
        return self

    @property
    def mu(self) -> float:
        return math.log(self.median_s)

    @property
    def sigma(self) -> float:
        return math.log(self.p90_s / self.median_s) / Z90


class EntryRule(Strict):
    """What the safety system does when the robot enters the (dilated) danger polygon in a press state.
    trip = robot-caused press stop (reset needed); hold = the press restart is delayed while the robot is inside;
    allow = no effect. tripped / held / resetting are the states after a trip or during a hold or a manual reset;
    they need their own rule (the plant never infers one from running / idle / stopped)."""
    press_state: Literal["running", "idle", "stopped", "tripped", "held", "resetting"]
    action: Literal["trip", "hold", "allow"]


class PressSafety(Strict):
    press: str
    guard: Guard
    reset: ResetDist
    entry_rules: list[EntryRule] = Field(min_length=len(PRESS_STATES), max_length=len(PRESS_STATES))

    @model_validator(mode="after")
    def _rules(self):
        unique([r.press_state for r in self.entry_rules], f"entry rule state of press {self.press}")
        if {r.press_state for r in self.entry_rules} != set(PRESS_STATES):
            raise ValueError(f"press {self.press}: entry rules must cover every press state {PRESS_STATES}")
        g = self.guard
        if g.type == "ASSUMED":
            if g.device != "light_curtain" or g.reset != "manual":
                raise ValueError(f"press {self.press}: an ASSUMED guard is a light curtain with manual reset (U4)")
            if g.evidence_clips:
                raise ValueError(f"press {self.press}: an ASSUMED guard has no evidence clips")
            if (self.reset.median_s, self.reset.p90_s) != (U4_MEDIAN_S, U4_P90_S):
                raise ValueError(f"press {self.press}: ASSUMED reset must be lognormal median 240 s, p90 900 s (U4)")
            if (self.reset.sweep_s.lo, self.reset.sweep_s.hi) != U4_SWEEP_S:
                raise ValueError(f"press {self.press}: ASSUMED reset sweep must be 60-900 s (U4: 1-15 min)")
        elif not g.evidence_clips:
            raise ValueError(f"press {self.press}: an OBSERVED guard needs evidence clips")
        return self


class Safety(Strict):
    contract: Literal["twin.safety/1"]
    site_id: str
    presses: list[PressSafety] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self):
        unique([p.press for p in self.presses], "press")
        return self
