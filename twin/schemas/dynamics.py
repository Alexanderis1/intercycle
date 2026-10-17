"""`twin/dynamics.json`: the twin's dynamics parameters (owner S3a; PLAN_TWIN §2, §5 DYN/DAYGEN).

Two blocks. OBSERVED: what is (or will be) fitted from train clips (path shapes, speeds, group size, forklift paths,
stroke impulse, in-clip intervals, P(press running), layout states, day/night share). ASSUMED: stated assumptions
(arrival rates, staffing, stops, violation share, social-force defaults). Every parameter has a source
(fitted | fitted_censored | prior | assumed), a sample size n and a sweep range. In v0 every value is a prior.

Parameter names are dotted: `<entity>.<name>` (e.g. `press.P9a.stroke_period_s`, `zone.z9.walker_rate_per_h`).
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Interval, Strict

ParamSource = Literal["fitted", "fitted_censored", "prior", "assumed"]
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)+$")


class Dist(Strict):
    family: Literal["lognormal", "beta", "gamma", "exponential", "poisson", "normal", "uniform", "empirical",
                    "kaplan_meier"]
    params: dict[str, float] = Field(default_factory=dict)


class Param(Strict):
    """`value` is the point value the twin uses (for a library or distribution, a scale factor or its median);
    `sweep` is the range of the sensitivity sweep and must contain it."""
    value: float
    unit: str
    source: ParamSource
    n: int = Field(ge=0)
    sweep: Interval
    dist: Dist | None = None
    ref: str | None = None                   # e.g. path / sha256 of a resampled-segment library
    clips: list[str] = Field(default_factory=list)   # train clip ids it was fitted on
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if not self.sweep.lo <= self.value <= self.sweep.hi:
            raise ValueError(f"value {self.value} outside its sweep [{self.sweep.lo}, {self.sweep.hi}]")
        if self.source in ("fitted", "fitted_censored") and self.n <= 0:
            raise ValueError(f"a {self.source} parameter needs n > 0")
        if self.source in ("prior", "assumed") and self.clips:
            raise ValueError(f"a {self.source} parameter lists no fit clips")
        return self


class Dynamics(Strict):
    contract: Literal["twin.dynamics/1"]
    site_id: str
    version: str
    observed: dict[str, Param]
    assumed: dict[str, Param]

    @model_validator(mode="after")
    def _blocks(self):
        both = set(self.observed) & set(self.assumed)
        if both:
            raise ValueError(f"parameters in both blocks: {sorted(both)}")
        for name in [*self.observed, *self.assumed]:
            if not NAME_RE.match(name):
                raise ValueError(f"bad parameter name {name!r} (want <entity>.<name>)")
        for name, p in self.assumed.items():
            if p.source not in ("prior", "assumed"):
                raise ValueError(f"ASSUMED parameter {name} cannot be {p.source}")
        return self

    def get(self, name: str) -> Param:
        if name in self.observed:
            return self.observed[name]
        return self.assumed[name]

    def train_clips(self) -> list[str]:
        return sorted({c for p in [*self.observed.values(), *self.assumed.values()] for c in p.clips})
