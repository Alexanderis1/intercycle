"""Scenario loader: one YAML file is the single source of truth for every tier (sim, render, dashboard)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "scenario" / "bodyshop.yaml"


def hhmm(s: str) -> float:
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


class Break(BaseModel):
    start: str
    minutes: float

    @property
    def t0(self) -> float:
        return hhmm(self.start)

    @property
    def t1(self) -> float:
        return self.t0 + self.minutes * 60


class Shift(BaseModel):
    name: str
    start: str
    end: str


class Line(BaseModel):
    takt_s: float
    jitter: float
    drift_per_h: float
    andon_stops_per_h: float
    andon_stop_s: tuple[float, float]
    recovery_after_trip_s: float


class Machine(BaseModel):
    id: str
    kind: str
    on_line: bool
    x: float
    z: float
    size: tuple[float, float]          # (dx, dz) of the hazard zone, centred on (x, z)
    busy: float
    offset_s: float = 0.0              # line stations: stagger within the takt
    period_s: float | None = None      # off-line cells
    jitter: float | None = None
    stops_per_h: float | None = None
    stop_s: tuple[float, float] | None = None

    @property
    def rect(self) -> tuple[float, float, float, float]:
        dx, dz = self.size
        return (self.x - dx / 2, self.z - dz / 2, self.x + dx / 2, self.z + dz / 2)


class Zone(BaseModel):
    id: str
    rect: tuple[float, float, float, float]


class Inspection(BaseModel):
    entry_m: float
    dwell_s: float
    robot_speed_mps: float
    safety_margin_s: float
    local_check_latency_s: float

    @property
    def inside_s(self) -> float:
        return 2 * self.entry_m / self.robot_speed_mps + self.dwell_s


class Ble(BaseModel):
    adv_interval_s: float
    range_m: float
    packet_loss: float
    broadcasters: list[tuple[float, float]]


class Scenario(BaseModel):
    name: str
    version: int
    site_id: int
    key_id: int
    hall: dict
    # Body shop only. A press-shop twin scenario (twin/compile.py) has no data hall: both stay empty, so code that
    # reads them in twin mode fails closed (KeyError) instead of using body-shop geometry.
    data_hall: dict = Field(default_factory=dict)
    shifts: list[Shift]
    breaks: list[Break]
    shift_change_stop_min: float
    line: Line
    machines: list[Machine]
    inspection: Inspection
    zones: list[Zone]
    traffic: dict
    robot: dict
    ble: Ble
    cameras: list[tuple[float, float]]
    data_hall_targets: dict = Field(default_factory=dict)
    seeds: dict = Field(default_factory=dict)

    def machine(self, mid: str) -> Machine:
        return next(m for m in self.machines if m.id == mid)

    @property
    def site_extent(self) -> tuple[float, float, float, float]:
        """(x0, z0, x1, z1) of the whole floor: the hall, extended east to the data hall's far wall in the body shop.
        A twin zone has no data hall, so it is the hall (the zone's extent). Twin-mode code reads this instead of
        the data hall (tests/test_twin_mode_tokens.py)."""
        x0, x1 = (float(v) for v in self.hall["x"])
        z0, z1 = (float(v) for v in self.hall["z"])
        if self.data_hall:
            x1 = max(x1, float(self.data_hall["x"][1]))
        return x0, z0, x1, z1

    def line_stops(self) -> list[tuple[float, float, str]]:
        """Planned line stops within one day: breaks and shift changes, as (t0, t1, kind)."""
        out = [(b.t0, b.t1, "break") for b in self.breaks]
        for s in self.shifts:
            t = hhmm(s.start)
            out.append((t, t + self.shift_change_stop_min * 60, "shift_change"))
        return sorted(out)


def load(path: str | Path = DEFAULT) -> Scenario:
    with open(path, encoding="utf-8") as f:
        return Scenario.model_validate(yaml.safe_load(f))
