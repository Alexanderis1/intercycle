"""`twin/tasks.yaml`: checkpoints and candidate inspection poses (owner S2; R-W10, E-W3).

The TwinSiteGraph (S1b) is built from this file: one dock and one fixed mission order per zone; candidate poses are
generated from the site plan, the target normal and the camera model (R-W10) and confirmed at commissioning.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import SiteSource, Strict, unique


class Pose(Strict):
    x: float
    z: float
    yaw: float                               # rad, plan frame


class CandidatePose(Strict):
    pose: Pose
    source: Literal["prior", "generated", "confirmed"]
    standoff_m: float | None = Field(default=None, gt=0)


class Checkpoint(Strict):
    id: str
    zone: str
    press: str | None = None                 # the press whose state gates this checkpoint, if any
    target_kind: Literal["press", "fixture", "panel", "other"]
    target_xzh: tuple[float, float, float]   # target point (x, z, height above floor) in metres
    normal_xz: tuple[float, float]           # outward target normal on the floor plane
    candidate_poses: list[CandidatePose] = Field(min_length=1)
    inside_danger: bool                      # the capture pose is inside the press danger polygon (needs a window)
    dwell_s: float = Field(gt=0)
    zoom: float = Field(ge=1, le=20)
    thermal: bool
    source: SiteSource


class ZoneTasks(Strict):
    zone: str
    dock: Pose
    mission_order: list[str] = Field(min_length=1)


class Inspection(Strict):
    defect_rate: float = Field(ge=0, le=1)   # 0.25 per PLAN_TWIN §4 (S2 TASKS)
    zoom_max: float = Field(ge=1, le=20)
    detectability: str = Field(min_length=1)  # the written detectability formula (E-W3)


class Tasks(Strict):
    contract: Literal["twin.tasks/1"]
    site_id: str
    inspection: Inspection
    zones: list[ZoneTasks] = Field(min_length=1)
    checkpoints: list[Checkpoint] = Field(min_length=1)

    @model_validator(mode="after")
    def _refs(self):
        unique([c.id for c in self.checkpoints], "checkpoint")
        unique([z.zone for z in self.zones], "zone")
        zone_ids = {z.zone for z in self.zones}
        for c in self.checkpoints:
            if c.zone not in zone_ids:
                raise ValueError(f"checkpoint {c.id}: zone {c.zone} has no ZoneTasks entry")
            if c.inside_danger and c.press is None:
                raise ValueError(f"checkpoint {c.id}: inside_danger requires the gating press")
        for z in self.zones:
            mine = {c.id for c in self.checkpoints if c.zone == z.zone}
            unique(z.mission_order, f"mission_order of zone {z.zone}")
            if set(z.mission_order) != mine:
                raise ValueError(f"zone {z.zone}: mission_order {z.mission_order} != its checkpoints {sorted(mine)}")
        return self
