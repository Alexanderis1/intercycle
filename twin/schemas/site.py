"""`twin/site.yaml`: geometry only (owner S2; PLAN_TWIN §2).

Plan frame: x, z in metres on the floor (the same convention as `intercycle.observer.mapping`). Pixel coordinates
(u, v) are in the framing's native image (1920x1080 for the UNV cameras). Every value carries a source tag; floor
outside a zone's `floor_polygon` is unseen and is modelled as a wall.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Camera, Mat3, Point, Polygon, Strict, Tagged, unique


class ScaleInterval(Strict):
    """Multiplicative factor on metric coordinates; the robot runs at both ends when estimates disagree (P6)."""
    lo: float = Field(gt=0)
    mid: float = Field(gt=0)
    hi: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self):
        if not self.lo <= self.mid <= self.hi:
            raise ValueError(f"scale interval must satisfy lo <= mid <= hi, got {self.lo}, {self.mid}, {self.hi}")
        return self


class Framing(Strict):
    """One camera framing (a registered view). H maps undistorted pixels (u, v, 1) to floor (x, z, 1)."""
    id: str
    camera: Camera
    image_size_px: Tagged[tuple[int, int]]
    K: Tagged[Mat3]
    k1: Tagged[float]                       # division-model radial distortion, one per framing (GeoCalib cut)
    H_img_to_floor: Tagged[Mat3]
    mount_height_m: Tagged[float]


class RamRoi(Strict):
    framing: str
    polygon_px: Polygon


class Press(Strict):
    id: str
    zone: str
    footprint: Tagged[Polygon]
    ram_roi: Tagged[list[RamRoi]]
    danger_polygon: Tagged[Polygon]
    dilation_m: Tagged[float]               # p95 reprojection error (CAL v1); the danger polygon is grown by it


class Zone(Strict):
    id: str
    camera: Camera
    framings: list[str] = Field(min_length=1)
    floor_polygon: Tagged[Polygon]          # visible floor; everything else is a wall


class Site(Strict):
    contract: Literal["twin.site/1"]
    site_id: str
    frame: Literal["plan_xz_m"] = "plan_xz_m"
    scale: Tagged[ScaleInterval]
    cal_sources: list[str]                  # train clip ids only (compile.py asserts it against the split)
    framings: list[Framing] = Field(min_length=1)
    zones: list[Zone] = Field(min_length=2, max_length=2)
    presses: list[Press] = Field(min_length=1)

    @model_validator(mode="after")
    def _refs(self):
        unique([f.id for f in self.framings], "framing")
        unique([z.id for z in self.zones], "zone")
        unique([p.id for p in self.presses], "press")
        fr = {f.id: f for f in self.framings}
        zones = {z.id: z for z in self.zones}
        for z in self.zones:
            for fid in z.framings:
                if fid not in fr:
                    raise ValueError(f"zone {z.id}: unknown framing {fid}")
                if fr[fid].camera != z.camera:
                    raise ValueError(f"zone {z.id} (camera {z.camera}) lists framing {fid} of camera {fr[fid].camera}")
        for p in self.presses:
            if p.zone not in zones:
                raise ValueError(f"press {p.id}: unknown zone {p.zone}")
            for roi in p.ram_roi.value:
                if roi.framing not in zones[p.zone].framings:
                    raise ValueError(f"press {p.id}: ram ROI framing {roi.framing} is not a framing of zone {p.zone}")
            if p.dilation_m.value < 0:
                raise ValueError(f"press {p.id}: dilation_m must be >= 0")
        for f in self.framings:
            if abs(f.K.value[2][2]) < 1e-12:
                raise ValueError(f"framing {f.id}: K[2][2] must be non-zero")
        stray = sorted(set(self.tagged_clips()) - set(self.cal_sources))
        if stray:
            raise ValueError(f"source clips not listed in cal_sources: {stray}")
        return self

    def tagged_clips(self) -> list[str]:
        """Every clip id referenced by a source tag anywhere in the site."""
        out: list[str] = list(self.scale.clips)
        for f in self.framings:
            for t in (f.image_size_px, f.K, f.k1, f.H_img_to_floor, f.mount_height_m):
                out += t.clips
        for z in self.zones:
            out += z.floor_polygon.clips
        for p in self.presses:
            for t in (p.footprint, p.ram_roi, p.danger_polygon, p.dilation_m):
                out += t.clips
        return out

    def zone_of(self, press_id: str) -> Zone:
        p = next(p for p in self.presses if p.id == press_id)
        return next(z for z in self.zones if z.id == p.zone)


def point_in_polygon(x: float, z: float, poly: list[Point]) -> bool:
    """Even-odd ray casting (stdlib only)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if (z0 > z) != (z1 > z):
            xc = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if x < xc:
                inside = not inside
    return inside


def polygon_bbox(poly, grow: float = 0.0) -> tuple[float, float, float, float]:
    """Axis-aligned (x0, z0, x1, z1) box of a floor polygon grown by `grow` metres on every side (stdlib only, no
    truth access: the plant and the real-clip pixel adapter share it)."""
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) - grow, min(zs) - grow, max(xs) + grow, max(zs) + grow
