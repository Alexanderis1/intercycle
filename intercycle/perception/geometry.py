"""Camera calibration, image -> floor projection and zone polygons.

Pure numpy; no torch/av import so it is usable (and unit-testable) anywhere.

Calibration format (NVIDIA PhysicalAI-SmartSpaces ``MTMC_Tracking_2026/*/calibration.json``)::

    {"sensors": [{"id": "Camera_0000",
                  "intrinsicMatrix": 3x3, "extrinsicMatrix": 3x4,
                  "cameraMatrix": 3x4,     # world (x, y, z, 1) -> pixel (homogeneous)
                  "homography": 3x3,       # floor (x, y, 1)   -> pixel (homogeneous)
                  "attributes": [{"name": "fps"|"frameWidth"|"frameHeight", "value": "..."}]}]}

World units are metres, floor plane is z = 0.  ``homography`` is the floor->image
map, i.e. columns [0, 1, 3] of ``cameraMatrix``; the image->floor map is its
inverse (verified against ground-truth 3D locations of persons, see
tests/test_perception.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CameraCalib",
    "Zone",
    "load_calibration",
    "foot_points",
    "load_zones",
    "save_zones",
    "grid_zones",
    "points_in_polygon",
    "assign_zones",
]


@dataclass(frozen=True)
class CameraCalib:
    cam_id: str
    width: int
    height: int
    fps: float
    P: np.ndarray  # (3, 4) world -> pixel
    H: np.ndarray  # (3, 3) floor (x, y, 1) -> pixel
    K: np.ndarray | None = None

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def image_to_floor(self, uv: np.ndarray) -> np.ndarray:
        """Map (N, 2) pixel coordinates to (N, 2) floor coordinates in metres."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        pts = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ self.H_inv.T
        return pts[:, :2] / pts[:, 2:3]

    def floor_to_image(self, xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) floor coordinates (metres) to (N, 2) pixel coordinates."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        pts = np.concatenate([xy, np.ones((len(xy), 1))], axis=1) @ self.H.T
        return pts[:, :2] / pts[:, 2:3]

    def world_to_image(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        pts = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1) @ self.P.T
        return pts[:, :2] / pts[:, 2:3]


def _attr(sensor: dict, name: str, default: float) -> float:
    for a in sensor.get("attributes", []):
        if a.get("name") == name:
            try:
                return float(a["value"])
            except (TypeError, ValueError):
                break
    return default


def load_calibration(path_or_dict: str | Path | dict) -> dict[str, CameraCalib]:
    """Load a SmartSpaces ``calibration.json`` into ``{camera_id: CameraCalib}``."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        with open(path_or_dict, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    out: dict[str, CameraCalib] = {}
    for s in raw.get("sensors", []):
        if s.get("type", "camera") != "camera":
            continue
        P = np.asarray(s["cameraMatrix"], dtype=np.float64)
        if "homography" in s:
            H = np.asarray(s["homography"], dtype=np.float64)
        else:
            H = P[:, [0, 1, 3]]
        K = np.asarray(s["intrinsicMatrix"], dtype=np.float64) if "intrinsicMatrix" in s else None
        out[s["id"]] = CameraCalib(
            cam_id=s["id"],
            width=int(_attr(s, "frameWidth", 1920)),
            height=int(_attr(s, "frameHeight", 1080)),
            fps=_attr(s, "fps", 30.0),
            P=P,
            H=H,
            K=K,
        )
    return out


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each (N, 4) xyxy box: the image point where the object meets the floor."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) * 0.5, xyxy[:, 3]], axis=1)


# --------------------------------------------------------------------------- zones


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: np.ndarray  # (M, 2) floor coordinates, metres


def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test. ``xy`` (N, 2), ``poly`` (M, 2) -> bool (N,)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(poly, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def assign_zones(xy: np.ndarray, zones: list[Zone]) -> np.ndarray:
    """Index of the first zone containing each point, -1 if none. Returns int array (N,)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    idx = np.full(len(xy), -1, dtype=np.int64)
    for k, z in enumerate(zones):
        free = idx < 0
        if not free.any():
            break
        hit = points_in_polygon(xy[free], z.polygon)
        sel = np.flatnonzero(free)[hit]
        idx[sel] = k
    return idx


def grid_zones(bounds: tuple[float, float, float, float], nx: int, ny: int, prefix: str = "Z") -> list[Zone]:
    """Regular ``nx`` x ``ny`` grid of rectangular zones over ``bounds = (xmin, ymin, xmax, ymax)``."""
    xmin, ymin, xmax, ymax = bounds
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    zones: list[Zone] = []
    for j in range(ny):
        for i in range(nx):
            poly = np.array(
                [[xs[i], ys[j]], [xs[i + 1], ys[j]], [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]],
                dtype=np.float64,
            )
            zones.append(Zone(f"{prefix}{j * nx + i:02d}", poly))
    return zones


def load_zones(path_or_obj: str | Path | dict | list) -> list[Zone]:
    """Load zones from JSON: ``{"zones": [{"name": str, "polygon": [[x, y], ...]}, ...]}`` (or the bare list)."""
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    else:
        obj = path_or_obj
    items = obj["zones"] if isinstance(obj, dict) else obj
    return [Zone(str(z["name"]), np.asarray(z["polygon"], dtype=np.float64)) for z in items]


def save_zones(zones: list[Zone], path: str | Path, meta: dict | None = None) -> None:
    obj = {"zones": [{"name": z.name, "polygon": z.polygon.tolist()} for z in zones]}
    if meta:
        obj["meta"] = meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
