"""Vehicle rule for the press-shop twin (PLAN_TWIN §3 probe P5, lane S3a).

Rule: suppress any person box whose foot point lies inside a forklift box. A seated or standing forklift driver is
inside the vehicle's box, so the person detector's box for the driver would otherwise create a pedestrian track that
moves at vehicle speed.

The foot point is the bottom-centre of the person box, ``((x1 + x2) / 2, y2)``, the same convention used to project
person tracks to the floor. "Inside" includes the forklift box border.
"""
from __future__ import annotations

import numpy as np

__all__ = ["foot_points", "suppress_people_on_forklifts"]


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre (u, v) of each box in an (N, 4) xyxy array, as an (N, 2) float array."""
    b = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], axis=1)


def suppress_people_on_forklifts(person_xyxy: np.ndarray, forklift_xyxy: np.ndarray) -> np.ndarray:
    """Boolean keep-mask over the person boxes: False where the foot point is inside any forklift box.

    Both inputs are (N, 4) / (M, 4) xyxy arrays in the same pixel frame. With no forklift boxes every person is kept.
    """
    feet = foot_points(person_xyxy)
    fk = np.asarray(forklift_xyxy, dtype=float).reshape(-1, 4)
    if len(feet) == 0 or len(fk) == 0:
        return np.ones(len(feet), dtype=bool)
    u, v = feet[:, 0:1], feet[:, 1:2]
    inside = (u >= fk[None, :, 0]) & (u <= fk[None, :, 2]) & (v >= fk[None, :, 1]) & (v <= fk[None, :, 3])
    return ~inside.any(axis=1)
