"""Cross-camera de-duplication of floor points and per-zone occupancy time series.

Pure numpy/pandas.  Input is a table of floor-plane points (one row per
per-camera detection/track at a sampled frame); output is occupancy per zone and
class, per sampled step and aggregated into time bins.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .geometry import Zone, assign_zones

__all__ = ["fuse_world_points", "occupancy_steps", "occupancy_bins", "auto_bin_seconds"]


def auto_bin_seconds(duration_s: float) -> float:
    """5-minute bins for long recordings, 1-minute bins for short clips."""
    return 300.0 if duration_s >= 900.0 else 60.0


def _fuse_group(xy: np.ndarray, cam: np.ndarray, score: np.ndarray, radius: float) -> list[tuple[float, float, int]]:
    order = np.argsort(-score, kind="stable")
    cents: list[np.ndarray] = []
    counts: list[int] = []
    cam_sets: list[set] = []
    for i in order:
        p = xy[i]
        best, best_d = -1, radius
        for k, c in enumerate(cents):
            if cam[i] in cam_sets[k]:
                continue  # one detection per camera per physical object
            d = float(np.hypot(c[0] - p[0], c[1] - p[1]))
            if d <= best_d:
                best, best_d = k, d
        if best < 0:
            cents.append(p.copy())
            counts.append(1)
            cam_sets.append({cam[i]})
        else:
            n = counts[best]
            cents[best] = (cents[best] * n + p) / (n + 1)
            counts[best] = n + 1
            cam_sets[best].add(cam[i])
    return [(float(c[0]), float(c[1]), n) for c, n in zip(cents, counts)]


def fuse_world_points(points: pd.DataFrame, radius_m: float = 0.8) -> pd.DataFrame:
    """Merge per-camera floor points that belong to the same physical object.

    ``points`` needs columns ``frame, cam, cls, wx, wy`` and optionally ``score``.
    Within each (frame, class), points from *different* cameras closer than
    ``radius_m`` are merged greedily in descending score order (at most one point
    per camera per merged object).  Returns ``frame, cls, wx, wy, n_cams``.
    """
    cols = ["frame", "cls", "wx", "wy", "n_cams"]
    if len(points) == 0:
        return pd.DataFrame({c: [] for c in cols})
    df = points
    if "score" not in df.columns:
        df = df.assign(score=1.0)
    rows: list[tuple] = []
    for (frame, cls), g in df.groupby(["frame", "cls"], sort=True):
        xy = g[["wx", "wy"]].to_numpy(dtype=np.float64)
        for wx, wy, n in _fuse_group(xy, g["cam"].to_numpy(), g["score"].to_numpy(dtype=np.float64), radius_m):
            rows.append((frame, cls, wx, wy, n))
    return pd.DataFrame(rows, columns=cols)


def occupancy_steps(
    points: pd.DataFrame,
    zones: Sequence[Zone],
    frames: Sequence[int],
    classes: Sequence[str],
) -> pd.DataFrame:
    """Dense per-step occupancy: ``frame, zone, cls, count`` incl. zeros for empty steps.

    ``points`` has columns ``frame, cls, wx, wy``.  ``frames`` is the full list of
    sampled frame indices (so frames with no detections contribute zeros).
    """
    zone_names = [z.name for z in zones]
    full = pd.MultiIndex.from_product([list(frames), zone_names, list(classes)], names=["frame", "zone", "cls"])
    if len(points) == 0:
        return pd.DataFrame({"count": 0}, index=full).reset_index()
    zi = assign_zones(points[["wx", "wy"]].to_numpy(dtype=np.float64), list(zones))
    p = points.assign(zone=np.where(zi >= 0, np.array(zone_names + [""], dtype=object)[zi], ""))
    p = p[(p["zone"] != "") & p["cls"].isin(list(classes))]
    cnt = p.groupby(["frame", "zone", "cls"]).size().rename("count")
    out = cnt.reindex(full, fill_value=0).reset_index()
    out["count"] = out["count"].astype(int)
    return out


def occupancy_bins(steps: pd.DataFrame, fps: float, bin_seconds: float) -> pd.DataFrame:
    """Aggregate per-step occupancy into time bins.

    Returns ``bin, bin_start_s, zone, cls, mean_count, max_count, n_steps``.
    ``mean_count`` is the time-average number of objects in the zone during the bin.
    """
    cols = ["bin", "bin_start_s", "zone", "cls", "mean_count", "max_count", "n_steps"]
    if len(steps) == 0:
        return pd.DataFrame({c: [] for c in cols})
    s = steps.assign(bin=np.floor(steps["frame"].to_numpy() / fps / bin_seconds).astype(int))
    g = s.groupby(["bin", "zone", "cls"], sort=True)["count"]
    out = g.agg(mean_count="mean", max_count="max", n_steps="size").reset_index()
    out["bin_start_s"] = out["bin"] * float(bin_seconds)
    return out[cols]
