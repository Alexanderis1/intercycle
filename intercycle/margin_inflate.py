"""M-W7 margin inflation: the ONE rule for growing the robot's map by the frozen safety margin (lane S7b).

Truth-free, numpy and the standard library only. The robot side (twin/observer_live.py, the policy) imports THIS module
and nothing of the offline calibration: it reads the frozen constant with `load_margin("twin/m7_margin.json")` and
applies it with `inflate(pred, margin, cell_m=<its grid cell size>)`. The offline calibration counts coverage with the
same rule (`radius2_cells`) and cross-checks its counts against `inflate`, so the certified risk applies to exactly what
the robot does.

The rule (written into the JSON as the `inflation` block):
  * pred = the cells the robot's map marks occupied OR UNKNOWN (UNKNOWN is treated as occupied), on a grid of square
    cells of side cell_m;
  * a cell is marked after inflation when the Euclidean distance between its centre and the centre of some pred cell
    is <= margin_m, i.e. for integer cell offsets (di, dj): di^2 + dj^2 <= radius2_cells;
  * radius2_cells = floor((margin_m / cell_m)^2 + 1e-9), an integer. Do NOT floor margin_m / cell_m itself: a radius of
    1.5 cells covers the diagonal neighbour (sqrt 2 cells), a radius of floor(1.5) = 1 does not;
  * only cells inside the grid exist; nothing outside the grid is inflated into it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "intercycle.m7_margin/v1"
KERNEL = "euclidean_disc"
DISTANCE = "cell_centre_to_cell_centre"
ROUNDING = "radius2_cells = floor((radius_m / cell_m)^2 + 1e-9); mark cell offset (di, dj) iff di^2 + dj^2 <= radius2_cells"
HELPER = "intercycle.margin_inflate.inflate"
R2_EPS = 1e-9


def radius2_cells(margin_m: float, cell_m: float) -> int:
    """Squared disc radius in whole cells^2 (see the module rule)."""
    margin_m, cell_m = float(margin_m), float(cell_m)
    if not (math.isfinite(margin_m) and margin_m >= 0):
        raise ValueError(f"margin_m must be finite and >= 0, got {margin_m}")
    if not (math.isfinite(cell_m) and cell_m > 0):
        raise ValueError(f"cell_m must be finite and > 0, got {cell_m}")
    return int(math.floor((margin_m / cell_m) ** 2 + R2_EPS))


@dataclass(frozen=True)
class Margin:
    """The frozen M-W7 margin: disc radius margin_m [m] on a grid of cell_m [m] cells."""
    margin_m: float
    cell_m: float

    def __post_init__(self) -> None:
        radius2_cells(self.margin_m, self.cell_m)  # validates

    @property
    def radius2_cells(self) -> int:
        return radius2_cells(self.margin_m, self.cell_m)


def inflation_spec(margin: Margin) -> dict:
    """The JSON `inflation` block that states exactly how to inflate."""
    return dict(kernel=KERNEL, distance=DISTANCE, radius_m=float(margin.margin_m), cell_m=float(margin.cell_m),
                radius2_cells=margin.radius2_cells, rounding=ROUNDING, includes_unknown=True, helper=HELPER)


def _hdilate(a: np.ndarray, w: int) -> np.ndarray:
    """1-D dilation along the last axis by half-width w (cells within |dj| <= w), inside the grid."""
    if w == 0:
        return a.copy()
    nx = a.shape[-1]
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,), np.int64), np.cumsum(a, axis=-1, dtype=np.int64)], axis=-1)
    j = np.arange(nx)
    lo = np.clip(j - w, 0, nx)
    hi = np.clip(j + w + 1, 0, nx)
    return (c[..., hi] - c[..., lo]) > 0


def inflate(pred: np.ndarray, margin: Margin, *, cell_m: float) -> np.ndarray:
    """Inflate `pred` (bool, (..., ny, nx): occupied OR UNKNOWN) by the Euclidean disc of `margin`.

    cell_m is the cell size of the caller's grid; it must equal margin.cell_m (the margin was certified on that grid),
    otherwise ValueError (fail closed: re-calibrate rather than rescale)."""
    if not math.isclose(float(cell_m), float(margin.cell_m), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"grid cell_m {cell_m} differs from the margin's cell_m {margin.cell_m}")
    p = np.asarray(pred, bool)
    if p.ndim < 2:
        raise ValueError("pred must be (..., ny, nx)")
    r2 = margin.radius2_cells
    if r2 == 0:
        return p.copy()
    ny = p.shape[-2]
    R = math.isqrt(r2)
    out = np.zeros_like(p)
    rows: dict[int, np.ndarray] = {}
    for dy in range(-R, R + 1):
        w = math.isqrt(r2 - dy * dy)
        if w not in rows:
            rows[w] = _hdilate(p, w)
        h = rows[w]
        # out[y] |= h[y + dy] for every y with 0 <= y + dy < ny
        if dy >= 0:
            if dy < ny:
                out[..., : ny - dy, :] |= h[..., dy:, :]
        elif -dy < ny:
            out[..., -dy:, :] |= h[..., : ny + dy, :]
    return out


def load_margin(path: str | os.PathLike) -> Margin:
    """Read the frozen margin JSON (robot side). Refuses (ValueError) anything but a frozen, valid margin whose
    inflation block is exactly the rule of this module."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict) or body.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not an {SCHEMA} file")
    if body.get("frozen") is not True or body.get("status") != "ok":
        raise ValueError(f"{path}: margin is not frozen with status ok")
    m, c = body.get("margin_m"), body.get("cell_m")
    if not isinstance(m, (int, float)) or not isinstance(c, (int, float)) or isinstance(m, bool) or isinstance(c, bool):
        raise ValueError(f"{path}: margin_m and cell_m must be numbers")
    margin = Margin(float(m), float(c))
    if body.get("inflation") != inflation_spec(margin):
        raise ValueError(f"{path}: the inflation block does not match the rule of {HELPER}")
    return margin
