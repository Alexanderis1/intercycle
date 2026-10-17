"""G0 MEASURED_SHIM camera adapter: realised state -> noisy detections at 2 Hz (stand-in until `twin/camera.py`
stat_v1, lane S3b).

Noise is `intercycle.observer.mapping.MEASURED_SHIM` (the error measured on real footage): floor-position noise
sigma_m = 0.285 m, per-detection miss p_drop, gross errors (share gross_p, radial U(gross_range_m)) and false-positive
people (fp_ratio per true person detection, fp_near of them 0.5-3 m from a true detection, the rest uniform on the
visible floor). Errors are iid per tick here (stat_v1 adds Markov miss bursts, persistent gross errors and class
confusion). DISCLOSED (M-W2): v0 passes the true class and the true moving flag (speed > 0.2 m/s) through.

Per press, `zone_motion` is the ram speed (|delta ram| per second) plus N(0, motion_sigma), NaN (MISSING) while a
true person stands within `OCCLUSION_M` of the press footprint (T-W6 occlusion = missing). The robot is emitted as
class `other`. Only movers inside a zone's visible floor are seen. The output feeds S4's Observation v2 adapter
(`intercycle/observer/mapping.py`); this module does not define Observation v2.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from intercycle.observer.mapping import LATENCY_S, MEASURED_SHIM
from twin.rng import key64
from twin.schemas.runs import RealisedTick, parse_realised
from twin.schemas.site import Site, point_in_polygon
from twin.truth_guard import open_truth

CLASS_OUT = {"person": "person", "forklift": "vehicle", "robot": "other"}
MOVING_MPS = 0.2
OCCLUSION_M = 0.5


@dataclass
class Detections:
    t: float                                  # delivery time (capture + latency)
    t_cap: float
    cls: np.ndarray                           # (n,) str: person / vehicle / other
    x: np.ndarray
    z: np.ndarray
    r: np.ndarray
    heading: np.ndarray                       # rad, NaN when not moving
    moving: np.ndarray                        # (n,) bool
    zone: np.ndarray                          # (n,) str zone id of the camera that saw it
    zone_motion: dict[str, float] = field(default_factory=dict)   # press id -> ram motion, NaN = MISSING
    cam_ok: dict[int, bool] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cls)


class CameraV0:
    def __init__(self, site: Site, seed: int, shim: dict[str, Any] | None = None, latency_s: float = LATENCY_S,
                 motion_sigma: float = 0.05, hz: float = 2.0):
        s: dict[str, Any] = dict(MEASURED_SHIM if shim is None else shim)
        self.sigma, self.p_drop = float(s["sigma_m"]), float(s["p_drop"])
        lo, hi = s["gross_range_m"]
        self.gross_p, self.gross_lo, self.gross_hi = float(s["gross_p"]), float(lo), float(hi)
        self.fp_ratio, self.fp_near = float(s["fp_ratio"]), float(s["fp_near"])
        self.site, self.latency, self.motion_sigma, self.hz = site, latency_s, motion_sigma, hz
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), key64("camera_v0")])))
        self.zones = {z.id: z for z in site.zones}
        self._area = np.array([abs(_shoelace(z.floor_polygon.value)) for z in site.zones])
        self._fp_bbox = {p.id: _bbox(p.footprint.value, OCCLUSION_M) for p in site.presses}
        self._prev_ram: dict[str, float] = {}

    def _floor_point(self) -> tuple[str, float, float]:
        k = int(self.rng.choice(len(self.site.zones), p=self._area / self._area.sum()))
        z = self.site.zones[k]
        poly = z.floor_polygon.value
        x0, z0, x1, z1 = _bbox(poly, 0.0)
        for _ in range(1000):
            x, y = float(self.rng.uniform(x0, x1)), float(self.rng.uniform(z0, z1))
            if point_in_polygon(x, y, poly):
                return z.id, x, y
        return z.id, (x0 + x1) / 2, (z0 + z1) / 2

    def observe(self, tick: RealisedTick) -> Detections:
        rng = self.rng
        seen = [m for m in tick.movers if m.zone is not None and m.zone in self.zones]
        keep = [m for m in seen if rng.random() >= self.p_drop]
        n = len(keep)
        x = np.array([m.x for m in keep], float) + rng.normal(0.0, self.sigma, n)
        z = np.array([m.z for m in keep], float) + rng.normal(0.0, self.sigma, n)
        gross = rng.random(n) < self.gross_p
        if gross.any():
            rr = rng.uniform(self.gross_lo, self.gross_hi, int(gross.sum()))
            aa = rng.uniform(0.0, 2 * math.pi, int(gross.sum()))
            x[gross] = np.array([m.x for m, g in zip(keep, gross) if g]) + rr * np.sin(aa)
            z[gross] = np.array([m.z for m, g in zip(keep, gross) if g]) + rr * np.cos(aa)
        r = np.maximum(np.array([m.r for m in keep], float) + rng.normal(0.0, 0.03, n), 0.1)
        moving = np.array([m.speed > MOVING_MPS for m in keep], bool)
        heading = np.where(moving, np.array([m.heading for m in keep], float) + rng.normal(0.0, 0.15, n), np.nan)
        cls = [CLASS_OUT[m.cls] for m in keep]
        zone = [str(m.zone) for m in keep]
        n_person = cls.count("person")
        k = int(rng.poisson(self.fp_ratio * n_person)) if n_person else 0
        fx, fz, fzone = [], [], []
        idx_person = [i for i, c in enumerate(cls) if c == "person"]
        for _ in range(k):
            if rng.random() < self.fp_near:
                i = idx_person[int(rng.integers(len(idx_person)))]
                d, a = rng.uniform(0.5, 3.0), rng.uniform(0.0, 2 * math.pi)
                fx.append(x[i] + d * math.sin(a))
                fz.append(z[i] + d * math.cos(a))
                fzone.append(zone[i])
            else:
                zid, px, pz = self._floor_point()
                fx.append(px)
                fz.append(pz)
                fzone.append(zid)
        if k:
            x, z = np.r_[x, fx], np.r_[z, fz]
            r = np.r_[r, np.full(k, 0.3)]
            heading = np.r_[heading, np.full(k, np.nan)]
            moving = np.r_[moving, np.zeros(k, bool)]
            cls += ["person"] * k
            zone += fzone
        motion: dict[str, float] = {}
        people = [(m.x, m.z) for m in seen if m.cls == "person"]
        for p in tick.presses:
            bx0, bz0, bx1, bz1 = self._fp_bbox[p.id]
            prev = self._prev_ram.get(p.id, p.ram)
            self._prev_ram[p.id] = p.ram
            if any(bx0 <= px <= bx1 and bz0 <= pz <= bz1 for px, pz in people):
                motion[p.id] = math.nan
            else:
                motion[p.id] = max(0.0, abs(p.ram - prev) * self.hz + float(rng.normal(0.0, self.motion_sigma)))
        return Detections(t=tick.t + self.latency, t_cap=tick.t, cls=np.array(cls, dtype=str), x=x, z=z, r=r,
                          heading=heading, moving=moving, zone=np.array(zone, dtype=str), zone_motion=motion,
                          cam_ok={zz.camera: True for zz in self.site.zones})

    def run(self, ticks: Iterable[RealisedTick]) -> Iterator[Detections]:
        for tk in ticks:
            yield self.observe(tk)


def stream(path: str | Path, site: Site, seed: int) -> Iterator[Detections]:
    """Read a realised_state.jsonl (truth; logged by the truth guard) and yield detections tick by tick."""
    with open_truth(path, "r") as f:
        _, ticks = parse_realised(f)
        yield from CameraV0(site, seed).run(ticks)


def detections_to_json(d: Detections) -> str:
    """One JSON line (NaN -> null) for debugging and fixtures."""
    def fl(a):
        return [None if not math.isfinite(v) else round(float(v), 4) for v in a]
    return json.dumps({"t": d.t, "t_cap": d.t_cap, "cls": d.cls.tolist(), "x": fl(d.x), "z": fl(d.z), "r": fl(d.r),
                       "heading": fl(d.heading), "moving": d.moving.tolist(), "zone": d.zone.tolist(),
                       "zone_motion": {k: (None if math.isnan(v) else v) for k, v in d.zone_motion.items()},
                       "cam_ok": {str(k): v for k, v in d.cam_ok.items()}})


def _bbox(poly, grow: float) -> tuple[float, float, float, float]:
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) - grow, min(zs) - grow, max(xs) + grow, max(zs) + grow


def _shoelace(poly) -> float:
    n = len(poly)
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
