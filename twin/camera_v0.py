"""G0 MEASURED_SHIM camera adapter: realised state -> noisy detections at 2 Hz (stand-in until `twin/camera.py`
stat_v1, lane S3b; it fills the twin/camera.py slot of the truth allowlist, `twin.truth_guard.STAND_INS`).

Noise is `intercycle.observer.mapping.MEASURED_SHIM` (the error measured on real footage): floor-position noise
sigma_m = 0.285 m, per-detection miss p_drop, gross errors (share gross_p, radial U(gross_range_m)) and false-positive
people (fp_ratio per true person detection in a zone, fp_near of them 0.5-3 m from a true detection, the rest uniform
on that zone's visible floor). Errors are iid per tick here (stat_v1 adds Markov miss bursts, persistent gross errors
and class confusion). DISCLOSED (M-W2): v0 passes the true class and the true moving flag (speed > 0.2 m/s) through.

Common random numbers: every draw comes from `twin.rng.keyed_rng(world_seed, day, entity, f"tick:{k}")`, with k the
absolute 2 Hz tick index (round(t * hz)) and entity `cam:<mover id>` (miss, position, gross error, radius, heading),
`cam_fp:<zone id>` (false positives) or `cam_motion:<press id>` (ROI motion noise). A mover's noise therefore does not
depend on which other movers exist: the robot, whose path differs between paired conditions, never shifts the
people's or the presses' noise (learned vs reactive share it, PLAN_TWIN §5 primary endpoint).

Per press, `zone_motion` is the ram speed (|delta ram| per second) plus N(0, motion_sigma), NaN (MISSING) while a true
person, forklift or the robot stands within `OCCLUSION_M` of the press footprint (T-W6 occlusion = missing; S4's
occluder classes). The robot is emitted as class `other`. Only movers inside a zone's visible floor are seen. The
output feeds S4's Observation v2 adapter (`intercycle/observer/mapping.py`); this module does not define Observation v2.
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
from twin.rng import keyed_rng
from twin.schemas.runs import RealisedMover, RealisedTick, parse_realised
from twin.schemas.site import Site, point_in_polygon
from twin.truth_guard import open_truth

CLASS_OUT = {"person": "person", "forklift": "vehicle", "robot": "other"}
OCCLUDER_CLASSES = ("person", "forklift", "robot")
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
    def __init__(self, site: Site, world_seed: int, day: int = 1, shim: dict[str, Any] | None = None,
                 latency_s: float = LATENCY_S, motion_sigma: float = 0.05, hz: float = 2.0):
        s: dict[str, Any] = dict(MEASURED_SHIM if shim is None else shim)
        self.sigma, self.p_drop = float(s["sigma_m"]), float(s["p_drop"])
        lo, hi = s["gross_range_m"]
        self.gross_p, self.gross_lo, self.gross_hi = float(s["gross_p"]), float(lo), float(hi)
        self.fp_ratio, self.fp_near = float(s["fp_ratio"]), float(s["fp_near"])
        self.site, self.latency, self.motion_sigma, self.hz = site, latency_s, motion_sigma, hz
        self.world_seed, self.day = int(world_seed), int(day)
        self.zones = {z.id: z for z in site.zones}
        self._fp_bbox = {p.id: _bbox(p.footprint.value, OCCLUSION_M) for p in site.presses}
        self._prev_ram: dict[str, float] = {}

    def _rng(self, entity: str, k: int) -> np.random.Generator:
        return keyed_rng(self.world_seed, self.day, entity, f"tick:{k}")

    def _detect(self, m: RealisedMover, k: int) -> tuple[float, float, float, float] | None:
        """(x, z, r, heading) of one mover's detection, or None if missed. A fixed number of draws per call."""
        g = self._rng(f"cam:{m.id}", k)
        u_drop, nx, nz, u_gross, rr, aa, nr, nh = (g.random(), g.normal(0.0, self.sigma), g.normal(0.0, self.sigma),
                                                   g.random(), g.uniform(self.gross_lo, self.gross_hi),
                                                   g.uniform(0.0, 2 * math.pi), g.normal(0.0, 0.03),
                                                   g.normal(0.0, 0.15))
        if u_drop < self.p_drop:
            return None
        if u_gross < self.gross_p:
            x, z = m.x + rr * math.sin(aa), m.z + rr * math.cos(aa)
        else:
            x, z = m.x + nx, m.z + nz
        heading = m.heading + nh if m.speed > MOVING_MPS else math.nan
        return x, z, max(m.r + nr, 0.1), heading

    def _floor_point(self, g: np.random.Generator, zone_id: str) -> tuple[float, float]:
        poly = self.zones[zone_id].floor_polygon.value
        x0, z0, x1, z1 = _bbox(poly, 0.0)
        for _ in range(1000):
            x, y = float(g.uniform(x0, x1)), float(g.uniform(z0, z1))
            if point_in_polygon(x, y, poly):
                return x, y
        return (x0 + x1) / 2, (z0 + z1) / 2

    def observe(self, tick: RealisedTick) -> Detections:
        k = int(round(tick.t * self.hz))
        seen = [m for m in tick.movers if m.zone is not None and m.zone in self.zones]
        cls: list[str] = []
        xs: list[float] = []
        zs: list[float] = []
        rs: list[float] = []
        hs: list[float] = []
        mv: list[bool] = []
        zone: list[str] = []
        real: list[bool] = []                                    # internal only: true detection vs false positive
        for m in seen:
            d = self._detect(m, k)
            if d is None:
                continue
            cls.append(CLASS_OUT[m.cls])
            xs.append(d[0])
            zs.append(d[1])
            rs.append(d[2])
            hs.append(d[3])
            mv.append(m.speed > MOVING_MPS)
            zone.append(str(m.zone))
            real.append(True)
        for zid in self.zones:                                   # false positives, one keyed stream per zone
            persons = [i for i, (c, zz) in enumerate(zip(cls, zone)) if c == "person" and zz == zid and real[i]]
            if not persons:
                continue
            g = self._rng(f"cam_fp:{zid}", k)
            for _ in range(int(g.poisson(self.fp_ratio * len(persons)))):
                if g.random() < self.fp_near:
                    i = persons[int(g.integers(len(persons)))]
                    dd, a = g.uniform(0.5, 3.0), g.uniform(0.0, 2 * math.pi)
                    px, pz = xs[i] + dd * math.sin(a), zs[i] + dd * math.cos(a)
                else:
                    px, pz = self._floor_point(g, zid)
                cls.append("person")
                xs.append(px)
                zs.append(pz)
                rs.append(0.3)
                hs.append(math.nan)
                mv.append(False)
                zone.append(zid)
                real.append(False)
        motion: dict[str, float] = {}
        occluders = [(m.x, m.z) for m in seen if m.cls in OCCLUDER_CLASSES]
        for p in tick.presses:
            bx0, bz0, bx1, bz1 = self._fp_bbox[p.id]
            prev = self._prev_ram.get(p.id, p.ram)
            self._prev_ram[p.id] = p.ram
            noise = float(self._rng(f"cam_motion:{p.id}", k).normal(0.0, self.motion_sigma))
            if any(bx0 <= px <= bx1 and bz0 <= pz <= bz1 for px, pz in occluders):
                motion[p.id] = math.nan
            else:
                motion[p.id] = max(0.0, abs(p.ram - prev) * self.hz + noise)
        return Detections(t=tick.t + self.latency, t_cap=tick.t, cls=np.array(cls, dtype=str),
                          x=np.array(xs, float), z=np.array(zs, float), r=np.array(rs, float),
                          heading=np.array(hs, float), moving=np.array(mv, bool), zone=np.array(zone, dtype=str),
                          zone_motion=motion, cam_ok={zz.camera: True for zz in self.site.zones})

    def run(self, ticks: Iterable[RealisedTick]) -> Iterator[Detections]:
        for tk in ticks:
            yield self.observe(tk)


def stream(path: str | Path, site: Site, world_seed: int) -> Iterator[Detections]:
    """Read a realised_state.jsonl (truth; logged by the truth guard) and yield detections tick by tick. The day
    comes from the stream's header."""
    with open_truth(path, "r") as f:
        header, ticks = parse_realised(f)
        yield from CameraV0(site, world_seed, header.day).run(ticks)


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
