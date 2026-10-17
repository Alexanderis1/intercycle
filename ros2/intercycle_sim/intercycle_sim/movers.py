"""Moving bodies (people, AGVs, tuggers, forklifts, weld-arm envelopes): the hook R3 drives.

R3 supplies a `MoverSource` (any object with `spec()` and `poses(t)`), named on the command line as
`--movers package.module:factory` (the factory is called with the episode cfg dict).  Every mover becomes
one or more KINEMATIC rigid bodies in Isaac (the robot physically collides with them; nothing yields here) and the
same primitives are hit analytically by the lidar and the camera model, so what the robot senses = what it bumps into.

Poses are in the ROS/Isaac frame (x = plan x, y = plan z, heading psi from +x, counter-clockwise).
Use `plan_to_ros(x, z, yaw)` to convert from intercycle.truth (plan yaw = atan2(dx, dz)).
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np

# cls -> parts (shape, forward offset of the part centre, length along heading, width, height); "cyl" uses length = 2r.
# Footprints match intercycle.sim.geometry.DIMS; heights are design assumptions (loads/masts included so a 2D lidar
# at ~0.8 m sees them).  Forks are modelled with their load (h 1.0 m); bare forks below the lidar plane are a stated
# limitation.
PARTS: dict[str, list[tuple[str, float, float, float, float]]] = {
    "person": [("cyl", 0.0, 0.6, 0.6, 1.75)],
    "agv": [("box", 0.0, 1.6, 1.0, 1.2)],
    "tugger": [("box", 0.0, 1.4, 1.0, 1.6)] + [("box", -(0.7 + 0.2 + 0.8 + 1.8 * k), 1.6, 1.0, 1.4) for k in range(3)],
    "forklift": [("box", 0.0, 1.9, 1.1, 2.2), ("box", 1.5, 1.1, 0.9, 1.0)],
    "weld_arm_active": [("cyl", 0.0, 4.4, 4.4, 2.5)],
    "weld_arm_idle": [("cyl", 0.0, 1.4, 1.4, 2.0)],
}
ALIASES = {"tugger_train": "tugger", "train": "tugger", "weld_arm": "weld_arm_idle"}
PARK_XY = (-60.0, -60.0)                 # parked slots live far outside the plant


def plan_to_ros(x: float, z: float, yaw: float) -> tuple[float, float, float]:
    return float(x), float(z), math.pi / 2 - float(yaw)


@dataclass
class Part:
    mover_id: str
    cls: str
    shape: str          # "box" | "cyl"
    fwd: float
    length: float
    width: float
    height: float


def parts_for(spec: list[dict]) -> list[Part]:
    out = []
    for m in spec:
        cls = ALIASES.get(m["cls"], m["cls"])
        for shape, fwd, ln, wd, h in PARTS[cls]:
            out.append(Part(m["id"], cls, shape, fwd, ln, wd, h))
    return out


def part_poses(parts: list[Part], poses: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """(P, 3) x, y, psi per part; movers missing from `poses` are parked."""
    out = np.empty((len(parts), 3))
    for k, p in enumerate(parts):
        q = poses.get(p.mover_id)
        if q is None:
            out[k] = (PARK_XY[0] + 3.0 * k, PARK_XY[1], 0.0)
            continue
        x, y, psi = q
        out[k] = (x + p.fwd * math.cos(psi), y + p.fwd * math.sin(psi), psi)
    return out


def primitives(parts: list[Part], pp: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic primitives for the sensors: circles (cx, cy, r, zmax) and boxes (cx, cy, hw, hd, yaw, zmax)."""
    circ, box = [], []
    for p, (x, y, psi) in zip(parts, pp):
        if p.shape == "cyl":
            circ.append((x, y, p.length / 2, p.height))
        else:
            box.append((x, y, p.length / 2, p.width / 2, psi, p.height))
    return {"circles": np.array(circ, dtype=float).reshape(-1, 4), "boxes": np.array(box, dtype=float).reshape(-1, 6)}


class NoMovers:
    def spec(self) -> list[dict]:
        return []

    def poses(self, t: float) -> dict[str, tuple[float, float, float]]:
        return {}


class ScriptedMovers:
    """Constant-velocity straight-line movers, for smoke tests: [{"id","cls","x","y","psi","v"}]."""

    def __init__(self, movers: list[dict]):
        self.m = movers

    def spec(self) -> list[dict]:
        return [{"id": m["id"], "cls": m["cls"]} for m in self.m]

    def poses(self, t: float) -> dict[str, tuple[float, float, float]]:
        return {m["id"]: (m["x"] + m.get("v", 0.0) * t * math.cos(m["psi"]),
                          m["y"] + m.get("v", 0.0) * t * math.sin(m["psi"]), m["psi"]) for m in self.m}


def load_source(ref: str | None, cfg: dict):
    """'pkg.mod:factory' -> factory(cfg); None -> scripted movers from cfg['movers'] (or none)."""
    if not ref:
        return ScriptedMovers(cfg["movers"]) if cfg.get("movers") else NoMovers()
    mod, _, fn = ref.partition(":")
    return getattr(importlib.import_module(mod), fn or "make")(cfg)
