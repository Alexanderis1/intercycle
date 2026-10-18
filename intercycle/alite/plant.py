"""The A-lite simulated plant and its noisy camera (truth side; SIMULATED body shop, no real data).

AlitePlant = KeyedWorld (W-W1) + ProgramKinematics (W-W2) + the measured-error camera model (mapping.CameraShim with
MEASURED_SHIM: sigma 0.285 m, 32 % misses, 15 % gross errors, false positives, blackouts). The camera sees machine
parts and skids only (A-lite has no people or vehicles on the floor: perception is a truth-derived noise model without
occlusion by movers, as the PLAN_TWIN §7 claim wording says). `extra_parts(t) -> {name: (k, 3) circles}` adds bodies
the plant does not know (a newly installed arm, M-W4). The observer only ever receives `observe(t)`'s Observation.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from intercycle.alite.programs import ProgramKinematics
from intercycle.alite.world import KeyedWorld
from intercycle.observer import mapping as M
from intercycle.scenario import Scenario

SHIM_SALT = 7919


class AlitePlant:
    def __init__(self, scn: Scenario, world_seed: int, t0: float, t1: float,
                 program_change: dict[str, float] | None = None,
                 extra_parts: Callable[[float], dict[str, np.ndarray]] | None = None, shim_kw: dict | None = None):
        self.scn = scn
        self.world = KeyedWorld(scn, world_seed, t0, t1)
        self.kin = ProgramKinematics(scn, self.world, program_change=program_change)
        kw = dict(M.MEASURED_SHIM)
        kw.update(shim_kw or {})
        self.cam = M.CameraShim(scn, seed=world_seed * SHIM_SALT + int(t0 // 3600), **kw)  # pyright: ignore[reportArgumentType]
        self.mids = [m.id for m in scn.machines]
        self.extra_parts = extra_parts

    def state(self, t: float) -> M.SimState:
        bd = self.kin.bodies(t)
        parts = {mid: bd[mid] for mid in self.mids}
        if self.extra_parts is not None:
            parts.update(self.extra_parts(t))
        return M.SimState(t, parts=parts, skids=bd["SKIDS"])

    def observe(self, t: float) -> M.Observation:
        """Camera tick delivered at t (the frame shows the floor at t - latency, parts motion-blurred)."""
        return self.cam.observe(self.state, t)
