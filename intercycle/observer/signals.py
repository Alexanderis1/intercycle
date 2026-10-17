"""Camera shim: what a ceiling camera's motion detector would output (SYNTHETIC).

The detector watches the ROI of one hazard zone and emits a binary motion flag at 2 Hz:
y=1 motion (zone ACTIVE), y=0 no motion. Real-world imperfections modelled here: 0.2 s pipeline
latency, occlusion dropouts in bursts (a person or tugger walks through the view -> reads "no
motion", p~8 % of the time, ~1.5 s per burst) and isolated false flips (p~3 % per sample).
This module may use ground truth; the learner only ever sees (t, y).
"""
from __future__ import annotations

import numpy as np

from intercycle.scenario import Scenario
from intercycle.truth import machines as truth_machines
from intercycle.truth import traffic as truth_traffic
from intercycle.types import Window

HZ = 2.0
LATENCY_S = 0.2
P_OCCLUDE = 0.08       # stationary fraction of time the ROI is occluded
BURST_SAMPLES = 3.0    # mean occlusion burst length in samples (1.5 s)
P_FLIP = 0.03
DWELL_S = 60.0         # mean time a tracked object spends in a zone: counts are track entries per bin


def _occlusion(n: int, rng: np.random.Generator, p: float, burst: float) -> np.ndarray:
    """Two-state Markov chain: occluded fraction p, mean occluded run `burst` samples."""
    out = np.zeros(n, bool)
    if p <= 0:
        return out
    p_end = 1.0 / max(burst, 1.0)
    p_start = p * p_end / (1 - p)
    i, occ = 0, bool(rng.random() < p)
    while i < n:
        run = int(rng.geometric(p_end if occ else p_start))
        out[i:i + run] = occ
        i += run
        occ = not occ
    return out


def camera(ws: list[Window], t0: float, t1: float, seed: int = 0, hz: float = HZ, latency_s: float = LATENCY_S,
           p_occlude: float = P_OCCLUDE, burst: float = BURST_SAMPLES, p_flip: float = P_FLIP) -> tuple[np.ndarray, np.ndarray]:
    """Motion-detector output (t, y) at `hz` for truth windows over [t0, t1). y=1 while the zone is ACTIVE."""
    n = int(round((t1 - t0) * hz))
    t = t0 + np.arange(n) / hz
    active = np.zeros(n, bool)
    if ws:
        bs = np.array([w.bs for w in ws])
        is_ = np.array([w.is_ for w in ws])
        k = np.searchsorted(bs, t - latency_s, side="right") - 1
        active = (k >= 0) & (t - latency_s < is_[np.maximum(k, 0)])
    rng = np.random.default_rng([seed, 3])
    y = active & ~_occlusion(n, rng, p_occlude, burst)
    y = y ^ (rng.random(n) < p_flip)
    return t, y.astype(np.int8)


def machine_signal(scn: Scenario, machine_id: str, seed: int, t0: float, t1: float, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Camera signal of one scenario machine (truth windows from truth/machines.py, noise seeded per machine)."""
    ws = truth_machines.windows(scn, machine_id, seed, t0, t1)
    salt = sum(map(ord, machine_id))
    return camera(ws, t0, t1, seed=seed * 1000 + salt, **kw)


def occupancy(scn: Scenario, seed: int, t0: float, t1: float, bin_s: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-zone track-entry counts per bin (detector + tracker output): Poisson around expected occupancy * bin_s / DWELL_S.
    Returns (bin_start_times, counts[zone][bin])."""
    t = np.arange(t0, t1 - bin_s / 2, bin_s)
    rng = np.random.default_rng([seed, 4])
    lam = np.stack([truth_traffic.expected(scn, z.id, t + bin_s / 2) for z in scn.zones]) * bin_s / DWELL_S
    return t, rng.poisson(lam).astype(np.int32)
