"""Keyed random streams for the twin (W-W1).

Every random draw of a generated day comes from `keyed_rng(world_seed, day, entity, event)`: a fresh PCG64 stream
seeded by the SeedSequence of (world_seed, day, blake2b64(entity), blake2b64(event)). A stream depends only on its
key, never on how many draws other entities made or in which order they were generated, so the same (world, day,
entity, event) is reproduced exactly when an entity is added, removed or generated in another order, and day d of a
world never replays day d' (the body-shop W-W1 defect: one stream restarted per query window).

Seed rule (PLAN_TWIN §5): dev world seeds are 0..999; any other seed is a test seed and is refused unless
INTERCYCLE_SEALED_RUN=1.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

DEV_SEED_MAX = 999


def key64(s: str) -> int:
    """Stable 64-bit key of a string (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def keyed_rng(world_seed: int, day: int, entity: str, event: str) -> np.random.Generator:
    if world_seed < 0 or day < 0:
        raise ValueError("world_seed and day must be non-negative")
    ss = np.random.SeedSequence([int(world_seed), int(day), key64(entity), key64(event)])
    return np.random.Generator(np.random.PCG64(ss))


def check_world_seed(world_seed: int) -> None:
    """Refuse test-world seeds outside the sealed run."""
    if 0 <= world_seed <= DEV_SEED_MAX:
        return
    if os.environ.get("INTERCYCLE_SEALED_RUN") != "1":
        raise PermissionError(f"world seed {world_seed} is not a dev seed (0..{DEV_SEED_MAX}); "
                              "test seeds need INTERCYCLE_SEALED_RUN=1")
