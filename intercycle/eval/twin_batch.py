"""The CPU-double batch plan of the press-shop twin (PLAN_TWIN §5 "Design" / "Rules"; USER DECISION Sun ~07:00 CEST, the
CUT design): which missions run, in which order, under which run ids, how infrastructure failures are re-run, where
the outputs land and what it costs. Pure planning: no Modal, no truth, no plant (the orchestrator is
scripts/run_double.py, which starts the generator, the observer phase and the plant).

Design (twin_design.SEALED_CONDITIONS x worlds x days 9, 10 x the tasks.yaml zones): one mission per (world, day, zone,
condition), CRN-paired: the three conditions of a (world, day, zone) cell run on the SAME day script, the same mission
start `slot_t0` and the same plant draws (the plant keys its draws by (world seed, day, plant_t0)).

Mission start (`day_slots`, LANE INTERPRETATION, to be hashed into PREREG): one slot per (world, day, zone), drawn
uniformly on the 2 Hz grid of the day from twin.rng.keyed_rng(world_seed, day, "mission_slot", zone) inside
[SLOT_PAD_S, day_s - MISSION_T_S - SLOT_PAD_S]; a new key, so no other draw of the world changes (W-W1). NO OVERLAP:
the zones of a world-day are drawn in the given (tasks.yaml) order and a later zone redraws from its OWN keyed stream
until its slot is >= MISSION_T_S away from every earlier zone's slot of that day (the reactive state is per world,
across zones, so an overlapping slot would load own-camera history from after its own start); the first zone's slot,
and every slot that did not overlap, is exactly the first draw. rounds() re-checks that each reactive chain step
starts at or after the previous step's t0 + MISSION_T_S.

Order (`rounds`): the reactive arm persists its own-camera history within the world (intercycle_nav.reactive: each
mission loads the state its previous mission saved), so a world's reactive missions are a CHAIN in time order (day,
slot): round r holds chain step r of every world. learned and static missions are independent and are spread evenly
over the rounds (each round <= the wave size). Within a round every mission runs in parallel.

Re-runs (classify_invalid's rule, PLAN_TWIN §5): after each round, scripts/classify_invalid.py labels its episodes
from the logs alone; every (world, day, zone) with an infrastructure-invalid run is re-run with ALL its conditions
under attempt k + 1 (`__a<k>`), before the next round, at most MAX_ATTEMPTS attempts. A reactive re-run starts from the
same input state as the attempt it repeats; the next chain step takes the state of the cell's scored attempt (the first
clean one, twin_design's rule).

Layout (what twin_eval scores): <runs_root>/<batch>/ holds batch.json (the cfgs), <tag>.json (the episode record) and
<tag>/ (the pulled episode dir: summary.json, log.parquet, ...), tag = run_id = w<W>_d<D>_<zone>_<condition>[__a<k>];
<runs_root>/invalid_all.json is the union of the batches' invalid lists (twin_eval --invalid).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from intercycle.eval.twin_design import SEALED_CONDITIONS, TEST_DAYS
from twin.rng import keyed_rng
from twin.schemas.common import MISSION_T_S, TICK_HZ

DAY_S = 86_400.0
OBSERVER_DAYS = tuple(range(1, 9))     # the observer phase: days 1-8, robot off (PLAN_TWIN §5 Design)
DEV_WORLD_SEEDS = (0, 1, 2, 3)          # dev worlds 0-3 (dev seeds only; --dev)
SLOT_PAD_S = 600.0
MAX_ATTEMPTS = 3                        # attempt 0 + 2 re-runs of an infrastructure-invalid cell
CPU = 4.0                               # H1: cpu 4, max_rtf 2 (docs/probes/H1.md)
MAX_RTF = 2.0
MEMORY_GIB = 8.0
RATE_CPU_PER_CORE_S = 0.0000131         # Modal pricing used by H1 (docs/probes/H1.md "Rates used")
RATE_MEM_PER_GIB_S = 0.00000222
WAVE_SIZE = 50                          # missions in flight at once (one Modal container each)
INVALID_ALL = "invalid_all.json"


class BatchError(ValueError):
    """The batch plan is inconsistent (fail closed)."""


@dataclass(frozen=True)
class World:
    index: int                          # the design's world index (twin_eval --worlds)
    seed: int                           # the world seed (dev 0..999; test seeds only through sealed_run)
    split: str                          # "dev" | "test"


@dataclass(frozen=True)
class Mission:
    world: World
    day: int
    zone: str
    condition: str
    t0: float                           # absolute plant_t0 (engine seconds)
    attempt: int = 0

    @property
    def cell(self) -> tuple[int, int, str]:
        return self.world.index, self.day, self.zone

    @property
    def run_id(self) -> str:
        return run_id(self.world.index, self.day, self.zone, self.condition, self.attempt)

    def record(self) -> dict:
        return {"run_id": self.run_id, "world": self.world.index, "day": self.day, "zone": self.zone,
                "condition": self.condition, "split": self.world.split, "engine": "double", "seed": self.world.seed,
                "t0": self.t0, "attempt": self.attempt}


def run_id(world: int, day: int, zone: str, condition: str, attempt: int = 0) -> str:
    return f"w{world}_d{day}_{zone}_{condition}" + (f"__a{attempt}" if attempt else "")


def day_t0(day: int) -> float:
    """Absolute start of a generated day (the generators' t0_s = (day - 1) x DAY_S, whatever the day's length)."""
    return (day - 1) * DAY_S


MAX_SLOT_DRAWS = 10_000


def _slot_draws(world_seed: int, day: int, zone: str, day_s: float, pad_s: float):
    """The keyed stream of candidate starts of (world, day, zone) on the 2 Hz grid."""
    lo, hi = pad_s, day_s - MISSION_T_S - pad_s
    if hi < lo:
        raise BatchError(f"a {day_s} s day has no room for a {MISSION_T_S} s mission and {pad_s} s pads")
    rng = keyed_rng(world_seed, day, "mission_slot", zone)
    for _ in range(MAX_SLOT_DRAWS):
        yield day_t0(day) + math.floor(float(rng.uniform(lo, hi)) * TICK_HZ) / TICK_HZ


def slot_t0(world_seed: int, day: int, zone: str, *, day_s: float = DAY_S, pad_s: float = SLOT_PAD_S) -> float:
    """The FIRST draw of (world, day, zone) (module doc); day_slots applies the no-overlap rule on top of it."""
    return next(_slot_draws(world_seed, day, zone, day_s, pad_s))


def day_slots(world_seed: int, day: int, zones: Sequence[str], *, day_s: float = DAY_S,
              pad_s: float = SLOT_PAD_S) -> dict[str, float]:
    """zone -> absolute mission start of one world-day, no two slots closer than MISSION_T_S (module doc). day_s =
    the generated day's length (DAY_S; a shorter dev-only day in the tests)."""
    out: dict[str, float] = {}
    for z in zones:
        t0 = next((t for t in _slot_draws(world_seed, day, z, day_s, pad_s)
                   if all(abs(t - u) >= MISSION_T_S for u in out.values())), None)
        if t0 is None:
            raise BatchError(f"no non-overlapping slot for world seed {world_seed} day {day} zone {z} "
                             f"in {MAX_SLOT_DRAWS} draws")
        out[str(z)] = t0
    return out


def plan(worlds: Sequence[World], zones: Sequence[str], *, days: Sequence[int] = TEST_DAYS,
         conditions: Sequence[str] = SEALED_CONDITIONS, day_s: float = DAY_S) -> list[Mission]:
    """Every attempt-0 mission of the design (module doc), CRN-paired per (world, day, zone)."""
    if tuple(conditions) != tuple(SEALED_CONDITIONS):
        raise BatchError(f"the CPU-double batch runs exactly {SEALED_CONDITIONS} (cut design), got {list(conditions)}")
    if len({w.index for w in worlds}) != len(worlds) or len({w.seed for w in worlds}) != len(worlds):
        raise BatchError("worlds repeat an index or a seed")
    if len({w.split for w in worlds}) > 1:
        raise BatchError("a batch is on one split")
    if not zones or not days:
        raise BatchError("no zones or no days")
    out = []
    for w in worlds:
        for d in days:
            slots = day_slots(w.seed, int(d), list(zones), day_s=day_s)
            for z in zones:
                out += [Mission(w, int(d), str(z), c, slots[str(z)]) for c in conditions]
    return out


def reactive_chains(missions: Iterable[Mission]) -> dict[int, list[Mission]]:
    """world index -> its reactive missions in time order (the state chain)."""
    ch: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        if m.condition == "reactive":
            ch[m.world.index].append(m)
    return {w: sorted(v, key=lambda m: (m.t0, m.zone)) for w, v in ch.items()}


def rounds(missions: Sequence[Mission], wave_size: int = WAVE_SIZE) -> list[list[Mission]]:
    """The run order (module doc): round r = chain step r of every world's reactive chain + an even share of the
    independent missions; a round larger than wave_size is refused (raise the wave size or split the batch)."""
    chains = reactive_chains(missions)
    n = max((len(c) for c in chains.values()), default=1)
    if len({len(c) for c in chains.values()}) > 1:
        raise BatchError("worlds have reactive chains of different lengths")
    for w, c in chains.items():
        for a, b in zip(c, c[1:]):
            if b.t0 < a.t0 + MISSION_T_S - 1e-9:
                raise BatchError(f"world {w}: chain step {b.run_id} starts before {a.run_id} ends (overlap)")
    out: list[list[Mission]] = [[c[r] for c in chains.values()] for r in range(n)] if chains else [[]]
    indep = sorted((m for m in missions if m.condition != "reactive"), key=lambda m: (m.t0, m.world.index, m.zone,
                                                                                      m.condition))
    # the independent missions of the cell of chain step r go to round r (a cell's conditions run together)
    step = {m.cell: r for r, rr in enumerate(out) for m in rr}
    for m in indep:
        out[step.get(m.cell, 0)].append(m)
    big = [len(r) for r in out if len(r) > wave_size]
    if big:
        raise BatchError(f"rounds of {big} missions exceed the wave size {wave_size}")
    return out


def reruns(round_missions: Sequence[Mission], invalid_ids: Iterable[str]) -> list[Mission]:
    """The next attempt of every (world, day, zone) with an infrastructure-invalid run: ALL its conditions at k + 1.
    Refused past MAX_ATTEMPTS (the cell stays unresolved: twin_eval claims no gate)."""
    bad = set(invalid_ids)
    cells = sorted({m.cell for m in round_missions if m.run_id in bad})
    out = []
    for c in cells:
        ms = [m for m in round_missions if m.cell == c]
        k = max(m.attempt for m in ms) + 1
        if k >= MAX_ATTEMPTS:
            continue
        out += [Mission(m.world, m.day, m.zone, m.condition, m.t0, k) for m in ms if m.attempt == k - 1]
    return out


def scored_attempt(missions: Sequence[Mission], invalid_ids: Iterable[str]) -> dict[tuple[int, int, str], int | None]:
    """(world, day, zone) -> the first attempt with no invalid run (twin_design's scored attempt), None if none."""
    bad = set(invalid_ids)
    by: dict[tuple[int, int, str], dict[int, list[Mission]]] = defaultdict(lambda: defaultdict(list))
    for m in missions:
        by[m.cell][m.attempt].append(m)
    return {c: next((a for a in sorted(atts) if not any(m.run_id in bad for m in atts[a])), None)
            for c, atts in by.items()}


def merge_invalid(lists: Iterable[Mapping], records: Mapping[str, dict]) -> dict:
    """The union of classify_invalid lists as twin_eval's --invalid input: {"invalid": [records]} (each id placed in its
    design cell by its record, so a run without a directory is still placed)."""
    ids = sorted({str(i) for lst in lists for i in lst.get("invalid", [])})
    unknown = [i for i in ids if i not in records]
    if unknown:
        raise BatchError(f"invalid ids that the batch never planned: {unknown[:5]}")
    keys = ("run_id", "world", "day", "zone", "condition", "split", "engine")
    return {"contract": "twin.invalid_all/1", "invalid": [{k: records[i][k] for k in keys} for i in ids], "crashed": []}


# ------------------------------------------------------------------------------------------------ cost


def container_usd(wall_s: float, cpu: float = CPU, memory_gib: float = MEMORY_GIB) -> float:
    """Modal bill of one container: reserved cores and memory x wall seconds (H1: within 5 % of the real bill)."""
    return float(wall_s) * (cpu * RATE_CPU_PER_CORE_S + memory_gib * RATE_MEM_PER_GIB_S)


def projection(n_worlds: int, n_missions: int, *, observer_wall_s: float, mission_wall_s: float, n_rounds: int,
               cpu: float = CPU, observer_cpu: float = CPU) -> dict:
    """Projected wall clock and $ of a batch: the observer phase (one container per world, in parallel) then
    n_rounds rounds of missions (each round in parallel, ~ one mission wall)."""
    obs_usd = n_worlds * container_usd(observer_wall_s, observer_cpu)
    mis_usd = n_missions * container_usd(mission_wall_s, cpu)
    return {"n_worlds": n_worlds, "n_missions": n_missions, "n_rounds": n_rounds,
            "wall_s": round(observer_wall_s + n_rounds * mission_wall_s, 1),
            "usd_observer": round(obs_usd, 3), "usd_missions": round(mis_usd, 3), "usd": round(obs_usd + mis_usd, 3),
            "assumes": {"observer_wall_s": observer_wall_s, "mission_wall_s": mission_wall_s, "cpu": cpu,
                        "memory_gib": MEMORY_GIB, "rates": [RATE_CPU_PER_CORE_S, RATE_MEM_PER_GIB_S]}}


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    return path


def mission_records(missions: Iterable[Mission]) -> dict[str, dict]:
    return {m.run_id: m.record() for m in missions}


__all__ = ["BatchError", "DAY_S", "DEV_WORLD_SEEDS", "INVALID_ALL", "MAX_ATTEMPTS", "MAX_SLOT_DRAWS", "Mission",
           "OBSERVER_DAYS", "World", "asdict", "day_slots", "container_usd", "day_t0", "merge_invalid", "mission_records", "plan", "projection",
           "reactive_chains", "reruns", "rounds", "run_id", "scored_attempt", "slot_t0", "write_json"]
