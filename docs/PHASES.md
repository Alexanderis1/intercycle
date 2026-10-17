# Unsupervised phase discovery (DIRECTION §4) — SYNTHETIC

Status: interface committed; results below are being filled in (dev numbers until the final run).

## Which machines expose named sub-phases in truth

`Kinematics.phase(mid, t)` returns only `("active"|"idle"|"hold"|"trip", frac)`. Named sub-phases are rebuilt **in the
scoring script only** (`scripts/phase_discovery.py`) from `phase()` + the kinematics constants:

| Machines | Kind | Truth sub-phases (fraction f of the active part) | Source |
|---|---|---|---|
| S10–S40 | weld_station | weld f < 0.80, swing_out 0.80–0.86, dress 0.86–0.92, return 0.92–1.0, idle | `T_WELD_END`, `T_DRESS_END` (+ the docstring's 0.86 swing/dress split) |
| S50 | framing | same | same |
| S60, S70 | respot | same | same |
| S80 | unload_lifter | lower 0.10–0.30, down 0.30–0.55, raise 0.55–0.80; carriage overhead (f < 0.10, f ≥ 0.80) is above `H_CLEAR` = 2.2 m and invisible on the floor → merged with idle as **clear** | `lift_height()` breakpoints |
| HEM1 | roller_hemming | approach (to the first path point, `arm_specs[].brk[1]`), path to 0.92, return 0.92–1.0, idle | `_timeline_path` |
| SEAL1 | sealer | same | same |
| AGVDOCK | not a machine in `phase()` (a parked vehicle, part of L2) | — | not scored |

So every machine exposes named sub-phases; all are scored on sub-phases **and** on cycle boundaries (onset error).
Coarse set for the weld family (the DIRECTION list "weld / tip-dress / transfer / wait"): weld / tip_dress (0.80–0.92) /
return / idle.

## Interface (for workstream M)

```python
from intercycle.observer.phases import PhaseModel
pm = PhaseModel.fit(observations, scn)          # iterable of mapping.Observation, a training window (>= ~1 h, ideally with a stop)
pm.update(obs)                                  # online, every tick in time order (causal forward filter)
pm.phase_bin(machine_id: str, t: float) -> int  # 0..n_bins-2 discovered sub-phases in cycle order (idle = n_bins-2),
                                                # n_bins-1 = blocked/stopped, -1 = unknown
pm.n_bins(machine_id: str) -> int               # sub-phases + 1 blocked bin (5..9)
pm.bin_names(machine_id: str) -> list[str]      # ["p0", ..., "idle", "blocked"]
pm.line_state(t: float) -> str                  # running | andon | break | shift_change ("unknown" before data)
pm.save(path); PhaseModel.load(path, scn)       # fit once, load in worker processes
```

Cost: fit ≈ 1 min for 3.5 h of 2 Hz data (10 machines, numpy motif engine; `phases.MOTIF_ENGINE = "stumpy"` adds a
one-off ~80–110 s numba JIT per process). Online `update` ≈ 1 ms per tick for all 10 machines.
