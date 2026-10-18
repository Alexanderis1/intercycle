"""Option A-lite (docs/PLAN_TWIN.md §7, lane S8): a small, fully SIMULATED body-shop scenario on the existing body-shop
code, CPU only, descriptive only, never part of the sealed claim.

Rules this package keeps (PLAN_TWIN §2 "File owners", §7 "Timing"):
- new modules only: nothing here changes a file that scripts/sealed_run.py imports, and sealed_run.py never imports
  this package (tests/test_alite_isolation.py);
- the observer side (`phase_origin`, `observer`) never imports ground truth (intercycle.truth, intercycle.sim, twin
  truth modules), the same rule as the rest of intercycle/observer; the truth side (`world`, `programs`, `episode`,
  `mw4`) is the simulated plant and the harness that scores it;
- observer = observer_live-style (learn_cycles + nowcast on the zone motion of a noisy camera), never
  intercycle.sim.observer_learned (it has a truth hook, R-W1).

Modules:
    world      W-W1: body-shop truth windows from the shared keyed day-script RNG (twin.rng.keyed_rng)
    programs   W-W2: 2 robot programs per station from a seeded model-mix sequence, tip-dress every N in [8, 30]
               cycles, +-3 cm path noise
    phase_origin  T-W2: one phase origin for the map's L1 bins and the robot's own_phase
    observer   the A-lite observer (noisy camera -> learn_cycles / nowcast, LayeredMap + M-W4 change detector)
    mw4        M-W4 scenario: a program change and a new arm 0.5-2 m from an existing cell, time to UNKNOWN
    episode    the CPU double for A-lite (conditions learned / best reactive / static)
"""
