# A-lite mini pre-registration (PLAN_TWIN §7, lane S8): SIMULATED, descriptive only

Status: v1, written before any pre-registered run. Committed with FREEZE (08:30); the FREEZE commit names the code sha.
Runs 09:00-10:30 on Modal CPU (`cloud/modal_alite.py`, app `s8-alite`). Not part of the sealed claim, never run by
`scripts/sealed_run.py` (tests/test_alite_isolation.py), never on Isaac.

## 1. What is simulated

Everything. A generated body shop (the existing body-shop scenario, `scenario/`), no real data:

- **World (W-W1):** `intercycle/alite/world.py` KeyedWorld. Every production day (06:05 to 06:05) draws its line and
  cell cycles from the keyed RNG `twin.rng.keyed_rng(world_seed, day, entity, event)`, so a day does not depend on the
  query window. The body-shop World (`intercycle/sim/world.py`) is unchanged; it replays one stream per query window
  (finding, tests/test_alite_world.py).
- **Robot programs (W-W2):** `intercycle/alite/programs.py` ProgramKinematics. Per station 2 programs (model A / B)
  from a keyed model-mix sequence (70 / 30, the body carries its model down the line), tip-dress every N cycles with N
  keyed in [8, 30], path noise U(±3 cm) per cycle and arm.
- **Perception:** the camera is `intercycle.observer.mapping.CameraShim` with MEASURED_SHIM (a truth-derived noise
  model: pose sigma 0.285 m, drop 0.32, gross errors 0.15, false positives; no occlusion model). The robot's own camera
  is the body-shop event sim's model (truth-derived: 5 Hz, occlusion bursts, misses, false positives).
- **Observer:** `intercycle/alite/observer.py`, observer_live-style (ObsV2Converter -> MapBuilder / LayeredMap ->
  learn_cycles), never `intercycle/sim/observer_learned.py`. Its imports never reach truth (test_alite_isolation.py).
- **Phase origin (T-W2):** `intercycle/alite/phase_origin.py`. The robot's own_phase (masks.own_phase formula) is put
  into the map's bin origin by a learned lag per machine (map phase = own_phase + lag / P), calibrated on day 0 by
  replaying the fast layer against the map phase. Test: within 0.5 bin (tests/test_alite_phase_origin.py).
- **Robot:** the A-lite CPU double `intercycle/alite/episode.py` (numpy, event-sim style; not Nav2, not Isaac).

## 2. Run A: the CPU double (item 5)

**Worlds:** 811, 812, 813, 814 (KeyedWorld seeds, dev range). They were never run before this file was written.
World 801 (and 802 in short runs) was used for development, tests and the dry run; it is not in the table.

**Episode** = one world x one condition, 2 generated days:
- day 0, 06:05 to day 1 06:05 (24 h): the observer learns; the robot is off. At day 1 06:05 the observer learns its
  slow layer (cycle models, line offsets, phase origin) on day 0 and keeps running.
- day 1: 8 missions of 720 s, starting 07:00, 08:00, 10:00, 11:00, 13:00, 15:00, 16:00, 18:00. Each mission is one
  round of the inspection tour (SiteGraph mission order, machine targets). At each target: a fence capture (1.0 s
  exposure from the hazard boundary), then a timed entry into the zone.

**Conditions** (same world, same camera noise, same robot speed / dwell factors in all three):
- `static`: captures on arrival; never enters (static keep-out of the whole zone).
- `reactive` (best reactive, own camera only): captures once its camera sees the machine still; enters at a still
  onset when the median of the last 5 complete still runs its camera saw there is >= inside_s + margin; declines after
  3 such runs say it is too short; waits at most 120 s.
- `learned` (InterCycle on the A-lite observer): captures at the start of a predicted idle window (fast layer,
  `robot.predict.idle_windows`) long enough for the exposure, when the map's L1 envelope at the predicted phase bins
  stays inside the machine's parked footprint and its own camera sees no motion; enters at the start of a predicted
  idle window >= inside_s + margin with its camera clear; declines when the slow layer says the idle never fits or when
  an UNKNOWN box (M-W4) it received at mission start touches the zone (the live push C-W7 is cut). No prediction:
  waits up to 120 s, then the reactive capture, and declines the entry. Any unknown input: not parked, no entry.

**Fixed parameters** (no tuning for any condition): CAPTURE_S 1.0, GUARD_S 0.3, WAIT_MAX_S 120, fast-layer issue every
2 s with 0.08 s latency, robot step 0.2 s, speed / dwell jitter ±5 %, parked footprint = smallest L1 bin envelope with
>= 8 visits dilated by 1 cell, reactive median of 5 runs, decline after 3; the scenario's inside_s, safety margin,
trip recovery and walkway rule (the body-shop camera rule, in every condition, with one change from E-W6: a transfer
counts as seen only on positive evidence, 3 visible samples with motion; in `intercycle/sim/run.py` an occlusion or a
false motion counts and the robot then steps on at a random point of the transfer cycle).

**Metrics** per condition, summed over the 4 worlds x 8 missions: captures attempted / passed (passed = the machine's
arms were stowed during the whole exposure, truth), entries attempted / clean / declined, trips (zone active when
entered, 90 s protective stop; walkway trips included), holds (the machine's next start had to wait for the robot),
line time lost (s), Y = passed captures + clean entries, set to 0 for a mission with any trip or hold; Y per mission.

**Table:** the one printed by `scripts/alite_run.py` / `cloud/modal_alite.py` (columns: condition, episodes,
missions, the metrics above, Y per mission), plus the per-world rows in the JSON. Descriptive only: no p-values, no
confidence intervals, no ranking claim beyond the numbers.

## 3. Run B: M-W4 change detection and the learned envelopes (item 4)

`intercycle/alite/mw4.py` via `cloud/modal_alite.py` (`scripts/alite_mw4.py` locally), worlds 811-814:
- the observer learns 24 h from day 0 06:05; at day 1 06:05 a program change (S60 switches to a third program,
  model C) and 4 new arms appear: HEM1 east 0.5 m (P 50 s), SEAL1 north 1.0 m (55 s), HEM1 west 1.5 m (37 s), SEAL1
  east 2.0 m (42 s) from the cell's zone; the run goes on for 30 min.
- **time to UNKNOWN** per change: the first time after which every 1 Hz sample of the next 60 s has all the cells the
  change occupies inside an UNKNOWN box (or an offline camera's area); inf if never in 30 min.
- **false alarms:** change events (`inzone_change`, `change_detected`) over the 24 h before the change.
- **envelopes:** over the learning window, at 1 Hz, the truth envelope of each machine and phase bin (the map's own
  bins) = the map's L1 cells the true arm circles occupy in >= 5 % of the samples of that bin (the map's own rule);
  scored on the learned map just before the change, over the bins it has learned: coverage = |truth & learned| /
  |truth|, IoU = |truth & learned| / |truth | learned|, micro-averaged; truth cells outside the map's grid count as
  missed.

## 4. Claim (filled in from Run B only, SIMULATED)

"In a fully simulated body shop (no real data; perception is a truth-derived noise model without occlusion), the
observer code (branch lane/S8-alite, <FREEZE sha>) learned time-binned arm envelopes covering X % (IoU Y); on N
injected new arms it marked the cells UNKNOWN after a median of Z min, with F false alarms over H simulated hours."

- X, Y: the median over the 4 worlds of coverage and IoU.
- N = 16 (4 new arms x 4 worlds); Z = median time to UNKNOWN over those 16 (an arm never marked counts as inf and is
  reported); the 4 program changes are reported next to it, not in N.
- F = false alarms summed over the 4 worlds; H = 96 (4 x 24 h).
- Run A is reported as its table with the same SIMULATED label; no claim sentence beyond the table.

## 5. Rules

- No change to code, parameters, worlds or metrics after FREEZE. A failed episode is reported as failed. Only an
  infrastructure failure (container lost, timeout) is re-run, once, same seed, disclosed.
- Budget: estimate ~$0.3 (Section 6); stop and report at $0.5.
- Dry run (done before FREEZE, local CPU, worlds 801 and 802): `scripts/alite_run.py --dry-run` (30 min of learning,
  one 300 s mission on day 0) runs for all three conditions; the same dry run on Modal (`--dry-run --worlds 801,802`)
  gave identical numbers (learned Y 8, reactive 7, static 2 over 2 missions; no trip, no hold).

## 6. Cost and time

Measured locally (world 801, one full learned episode, laptop CPU shared with other jobs): 1421 s wall, under 300 MB
of memory. The dry run took 33 s on Modal against 44 s locally. Per episode one Modal container (1 CPU, 2 GiB);
12 episodes + 4 M-W4 worlds run in parallel, about 25 min of wall time. Estimate: ~13 000 container-s for the
episodes + ~3 600 for M-W4, x ($0.0000131 / core-s + 2 x $0.00000222 / GiB-s, H1 prices) = about $0.3.

## 7. Known limits (disclose with the results)

- Everything is simulated; the camera and the robot camera are truth-derived noise models, no occlusion by bodies
  or people in the observer camera, and the robot never appears in the observer camera (W-W10).
- S50 (framing) has a bimodal nowcast from its dwell pauses: its learned lag is ~18 s off its offset and only 18 % of
  its samples fall within 0.5 bin (other stations 88-98 %). It is in the runs, not excluded.
- The CPU double is a numpy event sim, not Nav2 and not the Isaac double; the robot's motion is graph travel at fixed
  speed.
- The learned envelopes are larger than the truth ones (camera pose noise): the IoU is expected well below the
  coverage.
- Robots get UNKNOWN boxes at mission start only (C-W7 cut); in Run A no change is injected, so any UNKNOWN box there is
  a false alarm, which can only make `learned` decline more.
- Any time-of-day structure (mission times, the shift-change stop) is ASSUMED, from the body-shop scenario.
- The phase-origin calibration can leave a machine without a lag (S80 lifter on world 801: fewer than 20 usable
  pairs); the learned robot then never trusts the map's parked check there (fail closed) and uses its other rules.
- Run A's JSON lists the observer's change events on day 1 (world 801 dev run: 7 `inzone_change` events near 18:00,
  no change injected); they are false alarms and are reported with the table.
- The development run of world 801 (before the E-W6 walkway change and the mission-start UNKNOWN hand-over) gave
  learned Y 41 over 8 missions, 1 trip, 1 hold; it is not part of the results.
