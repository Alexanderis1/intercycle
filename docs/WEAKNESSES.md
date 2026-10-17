# InterCycle — weaknesses register (monitor session amd-ce, Sat 17 Oct 2026 ~20:30 CEST)

Written at the user's request after pausing the implementation session: "I can't accept those weaknesses: write them down and
do the same analysis for each part of this project." Every part below is explained (how it works, from the code), then
every weakness is listed with evidence (file:line), a REAL fix (not just a disclosure), the effort, and whether it changes
headline numbers. Sources: seven read-only code analyses of the committed state at `0af3e8a`, plus the earlier audit
(`docs/AUDIT.md`) and production brief (`docs/PRODUCTION_BRIEF.md`). Line numbers refer to that commit.

Standard used: **a product a real car body shop could install, with real people, using its existing ceiling cameras.**

## 0. The ones that matter most

### Safety-direction defects: the product could tell a robot "free" when it is not
1. **M-W4: blind inside known zones, no forgetting.** A new arm within 1 m of an existing zone is absorbed into it. Map counts
   never decay, so a newly occupied cell stays **broadcast as FREE for ≈ 9-30 h** after a program change or a new machine
   next to an old one.
2. **T-W2: phase-origin mismatch.** The map's envelope bins start at the skid index; the robot's phase starts at the zone's
   own motion. The gap is up to **2.8 bins** but the robot pads only ±1, so it can look up the wrong envelope.
3. **T-W6: occlusion reads as "no motion".** The timing signal treats an occluded camera as an idle machine, which is the
   dangerous direction (a person standing in front of the camera = "safe to enter").
4. **C-W5: key management fails open.** Without a configured keyring the robot silently trusts the PUBLIC demo key, so
   anyone could forge "idle" windows.
5. **M-W10: camera health is a heartbeat.** A frozen or obstructed camera reads as empty floor, and a camera with no visits
   gives p = 0 instead of UNKNOWN.

### Results that are not what they appear to be
6. **W-W1: the simulated "week" is one day replayed.** The truth RNG restarts with every query window (`machines.py:31`,
   verified), so all days of a seed are near-identical and held-out day 7 is correlated with training. Every convergence
   and held-out map number is affected.
7. **R-W1 = T-W1: the robot's "learned" timing isn't the audited observer.** It comes from a truth "zone active" flag with
   assumed 8 %/3 % noise, not from the measured-error camera (32 % miss) nor from pixels.
8. **W-W4 = R-W2: entering a guarded cell is too cheap.** Automatic hold/restart and a fixed 90 s trip recovery, where real
   cells need a manual reset or a safe-stop request. That rule is what makes entering pay off.
9. **E-W1, E-W3, E-W4: the robot experiment can't produce a meaningful result as designed.**
   - no secondary test can ever reach significance (minimum p 0.00195 > Holm threshold 0.00143);
   - the Inspect criterion needs ≤ 1.37 m range, so even the oracle may score 0;
   - the primary compares against a naive baseline that is sure to lose.
10. **M-W1, E-W6, T-W7: no current numbers.** The map has no scores for its published configuration; event-sim numbers
    predate the fixes; phase discovery is headlined on the ideal camera.

### Capabilities claimed but not running
11. **V-W1, M-W3: no pixels → map path**, no field of view, occlusion or parallax; the detector finds people only.
12. **C-W1: neither Zenoh nor BLE is exercised**; workstream C isn't implemented.
13. **G-W2: nothing consumes Gemma's output**; `verify_state` is unmeasured; the privacy mask uses truth (G-W1).
14. **W-W2, T-W3, X-W1: the same robot program every cycle, identical across seeds.** Phases and envelopes are trivially
    periodic, and the pixel demo is circular.

### Effort, honestly
Everything below adds up to roughly **260 hours** of work. The freeze is Sun 12:30 (≈ 16 h away), with ≤ 5 agents.
So not everything can be fixed before submission. Proposed split:

**Must fix before any number is shown (≈ 36 h of agent work + compute; ≈ 8-10 h wall with 4-5 parallel agents):**

| Item | Hours |
|---|---|
| W-W1 RNG keyed by absolute time + rebuild | 3 |
| M-W4 in-zone change detection + forgetting | 5 |
| T-W2 phase origin | 2 |
| T-W6 occlusion = missing | 3 |
| R-W1 / T-W1 robot timing from the measured-error camera | 5 |
| W-W4 / R-W2 per-zone safeguarding classes (needs your decision) | 3 |
| E-W1 + E-W2 Amendment 2 | 1 |
| E-W3 real Inspect targets + zoom head | 4 |
| E-W4 best-tuned reactive baseline | 3 |
| C-W5 fail-closed keys | 2.5 |
| M-W7 margin calibrated at the real latency | 4 |

Plus re-running all numbers (M-W1, E-W6, T-W7).

**Second tier (realism, ≈ 30 h):**

| Item | Hours |
|---|---|
| W-W2 / T-W3 program variety and model mix | 13 |
| M-W2 realistic detection noise | 5 |
| T-W5 multi-segment cycles from P's HMM | 6 |
| W-W3 stop model | 4 |
| R-W3 camera-gate motion model | 1.5 |

**The rest:** written as the roadmap, with this register as the honest record.

Which tier fits is the user's call for the new lead session.

---

## W. The simulated plant (ground truth every result is measured in)

### How it works
1. **Layout** (`scenario/bodyshop.yaml:9-15,63-70`):
   - a 96 × 48 m hall + a 24 × 24 m data hall;
   - the line along z = 24 (conveyor x 6-86);
   - aisles N (z 30-34), S (14-18), a transport aisle (6-10), a market (0-5), closures (z 36-48).
2. **10 machines** (`yaml:44-53`):
   - **8 line stations** on one clock (takt 60 s, σ 1 % = 0.6 s, drift 0.1 %/h, 1.5 s stagger): S10-S40 weld, S50 framing,
     S60/S70 respot, S80 unload lifter;
   - **2 off-line cells** with their own clocks: HEM1 (45 s, σ 0.8 %) and SEAL1 (60 s, σ 1.2 %), micro-stops 2/h of 5-30 s.
3. **Cycle** (`kinematics.py:13-18,64-70`):
   - two 2-link arms per station (2.6 m links);
   - fixed fractions: weld 0-0.80, swing-out 0.80-0.86, tip-dress 0.86-0.92, return 0.92-1.0. **Tip-dress every cycle**;
   - skid transfer 8 s; lifter 6 m → 0.35 m; AGV dock every 6th cycle for 20 s.
4. **Andon** (`machines.py:44`): Bernoulli per cycle (≈ 4/h), U(20, 120) s, always the whole line, no cause.
5. **Calendar:** 3 shifts, 6 breaks, 5-min shift-change stops, identical every day; cells stop too.
6. **People** (`agents.py:206-225`):
   - 10 walkers on straight back-and-forth segments (z ≈ 32 or 16), 0.9-1.4 m/s, pauses 5-60 s;
   - 6 people per break crowd, fully deterministic;
   - 2 data-hall techs.
7. **Vehicles:** 4 AGVs on a fixed loop (1.2 m/s), 2 tuggers (1.8 m/s), 3 forklifts (1.6 m/s, 25 s pauses).
8. **What changes per seed:** line jitter/andon, cell timing, walker phases/endpoints/speeds, impassive flags, episode start.
   **What is identical across seeds and days:** layout, arm paths (`rng([7,id,r])`, `kinematics.py:252`), all vehicle
   routes, break crowds, calendar.
9. **Freezes** (`world.py`): a trip freezes the line mid-motion; a hold shows the arms at home; planned stops are pinned to
   the wall clock.
10. **Robot plant** (`plant.py`, 0.1 s tick):
    - **trip:** footprint (3 circles, r 0.32) enters an ACTIVE zone → that group frozen for a fixed 90 s, automatic restart;
    - **hold:** inside a zone that is due to start → hold extended tick by tick, released when the robot leaves;
    - **people:** PySocialForce (τ 0.5, A 5.1, λ 2.0), 20 % impassive;
    - **forklifts:** brake after 1.2 s at 1.5 m/s²; **AGVs:** ISO 3691-4-style fields;
    - **shapes:** person cylinder r 0.3 × 1.75 m; arm columns r 0.5-0.65 × 2.0 m; forklift 2.2 m body + 1.0 m load box.

### Real / fair stand-ins (disclose)
- Every timing, rate and headcount is SYNTHETIC.
- Social-force parameters come from the literature.
- AGV fields follow ISO 3691-4 logic.
- Arm envelopes are plan-view circles.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort | Changes headline? |
|---|---|---|---|---|---|
| W-W1 | **Truth depends on the query window, not absolute time.** The RNG restarts each time a `World` is built, and drift is measured from the window start. The same hour on day 0 and day 6 differs by ≤ 6e-10 s. The "1-week" map is ≈ 28 replays of one random stream, and held-out day 7 is correlated with training. A single 7-day World instead drifts takt 60 → 69.3 s. | `machines.py:42` | Counter-based RNG keyed by (seed, day, cycle index); drift as a bounded mean-reverting process; rebuild the map and all splits. | 3 h + 1 heavy job | **Yes, all map metrics** |
| W-W2 | **Robot programs are unrealistic.** Paths are seed-independent and identical every cycle; arm speed scales with jitter; tip-dress every cycle; no model mix. Phases and L1 are trivially periodic. | `kinematics.py:252`, `:64-70` | 2-4 program variants per station from a seeded model-mix sequence (e.g. 60/30/10); fixed program speed, jitter absorbed in waits; tip-dress every N ∈ [8, 30] welds + cap changes; ±3 cm path noise. | 5 h | **Yes, phase ARI, L1 IoU** |
| W-W3 | **Andon is too simple.** Uniform 20-120 s, always whole-line, no cause; no micro-stops < 10 s, no starving/blocking, no long stops. | `machines.py:44` | Per-station faults with lognormal durations (median 60 s, tail 30 min); 20-60 micro-stops/h of 2-10 s; buffer starve/block; cause labels. | 4 h | Yes, states, robot gain |
| W-W4 | **Safety semantics are optimistic.** A trip auto-restarts after a fixed 90 s; a hold costs only the overlap. Real practice: protective stop + manual reset from outside the guarding (ISO 10218-2, ISO 13849, IEC 62046), minutes of recovery; perimeter light curtains cannot sense presence inside, so holds need area scanners; entering a running cell is a reportable incident. | `world.py:144`, `plant.py:402-409` | Per-zone `guard: curtain \| scanner` from the site plan. Trip = lognormal recovery (median 4 min, p90 15 min) + an incident flag that fails the episode. Curtain zones enterable only after a requested safe operating stop (or not at all); scanner holds need a 10-30 s acknowledgement. | 3 h | **Yes, robot results** |
| W-W5 | **People are sparse, scripted and inconsistent.** YAML says 34/30/14 per shift; `agents.py:207` makes 10 walkers + 2 techs; `traffic.py` uses the YAML numbers, so two inconsistent truths. No operators at stations/cells, no crossings, groups or carts, no shift-change surge. Break crowds identical every day and seed. | `agents.py:206-225` | One generator from `people_per_shift` with roles (operators, team leaders, maintenance called to andons, logistics) and a per-day sampled origin-destination matrix; derive traffic counts and the render from it. | 6 h | Yes, L3, robot safety metrics |
| W-W6 | **Vehicles are deterministic.** Fixed routes/speeds across seeds and days; forklifts never reverse and ignore other vehicles. This is why the vehicle map shows no skill over climatology. | `plant.py:456-459` | Dispatch tasks driven by line consumption; 30-50 % reverse legs; vehicle-vehicle yielding; temporary parked loads. | 4 h | Yes, L2 |
| W-W7 | **Crowd behaviour is narrow** (force term only). | `social.py` | Stop-and-watch (10 %), approach (5 %), standing groups at crossings, impassive share swept 20-40 %, reduced vision when carrying. | 3 h | Robot metrics |
| W-W8 | **Collision geometry gaps.** Arms are 2 m columns with no overhang; bare forklift forks sit below the lidar; the 22-slot arm-collider pool can overflow (three stations within 8 m need 24), so physical contacts are missed. | `plant.py:48`, `movers.py:159-166` | 3D capsules from the kinematics; low fork-tine boxes; dynamic pool + overflow assert. | 2.5 h | Possibly collisions |
| W-W9 | **The world never changes.** Static geometry exactly equals the map: no clutter, maintenance or teach mode with a person inside a cell, weekends, overtime or changeovers. | `world.py` | Daily clutter generator; PM and teach-mode events; realistic calendar. | 4 h | Yes, unsafe-free, new-machine false alarms |
| W-W10 | **The observer never sees the robot**; no field of view or occlusion. Robot-caused holds look like an idle machine. | `broadcaster.py:8` | Inject the robot into the observer's camera stream as an unknown mover; test it doesn't pollute L3 or trigger the new-machine detector; per-camera FOV + overhead occluders. | 3 h | Possibly |
| W-W11 | **The new machine is too kind.** One arm, 30 min of commissioning, ignores breaks, nothing static changes around it. | `newmachine.py:368-379` | Weeks of teach-mode motion with people inside, new fence geometry, stops at breaks. | 2.5 h | Yes, new-machine numbers |
| W-W12 | **Cells behave like CNC machines** (σ 0.8-1.2 %) although they are manually loaded. | `yaml:44-53` | Operator-paced loading, lognormal σ ≈ 15 %, operator present at the cell. | 1.5 h | Phases |

### What only a real site can show
- The real stop distribution, model mix and program edits (2-4 weeks of passive recording).
- EHS sign-off on entry rules, reset procedure and interlock design.
- Real camera coverage, occlusion, lighting and weld flash.
- How workers react to a legged robot (and works-council consent).
- Wi-Fi/BLE in RF-noisy welding halls.
- Behaviour through holidays, retooling and night maintenance.

---

## M. Observer map chain: what the cameras "see" → map → metrics, and the new-machine detector

### How it works
**Camera** (`CameraShim`, `intercycle/observer/mapping.py:274-487`):
- `build_map.run_window` (`scripts/build_map.py:139-156`) fills a `SimState` from the truth simulator:
  - movers (true class, footprint circles, heading, speed);
  - machine part circles (`kin.bodies()`; the S80 carriage reported only below 2.2 m);
  - skids (car bodies, 3 circles each); AGV dock as a vehicle.
- **Taken straight from truth:** class; radius (+N(0, 0.03)); heading (+N(0, 0.15 rad) when speed > 0.2); the `moving`
  flag (true speed, or true displacement > 2 cm for parts); `machine` = the zone (+1 m) the noisy point lands in.
- **MEASURED_SHIM** (`:57`):
  - 32 % iid miss per circle per frame;
  - position error 85 % Gaussian σ 0.285 m + 15 % gross error U(5, 20) m, drawn fresh every frame;
  - false positives 0.124 × person detections, PERSON class only.
- **No field of view, occlusion or parallax:** each detection is assigned to the nearest of 10 wall cameras (Voronoi); the line
  at z = 24 is the seam between camera cells.
- **Timing:** 2 Hz; 0.2 s constant latency.
- **Offline cameras:** Poisson 10 s blackouts, 4 per hour.

**Conditioning** (`ConditionTracker`, `:506-696`):
- machine bit = any moving PART in its zone; line bit = any moving SKID;
- phase = (t − last rise)/P; **16 equal bins** (P's discovered phases are off, `use_bins=False`, `:1150`);
- line stations use the skid clock; production state = P's `line_state` name, else the ≥ 50 %-overdue → ANDON rule;
- under measured noise the 09:00 break is labelled andon 100 % of ticks, so the map has no break slices.

**Map** (`LayeredMap`):
- **L0** is not in the observer (the robot's CAD map);
- **L1:** 0.25 m grid over each zone + 1 m; PART/SKID hits per bin; p = hits/visits; envelope = p ≥ 0.05; `shrink_m` was tuned
  for σ 0.15 and never re-tuned (`:937`); broadcast as ≤ 12 outer boxes; `predicted_bins` widens by jitter;
- **L2/L3:** a 1 m grid of 120 × 48 cells × 48 times of day × 4 states; p = hits/visits, not corrected for miss or false
  positives; a slice is trusted from ≥ 300 ticks;
- **UNKNOWN:** camera regions with < 300 ticks, change candidates, machines without an envelope, offline cameras,
  unresolvable slices;
- **hand-given:** hall extents, machine rects/kind/`on_line`, camera positions, the 1 m margin.

**Metrics** (`intercycle/eval/map_metrics.py`):
- L1 unsafe-free = FN/(TP+FN) over cell-instants (cell-centre and strict); false-occupied; IoU/precision/recall per bin;
- L2/L3: Brier, AUROC, skill vs pooled climatology (block bootstrap), PR-AUC;
- **conformal margin** (`:609-661`): 120 s episodes, λ = min m with n/(n+1)·R̂ + 1/(n+1) ≤ α; calibrated with map_5d on
  days 5-6, tested on day 7 of seeds 0/1/2; only 4 windows per split.

**Current numbers for the published (measured-shim, calendar-free) configuration: NONE.** The maps and split recordings exist;
no score/conformal files. The superseded assumed-shim run: L1 IoU flat 0.78 from 1 h to 1 week; strict unsafe-free 0.0082
without margin; vehicle Brier = climatology.

**Change detection** (`intercycle/observer/change.py`):
- residual = moving PART detections outside zones + 1 m; CUSUM S = clip(S + x − 0.05, 0, 20), alarm at h = 10 (hand-picked);
- candidate region → UNKNOWN at once; spawn when `learn_cycles` is confident (≥ 6 cycles, two estimates within 2 %, not
  phase-locked to a neighbour); converged when all 16 bins have ≥ 8 visits + 4 cycles;
- bump check: envelope matching over shifts ≤ ±2 m;
- **truth hook:** NEW1's circles are injected as perfectly classified PARTs, at the easiest placement (7.75 m from S80's zone,
  50 s period vs 60 s takt);
- **the only run** (seed 0, 8 h, written 18:34 on the OLD shim): UNKNOWN after 5.8 s, spawned at 40 min, converged at 44 min,
  transition unsafe-free 0, **post-convergence 1.57 % strict**, 1 false alarm.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort | Changes headline? |
|---|---|---|---|---|---|
| M-W1 | **No headline numbers exist for the current configuration.** | `out/timeline/` | Run `map_timeline.py replay` + `report`, and `new_machine.py` on the measured shim. | 0.5 h + 2-3 h compute | **Yes, all** |
| M-W2 | **Truth class and truth `moving` flag; false positives for people only; iid errors.** A skid's 3 circles × ~8 bodies are essentially never all missed, so the line clock and phases are immune to noise; per-frame gross errors average out (real ID switches persist). | `mapping.py:349-367, 450-481` | Markov miss bursts (occlusion), gross errors persisting 2-10 s, false positives for every class including PART, a class confusion matrix, motion flag computed from noisy positions; rebuild. | 5 h | **Yes, worse** |
| M-W3 | **No field of view, occlusion or parallax.** A point 1.5 m up on an arm, 22 m from a 9 m camera, projects 4.4 m away; σ 0.285 m was measured on people's feet. | `mapping.py:334` | Frustum + ray-cast against L0, car bodies and stations; occupy the height-aware shadow swath along the camera ray; never-visible cells → UNKNOWN. | 10 h | **Yes** |
| M-W4 | **Blind inside known zones; no forgetting.** A new arm within 1 m of a zone is absorbed (`change.py:95-100, 149`). Counts never decay: against a 1-week denominator a newly always-occupied cell needs ≈ 9 h (a 30 %-swept cell ≈ 30 h) to reach p ≥ 5 %, and until then it is broadcast as **FREE**. A robot program change behaves the same. **This is a safety-direction defect.** | as cited | CUSUM on in-zone hits in cells with p < p_min → UNKNOWN immediately; exponential count decay (24 h half-life). | 5 h | New failing tests until fixed |
| M-W5 | **New-machine evidence = one easy case, on the old shim**; convergence = 4 s of data per bin; leaks 1.57 % strict after convergence. | `out/newmachine/result.json` | Placements inside / adjacent / takt-synchronised / on the camera seam × 3 seeds, measured shim; converge only when the per-bin envelope changes < 2 % Jaccard over 3 cycles; margin + ±1-bin widening on spawned models. | 7 h | **Yes** |
| M-W6 | **Thresholds and stress model are from the same author**; false alarms judged on 8 h only. | `change.py:9-10` | h from a target of ≤ 1 false alarm/week on training residuals under the measured shim; false alarms/h over 7 days × 3 seeds. | 3 h | **Yes** |
| M-W7 | **The conformal margin is optimistic:** calibrated at zero latency (parts move at p99 ≈ 3 m/s, so 1 s ≈ 3 m); controls the mean, not a per-episode bound; correlated episodes; calibration and test on the same 4 windows; unweighted vs weighted means. | `map_metrics.py:507-508, 609-661` | Shift truth by the real fast-layer age; Learn-then-Test or a per-episode quantile; calibrate over all 48 time slices; weight consistently. | 4 h | **Yes, the margin grows** |
| M-W8 | **Breaks become andon**: the robot never gets a break slice; andon slices mix breaks and real andons; P's fit window has no shift change. | MAP_RESULTS | Key L2/L3 on P's `line_cluster` ids, not names; fit P on a full day; broadcast the cluster + its learned duration. | 5 h | **Yes** |
| M-W9 | **Traffic layers uncalibrated; people never scored.** p not corrected for the 32 % miss (the 0.3 hot-cell threshold means true occupancy ≈ 0.44); no people/vehicle unsafe-free; out-of-building cells inflate AUROC; vehicles show no skill. | `mapping.py`; `map_metrics.py` | Isotonic calibration on calibration days; floor mask; a people/vehicle unsafe-free metric for map + live tracks inflated by speed × latency. | 5 h | **Yes** |
| M-W10 | **Camera health is a heartbeat only.** A frozen/obstructed camera reads as empty floor; a camera with no visits in a slice gives p = 0, not UNKNOWN (`:1115`); bumps only checkable on cameras that see a machine (cameras 4, 7, 8, 9 see none); shifts only, ±2 m; UNKNOWN never lifted. | `mapping.py:1115`; `change.py` | Image watchdog (frame hash, variance, sharpness); UNKNOWN per camera per slice; homography re-registration against envelopes and landmarks. | 7 h | Small numeric, critical for deployment |
| M-W11 | **Evaluation coverage gaps:** half the breaks and the late shift never scored; robot programs identical across seeds; L1 truth clipped to zone + 1 m, so reach beyond the margin is never a miss. | `map_metrics.py` | Score 24 h of day 7 per seed on the site grid; a ±5 % kinematics/program-variant seed. | 3 h + 2 h compute | **Yes** |
| M-W12 | **Static changes are invisible** (new fences, pallets, racks), so L0 goes stale. | `change.py` | Persistent non-moving detections or background differences in L0-free space → UNKNOWN until an operator acknowledges. | 5 h | New metric |

---

## T. Cycle learning, phase discovery, production states and the robot's timing signal

### How it works
**learn_cycles** (`intercycle/observer/learn_cycles.py`), the slow layer. Input: a per-zone binary motion signal (t, y) at 2 Hz.
1. **Cleaning:** Schmitt trigger (3-sample mean, 0.6/0.4) + debounce (idle < 4 s and active < 2 s merged) (`:126-156`).
2. **Period:** autocorrelation (shortest peak ≥ 0.85 × best, parabolic refine), confirmed by Lomb-Scargle 5-300 s at FAP < 1e-3 (`:81-117, :344`).
3. **Cycle tracking:** idle pulses followed with tolerance ±max(6 s, 0.15·active), ≤ 2 missed cycles (`:168-272`).
4. **Statistics:**
   - period = mean of cycles near the median;
   - jitter corrected for edge noise (σ² = var + 2·cov₁);
   - busy fraction = median active / P;
   - stops/h counted; confidence = 15 × the weakest of 6 scores (≤ 3 if jitter ≥ 5 %).
5. **Line pooling** (`:400-464`): stations with confidence ≥ 4 and a period within 3 % get offsets to a reference station and
   one shared index clock.
6. **Nowcast** (`:484-542`):
   - ACTIVE → next-idle-in = rise + busy·P − now; idle length always (1 − busy)·P;
   - IDLE → idle left = idle − age; STOPPED when age > idle + 3σ;
   - TTL = ½·P·(0.025 / 1.96 j)², capped at 10 / stops-per-hour (≈ 2.5 s for the line).
7. **`idle_windows`** (`intercycle/robot/predict.py:9-38`): windows over 900 s, each shrunk by u = 2·j·P·√(k+1) + 0.05
   (1.25 s at 1 % jitter). Assumes fixed P and busy fraction, Gaussian jitter, no andon inside the horizon.

**What actually feeds the robot** (event sim robot B *and* the Isaac learned broadcast), `intercycle/sim/observer_learned.py`:
- **training:** seed 99, 06:10-10:00, from truth windows + noise;
- **live:** `_Camera.upto` = `world.active_many(t − 0.2)` (a truth "zone active" flag) with 8 % occlusion in ≈ 1.5 s bursts
  and 3 % flips (`:87-97`; `signals.py:18-22`), through `line_nowcast` / `nowcast`.

**phases.py** (workstream P):
- **input:** CameraShim observations; per zone (rect + 1 m) a count of moving PART detections + a 4×4 PART occupancy grid
  → log1p, standardised, PCA to 6 dims;
- **template:** period from learn_cycles; matrix-profile motif; open-end DTW; median template rotated to the motion onset;
  exact DP change points, k 4-8 by elbow (every machine ended at k = 8);
- **HMM:** stay / advance / skip 0.15 / 0.70 / 0.15; BLOCKED state; forward filter only;
- **line states:** 7 causal features, k-means++ (k 4-6 by silhouette); rule names + a "gathering" rule splitting break from
  shift change; operator labels carry over by Hungarian matching;
- **seed-2 headline (ideal shim):** welders segment F1 0.85, boundary error 0.57 s. Line states: k = 6, silhouette 0.31,
  ARI 0.35, named accuracy 0.973 vs an all-"running" baseline of 0.80; shift_change recall 0.70 comes only from the
  gathering rule. Not scored under measured noise.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort | Changes headline? |
|---|---|---|---|---|---|
| T-W1 | **The measured perception error never reaches the robot's timing.** The robot's input is a truth "zone active" flag with synthetic 8 %/3 % noise, while the map now uses 32 % miss, 15 % gross error, 11 % false positives. | `observer_learned.py:93-97`; `signals.py:18-22` vs `mapping.py:57` | Drive `_Camera` from moving-PART counts per zone of `CameraShim(MEASURED_SHIM)` (phases' X[:,0]), and from pixel motion when A1 lands, in both the event sim and the broadcaster. | 5 h + reruns | **Yes** (B trips, holds, insp/h) |
| T-W2 | **Map and robot use different phase origins.** The map's L1 bins for line stations start at the skid index (`mapping.py:676-686`); the robot's `own_phase` starts at the zone's own motion start (`masks.py:238-253`). The gap is the station offset, 0-10.5 s = up to **2.8 bins**, but the robot pads only **±1 bin** (`masks.py:262`). The robot can look up the wrong envelope. `phasecheck.py` checks against truth motion start, not the map's origin. | as cited | The observer learns each station's lag from the skid index and puts it in the slow layer; `own_phase` adds lag/P; a test that the robot's phase and the map's phase agree within 0.5 bin. | 2 h | Isaac collision/coverage |
| T-W3 | **Train and test are the same generator.** Even `eval_learner` learns 07:00-07:45 and scores 07:45+ on the same seed. | `eval_learner.py:37-68` | Test scenario variants (takt 66 s, busy ±10 %, jitter 2-5 %, model mix, tip-dress every N); CUSUM on cycle length that drops confidence and re-learns. | 8 h (waiver) | **Yes, down** |
| T-W4 | **`idle_windows` is uncalibrated.** k_sigma 2 under a Gaussian assumption; the only test checks the median next-idle-in; `eval_learner` compares with the nearest true fall (can't see whole-cycle errors) and never scores idle length, state or TTL. | `test_observer_learned.py:41-55`; `eval_learner.py:68` | Score the share of predicted-idle seconds that are truly idle, per machine and horizon; calibrate u conformally on a separate calibration day. | 3 h | Maybe fewer inspections |
| T-W5 | **A cycle is assumed to be one active block + one idle block.** On pixels, a weld/clamp/interlock dwell longer than 4 s reads as idle (the dangerous direction); car-body transfer, operators and sparks read as active. | `truth/machines.py:60-66` | Build `MachineNow` from P's HMM (time to idle = remaining template positions × learned durations); idle = arms in home cells AND no motion longer than the longest in-cycle still run, learned per machine. | 6 h | **Yes** |
| T-W6 | **The noise model was built to match the filter**: 4 s debounce vs 1.5 s mean bursts; independent flips; occlusion reads as "no motion" (fail-dangerous). | `signals.py`; `learn_cycles.py:126-156` | A sample is MISSING when a person/vehicle track overlaps the zone; stress cases: 30-300 s occlusions, an operator in the zone, the car-body transfer. | 3 h | **Yes** |
| T-W7 | **P's headline uses the ideal shim**; under measured noise the break was named andon 100 % of ticks. | `docs/PHASES.md` §0 | Run seed 2 with `--shim measured` (≈ 20 min) and headline it; if break can't be separated, report running/stopped only. | 1 h | **Yes** |
| T-W8 | **P's scoring is generous.** Bins named with test truth; K = 8 maps many-to-one onto 3 labels, so over-segmentation is free; boundary error recall-only; line accuracy without its 0.80 baseline. | `phase_discovery.py:195-201, 333` | Freeze bin→name on the training window; one-to-one (Hungarian) mapping; boundary precision/F1; macro recall; an equal-slice baseline. | 2 h | **Yes** |
| T-W9 | **Line states are unstable, untested across seeds and unused by the robot.** ARI drops 0.74 → 0.35 between seeds; training holds one break and one shift change; robot B still uses the public break calendar (`policies.py:30-34`). | as cited | `line_cluster` (operator-labelled) in the fast layer so robot and map key on the same slice; remove B's calendar input; fit/score on ≥ 3 training days and ≥ 3 test seeds. | 4 h | **Yes** |
| T-W10 | **One seed, hand tuning, a deterministic sim.** Debounce, Schmitt thresholds, k_sigma, the TTL rule, energy weight, gathering thresholds and HMM probabilities were all set by hand on dev data; one training day; one P seed. | various | Pre-register parameters; ±50 % sensitivity sweep; 3 training days; ≥ 5 test seeds with CIs; per-seed kinematics jitter (waiver). | 4 h + 2 h compute | CIs widen |
| T-W11 | **TTL ≈ 2.5 s against a 2 s issue interval**: one lost packet on a real radio kills availability. | `run.py:22`; `learn_cycles.py:469-481` | Separate phase validity (from jitter) from stop risk (already covered by the own camera and shrunk window end). | 1 h | B insp/h in Isaac |

---

## R. Legged robot on Nav2 (R1/R2/R3)

### How it works
Control chain, top to bottom:
```
timing_planner (ours: which target, WHEN to enter)  --Nav2 goals-->  Nav2 Humble (stock)
  bt_navigator -> SmacPlanner2D -> RotationShim -> MPPI Omni (20 Hz) -> velocity_smoother -> collision_monitor -> /cmd_vel
  -> ros_bridge_node (Unix socket, lockstep) -> Isaac Lab runner (Modal L4)
  -> stand mode -> pretrained ANYmal-D rsl_rl policy (50 Hz) -> 12 joint position targets -> PhysX
```
1. **Locomotion** (`ros2/intercycle_sim/intercycle_sim/isaac_backend.py`):
   - the published flat-terrain ANYmal-D velocity policy maps (vx, vy, wz) plus body state to joint targets every 0.02 s;
   - PhysX simulates legs, feet and contacts;
   - **stand mode** (`:162-187`): after 0.2 s of zero command, a P-loop holds the base pose (`:174`, true pose);
   - movers (people, vehicles, arm links) are kinematic bodies moved every 5 steps; contacts on non-foot bodies are logged.
2. **Bridge** (`runner.py`, `ros_bridge_node.py`):
   - each physics step the simulator sends sensor outputs; the bridge publishes `/clock /odom /tf /scan /imu` plus the camera
     observations, and replies with the latest `/cmd_vel` (zero if older than 0.5 s sim time);
   - pacing max_rtf; measured RTF ≈ 0.67.
3. **Sensors** (`sensors.py`), all computed from true state:
   - **lidar:** 2D, 360 beams, 10 Hz, 0.1-25 m, σ 1.5 cm, 0.5 % dropouts, levelled frame at +0.2 m; Warp ray-cast on the
     static mesh merged with analytic cylinders/boxes for movers (`:150-179`);
   - **odometry:** true body velocity × (1 + 2 % scale) + bias random walk + noise, integrated (`:227-250`);
   - **IMU:** published, unused.
4. **Nav2** (`ros2/intercycle_nav/config/nav2.yaml`):
   - AMCL Omni (alphas 0.4) on the static map;
   - SmacPlanner2D on the global costmap (static map + lidar obstacle layer + 1.1 m inflation + KeepoutFilter + SpeedFilter);
   - RotationShim (> 45° → rotate first), then MPPI Omni (500 samples × 2.8 s, vx 0.7 / vy 0.3 / wz 1.0);
   - collision monitor: stop polygon ±0.58 × ±0.38 m (≈ 8 cm beyond the body), slowdown ±1.0 × ±0.55 m to 50 %,
     approach = stop if the commanded motion hits within 1.2 s.
5. **observer_bridge** (`masks.py`, `observer_bridge.py`):
   - verifies the signed broadcast (Ed25519);
   - keep-out mask = zone rectangle unless a predicted idle window covers [now, now+6 s], + L1 envelopes at the projected
     phase bins (±1 bin, +0.2 m), + UNKNOWN cells;
   - when the AMCL pose is in a zone: machine "held" → stop-bin envelope + 0.75 m disc cleared under the robot;
   - speed mask from L2/L3 occupancy; production state = ANDON if ≥ 50 % of machines report STOPPED;
   - restrict-only clamp (`enforce_monotone`).
6. **timing_planner** (`timing_planner.py`, `timing.py`):
   - SIPP over a hand-built site graph (aisle nodes, a wait pose 1.05 m outside each zone, an inside inspection pose);
   - inside nodes carry the predicted idle windows as safe intervals;
   - next target = the one finishing earliest.

### One inspection, step by step
1. Initial pose = the known dock.
2. SIPP picks the target.
3. `NavigateToPose` to the wait pose.
4. Wait (max 180 s) until **(a)** an idle window is open with enter 5 s + dwell + exit 6.5 s + margin left (learned/oracle),
   **and (b)** the camera gate is clear: ≥ 2 fresh "visible, no motion" observations spanning ≥ 0.3 s.
   - none uses (b) only; static never enters.
5. Enter with a deadline of enter time + 3 s, else back out.
6. Dwell. The plant checks the capture after 1 s still:
   - target in image;
   - unoccluded;
   - incidence < 60°;
   - ≥ 1000 px/m (Amendment 1);
   - blur < 1 px.
   The dwell is cut if the window closes.
7. Exit with Nav2 BackUp at 0.35 m/s.
8. The plant penalises crossing into an ACTIVE zone (trip, 90 s line recovery) and staying when the machine wants to start
   (hold).

### Verified real
- physics walking with a learned controller;
- stock Nav2;
- AMCL on drifting odometry;
- lidar hits moving bodies;
- protective stop;
- no truth topic reaches Nav2/bridge/planner (`assert_no_truth`);
- the held rule uses the AMCL pose;
- social-force people (20 % ignore the robot), non-yielding forklifts, AGV protective fields;
- fixed camera aim point (truth-aim leak removed, e06ac35);
- pre-registered statistics.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| R-W1 | **The learned timing the robot uses does not come from the audited observer.** The windows/phase come from `intercycle/sim/observer_learned.py`, whose input is a per-zone *truth* flag "machine active" (`World.active_many`) at 2 Hz with assumed noise (0.2 s latency, 8 % occlusion bursts, 3 % flips; `observer/signals.py:18-22`). Not M's measured-error camera, not P's phase model, not pixels. The measured 32 % miss never reaches the robot. | `observer_learned.py:85-95` | Drive the robot's fast layer from the same observation stream as the map: per-zone activity = moving-part detections inside the zone from `MEASURED_SHIM` (or, once A1 lands, the pixel motion-energy detector), then `learn_cycles` on that. At minimum calibrate `_Camera` P(active seen \| active) and the false-active rate from A1's measured confusion. Re-run the dev check + batch. | 2-3 h + rerun |
| R-W2 | **Entering a guarded cell is too cheap.** Plant rule: robot inside an IDLE zone → automatic hold, automatic restart when it leaves. In real body shops, crossing a weld cell's light curtain gives a protective stop with manual reset, or entry needs a safe-operating-stop request (a safety-PLC interface, which contradicts "no PLC link"). This rule is what makes entering profitable. | `ros2/intercycle_eval/intercycle_eval/plant.py:12-14, 395-409` | Model per-zone safeguarding classes from the site plan: (a) open hazard areas with presence-sensing area scanners (lifter, transfer, AGV dock) → automatic hold, realistic; (b) fenced cells with light curtains → protective stop + manual reset (e.g. 120-300 s line loss) on ANY entry, so the robot inspects them from outside or only with a permit input. Choose Inspect targets accordingly; report both. **User decision.** | 1-2 h + rerun |
| R-W3 | **The camera gate is a near-perfect motion detector.** If ≥ 50 % of the moving parts are in view and unoccluded, motion = true part speed > 0.05 m/s, with p_detect 0.95, no false motion and no platform sway. | `camera.py:150-160` | Pixel-motion model: detect motion only if projected motion > k px per frame (speed·f/range·dt), add false motion from body sway (from Isaac body rates), range/occlusion-dependent detection probability, frame-difference latency. Optionally validate on R4 renders. | 1.5 h |
| R-W4 | **Perfect localisation priors.** The AMCL map is an exact copy of the simulator geometry and the start pose is exact. | `world.py` map export; `timing_planner.py:200` | Build the map with slam_toolbox in sim from noisy odometry; add unmapped clutter (pallets, carts, bins) to the world but not the map; perturb the start pose ±0.5 m / ±10°. | 1.5-2 h |
| R-W5 | **Idealised lidar.** Perfectly levelled (a legged body pitches/rolls); people/arms are simple shapes; arm links are floor-to-2 m columns, so the lidar "sees" arms a real 2D scanner would pass under; 2D only (ANYmal uses 3D lidar + depth cameras); low objects (forks, pallets) invisible. | `sensors.py:150-179`; `plant.py:48, 283-296` | Tilt the scan plane with the true body roll/pitch; arm links at their true heights; people as leg + torso shapes. Next step: 3D lidar (Warp pattern on mesh + analytic movers) into a voxel layer. | 2 h (2D realism); 4-6 h (3D) |
| R-W6 | **Stand mode holds against the TRUE pose** (perfect station-keeping). | `isaac_backend.py:173-187` | Hold against the robot's own estimate (odometry/AMCL) so drift shows up. | 0.5 h |
| R-W7 | **Policy observes the true base velocity.** Standard for Isaac Lab, but no estimator error. | `isaac_backend.py:5-6` | Inject estimator-like noise/bias/latency into the policy's velocity observation, within what the policy tolerates; measure walking quality. | 1 h |
| R-W8 | **Odometry = true velocity + statistical error.** No foot slip, no contact-based estimation. | `sensors.py:227-250` | Leg odometry from joint states + foot contacts (available in Isaac), fused with the IMU in robot_localization (EKF). This also makes the IMU useful. | 3-4 h |
| R-W9 | **Inspection "capture" is geometry only.** No image is produced or checked in the loop. | `camera.py:173-195` | R4: render the capture poses on RunPod (Isaac RTX) and run an image check (sharpness, target visible) on the real render; report agreement with the geometric pass. | 3-4 h + ≈ $2 |
| R-W10 | **Site graph and inspection poses are hand-made** in the scenario. | `intercycle/robot/sitegraph.py`, `timing.py:56-83` | Generate wait/inspection poses automatically from the site plan + target normals + the camera model (pose that satisfies DORI/incidence); a commissioning step with confirmation. | 2 h |
| R-W11 | **The observer never sees the robot**, though real ceiling cameras would. | `broadcaster.py:7-9` | Add the robot as an observed entity in the observer's camera stream; make sure it's excluded from L3 learning (or tagged) and appears in the live layer. | 1.5 h |
| R-W12 | **Nav2 gets ≈ 1.5× compute per sim second** (RTF 0.67), more than an onboard computer. | `runner.py` pacing | Pin Nav2 to a CPU budget matching ANYmal's onboard PC (cpuset / taskset) or report its timing margins. | 0.5 h |
| R-W13 | **No certified safety scanner modelled.** The collision monitor is the only stop, and its 8 cm field isn't a certified protective field. | `nav2.yaml:382-396` | A plant-side safety-scanner model: speed-dependent protective field per ISO 13855 (S = K·T + C); count protective stops and stopped time as a metric. | 1.5 h |

---

## C. Broadcast, codec and channels

### How it works
1. **Wire format** (`intercycle/codec/layout.py`):
   - a signed 12 B header: magic, version, layer, site_id, key_id, issue_time;
   - **slow** = cycle models (242 B); **fast** = state, next_idle_in, next_idle_len, TTL rounded down (160 B);
     **live** = up to 32 movers (241 B); each fits one 251 B BLE frame.
2. **BLE framing** (`ble.py:126-190`): company ID 0xFFFF, 16-bit msg_id, ≤ 15 fragments, CRC8; incomplete messages are
   dropped after 5 s.
3. **Map layer** (`maplayer.py`):
   - zlib(meta | L1 box covers | L2/L3 sparse deltas in 4-bit log-probability codes | UNKNOWN run lengths) = 29 kB raw,
     15.3 kB compressed, 64 chunks of 240 B + 8 XOR parity chunks;
   - a signed 417 B manifest (version, valid_s 6 h, payload SHA-256, a 4 B hash per chunk); chunks are accepted only if
     they match the table.
4. **Signing** (`sign.py`): Ed25519, key_id selects the verify key. Rejections: malformed, unknown key, bad signature, wrong site.
5. **Freshness:**
   - slow: max age 8 h; fast/live: 16-bit sequence window; live max age 1.5 s;
   - fast entries past TTL → UNKNOWN; the bridge drops a fast layer older than 30 s;
   - expired map → static keep-out of every hazard zone.
6. **Evaluation transport:**
   - a lossless local Unix socket on the shared sim clock: no latency, loss or skew;
   - only BLE-style frames go over it; the "Zenoh" whole-blob record is never sent;
   - the lossy `BleChannel` model runs only in tests.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| C-W1 | **No real channel. Zenoh and BLE are never exercised; workstream C is not implemented** (no Zenoh code, no `export/nav2.py`, no `INTERFACE.md`). "Two redundant channels" has no running code. | `broadcaster.py:126`, `feed.py:17`, STATUS | `ZenohTransport` (put/sub `intercycle/<site>/{map,fast,slow}`); a `BleSimTransport` feeding the bridge through `BleChannel` using the TF robot pose; a supervisor merging both; the drop-Wi-Fi / drop-both demo. Hardware next: nRF52840 + BlueZ periodic advertising. | 4-5 h (sim); ≈ 1 day (hardware) |
| C-W2 | **Map cache bug.** The cache is keyed by (site, version), not by content hash. A restarted observer that re-signs version 1 with different content is reported "cached": the robot keeps the old map with the new validity. | `maplayer.py:547-552, 650-658` | Key by sha; persistent monotonic version (or boot epoch + version); `latest()` uses the manifest whose sha matches; test. | 1 h |
| C-W3 | **Observer restart freezes fast/live.** seq resets to 1 and is rejected for up to 32,768 messages (≈ 4.5 h at 2 Hz): robots see UNKNOWN the whole time. Safe, but no service. | `ble.py:263, 322, 335` | Header v2 with a persisted boot_epoch; receiver state per key_id; accept a higher epoch. | 1.5 h |
| C-W4 | **No future-date or skew check; fast has no max age on accept** (TTL up to 300 s). A far-future slow layer or manifest blocks later genuine ones. | `ble.py:310-330`; `learn_cycles.py:470` | Reject issue − t_rx > 0.5 s; fast max_age ≈ 2 s; require chrony/NTP; skew monitor. | 0.5 h |
| C-W5 | **Key management is fail-open.** With no keyring parameter, the bridge silently trusts the PUBLIC demo key. No provisioning, rotation or revocation; key_id is a constant, not derived from the key. | `masks.py:52-54`; `observer_bridge.py:61-62` | `load_signing_key` (0600); `trust.py` with site-root-signed certs (key_id = sha256(pub)[:2], nbf/exp, revocation); the bridge refuses to start without a keyring unless dev=true. | 2.5 h |
| C-W6 | **The live layer (people/vehicles now) is never broadcast or used.** | `broadcast.py:196-230`; `masks.py:207-235` | live_fn from the tracker; movers → speed-scaled keep-out discs that expire after 1.5 s. | 2 h |
| C-W7 | **The map is broadcast once per episode and never updated**: the growing map never reaches robots. | `broadcaster.py:110` (`set_map` never called) | Re-pack with version++ on map change; test a mid-episode version switch. | 1 h |
| C-W8 | **The channel model is optimistic and inconsistent.** iid 10 % loss (no bursts, welding EMI or Wi-Fi coexistence); assembly computed at 20 fps vs 0.5 s advertising in the scenario; 251 B AdvData needs chained AUX PDUs; XOR parity repairs 1 chunk per 8. | `bodyshop.yaml:84-88`; `MAP.md:127` | Gilbert-Elliott burst channel; one consistent advertising rate; frames ≤ 229 B; Reed-Solomon/RaptorQ ≈ 25 %; hardware measurement. | 2 h |
| C-W9 | **Format limits are sized to this demo site** (live x ≤ 204.7 m, z ≤ 51.1 m; grid ≤ 255 cells per side; ≤ 64 machines; ≤ 31 boxes per bin). | `layout.py:12,29`; `maplayer.py:123-124,146` | Wire v2: 16-bit dimensions, site origin/scale, varints; keep the v1 decoder. | 2 h |
| C-W10 | **The robot guesses the production state** (≥ 50 % STOPPED → andon, else running; no fast layer → running), so break/shift slices are never selected although the map is keyed by them. | `masks.py:231-234` | Broadcast the observer's line-state cluster id in the fast layer's reserved bits; with no fast layer, take the max over all states. | 1 h |
| C-W11 | **Unverified buffering.** Chunks heard before a manifest are buffered unverified (5 map ids × 1024); decompression has no size cap. | `maplayer.py:572-576, 633` | LRU with a budget; cap decompression at raw_len. | 0.5 h |

### Needed for a real site
- Zenoh mTLS + ACL on `intercycle/**` and an OT VLAN.
- chrony to a site server.
- BLE broadcasters surveyed in a metal hall.
- Site root CA + TPM/HSM observer key; persisted epoch/version counters.
- Hardware fault tests (unplug, restart, replay, forge).

## D. Dashboard and stream

### How it works
- **Data path:** `StreamWriter` (`intercycle/observer/stream.py`) writes JSON snapshots (≈ 63 kB) of the in-process
  `LayeredMap`, plus events.
- **The page draws:**
  - L0 and hazard zones (site-plan import);
  - learned L1/L2/L3, UNKNOWN, period and phase-now;
  - events;
  - metric curves scored against simulator truth.
- **Clean:** the truth-period fallback is removed; SYNTHETIC / not safety-rated labels are present.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| D-W1 | **Every run the page shows is pre-audit:** newmachine (18:22, assumed shim, no state source) and week1 (18:25, calendar breaks, assumed shim); `week1` points at gitignored `out/`, so the published page can't load it. | `dashboard/runs.json`; STATUS:88 | Re-run `new_machine.py` + `map_timeline.py replay` on the measured shim; publish into `dashboard/data/`; rebuild the bundle; export guard refusing runs without `shim: measured` and a discovered state source. | 1 h + 45 min compute |
| D-W2 | **The page doesn't show what robots receive**: it reads the unsigned in-process map, not the quantised, box-covered, signed broadcast. | `export_static.py` | "Robot view": decode `pack_map` → `MapReceiver` → `DecodedMap`; show version, sha, age, verify status. | 2-3 h |
| D-W3 | **"Live" is a paced replay**; the websocket is on 127.0.0.1 with no auth. | `map_timeline.py:293` | Hook `StreamWriter` into the observer tick loop; TLS reverse proxy with read-only auth for a plant. | 2 h |
| D-W4 | **The phase strip comes from the stale pre-measured-noise model**; the Gemma strip is a fixture. | `phases.json` from `model_final_train.pkl`; `gemma_names.json fixture:true` | Re-export from `pm_measured.pkl`; run the gemma export once G's file exists, or drop the strip. | 0.5 h |
| D-W5 | **Metric labels are incomplete**; the curve uses one scoring record while milestones use the full set. | `map_timeline.py:283-287` | Label "vs simulator truth, day 7"; plot milestone values as dots. | 0.5 h |
| D-W6 | **Calendar data is still shipped** in `layout.json` (unused). | `export_static.py:51` | Drop it. | 0.1 h |

---

## G. Gemma 4 on AMD (semantic layer)

### How it works
- **Setup:** Gemma-4-31B-IT on Token Factory via `intercycle/llm/client.py` (json_object mode, record/replay cache,
  ≤ 15 req/min, 2 in flight); 159 live calls, 136 OK.
- **Frames:** every frame is from the PIL painter `scripts/gemma_frames.py` (a 9 m camera, 64°, 768×512, fake blur/noise/JPEG),
  not the Blender renders.
- **Privacy:** person boxes (+15 %) blacked out from SIMULATOR TRUTH (`gemma_keyframes.py:105-122`).
- **G1 commissioning:** one camera framed around each machine's true rectangle (`:71`); headline = no hatch, generic id,
  no hint; polygons on a 0-1000 grid scored by pixel IoU.
  - primary-zone IoU 0.30, hit@0.5 0.20;
  - all zones: recall@0.5 0.09, precision 0.10;
  - latency p50 62 s, p95 195 s.
- **G2 naming:** 3 keyframes per bin of P's discovered phases, closed vocabulary per family (family from the site plan);
  images still carry the hatch on the true zone and a `CAM-<machine id>` burn-in (`:198, :217`).
  - 0.753 of 89 bins (chance 0.27, per-family majority 0.51); with the family from G1: 0.73 of 63 bins;
  - return/transfer 0/5, approach 0/2, tip-dress 7/13; line states 2/4 vs the hand rule 4/4.
- **G3 triage:** M's 3 real change events → 2/3; oracle-framed crops → 20/24 (bump recall 0.5). Never frees space (asserted).
- **What consumes it:** nothing in observer/mapping/broadcast. Only dashboard captions (still a FIXTURE).
  `verify_state` has 0 results.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| G-W1 | **The privacy mask uses truth.** Production would mask with RF-DETR (recall 0.68), so ≈ 32 % of people would go out unmasked. The filter isn't enforced inside the client. | `gemma_keyframes.py:105-122` | Mask from the detector at a low threshold (0.1), dilated over track history; fail closed: send only machine-ROI crops and black out every motion blob outside them; the client refuses unfiltered images when mode=cloud. | 2 h |
| G-W2 | **Nothing consumes Gemma; `verify_state` is unmeasured.** The "semantic layer" changes nothing in the product. | `out/gemma_eval_verify.json` (6/6 cache misses) | Attach triage advice to `change_detected` events; run `gemma_eval_verify` on 160 frames (≈ 320 calls, 25 min); wire `suppress_window` (Gemma can only shorten or cancel a window, never open one) into the broadcast. | 3 h |
| G-W3 | **G1 geometry is poor, and its cameras don't exist in the plant** (each framed on one machine, not the real ceiling cameras); IoU penalises the prompt's own "draw larger" rule. | `tasks.py:157` | Geometry from pixels (A1-stretch periodicity zone discovery); Gemma only types and labels regions (type accuracy 1.0 on matched zones). Score on the Blender CCTV stills with GT masks (floor coverage + excess area). | 4 h |
| G-W4 | **G2 cannot separate phases that differ only in direction** (each keyframe from a different occurrence). G2 images still carry the true-zone hatch + machine id. | `:198, :217` | 3 frames 0.5 s apart from ONE occurrence + bin duration + position in the cycle; remove hatch/id; the G1-derived family as the headline. | 2 h, ≈ 90 calls |
| G-W5 | **G3 has n = 3 real events**; bumps are already detected by M. | `change.py:462` | ≥ 30 real events from M's detector over seeds, sites, parked-vehicle injections and bumps; an ECC-aligned difference image as a 4th input; M's bump score as context. | 4 h + compute |
| G-W6 | **Latency/limits:** p50 62 s, 20 RPM, provider OOM above 512 px; `VLM_MODE=local` behaves like cloud. | `client.py:429` | Serve Gemma on-prem (MI300X vLLM-ROCm, or E4B / 26B-A4B on an edge AMD GPU); measure p50/p95; implement local mode. | 3 h once hardware exists |

## V. Perception from pixels

### How it works
- RF-DETR-M (COCO), **person only** (`detector.py:122-124`, threshold 0.25), 800×1408 at 5 fps.
- ByteTrack per camera and class (lost after 3 s).
- Foot point → floor through the dataset's homography; no lens distortion.
- Greedy cross-camera merge at **0.8 m** (`occupancy.py:50`, `pipeline.py:73,379`).
- **Measured on SmartSpaces Warehouse_000** (photoreal SYNTHETIC warehouse, 19 cameras, 5 min):
  - precision/recall 0.89 / 0.68; MOTA 0.59; IDF1 0.34; 2,213 ID switches;
  - localisation median 0.40 m, mean 3.04 m, p90 9.7 m;
  - occupancy error 1.47 persons/zone-min; count r = 0.53.
- Only the fitted noise parameters (`MEASURED_SHIM`) reach the map; **no pixel output reaches it.**

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| V-W1 | **No pixels → map path.** | `mapping.py:57` | Adapter tracks → `Observation` per 0.5 s tick; run mapping on the A1 Blender clip and compare with the shim. | 5 h |
| V-W2 | **Domain gap**: a synthetic warehouse, not a body shop, not real. | `out/perception/*/metrics.json` | RF-DETR on the Blender body-shop CCTV sequence (COCO GT exists in `render/cctv/gt.py`); WILDTRACK (real, calibrated; research licence, evaluation only); re-fit the shim. | 4 h, ≈ $1 |
| V-W3 | **Heavy localisation tail** (truncated/occluded feet). | metrics.json | Reject boxes clipped at the frame edge; check the expected pixel height at the foot point; median across cameras. | 2 h |
| V-W4 | **Fusion radius 0.8 m in code**; 1.5 m was chosen in-sample. | `occupancy.py:50` | Tune on Warehouse_001, score on 000; set the default. | 1 h |
| V-W5 | **No world tracks** (IDF1 0.34), so flows can't come from pixels. | `tracking.py` | Kalman + Hungarian in metres over fused points; world-IDF1/HOTA. | 3 h |
| V-W6 | **Person only**: no vehicles, no machines. | `detector.py:122-124` | Fine-tune on Blender COCO data (forklift, AGV, tugger, robot_arm) checked on held-out renders, or OWLv2 (Apache-2.0) open-vocabulary. | 6 h |
| V-W7 | **No calibration tool, no distortion model** (`cctv_post` adds k1 −0.20..−0.35). | `geometry.py` | Click-floor-points UI + `findHomography` (RANSAC) + `undistortPoints` + GeoCalib for k1; validate on Blender renders with known cameras. | 3 h |
| V-W8 | **ROCm path untested**; one L4 ≈ 8 cameras. | `cloud/Dockerfile.rocm` | MIGraphX bench + sizing table. | 2 h once hardware exists |

## X. Renders and the pixels → phases demo (A1)

### How it works
- Blender 5.1 replays the trace; arms follow the simulator's kinematics (`weld_q`, sampled every 20 ms).
- **CCTV post:** barrel distortion, vignette, Poisson-Gaussian noise, x264 CRF 28-36, overlay, 0.35 px shake.
- **Use so far:** video only. No evaluation has used a render yet (G uses the painter, V never ran on them).
- **A1 inputs:** cctv_1, seed 3, 45 min at 2 fps; EEVEE 4 samples, **no shadows, no blur, no extras**; hand ROIs for
  S30/S40 projected through the known camera.
- **A1 features** (`intercycle/perception/pixel_phases.py`): ECC stabilisation, median-normalised luma, frame difference > 18,
  opening, blobs ≥ 12 px, flash skip → [motion, 4×4 grid, saturated] → the unchanged `PhaseModel`; fit 25 min, score 20 min.
- **A1 results: none yet** (PIXELS.md "pending"). Shim reference: S30 F1 0.975.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort |
|---|---|---|---|---|
| X-W1 | **Circular generator**: the frames are driven by the same kinematics the scorer uses as truth, with the same program every cycle and σ 1 % takt. A pixel result would still be on a trivially periodic machine. | `synth_trace.py --machines sim` | Real footage of a cyclic machine (user clip), and/or per-cycle pose/path variation + model mix in the render (with W-W2). | 3 h |
| X-W2 | **The nuisances are switched off**: no shadows, nobody crossing the ROI, near-zero shake. | RENDER.md §0 | A 10-min stress clip: shadows from the 8 nearest lights, 2 extras crossing, 2-5 px shake. | ≈ 1 h GPU |
| X-W3 | **Unfair baseline and lenient scoring**: the shim baseline is the optimistic default (σ 0.15 m, 5 % drop); bins named from test truth. | `scripts/pixel_phases.py:129` | Add a `MEASURED_SHIM` column; name bins on the training window; report K. | 0.5 h |
| X-W4 | **Motion alone can't tell "held" from "idle".** | `pixel_phases.py` | A learned static home-pose background + per-cell difference from it. | 2 h |
| X-W5 | **Hand ROIs.** | PIXELS.md | ±10 px sensitivity test; periodicity-discovered ROIs (A1-stretch). | 1-3 h |

---

## E. Evaluation: event simulation and the robot experiment design

### How it works
**Event simulation** (`intercycle/sim/run.py`, `sim/world.py`, `robot/policies.py`):
- The robot is a point on a topological site graph: edge time = length ÷ 1 m/s × a per-seed speed factor (±5 %).
  No people, vehicles or Nav2.
- An inspection inside a zone takes 7 s × a dwell factor.
- **Zones** are ACTIVE/IDLE from truth windows:
  - **trip** = zone active when crossed → 90 s group freeze;
  - **hold** = the next start falls while inside → freeze until exit;
  - the walkway at x = 52.5 trips during a skid transfer.
- **Mission:** 10 machines + 4 racks looped 07:00-13:00 (includes two breaks).
- **Policies:**
  - **A0** enters only in planned stops (public calendar) when its own camera sees no motion;
  - **A** enters whenever its own camera says "no motion";
  - **B** (InterCycle) = learned slow layer (seed 99) + nowcast from a per-zone ceiling signal (truth activity flag,
    2 Hz, 8 % occlusion, 3 % flips) + SIPP arrival timing + own-camera veto (5 Hz, 5 % miss, 2 % false motion,
    0.3-1 s latency);
  - **C** = true future windows (oracle, no camera, no margin).
- **Metrics:** inspections per production hour, trips, holds, line time lost, $ at $2.3M/h, gain retained = (A−B)/(A−C).

**Robot experiment** (`ros2/intercycle_eval/intercycle_eval/batch.py, metrics.py, stats.py`):
- 600 s episodes, seeds 1000-1009 × {none, static, learned, timing oracle}, paired; start 06:30 + U(0, 15 h) away from
  planned stops.
- need_s = enter 5.0 + dwell + exit 6.5 + margin ≈ 16.5 s, so only S60, S70, S80 and SEAL1 can ever fit in learned/oracle.
- **Primary:** line_time_lost_s, learned vs none, exact sign-flip (1024 flips).
- **Safety:** any_collision (Clopper-Pearson, Poisson/h).
- **Secondary:** 12 endpoints × 3 pairs, Holm.

### Weaknesses (not accepted, to fix)
| # | Weakness | Evidence | Real fix | Effort | Changes headline? |
|---|---|---|---|---|---|
| E-W1 | **No secondary test can ever be significant.** 35 tests; the minimum two-sided p at n = 10 is 2/1024 = 0.00195, but Holm's first threshold is 0.05/35 = 0.00143. | `stats.py:35-37` | Amendment 2 BEFORE the batch: a fixed-sequence gatekeeping chain of ≤ 4 hypotheses at α 0.05, the rest descriptive (or ≥ 13 seeds). | 0.5 h | What can be claimed |
| E-W2 | **The pre-registration contradicts itself** (§0 one primary; §4 3 endpoints × 2 pairs Holm-6; §3 still Identify + arm-base targets). | `docs/ROBOT_RESULTS.md` | Reconcile in the same Amendment 2 commit. | 0.5 h | – |
| E-W3 | **Amendment 1 is not implemented, and "static = 0" is an artefact.** Targets are still the nearest arm base at 1 m. At 70° HFOV, 1000 px/m needs range ≤ 1.37 m, so even the oracle may score 0. Real inspection PTZ heads have optical zoom. | `plant.py:143-159`; `camera.py:37-39` | Site-plan targets (gun tip, tip dresser, clamp, carriage underside) with guard/fixture occlusion meshes; the same zoom head in every condition, so a boundary capture fails only on line of sight; dev-check that the oracle passes each target. | 4 h + 1 dev seed (≈ $1.3, user OK) | **Yes** |
| E-W4 | **The primary is a strawman.** "none" enters after 0.3 s of camera-clear on every machine, including S10-S50 where idle (12 s / 6 s) < need (16.5 s), and the weld dwell fools its camera. Holds are near-certain, so p ≈ 0.002 is almost guaranteed. | `timing.py`; `camera.py:15-17` | A best-tuned reactive baseline (stillness threshold + idle length learned from its own camera, tuned on dev seeds 20-29). Primary = Inspect passes with zero line loss, learned vs best reactive. Line time vs static as non-inferiority (margin ≤ 5 s). | 3 h + ≈ $3.5 | **Yes** |
| E-W5 | **The comparison doesn't isolate the observer.** With 1 % jitter and fixed busy fractions, a robot watching 2 cycles learns idle exactly; no timing-vs-spatial ablation. | `world.py`; `yaml` | Event-sim policy **A-self** (own-camera period/idle fit persisted across visits, same `idle_windows`); a "timing only, no map masks" ablation on the CPU kinematic double. | 3-4 h | Likely shrinks B's edge |
| E-W6 | **Event-sim numbers are stale and the tests haven't been re-run** since 4660155. `test_safety.py:47-50` asserts 0 trips/holds on seeds 5-6 (can fail now B tripped on seed 10); the fault test compares totals with thin margins (long: 7 vs 8 trips). | `out/results_dev_0_19.json` (17:31) | Re-run; assert per seed; publish new numbers. | 1 h CPU | **Yes** |
| E-W7 | **"Gain retained" is degenerate** (A0 would score 96 %, static 100 %). | `run_seeds.py:46` | Replace with zero-loss inspections per production hour + a Pareto plot (inspections/h vs loss). | 0.5 h | Yes |
| E-W8 | **Line time lost ≠ lost production.** The event sim sums overlapping freezes (double counts); the 90 s recovery is optimistic; short holds are absorbed by takt slack/buffers; break inspections divided by production hours. | `run.py:299-308` | Union of freezes; jobs lost with a slack/buffer model; recovery sweep 30-600 s; split break vs production inspections. | 2 h | $ figures |
| E-W9 | **Production-only 10-minute windows** exclude break and andon restart transients, where the nowcast is weakest. | `batch.py:42-55` | Stratified starts at break end and andon end (free in the event sim; an Isaac stratum if budget allows). | 1 h | Possibly |
| E-W10 | **The event-sim abstraction is too clean:** no weld-dwell stillness for the own camera; ceiling signal not from measured perception. | `run.py:24-58` | Drive B's ceiling signal with A1's measured confusion; add weld-dwell stillness; per-edge travel jitter from Isaac logs. | 3 h | May raise B's trips |
| E-W11 | **The safety endpoint is uninformative**: 0/10 gives ≤ 0.31 per episode, ≤ 2.2/h. | `stats.py` | ≈ 200 CPU kinematic-double episodes for collision and near-miss (TTC) rates + the invisible-robot stress flag; Isaac for locomotion validity. | 2 h + CPU | Safety claim |
| E-W12 | **Conditions inspect different machines** (greedy choice). | `timing.py:142` | Per-target outcome table (declined / passed / held / tripped). | 1 h | Clarity |
| E-W13 | **need_s rests on unmeasured enter/exit times** (test-double 5.0/6.5 s), and decides which machines are feasible (HEM1's 15.75 s idle is just under 16.5 s). | `batch.py:74` | Measure on dev ANYmal logs and freeze. | 1 h | Feasible set |
| E-W14 | **Seed overlap**: event-sim held-out 1000-1199 contains the robot seeds. | `run_seeds.py` | Shift the event-sim range to 1010-1209 (or run the robot first). | 0 h | – |
| E-W15 | **Not measured:** mission value (no defects exist), people discomfort beyond distance, robot wear/energy, operator interventions. | – | Inject defects at targets with detectability = f(px/m, incidence) → defects found per hour; log energy/steps; count interventions. | 4-6 h | New headline |

### What a meaningful evaluation needs
1. One consistent pre-registration (gatekeeping; primary vs a best-tuned self-learning baseline; non-inferiority vs static).
2. Real Inspect targets with occlusion and a zoom head, checked on dev seeds.
3. Ablations that separate timing from spatial layers, and observer from self-learned timing.
4. Production units (jobs lost, interventions) with recovery-time sensitivity.
5. A large CPU batch for safety rates and transients; Isaac for realism.
6. Fresh event-sim numbers, then held-out once on disjoint seeds.
