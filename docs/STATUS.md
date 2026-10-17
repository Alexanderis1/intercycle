# InterCycle — status at clean stop (17 Oct 2026, ~13:50 CEST)

## Lead session 2 (started 17 Oct 16:20 CEST): live log
**Sat 17:30 checkpoint: met at 16:40.**
- The 5 failing tests had stale expectations: S50/S60 at x 47.5/57.5 with 7 m zones, AGV dock `DOCK_DX` east, and the tugger loop's 2.8 m corner shorter than the 6.5 m train offset the test probes. Test-only fix. **Full suite: 195 passed, 1 skipped** (6d59ccf, pushed).
- Desktop = Remote Control session `alexanderis-buzzing-planet`: clone at `D:\intercycle` (C: has 8.8 GB free), push works, RTX 4060 Ti driver 610.88, Blender 5.1.0. **The Blender MCP is not installed there**, so X runs Blender headless (`blender -b --python`).
- Active agents (5): M, P, R1, D on the laptop, X on the desktop. M and D share the stream contract the lead fixed in their prompts (`observer/stream.py` `StreamWriter.snapshot/event/close`; D commits it first).
- Laptop coordination: at most one heavy local job at a time, via the atomic lock dir `out/.heavy_lock/` (owner file inside). Agents commit only their own paths and never push; the lead pushes after a secret scan.
- Lead item in progress: the learned observer replaces ShimObserver in the event sim (timebox Sun 00:00).

### Spend log
| When | Where | What | GPU / min | $ | Running total |
|---|---|---|---|---|---|
| 17 Oct (before 16:20) | Modal | perception + sim + earlier tests | L4 etc. | ≈ 1.5 | Modal general ≈ 1.5 |
| 17 Oct 16:30 | Modal | Isaac physics smoke (monitor) | L40S | ≈ 1.5 | Isaac ≈ 1.5 / cap 25 (ask at 20) |
| 17 Oct 16:00 | RunPod | Isaac RTX render smoke (monitor), pod terminated | RTX 4090 | ≈ 0.18 | RunPod ≈ 0.18 / cap 10 |
| 17 Oct ~17:15 | Modal | R1 image build (CPU builders) + bridge/Nav2 CPU test with a kinematic test double | CPU | < 0.06 | Isaac ≈ 1.51 |
| 17 Oct ~17:30 | Modal | R1 ANYmal-D `/cmd_vel` smoke, Isaac headless physics-only | L4, ~100 s | ≈ 0.04 | Isaac ≈ 1.55 |
| 17 Oct 17:00–17:40 | Modal | R1 gate + movers + 15-min patrol (RTF 0.67 under Nav2 load, 23.6 container-min ≈ $0.54) | L4 | ≈ 0.75 | Isaac ≈ 2.3 |
| 17 Oct 17:09–17:37 | Modal | R2 test-double runs (6 × 8 CPU × 4–8 min) | CPU | ≈ 0.15 | Isaac ≈ 2.45 |

**Robot batch decision (user, 17 Oct ~17:45):** **10-min episodes**, 10 held-out seeds (1000–1009) × 4 conditions = 40 episodes in parallel L4 containers, ≈ $0.35 each ≈ $14. R1 measured ≈ $0.55 per 15-min episode at full container rate, so 15-min episodes would have broken the $20 Isaac stop line. R1 finished (both gates early). R3 launched 17:45.

**P done (~18:05, Sun 00:00 timebox): ffda177, 7185ca6, f86093c.** Held-out 13:30–20:00 (seed 1, trained 04:00–10:00, after M's camera fix). Online (causal) decoder: welder segment F1 0.86, boundary error median 0.57 s (offline DTW 0.93). S80 0.99; HEM1 0.66; SEAL1 0.81. Periods within 0.4 % of truth. Line states, calendar-free: ARI 0.74, purity 0.97 (running 0.99, andon recall 0.89, break 0.94, shift_change 0.57). hmmlearn/ruptures have no py3.14 wheels and were replaced by numpy equivalents. One held-out seed, no multi-seed sweep. Not done: §4.3 discovered-bin L1 integration (M's step). R2 done (~17:55): bridge masks + timing planner; robot runs so far used the camera stub, so no results yet. G launched 18:00, D resumed for wiring 18:05.

**Lead item done (1671f9e): the learned observer replaces ShimObserver in the event sim.** Robot B's broadcast = models learned by `learn_cycles` on a separate training day (seed 99, 06:10–10:00) + `line_nowcast`/`nowcast` on the episode's own camera signal (past samples only). C uses the scenario's true models. Dev seeds 0–19 (6 h): B 39.9 insp/h, 0 trips, 0.5 holds/shift, 1.5 s lost; A 50.4 insp/h, 5.0 trips, 568 s lost; C 46.7; A0 7.6. Gain retained 99.7 %, B/C throughput 0.854. With the shim (old): B 35.5, 0.05 holds, gain 99.9 %. A/C/A0 also moved since the shim run because the layout changed (S50/S60, dock, routes) and C now uses true models; A0 now shows 0.25 trips/shift from the walkway crossing (not observer-related; to check). Fault-injection tests 10/10 pass.

**Sat 20:00 checkpoint (R1): met early, ~17:30.** Modal image `intercycle-isaac-robot`: Isaac Sim 5.1, IsaacLab 2.3, torch 2.7.0 pinned, Humble + Nav2. Pretrained FLAT ANYmal-D policy. `/cmd_vel` 0.6 m/s goes through a Unix-socket lockstep bridge (Isaac py3.11 ↔ rclpy py3.10, `/clock` from Isaac). Result: 11.34 m in 20 s sim, heading drift 0.007 rad, RTF 0.78 on an L4. The CPU test double reached a 16 m Nav2 goal (SmacPlanner2D + MPPI Omni + AMCL Omni). Next: the 22:30 gate in the factory scene.

### Proposed (needs the user's OK)
- **APPROVED by the user directly (17 Oct ~17:40); scheduled after R3 and G are running.** From the monitor session `amd-ce`:
  1. Workstream V: a cross-camera **fused** tracker on SmartSpaces (floor point per box → per-frame cross-camera clustering → one world-frame tracker → offline tracklet stitching). L2/L3 = occupied / observed frames, unseen = UNKNOWN. HOTA + IDF1 before/after. No BEV training. ~6–8 h, Modal L4 (inside the $4 perception cap).
  2. Workstream C: a camera **placement planner** on the plant layout (≥2 views everywhere, ≥3 near machines, 40–90° ray angle, occlusion). Coverage heatmap, camera-count curve, recommended layout. CPU only, ~2–3 h.
  - Positioning notes for README/MAP_RESULTS: closest prior art Robinson et al. arXiv 2606.06762; keep "advisory"; licence traps (DUSt3R family non-commercial, etc.).
- **APPROVED by the user directly (17 Oct ~19:10): workstream T, trigger discovery, FULL version** (relayed by `amd-ce`). Unsupervised rules "event e → machine m within [τlo, τhi], P = X % (CI), support n". Event vocabulary comes from the observer's own outputs (phase transitions, line states, mover zone events, appearance clusters of tracked objects via DINOv2 Apache-2.0). Numpy Hawkes/lagged model with the machine's own-cycle baseline, surrogate null + Benjamini-Hochberg FDR. Validated on existing links (andon cascade, AGV dock every 6th cycle) plus an injected hidden cause via `scenario/bodyshop_triggers.yaml` + a minimal truth hook (**user waiver of rule 4 for this hook only**). Map use is add-only ("expected ACTIVE"), never frees space; the broadcast field only if time allows. Gemma may name a cause, never define it. No tigramite (GPL). ~6–8 h CPU, starts when D's wiring slot frees (~20:00).
- D: at freeze, copy the final stream runs into `dashboard/data/` so GitHub Pages works (accepted by the lead: inside D's static-hosting scope).


> **17 Oct 14:40:** the plan is now `docs/DIRECTION.md`. The restart plan that stood here was removed because it conflicted with the user's direction. The facts below (what's solid, failed, test status, resume commands) are still valid.

Deadline: 18 Oct 16:00 CEST (code freeze 12:30). Repo: `C:\AMD\intercycle` (git, MIT, not yet pushed).
Secrets: Token Factory key in `.env` (gitignored, never commit). Modal token in `%USERPROFILE%\.modal.toml`.
Nothing is running on Modal. All subagents were told to stop.

## Solid (committed, tested)
| Piece | Evidence |
|---|---|
| Scenario + ground truth (BIW line on a 60 s takt, cells, breaks, andon, data hall) | `scenario/bodyshop.yaml`, `intercycle/truth/` |
| Codec: slow / fast / live layers, Ed25519, BLE 5 framing, receiver (TTL rounds down) | 52 codec/BLE tests |
| Map layer codec (signed manifest + chunks) | 15.3 kB → 64 chunks, assembly 6.6 s mean at 10% loss |
| Cycle + traffic learner | period err 0.14 %, busy err 0.10 pt, nowcast ±1 s at 30 s: 74 % |
| Layered map of dynamics (L1 machine envelopes by phase, L2 vehicles, L3 people, by time of day × production state) | envelope coverage 99.4–100 %, 5–9 % false area; phase bin ±1: 96.5–99.9 % |
| Event-based sim + SIPP + policies A0/A/B/C + fault injection | 20 dev seeds: B 35.5 insp/h, 0 s line loss; A 48.2 insp/h, 397 s lost; C 42.0; A0 7.6. Gain retained 99.9 %. Sweep shows where it breaks (≥2 % jitter) |
| Collision geometry kernel (static map, footprints, lidar) | 25 geometry tests, map PNG `out/static_map.png` |
| Machine kinematics (arms incl. tip-dress, skids, lifter, AGV docks) | 20 tests |
| Gemma 4 on AMD Token Factory | probe OK: text, vision verify_state, JSON, tool calling (3.5–4.6 s) |
| Explainer (body shop + BLE) | https://claude.ai/artifact/MMWmiXFfJDQcTzm6WT2FGv |
| Brief for reviewers | https://claude.ai/artifact/AwKVafoEiN4SMFRXR6m1mo |
| Build log | https://claude.ai/artifact/R9hCCqNH82h3kEqLZJ8wQ9 (still shows the event-sim results) |

## Work in progress (committed as WIP at stop)
- **10 Hz collision-checked physics sim** (`sim/physics.py`, `sim/movers.py`): every entity has a body, robot senses only via scanner/camera/BLE, people and vehicles yield, DWB-style local planner, keep-out zones, recovery.
  - Robot B over 30 sim-min: 10 machine + 6 rack inspections, 0 trips, 0 holds, 0 collisions.
  - Robot A still stalls (last cause: 1 m walkway gap between S50/S60 — widened to 3 m in the YAML just before stop, **not re-run**).
  - Speed: ~100 s wall per 30 sim-min locally → a 6 h shift ≈ 20 min. `cloud/modal_sim.py` fans out on Modal CPUs (written, **not run**).
- Perception (RF-DETR + ByteTrack on SmartSpaces, Modal/ROCm): see agent report below.
- Gemma evaluation on sim-rendered frames: see agent report below.

## Failed / blocked
- **Isaac on Modal L40S (re-tested 17 Oct 16:30 by the monitor):** the RTX renderer segfaults even with the known-good precedent recipe (Modal driver 610). **Physics-only works:** pretrained ANYmal-D walked 7.36 m in 10 s at RTF 1.11. See DIRECTION §5.2. Monitor test spend ≈ $1.5.
- **Isaac rendering on RunPod RTX 4090 works** (17 Oct 16:00, monitor test, pod terminated, ≈ $0.18): ANYmal-D RTX camera image rendered. See DIRECTION §5.2 / §8.
- **Blender on Modal**: `Failed to create OptiX denoiser` → fix by switching the denoiser to OIDN/CPU or off. Local Blender worked earlier (render agent).

## Where we drifted (honest)
The plan was: learner + signed broadcast + SIPP + A/B/C evaluation, with the event-based sim as the metrics engine and a 3D replay for the video. Since 11:30 the scope grew three times on request — body shop + BLE (fine), a layered dynamic map (done, strong), and a full collision-checked physics sim. The physics sim then absorbed ~1.5 h of traffic-deadlock debugging (lanes, yielding, local-planner minima, keep-out margins, corridor widths). It is close but not yet trustworthy for robot A, and it is slow.

## Plan
See `docs/DIRECTION.md` (single source of truth). Modal spend so far: about $1.5 (17 Oct).

## Test status at stop: 190 passed, 1 skipped, 5 failed
All 5 failures come from geometry tweaks made in the last hour (not logic regressions); update the tests' expectations or revert the tweak:
- `test_kinematics.py::test_tip_dress_reaches_boundary_weld_does_not`, `::test_skids_stay_on_conveyor_and_move_one_pitch`,
  `::test_agv_dock_every_sixth_cycle_for_20_s` — S50/S60 narrowed to 7 m (3 m walkway gap) and AGV dock moved +2.6 m in x.
- `test_geometry.py::test_mover_routes_are_clear`, `::test_forklift_corner_at_west_wall` — tugger route changed to a one-way loop
  and FL0 corner moved to x=3; the tests still encode the old routes.

## Gemma evaluation at stop
- 0 Token Factory requests made by the eval (only the 4 probe calls in `out/llm_calls.jsonl`). No accuracy numbers yet.
- `scripts/gemma_frames.py` complete and tested (36 of 160 frames rendered before the stop; `labels.json` is written only at the end, so re-run it).
- `scripts/gemma_run_eval.py` written, never run. Not started: `scripts/gemma_ops_demo.py`, `docs/GEMMA_RESULTS.md`.
- Resume: `python scripts/gemma_frames.py --n 160 --composites 8`, then `set -a; . ./.env; set +a; INTERCYCLE_LLM_MODE=record python scripts/gemma_run_eval.py` (≤15 req/min).

## Perception at stop (NVIDIA PhysicalAI-SmartSpaces, CC-BY-4.0: photorealistic SYNTHETIC video rendered in Omniverse, not real footage)
- Dataset is public (not gated). Ingested 19 cameras of `MTMC_Tracking_2026/train/Warehouse_000` into Modal Volume `intercycle-data`.
- RF-DETR Medium (Apache-2.0) at 800x1408 + ByteTrack, 5 min of video at 5 fps on Modal L4:
  detection precision 0.89, recall 0.68 (persons, IoU 0.5); MOTA 0.59; IDF1 0.34; zone occupancy error 1.47 persons/zone-minute
  (1.00 with 1.5 m multi-camera fusion radius; code default is still 0.8 m — change it).
- Throughput ~42 frames/s per L4 (4 cameras on one L4 at 41 fps → ~8 cameras real-time at 5 fps). Modal spend ≈ $1.5.
- Code complete: `intercycle/perception/`, `cloud/modal_perception.py`, `scripts/perception_*.py`, 21 tests pass.
- Not done: `docs/PERCEPTION.md`; ROCm path (`cloud/Dockerfile.rocm`, `cloud/run_rocm.sh`) untested — needs MI300X access.
