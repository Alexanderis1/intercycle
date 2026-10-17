# InterCycle — status at clean stop (17 Oct 2026, ~13:50 CEST)

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
- **Isaac Sim 5.1 on Modal L40S**: segfault in the RTX renderer ("nvidia kernel module not loaded", driver 610 in Modal's sandbox). Gate was 17:00 → treat as **no-go**.
- **Blender on Modal**: `Failed to create OptiX denoiser` → fix by switching the denoiser to OIDN/CPU or off. Local Blender worked earlier (render agent).

## Where we drifted (honest)
The plan was: learner + signed broadcast + SIPP + A/B/C evaluation, with the event-based sim as the metrics engine and a 3D replay for the video. Since 11:30 the scope grew three times on request — body shop + BLE (fine), a layered dynamic map (done, strong), and a full collision-checked physics sim. The physics sim then absorbed ~1.5 h of traffic-deadlock debugging (lanes, yielding, local-planner minima, keep-out margins, corridor widths). It is close but not yet trustworthy for robot A, and it is slow.

## Recommended restart plan (decide before resuming)
1. **Freeze scope.** No new layers, entities or renderers.
2. **Headline numbers from what is solid:** event-sim A/B/C (20 dev seeds now, held-out 1000–1199 after freeze), learner + map accuracy, Gemma verify accuracy and latency on AMD, perception throughput if available.
3. **Physics sim = credibility demo, not the metrics engine:** fix robot A (re-run after the 3 m walkway change), run ~20 seeds × 4 robots × 2 h on Modal via `cloud/modal_sim.py` as a cross-check table ("collision-checked: 0 collisions, trips/holds agree with the event sim").
4. **Video:** Blender with denoiser fixed, or the Three.js replay (build log / dashboard) if Blender is not fixed in ≤1 h.
5. Day 2 as planned: AMD MI300X perception run (when access arrives), dashboard on Modal, held-out seeds, README, video, deck, submit by 15:30.

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

## Perception at stop (real video, NVIDIA PhysicalAI-SmartSpaces, CC-BY-4.0)
- Dataset is public (not gated). Ingested 19 cameras of `MTMC_Tracking_2026/train/Warehouse_000` into Modal Volume `intercycle-data`.
- RF-DETR Medium (Apache-2.0) at 800x1408 + ByteTrack, 5 min of video at 5 fps on Modal L4:
  detection precision 0.89, recall 0.68 (persons, IoU 0.5); MOTA 0.59; IDF1 0.34; zone occupancy error 1.47 persons/zone-minute
  (1.00 with 1.5 m multi-camera fusion radius; code default is still 0.8 m — change it).
- Throughput ~42 frames/s per L4 (4 cameras on one L4 at 41 fps → ~8 cameras real-time at 5 fps). Modal spend ≈ $1.5.
- Code complete: `intercycle/perception/`, `cloud/modal_perception.py`, `scripts/perception_*.py`, 21 tests pass.
- Not done: `docs/PERCEPTION.md`; ROCm path (`cloud/Dockerfile.rocm`, `cloud/run_rocm.sh`) untested — needs MI300X access.
