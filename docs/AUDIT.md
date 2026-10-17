# InterCycle cheat/drift audit — fix list for the lead (monitor amd-ce, Sat 17 Oct ~18:30)

## USER DECISIONS (the user answered directly to amd-ce, 17 Oct ~18:35)
- Send all 3 tiers: Tier 1 before the robot map is frozen, Tier 2 before the batch and at the freeze, Tier 3 as disclosures.
- Isaac budget: allow up to $25 for the full 40-episode batch (≈ $22 total). No further Isaac time after that without asking.
- Rule-4 waivers APPROVED for these sim changes only: (a) shim set to measured perception error (≈30 % miss, ≈0.35 m σ + heavy
  tail, false positives on), one map rebuild; (b) noisy own-camera (occlusion + latency) for robot B in the event sim;
  (c) ±5 % per-seed robot speed/dwell jitter in the event sim; (d) break-timing bug fix (pin planned stops to wall clock).
- NOTE: `map_1w.npz` was just declared final for R3 — it uses the calendar (Tier 1 #1) and the unmeasured shim. Rebuild it with
  line_state + measured shim before the batch (one heavy job), or the batch's learned condition inherits both.

Seven read-only audits (M, P, R, event sim, G, perception/dashboard/claims, planned work).
No truth reaches Nav2/bridge/timing_planner; learning code imports no truth. Issues below.

## Tier 1 — invalidates a number (fix before it is published)
1. **Calendar is an input.** `mapping.py:62-69,649` `calendar_state()` reads `scn.line_stops()` (the same schedule truth uses);
   `map_timeline.py:149` snapshots always use it; R2 `masks.py:235` picks SpeedFilter state from the "public break calendar";
   `map_1w.npz` (robot batch map) built that way. Fix: `use_line_state=True` default in build_map/Recorder/snapshot_layers/
   new_machine and rebuild BEFORE the robot map is frozen; or label "external input" everywhere. Delete MAP.md "break labelled 100 %".
2. **P tuned on its held-out window.** Final eval 17:36, naming patched + ARI re-scored on `final_test` 17:44-17:49, re-run 17:54.
   Onset metric redefined after the held-out run (HEM1 90 s -> 0.5 s, S80 NaN -> 0.0, commit f86093c). Fix: re-score once on an
   untouched seed (seed 2, ~15 min), headline that; report both onset definitions.
3. **Event sim: B's 0 trips rests on a truth own-camera.** `sim/run.py:97-99` `sees_motion = world.active(mid, t-lat)` (the §5.1
   cheat), walkway `run.py:149-151`. Faults "phase"/"flip" are no-ops for B (seed 10, 1 h: identical 34.0 insp/h, 0 trips).
   `test_safety.py:33` only checks B<=A trips. Fix: `_Camera`-style noise on sees_motion (occlusion reads "no motion", 0.3-1 s
   latency), a fault inflating `next_idle_len_s`, assert on line time lost. ~2 h.
4. **Map unsafe-free margin chosen on the scored held-out; held-out = same seed (day 7 of seed 0, deterministic shuttles).**
   Vehicle Brier 0.0115 = climatology (no skill). Fix: margin on days 5-6 or ISO formula, score day 7; held-out on seeds 1-2;
   headline strict-overlap unsafe-free + Brier skill vs climatology, not AUROC. New machine: report 2.4 % post-convergence unsafe-free.
5. **G1 answer in the prompt and painted on the image** (`gemma_tasks.py:130` machines_in_view from truth zones; hatch band on the
   true rect `gemma_frames.py:363-379`; `CAM-S30` burn-in). G3 crops centred on truth (`gemma_keyframes.py:245-283`).
   Fix: hint=None as headline; G3 from M's real change events incl. false alarms, or label "oracle-framed"; G2 report chance (25 %)
   and majority baseline (50.6 %). Also: G1 calls currently fail HTTP 500 — no G result exists.

## Tier 2 — misleading claims / robot stats (cheap, before 01:30 batch where noted)
- Robot (before batch): declare primary endpoints (line_time_lost_s, inspections_ok, any_collision) + Holm + exact sign-flip
  permutation (`stats.py:20-26`); `safe_success` must use `inspections_ok` not `n_inspected` (`metrics.py:280`); drop non-Isaac
  backend rows (`batch.py:133` allow_fake); remove truth pose from `/intercycle/inspection/capture` (`ros_bridge_node.py:15,161`);
  AIM_SIGMA 2-3 deg (aims from true pose, `camera.py`); measure enter/exit_s on ANYmal logs (5.0/6.5 s are test-double values);
  label oracle "timing oracle"; 10-min episodes = "in production only", no per-shift extrapolation; N=10, 0 collisions -> CP upper 0.31.
- "Gain retained 99.7 %" alone is misleading (A0 would score ~96 %); always with B/C throughput 0.854 and B vs A insp/h (-21 %).
- Event sim baselines: give A/A0 the same local walkway rule as B; ±5 % speed/dwell jitter (A's S30 trips are phase-locked).
  Bug: breaks run on machine time after a trip (`world.py:63-70`). STATUS "Solid" still lists shim numbers; shim comparison crosses layouts.
- Line-state naming encodes the sim (andon rule = sim design; "break = gathering" because sim spawns BRK crowd); merge break +
  shift_change into planned_stop or disclose. S80 F1 0.99 is tautological (shim shows carriage only below H_CLEAR) — drop from headline.
  Scoring maps bins with test truth; K=8 for all; boundary error recall-only, capped at 30 s — disclose.
- "No manual setup" (BRIEF.md:7) false: machine rects, on_line, camera poses, hall extents come from the scenario. Reword:
  "no PLC link; one-time site-plan import (zones, camera poses)". Dashboard header "Production state (discovered)" (index.html:41).
- "Video never leaves the edge box" vs Gemma frames to cloud Token Factory: disclose on-prem path (vLLM-ROCm / edge model, blurred crops).
- Dashboard: published bundle is the 16:40 non-growing stand-in; runs.json points at gitignored `out/`. Export week1+newmachine into
  dashboard/. Remove `periodOf` fallback to true `period_nominal` (app.js:255).
- Fusion radius still 0.8 m (occupancy.py:50, pipeline.py:73,379) — DIRECTION says 1.5. "Photoreal replay" (render/blender/README.md:3).
- Gemma: `gemma_eval_verify.py` uses SDK max_retries=2 unpaced, uncapped concurrency — route through the limiter.

## Tier 3 — simulation shortcuts to DISCLOSE (gap to a real factory)
- No pixels -> map path exists. Detector finds people only. Every map/phase/new-machine result runs on CameraShim sampling truth:
  true class, radius, `moving` flag before noise, no FOV (nearest camera covers everything), no occlusion, no ID switches, FP off.
- Shim is optimistic vs our own measurement: σ 0.15 m / 5 % drop vs measured median 0.40 m (heavy tail, p90 9.7 m) / 32 % miss / 11 % FP.
  Either re-run with measured params (combine with the calendar rebuild — one heavy job) or add a sensitivity line with explicit numbers.
- Same robot program every cycle (kinematics seeded [7,mid,r], seed-independent), σ 1 % takt, tip-dress every cycle, no model mix.
- Camera calibration assumed known; real install needs intrinsics + floor homography per camera.
- Codec: no key provisioning/rotation; no future-date check on fast frames; observer restart rejected as stale_seq.
- Robot: arm links are floor-to-2 m columns (conservative); AMCL map = exact sim geometry; stop polygon 8 cm; IMU unused.

## Planned work (T/V/C)
- T: never pass the skid "latent tag" truth id as the DINOv2 appearance cluster (circular). Either cluster real rendered crops,
  or a noisy channel with a stated confusion matrix labelled "simulated perception". Add triggers.py to test_no_truth_import;
  test add-only. Start T after the 01:30 batch launch (laptop contention), hard stop 08:00; cut DINOv2/broadcast field first.
- R4 (sensor-model validation, due Sun 08:00) was silently pushed behind T/V/C — not user-approved. Put R4 before V tracker and C.
- V: uses SmartSpaces supplied calibration — disclose. C: frame as coverage audit of existing cameras -> UNKNOWN cells.
- Budget: R3 estimates ~$16 for the batch -> Isaac ≈ $22, past the $20 stop line. Needs the user's decision now.
- Rule-4 note: turning on shim stress terms (clutter/persistent), randomising robot speed/dwell, or any sim change needs user OK.
