# InterCycle — status (press-shop twin, Option B; lead session intercycle-74 on the desktop)

## >>> Gate log, lead intercycle-74 (H+0 = Sat 17 Oct 22:48 CEST; FREEZE Sun 08:30 fixed) <<<
The monitor amd-ce was offline at 02:30, so gate hand-offs are written here.

**G0 (~02:30, green except the MANIFEST).** origin/main 65f2c26.
- User decisions (Sun ~00:45-01:40): M-W7 alpha 0.05 / delta 0.10; T-W6 onset gap disclosed; split Amendment 1 (A1+); double cpu=4, max_rtf 2, K=16; A-lite kept. At ~01:40 the user went to sleep and delegated decisions to the lead (budget caps unchanged).
- Split A1+ (prereg/split_a1.csv, AMENDMENTS.md): groups train 11+12, dev 6+12, test 4+12 (cam9+cam14); test 126 clips (cam9 114 = 18.2 min, cam14 12 = 1.2 min). Small-group rule applies to cam9. N = 55.6 train min, M = 7.2 cam14 train min. Test purge done; check_no_test_images clean.
- Probes: P1 96.8 min / 691 clips; P2 no timing information; P4 P9a + P9b LEARNED (cam9, 1.96 s / 1.69 s), cam14 presses ASSUMED; P5 no vehicles cam9, forklifts cam14 (RF-DETR truck>=0.5); P6 3 framings, scale interval cam14; P7 2 separate zones; P8 whole-box masking; H1 $0.026-0.028/run at cpu4 rtf2.
- Lead decisions: CAL points placed by an agent (U14 clause, disclosed); k1_v0/scale_v0 invalid, refit on train; forklift class cam14 only; S7a L1 adapter fails closed in unlearned bins.
- Open: no custodian session yet (bootstrap prompt given to the user), so no MANIFEST / test labels yet.
- Revised gates: G1 ~04:30, Movers ~05:45, PREREG v1 ~06:15, G2 ~07:00, Dry run 07:45, FREEZE 08:30.

# (older) InterCycle — status at clean stop (17 Oct 2026, ~13:50 CEST)

## >>> HAND-OFF for lead session 3 (written by lead session 2, Sat 17 Oct ~19:20 CEST) <<<
Read in this order: this section → `docs/DIRECTION.md` (still the plan of record) → `docs/AUDIT.md` + `docs/PRODUCTION_BRIEF.md`
(both adopted by the user, with owners and a cut order; see "Audit" / "Production-readiness brief" below) → `docs/BRIEF.md`.
Everything below is committed and pushed. The robot held-out batch has NOT been launched. Nothing of ours runs on Modal.

### User decisions since DIRECTION (all confirmed by the user directly)
- Scope added: the fused cross-camera tracker (V), the camera placement planner (C) and workstream **T** (trigger discovery, FULL; rule-4 waiver for its overlay + truth hook only).
- Audit Tier 1/2/3 (`docs/AUDIT.md`) and production brief A1–A9 (`docs/PRODUCTION_BRIEF.md`) adopted.
  **Cut order if hours run short:** A10/A1-stretch → V fused tracker → C placement planner → T DINOv2 + broadcast field → A8 reduced to 6 injections.
  A1, A2, A3, A4, A6, A9 always stay.
- Rule-4 waivers: (a) shim at the measured perception error + one map rebuild; (b) noisy own-camera in the event sim; (c) ±5 % robot speed/dwell jitter; (d) planned stops pinned to the wall clock.
- Robot batch: 10-min (600 s) episodes, held-out seeds 1000–1009 × 4 conditions, **Isaac budget up to $25 total for the full batch** (≈ $19–20 expected), ask before anything beyond.
  Primary inspection pass = **DORI Inspect (1000 px/m) with entry**; Identify-from-boundary is a secondary row (pre-registration Amendment 1, e06ac35).
- Only the first Token Factory key is used (the second key was removed from `.env`).
- Slot order for new agents: **R4 → V → T → C** (T after the batch launch, hard stop 08:00).

### State per workstream (what is final, what is left)
| WS | State | Left (in order) |
|---|---|---|
| Lead / event sim | Learned observer replaces ShimObserver (1671f9e). Audit #3 + waivers b/c/d done (4660155): noisy own camera, shared walkway rule, jitter, wall-clock breaks, a "long" fault, and the safety test asserts line time. Seed-10 smoke: B 1 trip (walkway), A 8 trips / 837 s, B-long 7 / 663 s. | **Run the full suite once (never done after 4660155) + `python -m intercycle.eval.run_seeds --seeds 0 19`** under the heavy lock; update numbers. Freeze docs A9 (safety / compliance / deployment wording in PRODUCTION_BRIEF §A9). Held-out seeds 1000–1199 once after the freeze. |
| M map | Calendar removed (line_state default), measured shim (M.MEASURED_SHIM), splits days 0–4 / cal 5–6 / test 7 + seeds 1–2. **Rebuild running at hand-off** → `out/timeline/map_1w.npz` (robot) + `map_5d`. Clean pre-audit 1 w numbers exist (stale now). New-machine run done (detect 5.8 s, 0 unsafe-free in transition, 2.4 % post-convergence). | See M's hand-off below: A2 conformal margin, held-out seeds 1–2, week1 replay (stream for D), MAP_RESULTS, A8 change-detection calibration. |
| P phases | Seed-2 one-shot re-score is the headline (welders F1 0.85, boundary 0.57 s; 947a878 frozen, 6a592c2). A7 done (neutral "State A.." + operator naming). | Re-score seed 2 under the MEASURED shim (~15 min, commands in P's report below) and headline it; state what the clusters separate under measured noise (likely stopped vs running). |
| R1/R2/R3 robot (final map: `out/timeline/map_1w.npz`, copied to `out/r3/maps/map_1w_37b6d78.npz`, sha256 4b01346dba3824da…; seed 0 days 0–6, measured shim, discovered line state, code 37b6d78; out/ is local, so rebuild with M's commands if it is lost) | Isaac + Nav2 + bridge + timing planner + plant + camera gate + broadcasters + detached batch + stats + pre-registration (954d030, 20df131, c5b5c68, e06ac35). Truth leak in the gate aim removed. | 1) implement the Inspect-level target points (fixed site-plan points + normals) and check the oracle can pass each; commit as an Amendment-1 follow-up. 2) dev seed 22, one episode per condition (~$1.3; R3's dev budget goes to ~$3.3, so **ask the user**). 3) relay M's new map. 4) launch detached (command below). 5) metrics → stats → ROBOT_RESULTS. |
| G Gemma (AMD) | G1 headline 10/10 cameras (centre-zone IoU 0.30, weak but honest), G2 0.75 vs chance 0.27 / majority 0.50 (family given), G3 real events 2/3, oracle-framed 20/24. Privacy filter + VLM_MODE (6dbc153, 051d4fc). | 7 hinted G1 calls, G2 re-ask of 17 bins with G1's family, score, then **write docs/GEMMA_RESULTS.md** with disclosures 1 (GT masks vs RF-DETR recall 0.68) and 2 (family source). Commands in G's report below. Provider is slow/unstable (60–230 s per commission call); quota is not the limit. |
| D dashboard | Wired to week1 + newmachine + P phases + Gemma names (fixture until G's file). Runs export into `dashboard/data/` (force-added; root .gitignore ignores data/). | After M's final week1: `dashboard/export_static.py publish --run week1`, re-publish newmachine, `export_static.py gemma`, record scene 2 (record_frames.mjs), rebuild the bundle + artifact; the lead publishes it as an artifact. Per-track export guard (A9 compliance). |
| X + A1 (desktop session `alexanderis-buzzing-planet`, separate, keeps running) | Real assets, guns, fixtures, CCTV look, extras, cctv_1 sequence, wall/film code, wording audit. A1: trace seed 3 10:00–10:45 with sim kinematics, OpenCV extractor + tests, shim baseline S30 F1 0.975. | A1 clip (EEVEE too slow, so Workbench) → extract → score → docs/PIXELS.md; cctv_4; wall; 3 cinematic shots (24 fps). Status lives at the top of docs/RENDER.md §0 and in docs/PIXELS.md. **Tell it the new lead session's name.** |
| V / T / C / R4 | Not started. | V: SmartSpaces shim calibration (handed to M already as MEASURED_SHIM), fusion radius 0.8 → 1.5 m, PERCEPTION.md, fused tracker (cut-order #2). T: brief in STATUS "Proposed/approved" (no latent-tag truth as the appearance cluster; triggers.py in test_no_truth_import). C: Zenoh + BLE demo, A5 broadcast hardening, placement planner (cut-order #3). R4: RunPod RTX validation + robot shots (cap $10), after batch episodes exist. |

### Robot batch launch (only when every go condition holds)
`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/batch.py --name heldout1 --seeds 1000-1009 --backend l4 --seconds 600 --map out/r3/maps/map_1w_37b6d78.npz --detach`
then `--collect` (same args) if the client died, then `metrics.py out/robot/heldout1` and `stats.py out/robot/heldout1`.

### Budget at hand-off
Modal general ≈ $1.5. Isaac ≈ $4.6 ($1.5 monitor + ≈ $0.8 R1 + ≈ $0.35 R2 + ≈ $2.0 R3) of $25. RunPod ≈ $0.18 of $10. Token Factory: $0.006 today (cap $1/day, resets 18:00 CEST).

### Laptop rules that worked
One heavy job at a time via `mkdir out/.heavy_lock` (owner file inside; every agent including M must take it). Agents commit only their own paths with
`git commit -- <paths>` and never push; the lead merges `origin/main` (desktop pushes) with `git pull --no-rebase`, runs a secret scan, and pushes.

### Folder clean-up for the next session (not done; nothing here is needed)
- `out/`: dbg2–5.py, dbg_phys.py, patch3–7.py, smoke_phys.py, prof.out (old physics-sim debugging; 811 MB in out/ overall).
- `docs/STATUS.md` has grown into a log: the next lead should rewrite it as a short current-state page and move this log to `docs/archive/STATUS_log_17oct.md`.
- `docs/GEMMA.md` vs `docs/GEMMA_RESULTS.md` (to write); `docs/archive/PLAN-v2.md` is stale.
- `intercycle/sim/physics.py`, `sim/movers.py` and `sim/observer_shim.py` are retired (kept only because the old physics sim imports them); mark them or move them under `intercycle/sim/legacy/`.
- `render/isaac/` is legacy (Isaac rendering moved to RunPod/R4).
- The root `.gitignore` ignores every `data/` dir; `dashboard/data` needs `git add -f` (or rename it to `dashboard/runs/`).
- `C:\AMD\_archive\` (outside the repo) holds the Isaac recipes; keep it.

### Resume commands from the agents' hand-offs (Git Bash, from `C:/AMD/intercycle`)
**G (Gemma):** checkpoints skip finished items, and `.env` is read automatically.
```
.venv/Scripts/python.exe scripts/gemma_tasks.py run g1 --retry-errors        # 7 hinted (variant B) G1 calls left, 1-4 min each
.venv/Scripts/python.exe scripts/gemma_tasks.py run g2 --family-source g1    # re-ask the 17 S80/HEM1 bins with G1's family
.venv/Scripts/python.exe scripts/gemma_tasks.py score                        # out/gemma/results_g.json + phase_names.json
INTERCYCLE_LLM_MODE=replay .venv/Scripts/python.exe scripts/gemma_tasks.py run g3   # offline replay check
```
Logs: `out/llm_calls.jsonl`, `out/llm_cache` (135 entries), `out/gemma/{usage_log,request_log,batches_g}.jsonl`. GEMMA_RESULTS.md must say "Gemma inference via AMD Token Factory"
and "provider retention per AMD terms", carry the A6 privacy sentence plus disclosures 1 and 2, and note that G1 sends the full masked camera view and light-curtain posts remain as a cue.

**P (phases, measured shim):**
```
.venv/Scripts/python.exe scripts/phase_discovery.py sim --name s2m_train --seed 2 --t0 4 --t1 10 --shim measured
.venv/Scripts/python.exe scripts/phase_discovery.py sim --name s2m_test --seed 2 --t0 13.5 --t1 20 --shim measured
.venv/Scripts/python.exe scripts/phase_discovery.py eval --train s2m_train --test s2m_test --engine stumpy
.venv/Scripts/python.exe scripts/phase_discovery.py table out/phases/results_s2m_train__s2m_test.json
```
Then headline it in PHASES.md §0, with the ideal-shim run alongside, and use `line_catalogue()` + a 2-state (stopped vs running) purity/ARI. M should key the map on `line_cluster`, not the names.

**R3 (robot):** the dev map copy is `out/r3/maps/dev_layered_map.npz` (old map, dev only). Dev check: `batch.py --name dev22 --seeds 22 --backend l4 --seconds 600 --map out/r3/maps/dev_layered_map.npz`.
Key files: `ros2/intercycle_eval/intercycle_eval/{plant,camera,social,broadcaster,batch,metrics,stats,phasecheck}.py`, `docs/ROBOT_RESULTS.md` (§0 pre-registration + Amendment 1).
ROBOT_RESULTS still needs:
- the results block from `tables.md`;
- every invalid or timed-out episode;
- impassive-walker contacts broken out;
- the 0-collision bounds (0.31 per episode, ~2.2/h);
- the phase-check table on the final map;
- RTF and cost;
- the final map's name and training days;
- the stand-mode disclosure.

**M (map), final state at hand-off.** Commits 85f2de1, a0aa595, 37b6d78, 78b2254. The heavy lock is released.
- Final files:
  - `out/timeline/map_1w.npz` (robot, days 0–6) and `map_5d.npz` (days 0–4, conformal calibration);
  - `out/timeline/phases/pm_measured.pkl`;
  - `out/timeline/heldout/` (cal days 5–6, test day 7 on seeds 0/1/2, `splits.json`).
- **Every number reported today is STALE:** it came from the old assumed shim and the calendar input; old outputs are kept in `out/timeline_shim015/`. The current `out/stream/week1` and `out/newmachine` are also old-shim.
- Under measured noise the discovered line state calls the 09:00 break "andon" (conservative). test_mapping.py was not re-run after 37b6d78; its fast suites pass.
- Remaining steps, each under the heavy lock:
  1. `scripts/map_timeline.py replay --run week1` (~25–30 min): curves, milestones, rewrites week1 for D.
  2. `scripts/map_timeline.py report` (~20–30 min): A2 conformal margin, α 0.01 headline (0.005 too), tables.md + conformal.json.
  3. `scripts/new_machine.py --hours 8` on the measured shim (~15 min); watch false alarms/h.
  4. `docs/MAP_RESULTS.md` (~45 min): strict-overlap unsafe-free with the conformal margin, its CI and α, false-occupied as the price, Brier skill vs climatology, the new machine's post-convergence figure, discovered bins rerun via `map_timeline.py phases --step build|heldout|score`.
  5. A8 (~2.5 h): the injection harness doesn't exist yet; disclose if short.
- Caveat: S80's per-bin IoU is weak (one bin has an empty envelope).

---------------------------------------------------------------------------------------------------

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

### Audit (monitor `amd-ce`, ~18:30): confirmed by the user directly (~19:20)
Full list in `docs/AUDIT.md`. The user confirmed: all 3 tiers (Tier 1 before the robot map freezes; Tier 2 before the batch and at the freeze; Tier 3 as disclosures). **Isaac budget up to $25 for the full 40-episode batch** (≈ $22 total; ask before anything beyond). Rule-4 waivers: (a) shim at the measured perception error + one map rebuild; (b) noisy own-camera for B in the event sim; (c) ±5 % per-seed robot speed/dwell jitter; (d) break-timing fix. Slot order: **R4, V, T, C** (T after the batch launch, hard stop 08:00, cut DINOv2/broadcast field first).
Owners: M (#1 calendar → line_state, measured-shim rebuild, #4 margin on days 5–6, held-out seeds 1–2, Brier skill); P (#2 re-score on untouched seed 2 + disclosures); lead (#3 event-sim own-camera noise, faults, jitter, break fix, A/A0 walkway rule, wording); G (#5 hint=None headline, real change events, baselines, HTTP 500); R3 (Tier 2 robot items, SpeedFilter state from the observer not the calendar, rebuilt map); D (dashboard export into dashboard/, no true-period fallback); X (wording).

### Production-readiness brief (monitor, ~19:00): confirmed by the user directly (~18:45 wall clock)
Full text: `docs/PRODUCTION_BRIEF.md`. Adopted A1–A9. Owners: A1 pixels→phases = desktop (V-A1, its CPU + the X clip, which renders before the cinematic shots); A2 conformal margin = M (replaces "margin on days 5–6"); A3 pre-registration + A4 restrict-only clamp = R3; A5 broadcast hardening = C (Sunday) or lead; A6 privacy filter = G; A7 state naming = P; A8 change-detection calibration = M after 00:30; A9 docs = lead at the freeze; the per-track export guard = D. **Cut order if hours run short:** A10/A1-stretch → V fused tracker → C placement planner → T DINOv2 + broadcast field → A8 reduced to 6 injections. A1, A2, A3, A4, A6, A9 always stay.

### Robot pass criterion: user decision (~19:20, before any held-out episode)
On dev seed 20 the static keep-out robot passed 10/10 inspections from the zone boundary at DORI Identify (250 px/m), which makes the static baseline trivially pass. **Primary pass = DORI Inspect (1000 px/m) on targets inside the guarding that need entry** (weld gun tips, tip dresser, clamps, the lifter carriage). Identify-from-boundary is kept as a secondary row. Amended in the ROBOT_RESULTS pre-registration with its own commit before the batch. Also fixed before the batch: the camera head aimed at the TRUE parts centroid (truth leak), now a fixed per-machine aim point.

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
