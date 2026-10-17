# InterCycle — DIRECTION (from the user, 17 Oct 2026 ~14:30 CEST)

This document is the single source of truth until submission. It overrides the restart plan in `docs/STATUS.md` and
everything in `docs/archive/`. `docs/BRIEF.md` holds the engineering facts (layout, conventions, safety semantics) and
has been aligned with this document.

---------------------------------------------------------------------------------------------------------------------------

## 0. Anti-drift rules (read before every task)

1. **Only work on items in this document.** If something seems necessary but isn't listed, write it under "Proposed" in
   `docs/STATUS.md` and **ask the user before doing it**.
2. **Every item has a timebox.** When it runs out:
   - commit what works;
   - write down what doesn't;
   - take the item's listed fallback;
   - move on.
   Never "one more fix" past the timebox.
3. **Debugging cap: 30 minutes on one bug.** After that, take the fallback or ask the user. (Today, 1.5 h went into robot
   traffic deadlocks. That must not repeat.)
4. **No new entities, renderers or simulator features** unless an item below names them.
5. **Commit at the end of every item.** Keep tests green, or mark failures xfail with a reason.
6. **Status line to the user at each checkpoint (section 9):**
   - done;
   - numbers;
   - next;
   - blocked.
7. **The plan must be complete without the MI300X.** It may arrive later, tomorrow or never. If it arrives, item G5 applies. Nothing waits for it.
8. **Secrets:**
   - The Token Factory key lives only in `.env` (gitignored).
   - Never write it in docs, code, logs, commits, artifacts or agent prompts.
   - Agents read it from the environment.

---------------------------------------------------------------------------------------------------------------------------

## 1. What we are building (one paragraph — judge every task against it)

**InterCycle's product is the OBSERVER.** It uses a plant's existing ceiling cameras. With no PLC link and no manual setup, it
learns how the whole plant moves:
- every machine's cycle and the distinct phases inside it;
- the production states of the line;
- the flows of vehicles and people by time of day.

From that it builds a **precise layered collision map** that any robot can use. The map gets sharper over hours, days and
weeks, and it adapts when new machinery is installed. The visiting inspection robot is the **consumer that proves the map is useful**:
it times inspections between machine cycles without touching line time.

Positioning: an **advisory planning layer, not safety-rated**.
- The robot keeps its certified scanner.
- UNKNOWN or stale space is always treated as occupied.
- Video never leaves the edge box, and people are anonymous tracks.
- Everything simulated is labelled SYNTHETIC.

**What is NOT the product:**
- a better robot local planner;
- traffic simulation fidelity;
- the A/B/C robot table as the headline (it is supporting evidence).

---------------------------------------------------------------------------------------------------------------------------

## 2. Workstreams, owners, file ownership

Each workstream has one owner (the lead or one subagent). Agents touch only their files and coordinate shared types through the lead.

| ID | Workstream | Owner | Files it may create/modify |
|---|---|---|---|
| M | Map quality, convergence, new machine | agent | **`intercycle/observer/mapping.py`, `intercycle/codec/maplayer.py`, `scripts/build_map.py`**, `intercycle/eval/map_metrics.py`, `intercycle/observer/change.py`, `scripts/map_timeline.py`, `scripts/new_machine.py`, scenario variant `scenario/bodyshop_newmachine.yaml`, truth hook for the new machine, `docs/MAP.md`, `docs/MAP_RESULTS.md`, tests |
| P | Unsupervised phase discovery | agent | `intercycle/observer/phases.py` (new; may call but not rewrite `learn_cycles.py`), `scripts/phase_discovery.py`, `docs/PHASES.md`, tests. Phases reach the map only through a function M calls (`phases.phase_bin(machine_id, t) -> int`). P does not edit `mapping.py`. |
| R | Legged robot on Nav2, cheat-free (§5) | lead + agents R1–R3 | see §5.7 |
| G | Gemma 4 semantic layer | agent | `intercycle/llm/*`, `scripts/gemma_*.py`, `docs/GEMMA_RESULTS.md` |
| C | Two comm channels + robot interface | agent (Sunday) | `intercycle/comms/*`, `intercycle/export/nav2.py`, `docs/INTERFACE.md` |
| V | Pixel path + perception realism | agent | `intercycle/perception/*`, `scripts/perception_*.py`, `docs/PERCEPTION.md` |
| X | Visual realism (renders + video) | agent | `render/*`, `docs/RENDER.md` |
| D | Live map view (map growing) | agent | `dashboard/*`, `intercycle/observer/stream.py` (snapshot writer + live feed; M calls it, D owns it), `docs/DASHBOARD.md` |
| L | Lead: tests, integration, docs, build log | **lead** | BRIEF, README, STATUS, build-log artifact |

---------------------------------------------------------------------------------------------------------------------------

## 3. Workstream M — map quality, learning over time, new machine (P0, the core)

**Goal:** prove the map is precise, show it learning, and show it adapting safely.

**Build:**
1. `map_metrics.py`: compare the learned layers with ground truth on **held-out time** (data the map was not built from).
   - L1 per machine and per phase bin: IoU, precision, recall of the predicted envelope against the true swept area.
   - L2/L3 per time-of-day slice and production state: Brier score and AUROC of occupancy.
   - **Unsafe-free rate**: share of truly occupied cells predicted free. Target ≈ 0.
   - Report the cost of safety too: false-occupied area.
2. Close or explain the current up to 0.6 % missed coverage. Add an inflation margin S = K·T + C (ISO 13855 style) and show
   the trade-off curve between unsafe-free and false-occupied.
3. `map_timeline.py`: build the map from 1 h, 6 h, 1 day, 3 days and 1 week of observation. For each, save snapshots (PNG)
   and the metrics. Output: convergence curves. Write snapshots through `observer/stream.py` (D's data contract) so the
   live view can replay the run. `--live` also pushes them to the websocket.
4. **Add an explicit UNKNOWN cell state** to the map and to the `maplayer.py` broadcast. It doesn't exist today: cells are
   only probabilities, and heading uses -1. UNKNOWN covers never-observed cells, cells under an offline camera, and new-machine cells.
   The robot side treats UNKNOWN as occupied.
5. **New machine** (`new_machine.py`): on day N a new moving machine is installed. Only the truth knows; the observer is not told.
   - `observer/change.py` detects a persistent motion blob in space L0 says is static (CUSUM or Page-Hinkley on per-cell residuals).
   - Its cells become **UNKNOWN (= occupied)** until K stable cycles have been seen and the period has converged. Then an L1 model is spawned for it.
   - Add a camera-bump check (a shifted camera must not look like a new machine).
   - Report: detection delay, false alarms per hour, time to re-converge, unsafe-free rate during the transition (target 0).
6. `docs/MAP_RESULTS.md` with the tables and figures. These become the **headline numbers**.

**Compute:** 1 week ≈ 20 min locally (3 days took 8 min). Use Modal CPU fan-out only if local is too slow (budget in section 8).

**Done when:**
- convergence curves exist for 1 h → 1 week;
- the new-machine run shows its detection delay and an unsafe-free rate of about 0 during the transition;
- the tests pass.

**Do NOT:** tune the simulator or movers; change the robot.

**Timebox:**
- metrics + timeline by Sat 19:30;
- new machine by Sat 22:30.

**Fallback:** if change detection isn't reliable, use the simplest version (a persistent motion blob outside known envelopes
for more than X minutes → UNKNOWN) and report it honestly.

---------------------------------------------------------------------------------------------------------------------------

## 4. Workstream P — unsupervised phase discovery (P0)

**Goal:** the observer discovers phases itself, at three levels.

**Build:**
1. **Per machine:** discover the distinct sub-phases of its cycle (weld / tip-dress / transfer / wait …), not 16 equal time slices. Pipeline:
   - motion-energy signal per machine region (from observations, not truth);
   - period (ACF/FFT, already in `learn_cycles.py`);
   - cycle motif (matrix profile, `stumpy`, BSD-3);
   - DTW alignment;
   - 4–8 state HMM (`hmmlearn`) or change points (`ruptures`), with an explicit idle/blocked state.
   Score against the true named sub-phases from `Kinematics.phase(mid, t) -> (name, frac)` in `truth/kinematics.py`
   (weld / dress boundaries at `T_WELD_END`, `T_DRESS_END`), as segment F1 and boundary error in seconds.
   Truth is imported only in the scoring script, never in `phases.py` (`tests/test_no_truth_import.py` must cover it).
2. **Whole line:** discover production states (running / andon / break / shift change) by clustering the vector of machine
   phases plus flow features. **The calendar may only be used to validate, never as an input.** Score: purity / ARI against the true states.
3. Feed the discovered phases into L1 binning as an option, and compare map metrics between discovered and equal-slice bins (with workstream M).

**Done when:**
- per-machine sub-phases are scored against truth for all machines;
- line states are scored against truth;
- results are in `docs/PHASES.md`.

**Do NOT:** use truth or the calendar as inputs; build worker routines (they are P2).

**Timebox:** Sat 22:00.

**Fallback:** if the HMM is unstable, use change points only. If line clustering fails, report it, with the calendar clearly
marked "external input".

---------------------------------------------------------------------------------------------------------------------------

## 5. Workstream R — legged inspection robot, realistic and cheat-free (P0; user requirement 17 Oct 15:30)

**Goal:** a scientifically defensible robot result. A **legged** inspection robot (ANYmal/Spot class: what industrial
inspection actually uses) walks with a real locomotion controller. It navigates with the **standard ROS 2 Nav2 stack**
using only its own sensors plus the observer's broadcast map, interacts realistically with people, vehicles and machines,
and performs a real inspection capture. **No hand-written planners, no ground truth in the robot loop.**

### 5.1 Cheats found in the current code (all must be gone from every robot result)
| Cheat | Where | Replacement |
|---|---|---|
| Robot B's broadcast comes from `ShimObserver` (reads past truth, "slightly wrong model"), not the learned observer | `sim/observer_shim.py`, used by `sim/run.py` and `sim/physics.py` | The broadcast comes from the **learned** observer (`learn_cycles` + `mapping`) trained on separate days |
| Perfect robot pose (no localisation) | `sim/physics.py` | Leg/IMU odometry with drift + AMCL (Omni model) on the static map |
| "Scanner" = exact geometric clearance to true bodies | `scanner_gap()` | A ray-cast lidar with noise, range limits and occlusion |
| "Own camera" = true machine state with latency | `sees_motion()` | A camera sensor model: moving parts must be inside the frustum and unoccluded to be seen, with detection noise and latency (Branch I). Frame differencing on rendered images where rendering exists (Branch M) |
| People step aside on command | `movers.ask_yield()` | Social force with the robot as an agent; about 20 % distracted/impassive people who don't react |
| Hand-written DWB-style local planner, graph SIPP | `physics.py`, `robot/sitegraph.py` | Nav2: SmacPlanner2D + MPPI (Omni) + collision monitor |
| Base driven by a diff-drive kinematic model | `physics.py` | Real legged locomotion (see 5.2) |

The custom 10 Hz physics sim is **retired for robot results**. The event sim stays only as a clearly labelled
*scheduling-level* analysis (line time lost), with B fed by the **learned** observer.

### 5.2 Simulator + locomotion: gate decided by the Isaac smoke test (17 Oct ~16:00)
- **Branch I — PRIMARY (verified by the monitor session, 17 Oct 16:30, on Modal L40S):**
  - **What works:** Isaac Lab 2.3 / Isaac Sim 5.1 **physics-only headless** (no `--enable_cameras`). The pretrained
    ANYmal-D rough policy downloaded (`get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Rough-Anymal-D-v0")`).
    Commanded 0.8 m/s, it walked **7.36 m in 10 s, RTF 1.11** with 1 env.
    - Sensors available without rendering: `RayCaster`, `RayCasterCamera`, `LidarPatternCfg`, contact sensors.
  - **What does NOT work on Modal:** anything using the RTX renderer (cameras, RTX lidar, Isaac rendering).
    It segfaults ("nvidia kernel module not loaded", Modal driver 610), **even with the precedent's exact recipe**.
    Do not spend time on it. **Visuals stay in Blender (workstream X)**, replaying logged ANYmal poses.
  - **Known-good recipe** (reference only): `C:\Users\anzel\AppData\Local\Temp\claude\C--AMD\76cf8262-e846-4133-a3c9-8a2e7bb4133f\scratchpad\isaac_smoke\main_policy.py`
    and `smoke_policy.py`.
    - Image: `nvidia/cuda:12.8.1-devel-ubuntu22.04` + py3.11, torch 2.7.0 cu128, `isaacsim[all,extscache]==5.1.0`,
      IsaacLab v2.3.0 editable, then `pip install -e isaaclab_rl 'rsl-rl-lib==3.0.1' 'torch==2.7.0' 'torchvision==0.22.0'`.
    - **Pin torch.** An unpinned `tensordict` upgrades torch and breaks the stack.
    - The precedent repo has **no licence**: write our own version, don't copy it into the MIT repo.
    - Never add the custom Vulkan ICD / `NVIDIA_DRIVER_CAPABILITIES` overrides.
  - **Robot:** ANYmal-D with the pretrained **flat** policy on the factory floor (verify its checkpoint exists; otherwise
    rough). The policy maps velocity commands to joint targets.
    - **Disclose:** the policy observes the true base velocity, standing in for an onboard estimator. Nav2 never sees it.
  - **Factory in Isaac:** L0 static geometry as one static mesh (walls, conveyor, racks, machine bases).
    People, forklifts, AGVs and arm links are **kinematic rigid bodies** driven each step from the mover/kinematics models.
    The robot **physically collides** with them (contact sensor logs collisions).
  - **Sensors (no RTX), stated exactly in the README:**
    - **Lidar:** Isaac Lab 2.3's `RayCaster` hits only **static** meshes. So `/scan` = Warp ray-cast against the static mesh
      **merged** with analytic ray–primitive intersection against the moving bodies' capsules and boxes. Same rays,
      noise, range and occlusion. This is physically equivalent for a 2D lidar.
    - **Robot camera** (no-motion check + inspection): a geometric camera model (frustum + occlusion ray-cast +
      pixel-density / incidence computation + detection noise and latency).
    - RGB evidence images are rendered **offline** in Blender from the logged camera pose and labelled as such.
    - IMU from body acceleration/rates + noise.
    - `/odom` = leg/IMU odometry, modelled as integrated body velocity with a bias random walk. **Label it.**
  - **ROS 2 / Nav2:**
    - ROS 2 Humble installed by apt in the same Ubuntu 22.04 image. Nav2 runs as separate processes with `use_sim_time`;
      Isaac publishes `/clock` and steps in lockstep.
    - Bridge Isaac (py3.11) ↔ ROS 2 (py3.10) with a small local socket/zenoh bridge, or Isaac's bundled ROS 2 bridge
      libraries if they load headless. Pick whichever works first, timeboxed to 1 h.
    - CPU: give the container 8 cores.
    - Optional throughput: N envs in one Isaac process, each with its own namespaced Nav2.
  - **Cost:** about $0.45 per 15-min episode on an L40S (RTF ≈ 1.1). Separate Isaac cap in section 8.
  - **RENDERING on RunPod (verified by the monitor session, 17 Oct 16:00).** User decision: physics-only on Modal (credits),
    rendering on RunPod (the user's money, so keep it lean).
    - The test: RunPod Secure **RTX 4090** (`NVIDIA GeForce RTX 4090`, $0.89/h; Community $0.34/h when in stock), driver
      580.159, image `nvidia/cuda:12.8.1-devel-ubuntu22.04` + apt python3.11 venv + the same pins as Modal, env
      `NVIDIA_DRIVER_CAPABILITIES=all`.
    - Result: Isaac Lab `Isaac-Velocity-Flat-Anymal-D-v0` with `--headless --enable_cameras` rendered a 640×480 RTX camera
      image (SMOKE_OK).
    - Install about 6 min, first render about 5 min (shader compile). The reference image is
      `...\scratchpad\isaac_smoke\anymal_runpod_render.jpg`.
    - **RunPod uses (rendering only):**
      1. **Sensor-model validation:** replay a few logged Modal episodes on RunPod with **RTX lidar + real robot cameras**.
         Compare against the merged ray-cast lidar and the geometric camera model (range-error histogram, detection
         agreement). Report it in ROBOT_RESULTS. This turns the sensor-model caveat into a measured result.
      2. **Photoreal shots for the video and the build log:** the ANYmal-D episodes + the factory built from NVIDIA
         SimReady/robot assets (CC-BY), rendered natively in Isaac. They replace the Blender robot shots;
         Blender stays for anything Isaac can't do in time.
      3. **CCTV-view frames with ground truth** (Replicator) for workstreams V and G, if time allows.
    - **Mechanics:** no file transfer through MCP. Use `runpodctl send/receive`, or SSH with a registered key, for code
      and results. Bake the install into a reusable script or template so each pod is ready in about 6 min.
      **Terminate every pod right after its job.** Never leave one idling.
  - **Gate:** ANYmal walks to a Nav2 goal in a simple scene on Modal **by Sat 20:30**. Otherwise switch to Branch M,
    keeping the bridge, sensor models and Nav2 config, which are simulator-agnostic by design.
- **Branch M (if Isaac fails, or as the fallback):**
  - MuJoCo with a Menagerie quadruped and MuJoCo Playground's pretrained **Go1** joystick policy
    (`experimental/sim2sim/onnx/go1_policy.onnx`, ONNX Runtime on CPU). Label it "Unitree Go1 stand-in, ANYmal-class task".
  - A thin ROS 2 Humble bridge in WSL2: `/cmd_vel` → policy, `/clock`, `/imu` with noise, `/odom` with drift, `/scan` from
    `mj_multiRay` on a gravity-levelled frame, offscreen camera.
  - **Disclose:** the policy observes the sim's true body velocity as a stand-in for an onboard estimator. Nav2 never sees it.
  - Smoke test by Sat 18:30.
- **Branch A (last resort):** CHAMP quadruped on Gazebo Classic + Humble Nav2 (an open-loop gait: weaker locomotion
  credibility, so say so). Swap its DWB config for MPPI Omni.

### 5.3 Navigation stack (identical in every branch)
- **Nav2 Humble:**
  - SmacPlanner2D;
  - MPPI with `motion_model: Omni` (vx 0.6–0.8, vy ≈ 0.3, wz ≈ 1.0, ax/ay ≈ 1–1.5; batch 500 on CPU);
  - Rotation Shim;
  - AMCL `OmniMotionModel` with raised alpha noise (slam_toolbox localisation as fallback);
  - collision monitor (slowdown + stop polygons = the protective stop);
  - footprint = body + leg swing.
- **Observer → Nav2 bridge (`observer_bridge` node):** decodes the **signed broadcast** (Zenoh, BLE fallback) and publishes
  standard costmap-filter masks. This is Nav2's documented mechanism; `maskCallback` updates at runtime.
  - **KeepoutFilter mask** = union over [now, now+H] of the L1 envelopes projected with the received phase, plus UNKNOWN cells.
  - **SpeedFilter mask** from L2/L3 flow probability for the current time-of-day and production state.
- **Timing layer (`timing_planner` node):** SIPP (Phillips & Likhachev 2011; reuse `robot/sipp.py` or libMultiRobotPlanning, MIT)
  over the observer map. It chooses *when* to depart and wait and which inspection target comes next, then sends goals via
  `nav2_simple_commander`. Nav2 executes. MPPI and the collision monitor handle residual dynamics. Cite SIPP; don't call it novel.

### 5.4 World and interactions (the robot only ever perceives them through its sensors)
- **People:** PySocialForce (MIT), with the robot as a repulsive agent. 20 % impassive/distracted.
  Plus an "invisible robot" stress condition where no one reacts.
- **Forklifts:** follow routes and **do not yield**. They brake only after a 1.2 s driver reaction with bounded deceleration.
- **AGVs/tuggers:** stop on ISO 3691-4-style protective fields (realistic, not a cheat).
- **Machines:** arm envelopes replayed kinematically from `truth/kinematics.py`.
  - Light-curtain trip and hold semantics are enforced by the sim (BRIEF).
  - The robot enters a hazard zone only in a predicted idle window **and** after its own camera confirms no motion.
- All movers are collision bodies the lidar really hits (mocap bodies in MuJoCo / kinematic prims in Isaac). No animated
  actors without collision.

### 5.5 Inspection task
- Reach the inspection pose. Hold body speed below a threshold for dwell T. Aim and capture with the robot camera.
- Pass checks, computed from the camera model in the loop. The RGB evidence image is rendered offline in Blender from
  the logged pose for the report and video:
  - the target centre projects inside the image;
  - the line of sight is unoccluded;
  - incidence angle < 60°;
  - pixel density meets IEC 62676-4 DORI (state the level used);
  - motion blur < 1 px.
- Metrics: inspection success rate, time per target, image-quality pass rate.

### 5.6 Evaluation protocol (Francis et al. 2023, arXiv 2306.16740)
- **Conditions (paired by seed):**
  - (1) no observer map (Nav2 + sensors only);
  - (2) static keep-out of all hazard zones;
  - (3) **learned observer map** (InterCycle);
  - (4) oracle map from truth, an upper bound labelled as such.
- **Data hygiene:**
  - The observer map is learned on **training days only**.
  - Test episodes use held-out seeds with cycle jitter, andon stops and unscheduled events.
  - Never tune on test seeds.
  - Report every failure and timeout.
- **Metrics:**
  - success;
  - collisions (people / vehicles / static / envelope intrusions);
  - minimum distance to people and minimum time-to-collision;
  - personal-space violations;
  - protective stops and stalled time;
  - time-to-goal, path length, SPL, path irregularity;
  - inspections completed, windows hit or missed;
  - line time lost (trips + holds);
  - real-time factor.
- **Statistics:**
  - Wilcoxon signed-rank for continuous metrics, McNemar for binary ones, bootstrap 95 % CIs.
  - Clopper-Pearson for collision rates. Zero collisions in N runs means an upper bound of ≈ 3/N, so say so.
- **Size:** at least 10 seeds × 4 conditions × 15-min episodes; more if compute allows (Modal CPU for Branch M, L40S for I).
- **Limitations stated in the README:**
  - simulation only; 2D lidar;
  - kinematic movers;
  - simulated humans are more constrained than real ones;
  - a stand-in robot model in Branch M;
  - no real-robot validation.

### 5.7 Ownership, files, timeboxes
| Sub-stream | Owner | Files | Done by |
|---|---|---|---|
| R1 sim + locomotion + sensor bridge (branch per 5.2) | agent | `ros2/intercycle_sim/*`, `cloud/modal_isaac_robot.py` | Sat 20:30 gate |
| R2 Nav2 config + `observer_bridge` + `timing_planner` | agent | `ros2/intercycle_nav/*` | Sat 23:30 |
| R3 movers/interaction + inspection checks + evaluation runner + stats | agent | `ros2/intercycle_eval/*`, `docs/ROBOT_RESULTS.md` | runs overnight; results Sun 10:30 |
| Integration, learned observer → broadcast (replaces ShimObserver), event-sim re-run | **lead** | `sim/`, `observer/` glue | Sat 22:00 |
| R4 RunPod rendering: the install script/template, sensor-model validation vs RTX lidar/cameras, photoreal ANYmal shots (coordinate the shot list with X) | agent (starts after R1 has logged episodes) | `cloud/runpod_isaac/*`, `docs/RENDER.md` (Isaac section) | validation by Sun 08:00, shots by Sun 11:00 |

- Keep the ROS 2 workspace on WSL ext4 (`~/intercycle_ws`) and sync it to `ros2/` in the repo. Use CycloneDDS,
  `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, and kill the process group per episode.
- **Do NOT:**
  - fix the old custom physics sim;
  - write a local planner;
  - give the robot or Nav2 any ground-truth topic;
  - make movers yield to the robot unconditionally;
  - port HuNavSim (time sink).
- **Debug cap:** 30 minutes per bug, then take the next branch down.

---------------------------------------------------------------------------------------------------------------------------

## 6. Workstream G — Gemma 4 as the observer's semantic layer (P1, the AMD workload)

**Facts (from the Token Factory guide):**
- Base URL `https://developer.amd.com.cn/radeon/api/v1`. Model `Gemma-4-31B-IT` (VLM: text + image).
  - The guide's image paragraph doesn't list Gemma, but its model table does.
  - Our probe confirmed vision works (`out/llm_calls.jsonl`). Re-probe once before a batch.
- Already configured: `.env` holds `INTERCYCLE_LLM_BASE_URL`, `_API_KEY`, `_MODEL`, `_JSON_MODE`. The client is
  `intercycle/llm/client.py`, tasks are in `llm/tasks.py`, and frames come from `scripts/gemma_frames.py`, with
  `scripts/gemma_run_eval.py` written but never run. **Reuse them; don't rewrite.**
- Gemma rules:
  - a `system` message must come first;
  - no thinking;
  - images go in as data URLs;
  - use `response_format: {"type":"json_object"}`.
- `seed` and `stop` are silently dropped, so reproducibility comes from the replay cache.
- **Limits (shared by the whole key):** 30 RPM per key, 20 RPM per account, 8 concurrent, plus a daily spend cap.
  - **Measured by the monitor via `/v1/usage` on 17 Oct:** `rpm_limit` 20, `daily_cost_limit_usd` 1.00, resets daily at
    16:00 UTC (18:00 CEST).
  - Gemma cost observed: about $0.034 per 1M prompt tokens. So the $1/day cap allows millions of tokens, and **RPM (20) is
    the binding limit**: about 900 requests/hour at ≤ 15 req/min.
  - Read `/v1/usage` before and after each batch.
  - Retry only on 429, honouring `Retry-After`, with backoff.
  - Run **≤ 15 req/min**.
- Check `GET /v1/models` at runtime. The catalog changes.
- Prompts are logged for about 2 days on AMD's side. Send only synthetic frames, never real people's faces.

**Build (in this order; stop after G3 if time is short):**
- **G1. Commissioning:** one keyframe per camera → JSON with machine regions, machine types, zones and inspection targets.
  Score: IoU against the true zones. The regions can feed workstream P.
- **G2. Naming discovered phases:** keyframes of each discovered sub-phase and line state → a label. Score: accuracy against true phase names.
- **G3. New-machine triage:** the change-detection blob crop → one of: new moving machine / parked object / camera bump.
  **Gemma never frees space.** Cells stay UNKNOWN until the cycle is learned, whatever Gemma says.
- G4. State verification (arm moving? person in zone?) → broadcast confidence. Only if time allows.
- G5. **Only if the MI300X arrives:** serve Gemma with vLLM-ROCm on it, re-run G1–G3 and add a latency/throughput column.
  Token Factory stays the default.
- Out of scope: the ops Q&A agent (demo nicety, P2).

**Deliverable:** `docs/GEMMA_RESULTS.md`, with accuracy, latency and calls per task, and the line "runs on AMD GPUs via Token Factory".
Record mode with the replay cache, so the demo works offline.

**Timebox:**
- G1–G3 by Sun 10:00;
- re-run frame generation first (`scripts/gemma_frames.py --n 160 --composites 8`).

**Fallback:** if the quota runs out, report on the subset already cached.

---------------------------------------------------------------------------------------------------------------------------

## 7. Workstreams C, V, X — deployability and realism (P1)

### C — two redundant channels + robot interface (Sunday morning)
- **Primary:** Wi-Fi + Zenoh (`eclipse-zenoh` python, Apache-2.0; rmw_zenoh is the ROS 2 path). Map, fast and live layers go out as messages.
- **Fallback:** the existing BLE 5 signed advertising.
- **Both channels carry the same Ed25519-signed payload.**
- Demo: drop Wi-Fi, B continues on BLE. Drop both, and after the TTL every dynamic zone counts as occupied and B waits safely.
- **Robot interface:**
  - `export/nav2.py` writes time-indexed `nav_msgs/OccupancyGrid`-format files for a Nav2 StaticLayer;
  - `docs/INTERFACE.md` specifies the `get_safe_windows(waypoint) -> [t_start, t_end]` service.
  - A C++ Nav2 plugin is out of scope.
  - STVL is LGPL: runtime dependency only.
- **Timebox:** Sun 10:30.
- **Fallback:** Zenoh in loopback with simulated drops is enough.

### V — the map is learnable from pixels
- Run the pixel path on a few rendered hours: workstream X frames → RF-DETR (Apache-2.0, sizes N–L only) + ByteTrack
  → homography ground-plane fusion → `Observation` → mapping.
- **Measure the perception error.** Calibrate `CameraShim`'s noise, dropout and latency from it. The long-horizon map then
  runs on a shim with measured, not assumed, error. Say this explicitly in MAP_RESULTS.
- Perception on PhysicalAI-SmartSpaces (CC-BY-4.0) has already run: Warehouse_000, 19 cameras, RF-DETR-M + ByteTrack on an L4.
  Results: P/R 0.89/0.68, MOTA 0.59, IDF1 0.34, about 42 fps per L4.
  - **Honesty fix:** SmartSpaces is **photorealistic synthetic** video (rendered by NVIDIA in Omniverse), **not real video**.
    STATUS.md calls it "real video". Correct that everywhere.
  - Set the fusion radius default to the measured best (1.5 m; the code still says 0.8 m).
  - Next: a body-shop-like scene if SmartSpaces has one, plus the ground-plane occupancy error used to calibrate the shim.
- **Compute:** Modal GPU now. If the MI300X arrives, ONNX Runtime ROCm/MIGraphX on it.
- Real video: if the user provides a clip of a real cyclic machine, run per-machine phase discovery (workstream P) on it.
  It's the strongest realism proof.
- **Timebox:** Sun 11:00.
- **Fallback:** SmartSpaces numbers + shim calibration from them.

### X — visual realism: as close to photoreal as possible
Blender Cycles is a path tracer. What looks fake is the content (procedural block arms, block car bodies, capsule people). Fix the content:
1. Real assets via glTF/USD import, with credits in `credits.json`:
   - NVIDIA 6.0 CC-BY models: Fanuc R-2000iC, forklifts, conveyors, SimReady guards, racks, structures.
   - **Car body-in-white** assembled from the SimReady car frame + body parts (CC-BY). No Kenney low-poly.
   - Replace MDL materials with CC0 PBR materials (Poly Haven / ambientCG).
2. Rigged humans (MPFB2/MakeHuman, CC0 output) with walk / idle / work clips, wearing PPE. No capsules.
3. Plant look:
   - high-bay LED grids, skylights, an industrial HDRI;
   - clutter (cable trays, columns, signage, floor markings, pallets);
   - weld sparks and flashes.
4. **CCTV look** for the observer views:
   - ceiling viewpoint, wide-angle distortion, motion blur;
   - sensor noise, H.264 artefacts, timestamp overlay;
   - per-frame randomisation.
   These frames feed workstream V.
5. Ground truth passes (segmentation, depth, boxes).
6. Modal: OIDN denoiser (OptiX fails in Modal's sandbox). Fix this first and re-run `modal_blender.py::smoke`.
   The laptop GPU (MX250, 2 GB) is for tiny smoke renders only. Real-asset scenes render on a Modal L4.
7. **Acceptance:** side-by-side stills against a real body-shop photo in the build log. The user decides; the target is "could pass as real CCTV".

**Timebox:**
- asset import: 3 h;
- first realistic stills by Sat 21:00;
- final shots overnight.

**Fallback:** an asset class that fails stays procedural but gets real materials. Last resort for the video: the Three.js replay.

### D — live view of the map growing (user requirement, 17 Oct 14:50)
**Goal:** one web page where you watch the observer's map being built and sharpening over time. It doubles as the demo
judges can open and as a video scene (screen-recorded, so no render budget).

**Data contract (agree this first; M writes it, D reads it):** `observer/stream.py` writes, every N simulated minutes:
- `out/stream/<run>/snap_<t>.json`:
  - observation time;
  - the current production state, as discovered;
  - L1 envelopes per machine and phase bin, as boxes or run-length masks;
  - L2/L3 occupancy + heading grids for the current time-of-day slice;
  - UNKNOWN cells;
  - per-machine phase now;
  - map metrics so far (IoU, unsafe-free, Brier).
- `out/stream/<run>/events.jsonl`: change detected, new machine spawned, phase discovered or named by Gemma, camera offline.

In live mode the same messages go over a local websocket. Keep snapshots small (≤ 200 kB each); downsample if needed.

**The page (`dashboard/`, static HTML + JS, no build step):**
1. Top-down plant view (L0 walls, conveyor, machines) with toggles for L1 / L2 / L3 / UNKNOWN. A 2.5D tilt that extrudes
   collision volumes is optional.
2. **Two modes:**
   - **Live:** connects to the websocket while a sim + observer run (`scripts/map_timeline.py --live`). The map fills in as you watch.
   - **Replay:** loads a recorded run. A time slider runs from 1 h → 1 day → 1 week with play / speed controls.
     It hosts statically (GitHub Pages from the repo), so judges can open it.
3. Metric curves (IoU, unsafe-free rate) with a cursor synced to the slider.
4. Event timeline: the new machine appears as an UNKNOWN blob, gets learned, then the envelope settles. Gemma labels show as captions.
5. Per-machine strip: the discovered sub-phases for the selected machine, with the phase-now marker moving.
6. Optional: the legged robot's Nav2 path and the keep-out mask it received (from R3 episode logs).
7. Label everything "SYNTHETIC: simulated plant".

**Done when:**
- replay of the 1-week run and the new-machine run works from the static page;
- live mode works locally;
- a 30-second screen recording exists for the video.

**Do NOT:**
- build a backend beyond a tiny websocket relay;
- add a frontend framework that needs a build step;
- host on Modal (static hosting only).

**Timebox:** 4 h total.
- Skeleton against the existing `scripts/build_map.py` outputs: Sat 17:00–19:00.
- Wire to M's real snapshots: Sat 22:30–23:30.
- Polish: Sun 09:00–10:00.

**Fallback:** replay mode only, with the PNG snapshots M already produces plus the metric curves.

---------------------------------------------------------------------------------------------------------------------------

## 8. Compute and budget

**Modal: about $30 of credits, already logged in on this workstation. About $1.5 was spent on 17 Oct before the stop. Hard rules:**
- **Ceilings from now on** (prices: L4 about $0.80/h, L40S about $1.95/h):
  - Blender renders ≤ $10;
  - perception GPU ≤ $4;
  - CPU sweeps and map builds ≤ $4.
  - Total $18, which with the $1.5 already spent means **stop and ask the user at $20 cumulative**.
  - The rest is a reserve the user releases.
- Before every Modal run: a smoke test at the smallest size, then an estimated cost written in STATUS.
- After every run: `modal app stop`. Log GPU type, minutes and cost in STATUS.
- Cheapest GPU that works (L4 before L40S). Low samples + denoiser. Render only the shots in the shot list.
- Local first: the laptop GPU (MX250) and CPU handle tests, smoke renders and map builds up to 1 week.

**Isaac (only if Branch I passes the gate, §5.2):** the user tops up Modal for it. Separate cap: **$25 of L40S time**
(≈ 12 h at $1.95/h), logged separately in STATUS. Stop and ask at $20 of Isaac spend. Isaac cannot render on Modal,
so all visuals stay in Blender. The monitor's Isaac tests on 17 Oct cost about $1.5 (log it).

**User's desktop (free; connected 17 Oct ~16:15 as a Remote Control session, likely `alexanderis-buzzing-planet`):**
- Hardware: RTX 4060 Ti 8 GB, 32 GB RAM, **Blender installed with its MCP server**.
- Reach it by `SendMessage` to that session. Confirm the name with `ListAgents`.
- **Use it for:**
  1. **Workstream X (primary render machine):** build and iterate the plant scene interactively through the Blender MCP
     (asset import, CC0 materials, lighting, CCTV cameras), then render Cycles/OptiX stills and CCTV frames locally.
     Modal's Blender budget becomes **overflow only**.
  2. **Workstream V:** RF-DETR (N–M) + ByteTrack inference on CUDA, locally. The Modal perception budget becomes overflow only.
  3. Branch M fallback (MuJoCo + Nav2), only if R falls back. 32 GB RAM beats the laptop.
- **Not for Isaac:** 8 GB is below Isaac Sim's 16 GB minimum. Isaac stays on Modal (physics) and RunPod (rendering).
- **One GPU, one owner at a time.** X owns the desktop GPU and the Blender MCP scene. V runs batches only when X is idle,
  and coordinates through the lead. Never run two GPU jobs at once (8 GB).
- **Code and data sync:** the repo isn't pushed yet. The desktop needs it via a git remote (see the open question to the user).
  Results come back by git (small files) or a shared folder. Never put secrets in what's synced; the desktop needs its own `.env` only if it calls Gemma.

**RunPod (Isaac rendering only; the user's own money):**
- RTX 4090 Secure $0.89/h; Community $0.34/h when available.
- **Cap: $10 total until the user raises it.** Estimate before each pod and log the actual figure in STATUS.
- Terminate pods immediately after use.
- The monitor's render test on 17 Oct cost about $0.18.

**AMD (what makes us eligible):**

| Workload | Now | If the MI300X arrives |
|---|---|---|
| Gemma 4 (workstream G) | **Token Factory (AMD GPUs)**: the required AMD workload | vLLM-ROCm on MI300X |
| Perception (workstream V) | Modal GPU | ONNX Runtime ROCm/MIGraphX on MI300X |
| Map builds, sweeps | local / Modal CPU | unchanged |

The README states exactly what ran on AMD. Never claim the MI300X if it didn't arrive.

---------------------------------------------------------------------------------------------------------------------------

## 9. Schedule and checkpoints (CEST)
**The user is available overnight.** Keep working through the night. Escalate blockers, budget approvals (section 8) and
branch decisions immediately instead of parking them until morning. Overnight priorities, in order:
- R evaluation batch;
- M 1-week / new-machine runs;
- X renders;
- G batches within the quota.

| When | Must be true | Status to user |
|---|---|---|
| Sat 15:30 | The 5 failing tests (listed in STATUS.md) fixed; the doc tidy-up committed; agents M, P, G, X, D launched, each given its section of this doc + section 0 + section 1. M and D agree the stream data contract | ✔ |
| Sat 19:30 | M: metrics + 1 h → 1 week curves. X: first realistic stills. D: page skeleton replaying existing map outputs | ✔ + build log update |
| Sat 18:30 | R1: Isaac image + ROS 2 Humble/Nav2 builds on Modal; ANYmal-D (flat policy) follows `/cmd_vel` in an empty scene. Start Branch M only if R1 is blocked | ✔ |
| Sat 20:30 | **R gate:** the legged robot walks to a Nav2 goal (Branch I on Modal, else Branch M). Pick the branch, then stop the other | ✔ |
| Sat 22:30 | M: new machine. P: phases scored. Lead: learned observer replaces ShimObserver | ✔ |
| Sat 23:30 | D: live + replay working on M's real snapshots | ✔ |
| Sat 23:30 | R2: observer_bridge masks + timing planner working; R3: movers + inspection checks. Overnight: evaluation batch (4 conditions × ≥10 seeds) + renders, within budget | ✔ + build log update |
| Sun 10:00 | G1–G3 done. C started | ✔ |
| Sun 10:30 | R3: `docs/ROBOT_RESULTS.md` with paired stats | ✔ |
| Sun 11:00 | C, V done or fallen back. Held-out seeds 1000–1199 run (event-sim scheduling analysis) | ✔ |
| **Sun 12:30** | **CODE FREEZE.** README, MAP_RESULTS, GEMMA_RESULTS, PHASES final | ✔ |
| Sun 12:30–15:30 | Video (≤ 3 min), deck, submission form. Submit by 15:30 | ✔ |

**P2 — only if everything above is green:**
- worker routines (trajectory clustering per time of day; ASOT on pose);
- a FreMEn spectral model per cell for L2/L3;
- a physics-sim sweep on Modal;
- the ops Q&A agent.

## 10. Video story (what the judges must see, about 3 min)
1. The plant in realistic CCTV views. "No PLC, no setup."
2. The observer's map growing: 1 h → 1 day → 1 week, with metric curves, screen-recorded from the live view (workstream D).
3. Discovered phases of one machine, named by Gemma on AMD.
4. A new machine is installed: it goes UNKNOWN, is learned, then re-converges.
5. The legged robot walks through the plant on Nav2 with the observer's masks, waits for a window, inspects, and leaves.
   Show the paired results table (learned map vs no map vs static keep-out vs oracle).
6. Robustness: a channel drops, and the robot falls back safely.
7. Honesty card:
   - synthetic data;
   - advisory, not safety-rated;
   - what ran on AMD.

## Cuts (agreed with the user)
- Isaac: **physics** on Modal (credits) for the robot evaluation. **Rendering** on RunPod RTX 4090 only (sensor validation +
  photoreal shots + CCTV frames, §5.2). Never on Modal (it crashes there).
- The custom 10 Hz physics sim is retired for robot results.
- No wheeled robot in results.
- Worker routines: P2.
- No new entities.
- Data hall: only what already works.
- Ops Q&A agent: P2.
