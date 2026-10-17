# Robot results: legged inspection robot on Nav2 with the observer map (SYNTHETIC; simulation only)

Workstream R3 (DIRECTION section 5.4-5.6). Every number on this page comes from simulated episodes. ANYmal-D runs in
Isaac Lab 2.3 / Isaac Sim 5.1, physics only, on Modal L4. Navigation is ROS 2 Humble Nav2. Nothing here was run on a
real robot.

## 0. Pre-registration (committed before the held-out batch; the commit timestamp is the record)

This analysis plan is fixed before any held-out episode runs. It is not changed after seeing results.

- **Data.**
  - Held-out seeds 1000-1009, 4 conditions each (none, static, learned, timing oracle), paired by seed.
  - One fixed 600 s episode per seed and condition (section 4), run once.
  - Invalid episodes are listed and excluded; this is decided only by the cheat guard and episode errors. Nothing is
    re-run or replaced.
- **Primary endpoint.**
  - `line_time_lost_s`: union of line-group freeze time from robot-caused trips (90 s recovery each) and holds inside the
    episode.
  - Comparison: **learned vs none** (no map).
  - Test: exact paired sign-flip permutation test on the 10 per-seed differences, two-sided, alpha 0.05.
    SciPy `permutation_test(permutation_type="samples")` enumerates all 2^10 = 1024 sign flips.
  - All 10 paired differences are shown, as a table and a plot. Bootstrap intervals are not the lead statistic at n = 10.
- **Safety endpoint.**
  - `any_collision` per condition: episodes with at least one robot collision (person / vehicle / machine part /
    static), with a Clopper-Pearson 95 % interval.
  - Collision events per hour of episode time get an exact Poisson 95 % upper bound.
  - With 0 collisions in the 10 learned episodes (100 min), the 95 % upper bounds are **0.31 per episode**
    (Clopper-Pearson) and about **2.2 per hour** (Poisson). No strong safety claim is made from this.
  - Collisions where the robot itself was moving (> 0.1 m/s), and person contacts by impassive walkers, are broken out.
- **Secondary endpoints, Holm-corrected as one family.**
  - Endpoints: `inspections_ok`, `line_time_lost_s` (other pairs), `machine_time_lost_s`, `trips`, `windows_hit`,
    `any_collision`, `collisions_robot_moving`, `min_dist_person_m`, `stalled_s`, `spl`, `time_per_target_s`.
  - Pairs: learned vs none, learned vs static, timing oracle vs learned.
  - Each uses the same exact sign-flip test.
  - Everything else (Francis et al. metric list, section 4) is descriptive.
- **Scope.** 10-minute episodes in production only. No per-shift or per-day extrapolation.

### 0.1 Amendment 1 (17 Oct, decided by the user on DEV data only, before any held-out episode)

- **Change.** The primary inspection pass becomes DORI **Inspect, 1000 px/m** (IEC 62676-4), with all other checks
  unchanged: in image, unoccluded, incidence < 60 deg, blur < 1 px.
  - It applies to targets inside the guarding that need line of sight from inside the zone: weld-gun tips, the tip
    dresser, clamp units, and the lifter carriage underside (S80).
  - Target points are fixed per machine from the site plan, never tracked from truth. They are justified by what a
    maintenance inspection must see.
  - `inspections_ok` = captures passing at the Inspect level. The former criterion, Identify (250 px/m), stays as a
    secondary row (`inspections_ok_identify`) for every condition, so the change is transparent.
- **Reason.** Dev seed 20 showed boundary captures passing Identify, so the static baseline passed 10/10 from the
  boundary. That does not test the inspection need: these targets need a close in-zone view.
- **Consequence, stated up front.** `static` never enters a zone, so it gets **0 Inspect passes by construction**. This
  is not a static "failure". `static` and `none` stay in every table for line time lost and safety.
- **Unchanged.** The primary endpoint (`line_time_lost_s`, learned vs none), the safety endpoint, the secondary family
  and the tests.

<!-- RESULTS -->
_Results pending: the held-out batch has not run yet._
<!-- /RESULTS -->

## 1. What is compared

There are four conditions, paired by seed. Each seed has the same plant start time, the same machine cycles, the same
movers and the same sensor-noise seed in all four conditions:

| # | Condition | What the robot receives | Entry rule at a hazard boundary (R2's timing_planner) |
|---|---|---|---|
| 1 | `none` | Nav2 + its own sensors only | the robot camera's no-motion check alone (reactive) |
| 2 | `static` | static keep-out of every hazard zone (provisioned site map) | never enters; inspects from the boundary |
| 3 | `learned` (InterCycle) | signed observer broadcast: learned map layer + learned cycle models + live nowcast | a predicted idle window with `need_s` left AND the camera's no-motion check |
| 4 | `oracle` = **timing oracle** (UPPER BOUND) | true cycle timing (PLC-equivalent) + the same learned spatial layers | same as 3 |

- Condition 4 is a **timing oracle**, labelled as an upper bound. Only its timing is oracle. Its spatial layers (L1-L3) are the same learned
  map as condition 3.
- No ground truth reaches Nav2 or R2's nodes in conditions 1-3:
  - The robot side gets `/scan`, `/imu`, `/odom`, TF, `/clock`, the static map, the verified broadcast and the camera
    model's observations.
  - `observer_bridge` and `timing_planner` refuse to run if `intercycle.truth` / `intercycle.sim` are imported.
- **Cheat guard (DIRECTION 5.1).**
  - The timing planner's entry gate reads `/intercycle/camera/observations`.
  - In every evaluation episode that topic is fed by R1's geometric camera model (section 3).
  - It is never fed by the `camera_stub` test double, which is not launched.
  - The runner (`intercycle_sim.episode.eval_check`) marks an episode **invalid** if:
    - goal.json reports `camera_stub_used`;
    - `require_camera` is false;
    - goal.json has no `camera_sources`;
    - no `camera_model` source fed the gate (`static` never enters, so it is exempt from this last check);
    - the episode ran on the kinematic test double.
  - Invalid episodes are listed below and excluded from the statistics.

## 2. World and interactions (sim side; the robot perceives them only through its sensors)

The plant is `ros2/intercycle_eval/intercycle_eval/plant.py`. It runs inside the simulator process (Isaac's python
3.11). Plant time is `plant_t0` + sim time.

- **Machines.**
  - Arm envelopes (5 circles per arm), car bodies on skids, the lifter carriage and docked AGVs are replayed from
    `intercycle/truth/kinematics.py` on the episode's `World`.
  - The `World` includes cycle jitter, takt drift, Poisson andon stops (4/h, 20-120 s) on the line, cell micro-stops
    (2/h, 5-30 s) and planned stops.
  - The lidar and the camera see all about 120 machine-part circles analytically.
  - In Isaac, the parts nearest the robot (within 8 m) get a pool of 41 kinematic cylinders, so the robot physically
    collides with them.
- **Light curtains (BRIEF safety semantics), enforced by the sim.**
  - The robot footprint is 3 discs covering the 1.0 x 0.64 m body.
  - **Trip:** the footprint crosses into an ACTIVE hazard zone. This is a protective stop: `World.trip` freezes the
    machine group for `recovery_after_trip_s` = 90 s, and a line station stops the whole line.
  - **Hold:** the footprint is inside an IDLE zone when the next start is due. The group's freeze is extended until the
    robot leaves; line stations hold the whole line.
  - The line walkway (x = 52.5 +- 1 m across the conveyor) is guarded during skid transfers, as in the event sim.
  - The freezes feed back into the kinematics, so arms freeze mid-motion. They also feed back into the observer's
    ceiling-camera signal: the episode's observer sees the floor the robot actually changed.
- **People.**
  - Social force along their truth routes: aisle walkers, break crowds, data-hall technicians. The force terms and
    default parameters are those of PySocialForce (MIT):
    - desired force, relaxation 0.5 s;
    - Moussaid 2009 social force (5.1, lambda 2, gamma 0.35, n 2, n' 3);
    - exponential obstacle force from the static geometry (sigma 0.2, factor 10).
  - They are **re-implemented in numpy** (`social.py`), because the PySocialForce package needs numba, which Isaac's
    python does not have.
  - The robot is one more agent in the social force. 20 % of people are impassive or distracted and ignore it; they
    still avoid each other and walls.
  - Stress flag `invisible_robot`: nobody reacts.
  - Nobody yields on command.
- **Forklifts** follow their routes and do **not** yield. When the robot or a person is in the 10 m corridor ahead, the
  driver brakes after a **1.2 s reaction**, at a bounded **1.5 m/s^2**, and restarts 1 s after the corridor clears.
- **AGVs and tugger trains** have ISO 3691-4-style scanner fields:
  - protective field = v * 0.1 s + v^2 / (2 * 1.0 m/s^2) + 0.3 m: stop at 1.0 m/s^2;
  - warning field = protective field + 1.5 m: slow to 0.3 m/s;
  - automatic restart 1 s after the protective field clears.
- **Isaac coupling.**
  - Every mover part is a kinematic rigid body. Its pose target is written at 10 Hz, with velocity extrapolation in
    between for the sensors.
  - Lidar and camera hit the same shapes analytically every step.
  - Contacts on non-foot robot bodies come from Isaac's contact sensor.

## 3. Robot camera (R1's geometric model) and inspection

- **Camera.** A pan-tilt inspection head, 1920 x 1080, 70 x 45 deg, mounted 0.45 m ahead of and 0.35 m above
  base_link. It aims at site-map locations (zone centre, inspection target) with a 2.5 deg Gaussian aim error. Aiming
  uses the simulator's true pose plus that error, as a stand-in for the robot's pose estimate.
- **No-motion gate** (`camera.py`; published by the sim bridge on `/intercycle/camera/observations` with source
  `camera_model (R1 geometric, R3 gate)`).
  - The view is the robot's wide-angle camera (110 x 80 deg), aimed at a FIXED site-map point per machine: the zone
    centre at part height, or for the lifter the midpoint of its carriage travel (3.2 m).
  - The aim is never at the moving parts. A dev build that aimed at the parts' true centroid was a truth leak; it was
    removed before any held-out episode. Parts that fall outside this fixed view count as not visible, so the gate
    stays closed; this is reported as a limitation.
  - Every 0.2 s it covers each machine whose zone is within 6 m.
  - The targets are the machine's moving parts. Static arm bases are excluded.
  - **visible** = at least 50 % of the parts are inside the frustum and unoccluded, AND the detector fires (p 0.95).
    Occlusion is a ray-cast against the static L0 mesh and every other moving body.
  - **motion_seen** = visible AND a visible part moves faster than 0.05 m/s.
  - Each observation is published 0.15 s later (latency).
  - The planner enters only after at least 2 consecutive clear observations spanning at least 0.3 s, the newest at
    most 1 s old.
  - **Known weakness, left in on purpose:** a weld arm dwelling at a weld point is momentarily still, and a camera-only
    check cannot tell that from idle.
- **Inspection capture** (DIRECTION 5.5).
  - It is triggered by the planner's `dwell_start`. The frame is taken once the true body speed has stayed below
    0.05 m/s and |yaw rate| below 0.05 rad/s for 1 s. Otherwise it is taken at `dwell_end`, flagged `settled = false`.
  - Target: the nearest arm base (cable dress pack / J1) to the inspection pose, at 1.0 m height, surface normal facing
    the approach. For the lifter, the lifter frame.
  - Pass checks, all required:
    - the target centre projects inside the image;
    - the line of sight is unoccluded;
    - incidence < 60 deg;
    - pixel density >= **250 px/m (IEC 62676-4 DORI "Identify")**; the 1000 px/m "Inspect" level is reported as well;
    - motion blur < 1 px at a 4 ms exposure.
  - The camera pose of every capture is logged for offline RGB rendering (R4).

## 4. Protocol (Francis et al. 2023, arXiv 2306.16740)

- **Episode.** A fixed **600 s of sim time**, identical for every condition, with inspections counted inside that
  window.
  - Nav2 boots inside the window (first few seconds).
  - The timing planner runs the default mission: all 10 machine targets in walking order. Each target takes Nav2
    walking time, waiting for a window and the camera, entry, dwell and exit.
  - At 600 s the sim stops and the planner is interrupted. Its events are the record.
  - The mission is longer than 600 s, so episodes measure throughput.
- **Plant start time.**
  - `plant_t0` = 06:30 + U(0, 15 h) of day 0, drawn from `default_rng([seed, 5])`.
  - It is re-drawn until [t0 - 60 s, t0 + 660 s] overlaps no planned line stop (breaks, shift changes).
  - So every episode runs in production with machine cycles and people/vehicle traffic.
- **Data hygiene.**
  - The observer map is `out/timeline/map_1w.npz` (workstream M), trained on **seed 0, sim days 0-6
    (t = 0-604 800 s), 28 x 6 h windows**. M's held-out day-7 numbers: unsafe-free 0.82 % strict / 0.005 %
    cell-centre, false-occupied share 0.29. It uses equal-slice phase bins.
  - The cycle models (slow layer) are learned by `observer_learned.learned_models` on training seed 99,
    06:10-10:00.
  - Both training seeds are disjoint from the dev seeds (20-29) and the test seeds (1000-1009).
  - Development used dev seeds only. The final batch uses held-out seeds **1000-1009**, run once, never tuned on.
- **Pre-registered primary endpoints** (declared before the held-out batch):
  - endpoints: `line_time_lost_s`, `inspections_ok`, `any_collision`;
  - comparisons: learned vs none, learned vs static;
  - test: exact two-sided sign-flip permutation test on the per-seed paired differences (all 2^10 sign assignments),
    Holm-corrected over those 6 tests.
  - Everything else is secondary and descriptive.
- **Scope of the claim.** Results are for **10-minute episodes in production only** (no breaks, shift changes or line
  start-up inside a window). Nothing is extrapolated per shift or per day.
- **Statistics.**
  - Continuous metrics: per-seed paired differences, Wilcoxon signed-rank and paired bootstrap 95 % CIs.
  - Binary metrics: exact McNemar.
  - Rates: Clopper-Pearson 95 %. Zero events in N gives an upper bound of about 3/N. With N = 10 and 0 collisions, the
    two-sided 95 % upper bound is **0.31**; the table says so.
  - Every invalid, failed or timed-out episode is listed.

## 5. Limitations

- Simulation only. There is no real-robot validation.
- 2D lidar, about 0.8 m above the floor. Bodies below the plane are not seen.
- Movers are kinematic. Simulated humans follow routes with a social-force model and are more constrained than real
  people. Machine parts are plan-view circles with assumed heights.
- The locomotion policy observes the true base velocity, as a stand-in for an onboard estimator. Nav2 never sees it.
- **Stand mode (an estimator stand-in, like the policy's velocity input).** Under a zero `/cmd_vel` the pretrained rsl_rl policy creeps, about 5 cm/s and 1.5 rad/min. In the
  first dev dry run this drifted the robot from its wait pose into the S80 hazard zone (holds and a trip with no entry
  decision). The locomotion layer therefore has a stand state: after 0.2 s of zero command it holds the stance pose with
  a P-loop on the TRUE base pose; Nav2 never sees it. Any non-zero command releases it. A real ANYmal stance controller
  does not creep, but it is not perfectly station-keeping either, so this is optimistic about drift while standing.
- **People model.** A reacting person treats the robot's footprint as an obstacle (PySocialForce obstacle term) plus a
  keep-right tangential share, because the Moussaid term alone let walkers creep into a standing robot. The 20 %
  impassive walkers ignore the robot completely and can walk into it while it stands; such contacts are counted as
  collisions and broken out.
- `/odom` is a labelled leg/IMU-odometry model with drift; localisation is AMCL.
- The camera is a geometric model (frustum, occlusion, pixel density, noise, latency), not rendered images. The camera
  head is aimed from the TRUE pose plus a 2.5 deg Gaussian aim error, a stand-in for the pose-estimate and head error.
  The capture outcome published on ROS carries no pose.
- The oracle's spatial layers are learned. Only its timing is true.
- R2's planner keeps the whole hazard rectangle keep-out unless a predicted idle window covers [now, now + 6 s]. The
  reason is the BRIEF rule that crossing an ACTIVE boundary trips. HEM1's idle (15.75 s) is shorter than `need_s`
  (16.5 s), so in `learned` / `oracle` HEM1 is never entered. This is a real constraint.
- `static` inspects from the boundary only.
- The SpeedFilter's production-state slice comes only from the broadcast. A line where at least half the machines report
  STOPPED counts as a stopped line; otherwise it is RUNNING. No calendar is used (AUDIT Tier 1).
- Further disclosures:
  - arm links are floor-to-2 m columns (conservative);
  - the AMCL map is the exact sim geometry;
  - the collision-monitor stop polygon is 8 cm;
  - the IMU is unused by Nav2;
  - every cycle runs the same robot program (truth kinematics), with 1 % takt jitter.
- The map's phase convention (phase 0 = motion start, the same as the fast layer) is assumed with +-1 bin padding. It is
  verified on a dev seed below.
