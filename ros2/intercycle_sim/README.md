# intercycle_sim: legged-robot simulation bridge (R1, DIRECTION §5.2 Branch I)

The robot is ANYmal-D in Isaac Lab 2.3 / Isaac Sim 5.1. It runs **physics only** (headless, no RTX, no cameras) on Modal,
and the locomotion controller is the pretrained rsl_rl **flat** velocity policy. The robot is driven by ROS 2 Humble +
Nav2 through a lockstep socket bridge. The robot loop gets no ground truth: Nav2 sees only `/scan`, `/imu`, `/odom`, TF,
`/clock` and the static map.

**Disclosure.** The locomotion policy observes the true base velocity, which stands in for an onboard state estimator.
That value stays inside the controller. `/odom` is a labelled leg/IMU-odometry model that drifts.

## Processes (one Modal container per episode, 8 CPU + 1 GPU)

| Process | Interpreter | What it does |
|---|---|---|
| `intercycle_sim/runner.py` | Isaac python 3.11 (clean env, no ROS vars) | Steps Isaac. Computes sensors (`sensors.py`). Drives movers. Logs truth for evaluation only. |
| `intercycle_sim/ros_bridge_node.py` | `/usr/bin/python3` 3.10 + Humble | Unix-socket server. Publishes topics. Replies with the latest `/cmd_vel`. |
| `ros2 launch nav2_bringup bringup_launch.py` | Humble | map_server + AMCL + planner/controller/BT/behaviors/smoother/velocity_smoother, `use_sim_time:=True`, `use_composition:=False` |
| `intercycle_sim/send_goal.py` | Humble | nav2_simple_commander: sets the initial pose once (the known dock pose), sends the goal, writes the result JSON |
| `cfg["extra_launch"]` | Humble, shell | R2/R3 nodes (observer_bridge, timing_planner, loggers), started after Nav2 |

`episode.py` orchestrates them. Every process is its own process group, and all groups are killed at the end of the
episode.

DDS is CycloneDDS pinned to `lo` with unicast discovery (`cyclonedds.xml`) and `ROS_DOMAIN_ID=17`.
`ROS_LOCALHOST_ONLY` is deliberately **not** set: rmw_cyclonedds would add `lo` twice and fail.
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` is set.

Lockstep works like this. Isaac sends one frame per 20 ms policy step and waits for the reply, so `/clock` has no gaps.
The sim is paced to `max_rtf` (default 1.0), which gives Nav2 real compute time.

## Topics and frames

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/clock` | rosgraph_msgs/Clock | 50 Hz | Sim time. Every node uses `use_sim_time`. |
| `/scan` | sensor_msgs/LaserScan | 10 Hz | Frame `laser`. 360 rays, angle_min = -pi, 0.1 to 25 m, σ 1.5 cm, 0.5 % dropouts, `inf` = no return. Gravity-levelled 2D plane about 0.8 m above the floor. |
| `/imu` | sensor_msgs/Imu | 50 Hz | Frame `imu_link`. Gyro + accelerometer with bias random walk and noise. No orientation (`orientation_covariance[0] = -1`). |
| `/odom` | nav_msgs/Odometry | 50 Hz | `odom` → `base_link`. Labelled leg/IMU-odometry stand-in: true planar body velocity × (1 + scale error) + bias random walk + noise, integrated. Starts at (0, 0, 0). |
| `/tf` | | 50 Hz | `odom` → `base_link` (bridge). `map` → `odom` comes from AMCL. |
| `/tf_static` | | once | `base_link` → `laser` (0.30, 0, 0.20), `imu_link` (0, 0, 0), `camera_link` (0.45, 0, 0.35) |
| `/cmd_vel` | geometry_msgs/Twist | subscribed | Clipped to ±1 m/s and ±1 rad/s (the policy's training range). Zeroed if older than 0.5 s of sim time. |
| `/intercycle/contacts` | std_msgs/String (JSON) | on contact | `{"t", "contacts": [{"body", "force_n"}]}` for non-foot robot bodies. For R3's logger only. |

The world frame is ROS `map`, with x = plan x (along the line) and y = plan z (across the hall). A plan yaw
(`atan2(dx, dz)`) becomes a ROS heading with ψ = π/2 − yaw (`movers.plan_to_ros`). The robot starts at the dock
(4, 32), facing +x.

## Static world (built locally from `scenario/bodyshop.yaml`, `intercycle_sim/world.py`)

- `factory_l0.usda` holds the L0 plant as **one** static triangle-mesh collider (`/World/Factory/L0`, 2088 triangles).
  Box heights per class are set in `world.HEIGHT`; every class is taller than the lidar plane.
- `mesh.npz` is the same mesh for the Warp lidar (`wp.mesh_query_ray`). It is cross-checked at boot against an
  analytic box cast: maximum difference 4e-6 m.
- `map.pgm` + `map.yaml` are the Nav2 static map: 0.1 m, origin (-2, -2), from `intercycle.sim.geometry.StaticMap`.
- `boxes.json` holds the same bodies as upright boxes.

## Nav2 configuration

- The gate config lives in `config/nav2_gate.yaml`: SmacPlanner2D, MPPI `motion_model: Omni` (batch 500), AMCL
  `OmniMotionModel` with alphas 0.3, obstacle + inflation costmaps, and an ANYmal footprint of about 1.0 × 0.64 m.
  This is a basic config only.
- **R2 owns the real one.** Put it in `ros2/intercycle_nav/config/...` and select it with `cfg["nav2_params"]`
  (an absolute container path) or `--nav2-params intercycle_nav/config/nav2.yaml` (relative to `ros2/`).
- The whole repo `ros2/` tree is mounted at `/root/ros2` at container start, so editing it needs no image rebuild.
  Every package dir under it is on `PYTHONPATH` for both the ROS side and the Isaac side.
- Start your nodes with `cfg["extra_launch"]`. It takes a list of shell commands, run in the Humble env with
  cwd `/root/ros2`. The env adds `IC_WORK` (temp dir), `IC_WORLD` (world files: `map.yaml`, `boxes.json`, ...) and
  `IC_OUT` (episode output dir on the Volume). Example:
  `["ros2 launch /root/ros2/intercycle_nav/launch/observer.launch.py use_sim_time:=True"]` or
  `["python3 -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"]`.
- **`cfg["nav2_cmd"]`** (a shell string, or a list with `$VAR` expansion) **replaces** the stock
  `ros2 launch nav2_bringup bringup_launch.py ...`. Use it, for example, to chain velocity_smoother -> collision_monitor
  -> `/cmd_vel`:
  `"ros2 launch /root/ros2/intercycle_nav/launch/nav.launch.py map:=$IC_WORLD/map.yaml use_sim_time:=True ..."`.
  When it is unset, the default above is used. Whatever it launches must end in `/cmd_vel` (geometry_msgs/Twist), which
  is the only command the bridge reads.
- **`cfg["goal_cmd"]`** (same forms) **replaces** `send_goal.py`. The episode ends when this process exits, after a
  2 s settle. It must set the initial pose (the dock, `cfg["start"]`) itself and write `$IC_WORK/goal.json` in the
  `send_goal.py` shape: `{"goal", "init", "result": "SUCCEEDED|FAILED|CANCELED", "time_to_goal_s", "t_goal_sent",
  "t_done", ...}`. Extra keys are kept, so a multi-target mission can add e.g. `"targets": [...]`.
- ROS-side python: in every ROS-env command, `python3` is the system **3.10** (`/usr/bin` comes first on PATH).
  `/usr/bin/python3` also has `pynacl==1.5.0` and `pydantic 2`. The repo's `intercycle/` and `scenario/` are mounted
  at `/root/repo` and are on the ROS `PYTHONPATH` (verified: `import intercycle.codec` and
  `intercycle.scenario.load("/root/repo/scenario/bodyshop.yaml")`). R2's nodes must not import `intercycle.truth` or
  `intercycle.sim`.
- Extra cfg keys from the CLI: `--cfg-json '{"nav2_cmd": "...", "goal_cmd": [...]}'` or `--cfg-json @file.json`.
  This was tested on CPU with the test double: `nav2_cmd`, `goal_cmd` and `extra_launch` all ran, and the goal
  SUCCEEDED.

## Movers (R3): kinematic bodies that Isaac collides with and the lidar sees

- Per step the runner calls `source.poses(t) -> {id: (x, y, psi)}` (the hook). Every mover part is written to a
  KINEMATIC rigid body (`/World/Movers/m_XX`, `IsaacBackend.set_movers`).
- The same primitives are hit analytically by the lidar and by the camera model. No mover ever yields in R1 code;
  behaviour is R3's job.
- Shapes per class (`movers.PARTS`): person cylinder r 0.3 × 1.75 m, agv, tugger (tractor + 3 carts), forklift
  (body + forks with load), weld_arm_active/idle.
- Ways to supply movers, in the order `movers.load_source` checks them:
  1. `cfg["mover_source"] = "intercycle_eval.movers:make"`: a factory `make(cfg)` returning an object with
     `spec() -> [{"id", "cls"}]` (a fixed set per episode) and `poses(t)`. It runs inside the **Isaac python 3.11**,
     which has numpy/torch but **not** scipy/pydantic, so keep it light.
  2. **Recommended:** `cfg["mover_tracks"] = "movers.npz"`. These are precomputed trajectories, added to the `world`
     dict you pass to the Modal function. Keys: `ids` (M,), `cls` (M,), `t` (T,), `xy_psi` (T, M, 3) in the ROS frame;
     a NaN row means absent/parked. They are interpolated linearly in time.
  3. `cfg["movers"] = [{"id", "cls", "x", "y", "psi", "v"}]`: constant-velocity scripted movers (smoke tests).
- Contact events go to `contacts.json` in the episode dir: `{t, contacts: [{body, force_n}], x, y,
  nearest_mover, nearest_mover_gap_m}`. They are also published on `/intercycle/contacts`.
  Contacts in the first 0.5 s (spawn settle) are ignored.
- The evaluation-only truth clearance `min_mover_gap_m` is in `sim_summary.json`.
- Both behaviours were verified on an L4 (17 Oct):
  - `--movers demo`: Nav2 walked past a standing person and a parked AGV (goal SUCCEEDED, min truth gap 0.31 m,
    0 contacts).
  - `--action cmdvel --movers bump`: the robot walked blindly into a parked AGV. Its base contact was logged at
    t = 8.18 s (`nearest_mover` AGV1, gap -0.03 m); the robot was stopped by the body and did not fall.
- Camera model (stub for the inspection/no-motion checks): `sensors.CameraModel.observe(t, cam_xyz, yaw, pitch,
  targets, prims)` returns frustum, occlusion (Warp static + mover primitives), range, incidence, px/m and a detection
  with noise and latency.

## Modal entry points (`cloud/modal_isaac_robot.py`, app `intercycle-isaac-robot`, Volume `intercycle-robot`)

```
modal run cloud/modal_isaac_robot.py --action check                       # CPU: image contents
modal run cloud/modal_isaac_robot.py --action fake --goal 12,16           # CPU: kinematic TEST DOUBLE + Nav2 (wiring only)
modal run cloud/modal_isaac_robot.py --action cmdvel --seconds 20 --vx 0.6
modal run cloud/modal_isaac_robot.py --action nav --goal 12,16 [--movers demo] [--nav2-params intercycle_nav/config/x.yaml]
modal run cloud/modal_isaac_robot.py --action nav --patrol-s 900 --movers demo   # 15-min patrol (RTF / load test)
modal run cloud/modal_isaac_robot.py --action pull --tag NAME             # Volume episodes/NAME -> out/robot/NAME
```

From your own Modal code, look the functions up by name:
`modal.Function.from_name("intercycle-isaac-robot", "isaac_episode_l4")` (needs `modal deploy`), or import the module
and call `.remote` / `.starmap` inside its app.

- `isaac_episode_l4(cfg: dict, world: dict[str, bytes]) -> dict`: L4, 8 CPU, 24 GiB, timeout 1 h
- `isaac_episode_l40s(cfg, world) -> dict`: fallback
- `fake_episode(cfg, world) -> dict`: CPU test double
- `world = build_world_bytes(empty=False)`: run locally; add `"movers.npz": bytes` for R3

`cfg` keys:

| Key | Meaning |
|---|---|
| `tag` | Unique episode name |
| `mode` | `"nav"` or `"cmdvel"` |
| `start` | `[x, y, psi]` |
| `goal` | `[x, y, psi]`, or a list of them (a patrol, repeated until `patrol_until_s` sim seconds) |
| `timeout_s` | Goal timeout in sim seconds |
| `seconds` | Sim-time limit |
| `nav2` | Bool |
| `nav2_params` | Params file path |
| `extra_launch` | List of shell commands |
| `nav2_cmd` | Replaces the stock Nav2 bringup |
| `goal_cmd` | Replaces `send_goal.py`; must write `$IC_WORK/goal.json` |
| `max_rtf` | Pacing limit (default 1.0) |
| `seed` | Sensor noise seed |
| `mover_source` / `mover_tracks` / `movers` | See "Movers" |
| `boot_timeout_s` | Boot wait limit |

Each episode writes to the Volume `/vol/episodes/<tag>/`:
- `episode.json`: cfg, phases, sim summary, goal result.
- `sim_summary.json`: RTF, boot, displacement, path, falls, contacts, min gaps.
- `truth_log.parquet|json`: 10 Hz true pose + cmd + odom, EVALUATION ONLY.
- `contacts.json`.
- `*.log`: one per process.

## Measured throughput (Modal L4, 8 CPU, 24 GiB; 17 Oct)

| Run | Result | RTF | Container wall |
|---|---|---|---|
| Constant `/cmd_vel` 0.6 m/s, empty scene | 11.3 m in 20 s | 0.78 | about 100 s |
| One Nav2 goal | SUCCEEDED in 37 s of sim time | 1.0 (paced) | about 90 s |
| 15-min patrol (`patrol15_l4_1`), demo movers | 20/20 legs SUCCEEDED, 0 recoveries, 483 m walked, 0 contacts, no fall, min truth gap to a mover 0.16 m | 0.67 (0.79 counting physics steps only) | 23.6 min |

Boot time is 21–25 s. A 15-min episode therefore costs about 24 container-minutes. At about $1.37/h
(L4 $0.80 + 8 CPU + 24 GiB) that is about $0.54 per episode.

## Limits (state these in the README/ROBOT_RESULTS)

- 2D lidar only. Bodies below the lidar plane are not seen.
- Movers are kinematic.
- No RTX sensors on Modal. RGB evidence is rendered offline on RunPod from the logged poses.
- The policy observes true base velocity (see the disclosure above).
- `fake_sim.py` is a test double and never a result.

## R3 additions (plant hooks, camera topics; maintained by R3 since R1 finished)

- **Plant hooks.** A `mover_source` may also implement `attach(caster, dt)`, `sensor_prims(t)`,
  `after_step(t, state, contacts)`, `frame_extras(t)`, `on_ros(t, events)` and `finish(out_dir, summary)`.
  `intercycle_eval.plant:make` implements them all:
  - `sensor_prims(t)`: ALL moving bodies for the lidar / camera (every step);
  - `after_step`: the plant's dynamics and light-curtain physics;
  - `frame_extras`: camera observations / capture outcomes for the bridge;
  - `on_ros`: the planner's `/intercycle/inspection` events, forwarded by the bridge in its reply.
- `cfg["mover_every"]` (default 5): kinematic pose targets are written to PhysX every N policy steps (10 Hz). The
  sensors still see every step analytically.
- **Bridge topics.**
  - `/intercycle/camera/observations`: the robot camera's no-motion check (R3 camera model; R2's gate contract).
  - `/intercycle/inspection/capture`: capture outcome, with no pose.
- **Sim-side python** (Isaac 3.11) has pydantic, pyyaml, pynacl 1.5.0 and astropy 6.1.7 (numpy 1.26.0 and torch 2.7.0
  unchanged). `sim_env` puts `/root/repo` on its PYTHONPATH. The ROS env exports `IC_SIM_PY` / `IC_SIM_PYTHONPATH` for
  episode-side helpers (R3's observer process).
- `episode.json` carries `plant` (plant summary) and `eval_check` (the cheat guard).
- `sim_summary.json` carries `wall_profile_s` (set_movers / sensors / exchange).
