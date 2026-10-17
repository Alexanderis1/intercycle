> **Partly outdated (17 Oct 14:40).** Isaac Sim is a **no-go** (it segfaults on Modal and needs NVIDIA RTX). Ignore §1 recommendation, §3 gate and every Isaac-only step. The Kenney low-poly car is dropped. **Still valid:** the asset and licence research (§4), the trace contract (§5), cost (§7) and the shot list (§8). The current plan is `docs/DIRECTION.md` §7 X; the pipeline that implements it is **§0 below**.

## 0. Workstream X: realistic renders on the user's desktop (current, 17 Oct evening)

> **Status for the lead (17 Oct 18:50).**
> - **Done (pushed):**
>   - line-side extras;
>   - cctv_1 sequence + post (no extras):
>     - frames `D:\icdata\renders\seq_sim1\cctv_1\frames`, GT `...\cctv_1\gt`;
>     - CCTV frames + COCO + mp4 `...\cctv_1\cctv`;
>     - preview `render/stills/seq_sim1/cam02_preview.mp4`;
>   - film / wall code;
>   - wording audit;
>   - A1 code (36d1d2c): arms follow the sim's kinematics, tip dressers at the dress tip, fast EEVEE clip mode, pixel features + scoring.
> - **In flight:** the A1 clip.
>   - Settings: cctv_1, seed 3, 10:00-10:45, 640x360, **2 fps** (user's choice: 6-10 fps would take ~8 h of GPU), EEVEE, 4 samples, no shadows (EEVEE's shadow buffer overflows with 136 lights).
>   - Output: `D:\icdata\renders\a1`, ETA ~21:20. A1 status: `docs/PIXELS.md`.
> - **Next:**
>   1. A1: post, extract, score, PIXELS.md;
>   2. cctv_4 sequence + post;
>   3. wall tiles + wall;
>   4. the 3 cinematic shots.
>   - GPU window for V around 00:00-02:00 when asked.

Machine: user's desktop, standalone Blender 5.1 (Python 3.13), Cycles + **OptiX** on the RTX 4060 Ti 8 GB. No Modal spend.
Large files never go to git: assets in `D:\icdata\assets\`, renders in `D:\icdata\renders\`, extra Python packages in `D:\icdata\pylib\`.

| Piece | File | What |
|---|---|---|
| launcher | `render/blender/bl_run.py` | `blender -b --factory-startup --python render/blender/bl_run.py -- <script> [args]`; adds the package folder (`INTERCYCLE_PYLIB`) |
| plant | `render/blender/plant.py` | hall shell, roof trusses + skylight strips, 128 LED high-bays, column grid between the sim zones, aisle markings from `zones`, overhead trays/ducts/sprinklers, clutter, dressing only on floor no sim zone uses |
| scene | `render/blender/scene_real.py` | `RealScene(BScene)`: stations (roller-skid conveyor, fixtures, mesh guards with light-curtain openings, stack lights from machine state, signs, controller platform), FANUC arms, spark bursts, line-side extras (`extras.py`); stills, CCTV sequences with motion blur, ground truth |
| robot arm | `render/blender/fanuc.py` | FANUC R-2000iC/210F from the NVIDIA USD (CC-BY 4.0) as a joint chain a1..a6 + servo C-gun; `weld_q(phase, k)` |
| CCTV look | `render/cctv/` | distortion, vignetting, sensor noise, burned-in timestamp + **SYNTHETIC**, real x264 round trip; GT warped by the same map -> boxes + COCO (`pytest render/cctv/test_cctv.py`) |
| assets | `render/blender/fetch_assets.py` | NVIDIA bucket downloader that records each asset's licence file (`manifest.json`) |
| props + BIW | `render/blender/assets.py` | NVIDIA SimReady props (CC-BY 4.0) -> `D:\icdata\assets\library\nvidia_props.blend`; the `BIW` collection (SimReady car frame + closures on a skid) replaces the placeholder bodies, `forklift_c` the box forklift; writes `render/credits.json` |
| materials | `render/blender/materials.py` | CC0 PBR (Poly Haven, ambientCG) with box projection; `plant(key, coords="world")` for the scaled plant boxes. `plant.PBR_KEYS` says which plant keys use it |
| workers | `render/blender/humans.py` | rigged MakeHuman (CC0) workers in PPE, walk/idle/work clips, `D:\icdata\assets\library\workers.blend` |
| acceptance | `render/side_by_side.py` | real body-shop photo (credit from `reference.json`) next to a render, both labelled |

What the realistic look deliberately does **not** show: glowing hazard-zone decals and curtain planes (map overlays, not
visible to a camera), capsule people, the wheeled AMR (robot shots are R4's).

Commands (repo root, PowerShell; `$B` = `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`):
```powershell
& $B -b --factory-startup --python render\blender\bl_run.py -- render\blender\scene_real.py --trace <trace.parquet> --synth-bodies `
    --out D:\icdata\renders\stills --stills compare_line,wide,weld_close,cctv_1,cctv_5 --t 36562.8167 --samples 128 --res 1920x1080 --gt
& $B ... scene_real.py --trace <trace> --synth-bodies --out D:\icdata\renders\seq --cctv-seq cctv_1 --t 36560 --seconds 10 --fps 25 --samples 64 --gt
$env:PYTHONPATH="D:\icdata\pylib"; python -m render.cctv.cctv_post --in D:\icdata\renders\seq\cctv_1\frames --gt D:\icdata\renders\seq\cctv_1\gt `
    --out D:\icdata\renders\seq\cctv_1\cctv --cam CAM02 --cam-label "BODY SHOP S20" --t0 "2026-10-17 10:10:20" --fps 25 --overscan 0.055 --video
```
`cctv_<i>` is camera i of `bodyshop.yaml` `cameras` at 9 m, ~91° horizontal; CCTV renders carry 5.5 % overscan per side
(`--overscan`, pass the same value to `cctv_post`). Ground truth per frame: `<stem>_inst.png` (uint16 object index, 0 =
background/static plant), `<stem>_depth.npy` (m), `<stem>_ids.json` (`{id: {cls, entity_id}}`, plus `extra: true` for
line-side extras; classes: robot_arm, body, forklift, agv, tugger, person). Speed: ~5-10 s per 1080p frame at 64 spp.

First stills (17 Oct, `render/stills/first/`, trace `trace_synth15.parquet`): `--t 36562.95` is a frame where the near
middle robot of S30 welds, so the `weld_close` shot shows sparks (spatter leaves the gun on the robot side). Side-by-sides
against the Jurvetson photos (CC BY 2.0): `compare_*.jpg`.
Brightness + density pass (`render/stills/dense/`, `--t 36562.8167`): light-grey epoxy floor and columns, LED high-bays
150 W each (`--light-e`), 8 robots per 8 m weld/respot cell (4 corner, 2 middle, 4 angled onto the body ends; the 7 m
cell gets 8 without the middle pair), a tip dresser beside every robot, and an overhead cell frame (4.4 m) with cable
trays and a dress-cable drop to each robot. All of it stays inside the station rects of `bodyshop.yaml`.
`compare_line` (still name) is the reference photo's viewpoint: 7 m above the line axis at x 15.5, looking down +x, 16.5 mm
(~95 deg); `compare_wide.jpg` uses it.
Detail pass: servo C-gun v2 (two C side plates, ribbed transformer, servo actuator) and a dress pack (retract unit, main
hose, blue/red cooling water) on every robot; fixtures with locator blocks, pins, toggle clamps and gussets;
`RealScene.rounded()` rounds edges in shading (Cycles Bevel node, radius in metres; the first render after a fresh install
compiles the OptiX kernel once, ~2 min); closures carts with door panels at each station's line side; `cctv_post`
`sensor_scale` (per camera 0.65-0.85 of 1080p, `--sensor-scale`) for CCTV sensor softness.

Trace for the CCTV sequences: `python render/synth_trace.py --movers sim --minutes 15 --out D:\icdata\renders\trace_sim15.parquet`
(repo Python). People and vehicles are the sim's own movers (`intercycle/truth/agents.py`): at 10:00-10:15 that is 12
people (10 aisle walkers + 2 data-hall techs; break crowds only around breaks), 4 AGVs, 2 tuggers, 3 forklifts, sampled
every 0.25 s on their unobstructed paths. Machines and the hero AMR stay synthetic. Note: `bodyshop.yaml`
`people_per_shift` (early 34) is only used by the statistical occupancy model `truth/traffic.py`; `agents.py` does not
spawn that many, so the render shows the sim's 12, not 34.

Line-side extras (`render/blender/extras.py`, `--extras`, default 2): rendered workers who are **not** in the trace, so a
view can show line-side operators next to the sim's 12. Rules:
- only outside the guarding: an operator at a light-curtain opening of a loaded on-line cell (weld / framing / respot;
  south side with a walk to the part cart and back, north side with a short walk along the guard), an inspector on the
  S80 south side, a maintenance tech at the north opening of a cell that is STOPPED / HELD / TRIPPED on every frame of
  the clip. Never inside a cell, active or not;
- per view: at most `--extras` (2) and never more than the trace persons inside the nominal frame, taking the fewest over
  the clip, so the cap holds on every frame. Chosen once per clip: the work spot must be in frame at 4-30 m, keep 1.2 m
  from every trace mover on every frame and 1.5 m from the other extra, one extra per cell, maintenance first, then nearest;
- mostly standing / working (work and idle clips) with a short walk;
- ground truth: class `person`, `entity_id` `extra_<n>`, `"extra": true` in `<stem>_ids.json`, in `<stem>_boxes.json`
  and in the COCO annotations, so they can be filtered out of anything that compares against the trace.
Rules are tested in `render/blender/test_extras.py`. `--extras 0` renders the trace persons only.
Second V sequence: `--cctv-seq cctv_4 --t 36245` (north ceiling camera at (6, 46)). It shows weld cells S10, S20 and S30, all
ACTIVE, and forklift FL0 on the north aisle for the whole 10 s.

Final video shots (build lead, 17 Oct). There is no robot/AMR in any of them.
- **Cinematic shots:** 1080p, 24 fps, 64 spp, motion blur on the moving camera too, no extras.
  - Every frame is labelled "SYNTHETIC REPLAY OF OUR SIMULATION". The paths are in `scene_real.film_cam`.
  - Output goes to `<out>/<shot>/frames/` and `<out>/<shot>.mp4`.
  - Commands:
```powershell
& $B ... scene_real.py --trace <trace> --synth-bodies --out D:\icdata\renders\film --film establishing --t 36560 --seconds 8 --fps 24 --samples 64 --res 1920x1080
#   --film flyalong --seconds 12  (3.5 m over aisle S, x 10 -> 58, looking into the cells)
#   --film weldclose --seconds 6  (push-in through the S30 north opening onto the weld_close framing, 28 -> 35 mm, sparks)
```
- **CCTV wall:** one shared clock, 36560-36570 at 25 fps.
  - Tiles: `cctv_0`, `cctv_1`, `cctv_5`, `cctv_6`.
  - `cctv_1` is the 1080p V sequence, downscaled. The other three are rendered at 960x540.
  - The 960x540 tiles are post-processed with `--size 960x540 --sensor-scale 1.0`, because the half-size render already does
    the softening. Their extras follow the per-view rule.
  - `python -m render.cctv.wall --tiles <TL> <TR> <BL> <BR> --out D:\icdata\renders\wall\out` tiles them 2x2, keeps each
    tile's burned-in text, and adds one label.

# Path-traced render tier (synthetic replay of our simulation): decision, research, shot list

*Written 2026-10-17 12:40 CEST for the 18 Oct 16:00 submission. Owner: render pipeline. Everything here is
**replay of `trace.parquet`**; the render tier never simulates and never feeds back into the sim. Every frame is a
synthetic replay of our simulation and must be captioned that way.*

## 1. Recommendation

1. **Blender/Cycles on Modal is the primary path.** It is already proven end to end on a synthetic trace (local `bpy 4.5.14`,
   5 shots -> PNG -> mp4 -> A/B split, frames inspected) and carries no third-party licence risk (all geometry procedural).
2. **Isaac Sim 5.1 on Modal L40S is the stretch**, gated at **17:00 today: one rendered frame of a stock scene saved by
   `modal_app.py::smoke`, otherwise stop and stay on Blender.** Isaac buys real Fanuc/KUKA robots, Nova Carter, forklifts and the NVIDIA
   data-hall racks; it costs a 10-25 min image build, a cold shader compile, and an untested Kit half of `replay.py`.
3. **Modal is not authenticated on this machine** (`%USERPROFILE%\.modal.toml` absent, no `MODAL_TOKEN_*`), so neither smoke test has
   run on Modal. The user must run `C:\AMD\intercycle\.venv\Scripts\modal.exe setup` once (opens a browser).

## 2. Decision table

| Criterion | Isaac Sim 5.1 (RTX, Kit) | Blender 4.5 LTS (Cycles, OptiX) |
|---|---|---|
| Status today | smoke app written + `modal_app.py` imports cleanly; Kit half of `replay.py` **untested** | **ran end to end locally** (CUDA, MX250); OptiX not yet seen |
| Time to first frame on Modal | image 10-25 min (isaacsim[all,extscache] is >10 GB) + 5-10 min cold shaders, then cached in Volume `intercycle-isaac-cache` | image ~3-5 min (373 MB wheel), first frame < 2 min |
| Real industrial assets | Fanuc R-2000iC spot-welders, KUKA KR210, Nova Carter, iw.hub, forklifts, NVIDIA rack/CRAH/data-hall stage | procedural (KR210-proportioned arms, racks, forklift, people capsules); optional CC0/CC-BY drop-ins |
| Arm animation | articulation joint teleport + 1 physics tick per frame; **base-pose behaviour must be verified**, fallback = home pose + weld sparks | hierarchical joint empties, deterministic, no physics |
| Look | real-time RTX 'fast' or path-traced 'final'; materials from NVIDIA MDL | Cycles path tracing + OptiX denoise; materials are ours (plain) |
| Licence exposure for a public MIT repo | CC-BY credits + NVIDIA licence reading (section 4) | none beyond CC0 HDRI (and any optional CC-BY asset) |
| Cost per 2 min cut (3 600 frames) | about $3 'fast' / about $19 fully path-traced (hero shots only: about $5) + debug runs | about $4-8 (L4 or L40S) |
| Failure modes | driver/Vulkan on Modal, 6.0-bucket USD not opening in 5.1, EULA/auth, articulation base snap | OptiX libs missing (falls back to CUDA/CPU), art direction |
| Pitch value | "NVIDIA stack, real robot models" | "path-traced, fully reproducible, zero licence risk" |

## 3. Go / no-go gate

| When | Gate | If it fails |
|---|---|---|
| now | `modal setup` done (user) | no Modal => local Blender only (works today, slow on the MX250) |
| **<= 17:00 today** | `modal run render/isaac/modal_app.py::smoke` writes `render/out/smoke.png` (non-blank, warehouse visible). Cap about $3 / 35 min: if image build or first render exceeds ~35 min, stop and report | **Blender only**, no further Isaac work |
| in parallel, now | `modal run render/blender/modal_blender.py::smoke` prints `Cycles device: OPTIX` and writes `render/out/bpy_smoke.png` | accept CUDA; CPU is not acceptable for finals |
| 17:00-21:00 | (only if Isaac passed) first `replay` frames: arms stay on their bases, units sane, one shot at `--quality fast` | drop Isaac, finish on Blender |
| by 12:30 tomorrow | all hero shots rendered (code freeze) | ship whatever exists; the dashboard replay remains the fallback visual |

## 4. Research: what exists, where, under which licence

Method note: the NVIDIA asset bucket is **anonymously listable** (`https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&prefix=...`),
so the paths below were taken from the bucket listing on 2026-10-17, not from docs. Two roots matter:
`.../Assets/Isaac/5.1/Isaac/` (what the 5.1.0 pip build resolves by default) and `.../Assets/Isaac/6.0/Isaac/` (also 6.1, 6.2 exist;
6.0 adds `SimReady/`, ~90 Fanuc models, and per-asset `LICENSE` files). `isaacsim` on PyPI: 5.1.0.0 needs Python 3.11; 6.0.0.0 through 6.2.0.0
(13 Oct 2026) need Python 3.12; the 5.1 docs now label 5.1 "unsupported". Paths below are relative to the root shown.

| Need | Exact path | Licence (where seen) | Used by |
|---|---|---|---|
| **Spot-welding arms** | 6.0: `Robots/Fanuc/r2000ic_210f/r2000ic_210f.usd`, `.../r2000ic_165f/r2000ic_165f.usd` (+ `r2000ic_210l`, `r2000ic_270f`, `r2000id_*`, `m710ic_*`, `m900ib*`, `m-2000ia`, `lrmate*`, `crx*`) | **CC-BY-4.0** (`LICENSE` next to the asset). 5.1 docs list FANUC under "3D Content Sharing Agreement"; 5.1 bucket has only `Robots/Fanuc/CRX10IAL/crx10ial.usd` | Isaac `arm_weld`, `arm_respot` |
| KUKA | 5.1+6.0: `Robots/Kuka/KR210_L150/kr210_l150.usd`; 6.0: `Robots/Kuka/LBR_iiwa/` | **Apache-2.0** (ros-industrial/kuka_experimental) | Isaac `arm_kuka` |
| Universal Robots | `Robots/UniversalRobots/{ur3,ur3e,ur5,ur5e,ur10,ur10e,ur16e,ur20,ur30}/<name>.usd` | BSD-3 (`LICENSE` per asset) | not used (cobots) |
| **ABB** | **none** in 5.1 or 6.0 (`RobotStudio/` holds SO-100/101 hobby arms, not ABB); Yaskawa `Robots/Yaskawa/Motoman Next`, Kawasaki `RS007L/N, RS013N, RS025N, RS080N`, Comau (6.0 `comau/`), Denso exist | Yaskawa/Denso: "3D Content Sharing Agreement" (docs) | not used |
| **Conveyors** | `Props/Conveyors/ConveyorBelt_A01..A42.usd` (4-110 MB each); SimReady 6.0: `SimReady/Industrial/Warehouse/Equipment/Conveyor_Belt_{A02,A35,A61}/sm_conveyor_belt_a35_01.usd` | **CC-BY-4.0** (6.0 `LICENSE`) | market feed (optional) |
| **Car bodies** | **no full vehicle / BIW.** 6.0 `SimReady/Industrial/Machines/Automotive/Body_Parts/*` (doors, hood, fenders, roof panel, bumpers), `.../Frames/Car_Frame_Car_Frame_Assembled_A01/sm_carFrame_car_frame_assembled_a01_01.usd`, `.../Wheels/*` | **CC-BY-4.0** (per-asset `LICENSE.txt`) | alternatives in the next table |
| **Environments** | `Environments/Simple_Warehouse/{warehouse,warehouse_with_forklifts,warehouse_multiple_shelves,full_warehouse}.usd`, `Digital_Twin_Warehouse/small_warehouse_digital_twin.usd`, `Modular_Warehouse/`; 6.0 adds `Modular_Warehouse_New`, `warehouse_20x20_envi`. No factory/body shop | **Limited Use Content**: "for your use only, **without modifications**" (`environment-supplement-LICENSE.txt`, 2021). Fine for the smoke test; do not build the final scene from it | smoke test only |
| Structural/safety bits | 6.0 `SimReady/Industrial/Warehouse/Barriers/{Machine_Guard_A01,A02, Wire_Machine_Guard_A01-A04, Barrier_Fence_Metal_A02,A03, Barrier_Gate_Safety_A02, Environment_Fence_A01-C01}/sm_*.usd`, `.../Racks/*`, `.../Signs/*`, `.../Structures/*` (walls, columns, beams) | **CC-BY-4.0** (per-asset) | Isaac `guard` |
| **Light curtains** | **none stock.** Built procedurally: two yellow posts + translucent emissive plane | ours | both |
| **Nova Carter** | `Robots/NVIDIA/NovaCarter/nova_carter.usd` (+ `NovaCarterDevKit/` in 6.0, `Props/NVIDIA/Carter/ChargingDock/`) | **CC-BY-4.0** (6.0) | hero AMR base |
| **iw.hub** | `Robots/Idealworks/iwhub/iw_hub.usd`, `iw_hub_sensors.usd`, `iw_hub_static.usd` | **CC-BY-4.0** (6.0) | AGV |
| **Forklifts** | `Props/Forklift/forklift.usd`; `Robots/IsaacSim/ForkliftB/forklift_b.usd`, `ForkliftC/forklift_c.usd`; 6.0 SimReady `Warehouse/Equipment/Vehicle_Forklift_{Blue_A01,Blue_C01,Red_B01,Red_C01}/sm_vehicle_forklift_*_01.usd`; pallet jacks too | **CC-BY-4.0** (6.0) | forklift |
| **People (Replicator Agent)** | `isaacsim.replicator.agent` (default `asset_path` = `Isaac/People/Characters/`). Characters: `People/Characters/original_male_adult_construction_{01,02,03,05}/male_adult_construction_NN.usd`, `original_male_adult_police_04`, `original_female_adult_{police_01,02,03, business_02, medical_01}`, `original_male_adult_medical_01`, `*_new` variants, `biped_demo/biped_demo_meters.usd`, `DH_Characters/<uuid>/`. Animations: `People/Animations/stand_walk_loop.skelanim.usd` (+ `stand_walk_1..7`, `stand_idle_loop`, `push_button`, `type_keyboard`, `Sit`, mirrored variants) | **no licence file in the bucket**; falls under the Isaac Sim Additional Software and Materials Licence | people (slide) |
| **Server racks / data hall** | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Datacenter/`: `Racks/Rack_42U_A/Rack_42U_A_01.usd`, `Server_Nodes/Servers/Server_{1U_A,2U_B,2U_C}_01.usd`, `Server_Nodes/DGX/DGX_A100_01.usd`, `Network_Switches/NVIDIA/*`, `Liquid_Cooling/Data_Hall/DCP_A/DCP_A_01.usd`, `Facilities/Cable_Tray/*`, **ready-made stage** `Facilities/Stages/Data_Hall/DataHall_Full_01.usd` (+ `_Full_B01`, `_NoRacks_01`, `_Rack_42U_01`). No CRAH asset (we draw boxes) | **no licence file**; Omniverse content terms | racks (Isaac) |
| DSX blueprint (AI-factory) | NGC resource `dsx_dataset` (GB300 NVL72 rack, Vertiv XDU2300 CDU), https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/dsx_dataset | NVIDIA Software License Agreement + Omniverse product terms, NGC login; **not CC** | not used |

### Licence reading for a public MIT repo (rendered images/video only, no asset redistribution)

This is a reading of the sources, **not legal advice**; the lead owns the call.

* **Isaac Sim software.** Source on GitHub is Apache-2.0. The pip/Kit binaries and NVIDIA-provided assets fall under the *NVIDIA Isaac Sim Additional
  Software and Materials License*: non-exclusive, revocable, NVIDIA GPUs only, no redistribution, no derivative works of the Software, and section 3
  limits users to employees/contractors of the licensee entity (plus students). The NVIDIA licence FAQ says Isaac Sim is free for internal R&D incl.
  commercial use and that selling only **outputs** (videos, reports, datasets) needs no AI Enterprise licence. Nothing found restricts *publishing*
  outputs, but no sentence explicitly grants it either. We redistribute nothing from Isaac (assets are read by URL at render time; `*.usd` is never
  committed). Accepting the EULA via `ACCEPT_EULA=Y` makes the person who runs the app the licensee.
* **CC-BY-4.0 assets** (Fanuc R-2000iC, Nova Carter, iw.hub, forklifts, conveyors, all `SimReady/`): stills and video are fine with **attribution**. Credit line:
  "Robot, vehicle and prop models: NVIDIA Corporation, Isaac Sim / SimReady assets, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)". The Isaac
  script writes `credits.json` with what it actually loaded. Do not imply FANUC/KUKA/NVIDIA endorsement (trademarks).
* **KUKA** Apache-2.0, **UR** BSD-3: credit in the README.
* **Limited Use Content** (`Isaac/Environments/*`): not for the final scene.
* **No licence file** (People, DigitalTwin Datacenter, Props/Dolly, Props/Pallet): covered by Omniverse content terms. Older Omniverse agreements allow distributing
  user-generated renders/videos/screen captures but not the content standalone; the current text could not be fetched (HTTP 403). If you want zero ambiguity, use the
  Blender path (procedural) for any shot that shows these.
* Every element is **synthetic** (machines, schedules, traffic, temperatures); the video must say so.

### Gaps and CC0 / CC-BY substitutes

| Gap | Substitute | Licence | URL | Path to USD |
|---|---|---|---|---|
| Car body-in-white | **Kenney Car Kit** (45 low-poly cars) | CC0 | https://kenney.nl/assets/car-kit (zip https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip) | strip wheels/glass, paint bare steel (`replay_bpy.py --body-glb`) |
| | "Car Body" B L A V E N, 1.4k tris unfinished shell | CC-BY | https://sketchfab.com/3d-models/car-body-f11ff10659cd4282b3a1535288b86bfe | download glTF |
| | Poly Haven "Covered Car" | CC0 | https://polyhaven.com/a/covered_car | not a BIW |
| | assemble NVIDIA SimReady `Car_Frame_Car_Frame_Assembled_A01` + roof/door/fender parts | CC-BY-4.0 | bucket paths above | already USD |
| Server racks (if the NVIDIA ones are unwanted) | "Server Rack" Anais3Dcraft 1.1k tris; Spellkaze 744 tris; Khoa Nguyen "Network Server Rack" | CC-BY | https://sketchfab.com/3d-models/server-rack-1348f8c5313f4059b9a2b5b3f37c81f3 , https://sketchfab.com/3d-models/server-rack-62f6779cb7e448b19aaf58544c3c7218 , https://sketchfab.com/3d-models/network-server-rack-3b15fb8779274844b54e63a40194d450 | glTF |
| | Poly Haven | CC0 | has no server rack (only `utility_box_01`); Kenney/NASA none | - |
| Ambient light | Poly Haven HDRIs `autoshop_01`, `aircraft_workshop_01`, `aerodynamics_workshop` | CC0 | https://polyhaven.com/a/autoshop_01 | `UsdLux.DomeLight` texture or Blender world |
| Arm meshes (non-NVIDIA) | KUKA KR210 L150 `kuka_kr210_support/meshes/kr210l150/visual/*.dae` | Apache-2.0 | https://github.com/ros-industrial/kuka_experimental | Blender import |

**Conversion to USD** (any of): (1) Isaac Sim's asset converter (`omni.kit.asset_converter`, File > Import) for glTF/GLB/FBX/OBJ; (2) headless Blender 4.5 (already in our
container): `bpy.ops.import_scene.gltf(filepath=in); bpy.ops.wm.usd_export(filepath=out)`; (3) `usd-core` for composition afterwards. After import, check units (cm vs m):
`replay.py` auto-fits assets to a real-world length/height, so a wrong scale cannot break the layout.

## 5. Trace contract (what the sim must write)

`trace.parquet`: `t` (sim s since 00:00), `entity_id`, `cls`, `x`, `z` (m, scenario frame, +z north), `yaw` (rad, atan2(dz, dx); NaN = derive from motion), `state`.
Poses are linearly interpolated, states are held. Full text in the docstring of `render/isaac/replay.py`; class aliases and state names are normalised there (edit `CLASS_RULES`).
**Gap to close today:** `intercycle/sim/run.py` currently logs only the robot as `(t, x, z, state)` at graph nodes (`dock/move/enter/exit`). For the shots we also need
(a) one row per **state transition** for every machine in `bodyshop.yaml` (`ACTIVE`/`IDLE`/`STOPPED`/`HELD`/`TRIPPED`, or the `State` ints), (b) the hero robot's `wait`
state, and (c) nice-to-have: people/AGV/tugger/forklift tracks and `body` rows (car bodies). Without (c) the replay uses `--synth-bodies` (a labelled visual guess). Run the
same seed for policies A and B and save two traces (`--label A`, `--label B`) for the split screen. `python render/synth_trace.py` writes a stand-in with the same schema.

## 6. Files

| File | What |
|---|---|
| `render/isaac/modal_app.py` | Modal app (modal 1.x): `nvidia/cuda:12.8.1-devel-ubuntu22.04` + py3.11, requested apt set (+ `vulkan-tools ffmpeg`, `libglu1-mesa`), `isaacsim[all,extscache]==5.1.0`, EULA env, L40S, 3600 s, Volume `intercycle-isaac-cache`. `smoke` entrypoint -> `render/out/smoke.png` (function `smoke_gpu`); `replay` entrypoint |
| `render/isaac/replay.py` | trace contract, tracks, layout, shots, `--dry-run`, ffmpeg, split-screen `compose`; Kit half (stage build, per-frame animation, Replicator `BasicWriter` -> PNG -> mp4) |
| `render/blender/replay_bpy.py`, `modal_blender.py`, `README.md` | Blender fallback (see its README) |
| `render/synth_trace.py` | stand-in trace |

Commands (repo root, PowerShell): `.venv\Scripts\modal.exe setup`, then `.venv\Scripts\modal.exe run render/isaac/modal_app.py::smoke`,
`...render/blender/modal_blender.py::smoke`, `python render/synth_trace.py`, `python render/isaac/replay.py --trace render/out/trace_synth.parquet --dry-run`.

## 7. Cost (Modal pricing 2026-10-17: L4 $0.80/h, L40S $1.95/h, per second; Starter plan includes $30/month credit)

| Item | Estimate |
|---|---|
| Isaac smoke (image build on CPU + 15-25 min L40S incl. cold shaders) | $0.5-1.0; hard cap $3 / 35 min |
| Isaac replay, 'fast' RTX, 1080p, 1-2 s/frame x 3 600 | $2-4 |
| Isaac replay, path-traced 'final' for hero shots only (about 900 frames x 8-15 s) | $4-7 |
| Blender smoke | < $0.2 |
| Blender 3 600 frames, 1080p, 64 spp, OptiX (L4 5-8 s/frame, L40S 2-4 s/frame) | $4-8 either GPU |
| Debug re-runs (budget) | $5-10 |

Per-frame times are estimates; the smoke logs print wall times so the first real run replaces these numbers. Wall-clock is cut by one container per shot, not by a bigger GPU.

## 8. Shot list for the 2:30 pitch cut (all captioned "synthetic replay of our simulation")

| Time | Shot id | On screen | Needs from the trace | Overlay / audio beat |
|---|---|---|---|---|
| 0:00-0:12 | `overview` | crane move NW -> SE inside the hall: 8 stations, red/green hazard decals, arms welding, bodies indexing | machine states, `body` rows (or `--synth-bodies`) | "Inspect between cycles, not between shutdowns." |
| 0:12-0:35 | `ab_cam` (A only) | fixed camera at S40: reactive robot arrives, waits, enters late, line holds | trace A: hero `wait/enter`, S40 states, `HELD`/`TRIPPED` | cost counter ($639/s context line, labelled "Siemens 2024, context only") |
| **0:35-1:05** | **`curtain`** (hero shot) | low tracking shot: AMR **waits at the light curtain** while the welding robots finish; sparks stop, zone turns green, curtain turns green, robot rolls in | hero `wait` then `enter`; zone/curtain colours from machine `ACTIVE -> IDLE` | caption: "local check 0.3 s, then cross"; insert 4 s of `pov` (mast camera) |
| 1:05-1:25 | `ble` | camera beside the AMR in the aisle: ceiling broadcasters pulse every 0.5 s, cyan packets fly to the mast, red ones die mid-air (10 % loss) | hero pose (broadcasters come from the YAML; loss is seeded, visual only) | dashboard panel composited: "Ed25519 verified, TTL 8 s"; "one-way, no pairing, no PLC link" |
| 1:25-1:45 | `datahall` | thermal pass down the hot aisle (3 x 12 racks, CRAH wall, cold-aisle containment), hero inspects a hot rack | hero tracks into the data hall; rack temperatures are synthetic (`rack_temp_c`) | "thermal view - SYNTHETIC temperatures"; "no machine hazard here, only traffic and the door" |
| 1:45-2:15 | `ab_cam` A vs B split | same seed, same camera: left A (waits, holds the line), right B (enters at the start of the predicted idle window) | two traces, same seed, `--label A|B`, then `replay.py compose` | live metrics bar from the eval: line time lost, trips, inspections/hour, gain retained; **entries into an active zone: 0** |
| 2:15-2:30 | end card | title, repo (MIT), credits from `credits.json` (NVIDIA CC-BY, Poly Haven CC0, Kenney CC0 as used), "all data synthetic" | - | licences/attribution |

Frame budget: 150 s x 30 fps = 4 500 output frames, about 3 600 rendered (B-roll and holds reuse frames). Shot windows default to the trace (`win_*` in `replay.py`); override with `--t0/--t1`.

## 9. Known unknowns (ranked)

1. **Isaac Kit half is unrun.** Most likely first-contact fixes (all marked `# VERIFY`): `isaacsim.core.utils.stage.create_new_stage` path, `BasicWriter` file names, `/rtx/rendermode` enum for path tracing,
   Fanuc articulation base staying put under a translated parent (fallback `--arms static`), 6.0-bucket USD opening in the 5.1 runtime (fallback: the 5.1 URLs already listed in `ASSETS`, or bump to `isaacsim==6.0.1.0` + Python 3.12).
2. **Modal Vulkan.** The CUDA base image has no Vulkan ICD; the app writes `nvidia_icd.json` and sets `NVIDIA_DRIVER_CAPABILITIES=all`; if `vulkaninfo --summary` in the smoke log shows no NVIDIA device, Isaac cannot render and the gate fails fast.
3. **Sim trace** lacks machine rows today (section 5).
4. Licence reading (section 4) for People/Datacenter assets is unverified at the source text level.

## 10. Sources

NVIDIA asset bucket listings and per-asset `LICENSE` files (fetched 2026-10-17). Isaac Sim docs: robots https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html ,
manipulators https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots_manipulator.html , environments https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_environments.html ,
asset root / cloud https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html , licence FAQ https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-faq.html ,
Additional Software and Materials License https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-isaac-sim-additional.html . DSX https://github.com/NVIDIA-Omniverse-blueprints/omniverse-dsx-blueprint-for-ai-factories ,
https://docs.omniverse.nvidia.com/dsx/latest/simready-assets.html . SimReady warehouse dataset (CC-BY-4.0, mostly hand-manipulation props; not used) https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01 .
PyPI https://pypi.org/project/isaacsim/ , https://pypi.org/project/bpy/ . Modal pricing https://modal.com/pricing . Modal-on-Isaac template https://github.com/TheFloatingString/isaac-lab-on-modal .
Poly Haven licence https://polyhaven.com/license . Kenney https://kenney.nl/assets/car-kit .
