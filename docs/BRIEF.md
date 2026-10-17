# InterCycle — engineering brief (read this first, then `docs/DIRECTION.md`)

Hackathon: AMD Developer Hackathon ACT III, Track 1 Intelligent Industry. Submission **18 Oct 2026 16:00 CEST**, code freeze 12:30. Public **MIT** repo. Solo developer + Claude.
**Plan, priorities, timeboxes and anti-drift rules live in `docs/DIRECTION.md`.** This brief holds only the engineering facts.

## Product (one paragraph; DIRECTION §1 is authoritative)
The product is the **observer**. Existing ceiling cameras in an automotive body shop (BIW, plus an on-site data hall) learn, with no PLC link and only a one-time site-plan import (zone outlines, camera poses):
- every machine's cycle and the distinct phases inside it;
- the line's production states;
- vehicle and people flows by time of day.

From these the observer builds a **layered map of dynamic collision volumes**. The map converges over hours, days and weeks and adapts when new machinery is installed.
It goes to robots over **two redundant signed channels**: Wi-Fi/Zenoh as primary, BLE 5 advertising as fallback.

A visiting inspection robot plans through the layers to inspect **between machine cycles instead of during shutdowns**.

Positioning:
- an **advisory planning layer, not safety-rated**;
- UNKNOWN or stale space counts as occupied;
- the robot's certified scanner stays the safety function.

Buyer: maintenance/reliability manager. EHS co-signs.

## Safety semantics (design assumptions — say so in docs)
- Machines are **not fenced boxes**. Each machine has a **hazard zone** (robot reach envelope, lifter drop zone, hemming/sealer cell) guarded by **light curtains / area scanners**.
- **ACTIVE zone** = machine moving. A robot crossing the boundary of an ACTIVE zone **trips** the safety system: protective stop, then a line stop with a `recovery_after_trip_s` (90 s) restart. Line stations stop the whole line.
- **IDLE zone:** a robot inside it **inhibits the next start** (hold) until it leaves. Holds on line stations hold the whole line.
- The prediction only changes *when* the robot tries. Before crossing, the robot's own camera confirms "no motion" (`local_check_latency_s`). The safety system is always the final layer.
- Cost framing: Siemens *True Cost of Downtime 2024*, about $2.3M/h for a large automotive plant (≈ $639/s). This is context only, not our measured gain.

## Layered map of dynamics (details: `docs/MAP.md`)
| Layer | Content | Indexed by |
|---|---|---|
| L0 static | walls, conveyor, racks, robot bases, posts (CAD/SLAM) | — |
| L1 machine envelopes | floor swept by moving parts per machine (weld arms incl. tip-dress, skids, lifter, hemming head, sealer) | machine, cycle-phase bin (discovered sub-phases, see DIRECTION §4), production state |
| L2 vehicle flows | per 1 m cell: occupancy probability + dominant heading (8 dirs), AGVs / tuggers / forklifts | time-of-day bin (30 min), production state |
| L3 people flows | same, for people | time-of-day bin, production state |
| UNKNOWN | never observed, camera offline, or a newly detected machine still being learned. Treated as occupied | — |

**Production state** ∈ {running, andon, break, shift_change}. It is **discovered from observations**; the public calendar is used only to validate.

**Broadcast** (the same Ed25519-signed payloads on both channels):
- **slow/map layer:** L1–L3, a signed manifest plus chunks, valid for hours;
- **fast layer:** each machine's phase-in-cycle and the production state;
- **live layer:** tracked movers every 0.5 s.

## Robots compared (same seeds; supporting evidence, not the headline)
- **A0 shutdown-only** (status quo): inspects only during planned breaks and shift changes.
- **A reactive:** waits until its own camera sees the machine stop, then enters. It doesn't know how long the idle lasts, so it causes holds and occasional trips.
- **B InterCycle:** fed by the **learned** observer (not the old ShimObserver). Its zone windows come from L1.
- **C PLC oracle:** knows the true future (upper bound; needs plant integration).

Metrics:
- inspections per production hour;
- line time lost to the robot;
- trips per shift;
- gain retained = (A − B)/(A − C);
- entries into an active zone: must be 0 for B under fault injection;
- These A0/A/B/C numbers are a *scheduling-level* analysis. The robot result is the **legged robot on Nav2** (DIRECTION §5: 4 conditions, Francis et al. metrics).

## Repo layout
```
intercycle/types.py        shared dataclasses          intercycle/scenario.py   YAML loader (scenario/bodyshop.yaml = single source of truth)
intercycle/truth/          ground truth (machines, traffic, agents, kinematics). Learner/codec/robot MUST NOT import it (tests/test_no_truth_import.py)
intercycle/observer/       signals.py (camera shim, may import truth), learn_cycles.py, learn_traffic.py, mapping.py (no truth!)
intercycle/codec/          layout.py (bits), sign.py (Ed25519), ble.py (framing, channel sim, receiver), maplayer.py (map chunks)
intercycle/robot/          sipp.py, sitegraph.py, predict.py, policies.py (A0/A/B/C) — no truth imports
intercycle/sim/            event sim (world.py, run.py, trace.py), 10 Hz physics demo (physics.py, movers.py, geometry.py)
intercycle/perception/     RF-DETR + ByteTrack + ground-plane fusion        intercycle/llm/  Gemma 4 client, tasks, replay cache
intercycle/eval/           seed sweeps (run_seeds.py; held-out seeds 1000-1199 run once after freeze)
cloud/                     Modal apps (sim fan-out, perception), ROCm Dockerfile (untested, needs MI300X)
render/blender             Blender Cycles replay (primary). render/isaac/replay.py is kept ONLY because replay_bpy imports its pure-python half; Isaac is used physics-only on Modal for the legged robot (DIRECTION §5.2); its rendering crashes on Modal
scripts/                   build_map, gemma_*, perception_*, plots
dashboard/                 live view of the map growing (static HTML/JS; replay + local websocket live mode) — DIRECTION §7 D
```
Python: `C:\AMD\intercycle\.venv\Scripts\python.exe` (3.14). Tests: `.venv\Scripts\python.exe -m pytest -q`. Secrets: `.env` (gitignored): never print or commit it.

## Time convention
Sim time `t` = seconds since 00:00 local of sim day 0; unix = `DAY0_UNIX + t` (see types.py).

## Licences
- MIT repo.
- Avoid AGPL: no Ultralytics.
- Use RF-DETR N–L (Apache-2.0; not XL/2XL), roboflow `trackers` (Apache-2.0), supervision (MIT), stumpy (BSD-3), ruptures (BSD-2), hmmlearn (BSD), eclipse-zenoh (Apache-2.0/EPL-2.0).
- STVL is LGPL: a runtime dependency only.
- SmartSpaces is CC-BY-4.0 and is **synthetic** video (attribute it).
- NVIDIA SimReady/robot assets are CC-BY-4.0 (credit them in `credits.json`).
- Every synthetic element is labelled SYNTHETIC.
