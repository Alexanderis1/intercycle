# InterCycle — engineering brief (read this first)

Hackathon: AMD Developer Hackathon ACT III, Track 1 Intelligent Industry. Submission **18 Oct 2026 16:00 CEST**, code freeze 12:30. Public **MIT** repo. Solo developer + Claude.

## Product
Existing ceiling cameras in an **automotive body shop (BIW) with an on-site data hall** learn
(1) hour-of-day traffic per zone (people, AGVs, tuggers, forklifts, data-hall technicians) and
(2) the cycle of every automatic machine (period, busy fraction, jitter, drift, stop rate, and *where it is in its cycle right now*).
An observer box broadcasts this as **Ed25519-signed BLE 5 advertisements** (one-way broadcast, no pairing, no plant-network or PLC/DCS link). Any visiting inspection robot with a BLE receiver listens, verifies, and uses **SIPP** (safe interval path planning) to time inspections **between machine cycles instead of during shutdowns**.

Headline: *"Inspect between cycles, not between shutdowns."* Buyer: maintenance/reliability manager; EHS co-signs.

## Safety semantics (design assumptions — say so in docs)
- Machines are **not fenced boxes**. Each machine has a **hazard zone** (robot reach envelope, lifter drop zone, hemming/sealer cell) guarded by **light curtains / area scanners**.
- Zone ACTIVE = machine moving. A robot crossing the boundary of an ACTIVE zone **trips** the safety system → protective stop → line stop with `recovery_after_trip_s` (90 s) restart. Line stations stop the whole line.
- A robot inside an IDLE zone **inhibits the next start** (hold) until it leaves. Holds on line stations hold the whole line.
- The prediction only changes *when* the robot tries. Before crossing, the robot's own camera confirms "no motion" (`local_check_latency_s`). The safety system is always the final layer.
- Cost framing: Siemens *True Cost of Downtime 2024*: ~$2.3M/h for a large automotive plant (≈ $639/s). Context only, not our measured gain.

## Robots compared (same seeds)
- **A0 shutdown-only** (status quo): inspects only during planned breaks / shift changes.
- **A reactive**: drives to the zone, waits until its own camera sees the machine stop, enters. Doesn't know how long the idle lasts → holds and occasional trips.
- **B InterCycle**: receives slow + fast layers over BLE, plans with SIPP, enters only at the start of a predicted idle window that fits `inside_s + safety_margin_s`, after a local check.
- **C PLC oracle**: knows the true future (upper bound; needs plant integration).
Metrics: inspections per production hour, line time lost to robot (holds + trips × recovery), trips per shift, robot waiting, gain retained = (A − B)/(A − C) on line-time lost, entries into an active zone (must be 0 for B under fault injection).

## Repo layout & ownership
```
intercycle/types.py        shared dataclasses (Window, MachineModel, MachineNow, SlowLayer, FastLayer, Decoded, State)
intercycle/scenario.py     YAML loader (scenario/bodyshop.yaml is the single source of truth)
intercycle/truth/          ground truth generators — learner/codec/robot MUST NOT import (tests/test_no_truth_import.py)
intercycle/observer/       signals.py (camera shim, may import truth), learn_cycles.py, learn_traffic.py (no truth!)
intercycle/codec/          layout.py (bits), sign.py (Ed25519), ble.py (framing, channel sim, receiver)
intercycle/robot/          sipp.py, policies.py (A0/A/B/C)       — no truth imports
intercycle/sim/            world.py (applies holds/trips to truth timelines), run.py
intercycle/eval/           seeds, metrics, figures
render/isaac, render/blender   photoreal replay of sim traces (trace.parquet)
dashboard/                 FastAPI + Three.js
```
Python: `C:\AMD\intercycle\.venv\Scripts\python.exe` (3.14; numpy, scipy, astropy, pydantic, pynacl, pandas, pyarrow, pytest, modal, fastapi installed). Run tests with `.venv\Scripts\python.exe -m pytest -q`.

## Time convention
Sim time `t` = seconds since 00:00 local of sim day 0; unix = `DAY0_UNIX + t` (see types.py).

## Licences
MIT repo. Avoid AGPL (no Ultralytics). Use RF-DETR (Apache-2.0), roboflow `trackers` (Apache-2.0), supervision (MIT). SmartSpaces data is CC-BY-4.0 (attribute). Every synthetic element is labelled synthetic.
