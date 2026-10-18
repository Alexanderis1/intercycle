# InterCycle

**The cameras a factory already has, guiding the robots it is deploying.**

InterCycle turns ceiling-camera video into a signed, continuously updated map of when and where an inspection robot
can safely go. It learns each machine's cycle and how people and forklifts move, from video alone, and broadcasts
that map to the robot, signed. Same cameras, same robots: only the observer layer is added. No new sensors on the
floor, no link to the machine controllers (PLCs). The full chain is shown end to end in simulation; the perception
is checked on real factory CCTV.

In an event simulation of an automotive body shop (20 seeds x 6 h), a robot guided by InterCycle did **5.4x the
inspections per production hour of a shutdown-only robot**, and compared with a robot that reacts to its own camera
it had **3.3x fewer safety trips and 3.7x less line time lost**. On real factory CCTV, **every one of the 23 press
strokes it detected in a sealed blind test was real**.

InterCycle is an advisory planning layer. It is not safety-rated. The robot's own certified safety scanner stays
the safety function.

Built for the AMD Developer Hackathon ACT III, Track 1 (Intelligent Industry).

---

## How it works

**Watch -> Learn -> Broadcast -> Inspect**

1. **Watch.** The CCTV already on the ceiling. A one-time import of the site plan (zones, camera poses).
   Our pipeline masks every detected person before an image is sent to the vision model.
   Code: `intercycle/perception/` (detector, tracking, ground-plane geometry, `privacy_mask.py`).
2. **Learn.** Each machine's cycle and the phases inside it, whether each machine runs or stops, and the flows of
   people and vehicles by time of day. Code: `intercycle/observer/` (`learn_cycles.py`, `learn_traffic.py`,
   `phases.py`, `mapping.py`).
3. **Broadcast.** A layered map, signed with Ed25519, sent in small chunks that fit a radio frame.
   Code: `intercycle/codec/` (`sign.py`, `trust.py`, `maplayer.py`, `ble.py`).
4. **Inspect.** The robot's standard ROS 2 Nav2 stack takes the map as keep-out and speed zones, and each approach
   is timed to a predicted idle window. Code: `ros2/intercycle_nav/` (Nav2 config in `config/nav2.yaml`),
   `intercycle/robot/`. The Nav2 integration is tested in NVIDIA Isaac Sim; the inspection results below come from
   the event simulation.

### The layered map

| Layer | What it holds |
|---|---|
| L0 | Static plant: walls, racks, machine bases |
| L1 | Machine envelopes: the floor each machine sweeps, per phase of its cycle |
| L2 | Forklift and AGV flows, per 1 m cell, by time of day and production state |
| L3 | People flows, same format |
| UNKNOWN | Never seen, camera offline, or a new machine still being learned. Counts as occupied |

Format: 1 m cells, 8 headings, half-hour time-of-day slices, production-state slices. The flow layers are built in
simulation; on real CCTV the code detects people and places them in the right floor zone. Details: `docs/MAP.md`.

### Signed broadcast

- Every layer is signed with Ed25519. The observer's key is trusted only through a certificate from an offline
  site root key, with revocation.
- Forged, tampered, expired and replayed messages are rejected (`tests/test_codec.py`, `tests/test_ble.py`,
  `tests/test_trust.py`).
- Compact: a 3-day map is 15.3 kB and a robot rebuilds it in 6.6 s over a simulated Bluetooth LE link at 10 %
  packet loss (`docs/MAP.md`, section 5).
- Compact enough for Bluetooth LE (simulated today); any IP link such as Wi-Fi can carry the same signed payload.

### On the robot: standard Nav2

- Nav2's own `KeepoutFilter` and `SpeedFilter` (`ros2/intercycle_nav/config/nav2.yaml`). No custom navigation stack.
- Keep-out is projected ahead along each machine's learned cycle.
- Lower speed where people and forklifts are likely.
- Each approach is timed to a predicted idle window. The robot's own camera confirms before it enters.

### Safe by design

- **Unknown = occupied.** Unseen space, an offline camera or stale data all count as occupied.
- **Restrict only.** The broadcast can add keep-out and lower speed, never remove them
  (`ros2/intercycle_nav/intercycle_nav/masks.py`, `ros2/intercycle_nav/test/test_monotone.py`).
- **Camera watchdog.** A frozen, black, blocked or blurred image turns that floor unknown
  (`intercycle/observer/watchdog.py`, `tests/test_watchdog.py`).
- **Hidden is not idle.** A press hidden by a person is marked MISSING, not "stopped" (`intercycle/observer/occlusion.py`).
- **No trusted key, no start.** A robot without a valid keyring refuses to trust any map
  (`ros2/intercycle_nav/test/test_keyring_failclosed.py`).
- **AI adds caution only.** The vision model (Gemma 4) can lower confidence or close a window, never open one
  (`intercycle/llm/`).
- **The observer is walled off from the simulator's truth.** Import-graph tests block new paths from the observer
  to the simulator truth; the remaining known exceptions are listed in the tests (`tests/test_no_truth_import.py`,
  `tests/test_truth_allowlist.py`).

---

## Results

### 1. Automotive body shop (simulated): the headline

An event simulation of an automotive body shop: welding robots, hemming, sealing, AGVs and people. Four robot
strategies run on the same seeds:

- **A0 shutdown-only:** inspects only during planned breaks.
- **A reactive:** waits until its own camera sees a machine stop, then enters.
- **B InterCycle:** guided by the learned observer.
- **C PLC oracle:** knows the true future from every machine controller. An upper bound.

Seeds 0-19 (n = 20 per strategy), 6 h shift window from 07:00, code commit `80fb1a8`. Means per run with 95 %
confidence half-widths (1.96 x standard error over seeds). Source: `results/bodyshop/RESULTS.md`,
`results/bodyshop/summary.json`.

| strategy | insp/h | trips/run | holds/run | line time lost s/run | insp/h vs C |
|---|---|---|---|---|---|
| A0 shutdown-only | 7.7 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.164 |
| A reactive | 48.1 ± 0.6 | 10.95 ± 2.25 | 44.55 ± 5.22 | 1135.3 ± 216.4 | 1.028 |
| B InterCycle (learned observer) | 41.5 ± 0.4 | 3.35 ± 1.39 | 0.90 ± 0.35 | 304.6 ± 124.7 | 0.886 |
| C PLC oracle | 46.8 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 | 1.000 |

- B/C throughput (InterCycle inspections per production hour / PLC oracle): **0.886**
- Line-time gain retained by B, (A - B) / (A - C): 0.732
- insp/h = machine inspections per production hour; trips = robot crossed into an active zone (protective stop of
  the line); holds = inspections where the line waited for the robot; line time lost = seconds of production
  stopped by the robot.

The reactive robot inspects more often than InterCycle, but it pays for it with 3.3x the safety trips and 3.7x the
line time lost. InterCycle's remaining trips are a known open issue (E-W6 in `docs/WEAKNESSES.md`, xfail test in
`tests/test_safety.py`): the walkway rule can count an occlusion or a false motion of the robot's own camera as a
seen transfer.

How to reproduce it (about 5 min on 6 processes):

```bash
python -m intercycle.eval.bodyshop --seeds 0-19 --hours 6
```

It writes `results/bodyshop/RESULTS.md` and `results/bodyshop/summary.json` (seeds, hours, commit, whether the code
had uncommitted changes, means with 95 % CIs, per-run rows). `--workers` and `--out` are optional.

**Earlier sim version.** At commit `1671f9e` the robot's own camera was ideal (no occlusion, missed or false
motion, no latency). There, B had 39.9 insp/h, 0 trips and 1.5 s line time lost against A's 5.0 trips and 568 s,
and B/C was 0.854. Commit `4660155` then made the robot camera noisy, added speed jitter and gave all strategies the
same walkway rule, which gives the table above. The earlier output is committed unedited in
`results/bodyshop/eventsim_1671f9e/` with its provenance.

What the observer learned from camera signals alone in this simulation (one held-out seed, commit `947a878`): welder
phases at segment F1 0.85, with a median boundary error of 0.57 s; line states found with no calendar. Code:
`intercycle/sim/`, `intercycle/eval/`. Status and history: `docs/STATUS.md`.

### 2. Real factory CCTV: perception on the development set

CCTV of a real metal and plastics press shop from a published data set (96.8 minutes in 691 clips from 2 cameras
in all; see Credits). The development set is the part not held out for the sealed test. Scored against independent AI-agent labels (a vision model), not human labels. Source:
`results/dev/realclip_dev.json`.

| Measure | Result |
|---|---|
| Press-stroke precision | 1.00 (53 of 53 detected strokes were real) |
| Press-stroke F1, all presses | 0.80 |
| Press-stroke F1, best-visible press (P9a) | 0.92 |
| People placed in the right floor zone | 1.00 (81 of 81 matched people) |
| Press cycle periods learned from training clips | 1.96 s and 1.69 s (`docs/probes/S3a-P4.md`) |

### 3. Real factory CCTV: sealed blind test

126 real clips were held out and fixed by hash before any result. The pass levels were written down in advance
(`prereg/`). The test ran once. Source: `results/sealed/realclip_test_attempt_001.json`, run log
`results/sealed/RUNLOG.json`.

| Measure | Result |
|---|---|
| Press-stroke precision | 1.00 (23 of 23 detected strokes were real) |
| People placed in the right floor zone | 0.99 (82 of 83 matched people) |

The pre-registered person-recall test (H4) is not claimed: average recall was 0.65 (95 % CI 0.56-0.74), but the
lowest clip group scored 0.33 against a pass bar of 0.53. We wrote down before the test that this outcome was
expected, because camera 9 has only 4 test groups.

### 4. Press-shop digital twin

From the same footage we built a digital twin of the press shop. Its plant days are generated by resampling the real
motion, so a robot can be tested over many days without stopping a line.

- **5 of 5** distribution checks passed against held-out real tracks: walking speed in both zones, forklift speed,
  and stroke intervals of both presses (`docs/DYN_VALIDATION.md`).
- **2 of 2** Nav2 legs walked by a legged ANYmal-D in NVIDIA Isaac Sim physics: 20.6 m, no fall, zero contacts
  (`docs/probes/S5a_isaac_smoke_verdict.json`).
- **2.6 to 2.8 US cents** of cloud compute per simulated 12-minute mission on our CPU robot double (Nav2 with a
  kinematic test double, no physics engine; Modal, 4 CPU) (`docs/probes/H1.md`). Isaac physics runs cost more.

The sealed robot batch on the twin was not run. No robot claim is made from the twin.

---

## Installation

**Requirements**

- Python 3.11 or newer (`pyproject.toml`). Development used Python 3.14.
- Git.
- CPU is enough for the tests and the body-shop simulation.

**Set up (Linux / macOS)**

```bash
git clone https://github.com/Alexanderis1/intercycle.git
cd intercycle
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PWD/ros2/intercycle_nav:$PWD/ros2/intercycle_eval"
```

**Set up (Windows, Git Bash)**

```bash
git clone https://github.com/Alexanderis1/intercycle.git
cd intercycle
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
export PYTHONUTF8=1
export PYTHONPATH=".;ros2/intercycle_nav;ros2/intercycle_eval"   # Windows uses ';' as separator
```

On Windows replace `python` in the commands below with `.venv/Scripts/python.exe`.

**Optional extras** (all in `pyproject.toml`)

| Extra | What it adds |
|---|---|
| `.[phases]` | `stumpy` matrix-profile engine for phase discovery (a numpy fallback exists) |
| `.[dashboard]` | live map view relay and screen recording |
| `.[cloud]` | Modal (cloud fan-out of simulations, perception, Isaac runs) and an OpenAI-compatible client for Gemma 4 |

**Optional tools**

- **ROS 2 Humble + Nav2** for the robot side (`ros2/intercycle_nav`, launch files in `ros2/intercycle_nav/launch/`).
- **Modal** account and token for cloud runs (`cloud/`). Keep keys in `.env`, which git ignores. Never commit keys.
- **Blender 5.1** for the rendered scenes (`render/blender/`).
- **NVIDIA Isaac Sim** (run on Modal) for the legged-robot physics check (`cloud/modal_isaac_robot.py`).

---

## Quickstart

Run from the repository root, after the installation step. Timings are on a Windows laptop CPU.

1. Check the signed broadcast, the trust chain, the camera watchdog and the truth wall (about 10 s,
   110 tests pass):

   ```bash
   python -m pytest -q tests/test_codec.py tests/test_ble.py tests/test_trust.py tests/test_watchdog.py tests/test_no_truth_import.py
   ```

2. Run the automotive body-shop simulation, quick version (2 seeds x 1 h, under a minute; writes to
   `out/bodyshop_quick/`). It is a smoke check: 1 h is too short for the shutdown-only robot to reach a break, and
   2 seeds are too few for a result.

   ```bash
   python -m intercycle.eval.bodyshop --quick
   ```

   The full run behind the Results table is `python -m intercycle.eval.bodyshop --seeds 0-19 --hours 6`
   (about 5 min on 6 processes).

3. Run the full test suite (about 1,240 test functions; some files run simulations and take several minutes):

   ```bash
   python -m pytest -q
   ```

---

## Repository layout

```
intercycle/
  perception/   detection, tracking, ground-plane geometry, privacy masking
  observer/     learns cycles, phases, traffic flows; builds the layered map; watchdog, occlusion
  codec/        map layers, Ed25519 signing, trust chain, Bluetooth LE framing
  robot/        planners and the A0 / A / B / C policies
  sim/          automotive body-shop event simulation
  eval/         body-shop results (bodyshop.py), seed sweeps (run_seeds.py), scoring (real clips, twin)
  llm/          Gemma 4 client, observer tasks, privacy filter
  truth/        simulator ground truth (the observer may not import it)
twin/           press-shop digital twin: site, generated days, plant, camera model
ros2/           ROS 2 packages: Nav2 bridge and filters (intercycle_nav), evaluation (intercycle_eval), sim (intercycle_sim)
scenario/       body-shop scenario files (bodyshop.yaml is the single source of truth)
cloud/          Modal apps (simulation fan-out, perception, Isaac robot runs)
render/         Blender scenes for the demo video
dashboard/      static web view of the map growing over time
scripts/        command-line tools (map build, perception, sealed run, plots)
prereg/         pre-registration, amendments, freeze records, data split
results/        committed result files (body shop, dev and sealed real-clip runs)
sealed/         hashes of the sealed test clips and labels
docs/           design notes, status, validation and probe reports
tests/          automated tests
```

---

## Media

- Pitch deck: https://claude.ai/artifact/6gGWqcaffwDxej4V7EwHVu
- Demo video: [[VIDEO_LINK]]

Renders are illustrative: the rendered robot routes are scripted animations of the digital twin, and the Isaac Sim
clip is a physics simulation. Frames with generated content are labelled as such. All people are masked.

---

## Validation and limits

- **The body-shop results are a simulation.** They are an event simulation with a model of the line, not a real plant.
- **The twin's days are generated**, from real motion. A win on the twin is not a win on a real line.
- **Labels are AI-agent labels** (a vision model), not human labels (`docs/LABEL_AGREEMENT.md`).
- **Stroke detection is precise but misses strokes.** In the sealed test it found 23 of 51 labelled strokes
  (recall 0.45), and every one it found was real.
- **Person recall did not pass its pre-registered bar** (see Results, section 3).
- **Not shown:** a real robot among real cameras, a real shop floor, safety certification.
- **Rigour:** hypotheses and pass levels were committed before any test run (`prereg/PREREG_TWIN.md`,
  `prereg/AMENDMENTS.md`); the sealed clips and labels are fixed by hash (`sealed/`).
- Full status and history: `docs/STATUS.md`, `STATUS.md`. Known weak spots: `docs/WEAKNESSES.md`.

---

## Credits and licences

- **Code:** MIT (`LICENSE`), Copyright (c) 2026 InterCycle contributors.
- **CCTV clips:** Önal, O. and Dandıl, E. (2024), *Data in Brief* 56, 110791, through the Hugging Face set
  Voxel51/Safe_and_Unsafe_Behaviours. Licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes:
  trimmed, re-encoded, people masked. Tracks, labels and figures are derived by InterCycle. We have no affiliation
  with the plant or the authors, and no endorsement is implied. The plant is not named in this repository.
- **ANYmal-D robot model:** BSD-3-Clause, ANYbotics.
- **NVIDIA assets** (forklifts and other SimReady assets): CC-BY-4.0, NVIDIA (`render/blender/assets.py`).
- **Gemma 4** (Apache-2.0), served through AMD Token Factory.
- Stack: permissive licences only (MIT, Apache-2.0, BSD). No AGPL.

Author: Alessandro Angelo Anzellini. Special thanks to Ing. Emanuele Musumeci.
