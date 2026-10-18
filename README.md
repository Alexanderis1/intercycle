# InterCycle

**The cameras a factory already has, guiding the robots it is deploying.**

InterCycle watches the plant's existing ceiling CCTV, learns when and where each machine, person and vehicle
moves, and broadcasts a signed map to the inspection robot's standard ROS 2 Nav2 stack, so the robot reaches
its inspection spots without stopping the line. Only the observer layer is added: no new floor sensors and no
link to the machine controllers (PLCs). It is an advisory planning layer, not a safety function.

Built for the AMD Developer Hackathon ACT III, Track 1 (Intelligent Industry).

## Results

**Automotive body shop** (event simulation, 20 seeds x 6 h, commit `1671f9e`):

| Robot strategy | Inspections / production hour | Safety trips / run | Line time lost / run |
|---|---|---|---|
| **InterCycle** | **39.9** | **0** | **1.5 s** |
| Reacts to its own camera | 50.4 | 5.0 | 568 s |
| Inspects only during shutdowns | 7.6 | 0.25 | 23 s |
| PLC oracle (knows the future, upper bound) | 46.7 | 0 | 0 s |

- **5.2x the inspections** of a shutdown-only robot.
- **Zero safety trips** and **380x less line time lost** than a robot that reacts to its own camera.
- **85 %** of the throughput of a PLC oracle that knows every machine's future, from video alone.

**Real factory CCTV** (sealed blind test, 126 held-out clips, pass levels committed in advance):

- **23 of 23** detected press strokes were real (precision 1.00).
- **82 of 83** people placed in the right floor zone.

Limits are listed at the end of this file.

## Reproduce

### Install

Python 3.11 or newer, and Git. A CPU is enough.

```bash
git clone https://github.com/Alexanderis1/intercycle.git
cd intercycle
python -m venv .venv
source .venv/bin/activate            # Windows (Git Bash): source .venv/Scripts/activate
pip install -e ".[dev]"
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PWD/ros2/intercycle_nav:$PWD/ros2/intercycle_eval"
# Windows: export PYTHONPATH=".;ros2/intercycle_nav;ros2/intercycle_eval"
```

### Quickstart (about 1 minute)

```bash
# Signed broadcast, trust chain, camera watchdog, observer/simulator wall: 110 tests, ~10 s
python -m pytest -q tests/test_codec.py tests/test_ble.py tests/test_trust.py tests/test_watchdog.py tests/test_no_truth_import.py

# Body-shop simulation, smoke version (2 seeds x 1 h), writes out/bodyshop_quick/
python -m intercycle.eval.bodyshop --quick
```

### One command per result

| Result | Command | Output |
|---|---|---|
| Body-shop table above (commit `1671f9e`) | `git switch --detach 1671f9e && python -m intercycle.eval.run_seeds --seeds 0 19` | table on screen and `out/results_dev_0_19.json`; committed copy: `results/bodyshop/eventsim_1671f9e/` |
| Body shop, current harsher sim (noisy robot camera) | `python -m intercycle.eval.bodyshop --seeds 0-19 --hours 6` | `results/bodyshop/RESULTS.md`, `summary.json` |
| Sealed real-clip test | already run once, by design; scores committed | `results/sealed/realclip_test_attempt_001.json`, `results/sealed/RUNLOG.json` |
| Real-clip development set | committed scores | `results/dev/realclip_dev.json` |
| Full test suite | `python -m pytest -q` | several minutes |

On the current sim, where the robot's own camera can miss, misread or lag, InterCycle keeps 3.35 trips and
305 s lost per run, against 10.95 trips and 1,135 s for the reactive robot (3.3x fewer trips, 3.7x less line
time lost), at 0.886 of the oracle's throughput. The remaining trips are tracked as E-W6 in
`docs/WEAKNESSES.md`.

The sealed test cannot be re-run from this repository: the held-out clips stay outside it, and only their
hashes are committed (`sealed/`, `prereg/`).

## How it works

**Watch -> Learn -> Broadcast -> Inspect**

1. **Watch** (`intercycle/perception/`): the existing CCTV. Every detected person is masked before an image
   leaves the pipeline.
2. **Learn** (`intercycle/observer/`): each machine's cycle and phases, whether it runs or stops, and the flows
   of people and vehicles by time of day.
3. **Broadcast** (`intercycle/codec/`): a layered map signed with Ed25519, in chunks small enough for
   Bluetooth LE. Forged, tampered, expired and replayed messages are rejected.
4. **Inspect** (`ros2/intercycle_nav/`): Nav2's own `KeepoutFilter` and `SpeedFilter` take the map. Each
   approach is timed to a predicted idle window.

Safety rules: unseen or stale space counts as occupied; the broadcast can only add keep-out and lower speed,
never remove them; a frozen or blocked camera turns its floor unknown; a robot without a trusted key ignores
every map. Details: `docs/MAP.md`, `docs/STATUS.md`.

## Limits

- The body-shop results come from an event simulation, not a real plant.
- In the sealed test, stroke detection found 23 of 51 labelled strokes (recall 0.45). The pre-registered
  person-recall test (H4) did not pass: the lowest clip group scored 0.33 against a bar of 0.53.
- Labels are AI-agent labels, not human labels.
- Not shown: a real robot among real cameras, safety certification.
- Rendered videos are illustrative animations of the scenes, not recorded runs.

## Credits and licences

- Code: MIT (`LICENSE`).
- CCTV clips: Önal, O. and Dandıl, E. (2024), *Data in Brief* 56, 110791, via the Hugging Face set
  Voxel51/Safe_and_Unsafe_Behaviours, CC BY 4.0. Trimmed, re-encoded, people masked. No affiliation or
  endorsement.
- ANYmal-D model: BSD-3-Clause, ANYbotics. NVIDIA SimReady assets: CC-BY-4.0. Gemma 4: Apache-2.0, served
  through AMD Token Factory.

Author: Alessandro Angelo Anzellini. Special thanks to Ing. Emanuele Musumeci.
