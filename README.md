# InterCycle

InterCycle is an observer layer that runs on existing CCTV cameras. It learns when machines cycle and where people
and vehicles move. It then gives an existing inspection robot signed keep-out maps and timing maps. If a map is
missing, stale or not signed, the robot treats the space as occupied (fail closed). The layer is advisory. It is not
safety-rated, and the robot's own certified scanner stays the safety function.

The work in this repository is tested on a digital twin of a press shop. The twin is built from public CCTV clips.
Its days are generated, not recorded.

## What was tested

Two tests were pre-registered before any dev-world run. Each has its own sealed run.

1. **Sealed real-clip test (H4): person recall.** 126 held-out public clips (114 from camera 9, 12 from camera 14).
   A detector finds people. We check how many labelled people it finds. The pass level is r0 = 0.5272, set from the
   19 scored blind dev frames only (of 20; one is a duplicate). The labels are made by Claude agents, not humans.
   Stated in advance: camera 9 has only 4 test groups, so every per-group recall must reach r0, and H4 is expected
   not to be claimed if any one group falls below it. That is the pre-registered rule, not a failure of the run.
2. **Sealed twin robot batch (H1-H3).** 192 missions on the CPU double: 3 conditions (static, learned, best reactive)
   x 16 test worlds x test days 9-10 x zones 9 and 14.
   - H1: the learned robot completes more inspections per mission-hour than the reactive robot.
   - H2: the learned robot costs at most 30 s more lost press time per mission-hour than the static robot
     (non-inferiority).
   - H3: the map's miss share at a fixed margin stays below 0.06 (mean over worlds) on new worlds (a consistency check).

   This batch is a claim only if the robot part is frozen in time (`prereg/FREEZE`, planned about 10:15 CEST on
   18 Oct 2026). If it is not, the robot numbers are descriptive dev results from dev worlds 0-3, and nothing in
   H1-H3 is claimed.

The gate order is fixed: H1, then H2, then H3, then H4. A hypothesis counts as claimed only if it and every earlier
one passed. Details: `prereg/PREREG_TWIN.md` (sections 3 and 4). Time of each step: `STATUS.md`.

## Headline results

Both lines stay as placeholders until the sealed runs are scored. Each is filled with one of the wordings the
pre-registration allows (rules in the comments, and in `STATUS.md`).

- **Real-clip test, person recall (H4):** [[RESULT_H4]]
- **Twin robot batch (H1-H3):** [[RESULT_ROBOT]]

<!--
How to fill [[RESULT_H4]] (PREREG sections 4, 4.1, 10). Always say: "against independent Claude-agent labels (vision
model), not human labels"; give n (frames, people, groups); r0 = 0.5272; the figures per stratum (safe and unsafe,
per camera). Camera 9 has 4 test groups, so the small-group rule applies: give per-group recall, and the frame-level
Clopper-Pearson interval marked "ignores clustering (only 4 groups)". Also say that near-duplicate footage can sit on
both sides of the train/test line (A1+).
  (a) "H4 claimed: <pooled recall>, lowest per-group recall <x>, both at or above r0."
      Allowed only if H1, H2 and H3 were claimed first and h4_recall rejects.
  (b) "H4 computed, not claimed: <figures>. A hypothesis is claimed only if every earlier one passed."
      Use when the robot part is descriptive or an earlier gate did not pass.
  (c) "H4 not claimed: the lowest per-group recall <x> is below r0 = 0.5272. This is the pre-registered rule, not a
      failure of the run."

How to fill [[RESULT_ROBOT]] (PREREG sections 0, 2.1, 3, 4, 8, 9). Always say: the effect holds only inside the
assumed stop model; zone 14 presses are assumed; neither robot arm was tuned; the learned robot never enters a
stopped press and the reactive robot can; the margin is a fixed latency margin, not a certified one.
  (a) Sealed and frozen: "H1 <claimed | not claimed>: mean difference <d> inspections per mission-hour (learned minus
      reactive), exact sign-flip p = <p> over 16 worlds. H2 <...>. H3 <...>." Add learned minus static with its 95 %
      world-clustered interval. If that point estimate is 0 or below, add on the first screen: "static boundary
      inspection matched or beat the learned policy."
  (b) Not frozen in time: "Descriptive dev results from dev worlds 0-3. No H1, H2 or H3 claim is made."
  (c) Any design cell unresolved after the last re-run: "Not evaluable: unresolved cells. No H1, H2 or H3 claim is
      made."
-->

## What this can and cannot show

- **H1 holds only inside the assumed stop model.** The twin's press stops, their timing and their lengths come from a
  generator we wrote. No real clip has timing information. A win on the twin says the learned robot uses that stop
  model better than a camera-only reactive rule. It does not say the same win happens on a real press line.
- **Zone 9 carries the learned evidence. Zone 14 is fully assumed.** Two zone-9 presses have stroke periods learned
  from train clips. The four zone-14 presses are assumed. The gate pools both zones with equal weight. The per-zone
  split is reported next to it.
- **H3 is a consistency check on the same generator.** The margin and the test worlds come from one generator family.
  H3 is not a safety proof for a real shop floor.
- **Real-clip figures are against Claude-agent labels**, not human labels. The agreement between two labeller
  agents is their noise floor (`docs/LABEL_AGREEMENT.md`).
- **Neither robot was tuned.** Both run their committed default settings. The comparison is equal, but neither is at
  its best setting.
- **Stop entry is asymmetric, by design.** The learned robot never enters a stopped press. The reactive robot can.
- **The robot uses the maps in a simple way.** It treats them as a yes/no keep-out mask, looking 6 s ahead. On top it
  uses timing gates (enter a press area only in a predicted idle window) and lower speed in busy areas. It does not
  weigh areas by risk. A win or a loss is a result of that simple use, not of the best possible use.
- **The observer learns by clock time only.** It learns what usually happens in each time-of-day bin. It does not
  learn effects triggered by events, such as "after a forklift passes, the press stops". That is future work. Nothing
  here tests it.
- **If the robot part is not ready by the freeze, there is no H1-H3 claim.** The robot results are then descriptive.
- **Nothing was tuned after looking at results.** No threshold was chosen from dev or test data. The one approved
  change to a learning limit (fix (a), disclosure 16) is chosen from the tick period, not from a result.
- **Not shown at all:** a real robot among real cameras (the real cameras never saw a robot), a real shop floor,
  safety certification.

## Disclosures

Sources: `prereg/AMENDMENTS.md` (A1-A6, including the A5 erratum) and `prereg/PREREG_TWIN.md` section 11, which holds
the full list of 36 items. The ones below are the ones a reader needs first.

Data and labels

1. **All labels are made by Claude agents** (vision models), not by humans. This holds for dev and test labels. The
   blind agreement between labeller agents covers camera 9 only, and its intervals ignore clustering.
2. **Two test labeller agents.** One labelled person frames, one labelled stroke windows. Both read only
   `labels/TEST_RULES.md` and the test images.
3. **Label file format fix.** The person-frame file `labels_frames.json` had no `stroke_windows` field. A fresh
   labeller agent added an empty list. No label changed. The file hash before and after is in the custodian's access
   log, which no builder reads.
4. **P9a has 1 of 13 planned test stroke windows.** The test split holds only one clip of that framing. P9a's stroke
   figures rest on n = 1.
5. **Split risk.** Groups are clusters by background, lighting and layout, without the near-copy link. Real recording
   sessions may be split across groups, so near-duplicate footage can sit on both sides of the train/test line.
   Also, 26 test clips belong to byte-identical sets (13 pairs inside test), and those sets carry different class
   labels.
6. **The split is unbalanced.** Camera 9 test has 109 unsafe and 5 safe clips, and no class-2 or class-4 clip.
   Camera 14 has 3 of 40 safe clips in test. Figures are reported per stratum.
7. **Four test clips were processed by machine without an agent looking at them** (7_tr1, 3_te7, 3_tr42, 6_tr12).
   Only clips whose pixels an agent viewed were kept out of test.
8. **Calibration was refit.** The first calibration files used sources that are now dev or test. They were declared
   invalid and refit on train clips. The calibration points were placed by an agent, not clicked by the user.

Seal and procedure

9. **The seal is procedural, on one machine.** The custodian and labeller agents run on a desktop. The builder lanes
   never read the sealed folder. The seal rests on agent separation, an access log and committed hashes. It is not
   a machine boundary, and it is not cryptographic.
10. **Two freeze files instead of one.** The real-clip part is frozen by `prereg/FREEZE_REALCLIP`, the robot part by
    `prereg/FREEZE`. Test seeds come from a public random beacon (drand) only after the robot freeze. The real-clip
    freeze moved from 08:30 to as late as 10:15 CEST by the user's decision, and the user then chose both freezes on
    one commit at 10:15 (A5). Real-clip outputs go only to the custodian until `FREEZE` is committed.
11. **Cut design.** 3 conditions and 192 missions, not 5 conditions and 320. There is no timing-oracle condition and
    no no-map condition. Isaac, mover and body-shop (A-lite) runs are descriptive only and sit outside the sealed run.
12. **Code hashes re-recorded by an append-only amendment.** The pre-registration text lists a git hash for 15 code
    paths. By the time it reached main, 8 of the 15 had changed. The new hashes are listed in Amendment 5 and its
    erratum, and the pre-registration text is untouched. The list replaces the table update that the text called for.

Observer and twin

13. **Twin camera.** On the real site files the twin's camera is `stat_v1`, a statistical model of detector noise
    fitted on dev labels and dev detector tracks (plus train ROI data). It is not rendered pixels. The older `camera_v0` stays on main only for the stub site and as
    a noise reference.
14. **Learning time is 20 minutes, not days 1-8.** The user chose "Try days 1-8, else 20 min". Days 1-8 were measured at
    about 90 s of wall time per 1,200 s simulated. That is about 14.4 hours per world, so it cannot finish before
    submission. The observer learns for 20 minutes before each mission. With 20 minutes of learning, the v1 slice
    gave confidence 0 on every press before fix (a).
15. **`keep_max_s` = 1200 s** (user decision). A confident model kept over mostly-missing data lives at most 1200 s
    after its last confident fit.
16. **Fix (a), approved and still being tried.** It matches the press timing limits to the 2 Hz observation tick
    (jitter and consistency limits) so that press P9b can be learned. It is chosen from the tick period, not from a
    result. It is not on main at this commit. If it is merged before the freeze, it gets its own amendment with old
    and new code hashes.
17. **ROI-only occlusion, in clips and in the twin.** A person or vehicle box that overlaps a press's ram region
    marks that press MISSING. The floor-plane rule is off for real clips (user decision) and in the twin (A4).
    Two differences stand and were not changed (A6): the hold after an overlap is 1.33 s on real clips but 34 s
    (camera 9) and 7.5 s (camera 14) in the twin; and in the twin only movers the camera detects that tick occlude,
    while on real clips every confirmed track box occludes, plus every person or forklift detection at conf 0.4 or
    above.
    Limit, dev figures: inside hidden spans, 336 of 428 samples (0.785) were flagged, so 21.5 % were not. On the 2 Hz
    ticks inside hidden spans, 31 of 35 were MISSING. Outside hidden spans, 145 of 292 (0.497) were also MISSING,
    mostly from the watchdog.
18. **Strokes are scored at the bottom of the stroke (BDC)**, the label definition. The tolerance is +/- 0.25 s.
19. **Fixed latency margin, not a certified one.** The margin is the 99th-percentile map age times the top speed,
    rounded up to whole cells. There is no conformal certificate. If the margin does not fit the grid (at most 1.0 m,
    so a map age p99 of at most 0.5578 s), the robot part is descriptive. The margin is never cut to fit.
20. **Stop-model limits.** Press timing is learned for two presses only (periods 1.96 s and 1.69 s). All zone-14 press
    timing is assumed. Guarding and reset are assumed, not observed.
21. **r0 is a camera-9 value.** It was set from the blind dev frames (19 scored of 20), all from camera 9, and applied
    to both cameras.

## Reproduce

All commands run from the repository root. Use Python 3.11 or newer. The real-clip stage and the Modal backend need
a Modal account and token. Do not put keys in the repository. `.env` is ignored by git.

Set up and run the tests:

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,cloud]"
export PYTHONUTF8=1
export PYTHONPATH="$PWD:$PWD/ros2/intercycle_nav:$PWD/ros2/intercycle_eval:$PWD/ros2/intercycle_sim"
.venv/Scripts/python.exe -m pytest -q
```

Dev worlds (robot development only; seeds 0-3). The script refuses unless the tree is clean and PREREG is on main:

```
.venv/Scripts/python.exe scripts/gen_dev_worlds.py --set robot --out-dir out/dev_worlds --suffix .day_script.json
```

Dev robot batch on the CPU double (dev worlds 0-3 only):

```
.venv/Scripts/python.exe scripts/run_double.py --dev --out out/double/dev1 --worlds 0,1,2,3 --backend modal
```

Fixed margin from the learned-arm dev runs (only before the robot freeze; it refuses to overwrite a frozen file;
`--grid-margin-m 1.0` is the PREREG section 7 (b) limit and must be given, the script has no default):

```
.venv/Scripts/python.exe scripts/write_fixed_margin.py --runs RUN_DIR [RUN_DIR ...] --grid-margin-m 1.0     --out twin/m7_margin.json
```

Real-clip dev dry run (dev clips and dev labels only, never the sealed folder):

```
.venv/Scripts/python.exe scripts/sealed_run.py dev-dry-run --dev-root out/rc_final_dryrun --clips <dev ids> --labels labels/dev
.venv/Scripts/python.exe scripts/sealed_run.py dev-score --dev-root out/rc_final_dryrun --labels labels/dev
```

Sealed real-clip test. Only the custodian runs it, from a detached worktree at the commit named in
`prereg/FREEZE_REALCLIP`. Full steps: `docs/probes/REALCLIP_RUN.md`.

```
git worktree add --detach <dir> <freeze-realclip commit>
cd <dir>
.venv/Scripts/python.exe scripts/sealed_run.py check --stages clips
.venv/Scripts/python.exe scripts/sealed_run.py run --stages clips
.venv/Scripts/python.exe scripts/sealed_run.py score
```

Sealed robot batch. Only the custodian runs it, after `prereg/FREEZE` is committed. `derive` makes the 16 test seeds
from the first drand round after the freeze commit time (`--fallback` uses `secrets.token_hex(16)` and must be
disclosed). `run --stages double` refuses unless every freeze check passes:

```
.venv/Scripts/python.exe scripts/sealed_run.py derive
.venv/Scripts/python.exe scripts/sealed_run.py run --stages double
```

Scoring (the first step is the only read of truth; the second never opens it):

```
.venv/Scripts/python.exe -m twin.eval_metrics --runs RUNS --margin-file twin/m7_margin.json \
    --tasks twin/tasks.yaml --site twin/site.yaml --worlds W1,...,W16 \
    --invalid results/sealed/invalid.json --out results/sealed/h3_tick_counts.json
.venv/Scripts/python.exe scripts/twin_eval.py --runs RUNS --tasks twin/tasks.yaml --split test --engine double \
    --worlds W1,...,W16 --invalid results/sealed/invalid.json --h3-counts results/sealed/h3_tick_counts.json \
    --margin-file twin/m7_margin.json --recall results/sealed/recall_frames.json --r0 0.5272 \
    --out results/sealed/eval.json
```

A reader cannot re-run the sealed steps without the sealed test clips and labels. What a reader can check: the
committed hashes (`sealed/MANIFEST.sha256`, `sealed/LABELS.sha256`), the run log (`results/sealed/RUNLOG.json`), the
pre-registration and its amendments, and the code at the frozen commits.

## Media

These are illustrative. They show how the system works. They are not recordings of the sealed runs.

- Design deck: [[DECK_LINK]]
- Video, how the observer and the robot fit together (illustrative 3D, moving camera): [[VIDEO_LINK_OVERVIEW]]
- Video, one generated day with the observer map: [[VIDEO_LINK_TWIN_DAY]]

Tags on every frame: `REAL` (a real clip, people masked), `GENERATED DAY` (a twin day, not recorded) and
`SIMULATED body shop`. A frame with any generated content never carries `REAL`. All worker imagery is masked.

## Data credit and licence

Clips: Önal, O. and Dandıl, E. (2024), Data in Brief 56, 110791, doi:10.1016/j.dib.2024.110791 [[VERIFY_DOI]].
Data set: Mendeley Data, doi:10.17632/xjmtb22pff.1, through the Hugging Face set Voxel51/Safe_and_Unsafe_Behaviours,
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: trimmed, re-encoded, people masked. Tracks, labels
and figures are derived by InterCycle. We have no affiliation with the plant or the authors, and no endorsement is
implied. The plant is not named in this repository. This is a non-commercial research demo.

Code: MIT (`LICENSE`).

## Where to read more

- `prereg/PREREG_TWIN.md`: the pre-registration (what is fixed and what cannot be amended).
- `prereg/AMENDMENTS.md`: amendments A1-A6 with time and evidence.
- `STATUS.md`: gate log with times and commits.
- `docs/probes/REALCLIP_RUN.md`: custodian commands for the real-clip run.
- `docs/LABEL_AGREEMENT.md`: agreement between the two labeller agents.
