# Unsupervised phase discovery (DIRECTION §4). Everything here is SYNTHETIC.

The observer finds the phases itself, from ceiling-camera observations only (`mapping.Observation`). No PLC, no truth, no
calendar. Code: `intercycle/observer/phases.py`, which never imports truth (`tests/test_no_truth_import.py`;
`tests/test_phases.py` also checks it never reads the calendar). Scoring: `scripts/phase_discovery.py` is the only place
truth is read.

## 0. Audit status (#2, A7). Read this first

**The seed-1 numbers in §3 are tainted.** After the 17:36 run on `final_test`, three things were changed and re-scored on
that same window:
* the line-state naming was patched and the ARI re-scored (17:44–17:49);
* the eval was re-run (17:54);
* the onset metric was redefined (f86093c).

**Headline instead = seed 2, scored once** with the code frozen at 947a878. That run is the ideal shim (σ 0.15 m, 5 %
drop), train 04:00–10:00, test 13:30–20:00 (14:00 shift change, breaks at 17:00 and 19:30, andons).
`out/phases/results_s2_train__s2_test.json`.

Welders, online causal HMM, labels weld / tip_dress / return / idle:

| Machine | Seg F1 | Boundary err, median (s) | Onset err (s), resting→working | Onset err (s), v1: active start → bin 0 |
|---|---|---|---|---|
| S10 | 0.99 | 0.5 | 0.5 | 0.5 |
| S20 | 0.97 | 0.5 | 0.5 | 0.5 |
| S30 | 0.84 | 0.5 | 0.5 | 0.5 |
| S40 | 0.98 | 0.5 | 0.5 | 0.5 |
| S50 | 0.74 | 1.0 | 1.0 | 1.0 |
| S60 | 0.67 | 0.5 | 0.5 | 0.5 |
| S70 | 0.78 | 0.5 | 0.5 | 0.5 |
| **Mean** | **0.85** | **0.57** | | |

Offline DTW mean for the welders is 0.87.

Other machines:

| Machine | Seg F1 | Boundary err, median (s) | Onset err (s), resting→working | Onset err (s), v1 |
|---|---|---|---|---|
| HEM1 | 0.78 | 0.5 | 0.5 | 0.5 |
| SEAL1 | 0.61 | 2.0 | 1.5 | 72 |

S80 is left out of the headline. Its 1.00 is tautological: the shim draws the carriage only below H_CLEAR, which is
exactly the on_floor label.

Line states: ARI 0.35, purity 0.967, named accuracy 0.973 (k = 6; running splits into 4 clusters).

| State | Recall |
|---|---|
| running | 0.99 |
| andon | 0.90 |
| break | 0.94 |
| shift_change | 0.70 |

**Not yet scored: seed 2 under `M.MEASURED_SHIM`.** The lead wants it as the headline (measured perception error).
M's refit under measured noise found the 09:00 break named andon on 100 % of ticks: the break crowd is not separable
under that noise. "Stopped vs running" may be the honest level there. Commands are in the hand-off below.

Disclosures:
* **The line-state naming rules encode the simulator.**
  * The andon rule is the simulator's design: the line is overdue while the off-line cells keep working.
  * "break = gathering" holds only because the simulator spawns a break crowd.
  * Chosen and stated: we do **not** merge break and shift_change into planned_stop, because M's map indexes 4 named
    states. The names are therefore *suggestions*.
  * Since A7 the identity is the cluster id (`line_cluster(t)`), and a name never changes the clustering. Each cluster is
    shown as "State A..", with an auto description, until an operator names it (`set_line_label`). Labels persist and
    carry over on refit by Hungarian matching of centroids. Model (Gemma) suggestions are stored as unverified and never
    applied.
* **Scoring maps the discovered bins to truth names using the test truth** (many-to-one majority).
* **K = 8 for every machine.** The elbow never stopped earlier, so every machine has 9 bins.
* **Boundary error is recall-only and capped at 30 s.** It is measured from each true boundary to the nearest predicted
  boundary; extra predicted boundaries count only in the boundary F1.
* **Choices made on dev seed 0:** the energy weight, the gathering thresholds and the variance floor. The S80
  floor-level scoring and the onset v2 metric were added after seeing seed-1 results; both onset definitions are reported.

## 1. Which machines expose named sub-phases in truth

`Kinematics.phase(mid, t)` returns only `("active"|"idle"|"hold"|"trip", frac)`. The scoring script rebuilds named
sub-phases from `phase()` plus the kinematics constants:

| Machines | Kind | Truth sub-phases (fraction f of the active part) | Source |
|---|---|---|---|
| S10–S40 | weld_station | weld f < 0.80, swing_out 0.80–0.86, dress 0.86–0.92, return 0.92–1.0, idle | `T_WELD_END`, `T_DRESS_END` (+ the docstring's 0.86 swing/dress split) |
| S50 | framing | same | same |
| S60, S70 | respot | same | same |
| S80 | unload_lifter | lower 0.10–0.30, down 0.30–0.55, raise 0.55–0.80, carriage overhead otherwise | `lift_height()` breakpoints |
| HEM1 | roller_hemming | approach (to the first path point, `arm_specs[].brk[1]`), path to 0.92, return 0.92–1.0, idle | `_timeline_path` |
| SEAL1 | sealer | same | same |
| AGVDOCK | not a machine in `phase()` (a parked vehicle, part of L2) | — | not scored |

**Every machine exposes named sub-phases.** All of them are scored on sub-phases **and** on cycle boundaries (onset error).

Scoring sets:
* **Weld family (S10–S70): weld / tip_dress / return / idle.** tip_dress = 0.80–0.92 (swing-out + dress). This is
  DIRECTION's "weld / tip-dress / transfer / wait". The 5-label set that splits swing_out from dress is reported as "fine".
* **S80: on_floor / clear.** The carriage is on the floor while `lift_height(f) < H_CLEAR` (2.2 m). In plan view
  lower / down / raise look **identical**: the carriage circles do not move, only the height does. A ceiling camera
  therefore cannot see the 4-way split; its fine-label F1 (0.47) is reported only for completeness.
* **HEM1, SEAL1: approach / path / return / idle.**

## 2. Method

**Per machine** (`MachinePhases.fit`), with no machine-specific settings:
1. **Signal.** Each 2 Hz tick, per machine zone: motion energy (number of moving machine-part detections) plus a 4 × 4
   occupancy grid of all part detections (where the arms are). log1p, standardised, PCA to 6 dimensions.
   * A frame where the camera is blacked out counts as missing.
   * So does a frame where every part is flagged moving at once (a frame-differencing glitch: robot bases never move).
2. **Period.** `learn_cycles.learn` on the motion signal, cross-checked against the ACF of the latent signal (the ACF
   is the fallback). S80's carriage only appears and disappears, so `learn_cycles` has no confidence there and the ACF
   is used.
3. **Cycle motif.** Matrix profile of the first latent dimension over windows with activity (`stumpy.stump`, BSD-3;
   numpy fallback), computed on the first 60 periods.
4. **DTW alignment.** Anchors are minima of the distance profile to the motif, kept only if they contain a cycle's worth
   of activity, so a stop is never a cycle. Every anchor-to-anchor cycle gets an open-end DTW to the template
   (steps: repeat / advance / skip). Then the template is rebuilt as the per-position **median** (robust DBA), with a
   rotation so that position 0 is the motion onset. The onset is found where the template stops looking like the
   machine at rest. The resting look is the mode of the frames inside long debounced still runs (real stops); the
   positions that look stopped are those on the near side of an Otsu split of the log-distance to it.
5. **Sub-phases.** The trailing at-rest positions form the explicit **idle** sub-phase. The rest of the template gets
   exact optimal change points (dynamic programming with the L2 cost of `ruptures.Dynp`, written in numpy). The cost
   space is the whitened template plus motion energy (weight 3, `SEG_MODE = "whiten+e3"`). Motion energy is what
   separates still sub-phases (weld dwell, tip dressing) from moving ones (swing, return): the dresser and the robot
   base share an occupancy cell. The weight was chosen on the dev window (seed 0); the numbers below are on seed 1.
   There are 4–8 sub-phases including idle, chosen by an elbow on the cost; every machine ended at 8. So each machine has
   **9 bins: 7 active sub-phases + idle + blocked**. The long weld stretch is split into weld-point groups, which
   scoring maps many-to-one onto "weld".
6. **Online decoder (causal).** A left-to-right HMM over the template positions:
   * Moves: stay / advance / skip with probability 0.15 / 0.70 / 0.15. The last position wraps to 0 or enters
     **BLOCKED**, an explicit state for "the idle lasts longer than the cycle allows" (andon, break, micro-stop).
     BLOCKED leaves back to the onset.
   * Emissions: diagonal Gaussians per position, with the variance of that position's sub-phase.
   * Forward filter only, so it is causal. A numba kernel (numpy fallback) costs about 0.7 ms per tick for all 10
     machines plus the line state.
   * This is the "4–8 state HMM": each discovered sub-phase is a chain of template positions (a segmental HMM with
     learned durations).

**Whole line** (`LineStates`):
* Causal per-tick features:
  * index overdue: time since the last skid motion, in learned line periods;
  * share of line stations working and share blocked;
  * share of off-line cells working (90 s window);
  * people excess and crowd excess over their own trailing 1 h median, so the shift head count does not split running
    into day and night clusters;
  * share of people moving.
* Clustering: k-means++ on the standardised features, k ∈ {4, 5, 6} chosen by silhouette (k = 4 was chosen).
* Naming, by rules on the cluster centroids:
  * most line stations not blocked → running;
  * line blocked while the cells still work → andon;
  * line and cells stopped → a planned-stop cluster. Each tick in it is named break if it shows a gathering
    (≥ 3 extra people or ≥ 1 extra crowding), else shift_change.
* **The calendar is never an input.** The truth line state (calendar for break / shift_change, overdue line index for
  andon) is used only in the scoring script.

## 3. Results on seed 1 (TAINTED, see §0; kept for the record)

Setup:
* Train: seed 1, 04:00–10:00. It contains 19 andons, the 06:00 shift change and the 09:00 break.
* Test: seed 1, 13:30–20:00, a window the model never saw. It contains the 14:00 shift change, the 17:00 break
  (15 min), the 19:30 break (30 min) and 19 andons.
* Camera: 2 Hz `CameraShim` after M's per-group motion fix (85f2de1). It adds 0.15 m noise, 5 % dropouts, 10 s
  blackouts (about 1.3 % of frames) and 0.2 s latency.
* Motif engine: stumpy. Fit 110 s for 6 h of data; decode 8 s for 6.5 h.
* Reproduce: `python scripts/phase_discovery.py sim --name final_train --seed 1 --t0 4 --t1 10`, the same with
  `--name final_test --t0 13.5 --t1 20`, then `eval --train final_train --test final_test --engine stumpy`, then
  `table out/phases/results_final_train__final_test.json`.

How the numbers are computed:
* Discovered bins are named many-to-one by the truth label they overlap most. This is the standard unsupervised mapping
  and is used in scoring only.
* **Segment F1:** a true segment counts as found if a predicted segment with the same name overlaps it with temporal
  IoU ≥ 0.5, matched one-to-one.
* **Boundary error:** for every true boundary, the distance to the nearest predicted boundary. At 2 Hz one tick is
  0.5 s.
* **Onset error:** true resting → working (the first frame a floor camera can see) vs predicted idle/blocked → any
  other bin.

**Headline = online (causal HMM, what M gets through `phase_bin`).** The offline DTW labelling is non-causal and is
shown as supporting evidence.

### 3.1 Welders: named sub-phases weld / tip_dress / return / idle

| Machine | Period (s) | **Online seg F1** | Online frame acc | **Online boundary err, median (s)** | Online boundary err, mean (s) | Boundary F1 ±1 s | Offline DTW seg F1 | Fine-label (5-way) online F1 | Cycle onset err, median (s) |
|---|---|---|---|---|---|---|---|---|---|
| S10 | 60.12 | **0.97** | 0.98 | 0.5 | 0.43 | 0.98 | 0.99 | 0.98 | 0.5 |
| S20 | 60.14 | **0.92** | 0.97 | 0.5 | 0.47 | 0.94 | 0.99 | 0.94 | 0.5 |
| S30 | 60.16 | **0.84** | 0.93 | 0.5 | 1.19 | 0.85 | 0.84 | 0.50 | 0.5 |
| S40 | 60.18 | **0.86** | 0.97 | 0.5 | 0.52 | 0.88 | 0.98 | 0.88 | 0.5 |
| S50 (framing) | 60.17 | **0.68** | 0.91 | 1.0 | 1.20 | 0.75 | 0.81 | 0.47 | 1.0 |
| S60 (respot) | 60.12 | **0.80** | 0.97 | 0.5 | 0.47 | 0.93 | 0.93 | 0.81 | 0.5 |
| S70 (respot) | 60.12 | **0.98** | 0.97 | 0.5 | 0.46 | 0.98 | 0.99 | 0.98 | 0.5 |
| **mean S10–S70** | | **0.86** | 0.96 | **0.57** | 0.68 | 0.90 | 0.93 | 0.79 | 0.57 |

Online segment F1 per label:

| Machine | weld | tip_dress | return | idle |
|---|---|---|---|---|
| S10 | 0.96 | 0.99 | 0.98 | 0.96 |
| S20 | 0.87 | 0.99 | 0.99 | 0.87 |
| S30 | 0.99 | 0.99 | **0.00** | 0.98 |
| S40 | 0.76 | 0.98 | 0.99 | 0.76 |
| S50 | 0.80 | 0.80 | **0.00** | 0.73 |
| S60 | 0.90 | 0.79 | 0.76 | 0.81 |
| S70 | 0.96 | 0.99 | 0.98 | 0.96 |

### 3.2 Other machines: named sub-phases that are physically observable, plus cycle boundaries

| Machine | Scored labels | Period (s) | **Online seg F1** | Online frame acc | **Online boundary err, median (s)** | Offline DTW seg F1 | Cycle onset err, median / p90 (s) | Per-label online F1 |
|---|---|---|---|---|---|---|---|---|
| S80 lifter | on_floor / clear | 60.23 (ACF) | **0.99** | 1.00 | 0.0 | 0.96 | 0.0 / 0.0 | on_floor 0.99, clear 0.99 |
| HEM1 | approach / path / return / idle | 45.00 | **0.66** | 0.91 | 1.25 | 0.56 | 0.5 / 0.5 | approach 0.00, path 0.98, return 0.00, idle 0.98 |
| SEAL1 | approach / path / return / idle | 60.04 | **0.81** | 0.93 | 1.0 | 0.64 | 2.0 / 57.5 | approach 0.92, path 0.97, return 0.00, idle 0.93 |

Periods: true takt 60 s (+0.1 %/h drift), HEM1 45 s, SEAL1 60 s. Every discovered period is within 0.4 %.

### 3.3 Line production states (calendar used only to validate)

ARI **0.74**, purity **0.97**, named accuracy **0.974**; k = 4 clusters (running, andon, planned stop, and a second
running cluster that holds the minutes around break crowds).

| True \ predicted | running | andon | break | shift_change | Recall |
|---|---|---|---|---|---|
| running | 37,839 | 357 | 7 | 0 | **0.99** |
| andon | 273 | 2,325 | 0 | 0 | **0.89** |
| break | 47 | 231 | 5,097 | 24 | **0.94** |
| shift_change | 23 | 146 | 89 | 342 | **0.57** |

Precision: running 0.99, andon 0.76, break 0.98, shift_change 0.93. Ticks are at 2 Hz.

How the errors arise:
* **Andon** is named once the line index is 15 % of a period overdue, so the first seconds of every andon read as
  running.
* **Break and shift-change starts** first read as andon. At the calendar start the line is still finishing its cycle
  and the cells finish theirs: up to about 1 min.
* **Shift change** is the weakest state:
  * In this simulation nothing but the stop's duration (5 min vs 15–30 min) and the missing break crowd distinguishes
    it from a break.
  * Duration cannot be known causally.
  * So 15 % of its ticks are named break: the gathering rule fires on some of them (cause not investigated).
  * 24 % are named andon: the cells finish their cycle first.

## 4. What fell back, limits

* **`hmmlearn` and `ruptures` have no Python 3.14 wheels**: the source build fails. Their replacements are written in
  numpy:
  * the HMM forward filter (template-position HMM, numba-accelerated);
  * exact optimal change points (Dynp L2 cost).

  `stumpy` 1.14.1 installed (numba 0.68 / llvmlite 0.50 come with it). Its numba JIT costs 80–110 s once per process
  and the numba cache does not persist on this laptop, so `MOTIF_ENGINE = "numpy"` is the fast option; the tests use it.
  The motif only seeds the template, so the engine matters little.
* **Short sub-phases that look alike are merged on some machines:**
  * return on S30 and S50: S50's long weld dwells make the elbow put a cut inside the weld instead;
  * return and approach on HEM1 and SEAL1: about 3.6 s and less than 1 s of motion near the fixture.

  The rest of each cycle is still segmented correctly. In practice those bins are named tip_dress or path.
* **SEAL1 onset.** SEAL1's discovered last sub-phase (the bin treated as idle) is mostly truth "approach": the rotation
  to the onset was off by a few seconds. The onset metric (idle/blocked → other) therefore misfires (p90 57 s), but frame
  accuracy is 0.93.
* **S80**: lower / down / raise cannot be told apart in plan view (the carriage moves vertically). Only on_floor vs clear
  is observable, and that is what matters for the floor envelope.
* **Camera shim artefacts:**
  * The pre-fix shim flagged every part as moving when any body group changed size. M fixed it in 85f2de1; the final
    data was re-extracted after the fix.
  * Frames where every part moves at once are still dropped defensively.
  * The DBA median guards against partial glitches.
* **Choices made on the dev window, held fixed for the final run:**
  * the change-point energy weight (3);
  * the gathering thresholds (3 people / 1 crowd);
  * the variance floor.

  There is a single held-out seed and window; no multi-seed sweep was done in the timebox.
* Worker routines (P2) were not built.

## 5. Interface (for workstream M)

```python
from intercycle.observer.phases import PhaseModel
pm = PhaseModel.fit(observations, scn)          # iterable of mapping.Observation, a training window (>= ~1 h, ideally with a stop)
pm.update(obs)                                  # online, every tick in time order (causal forward filter)
pm.phase_bin(machine_id: str, t: float) -> int  # 0..n_bins-2 discovered sub-phases in cycle order (idle = n_bins-2),
                                                # n_bins-1 = blocked/stopped, -1 = unknown
pm.n_bins(machine_id: str) -> int               # sub-phases + 1 blocked bin (9 for every machine here)
pm.bin_names(machine_id: str) -> list[str]      # ["p0", ..., "idle", "blocked"]
pm.line_state(t: float) -> str                  # running | andon | break | shift_change ("unknown" before data)
pm.save(path); PhaseModel.load(path, scn)       # fit once, load in worker processes
```

`phase_bin` and `line_state` answer for the latest fed tick with `t_cap <= t` (2 h history). For batch work use
`phases.extract_all(observations, Layout.from_scenario(scn))` then `pm.decode_features(T, X, F)`, which returns
`(bins, line_state_names, line_features, cluster_ids)`.

Cost:
* fit: about 2 min for 6 h of 2 Hz data, 10 machines, single core (with stumpy, plus its one-off JIT);
* online: about 0.7 ms per tick for all machines plus the line state, so 1 week at 2 Hz is about 15 min of single-core
  CPU.

The fitted model from the final run is `out/phases/model_final_train.pkl`.

Not done (DIRECTION §4.3): feeding the discovered bins into L1 binning and comparing map metrics against equal-slice
bins. That is M's integration step and uses this interface.
