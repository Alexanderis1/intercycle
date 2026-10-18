# PREREG v1: press-shop twin, sealed evaluation

Written by lane S1a-prereg (PLAN_TWIN §4 S1a item 5), Sun 18 Oct 2026, for the lead to commit BEFORE any dev-world
run (PLAN_TWIN §4 gate "PREREG v1"). Every value carries its source in brackets:
[PLAN §n] = docs/PLAN_TWIN.md, [U#] = the user's answer in PLAN §0, [STATUS G0] = docs/STATUS.md gate log,
[A1+] = prereg/AMENDMENTS.md, [code: path::name] = the implementation on main at the base of this branch (4afd810),
[lead] = a lead decision under the user's delegation of Sun ~01:40.

**Not amendable** [PLAN §5 "EVAL"]: §3 (primary), §4 (NI margin, gate order and each test, r0, unsafe-free
threshold), §5 (inspection window), §6 (exclusion and re-run rules) and §7 (S7b's α and δ). Later amendments
(prereg/AMENDMENTS.md) may change implementation details only, each with a timestamp and the dev run that prompted it.
No amendment after FREEZE. The 11:15 auditor diffs this v1 against the final version and lists every amendment in the
README.

## 1. Code that implements this pre-registration

| What | Where (function names quoted from the code) |
|---|---|
| EVAL runner, statistics, gates | `intercycle/eval/twin_eval.py`: `run_eval`, `select`, `score_run`, `missing_mission`, `per_world`, `paired`, `signflip_p`, `t_interval`, `poisson_exact`, `clopper_pearson`, `world_bootstrap`, `h3_world_shares`, `check_h3_provenance`, `load_frozen_margin`, `h4_recall`, `evaluate`, `_gate`, `_blocked` |
| Design, cells, re-runs | `intercycle/eval/twin_design.py`: `make_design`, `index_cells`, `find_runs`, `load_classified`, `CONDITIONS`, `K_TEST`, `TEST_DAYS` |
| Constants | `twin_eval.ALPHA = 0.05`, `NI_DELTA_S_PER_H = 30.0`, `ALPHA_TARGET = 0.05`, `H3_SLACK = 0.01`, `MIN_GROUPS = 5`, `BOOT_B = 10000`, `BOOT_SEED = 20261018`, `H3_CONDITION = "learned"`; `twin.schemas.common.MISSION_T_S = 720.0` |
| M-W7 margin (S7b) | `intercycle/eval/conformal_margin.py`: `calibrate`, `calibrate_files`, `stream_units`, `per_world_miss_share`, `write_margin_file`; frozen output `twin/m7_margin.json` |
| Exclusion list | `scripts/classify_invalid.py` (lane/S1a-slice-fix1, not yet on main at 4afd810) |
| Sealed run | `scripts/sealed_run.py` (lane/S1a-slice-fix1, not yet on main at 4afd810) |
| r0 | `scripts/prereg_r0.py` -> `prereg/r0_v1.json` (this lane) |
| Dev worlds | `scripts/gen_dev_worlds.py` (this lane) |
| Reactive baseline space | `ros2/intercycle_nav/intercycle_nav/reactive.py`: `PARAM_SPACE`, `ReactiveConfig`, `candidate_configs`, `space_json`, `space_sha256` |

If the code and this text disagree on a not-amendable item, this text holds and the code is a bug to fix before FREEZE.

## 2. Design

| Item | Value | Source |
|---|---|---|
| Test worlds | K = 16; `K_TEST = 16` | [PLAN §5 Design], [STATUS G0: user, K = 16], [code: twin_design.K_TEST] |
| Test seeds | Derived at FREEZE by the custodian: seed_i = sha256(R ‖ i), i = 0..15, R = randomness of the first drand round published after the FREEZE commit time; fallback `secrets.token_hex(16)`, said in the README; TEST_SEED_COMMIT = sha256(salt ‖ seeds) committed by the custodian. A test seed whose integer value falls in 0..999 is refused. | [PLAN §4 FREEZE], [code: sealed_run.derive_seeds, seed_int, seed_commit] |
| Dev worlds (robot development and tuning) | world seeds 0, 1, 2, 3; robot development and tuning use days 2-8 only | [PLAN §5 Design] |
| Dev worlds (S7b calibration only) | world seeds 100-131 (32 worlds) from the dev namespace (0..999), CPU engine only (observer, no robot). The M-W7 margin is certified on twin worlds, not on real footage. | [lead] |
| Other dev seeds | 900-963 are DYN validation seeds (`twin.daygen.VALIDATION_SEEDS`), not dev worlds of this design | [code: twin/daygen.py] |
| Dev-world generator | `scripts/gen_dev_worlds.py` accepts only the two sets above and refuses any other seed and any run with `INTERCYCLE_SEALED_RUN` set; `twin.rng.check_world_seed` refuses seeds outside 0..999 unless the sealed flag is set | this lane; [code: twin/rng.py] |
| Days | every world (dev and test): days 1-8 observer only on the CPU engine (numpy, no Nav2); test worlds: robot off on days 1-8; days 9-10 scored | [PLAN §5 Design], [code: twin_design.TEST_DAYS = (9, 10)] |
| Visit-count check | the slice asserts that mean visit counts per slice at day 9 match between dev and test worlds within 10 % | [PLAN §5 Design] |
| Time slices | 48 (`--slices 0-47`): the M-W7 calibration must cover every slice the robot can run in | [PLAN §4 S7b], [WEAKNESSES M-W7], [code: conformal_margin CLI `--slices`] |
| Conditions on the CPU double | 5: no-map, static, learned, best reactive, timing oracle (`no_map`, `static`, `learned`, `reactive`, `oracle`) | [PLAN §5 Design], [code: twin_design.CONDITIONS] |
| Sealed double runs | one run = one mission in one zone; both zones on every test world-day, same for every condition: 5 × 16 × 2 × 2 = 320 | [U12], [code: twin_eval module doc] |
| Double compute | Modal cpu = 4, max_rtf 2 | [STATUS G0: user], [docs/probes/H1.md] |
| Mission | one round of the zone's checkpoints (`tasks.yaml` mission_order), fixed denominator T = 720 s of sim time; the robot idles after its round | [PLAN §5 Primary], [code: MISSION_T_S] |
| Isaac (descriptive, no p-values) | 30 missions: 3 conditions (learned, best reactive, static) × test worlds 1-2 × days 9-10 × both zones (24) + test world 3 day 9 both zones × 3 conditions (6); unbalanced, descriptive only | [PLAN §5 Isaac], [U1 Option B] |
| Times | from container PTS, never frame_index/24 | [PLAN §5 Design] |

## 3. Primary endpoint (not amendable)

- Per mission: Y = the checkpoints of the zone's mission_order whose capture passed Inspect (a pass row of an attempted
  checkpoint), set to 0 if the mission had any robot-caused press stop, hold (press restart delayed by robot presence)
  or incident. Declined and unattempted checkpoints score 0. T = 720 s per mission. [PLAN §5], [code: score_run]
- Per world and condition: V_w(c) = ΣY / ΣT × 3600 (inspections per mission-hour), pooled over test days 9-10 and both
  zones. [PLAN §5], [code: per_world(..., "Y")]
- Effect: d_w = V_w(learned) − V_w(best reactive), paired by common random numbers inside each world. [code: paired]
- Test: exact one-sided sign-flip over the 16 worlds (all 2^16 sign vectors; ties count against H1), H1: mean d > 0,
  α = 0.05. [PLAN §5], [code: signflip_p(d, "greater"), ALPHA]
- Published with it: the per-checkpoint outcome table, tripped > held > passed > failed > declined > unattempted
  (E-W12). [code: _outcomes, OUTCOME_ORDER]

## 4. Gatekeeping (not amendable)

Fixed sequence, each one-sided at α = 0.05; a hypothesis is claimed only if it and every earlier one rejected
(`_gate(..., prev)`, `claimed = prev and reject`). While any design cell is unresolved (infrastructure-invalid without a
clean re-run) no gate is evaluated or claimed (`_blocked`, status "not evaluable: unresolved cells"). [PLAN §5], [code:
evaluate]

| # | Hypothesis | Test as implemented |
|---|---|---|
| H1 | Primary (§3) | `signflip_p(d_w, "greater") <= ALPHA` |
| H2 | Non-inferiority on press time: L_w(c) = robot-caused press time lost (stops + holds incl. reset time, plus the remaining T of errored missions) per mission-hour [s/h]; learned − static ≤ δ, **δ = 30 s/h** [U14], [PLAN §5] | e_w = L_w(learned) − L_w(static) − 30; `signflip_p(e_w, "less") <= ALPHA` (H1: mean e < 0) [code: per_world(..., "lost"), NI_DELTA_S_PER_H] |
| H3 | Map unsafe-free share on test days. **α_target = 0.05** (S7b's conformal α, §7). H0: mean miss share ≥ α_target + 0.01 = **0.06** | per world: the unweighted mean over its test-day units of the unit miss share at the frozen M-W7 margin (`stream_units` + `per_world_miss_share`, the calibration's estimand), from the learned runs EVAL scored; e_w = share_w − 0.06; `signflip_p(e_w, "less") <= ALPHA`. A world missing a stream of any of its (day, zone) cells makes H3 not evaluable (fail closed). Robot safety events are descriptive only. [code: h3_world_shares, check_h3_provenance, ALPHA_TARGET, H3_SLACK] |
| H4 | Blind test person recall ≥ r0 (r0 in §4.1) | `h4_recall`: if any camera has fewer than `MIN_GROUPS = 5` test groups (the small-group rule; camera 9 has 4 [A1+]) it rejects only if the lowest per-group recall AND the pooled recall are both ≥ r0, with the frame-level Clopper-Pearson 95 % interval reported as "ignores clustering (only k groups)"; otherwise the one-sided 95 % group-bootstrap lower bound of the pooled recall must be ≥ r0 |

### 4.1 Recall threshold r0 (not amendable)

- **Rule.** r0 is set from the 20 blind DEV frames only, never from test data [PLAN §5]:
  - reference = labeller B's blind set `labels/dev/labels_blind_B.json` (sha256 of the bytes on disk
    `3d5f72efafbee704b5814029b9ecf5eb153ee5ba6201a69120e75fb0e613222d`; scored.json
    `9e68d146bd0807b91b8a1aa119d83325f684fafc24b43e504b40cf90b9d78380`; both recorded in `prereg/r0_v1.json`), on the
    19 frames `labels/dev/scored.json` marks `scored`
    (`dev-f-c9-14` is a byte-and-PTS duplicate of `dev-f-c9-03`) [docs/LABEL_AGREEMENT.md];
    every person box counts, occluded and truncated included;
  - detections = the PERC ingest's stored detector boxes (`intercycle.perception.esk_ingest`, RF-DETR-M), class
    person, not suppressed, conf ≥ 0.4 (`TrackerConfig.activation`, the confidence that starts a track), on the
    sampled frame nearest the label PTS, at most 0.05 s away (else every labelled person of that frame is a miss);
  - matching = greedy one-to-one by IoU, highest first, IoU ≥ 0.5 (`scripts/label_agreement.match`) [PLAN §5
    Real-clip test];
  - **r0 = the one-sided 95 % Clopper-Pearson lower bound** of the pooled recall = Beta.ppf(0.05, k, n − k + 1).
- **The test side uses the same detections, the same matching and the same IoU** on the test labels (same per-clip
  ingest code inside `sealed_run.py`), so H4 compares like with like.
- **Value now** (`python scripts/prereg_r0.py`, inputs on disk at this branch): 54 of 87 labelled persons matched on
  the 19 scored frames, recall 0.621, **r0 = 0.5272**. (All 20 frames: 57 / 93 = 0.613, bound 0.523; reported, not
  used.) The exact counts, hashes and per-frame rows are in `prereg/r0_v1.json`.
- Disclosed: the 20 blind frames are all camera 9 [docs/LABEL_AGREEMENT.md Limits], so r0 is a camera-9 value applied
  to both cameras; 9 of the 20 label PTS are one native frame (0.04 s) from the nearest 12 fps detector frame.

## 5. Press inspection window (not amendable)

A press-shop inspection window is a press stop, or a gap between strokes of at least inside_s + margin, where inside_s =
2 × entry_m / robot_speed_mps + dwell_s (`intercycle.scenario.Inspection.inside_s`) and the margin is the planner's
`margin_s` (`timing.TimingCore.need_s` = enter + dwell + exit + margin). The share of valid inspections that fall inside
ASSUMED stops is reported. [PLAN §5 Design], [code: intercycle/scenario.py, ros2/intercycle_nav/intercycle_nav/timing.py]

## 6. Exclusion and re-run rules (not amendable)

- **Infrastructure-invalid** = the container failed, was pre-empted or ran out of memory before the robot's first
  command; or logs are missing; or Modal/RunPod returned an error code. `scripts/classify_invalid.py` labels episodes
  from the logs alone (rule 1 logs missing, rule 2 provider error code / status FAILED, CANCELLED, TIMED_OUT, rule 3 no
  first command AND an infrastructure sign) and writes `<batch>/invalid.json` once, committed before any metric is
  computed; it refuses if a metric file or the list already exists. [PLAN §5 Rules], [code: classify_invalid.py doc]
- **Every other error is a failure**: Y = 0 with the full T, and the remaining T (T − time of the first error) counts as
  press time lost on H2. Errored episodes are never dropped and the pair is never dropped. This overrides episode.py
  eval_check and stats.py. [PLAN §5 Rules], [code: score_run]
- On the test split a run whose logs are missing or unreadable that classify_invalid does not list is refused
  (EvalError): it must be listed and its cell re-run. [code: select]
- **A re-run repeats ALL conditions** of the affected (world, day, zone); for each cell the first clean attempt is
  scored; a partial re-run is refused on the test split. First-attempt results and the re-run count per condition are
  reported. [PLAN §5 Rules], [code: select, Selection.reruns, first_attempt_per_world]
- Last re-run at 09:55 [PLAN §4 Sealed]; a cell still unresolved then blocks every gate (§4).

## 7. S7b conformal margin (not amendable)

- Risk level **α = 0.05**, confidence **δ = 0.10** (user decision, Sun ~00:45-01:40) [STATUS G0], [PLAN §4 S7b].
- Learn-then-Test at the real fast-layer latency, over all 48 time slices, offline on the calibration dev worlds
  100-131 only (§2); output frozen in `twin/m7_margin.json` and read through `intercycle.margin_inflate.load_margin`.
  If no margin is certified, the fallback is a fixed latency-p99 margin, disclosed [PLAN §6 Fallback].
  [code: conformal_margin module doc]

## 8. Reported outside the gate (descriptive)

[PLAN §5], [code: evaluate]
- Learned − static on the primary with the world-clustered 95 % t interval (`t_interval`); if its point estimate is
  ≤ 0, the README's first screen says that static boundary inspection matched or beat the learned policy.
- Safety outcomes per condition (trips, holds, incidents, contacts) as Poisson rates per mission-hour with exact
  (Garwood) 95 % intervals (`poisson_exact`; E-W11).
- Share of the oracle gain Σ_w(L − R) / Σ_w(O − R), only if Σ_w(O − R) > 0 and its 2.5 % world-bootstrap quantile is
  > 0; otherwise the two paired differences with t intervals.
- The scale +10 % variant and the reset-time sweep (1-15 min).
- Isaac (§2): descriptive table, no p-values.

## 9. Baseline fairness and tuning budget

- Reactive uses the same window planner (SIPP, `timing.TimingCore`) and entry logic as learned; only its windows come
  from its own camera, persisted within the world; no camera history before the world's first mission (by design,
  stated in the claim). Same sensors, zoom head, rules and checkpoints. [PLAN §5], [code: reactive.py doc]
- **Frozen parameter space** [PLAN §4 S5b], [code: reactive.PARAM_SPACE], checked by the G2 auditor:
  - `space_sha256()` = `c4053aaa6b33af1fc49c2fee1022920b2a3941e00df8b00c597d0abaeccdc18a` (hash of `space_json()`:
    the space, fresh_s, max_runs, candidate seed, presence pad 0.81 m, the default and the 8 candidate hashes);
  - git blob of `reactive.py` at this base: `5037499ceb910c9662c6ff8e11b90356c30bff23` (last change 3efae07);
  - PARAM_SPACE: debounce_s {0.0, 0.4, 1.0}; min_still_s {0.5, 2.0, 5.0, 10.0}; quantile {0.05, 0.10, 0.20, 0.35};
    min_at_risk {1, 3, 5}; extra_margin_s {0.0, 2.0, 5.0}; max_gap_s {1.0, 3.0}; use_left_censored {False, True};
  - the 8 tuning candidates = `candidate_configs()` (default first, then 7 seeded draws, CANDIDATE_SEED = 13855);
    their sha256 (first 16 hex): 8e9dbfee6379f212, 14e374e8a03891cc, 73282d8103ac4f1d, 56c0454c09b73870,
    5cc956572ebf4acb, 8f87313279e1a366, 826fc6418108226c, 8ae54cfbbc50c556.
- **Equal tuning budget per arm** [PLAN §5]: 8 configs × 4 dev worlds (0-3, one fixed dev day each) = 32 runs, run as
  2 parallel rounds of 4 configs; the config is chosen by its mean over world-days. Each arm also gets at most 20 dev
  runs, and InterCycle's debug runs count there.
- A separate agent tunes the baseline and never sees InterCycle's dev results. InterCycle lanes do not read
  `results/tuning/baseline` before FREEZE. Both tuning curves and the chosen configs' per-world-day scores are published
  in `results/tuning/`. [PLAN §5]
- Order [PLAN §4]: this PREREG committed → dev worlds generated (`scripts/gen_dev_worlds.py`) → tuning runs → G2 →
  dry run (~08:00) → FREEZE 08:30.

## 10. Real-clip test (§5 "Real-clip test")

- Metrics, n reported for every figure: person precision/recall at IoU 0.5; foot-point pixel error; stroke F1 at
  ±0.25 s tolerance (greedy one-to-one matching); stroke-count error; occlusion → MISSING rate; zone accuracy.
  [PLAN §5]
- Intervals: bootstrap-by-group CIs only if a camera has ≥ 5 test groups. Otherwise per-group recall with n, plus a
  frame-level Clopper-Pearson interval labelled "ignores clustering (only k groups)"; gate H4 then passes only if the
  lowest per-group recall and the pooled estimate are both ≥ r0. **Camera 9 has 4 test groups** (114 clips, 18.2 min),
  so the small-group rule applies; camera 14 has 12 test groups of one clip each (12 clips, 1.2 min). [A1+], [code:
  h4_recall]
- A test framing with no matching train framing gets no fitted H; its clips are scored for P/R and strokes only and
  excluded from foot-point error and zone accuracy, with n reported (under A1+ every test framing has train clips).
  [PLAN §5], [A1+]
- Run once, in an ephemeral Modal container launched by the custodian, through the same per-clip ingest as dev
  (`sealed_run.py`); test labels are made blind by a fresh labeller agent in the custodian session and only their hash
  (`sealed/LABELS.sha256`) enters the repo. Every figure is stated "against independent Claude-agent labels (vision
  model), not human labels", with the inter-agent agreement (docs/LABEL_AGREEMENT.md) as its noise floor. [U14]
- Reported per stratum (safe/unsafe per camera) because the split's safe strata fall short [A1+ lead notes].

## 11. Disclosures so far (collected; each goes to the README or the results)

From prereg/AMENDMENTS.md (A1+ and lead notes):
1. Split A1+: groups are background/lighting/layout clusters without the pHash link; real sessions may be split across
   groups, so near-duplicate footage can sit on both sides of the train/test line.
2. The committed split rule ran once and gave a degenerate split; the split ran twice in total (committed rule, A1+).
3. Pass 2 re-fetched train+dev clips by id from HF (pinned revision, sha-checked), against PLAN §3 step 2's wording.
4. A1+ aggregates (P1/P3) were computed before the test purge.
5. "Viewed" = an agent looked at the pixels (96 ids, kept out of test); 4 clips processed only automatically are in
   test: 7_tr1 (P7), 3_te7, 3_tr42, 6_tr12 (P8).
6. k1_v0 / scale_v0 were fitted before A1+ with sources now dev or test; invalid, refit on A1+ train clips (S2).
7. P5's detector operating points were read off 40 frames of which 12 are now dev.
8. 60/20/20 holds in the unsafe strata but not in the safe ones (camera 9 safe: train 31 / dev 21 / test 1; camera 14
   safe: test 3 of 40); camera-9 test is 109 unsafe / 5 safe and has no class-2 or class-4 clip.
9. 26 test clips belong to byte-identical sets (13 pairs inside test); all 32 byte-identical sets carry different class
   labels.

From docs/STATUS.md (G0) and docs/probes/*.md:
10. P2: no timing information in the clips (time-of-day structure is ASSUMED).
11. P4: presses P9a and P9b LEARNED (camera 9, 1.96 s / 1.69 s); camera-14 presses ASSUMED. P4's trigger floor
    `min_thr` = 4 px/s was set from train data (a probe choice).
12. P5: no vehicles observed in zone-9 train clips; forklifts on camera 14 only (RF-DETR truck ≥ 0.5).
13. P6: 3 framings; the camera-14 scale carries an interval.
14. P7: 2 separate zones, no corridor. P8: every person box is black-boxed whole.
15. CAL points were placed by an agent, not clicked by the user (U14 clause).
16. T-W6 onset gap (user decision). S7a's L1 adapter fails closed in unlearned bins.
17. All labels are made by Claude agents (vision models), not humans; the blind agreement covers camera 9 only and its
    intervals ignore clustering (6 groups) [docs/LABEL_AGREEMENT.md].

From this pre-registration:
18. r0 is a camera-9 value (§4.1) and uses the nearest 12 fps detector frame (up to one native frame off the label).
19. The M-W7 margin is certified on twin worlds 100-131, not on real footage (§2, §7).
20. Isaac is a descriptive physics replicate (30 missions, unbalanced); the primary comes from the CPU double.
