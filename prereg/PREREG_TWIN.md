# PREREG v1: press-shop twin, sealed evaluation

Written by lane S1a-prereg (PLAN_TWIN §4 S1a item 5), finalised by lane PREREG-final, Sun 18 Oct 2026 ~07:50 CEST,
and made commit-ready by lane PREREG-v1 (~08:10 CEST), for the lead to commit on main BEFORE any dev-world run and
before FREEZE_REALCLIP (PLAN_TWIN §4 gate "PREREG v1").
Every value carries its source in brackets:
[PLAN §n] = docs/PLAN_TWIN.md, [U#] = the user's answer in PLAN §0, [STATUS G0] = docs/STATUS.md gate log,
[A1+] = prereg/AMENDMENTS.md Amendment 1, [A-cust] = prereg/AMENDMENTS.md "Custodian on the desktop",
[A2] = prereg/AMENDMENTS.md Amendment 2 (the user decisions of Sun ~07:00 CEST), [A3] = prereg/AMENDMENTS.md
Amendment 3 (the user decisions of Sun ~08:00 CEST and the PREREG-v1 disclosures), [code: path::name] = the
implementation on main at 6b0ea6b, [lead] = a lead decision under the user's delegation of Sun ~01:40.

**Not amendable** [PLAN §5 "EVAL"]: §3 (primary), §4 (NI margin, gate order and each test, r0, unsafe-free
threshold), §5 (inspection window), §6 (exclusion and re-run rules) and §7 (S7b's α and δ, and the fixed-margin
fallback). Later amendments (prereg/AMENDMENTS.md) may change implementation details only, each with a timestamp and
the dev run that prompted it. No amendment of the real-clip part after FREEZE_REALCLIP, and no amendment of anything
after FREEZE (§2.1). The 11:15 auditor diffs this v1 against the final version and lists every amendment in the README.

## 0. What this can and cannot show (plain words)

- **H1 (learned beats the best reactive baseline) holds only inside the assumed stop model.** The twin's press stops,
  their timing and their lengths come from a generator we wrote (twin/daygen.py, twin/dynamics.json); no real clip has
  timing information (P2). A win on the twin says the learned policy uses that stop model better than a camera-only
  reactive rule does. It does not say the same win happens on a real press line.
- **Zone 9 carries the learned evidence; zone 14 is fully assumed.** The camera-9 presses (P9a, P9b) have stroke
  periods learned from train clips (P4); the camera-14 presses are ASSUMED (P4). The gate pools both zones with equal
  weight per world, as `twin_eval.per_world` does (§3) [lead reading of the user's "zone 9 primary"]; the per-zone
  split is reported beside it, and zone 14's share of any effect rests wholly on assumed presses.
- **H3 is a consistency check on the same generator.** The safety margin and the test worlds come from the same
  generator family; H3 checks that the margin keeps its miss share on new worlds of that family. It is not a safety
  proof for a real shop floor.
- **Real-clip figures (person recall, strokes, occlusion) are against Claude-agent labels** (vision models), not human
  labels, with the inter-agent agreement (docs/LABEL_AGREEMENT.md) as their noise floor [U14].
- **Neither robot arm was tuned** (§9): both run their committed default configs. The comparison is equal, but neither
  arm is at its best possible setting.
- **Stop entry is asymmetric, by design of the two arms, and disclosed** (§9): the learned arm never enters a STOPPED
  press; the reactive arm can.
- **How the robot uses the maps, said simply** [A3]: the robot treats the observer's maps as a yes/no "keep out"
  mask, looking 6 s ahead; on top of that it uses timing gates (enter a press area only in a predicted idle window)
  and lower speeds in busy areas. It does not weigh areas by how risky they are (no graded costs). A win or a loss
  is a result of that simple use of the maps, not of the best possible use.
- **The observer learns by clock time only** [A3]: it learns what usually happens in each time-of-day bin. It does
  not learn effects triggered by events (for example "after a forklift passes, the press stops"). That is future
  work, and nothing here tests it.
- **If the robot part is not ready by FREEZE (~10:15), there is no H1-H3 claim at all**: the robot results are then
  descriptive dev results (§2.1). The robot batch's launcher is still being finished (§2.1 "Robot launcher").
- **Nothing was tuned after looking at results** (§9): no tuning run, no threshold chosen from dev or test data.

## 1. Code that implements this pre-registration

| What | Where (function names quoted from the code) |
|---|---|
| EVAL runner, statistics, gates | `intercycle/eval/twin_eval.py` (git blob `61a671af115bbcac309b6be21880136b99951879`): `run_eval`, `select`, `score_run`, `missing_mission`, `per_world`, `paired`, `signflip_p`, `t_interval`, `poisson_exact`, `clopper_pearson`, `world_bootstrap`, `h3_world_shares`, `check_h3_provenance`, `load_frozen_margin`, `h4_recall`, `evaluate`, `_gate`, `_blocked` |
| Design, cells, re-runs | `intercycle/eval/twin_design.py` (blob `b2a2c0c2aa959cbca400ce0b005a9cdf24924026`): `make_design`, `index_cells`, `find_runs`, `load_classified`, `CONDITIONS`, `REQUIRED`, `K_TEST`, `TEST_DAYS` |
| Constants | `twin_eval.ALPHA = 0.05`, `NI_DELTA_S_PER_H = 30.0`, `ALPHA_TARGET = 0.05`, `H3_SLACK = 0.01`, `MIN_GROUPS = 5`, `BOOT_B = 10000`, `BOOT_SEED = 20261018`, `H3_CONDITION = "learned"`; `twin.schemas.common.MISSION_T_S = 720.0` |
| M-W7 margin (S7b) | `intercycle/margin_inflate.py`: `load_margin`, `inflation_spec`, `inflate`; frozen file `twin/m7_margin.json` (here: the fixed-margin fallback, §7) |
| Exclusion list | `scripts/classify_invalid.py`, git blob `b9c293f970fc52c486c388c714c54657f7a46ecb` |
| Sealed run | `scripts/sealed_run.py`, git blob `c6e7d5306b06e3a8083c4db0ace190e9348307fe` (clips stage only, `STAGES = ("clips",)`); container side `scripts/sealed_ingest.py`; image `cloud/modal_perc.py` |
| Robot batch launcher | not on main: `scripts/run_double.py::run_sealed` and `intercycle/eval/twin_batch.py`, called by the `double` stage of `sealed_run.py` (lane double-modal, being finished; §1 must-change item 7) |
| Real-clip scoring | `intercycle/eval/realclip.py` (blob `ad04ba3f889601a70f9b5ed64b092c0fd3e83290`); occlusion `intercycle/observer/occlusion.py` (blob `981b697491105f93b20c32c83d31d27334b74852`): `box_hits_roi`, `roi_missing_series`; strokes `intercycle/perception/press_motion.py::detect_strokes` (blob `6fe191a2275975cb95b2bc1e329017809dec1265`) |
| r0 | `scripts/prereg_r0.py` -> `prereg/r0_v1.json` |
| Dev worlds | `scripts/gen_dev_worlds.py` |
| Reactive arm | `ros2/intercycle_nav/intercycle_nav/reactive.py` (blob `5037499ceb910c9662c6ff8e11b90356c30bff23`): `ReactiveConfig`, `PARAM_SPACE`, `candidate_configs`, `space_json`, `space_sha256` |
| Learned arm | `ros2/intercycle_nav/intercycle_nav/timing.py` (`0c3555c25bbbfcda748b869ae1642b312ae95ede`), `timing_planner.py` (`2ea9a0e0f3396c0fa01450ae4a1ed7b6d6ae7e13`), `masks.py` (`a9c67f66a2b80272e6e971bf9ae2c24838e0ee54`), `intercycle/robot/predict.py` (`1a55ec74f38777577425b0aebaf2df8a7147be8d`), `intercycle/observer/learn_cycles.py` (`ee28cccf08d9731fab70ce0a2181a4a04f578504`), `twin/observer_live.py` (`a99a99dad5c28ae2c6f32031938258e6ca5db473`), `ros2/intercycle_nav/config/nav2.yaml` (`156409cdb7332bed843b937868ba67c114fda619`) |

All blobs are those at main 6b0ea6b. **Condition on the commit of this file** [lead]: at the main commit that adds
this file, `git rev-parse HEAD:<path>` gives each blob above. If one differs, the blob in this table is updated to the
merged one in the same commit (and the change listed in prereg/AMENDMENTS.md) before any dev-world run; the rules
quoted in §2 (seed derivation), §6 (exclusion) and §9 (default configs) must still read as written here. The learned
and reactive blobs are re-recorded at FREEZE (§9).

**Code that must change to match this text** (the text holds, PLAN §5; each is a bug until fixed).

Must be merged **before FREEZE_REALCLIP** (target ~09:15, at the latest 10:15 CEST, §2.1), because the real-clip
run executes them. They touch only real-clip paths of §2.1 (the "Covers" row), and every real-clip path is frozen by
that commit:
2. `scripts/sealed_run.py` (main: blob `c6e7d530…`) reads one `prereg/FREEZE` for every stage and requires seeds;
   the clips stage must be gated by `prereg/FREEZE_REALCLIP` alone and run without seeds, the robot stages by
   `prereg/FREEZE`, and the seeds must be derived only after `prereg/FREEZE` (§2.1). The clips stage (and its
   scoring command) must refuse unless no path changed since the FREEZE_REALCLIP commit (it runs from a detached
   worktree at that commit), and the robot stages must refuse unless every real-clip path of §2.1 has the same blob
   at FREEZE_REALCLIP and at FREEZE. It must also carry the sealed real-clip scoring (`sealed_run.py score`: the
   dev scorer `intercycle/eval/realclip.py::run` on the LABELS-checked test labels, `twin/dev_labels.py`). [lane
   RC-final implements this: `preflight_clips`, `import_closure`, `CLIPS_STAGE_ROOTS`, `REALCLIP_DATA_PATHS`, `score`]
3. `intercycle/eval/realclip.py`: `FRAME_TOL_S = 1/12 s` (main and lane RC-final); the H4 test side must use the r0
   rule's nearest-frame tolerance of 0.05 s (§4.1), or the r0 rule and the test side differ.
4. The real-clip occlusion figure (`intercycle/eval/realclip.py`, `intercycle/observer/pixel_adapter.py`) must use the
   image-space ram-ROI rule (§10, `intercycle/observer/occlusion.py`), and the PERC module
   `intercycle/perception/press_motion.py::detect_strokes` must emit the BDC (bottom) time of each stroke (§10)
   [lane RC-final: `_bdc_index`].
6. The H4 frames file that `twin_eval.evaluate(..., recall_frames=...)` passes to `twin_eval.h4_recall` must be
   written from the sealed scoring output of item 2 (one row per labelled test frame: camera, group, tp, n_gt); no
   other part of `intercycle/eval/twin_eval.py` belongs to the real-clip test.

Must be merged **before FREEZE (~10:15)**:
1. `twin_design.make_design` enforces the five-condition design on the sealed CPU double (`cs != CONDITIONS`); it must
   enforce the cut design of §2 (static, learned, reactive; 16 × 2 × 2 × 3 = 192 cells).
5. No writer enforces the fixed-margin rule of §7 (`conformal_margin.write_margin_file` refuses non-LTT results):
   `scripts/write_fixed_margin.py` must compute margin_m exactly as §7 states and write `twin/m7_margin.json`.
7. The robot batch has no sealed launcher on main (`scripts/sealed_run.py` on main: `STAGES = ("clips",)`). The
   `double` stage of `sealed_run.py` (directly, as in lane double-modal, or through `scripts/sealed_robot.py`, as lane
   RC-final dispatches it) must call `scripts/run_double.py::run_sealed` (plan, order, re-runs, layout and
   cost in `intercycle/eval/twin_batch.py`) with the 16 revealed seeds, and `run_double.SEALED_BACKEND_READY` must be
   true; both are being finished in lane double-modal (there `SEALED_BACKEND_READY = False`, so the stage refuses).
   Lanes double-modal and RC-final both change `sealed_run.py` (a real-clip path): their `sealed_run.py` changes
   must be reconciled and merged before FREEZE_REALCLIP (§2.1 "Shared files").
   If this item is not merged and dry-run on dev worlds by 10:15, §2.1 "not ready" applies.

If the code and this text disagree on a not-amendable item, this text holds and the code is a bug to fix before the
freeze that covers it.

## 2. Design

| Item | Value | Source |
|---|---|---|
| Test worlds | K = 16; `K_TEST = 16` | [PLAN §5 Design], [STATUS G0: user, K = 16], [code: twin_design.K_TEST] |
| Test seeds | Derived **after FREEZE (the robot freeze, §2.1)** by the custodian agent: seed_i = sha256(R ‖ i), i = 0..15, R = randomness of the first drand round published after the FREEZE commit time; fallback `secrets.token_hex(16)`, said in the README; TEST_SEED_COMMIT = sha256(salt ‖ seeds) committed by the custodian agent. **Encoding**: R = the bytes of the drand `randomness` hex string (`bytes.fromhex`); i = 4-byte unsigned big-endian (`int(i).to_bytes(4, "big")`); seed_i = the 32-byte digest, written as hex; the integer world seed = the first 8 bytes of seed_i read big-endian (`int(seed_hex[:16], 16)`); a value ≤ `DEV_SEED_MAX` = 999 (the dev range 0..999) is refused; TEST_SEED_COMMIT = sha256 hex of (salt bytes ‖ seed_0 bytes ‖ … ‖ seed_15 bytes). No test seed exists before FREEZE; FREEZE_REALCLIP does not derive seeds. | [PLAN §4 FREEZE], [A2], [code: sealed_run.seed_digest, derive_seeds, seed_int, seed_commit, N_SEEDS = 16; twin/rng.py DEV_SEED_MAX] |
| Dev worlds (robot development) | world seeds 0, 1, 2, 3; robot development uses days 2-8 only; days 9-10 of dev worlds exist for the dry run and the visit-count check | [PLAN §5 Design] |
| Dev worlds 100-131 | **not run**: the 32-world S7b calibration is replaced by the fixed-margin fallback (§7). `scripts/gen_dev_worlds.py --set calib` stays in the code but is not used for this evaluation | [A2] |
| Other dev seeds | 900-963 are DYN validation seeds (`twin.daygen.VALIDATION_SEEDS`), not dev worlds of this design | [code: twin/daygen.py] |
| Dev-world generator | `scripts/gen_dev_worlds.py` needs `--set robot` (or `--set calib`, unused) on every run (`--seeds` only as a subset of that set), refuses any other seed and any run with `INTERCYCLE_SEALED_RUN` set, and (`prereg_gate`) refuses unless the working tree is clean, this file is tracked and unmodified at HEAD, the commit that added it is on main, and HEAD's copy equals main's; `dev_worlds.json` records the set, HEAD, the main commit, that commit, this file's git blob and sha256. `twin.rng.check_world_seed` refuses seeds outside 0..999 unless the sealed flag is set | [code: scripts/gen_dev_worlds.py, twin/rng.py] |
| Days | every world (dev and test): days 1-8 observer only on the CPU engine (numpy, no Nav2); test worlds: robot off on days 1-8; days 9-10 scored | [PLAN §5 Design], [code: twin_design.TEST_DAYS = (9, 10)] |
| Zones | zone 9 (camera 9, presses P9a / P9b LEARNED) and zone 14 (camera 14, presses ASSUMED); both on every test world-day | [U12], [STATUS G0 P4] |
| Visit-count check | the slice asserts that mean visit counts per slice at day 9 match between dev and test worlds within 10 % | [PLAN §5 Design] |
| Conditions on the CPU double (**cut design**) | 3: `static`, `learned`, `reactive` (best reactive = the reactive arm at its committed default config, §9). **No timing oracle, no no-map condition** | [A2], [code: twin_design.REQUIRED = ("static", "learned", "reactive")] |
| Sealed double runs | one run = one mission in one zone, same for every condition: 3 conditions × 16 test worlds × days 9-10 × zones 9 and 14 = **192 cells** | [A2] |
| Double compute | Modal cpu = 4, max_rtf 2 | [STATUS G0: user], [docs/probes/H1.md] |
| Mission | one round of the zone's checkpoints (`tasks.yaml` mission_order), fixed denominator T = 720 s of sim time; the robot idles after its round | [PLAN §5 Primary], [code: MISSION_T_S] |
| Isaac, movers, A-lite | **descriptive only, outside `sealed_run.py`**, no p-values, never part of a gate or of the sealed claim: Isaac physics replicate (planned 30 missions, 3 conditions, unbalanced), the mover variant (S5a replayed train movers) and A-lite (S8 body shop) | [A2], [PLAN §5 Isaac], [U1 Option B], [U9] |
| Times | from container PTS, never frame_index/24 | [PLAN §5 Design] |

### 2.1 Split freeze [A2]

| | FREEZE_REALCLIP | FREEZE |
|---|---|---|
| File | `prereg/FREEZE_REALCLIP` = the 40-hex sha of the commit it freezes | `prereg/FREEZE` = the 40-hex sha of the commit it freezes |
| When | target **~09:15 CEST**, at the latest **10:15 CEST** (user allowance, Sun ~08:00, [A3]; it was 08:30 under [A2]) | ~10:15 CEST |
| Covers | the real-clip test, frozen as these exact paths (the "real-clip paths"). **Code the run executes**: the sealed run `scripts/sealed_run.py` (clips stage and `score`) and its container side `scripts/sealed_ingest.py`; the scorer `intercycle/eval/realclip.py` and `intercycle/observer/pixel_adapter.py`; the PERC modules `cloud/modal_perc.py` (image), `intercycle/perception/esk_ingest.py`, `intercycle/perception/detector.py`, `intercycle/perception/press_motion.py`, `intercycle/perception/tracking.py`, `intercycle/perception/vehicles.py`; the modules these import: `intercycle/observer/occlusion.py`, `intercycle/observer/mapping.py`, `intercycle/observer/watchdog.py`, `intercycle/observer/learn_cycles.py`, `intercycle/framing.py`, `intercycle/scenario.py`, `intercycle/types.py`, `intercycle/esk.py`, `twin/dev_labels.py`, `twin/rng.py`, and the schemas `twin/schemas/*.py` (among them `labels.py` and `prereg.py`; all of `twin/schemas` because `twin/schemas/__init__.py` imports every schema module), plus the `__init__.py` of `intercycle`, `intercycle/eval`, `intercycle/observer`, `intercycle/perception`, `twin`, `twin/schemas`; and **only the function `h4_recall`** of `intercycle/eval/twin_eval.py` (the gate H4). **Data**: `prereg/r0_v1.json`, `sealed/MANIFEST.sha256`, `sealed/LABELS.sha256`, `prereg/ACTIVE_SPLIT` and the split csv it names, `docs/probes/S3a-P4_rois.json`, `twin/site.yaml`, `labels/TEST_RULES.md`. The list is the static import closure of the roots `sealed_run.py`, `sealed_ingest.py`, `modal_perc.py`, `esk_ingest.py`, `realclip.py`, `pixel_adapter.py`, `dev_labels.py` (the roots of lane RC-final's `CLIPS_STAGE_ROOTS`; walked by lane PREREG-v1 over the static imports at lane/RC-final a7a8cdd, ~08:05; RC-final's `sealed_run.import_closure` recomputes it); the FREEZE_REALCLIP commit message lists every path with its blob, recomputed at that commit, and a path the recomputation adds is covered too | the robot part: twin generator and dynamics (G2 hashes), CPU double, both robot arms at their default configs (§9), the fixed M-W7 margin (§7), EVAL (`twin_eval.py` except `h4_recall`, `twin_design.py`), `classify_invalid.py`, the robot launcher (`run_double.py`, `twin_batch.py`, the `double` stage of `sealed_run.py`) |
| Run | once, **right after the FREEZE_REALCLIP commit**, from a detached worktree at that commit: `sealed_run.py run --stages clips` (ephemeral Modal container launched by the custodian agent, test clips only), then `sealed_run.py score` | the 192 sealed CPU-double cells, after the seeds, through the `double` stage (§2.1 "Robot launcher") |
| Seeds | none | derived only after the FREEZE commit (§2 "Test seeds") |

- No change to the real-clip code or inputs after FREEZE_REALCLIP; no change to the robot code or inputs after FREEZE.
  The FREEZE commit contains the FREEZE_REALCLIP commit (it is an ancestor).
- **Shared files.** The real-clip run is made from a detached worktree at the FREEZE_REALCLIP commit, so it runs
  exactly the frozen code. `scripts/sealed_run.py` is a real-clip path, so after FREEZE_REALCLIP it cannot change:
  any wiring of the robot `double` stage inside `sealed_run.py` (lane double-modal changes it) must be merged before
  FREEZE_REALCLIP, and later robot work goes only into files outside the real-clip paths (`scripts/run_double.py`,
  `intercycle/eval/twin_batch.py`, and `scripts/sealed_robot.py`, the module lane RC-final's `sealed_run.py`
  dispatches the robot stages to). `intercycle/eval/twin_eval.py` is shared only through `h4_recall`, which must
  stay textually unchanged after FREEZE_REALCLIP (the auditor diffs it against that commit).
- **Robot code frozen early** (a consequence, disclosed): `intercycle/observer/learn_cycles.py` (learned arm),
  `intercycle/observer/mapping.py`, `intercycle/observer/occlusion.py`, `intercycle/scenario.py` and
  `twin/schemas/*.py` are real-clip paths, so from FREEZE_REALCLIP on they cannot change for the robot part either
  (the robot stages refuse when a real-clip path differs between FREEZE_REALCLIP and FREEZE).
- **Robot launcher** [A3]: `scripts/sealed_run.py` on main has no robot stage (`STAGES = ("clips",)`). The sealed
  192-cell batch is launched by the `double` stage, which calls `scripts/run_double.py::run_sealed` with the 16
  revealed seeds (`intercycle/eval/twin_batch.py`: plan, order, re-runs, layout, cost); both are being finished in
  lane double-modal (§1 must-change item 7). **If that launcher is not merged and dry-run on dev worlds by 10:15**,
  the robot part is "not ready" (bullet "If the robot part is not ready by 10:15" below): descriptive dev results,
  no H1-H3 claim.
- **The real-clip outputs go only to the custodian agent until FREEZE is committed** (or until the lead declares the
  robot part descriptive at 10:15): the lead receives only "run completed" and the output hash, so the real-clip
  outcomes cannot inform any robot-part choice (margin, bug-fix amendments, the go/no-go on FREEZE). Any amendment
  between the real-clip run and FREEZE names the dev run that prompted it.
- **If the robot part is not ready by 10:15** (among the causes: the robot launcher of §1 item 7 not merged and
  dry-run on dev worlds), FREEZE is not committed for the robot part, no test seed is derived and
  no sealed robot run happens: the robot results are reported as **descriptive dev results** (dev worlds 0-3), and
  **no H1, H2 or H3 claim is made**. H4 is then still computed on the real-clip run and reported, but it is not
  claimed either, because a hypothesis is claimed only if every earlier one rejected (§4).
- Last re-run of a robot cell (§6): the time is written by the lead in the FREEZE commit message and in
  `prereg/AMENDMENTS.md` before the seeds exist; a cell still unresolved then blocks every gate (§4).

## 3. Primary endpoint (not amendable)

- Per mission: Y = the checkpoints of the zone's mission_order whose capture passed Inspect (a pass row of an attempted
  checkpoint), set to 0 if the mission had any robot-caused press stop, hold (press restart delayed by robot presence)
  or incident. Declined and unattempted checkpoints score 0. T = 720 s per mission. [PLAN §5], [code: score_run]
- Per world and condition: V_w(c) = ΣY / ΣT × 3600 (inspections per mission-hour), pooled over test days 9-10 and both
  zones. [PLAN §5], [code: per_world(..., "Y")]
- Effect: d_w = V_w(learned) − V_w(reactive), paired by common random numbers inside each world. [code: paired]
- Test: exact one-sided sign-flip over the 16 worlds (all 2^16 sign vectors; ties count against H1), H1: mean d > 0,
  α = 0.05. [PLAN §5], [code: signflip_p(d, "greater"), ALPHA]
- Published with it: the per-checkpoint outcome table, tripped > held > passed > failed > declined > unattempted
  (E-W12) [code: _outcomes, OUTCOME_ORDER], and the per-zone values V_w(c) for zone 9 and zone 14 (descriptive, §0).

## 4. Gatekeeping (not amendable)

Fixed sequence, each one-sided at α = 0.05; a hypothesis is claimed only if it and every earlier one rejected
(`_gate(..., prev)`, `claimed = prev and reject`). While any design cell is unresolved (infrastructure-invalid without a
clean re-run) no gate is evaluated or claimed (`_blocked`, status "not evaluable: unresolved cells"). All of H1-H3 are
computed on the 192-cell design of §2 (16 worlds, both test days, both zones, conditions static / learned / reactive).
[PLAN §5], [A2], [code: evaluate]

| # | Hypothesis | Test as implemented |
|---|---|---|
| H1 | Primary (§3): learned − best reactive | d_w = V_w(learned) − V_w(reactive) over the 16 worlds; `signflip_p(d_w, "greater") <= ALPHA` |
| H2 | Non-inferiority on press time: L_w(c) = robot-caused press time lost (stops + holds incl. reset time, plus the remaining T of errored missions) per mission-hour [s/h], pooled like V; learned − static ≤ δ, **δ = 30 s/h** [U14], [PLAN §5] | e_w = L_w(learned) − L_w(static) − 30; `signflip_p(e_w, "less") <= ALPHA` (H1: mean e < 0) [code: per_world(..., "lost"), NI_DELTA_S_PER_H] |
| H3 | Map unsafe-free share on test days at the frozen fixed margin (§7). **α_target = 0.05** (S7b's α). H0: mean miss share ≥ α_target + 0.01 = **0.06** | per world: the unweighted mean over its test-day units (`episode_s` from the margin file, §7) of the unit miss share at the frozen margin (`stream_units` + `per_world_miss_share`), from the 4 learned runs of each world (days 9-10 × zones 9, 14) that EVAL scored (`check_h3_provenance`); e_w = share_w − 0.06; `signflip_p(e_w, "less") <= ALPHA`. A world missing a stream of any of its 4 (day, zone) cells makes H3 not evaluable (fail closed). Robot safety events are descriptive only. [code: h3_world_shares, check_h3_provenance, ALPHA_TARGET, H3_SLACK, H3_CONDITION] |
| H4 | Blind test person recall ≥ r0 (r0 in §4.1), from the real-clip run (§2.1) | `h4_recall`: if any camera has fewer than `MIN_GROUPS = 5` test groups it rejects only if the lowest per-group recall AND the pooled recall are both ≥ r0, with the frame-level Clopper-Pearson 95 % interval reported as "ignores clustering (only k groups)"; otherwise the one-sided 95 % group-bootstrap lower bound of the pooled recall must be ≥ r0. **Scope**: a group is (camera, test group); the minimum runs over EVERY test group of BOTH cameras with at least one labelled person (n_gt > 0), so camera 14's 12 single-clip groups are included; the pooled recall is over all labelled frames of both cameras; groups with n_gt = 0 are left out of both. |

**Stated in advance for H4** [A1+], [A2]: camera 9 has **4 test groups** (fewer than 5), so the small-group rule
applies. H4 then needs every one of the (up to 4 + 12) per-group recalls to be ≥ r0; **H4 is expected not to be
claimed if any per-group recall is below r0** (one camera-14 single-clip group with recall 0 is enough). This is the
pre-registered rule, not a failure of the run.

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
- **The test side uses the same detections, the same matching, the same IoU and the same 0.05 s nearest-frame
  tolerance** on the test labels (same per-clip ingest code inside `sealed_run.py`), so H4 compares like with like
  (§1 "must change" item 3).
- **Value** (`python scripts/prereg_r0.py`, inputs on disk at lane S1a-prereg): 54 of 87 labelled persons matched on
  the 19 scored frames, recall 0.621, **r0 = 0.5272**. (All 20 frames: 57 / 93 = 0.613, bound 0.523; reported, not
  used.) The exact counts, hashes and per-frame rows are in `prereg/r0_v1.json`.
- **Detector inputs** (they live outside the repo): `prereg/r0_v1.json` records the sha256 of each of the 15 boxes
  files read (`boxes_sha256`); `scripts/prereg_r0.py` refuses a file whose sha256 differs from its line in the
  committed list `docs/perc_sha256.txt`. It also records the ingest code: commit
  225f18d6b25d64913e5ab64c8037bbad705bfa5a (the last change to the ingest code and the hash list) and the git blobs
  of `intercycle/perception/esk_ingest.py`, `intercycle/perception/detector.py` and `cloud/modal_perc.py`; detector
  RF-DETR-M (rfdetr 1.11.1 RFDETRMedium, COCO-pretrained weights, fp32, raw threshold 0.1).
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
- **A re-run repeats ALL three conditions** of the affected (world, day, zone); for each cell the first clean attempt is
  scored; a partial re-run is refused on the test split. First-attempt results and the re-run count per condition are
  reported. [PLAN §5 Rules], [code: select, Selection.reruns, first_attempt_per_world]
- Last re-run at the time fixed in the FREEZE commit (§2.1; it was 09:55 under the single 08:30 freeze [PLAN §4
  Sealed]); a cell still unresolved then blocks every gate (§4).

## 7. S7b safety margin (not amendable)

- Risk level **α = 0.05**, confidence **δ = 0.10** (user decision, Sun ~00:45-01:40) [STATUS G0], [PLAN §4 S7b].
  α is H3's α_target (§4).
- **The fixed-margin fallback is used** [A2], [PLAN §6 Fallback]: the 32-world Learn-then-Test calibration on dev
  worlds 100-131 is not run. M-W7 is a **fixed latency-p99 margin**, fixed exactly here [lead, fixed by committing
  this file; terms from the S7b code on main]:

  k = ceil(age_p99 × v_max / cell_m − 1e-9) (an integer number of cells; the 1e-9 only absorbs float error, as
  `margin_inflate.R2_EPS` does), margin_m = round(k × cell_m, 2) [m];

  the robot then marks every cell offset (di, dj) with di² + dj² ≤ radius2_cells = floor((margin_m / cell_m)² + 1e-9)
  = k² (`margin_inflate.radius2_cells`, `ROUNDING`; a disc of k cells, UNKNOWN cells included). Worked check: with
  v_max = 1.792664 m/s and cell_m = 0.25 m, age_p99 = 0.40 s gives 0.717 m → k = 3 → margin_m = 0.75 m. Where:
  - **age_p99** = `conformal_margin.age_summary(age_s, source)["p99"]` (numpy linear percentile) of the fast-layer
    age the robot holds at each tick (`DAY0_UNIX + t − issue_unix` of the fast layer in use,
    `twin/observer_live.py`, the quantity checked against `ROBOT_FAST_MAX_AGE_S = 30 s`), pooled over **every tick of
    every learned-arm mission on dev worlds 0-3, days 2-8, zones 9 and 14** (the reactive arm has no fast layer);
    a tick with no fast layer is left out and the count is recorded;
  - **v_max** = the largest `sweep.hi` over the three mover speed keys of `twin/dynamics.json` `observed`:
    `forklift.z14.speed_mps` (1.792664), `person.z14.speed_mps` (1.527327), `person.z9.speed_mps` (1.011488), i.e.
    **1.792664 m/s** (forklift, zone 14) at the dynamics file of the FREEZE commit; one value for both zones;
  - **cell_m** = the robot's inflation grid cell, `observer_live` spec `env_res` (default 0.25 m), the `cell_m` that
    `inflate` is called with.
  **Fail closed, no clipping**: (a) margin_m must lie on the H3 grid `conformal_margin.DEFAULT_MARGINS_M`
  (0 to 4 m in 0.05 m steps; `_margin_index` refuses any other value), which every multiple of 0.25 m up to 4 m does;
  (b) margin_m must fit in the observer's L1 grid margin, `twin_spec(margin=1.0)` as `scripts/slice.py::observer_spec`
  calls it (`observer_live` refuses `margin_m > spec["margin"]`), so with v_max above the robot accepts at most
  k = 4, margin_m = 1.0 m, i.e. age_p99 ≤ 0.5578 s. If (a) or (b) fails, no margin file is written and the robot part
  is "not ready" (§2.1): descriptive dev results, no H1-H3 claim. The margin is never cut to fit.
  Disclosed as "fixed latency-p99 margin, not certified". It is computed from dev runs only, before FREEZE, by
  `scripts/write_fixed_margin.py` (not on main yet: §1 "must change" item 5), which computes k and margin_m exactly
  as above, applies (a) and (b), writes `twin/m7_margin.json` and refuses to overwrite a frozen file.
- The fallback is written to `twin/m7_margin.json` in the format `twin_eval.load_frozen_margin` reads
  (`margin_inflate.load_margin`: the margin schema, `frozen: true`, `status: "ok"`, `margin_m`, `cell_m`, the
  `inflation` block of `margin_inflate.inflation_spec`, and `unit.episode_s = 720` = one mission), with a `method`
  block naming the fallback, the p99 age, the speed and their dev-run sources. The robot reads it through
  `load_margin` and applies it with `inflate`; H3 counts misses at that same margin.
- Disclosed: no margin is certified (no LTT bound); H3 is then the only check of the margin, on test worlds of the
  same generator (§0).

## 8. Reported outside the gate (descriptive)

[PLAN §5], [code: evaluate]
- Learned − static on the primary with the world-clustered 95 % t interval (`t_interval`); if its point estimate is
  ≤ 0, the README's first screen says that static boundary inspection matched or beat the learned policy.
- Safety outcomes per condition (static, learned, reactive: trips, holds, incidents, contacts) as Poisson rates per
  mission-hour with exact (Garwood) 95 % intervals (`poisson_exact`; E-W11).
- **No oracle share**: the cut design has no oracle condition, so `evaluate` does not compute it (`if "oracle" in
  conds`); no no-map comparison either. [A2]
- The per-zone values (zone 9, zone 14), the scale +10 % variant and the reset-time sweep (1-15 min).
- Isaac, movers, A-lite (§2): descriptive tables, outside `sealed_run.py`, no p-values.

## 9. Baseline fairness, default configs, no tuning

- Reactive uses the same window planner (SIPP, `timing.TimingCore`) and entry logic as learned; only its windows come
  from its own camera, persisted within the world; no camera history before the world's first mission (by design,
  stated in the claim). Same sensors, zoom head, rules and checkpoints. [PLAN §5], [code: reactive.py doc]
- **No tuning for either arm** (lead decision, Sun ~07:00 CEST, [A2]): both arms run their committed default configs;
  no tuning run is made, the 8 tuning candidates are not run, and nothing is chosen from dev results. Equal for both
  arms and disclosed. PLAN §5's 32-run tuning budget per arm does not apply.
  - **Reactive default** = `ReactiveConfig()`: `{"debounce_s":0.4,"extra_margin_s":2.0,"max_gap_s":3.0,
    "min_at_risk":3,"min_still_s":2.0,"quantile":0.1,"schema":"intercycle.reactive.v1","use_left_censored":false}`,
    sha256 `8e9dbfee6379f212931f6da750c27632f14df8705e270afb0a52da70258c262a` (`ReactiveConfig().sha256()`); space
    hash `space_sha256()` = `c4053aaa6b33af1fc49c2fee1022920b2a3941e00df8b00c597d0abaeccdc18a`; reactive.py blob above.
  - **Learned default** = the committed defaults of the learned arm's code, hashed as the git blobs of §1 "Learned arm"
    (timing.py, timing_planner.py, masks.py, predict.py, learn_cycles.py, observer_live.py, nav2.yaml) at main 6b0ea6b.
  - Bug fixes before FREEZE are allowed only as amendments with the dev run that prompted them; a change of a default
    value is not a bug fix. The FREEZE commit re-records both arms' blobs and the reactive config hash.
- **Stop-entry asymmetry, kept and disclosed** (user decision, Sun ~07:00 CEST, [A2]): the learned arm treats a twin
  press's own confident STOPPED as an idle window only inside the learner's TTL. The TTL is set by
  `intercycle/observer/learn_cycles.py::ttl_s`: ttl [s] = clip(min(0.5 · P · (0.025 / (1.96 · jitter))², 10 / stops_per_h),
  2 s, 300 s), with P the period [s], jitter dimensionless and stops_per_h the stop rate [1/h] (so 10 / stops_per_h is
  in seconds with the constant 10 in s/h); a STOPPED forecast is further capped at 10 s (`learn_cycles.py`, the
  `State.STOPPED` cap in the forecast). The pinned fact is that this 10 s cap is below need_s = 16.5 s
  (ros2/intercycle_nav/test/test_stop_entry.py), so **the learned arm never enters a STOPPED press**; the user-stated
  typical learner TTL is ~2.6 s [A2]. The reactive arm's own-camera still-run (KM) window does, so **the reactive arm
  can enter a STOPPED press**. [code: test_stop_entry.py, twin/observer_live.py tags["stop_entry"]]
- **How the learned arm uses the maps, disclosed (no code change)** [A3]: the observer maps enter the robot as a
  binary keep-out mask over a 6 s look-ahead, plus timing gates (entry only in a predicted idle window, §5) and speed
  limits in busy areas; there are no graded costs. **The observer learns clock-time bins only** (time-of-day
  slices), not event-triggered effects; event-triggered learning is future work.
- Order [PLAN §4], [A2], [A3]: this PREREG committed → dev worlds generated (`scripts/gen_dev_worlds.py --set
  robot`) → dev runs (descriptive, no tuning) → FREEZE_REALCLIP (~09:15, at the latest 10:15) → real-clip run right
  after it → fixed margin (§7) and G2 → FREEZE ~10:15 → seeds (drand) → sealed CPU-double runs.

## 10. Real-clip test (§5 "Real-clip test"), frozen by FREEZE_REALCLIP, run right after it

- Metrics, n reported for every figure: person precision/recall at IoU 0.5; foot-point pixel error; stroke F1 at
  **±0.25 s** tolerance (greedy one-to-one matching); stroke-count error; occlusion → MISSING rate; zone accuracy.
  [PLAN §5], [code: realclip.IOU_MIN, STROKE_TOL_S]
- **Stroke time = BDC** (user decision, Sun ~07:00 CEST, [A2]): `detect_strokes` emits the bottom-dead-centre time of
  each stroke, the label definition (labels mark BDC); the tolerance stays ±0.25 s.
- **Occlusion on REAL clips = the image-space ram-ROI rule** (user decision, Sun ~07:00 CEST, [A2]): a frame is
  MISSING for a press when a person or vehicle box of that frame, grown by `ROI_PAD_FRAC` of its size per side,
  overlaps the press's ram ROI polygon in the same framing (`occlusion.box_hits_roi`), held for `hold_s` seconds of
  container-PTS time after the last overlapping frame (`occlusion.roi_missing_series`). It replaces the floor-plane
  rule for real clips only; the twin keeps its own (floor) rule.
- Intervals: bootstrap-by-group CIs only if a camera has ≥ 5 test groups. Otherwise per-group recall with n, plus a
  frame-level Clopper-Pearson interval labelled "ignores clustering (only k groups)"; gate H4 then passes only if the
  lowest per-group recall and the pooled estimate are both ≥ r0 (§4 H4 "Scope"). **Camera 9 has 4 test groups**
  (114 clips, 18.2 min), so the small-group rule applies; camera 14 has 12 test groups of one clip each (12 clips,
  1.2 min). [A1+], [code: h4_recall]
- A test framing with no matching train framing gets no fitted H; its clips are scored for P/R and strokes only and
  excluded from foot-point error and zone accuracy, with n reported (under A1+ every test framing has train clips).
  [PLAN §5], [A1+]
- **Custodian and labels on the desktop** [A-cust]: the custodian steps (test-clip download to the sealed folder,
  MANIFEST, hashed test sample list, test labellers, LABELS.sha256, seeds after FREEZE, `sealed_run.py`) run on the
  desktop in dedicated custodian and labeller agents whose outputs to the lead carry only ids, counts and hashes. The
  lead session and every builder lane never read the sealed folder. **The seal is procedural on one machine** (agent
  separation, the sealed folder's access log, committed hashes), not a machine boundary as PLAN §4/§6 planned. The test
  labels are made blind by **two fresh labeller agents** (one for person frames, one for stroke windows), both reading
  only `labels/TEST_RULES.md` and the test images; only their hash (`sealed/LABELS.sha256`) enters the repo.
- **Test-label format fix, disclosed** [A3]: the person-frame file `labels_frames.json` had no `stroke_windows`
  field (label contract `twin/schemas/labels.py`). A fresh labeller agent added an empty `stroke_windows` list to it;
  no label was changed; the file's sha256 before and after are in `C:\intercycle_sealed\access.log` (custodian side;
  no builder reads it).
- **P9a test stroke windows: 1 of 13** [A3]: the test split holds only one c9_f0 clip (A1+ "Framings": test c9_f0 1),
  so the stroke labeller marked 1 test stroke window for press P9a, not the 13 planned. P9a's test stroke figures
  rest on that one window and are reported with n = 1; a press with fewer than 10 scored windows is reported in
  `counts.short_presses`, not refused (lane RC-final scoring rule, its AMENDMENTS entry "Sealed real-clip scoring").
- Run once, right after the FREEZE_REALCLIP commit (§2.1), in an ephemeral Modal container launched by the
  custodian agent, through the same per-clip ingest as dev (`sealed_run.py`), then scored once by `sealed_run.py
  score`. Every figure is stated "against independent Claude-agent labels (vision model), not human labels", with the inter-agent agreement (docs/LABEL_AGREEMENT.md) as its noise floor. [U14]
- Reported per stratum (safe/unsafe per camera) because the split's safe strata fall short [A1+ lead notes].

## 11. Disclosures so far (collected; each goes to the README or the results)

From prereg/AMENDMENTS.md (A1+, lead notes, A-cust, A2):
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
10. The seal is procedural on one machine (desktop custodian and labeller agents), not a machine boundary; two test
    labeller agents instead of one [A-cust].
11. Split freeze (FREEZE_REALCLIP planned 08:30, moved to ~09:15 and at the latest 10:15 by user allowance [A3];
    FREEZE ~10:15) and the cut design: 3 conditions, 192 cells, no oracle, no no-map; Isaac, movers and A-lite
    descriptive outside sealed_run [A2].
12. No tuning for either arm; both run their committed defaults [A2].
13. Fixed latency-p99 margin instead of a certified conformal margin [A2].
14. Learned never enters a STOPPED press; reactive can [A2].
15. Real-clip occlusion uses the image-space ram-ROI rule, not the twin's floor rule; strokes are scored at BDC [A2].

From docs/STATUS.md (G0) and docs/probes/*.md:
16. P2: no timing information in the clips (time-of-day structure is ASSUMED).
17. P4: presses P9a and P9b LEARNED (camera 9, 1.96 s / 1.69 s); camera-14 presses ASSUMED. P4's trigger floor
    `min_thr` = 4 px/s was set from train data (a probe choice).
18. P5: no vehicles observed in zone-9 train clips; forklifts on camera 14 only (RF-DETR truck ≥ 0.5).
19. P6: 3 framings; the camera-14 scale carries an interval.
20. P7: 2 separate zones, no corridor. P8: every person box is black-boxed whole.
21. CAL points were placed by an agent, not clicked by the user (U14 clause).
22. T-W6 onset gap (user decision). S7a's L1 adapter fails closed in unlearned bins.
23. All labels are made by Claude agents (vision models), not humans; the blind agreement covers camera 9 only and its
    intervals ignore clustering (6 groups) [docs/LABEL_AGREEMENT.md].

From this pre-registration:
24. r0 is a camera-9 value (§4.1) and uses the nearest 12 fps detector frame (up to one native frame off the label).
25. H1 holds only inside the assumed stop model; zone 14 is fully assumed; H3 is a consistency check on the same
    generator (§0).
26. With 4 camera-9 test groups the small-group rule applies and H4 is expected not to be claimed if any per-group
    recall is below r0 (§4).

From Amendment 3 (prereg/AMENDMENTS.md, [A3]):
27. Custodian on the desktop (procedural seal on one machine) and two test labeller agents (person frames; stroke
    windows) instead of one (repeats 10, with the A3 record).
28. `labels_frames.json` format fix: a fresh labeller agent added an empty `stroke_windows` list; before/after
    sha256 in `C:\intercycle_sealed\access.log`; no label changed (§10).
29. P9a test stroke windows: 1 of 13, because the test split has only one c9_f0 clip (§10).
30. The robot uses the observer maps as a binary keep-out mask (6 s look-ahead) plus timing gates and speed limits in
    busy areas; no graded costs (§0, §9).
31. The observer learns clock-time bins only, not event-triggered effects (future work) (§0, §9).
32. FREEZE_REALCLIP moved from 08:30 to ~09:15 (at the latest 10:15) by user allowance; the real-clip run follows it
    directly (§2.1).
33. No tuning of either arm and no value chosen from results (repeats 12).
34. The robot batch launcher (`run_double.py`, `twin_batch.py`) is not on main at PREREG v1; without it merged and
    dry-run by 10:15, the robot results are descriptive dev results with no H1-H3 claim (§2.1).
35. Freezing the real-clip import closure also freezes `learn_cycles.py`, `mapping.py`, `occlusion.py`, `scenario.py`
    and `twin/schemas` for the robot part from FREEZE_REALCLIP on (§2.1).
36. The fixed margin must fit the observer grid margin of 1.0 m (age_p99 ≤ 0.5578 s) or the robot part fails closed
    (§7).
