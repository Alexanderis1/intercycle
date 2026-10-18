# Amendments to the pre-registration

Each amendment has a timestamp, the evidence behind it, the exact rule change, and every deviation from the plan
(PLAN_TWIN §2: "Amendments carry a timestamp and the evidence behind them"). `prereg/split.csv` is the original
record and is never edited.

## Amendment 1 (A1+): session groups and the train/dev/test split

- **Decided:** Sun 18 Oct 2026, ~00:45 CEST, by the user, on the lead's report (`docs/probes/G0_PROBES.md`, "STOP").
- **Run:** Sun 18 Oct ~01:35 CEST, once, on Modal (app `s3a-stage-esk`, volume `intercycle-esk`), lane S3a-a1.
- **User decision, verbatim:** "A1+ amendment: drop the near-copy (pHash) link and group by background/lighting/layout
  clusters (21 + 36 groups). Byte-identical files (33 found) stay in one group. The ~150 train clips agents already
  looked at stay out of test. Every test camera framing must also exist in train."
- **Risk, disclosed:** real recording sessions may be split across groups, so near-duplicate footage can sit on both
  sides of the train/test line (near-duplicate leakage). The real-clip test numbers must say so.

### Evidence

1. **The committed rule gave a degenerate split.** `scripts/esk_split.py` (commit 55ffd1e, before Pass 1) ran once and
   gave 5 groups in total: c9_g000 (271 clips, dev), c9_g001 (312, train), c9_g002 (22, test), c14_g000 (85, train),
   c14_g001 (1, train). Camera 14 had no dev and no test clips; test was 22 camera-9 clips from one group (4.0 min).
2. **pHash link rates.** The link (≥ min(3, n) frames at Hamming ≤ 6) joined 69,948 of 182,710 camera-9 pairs (38 %)
   and 1,564 of 3,655 camera-14 pairs (43 %). On a fixed CCTV camera the static background dominates the 8×8 DCT, so
   the rule cannot separate sessions.
3. **Cluster counts.** The agglomerative step alone, at the committed cut of 8 grey levels, gives 21 clusters
   (camera 9) and 36 clusters (camera 14). The A1+ run reproduced 21 + 36.
4. **Byte-identical files.** 32 sha256 values are shared by 65 files (33 extra copies), all on camera 9. Several sets
   carry different class labels (for example 0_tr123 / 1_tr55 / 4_te4, classes 0 / 1 / 4). In the A1+ run every set
   was already inside one cluster (0 cluster merges were needed).
5. **Framings are a session marker** (`docs/probes/S3a-P6.md`): all 271 original camera-9 dev clips are framing c9_f1
   and all 312 original camera-9 train clips are c9_f0, so under the committed split camera-9 dev had no train framing.

### Rule change (A1+), implemented in `scripts/esk_split.py --rule a1plus` (the default path is unchanged)

1. **Groups** = the agglomerative clusters per camera (same features, weights, average linkage and cut
   `CLUSTER_DISTANCE = 8.0`), with **no pHash link**. Clusters that share a byte-identical sha256 are merged.
   Group ids are `c<cam>_a<k>`.
2. **Viewed clips.** A group that contains a clip listed in `prereg/viewed_clips.txt` (a clip whose image an agent
   looked at before this amendment) cannot go to test.
3. **Framing.** Every clip is registered on Modal to the P6 framing references (SIFT 3000, ratio 0.75, RANSAC 3 px,
   ≥ 60 inliers, corner displacement ≤ 12 px at 960×540; references 0_te21 → c9_f0, 0_te1 → c9_f1, 3_te1 → c14_f0,
   in that order; a clip matching none would found a new framing). Only the framing id leaves Modal. A group can go to
   test only if every framing it contains already has a TRAIN clip.
4. **Split** = the committed rule 4 unchanged: 60/20/20 by group, strata camera × safe/unsafe, seed 20261017, groups
   largest first, each to the split with the largest clip deficit (ties train, dev, test). The one change: when that
   split is test and the group is not eligible for test (rule 2 or 3), the group goes to the next-largest deficit among
   train and dev.
5. **Output:** `prereg/split_a1.csv` (clip_id, group_id, split, sha256, framing_id). `prereg/ACTIVE_SPLIT` names it, and
   `intercycle/esk.py` reads the active split from that pointer.

### Result

| split | camera | groups | clips | minutes | unsafe clips |
|---|---|---|---|---|---|
| train | 9 | 11 | 361 | 48.32 | 260 |
| train | 14 | 12 | 55 | 7.23 | 38 |
| dev | 9 | 6 | 130 | 20.09 | 91 |
| dev | 14 | 12 | 19 | 1.74 | 9 |
| test | 9 | 4 | 114 | 18.17 | 109 |
| test | 14 | 12 | 12 | 1.22 | 9 |

- **Small-group rule (§5).** Camera 9 has 4 test groups (fewer than 5), so the rule applies there: per-group recall
  with n plus a frame-level Clopper-Pearson interval labelled "ignores clustering (only k groups)". Camera 14 has
  12 test groups, but each holds a single clip (73 s in total).
- **Framings.** All 3 framings now have train clips (c9_f0 252, c9_f1 109, c14_f0 55). Test: c9_f1 113, c9_f0 1,
  c14_f0 12. The registration reproduced P6's framing id on 669 of 669 train+dev clips.
- No viewed clip is in test (66 viewed clips are train, 30 are dev). No byte-identical set spans two groups. 26 test
  clips belong to byte-identical sets (13 pairs inside test); `inventory.parquet` marks them in `dup_of` / `dup_classes`.
- Test is unbalanced on safe/unsafe for camera 9 (109 unsafe, 5 safe) and has no class-2 or class-4 clip.

### Deviations from the plan

1. **Pass 2 re-fetches train+dev ids from HF by script.** PLAN §3 step 2 says agents never call the HF dataset after
   Pass 1. Pass 1 kept no video, so Pass 2 (`cloud/modal_esk.py --action pass2 --split-name split_a1.csv`) downloads
   each train/dev clip by id from the pinned revision 7369ff18 on Modal, checks its sha256 against split_a1.csv and
   refuses any id that is not train/dev. No test id is ever requested.
2. **The split ran twice in total** (the committed rule once, A1+ once). Each run refuses to run again
   (`split.csv` / `split_a1.csv` already on the volume).
3. **The A1+ aggregates (P1/P3) were computed before the purge**, because the purge deletes the test features they need.
   Only aggregates left Modal.
4. **"Viewed" means an agent looked at the pixels (vision).** P5 (62 frames) and P6 (50 median-background sources) are
   viewed. P7 (16 frames, matcher counts only) and P8 (100 frames, detector aggregates only, "no image was viewed by
   anyone" in its report) were processed automatically and stay eligible; 1 P7 clip and 3 P8 clips are in test. The
   user's "~150" was an estimate; the recoverable viewed list has 96 ids. Staging and P4 looked at no frame.
5. **Calibration inputs that are no longer train.** P6 drew its k1 median backgrounds from the original train split:
   under A1+, 8 of the 25 c9_f0 sources and 11 of the 25 c14_f0 sources are dev, and the c9_f0 reference 0_te21 is
   dev. P6's person-height scale fit used all 398 original train frames, 13 of which are now test. Under PLAN §5 ("CAL
   uses only train frames"), `twin/cal/k1_v0.json` and `twin/cal/scale_v0.json` must be refit on A1+ train clips or the
   deviation disclosed. P5's detector operating points were read off 40 frames of which 12 are now dev.

### Purge and local clean-up (PLAN §3 step 2, A1+ steps 4-5)

- **Volume:** the t = 0.5 s frame, the features (pHash, median background, luma), the ffprobe meta and the RF-DETR
  detections of all 126 A1+ test clips were deleted; sha256 and camera/class counts are kept in
  `esk/split/test_manifest_a1.json`. `check_no_test_images` (now over the whole volume, except the id/hash tables in
  `esk/split/`) passed: 2,263 files, 0 test files. The local half of the check (`$INTERCYCLE_DATA` and the session
  scratch root) passed: 619 files, 0 test traces.
- **Desktop files deleted** (ids only; all are P5 detector copies of t = 0.5 s frames, never viewed by an agent):
  `$INTERCYCLE_DATA/p5/frames/<id>.jpg` for 3_te3, 3_te7, 3_tr15, 3_tr17, 3_tr19, 3_tr29, 3_tr37, 3_tr42, 3_tr43,
  6_tr12, 7_te5, 7_tr1, 7_tr14 (13 files). The same 13 ids were removed from `$INTERCYCLE_DATA/p5/detections.json`.
- **Scratch tables scrubbed:** the per-clip registration numbers (inliers, corner displacement) of the 104 test clips
  that P6 had registered were removed from the session's P6 result files (`p6/p6.json`, `p6/p6b.json`).
- Scratch images of P5 and P6 are named by index or framing and show only viewed clips, which are never test.
- **After Pass 2** (565 train+dev clips and 1,130 keyframes on the volume and in `$INTERCYCLE_DATA`), the check was run
  again: volume 3,958 files, local 2,329 files, 0 test traces. `inventory.parquet` equals `split_a1.csv`.

## Amendment 1, lead notes (Sun 18 Oct ~02:25 CEST, lead session intercycle-74, under the user's autonomy grant of ~01:40)

- **Reading of "clips agents already looked at".** The lead confirms the vision reading used by the builder:
  only clips whose pixels an agent viewed (P5's checked frames, P6's median backgrounds; 96 ids) are excluded from
  test. P7 (matcher counts) and P8 (detector-only face-size histogram) processed clips automatically, without an agent
  viewing them. Four such clips are now in TEST and are named here for the record: `7_tr1` (P7), `3_te7`, `3_tr42`,
  `6_tr12` (P8). The split cannot be re-derived after the purge, so this is disclosed, not amended again.
- **Calibration files invalid until refit.** `twin/cal/k1_v0.json` and `twin/cal/scale_v0.json` were fitted before
  A1+. Some k1 sources are now dev, the c9_f0 reference clip `0_te21` is now dev, and the person-height scale fit used
  RF-DETR boxes of 13 clips that are now TEST. Both files are INVALID for CAL v0. Lane S2 refits k1 (per framing,
  including c9_f1) and scale on A1+ TRAIN clips only, and the train-only provenance assertion covers their source ids.
- **Wording fix.** All 32 byte-identical sets carry different class labels; 13 such pairs lie inside test.
- **Per-stratum shortfall (to disclose with the real-clip results).** 60/20/20 holds in the unsafe strata but not in
  the safe ones (camera 9 safe: train 31 / dev 21 / test 1; camera 14 safe: test 3 of 40), because the viewed-group
  rule moved groups out of test. The real-clip test reports per stratum.
- **custodian_fetch.py hardened** (extra-file check by full name, no mkdir in --verify-only, CRLF-normalised manifest
  compare) and `.gitattributes` pins LF for `sealed/*`, `prereg/*.csv` and `prereg/ACTIVE_SPLIT`.
- **Custodian on the desktop (user approval, Sun ~06:25 CEST: "Run the sealed clip test from this machine, you have my
  approval").** No laptop custodian session appeared. The custodian steps (test-clip download to
  `C:\intercycle_sealed`, MANIFEST, hashed test sample list, test labellers, LABELS.sha256, seeds at FREEZE,
  sealed_run.py) now run on the desktop, in dedicated custodian and labeller agents whose outputs to the lead carry only
  ids, counts and hashes. The lead session and every builder lane still never read `C:\intercycle_sealed`. The seal is
  therefore procedural on one machine (agent separation, `C:\intercycle_sealed\access.log`, committed hashes), not a
  machine boundary as PLAN §4/§6 planned. The test labels are made by two fresh labeller agents (person frames;
  stroke windows) instead of one, to fit before FREEZE; both read only `labels/TEST_RULES.md` and the test images.

## Amendment 2 (A2): split freeze, cut design, three user decisions, no tuning

- **Decided:** Sun 18 Oct 2026, ~07:00 CEST. Items 1-5 by the user; item 6 by the lead (session intercycle-74, under
  the user's autonomy grant of ~01:40). Written into PREREG v1 (`prereg/PREREG_TWIN.md`) by lane PREREG-final before
  PREREG v1 is committed on main, so before any dev-world run, any dev robot run and any test seed.
- **Evidence:** the time left before the planned single FREEZE at 08:30 (PLAN_TWIN §4) is too short for the five-
  condition, 320-cell robot design, the 32-world S7b calibration and the tuning runs, while the real-clip part
  (PERC, labels, scoring) is ready earlier. No dev-world or test result was seen before this amendment.

1. **Split freeze (user).** `prereg/FREEZE_REALCLIP` = the sha of the commit that freezes the real-clip test, at 08:30
   CEST; the real-clip test runs once at 09:15 CEST. `prereg/FREEZE` = the sha of the commit that freezes the robot
   part, at ~10:15 CEST. Test seeds come from drand only after the robot FREEZE. If the robot part is not ready by
   10:15 it is reported as descriptive dev results, with no H1-H3 claim (PREREG §2.1).
   *Deviation:* PLAN §4 has one FREEZE at 08:30 for everything; the 09:55 last re-run moves to the time the lead
   writes into the FREEZE commit (PREREG §6).
2. **Cut design (user).** CPU double: 3 conditions (static, learned, best reactive) × 16 test worlds × days 9-10 ×
   zones 9 and 14 = 192 cells; no timing oracle, no no-map. H1-H3 as `twin_eval` implements them on that design; the
   oracle share is not reported. The S7b fixed-margin fallback (latency p99, PREREG §7) replaces the 32-world
   calibration on dev worlds 100-131 (not run). Isaac, movers and A-lite are descriptive only and outside
   `sealed_run.py`. *Deviation:* U12's 320 runs (5 conditions) and PLAN §5's oracle share; S7b's LTT certificate.
3. **Stop entry (user): keep as is and disclose.** The learned arm never enters a STOPPED press (learner TTL ≤ ~2.6 s,
   below need_s); the reactive arm does (own-camera KM still-run window). Pinned by
   `ros2/intercycle_nav/test/test_stop_entry.py` (lane S5b-entry fix1).
4. **Real-clip occlusion (user).** For REAL clips the image-space ram-ROI rule
   (`intercycle/observer/occlusion.py` `box_hits_roi` / `roi_missing_series`) replaces the floor rule; the twin keeps
   its rule.
5. **Stroke time (user).** PERC `detect_strokes` emits the BDC (bottom) time, the label definition; the stroke
   tolerance stays ±0.25 s.
6. **No tuning (lead).** Neither arm is tuned: both run their committed default configs, hashed in PREREG §9
   (reactive `ReactiveConfig().sha256()` = `8e9dbfee6379f212931f6da750c27632f14df8705e270afb0a52da70258c262a`; learned
   = the git blobs of its code at main 6b0ea6b). Equal for both arms and disclosed. *Deviation:* PLAN §5's equal
   tuning budget (8 configs × 4 dev worlds per arm) and the separate baseline tuner are not used.

- **PREREG v1 also records** (same lane, no rule change): the desktop-custodian amendment above (procedural seal on one
  machine, two test labeller agents), the `scripts/sealed_run.py` blob on main (`c6e7d530…`, replacing the
  lane/S1a-slice-fix1 blob `6d87cefc…` that the S1a-prereg draft cited), and the code that must change before FREEZE to
  match the text (PREREG §1).

Lane notes under A2 item 4 and the split freeze (lane RC-freeze-fix1):

- **Real-clip occlusion -> MISSING, limitation (lane RC-freeze-fix1, dev figures, results/dev/realclip_dev.json).**
  Under the image-space ram-ROI rule (USER DECISION Sun ~07:00) the T-W6 claim is partial on the dev clips: inside the
  labelled hidden spans 336/428 PERC samples (0.785) are flagged missing, so 92 (21.5 %) are not, and 31/35 2 Hz
  Observation ticks (0.886) are MISSING, so 4 stay finite: a stroke can be emitted from an occluded ram ROI. On the
  windows' non-hidden ticks 145/292 (0.497) are also MISSING, mostly from the M-W10 watchdog (ok on 487/763 = 0.638 of
  all adapter ticks) and frame 0; the split of that base rate into watchdog, frame-0 and no-flow parts is not computed
  yet. Both rates are reported beside the test figure.
- **Sealed real-clip scoring (lane RC-freeze-fix1).** `scripts/sealed_run.py score` is the pre-registered analysis of
  the real-clip test: same refusals as the clips stage, then the dev scorer (intercycle.eval.realclip.run) on the
  LABELS-checked test labels with split test. Test stroke windows are scored by the dev status rules; reserves
  (`-rN`) count only while their press has fewer than 10 scored windows, in list order, and a press that ends below
  10 is reported (counts.short_presses), not refused. The watchdog's TRAIN keyframes and the inventory are read from
  `$INTERCYCLE_DATA` (outside the repo, so not hash-guarded by the freeze).

## Amendment 3 (A3): real-clip freeze time, robot launcher, exact fixed margin, PREREG-v1 disclosures

- **Decided / recorded:** Sun 18 Oct 2026, ~08:00-08:10 CEST. Item 1 by the user (allowance relayed by the lead,
  session intercycle-74, to lane PREREG-v1 at ~08:00 CEST); items 2-3 by the lead under the user's autonomy grant of
  ~01:40 (fixed by committing PREREG v1); items 4-9 are disclosures with no rule or code change (user decisions of
  ~07:00 and ~08:00 as relayed by the lead). Written into PREREG v1 by lane PREREG-v1 before PREREG v1 is committed on
  main, so before any dev-world run, any dev robot run, any test seed and FREEZE_REALCLIP.
- **Evidence:** the real-clip code is still being merged (lane RC-final: split-freeze checks, sealed scoring, BDC
  strokes); `scripts/sealed_run.py` on main has `STAGES = ("clips",)` (no robot stage), and the robot launcher
  (`scripts/run_double.py`, `intercycle/eval/twin_batch.py`) is on lane double-modal with
  `SEALED_BACKEND_READY = False`. No dev-world, real-clip test or robot test result was seen before this amendment.

1. **FREEZE_REALCLIP time (user).** The real-clip freeze may move from 08:30 up to 10:15 CEST; target ~09:15, with
   the real-clip run (`sealed_run.py run --stages clips`, then `score`, from a detached worktree at the
   FREEZE_REALCLIP commit) right after it. *Deviation:* A2's 08:30 freeze and 09:15 run.
2. **Real-clip paths named in full (lead).** PREREG §2.1 "Covers" lists every path the real-clip run executes (the
   static import closure of sealed_run.py, sealed_ingest.py, modal_perc.py, esk_ingest.py, realclip.py,
   pixel_adapter.py, dev_labels.py, incl. the PERC modules and `twin/schemas`) and, of `twin_eval.py`, only
   `h4_recall`. Consequence, disclosed: `learn_cycles.py`, `mapping.py`, `occlusion.py`, `scenario.py` and
   `twin/schemas` are frozen for the robot part too from FREEZE_REALCLIP on.
3. **Robot launcher and the 10:15 condition (lead).** The sealed robot batch runs through the `double` stage of
   `sealed_run.py` calling `scripts/run_double.py::run_sealed` (`intercycle/eval/twin_batch.py`). If it is not merged
   and dry-run on dev worlds by 10:15, the robot results are descriptive dev results and no H1-H3 claim is made.
4. **Exact fixed margin (lead).** k = ceil(age_p99 × v_max / cell_m − 1e-9), margin_m = round(k × cell_m, 2), with
   age_p99, v_max = 1.792664 m/s and cell_m = 0.25 m as PREREG §7 defines them; fail closed (no file, robot part
   descriptive) if margin_m is off the H3 grid (0-4 m, 0.05 m steps) or above the observer's 1.0 m grid margin
   (`twin_spec(margin=1.0)`), i.e. age_p99 > 0.5578 s. Replaces "ceil(age_p99 × v_max / cell_m) × cell_m" of the
   PREREG-final draft (same value except on float ties).
5. **Custodian on the desktop and two test labeller agents** (record of the A1 lead-notes entry above, unchanged).
6. **`labels_frames.json` format fix.** A fresh labeller agent added an empty `stroke_windows` list to the test
   person-frame file; no label was changed; its sha256 before and after are in `C:\intercycle_sealed\access.log`.
7. **P9a test stroke windows: 1 of 13**, because the test split holds only one c9_f0 clip.
8. **Robot use of the maps.** Binary keep-out mask over a 6 s look-ahead, timing gates and speed limits in busy
   areas; no graded costs. **No event-triggered learning:** the observer learns clock-time bins only (future work).
9. **No tuning** (A2 item 6, repeated): no tuning run, no value chosen from dev or test results.

## Amendment 4 (A4): the twin's Observation uses the real clips' ram-ROI occlusion rule (T-W6)

- **Numbering (integration):** lane twin-roi wrote this as "Amendment 2"; it is renumbered A4 at the merge into
  lane/integration because A2 (split freeze) and A3 (PREREG-v1) already exist. Text otherwise unchanged.

- **Decided:** Sun 18 Oct 2026, ~08:25 CEST, by the user. **Implemented:** lane twin-roi (branch lane/twin-roi).
- **User decision, verbatim:** "Same ROI rule in twin": the twin's observations use the SAME image-space ram-ROI
  occlusion rule as the real clips (a person's image box covering the press's ram ROI makes that press's zone_motion
  MISSING, with the same hold), not the floor rule.
- **Reason (evidence):** on the v1 artifacts the floor rule (a person / vehicle detection within r + 0.5 m of the
  press's danger rect + 1 m) marked zone_motion MISSING on most learning ticks, because operators stand at the
  presses, so the learned observer learned nothing (period 0 on P9a / P9b, confidence 0 on every press).
- **Rule change:** `intercycle/observer/occlusion.RoiOcclusion` (`box_hits_roi`, box padded ROI_PAD_FRAC = 0.15, plus
  the track hold) is the streaming form of `roi_missing_series`, the real-clip ingest's rule (one implementation;
  tests/test_occlusion_missing.py checks equal output). twin/camera.py stat_v1 applies it per press with
  `hold_s_by_press` (camera 9: 34 s, camera 14: 7.5 s, unchanged). The plant's ObsAdapter takes an explicit rule:
  `occlusion="roi"` for stat_v1 turns the floor rule off on every press (`ObsV2Converter(floor_rule=...)`) and keeps
  the camera's MISSING; a press without an ROI is MISSING on every tick (fail closed). `"floor"` stays only for the
  G0 stand-in camera_v0 (stub site) and the body-shop shim. cam_floor_ok and the M-W10 watchdog are unchanged.
- **Measured** (v1 slice, world 0, day 9, the 20 min learning period, 2,400 ticks; `scripts/slice.py --world 0
  --day 9 --zone z14`; "before" = the same code with the ObsAdapter on the floor rule):

  | press | zone_motion MISSING, floor rule (before) | ROI rule (after) | learned period after (s) | dynamics.json period (s) | confidence after |
  |---|---|---|---|---|---|
  | P9a | 100.0 % | 80.0 % | 1.60 | 1.94 (observed) | 0 |
  | P9b | 99.9 % | 0.0 % | 1.54 | 1.69 (observed) | 0 |
  | P14a | 80.1 % | 66.8 % | 4.95 | 4.0 (assumed prior) | 0 |
  | P14b | 78.2 % | 51.1 % | 53.86 | 4.0 (assumed prior) | 0 |
  | P14c | 79.9 % | 43.0 % | 9.87 | 4.0 (assumed prior) | 0 |
  | P14d | 73.7 % | 34.6 % | 26.62 | 4.0 (assumed prior) | 0 |

- **Still open (not tuned here):** every press stays at confidence 0 and the slice's stop-in-Observation assertion is
  still red (the zone-14 trip probe waits 240 s at P14d and sees no stroke). The remaining MISSING is the camera's own
  image-space ROI rule with its hold; on zone 14 only 8-16 % of the finite samples exceed the motion threshold, about
  the idle false-positive rate (0.12), so those presses look idle in this window.

## Amendment 5 (A5): PREREG v1 on main, blob re-record (append-only), user decisions of 18 Oct, disclosures

- **Decided / recorded:** Sun 18 Oct 2026, ~09:40 CEST, lead session intercycle-74, at the main commit that merges
  lane/integration and so adds prereg/PREREG_TWIN.md to main. Before any dev-world run, any dev robot run, any test
  seed, FREEZE_REALCLIP and FREEZE.
- **Why append-only:** PREREG §1 says that if a blob at this main commit differs from the §1 table, the table blob is
  updated in the same commit and the change listed here. The automated permission check blocked edits of the hashes
  inside PREREG_TWIN.md; the user chose, verbatim, "Append-only amendment now": "Merge to main, keep the PREREG text
  untouched, and list every old to new hash in a new AMENDMENTS entry, stated as replacing the table update".
  **This list replaces the §1 table update.** For each path below, the "new" blob is the binding one;
  `git rev-parse HEAD:<path>` at this commit gives it. The PREREG_TWIN.md text is unchanged.

  | path | §1 table blob (old) | blob at this main commit (binding) |
  |---|---|---|
  | intercycle/eval/twin_eval.py | 61a671af115bbcac309b6be21880136b99951879 | cb4336d52d61317caebbf14a900b976bc07aec4a |
  | intercycle/eval/twin_design.py | b2a2c0c2aa959cbca400ce0b005a9cdf24924026 | 323e99b3327f45becf7a2c9ac91071520405ea49 |
  | scripts/sealed_run.py | c6e7d5306b06e3a8083c4db0ace190e9348307fe | 2bbcabf0a1c04139b63f0c5ec9ce7e5bda83492e |
  | intercycle/eval/realclip.py | ad04ba3f889601a70f9b5ed64b092c0fd3e83290 | 95ca740bba59c943f9087d4aa9ee285487722626 |
  | intercycle/observer/occlusion.py | 981b697491105f93b20c32c83d31d27334b74852 | 7ebe879be2eb6f7e82fc5cf252e817d0dd60c003 |
  | intercycle/perception/press_motion.py | 6fe191a2275975cb95b2bc1e329017809dec1265 | 8ff28d13ee9552ad39301f74a8efe3ccc6ce96b5 |
  | ros2/intercycle_nav/intercycle_nav/timing.py | 0c3555c25bbbfcda748b869ae1642b312ae95ede | b879b2e330824294be3b819e4567b0045256c62d |
  | ros2/intercycle_nav/intercycle_nav/timing_planner.py | 2ea9a0e0f3396c0fa01450ae4a1ed7b6d6ae7e13 | ae4c57d66a516fa4f671e295898d439d84538d9d |
  | ros2/intercycle_nav/intercycle_nav/masks.py | a9c67f66a2b80272e6e971bf9ae2c24838e0ee54 | 552c3a083400f690e7e0173d9433543713f6c2d3 |
  | intercycle/observer/learn_cycles.py | ee28cccf08d9731fab70ce0a2181a4a04f578504 | de8d4233aacd7d9d40fda21367b7d8606315811c |
  | twin/observer_live.py | a99a99dad5c28ae2c6f32031938258e6ca5db473 | f8e07701be17ab4aba7ad4f1504adec43d8b532c |
  | ros2/intercycle_nav/config/nav2.yaml | 156409cdb7332bed843b937868ba67c114fda619 | 9342e36d3730996041519581b380a73fe44ab747 |

  Unchanged (table blob = blob at this commit): scripts/classify_invalid.py, reactive.py, intercycle/robot/predict.py.
  The changes come from the lanes merged in lane/integration after the §1 table was written (RC-final,
  double-modal-fix1, observer-fix-fix1, obs-speed, observer-learn, freeze-glue, twin-roi as A4).

- **User decisions of Sun 18 Oct (verbatim answers, CEST, approximate):**
  1. Freezes: "Both at 10:15": FREEZE_REALCLIP and FREEZE on one commit at 10:15 CEST (A3 item 1 allowed up to 10:15).
  2. Stop entry: "Keep as is, disclose".
  3. Real-clip occlusion: "Image ROI rule"; then "Make real clips ROI-only" (the floor rule is off on real clips too).
  4. Twin occlusion: "Same ROI rule in twin" (A4).
  5. Strokes: "Detector emits BDC".
  6. Observer learning (~08:35): "Try days 1-8, else 20 min". Days 1-8 were measured at ~90 s wall per 1,200 s
     simulated, about 14.4 h per world, so they cannot finish before submission: **the observer learns for 20 min
     before each mission** (the fallback the user chose).
  7. "Approve 1200 s" for keep_max_s.
  8. Robot part: "Approve fix (a), still try": the press timing limits are matched to the 2 Hz observation tick
     (jitter and LS/ACF consistency limits) so P9b can be learned. This is the only change to a learning limit; it
     is chosen from the tick period, not from a dev or test result.
  9. Claude vision test: "Yes, after the freeze": a descriptive running/stopped test on dev press crops only, not part
     of any claim.
- **No tuning** (A2 item 6, A3 item 9): apart from item 8, no value was chosen from dev or test results.
- **Disclosures (for the README / results):** two test labeller agents; the `labels_frames.json` format fix (A3
  item 6); P9a has 1 of 13 test stroke windows (A3 item 7); the custodian runs on the desktop (A1 lead notes); the
  observer learns clock-time bins only, so event-triggered effects "cannot show" (A3 item 8); with 20 min learning,
  the v1 slice gave confidence 0 on every press before fix (a) (A4 "Still open").

### A5 erratum (same day, ~09:45 CEST, lead): five rows of the A5 table corrected, checked against HEAD of d2b3a56

The A5 table was computed from a working tree that did not match the merge commit. Checked with
`git rev-parse d2b3a56:<path>` for all 15 §1 paths; **this erratum replaces the A5 table where they differ**:

- intercycle/observer/learn_cycles.py: binding blob is `03b54be1a905de3c371f498adf94c3bbb86f34b7` (not de8d4233aacd7d9d40fda21367b7d8606315811c).
- ros2/intercycle_nav/intercycle_nav/timing.py, timing_planner.py, masks.py and ros2/intercycle_nav/config/nav2.yaml
  are **unchanged** at this commit (the §1 table blobs 0c3555c2…, 2ea9a0e0…, a9c67f66…, 156409cd… are binding);
  their A5 "new" blobs are void.
- The other seven A5 rows are correct. So 8 of the 15 §1 paths changed, not 12.
- Any later change to a §1 path (for example fix (a), A5 item 8, if it is merged before FREEZE) gets its own entry
  listing the old and new blobs.

## Amendment 6 (A6): real clips ROI-only (already in force), corrections to A2 item 4 and A4, two open differences

- **Recorded:** Sun 18 Oct 2026, ~09:50 CEST, lead session intercycle-74, before FREEZE_REALCLIP. Text only, no code.
- **User decision (~09:35 CEST), verbatim:** "Make real clips ROI-only". Checked on main by lane rc-roi-only and its
  auditor: `intercycle/observer/pixel_adapter.py` has applied only the image-space ram-ROI rule since 1205db0
  ("RC-freeze item 1"); it builds no `ObsV2Converter` and calls no floor rule. So nothing changes, and the dev numbers
  stand (a re-score from the same PERC outputs gave the identical results/dev/realclip_dev.json): MISSING under
  occlusion 31/35 = 0.886, not occluded 145/292 = 0.497, stroke F1 0.803 (full dev set) / 0.886 (base pair), person
  recall 0.614.
- **Correction to A2 item 4:** "the twin keeps its rule" is superseded by A4: the twin and the real clips both use the
  ROI rule (the floor rule remains only for the G0 stand-in camera_v0 and the body-shop shim).
- **Correction to A4:** "with the same hold" means the same rule with different hold values (next item).
- **Differences still standing, not changed (no tuning):**
  1. Hold: the real ingest holds 1.33 s (16 frames at 12 fps, `esk_ingest._default_hold`); the twin holds 34 s on
     camera 9 and 7.5 s on camera 14.
  2. Occluder input: in the twin, only movers the camera detects that tick occlude; on real clips every confirmed track
     box occludes, plus every person or forklift detection at conf >= occ_conf 0.4.

## Amendment 7 (A7): real-clip score uses the FIRST done attempt (sealed_run.py blob)

- **Recorded:** Sun 18 Oct 2026, ~09:55 CEST, lead, before FREEZE_REALCLIP. Lane rc-first (e050251, audit pass), merged
  at 9f96abf.
- **Change:** `scripts/sealed_run.py score` without `--attempt` now scores the FIRST clips attempt whose closing record
  is `done` (it took the last one), matching PREREG "the first clean attempt is scored". `--attempt N` is unchanged.
  The RC-final audit had flagged this as a minor issue. Runbook: one safe.directory line in docs/probes/REALCLIP_RUN.md.
- **§1 blob, scripts/sealed_run.py:** 2bbcabf0a1c04139b63f0c5ec9ce7e5bda83492e (A5) -> **d95f72f050e3b7a74d77334410a49e01de2f6ce6** (binding, = `git rev-parse 9f96abf:scripts/sealed_run.py`).

## Amendment 8 (A8): robot part at FREEZE: 20-min learning in code, fix (a) not built, freeze mechanics

- **Recorded:** Sun 18 Oct 2026, ~10:10 CEST, lead session intercycle-74, at the main commit RC that merges
  lane/robot-freeze, before prereg/FREEZE_REALCLIP and prereg/FREEZE.
- **Observer learning = 20 min (A5 item 6, now in code).** PREREG v1 "days 1-8" (§ on observer learning) is replaced: for each learned mission a
  fresh observer reads the robot-free plant stream of the same world day over [t0-1200 s, t0) (lane learn20b, c4b8df3,
  audit pass; `scripts/run_double.py` default `learn20`; the days 1-8 path stays behind the dev-only flag
  `--observer-days18`, refused in a sealed run). Nothing carries over between days, zones or missions.
- **Short window (user decision ~10:00 CEST, verbatim "Keep the shorter window, disclose"):** a mission starting less
  than 1200 s into its day (SLOT_PAD_S = 600 s) learns from [day start, t0) only; the run record marks `clipped = true`
  and `learn_s` < 1200. Expected ≈ 0.7 % of slots (< 1 of 64 learned missions).
- **Fix (a) NOT built (A5 item 8 void).** Measured before building: the 2 Hz tick does not block learning (a clean
  synthetic 1.69 s stroke learns at confidence 15 under the current limits; the estimator already removes the tick
  quantisation). Real P9b fails because tracked strokes are split / merged (cycles 1.0-4.0 s; learned 1.518 s vs
  generator ~1.70 s; jitter 0.203, consistency 0.103). Loosening the limits to pass it would be tuning, so no learning
  limit was changed. Consequence, disclosed: the learned arm most likely has confidence 0 on every press and acts like
  the static arm; the user decided (~10:00 CEST, verbatim "Freeze and run it, report honestly").
- **Freeze mechanics (user decision "Both at 10:15"):** ONE frozen code commit RC for both parts. The files are written in
  two consecutive commits because the checks require it: commit A writes prereg/FREEZE_REALCLIP = RC (the real-clip run
  uses a detached worktree at A; C3 allows only that file after RC); commit B writes prereg/FREEZE = A (robot check 3
  allows only prereg/FREEZE, TEST_SEED_COMMIT and results/sealed/** after A; check 5: RC is an ancestor of A and no
  real-clip input differs). Seeds: drand round after commit B's time.
- **Sealed batch cost:** projected ≈ $5.9 on Modal (observer $0.41 + missions $5.50), inside the session's Modal cap.
- **Dev dry run of this path:** `scripts/run_double.py` on Modal, dev world 0, day 9, 3 missions, 1 round:
  0 invalid, 0 crashed, $0.08, 459 s wall (out/double/freeze_dry2_w0d9z9, ~09:53-10:01 CEST).
