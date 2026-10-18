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
