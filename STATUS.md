# InterCycle: status and gate log

This page is the gate log of the press-shop twin evaluation. It records what git shows. Rows with a `[[...]]`
placeholder have not happened yet; they are filled by hand at the time, never guessed.

State when this page was written: main at `b80f883` (PREREG v1 on main); since then A6 (`57b547d`, text only) is on
main. There is no `prereg/FREEZE_REALCLIP`, no
`prereg/FREEZE`, no `prereg/TEST_SEED_COMMIT`, no `results/sealed/`, and no `twin/m7_margin.json` on main.

Older log (before the press-shop plan, and the G0 report): `docs/STATUS.md`.

## Times

- All times are CEST, 17-18 Oct 2026.
- Git stamps on this machine read +01:00 (WEST). They are shown here as CEST, which is the git stamp plus one hour.
  Check with `git log --date=iso-strict --format='%h %cd %s'`.
- A time marked `~` is the time the lead or the amendment text states, not a git stamp.
- Planned times come from `docs/PLAN_TWIN.md` section 4 and from the changes in `prereg/AMENDMENTS.md` (A2, A3).

## Gate log

| Gate | Time (CEST) | Commit | Result |
|---|---|---|---|
| Plan of record (digital twin, Option B) | Sat 17 Oct 21:52 | `8c52e12` | Plan, decisions U1-U15 and the defect register committed. Freeze planned Sun 08:30, submission Sun 12:30. |
| Early lanes merged (S1b, S3b, S4, S7a, S7b, S5a-H1) | Sun 18 Oct 01:28 | `726bab0` (also `2c21e59`, `d8a1719`, `62ecb87`, `f754294`, `14d2931`) | Merged to main. |
| Lanes merged (S3a-a1, S3a-P4b, S2-calprep), Amendment 1 lead notes | 02:19-02:21 | `a58317b`, `387d7d4`, `7564c12`, `7480db3` | Split A1+ and its notes on main. Custodian fetch hardened. |
| G0 | ~02:30 | `65f2c26`, gate log `b028a65` (02:24) | Green except the MANIFEST (no custodian session yet). User decisions (alpha 0.05, delta 0.10, K = 16, A-lite kept) and probe results recorded in `docs/STATUS.md`. |
| Lanes 1 merged (S1a-compile, S1b-engine, S5a-plant, S6-observer-live, S5b-policy) | 04:05 | `e24b2c6`, `6d7987a`, `0bfb53b`, `ccb250c`, `407dbe3` | Merged to main. The planned G1 (03:00, revised 04:30) has no separate gate record in git. |
| Lanes 2 merged (S5a-isaac, S6-optB, S1b-eval, S2-cal, S3a-perc, S4-labels, S8-alite, S3a-dyn) | 06:11-06:14 | `35e6398`, `a95d33d`, `172bfe8`, `c1e915e`, `77dfee2`, `262e776`, `d0bad7f`, `e469a5b`, `4afd810` | Merged to main. |
| Custodian: test MANIFEST | 06:43 | `99ef287` | `sealed/MANIFEST.sha256` committed (126 test clips). |
| Custodian: test sample list | 06:51 | `c0b8919` | `sealed/SAMPLE.sha256` committed. |
| Custodian: test labels hashed | 07:27 | `c9a0abc` | `sealed/LABELS.sha256` committed. Custodian and labellers run on the desktop (A1 lead notes). |
| A-lite prereg, Modal dry run repeated on main | 07:32 | `4b3611e` | Identical to the first dry run. |
| Amendment 2 (split freeze, cut design, no tuning) | ~07:00 (decided) | in `d06157e` (PREREG-final, 07:08) | Five user decisions and one lead decision recorded. |
| Amendment 3 (real-clip freeze time, robot launcher, exact margin, disclosures) | ~08:00-08:10 | `0702396` (08:08, lane PREREG-v1) | Recorded in the lane; reached main with `d2b3a56`. |
| Amendment 4 (twin uses the ROI occlusion rule) | ~08:25 (decided) | `f79cd8c` (08:22), `a23c7b9` (08:27) | Implemented in lane twin-roi; reached main with `d2b3a56`. |
| Lane merges into `lane/integration`: RC-final, PREREG-v1, double-modal, observer-fix-fix1, obs-speed, freeze-glue, observer-learn (with twin-roi) | 08:20-09:06 | `30f8a4b`, `918774a`, `6f0a117`, `eeb0685`, `d331d1a`, `c9c4e32`, `15846ee`, `b0e3c2f` | Integrated. |
| Glue: robot check 5 guards `prereg/r0_v1.json`; `run_double` passes the camera's occlusion rule | 09:16 and 09:22 | `6260768`, `f69b4a2` | Integration branch at `f69b4a2`. |
| **PREREG v1 on main** | ~09:40 | `d2b3a56` (09:39) | Merge of `lane/integration`. `prereg/PREREG_TWIN.md` text unchanged. Amendment 5 records old and new code hashes as an append-only list. Per A5, this is before any dev-world run, any dev robot run, any test seed, `FREEZE_REALCLIP` and `FREEZE`. |
| PREREG v1 erratum (A5) | ~09:45 | `b80f883` (09:43) | 5 rows of the A5 table corrected, checked against `d2b3a56`. 8 of the 15 code paths changed, not 12. |
| Amendment 6 (real clips ROI-only already in force; A2 item 4 / A4 corrections; hold and occluder differences) | ~09:50 | `57b547d` (09:48) | Text only, no code. Dev numbers unchanged. Before FREEZE_REALCLIP. |
| README and STATUS first draft | [[TIME_DRAFT_DONE]] | [[COMMIT_SUBDOCS]] | Branch `lane/subdocs`, not merged. |
| Dev worlds 0-3 generated | [[TIME_DEV_WORLDS]] | [[COMMIT_DEV_WORLDS]] | [[RESULT_DEV_WORLDS]] |
| Dev robot runs (descriptive, no tuning) | [[TIME_DEV_RUNS]] | [[COMMIT_DEV_RUNS]] | [[RESULT_DEV_RUNS]] |
| Fixed margin written (`twin/m7_margin.json`) | [[TIME_MARGIN]] | [[COMMIT_MARGIN]] | [[RESULT_MARGIN]]: age_p99 = [[AGE_P99_S]] s, margin = [[MARGIN_M]] m. Fails closed above 0.5578 s. |
| **FREEZE_REALCLIP** | target ~09:15 (A3), latest 10:15; user says both freezes at 10:15 (A5) | [[COMMIT_FREEZE_REALCLIP]] | [[RESULT_FREEZE_REALCLIP]] |
| Real-clip sealed run (clips stage), attempt 1 | [[TIME_RC_RUN]] | [[COMMIT_RC_RUN]] | [[RESULT_RC_RUN]] (output hash only goes to the lead until `FREEZE`) |
| Real-clip sealed score | [[TIME_RC_SCORE]] | [[COMMIT_RC_SCORE]] | [[RESULT_RC_SCORE]] |
| G2 (slice on all v1 artifacts, parity, DYN report, hashes) | [[TIME_G2]] | [[COMMIT_G2]] | [[RESULT_G2]] |
| Dry run (sealed path on dev; FREEZE check on a throwaway commit) | [[TIME_DRY_RUN]] | [[COMMIT_DRY_RUN]] | [[RESULT_DRY_RUN]] |
| **FREEZE (robot part) or the declaration "not ready"** | planned ~10:15 | [[COMMIT_FREEZE]] | [[RESULT_FREEZE]]. If not ready: descriptive dev results, no H1-H3 claim. |
| Last re-run time (written in the FREEZE commit message) | [[TIME_LAST_RERUN]] | [[COMMIT_FREEZE]] | [[RESULT_LAST_RERUN]] |
| Test seeds (drand) | [[TIME_SEEDS]] | [[COMMIT_TEST_SEED_COMMIT]] | drand round [[DRAND_ROUND]]; `TEST_SEED_COMMIT` = [[TEST_SEED_COMMIT]]; source [[SEED_SOURCE]] (drand or fallback, said in the README) |
| Sealed robot batch (192 cells), attempt 1 | [[TIME_ROBOT_RUN]] | [[COMMIT_ROBOT_RUN]] | [[RESULT_ROBOT_RUN]]; infrastructure-invalid re-runs: [[N_RERUNS]] |
| `classify_invalid` list | [[TIME_INVALID]] | [[COMMIT_INVALID]] | [[RESULT_INVALID]] (committed before any metric) |
| Scoring (`twin_eval`: H1-H4, outside-gate reports) | [[TIME_SCORING]] | [[COMMIT_SCORING]] | [[RESULT_SCORING]] |
| Independent audit (PREREG v1 against the final text; amendments listed) | planned ~11:15 | [[COMMIT_AUDIT]] | [[RESULT_AUDIT]] |
| README headline lines filled | [[TIME_README_FILLED]] | [[COMMIT_README_FILLED]] | [[RESULT_H4]] and [[RESULT_ROBOT]] replaced |
| Submission | 12:30 | [[COMMIT_SUBMISSION]] | [[RESULT_SUBMISSION]] |

## Amendments

| Amendment | Decided (CEST) | What it changed |
|---|---|---|
| A1 / A1+ | Sun ~00:45 | Split by background clusters, no near-copy link. Test: 4 groups on camera 9, 12 on camera 14. |
| A2 | Sun ~07:00 | Split freeze, cut design (192 cells), ROI occlusion on real clips, BDC strokes, no tuning. |
| A3 | Sun ~08:00-08:10 | Real-clip freeze may move to 10:15, robot launcher, exact fixed margin, disclosures. |
| A4 | Sun ~08:25 | The twin uses the same ROI occlusion rule as the real clips. |
| A5 | Sun ~09:40, erratum ~09:45 | Old and new code hashes (append-only), user decisions of 18 Oct, disclosures. |
| A6 | Sun ~09:50 | Real clips ROI-only (already in force since `1205db0`); corrects A2 item 4 and A4; lists the hold and occluder differences between twin and real clips. |
| [[NEXT_AMENDMENT]] | [[TIME_NEXT_AMENDMENT]] | [[WHAT_NEXT_AMENDMENT]] (name the dev run that prompted it; fix (a) goes here if merged) |

## Filling rules for the README result lines

The README has two placeholder lines. Fill them only with the wordings below (`prereg/PREREG_TWIN.md` sections 0, 2.1,
4, 4.1, 8, 9, 10).

`[[RESULT_H4]]` (real-clip person recall):
- Always: "against independent Claude-agent labels (vision model), not human labels"; n; r0 = 0.5272; the figures per
  stratum; the camera-9 interval marked "ignores clustering (only 4 groups)"; near-duplicate footage can sit on both
  sides of the train/test line (A1+).
- Claimed only if H1, H2 and H3 were claimed first and the H4 rule rejects.
- If the robot part is descriptive or an earlier gate did not pass: "computed, not claimed".
- If any per-group recall is below r0: "not claimed: this is the pre-registered rule, not a failure of the run".

`[[RESULT_ROBOT]]` (twin robot batch):
- Always: holds only inside the assumed stop model; zone 14 assumed; neither robot tuned; learned never enters a
  stopped press and reactive can; fixed latency margin, not certified.
- If frozen and sealed: H1, H2, H3 in gate order, each "claimed" or "not claimed".
- If learned minus static is 0 or below: say on the first screen that static boundary inspection matched or beat the
  learned policy.
- If not frozen in time: "Descriptive dev results from dev worlds 0-3. No H1, H2 or H3 claim is made."
- If any design cell is still unresolved after the last re-run: "Not evaluable: unresolved cells", no H1-H3 claim.

## Other placeholders in the README

`[[DECK_LINK]]`, `[[VIDEO_LINK_OVERVIEW]]`, `[[VIDEO_LINK_TWIN_DAY]]` (media; illustrative, not recorded runs) and
`[[VERIFY_DOI]]` (check the data paper DOI on doi.org before submission; the dataset card carries a wrong number).
