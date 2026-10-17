# Probe P6: framing and scale (lane S3a-P6)

Sat 17 Oct, ~23:30-00:30 CEST. Plan row: PLAN_TWIN §3 P6. Code: `intercycle/framing.py`, `cloud/modal_p6.py`,
`scripts/esk_p6_k1.py`, `scripts/esk_p6_apply.py`, tests `tests/test_framing.py`. Outputs: `twin/cal/k1_v0.json`
(CAL v0 k1 per framing, with source clip ids), `twin/cal/k1_hints.json` (the lines used), `twin/cal/scale_v0.json`.

## Short answer

| | c9_f0 | c14_f0 | c9_f1 |
|---|---|---|---|
| clips | 312 train | 86 train | 271 dev (0 train) |
| k1 (division model, normalised by f = 1330 px) | **−0.294** (3 lines, leave-one-out −0.290 to −0.295) | **−0.118** (2 lines, weak) | none (dev only) |
| camera height from people (f = 1330) | 3.52 m (90 % CI 3.47-3.57) | 5.71 m (5.56-5.93) | — |
| camera height from 10 cm stripes (f = 1330) | 3.95 / 3.27 / 3.96 m | 5.51 m | — |
| prior | 4.0 m | 4.0 m | — |
| largest disagreement | 13.6 % | **43 %** | — |
| verdict | pass | **fallback: scale interval 4.0-5.93 m** | no CAL (§5 rule) |

- **Framings:** camera 9 has 2 framings, camera 14 has 1. Every train and dev clip registered.
- **Big finding:** all 271 camera-9 dev clips (group c9_g000) are a different framing from all 312 camera-9 train
  clips (group c9_g001). The corners move 276-335 px at 960 px width, so the camera was re-aimed or re-zoomed between
  the two sessions. Under the committed split, camera-9 dev has no train framing, so §5 gives it no fitted H. That
  means no foot-point error or zone accuracy on dev, the same rule the plan sets for test. This matters for the split
  decision in `G0_PROBES.md`: framing is a clean session marker.
- **Guard hardware:** camera 9 shows a fixed wire-mesh guard and railings on the coil line, but no light curtain.
  Camera 14 shows yellow vertical housings beside the die opening on 3 presses. That is consistent with light curtains
  but cannot be confirmed at this resolution. U4 (ASSUMED light curtain plus manual reset on every press) is unchanged.
- **Modal:** $0.073 (app `s3a-p6-esk`, CPU only, 4 app runs including one crash and one image fetch).

## Inputs and what was looked at

- `inventory.parquet`, the clips and the keyframes are not in `$INTERCYCLE_DATA`. The split is still on STOP and
  Pass 2 has not run. So "train" here means `split == train` in the committed `prereg/split.csv` (c9_g001, c14_g000,
  c14_g001), and the frames are the full-resolution t = 0.5 s frames (`/esk/frame05`) and their RF-DETR boxes
  (`/esk/det`) already on the Modal volume. Only train and dev ids were passed to the job, and no test id was read.
- Only numbers came back from the registration and scale job. Two derived images per train framing stayed on the
  volume (`/esk/p6/<framing>_bg.jpg`, `_chains.jpg`). The median backgrounds of c9_f0 and c14_f0 were downloaded to
  the session scratch and **viewed** to pick the k1 lines and look for guard hardware. Each is the per-pixel median of
  25 train frames, so most people disappear. The 25 source clip ids per framing are listed in `twin/cal/k1_v0.json`.
  **If the split is redone (A1/A2), those 50 ids must stay out of test.** Nothing image-like is committed.

## Method

1. **Registration (framing_id).** SIFT (3000 features) on each frame at 960×540, ratio test 0.75, RANSAC homography
   (3 px). Train clips are processed first, in sorted order, then dev clips. A clip joins the first framing of its
   camera with ≥ 60 inliers and a maximum corner displacement ≤ 12 px at 960 px (24 px full-res). Otherwise it starts
   a new framing. A dev clip may start a framing (registration only). Such a framing gets no k1, no CAL and no H.
2. **CAL v0 k1** (one per framing, division model `x_u = c + (x_d − c)/(1 + k1·r_d²)`, c = image centre, r in px of
   1920×1080). Straight world lines were picked by eye on the train median background (`twin/cal/k1_hints.json`).
   `framing.trace_edge` snaps each hint to the sub-pixel edge in two passes (strongest gradient within ±40 px, then
   locked within ±5 px of a robust quadratic). k1 minimises the summed squared distance to each line after
   undistortion (`framing.fit_k1`). Lines nearly radial to the image centre carry no k1 information and were dropped.
   `k1_v0.json` also gives K (f = 1330) and an OpenCV rational distortion vector `[k1,k2,0,0,k3,k4,k5,k6]` that
   reproduces the division model within 1e-5 px. So S2 can call `cv2.undistortPoints(pts, K, dist, P=K)` before
   `findHomography`, as the plan says.
3. **Scale.** The ground scale is proportional to the camera height h, so every check is written as h. The factor is
   h / 4.0 m.
   - *People:* upright, unoccluded, non-border person boxes (conf ≥ 0.5) on train frames, undistorted with k1. A
     zero-roll pinhole is fitted with f fixed (1020, 1330 and 1430 px), giving h and tilt by robust least squares on
     head-top rows with walker height 1.72 m. The CI is a 100× clip bootstrap.
   - *Stripes:* both edges of painted floor lines are traced, undistorted, and projected to the floor with the people
     fit's tilt. The line is assumed to be 10 cm wide.
   - *Pallets:* not measured. See "Not done".

## Numbers

**Registration.** Each cell gives inliers and the corner displacement at 960 px.

| framing | ref clip | members | vs own reference | vs c9_f0 reference |
|---|---|---|---|---|
| c9_f0 | 0_te21 (train) | 312 train | within threshold (p50 displacement 0.6 px) | — |
| c9_f1 | 0_te1 (dev) | 271 dev | inliers min 346, median 924; displacement ≤ 1.2 px | 42-53 inliers; displacement 276-335 px |
| c14_f0 | 3_te1 (train) | 86 train | inliers ≥ 696; displacement p50 0.2, max 2.7 px | — |

The clip ids follow the authors' own tr/te naming, not our split.

**k1** (k1_norm = k1·1330²; barrel distortion gives k1 < 0)

| framing | line | length px | RMS raw → after (px) | single-line k1_norm |
|---|---|---|---|---|
| c9_f0 | A centre yellow stripe (dashed) | 780 | 1.35 → 1.02 | −0.231 |
| c9_f0 | B left yellow line of the green walkway | 1116 | 8.99 → 0.49 | −0.294 |
| c9_f0 | C inner yellow line of the green walkway | 481 | 1.89 → 0.34 | −0.310 |
| c9_f0 | **joint** | | | **−0.294** (k1 = −1.66e-7 px⁻²) |
| c14_f0 | A yellow floor line | 768 | 1.32 → (1.24 at its own optimum) | −0.12 |
| c14_f0 | C white column, left edge (top 100 px) | 103 | 0.36 | — |
| c14_f0 | **joint** | | | **−0.118** (k1 = −6.68e-8 px⁻²) |

- **Cross-check c9:** a blind run over automatic Canny chains on the same background gives −0.276 (3 chains,
  leave-one-out −0.25 to −0.30).
- **Prior check:** if f were exactly 1330, the vendor 86.5° HFOV would imply k1_norm ≈ −0.45. The measured −0.29
  therefore points to slightly less barrel than the vendor sheet implies at that f, or to a somewhat different f. This
  is a consistency note only.
- **c14 is weak.** Its only long informative line is a painted, partly faded floor line. The c9 value −0.294 would
  raise that line's residual only from 1.24 to 1.41 px.
- **Rejected c14 lines:** the plate edges are radial, so they carry no k1 information, and the right column's trace
  followed a ladder.
- **Choice:** following the plan text, k1_v0 keeps c14's own fit. The c9 value is the same-lens alternative if CAL v1
  needs one.

**Scale.** All values are camera height in m.

| framing | f px | people h (tilt) | people median \|resid\| px | stripes h |
|---|---|---|---|---|
| c9_f0 | 1020 | 3.12 (32.1°) | 12.3 | 3.59 / 2.99 / 3.70 |
| c9_f0 | 1330 | **3.52** (27.2°), boot 3.47-3.57 | 13.5 | **3.95 / 3.27 / 3.96** (widths 14.6 / 10.7 / 14.4 px) |
| c9_f0 | 1430 | 3.63 (25.8°) | 13.7 | 4.03 / 3.33 / 4.02 |
| c14_f0 | 1020 | 5.00 (33.4°) | 10.1 | 5.56 |
| c14_f0 | 1330 | **5.71** (28.1°), boot 5.56-5.93 | 9.9 | **5.51** (width 10.3 px) |
| c14_f0 | 1430 | 5.89 (26.7°) | 9.9 | 5.46 |

- **Without k1**, the c9 people fit gives 3.49 m, but its residuals rise from 13.5 to 19.3 px. k1 matters for tilt,
  not much for h.
- **Spread at f = 1330** (prior vs people vs median stripe):
  - c9: 4.00 / 3.52 / 3.95, so the largest disagreement is 13.6 % (≤ 20 %: **pass**). Line B alone (3.27) is 18 %
    below the prior; it borders a green band, so its second edge may be the green edge.
  - c14: 4.00 / 5.71 / 5.51, which is **43 %**. The two independent image checks agree with each other within 4 %,
    so the 4 m mount prior looks wrong for camera 14.

**Scale published** (`twin/cal/scale_v0.json`):

| framing | h interval | scale factor vs the 4 m prior | point estimate | robot at both ends |
|---|---|---|---|---|
| c9_f0 | 3.47-4.0 m | ×0.87-1.00 | 3.75 m (midpoint) | not required (pass) |
| c14_f0 | 4.0-5.93 m | ×1.00-1.48 | 5.6 m | **required (fallback)** |

## Guard hardware seen (train median backgrounds; nothing committed)

**Camera 9 (c9_f0).**
- One eccentric press. Its data plate is legible: 100 t, table 670 × 1100 mm, so the table is a further possible
  scale reference.
- It is fed by an automatic coil line (decoiler, then straightener/feeder).
- Visible guarding:
  - a fixed yellow wire-mesh panel at the lower die side;
  - yellow tubular railings around the straightener/feeder and the decoiler;
  - a danger sign forbidding hands in the press while it runs.
- The die opening is not in view. **No light curtain is visible.**

**Camera 14 (c14_f0).**
- Four C-frame presses in the middle distance, with a fifth large press at the top right.
- On 3 of them, yellow vertical housings stand on both sides of the die opening. They are consistent with
  light-curtain transmitter/receiver columns, but only a few px wide, so they are **not confirmable**.
- Control panels with displays stand beside two presses.
- No fences in front of the c14 presses.

**Consequence.** Nothing here contradicts U4. Guard type stays ASSUMED (light curtain plus manual reset) on every
press. At most, c14 could be written as "consistent with light curtains (visual, unconfirmed)".

## Verdict against the plan

- **Registration (pass).** Every framing registered to a reference.
- **CAL v0 k1.** It is delivered per train framing in `twin/cal/k1_v0.json`, with its source train clip ids.
  - c9: solid.
  - c14: weak (2 lines), with a same-lens alternative noted.
- **Scale.**
  - c9 passes: 13.6 % ≤ 20 %.
  - **c14 takes the fallback.** Scale interval h 4.0-5.93 m (×1.00-1.48). Run the robot at both ends for the zone-14
    world.
- **No train framing for c9_f1 (all camera-9 dev clips).** Per §5, no fitted H, so it is excluded from foot-point
  error and zone accuracy. This needs the lead's attention together with the split decision.

## Framing ids written

- `$INTERCYCLE_DATA/p6_framing.csv` has 669 rows: every train and dev clip of the committed split.
  - train camera 9 → `c9_f0`
  - camera 14 → `c14_f0`
  - dev camera 9 → `c9_f1`
- `inventory.parquet` does not exist yet, because S3a deliberately did not write it under the STOP. Once it is built,
  `python scripts/esk_p6_apply.py --p6-json <p6 result>` merges the ids. It refuses test ids and re-checks against
  split.csv.

## Not done / interpretations

- **Pallets:** not measured (timebox). The c9 coil pallet and the press table (670 × 1100 mm) are candidates for CAL v1.
- **Matcher:** SIFT only (CPU). LightGlue was not needed: the inlier margins are large.
- **Keyframes:** the t = 0.5 s frames replace the keyframes, because Pass 2 has not run.
- **Hints:** k1 lines were picked by an agent viewing the train median background, not by the user. CAL clicks for H
  are still S2's.
- **Split change:** if the split changes, the framing ids stay valid, because they are per clip and come from the
  image. The k1 and scale sources must then be re-checked to still be train.
