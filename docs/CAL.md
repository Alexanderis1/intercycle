# CAL v1: camera calibration and site tracing (lane S2)

Sun 18 Oct, ~02:30-03:30 CEST. Plan rows: PLAN_TWIN §2 (`twin/site.yaml`), §3 P6, §4 S2 (CAL v0 + CAL v1), §5
("CAL uses only train frames"). Code: `intercycle/calib.py`, `scripts/cal_v1.py`; tests `tests/test_calib.py`,
`tests/test_site_v1.py`, `tests/test_cal_v1_script.py`. Fix round 1 (audit of the first version): see
"Fix round 1" at the end; every number below is the regenerated one.

> **Disclosure (user decision, U14 clause).** CAL points placed by a Claude agent from train keyframes (vision), not
> by a human. The user was asleep and delegated this step. The same agent also traced the floor polygons and press
> footprints.

## Short answer

| | c9_f0 | c9_f1 | c14_f0 |
|---|---|---|---|
| train clips (A1+) | 252 | 109 | 55 |
| k1_norm (division model, f = 1330) | **-0.2945** (3 lines, leave-one-out -0.291 to -0.295) | **-0.2747** (3 lines, LOO -0.2745 to -0.2765) | **-0.111** (2 lines, weak) |
| tilt / roll from the floor relations | 32.66° / 2.41° | 16.19° / -0.47° | 25.14° / -1.21° |
| relation residuals | 0.26-1.42° | 0.19-1.35° | 0 (2 relations, 2 unknowns) |
| tilt 90 % bootstrap | 32.5-32.9° | 15.9-16.4° | 21.8-27.3° (k1 drawn too) |
| roll 90 % bootstrap | 2.2-2.7° | -0.7 to -0.3° | -7.7 to -0.7° (k1 drawn too) |
| camera height from people (1.72 m walker), nominal orientation | 4.24 m | 4.42 m | 4.33 m |
| camera height from people, joint bootstrap p5 / p50 / p95 | 4.19 / 4.26 / 4.35 m | 4.37 / 4.42 / 4.48 m | **2.04 / 3.50 / 4.85 m** |
| camera height from 10 cm stripes, nominal | 4.76 / (7.56*) / 4.58 m | 5.11 / 5.22 m | 4.81 m |

\* Line B borders the green walkway band, so one of its two "edges" is the green edge. P6 saw the same effect. It is
reported but left out of the stripe median (fix round 1).

**Scale (camera height h; H is built at the 4 m prior).** Joint bootstrap: in each of the 200 draws the people and
stripe heights are re-solved at that draw's k1, tilt and roll, and the draw's h is taken from that draw's estimates.

| camera | prior | people, joint p5-p95 | stripes median, joint p5-p95 | largest disagreement (nominal) | published interval | factor |
|---|---|---|---|---|---|---|
| 9 | 4.0 m | 4.19-4.48 m (both framings) | 4.70-5.06 m | 23.4 % | **4.0-5.06 m** | ×1.00-1.27 |
| 14 | 4.0 m | 2.04-4.85 m | 3.71-5.18 m | 20.2 % | **2.04-5.18 m** | ×0.51-1.30 |

Both cameras disagree by more than 20 %, so the P6 fallback applies to both: publish the interval and run the robot at
both ends. `twin/site.yaml` carries one site-level interval, the union ×0.51-1.30 with mid 1.0; the per-camera
intervals are in `twin/cal/scale_v1.json` and in the scale note.

- **Camera 14 is the weak camera.** Its tilt rests on 2 assumed floor relations for 2 unknowns, and its k1 on one
  floor line plus a 100 px column edge. Where the camera-14 spread comes from (same 200 draws, scratch decomposition):
  - tilt / roll / clicks only (k1 fixed at its own fit): people 3.01 / 4.30 / 5.66 m, interval 3.01-5.66 m (the
    audit's own re-solve gave 3.09 / 4.33 / 5.69 m);
  - plus the k1 draw between the c14 fit (-0.111) and the camera-9 fit (-0.295): people 2.04 / 3.50 / 4.85 m,
    interval 2.04-5.18 m (published).
- **P6's camera-14 interval (4.0-5.93 m) is not reused.** It is `scale_v0.json`, which the lead declared INVALID: it
  was fitted on pre-A1+ frames, 13 of them now TEST. The published interval is recomputed on A1+ train only.
- **Camera 9:** P6 had people at 3.52 m (tilt 27.2° fitted with h); with the floor-line tilt (32.7°) people give
  4.24-4.42 m.

**Press danger margin `dilation_m`** (p95 floor displacement of the danger-polygon vertices over 200 joint bootstrap
draws of H, in the framing the footprint was placed in and every framing with a P4 ROI of the press):

| press | status (P4) | dilation_m (site.yaml, scale-inclusive) | fixed-scale p95 (same draws at 4 m) | shape-only p95 (centroid pinned) | footprint |
|---|---|---|---|---|---|
| P9a | OBSERVED | **1.81 m** | 0.09 m | 0.51 m | traced, 2.8 × 1.6-1.9 m |
| P9b | OBSERVED | **5.44 m** | 4.01 m | 0.38 m | ASSUMED 1.4 m square (base hidden), centre ±60 px in c9_f0 |
| P14a | ASSUMED | **3.64 m** | 4.09 m | 1.30 m | traced, 1.4 × 4.2 m (back edge hidden) |
| P14b | ASSUMED | **2.68 m** | 3.66 m | 1.08 m | traced, 1.4 × 2.9 m |
| P14c | ASSUMED | **2.49 m** | 3.12 m | 0.91 m | traced, 1.2 × 1.8 m |
| P14d | ASSUMED | **3.03 m** | 3.47 m | 1.19 m | traced, 1.5 × 1.5 m |

- Camera 9: most of `dilation_m` is the scale interval (P9a: 1.81 m with scale, 0.09 m without).
- P9b: most of it is the stated ±60 px placement of its centre at a far pixel of c9_f0 (4.0 m without scale).
- Camera 14: the fixed-scale value is LARGER than the joint one. In the joint draws h moves with the tilt
  (correlation 0.67), and the people boxes pin the floor positions, so part of the orientation error cancels. At a
  fixed 4 m it does not.
- site.yaml uses the literal joint p95 (fail closed). Its `dilation_m` note tells consumers that run the robot at both
  scale ends to use `dilation_fixed_scale_p95_m` (twin/cal/cal_v1.json) instead, so scale is not counted twice.

## Method

1. **Images (train only).** `scripts/cal_v1.py gather` reads only `esk.load("train")` rows. For each framing it builds
   a per-pixel median of one full-resolution frame (at 1/3 of the clip) from up to 61 seeded train clips. Most people
   disappear. It also runs RF-DETR-M (conf ≥ 0.5) on the full-resolution frames at 1/3 and 2/3 of every train clip
   (the keyframe positions; local GPU, under the heavy lock). The images stay in `$INTERCYCLE_DATA/cal/`, which is
   gitignored. Nothing image-like is committed.
2. **k1 per framing** (`k1` step, `twin/cal/k1_hints_v1.json`). The agent picked 2-3 straight lines per framing on
   the train median. `framing.trace_edge` snaps each one to the sub-pixel edge, and `framing.fit_k1` (plumb line,
   division model) fits k1.
   - c9_f0 reuses the three P6 stripes.
   - c9_f1 (no k1 before) uses the centre stripe, the walkway line and the roof-beam pipe. The pipe is long and
     near the top edge, so it carries the most k1 information. Its two halves were traced with a fixed edge polarity
     and then joined.
   - c14_f0 has only the long yellow line plus a short column edge, so its k1 is weak. A white floor line, a ladder
     rail and a longer column trace were tried and dropped: residuals of 3-4 px, wrong sign.
3. **Points and relations** (`twin/cal/cal_points_v1.json`). Per framing the agent placed 4 floor points on 2-3×
   zoomed crops of the median, with a pixel grid, and cross-checked them on the masked click image. Each point has
   its location, the floor feature and a 1-σ uncertainty; see the table below. No lengths are typed. The points
   carry floor relations instead (two directions are parallel or perpendicular on the floor):
   - the L's long side and the walkway's yellow lines run along the aisle (parallel);
   - the L's arms and the recess arm are perpendicular to the aisle. Their directions come from stripe edges traced
     on the short arms, which are much more precise than two clicks.
4. **H** (`fit` step, `intercycle/calib.py`). Ground-plane camera model: f = 1330 px, principal point at the image
   centre, and tilt θ and roll φ fitted by least squares so that the relations hold. Height h = 4 m (P6 prior). The
   zone frame puts the origin at q1 and +z along the axis line. The points are undistorted with
   `cv2.undistortPoints` (k1 via its rational-model equivalent, iterated to convergence), mapped to the floor, and
   `cv2.findHomography` on the 4 undistorted points reproduces the model H to < 1 mm (checked on every run).
   **c9_f0 and c9_f1 share one zone-9 frame**: the same physical origin (corner of the yellow L) and the same axis
   (the walkway's left yellow line).
5. **Scale cross-checks** (at a given k1, θ and φ):
   - people: per box, solve h so that a 1.72 m walker standing at the foot pixel has its head at the box top
     (closed form, `calib.person_height_h`). Upright boxes only (height ≥ 80 px and ≥ 2× width, not at the border).
     Median over boxes.
   - stripes: both edges of each painted line, assumed 10 cm wide (P6 assumption). Median over the camera's stripes,
     with the flagged c9_f0 line B left out.
   - The `fit` step reports both at the nominal orientation (and a 1000× clip bootstrap at that orientation only).
6. **CAL v1 joint bootstrap** (`site` step, `scripts/cal_v1.py joint_bootstrap`). 200 draws per camera. The framings
   of one camera share draw i and one h (one mount). Per draw:
   - every placed point jittered by N(0, σ_stated);
   - every traced line resampled with replacement;
   - c14_f0 only: k1 drawn uniformly between its own fit and the c9_f0 fit (same lens model), lines re-undistorted;
   - θ and φ refitted;
   - at that draw's k1, θ and φ: people height (clip-resampled boxes) per framing and stripe height (edge points
     resampled);
   - h drawn uniformly over the span of {4 m prior, people, stripes} of that draw, so h moves with the tilt;
   - H rebuilt at that h.

   Published interval per camera = [min(prior, p5 of each estimate), max(prior, p95 of each estimate)].
   `dilation_m` = p95 of the floor displacement of each press's danger-polygon vertices against the nominal H. Pixels
   are kept RAW, and each draw undistorts them with its own k1. For P9b the centre pixel is also jittered by its
   stated σ = 60 px in c9_f0, the framing it was placed in. `dilation_fixed_scale_p95_m` = the same draws at 4 m.
7. **Site** (`twin/site.yaml`, from `twin/cal/site_trace_v1.json`).
   - The agent traced the visible floor per framing; zone 9 = the union of c9_f0 and c9_f1. The floor is cut at 25 m
     from the camera nadir.
   - Zone 14 is shifted by x + 21 m: no shared floor (P7), 10 m gap.
   - Press footprints: P9a and P14a-d traced at their floor contact.
   - P9b's base is hidden behind P9a in both framings, so its footprint is ASSUMED: a 1.4 m square under its column.
   - Danger polygon = footprint grown by 1.0 m. The 1.0 m is ASSUMED: no guard geometry is observed.
   - Ram ROIs come from P4 unchanged. Their source tag names the 10 train clips P4 measured each press on
     (`site_trace_v1.json` `p4_clips`). P4 did not record the ids of the clips it drew the ROIs from.
   - Every field has a source tag. `cal_sources` = the 369 train ids used: median sources, people-box clips, the
     click-sheet clips the points were cross-checked on (2_tr26, 4_tr31, 7_tr19), and the P4 clips.
   - The schema (`twin.schemas.site.Site`) and `twin/provenance.assert_train_only` pass.

## The points (full-resolution RAW pixels, stripe centre lines)

| framing | id | uv | σ px | floor feature |
|---|---|---|---|---|
| c9_f0 | q1 (origin) | (1672.0, 522.0) | 3 | corner of the yellow L right of the green walkway, below the waste-bin recess |
| c9_f0 | q2 | (1765.0, 530.0) | 4 | right end of the L's short top arm, at the wall |
| c9_f0 | q3 | (1830.7, 979.0) | 3 | bottom corner of the same L |
| c9_f0 | q4 = R2 | (1645.0, 454.0) | 3 | bottom corner of the waste-bin recess (check point) |
| c9_f1 | q1 (origin) | (1747.5, 926.0) | 3 | the same physical L corner |
| c9_f1 | q2 | (1840.0, 929.0) | 5 | right end of the L's top arm |
| c9_f1 | q4 = R2 | (1720.0, 862.5) | 3 | bottom corner of the waste-bin recess (same as c9_f0 q4) |
| c9_f1 | q5 = R1 | (1660.0, 680.0) | 4 | top corner of the waste-bin recess |
| c14_f0 | q1 (origin) | (505.0, 478.0) | 10 | estimated crossing of the worn dashed line and the long yellow line |
| c14_f0 | q2 | (180.0, 590.0) | 6 | left end of the dashed yellow line |
| c14_f0 | q3 | (505.0, 612.0) | 4 | left end of a straight slab seam |
| c14_f0 | q4 | (850.0, 800.0) | 5 | further right on the same seam |

**Cross-framing check.** R2 sits at (-0.01, -0.85) m in c9_f0 and (0.03, -0.77) m in c9_f1, 9 cm apart (at 4 m).
The L's top arm measures 0.62 m and 0.60 m.

**The click sheet's c9_f1 points were on a different L.** `cal_click_plan.json` put them on the waste-bin recess,
not on the L used for c9_f0: the re-aim moved the picture by about 400 px. This file uses the same physical features
in both framings.

**Camera-14 relations are assumptions:**
- the dashed line meets the long line at 90° on the floor;
- the slab seam is parallel to the long line.

With 2 relations for 2 unknowns, their residual is 0 and says nothing. The tilt spread (21.8-27.3°) comes from the
click jitter and the k1 draw.

**Camera-14 lens check.** The nominal fit redone with the c9_f0 k1 (-0.295 instead of -0.111) moves the tilt from
25.1° to 24.4°, the roll from -1.2° to -7.6°, and the P14a-d footprint centroids by 0.89 / 0.83 / 1.18 / 1.60 m.
The bootstrap draws k1 over this whole range, so this shift is inside `dilation_m` and the scale interval.

## Honest limits

- CAL points, floor polygons and footprints were placed by a Claude agent (vision), not a human.
- f is fixed at the 1330 px prior. The 1020-1430 px range is not propagated, either into H or into the bootstrap.
- c14's k1 rests on one painted line (weak). The bootstrap draws it between its own fit and the camera-9 value
  (-0.29); with the camera-9 value the roll moves from -1.2° to -7.6° and the P14a-d centroids by 0.8-1.6 m.
- The scale interval is wide on both cameras: ×1.00-1.27 on camera 9, and ×0.51-1.30 on camera 14 (h 2.04-5.18 m).
  The camera-14 low end (2.0 m) comes from the k1 draw plus the 2-relation tilt; it is the honest spread, not a
  plausible mount height. The 10 cm stripe width is an assumption, and people boxes include partly hidden feet and
  are taken as standing.
- P9b's `dilation_m` (5.44 m) is dominated by its ASSUMED centre placement (±60 px at a far pixel of c9_f0).
- Danger polygons use an ASSUMED 1.0 m clearance. P9b's footprint is ASSUMED. The back edges of the c14 footprints
  are hidden.
- Only the floor the cameras see is modelled. The far floor is cut at 25 m.
- Rejected on the way: vertical world edges as a roll constraint. Their traces on the median wandered (3-5 px RMS),
  and the fit then pulled the floor relations off by 4-14°.

## Decisions for the lead

1. **dilation_m size.** The literal joint p95 (1.8-5.4 m) is in site.yaml, because the task says the bootstrap
   includes the scale interval. The robot also runs at both scale ends, so a consumer that does both counts scale
   twice. The `dilation_m` note in site.yaml says so and names the alternative, `dilation_fixed_scale_p95_m`
   (0.09-4.1 m, the same draws at 4 m; `cal_v1.json`). That one is the right margin when the scale end is applied
   separately. The shape-only `dilation_local_p95_m` (0.4-1.3 m) also removes the press position error from the
   clicks and k1, so it is NOT recommended as a safety margin.
2. **Camera-14 scale low end.** The published camera-14 interval goes down to h = 2.04 m (×0.51), from the k1 draw.
   Running the robot at ×0.51 is what the evidence allows; the lead may prefer to cap it. That would be a decision,
   and it would be disclosed. Without the k1 draw the interval is 3.01-5.66 m.
3. **Provenance validator bug (outside this lane, still on main).** `twin.schemas.prereg.active_split_path` resolves the
   `prereg/ACTIVE_SPLIT` entry (`split_a1.csv`) against the repo root, not `prereg/`, so `assert_train_only()` with
   no explicit split raises FileNotFoundError on main. `scripts/cal_v1.py` and `tests/test_site_v1.py` pass the split
   explicitly (`esk.active_split_path`). The fix belongs to the owner of `twin/schemas/prereg.py` (S1a).

## WEAKNESSES touched

- **V-W7** (no calibration tool, no distortion model): partly closed. There is now a k1 per framing, H via
  undistortPoints and findHomography, and a bootstrap. It is validated on a synthetic camera in `tests/test_calib.py`
  (known tilt, roll, k1, h are recovered), not on Blender renders. GeoCalib remains cut.
- **R-W3** (camera gate) and **R-W10** (site graph): not in this item. They are later S2 items (camera gate module,
  TASKS).

## Fix round 1 (audit of the first CAL v1)

| finding | fix | evidence |
|---|---|---|
| major: camera-14 scale ignored the tilt uncertainty; h drawn independently of the tilt | `joint_bootstrap`: people and stripe heights re-solved at every draw's k1 / tilt / roll; h drawn from that draw's span; one h per camera per draw | camera 14 now 2.04-5.18 m (was 4.0-4.81 m); without the k1 draw 3.01-5.66 m, matching the audit's 3.09-5.69 m |
| keep P6's 4.0-5.93 m if wider? | not reused | it is `scale_v0.json`, INVALID per the lead (13 now-TEST clips) |
| minor: c14 k1 uncertainty not propagated | k1 drawn U(c14 fit, c9_f0 fit) per draw; shift stated in the c14_f0 H note and in `cal_v1.json` `c14_f0_k1_alt_check` | roll -1.2° → -7.6°, centroids 0.83-1.60 m |
| minor: P9b dilation only from c9_f1 | dilation over c9_f0 (placement) ∪ c9_f1 (ROI), with the centre pixel jittered by its σ = 60 px | P9b 3.62 → 5.44 m |
| minor: flagged line-B stripe in the median | `STRIPE_FLAGGED`, reported in `stripes_flagged_out` | camera-9 nominal stripe median 5.11 → 4.94 m; interval top 5.11 → 5.06 m |
| minor: failed traces dropped silently | `k1` stops unless every hinted line traces (and ≥ 2); relation lines too; dropped stripes recorded (`stripes_dropped`) | `tests/test_cal_v1_script.py` |
| minor: provenance incomplete | click-sheet clips (2_tr26, 4_tr31, 7_tr19) in `cal_sources`; `ram_roi` tagged with P4's own 10 train clips per press | `tests/test_site_v1.py` |
| minor: dilation_m double-counts scale for consumers | consumer note on every `dilation_m`; `dilation_fixed_scale_p95_m` added | see "Decisions for the lead" 1 |

**Bug found on the way (root cause fixed).** The first version solved each person's h by bisection on [0.5, 30] m.
At h = 0.5 m the 1.72 m head is above the camera, so the projected row wraps and has the same sign as at 30 m. The
bracket check then dropped the box silently. That hit 202 of 532 c9_f0 boxes, the lower-h ones, so the c9_f0 people
estimate was biased up: 4.57 m, now 4.24 m. c9_f1 and c14_f0 were not affected. `person_height_h` now uses the
closed form. It matches a correct bracket (h > 1.72 m) to 1e-14 m, and it is checked by a regression test
(`test_person_height_solves_every_visible_upright_person`: the old solver solves 23 of 35 synthetic people).
