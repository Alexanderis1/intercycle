# CAL v1: camera calibration and site tracing (lane S2)

Sun 18 Oct, ~02:30-03:30 CEST. Plan rows: PLAN_TWIN §2 (`twin/site.yaml`), §3 P6, §4 S2 (CAL v0 + CAL v1), §5
("CAL uses only train frames"). Code: `intercycle/calib.py`, `scripts/cal_v1.py`; tests `tests/test_calib.py`,
`tests/test_site_v1.py`.

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
| tilt 90 % bootstrap | 32.4-32.9° | 15.9-16.4° | 22.6-27.8° |
| camera height from people (1.72 m walker) | 4.57 m | 4.42 m | 4.33 m |
| camera height from 10 cm stripes | 4.76 / 7.56* / 4.58 m | 5.11 / 5.22 m | 4.81 m |

\* Line B borders the green walkway band, so one of its two "edges" is the green edge. P6 saw the same effect.

**Scale (camera height h; H is built at the 4 m prior).**

| camera | prior | people (90 % CI) | stripes (median) | largest disagreement | published interval | factor |
|---|---|---|---|---|---|---|
| 9 | 4.0 m | 4.50 m (4.38-4.61) | 5.11 m | 27.8 % | **4.0-5.11 m** | ×1.00-1.28 |
| 14 | 4.0 m | 4.33 m (4.31-4.35) | 4.81 m | 20.2 % | **4.0-4.81 m** | ×1.00-1.20 |

Both cameras disagree by more than 20 %, so the P6 fallback applies to both: publish the interval and run the robot at
both ends. `twin/site.yaml` carries one site-level interval, the union ×1.00-1.28 with mid 1.0; the per-camera
intervals are in `twin/cal/scale_v1.json` and in the scale note.

- **Camera 14 changed from P6.** P6 published 4.0-5.93 m. That fit used pre-A1+ train frames (13 of them now test)
  and fitted tilt and height together from people. Here the tilt comes from the floor lines (25.1°, P6 had 28.1°) and
  only h comes from people, which gives 4.33 m. Recomputed on A1+ train only, as the task asks.
- **Camera 9** moved up as well: P6 had people at 3.52 m (tilt 27.2° fitted with h); with the floor-line tilt
  (32.7°) people give 4.5 m.

**Press danger margin `dilation_m`** (p95 floor displacement of the danger-polygon vertices over 200 bootstrap draws
of H, in every framing that sees the press):

| press | status (P4) | dilation_m (written to site.yaml) | shape-only p95 (centroid pinned) | footprint |
|---|---|---|---|---|
| P9a | OBSERVED | **2.36 m** | 0.65 m | traced, 2.8 × 1.6-1.9 m |
| P9b | OBSERVED | **3.62 m** | 0.52 m | ASSUMED 1.4 m square (base hidden) |
| P14a | ASSUMED | **5.14 m** | 1.75 m | traced, 1.4 × 4.2 m (back edge hidden) |
| P14b | ASSUMED | **3.65 m** | 1.33 m | traced, 1.4 × 2.9 m |
| P14c | ASSUMED | **2.51 m** | 0.95 m | traced, 1.2 × 1.8 m |
| P14d | ASSUMED | **2.49 m** | 0.84 m | traced, 1.5 × 1.5 m |

The large values come mostly from the scale interval: a +28 % scale moves a press 6-12 m from the zone origin by
1.7-3.4 m. The "shape-only" column removes the press's own shift and keeps only the error in its size and shape.
site.yaml uses the literal p95 (fail closed). See "Decisions for the lead".

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
5. **Scale cross-checks.** With θ and φ fixed:
   - people: per box, solve h so that a 1.72 m walker standing at the foot pixel has its head at the box top. Upright
     boxes only (height ≥ 80 px and ≥ 2× width, not at the border). Median, 1000× clip bootstrap.
   - stripes: both edges of each painted line, assumed 10 cm wide (P6 assumption).
6. **CAL v1 bootstrap** (`site` step). 200 draws per framing:
   - every placed point jittered by N(0, σ_stated);
   - every traced line resampled with replacement;
   - h drawn uniformly over the camera's interval;
   - θ and φ refitted each time.

   `dilation_m` = p95 of the floor displacement of each press's danger-polygon vertices against the nominal H.
7. **Site** (`twin/site.yaml`, from `twin/cal/site_trace_v1.json`).
   - The agent traced the visible floor per framing; zone 9 = the union of c9_f0 and c9_f1. The floor is cut at 25 m
     from the camera nadir.
   - Zone 14 is shifted by x + 21 m: no shared floor (P7), 10 m gap.
   - Press footprints: P9a and P14a-d traced at their floor contact.
   - P9b's base is hidden behind P9a in both framings, so its footprint is ASSUMED: a 1.4 m square under its column.
   - Danger polygon = footprint grown by 1.0 m. The 1.0 m is ASSUMED: no guard geometry is observed.
   - Ram ROIs come from P4 unchanged.
   - Every field has a source tag. `cal_sources` = the 331 train ids used (median sources + people-box clips).
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

With 2 relations for 2 unknowns, their residual is 0 and says nothing. The tilt spread (22.6-27.8°) comes only from
the click jitter.

## Honest limits

- CAL points, floor polygons and footprints were placed by a Claude agent (vision), not a human.
- f is fixed at the 1330 px prior. The 1020-1430 px range is not propagated, either into H or into the bootstrap.
- c14's k1 rests on one painted line (weak). The same-model value of camera 9 (-0.29) is the alternative.
- The scale interval is wide on both cameras (×1.00-1.28 and ×1.00-1.20). The 10 cm stripe width is an assumption,
  and people boxes include partly hidden feet and are taken as standing.
- Danger polygons use an ASSUMED 1.0 m clearance. P9b's footprint is ASSUMED. The back edges of the c14 footprints
  are hidden.
- Only the floor the cameras see is modelled. The far floor is cut at 25 m.
- Rejected on the way: vertical world edges as a roll constraint. Their traces on the median wandered (3-5 px RMS),
  and the fit then pulled the floor relations off by 4-14°.

## Decisions for the lead

1. **dilation_m size.** The literal p95 (2.4-5.1 m) is in site.yaml, because the task says the bootstrap includes the
   scale interval. Most of it is scale, and the robot already runs at both scale ends. If the lead prefers the
   shape-only margin (0.5-1.7 m), `cal_v1.json` has it per press (`dilation_local_p95_m`).
2. **Provenance validator bug (outside this lane).** `twin.schemas.prereg.active_split_path` resolves the
   `prereg/ACTIVE_SPLIT` entry (`split_a1.csv`) against the repo root, not `prereg/`, so `assert_train_only()` with
   no explicit split raises FileNotFoundError on main. `scripts/cal_v1.py` and `tests/test_site_v1.py` pass the split
   explicitly (`esk.active_split_path`). The fix belongs to the owner of `twin/schemas/prereg.py` (S1a).

## WEAKNESSES touched

- **V-W7** (no calibration tool, no distortion model): partly closed. There is now a k1 per framing, H via
  undistortPoints and findHomography, and a bootstrap. It is validated on a synthetic camera in `tests/test_calib.py`
  (known tilt, roll, k1, h are recovered), not on Blender renders. GeoCalib remains cut.
- **R-W3** (camera gate) and **R-W10** (site graph): not in this item. They are later S2 items (camera gate module,
  TASKS).
