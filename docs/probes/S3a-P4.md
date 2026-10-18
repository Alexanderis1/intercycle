# Probe P4: press strokes (lane S3a-P4, rerun S3a-P4b)

Sun 18 Oct, ~01:00-02:00 CEST. Plan: PLAN_TWIN §3 P4 and the §2 machines contract. Split in force: `prereg/ACTIVE_SPLIT`
(A1). Clips read only through `esk.load("train")`; every script refuses a clip id outside train. No test clip, frame
or keyframe was opened.

## Result

**2 presses pass the rule (P9a, P9b), both on camera 9. "Press shop" wording: eligible (≥ 2 presses pass).**
The four camera-14 presses do not pass: they are idle in every sampled train clip, so they take the ASSUMED fallback.

| press | what | framings (ROI) | clips | running (≥ 2 strokes) | learned | intervals | period median | p10-p90 | running fraction |
|---|---|---|---|---|---|---|---|---|---|
| P9a | 100 t eccentric press, camera-9 foreground (coil-fed) | c9_f0, c9_f1 | 10 (8 c9_f0, 2 c9_f1) | 9 | **yes** | 24 | **1.96 s** | 1.77-2.00 s | 0.9 |
| P9b | C-frame press behind P9a (white, round dial) | c9_f1 only | 10 | 8 | **yes** | 31 | **1.69 s** | 1.61-1.73 s | 0.8 |
| P14a | camera-14 C-frame press 1 (left) | c14_f0 | 10 | 0 | no (ASSUMED) | 0 | — | — | 0.0 |
| P14b | camera-14 C-frame press 2 | c14_f0 | 10 | 0 | no (ASSUMED) | 0 | — | — | 0.0 |
| P14c | camera-14 C-frame press 3 | c14_f0 | 10 | 0 | no (ASSUMED) | 0 | — | — | 0.0 |
| P14d | camera-14 press 4 (right, blue guard) | c14_f0 | 10 | 0 | no (ASSUMED) | 0 | — | — | 0.0 |

- The two learned periods differ (1.96 vs 1.69 s), so they are two machines, not one artefact seen twice.
- Short intervals (1.32, 1.41, 1.49, 1.52 s) are single outliers at clip starts. Every clip also has a right-censored
  tail (1.2-3.6 s); these are in `out/p4/summary.json` per clip.
- The 2 non-running P9a/P9b clips (0_tr114, 0_te25) have ROI flow p99 ≈ 3 px/s: the press is stopped there, which is
  not a detector miss. P9b 0_tr44 (5 s) has 1 stroke only.
- Camera 14: in all 10 clips the ROI flow p99 is 1-3 px/s when no forklift or person crosses the ROI. There are 3
  single strokes in 3 clips, on a forklift or person pass (p99 24-42 px/s, no person boxes yet). No clip has 2.
  The 15 other camera-14 train clips used for the ROI search also show no periodic motion on any press.
- Not measured: the fifth large press at the top right of c14_f0 (no ram identifiable in the keyframe), and P9b in
  c9_f0 (its slide is hidden behind the P9a crown in that framing).

## Lamp beat

Global luma beat (ROI masked out), per clip: 0.10-0.57 Hz, changing from clip to clip, with power share 0.2-0.85.
Stroke rates are 0.51 Hz (P9a) and 0.59 Hz (P9b). **No press has a stroke rate equal to the global beat**:
- The closest clips are P9a 2_tr18 (beat 0.57 Hz against 0.51 Hz) and P9b 0_tr177 (0.51 Hz against 0.58 Hz).
  Those differences are larger than the frequency resolution.
- The two presses have different, fixed periods in clips whose global beat varies, which a lamp would not give.
- The ROI luma beat does equal the stroke rate (0.51-0.61 Hz, share 0.6-0.86), as expected from a moving ram.

## How the presses and ROIs were found

1. Train keyframes of the three train framings (c9_f0, c9_f1, c14_f0) named the machines (P6 also lists them).
2. Per-cell periodicity of stabilised vertical Farneback flow, on 15-16 train clips per framing (cells 96 px, then
   36 px full-res). A cell counts when one 0.15-3 Hz peak holds ≥ 40 % of the non-DC power and sd ≥ 4 px/s.
   - c9_f0: the cells at (468-576, 612-720) count in 8/8 clips at 0.48-0.60 Hz (P9a).
   - c9_f1: the cells at (396-468, 360-432) count in 8/8 clips at 0.56-0.62 Hz (P9b). The slide of the press behind
     is visibly moving in a train time-montage.
   - c9_f1: the P9a cell sits at the bottom edge (576-720, 960-1078), 0.5-0.6 Hz. Fewer clips count there, because
     people stand in front of it.
   - c9_f0 also has a fixed 2.01 Hz signal at (110-180, 470-580), on the P9a flywheel guard, in 15/16 clips. That is
     rotation, not a stroke, and it is outside every ROI.
   - c14_f0: no press cell is periodic across clips. The only periodic areas are forklift passes.
3. The ROIs are in `docs/probes/S3a-P4_rois.json`, in full-res px per framing. The P9a c9_f0 ROI is on ram-coupled
   motion at the side of the press: the die opening is not in view (P6). The montage alone cannot confirm that it is
   the slide, but the 0.5 Hz periodicity in every clip is clear.

## Detector fix (root cause, in `intercycle/perception/press_motion.py`)

The first run gave 0 strokes everywhere, also on P9a and P9b, where the ROI luma beat was clear. The cause had two
parts:
1. **The noise estimate.** The noise sigma came from the 20th percentile of |v|. A press in continuous run moves on
   every frame, as a smooth near-sinusoid with no dwell, so that estimate is the stroke amplitude itself. The
   threshold (25-45 px/s) sat above the whole signal (p99 ≈ 10 px/s).
2. **The fixed floor.** The 25 px/s floor was above the real ram speed. At this distance and with an ROI mean, the
   ram moves 8-17 px/s.

What changed:
- **Noise.** It is now the MAD of the frame-to-frame differences of v, divided by √2. A smooth stroke barely changes
  between frames, so the differences carry the noise.
- **Smoothing.** `flow_v` gets a centred 0.2 s moving mean before the trigger (strokes are below 3 Hz). The noise
  sigma is divided by √(samples in the window).
- **Floor.** `min_thr` is now 4 px/s. Idle camera-14 ROIs with no occluder have p99 1-3 px/s. This floor was set
  from train data: it is a probe choice, disclosed here.
- **Test.** New test `test_continuous_sinusoidal_ram_is_counted`: a continuous-run ram with 3 px working travel
  (peak ≈ 16 px/s full-res) gives 5 strokes at the true period. The old code gave 0. All earlier tests still pass.

`flow_v` in the machines table is still the raw (unsmoothed) value; only the trigger sees the smoothed one.

## Run

- `python scripts/p4_press_probe.py run --rois docs/probes/S3a-P4_rois.json --per-camera 10` (GPU env, under the
  heavy lock). It writes `out/p4/summary.json` and `$INTERCYCLE_DATA/machines/<press>/<clip>.parquet`.
- The script now takes ROIs per framing (`"rois": {framing: polygon}`). Each press draws 10 train clips (seed
  20261017, a fresh generator per press) from its own framings. Each clip uses the ROI of its own framing (`framing`
  per row). The old version sampled by camera only, so it would have put a c9_f0 ROI on c9_f1 clips.
- Person boxes: none (PERC not available), so MISSING is 0 % everywhere.

## Interpretations

1. "10 camera-9 and 10 camera-14 clips" is read as 10 clips per press, drawn from that press's framings. P9b exists
   only in c9_f1, so its 10 clips are all c9_f1. The 4 camera-14 presses share the same 10 clips.
2. "Running" means ≥ 2 detected strokes in the clip (the pass rule's own threshold).
3. Fallback for P14a-d: running fraction 0/10 in sampled train clips, plus the censored period prior, labelled
   ASSUMED. Nothing was measured for them, so no period is reported.

## Modal cost

$0.00. Everything ran locally.
