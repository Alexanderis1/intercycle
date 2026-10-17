# Probe P4: press strokes (lane S3a-P4)

Sat 17 Oct, ~23:15-00:00 CEST. Plan: PLAN_TWIN §3 P4 and the §2 machines contract.

## Status: BLOCKED on data. The code is ready; no press was measured.

No train video is available to this lane:
- `$INTERCYCLE_DATA` (`F:\intercycle\data\esk`) has no `inventory.parquet`, no `clips/` and no `keyframes/`.
- The Modal volume `intercycle-esk` holds only `esk/{meta,frame05,feat,det,split}`. There are no clips, because
  Pass 2 is on hold (see `docs/probes/G0_PROBES.md`, "STOP": the committed split is degenerate and the lead and user
  must choose A0, A1 or A2).
- I waited 20 minutes for Pass 2 (polling every 60 s). It did not land.

I did not work around this, for two reasons:
1. **Sealed-test risk.** Under A1 or A2, clips that are train under the committed `prereg/split.csv` can become
   test. That is certain for camera 14, which has no test clips today but would get about 7 test groups under A1.
   Looking at any camera-14 frame now could mean looking at a future test clip, and that cannot be undone.
2. **No HF access.** The plan says "agents never call the HF dataset after Pass 1", so I did not download clips myself.

Because of this, the presses are **not identified yet**. There are no ROI polygons, period estimates or
running-clip counts. **The pass rule was not evaluated.** The plan's fallback (fewer than 2 presses pass: running
fraction plus a censored period prior, labelled ASSUMED) is **not taken by me**. Without measurements it would only be
a prior. Whether to declare it now or re-run P4 after Pass 2 is the lead's call.

## What is ready

| Path | What |
|---|---|
| `intercycle/perception/press_motion.py` | The per-clip press-motion ingest, shared by dev and test. Steps: (1) decode with PyAV; times come from the container PTS (`frame.time`), never index/fps. (2) ECC (Euclidean) stabilisation onto frame 0, with the ROI masked out of the fit. (3) Farneback vertical flow inside the ROI polygon, giving `flow_v` in full-res px/s (+ = down), plus `roi_energy`. (4) MISSING (NaN, no stroke) on every frame whose ROI touches a person box. (5) Stroke starts: a Schmitt trigger on the down lobe, with an adaptive threshold (5 × noise σ from the 20th percentile of \|v\|, floor 25 px/s), confirmed by an up lobe within 3 s and debounced at 0.3 s. (6) Lamp beat: the dominant frequency and power share of the ROI-masked global luma, and of the ROI luma. (7) In-clip intervals plus the right-censored tail. The output columns are exactly the `MachineRow` contract (`twin/schemas/tables.py`, lane S1a). |
| `tests/test_press_motion.py` | Synthetic clips: a textured ram, camera shake, a 0.5 Hz lamp beat and person boxes. 7 pass; the contract check is skipped on this branch (`twin` is not merged here). |
| `scripts/p4_press_probe.py` | `motion`: per-clip vertical-flow σ grids (the top cells in full-res px), to find rams with numbers before an ROI is drawn on a keyframe. `run`: a seeded sample (seed 20261017) of N train clips per camera, the ingest per press ROI, `$INTERCYCLE_DATA/machines/<press>/<clip>.parquet`, and a JSON summary with the pass rule (≥ 8 clips with ≥ 2 strokes). Both commands refuse any clip id that is not in `esk.load("train")`. |

A smoke test on a synthetic H.264 mp4 (1920×1080, 25 fps, PyAV encode → `ingest_clip`) recovered PTS times
0, 0.04, 0.08 … and strokes at 0.04, 1.64 and 3.24 s, for a true period of 1.6 s.

## How to finish P4 once Pass 2 has staged train clips and keyframes (about 35 min)

1. Look at train keyframes of camera 9 and camera 14 (train only, after the split is final). Name each press and
   draw its ram ROI in full-res px per framing in `docs/probes/S3a-P4_rois.json`.
2. Optional: `python scripts/p4_press_probe.py motion --clips <5 train clips per camera>` to confirm that the ROI
   sits on the highest vertical-flow σ cells.
3. Take the heavy lock, then run `python scripts/p4_press_probe.py run --rois docs/probes/S3a-P4_rois.json
   --per-camera 10`. Run it with the GPU env python, which has PyAV, and with `PYTHONPATH` set to the worktree.
4. Fill this report: presses found, which passed, period median and p10-p90, running fraction, and lamp beat (flag
   any press whose stroke rate equals the global luma beat).

## Interpretations

1. "Running" = ≥ 2 detected strokes in the clip (the pass rule's own threshold).
2. The probe runs without person boxes, because the PERC batch is not available. Its MISSING share is therefore 0.
   The ingest takes boxes once PERC exists.
3. The sealed rule is read as covering any clip that may become test under the pending split decision, not only
   the current test rows.

## Modal cost

$0.00 (only `modal volume ls` was run).
