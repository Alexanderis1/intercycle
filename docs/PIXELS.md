# A1: pixels -> phases (rendered CCTV frames into the unchanged phase model)

> **Status (17 Oct 18:50):** the clip is rendering (ETA ~21:20). Code, ROIs and tests are pushed (36d1d2c). Scores: pending.

## Question

Can the phase model in `intercycle/observer/phases.py`, unchanged, recover a weld cell's cycle and sub-phases from
**pixels** (a ceiling CCTV view) as well as it does from the camera **shim** (counts of sim parts per zone)?

## Setup

| Item | Value |
|---|---|
| Sim | `scenario/bodyshop.yaml`, **seed 3** (outside P's train/test seeds), 10:00-10:45 (t 36000-38700 s) |
| What moves | robots follow the sim's kinematics (`render/synth_trace.py --machines sim` sidecar, active fraction every 20 ms): 3 spot holds (sparks) -> swing to the tip dresser -> dress -> return; home when idle/held |
| Camera | `cctv_1` (CAM02), ceiling 9 m, S30 about 20 m away; S40 also in view |
| Render | Blender 5.1 EEVEE, 640x360 (+ overscan), 4 samples, no shadows, no motion blur, no extras, **2 fps** (5,400 frames) |
| CCTV look | `render/cctv/cctv_post` (lens distortion, noise, softening, x264, timestamp, SYNTHETIC label), 640x360 |
| Per-frame time | `D:\icdata\renders\a1\cctv_1\frames.csv` (sim time of every frame) |
| ROIs | `render/a1/cam02_rois.json`: one box per machine (S30, S40), placed by projecting the layout through the camera and checked on a frame (the one-time commissioning step) |
| Split | fit on the first 25 min, score the last 20 min (held out); window holds S30 andon idles of 132 s and 78 s |

Speed note: EEVEE with shadows took 14 s per frame here (its shadow buffer overflows with the hall's 136 lights);
without shadows, 4 samples: 1.85 s. The approved 6-10 fps would have needed ~8 h of GPU; the user chose 2 fps
(one frame per 2 Hz observer tick).

## Method (OpenCV only, Apache-2.0; never sees ground truth)

`intercycle/perception/pixel_phases.py`, per frame:
1. ECC (Euclidean) stabilisation of a half-size frame against the first frame;
2. luma normalised by the frame median;
3. difference to the previous frame > 18 grey levels;
4. 3x3 opening, connected blobs < 12 px dropped (sparks, noise);
5. a ROI where > 50 % of the pixels change at once is skipped for that frame (flash / exposure jump);
6. per ROI: motion pixels, the same pixels in a 4x4 grid, saturated pixels (luma >= 250: the weld-on cue).

Frames are averaged into the observer's 2 Hz ticks. The block per machine is `[motion, 16 grid cells, saturated]`
(18 columns). `PhaseModel.fit_features` takes any width, so phases.py is used as is.

Scoring (`scripts/pixel_phases.py score`): the same protocol and functions as `scripts/phase_discovery.py` eval
(online HMM and offline DTW decode, bins named by majority overlap, segment F1, frame accuracy, boundary F1 at
+-1 s, cycle onset error, period error). Truth = `Kinematics.phase` at each tick (scoring only). The shim runs on the
same seed, window and split.

## Findings so far

- **Weld light is not visible from this camera.** On the test clip the brightest pixel in the S30 box stays at
  200-215 whether the robots weld or not: at 20 m and 640x360, after the CCTV softening, the sparks are a few pixels
  and never saturate. The saturated-pixel column is ~0; only motion carries the signal in this view.
- Shim baseline (seed 3, same split): S30 period 60.3 s (truth 60.5), online coarse segment F1 0.975, accuracy
  0.972, boundary F1 (+-1 s) 0.987, fine accuracy 0.966. S40 coarse F1 0.726, accuracy 0.92.

## Results

Pending (clip rendering).

## Reproduce

```bash
python render/synth_trace.py --movers sim --machines sim --seed 3 --minutes 45 --out D:/icdata/renders/trace_a1.parquet
blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/scene_real.py \
  --trace D:/icdata/renders/trace_a1.parquet --kin D:/icdata/renders/trace_a1.kin.parquet --synth-bodies \
  --out D:/icdata/renders/a1 --cctv-seq cctv_1 --t 36000 --seconds 2700 --fps 2 \
  --engine eevee --samples 4 --no-shadows --res 640x360 --no-blur --extras 0
python -m render.cctv.cctv_post --in D:/icdata/renders/a1/cctv_1/frames --out D:/icdata/renders/a1/cctv_1/cctv \
  --cam CAM02 --cam-label "BODY SHOP S20" --t0 "2026-10-17 10:00:00" --fps 2 --size 640x360 --overscan 0.055 --sensor-scale 1.0
python scripts/pixel_phases.py extract --frames D:/icdata/renders/a1/cctv_1/cctv/frames \
  --csv D:/icdata/renders/a1/cctv_1/frames.csv --rois render/a1/cam02_rois.json --out out/pixels/a1_px.npz
python scripts/pixel_phases.py score --px out/pixels/a1_px.npz --seed 3 --t0 36000 --t1 38700 --split 1500 \
  --out out/pixels/a1_scores.json
python -m pytest tests/test_pixel_phases.py
```
