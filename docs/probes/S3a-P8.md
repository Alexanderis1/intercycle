# Probe P8 (faces), lane S3a-P8

Sat 17 Oct, about 23:30 CEST. PLAN_TWIN §3 P8, §2 privacy row, U7, WEAKNESSES G-W1.

## Verdict

**Faces are recognisable at this size, so faces-only blurring is not enough. The plan's rule applies:** every person box
at conf ≥ 0.1 is black-boxed whole, padded 15 %, and held over its track. The rule is now in
`intercycle/perception/privacy_mask.py`, has unit tests, and `scripts/esk_mask.py` writes masked frames for train
and dev only.

## Method

- **Frames.** The t = 0.5 s frame (1920×1080) of 100 TRAIN clips of the committed `prereg/split.csv`. Clips were drawn
  with `numpy.default_rng(20261025)`, stratified by camera in proportion to the 398 train clips: 78 camera-9 frames and
  22 camera-14 frames. Everything ran on Modal next to the frames (`cloud/modal_p8_faces.py`, app `s3a-p8-faces`).
  Only aggregate numbers left Modal, and no image was viewed by anyone.
- **People.** RF-DETR-M (COCO) person boxes. Boxes at conf ≥ 0.1 are counted. The size statistics use only boxes at
  conf ≥ 0.5, because many 0.1-0.5 boxes are not people.
- **Face size estimated from the person box.** We use the plan's walker height of 1.72 m. A standing adult's head is
  about 0.23 m tall and the face about 0.15 m wide, so:
  - pixel density on the person = h / 1.72 px/m;
  - face width ≈ 0.087 × h px.
  Only "full-body" boxes are used for this: boxes that do not touch the frame border and have height/width ≥ 1.8.
- **Faces measured directly.** We ran the YuNet face detector (OpenCV Zoo, score ≥ 0.6) on the top 45 % of every
  conf ≥ 0.5 person box, enlarged 2×. For each box we record the width of the widest face found, in source pixels.
- **What counts as identifiable.** We use the IEC 62676-4 (DORI) pixel densities: 125 px/m to **recognise** a known
  person and 250 px/m to **identify** someone.

## Numbers

| | camera 9 (78 frames) | camera 14 (22 frames) |
|---|---|---|
| person boxes, conf ≥ 0.1 / ≥ 0.5 | 781 / 187 | 264 / 32 |
| full-body boxes | 88 | 14 |
| px/m on the person, median (p10-p90, max) | **146** (76-229, max 263) | **80** (49-128, max 138) |
| estimated face width px, median / p90 | **22.0 / 34.3** | **12.0 / 19.2** |
| estimated head height px, median / p90 | 33.7 / 52.6 | 18.4 / 29.5 |
| boxes where YuNet found a face | 86 / 187 (46 %) | 9 / 32 (28 %) |
| YuNet face width px, median / p90 (max) | **22.8 / 36.7** (49.8) | **18.9 / 22.6** (23.0) |

Histogram of the estimated face width (px), bins 0-5-10-15-20-25-30-40-60-100:
- camera 9: 0, 2, 27, 11, 10, 17, 21, 0, 0;
- camera 14: 0, 6, 3, 3, 2, 0, 0, 0, 0.

Histogram of the YuNet face width (px), same bins:
- camera 9: 0, 0, 22, 14, 11, 11, 21, 7, 0;
- camera 14: 0, 0, 1, 5, 3, 0, 0, 0, 0.

**Are faces identifiable?**
- **Camera 9:** yes, in part.
  - The median person is above the "recognise" density (146 vs 125 px/m).
  - The closest 10 % come close to the "identify" density (p90 229, max 263 vs 250 px/m).
  - Faces measure 20-50 px wide, and a face detector finds a face in almost half of the person boxes.
  - Co-workers could recognise each other, and the closest people may be identifiable.
- **Camera 14:** mostly below the "recognise" density (median 80 px/m). The p90 (128 px/m) reaches it.
- **Faces-only blur is not safe here.** A face detector misses many faces: YuNet found a face in only 46 % of camera-9
  boxes, including people seen from behind. Body, clothing and gait still show who someone is.

## Verdict against the plan

- §3 P8 sets no numeric pass rule. Its listed fallback is the rule that U7 tightened and that §2 makes mandatory
  ("every person box is black-boxed whole").
- The histogram confirms the fallback is needed, and we take it: whole-box black, conf ≥ 0.1, 15 % padding, held over
  the track.

## Implementation

**`intercycle/perception/privacy_mask.py`** (numpy only)
- `MaskConfig`: conf 0.1, pad 0.15, hold_s 1.0, grow_per_s 0.5.
- `TrackHoldMasker.update(t, xyxy, score, track_id=None)` returns the boxes to black out in each frame:
  - Every box at conf ≥ 0.1 is masked in its own frame, whether or not a tracker has confirmed it.
  - Boxes without a track id are joined to held tracks by greedy IoU, so a 0.1-conf box below ByteTrack's activation
    threshold still belongs to a track.
  - After a missed detection, the track's last padded box stays masked for up to 1 s. It grows by 0.5 box-heights per
    second of gap on each side.
  - Frames must arrive in time order, and `reset()` is called between clips.
- `apply_mask(image, boxes)` fills each box with black, rounding the box edges outwards.

**`scripts/esk_mask.py`**
- Reads `$INTERCYCLE_DATA/inventory.parquet` through `esk.load`, which refuses TEST. `--split` accepts only train and
  dev, every row is checked again, and test clip ids passed with `--clips` are rejected.
- Decodes `clips/<clip>.mp4` on a fixed PTS grid (default 2 fps) and runs RF-DETR (conf 0.1, person) on every frame
  it writes.
- Writes `blurred/<split>/<clip>/<t_ms>.jpg` plus `mask.json`. With `--crop`, the frame is masked before it is cropped.
  With `--keyframes`, it also writes `blurred/keyframes/<clip>_k{1,2}.jpg`.

**Tests**
- `tests/test_privacy_mask.py` (9 tests): the 0.1 threshold, 15 % padding, low-conf boxes without an id, holding and
  growing through a gap, the hold expiring, IoU continuation, two people, time order and reset, clipping and filling.
- `tests/test_esk_mask.py` (4 tests): a synthetic mp4 with a stub detector. Checks that the person is blacked out, file
  names follow PTS, the test split and test ids are rejected, and the crop is taken after masking. These tests skip
  when PyAV is missing (the repo `.venv`) and run in the GPU env.

## Not done / open

- `esk_mask.py` has not run on real clips. Pass 2 (local clips and keyframes) is held until the split decision in
  `G0_PROBES.md`, and `inventory.parquet` is not yet in `$INTERCYCLE_DATA`. Run it after Pass 2 under the heavy lock
  (GPU env).
- The plan requires a human review of every masked real frame that is published (§6 video). That review is not done.
- If the lead takes split amendment A1 or A2, some of these 100 train clips may move to test. No image left Modal, so
  only aggregate numbers would have touched them, which the sealed rule allows.

## Cost

Modal app `s3a-p8-faces`: **$0.010** (L4, one run, including the image layer for OpenCV and YuNet), from
`modal billing report --for today`.
