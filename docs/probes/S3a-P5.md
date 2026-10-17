# Probe P5: forklifts (lane S3a-P5)

Sat 17 Oct, about 23:15-23:45 CEST. Plan: PLAN_TWIN §3 P5. TRAIN clips only (prereg/split.csv, `split == train`).

## Verdict

- **Camera 14: forklifts are present and detectable.** 13 of the 20 checked camera-14 frames show a forklift. RF-DETR
  "truck" at conf ≥ 0.5 and OWLv2 "forklift" at conf ≥ 0.3 were both right on every box they drew in the check
  (8/8 and 10/11, with the 11th box a second box on the same forklift). Each found 8-10 of the 13 forklifts.
- **Camera 9: no vehicles observed in zone-9 train clips.** Every RF-DETR vehicle box on camera 9 is a fixed object.
  305 of 312 frames have a "car"/"motorcycle" box on a steel coil lying on the floor, or a "truck" box on the
  press. OWLv2 never scored above 0.26 on camera 9, and the boxes it drew were machine parts and a control
  cabinet. The camera-9 road showed only people in every frame checked.
- **Result: P5 passes for camera 14, with the disclosure for camera 9.** Forklift paths for zone 14 can come from
  detections. Zone 9 has no forklifts in its training data. The fallback (a path library from motion blobs) was
  **not taken**.
- The rule "suppress any person box whose foot point is inside a forklift box" is implemented in
  `intercycle/perception/vehicles.py` (`suppress_people_on_forklifts`) and tested in `tests/test_vehicles.py`
  (5 tests pass). On all 398 train frames it mostly removes forklift drivers, with one wrong suppression per
  detector (table below).

## Method

- **Frames.** Pass 2 (clips and keyframes) has not run, and `$INTERCYCLE_DATA` had no clips or inventory. So the probe
  used the one frame per clip that exists: the t = 0.5 s frame from Pass 1 (`esk/frame05/` on the Modal volume
  `intercycle-esk`). 398 train frames were copied (312 from camera 9, 86 from camera 14), read one file at a time by
  train id from prereg/split.csv. No dev or test id was read. They are stored in `$INTERCYCLE_DATA/p5/` (gitignored).
- **Detectors** (local GPU, RTX 4060 Ti, run under the heavy lock), from `scripts/p5_forklift.py`:
  - RF-DETR-M (COCO, rfdetr 1.11.1) at conf ≥ 0.3, keeping car, truck, bus, motorcycle and bicycle. Person boxes
    are kept for the suppression check.
  - OWLv2 `google/owlv2-base-patch16-ensemble` (transformers 5.18), query "a photo of a forklift", conf ≥ 0.1.
    Boxes are scaled back through the padded square input.
- **Agent check** (reported as an agent check, **not a human check**). This agent looked at 40 frames: 20 per
  camera, seeded sample (seed 20261017), with the detector boxes drawn on them (RF-DETR ≥ 0.3, OWLv2 ≥ 0.2). For each
  frame it judged whether a forklift body was visible, and for each box whether it covers one.
  - A second box on a forklift that already has a correct box counts as a **duplicate**.
  - A load carried without any visible forklift body counts as **no forklift**.
  - The agent also looked at all 11 other camera-9 frames where a box fell outside the coil and press positions
    (10 OWLv2 boxes ≥ 0.2 and 1 RF-DETR "bicycle"), and at every frame where the suppression rule fired.

## Numbers

**Detections on all train frames** (frames with at least one box / number of boxes):

| camera (frames) | RF-DETR vehicle ≥ 0.3 | RF-DETR vehicle ≥ 0.5 | RF-DETR truck ≥ 0.5 | OWLv2 ≥ 0.1 | OWLv2 ≥ 0.2 | OWLv2 ≥ 0.3 |
|---|---|---|---|---|---|---|
| 9 (312) | 305 / 470 | 218 / 219 | 1 / 1 | 283 / 729 | 10 / 11 | 0 / 0 |
| 14 (86) | 54 / 79 | 38 / 39 | 38 / 39 | 85 / 600 | 59 / 112 | 43 / 51 |

Classes at ≥ 0.3:
- Camera 9: car 274, truck 130, motorcycle 65, bicycle 1. These are the coil, the press and one far walker.
- Camera 14: truck 79.
- Highest score on camera 9: RF-DETR truck 0.509 (the press); OWLv2 0.257.

**Agent check on 40 frames.** Forklift visible in camera 9: 0 of 20 frames. Camera 14: 13 of 20 frames.

| detector, threshold | camera | boxes | correct | duplicate | false | precision (correct / boxes) | forklifts found |
|---|---|---|---|---|---|---|---|
| RF-DETR vehicle ≥ 0.3 | 9 | 30 | 0 | 0 | 30 | 0.00 | 0 of 0 |
| RF-DETR vehicle ≥ 0.3 | 14 | 15 | 10 | 4 | 1 | 0.67 (0.91 without duplicates) | 10 of 13 |
| RF-DETR truck ≥ 0.5 | 9 | 0 | 0 | 0 | 0 | n/a | 0 of 0 |
| RF-DETR truck ≥ 0.5 | 14 | 8 | 8 | 0 | 0 | 1.00 | 8 of 13 |
| OWLv2 ≥ 0.2 | 9 | 0 | 0 | 0 | 0 | n/a | 0 of 0 |
| OWLv2 ≥ 0.2 | 14 | 27 | 13 | 8 | 6 | 0.48 (0.68 without duplicates) | 13 of 13 |
| OWLv2 ≥ 0.3 | 9 | 0 | 0 | 0 | 0 | n/a | 0 of 0 |
| OWLv2 ≥ 0.3 | 14 | 11 | 10 | 1 | 0 | 0.91 (1.00 without duplicates) | 10 of 13 |

How these misses and false boxes break down:
- Missed forklifts were mostly far away or hidden behind their load.
- The single false RF-DETR box on camera 14 is a stack of bins with no visible forklift.
- 5 of the 6 false OWLv2 boxes at ≥ 0.2 sit on one fixed object at the far right, near (1635-1755, 55-245) px.
- n is small: 20 frames per camera from one session group each. These are not calibrated precision estimates.
- The operating points (truck ≥ 0.5, OWLv2 ≥ 0.3) were read off these same 40 frames, so they are not independent
  of this check.

**Suppression rule on all 398 train frames.** Every suppressed box was checked by eye. Several frames have 2
person boxes on one driver.

| forklift source | camera 9 suppressed | camera 14 suppressed | wrong suppressions |
|---|---|---|---|
| RF-DETR truck ≥ 0.5 | 1 of 1139 person boxes | 11 of 163 (all drivers) | 1: camera 9 `2_tr107`. The "truck" box (0.509) is the press, and a worker at the press loses his box. |
| OWLv2 ≥ 0.3 | 0 of 1139 | 15 of 163 | 1: camera 14 `7_te7`. A seated press operator's foot point falls inside the box of a blurred forklift passing in front of him. |

## Interpretations and caveats

- **Local GPU instead of Modal L4.** The lead's task asked for local GPU, while the plan says Modal L4. Modal cost $0.
  The only Modal use was reading 398 small files from the volume through the client.
- **One frame per clip.** Pass 2 had not run, so each clip gives only its t = 0.5 s frame. A forklift that crosses
  camera 9 later in a clip would be missed. "No vehicles observed" therefore means "none in the t = 0.5 s frames of
  the 312 camera-9 train clips".
- **"Truck/car classes"** was read as all COCO vehicle classes (car, truck, bus, motorcycle, bicycle).
- **Degenerate split.** The split in prereg/split.csv is degenerate and still waits on a decision (G0_PROBES.md). If
  the split is amended, an amendment must not move any clip this agent looked at into TEST. Those clips are:
  - The 40 checked frames: 2_tr46, 0_tr134, 4_tr16, 2_tr117, 2_te11, 5_te11, 5_tr17, 5_tr2, 5_tr16, 6_te13, 6_te12,
    2_tr69, 6_tr16, 2_tr114, 4_tr8, 5_tr11, 4_te2, 6_te10, 6_te5, 1_tr53, 3_tr22, 3_tr23, 3_tr20, 3_tr31, 3_tr39,
    7_te2, 7_tr17, 3_tr18, 3_te1, 3_tr4, 3_te2, 7_tr16, 3_tr11, 3_tr44, 3_tr46, 7_tr10, 3_tr41, 7_te8, 3_tr21, 3_tr3.
  - The other frames viewed: 0_tr123, 0_tr142, 1_tr55, 1_tr59, 2_tr105, 2_tr18, 2_tr56, 4_te4, 5_te3, 5_tr3, 6_tr11,
    2_tr107, 3_tr1, 3_tr36, 3_tr45, 7_te6, 7_tr18, 7_tr19, 7_tr3, 3_tr26, 7_te7, 7_tr13.
  - The `te`/`tr` in these ids is the authors' naming. All of them are TRAIN in prereg/split.csv.
- **Privacy.** No image is committed. The frames and overlays stay in the gitignored data root and the session
  scratchpad.
- **Not decided here (for the lead).** This probe does not decide:
  - which forklift source PERC uses;
  - whether the suppression rule should run only on camera 14 (that would avoid the camera-9 press error).
