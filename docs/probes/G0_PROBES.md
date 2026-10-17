# G0 probes, lane S3a (staging, P1, P2, P3, P7)

Sat 17 Oct, 22:55-23:25 CEST. Dataset: HF `Voxel51/Safe_and_Unsafe_Behaviours`, revision `7369ff18…`, 691 mp4 clips
(1920×1080, H.264), 96.78 min in total. Credit: Önal & Dandıl (2024), Data in Brief 56, 110791, CC BY 4.0.

## STOP: the pre-registered split is degenerate, and a decision is needed before Pass 2

`scripts/esk_split.py` (commit 55ffd1e, before Pass 1) ran **once** on Modal. Its output is committed as
`prereg/split.csv`, unchanged. It gives **5 groups in total**:

| group | camera | clips | unsafe clips | split |
|---|---|---|---|---|
| c9_g000 | 9 | 271 | 240 | dev |
| c9_g001 | 9 | 312 | 202 | train |
| c9_g002 | 9 | 22 | 18 | test |
| c14_g000 | 14 | 85 | 56 | train |
| c14_g001 | 14 | 1 | 0 | train |

**Cause.** The pHash link rule (≥ min(3, n) frames at Hamming ≤ 6) cannot tell sessions apart on a fixed CCTV
camera. The static background dominates the 8×8 DCT, so any two clips of one camera look like near-duplicates. It
linked 69,948 of 182,710 camera-9 pairs (38 %) and 1,564 of 3,655 camera-14 pairs (43 %). The agglomerative step on
its own, at the committed cut of 8 grey levels, found **21 clusters (camera 9) and 36 clusters (camera 14)**.

**Consequence.** Camera 14 has no dev clips and no test clips. Test is 22 camera-9 clips from one group (4.0 min).

**What I did NOT do, and why.** I did not purge test features, run Pass 2, copy clips, or build keyframes. Purging
would delete the features any re-split needs. Pass 2 would stage clips under a split that may change. I also did not
write `inventory.parquet` into the shared `$INTERCYCLE_DATA`, so no lane starts on this split; a checked copy is in
scratch. Changing the rule after the run is an amendment, so it is the lead's and the user's decision.

**Options for the lead and user:**
- **A0. Keep the split** (the literal §3 P3 fallback). The small-group rule applies with k = 1 test group (camera 9)
  and k = 0 (camera 14). There would be no camera-14 dev or test, so DYN validation and the real-clip test would run
  on camera 9 only.
- **A1. Minimal amendment (recommended).** Drop the pHash link and keep everything else as committed (clustering
  features, the 8-level cut, seed 20261017, 60/20/20, strata). Groups = the agglomerative clusters, 21 + 36.
  Expected test groups: about 4 for camera 9 (< 5, so the small-group rule still applies) and about 7 for camera 14.
  - Risk: 36 clusters from 86 camera-14 clips probably splits real sessions, so near-duplicate leakage across splits
    is possible. This would be disclosed.
  - Code: about 10 lines (a flag in `esk_split.py`, and `run_split` writing `split_a1.csv`). One more run on the
    features still on the volume, about $0.02.
- **A2. Stricter link.** pHash on the foreground (frame minus the clip's median background) instead of the raw frame.
  It needs new code and a dry check on train-only clips, about 30 min.

## P1. Camera and volume

Cameras are taken from the class prefix (exact): 0, 1, 2, 4, 5, 6 → camera 9 and 3, 7 → camera 14. Durations are from
the container PTS.

| | camera 9 | camera 14 | total |
|---|---|---|---|
| all clips | 605 clips, 86.58 min | 86 clips, 10.19 min | 691 clips, 96.78 min |

Under the committed split (`prereg/split.csv`):

| split | camera | clips | clip-seconds | minutes | groups |
|---|---|---|---|---|---|
| train | 9 | 312 | 2318.3 | 38.64 | 1 |
| train | 14 | 86 | 611.6 | 10.19 | 2 |
| dev | 9 | 271 | 2636.7 | 43.94 | 1 |
| dev | 14 | 0 | 0 | 0 | 0 |
| test | 9 | 22 | 240.1 | 4.00 | 1 |
| test | 14 | 0 | 0 | 0 | 0 |

Under this split, N (train minutes) = 48.8 and M (camera-14 train minutes) = 10.2. Both change if A1 or A2 is taken.
Clip-seconds per press ROI are not measured here: ROIs come from CAL and P4.

## P2. Hidden time signal

Verdict: **no timing information.**
- ffprobe on all 691 clips: 0 have a format `creation_time` and 0 have a stream `creation_time`.
- 0 clips have an audio stream.
- The only tags are `major_brand`, `minor_version`, `compatible_brands`, `encoder` and the stream's
  `handler_name`, `language` and `vendor_id`. The encoder tag names a desktop cutting tool (Bandicut), not a recorder.
- HF ships no zip files (691 mp4 + 5 metadata files + 1 gif), so there are no zip mtimes to check. The Mendeley zip
  was not downloaded.
- Container-PTS fps is 24.8-25.1, not 24. The paper says 24 fps and the FiftyOne metadata says 24.83 for 0_tr1.
  The tagged average frame rate agrees with the PTS fps within 1 % on all 691 clips.
  The plan's rule "times from container PTS, never frame_index/24" is needed.

## P3. Sessions, split and balance

- **Group counts per split × camera** (committed split): train 1 (camera 9) + 2 (camera 14); dev 1 + 0; test 1 + 0.
  **That is fewer than 5 test groups per camera on both cameras.** The §5 small-group rule applies: report "held-out
  clips from few sessions". Camera 14 has no test at all; see the STOP section.
- **Authors' tr/te overlap.** All 125 of the authors' test clips sit in a group that also holds authors' train
  clips: 4 of the 5 groups mix tr and te. With groups this coarse, this only shows that the pHash rule links the
  authors' test clips to their train clips. It is not evidence about sessions. The paper does not say how its split
  was made.
- **Balance.** Computed on Modal; for test, only these aggregates left Modal.
  - People per frame: RF-DETR-M (COCO) on the t = 0.5 s frame, person boxes with conf ≥ 0.5.
  - Occlusion share (definition): the share of person boxes (conf ≥ 0.5) with ≥ 10 % of their area covered by any
    other detected box (any class, conf ≥ 0.5).
  - Luma: the mean of the 160×90 luma over the sampled frames.

| split × camera | clips | median length s | luma mean (p10-p90) | people/frame mean (p50, p90) | frames with ≥1 person | occlusion share (boxes) |
|---|---|---|---|---|---|---|
| train × 9 | 312 | 7.0 | 109.2 (101.9-116.5) | 2.40 (2, 4) | 305 | 0.178 (133/748) |
| train × 14 | 86 | 6.5 | 109.5 (104.4-115.0) | 1.38 (1, 2) | 76 | 0.311 (37/119) |
| dev × 9 | 271 | 9.0 | 116.3 (109.5-122.0) | 2.30 (2, 4) | 236 | 0.071 (44/622) |
| test × 9 | 22 | 10.5 | 103.8 (102.1-107.3) | 2.32 (2, 4) | 20 | 0.059 (3/51) |

The splits are not balanced: dev is brighter (+7 levels) and test darker (−5) than train, and occlusion differs
2-3×. This is what single-session splits produce.

## P7. Overlap between the cameras

- **DOI verified on doi.org.** 10.1016/j.dib.2024.110791 resolves to PII S235234092400756X, the article the HF card
  links (Europe PMC: PMC11367630). The HF card's 110756 resolves to a different article (PII S2352340924007224).
- **Paper Fig. 3** is a schematic of the work area, viewed by us and to be redrawn, not reproduced.
  - Camera 9 sits at the junction of the safe walkway and the east-west forklift way and looks west, toward the press
    machine.
  - Camera 14 is on the east wall and looks south, down the north-south forklift way.
  - The viewing directions are roughly perpendicular, and the text never says the views overlap.
- **Matcher.** DISK + LightGlue (kornia 0.8.1; LightGlue was installable), cross-checked with SIFT + ratio test 0.75.
  RANSAC (MAGSAC) fundamental-matrix inliers at 960×540.
- **Frames.** t = 0.5 s frames of 8 camera-9 and 8 camera-14 TRAIN clips (committed split, seeded choice), giving
  64 cross pairs. Only counts left Modal.

| pairs | n | LightGlue inliers min / median / max | SIFT inliers min / median / max |
|---|---|---|---|
| camera 9 × camera 14 | 64 | 7 / 11 / 19 | 8 / 11 / 19 |
| camera 9 × camera 9 (control) | 7 | 1356 / 1467 / 1497 | 801 / 981 / 1061 |
| camera 14 × camera 14 (control) | 7 | 1084 / 1332 / 1401 | 716 / 903 / 1236 |

Verdict: **every cross-camera pair has fewer than 20 inliers**, which is the RANSAC floor on random matches. The
default applies: **2 separate zones, no corridor.** P7 used t = 0.5 s frames instead of keyframes, because the keyframe
pack waits on Pass 2.

## Modal cost (lane S3a staging, app `s3a-stage-esk`, volume `intercycle-esk`)

From `modal billing report --for today`, read at ~23:20 CEST; the last minutes may not be billed yet.

| item | $ |
|---|---|
| Pass 1 (691 clips, 24 containers, 88 s wall) + 3-clip smoke + failed image build | 0.089 |
| split run | 0.015 |
| **staging subtotal** (budget 0.6) | **0.10** |
| RF-DETR on 691 frames (L4) | 0.137 |
| P7 (L4, incl. one failed start) | 0.016 |
| report | 0.012 |
| **probes subtotal** (budget 0.3) | **0.17** |
| **total** | **0.27** |

Still to spend if Pass 2 goes ahead: about $0.1 of compute plus about $0.30 of egress to copy about 7.5 GiB of
train+dev clips to the desktop (Modal egress is $0.04/GiB). That fits inside the staging budget.

## Interpretations

1. Pass 1 downloads with plain HTTPS from the pinned HF revision, without a token; there is no HF secret on Modal.
   Videos lived only in container /tmp. Only features, the t = 0.5 s frame and metadata reached the volume.
2. The split script ran on Modal, next to the features, so no test-clip image feature ever reached the desktop.
3. "Object layout" = a 16×9 grid of the Sobel gradient magnitude of the clip's median background (where static
   objects stand), not person positions, so the layout does not change with who walks by within a session.
4. For test clips, the §2 inventory columns (clip_id, camera, class, group, split, sha256, PTS duration) leave Modal,
   because the §2 contract requires them and the duration is already public in the HF metadata. Nothing else
   per-clip does: no pHash, luma, people counts or ffprobe tags.
5. Volume name `intercycle-esk` (shared press-shop store, per the §2 note "mirrored on the Modal volume"); app name
   `s3a-stage-esk`.
6. framing_id is provisional: `c9_f0` / `c14_f0`, one framing per camera until P6.
7. The keyframes would be at 1/3 and 2/3 of each clip's PTS span, 960 px wide (code is ready in `pass2_clip`).
