# Label agreement: two independent Claude-agent labellers on 20 DEV frames (U14)

Lane S4-labels-blind, Sun 18 Oct 2026 ~03:15 CEST. PLAN_TWIN Â§0 U14 rule (1) and Â§5 "Split and labels".

## What was compared

- **Labeller A** (`agent:dev-labeller-a`): the DEV label set `labels/dev/labels.json`.
- **Labeller B** (`agent:dev-labeller-b`, this lane): `labels/dev/labels_blind_B.json`. It is **blind**. B labelled from
  the frames alone, under `labels/RULES.md` (r2). B never saw A's labels, any detector proposal or track, or any
  result. B's set was committed (06cd03a) **before** A's labels were opened for this comparison.
- **Frames:** the first 20 frame items of the hashed sample list `labels/dev/sample.json`, in list order
  (`dev-f-c9-01` â€¦ `dev-f-c9-20`). All 20 are camera 9, from 6 dev groups (c9_a000, a001, a004, a012, a015, a019) and
  both camera-9 framings. B extracted them with PyAV (`scripts/label_extract_b.py`) to its own private store
  `$INTERCYCLE_DATA/labelling_b/`. That store is gitignored and holds unmasked people, so it is never committed.
- **Duplicate:** `dev-f-c9-14` is the same clip bytes and the same PTS as `dev-f-c9-03`; the scored set
  (`labels/dev/scored.json`) marks it `duplicate`. The main figures below use the **19 scored frames**. The 20-frame
  figures are given too.
- **Method** (`scripts/label_agreement.py`, tests in `tests/test_label_agreement.py`):
  - person boxes are matched one to one, greedily by IoU (highest first), and a match needs IoU â‰¥ 0.5 (the same rule
    as the real-clip test, PLAN_TWIN Â§5);
  - recall of B against A = matched / A boxes;
  - precision of B against A = matched / B boxes;
  - foot-point error = the pixel distance between the two foot points of a matched pair;
  - 95 % intervals are frame-level Clopper-Pearson. They **ignore clustering**: the boxes come from only 6 groups,
    and the same seated operators appear in several frames.
- The machine-readable result is `labels/dev/agreement_blind_B.json`. It records the sha256 of both label files and of
  scored.json. A's files are read through `twin.dev_labels.load_scored`, which fails closed on any hash mismatch.

## Results

| Set | Frames | A boxes | B boxes | Matched | Recall B vs A (95 % CI) | Precision B vs A (95 % CI) |
|---|---|---|---|---|---|---|
| **Scored frames** | 19 | 89 | 87 | 85 | **0.955** (0.889-0.988) | **0.977** (0.919-0.997) |
| All 20 frames (incl. duplicate) | 20 | 95 | 93 | 91 | 0.958 (0.896-0.988) | 0.978 (0.925-0.997) |
| Scored frames, A boxes neither occluded nor truncated | 19 | 26 | â€“ | 26 | 1.000 (0.868-1.000) | â€“ |

- **Box overlap of matched pairs** (scored frames): median IoU 0.88, mean 0.85.
- **Both labellers flag most boxes:** A marks 51 of 89 boxes occluded and 30 truncated; B marks 48 of 87 occluded and
  29 truncated. Most people in these camera-9 frames are operators behind machines or cut by the image border.

**Foot-point pixel error** (matched pairs, native 1920 Ã— 1080 pixels):

| Pairs | n | Median | Mean | p90 | Max |
|---|---|---|---|---|---|
| Scored frames, all matched | 85 | 4.1 px | 7.5 px | 12.6 px | 56.0 px |
| Scored frames, A box neither occluded nor truncated | 26 | 5.9 px | 8.5 px | 10.7 px | 56.0 px |

**Per group** (scored frames; recall and precision of B against A):

| Group | Frames | A boxes | B boxes | Matched | Recall | Precision |
|---|---|---|---|---|---|---|
| c9_a000 | 4 | 20 | 18 | 18 | 0.90 | 1.00 |
| c9_a001 | 3 | 17 | 17 | 17 | 1.00 | 1.00 |
| c9_a004 | 4 | 12 | 12 | 12 | 1.00 | 1.00 |
| c9_a012 | 3 | 12 | 12 | 11 | 0.92 | 0.92 |
| c9_a015 | 3 | 18 | 18 | 18 | 1.00 | 1.00 |
| c9_a019 | 2 | 10 | 10 | 9 | 0.90 | 0.90 |

## Where the two labellers disagree

**Unmatched boxes** (scored frames; 4 A-only, 2 B-only). All of them are partly hidden or tiny, and each one is a
judgement call that the labelling rule leaves open:

| Frame | Who | Box (x0, y0, x1, y1) | What it is |
|---|---|---|---|
| dev-f-c9-01 | A only | 0, 350, 15, 470 | A narrow dark shape at the far-left press, cut by the left border. B's note lists it as uncertain (no face or limb seen) and did not box it. |
| dev-f-c9-07 | A only | 0, 348, 13, 470 | The same spot in another clip. B's note: uncertain at this frame, a person appears there 1 s later. |
| dev-f-c9-05 | A only | 182, 136, 218, 200 | The seat at the upper-left press. B boxed a seated operator there in 7 other frames, but in this frame saw no head or limb and listed it as uncertain. |
| dev-f-c9-05 | B only | 1442, 8, 1472, 60 | A person walking away in the far aisle near the top border (about 52 px tall). A did not box it. |
| dev-f-c9-08 | A and B | A 302, 97, 377, 218; B 292, 102, 346, 215 | The same operator at the upper-left press, boxed by both, but IoU 0.487 is just under 0.5. A's box reaches further right (to x 377, B's to x 346). |

**Foot points.** The median error is small (about 4-6 px). The large errors come from a few people whose feet are
hidden, where the two labellers drew the bottom of the box at different places:

- **Seated operator on a stool at the press**, seen from behind (dev-f-c9-04, -10, -15; 50 px each). A's box runs
  down to y â‰ˆ 985, B's stops at y â‰ˆ 935, where B judged the visible body ends above the stool.
- **Person at the coil line behind a forklift** (dev-f-c9-06; 56 px). A marked this person not occluded, with feet at
  y â‰ˆ 588. B judged the legs hidden by the forklift and stopped the box at y â‰ˆ 532.
- **Operator at the press behind the column** (dev-f-c9-03 and its duplicate -14; 35 px). The two boxes end at
  different heights.
- **Walker in the aisle** (dev-f-c9-02; 34 px): A put the feet at y â‰ˆ 750, B at y â‰ˆ 716.

## What this means

- **Noise floor.** These numbers are the inter-agent noise floor that U14 rule (3) asks for. Every real-clip figure is
  stated "against independent Claude-agent labels (vision model), not human labels". On camera-9 DEV frames, two such
  labellers agree on about 95-96 % of A's person boxes and 98 % of B's at IoU 0.5. Every disagreement is a partly
  hidden or border-cut person.
- **Foot points.** Inter-labeller foot-point error has a median of about 4 px and a p90 of about 13 px. A tail of about
  35-56 px comes from people whose feet are hidden.
- **Recall threshold r0.** PLAN_TWIN Â§5 says r0 is set from these 20 blind DEV frames only. This lane reports the
  numbers and does **not** choose r0.

## Limits

- Only camera 9 is covered. The first 20 frame items of the sample list are all camera-9 frames; the camera-14 frames
  come after them in list order. So this comparison says nothing about camera 14 or forklifts. On camera 9, the rule
  labels no vehicle class.
- The frames come from 6 groups, and several frames show the same seated operators. The intervals above ignore that
  clustering.
- Both labellers are Claude agents (vision models), not humans. They may share failure modes, so agreement between
  them is not accuracy.
- A's set was made before r1/r2 of the rule and re-checked under r2 (`labels/RULES.md` "Revisions"). B labelled under
  r2 directly.
