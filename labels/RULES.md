# Labelling rule (DEV and TEST labels, PLAN_TWIN §5, U14)

Committed before any dev frame is labelled. Every labeller follows it as written. `rules_sha256` in each label set is
the sha256 of this file.

## Who labels, and from what

- The labeller is a Claude agent (vision model), separate from every lane that tunes PERC/OBS/DYN. It labels from the
  sampled frames alone: no detector proposal, track, motion-detector output, result or other label set is shown.
  Its label set says `blind: true`, `proposals_shown: false`, and every box has `src: "labeller"`.
- Every sampled item in the hashed sample list is labelled, including frames with nobody in them (`boxes: []`).
- Coordinates are native clip pixels (1920 × 1080, origin top-left, x right, y down), on the frame as decoded
  (no undistortion). Times are container PTS seconds.

## What counts

- **Person:** any real human, any pose (walking, standing, bending, sitting, kneeling), including a person seen only
  in part. Not a person: pictures, posters, mannequins, reflections, shadows, clothing on a hook.
- **Forklift (camera 14 only):** a powered industrial truck (counterbalance or reach forklift, with or without a
  load). Its box covers the truck body, mast, forks and load as visible. A hand pallet truck is not a forklift.
  A person riding on a forklift is part of the forklift box and gets no separate person box (the P5 rule removes
  person detections whose foot point is in a forklift box); the frame note says "driver in forklift".
  On camera 9 no vehicle class is labelled: a vehicle seen there is written in the note only, and anyone riding it
  is boxed as a person.

## How to box

- **Visible extent.** The box is the tightest axis-aligned rectangle around the visible pixels of the object
  (head to feet, arms included, carried objects excluded for people; the whole visible truck for a forklift).
- **Occlusion.** If an object is hidden by something in front of it (machine, pallet, another person) and
  more than about 20 % of its full extent is hidden, set `occluded: true`. The box still covers the visible part
  only; it is not extended to guess the hidden part. A person hidden by more than about 90 % (only a hand, or only
  the top of a head with no body) is not boxed; the note says so.
- **Truncation.** An object cut by the image border is boxed up to the border and gets `truncated: true`.
- **Minimum size.** Box a person whose visible box is at least 20 px tall in native pixels, and a forklift at least
  30 px in its larger side. Smaller ones are not boxed; the frame note lists them ("tiny person near (u, v)").
- **Groups.** One box per person, also in a crowd; where two people overlap, each gets its own visible-extent box.
- **Foot point (person).** The point on the floor midway between the two feet where they touch the ground, in
  pixels. If the feet are hidden or cut off, the foot point is the bottom centre of the visible box and the person is
  `occluded: true` (or `truncated: true`); the frame note says "feet not visible". The foot point always lies
  inside its box.
- **Uncertain object.** If the labeller cannot tell whether something is a person (or a forklift), it is not boxed
  and the note describes it with its approximate position, so the ambiguity is reported, not guessed.
- **Blur.** Motion blur does not stop boxing: box the blurred extent of the body.
- **Border check.** Before a frame is closed, the labeller views the four image borders (strips about 200 px
  wide) as full-resolution crops; people sitting or standing at the border are often only partly in view. A dark or
  low-contrast shape in a border strip is judged by stepping through neighbouring frames of the same clip: a shape
  whose pose or position changes is a person (boxed, or named in the note as uncertain); a shape that stays the
  same is background.

## Strokes

- A stroke window is a time window `[t0, t1]` on one press (P9a or P9b, ROIs in `docs/probes/S3a-P4_rois.json`).
- **Stroke time = the bottom dead centre (BDC) of the ram:** the PTS of the frame where the ram (slide), or the
  ram-coupled part visible inside the ROI, reaches the lowest point of its down stroke, i.e. the moment it stops
  going down and starts going up. One stroke per BDC. If the BDC lies between two frames, the earlier frame's
  PTS is used; if the ram dwells at the bottom for several frames, the middle frame of the dwell is used.
- A BDC within the first or last frame of the window counts only if the ram is seen going down before it and up
  after it (otherwise it is not clicked; the note says "edge, not clicked" and gives the excluded span as
  "edge a-b s", from about midway between the nearest clicked BDC and the unclicked one up to the window edge).
- If a person or object hides the ROI so that a BDC cannot be seen, no stroke is clicked for that stretch and the
  note gives the hidden time span as "hidden a-b s".
- Spans are written in seconds with two decimals and rounded outward; a span that runs to a window edge is written
  so that it reaches it (a <= t0, b >= t1), so no frame is left outside it. A scorer ignores detections inside
  hidden and edge spans (`twin/dev_labels.py`, `ScoredDev.ignore`).
- **Running.** A press is running in a window if at least one BDC is seen. A window with the press stopped is
  labelled `strokes: []` with the note "press not running"; it stays in the label set, and the reserve windows
  (`...-r1`, `-r2`, ...) replace such windows in list order for the 20 scored windows.
- **Not scorable (ambiguous).** If the ram (or a ram-coupled part) cannot be identified in the ROI, so that BDCs
  cannot be clicked even though something may move, the window is neither running nor stopped: it is labelled
  `strokes: []`, its note starts with "NOT SCORABLE:", and it is excluded from stroke scoring (status `ambiguous`).
  Reserves replace it like a stopped window.
- **Duplicates.** A byte-identical copy (same sha256, Amendment 1 `dup_of`) of an item already labelled is labelled the
  same way but scored once: status `duplicate` with `dup_of` (for frames: same sha256 and same PTS).
- **The scored set is machine-readable.** `labels/dev/scored.json` gives every labelled item one status: frames
  `scored` | `duplicate`; stroke windows `scored` (running, at least one BDC clicked) | `not_running` | `ambiguous` |
  `duplicate` | `not_needed`. It records the sha256 of each label file and of each sample list it covers.
  Scorers read labels only through `twin/dev_labels.py`, which checks those hashes, rebuilds the statuses from the
  hashed files and fails closed on any difference. The 20 scored windows are exactly 10 per press with distinct clip
  sha256.

## Supplementary lists

- A supplementary sample list (`labels/dev/sample_s1.json`, `scripts/label_sample.py --supplement`) is drawn with a
  seeded RNG and hashed (`sample_s1.sha256`) and committed BEFORE any of its items is viewed. It is drawn from the dev
  inventory reduced to one clip per sha256, and excludes every clip sha256 already sampled for the same press.
- Its frames and its primary stroke windows are all labelled. Its reserve windows are labelled in list order, per
  press, only while that press has fewer than 10 `scored` windows over the base and supplementary lists together;
  the remaining reserves are not labelled and get the status `not_needed`. The list order is seeded random, so this
  stopping rule does not select on content.
- **Reproducing the lists** (GPU env, PyAV; nothing is written): `python scripts/label_sample.py --legacy-base
  --check labels/dev/sample.json` (the base list was drawn before the sha256 dedupe) and `python
  scripts/label_sample.py --supplement labels/dev/sample.json --check labels/dev/sample_s1.json`. The script never
  overwrites an existing list; a new list needs `--out` to a new path.
- Its labels go to a separate label set (`labels/dev/labels_s1.json`) that answers its hash.
- How strokes are seen is written in the label set notes (frame stepping, space-time strip of the ROI pixels).

## Never

- No test clip or frame is opened by the dev labeller. Frames are extracted to the private store
  `$INTERCYCLE_DATA/labelling/` (gitignored) and never committed or published: they show unmasked people.

## Revisions

- **r1 (fix round 1, Sun 18 Oct ~02:50 CEST).** Added: Border check, Not scorable, Duplicates, the scored set and
  Supplementary lists. The base labels (`labels.json`, first made under the previous version of this file, sha256
  `8a3b9ab5984e13350444b26db853619f807738a31c298f75bfa4514e27695b63`) were re-checked under this version, and their
  `rules_sha256` now names this version.
- **r2 (fix round 2, Sun 18 Oct ~03:30 CEST).** Added: stepping through neighbouring frames in the Border check, the
  "edge a-b s" span, spans that reach the window edge, exactly 10 scored windows per press with reserves labelled
  only below the target, and the reproduce commands. Both DEV label sets (made under r1, sha256
  `ba6242456c7397038781d182b576950f1cb828620ebbe58d9ea2cac6dc72ca92`) were re-checked in the left 200 px strip of
  every c9_f1 frame and at the window edges, and their `rules_sha256` now names this version.
