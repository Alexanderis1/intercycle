# Labelling rule for the TEST labeller (PLAN_TWIN §5, U14)

This is the only file the test labeller reads. It is committed before any test item is labelled. `rules_sha256` in
the test label set is the sha256 of this file's bytes.

## Who labels, and from what

- The labeller is a fresh Claude agent (vision model) started inside the custodian session. It labels from the
  extracted test frames, crops and strips alone: no detector proposal, track, motion-detector output, result, code or
  other label set is shown, and none is looked for.
- Its label set says `split: "test"`, `labeller_kind: "claude_agent"`, `blind: true`, `proposals_shown: false`, and
  every box has `src: "labeller"`.
- Every item of the hashed sample list is labelled exactly once: every frame, including frames with nobody in them
  (`boxes: []`), and every stroke window, reserve windows (`...-r1`, `-r2`, ...) included.
- Coordinates are native clip pixels (1920 × 1080, origin top-left, x right, y down), on the frame as decoded (no
  undistortion). Times are container PTS seconds, copied from the sample list and the extracted files.

## What counts

- **Person:** any real human, any pose (walking, standing, bending, sitting, kneeling), including a person seen only
  in part. Not a person: pictures, posters, mannequins, reflections, shadows, clothing on a hook.
- **Forklift (camera 14 only):** a powered industrial truck (counterbalance or reach forklift, with or without a
  load). Its box covers the truck body, mast, forks and load as visible. A hand pallet truck is not a forklift.
  A person riding on a forklift is part of the forklift box and gets no separate person box; the frame note says
  "driver in forklift". On camera 9 no vehicle class is labelled: a vehicle seen there is written in the note only,
  and anyone riding it is boxed as a person.

## How to box

- **Visible extent.** The box is the tightest axis-aligned rectangle around the visible pixels of the object
  (head to feet, arms included, carried objects excluded for people; the whole visible truck for a forklift).
- **Occlusion.** If an object is hidden by something in front of it (machine, pallet, another person) and more than
  about 20 % of its full extent is hidden, set `occluded: true`. The box still covers the visible part only; it is not
  extended to guess the hidden part. A person hidden by more than about 90 % (only a hand, or only the top of a head
  with no body) is not boxed; the note says so.
- **Truncation.** An object cut by the image border is boxed up to the border and gets `truncated: true`.
- **Minimum size.** Box a person whose visible box is at least 20 px tall in native pixels, and a forklift at least
  30 px in its larger side. Smaller ones are not boxed; the frame note lists them ("tiny person near (u, v)").
- **Groups.** One box per person, also in a crowd; where two people overlap, each gets its own visible-extent box.
- **Foot point (person).** The point on the floor midway between the two feet where they touch the ground, in pixels.
  If the feet are hidden or cut off, the foot point is the bottom centre of the visible box and the person is
  `occluded: true` (or `truncated: true`); the frame note says "feet not visible". The foot point always lies inside
  its box.
- **Uncertain object.** If the labeller cannot tell whether something is a person (or a forklift), it is not boxed
  and the note describes it with its approximate position, so the ambiguity is reported, not guessed.
- **Blur.** Motion blur does not stop boxing: box the blurred extent of the body.
- **Border check.** Before a frame is closed, the labeller views the four image borders (strips about 200 px wide) as
  full-resolution crops (`crop` command below); people sitting or standing at the border are often only partly in
  view. A dark or low-contrast shape in a border strip that cannot be told apart from background is an uncertain
  object (not boxed, described in the note).

## Strokes

- A stroke window is a time window `[t0, t1]` on one press (P9a or P9b); the extracted window file holds the press
  ROI crop of every frame of the window.
- **Stroke time = the bottom dead centre (BDC) of the ram:** the PTS of the frame where the ram (slide), or the
  ram-coupled part visible inside the ROI, reaches the lowest point of its down stroke, i.e. the moment it stops going
  down and starts going up. One stroke per BDC. If the BDC lies between two frames, the earlier frame's PTS is used;
  if the ram dwells at the bottom for several frames, the middle frame of the dwell is used.
- A BDC within the first or last frame of the window counts only if the ram is seen going down before it and up after
  it (otherwise it is not clicked; the note says "edge, not clicked" and gives the excluded span as "edge a-b s", from
  about midway between the nearest clicked BDC and the unclicked one up to the window edge).
- If a person or object hides the ROI so that a BDC cannot be seen, no stroke is clicked for that stretch and the note
  gives the hidden time span as "hidden a-b s".
- Spans are written in seconds with two decimals and rounded outward; a span that runs to a window edge is written so
  that it reaches it (a <= t0, b >= t1), so no frame is left outside it.
- **Running.** A press is running in a window if at least one BDC is seen. A window with the press stopped is labelled
  `strokes: []` with the note "press not running".
- **Not scorable (ambiguous).** If the ram (or a ram-coupled part) cannot be identified in the ROI, so that BDCs
  cannot be clicked even though something may move, the window is labelled `strokes: []` and its note starts with
  "NOT SCORABLE:".
- Reserve windows are labelled like every other window; which windows are scored is decided later from the list
  order, not by the labeller.
- How strokes are seen is written in the window note (for example "space-time strip of the ROI columns x0-x1").

## Tools (the only commands the labeller runs)

The custodian has already extracted the frames and windows into `C:\intercycle_sealed\labelling\`: `grid\<sample_id>.jpg`
(a 1280 px viewing copy with a 100-native-px grid), `frames\<sample_id>.png` (full resolution) and
`windows\<sample_id>.npz`. From the repo root, in the custodian's Python environment:

```
python scripts/label_extract.py --sealed crop <sample_id> <x0> <y0> <x1> <y1> --step 20
python scripts/label_extract.py --sealed kymo <sample_id> <x0> <x1>
python scripts/label_extract.py --sealed kymo <sample_id> <x0> <x1> <y0> <y1>
```

- `crop` writes `labelling\crops\<sample_id>_<x0>_<y0>_<x1>_<y1>.jpg`: an upscaled crop of the frame with a fine grid
  labelled in native pixels (every `--step` px), for exact box edges and foot points.
- `kymo` writes `labelling\kymo\<sample_id>_k<x0>_<x1>_y<y>_<start>.png`: a space-time strip of the window, one column
  per frame (the mean grey of native columns x0..x1), native y on the left and the frame PTS on top. The BDC is the
  lowest image position of the ram band. With `<y0> <y1>` only those rows are shown, at a larger scale. The columns
  must lie inside the window's ROI crop (the command says so if they do not).

## Label file

One JSON file (UTF-8) in `C:\intercycle_sealed\labels\`, for example `labels_test.json` (never `sample.json`), contract
`twin.labels/1`. `sample_list_sha256` is the `items_sha256` of `C:\intercycle_sealed\labels\sample.json`;
`rules_sha256` is the sha256 of this file. Each frame label copies `sample_id`, `clip_id` and `t_pts` from the sample
list; each stroke window copies `sample_id`, `clip_id`, `press`, `t0` and `t1`. Stroke times increase strictly and lie
inside `[t0, t1]`. A synthetic example (values invented, not from any clip):

```json
{
 "contract": "twin.labels/1",
 "split": "test",
 "labeller_id": "agent:test-labeller",
 "labeller_kind": "claude_agent",
 "blind": true,
 "proposals_shown": false,
 "rules_sha256": "<sha256 of labels/TEST_RULES.md>",
 "sample_list_sha256": "<items_sha256 of sample.json>",
 "frames": [
  {"sample_id": "test-f-c9-01", "clip_id": "<clip_id>", "t_pts": 3.0,
   "boxes": [{"cls": "person", "x0": 100.0, "y0": 200.0, "x1": 160.0, "y1": 380.0, "foot_u": 130.0, "foot_v": 378.0,
              "truncated": false, "occluded": false, "src": "labeller"}],
   "note": null},
  {"sample_id": "test-f-c9-02", "clip_id": "<clip_id>", "t_pts": 7.0, "boxes": [], "note": "nobody in view"}
 ],
 "stroke_windows": [
  {"sample_id": "test-sw-P9b-01", "clip_id": "<clip_id>", "press": "P9b", "t0": 0.0, "t1": 10.0,
   "strokes": [1.24, 3.48, 5.72], "note": "space-time strip of the ROI columns x0-x1; edge, not clicked; edge 8.80-10.00 s"}
 ]
}
```

A forklift box has `"cls": "forklift"` and `"foot_u": null, "foot_v": null`.

## Never

- The labeller opens only the files named above for the items of the sample list. It never opens a clip file, the
  repo's code, results or other label sets, and nothing it sees leaves `C:\intercycle_sealed`: the images show
  unmasked people. Only the hash of the label file enters the repo (`sealed/LABELS.sha256`, written by the custodian).
