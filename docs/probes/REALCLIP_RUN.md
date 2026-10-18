# Real-clip run (custodian commands)

The sealed real-clip test (PLAN_TWIN §5, split freeze: USER DECISION Sun ~07:00 CEST). It runs from a DETACHED
WORKTREE at the commit that writes `prereg/FREEZE_REALCLIP`, never from main. Every command below is the custodian's;
the lead reads only the committed summary. `scripts/sealed_run.py` refuses (exit 2, `sealed_run REFUSED: ...`) on any
failed check: then STOP and hand the message to the lead; do not edit anything in the worktree.

## 0. What the lead hands over

- `<RCW>`: the sha of the commit on main that writes `prereg/FREEZE_REALCLIP` (one line: the sha of the frozen code
  commit `<RC>`). `<RC>` must already contain `sealed/MANIFEST.sha256` and `sealed/LABELS.sha256`.
- Target time ~09:15 CEST (latest allowed 10:15 for the freeze, the run right after).

## 1. Environment (PowerShell, on the custodian laptop)

```powershell
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:INTERCYCLE_DATA = 'F:\intercycle\data\esk'     # dev data root: inventory + the watchdog's TRAIN keyframes only
$py = 'F:\intercycle\.venv\Scripts\python.exe'      # the Modal client is installed there; Modal token configured
```

## 2. Detached worktree at the freeze commit

```powershell
git -C F:\intercycle worktree add --detach C:\intercycle_rc <RCW>
cd C:\intercycle_rc
$env:PYTHONPATH = 'C:\intercycle_rc;C:\intercycle_rc\ros2\intercycle_nav;C:\intercycle_rc\ros2\intercycle_eval'
git rev-parse HEAD                                  # must print <RCW>
type prereg\FREEZE_REALCLIP                         # must print <RC>
```

## 3. Checks only (no writes)

```powershell
& $py scripts\sealed_run.py check --stages clips
```

Expected: a JSON block (head, freeze `prereg/FREEZE_REALCLIP`, freeze_sha `<RC>`, MANIFEST and LABELS hashes,
`n_clips` 126) and `sealed_run: every refusal passed (0 RUNLOG records so far)`. It checks C1 clean tree, C2 the
freeze sha is an ancestor, C3 nothing changed since `<RC>` but `prereg/FREEZE_REALCLIP` and `results/sealed/**`
(this covers every file of the clips stage, `scripts/sealed_ingest.py` included), C3b RUNLOG append-only, C4
MANIFEST / LABELS equal their `<RC>` blobs, every clip in `C:\intercycle_sealed\clips` and every label file under
`C:\intercycle_sealed\labels\` hashes to its line (a LABELS path outside `labels/` is refused).

## 4. The clips stage (ephemeral Modal app, no volume)

```powershell
& $py scripts\sealed_run.py run --stages clips
```

It appends a `started` record to `results/sealed/RUNLOG.json`, uploads the 126 MANIFEST-checked clips as bytes to an
ephemeral Modal app (L4, the dev PERC image of `cloud/modal_perc.py` plus `scripts/sealed_ingest.py`), re-hashes them
in the container, runs the dev per-clip ingest (`intercycle.perception.esk_ingest.ingest_clip`, PERC emits BDC), writes
the tables to `C:\intercycle_sealed\perc\attempt_001\` and appends the `done` record with their sha256. Cost: the dev
dry run of 51 dev clips took 157 s wall (well under $1); 126 clips run in 8 calls of 16, about the same wall time.
Expected last line: `sealed_run: attempt 1 done; results/sealed/RUNLOG.json appended (clips)`.

## 5. The score (the pre-registered real-clip analysis)

```powershell
& $py scripts\sealed_run.py score
```

It re-runs C1-C4, takes the last `done` clips attempt of RUNLOG, re-hashes its PERC tables against the RUNLOG
`outputs_sha256`, reads the test labels ONLY from `C:\intercycle_sealed\labels\` as listed in `sealed/LABELS.sha256`
(each file re-hashed), builds the scored test set (`twin.dev_labels.load_scored_test`, dev status rules, reserves by
list order) and runs the dev scorer `intercycle.eval.realclip.run` with split `test` (pixel adapter + watchdog on the
decoded test clips; image-ROI occlusion rule). Writes:

- `C:\intercycle_sealed\scores\attempt_001_realclip_test.json` (full: per-item records carry label content; stays in
  the sealed root);
- `results/sealed/realclip_test_attempt_001.json` (summary: figures and n only).

Expected last line: `sealed_run score: attempt 1 scored -> ...realclip_test_attempt_001.json (full result in the sealed
root)`. It runs on the CPU (decoding), a few minutes. Score each attempt once (a second `score` of the same attempt is
refused).

## 6. Commit the results (only `results/sealed/**`)

```powershell
git switch -c custodian/realclip-results
git add results/sealed/RUNLOG.json results/sealed/realclip_test_attempt_001.json
git status --porcelain                              # must list only those two files
git commit -m "custodian: real-clip sealed run attempt 1 (RUNLOG + score summary)" -- results/sealed
```

Then tell the lead the branch name and the commit sha; the lead merges it into main (it changes only
`results/sealed/**`, which robot check 5 and C3 allow). Do not push other branches, do not commit anything from
`C:\intercycle_sealed`.

## 7. If something fails

- A refusal before `run`: nothing was written; report the message.
- `run` fails after `started`: RUNLOG has an `error` record; a re-run is attempt 2 (fresh `perc\attempt_002\`), only
  after the lead says so. Commit the RUNLOG first if a summary file exists (an untracked `results/sealed/*.json` other
  than RUNLOG makes C1 refuse).
- `score` fails: report the message; `score --attempt N` scores a given done attempt.
- Remove the worktree only after the lead has merged: `git -C F:\intercycle worktree remove C:\intercycle_rc`.

## Dev dry run of this exact path (RC-final, before the freeze)

The same code (clips stage on Modal, then the score core `score_labels_and_run`) on DEV clips and DEV labels, never on
the sealed root:

```powershell
& $py scripts\sealed_run.py dev-dry-run --dev-root F:\intercycle\out\rc_final_dryrun --clips <dev ids> --labels labels\dev
# or, on an existing dry run:  & $py scripts\sealed_run.py dev-score --dev-root F:\intercycle\out\rc_final_dryrun --labels labels\dev
```

### Dev numbers (RC-final, Sun 07:20 WEST; against Claude-agent labels, not human labels)

Run: 51 dev clips (every clip of the dev base sample list plus every clip with a scored dev label) through the clips
stage on Modal (157 s wall, one ephemeral app, cost well under the $1 cap), then `dev-score`. Output in
`F:\intercycle\out\rc_final_dryrun\scores\dev_realclip.json` (outside the repo).

PERC reproducibility: of the 220 tables the container wrote, all 51 `tracks` and 51 `boxes` are byte-identical to the
dev batch in `$INTERCYCLE_DATA`; 102 of 118 `machines` tables are identical, 16 differ only in float noise of
`flow_v` / `roi_energy` (max 0.11) with 16 flipped `stroke` / `missing` samples in total, none in a scored window
(the full-set re-score below is identical to the committed dev result).

1. Sealed path (`dev-score`: dev LABELS -> `load_scored_test(split="dev")` -> `realclip.run`), dev BASE pair only
   (`labels.json` + `sample.json`; the test loader takes one sample list, so the dev supplement is not in it):
   - person (tracked, primary), pooled: 39 frames, 126 GT, 103 pred, 79 TP: recall 0.627 (bootstrap-by-group CI
     0.559-0.712), precision 0.767; c9 R 0.573 P 0.773 (19 frames), c14 R 0.757 P 0.757 (20 frames);
   - foot point: median 8.98 px, p90 42.1 px (n 79); zone accuracy 1.00 (n 79);
   - strokes (BDC): 11 windows, 44 GT, 35 pred, 35 TP: P 1.00, R 0.795, F1 0.886 (P9a 0.947 over 6 windows, P9b 0.829
     over 5); stroke-count error mean signed -0.82, mean abs 0.82;
   - occlusion -> MISSING (image-ROI rule): NOT exercised here: no base window has a hidden span (n 0); base non-hidden
     ticks reported MISSING 74/162 (0.457);
   - status counts: 39 frames scored, 11 windows scored, both presses short of the 10-window target (reported as
     `short_presses`, as the test path does).
2. Same re-ingested PERC, full dev set with the dev loader (`realclip.run(labels/dev)`, base + supplement):
   identical to the committed `results/dev/realclip_dev.json`: person R 0.614 P 0.764 (40 frames, 132 GT), foot px
   median 8.69 / p90 41.6, zone accuracy 1.00 (n 81); strokes 20 windows F1 0.803 (P 1.00, R 0.671), count error -1.3;
   occlusion -> MISSING 31/35 observation ticks (0.886), base non-hidden 145/292, PERC samples 336/428.

Sanity: the person figures of (1) and (2) agree within the one-frame difference; precision 1.00 of strokes and the
negative count error (PERC misses strokes, never invents them) match the dev result; the sealed scoring path ran end to
end on real files (real loader, real scorer, LABELS hash checks, PERC re-hash) with no error.
