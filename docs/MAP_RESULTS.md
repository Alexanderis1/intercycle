# Map results: precision, convergence, adaptation (workstream M) — DRAFT, final numbers pending

**Everything here is SYNTHETIC**: a simulated body shop, simulated ceiling cameras. The map is an advisory planning layer,
not a safety function; UNKNOWN or stale space is always occupied, and the robot keeps its certified scanner.

## Status (17 Oct ~19:50, hand-off)

The published configuration (audit 17 Oct, Tier 1 #1 and waiver (a)) is:
- **camera shim at the perception error measured by workstream V** (`mapping.MEASURED_SHIM`):
  - 32 % miss;
  - floor-position error: 85 % Gaussian σ 0.285 m per axis + 15 % gross errors, radial U(5, 20) m (median 0.38 m, p90 10.2 m);
  - false positives: 0.124 person detections per true person detection, half within 0.5–3 m of a true detection, half anywhere;
  - applied to every class (V measured people only, so this is an assumption for machine parts and vehicles);
- **production state discovered by workstream P** (`PhaseModel.line_state`, fitted on training day 0 06:00–09:45 with the
  measured shim). The public calendar is not an input. Under the measured noise the discovered state labels the 09:00 break
  as "andon" (stopped) on 100 % of ticks and never as "break", so the map has no break slices: break crowds land in the
  stopped/andon slices. P reports shift_change recall 0.57 even on the assumed shim.

Splits:
- train: days 0–4 (`out/timeline/map_5d.npz`) and days 0–6 (`out/timeline/map_1w.npz`, the robot batch map);
- conformal calibration: days 5–6;
- test: day 7 of seed 0, plus day 7 of seeds 1 and 2;
- each split has 4 windows: 08:15–10:15, 11:00–12:45, 13:24–15:00, 02:36–04:30.

**The map files and split recordings exist. The scoring (conformal margin, test, traffic skill, convergence replay) has
not been run on them yet**; the numbers below will be filled from `scripts/map_timeline.py report`.

Numbers measured earlier on the ASSUMED shim (0.15 m, 5 % miss, calendar break labels) are superseded; they are kept in
`out/timeline_shim015/` for the record only:
- 1-week map, held-out day 7:
  - strict any-overlap unsafe-free: robot boxes 0.0082, 3.0e-5 with a 0.25 m margin;
  - false-occupied share: 0.29 without the margin, 0.46 with it;
- vehicle Brier equals climatology (no skill); people Brier skill about 0.09–0.11;
- new machine:
  - UNKNOWN after 5.8 s, reported after 36 s;
  - spawned at 40 min, converged at 44 min;
  - transition unsafe-free 0 (2,601 instants);
  - post-convergence unsafe-free 1.6 % strict (0.03 % with 0.25 m); an earlier start-gating version measured 2.4 % / 4.1 %;
  - camera bump recognised in 60 s.

## Tier 3 disclosures (gap to a real factory)
- No pixels → map path exists yet (workstream V). Every map, phase and new-machine result runs on the camera shim. The shim
  samples truth: true class, true radius, a `moving` flag computed before noise, no field of view (the nearest camera sees
  everything), no occlusion, no ID switches. The measured miss / error / false-positive rates are layered on top.
- The same robot program runs every cycle (kinematics are seed-independent): σ 1 % takt, tip-dress every cycle, no model mix.
- Camera calibration (intrinsics, floor homography) is assumed known.
- The change-detection stress terms (clutter, flicker, camera bump) are synthetic assumptions.
