# InterCycle — production-readiness brief (monitor amd-ce → lead amd-4d, Sat 17 Oct ~19:00)
The user asked: "instruct the implementing session about the production-ready flaws, after researching how to optimally solve them".
Research: 3 agents (pixels/calibration, security/privacy/compliance, learning robustness/eval). This is ON TOP of the Tier 1/2 audit plan
(unchanged). Licences checked where marked; verify others before citing.

## A. Before the freeze, ranked by value per hour

### A1. "Pixels → phases" demo: closes the biggest gap (no pixel path for machines). Owner V (perception/*), ~5-6 h, no cloud
`phases.py` already consumes a per-machine feature vector per tick (motion energy + 4x4 occupancy). Compute it from PIXELS:
1. X renders one CCTV clip of a FANUC weld cell after the `cctv_post` pass (noise, blur, distortion), ≥ 20 cycles (~20 min).
   640x360, EEVEE/Workbench, desktop 4060 Ti. This takes PRIORITY over cinematic shots on the desktop GPU. Truth = the trace (`weld_q(phase)`).
2. Image ROI per robot, drawn once (disclose as the commissioning step, or use A1-stretch zone discovery).
3. Per frame:
   - `cv2.findTransformECC` stabilisation (vibration);
   - luma normalised by the frame median;
   - MOG2/KNN or frame difference;
   - morphological opening + min blob area (sparks);
   - skip frames where most of the ROI changes at once (weld flash);
   - add "saturated-pixel fraction" as a weld-on feature;
   - foreground fraction per ROI + a 4x4 grid → the same feature vector.
4. Feed it into the UNCHANGED `PhaseModel.fit_features`.
5. Score period error, sub-phase boundary F1 (±1 s) and online phase accuracy vs the trace. Report SIDE BY SIDE with the shim result.
6. Licence: OpenCV Apache-2.0. Run on the desktop CPU/32 GB (not the laptop). NO CoTracker3 (CC-BY-NC). RAFT/SEA-RAFT (BSD-3)
   only as optional direction features.
- **Stretch (2-3 h, only if A1 is done):** zone discovery without hand ROIs. Per-pixel FFT/ACF periodicity on an 80x45
  motion-energy stack → threshold → connected components clustered by period/phase → candidate machine ROIs; Gemma labels, a human
  confirms ("5-min confirmation", not "zero setup").
- This moves ahead of V's fused tracker. The tracker becomes "if time remains" (confirm the order with the user).

### A2. Conformal margin instead of a hand-picked margin. Owner M, ≈ 0 extra hours (replaces "margin on days 5-6")
- Loss L(λ) = unsafe-free share at inflation λ; it never increases with λ.
- Conformal risk control (Angelopoulos et al., arXiv 2208.02814): λ̂ = inf{λ : (n/(n+1))·R̂_cal(λ) + 1/(n+1) ≤ α}, with calibration units
  = DAYS or episodes (exchangeable), not frames. This gives E[unsafe-free] ≤ α. Learn-then-Test (2110.01052) if a high-probability bound is wanted.
- Split train / calibrate / test (days 0-4 / 5-6 / 7, plus held-out seeds 1-2).
- Headline: test unsafe-free (strict overlap) with a day-level block-bootstrap CI and the α used; false-occupied share as the price.
- Traffic: Brier SKILL vs a named climatology (time-of-day × state), block-bootstrap CI, a reliability diagram, PR-AUC on ever-occupied cells.
  If the simulator's traffic is stationary, "no skill over climatology" is a valid, honest finding.

### A3. Robot statistics, pre-registered. Owner R3, +0.5 h on the Tier 2 item
- Commit a pre-registration paragraph in docs/ROBOT_RESULTS.md BEFORE the batch, with a commit timestamp.
  - Primary endpoint: line_time_lost_s, learned vs no-map.
  - Safety endpoint: any_collision, reported with Clopper-Pearson.
  - Everything else is secondary, with Holm.
- Exact paired sign-flip permutation test (2^10 = 1024 flips; SciPy `permutation_test(permutation_type='samples')`).
  Plot all 10 paired differences. Do not lead with percentile/BCa bootstrap at n=10.
- 0 collisions in 100 min → 95 % upper bound ≈ 2.2/h. State it; no strong safety claim.

### A4. Restrict-only clamp + low-confidence fallback. Owner R3 (owns intercycle_nav), ~1 h
- In observer_bridge: KeepoutFilter may only ADD keep-out; SpeedFilter may only LOWER speed. Missing / stale / bad-signature /
  expired data = occupied. Unit test: a broadcast that tries to free a static keep-out or raise a speed limit is clamped.
- If the phase posterior is uncertain (or the phase is unknown), project the UNION of all L1 bins, not one bin. Verify, add if missing, test.

### A5. Broadcast hardening. Owner C (Sunday slot, before the placement planner) or lead, ~2-2.5 h, pynacl
1. `sign.py`: `load_signing_key(path)` (key generated on the box, 0600). `demo_key` in tests only.
2. New `trust.py`: an offline site root key signs a small observer cert {key_id = sha256(pub)[:4], pub, site_id, nbf, exp, layers}.
   `TrustStore.from_json()` verifies certs against the root, drops expired/revoked ones and returns the keyring. CLI: `keygen`, `cert issue`.
3. `ble.py` Receiver:
   - state keyed by key_id (today it is one global state);
   - accept if boot_epoch is higher (a persisted counter, incremented at start), or the epoch is equal and seq is newer;
   - reject if t_rx − issue > max_age (fast ≈ 2 s) or issue − t_rx > 0.5 s (future-dated / clock skew);
   - if the header has no spare field, "issue_time newer than last" accepts a seq reset.
4. ROS broadcaster loads a persistent key (not one per episode).
5. Tests: restart accepted, replay rejected, future-dated rejected, expired/revoked cert rejected, two observers coexist.
- Zenoh: document mTLS + ACL (put on `intercycle/**` only for observer certs); chrony/NTP to a site server (PTP not needed); robots on
  an OT VLAN (IEC 62443-3-3 zone/conduit). Precedents to cite: IEEE 1609.2 short-lived certs, ASTM F3411 / DRIP paged auth, SROS2.

### A6. Privacy filter before any frame leaves. Owner G, ~1-1.5 h
- Before a Gemma call: black-box (not blur) every person box from the existing detector, padded 15 %; send only machine-zone crops,
  downscaled, metadata stripped. Optional face pass: YuNet (MIT). NOT InsightFace/SCRFD (non-commercial weights).
- `VLM_MODE = off | local | cloud` + an OpenAI-compatible base_url, so going on-prem is a config change.
- Wording: "In this demo, privacy-filtered machine-zone crops (people masked) of SYNTHETIC frames are sent to Gemma 4 on AMD Token Factory.
  Production runs the same model on-prem on AMD hardware (vLLM-ROCm on Instinct/Radeon, or llama.cpp on Ryzen AI Max); no video leaves the site."
  (AMD announced day-0 Gemma 4 support on Instinct/Radeon/Ryzen AI. It's a good AMD story for the judges.)
- Do not state "AMD logs prompts ~2 days" as fact unless the guide says it. Otherwise say "provider retention per AMD terms".

### A7. Line-state naming without hand rules. Owner P, ~1 h (with its disclosure item)
- The map is keyed on cluster ID only; a name never changes the map.
- Each cluster is shown as "State C" with an auto description (e.g. "85 % cells idle, people 3× baseline, typical 12 min").
- An operator names it once and the label persists. On refit, labels carry over by Hungarian matching on centroids. Gemma only suggests (marked unverified).
- Merge break + shift_change into planned_stop, or disclose that the rule encodes the simulator.

### A8. Change-detection calibration. Owner M after A2 (00:30+), ~2.5 h + compute; otherwise disclose
- Set h empirically: replay NO-change data (new seeds + clutter/occlusion bursts) and record the site-wide max CUSUM per hour.
  h = the quantile giving ≤ 1 false alarm per 8 h shift. k = the smallest change worth detecting.
- Inject ≥ 6 (ideally 20) new machines, incl. 0.5-2 m from existing machines (today's `machine_of` absorbs them).
  Report detection probability (Clopper-Pearson), the delay distribution, and false alarms/h with an exact Poisson CI.

### A9. Docs at the freeze. Lead, ~1.5 h
- **Safety statement:** "Advisory planning layer. Not a safety function (not part of any SRP/CS under ISO 13849-1 / IEC 62061);
  does not replace or modify certified safety devices (IEC 61496-3 scanner, ISO 13855 separation distance). It can only make robot
  motion more conservative; missing, stale or unauthenticated data is treated as occupied." ISO 3691-4 / ANSI R15.08 cited as
  guidance only (ANYmal is a legged robot). A plant would require: an ISO 12100 risk assessment with and without the map, an FMEA
  showing a wrong "free" cell is no worse than no map, and an unplug test.
- **Compliance:**
  - GDPR: analytics on CCTV frames is personal-data processing even with anonymous output. Lawful basis Art 6(1)(f) or (c) via
    D.Lgs 81/2008, not consent (EDPB Guidelines 3/2019). A DPIA is required (Art 35).
  - Minimisation: frames in memory only, no re-ID embeddings, short-lived track IDs, aggregated occupancy stored.
  - Italy, Statuto dei Lavoratori Art. 4: an RSU/RSA agreement or INL authorisation BEFORE activation, even for safety purposes;
    individual consent doesn't replace it.
  - EU AI Act: not prohibited (no emotion recognition, no biometric identification). Annex III 4(b) worker monitoring is high-risk,
    so per-worker metrics are excluded by design (record an Art 6(3) assessment).
  - Add a code guard refusing per-track history export (G or D, 0.25 h).
- **Deployment section:**
  - calibration: commissioning click-UI homography, 4-8 floor points vs the DXF, ≈ 10-15 min/camera;
  - lower-setup option: GeoCalib (arXiv 2409.06704, code Apache-2.0, weights CC-BY-4.0) gives focal/roll/pitch/distortion
    from one frame + the mounting height;
  - multi-view only with `facebook/map-anything-apache` or VGGT-1B-Commercial (other variants non-commercial);
  - zones = "5-min confirmation" of discovered ROIs;
  - on-prem Gemma on AMD.
- **Measured vs assumed perception:** state the shim parameters used, our measured RF-DETR numbers, and that machine-part
  detection from pixels is demonstrated only by A1 (per-zone motion), not by a detector.

### A10. Stretch, if V finishes A1 early: GeoCalib accuracy on rendered frames vs Blender truth (focal, pitch, roll, k1). 1.5 h, desktop GPU.

## B. Roadmap (README "Next steps", no code now)
- **Phase model:**
  - model mix via DTW k-means/DBA, K parallel left-to-right chains with a shared start state;
  - profile-HMM delete edges for optional phases (Eddy 1998); per-position dwell self-loops; insert branches for every-N-cycles tip dress;
  - a garbage state → union envelope;
  - ADWIN/Page-Hinkley drift → DBA refit;
  - simulator variability (2-3 programs, σ 5-10 %, retries, desynced paired robots) to test it.
  - Libraries: tslearn BSD-2, stumpy BSD-3, dynamax MIT, ssm MIT. ASOT (arXiv 2404.01518, MIT) as an offline check for irregular cells.
- **L1 from pixels:** conservative visual hull per phase bin (voxel occupied only if inside the silhouette in every camera seeing it;
  seen by < 2 cameras = UNKNOWN; dilate by mask + calibration error; Hänel/Henrich arXiv 2103.11211), SAM2 (Apache-2.0) arm masks.
  This removes floor-homography parallax.
- **Perception model:** a fitted shim: p_detect(class, range, visibility, FOV) logistic + occlusion ray tests + Gaussian/Student-t
  position noise + Poisson false positives + a Markov ID-switch rate, fitted on rendered CCTV with GT (TrackEval MIT for HOTA/IDF1).
- **Flows:** FreMEn per cell (Krajník T-RO 2017; reimplement — repo licence unconfirmed), STeF per heading bin, CLiFF-maps later.
- **Calibration:** pedestrian self-calibration (Lv/Zhao/Nevatia PAMI 2006; Liu/Collins/Liu BMVC 2011).
- **Security:** TPM/HSM keys, trust-store distribution, IEC 62443-4-2 assessment.
- **Other:** independent safety assessment and pilot; local Gemma appliance benchmark; DPIA template + model union-agreement annex.

## C. Scheduling and cuts (confirm cuts with the user)
- New hours on agents: A1 6, A3 +0.5, A4 1, A5 2.5, A6 1.5, A7 1, A8 2.5, A9 1.5 (+A1 stretch 2-3, A10 1.5).
- Desktop GPU order: the A1 clip render first, then X's cinematic shots.
- Laptop: one heavy job at a time. M's rebuild (→22:00) first. A1 processing runs on the desktop. A8 compute after M's 00:30 items.
- If hours don't fit, the cut order is:
  1. A10 / A1-stretch;
  2. V fused tracker;
  3. C placement planner;
  4. T DINOv2 + broadcast field;
  5. A8 reduced to 6 injections.
  A1, A2, A3, A4, A6, A9 stay.
