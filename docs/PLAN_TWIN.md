> **Read this first (new lead session).** This plan supersedes `docs/DIRECTION.md`'s body-shop scope. It was decided with
> the user on Sat 17 Oct ~20:50-21:05 in the monitor session (amd-ce): Option B = a digital twin of the real Eskişehir press
> shop, the user's 10-day protocol, realistic inspection rounds, freeze Sun 12:30, executed as ultracode workflows. It was
> adversarially reviewed by 4 critics (timeline/budget, honesty, parallelism, data risk) and synthesised. DIRECTION §0 rules
> (secrets, budgets, one heavy local job, commit per item, user approves scope) still hold. `docs/WEAKNESSES.md` is the
> defect register; `docs/AUDIT.md` and `docs/PRODUCTION_BRIEF.md` hold the earlier fixes.
> **Clock:** H+0 is when the new lead starts (planned 21:15). If it starts later, keep FREEZE at 08:30 and the sealed window
> fixed, and absorb the delay in robot development time, not in the sealed test.

# InterCycle: PLAN for a digital twin of a real press shop (Option B), v1

This replaces draft v0. H+0 is Sat 21:15 CEST, FREEZE (code that feeds results) is Sun 08:30 and SUBMISSION is Sun 12:30. DIRECTION §0 still applies, agent slots per U2, and `docs/WEAKNESSES.md` is the register this plan must close or disclose.

## 0. Decisions needed from the USER at G0 (22:45)

**ANSWERED by the user (Sat 17 Oct ~21:10, in the monitor session amd-ce):**

| # | Answer |
|---|---|
| U1 | OK. Option B: up to the $25 Isaac cap, 30 sealed missions. |
| U2 | "Use as many agents as you need." Agent slots are no longer capped at 5. The physical limits still apply: 8 GB laptop RAM (watch free memory), the workflow concurrency cap, and one heavy local job via the lock. |
| U3 | Approved. |
| U4 | Approved. |
| U5 | The user is available all night, as much as needed. |
| U6 | OK. The desktop is the custodian. |
| U7 | OK. The user notes this is the last step, done at the end. |
| U8 | OK. |
| U9 | OK. Option A-lite is in. |

| # | Decision | Recommendation |
|---|---|---|
| U1 | **Isaac budget.** Option A: stay at the $20 stop-and-ask line. That leaves $15.4 of headroom and pays for 24 sealed missions with about $1.4 of reserve. Option B: release up to the $25 cap. That leaves $20.4 of headroom and pays for 30 sealed missions with about $2 of reserve. | **B** |
| U2 | **A 6th agent slot** (DIRECTION §0.9 allows 5). Superseded by U2's answer. The builder work was about 62 agent-h (68 h with A-lite) on a critical path of 12 h. After the review fixes and the monitor's overrides (§8) it is about 75 agent-h before FREEZE (plus 1.5 h GEMMA after) across 12 lanes (S1a-S8), with no lane over 8.25 h. M-W4 change detection (S7a), the M-W7 conformal margin (S7b) and A-lite (S8) keep their own lanes and are not pre-cut. Physical limits still bind: 8 GB laptop RAM (below 1.5 GB free, pause S8 first), the workflow concurrency cap, and one heavy local job via the lock. | Approve 12 lanes; the heavy-job lock still applies |
| U3 | **Approve the scope cuts in §6** (DIRECTION §0.1). | Approve |
| U4 | **Guard rules.** Guard types cannot be seen from 4 m, so every ASSUMED press gets a light curtain plus manual reset, with reset time lognormal (median 4 min, p90 15 min). A sweep over 1-15 min is reported as descriptive only. | Approve |
| U5 | **About 2 h of the user's time.** 23:30: click CAL points (10 min). 23:40-23:50: label 20 dev frames blind, before seeing any proposals. 23:50-00:40: correct the remaining dev labels. 00:45-02:00: label the test split blind (40 person frames + 20 stroke windows), at the desktop or by remote desktop. | Approve |
| U6 | **The desktop session becomes the custodian** of test clips, test labels and test seeds, and stops the obsolete body-shop renders now. The old body-shop heldout1 Isaac launch is cancelled. | Approve |
| U7 | **New claim wording (§1)** and the privacy rules: data/esk/ is gitignored, the plant's name is never used, faces are blurred and a human reviews the cut (tightened after validation: every person box is black-boxed whole, §3 P8). | Approve |
| U8 | **Optional:** 1 h of the user's time for forklift polylines, needed only if probe P5 fails. | Decide at G0 |
| U9 | **Option A-lite** (§7): a small, clearly SIMULATED body-shop scenario, built in its own lane (S8) from 21:15 and run 09:00-10:30 after FREEZE, to cover what the press shop lacks (large swept robot-arm envelopes, per-phase L1 envelopes, fence captures while an arm is parked). Descriptive only, never part of the sealed claim; first item in the cut order (§6). | Approve (user proposed it, Sat ~21:00) |

**Open decisions after validation (Sat ~21:45).** Raised by the validated edit list; answer at G0 (22:45). U11 of the edit list (defer A-lite) is withdrawn by the monitor's override: U9 stands and A-lite runs in lane S8.

| # | Decision | Recommendation |
|---|---|---|
| U10 | **Modal sub-ceilings** (DIRECTION §8): CPU sweeps and double ≤ $13 (incl. A-lite ≈ $0.5), perception ≤ $4, Blender ≤ $1 (renders are on RunPod). Stop-and-ask at $20 unchanged. | Approve |
| U12 | **Zone rule.** Default: day 9 = zone 9, day 10 = zone 14, 160 double runs. Alternatives: zone 9 only (smaller; zone 14 has about 6 min of training footage), or both zones every day (320 runs ≈ $21, over the $20 line). | Default |
| U13 | **Gemma input** = masked dev press crops; amend DIRECTION §6 "synthetic frames only" to match. | Approve |
| U14 | **Label volumes** cut to 40 + 40 person frames and 20 + 20 stroke windows, plus 20 blind dev frames (10 min of the user's time). NI margin δ = 30 s/h. | Approve |
| U15 | **Custodian setup** (changes U6): the real-clip test runs in an ephemeral Modal container launched from the desktop (§5), not in a new desktop environment; the custodian is a NEW desktop session that does no code work (§4). | Approve |

**ANSWERED by the user (Sat ~21:55, monitor session):**

| # | Answer, and what changes in this plan |
|---|---|
| U12 | **Both zones every world-day**: 320 sealed double runs (16 worlds × 2 days × 2 zones × 5 conditions), ≈ $12.8. The user chose this knowing it crosses the $20 Modal line, so that overage is approved. New Modal general plan ≈ $21.5 with a **new stop-and-ask line at $24** (Isaac caps unchanged). §5 and §6 run counts and totals follow 320, not 160. |
| U10 | **Approved**, with the CPU sub-ceiling raised to cover U12: CPU sweeps/double ≤ $19.5 (incl. A-lite ≈ $0.5), perception ≤ $4, Blender ≤ $1. |
| U13 | **Approved**: Gemma sees masked crops of the press area from DEV clips only (people black-boxed whole), never test clips. DIRECTION §6 "synthetic frames only" is amended accordingly. |
| U14 | **Changed by the user: "a Claude agent labels the volumes"**, not the user. The volumes stay (40 + 40 person frames, 20 + 20 stroke windows; NI margin δ = 30 s/h). Rules that keep this honest: (1) DEV labels: a dedicated labeller agent, separate from every lane that tunes PERC/OBS/DYN, labels from frames alone; a second independent labeller agent relabels 20 dev frames blind and agreement (IoU / foot-point px / stroke match) is reported. (2) TEST labels: a fresh labeller agent spawned inside the custodian session, which sees only the test frames, never detector proposals, dev results or code; its labels are hashed into `sealed/LABELS.sha256` before FREEZE. (3) Every real-clip figure is reported as "against independent Claude-agent labels (vision model), not human labels", with the inter-agent agreement as its noise floor. (4) The user may spot-check any frames at any time; spot-checks are logged. U5's labelling time slots are removed; the user still clicks the CAL points (10 min) unless a CAL agent does it from the stripes with the same disclosure. |
| U15 | Not answered separately; the custodian is required for the sealed test and for the test labeller in U14, so the recommended setup applies (NEW desktop session, no code work; real-clip test in an ephemeral Modal container). The user can override. |

**MACHINE ROLES (user decision, Sat ~22:05; supersedes U6/U15 where they say "desktop"):** the LEAD runs on the DESKTOP
(D:\intercycle, 32 GB RAM, RTX 4060 Ti) because the 8 GB laptop is the bottleneck for 12 lanes. The CUSTODIAN runs as a NEW
session on the LAPTOP (sealed clips in C:\intercycle_sealed, outside the repo; test-label agent; seed derivation; it
launches the ephemeral Modal real-clip test with the laptop's Modal credentials). The monitor amd-ce also stays on the laptop
and watches the desktop lead through git pushes, Modal/RunPod state and the lead's gate reports (send every gate hand-off to
amd-ce). Desktop setup by the user: `git pull`, Python env with the repo deps, `modal token new` (interactive), copy `.env` by
hand (never via chat/git), stop the old desktop session's body-shop renders.

## 1. The claim at 12:30

"From about N minutes of training clips (out of about 97 minutes of short public CCTV clips from 2 cameras, Önal & Dandıl 2024, CC BY 4.0; the rest held out for validation and a sealed test; we have no affiliation with the plant), InterCycle hand-traced the visible floor layout using a calibration whose points a Claude agent placed on training keyframes, extracted real walking [and forklift, if P5 passed] paths and [machine stroke timing, if P4 passed | an assumed machine-timing prior, if P4 failed], and built a digital twin of a real metal/plastics plant from them [the press(es) were identified by us from the footage, not stated by the authors]. Layout and motion paths were not scored against real truth. **Its days are synthetic.** Trajectories and stroke timing are resampled from real clips. Frequencies, staffing, long stops and day-to-day variation are stated assumptions, fixed at stated values. A kinematic robot double [plus N descriptive physics-simulated ANYmal missions] inspects the twin using its own simulated sensors plus the observer's signed map. Inference is over 16 held-out generated worlds from our generator (in distribution), not over real days. On held-out real clips, scored once against independent Claude-agent labels (vision model, not human labels), person recall was X [CI, or the per-group rule in §5]. The camera-14 zone rests on about M minutes of training footage."
N and M are filled from inventory.parquet after P3 (estimated 58 and 6). Write "press shop" only if ≥2 presses pass P4.

Honest limits go on the README's first screen and the final video card:
- 2 non-overlapping zones, and only the floor the cameras see is modelled;
- the footage is event-selected;
- the scale carries an interval of ±X %;
- the twin's camera is a statistical model (`stat_v1`), not rendered pixels;
- the primary result comes from the kinematic CPU double, and Isaac is a descriptive physics replicate;
- the real cameras never saw a robot;
- time-of-day layers, forgetting and change detection are evaluated only in simulation (the twin, plus A-lite's simulated body shop);
- the layer is advisory, not safety-rated;
- guarding and reset on every press are assumed (light curtain + manual reset, lognormal reset), not observed;
- inspection targets, defects (25 %) and thermal readings are simulated;
- press timing and forklift paths are [learned / assumed] per probes P4/P5;
- lens distortion is one fitted k1 per framing (GeoCalib cut);
- on the double, robot-mover contacts are geometric overlaps; safety rates are reported per engine;
- the test clips are public; the seal is procedural, not cryptographic;
- the plant's permission, per the authors, covers academic use; this is a non-commercial research demo; CC BY covers copyright only, and all worker imagery we publish is masked;
- if stat_v1 misses G2, the twin observer used the true class and moving flag (M-W2);
- one line per cut item (§6).

## 2. Contracts (schemas plus v0 stubs frozen at G0; v1 swapped in at G2)

| Path | Owner | Content / rule |
|---|---|---|
| `data/esk/inventory.parquet` | S3a | clip_id, camera (9 or 14, taken from the class prefix: 0,1,2,4,5,6 → 9 and 3,7 → 14), framing_id, group_id, class, duration, sha256, split. Loader `esk.load(split=)` raises on `test` unless `INTERCYCLE_SEALED_RUN=1`. |
| `sealed/MANIFEST.sha256`, `sealed/LABELS.sha256` | Custodian | sha256 per test file, the sorted ids, counts per camera and class, and the hash of the label files. |
| `prereg/PREREG_TWIN.md`, `AMENDMENTS.md`, `FREEZE`, `TEST_SEED_COMMIT` | S1a | PREREG v1 before the first dev run. Amendments carry a timestamp and the evidence behind them. FREEZE holds the commit sha. TEST_SEED_COMMIT = sha256(salt‖seeds). |
| `twin/site.yaml` | S2 | Geometry only. Per-framing K/H; 2 zones; press footprints, ram ROI and danger polygon with `dilation_m`; scale interval; `cal_sources` (train clip ids only); a source tag on every field. |
| `twin/tasks.yaml` / `twin/safety.yaml` | S2 / S5a + user | Checkpoints and candidate poses / guard type per press (OBSERVED or ASSUMED), reset distribution, entry rules. |
| `twin/compile.py` | S1a | Turns site.yaml into a `Scenario` (presses on_line=False), rects for `intercycle.sim.geometry`, and a SiteGraph built from tasks.yaml (R-W10). A grep test bans CROSS_X, S80 and data_hall in twin mode. |
| `twin/<zone>/{boxes.json,static.usda,mesh.npz,map.pgm,map.yaml}` | S2 via `intercycle_sim/world.py` | Unseen floor is a wall in both the USD and the map. |
| `$INTERCYCLE_DATA/tracks/<clip>.parquet` (default C:\AMD\intercycle\data\esk\, gitignored, mirrored on the Modal volume; only sha256 goes in the repo) | S3a | split, t, track_id, class, u, v, then x, y (re-projected when H changes), conf, framing. Train and dev only until the sealed run. |
| `$INTERCYCLE_DATA/machines/<press>/<clip>.parquet` (same rule; the public repo ships aggregate statistics plus code to regenerate) | S3a | Vertical flow on the ram ROI, stroke events, MISSING while a person box overlaps the ROI. |
| `labels/dev/` (repo); `C:\intercycle_sealed (LAPTOP)\labels\` (test) | Labeller agents (U14; test labellers inside the custodian context, on the desktop since Sun ~06:25 by user approval, prereg/AMENDMENTS.md) | 40 + 40 person frames (plus forklift boxes wherever P5 finds vehicles), 20 + 20 stroke windows = the whole clip up to 10 s with a running press in the ROI (n confirmed after P1 from the eligible count; total labelled seconds reported), each stroke clicked. |
| `twin/dynamics.json` | S3a | An OBSERVED block and an ASSUMED block. Every parameter has a source (fitted, fitted_censored, prior or assumed), n and a sweep range. |
| Day event script | S3a | Press events, mover intents and resampled segments. RNG keyed by (world_seed, day, entity, event), which fixes W-W1. Executed by `twin/engine.py` with `apply_freeze(press,t0,t1)`, numpy only. |
| `runs/<id>/realised_state.jsonl`, `log.parquet`, `summary.json` | S5a | The plant runs the engine and writes the realised stream at 2 Hz, which is the truth for EVAL. Only twin/camera.py (stat_v1, run in the plant process) reads realised_state.jsonl and writes runs/<id>/obs_v2.jsonl. The broadcaster and observer_live read only obs_v2.jsonl and never open realised_state or plant_state. |
| Observation v2 (`mapping.py`) | S4 | Adds `zone_motion` (M floats, NaN = missing), `zone_vis` and `cam_floor_ok` per camera polygon. Keeps the v1 fields, with a v1→v2 converter and golden fixtures `tests/fixtures/obs_v2_*.npz`. |
| `twin/camera.py` (`stat_v1`) | S3b | Frustum through CAL, visible floor, occlusion by footprints and movers, Markov burst misses fitted to dev track gaps, ID-switch and class-confusion rates from dev, real ROI-energy distributions, and the robot emitted as `other`. Per-detection position noise: foot-point pixel error fitted on dev labels, projected through one H bootstrap draw per world, with σ ≥ MEASURED_SHIM's 0.285 m. MEASURED_SHIM's gross-error and false-positive model (gross_p, fp_ratio) on every class. The moving flag is computed from the noisy positions. tests/test_stat_v1_noise.py asserts that the median position error is ≥ MEASURED_SHIM's. |
| `twin/observer_live.py` | S6 (M-W4 detector from S7a, M-W7 margin from S7b) | R-W1: fast layer = learn_cycles plus nowcast on adapter `zone_motion`. `observer_learned` is deleted from the twin path. |
| `llm/window_edits.jsonl`, `data/esk/blurred/`, `results/truth_access.log` (tracked) | S3b / custodian / all | Gemma output may only shorten a window, behind a clamp. Unblurred frames never leave the private store. Every opening of truth is logged. |

**Truth guards.**
- Truth = realised_state*, plant_state, the day event script, per-world dynamics draws, world seeds and defect state. Only these may open it: twin/engine.py, twin/plant.py, twin/camera.py, twin/oracle_feed.py, the EVAL metrics module (after run only, once log.parquet is closed), and S7b's offline M-W7 calibration on dev worlds only (it writes a frozen margin constant and is never imported by observer_live, the policy or the broadcaster).
- `twin.` is added to the forbidden prefixes in masks.assert_no_truth and test_no_truth_import.py. The test asserts that the import graph of observer_live, the policy and the broadcaster never reaches truth. It also scans ros2/intercycle_eval and ros2/intercycle_nav for the string literals "realised_state" and "plant_state".
- `tests/test_truth_allowlist.py` greps imports and open() calls, and the runtime assert writes to the access log.
- `tests/test_isaac_deps.py` runs an import hook that allows only numpy, astropy, pydantic, nacl, yaml and the standard library.

**File owners.**
- masks.py: S3b (COMMS).
- plant.py: S5a. The camera gate (R-W3) is owned by S2, delivered as a module plant.py imports, and reviewed by S5a.
- mapping.py: S4.
- broadcaster.py, geometry.py, world.py, sitegraph: S1b.
- M-W4 change detector: S7a; M-W7 margin and its calibration: S7b; A-lite modules: S8 (new modules only, none imported by sealed_run.py).
- Other lanes add new modules or send patches through the owner.

## 3. First 90 minutes: probes (21:15-22:45) with fallbacks

Video never touches the laptop. Staging works in two passes:
1. **Pass 1.** A Modal CPU job streams all 691 clips from HF and stores only features: sha256, 1 fps pHash, a 64×36 median background, luma, and one frame at t=0.5 s.
2. **Split.** The split script, its thresholds and its seed are committed before Pass 1 finishes, and the script runs once. The split (clip_id, group_id, split, sha256; no video) is committed as prereg/split.csv, and inventory.parquet is checked against it. Pass 1 then deletes the t=0.5 s frame, the median background and the features of every test clip from the Modal volume, keeping only sha256, camera and class counts. A check asserts that no test clip_id has image data on the volume or in the keyframe pack. Agents never call the HF dataset after Pass 1.
3. **Pass 2.** Only train and dev clips are written to the volume. The laptop gets a keyframe pack (2 frames per clip at 960 px, train and dev only).
4. **Custodian copy.** The custodian downloads every clip to `C:\intercycle_sealed (LAPTOP)\clips\` in the background.

| Probe | When / where | Pass | Fallback |
|---|---|---|---|
| P1 camera and volume | 0:00-0:20, Modal | Cameras taken from the prefix (exact). Clip-seconds per camera × press ROI. | none needed |
| P2 hidden time signal | 0:00-0:15 | Look for ffprobe creation_time or audio in 20 clips, and check zip mtimes. | Declare "no timing information" |
| P3 sessions and split | 0:20-0:45, Modal CPU | pHash (4 fps for clips < 3 s) links pairs with ≥min(3, n_frames) frames at Hamming distance ≤6. Agglomerative clustering on background, luma and object layout gives groups. Split 60/20/20 by group, stratified by camera and safe/unsafe class, with a committed seed. Report the authors' te/tr session overlap. The G0 report includes train/dev/test balance on luma, people per frame, clip length and occlusion share. | Fewer than 5 test groups per camera: the §5 small-group rule applies; report "held-out clips from few sessions" |
| P4 press strokes | 0:45-1:20, Modal CPU, train only | 10 camera-9 clips and 10 camera-14 clips. ECC stabilisation, vertical flow on the ram ROI, check for lamp beat. A press counts as learned if ≥8 train clips show it running with ≥2 strokes each. | Fewer than 2 presses pass: running fraction plus a censored period prior, labelled ASSUMED and shown grey |
| P5 forklift | 0:45-1:15, Modal L4 | RF-DETR truck/car against OWLv2 "forklift" on camera-9 AND camera-14 train frames; report vehicle detections per camera (camera 9 shows a vehicle road). If camera 9 has none, disclose "no vehicles observed in zone-9 train clips". An agent checks 40 frames (reported as an agent check). Suppress any person box whose foot point is inside a forklift box. | Forklift path library from motion blobs, no claim of forklift detection; optionally U8 |
| P6 framing and scale | 0:45-1:30 | SIFT/LightGlue registers each framing to a reference. Priors: UNV IPC2122CR3-PF40-A, 4.0 mm, 1/2.7in, 1920×1080, 4 m mount height. The vendor's 86.5° H / 44.1° V includes barrel distortion, so it is not used as a pinhole: focal prior f ≈ 1330 px (from VFOV), range 1020-1430 px. CAL v0 fits one k1 (division model) per framing from 2-3 straight floor stripes or wall edges and applies cv2.undistortPoints before findHomography. Scale cross-checked by walker height ≈1.72 m, pallets, and stripes at 10 cm. Look for guard hardware. | Estimates disagree by more than 20 %: publish a scale interval and run the robot at both ends |
| P7 overlap | 0:30-0:40 | Paper Fig. 3, plus fewer than 20 LightGlue inliers between views. | Default: 2 separate zones, no corridor |
| P8 faces | 1:00-1:10 | Face-size histogram on 100 frames. | Every person box at conf ≥0.1 is black-boxed whole, padded 15 % and held over the track |
| H1 harness | 0:00-0:45, Modal CPU | Body-shop `fake_episode` with a new cpu= parameter (the decorator defaults to cpu=8, memory=8192), at cpu=4 and 8 and max_rtf 1 and 2: measure wall-min per 720 s mission, boot time and $/run. Before G0, rewrite every double row as runs × measured $/run. | If tuning + dev + sealed > $10.6: (1) cpu=4; (2) tuning drops to 6 configs per arm; (3) no-map and timing oracle on day 10 only; (4) K 16→12; (5) ask the user before G2 |

## 4. Lanes, workflows and gates (12 builder lanes per U2; no lane plans more than 9 agent-h before FREEZE)

| Lane | Work, in order (agent-h; H+0 = 21:15) |
|---|---|
| **S1a INTEG** | Contracts and fixtures (1.5); compile.py incl. per-zone axis-aligned danger rects, plus the grep test extended to plant, broadcaster, timing, masks, mapping and sitegraph in twin mode (2); slice (2, green by 02:45); `scripts/sealed_run.py`, loader guards, `scripts/classify_invalid.py`, twin/ added to the `add_local_dir` mounts in cloud/modal_sim.py and cloud/modal_isaac_robot.py (1.5); PREREG v1 (1, due 05:30). Total 8. |
| **S1b ENGINE/ADAPT** | `geometry.StaticMap(scn, rects=None, extent=None)` and `world.build_world_files(scenario_path=, rects=)` (1.5, from 21:15); engine and realised stream against the G0 schemas (1.5); TwinSiteGraph from tasks.yaml (SiteGraph interface, any approach normal, mission order from tasks.yaml), with `S.load(path)` threaded through broadcaster, timing and masks via run_cfg (2.5); EVAL runner, stats, oracle-share and failure scoring (2.5). Total 8. |
| **S2 CAL/TASKS** | CAL v0 incl. one k1 per framing (1.25; clicks 23:30, frozen 00:15); CAL v1 reduced to 200 bootstrap draws of H from the v0 clicks, the scale interval and `dilation_m` = p95 reprojection error (1); USD and maps through world.py (1); TASKS (3.5: 8 checkpoints, 5 in zone 9 and 3 in zone 14, 20× zoom plus thermal model, defect rate 25 % with a written detectability formula; E-W3); R-W3 camera gate as a module plant.py imports (1.5). Total 8.25. |
| **S3a DATA/PERC/DYN** | Staging and probes (1.5); PERC batch, press motion and per-clip ingest shared by dev and test (fps from container PTS, fresh tracker per clip, explicit class_map) (2); DYN/DAYGEN by resampling (3, by ~03:45). Total 6.5. |
| **S3b COMMS/CAMERA** | COMMS C-W5 (2.5, from 21:15); camera.py `stat_v1` incl. the noise model (3, from ~23:45, fitted once dev tracks land at 00:00). After FREEZE: GEMMA minimum (1.5, 10:30-12:00). Total 5.5 before FREEZE. |
| **S4 OBS** | Labeller HTML (0.5, by 23:30); Observation v2 (1); T-W6 occlusion = missing (2.5); M-W10 watchdog that sets UNKNOWN per camera (1.5); pixel adapter, spatial only (1). Total 6.5. |
| **S5a ROBOT-PLANT** | H1 (1); Isaac smoke on compile(site_v0) (1); twin plant implementing the World interface (active, trip, group, freezes, true_idle_windows) on twin/engine, with no WALK zone, CROSS_X, Kinematics or truth_agents, plus the press-ram mover (3); safety.yaml rules plus lognormal reset (W-W4/R-W2) (2). Total 7. |
| **S5b ROBOT-POLICY** | Reactive baseline policy (3, per E-W4, from 22:45; its parameter space is frozen in PREREG v1); R-W13 as a fixed ISO 13855 field (1); R-W5 2D lidar tilt (1); R-W6 (0.5); start noise plus unmapped clutter (0.5). Total 6. |
| **S6 OBS-2** | observer_live R-W1 (3); learn_cycles constants per machine kind (press: Schmitt w=1, FLIP_S 0.25 s, DEBOUNCE_S and P_MIN from P4 periods); `_prep`, `_rows` and nowcast mask NaN instead of thresholding it; unit test on a synthetic press stream with real P4 intervals and 10 % NaN bursts, requiring conf ≥ MIN_CONF and a period within 5 % (1.5); verifier duty. Total 4.5. |
| **S7a SAFETY-CHANGE** | M-W4 full (5, from 21:15, merged by ~02:30): CUSUM on in-zone hits in cells with p < p_min → UNKNOWN immediately, plus count forgetting decayed per observed tick of coverage. Delivered as a module observer_live imports (S6 reviews). Total 5. |
| **S7b SAFETY-MARGIN** | M-W7 conformal margin (4, from 21:15): calibrated at the real fast-layer latency (truth shifted by the real fast-layer age), Learn-then-Test high-probability bound, calibrated over all time slices, offline on dev worlds only. Its risk level α and confidence δ (not the H2 NI margin) are frozen in PREREG v1. Total 4. |
| **S8 A-LITE** | Option A-lite (§7) on the existing body-shop code, CPU only (≈6, from 21:15): W-W1 check (0.5); minimal W-W2 program variety (2.5); T-W2 phase-origin fix + test, in an A-lite-only module (2); M-W4 scenario using S7a's code once merged (0.5); run plan and mini pre-registration, committed at FREEZE 08:30 (0.5). Runs 09:00-10:30 on Modal CPU next to the sealed batch, never inside sealed_run.py; descriptive only. Total 6. |
| **Custodian** (NEW LAPTOP session, not a slot; moved from the desktop per the user, Sat ~22:05) | Its only jobs are downloads, serving the test labeller, deriving seeds and running sealed_run.py. It does not push code, it takes no P4/P5 overflow or other code work, and only the lead sends it messages. Its transcript goes to the 11:15 auditor. Desktop overflow runs in a different session that cannot read C:\intercycle_sealed (LAPTOP). |

Pre-cut at H+0, each disclosed: CAL v1 refinement beyond bootstrap and interval, stress family and sensitivity sweeps, the offline event-sim merge check. A-lite (S8), M-W4 change detection (S7a) and the M-W7 conformal margin (S7b) are NOT pre-cut: per U2 they run in their own lanes from 21:15. GEMMA moves to after FREEZE. If free RAM on the laptop drops below 1.5 GB, pause S8 first; if it is still below, S6 merges into S4 after S4's watchdog item. Twelve lanes may exceed the workflow concurrency cap: then S8 starts when the first lane finishes, and S7a/S7b keep priority over S8.

Verifiers take a slot only while a builder is waiting on remote compute. Auditors at the gates are fresh agents that see only the commit, prereg/, sealed/*.sha256, the truth-access log and a checklist.

| Gate | Clock | Acceptance (fails closed: no merges past a red gate) |
|---|---|---|
| **G0** | 22:45 | Probes reported; U1-U9 answered and the open decisions U10, U12-U15 answered; split and MANIFEST committed; schemas and v0 stubs (site_v0, dynamics_v0 from priors, a stub day script, MEASURED_SHIM adapter) committed; worktrees and heavy lock in place. |
| CAL v0 | 23:30 clicks; frozen 00:15 | A CAL agent placed 4 floor points per framing on train keyframes (U14 fallback, disclosed; the user did not click); k1 applied; H scaled by the 4 m prior; site_v0 traced from train keyframes. |
| PERC | 22:30 / 00:00 | Label-sample proposals by 22:30. Full train and dev batch at 12 fps over 6 L4 containers (train+dev seconds from P1 × 12 frames, about 56k; at about 42 fps per L4 over 6 containers) by 00:00. Pixel tracks first, then projected through v0. |
| Labels | 02:15 | Dev labels in (incl. the 20 blind dev frames); test labels hashed by the custodian. Not merge-blocking. |
| **G1** | 03:00 | Slice green on stubs (see the next section). Static Isaac scene walking on compile(site_v0). Dev labels in. |
| Movers | 05:00 | Isaac and double driven by replayed real train track segments through the day-script interface. The double's `set_movers` is implemented and tested (movers block paths and count in overlap and TTC). CAL v1, DAYGEN v1 and stat_v1 merged. |
| PREREG v1 | 05:30 | Committed before any dev-world run. Dev worlds 0-3 are generated after it. Tuning runs 05:30-07:00. |
| **G2** | 06:00 | Slice re-run on all v1 artifacts. Parity: 3 matched missions (same script, seed and pose) on Isaac and the double, compared on inspections, trips, holds, path length, overlap counts by class, minimum person and vehicle gaps, and TTC minima, within the pre-registered tolerances, at the same max_rtf as the sealed double. DYN validation report. Generator and dynamics hashes committed. The auditor checks the frozen baseline parameter space. |
| Dry run | 07:30 | sealed_run.py end to end on 1 short dev mission per condition, AND the real-clip path on 20 dev clips through the same code, AND the FREEZE check run against a throwaway FREEZE commit on a scratch branch. |
| **FREEZE** | 08:30 | All code that feeds a result is frozen and prereg/FREEZE is written. The custodian derives 16 seeds as seed_i = sha256(R ‖ i), where R is the randomness of the first drand round published after the FREEZE commit time. The round number goes in RUNLOG.json. If drand is unreachable, it uses `secrets.token_hex(16)` and says so in the README. It then commits TEST_SEED_COMMIT. A-lite's mini pre-registration is committed with FREEZE. |
| Sealed | 08:45-10:20 | 08:45: Isaac batch in waves of 10 (about 65 min). 09:00: CPU-double batch. 09:15: real-clip test (ephemeral Modal, launched from the desktop). 09:55: last re-run under the §5 rule. In parallel, outside sealed_run.py: A-lite (S8) on Modal CPU 09:00-10:30, descriptive. |
| Close | 10:30-12:30 | 10:30 stats; 10:30-12:00 GEMMA minimum (descriptive); 11:15 independent audit (each WEAKNESSES id marked closed with its test, or disclosed with where; PREREG v1 vs final diffed); 12:00 docs; 12:30 SUBMISSION. |

**Thin vertical slice.** It lives in `scripts/slice.py` with the test `tests/test_slice_e2e.py`.

Pipeline:
1. site_v0 goes through compile and world.py to boxes.json, usda, mesh.npz and map.pgm/.yaml.
2. A stub day runs for 30 minutes: 2 presses, Poisson walkers, 1 forklift.
3. The engine produces the realised stream.
4. The adapter turns it into Observation v2 at 2 Hz.
5. The observer learns for 20 minutes and produces signed slow, fast and map layers.
6. The fake backend plus Nav2 runs on Modal CPU: 1 mission of 10 minutes in the `learned` condition.
7. The plant writes the run log.
8. The primary-metric JSON is computed.

Assertions:
- the truth allowlist and Isaac-dependency tests pass;
- signatures verify with an explicit keyring, and the check fails closed;
- at least 1 inspection attempt happens;
- the metric JSON validates against its schema;
- a press stop caused by the robot appears in the Observation within 1 tick plus latency.

No offline event-sim variant (intercycle.sim is body-shop specific). The Modal slice runs at G1, G2 and FREEZE, and the full test suite runs only at those gates.

## 5. Method details

**Sealed-test enforcement.**
- Test clips are stored only on the laptop (C:\intercycle_sealed), outside the repo and away from the desktop lead. Test perception runs only inside `sealed_run.py`, which uploads them to an ephemeral Modal container (no volume; upload hashes checked against MANIFEST) running the same per-clip ingest code as dev. The test clips are public, so the seal is procedural (sealed labels, hashes, an access log), not cryptographic.
- `sealed_run.py` refuses to start unless: the tree is clean; FREEZE_sha is an ancestor of HEAD; `git diff --name-only $(cat prereg/FREEZE) HEAD` lists nothing but prereg/FREEZE and prereg/TEST_SEED_COMMIT; and every input hash matches (MANIFEST, sealed/LABELS.sha256, generator, dynamics, TEST_SEED_COMMIT).
- It writes `results/sealed/RUNLOG.json` (tracked) with a timestamp, HEAD, the input hashes, the drand round, and the revealed salt and seeds.
- The generator asserts that dev seeds come from 0-999 and rejects test seeds unless the sealed flag is set.
- CAL uses only train frames. compile.py asserts that every source id in site.yaml and dynamics.json is a train id.
- A test framing with no matching train framing gets no fitted H. Its clips are scored for P/R and strokes only and are excluded from foot-point error and zone accuracy, with n reported.

**Split and labels.**
- Frame sampling uses a seeded RNG over (group, clip), 1 frame every 2 s, stratified by camera. The sample list is hashed before labelling starts.
- Dev labels use detector proposals that the user accepts or corrects. Agents may pre-fill boxes but never accept them.
- A fresh agent relabels 12 dev frames blind, and the agreement is reported.
- Before seeing any proposals, a second independent labeller agent labels 20 dev frames blind (U14). Blind-vs-anchored recall on those 20 is reported.
- Every sampled frame is labelled, including frames with no people. Stroke windows start at seeded-random times per press clip, never where the motion detector fires, and are hashed with the frame list.
- A one-page labelling rule (occlusion, truncation, minimum box size) is committed before dev labelling.
- Test labels are made blind by the user, with no proposals shown, and only their hash enters the repo.

**DYN/DAYGEN (resampling is the default).**
- OBSERVED: path-shape library, speeds, group size, forklift paths, stroke impulse shape, in-clip intervals (Kaplan-Meier, censored at the end of the clip), P(press running) from a Beta posterior per press, layout states taken from the session clusters, day/night share.
- ASSUMED: arrival rates (corrected by recall), staffing ±20 %, stops using the W-W3 lognormal, violation share, social-force parameters from published defaults.
- A day draws its layout state, its lighting and its assumed rates, then resamples the real segments.
- Validation pushes twin output through stat_v1 into tracks, applies the dataset's selection rule, and compares with real dev tracks using speed and stroke-period KS only. It passes if each statistic is within 1.5× the train-vs-dev noise floor and no worse than a bootstrap of train tracks.
- Stress family (staffing +50 %, press period −15 %, stops of 5-30 min): pre-cut at H+0 and disclosed (§4, §6).

**EVAL (PREREG v1 by 05:30, before any dev-world run).**

PREREG v1 fixes the following, and later amendments may change implementation details only, never these: the primary formula, the NI margin, the gate order and each hypothesis test, the recall threshold r0, the unsafe-free threshold, the press inspection-window definition, the exclusion and re-run rules, and S7b's conformal risk level α and confidence δ. Each amendment names the dev run that prompted it. No amendments after FREEZE. The 11:15 auditor diffs PREREG v1 against the final version and lists every amendment in the README.

Design:
- K = 16 test worlds plus 4 dev worlds.
- In every world (dev and test) the observer runs days 1-8 on the CPU engine (numpy, no Nav2). In test worlds the robot is off on days 1-8. Robot development and tuning use days 2-8 of the dev worlds only. Days 9-10 are scored. The slice asserts that mean visit counts per slice at day 9 match between dev and test worlds within 10 %.
- Conditions on the CPU double: no-map, static, learned, best reactive, timing oracle.
- A double run is one mission in one zone. Per U12 (user, Sat ~21:55) BOTH zones run on every test world-day, the same for every condition: 5 × 16 × 2 × 2 = 320 runs.
- All times come from container PTS, never frame_index/24.
- A press-shop inspection window is a press stop, or a gap between strokes of at least inside_s + margin. The share of valid inspections that fall inside ASSUMED stops is reported.

Primary endpoint:
- Per mission, Y = checkpoints whose capture passes Inspect, set to 0 if the mission had any robot-caused press stop, hold (press restart delayed by robot presence) or incident. Every mission has the fixed denominator T = 720 s of sim time, and the robot idles after its round. Declined and unattempted checkpoints score 0.
- Per-world value = ΣY / ΣT (per hour), pooled over both test days. The effect is learned minus best reactive, paired by common random numbers and averaged within each world, tested with an exact one-sided sign-flip over the 16 worlds at α 0.05.
- A per-checkpoint outcome table (declined / passed / held / tripped; E-W12) is published.

Gatekeeping (fixed sequence, each one-sided at α 0.05):
1. The primary.
2. Non-inferiority: robot-caused press time lost (stops + holds, incl. reset time) per mission-hour, learned minus static ≤ δ = 30 s/h. Sign-flip on (d_w − δ) over the 16 worlds. δ is fixed here and cannot be amended.
3. Map unsafe-free share on test days, per world. H0: mean miss share ≥ α_target + 0.01 (α_target = S7b's conformal α). Sign-flip over the 16 worlds. Robot safety events are descriptive only.
4. Blind test person recall ≥ r0, where r0 is set from the 20 blind dev frames only. Decision rule as in Real-clip test.

Reported outside the gate:
- Learned minus static on the primary, with a world-clustered 95 % CI, in the same table row as the primary. If its point estimate is ≤ 0, the README's first screen says that static boundary inspection matched or beat the learned policy.
- Safety outcomes as Poisson rates with exact CIs.
- Share of the oracle gain = Σ_w(L − R) / Σ_w(O − R) (a ratio of sums), reported only if Σ_w(O − R) > 0 and its world-bootstrap lower bound is > 0. Otherwise the two paired differences are reported with CIs.
- The scale +10 % variant and the reset-time sweep, as descriptive.

Rules:
- Infrastructure-invalid = the container failed, was pre-empted or ran out of memory before the robot's first command; or logs are missing; or Modal/RunPod returned an error code. `scripts/classify_invalid.py` labels episodes from the logs alone and commits the list before any metric is computed.
- Every other error is a failure: Y = 0 with full T, and the remaining T counts as time lost on H2. The EVAL runner scores errored episodes as failures and never drops the pair. This overrides episode.py eval_check and stats.py.
- A re-run repeats ALL conditions of the affected (world, day, slot). First-attempt results and the re-run count per condition are reported.

Baseline fairness:
- Reactive uses the same window planner and entry logic as learned. Only its timing and idle estimates come from its own camera, persisted within the world. The claim states that it has no camera history before the mission, by design.
- It has the same sensors, zoom head, rules and checkpoints.
- Its parameter space is frozen in PREREG v1 and checked by the G2 auditor.
- Equal tuning budget per arm: 8 configs × 4 dev worlds (one fixed dev day each) = 32 runs, run as 2 parallel rounds of 4 configs (about 15 min each). The config is chosen by its mean over world-days. Each arm also gets at most 20 dev runs, and InterCycle's debug runs count there.
- A separate agent tunes the baseline and never sees InterCycle's dev results. InterCycle lanes do not read results/tuning/baseline before FREEZE. Both tuning curves and the chosen configs' per-world-day scores are published in `results/tuning/`.

Real-clip test:
- Person P/R at IoU 0.5, foot-point pixel error, stroke F1 with ±0.25 s tolerance (greedy one-to-one matching), stroke-count error, occlusion→MISSING rate, zone accuracy. n is reported for every figure.
- Bootstrap-by-group CIs only if there are ≥ 5 test groups per camera. Otherwise: per-group recall with n, plus a frame-level Clopper-Pearson interval labelled "ignores clustering (only k groups)". Gate 4 then passes only if the lowest per-group recall and the pooled estimate are both ≥ r0.

**Isaac (descriptive physics replicate, no p-values).**
- 3 conditions: learned, best reactive, static.
- Test worlds 1-2, days 9 and 10, 1 slot per zone per day: 24 missions. Option B adds world 3, day 9, both zones (+6 missions = 30); unbalanced, descriptive only.
- A mission is one round of the zone's checkpoints, capped at 720 s.

## 6. Compute, budget, cuts

**Where things run.**
- Laptop: orchestration only.
- Desktop: the LEAD session (orchestration + local GPU/CPU work: P4/P5, perception overflow, face blur, tests, map builds). Laptop: the CUSTODIAN session only (sealed clips/labels/seeds, sealed_run.py launches the ephemeral Modal real-clip test) and the monitor amd-ce. The lead never reads C:\intercycle_sealed and never messages the custodian except to request the scripted sealed steps.
- Modal: staging, perception, the double, map builds, the sealed real-clip test (ephemeral container), A-lite (CPU).
- RunPod: renders after the freeze only.

**Isaac.** Measured rate $1.37 per container-hour, so a 12-min mission ≈ 21 container-min ≈ $0.48.

| Item | A ($15.4) | B ($20.4) |
|---|---|---|
| Smoke runs at G1 (03:00) and Movers (05:00) | 0.55 | 0.55 |
| Dev and parity (3 / 6 missions; B adds a second parity seed) | 1.45 | 2.90 |
| Dry run (3 short missions) | 0.50 | 0.50 |
| Sealed (24 / 30 missions) | 11.50 | 14.40 |
| Reserve | 1.40 | 2.05 |

**Modal general** ($18.5 of headroom to the ask line):

| Item | Cost |
|---|---|
| Staging and keyframes | $0.6 |
| Probes | $0.3 |
| Perception (train and dev) | $1.5 |
| Harness and slices | $0.8 |
| Tuning (2 arms × 8 configs × 4 dev worlds = 64 runs) | $2.6 |
| Dev runs (≤20 per arm) | $1.6 |
| Sealed double (320 runs: 5 conditions × 16 worlds × 2 days × 2 zones, per U12) | ≈$12.8 |
| Map builds and day generation | $0.8 |
| A-lite (S8, Modal CPU, 09:00-10:30) | ≈$0.5 |
| **Total** | **≈$21.5** (U12 answer: stop-and-ask raised to $24, reserve ≈$2.5). Every double row assumes $0.04/run until H1 measures the real cost; if H1 measures more than $0.05/run, K drops 16 → 12 before anything else. Sub-ceilings per the U10 answer. |

Double runs are logged against the general cap. RunPod: at most $2. Gemma on Token Factory is the AMD workload.

**Cut now, each disclosed against its WEAKNESSES entry:**
- R-W4 slam_toolbox (start noise and clutter are kept);
- 3D lidar;
- R-W13 reduced to a fixed field;
- M-W10 homography re-registration;
- C-W1 Zenoh/BLE;
- C-W6 live layer;
- C-W7;
- C-W2 and C-W4 become stretch goals;
- GeoCalib;
- cross-camera fusion and the corridor;
- social-force fitting;
- fully generative days;
- real-clip map learning beyond spatial occupancy;
- Gemma change triage. GEMMA minimum (kept, after FREEZE): dev-split press-ROI crops only, with persons black-boxed by intercycle/llm/privacy.py before the call. Task: running / stopped / person-at-press, scored against the user's dev stroke labels. Record mode now, replay mode for anything shown; descriptive and outside the sealed numbers. Guard (0.25 h): client.chat raises if mode != replay and any image lacks the privacy-filter marker;
- T-W2 superseded in the press shop by the topology (guard test: own_phase equals the map phase within 0.5 bin); A-lite's T-W2 fix lives in an A-lite-only module (S8);
- M-W8 and C-W10 disclosed: press-shop slices are time-of-day only, and the robot does not select production-state slices;
- E-W5 timing-only (no map masks) ablation;
- pre-cuts at H+0: CAL v1 refinement, stress family and sensitivity sweeps, the offline event-sim merge check. (A-lite, M-W4 change detection and the M-W7 conformal margin are not pre-cut; they run in lanes S8, S7a and S7b.)

**Cut order if late** (cuts before FREEZE start at item 1; renders only matter after 12:30):
1. Option A-lite (S8, §7). Independently of lateness: if laptop free RAM < 1.5 GB, pause S8 first.
2. Isaac sealed reduced to 12 missions, then to none, labelled as such.
3. Robot rounds in the camera-14 zone (primary becomes zone 9 only, disclosed).
4. Dashboard panel reduced to stills.
5. Renders (after 12:30).

Fallback, not a pre-cut: if S7a's M-W4 or S7b's M-W7 is not merged by G2 (06:00), M-W4 falls back to forgetting only and M-W7 to a fixed latency-p99 margin, both disclosed. S8's M-W4 scenario then reports forgetting only.

**Never cut:** sealed enforcement, C-W5, T-W6, the M-W10 watchdog, the real-clip test, honest labels, the Gemma minimum.

**Video (12:30-15:30).** Only frozen logs are used.
- Every twin frame carries the tag "GENERATED DAY · world W day D · not recorded".
- Every real frame carries "REAL · clip id · people masked". Real footage totals ≤ 20 s, and a human checks every frame of it. Otherwise, real clips are shown as tracks over the median background.
- An export guard refuses twin output without `generated:true`.
- Paper Fig. 3 is redrawn, not reproduced.
- A human does a frame-strip review of the twin footage at 13:00.
- DIRECTION §10 is superseded. Shot list: real clips (masked) → traced layout and twin → generated day with observer map → robot round on the double/Isaac → sealed table → honesty card (synthetic days, advisory, what ran on AMD vs NVIDIA). No "no setup", no week-long curve, no channel-drop scene.
- Three tags: REAL / GENERATED DAY / SIMULATED body shop. A frame with any generated content never carries REAL, and the export guard checks the tag field.
- Credit block (README and video): Önal, O. & Dandıl, E. (2024), Data in Brief 56, 110791, doi:10.1016/j.dib.2024.110791 (verify on doi.org before writing; the HF card's 110756 is wrong); clips from Mendeley Data doi:10.17632/xjmtb22pff.1 via HF Voxel51/Safe_and_Unsafe_Behaviours, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: trimmed, re-encoded, people masked; tracks, labels and figures derived by InterCycle; no affiliation or endorsement.

## 7. Option A-lite: large moving machinery (simulated, descriptive)

**Why.** The press shop has rams, flywheels, forklifts and people, but nothing that sweeps a large envelope through the air
like a 2.6 m spot-weld arm. The per-phase L1 envelopes, the in-zone change detector (M-W4) and "capture from the fence while
the arm is parked" are best shown on large arms. A-lite reuses the existing body-shop scenario for exactly that, labelled
SIMULATED everywhere, and is never part of the sealed claim.

**Scope (≈ 6 agent-h; lane S8 A-LITE from 21:15, on the existing body-shop code, CPU only; first item in the cut order).**
1. W-W1 is already fixed by the shared day-script RNG; check the body-shop World uses it too (0.5 h).
2. Minimal W-W2: 2 robot programs per station from a seeded model-mix sequence, tip-dress every N ∈ [8, 30] cycles,
   ±3 cm path noise (2.5 h).
3. T-W2 phase origin: one origin for map bins and the robot's own_phase, test within 0.5 bin (2 h). It is superseded in
   the press shop but required here.
4. M-W4 in-zone change detection exercised: a program change and a new arm placed 0.5-2 m from an existing cell; report
   the time until the observer's map marks those cells UNKNOWN (uses S7a's M-W4 code once merged); robots receive it at the next mission start, because C-W7 is cut (0.5 h).
5. Run on the CPU double: 2 generated days (learn, then 1 day of fence captures + timed entries into open zones) per
   condition learned / best reactive / static; descriptive table, no p-values (0.5 h + Modal CPU ≈ $0.5).

**Timing.** Built from 21:15 in lane S8. It uses twin/observer_live.py on a noisy camera (never observer_learned, which has a
truth hook, R-W1) and changes no file that `sealed_run.py` imports; T-W2 goes into an A-lite-only module. Its mini
pre-registration is committed with FREEZE (08:30). It runs 09:00-10:30 on Modal CPU (≈$0.5) next to the sealed batch,
never on Isaac and never inside `sealed_run.py`. It is reported as SIMULATED and descriptive only. If laptop free RAM
drops below 1.5 GB, S8 is paused first.

**Claim wording.** "In a fully simulated body shop (no real data; perception is a truth-derived noise model without occlusion), the observer code (branch <sha>) learned time-binned arm envelopes covering X % (IoU Y); on N injected new arms it marked the cells UNKNOWN after a median of Z min, with F false alarms over H simulated hours."

## 8. Validation log (Sat 17 Oct ~21:45)

- **Review.** 81 findings from 6 lenses (arithmetic, leakage, code feasibility, statistics, facts, claims/privacy). Skeptics confirmed 71 as real (2 blockers, 23 major, 46 minor) and rejected 10. The confirmed findings were merged into one edit list and applied to this file.
- **Monitor overrides (U2: "use as many agents as you need").**
  1. M-W4 is not pre-cut or reduced to forgetting only. New lane **S7a SAFETY-CHANGE**: CUSUM on in-zone hits in cells with p < p_min → UNKNOWN immediately, plus count forgetting decayed per observed tick (5 agent-h, from 21:15, merged by ~02:30). S4 loses its "forgetting only" item.
  2. M-W7 is not reduced to a fixed margin. New lane **S7b SAFETY-MARGIN**: conformal margin calibrated at the real fast-layer latency (truth shifted by the real fast-layer age), Learn-then-Test high-probability bound, calibrated over all time slices (4 agent-h, from 21:15); its α and δ are frozen in PREREG v1. S6 loses "M-W7 as a fixed latency-p99 margin".
  3. A-lite is not deferred to after submission. New lane **S8 A-LITE** (≈ 6 agent-h, from 21:15, body-shop code, CPU only): minimal W-W2, T-W2 phase-origin fix + test, S7a's M-W4 once merged, mini pre-registration at FREEZE 08:30, runs 09:00-10:30 on Modal CPU (≈$0.5) next to the sealed batch, never inside sealed_run.py, descriptive only, honest claim wording (honest-08). First in the cut order; paused first if laptop free RAM < 1.5 GB. Edit-list row U11 (defer A-lite) is withdrawn; U9 stands.
- **Edits applied:** E1a, E1b (+ overrides), E1c, E1d, E1e, E2a, E2b (rewritten per overrides), E2c, E3, E4a, E4b, E5, E6, E7a, E7b, E7c, E7d, E7e (A-lite budgeted: total ≈$15.1, reserve $3.4), E7f, E7g, E8a/E8b/E8c (rewritten per override 3), E8d, E9, E10, E11 (moved to "Open decisions after validation" in §0, U11 dropped, U15 added from E6), E12a, E12b, E12c, E13, E14a, E14b, E14c, E15a, E15b, E15c, E16a, E16b, E17a, E17b, E17c, E17d, E17e, E18, E19a, E19b (both per overrides), E20a, E20b.
