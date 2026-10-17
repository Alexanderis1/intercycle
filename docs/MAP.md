# The observer's layered map of dynamics (v4)

The observer does not only say "busy / idle". Over days it builds layers on top of the static map (L0, from CAD / SLAM).
Each layer holds **dynamic collision volumes** that depend on **time of day** and on the **production phase** the observer
detects itself. A visiting robot downloads the layers once per version over BLE (signed), receives the machines' current
phase and the production state in the fast layer, and indexes the layers with them.

| Layer | Content | Grid | Indexed by |
|---|---|---|---|
| L1 machine envelopes | floor swept by moving parts and skids, per machine | local 0.25 m over zone + 1 m margin | machine, **phase bin** (16 + 1 "stopped") |
| L2 vehicle flows | occupancy probability + dominant heading (8 dirs): AGVs, tuggers, forklifts, docked AGVs | site 1 m (120 x 48) | **time of day** (48 x 30 min), **production state** (4) |
| L3 people flows | same, for people (aisle walkers, break crowds, data-hall techs) | site 1 m | same |
| UNKNOWN | cells never observed, under an offline / bumped camera, a new machine still being learned, a machine with no envelope yet, or a whole slice nothing comparable was seen for. **Treated as occupied** | site 1 m | live state + map version |

Production state = `running | andon | break | shift_change`. Everything below is SYNTHETIC (simulated floor, simulated cameras).

Files: `intercycle/observer/mapping.py` (observations, camera shim, conditioning, learned map),
`intercycle/codec/maplayer.py` (compression, signed manifest, chunks, receiver),
`scripts/build_map.py` (floor simulation, build, render), `tests/test_mapping.py`.

## 1. What the observer sees (no ground truth)

`Observation` per 2 Hz tick: body footprints as circles `(x, z, r)` with a coarse class (`person`, `vehicle`, `part` =
machine part with the id of the zone it falls inside, `skid`), a heading for moving bodies, a `moving` flag (frame
differencing) and a per-camera heartbeat `cam_ok`. `CameraShim.observe(sim_state)` produces it from a simulated floor:

| Imperfection | Value |
|---|---|
| floor-projection noise | sigma 0.15 m on every circle centre (one error per body circle per frame, shared by the sub-frames) |
| dropouts | 5 % of detections |
| occlusion | random blackout of one camera region (Voronoi cell of the 10 ceiling cameras), 10 s long, 4 per hour per camera (~1.1 % downtime); the missing frames are visible in `cam_ok`, so the observer does not count them as "empty floor" |
| latency | 0.2 s; every frame carries its capture time, the observer conditions on that |
| motion blur | machine parts are integrated over 3 sub-frames in the last 0.5 s, so a fast arm leaves a continuous swept footprint |

The shim is the only place that is fed ground truth (`SimState` built by the caller). `mapping.py` and `maplayer.py` never
import `intercycle.truth` or `intercycle.sim` (tested).

## 2. Conditioning variables the observer computes itself

`ConditionTracker.step(obs) -> Cond` (time-of-day bin, production state, per-machine phase bin):

* **Time of day**: `floor((t mod 24 h) / 30 min)`.
* **Motion signals**: machine m = "a part inside zone m is moving"; LINE = "a skid is moving" (the line is indexing).
  Debounced online (rise after 2 still-to-moving ticks, 3 for the line, fall after 4 moving-to-still ticks); blackouts pause
  the debouncing and a rise that surfaces after a blackout is placed where the learned period predicts it.
* **Phase** = (t - last start) / learned period. The start is the **line index** for the eight line stations (all stations are
  indexed by the same line phase; dwell pauses inside a station cycle cannot fool it - a framing station's weld dwell is longer
  than its idle) and the machine's own rise for the two off-line cells. 16 bins of the cycle; once the phase exceeds
  1 + max(6 %, 3 s / period) the machine is **overdue = bin 16 ("stopped")**. Period, jitter, busy fraction come from the
  rise-to-rise intervals (stops excluded). `observer.learn_cycles.learn` is run on the buffered signals every 15 min and seeds
  rows the online estimate cannot resolve; its agreement is kept in `tracker.learner_agreement`.
* **Production state**: the public calendar (`scn.line_stops()`) gives `break` and `shift_change`; otherwise the line index
  being overdue (stopped longer than its learned cycle allows) gives `andon`; else `running`.

## 3. The learned map

`LayeredMap` holds counts, so it can be merged across windows or processes (`merge`) and saved as `.npz` (`save` / `load`).

* **L1**: `hits[machine][17, nx, nz]` (cells counted once per tick) and `visits[machine, 17]`; `p = hits / visits`.
  `envelope(machine_id, phase_bin, p_min=0.05)` -> boolean local grid (a bin with fewer than 8 observed ticks falls back to the
  union of the well-observed bins, conservative). `box_cover(...)` -> axis-aligned boxes by greedy rectangle cover: every box
  lies inside the envelope (no false area), each step takes the box covering most still-uncovered cells, stopping at
  `max_boxes` or at 98.5 % coverage. Detections are rasterised with their radius shrunk by 0.1 m (`shrink_m`), a bias correction for
  the camera noise: without it the p >= 5 % threshold dilates every envelope by about one cell.
* **L2 / L3**: `hits[tod * 4 + prod]` per class on the 1 m grid (a 1 m cell is hit if a body circle intersects it), a heading
  histogram (8 compass bins, 0 = +z clockwise towards +x), and per-camera visit counts (blackouts are not visits).
  `traffic(cls, tod_bin, prod_state)` -> `(probability [nx, nz], dominant heading [nx, nz] int8, -1 = unknown)`. A slice with
  fewer than 300 observed ticks is not trusted: the read falls back to the nearest observed time of day of the same production
  state, else the same time of day while running.
* **`predict_obstacles(t_now, horizon_s, machine_phase_now, prod_state)`** -> list of `PredStep(t, boxes, box_machine, circles,
  circle_p, circle_cls)`, one per 0.5 s. `machine_phase_now = {machine_id: phase fraction}` (what the fast layer carries; `None`
  / negative = stopped or unknown). Each machine's phase is advanced with its learned period and widened by
  `2 * sqrt(0.03^2 + jitter^2 * cycles)` (union of the neighbouring bins). In a break / shift change the STOP envelope holds for
  `prod_until_s`; in andon only 3 s (the machine may restart at any moment); stopped / unknown machines return the union of all
  bins. `circles` are the traffic cells with p >= 0.3 at the predicted time of day (soft costs come from `traffic()`).
  `DecodedMap` (robot side) has the same API.

## 4. BLE map layer (layer id 3)

Payload = `zlib( meta | L1 | L2/L3 )`, cut into 240 B chunks (one 251 B extended-advertising frame each).

* **L1**: per machine, per bin, the box cover (count 5 bits, then 4 coordinates of 6 bits in local 0.25 m cells).
* **L2 / L3**: per class and production state, the 48 time-of-day slices in order. A slice is `0` (not observed), `1` (same as
  before) or `2` + a sparse delta against the previous observed slice: for every changed cell a LEB128 gap and one byte =
  4-bit log-quantised probability (0 = below 2 %, 1..15 = 2 % * 50^((c-1)/14), step 32 %) | 3-bit heading | heading-valid. Only cells
  above 2 % are ever sent. The encoder keeps a cell unchanged unless it moved by more than max(1 step, 2.5 Poisson sigmas of its
  hit count) and switches a heading only if the new direction is 1.5x the old one in the histogram, so the ~90 observed slices per
  class cost only their real changes.
* **Manifest** (signed, Ed25519, `codec/sign.py`): the common 12 B header with layer 3, map version, `valid_s` (6 h), raw and
  compressed length, chunk count, parity group, SHA-256 of the payload, and a 4-byte hash of every chunk. It travels as ordinary
  `ble.frame()` fragments. Chunk frames are unsigned but are accepted only if their hash is in the signed table, so a forged or
  corrupted chunk can never enter the map; the assembled payload is checked against the signed SHA-256 again.
* **Carousel** (periodic advertising): `[manifest fragments, 16 chunks]` repeated; after every 8 data chunks an XOR parity chunk repairs
  one loss per group. `is_map_frame()` routes layer-3 frames away from the slow / fast / live `ble.Receiver` (which would count chunk frames as bad
  frames and reject a reassembled manifest as `unknown layer 3`; route before it).
* **`MapReceiver`**: any order, any loss, junk and forgeries; chunks heard before the manifest are held and verified afterwards.
  Rejection reasons are counted (`bad_signature`, `unknown_key`, `wrong_site`, `expired`, `stale_version`, `bad_chunk`,
  `bad_payload`). A cached version is never downloaded again (`MapCache`, keyed by `(site_id, version)`), and a partial
  download is kept per manifest so it resumes when the robot comes back into range.

Robot fast-layer addition proposed (not implemented here, `layout.py` is not in this change): the fast record's 16 reserved bits can
carry `phase` (8 bits, 0..254 = 0..1.27 of the cycle, 255 = stopped / unknown) and the 2-bit production state.

## 5. Results (3 simulated days, 12 windows of 6 h)

`python scripts/build_map.py` (3 days = 12 windows of 6 h, 8 processes, 518,400 ticks at 2 Hz, 8 min wall on a shared machine) writes
`out/layered_map.npz` (1.2 MB of counts) and the figures in `out/map/`. Time of day x production state slices with enough data
(>= 300 observed ticks): 86 of 192 for people and for vehicles (running 45 of 48 times of day, andon 32, break 6, shift change 3 at
the calendar times).

**Size on the air** (`summary.json`, figure `map_size_and_ble_assembly.png`)

| Section | Raw bytes |
|---|---|
| meta (10 machines) | 124 |
| L1 box covers (10 machines x 17 bins, <= 12 boxes) | 4,823 |
| L2 vehicles (sparse deltas) | 15,108 |
| L3 people (sparse deltas) | 9,004 |
| payload | 29,068 raw -> **15,323 B zlib** = **64 chunks** of 240 B |
| signed manifest | 417 B in 2 frames (per-chunk hashes, Ed25519) |
| XOR parity (1 per 8 chunks) | +8 chunks; 18,470 B on air per carousel loop in total |

**Assembly time over BLE** at 20 frames/s with 10 % iid loss, robot enters at a random point of the carousel, 100 trials:

| | loop | mean | median | p95 | max |
|---|---|---|---|---|---|
| no parity | 72 frames, 3.6 s | 8.1 s | 7.2 s | 11.7 s | 14.4 s |
| XOR parity 1/8 (default) | 82 frames, 4.1 s | **6.6 s** | 6.9 s | **8.1 s** | 12.0 s |

All 100 trials completed in both cases. The map is valid for 6 h and is cached per version, so a robot pays this once per version
(a robot that already holds the version downloads nothing).

**Accuracy** (`tests/test_mapping.py`, 4 h of simulated floor: two breaks, a shift change, running, and a traced window with three andon stops;
ground truth = `truth/kinematics.py` sampled every 0.1 s over 5 min, used only for scoring)

| Check | Result |
|---|---|
| L1 envelope (p >= 5 %) covers true swept cells | 99.4 - 100 % per machine (target >= 95 %) |
| L1 false area (learned cells outside the true swept set) | 5.1 - 8.8 % per machine, 6.0 % aggregate (target <= 20 %) |
| broadcast box covers (<= 24 boxes) | coverage 99.4 - 100 %, false area 5.7 % |
| phase specificity (S30, S80, HEM1, SEAL1) | the bin's envelope covers 96.6 % of what moves in that phase, the opposite phase 59 %; one bin = 44 % of the all-phase union |
| phase bin vs kinematics' own cycle phase (+-1 bin) | 96.5 - 99.9 % of ticks, phase known 100 % after warm-up |
| production state | break labelled 100 %, andon recall 100 % (ticks after the missed index), 0 false andon while running |
| break people crowd, north aisle x < 60 m, z 30-34 m | expected people 27.7 in break vs 8.6 running (3.2x), peak cell occupancy 0.92 vs 0.38 |
| codec round trip | box covers identical; 14,380 L2/L3 cells with p >= 3 %: 0 off by more than 2 codes; heading mismatches 9.8 % (hysteresis); same fallback behaviour |
| tamper | flipped manifest bit, forged map (other key), unknown key id, altered chunk with valid CRC, flipped chunk bit, expired and stale-version replay, wrong site: all rejected, no map accepted |
| loss | 40 % iid loss completes in a few loops; first loop lost completes; chunks before manifest verified later; parity rebuilds one lost chunk per group without receiving it |

Figures (`out/map/`): `L1_machine_envelopes.png`, `L2_vehicle_flows.png`, `L3_people_flows.png` (one per layer),
`L3_people_slices.png` / `L2_vehicle_slices.png` (running 10:00 and 03:00, break 09:00 and 11:30, shift change 14:00, andon),
`S30_phase_bins.png` (S30 envelope for every phase bin with its box cover), `map_size_and_ble_assembly.png`.

Observations: the vehicle layer hardly depends on the time of day because `truth/agents.py` moves vehicles on fixed loops with no
schedule; the people layer changes strongly with the production state (break crowds standing along the north aisle at the station
positions, data-hall technicians only 07-19 h) and with the shift (night: three walkers).

## 6. Limits and assumptions

* The floor simulation here has movers on unobstructed routes (no robot, no mover-mover blocking) and the kinematics of
  `truth/kinematics.py`; the people and vehicle layers therefore learn the routes of one seed. A real site changes slowly; the
  map is rebuilt and re-versioned, robots keep the cached version until a newer signed one is heard.
* Line phase relies on the skids being visible (line index). Without them a line station falls back (after 10 min) to its own
  motion rises, which is unreliable for stations with long dwell pauses.
* Cameras are modelled as perfect classifiers (person / vehicle / part / skid): the perception model (RF-DETR + tracker) is
  upstream of this module.
* `demo_key` signing keys in the tests and the script are public (seeded); a deployment uses a provisioned key.
* Heading bins are only filled for moving bodies (speed > 0.2 m/s), so stationary crowds and docked AGVs have probability but no heading.
* Safety semantics are unchanged: the map is a prior for planning; light curtains / area scanners remain the final layer.

## 7. v4 additions (workstream M, 17 Oct): UNKNOWN, conservative boxes, margin, held-out metrics, change detection

Held-out results, convergence curves and the new-machine run are in `docs/MAP_RESULTS.md`.

* **UNKNOWN cell state.** `MapView.unknown_mask(tod, prod)` (site grid) = `_unknown_static()` (camera regions with fewer
  than `min_ticks` observed ticks in total, `unknown_rects` such as a new machine still being learned, the zone of any
  machine without a single well-observed phase bin) + every cell of a camera in `cam_offline` + the whole site when the
  traffic slice cannot be resolved for either class. `predict_obstacles` adds UNKNOWN as boxes owned by `"UNKNOWN"`
  (`with_unknown=True`), so a consumer that treats boxes as occupied treats UNKNOWN as occupied. An unseen machine's
  `envelope()` is its whole zone grid (it used to be empty, i.e. free). Detections from an offline / bumped camera are
  no longer counted into the map (`LayeredMap.update`).
* **Broadcast payload v2** (`codec/maplayer.py`): v1 + an UNKNOWN section, LEB128 run lengths over the flattened [nx, nz]
  site grid (known / unknown alternating, known first; ~10-80 B). Offline cameras are folded into it at packing time, so
  a camera going offline means a new map version. v1 payloads still decode (no UNKNOWN). `DecodedMap.unknown_mask` and
  its `predict_obstacles` behave like the learned map's.
* **Conservative box covers.** The L1 box cover that is broadcast is now an outer cover (`safe_box_cover`, `safe=True`
  by default): greedy exact boxes first, the uncovered rest as bounding boxes of its components, then least-area-cost
  merges down to `max_boxes`. Every envelope cell is inside a box (the old inner cover stopped at 98.5 % of the cells,
  i.e. sent up to 1.5 % of the envelope as free). Price: about 6 % extra box area on the 3-day map (mean over machines x bins).
* **Inflation margin** `S = K * T + C` (`iso13855_margin`, `predict_obstacles(inflate_m=S)`, `dilate` on grids). How the
  terms map here and the measured trade-off curve: `docs/MAP_RESULTS.md` section 3.
* **`predicted_bins(i, phase, prod, dt)`** is the single place where the robot-facing bin choice (phase advance, jitter
  widening, STOP in breaks / andon, all bins when unknown) lives; `predict_obstacles` and the metrics both call it.
* **Camera shim stress terms** (off by default, so map builds are unchanged): transient clutter blobs, persistent
  flicker sources, camera bumps (a camera's detections projected off by (dx, dz) from a time on).
* **`LayeredMap.add_machine(id, rect)`** appends a discovered machine (empty L1 over its zone + margin);
  `observer/change.py` fills it (`l1_view`).
* **Change detection** (`observer/change.py`, no truth): per-cell one-sided CUSUM on unexplained motion in L0-static
  space -> candidate region UNKNOWN from the first alarm -> `change_detected` after 30 s -> rise-to-rise intervals of the
  region's motion -> spawn an L1 model once the last K = 6 intervals agree within 5 % and the period is stable, unless the
  region's cycle is phase-locked to a known machine next to it (`change_explained`: its displaced image, e.g. a bumped
  camera) -> UNKNOWN lifted once every phase bin has `min_visits` ticks and 4 more cycles passed (`machine_converged`).
  Aperiodic persistent motion stays UNKNOWN and is cleared after 10 min without evidence (`change_cleared`). Camera
  bump: every 30 s the last 3 min of part detections near known machines are matched against their learned envelopes
  at shifts of +-2 m; a clearly better non-zero shift twice in a row = `camera_bump`: that camera's cells go UNKNOWN and
  its change candidates are dropped.
