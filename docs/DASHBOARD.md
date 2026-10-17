# Live map view (workstream D)

One static web page shows the observer's layered map being built and sharpening over time. It is the demo judges can
open and video scene 2 (`DIRECTION.md` §10). Everything on it is **SYNTHETIC: simulated plant**.

Files:
* `intercycle/observer/stream.py`: the snapshot writer and the live relay (M calls it). It never imports truth.
* `dashboard/`: `index.html`, `style.css` and `app.js` (no build step, no framework), plus:
  * `layout.json`: the L0 plant;
  * `runs.json`: the recorded runs;
  * `export_static.py`: layout, replay runs and the single-file bundle;
  * `record_frames.mjs`: headless capture;
  * `fake_feed.py`: a test fixture.
* Tests: `tests/test_stream.py`.

## Open it

```bash
# replay, served from the repo root (the page fetches ../out/stream/<run>/)
.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
#   http://127.0.0.1:8000/dashboard/                          registered runs (dashboard/runs.json)
#   http://127.0.0.1:8000/dashboard/?runpath=../out/stream/week1   any run folder
# no server: a single self-contained file (CSS, JS, layout and the runs inlined)
.venv/Scripts/python.exe dashboard/export_static.py bundle            # out/stream/dashboard_bundle.html (open via file://)
.venv/Scripts/python.exe dashboard/export_static.py bundle --artifact --out out/stream/dashboard_artifact.html   # claude.ai artifact
# live: the writer serves ws://127.0.0.1:8765 while it runs; open the page with ?mode=live
.venv/Scripts/python.exe scripts/map_timeline.py --live               # M's script (StreamWriter(live=True))
#   http://127.0.0.1:8000/dashboard/?mode=live      (ws=ws://host:port to override)
```

URL parameters:

| Parameter | Meaning |
|---|---|
| `run=<id>` | A run registered in `runs.json` |
| `runpath=<dir>` | An unregistered run folder |
| `mode=live` | Connect to the live websocket |
| `ws=<url>` | Websocket address (default `ws://127.0.0.1:8765`) |
| `autoplay=1` | Start playing on load |
| `speed=<x>` | Playback speed |
| `u=<0..1>` | Start position on the slider |
| `machine=<id>` | Machine shown in the phase strip |
| `layers=l1,l2,l3,unk,dir,cycle` | Layers switched on at load |
| `base=<dir>` | Folder holding the run folders (default `../out/stream`) |

**GitHub Pages:** `out/` is gitignored, so a published page needs its runs under `dashboard/` (or the bundle). This is
**Proposed** for the lead: at freeze, copy the final runs to `dashboard/data/` and point `runs.json` at them.

## The page

* **Plant view (top-down):** the L0 static map (walls, conveyor, robot bases, light-curtain posts, racks, columns, data
  hall, ceiling cameras) and the hazard zones. Layer toggles:
  * **L1** machine envelopes: all phase bins faint, the bin for the current phase bright, the STOP bin red;
  * **L2** vehicle occupancy and **L3** people occupancy (blue and aqua; alpha follows probability);
  * **UNKNOWN** (hatched; treated as occupied);
  * flow-direction arrows;
  * **cycle preview**: each envelope steps through its learned cycle at 4× real time, starting from the snapshot's
    phase-now. It is labelled as a preview.

  A machine that appears in `l1` but not in L0 (the new machine) is drawn in yellow and labelled "learned". Events with a
  location pulse on the map.
* **HUD:** sim time, observation time, the discovered production state (colour + text) and the UNKNOWN area.
* **Replay:** time slider with play/pause and ¼×–4× speed. A full sweep takes 30 s at 1×.
  * The axis is **log observation time** (ticks at 1 h, 6 h, 1 day, 3 days, 1 week) when `t_obs` grows across the run.
  * It is **sim clock time** when `t_obs` is constant, or when `meta.axis` says so.
  * The track shows snapshot ticks (coloured by production state) and event diamonds.
* **Live:** connects to the websocket, gets the index plus the last snapshot on connect, and appends snapshots and events
  as they arrive. The cursor follows the newest snapshot until you scrub. It reconnects every 2 s.
* **Metric curves:** up to 4 small multiples, each on its own y-axis. Priority: `l1_iou`, `unsafe_free`,
  `false_occupied`, `brier`, `auroc`, ... Each has a cursor synced to the slider and a hover crosshair with tooltip.
  Clicking a curve seeks.
* **Event timeline:** every event with kind badge and caption. Past events are bright and the latest is highlighted;
  clicking one seeks to it. The latest event is also shown as a caption on the map (the video captions: change detected →
  machine spawned → Gemma phase names).
* **Machine list and phase strip:**
  * The machine list shows phase-now (16 bins + STOP) and the learned period.
  * The phase strip shows the selected machine's L1 envelope area per equal-slice phase bin (bars), a moving phase-now
    marker, and a band of **discovered sub-phases**. The band comes from `meta.phases[mid]` if the run has it, else from
    `dashboard/phases.json`: workstream P's model (`export_static.py phases` reads `out/phases/model_final_train.pkl`),
    with names `p0..`, `idle` and cycle fractions.
  * **Gemma names:** `dashboard/gemma_names.json` holds Gemma's names per P bin. It is a copy of G's
    `out/gemma/phase_names.json` made by `export_static.py gemma`, with the shape
    `{model, provider, machines: {mid: {bin: {bin_name, name, confidence, notes}}}, line_clusters}`. The strip title
    then says "named by Gemma 4 on AMD", and the note names the model and Token Factory.
  * Until G's file exists, the same command writes a **labelled FIXTURE** (`fixture: true`, names `[fixture] p3`,
    strip title "FIXTURE names"). Re-run `export_static.py gemma` and rebuild the bundle once the real file lands.
  * A `phase_named` event with `machine` and `bin` (P's bin index) also renames a segment; `start` / `end` fractions
    work too.
  * When a run streams no `phase_now` (week1), the marker is a cycle preview from phase 0 at the learned period, and the
    note says so.

## Data contract: `intercycle.stream/1`

```python
from intercycle.observer.stream import StreamWriter
sw = StreamWriter(run, out_root="out/stream", live=False, port=8765, meta=None)   # meta: optional, stored in index.json
sw.snapshot(t_sim=, t_obs=, prod_state=, grid={x0, z0, res, nx, nz}, l1={mid: [(phase_bin, x0, z0, x1, z1), ...]},
            l2_occ=, l2_head=, l3_occ=, l3_head=, unknown=, phase_now={mid: bin or None}, metrics={name: number})
sw.event(t_sim=, kind=, **fields)   # change_detected, machine_spawned, phase_discovered, phase_named, camera_offline
sw.close()
```

* Grids are **(nz, nx)**, row = z. `LayeredMap.traffic()` returns `[nx, nz]`, so pass `.T`. A transposed input is
  accepted with a warning; any other shape raises.
* Headings: 0..7, where 0 = +z, clockwise towards +x; -1 = none.
* `phase_now`: bin 0..15, 16 = stopped, `None` = unknown.
* Non-finite metrics become `null`.
* A new `StreamWriter(run)` clears that run folder's old snapshots and events, so use one run name per run.

Files in `out/stream/<run>/`:
* `snap_<t_sim:09d>.json`, one per snapshot (≤ 200 kB);
* `events.jsonl`, appended;
* `index.json`, the manifest the page loads first. It holds the run meta, every snapshot (`file, t_sim, t_obs, prod_state,
  metrics, bytes`) and every event, and is rewritten atomically after each call. `complete: true` after `close()`.

Snapshot body: `{schema, label, run, seq, t_sim, t_obs, prod_state, phase_now, metrics, grid, grid_in, downsample, l1, layers}`.

| Field | Encoding |
|---|---|
| `layers.l2_occ`, `l3_occ` | `{"enc": "u8", "scale": 255, "b64"}`: `round(p*255)` as uint8, base64, row-major (nz, nx). Above 16 kB raw: zlib first, `"enc": "u8z"` (browser: `DecompressionStream("deflate")`) |
| `layers.l2_head`, `l3_head` | `{"enc": "u8"/"u8z", "none": 255, "b64"}`: 0..7, 255 = none |
| `layers.unknown` | `{"enc": "rle", "runs": [n0, n1, ...]}`: run lengths over the flattened mask, alternating False/True, starting with False |
| `l1` | `{mid: [[bin, x0, z0, x1, z1], ...]}`: metres, site frame, 3 decimals |
| over 200 kB | Grids are max-pooled by an integer factor `k` (conservative for occupancy and UNKNOWN; heading taken from the max-occupancy cell); `grid.res` = `k × grid_in.res` |

Measured sizes: the 120 × 48 site grid at 1 m with all 17 L1 phase bins for 10 machines (≤ 8 boxes each, ~1250 boxes)
gives a **62.9 kB** snapshot. About 30 kB of that is the L1 boxes; the 4 grids + mask are ≈ 30 kB.

Live websocket messages (JSON text):
* `{"type": "hello", "index", "last"}` on connect;
* `{"type": "snapshot", "snap"}`;
* `{"type": "event", "event"}`;
* `{"type": "close"}`.

The relay is the only backend: an asyncio `websockets` server in a daemon thread inside the writer's process, on
127.0.0.1. If `websockets` is missing or the port is taken, it warns and keeps writing files.

## Runs

| Run | What it is |
|---|---|
| `build_map_72h` (`export_static.py npz`) | **Skeleton stand-in:** the final `out/layered_map.npz` from `scripts/build_map.py` (72 h observed), replayed over one day: 48 snapshots every 30 min (+2 min, so the 5-min shift-change stops are hit). It does **not** grow. L2/L3 follow the time-of-day slice; the production state comes from the public calendar; phase-now is projected from the learned period (line stations STOP in breaks). Metrics are map statistics only (slice observations, hot area, mean occupancy, UNKNOWN share), not accuracy |
| `week1` (M, `scripts/map_timeline.py replay`) | The map growing: cumulative maps at checkpoints from 15 min to 1 week (log axis). Metrics on held-out day 7: `l1_iou`, `unsafe_free`, `false_occupied_share`, `l2/l3_auroc`, `l2/l3_brier`, ... No `phase_now` (the replay merges parallel window counts, so no tracker state exists; the phase strip runs a labelled cycle preview) and no events |
| `newmachine` (M, `scripts/new_machine.py`) | A new machine installed at `meta.t_install`; `t_obs` = time since install (negative before). The axis is sim time shown as T±h relative to install, with an install marker. Events, each with its own caption: `change_detected` / `change_cleared` / `change_explained` (candidates, `rect`), `camera_bump`, `machine_spawned` (`period_s`, `n_cycles`), `machine_converged`. It streams the observed `phase_now` every 5 sim-min, so use this run for the phase-strip scene. Metrics: `unsafe_free_{pre_detection, transition, learned}` |
| `_fixture_feed` / `_fixture_live` (`fake_feed.py`) | **TEST FIXTURE, fake numbers.** Exercises the log axis, events, UNKNOWN, a new machine and live mode. Not registered, never a result |

## Recording (video scene)

```bash
node dashboard/record_frames.mjs --url "file:///C:/AMD/intercycle/out/stream/dashboard_bundle.html?machine=S30" \
     --frames out/stream/rec --seconds 30 --fps 15        # headless Chrome via CDP; ~0.6 s per 1920x1080 frame (450 frames ≈ 4.5 min)
FF=$(.venv/Scripts/python.exe -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -framerate 15 -i out/stream/rec/f_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 out/stream/map.mp4
```

The recorder steps the slider from `u0` to `u1` and pins the page clock to `k / fps`. The capture is deterministic and
smooth however slow the screenshots are. `imageio-ffmpeg` (bundled ffmpeg) was installed into the venv for this.
