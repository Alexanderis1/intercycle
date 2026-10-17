# InterCycle — 2-day build plan (AMD Developer Hackathon ACT III)

*Working name. Plan v1, written 2026-10-17 11:00 CEST. Supersedes `CONSOLIDATED-v2-DELTA.md` where they disagree (see §2).*

**Deadline assumption:** submission closes **18 Oct 16:00 CEST** (Milan on-site cutoff; the online page says 12–17/18 Oct with no time). **Confirm on the lablab Discord / schedule tab today.** Internal code freeze: **18 Oct 12:30**.

**Operator decisions (made):** Go. Scope = full v2. Public **MIT** repo. Solo + Claude. Demo = video + live dashboard. Compute: Modal (own credits) now → AMD (Token Factory + Developer Cloud / ROCm notebook) when access lands.

---

## 1. The product in one paragraph

Fixed cameras already watching a plant floor learn **when each zone is busy**: time-of-day traffic of people and forklifts, and the **cycle** of each automatic machine (period, phase, busy fraction, jitter, micro-stop rate). An observer box publishes this as an **Ed25519-signed QR code on a small screen** at the cell entrance. Any visiting inspection robot with a camera reads it — no Wi-Fi, no pairing, no PLC/DCS integration — and a **safe-interval planner (SIPP)** times its approach so it inspects **between cycles, not between shutdowns**. The prediction only ever affects *timing*: entry into any machine envelope is decided by the cell's own **interlock**, from true machine state.

- **Track:** 1 — Intelligent Industry.
- **Buyer:** maintenance / reliability manager (EHS co-signs).
- **Headline:** *"Inspect between cycles, not between shutdowns."*
- **Context number (not our gain):** Siemens *True Cost of Downtime 2024*: an hour of downtime costs ~$2.3M in a large automotive plant, ~$36k at the FMCG low end.

## 2. What changed from the brainstorm (verified 17 Oct)

| Brainstorm said | Verified | Plan change |
|---|---|---|
| Use "SDG-Warehouse" for traffic | It's 10–15 s staged incident clips (24 TB), no time-of-day, no machines | Use **nvidia/PhysicalAI-SmartSpaces** (CC-BY-4.0, people/forklifts/pallet trucks, track IDs, calibration, `map.png`) for **perception validation only**. Time-of-day and machine layers are **synthetic and labelled as such**. |
| IPAD as real machine data | License not found | Not used. Optional "future work" mention only. |
| "First system that…" | FreMEn (T-RO 2017), *Now or later?* (ICRA 2015), Kairos (2026), camera cycle-time vendors, signed offline QR all exist | Claim the **integration**: air-gapped, camera-only, vendor-neutral hand-off to *any* visiting robot, plus SIPP under **uncertain, ageing periodic phase** (a gap in SIPP literature, which assumes known trajectories). Never say "first". |
| ANYbotics Shift "schedules windows via DCS" | Shift = DCS-*triggered* missions + calendar time windows (Shift 1.0, Oct 2026) | Describe it exactly that way. Our edge: no control-system link at all. |
| Phase valid ~50 min @0.5% jitter → re-issue ≤45 min, e-paper | Monte Carlo: 95%-valid time is 10–31 min @0.5%, 3–13 min @1%, **seconds** with micro-stops (1/10 min) | **Two-layer code** (§5): slow layer valid for hours; fast phase layer on a **live screen refreshed every ~2 s**, each entry carrying its own TTL. E-paper dropped. |
| Period at 0.1 s | Drifts up to 6.75 s in 45 min at T=20 s | **1 ms** period resolution. |
| QR v20-Q fits (cap "485") | v20-Q byte capacity is **482 B**; our sizes fit v19-Q | Target ≤ 482 B; encode with `boost_error=False`. Decode with **zxing-cpp** (OpenCV fails on binary). |
| Detector: YOLO | Ultralytics is **AGPL-3.0**, conflicts with MIT / "MIT-compliant" rule | **RF-DETR** (Apache-2.0) + `roboflow/trackers` ByteTrack (Apache-2.0) + `supervision` (MIT). |
| Isaac Sim for visuals | Needs RT-core GPU (Modal L40S), 10–14 h setup, cannot run on MI300X | **Not on the critical path.** Visuals = Three.js replay of the metrics sim. Isaac = stretch only. |

## 3. Architecture

```
            ┌───────────────────── OBSERVER (fixed cameras) ─────────────────────┐
 SmartSpaces│  video ─► RF-DETR ─► ByteTrack ─► zone polygons ─► occupancy(t)     │  ◄─ runs on AMD (ROCm)
 real video │                                   (validated vs ground truth)       │      / Modal L4 fallback
            │                                                                     │
 synthetic  │  zone activity(t) ─► Lomb-Scargle + ACF + epoch fold ─► cycles      │  CPU
 plant sim  │  zone occupancy(t)─► hour-of-day profile ─► traffic table           │
            │  VLM (Token Factory): label zones/ROIs, sanity-check machine state  │  ◄─ AMD Token Factory
            └───────────────┬─────────────────────────────────────────────────────┘
                            ▼
              CODEC: slow layer (hours) + fast layer (seconds) ─► Ed25519 ─► QR v≤20-Q
                            ▼   (one-way optical link: on-site screen)
            ┌──────────── VISITING ROBOT (sim) ───────────┐
            │ zxing-cpp decode ─► verify sig/TTL/conf ≥ 4 │
            │ mission order from slow layer               │
            │ SIPP on grid × time, unsafe intervals from  │
            │ fast layer (inflated by jitter & age)       │
            │ request entry ─► INTERLOCK (true state) ────┼──► open / deny
            └─────────────────────────────────────────────┘
                            ▼
              EVAL: robots A / B / C × held-out seeds ─► metrics ─► dashboard (Three.js) + report (LLM)
```

**Where things run**

| Piece | Day 1 | When AMD access lands |
|---|---|---|
| Sim, SIPP, codec, cycle learner, tests | Windows laptop (Python 3.14 via `py`; numpy/scipy present) | unchanged (CPU) |
| Seed sweeps | laptop multiprocessing, or Modal CPU `.map` | unchanged |
| Detect + track on SmartSpaces | Modal L4 (`.map` per camera) | **MI300X (ROCm PyTorch)** — the headline AMD workload, with fps + accuracy table |
| VLM / LLM | cached fixtures; Modal vLLM only if needed | **Token Factory** (OpenAI-compatible; `BASE_URL`/`API_KEY`/`MODEL` env) |
| Dashboard | local FastAPI | Modal `@asgi_app` → public app URL |

## 4. Reuse vs write

**Reuse (working, licence-clean):** RF-DETR (Apache-2.0), `trackers` ByteTrack (Apache-2.0), `supervision` PolygonZone (MIT), `astropy.timeseries.LombScargle` (BSD), PyNaCl (Apache-2.0), segno (BSD), zxing-cpp (Apache-2.0), FastAPI (MIT), Three.js (MIT), Modal SDK, SmartSpaces data (CC-BY-4.0, attribute). SIPP reference: `atb033/multi_agent_path_planning` (MIT) — adapt.

**Write ourselves (this is the contribution):** scenario schema + generators, cycle/traffic learner with micro-stop rejection and confidence, two-layer payload codec, SIPP with periodic + uncertain intervals, interlock gate + fault injection, robots A/B/C, evaluation harness, dashboard.

**Mission format (imitate Spot Autowalk):** `mission.yaml` = list of `{waypoint, zone, action: photo|gauge|thermal, dwell_s, expect_range}` → each step writes an artifact + pass/fail + timestamp → `report.json` → LLM-written inspection report.

**Planner granularity:** SIPP runs on a **topological site graph** (~20–30 nodes: corridor junctions, cell gates, inspection points), not the 0.5 m grid. Edge traversal times come from the grid sim. This is ~100–150 lines and matches how it would sit on a real robot.

**Real-robot story (pitch only, not built):** on a real AMR (e.g. Clearpath Husky + ROS 2 Jazzy, Nav2 1.3.x) our planner is a **mission-level timing layer**: it outputs `(leg, depart_t, wait_node)`, and `nav2_simple_commander` executes each leg (Nav2 paths are untimed, so timing acts at dispatch). The interlock maps 1:1 onto an **Open-RMF door** (`rmf_door_msgs/DoorState`): Open-RMF treats doors reactively; we add *predicted* open windows from a signed code. Nav2 / Open-RMF are Linux-only (WSL2/Docker), so they are **not** in the 2-day build. Stretch: Nav2 loopback sim in WSL2 driving the same mission, and a `DoorState` publisher for the gate.

## 5. The code (payload) — two layers

Everything signed with Ed25519 (64 B). Keys: the site's public key is provisioned to the robot once (out of band); `key_id` selects it.

**Slow layer — printed in the code at the zone gateway; valid hours (TTL 8 h):**
- header 12 B: magic+version, flags, site id (4), key id (2), issue time (4, unix s)
- traffic: per zone, 24 × 4-bit busy level (hour of day) = 12 B/zone
- per machine 8 B: zone 6 b · period 17 b (1 ms) · busy fraction 6 b · jitter σ 4 b (log) · drift 6 b · micro-stop rate 4 b · confidence 4 b · reserved 17 b

**Fast layer — live screen at each cell, refreshed every ~2 s:**
- header 12 B + per machine 8 B: zone 6 b · phase anchor 17 b (ms from issue time to next idle start) · state 2 b · TTL 8 b (2 s units, ≤ 510 s) · confidence 4 b · seq 16 b · reserved 11 b
- one gateway screen at the service-lane entrance carries the fast layer for **all** cells (robot plans its departure from the dock); each cell screen carries its own machine (last-metres re-check).
- TTL computed on the observer from that machine's jitter and micro-stop rate (MC result: ~5 min @≤0.5% no stops, ~1 min @1%, ~10 s with micro-stops).

Example sizes: site with 8 zones + 8 machines → slow = 12 + 96 + 64 + 64 = **236 B**; fast for 1 cell (≤4 machines) = 12 + 32 + 64 = **108 B**. Both far under 482 B, so the QR can drop to a lower version (bigger modules → reads from farther away).

**Robot rules:** reject on bad signature, expired TTL, or confidence < 4 → that machine is "unknown" → robot waits at the gate (behaves like robot A). Prediction never opens the gate.

## 6. Simulation & evaluation

**Scenario (`scenario.yaml`, single source of truth):** 40 × 25 m plant on a 0.5 m grid; 8 zones; 3 machine cells — arm (45 s, 70% busy), press (30 s, 60%), conveyor transfer (20 s, 50%); jitter σ ∈ {0.5, 1, 2}%, drift 0.1%/h, micro-stops Poisson 1/10 min, 2–10 s; people + forklifts on corridors driven by a shift timetable (shift change 06/14/22, breaks, dock waves); inspection point per cell with dwell 8 s.

**Interlock semantics (design assumption, stated in the README):** entry is granted only when the machine is truly idle; while the robot is inside, the machine cannot start — if its next cycle would start, it is **held** (counted as production loss). The robot never shares the envelope with a running machine. Near people/forklifts the robot does a protective stop (1.5 m).

**Robots**
- **A — static:** fixed route, requests entry on arrival, waits.
- **B — code:** reads slow + fast layers, orders the mission by traffic, SIPP-times each approach to arrive at the start of an idle window that fits the dwell.
- **C — PLC oracle:** knows the true future machine state (upper bound).

**Metrics:** waiting time per inspection, inspections/hour, % inspections with **zero machine hold**, hold seconds (production lost), gate denials, intrusions (**must be 0**), protective stops, and **gain retained = (B − A) / (C − A)**.

**Anti-circularity protocol:** generator parameters live in `truth/`, which the learner never imports (enforced by a test); learner sees only noisy activity signals (occlusion dropouts, false triggers); train seeds 0–99, **held-out seeds 1000–1199** run once, after code freeze; report period/phase/busy-fraction error vs ground truth and how the gain degrades with σ and micro-stop rate (we show where it *stops* working).

**Safety tests (CI):** fault injection — corrupt phase, flip state, replay old code, forged signature, stuck screen → assert **zero entries with gate closed** and zero intrusions across 10,000 episodes.

**Perception validation (real video):** SmartSpaces `MTMC_Tracking_2026/train/Warehouse_000` (19 cameras, 3.25 GB). Report per-zone occupancy error vs ground truth, and detector/tracker fps on AMD vs CPU. P0 = persons with COCO weights + forklifts from ground truth; P1 = fine-tune RF-DETR on SmartSpaces (Person/Forklift/PalletTruck) on MI300X.

## 7. AMD usage (what judges must see)

1. **Perception on AMD:** batched RF-DETR over many camera streams on MI300X (ROCm PyTorch, `torch.version.hip` printed), benchmark table "N cameras real-time on one GPU vs CPU", `rocm-smi` screenshot.
2. **Token Factory LLM/VLM in the product:** (a) VLM proposes zone polygons + machine ROIs from one keyframe per camera (JSON, human-approved); (b) LLM writes the inspection report and explains each robot decision ("waited 12 s because press idle window opens at 10:42:07, conf 12/15"). Calls logged in the dashboard (model, latency, tokens). Responses cached for replay.
3. Stretch: Gemma on AMD (Google partner prize) via Token Factory or vLLM-ROCm.

Code rules for portability: device-agnostic torch (`"cuda"` works on ROCm), CPU video decode (PyAV/ffmpeg), no TensorRT/NVDEC/DALI, ONNX path uses MIGraphX EP with NMS outside the graph, one OpenAI client configured by env vars.

## 8. Timeline

### Day 1 — Sat 17 Oct
| Time | Block | Done when |
|---|---|---|
| 11:00–11:45 | **Access & setup:** confirm deadline; request Token Factory key + AMD Dev Cloud/notebook; repo skeleton, MIT licence, `pyproject`, CI test runner; Modal token | repo pushed; access requests sent |
| 11:45–13:00 | **Scenario + truth generators** (machines, traffic timetable, seeds) | `scenario.yaml` produces deterministic traces |
| 13:00–14:00 | **Modal ingest** (CPU, background): SmartSpaces Warehouse_000 → Volume. In parallel: **codec** (layout, sign, QR encode, zxing decode, tests) | round-trip test passes, QR PNG |
| 14:00–16:30 | **Cycle + traffic learner** (Lomb-Scargle, ACF, folding, micro-stop rejection, jitter, confidence, TTL) | period error < 0.1%, phase error reported |
| 16:30–19:30 | **Sim core:** grid world, agents, interlock gate, **SIPP**, robots A/B/C | single episode runs all three |
| 19:30–20:15 | Dinner | |
| 20:15–22:00 | **Eval harness** + fault-injection tests + first metrics table | A/B/C table on train seeds; zero intrusions |
| 22:00–23:30 | **Perception job** on Modal L4 (RF-DETR + ByteTrack + zones), launched to run overnight | job running, writes occupancy parquet |

### Day 2 — Sun 18 Oct
| Time | Block | Done when |
|---|---|---|
| 07:30–09:30 | **Dashboard:** FastAPI + Three.js replay (reuse the explainer artifact's 3D scene), metrics, QR panel, LLM log | runs locally |
| 09:30–10:30 | **AMD run:** perception benchmark on MI300X / ROCm notebook; Token Factory wiring (fall back to cached + labelled if key absent) | AMD numbers in README |
| 10:30–11:30 | **Held-out evaluation** (seeds 1000–1199), sensitivity sweep σ × micro-stops, final figures | frozen results |
| 11:30–12:30 | Deploy dashboard to Modal (public URL); README (claims, non-claims, licences, attribution) | **code freeze 12:30** |
| 12:30–14:30 | **Video** (2–3 min) + slide deck + cover image | uploaded |
| 14:30–15:30 | Submit on lablab; buffer | submitted ≥ 30 min early |

**Stretch, only if ahead:** RF-DETR fine-tune on MI300X; Isaac Sim 5.1 render on Modal L40S (15-min smoke test first: `vulkaninfo` + one frame); Gemma prize; floor plan imported from SmartSpaces `map.png`.

**Cut order if behind:** Isaac → fine-tune → VLM zone labelling (keep LLM report) → traffic layer in SIPP (keep it in mission ordering only) → Modal deploy (submit local video + repo). **Never cut:** gate + fault-injection test, held-out seeds, the AMD perception run.

## 9. Repo layout

```
intercycle/
  scenario/       schema.py (pydantic), plant.yaml
  truth/          machines.py, traffic.py        # learner must not import
  observer/       perception_modal.py, zones.py, learn_cycles.py, learn_traffic.py
  codec/          layout.py, sign.py, qr.py
  sim/            world.py, sipp.py, gate.py, robots.py
  eval/           run_seeds.py, metrics.py, figures.py
  llm/            client.py (OpenAI-compatible), report.py, label_zones.py, cache/
  dashboard/      app.py (FastAPI), static/ (Three.js)
  tests/          test_codec.py, test_sipp.py, test_gate_faults.py, test_no_truth_import.py
  modal_app.py    README.md  LICENSE (MIT)  NOTICE (CC-BY-4.0 attribution)
```

## 10. Risks (say them out loud)

1. **Why not just read the PLC?** We win only in brownfield cells where a visiting/third-party robot has no control-system access. C shows what a PLC link would give; we report gain retained.
2. **Phase ages fast.** Hence the two-layer code + live screen + per-entry TTL; and the cycle layer only works for **tightly timed automatic machines** (presses, arms, conveyors), not human-paced work. We show the σ/micro-stop range where it breaks.
3. **Circular demo.** Synthetic machines learned by our own observer — mitigated by hidden generator, noise, held-out seeds, and reporting error vs truth. Real video is used only where we have it (perception).
4. **"Why not BLE?"** One-way optical link: no RF approval, no pairing, works in RF-restricted areas, robot-vendor neutral, acts like a data diode. Honest: BLE would also work in many plants.
5. **Design assumptions:** interlock semantics, jitter, thresholds and dwell are assumptions, not from a standard or a real cell.
6. **Access risk:** Token Factory key / MI300X may arrive late → everything runs on Modal/CPU first; AMD run is a swap of env vars + device.
7. **Deadline risk:** cutoff time unconfirmed → freeze at 12:30 on 18 Oct.

## 11. Submission checklist (lablab)

Title · short + long description · tags · cover image · 2–3 min demo video · slide deck · public GitHub repo (MIT) · app URL (Modal) · README with AMD benchmark table, licences, SmartSpaces attribution, synthetic-data labels, non-claims.

## 12. Open items

- [ ] Confirm exact submission time & timezone.
- [ ] Token Factory key + endpoint (`GET /models` to pick VLM/LLM).
- [ ] AMD Developer Cloud / notebook access path for MI300X.
- [ ] Final product name (InterCycle is a placeholder).

## Sources (selected)

SmartSpaces https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces · ACT III https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii · Siemens TCOD 2024 PDF https://assets.new.siemens.com/siemens/assets/api/uuid:1b43afb5-2d07-47f7-9eb7-893fe7d0bc59/TCOD-2024_original.pdf · ANYbotics Shift https://www.anybotics.com/news/shift-launch/ · SIPP doi:10.1109/ICRA.2011.5980306 · *Now or later?* doi:10.1109/ICRA.2015.7139315 · FreMEn doi:10.1109/TRO.2017.2665664 · Kairos arXiv:2609.27467 · RF-DETR licence https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/ · trackers https://github.com/roboflow/trackers · Ultralytics AMD https://docs.ultralytics.com/integrations/amd · Modal GPUs https://modal.com/docs/guide/gpu · Isaac on Modal https://github.com/TheFloatingString/isaac-lab-on-modal · QR capacity https://www.thonky.com/qr-code-tutorial/character-capacities · staleness Monte Carlo: `scratchpad/math/mc_phase.py` (this session).
