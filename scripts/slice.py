"""Thin vertical slice of the press-shop twin (PLAN_TWIN §4 "Thin vertical slice"; lane S1a item 3).

    python scripts/slice.py [--out out/slice/<tag>] [--world 0] [--day 9] [--zone z9]
                            [--modal timing|patrol] [--cpu 4] [--max-rtf 2] [--skip-guard-tests]

Pipeline (one record per step in <out>/slice_report.json; the process exits 1 if any step or assertion fails):
  1. build    site.yaml (or the G0 stub site_v0) -> twin/compile.py + intercycle_sim/world.py (twin/build_worlds.py)
              -> per zone boxes.json, static.usda, mesh.npz, map.pgm/.yaml. tasks.yaml falls back to the G0 stub
              twin/tasks_v0.yaml while S2 TASKS has not landed (compile.default_inputs(allow_stubs=True), logged).
  2. day      the stub day generator (twin/daygen_stub.py): 30 min, 2 presses, Poisson walkers, 1 forklift.
  3. engine   the twin plant (twin/plant.py) runs the engine and writes the realised stream (2 Hz) of the first
              20 min with the robot parked off the visible floor (robot off while the observer learns).
  4. adapter  camera_v0 (MEASURED_SHIM) + the plant's ObsAdapter write Observation v2 at 2 Hz (obs_v2.jsonl).
  5. observer twin/observer_live.py learns 20 min from obs_v2.jsonl only and signs the slow / fast / map layers; the
              receiver checks use an explicit keyring and must fail closed (empty keyring, foreign key, tampered
              blob, key outside the keyring).
  6. mission  one 10 min mission of the "learned" condition in one zone:
              * local (always): the LOCAL SCRIPTED DOUBLE drives the plant hooks the way intercycle_sim.runner
                calls them (poses / after_step / frame_extras / on_ros / finish, 10 Hz), along the zone's compiled
                TwinSiteGraph (R-W10) in mission order. At the first press checkpoint it is a TRIP PROBE: it waits at
                the generated wait pose until the Observation shows the press stroking and then enters, so the
                mission contains a robot-caused press stop for the propagation assertion. It is a wiring double, not
                the learned policy, and its mission scores Y = 0 by design.
              * --modal timing: the fake backend + Nav2 on Modal CPU (cloud/modal_isaac_robot.py fake path, cpu 4,
                max_rtf 2) with the robot-side learned stack (observer_bridge + timing_planner on the twin scenario).
              * --modal patrol: the same episode with send_goal.py driving Nav2 through the zone's graph nodes
                (wiring check of the twin world + Nav2 + plant; no feed, no inspection events).
  7. log      the plant writes summary.json and log.parquet (log.jsonl in the Isaac python); both are validated.
  8. metric   the primary-metric JSON (twin.schemas.runs.PrimaryMetric) of the mission, validated against the
              model and against the exported JSON schema (twin/schemas/json/primary_metric.schema.json).

Assertions (PLAN_TWIN §4): the truth-allowlist and Isaac-dependency tests pass; signatures verify with an explicit
keyring and fail closed; >= 1 inspection attempt; the metric JSON validates against its schema; a press stop caused
by the robot appears in the Observation within 1 tick + latency.

Truth: this script is an ORCHESTRATOR (twin/truth_guard.py): it imports the plant and the generator to start them and
never opens a truth file. Everything it checks is read from the non-truth outputs (obs_v2.jsonl, summary.json, the
run log). The day event script is written by the generator and opened by the plant only.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "ros2" / "intercycle_sim", REPO / "ros2" / "intercycle_nav", REPO / "ros2" / "intercycle_eval",
           REPO / "cloud"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from nacl.signing import SigningKey  # noqa: E402

from intercycle.codec import maplayer as ML  # noqa: E402
from intercycle.codec.trust import TrustError, TrustStore, issue_cert  # noqa: E402
from intercycle.margin_inflate import SCHEMA as MARGIN_SCHEMA, Margin, inflation_spec  # noqa: E402
from intercycle.observer import mapping as OM  # noqa: E402
from intercycle.types import DAY0_UNIX  # noqa: E402
from twin import build_worlds as BW  # noqa: E402
from twin import compile as C  # noqa: E402
from twin import daygen_stub as DG  # noqa: E402
from twin import observer_live as OL  # noqa: E402
from twin import plant as PL  # noqa: E402
from twin.schemas import json_schema, load  # noqa: E402
from twin.schemas.dayscript import FILE_SUFFIX  # noqa: E402
from twin.schemas.dynamics import Dynamics  # noqa: E402
from twin.schemas.runs import (CheckpointOutcome, MissionMetric, PrimaryMetric, Summary,  # noqa: E402
                               world_values)
from twin.schemas.site import Site  # noqa: E402
from twin.schemas.tables import LogRow, validate_table  # noqa: E402

LEARN_S = 1200.0                 # the observer learns 20 min (PLAN_TWIN §4 step 5)
MISSION_S = 600.0                # 1 mission of 10 min (step 6)
DAY_S = LEARN_S + MISSION_S      # the stub day: 30 min (step 2)
METRIC_T_S = 720.0               # Summary / MissionMetric fixed denominator T (PLAN_TWIN §5)
DT = 0.1                         # local double step (= the plant's SAFETY_DT)
SPEED = 0.5                      # local double walking speed (m/s; timing.TimingCore's nominal speed)
HOLD_EXTRA_S = 1.0               # the local double dwells dwell_s + this (the plant's capture needs HOLD_S of stillness)
PROBE_WAIT_S = 240.0             # trip probe: give up waiting for a stroking press after this long and enter anyway
PROBE_LOOKBACK_S = 4.5           # trip probe: "stroking" = a motion tick within this window (period prior 4 s)
PARK = (-5.0, -5.0)              # robot parked off every zone's visible floor while the observer learns
TICK_S = 0.5                     # Observation / realised stream tick (2 Hz)
GUARD_TESTS = ("tests/test_truth_allowlist.py", "tests/test_isaac_deps.py")
FILES_STEP1 = ("boxes.json", "static.usda", "mesh.npz", "map.pgm", "map.yaml")
REMOTE_REPO = "/root/repo"       # cloud/modal_isaac_robot.py mounts twin/, intercycle/ and scenario/ here
STUB_MARGIN = dict(margin_m=0.5, cell_m=0.25)


class SliceError(RuntimeError):
    """A slice step or assertion failed (the report says which)."""


# ------------------------------------------------------------------------------------------------ helpers

def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _state(x: float, y: float, psi: float, vx: float = 0.0, vy: float = 0.0) -> dict:
    """The runner's robot state dict (ROS frame: x = plan x, y = plan z, psi from +x counter-clockwise)."""
    return {"pos_w": [x, y, 0.6], "quat_wxyz": [math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)],
            "v_w": [vx, vy, 0.0], "omega_b": [0.0, 0.0, 0.0]}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_log(run_dir: Path) -> list[dict]:
    """The plant's run log: log.parquet (CPU double) or log.jsonl (Isaac python), validated against LogRow."""
    pq_path = run_dir / "log.parquet"
    if pq_path.exists():
        validate_table(pq_path, LogRow)
        import pyarrow.parquet as pq

        return pq.read_table(pq_path).to_pylist()
    rows = _read_jsonl(run_dir / "log.jsonl")
    for r in rows:
        LogRow.model_validate(r)
    return rows


def inputs(allow_stubs: bool = True) -> dict[str, Path]:
    """The contract files in force, plus safety.yaml. tasks.yaml absent (S2 TASKS later) -> the tasks_v0 stub."""
    paths = dict(C.default_inputs(allow_stubs=allow_stubs))
    paths["safety"] = C.TWIN / "safety.yaml"
    return paths


def plant_cfg(paths: dict[str, Path], script: Path, run_dir: Path, *, run_id: str, zone: str, world: int,
              t0: float, seconds: float, condition: str = "learned") -> dict:
    """The runner cfg the plant reads (the EVAL identity fields set explicitly; a short dev mission)."""
    p = dict(site=str(paths["site"]), safety=str(paths["safety"]), tasks=str(paths["tasks"]),
             dynamics=str(paths["dynamics"]), day_script=str(script), run_id=run_id, zone=zone, world=world,
             world_split="dev", condition=condition, code={"head": git_head()}, plant_t0=t0, run_dir=str(run_dir),
             allow_short_mission=True)
    return {"seconds": seconds, "sim": "fake", "plant": p}


class Driver:
    """Steps the plant hooks like intercycle_sim.runner (poses -> after_step -> frame_extras -> on_ros) at DT."""

    def __init__(self, plant: PL.Plant):
        self.plant = plant
        self.t = 0.0
        self.x, self.y, self.psi = PARK[0], PARK[1], 0.0
        self.path_m = 0.0

    def step(self, vx: float = 0.0, vy: float = 0.0, events: list[dict] | None = None) -> None:
        self.t = round(self.t + DT, 6)
        self.plant.poses(self.t)
        self.plant.after_step(self.t, _state(self.x, self.y, self.psi, vx, vy), [])
        self.plant.frame_extras(self.t)
        if events:
            self.plant.on_ros(self.t, events)

    def hold(self, seconds: float, until: Callable[[], bool] | None = None) -> None:
        end = self.t + seconds
        while self.t < end - 1e-9 and self.t < self.plant.T - DT + 1e-9:
            if until is not None and until():
                return
            self.step()

    def goto(self, x: float, y: float) -> None:
        d = math.hypot(x - self.x, y - self.y)
        if d < 1e-9:
            return
        self.psi = math.atan2(y - self.y, x - self.x)
        ux, uy = (x - self.x) / d, (y - self.y) / d
        n = max(1, math.ceil(d / (SPEED * DT)))
        step = d / n
        for _ in range(n):
            if self.t >= self.plant.T - DT + 1e-9:
                return
            self.x += ux * step
            self.y += uy * step
            self.path_m += step
            self.step(ux * step / DT, uy * step / DT)

    def face(self, tx: float, ty: float) -> None:
        self.psi = math.atan2(ty - self.y, tx - self.x)

    def run_out(self) -> None:
        while self.t < self.plant.T - DT + 1e-9:
            self.step()


def shortest(adj: dict[str, dict[str, float]], a: str, b: str) -> list[str]:
    """Dijkstra on the site graph (edge weights in seconds)."""
    dist, prev, q = {a: 0.0}, {}, [(0.0, a)]
    while q:
        d, n = heapq.heappop(q)
        if n == b:
            break
        if d > dist.get(n, math.inf):
            continue
        for m, w in adj[n].items():
            if d + w < dist.get(m, math.inf):
                dist[m], prev[m] = d + w, n
                heapq.heappush(q, (d + w, m))
    if b not in dist:
        raise SliceError(f"no graph path {a} -> {b}")
    out = [b]
    while out[-1] != a:
        out.append(prev[out[-1]])
    return out[::-1]


# ------------------------------------------------------------------------------------------------ steps

def step_build(out: Path, paths: dict[str, Path]) -> tuple[dict, dict[str, C.CompiledZone]]:
    zones = BW.build(paths["site"], paths["dynamics"], paths["tasks"], out / "worlds")
    missing = {z: [f for f in FILES_STEP1 if not (d / f).is_file() or (d / f).stat().st_size == 0]
               for z, d in zones.items()}
    if any(missing.values()):
        raise SliceError(f"world files missing: {missing}")
    compiled = C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])
    stubs = {k: C.is_stub(k, p) for k, p in paths.items() if k in ("site", "dynamics", "tasks")}
    return ({"zones": {z: str(d) for z, d in zones.items()}, "inputs": {k: str(p) for k, p in paths.items()},
             "stubs": stubs, "targets": {z: [t.id for t in cz.graph.mission()] for z, cz in compiled.items()}},
            compiled)


def step_day(out: Path, paths: dict[str, Path], world: int, day: int) -> tuple[dict, Path, Any]:
    site, dyn = load(Site, paths["site"]), load(Dynamics, paths["dynamics"])
    script = DG.generate_day(world, day, site=site, dynamics=dyn, duration_s=DAY_S)
    presses = list(script.presses)
    forklifts = sorted({m.entity for m in script.mover_intents if m.cls == "forklift"})
    walkers = sorted({m.entity for m in script.mover_intents if m.cls == "person"})
    if len(presses) != 2 or len(forklifts) != 1 or not walkers:
        raise SliceError(f"stub day: {len(presses)} presses, {len(forklifts)} forklifts, {len(walkers)} walkers "
                         "(want 2, 1, >= 1)")
    path = DG.write_day(script, out / f"w{world}_d{day}{FILE_SUFFIX}")
    rec = {"file": path.name, "duration_s": script.duration_s, "t0_s": script.t0_s, "presses": presses,
           "forklifts": len(forklifts), "walkers": len(walkers), "generator": script.generator}
    return rec, path, script


def run_learn(out: Path, paths: dict[str, Path], script_path: Path, t0: float, zone: str, world: int) -> tuple[dict, Path]:
    """Steps 3-4: the plant over the first LEARN_S s with the robot parked off the floor."""
    run_dir = out / "runs" / "learn"
    plant = PL.make(plant_cfg(paths, script_path, run_dir, run_id="slice_learn", zone=zone, world=world, t0=t0,
                              seconds=LEARN_S))
    drv = Driver(plant)
    drv.run_out()
    plant.finish(str(run_dir), {"backend": "local_scripted_double"})
    obs = OL.ObsV2Reader(plant.obs_path, len(plant.adapter.press_ids), len(plant.adapter.zone_ids)).poll()
    caps = np.array([o.t_cap for o in obs])
    if len(obs) < int(LEARN_S / TICK_S) - 1 or not np.allclose(np.diff(caps), TICK_S):
        raise SliceError(f"Observation v2 is not a contiguous 2 Hz stream ({len(obs)} lines)")
    if not all(o.is_v2 for o in obs):
        raise SliceError("non-v2 observation in obs_v2.jsonl")
    lat = float(np.median([o.t - o.t_cap for o in obs]))
    return ({"run_dir": str(run_dir), "n_obs": len(obs), "hz": 1.0 / TICK_S, "latency_s": lat,
             "t_cap0": float(caps[0]), "t_cap1": float(caps[-1]), "robot": "parked off the visible floor"},
            plant.obs_path)


def margin_path(out: Path) -> tuple[Path, str]:
    """S7b's frozen twin/m7_margin.json when it exists; otherwise a slice-local STUB margin (on stubs only)."""
    frozen = C.TWIN / "m7_margin.json"
    if frozen.exists():
        return frozen, "twin/m7_margin.json (S7b, frozen)"
    mg = Margin(**STUB_MARGIN)
    p = out / "m7_margin.json"
    p.write_text(json.dumps(dict(schema=MARGIN_SCHEMA, frozen=True, status="ok", inflation=inflation_spec(mg),
                                 note="SLICE STUB: S7b's frozen twin/m7_margin.json is not on this branch",
                                 **STUB_MARGIN)), encoding="utf-8")
    return p, "slice stub (0.5 m on 0.25 m cells; S7b's frozen margin not merged)"


def observer_spec(site: Site, compiled: dict[str, C.CompiledZone], adapter: PL.ObsAdapter) -> dict:
    """The site-wide observer spec in the plant adapter's order (machines = press danger rects, cameras = zones)."""
    zones = {z.id: z for z in site.zones}
    floors = [np.asarray(zones[z].floor_polygon.value, float) for z in adapter.zone_ids]
    allp = np.vstack(floors)
    extent = (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))
    machines = [dict(id=pid, kind="press", rect=tuple(r)) for pid, r in zip(adapter.press_ids, adapter.rects)]
    cams = [(float(f[:, 0].mean()), float(f[:, 1].mean())) for f in floors]
    scn = next(iter(compiled.values())).scenario
    return OL.twin_spec(machines, extent, cams, site_id=scn.site_id, key_id=scn.key_id)


def step_observer(out: Path, spec: dict, obs_path: Path, t_end: float) -> tuple[dict, OL.ObserverLive]:
    mp, msrc = margin_path(out)
    ob = OL.ObserverLive(spec, OL.ObserverLiveConfig(margin_path=str(mp)), obs_path=obs_path)
    n = ob.advance(t_end + 10.0)
    if n < int(LEARN_S / TICK_S) - 1:
        raise SliceError(f"observer ingested {n} ticks")
    diag = ob.diagnostics()
    return {"ticks": n, "margin": msrc, "models": diag["models"], "availability": diag["availability"],
            "tags": diag["tags"]}, ob


def step_signatures(ob: OL.ObserverLive, t: float) -> dict:
    """Sign with a key certified in an explicit keyring; every receiver check must fail closed."""
    sid = int(ob.spec["site_id"])
    now = DAY0_UNIX + t
    root, key, other = SigningKey.generate(), SigningKey.generate(), SigningKey.generate()

    def ring_of(k: SigningKey, at: float):
        st = TrustStore(root.verify_key, sid)
        st.add_cert(issue_cert(root, k.verify_key, sid, int(now - 3600), int(now + 3600)))
        return st.keyring_at(at)

    ring = ring_of(key, now)
    out = ob.sign_layers(key, t, ring)
    slow, fast = OL.verify_blob(out["slow"], ring), OL.verify_blob(out["fast"], ring)
    rx = ML.MapReceiver(ring, sid)
    got = None
    for k, fr in enumerate(out["map"].carousel()):
        got = rx.ingest(t + k / 20.0, fr) or got
    if got is None:
        raise SliceError("map bundle did not verify with the keyring")
    checks: dict[str, bool] = {}
    for name, bad in (("empty_keyring", {}), ("foreign_key", ring_of(other, now))):
        try:
            OL.verify_blob(out["slow"], bad)
            checks[name] = False
        except TrustError:
            checks[name] = True
        rx2 = ML.MapReceiver(bad, sid)
        checks[name + "_map"] = all(rx2.ingest(t + k / 20.0, fr) is None for k, fr in enumerate(out["map"].carousel()))
    tampered = bytearray(out["fast"])
    tampered[20] ^= 1
    try:
        OL.verify_blob(bytes(tampered), ring)
        checks["tampered"] = False
    except TrustError:
        checks["tampered"] = True
    try:
        ob.sign_layers(other, t, ring)
        checks["signer_outside_keyring"] = False
    except TrustError:
        checks["signer_outside_keyring"] = True
    try:
        ob.sign_layers(key, t, ring_of(key, now + 7200))
        checks["expired_keyring"] = False
    except TrustError:
        checks["expired_keyring"] = True
    if not all(checks.values()):
        raise SliceError(f"a receiver check did not fail closed: {checks}")
    return {"key_id": out["key_id"], "slow_machines": len(slow.machines), "fast_machines": len(fast.machines),
            "confident": [m.confidence for m in slow.machines], "fail_closed": checks}


def step_mission_local(out: Path, paths: dict[str, Path], script_path: Path, cz: C.CompiledZone, t0: float,
                       zone: str, world: int) -> dict:
    """Step 6 (local scripted double): one MISSION_S mission in mission order; the first press checkpoint is the
    trip probe (see the module docstring)."""
    run_dir = out / "runs" / "mission_local"
    plant = PL.make(plant_cfg(paths, script_path, run_dir, run_id=f"slice_w{world}_{zone}_learned_local", zone=zone,
                              world=world, t0=t0, seconds=MISSION_S))
    g = cz.graph
    drv = Driver(plant)
    drv.x, drv.y = g.pos["DOCK"]
    reader = OL.ObsV2Reader(plant.obs_path, len(plant.adapter.press_ids), len(plant.adapter.zone_ids))
    seen: list[Any] = []
    probe: dict = {}
    tasks = {c.id: c for c in plant.tasks.checkpoints} if plant.tasks else {}

    def stroking(pid: str) -> bool:
        seen.extend(reader.poll())
        i = plant.adapter.press_ids.index(pid)
        recent = [o for o in seen if o.t_cap >= plant.t0 + drv.t - PROBE_LOOKBACK_S]
        return any(np.isfinite(o.zone_motion[i]) and o.zone_motion[i] > OL.ObserverLiveConfig.motion_thr
                   for o in recent)

    here = "DOCK"
    for tg in g.mission():
        for n in shortest(g.adj, here, tg.node)[1:]:
            drv.goto(*g.pos[n])
        here = tg.node
        cp = tasks.get(tg.id)
        if cp is None:
            raise SliceError(f"checkpoint {tg.id} has no tasks entry")
        inside = tg.kind == "machine"
        if inside and not probe:
            waited0 = drv.t
            drv.hold(PROBE_WAIT_S, until=lambda: stroking(str(tg.machine)))
            probe = {"checkpoint": tg.id, "press": tg.machine, "waited_s": round(drv.t - waited0, 1),
                     "saw_stroke": stroking(str(tg.machine))}
        if inside:
            drv.goto(*g.inspect_pose(tg))
        drv.face(cp.target_xzh[0], cp.target_xzh[1])
        drv.step(events=[{"event": "dwell_start", "target": tg.id, "inside": inside}])
        drv.hold(float(cp.dwell_s) + HOLD_EXTRA_S)
        drv.step(events=[{"event": "dwell_end", "target": tg.id, "inside": inside}])
        if inside:
            drv.goto(*g.pos[tg.node])
        if drv.t >= plant.T - DT:
            break
    drv.run_out()
    plant.finish(str(run_dir), {"backend": "local_scripted_double", "path_length_m": drv.path_m})
    return {"run_dir": str(run_dir), "backend": "local_scripted_double", "trip_probe": probe,
            "path_m": round(drv.path_m, 2), "t_end": drv.t}


def mission_metric(run_dir: Path, planned: list[str], engine: str = "double") -> tuple[PrimaryMetric, dict]:
    """Steps 7-8: summary.json + the run log -> the primary-metric JSON (one mission). `planned` = the zone's
    mission_order (tasks.yaml). Outcome per checkpoint, in this precedence: tripped > held > passed > declined >
    failed (attempted, not passed) > unattempted."""
    s = Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    if s.checkpoints_planned != len(planned):
        raise SliceError(f"summary plans {s.checkpoints_planned} checkpoints, mission_order has {len(planned)}")
    rows = read_log(run_dir)
    by = {k: {r["checkpoint"] for r in rows if r["kind"] == k and r["checkpoint"]}
          for k in ("press_trip", "press_hold", "checkpoint_pass", "checkpoint_decline", "checkpoint_attempt")}
    outs = []
    for cp in planned:
        o = ("tripped" if cp in by["press_trip"] else "held" if cp in by["press_hold"]
             else "passed" if cp in by["checkpoint_pass"] else "declined" if cp in by["checkpoint_decline"]
             else "failed" if cp in by["checkpoint_attempt"] else "unattempted")
        outs.append(CheckpointOutcome(checkpoint=cp, outcome=o))  # type: ignore[arg-type]
    stop, hold, inc = s.robot_caused_press_stops > 0, s.holds > 0, s.incidents > 0
    err = s.status != "ok"
    y = 0 if (stop or hold or inc or err) else sum(o.outcome == "passed" for o in outs)
    m = MissionMetric(run_id=s.run_id, world=s.world, day=s.day, zone=s.zone, condition=s.condition, T_s=METRIC_T_S,
                      Y=y, robot_caused_press_stop=stop, hold=hold, incident=inc, error=err,
                      press_time_lost_s=s.press_time_lost_s, checkpoints=outs)
    pm = PrimaryMetric(contract="twin.primary_metric/1", engine=engine, split=s.world_split, missions=[m],  # type: ignore[arg-type]
                       per_world=world_values([m]))
    return pm, {"summary": s.model_dump(mode="json"), "n_log_rows": len(rows),
                "outcomes": {o.checkpoint: o.outcome for o in outs}, "Y": y}


def check_metric_schema(pm: PrimaryMetric, path: Path) -> dict:
    """The metric JSON round-trips through the model, and the exported JSON schema is the model's schema."""
    text = pm.model_dump_json(indent=1)
    path.write_text(text + "\n", encoding="utf-8")
    PrimaryMetric.model_validate_json(path.read_text(encoding="utf-8"))
    exported = json.loads((C.TWIN / "schemas" / "json" / "primary_metric.schema.json").read_text(encoding="utf-8"))
    if exported != json_schema("primary_metric"):
        raise SliceError("twin/schemas/json/primary_metric.schema.json is stale (python -m twin.schemas)")
    return {"file": str(path), "validated": "pydantic PrimaryMetric + exported JSON schema in sync"}


def check_stop_propagation(run_dir: Path, obs_path: Path, adapter: PL.ObsAdapter, thr: float) -> dict:
    """A robot-caused press stop (press_trip row; detail.t_abs0 = the freeze start ts) is in the realised stream from
    the first 2 Hz tick k0 >= ts. The camera measures motion between consecutive ticks, so the Observation can show it
    one tick later: the observation captured at k0 + 1 tick must report no motion for that press (finite zone_motion
    <= thr; MISSING does not count as showing the stop) and be delivered by k0 + 1 tick + the nominal latency
    (mapping.LATENCY_S). No observation captured inside the stop may report motion."""
    rows = [r for r in read_log(run_dir) if r["kind"] == "press_trip"]
    if not rows:
        raise SliceError("the mission has no robot-caused press stop to check")
    obs = OL.ObsV2Reader(obs_path, len(adapter.press_ids), len(adapter.zone_ids)).poll()
    out = []
    for r in rows:
        d = json.loads(r["detail"]) if isinstance(r["detail"], str) else dict(r["detail"] or {})
        ts = float(d["t_abs0"])
        stop_end = ts + float(d["tripped_s"]) + float(d["reset_s"])
        k0 = math.ceil(ts / TICK_S - 1e-6) * TICK_S
        i = adapter.press_ids.index(r["entity"])
        first = next((o for o in obs if abs(o.t_cap - (k0 + TICK_S)) < 1e-6), None)
        if first is None:
            raise SliceError(f"no observation captured at {k0 + TICK_S} (stop of {r['entity']} at {ts})")
        zm = float(np.asarray(first.zone_motion, float)[i])
        bound = k0 + TICK_S + OM.LATENCY_S
        during = [float(np.asarray(o.zone_motion, float)[i]) for o in obs if k0 + TICK_S - 1e-6 <= o.t_cap < stop_end]
        ok = math.isfinite(zm) and zm <= thr and first.t <= bound + 1e-6 and all(
            (not math.isfinite(v)) or v <= thr for v in during)
        out.append({"press": r["entity"], "t_abs0": ts, "first_frozen_tick": k0, "obs_t_cap": first.t_cap,
                    "obs_t": first.t, "bound_t": bound, "zone_motion": zm, "delay_s": round(first.t - ts, 3),
                    "ticks_in_stop": len(during), "missing_in_stop": int(sum(not math.isfinite(v) for v in during)),
                    "max_motion_in_stop": max((v for v in during if math.isfinite(v)), default=None), "ok": bool(ok)})
    if not all(x["ok"] for x in out):
        raise SliceError(f"press stop not visible in the Observation within 1 tick + latency: {out}")
    return {"stops": out}


def run_guard_tests() -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *GUARD_TESTS], cwd=REPO,
                       env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    if r.returncode != 0:
        raise SliceError(f"guard tests failed: {tail[0]}")
    return {"tests": list(GUARD_TESTS), "result": tail[0]}


# ------------------------------------------------------------------------------------------------ Modal (step 6)

def modal_cfg(tag: str, paths: dict[str, Path], script_name: str, cz: C.CompiledZone, t0: float, zone: str,
              world: int, planner: str, max_rtf: float) -> dict:
    """The episode cfg of the Modal fake-backend mission (intercycle_sim.episode.run_episode). The plant runs in the
    container from the mounted repo (twin/ under REMOTE_REPO) and the day script shipped with the world files."""
    rel = {k: Path(p).resolve().relative_to(REPO).as_posix() for k, p in paths.items()}
    wdir = f"/tmp/ic_{tag}/world"                      # episode.run_episode writes the world files here
    pc = dict(site=f"{REMOTE_REPO}/{rel['site']}", safety=f"{REMOTE_REPO}/{rel['safety']}",
              tasks=f"{REMOTE_REPO}/{rel['tasks']}", dynamics=f"{REMOTE_REPO}/{rel['dynamics']}",
              day_script=f"{wdir}/{script_name}", run_id=tag, zone=zone, world=world, world_split="dev",
              condition="learned", code={"head": git_head()}, plant_t0=t0, allow_short_mission=True)
    dock = cz.graph.pos["DOCK"]
    start = [float(dock[0]), float(dock[1]), 0.0]
    cfg: dict = {"tag": tag, "twin": True, "sim": "fake", "mode": "nav", "nav2": True, "seconds": MISSION_S,
                 "timeout_s": MISSION_S, "max_rtf": max_rtf, "start": start, "goal": start, "scenario": "scenario.yaml",
                 "mover_source": "twin.plant:make", "plant": pc,
                 "nav2_cmd": "ros2 launch /root/ros2/intercycle_nav/launch/nav.launch.py map:=$IC_WORLD/map.yaml"}
    from intercycle_nav.masks import scenario_ros_arg  # pyright: ignore[reportMissingImports]

    arg = scenario_ros_arg(cfg)
    if planner == "timing":
        sock = "unix://$IC_WORK/observer.sock"
        wait_vk = "for i in $(seq 1200); do [ -s $IC_WORK/observer_vk.txt ] && break; sleep 0.1; done; "
        keyring = " -p keyring:=\\\"$(cat $IC_WORK/observer_vk.txt)\\\""
        cfg["extra_launch"] = [f"{wait_vk}exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"
                               f" -p condition:=learned -p feed:={sock} -p plant_t0:={t0}{keyring}{arg}"]
        cfg["goal_cmd"] = (f"python3 -u -m intercycle_nav.timing_planner --ros-args -p use_sim_time:=true"
                           f" -p condition:=learned -p plant_t0:={t0} -p init:=[{start[0]},{start[1]},{start[2]}]"
                           f" -p mission_timeout_s:=100000.0 -p max_wait_s:=120.0 -p out:=$IC_WORK/goal.json{arg}")
        from intercycle_eval.broadcaster import check_robot_scenario  # pyright: ignore[reportMissingImports]

        check_robot_scenario(cfg)
    else:
        nodes, here = [], "DOCK"
        for tg in cz.graph.mission():
            for n in shortest(cz.graph.adj, here, tg.node)[1:]:
                nodes.append(n)
            here = tg.node
        for n in shortest(cz.graph.adj, here, "DOCK")[1:]:
            nodes.append(n)
        cfg["goal"] = [[float(cz.graph.pos[n][0]), float(cz.graph.pos[n][1]), 0.0] for n in nodes]
        cfg["timeout_s"] = 120.0
        cfg["patrol_until_s"] = MISSION_S + 60.0      # past the sim limit: the sim, not the goal client, ends the mission
        cfg["scenario"] = ""                           # send_goal.py takes no scenario parameter (body-shop free)
    return cfg


def world_bytes(zone_dir: Path, script_path: Path) -> dict[str, bytes]:
    files = {f: (zone_dir / f).read_bytes() for f in BW.FILES if (zone_dir / f).exists()}
    files["factory_l0.usda"] = files["static.usda"]   # the name the Isaac backend opens (isaac_backend.py)
    files[script_path.name] = script_path.read_bytes()
    return files


def run_modal(out: Path, cfg: dict, world: dict[str, bytes], cpu: float) -> dict:
    import modal_isaac_robot as R  # pyright: ignore[reportMissingImports]

    t0 = time.time()
    with R.app.run():
        res = R.fake_episode_cpu(cpu).remote(cfg, world)
    res["client_wall_s"] = round(time.time() - t0, 1)
    (out / f"{cfg['tag']}.episode.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    dest = R.pull(cfg["tag"])
    return {"tag": cfg["tag"], "pulled": str(dest), "error": res.get("error"), "phases": res.get("phases"),
            "client_wall_s": res["client_wall_s"], "goal": (res.get("goal") or {}).get("result"),
            "plant_status": (res.get("plant") or {}).get("status"), "log_tail": (res.get("log_tail") or "")[-3000:]}


# ------------------------------------------------------------------------------------------------ main

def run(out: Path, *, world: int = 0, day: int = 9, zone: str = "z14", modal: str = "", cpu: float = 4.0,
        max_rtf: float = 2.0, guard_tests: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "head": git_head(), "world": world, "day": day,
                 "zone": zone, "steps": {}, "assertions": {}}
    ctx: dict = {}

    def do(name: str, fn: Callable[[], Any], kind: str = "steps") -> Any:
        t = time.time()
        try:
            res = fn()
            rec = res[0] if isinstance(res, tuple) else res
            rep[kind][name] = {"ok": True, "wall_s": round(time.time() - t, 2), **(rec if isinstance(rec, dict) else {})}
            return res
        except Exception as e:  # noqa: BLE001 - every failure is reported, then the slice fails
            rep[kind][name] = {"ok": False, "wall_s": round(time.time() - t, 2), "error": repr(e)[:2000]}
            return None

    paths = inputs()
    r = do("1_build", lambda: step_build(out, paths))
    if r:
        ctx["compiled"] = r[1]
    r = do("2_day", lambda: step_day(out, paths, world, day))
    if r:
        ctx["script_path"], ctx["script"] = r[1], r[2]
    if "script" in ctx and "compiled" in ctx:
        t0 = float(ctx["script"].t0_s)
        r = do("3_4_engine_adapter", lambda: run_learn(out, paths, ctx["script_path"], t0, zone, world))
        if r:
            ctx["learn_obs"] = r[1]
            site = load(Site, paths["site"])
            ctx["adapter"] = PL.ObsAdapter(site)
            spec = observer_spec(site, ctx["compiled"], ctx["adapter"])
            r = do("5_observer", lambda: step_observer(out, spec, ctx["learn_obs"], t0 + LEARN_S))
            if r:
                ctx["observer"] = r[1]
                do("signatures", lambda: step_signatures(ctx["observer"], t0 + LEARN_S), "assertions")
        cz = ctx["compiled"][zone]
        planned = [t.id for t in cz.graph.mission()]
        r = do("6_mission_local", lambda: step_mission_local(out, paths, ctx["script_path"], cz, t0 + LEARN_S, zone, world))
        if r:
            rd = Path(r["run_dir"])
            m = do("7_8_log_metric", lambda: mission_metric(rd, planned))
            if m:
                pm, info = m
                rep["steps"]["7_8_log_metric"].update(info)
                do("metric_schema", lambda: check_metric_schema(pm, out / "primary_metric.json"), "assertions")
                do("inspection_attempt", lambda: _attempts(info), "assertions")
            if "adapter" in ctx:
                do("stop_in_observation", lambda: check_stop_propagation(
                    rd, rd / OL.OBS_FILE, ctx["adapter"], OL.ObserverLiveConfig.motion_thr), "assertions")
        if modal:
            tag = f"slice_{modal}_{time.strftime('%m%d_%H%M%S')}"
            cfg = modal_cfg(tag, paths, ctx["script_path"].name, cz, t0 + LEARN_S, zone, world, modal, max_rtf)
            wb = world_bytes(Path(out / "worlds" / zone), ctx["script_path"])
            mr = do(f"6_mission_modal_{modal}", lambda: run_modal(out, cfg, wb, cpu))
            if mr:
                rd = Path(mr["pulled"])
                if (rd / "summary.json").exists():
                    m = do("7_8_log_metric_modal", lambda: mission_metric(rd, planned))
                    if m:
                        rep["steps"]["7_8_log_metric_modal"].update(m[1])
                        do("metric_schema_modal", lambda: check_metric_schema(m[0], out / "primary_metric_modal.json"),
                           "assertions")
                        do("inspection_attempt_modal", lambda: _attempts(m[1]), "assertions")
                else:
                    rep["steps"][f"6_mission_modal_{modal}"].update(ok=False, error="no summary.json on the volume")
    if guard_tests:
        do("guard_tests", run_guard_tests, "assertions")
    rep["green"] = all(v["ok"] for k in ("steps", "assertions") for v in rep[k].values())
    (out / "slice_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep


def _attempts(info: dict) -> dict:
    n = int(info["summary"]["attempted"])
    if n < 1:
        raise SliceError("no inspection attempt in the mission")
    return {"attempted": n, "passed": info["summary"]["passed"], "planned": info["summary"]["checkpoints_planned"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--day", type=int, default=9)
    ap.add_argument("--zone", default="z14")
    ap.add_argument("--modal", default="", choices=("", "timing", "patrol"))
    ap.add_argument("--cpu", type=float, default=4.0)
    ap.add_argument("--max-rtf", type=float, default=2.0)
    ap.add_argument("--skip-guard-tests", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else REPO / "out" / "slice" / time.strftime("%m%d_%H%M%S")
    rep = run(out, world=a.world, day=a.day, zone=a.zone, modal=a.modal, cpu=a.cpu, max_rtf=a.max_rtf,
              guard_tests=not a.skip_guard_tests)
    for k in ("steps", "assertions"):
        for name, v in rep[k].items():
            print(f"{'OK  ' if v['ok'] else 'FAIL'} {k[:-1]:<9} {name:<26} {v['wall_s']:>7}s {(v.get('error') or '')[:300]}")
    print(f"slice {'GREEN' if rep['green'] else 'RED'} -> {out / 'slice_report.json'}")
    return 0 if rep["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
