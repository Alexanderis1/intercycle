"""The sealed / dev CPU-double batch launcher (PLAN_TWIN §5 "Design"; USER DECISION Sun ~07:00 CEST, the CUT design).

    python scripts/run_double.py --dev  --out out/double/<tag> [--worlds 0] [--backend local|modal] [--zones z9]
    (the sealed batch runs ONLY through `scripts/sealed_run.py run --stages double`, after the robot FREEZE)

Per world (intercycle.eval.twin_batch for the plan, the order, re-runs, layout and cost):
  1. day scripts: days 1-10 of the world from the generator in force (twin/daygen.py v1; the G0 stub generator
     twin/daygen_stub.py when the slice runs on the whole v0 stub set), written as plant-only files.
  2. observer phase (`observer_phase`): the robot-free plant stream (twin/engine.py -> the plant's camera model ->
     the plant's ObsAdapter, exactly the plant's per-tick path) of days 1-8 and of days 9-10 up to each mission start
     feeds ONE twin/observer_live.py observer; at each mission start its state is pickled (snapshot). The learned arm's
     broadcaster (twin/broadcast_state.py) starts from that snapshot and then tails the mission's live obs_v2.jsonl, so
     its layers come ONLY from days 1-8 plus live observations (DISCLOSED: the stream of a day 9 mission's 720 s is
     the robot-free stream in the observer phase, the same for all three conditions, so day 10's state stays paired).
  3. missions: twin_batch.rounds (the reactive chain per world in time order), each round in parallel; after a round,
     scripts/classify_invalid.py labels its batch dir from the logs alone and every (world, day, zone) with an
     infrastructure-invalid run is re-run with ALL conditions (twin_batch.reruns) before the next round.
  4. <out>/runs/ is the twin_eval runs root (twin_batch module doc); <out>/costs.json records wall and $ per container.

Backends: `local` = no Modal: the observer phase in this process and each mission as the plant driven by a scripted
wiring double (slice.Driver) in this process (a wiring double, not the policies: for the stub-world end-to-end test);
`modal` = the real policies (section "Modal backend"): the day scripts are generated HERE (the generator and
$INTERCYCLE_DATA stay local) and shipped as plant-only files; the observer phase runs in one Modal CPU container per
world (cloud/modal_isaac_robot.double_observer_phase, all worlds in parallel) and writes one pickled snapshot per
(day, zone) slot to the volume; each mission runs in its own Modal CPU container (double_episode: the Modal slice's
fake backend + Nav2 stack, cpu 4, max_rtf 2, the full T = 720 s); the learned arm's broadcaster starts from the
snapshot (twin/broadcast_live.py --snapshot-t0) and tails the live obs_v2.jsonl; the reactive arm loads the world's
previous chain step's saved state from the volume (episodes/<tag>/reactive_state.json, the scored attempt's), and a
re-run of a cell starts from the SAME input state; every pulled reactive state is checked with reactive.verify_chain
(<batch>/<tag>.reactive_chain.json). The z9 / z14 slots of a world-day never overlap (twin_batch.day_slots), because
the reactive state is ONE per world across zones (PLAN_TWIN §5 "Baseline fairness": "persisted within the world").

Modes: --dev runs dev worlds 0-3 only (dev seeds); the sealed batch (`run_sealed`) is reachable only from sealed_run.py
after every refusal passed (INTERCYCLE_SEALED_RUN=1 and the robot FREEZE checked there); this CLI refuses --sealed.

This file is an ORCHESTRATOR (twin/truth_guard.py): it imports the generators, the engine and the plant to start them
and never opens a truth file itself (the day scripts are written by the generators and read by the plant).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "ros2" / "intercycle_sim", REPO / "ros2" / "intercycle_nav",
           REPO / "ros2" / "intercycle_eval", REPO / "cloud"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from intercycle.eval import twin_batch as TB  # noqa: E402
from intercycle.eval.twin_design import SEALED_CONDITIONS  # noqa: E402
from twin import compile as C  # noqa: E402
from twin import observer_live as OL  # noqa: E402
from twin import plant as PL  # noqa: E402
from twin.engine import Engine  # noqa: E402
from twin.schemas import load  # noqa: E402
from twin.schemas.common import MISSION_T_S  # noqa: E402
from twin.schemas.dayscript import FILE_SUFFIX  # noqa: E402
from twin.schemas.dynamics import Dynamics  # noqa: E402
from twin.schemas.site import Site  # noqa: E402

import slice as SL  # noqa: E402  (scripts/slice.py: inputs, observer spec, margin, Driver, Modal cfg pieces)

SEALED_ENV = "INTERCYCLE_SEALED_RUN"
SEALED_BACKEND = "modal"                # the sealed batch runs the real policies on Modal CPU, never the wiring double
SEALED_BACKEND_READY = True             # the Modal backend is wired (dev smoke on the stub world: see the lane report)
SNAPSHOT = "observer_snapshot.pkl"      # the pickled observer state at a mission start (one per (world, day, zone))
SCORED_DAYS = (9, 10)


class DoubleError(RuntimeError):
    """The batch cannot go on (fail closed)."""


# ------------------------------------------------------------------------------------------------ inputs


def setup(out: Path, *, stubs: bool = False) -> dict:
    """Contract files in force, compiled zones, the site-wide observer spec and the frozen M-W7 margin. `stubs` = the
    whole G0 v0 stub set (slice.inputs(stubs=True); the stub-world end-to-end test only, never a sealed run)."""
    if stubs and os.environ.get(SEALED_ENV) == "1":
        raise DoubleError("the G0 stub inputs in a sealed run: refused")
    paths = SL.inputs(stubs=stubs)
    stub = all(C.is_stub(k, paths[k]) for k in ("site", "dynamics", "tasks"))
    if stubs and not stub:
        raise DoubleError(f"stubs requested but the inputs are not the whole G0 stub set: {paths}")
    site, dyn = load(Site, paths["site"]), load(Dynamics, paths["dynamics"])
    compiled = C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])
    # spec only: observer_spec reads the adapter's geometry (press_ids, zone_ids, rects), never its hold; this adapter
    # adapts no observation (each day's observations go through the world's own camera hold, run_world below); its
    # occlusion rule is never applied, so the geometry-only "floor" choice (no camera roi_vis here) changes nothing
    adapter = PL.ObsAdapter(site, hold_s=0.0, occlusion="floor")
    spec = SL.observer_spec(site, compiled, adapter)
    out.mkdir(parents=True, exist_ok=True)
    margin, msrc = SL.margin_path(out)
    zones = [z for z in compiled]
    return {"paths": paths, "stub": stub, "site": site, "dyn": dyn, "compiled": compiled, "spec": spec,
            "margin": margin, "margin_source": msrc, "zones": zones}


def generate_day(ctx: dict, seed: int, day: int, day_s: float):
    if ctx["stub"]:
        from twin import daygen_stub as DGS
        return DGS.generate_day(seed, day, site=ctx["site"], dynamics=ctx["dyn"], duration_s=day_s)
    from twin import daygen as DG
    if "dg_inputs" not in ctx:
        ctx["dg_inputs"] = DG.load_inputs(ctx["site"], ctx["dyn"])
    return DG.generate_day(seed, day, inputs=ctx["dg_inputs"], duration_s=day_s)


def write_days(ctx: dict, world: TB.World, days, day_s: float, out: Path) -> tuple[dict[int, Path], dict[int, Any]]:
    """The world's day scripts (plant-only files), written by the generator's own write_day; returns (paths, the
    generated script objects): the observer phase consumes the SAME objects the plant files were written from."""
    from twin import daygen_stub as DGS
    from twin import daygen as DG
    w = DGS.write_day if ctx["stub"] else DG.write_day
    objs = {d: generate_day(ctx, world.seed, d, day_s) for d in days}
    return {d: w(objs[d], out / f"w{world.index}_d{d}{FILE_SUFFIX}") for d in days}, objs


# ------------------------------------------------------------------------------------------------ observer phase


def _camera(ctx: dict, seed: int, day: int):
    """The camera the mission plant runs on these inputs (twin/plant.py Plant: StubCameraV0 on the G0 stub site file,
    stat_v1 otherwise), so the observer phase sees the same observation stream as the missions."""
    return PL.make_camera(ctx["site"], seed, day, stub_site=Path(ctx["paths"]["site"]).name == PL.STUB_SITE_FILE)


def observer_phase(ctx: dict, world: TB.World, slots: list[tuple[int, str, float]], day_s: float = TB.DAY_S,
                   scripts: dict[int, Any] | None = None) -> tuple[dict[tuple[int, str], bytes], dict]:
    """Days 1..max slot day of the robot-free plant stream through one observer; snapshot (pickle) at each slot
    (day, zone, t0): the state after exactly the observations the plant would have written to obs_v2.jsonl by t0.
    The plant's _deliver is a FIFO in CAPTURE order (popleft while the head's delivery time obs.t = t_cap + latency
    <= T), so what is out by t0 is the longest capture-order prefix whose every obs.t <= t0; the loop below takes that
    prefix (it snapshots at the first capture-order observation with obs.t > t0). Capture order is also what
    observer_live.ingest requires (t_cap strictly increasing). DISCLOSED: observations captured before t0 and not
    out by t0 are in no snapshot and in no mission stream (the mission plant writes from t0): a one-latency gap at
    the mission start, the same for all three conditions. `scripts` = the generated day objects the plant files were
    written from (write_days); a day missing there is generated here. Returns (snapshots, stats)."""
    t_wall = time.time()
    # the observer config follows the camera model that writes the observations (observer_live.config_for_camera, as
    # the slice and the twin broadcaster do): stat_v1's fitted motion threshold, never the 0.5 placeholder (fail closed)
    model = _camera(ctx, world.seed, 1).provenance()["model"]
    ob = OL.ObserverLive(ctx["spec"], OL.config_for_camera(model, str(ctx["margin"]), stub_run=ctx["stub"]))
    pending = sorted(slots, key=lambda s: s[2])
    last_day = max(d for d, _, _ in pending)
    snaps: dict[tuple[int, str], bytes] = {}
    n = 0
    per_day = {}
    for day in range(1, last_day + 1):
        t_day = time.time()
        script = scripts[day] if scripts and day in scripts else generate_day(ctx, world.seed, day, day_s)
        eng = Engine(script, ctx["site"], ctx["dyn"])
        cam = _camera(ctx, world.seed, day)
        # as the plant does (twin/plant.py): the camera's own per-press hold, ROI visibility (stat_v1, S3b fix2) and
        # occlusion rule (stat_v1 "roi", the real clips' image-space rule, AMENDMENTS A4; the G0 stand-in "floor")
        ad = PL.ObsAdapter(ctx["site"], hold_s=dict(cam.hold_s_by_press), roi_vis=cam.roi_vis,
                           occlusion=cam.occlusion_rule)
        for t in eng.tick_times():
            obs = ad(cam.observe(eng.tick(float(t))))
            while pending and obs.t > pending[0][2] + 1e-9:
                d, z, t0 = pending.pop(0)
                snaps[(d, z)] = pickle.dumps(ob, protocol=pickle.HIGHEST_PROTOCOL)
            ob.ingest(obs)
            n += 1
        per_day[day] = round(time.time() - t_day, 1)
    while pending:                                         # a slot at the very end of the last day
        d, z, _ = pending.pop(0)
        snaps[(d, z)] = pickle.dumps(ob, protocol=pickle.HIGHEST_PROTOCOL)
    return snaps, {"world": world.index, "ticks": n, "wall_s": round(time.time() - t_wall, 1), "per_day_wall_s": per_day,
                   "snapshot_bytes": {f"d{d}_{z}": len(b) for (d, z), b in snaps.items()}}


# ------------------------------------------------------------------------------------------------ missions


def plant_cfg(ctx: dict, m: TB.Mission, script: Path | PurePosixPath, run_dir: Path, *, remote: bool = False) -> dict:
    paths = ctx["paths"]
    root = SL.REMOTE_REPO if remote else None

    def p(k: str) -> str:
        return f"{root}/{Path(paths[k]).resolve().relative_to(REPO).as_posix()}" if root else str(paths[k])

    pc = dict(site=p("site"), safety=p("safety"), tasks=p("tasks"), dynamics=p("dynamics"), day_script=str(script),
              run_id=m.run_id, zone=m.zone, world=m.world.index, world_split=m.world.split, condition=m.condition,
              code={"head": SL.git_head()}, plant_t0=m.t0)
    if not remote:
        pc["run_dir"] = str(run_dir)
    return pc


def local_mission(ctx: dict, m: TB.Mission, script: Path, batch: Path, snapshot: bytes | None) -> dict:
    """The local backend: the plant over the full 720 s mission driven by the scripted wiring double (slice.Driver,
    mission order, no entry into a press zone); the learned arm's snapshot is unpickled and checked to predate t0."""
    t_wall = time.time()
    run_dir = batch / m.run_id
    rec: dict = {"tag": m.run_id, "cfg": {"twin": True, "tag": m.run_id}, "backend": "local_wiring_double"}
    if m.condition == "learned":
        ob = pickle.loads(snapshot) if snapshot else None
        if not isinstance(ob, OL.ObserverLive) or ob.t_last is None or ob.t_last >= m.t0:
            raise DoubleError(f"{m.run_id}: no observer snapshot from before the mission start")
        rec["observer"] = {"t_last": ob.t_last, "ticks": ob.n_ticks}
    plant = PL.make({"seconds": MISSION_T_S, "sim": "fake", "plant": plant_cfg(ctx, m, script, run_dir)})
    cz = ctx["compiled"][m.zone]
    drv = SL.Driver(plant)
    drv.x, drv.y = cz.graph.pos["DOCK"]
    drv.step()
    rec["goal"] = {"t_goal_sent": drv.t, "result": "wiring"}
    here = "DOCK"
    for tg in cz.graph.mission():
        if tg.kind == "machine":
            continue                                        # the wiring double never enters a press zone
        for node in SL.shortest(cz.graph.adj, here, tg.node)[1:]:
            drv.goto(*cz.graph.pos[node])
        here = tg.node
        cp = next(c for c in plant.tasks.checkpoints if c.id == tg.id) if plant.tasks else None
        if cp is not None:
            drv.face(cp.target_xzh[0], cp.target_xzh[1])
            drv.step(events=[{"event": "dwell_start", "target": tg.id, "inside": False}])
            drv.hold(float(cp.dwell_s) + SL.HOLD_EXTRA_S)
            drv.step(events=[{"event": "dwell_end", "target": tg.id, "inside": False}])
    drv.run_out()
    plant.finish(str(run_dir), {"backend": "local_wiring_double", "path_length_m": drv.path_m})
    wall = round(time.time() - t_wall, 2)
    (run_dir / "episode.json").write_text(json.dumps({"tag": m.run_id, "backend": rec["backend"]}), encoding="utf-8")
    (run_dir / "sim.log").write_text("LEG local wiring double\n", encoding="utf-8")
    rec.update(phases={"total_wall_s": wall}, sim={"backend": "local", "rtf": None}, total_wall_s=wall)
    TB.write_json(batch / f"{m.run_id}.json", rec)
    return {"run_id": m.run_id, "wall_s": wall, "container_usd": 0.0}


# ------------------------------------------------------------------------------------------------ Modal backend
# The real policies on Modal CPU (cloud/modal_isaac_robot.py double_episode: the fake kinematic backend + Nav2, the same
# stack as the Modal slice, cpu 4, max_rtf 2), one container per mission; the observer phase in one Modal CPU container
# per world (double_observer_phase), in the SAME image, so its pickled snapshots unpickle in the broadcaster.
# Per condition (twin_design.SEALED_CONDITIONS), the robot side is exactly the slice's stack with:
#   learned   twin/broadcast_live.py --snapshot-t0 t0 (starts from the world's observer snapshot at this mission's start,
#             then tails the mission's live obs_v2.jsonl) -> observer_bridge (condition learned, episode keyring) ->
#             timing_planner (condition learned)
#   static    observer_bridge condition static (no feed) -> timing_planner condition static (never enters)
#   reactive  observer_bridge condition reactive (own camera; reactive_config = the frozen default ReactiveConfig, no
#             tuning; world_id; mission_tag; reactive_state = the world's previous chain step's saved state, empty on
#             the world's first step) -> timing_planner condition reactive
SNAPSHOT_VOL_DIR = "double"                 # <VOL>/double/<batch id>/w<W>/snap_d<D>_<zone>.pkl
REACTIVE_IN = "reactive_state_in.json"      # the reactive arm's input state in its world dir
REACTIVE_OUT = "reactive_state.json"        # observer_bridge's default reactive_state_out ($IC_OUT/reactive_state.json)
REACTIVE_CFG = "reactive_config.json"
PULL_WORKERS = 8                            # parallel `modal volume get` of a round's episode dirs


def _remote(ctx: dict, k: str) -> str:
    return f"{SL.REMOTE_REPO}/{Path(ctx['paths'][k]).resolve().relative_to(REPO).as_posix()}"


def remote_observer_phase(job: dict, plant_dir: Path, vol_root: Path) -> dict:
    """In the observer-phase container (cloud/modal_isaac_robot.double_observer_phase): the day scripts written by the
    LOCAL generator (shipped as plant-only files; the engine opens them through the truth guard) -> observer_phase ->
    one pickled snapshot per slot on the volume. Returns the stats with the snapshots' volume-relative paths."""
    from twin.engine import load_day_script

    site, dyn = load(Site, REPO / job["site"]), load(Dynamics, REPO / job["dynamics"])
    margin = Path("/tmp") / f"{job['tag']}_m7_margin.json"
    margin.write_bytes(job["margin"])
    ctx = {"spec": job["spec"], "margin": margin, "site": site, "dyn": dyn, "stub": bool(job["stub"])}
    scripts = {int(d): load_day_script(plant_dir / name) for d, name in job["days"].items()}
    world = TB.World(**job["world"])
    snaps, st = observer_phase(ctx, world, [(int(d), str(z), float(t)) for d, z, t in job["slots"]], float(job["day_s"]),
                               scripts=scripts)
    rels = {}
    for (d, z), b in snaps.items():
        rel = f"{job['snap_dir']}/snap_d{d}_{z}.pkl"
        p = vol_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        rels[f"d{d}_{z}"] = rel
    st["snapshots"] = rels
    return st


def modal_cfg(ctx: dict, m: TB.Mission, script_name: str, *, max_rtf: float = TB.MAX_RTF,
              reactive_in: bool = False) -> dict:
    """The episode cfg of one batch mission on Modal (the slice's Modal cfg with the full T and the mission's real
    policy; module doc of this section)."""
    from intercycle_eval.broadcaster import check_robot_scenario  # pyright: ignore[reportMissingImports]
    from intercycle_nav.masks import scenario_ros_arg  # pyright: ignore[reportMissingImports]

    if m.condition not in SEALED_CONDITIONS:
        raise DoubleError(f"{m.run_id}: condition {m.condition!r} is not in the cut design {SEALED_CONDITIONS}")
    cz = ctx["compiled"][m.zone]
    pc = plant_cfg(ctx, m, PurePosixPath(SL.REMOTE_PLANT_DIR, m.run_id, script_name), Path("."), remote=True)
    dock = cz.graph.pos["DOCK"]
    start = [float(dock[0]), float(dock[1]), 0.0]
    cfg: dict = {"tag": m.run_id, "twin": True, "sim": "fake", "mode": "nav", "nav2": True, "seconds": MISSION_T_S,
                 "timeout_s": MISSION_T_S, "max_rtf": max_rtf, "start": start, "goal": start,
                 "scenario": "scenario.yaml", "mover_source": "twin.plant:make", "plant": pc,
                 "double": m.record(),
                 "nav2_cmd": "ros2 launch /root/ros2/intercycle_nav/launch/nav.launch.py map:=$IC_WORLD/map.yaml"}
    arg = scenario_ros_arg(cfg)
    x0, z0, x1, z1 = (float(v) for v in cz.extent)
    grid = f" -p mask_origin:=[{x0},{z0}] -p mask_size:=[{x1 - x0},{z1 - z0}]"
    c, t0 = m.condition, m.t0
    extra = []
    if c == "learned":
        extra.append("PYTHONPATH=$IC_SIM_PYTHONPATH exec $IC_SIM_PY -u -m twin.broadcast_live --work $IC_WORK"
                     f" --obs $IC_OUT/obs_v2.jsonl --world $IC_WORLD --snapshot-t0 {t0}")
        wait_vk = "for i in $(seq 1200); do [ -s $IC_WORK/observer_vk.txt ] && break; sleep 0.1; done; "
        bridge = (f"{wait_vk}exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"
                  f" -p condition:=learned -p feed:=unix://$IC_WORK/observer.sock -p plant_t0:={t0}"
                  f" -p keyring:=\\\"$(cat $IC_WORK/observer_vk.txt)\\\"{grid}{arg}")
    elif c == "static":
        bridge = (f"exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"
                  f" -p condition:=static -p feed:=none -p plant_t0:={t0}{grid}{arg}")
    else:
        rin = f" -p reactive_state:=$IC_WORLD/{REACTIVE_IN}" if reactive_in else ""
        bridge = (f"exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"
                  f" -p condition:=reactive -p feed:=none -p plant_t0:={t0}"
                  f" -p \"reactive_config:='$(cat $IC_WORLD/{REACTIVE_CFG})'\""
                  f" -p world_id:={reactive_world_id(m.world)} -p mission_tag:={m.run_id}{rin}{grid}{arg}")
    extra.append(bridge)
    cfg["extra_launch"] = extra
    cfg["goal_cmd"] = (f"python3 -u -m intercycle_nav.timing_planner --ros-args -p use_sim_time:=true"
                       f" -p condition:={c} -p plant_t0:={t0} -p init:=[{start[0]},{start[1]},{start[2]}]"
                       f" -p mission_timeout_s:=100000.0 -p max_wait_s:=120.0"
                       f" -p out:=$IC_WORK/goal.json{arg} ; exec sleep {SL.HOLD_GOAL_S}")
    check_robot_scenario(cfg)
    return cfg


def reactive_world_id(w: TB.World) -> str:
    """The reactive arm's world id (its state never crosses worlds): split + design index, never the seed."""
    return f"{w.split}{w.index}"


def zone_world_bytes(ctx: dict, out: Path, zone: str) -> dict[str, bytes]:
    """The world dir of a mission in `zone` (no truth): the built zone files + the observer spec record + the frozen
    margin + the frozen default reactive config. The learned snapshot and the reactive input state come from the
    volume (double_episode vol_world)."""
    from intercycle_nav import reactive as RX  # pyright: ignore[reportMissingImports]
    from twin import broadcast_live as BL

    zd = out / "worlds" / zone
    files = {f: (zd / f).read_bytes() for f in SL.BW.FILES if (zd / f).exists()}
    files["factory_l0.usda"] = files["static.usda"]
    cz = ctx["compiled"][zone]
    # the spec record names the camera model the mission plant runs (broadcast_live refuses a record without one);
    # it is the model the observer phase's config follows (_camera: stat_v1, or StubCameraV0 on the G0 stub site)
    model = _camera(ctx, 0, 1).provenance()["model"]
    files[BL.SPEC_FILE] = json.dumps(BL.spec_record(ctx["spec"], zone, [mm.id for mm in cz.scenario.machines],
                                                    model)).encode()
    files[BL.MARGIN_FILE] = Path(ctx["margin"]).read_bytes()
    files[REACTIVE_CFG] = json.dumps(json.loads(RX.ReactiveConfig().to_json()), separators=(",", ":")).encode()
    return files


def pull_episode(tag: str, dest: Path) -> Path | None:
    """modal volume get episodes/<tag> -> <dest>/<tag>/ (twin_eval's layout). None if the volume has no such dir."""
    import subprocess

    r = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "intercycle-robot",
                        f"episodes/{tag}", str(dest)], capture_output=True, text=True)
    p = dest / tag
    return p if r.returncode == 0 and p.is_dir() else None


class ModalRunner:
    """The Modal side of run_batch (one app.run() context for the whole batch)."""

    def __init__(self, ctx: dict, out: Path, batch_id: str, cpu: float = TB.CPU, max_rtf: float = TB.MAX_RTF):
        import modal_isaac_robot as R  # pyright: ignore[reportMissingImports]

        self.R, self.ctx, self.out, self.batch_id, self.cpu, self.max_rtf = R, ctx, out, batch_id, cpu, max_rtf
        self.worlds = {z: zone_world_bytes(ctx, out, z) for z in ctx["zones"]}

    def observer_phases(self, jobs: list[tuple[dict, dict[str, bytes]]]) -> list[dict]:
        fn = self.R.double_observer_phase
        if self.cpu != 4.0:
            fn = fn.with_options(cpu=self.cpu)
        calls = [fn.spawn(j, f) for j, f in jobs]
        out = []
        for (j, _), c in zip(jobs, calls):
            try:
                out.append(c.get())
            except Exception as e:  # noqa: BLE001 - an observer phase that fails stops the batch (fail closed)
                raise DoubleError(f"observer phase {j['tag']} failed on Modal: {e!r}") from e
        return out

    def missions(self, items: list[tuple[TB.Mission, dict, dict[str, bytes], dict[str, str]]],
                 batch: Path) -> list[dict]:
        fn = self.R.double_episode
        if self.cpu != 4.0:
            fn = fn.with_options(cpu=self.cpu)
        t0 = time.time()
        calls = [(m, cfg, fn.spawn(cfg, self.worlds[m.zone], pf, vw)) for m, cfg, pf, vw in items]
        recs = []
        for m, cfg, call in calls:
            res: dict
            try:
                res = dict(call.get())
            except Exception as e:  # noqa: BLE001 - a provider failure is recorded; classify_invalid labels it
                res = {"tag": m.run_id, "error": f"modal: {e!r}"}
            res.setdefault("tag", m.run_id)
            res["cfg"] = cfg
            res["client_wall_s"] = round(time.time() - t0, 1)
            recs.append((m, res))
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=PULL_WORKERS) as ex:     # the pulls are I/O: in parallel, not per mission
            pulled = list(ex.map(lambda mr: pull_episode(mr[0].run_id, batch) if "phases" in mr[1] else None, recs))
        out = []
        for (m, res), p in zip(recs, pulled):
            res["pulled"] = str(p) if p else None
            TB.write_json(batch / f"{m.run_id}.json", res)
            wall = float((res.get("phases") or {}).get("total_wall_s") or res.get("total_wall_s") or 0.0)
            out.append({"run_id": m.run_id, "wall_s": wall, "container_usd": TB.container_usd(wall, self.cpu),
                        "error": res.get("error"), "pulled": res["pulled"]})
        return out


def modal_batch(ctx: dict, out: Path, worlds: list[TB.World], missions: list[TB.Mission], rnds, scripts, objs_days,
                classify: Callable[[Path], dict], costs: list[dict], phase_stats: list[dict]) -> tuple[list, list]:
    """Observer phases (one container per world, all in parallel), then the rounds (module doc step 3) on Modal."""
    import modal_isaac_robot as R  # pyright: ignore[reportMissingImports]

    batch_id = f"{out.name}_{time.strftime('%m%d_%H%M%S')}"
    run = ModalRunner(ctx, out, batch_id)
    runs = out / "runs"
    jobs = []
    for w in worlds:
        slots = sorted({(m.day, m.zone, m.t0) for m in missions if m.world == w})
        days = {d: p.name for (wi, d), p in scripts.items() if wi == w.index and d <= max(s[0] for s in slots)}
        files = dict(PL.day_script_payload(scripts[(w.index, d)]) for d in days)
        jobs.append(({"tag": f"{batch_id}_obs_w{w.index}", "world": TB.asdict(w), "slots": [list(s) for s in slots],
                      "days": days, "day_s": objs_days, "stub": ctx["stub"], "spec": ctx["spec"],
                      "margin": Path(ctx["margin"]).read_bytes(),
                      "site": Path(ctx["paths"]["site"]).resolve().relative_to(REPO).as_posix(),
                      "dynamics": Path(ctx["paths"]["dynamics"]).resolve().relative_to(REPO).as_posix(),
                      "snap_dir": f"{SNAPSHOT_VOL_DIR}/{batch_id}/w{w.index}"}, files))
    all_missions: list[TB.Mission] = []
    invalid_lists: list[dict] = []
    with R.app.run():
        t = time.time()
        stats = run.observer_phases(jobs)
        for w, st in zip(worlds, stats):
            phase_stats.append(st)
            costs.append({"what": f"observer_phase w{w.index}", "wall_s": st["wall_s"],
                          "usd": TB.container_usd(st["wall_s"]), "client_wall_s": round(time.time() - t, 1)})
        snap_rel = {(w.index, k): v for w, st in zip(worlds, stats) for k, v in st["snapshots"].items()}
        # reactive chain: world -> (volume path of the previous step's scored saved state, its pulled local copy)
        chain_state: dict[int, tuple[str, Path] | None] = {w.index: None for w in worlds}
        for r, rnd in enumerate(rnds):
            todo = list(rnd)
            k = 0
            step_in = {m.world.index: chain_state[m.world.index] for m in rnd if m.condition == "reactive"}
            while todo:
                batch = runs / f"round{r}_try{k}"
                batch.mkdir(parents=True, exist_ok=False)
                items = []
                for m in todo:
                    sp = scripts[(m.world.index, m.day)]
                    vw: dict[str, str] = {}
                    rin = None
                    if m.condition == "learned":
                        vw[SNAPSHOT] = snap_rel[(m.world.index, f"d{m.day}_{m.zone}")]
                    elif m.condition == "reactive":
                        rin = step_in.get(m.world.index)
                        if rin is not None:
                            vw[REACTIVE_IN] = rin[0]
                    cfg = modal_cfg(ctx, m, sp.name, max_rtf=run.max_rtf, reactive_in=rin is not None)
                    cfg["reactive_parent"] = rin[0] if rin else None
                    items.append((m, cfg, dict([PL.day_script_payload(sp)]), vw))
                TB.write_json(batch / "batch.json", {"cfgs": [c for _, c, _, _ in items]})
                for rec in run.missions(items, batch):
                    costs.append({"what": rec["run_id"], **rec})
                lst = classify(batch)
                invalid_lists.append(lst)
                all_missions += todo
                bad = set(lst.get("invalid", []))
                for m in todo:                      # the chain advances on the scored (first clean) attempt only
                    if m.condition == "reactive" and m.run_id not in bad:
                        cell_bad = any(x.run_id in bad for x in todo if x.cell == m.cell)
                        if not cell_bad:
                            prev = step_in.get(m.world.index)
                            local = batch / m.run_id / REACTIVE_OUT
                            chk = chain_check(local, m, prev[1] if prev else None)
                            TB.write_json(batch / f"{m.run_id}.reactive_chain.json", chk)
                            chain_state[m.world.index] = (f"episodes/{m.run_id}/{REACTIVE_OUT}", local)
                todo = TB.reruns(todo, lst.get("invalid", []))
                k += 1
    return all_missions, invalid_lists


def chain_check(state_out: Path, m: TB.Mission, prev_local: Path | None) -> dict:
    """reactive.verify_chain on the pulled saved state (fail closed: an error is recorded, the next chain step then
    starts from that file, and its bridge refuses a missing one)."""
    from intercycle_nav import reactive as RX  # pyright: ignore[reportMissingImports]

    try:
        return {"ok": True, **RX.verify_chain(str(state_out), m.run_id, str(prev_local) if prev_local else None,
                                              reactive_world_id(m.world))}
    except (OSError, ValueError) as e:
        return {"ok": False, "error": repr(e)[:500]}


# ------------------------------------------------------------------------------------------------ the batch


def run_batch(out: Path, worlds: list[TB.World], *, backend: str = "local", zones: list[str] | None = None,
              day_s: float = TB.DAY_S, wave_size: int = TB.WAVE_SIZE,
              classify: Callable[[Path], dict] | None = None, stubs: bool = False, max_cells: int = 0) -> dict:
    """Steps 1-4 of the module doc for every world; returns the batch record (also <out>/double_batch.json).
    `stubs` (the stub-world end-to-end test only) runs on the whole G0 v0 stub set (setup), local backend only.
    max_cells > 0 (dev smoke only) keeps the first max_cells (world, day, zone) cells of the plan."""
    if backend not in ("local", "modal"):
        raise DoubleError(f"backend {backend!r} unknown (local | modal)")
    if stubs and backend != "local":
        raise DoubleError("the G0 stub set runs on the local wiring backend only (fail closed)")
    t_wall = time.time()
    ctx = setup(out, stubs=stubs)
    SL.BW.build(ctx["paths"]["site"], ctx["paths"]["dynamics"], ctx["paths"]["tasks"], out / "worlds")
    zs = zones or ctx["zones"]
    missions = TB.plan(worlds, zs, day_s=day_s)
    if max_cells:
        if worlds[0].split != "dev":
            raise DoubleError("max_cells (a smoke subset) is for dev worlds only")
        keep = sorted({m.cell for m in missions}, key=lambda c: next(m.t0 for m in missions if m.cell == c))[:max_cells]
        missions = [m for m in missions if m.cell in keep]
    rnds = TB.rounds(missions, wave_size)
    runs = out / "runs"
    days_dir = out / "plant"
    costs: list[dict] = []
    scripts: dict[tuple[int, int], Path] = {}
    snaps: dict[tuple[int, int, str], bytes] = {}
    phase_stats: list[dict] = []
    last_day = max(m.day for m in missions)
    objs_by_world = {}
    for w in worlds:
        sp, objs = write_days(ctx, w, range(1, last_day + 1), day_s, days_dir)
        scripts.update({(w.index, d): p for d, p in sp.items()})
        objs_by_world[w.index] = objs
    classify = classify or classify_batch
    if backend == "modal":
        all_missions, invalid_lists = modal_batch(ctx, out, worlds, missions, rnds, scripts, day_s, classify, costs,
                                                  phase_stats)
        return finish_batch(out, ctx, backend, worlds, zs, missions, rnds, runs, all_missions, invalid_lists,
                            phase_stats, costs, t_wall)
    for w in worlds:
        slots = sorted({(m.day, m.zone, m.t0) for m in missions if m.world == w})
        s, st = observer_phase(ctx, w, list(slots), day_s, scripts=objs_by_world[w.index])
        snaps.update({(w.index, d, z): b for (d, z), b in s.items()})
        phase_stats.append(st)
        costs.append({"what": f"observer_phase w{w.index}", "wall_s": st["wall_s"], "usd": 0.0})
    all_missions: list[TB.Mission] = []
    invalid_lists = []
    for r, rnd in enumerate(rnds):
        todo = list(rnd)
        k = 0
        while todo:
            batch = runs / f"round{r}_try{k}"
            batch.mkdir(parents=True, exist_ok=False)
            TB.write_json(batch / "batch.json", {"cfgs": [dict(m.record(), tag=m.run_id, twin=True) for m in todo]})
            for m in todo:
                try:
                    res = local_mission(ctx, m, scripts[(m.world.index, m.day)], batch,
                                        snaps.get((m.world.index, m.day, m.zone)))
                    costs.append({"what": m.run_id, **res})
                except DoubleError:
                    raise
                except Exception as e:  # noqa: BLE001 - the record says why; classify_invalid decides
                    TB.write_json(batch / f"{m.run_id}.json", {"tag": m.run_id, "error": repr(e)[:2000]})
            lst = classify(batch)
            invalid_lists.append(lst)
            all_missions += todo
            todo = TB.reruns(todo, lst.get("invalid", []))
            k += 1
    return finish_batch(out, ctx, backend, worlds, zs, missions, rnds, runs, all_missions, invalid_lists, phase_stats,
                        costs, t_wall)


def finish_batch(out: Path, ctx: dict, backend: str, worlds, zs, missions, rnds, runs: Path, all_missions,
                 invalid_lists, phase_stats, costs, t_wall: float) -> dict:
    inv = TB.merge_invalid(invalid_lists, TB.mission_records(all_missions))
    TB.write_json(runs / TB.INVALID_ALL, inv)
    rec = {"contract": "twin.double_batch/1", "backend": backend, "split": worlds[0].split,
           "worlds": [TB.asdict(w) for w in worlds], "zones": zs, "conditions": list(SEALED_CONDITIONS),
           "n_missions": len(missions), "n_rounds": len(rnds), "runs_root": str(runs),
           "invalid": str(runs / TB.INVALID_ALL), "observer_phase": phase_stats, "costs": costs,
           "usd": round(sum(c.get("usd", c.get("container_usd", 0.0)) or 0.0 for c in costs), 4),
           "wall_s": round(time.time() - t_wall, 1), "margin": ctx["margin_source"], "stub_inputs": ctx["stub"]}
    TB.write_json(out / "double_batch.json", rec)
    TB.write_json(out / "costs.json", {"costs": costs, "usd": rec["usd"]})
    return rec


def classify_batch(batch: Path) -> dict:
    """scripts/classify_invalid.py on one round's batch dir (its rule, from the logs alone); writes invalid.json."""
    import classify_invalid as CI  # scripts/classify_invalid.py

    out = CI.classify(batch)
    CI.write(batch, out)
    return out


def dev_worlds(indices: list[int]) -> list[TB.World]:
    bad = [i for i in indices if i not in TB.DEV_WORLD_SEEDS]
    if bad:
        raise DoubleError(f"--dev runs dev worlds {TB.DEV_WORLD_SEEDS} only, got {bad}")
    return [TB.World(i, i, "dev") for i in indices]


def run_sealed(out: Path, seed_ints: list[int], *, backend: str = "modal") -> dict:
    """The sealed batch: 16 test worlds (index i = seed_ints[i]). Only sealed_run.py calls this, after its refusals
    (it sets INTERCYCLE_SEALED_RUN=1 only then); refused otherwise."""
    if os.environ.get(SEALED_ENV) != "1":
        raise DoubleError("the sealed double runs only through sealed_run.py after the robot FREEZE (no sealed flag)")
    if len(seed_ints) != 16:
        raise DoubleError(f"the sealed design has 16 test worlds, got {len(seed_ints)} seeds")
    if backend != SEALED_BACKEND or not SEALED_BACKEND_READY:
        raise DoubleError(f"the sealed batch runs only on the {SEALED_BACKEND} backend (ready: {SEALED_BACKEND_READY}); "
                          f"{backend!r} refused (the local backend is a wiring double, not the policies)")
    return run_batch(out, [TB.World(i, s, "test") for i, s in enumerate(seed_ints)], backend=backend)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU-double batch launcher (PLAN_TWIN §5, cut design)")
    ap.add_argument("--dev", action="store_true", help="dev worlds 0-3 only")
    ap.add_argument("--sealed", action="store_true", help="refused here: use sealed_run.py run --stages double")
    ap.add_argument("--out", default="")
    ap.add_argument("--worlds", default="0")
    ap.add_argument("--zones", default="")
    ap.add_argument("--backend", default="local", choices=("local", "modal"))
    ap.add_argument("--day-s", type=float, default=TB.DAY_S, help="generated day length (dev smoke only)")
    ap.add_argument("--max-cells", type=int, default=0, help="dev smoke: keep the first N (world, day, zone) cells")
    a = ap.parse_args(argv)
    if a.sealed or not a.dev:
        print("run_double REFUSED: the sealed batch runs only through scripts/sealed_run.py (after the robot FREEZE); "
              "use --dev for dev worlds 0-3", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else REPO / "out" / "double" / time.strftime("%m%d_%H%M%S")
    rec = run_batch(out, dev_worlds([int(x) for x in a.worlds.split(",") if x]), backend=a.backend,
                    zones=[z for z in a.zones.split(",") if z] or None, day_s=a.day_s, max_cells=a.max_cells)
    print(json.dumps({k: rec[k] for k in ("n_missions", "n_rounds", "usd", "wall_s", "runs_root")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
