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
`modal` = NOT WIRED in this build (refused, fail closed): the observer phase in one Modal CPU container per world and
each mission on Modal CPU (cloud/modal_isaac_robot.py fake backend + Nav2, cpu 4, max_rtf 2) need a broadcaster that
starts from the pickled snapshot (twin/broadcast_live.py only re-ingests learn_obs.jsonl, which for 8 days is ~1.4 M
lines) and the reactive state chain through the Modal volume.

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
from pathlib import Path
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
SEALED_BACKEND_READY = False            # the Modal backend is NOT wired in this build: sealed_run refuses up front
SNAPSHOT = "observer_snapshot.pkl"      # the pickled observer state at a mission start (one per (world, day, zone))
SCORED_DAYS = (9, 10)


class DoubleError(RuntimeError):
    """The batch cannot go on (fail closed)."""


# ------------------------------------------------------------------------------------------------ inputs


def setup(out: Path) -> dict:
    """Contract files in force, compiled zones, the site-wide observer spec and the frozen M-W7 margin."""
    paths = SL.inputs()
    stub = all(C.is_stub(k, paths[k]) for k in ("site", "dynamics", "tasks"))
    site, dyn = load(Site, paths["site"]), load(Dynamics, paths["dynamics"])
    compiled = C.compile_paths(paths["site"], paths["dynamics"], paths["tasks"])
    adapter = PL.ObsAdapter(site)
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
    ob = OL.ObserverLive(ctx["spec"], OL.ObserverLiveConfig(margin_path=str(ctx["margin"])))
    pending = sorted(slots, key=lambda s: s[2])
    last_day = max(d for d, _, _ in pending)
    snaps: dict[tuple[int, str], bytes] = {}
    n = 0
    per_day = {}
    for day in range(1, last_day + 1):
        t_day = time.time()
        script = scripts[day] if scripts and day in scripts else generate_day(ctx, world.seed, day, day_s)
        eng = Engine(script, ctx["site"], ctx["dyn"])
        cam = PL.make_camera(ctx["site"], world.seed, day)
        ad = PL.ObsAdapter(ctx["site"])
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


def plant_cfg(ctx: dict, m: TB.Mission, script: Path, run_dir: Path, *, remote: bool = False) -> dict:
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


# ------------------------------------------------------------------------------------------------ the batch


def run_batch(out: Path, worlds: list[TB.World], *, backend: str = "local", zones: list[str] | None = None,
              day_s: float = TB.DAY_S, wave_size: int = TB.WAVE_SIZE,
              classify: Callable[[Path], dict] | None = None) -> dict:
    """Steps 1-4 of the module doc for every world; returns the batch record (also <out>/double_batch.json)."""
    if backend != "local":
        raise DoubleError(f"backend {backend!r} is not wired in this build (fail closed): only the local wiring backend")
    t_wall = time.time()
    ctx = setup(out)
    SL.BW.build(ctx["paths"]["site"], ctx["paths"]["dynamics"], ctx["paths"]["tasks"], out / "worlds")
    zs = zones or ctx["zones"]
    missions = TB.plan(worlds, zs, day_s=day_s)
    rnds = TB.rounds(missions, wave_size)
    runs = out / "runs"
    days_dir = out / "plant"
    costs: list[dict] = []
    scripts: dict[tuple[int, int], Path] = {}
    snaps: dict[tuple[int, int, str], bytes] = {}
    phase_stats = []
    for w in worlds:
        sp, objs = write_days(ctx, w, range(1, max(SCORED_DAYS) + 1), day_s, days_dir)
        scripts.update({(w.index, d): p for d, p in sp.items()})
        slots = sorted({(m.day, m.zone, m.t0) for m in missions if m.world == w})
        s, st = observer_phase(ctx, w, list(slots), day_s, scripts=objs)
        snaps.update({(w.index, d, z): b for (d, z), b in s.items()})
        phase_stats.append(st)
        costs.append({"what": f"observer_phase w{w.index}", "wall_s": st["wall_s"],
                      "usd": 0.0 if backend == "local" else TB.container_usd(st["wall_s"])})
    classify = classify or classify_batch
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
    a = ap.parse_args(argv)
    if a.sealed or not a.dev:
        print("run_double REFUSED: the sealed batch runs only through scripts/sealed_run.py (after the robot FREEZE); "
              "use --dev for dev worlds 0-3", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else REPO / "out" / "double" / time.strftime("%m%d_%H%M%S")
    rec = run_batch(out, dev_worlds([int(x) for x in a.worlds.split(",") if x]), backend=a.backend,
                    zones=[z for z in a.zones.split(",") if z] or None)
    print(json.dumps({k: rec[k] for k in ("n_missions", "n_rounds", "usd", "wall_s", "runs_root")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
