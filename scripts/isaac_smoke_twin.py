"""S5a item 2 (G1 "static Isaac scene walking on compile(site_v0)"): ONE short Isaac episode on Modal L4.

    python scripts/isaac_smoke_twin.py [--zone z9] [--checkpoint cp_P9a_side] [--tag NAME] [--dry]
    python scripts/isaac_smoke_twin.py --verdict-only NAME          # re-score a pulled episode (out/robot/NAME/)

1. compile(site_v0): twin/compile.py on the G0 v0 stubs (twin/site_v0.yaml, dynamics_v0.json, tasks_v0.yaml).
2. World files: intercycle_sim/world.py build_world_files(scenario_path, rects, extent) for the zone (the same bytes as
   twin/<zone>/, with the USD under the name the Isaac backend loads, factory_l0.usda).
3. One episode through cloud/modal_isaac_robot.py isaac_episode_l4 (twin/ is mounted there): ANYmal-D starts at the
   zone's dock, Nav2 walks it to one checkpoint pose of the zone's site graph and back to the dock. Static scene: no
   movers, no plant, no observer. The container timeout is capped at 15 min (budget <= $0.35 at $1.37/container-h).
4. The episode directory is pulled from the Volume and the walk is scored (smoke_verdict): no episode error, a sim
   summary, no fall, both Nav2 legs SUCCEEDED, the TRUE pose came within REACH_TOL_M of the checkpoint and ended
   within it of the dock, and the path is at least MIN_PATH_FACTOR x the straight dock-checkpoint-dock length.
   Any missing piece fails the verdict (fail closed).

Frames: ROS x = plan x, ROS y = plan z; a plan yaw (0 faces +z) maps to the ROS heading psi = pi/2 - yaw (world.py).
The truth log read here is the simulator's own robot pose (evaluation only, after the run); it is not plant truth.
This script runs locally only (it imports scipy through intercycle.sim.geometry); the container gets bytes.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "ros2" / "intercycle_sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from twin.compile import CompiledZone, compile_paths, scenario_yaml  # noqa: E402

log = logging.getLogger("isaac_smoke_twin")

V0 = {"site": REPO / "twin" / "site_v0.yaml", "dynamics": REPO / "twin" / "dynamics_v0.json",
      "tasks": REPO / "twin" / "tasks_v0.yaml"}
ISAAC_USD_PER_H = 1.37          # measured Modal L4 + 8 CPU + 24 GiB rate (PLAN_TWIN §6)
CONTAINER_TIMEOUT_S = 900       # hard cap on the one container: 15 min = $0.34
NAV2_XY_TOL_M = 0.35            # xy_goal_tolerance in ros2/intercycle_sim/config/nav2_gate.yaml
REACH_TOL_M = NAV2_XY_TOL_M + 0.25    # + an allowance for the AMCL error (Nav2 judges the goal on its estimate)
MIN_PATH_FACTOR = 0.9


def ros_psi(plan_yaw: float) -> float:
    return math.pi / 2 - float(plan_yaw)


def compile_zone(zone: str, inputs: dict[str, Path] = V0) -> CompiledZone:
    """compile(site_v0) for one zone (the G0 v0 stubs: a pre-G2 smoke, logged as such)."""
    log.warning("compiling the G0 v0 STUBS %s (pre-G2 smoke)", {k: str(Path(p).relative_to(REPO)) for k, p in
                                                               inputs.items()})
    zones = compile_paths(inputs["site"], inputs["dynamics"], inputs["tasks"])
    if zone not in zones:
        raise KeyError(f"zone {zone!r} not in the compiled site ({sorted(zones)})")
    return zones[zone]


def zone_world_bytes(cz: CompiledZone) -> dict[str, bytes]:
    """world.py's files for the compiled zone (build_world_files(scenario_path, rects, extent))."""
    from intercycle_sim import world as W

    with tempfile.TemporaryDirectory() as td:
        scn_path = Path(td) / "scenario.yaml"
        scn_path.write_text(scenario_yaml(cz.scenario), encoding="utf-8", newline="\n")
        return W.build_world_files(scenario_path=str(scn_path), rects=cz.rects, extent=cz.extent)


def mission_cfg(cz: CompiledZone, checkpoint: str, *, tag: str, timeout_s: float = 150.0,
                max_rtf: float = 1.0) -> dict:
    """Episode cfg: dock -> the checkpoint's site-graph pose -> dock, static scene (no movers), Nav2 + send_goal."""
    tgt = {t.id: t for t in cz.graph.targets}
    if checkpoint not in tgt:
        raise KeyError(f"checkpoint {checkpoint!r} is not a target of zone {cz.zone} ({sorted(tgt)})")
    dx, dz = cz.graph.pos["DOCK"]
    dock = [float(dx), float(dz), ros_psi(cz.scenario.robot.get("dock_yaw", 0.0))]
    cx, cz_ = cz.graph.pos[tgt[checkpoint].node]       # point target: the capture pose; machine target: the wait pose
    cp = [float(cx), float(cz_), ros_psi(cz.graph.yaw[checkpoint])]
    return {"tag": tag, "mode": "nav", "start": list(dock), "goal": [cp, list(dock)], "nav2": True,
            "timeout_s": float(timeout_s), "seconds": 2 * float(timeout_s) + 60.0, "max_rtf": float(max_rtf),
            "twin": True, "seed": 0, "boot_timeout_s": 600,
            "smoke": {"zone": cz.zone, "checkpoint": checkpoint, "node": tgt[checkpoint].node,
                      "inputs": {k: str(Path(p).relative_to(REPO).as_posix()) for k, p in V0.items()}}}


def smoke_verdict(res: dict, rows: list[dict], cp_xy: tuple[float, float], dock_xy: tuple[float, float],
                  tol: float = REACH_TOL_M, min_path_factor: float = MIN_PATH_FACTOR) -> dict:
    """Did the robot walk dock -> checkpoint -> dock? Every check must hold; a missing input fails it."""
    sim = res.get("sim") or {}
    goal = res.get("goal") or {}
    legs = goal.get("legs") or []
    straight = 2 * math.dist(cp_xy, dock_xy)
    d_cp = [math.dist((float(r["x"]), float(r["y"])), cp_xy) for r in rows]
    k_cp = min(range(len(d_cp)), key=d_cp.__getitem__) if d_cp else None
    end_d = math.dist((float(rows[-1]["x"]), float(rows[-1]["y"])), dock_xy) if rows else None
    path = sim.get("path_length_m")
    checks = {
        "no_episode_error": not res.get("error"),
        "sim_summary": bool(sim),
        "no_fall": bool(sim) and sim.get("fell") is False and sim.get("end_reason") != "fall",
        "nav2_legs_succeeded": len(legs) == 2 and all(lg.get("result") == "SUCCEEDED" for lg in legs),
        "truth_log": bool(rows),
        "reached_checkpoint": k_cp is not None and d_cp[k_cp] <= tol,
        "back_at_dock": end_d is not None and end_d <= tol and k_cp is not None and k_cp < len(rows) - 1,
        "path_length": path is not None and float(path) >= min_path_factor * straight,
    }
    return {"walked": all(checks.values()), "checks": checks, "tol_m": tol,
            "min_dist_to_checkpoint_m": None if k_cp is None else round(d_cp[k_cp], 3),
            "end_dist_to_dock_m": None if end_d is None else round(end_d, 3),
            "path_length_m": path, "straight_dock_cp_dock_m": round(straight, 3),
            "min_z_m": round(min(float(r["z"]) for r in rows), 3) if rows else None,
            "sim_s": sim.get("sim_s"), "rtf": sim.get("rtf"), "n_contact_steps": sim.get("n_contact_steps"),
            "legs": [{k: lg.get(k) for k in ("result", "time_to_goal_s", "timeout")} for lg in legs]}


def load_truth_rows(ep_dir: Path) -> list[dict]:
    pj, pq = ep_dir / "truth_log.json", ep_dir / "truth_log.parquet"
    if pj.exists():
        return json.loads(pj.read_text())
    if pq.exists():
        import pandas as pd

        return pd.read_parquet(pq).to_dict("records")
    return []


def score(tag: str) -> dict:
    out = REPO / "out" / "robot"
    res = json.loads((out / f"{tag}.json").read_text())
    cfg = res["cfg"]
    cp, dock = cfg["goal"][0], cfg["goal"][1]
    rows = load_truth_rows(out / tag)
    v = smoke_verdict(res, rows, (cp[0], cp[1]), (dock[0], dock[1]))
    wall = res.get("client_wall_s")
    v.update(tag=tag, zone=cfg["smoke"]["zone"], checkpoint=cfg["smoke"]["checkpoint"], goal=cfg["goal"],
             boot_wall_s=(res.get("phases") or {}).get("boot_wall_s"),
             episode_wall_s=(res.get("phases") or {}).get("total_wall_s"), client_wall_s=wall,
             usd_upper_bound=None if wall is None else round(wall / 3600 * ISAAC_USD_PER_H, 3),
             safety_scanner=(res.get("sim") or {}).get("safety_scanner"))
    (out / f"{tag}_verdict.json").write_text(json.dumps(v, indent=1))
    return v


def run(zone: str, checkpoint: str, tag: str, dry: bool) -> int:
    cz = compile_zone(zone)
    world = zone_world_bytes(cz)
    cfg = mission_cfg(cz, checkpoint, tag=tag)
    print(json.dumps({"cfg": cfg, "world": {k: len(v) for k, v in world.items()},
                      "world_meta": json.loads(world["world.json"])}, indent=1))
    if dry:
        return 0
    os.environ.setdefault("IC_MODAL_APP", "intercycle-isaac-s5a-smoke")
    import modal

    sys.path.insert(0, str(REPO / "cloud"))
    import modal_isaac_robot as R

    out = REPO / "out" / "robot"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with modal.enable_output(), R.app.run():
        res = R.isaac_episode_l4.with_options(timeout=CONTAINER_TIMEOUT_S).remote(cfg, world)
    res["client_wall_s"] = round(time.time() - t0, 1)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1, default=str))
    if res.get("log_tail"):
        print("---- log tail ----\n" + res["log_tail"])
    R.pull(tag)
    v = score(tag)
    print("VERDICT " + json.dumps(v))
    return 0 if v["walked"] else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--zone", default="z9")
    ap.add_argument("--checkpoint", default="cp_P9a_side")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true", help="build and print the cfg; no Modal call")
    ap.add_argument("--verdict-only", default="", metavar="TAG")
    a = ap.parse_args(argv)
    if a.verdict_only:
        v = score(a.verdict_only)
        print("VERDICT " + json.dumps(v))
        return 0 if v["walked"] else 1
    return run(a.zone, a.checkpoint, a.tag or f"twin_smoke_{a.zone}_{time.strftime('%m%d_%H%M%S')}", a.dry)


if __name__ == "__main__":
    raise SystemExit(main())
