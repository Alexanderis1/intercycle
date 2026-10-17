"""Evaluation runner (R3, DIRECTION 5.6): 4 conditions paired by seed, one Modal container per episode, in parallel.

    .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/batch.py --name dev1 --seeds 20 \
        --conds none,static,learned,oracle --backend fake|l4 [--seconds 600] [--print-cfg]

Episode = a fixed 600 s of sim time (`seconds`), identical for every condition: Nav2 boots inside it, the timing
planner (R2) runs the default mission (all 10 machine targets in walking order) until the sim stops, then it is
interrupted and its events are the record (a planner that finishes all targets earlier ends the episode early).
Everything a condition achieved in those 600 s is counted; nothing is dropped.

Plant start time rule (stated in ROBOT_RESULTS): plant_t0 = 06:30 + U(0, 15 h) of day 0, drawn from
default_rng([seed, 5]), re-drawn until [t0 - 60 s, t0 + cap + 60 s] overlaps no planned line stop (breaks, shift
changes), so every episode is in production with machine cycles and people/vehicle traffic (people are present
06:00-22:00). Same t0 for the 4 conditions of a seed (paired).

Data hygiene: the observer map (out/layered_map.npz, workstream M) and the cycle models (observer_models.json) are
learned on training seeds only (map: seed 0, models: seed 99; scenario train range [0, 99]); development on dev seeds
20-29; the final batch on held-out seeds 1000-1009, run once.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "cloud", REPO / "ros2" / "intercycle_sim", REPO / "ros2" / "intercycle_eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONDS = ("none", "static", "learned", "oracle")
FEED = ("learned", "oracle")
MISSION_S = 100000.0             # the planner never ends the mission itself: the episode is a fixed sim length
INIT = [4.0, 32.0, 0.0]


def plant_start(seed: int, cap_s: float) -> float:
    import numpy as np

    from intercycle import scenario as S

    scn = S.load()
    stops = scn.line_stops()
    rng = np.random.default_rng([seed, 5])
    for _ in range(1000):
        t0 = round(6.5 * 3600 + float(rng.uniform(0, 15 * 3600)))
        a, b = t0 - 60, t0 + cap_s + 60
        if not any(s < b and e > a for s, e, _ in stops):
            return float(t0)
    raise RuntimeError("no production start found")


def episode_cfg(name: str, seed: int, cond: str, seconds: float = 600.0, invisible: bool = False,
                timing: dict | None = None) -> dict:
    cap = seconds
    t0 = plant_start(seed, cap)
    feed = cond in FEED
    sock = "unix://$IC_WORK/observer.sock"
    wait_vk = "for i in $(seq 1200); do [ -s $IC_WORK/observer_vk.txt ] && break; sleep 0.1; done; " if feed else ""
    keyring = " -p keyring:=\\\"$(cat $IC_WORK/observer_vk.txt)\\\"" if feed else ""
    bridge = (f"{wait_vk}exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true"
              f" -p condition:={cond} -p feed:={sock} -p plant_t0:={t0}{keyring}")
    extra = []
    if feed:
        extra.append(f"PYTHONPATH=$IC_SIM_PYTHONPATH exec $IC_SIM_PY -u -m intercycle_eval.broadcaster --work $IC_WORK"
                     f" --condition {cond}")
    extra.append(bridge)
    extra.append("exec python3 -u -m intercycle_nav.nav_probe --ros-args -p use_sim_time:=true")
    tp = {"enter_s": 5.0, "exit_s": 6.5}
    tp.update(timing or {})
    tps = "".join(f" -p {k}:={v}" for k, v in tp.items())
    goal = (f"python3 -u -m intercycle_nav.timing_planner --ros-args -p use_sim_time:=true -p condition:={cond}"
            f" -p plant_t0:={t0} -p init:=[{INIT[0]},{INIT[1]},{INIT[2]}] -p mission_timeout_s:={MISSION_S}"
            f" -p max_wait_s:=120.0{tps} -p out:=$IC_WORK/goal.json")
    return {
        "tag": f"{name}_s{seed}_{cond}", "eval": True, "condition": cond, "seed": seed, "start": INIT, "mode": "nav",
        "goal": INIT, "nav2": True, "seconds": cap, "timeout_s": cap, "max_rtf": 1.0,
        "mover_source": "intercycle_eval.plant:make",
        "plant": {"seed": seed, "plant_t0": t0, "invisible_robot": invisible, "feed": feed},
        "nav2_cmd": "ros2 launch /root/ros2/intercycle_nav/launch/nav.launch.py map:=$IC_WORLD/map.yaml",
        "extra_launch": extra, "goal_cmd": goal, "mission_s": MISSION_S,
    }


def observer_models_bytes() -> bytes:
    from intercycle import scenario as S
    from intercycle.sim import observer_learned as OL

    models, offs = OL.learned_models(S.load())
    return json.dumps({"models": [asdict(m) for m in models], "offsets": list(offs),
                       "train": {"seed": OL.TRAIN_SEED, "window": list(OL.TRAIN_WINDOW)}}).encode()


def world_bytes(map_path: Path) -> dict[str, bytes]:
    import modal_isaac_robot as R

    w = R.build_world_bytes(empty=False)
    w["layered_map.npz"] = map_path.read_bytes()
    w["observer_models.json"] = observer_models_bytes()
    return w


def pull(tags: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "intercycle-robot",
                        f"episodes/{tag}", str(dest)], check=False, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="20")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--backend", default="fake", choices=("fake", "l4", "l40s"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--map", default=str(REPO / "out" / "layered_map.npz"))
    ap.add_argument("--invisible", action="store_true")
    ap.add_argument("--timing-json", default="")
    ap.add_argument("--print-cfg", action="store_true")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for part in a.seeds.split(",") for s in
             (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]
    conds = [c for c in a.conds.split(",") if c]
    timing = json.loads(a.timing_json) if a.timing_json else None
    cfgs = [episode_cfg(a.name, s, c, a.seconds, a.invisible, timing) for s in seeds for c in conds]
    if a.backend == "fake":
        for c in cfgs:
            c["allow_fake"] = True
    if a.print_cfg:
        print(json.dumps(cfgs, indent=1))
        return
    world = world_bytes(Path(a.map))
    out = REPO / "out" / "robot" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.json").write_text(json.dumps({"name": a.name, "seeds": seeds, "conds": conds, "backend": a.backend,
                                                "seconds": a.seconds, "map": a.map, "started": time.strftime("%F %T"),
                                                "cfgs": cfgs}, indent=1))
    import modal_isaac_robot as R

    fn = {"fake": R.fake_episode, "l4": R.isaac_episode_l4, "l40s": R.isaac_episode_l40s}[a.backend]
    t0 = time.time()
    with R.app.run():
        res = list(fn.starmap([(c, world) for c in cfgs], return_exceptions=True, order_outputs=True))
    wall = time.time() - t0
    for c, r in zip(cfgs, res):
        if isinstance(r, BaseException):
            r = {"tag": c["tag"], "error": repr(r), "cfg": c}
        r["client_wall_s"] = round(wall, 1)
        (out / f"{c['tag']}.json").write_text(json.dumps(r, indent=1, default=str))
        g = r.get("goal") or {}
        sim = r.get("sim") or {}
        pl = r.get("plant") or {}
        print(f"{c['tag']:<34} res={g.get('result')} insp={g.get('n_inspected')}/{g.get('n_targets')}"
              f" trips={pl.get('trips')} lost={pl.get('line_time_lost_s')} contacts={pl.get('contacts')}"
              f" rtf={sim.get('rtf')} wall={r.get('phases', {}).get('total_wall_s')} check={r.get('eval_check')}"
              f" err={r.get('error')}")
    if not a.no_pull:
        pull([c["tag"] for c in cfgs], out)
    print(f"batch wall {wall:.0f}s -> {out}")


if __name__ == "__main__":
    main()
