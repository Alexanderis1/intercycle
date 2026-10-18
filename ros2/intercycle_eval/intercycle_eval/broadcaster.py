"""Episode-side observer process (R3): signs and broadcasts the observer layers for one condition. NOT the robot.

    $IC_SIM_PY -m intercycle_eval.broadcaster --work $IC_WORK --condition learned|oracle
        (run with PYTHONPATH=$IC_SIM_PYTHONPATH: Isaac's python 3.11 + numpy, astropy, pydantic, pynacl)

It follows the plant (intercycle_eval.plant) through `$IC_WORK/plant_state.json`: the plant clock and the freezes the
robot caused (holds / trips). From those it rebuilds the episode's World exactly (same scenario, seed, plant_t0), so the
observer sees the floor the robot actually changed. It never sees the robot.

  learned (condition 3, InterCycle): map layer = M's LayeredMap learned on TRAINING days (world file layered_map.npz),
          slow layer = cycle models learned on the training day (observer_models.json, from
          intercycle.sim.observer_learned.learned_models), fast layer = LearnedObserver.fast(t): nowcasts from the
          episode's simulated ceiling-camera signal up to t (latency, occlusion bursts, false flips).
  oracle  (condition 4, UPPER BOUND, labelled): slow = the scenario's true cycle parameters (true_models), fast = the
          TRUE state and next idle window from the World (as a PLC link would give, no latency), map layer = the same
          learned spatial layers as condition 3 (only the timing is oracle; stated in ROBOT_RESULTS).
Conditions none / static need no feed (the bridge builds the static keep-out from the provisioned site map).

Keys: a fresh Ed25519 key per episode (never written to disk except the PUBLIC verify key `observer_vk.txt`, which the
bridge's keyring parameter reads). Its key_id is DERIVED from the key (codec.trust.key_id_of, C-W5): observer_vk.txt
reads "<key_id_of(vk)>:<vk hex>", the map manifest is packed under that id (pack_map_for_key) and Broadcaster writes it
into every slow / fast header, so the scenario's constant key_id never reaches the air.
Feed: R2's FeedWriter("unix://$IC_WORK/observer.sock") + Broadcaster(...).tick(plant_t).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from nacl.signing import SigningKey


def episode_key() -> tuple[SigningKey, str]:
    """(fresh signing key, its keyring line "key_id:hex" with key_id = key_id_of(verify key))."""
    from intercycle.codec.trust import key_id_of

    key = SigningKey.generate()
    return key, f"{key_id_of(key.verify_key)}:{key.verify_key.encode().hex()}"


def _models_patch(path: str):
    """Replace observer_learned.learned_models with the models learned locally on the training day (shipped as json:
    the in-container cache key differs, and re-training would cost minutes)."""
    from intercycle.sim import observer_learned as OL
    from intercycle.types import MachineModel

    d = json.load(open(path))
    models = tuple(MachineModel(**m) for m in d["models"])
    offs = tuple(d["offsets"])
    OL.learned_models = lambda scn, seed=OL.TRAIN_SEED, window=OL.TRAIN_WINDOW: (models, offs)  # noqa: E731
    return d


_ROS_SCENARIO = re.compile(r"(?:^|\s)(?:-p|--param)\s+scenario:=(\S*)")
BRIDGE_MODULE = "intercycle_nav.observer_bridge"


def check_robot_scenario(cfg: dict) -> None:
    """Fail closed on a half-threaded scenario (S1b item 3, audit fix1).

    The robot side (observer_bridge -> masks.Site, timing_planner -> TimingCore) gets the scenario ONLY through the
    ' -p scenario:=...' ROS argument in its launch strings (masks.scenario_ros_arg); it never reads run_cfg. If the
    broadcaster loaded run_cfg["scenario"] while those strings lack the argument, it would sign frames with the twin
    site_id, the robot would load the body shop, and every frame would be rejected as 'wrong_site' (a counted
    rejection, not an error): the learned / oracle conditions would silently run without a feed. So: goal_cmd and
    every observer_bridge command in extra_launch must carry exactly one scenario argument equal to
    scenario_ros_arg(cfg), and none in body-shop mode (the reverse mismatch). In twin mode both commands must exist.
    Raises ValueError (TypeError for non-string commands)."""
    from intercycle_nav.masks import run_cfg_scenario, scenario_ros_arg

    run_cfg_scenario(cfg)                                  # validates the key (type, relative path needs world_dir)
    arg = scenario_ros_arg(cfg)
    want = [arg.split(":=", 1)[1]] if arg else []
    goal = cfg.get("goal_cmd") or ""
    extra = cfg.get("extra_launch") or []
    if not isinstance(goal, str) or not isinstance(extra, list) or not all(isinstance(c, str) for c in extra):
        raise TypeError("run_cfg goal_cmd must be a string and extra_launch a list of strings")
    bridges = [c for c in extra if BRIDGE_MODULE in c]
    if want and not goal:
        raise ValueError(f"run_cfg scenario {cfg['scenario']!r} is set but there is no goal_cmd (timing_planner) to "
                         "carry it to the robot side")
    if want and not bridges:
        raise ValueError(f"run_cfg scenario {cfg['scenario']!r} is set but extra_launch has no {BRIDGE_MODULE} "
                         "command to carry it to the robot side")
    for name, cmd in [("goal_cmd", goal)] + [(BRIDGE_MODULE, c) for c in bridges]:
        got = _ROS_SCENARIO.findall(cmd)
        if got != want:
            raise ValueError(f"{name} carries scenario argument(s) {got}, run_cfg wants {want or 'none (body shop)'}: "
                             f"append masks.scenario_ros_arg(cfg) = {arg!r} to the robot-side launch strings")


def load_scenario(cfg: dict):
    """(Scenario, path) of the episode: run_cfg["scenario"] (twin mode: the compiled twin scenario; relative = inside
    world_dir), else the body-shop default (path None). The robot side gets the same path as a ROS parameter
    (intercycle_nav.masks.scenario_ros_arg), never by reading run_cfg; check_robot_scenario refuses the run first if
    its launch strings do not carry exactly that argument."""
    from intercycle import scenario as S
    from intercycle_nav.masks import run_cfg_scenario

    check_robot_scenario(cfg)
    sp = run_cfg_scenario(cfg)
    return (S.load(sp) if sp else S.load()), sp


class OracleFeed:
    def __init__(self, scn, world):
        from intercycle.sim.observer_learned import true_models

        self.scn, self.world = scn, world
        self.models = true_models(scn)
        self.seq = 0

    def slow(self, t: float):
        from intercycle.types import DAY0_UNIX, SlowLayer

        return SlowLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t, [[0] * 24 for _ in self.scn.zones], self.models)

    def fast(self, t: float):
        from intercycle.types import DAY0_UNIX, FastLayer, MachineNow, State

        self.seq = (self.seq + 1) & 0xFFFF
        out = []
        for i, m in enumerate(self.scn.machines):
            ws = self.world.true_idle_windows(m.id, t, horizon=600.0)
            if ws and ws[0][0] <= t:
                st, nin, nlen = State.IDLE, 0.0, ws[0][1] - t
            elif ws:
                st, nin, nlen = State.ACTIVE, ws[0][0] - t, ws[0][1] - ws[0][0]
            else:
                st, nin, nlen = State.UNKNOWN, 0.0, 0.0
            out.append(MachineNow(i, st, max(0.0, nin), max(0.0, nlen), 4.0, 15 if st != State.UNKNOWN else 0, self.seq))
        return FastLayer(self.scn.site_id, self.scn.key_id, DAY0_UNIX + t, self.seq, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--condition", required=True, choices=("learned", "oracle"))
    ap.add_argument("--map", default="layered_map.npz")
    ap.add_argument("--models", default="observer_models.json")
    ap.add_argument("--fast-every", type=float, default=0.5)
    ap.add_argument("--slow-every", type=float, default=10.0)
    a = ap.parse_args()
    work = a.work
    cfg = json.load(open(os.path.join(work, "run_cfg.json")))
    wdir = cfg["world_dir"]
    out_dir = cfg.get("out_dir", work)
    pcfg = cfg.get("plant") or {}
    log = open(os.path.join(out_dir, "broadcaster.jsonl"), "a")

    from intercycle.observer import mapping as M
    from intercycle.sim.world import Freeze, World
    from intercycle.types import DAY0_UNIX
    from intercycle_nav.broadcast import Broadcaster, pack_map_for_key
    from intercycle_nav.feed import FeedWriter

    scn, scn_path = load_scenario(cfg)
    seed, t0 = int(pcfg["seed"]), float(pcfg["plant_t0"])
    T = float(cfg.get("seconds", 600.0))
    world = World(scn, seed, t0, t0 + T + 120.0)
    key, vk = episode_key()
    with open(os.path.join(work, "observer_vk.txt.tmp"), "w") as f:
        f.write(vk)
    os.replace(os.path.join(work, "observer_vk.txt.tmp"), os.path.join(work, "observer_vk.txt"))

    lm = M.LayeredMap.load(os.path.join(wdir, a.map))
    bundle = pack_map_for_key(lm, key, version=1, issue_time=DAY0_UNIX + t0 - 60.0)
    info = {"t_wall": time.time(), "condition": a.condition, "seed": seed, "plant_t0": t0,
            "map": a.map, "map_bytes": bundle.stats.get("compressed_bytes"), "verify_key": vk,
            "scenario": scn_path or "default"}
    if a.condition == "learned":
        from intercycle.sim.observer_learned import LearnedObserver

        md = _models_patch(os.path.join(wdir, a.models))
        info["models_train"] = md.get("train")
        obs = LearnedObserver(scn, world, seed, t0, t0 + T + 60.0)
        slow_fn, fast_fn = obs.slow, obs.fast
    else:
        orc = OracleFeed(scn, world)
        slow_fn, fast_fn = orc.slow, orc.fast
        info["label"] = "ORACLE (upper bound): true cycle timing; spatial layers = learned map"
    log.write(json.dumps({"start": info}) + "\n")
    log.flush()
    w = FeedWriter(f"unix://{os.path.join(work, 'observer.sock')}")
    bc = Broadcaster(w, key, bundle, slow_fn, fast_fn, fast_every_s=a.fast_every, slow_every_s=a.slow_every)

    st_path = os.path.join(work, "plant_state.json")
    last_t, n_ticks, busy, n_fz = -1.0, 0, 0.0, 0
    stop = os.path.join(work, "STOP")
    t_wall0 = time.time()
    while not os.path.exists(stop):
        try:
            st = json.load(open(st_path))
        except (OSError, ValueError):
            time.sleep(0.02)
            continue
        t = float(st["t"])
        if t <= last_t:
            time.sleep(0.01)
            continue
        fz = st.get("freezes", {})
        nf = sum(len(v) for v in fz.values())
        if nf != n_fz or nf:
            for gname, lst in fz.items():
                g = world.groups.get(gname)
                if g is None:
                    continue
                if len(g.freezes) == len(lst) and all(f.t0 == q[0] and f.t1 == q[1] for f, q in zip(g.freezes, lst)):
                    continue
                for k, q in enumerate(lst):
                    if k < len(g.freezes):
                        g.freezes[k].t1 = float(q[1])      # a hold the plant is still extending
                    else:
                        g.freezes.append(Freeze(float(q[0]), float(q[1]), str(q[2]), str(q[3])))
            if nf != n_fz:
                log.write(json.dumps({"t": t, "freezes": nf}) + "\n")
            n_fz = nf
        w0 = time.time()
        bc.tick(t0 + t)
        busy += time.time() - w0
        last_t = t
        n_ticks += 1
        if n_ticks % 500 == 0:
            log.write(json.dumps({"t": t, "ticks": n_ticks, "busy_s": round(busy, 2), "sent": w.sent,
                                  "dropped": w.dropped, "wall_s": round(time.time() - t_wall0, 1)}) + "\n")
            log.flush()
        time.sleep(0.01)
    log.write(json.dumps({"end": {"t": last_t, "ticks": n_ticks, "busy_s": round(busy, 2), "sent": w.sent,
                                  "dropped": w.dropped}}) + "\n")
    log.close()
    w.close()


if __name__ == "__main__":
    main()
