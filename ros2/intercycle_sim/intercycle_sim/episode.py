"""Episode orchestrator, run inside the Modal container (stdlib only; any python).

Process layout (each one is its own process group, and every group is killed at the end of the episode):
  bridge   /usr/bin/python3 ros_bridge_node.py            ROS 2 Humble env (py3.10), listens on a Unix socket
  sim      python -m intercycle_sim.runner                Isaac python 3.11 (clean env, no ROS variables)
  nav2     ros2 launch nav2_bringup bringup_launch.py     use_sim_time, map = static L0 map, params = cfg["nav2_params"]
  goal     /usr/bin/python3 send_goal.py                  nav2_simple_commander: initial pose (dock) + goal
  cmdvel   ros2 topic pub /cmd_vel ...                    only in mode "cmdvel" (constant command smoke test)

DDS: CycloneDDS pinned to the loopback interface with unicast discovery (cyclonedds.xml; ROS_LOCALHOST_ONLY is NOT set
because rmw_cyclonedds would add lo a second time; ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
is also set for Iron+ tooling).

run_episode(cfg, world_files, vol_root) -> summary dict; logs go to <vol_root>/<tag>/.
cfg: tag, mode ("cmdvel" | "nav"), sim ("isaac" | "fake"), start [x,y,psi], seconds, cmd [vx,vy,wz] (cmdvel),
goal [x,y,psi] + timeout_s (nav), nav2 (bool), nav2_params (path inside the container; default config/nav2_gate.yaml),
nav2_cmd (replaces the stock Nav2 bringup), goal_cmd (replaces send_goal.py; must write $IC_WORK/goal.json),
max_rtf, seed, movers / mover_source (see movers.py), extra_launch (list of shell commands R2/R3 add, run in the
ROS env after Nav2 starts).
R-W4 (fix round 2): start_noise {world_seed, day, slot} and clutter {n, keep_clear, avoid_rects} (loc_noise.py
apply_loc_noise): the simulator gets the TRUE start (perturbed) and the cluttered world files; the planner keeps the
nominal "start" as its belief (send_goal --init; goal_cmd runners pass it as init) and gets the AMCL spread through
$IC_INIT_SPREAD_XY_M / $IC_INIT_SPREAD_YAW_DEG (0 when no start noise) plus $IC_INIT_BELIEF; the draw goes in the result
("loc_noise"). The keys depend on (world, day, slot) only, so every condition of a slot gets the same draw. A twin
evaluation episode without both is invalid (eval_check), as is a planner whose init or spread differs from the record.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../intercycle_sim (package dir parent)
ROS2_ROOT = os.path.dirname(PKG_ROOT)                                         # the repo's ros2/ tree (all packages)
ROS_SETUP = "/opt/ros/humble/setup.bash"
REPO_ROOT = os.environ.get("INTERCYCLE_REPO", "/root/repo")          # mounted intercycle/ + scenario/ (ROS env only)


def _pkg_paths() -> str:
    """Every package dir under ros2/ (intercycle_sim, intercycle_nav, intercycle_eval, ...) for PYTHONPATH."""
    try:
        return ":".join(os.path.join(ROS2_ROOT, d) for d in sorted(os.listdir(ROS2_ROOT))
                        if os.path.isdir(os.path.join(ROS2_ROOT, d)))
    except FileNotFoundError:
        return PKG_ROOT

CYCLONE_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="lo"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
      <Peers><Peer address="127.0.0.1"/></Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def ros_env(work: str) -> dict:
    out = subprocess.run(["bash", "-c", f"source {ROS_SETUP} && env -0"], capture_output=True, check=True).stdout
    env = dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)
    xml = os.path.join(work, "cyclonedds.xml")
    with open(xml, "w") as f:
        f.write(CYCLONE_XML)
    env.update(RMW_IMPLEMENTATION="rmw_cyclonedds_cpp", CYCLONEDDS_URI=f"file://{xml}",
               ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", ROS_DOMAIN_ID="17", PYTHONUNBUFFERED="1",
               RCUTILS_COLORIZED_OUTPUT="0")
    env["PYTHONPATH"] = _pkg_paths() + ":" + REPO_ROOT + ":" + env.get("PYTHONPATH", "")
    env["INTERCYCLE_ROS2"] = ROS2_ROOT
    env["INTERCYCLE_REPO"] = REPO_ROOT
    # `python3` in the ROS env must be the system 3.10 (rclpy), not Modal's 3.11 in /usr/local/bin
    env["PATH"] = "/opt/ros/humble/bin:/usr/bin:/bin:" + env.get("PATH", "")
    # the simulator's python (Isaac 3.11) for episode-side helpers launched from extra_launch (R3's observer process);
    # run it with IC_SIM_PYTHONPATH, never with the ROS PYTHONPATH (3.10 extension modules)
    import sys

    env["IC_SIM_PY"] = sys.executable
    env["IC_SIM_PYTHONPATH"] = _pkg_paths() + ":" + REPO_ROOT
    return env


def sim_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("ROS_") or k.startswith("AMENT") or k.startswith("COLCON") or k in ("RMW_IMPLEMENTATION",
                                                                                                   "CYCLONEDDS_URI"))}
    env["PYTHONPATH"] = _pkg_paths() + (":" + REPO_ROOT if os.path.isdir(REPO_ROOT) else "")   # R3's plant imports intercycle.*
    env["PYTHONUNBUFFERED"] = "1"
    return env


class Procs:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.p: dict[str, subprocess.Popen] = {}
        self.files = {}

    def start(self, name: str, cmd, env: dict, cwd: str | None = None, shell: bool = False) -> subprocess.Popen:
        f = open(os.path.join(self.logdir, f"{name}.log"), "w")
        self.files[name] = f
        pr = subprocess.Popen(cmd, env=env, cwd=cwd, shell=shell, stdout=f, stderr=subprocess.STDOUT,
                              start_new_session=True, executable="/bin/bash" if shell else None)
        self.p[name] = pr
        return pr

    def alive(self, name: str) -> bool:
        return name in self.p and self.p[name].poll() is None

    def kill_all(self, grace_s: float = 5.0) -> None:
        for sig, wait in ((signal.SIGINT, grace_s), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
            live = [n for n in self.p if self.p[n].poll() is None]
            if not live:
                break
            for n in live:
                try:
                    os.killpg(self.p[n].pid, sig)
                except ProcessLookupError:
                    pass
            t0 = time.time()
            while time.time() - t0 < wait and any(self.p[n].poll() is None for n in live):
                time.sleep(0.2)
        for f in self.files.values():
            f.close()

    def tail(self, name: str, n: int = 4000) -> str:
        try:
            with open(os.path.join(self.logdir, f"{name}.log"), errors="replace") as f:
                return f.read()[-n:]
        except FileNotFoundError:
            return ""


def _start_cmd(procs: "Procs", name: str, cmd, env: dict) -> None:
    """A user command: a shell string, or a list whose items get $VAR expansion from env; cwd = the ros2/ tree."""
    if isinstance(cmd, str):
        procs.start(name, cmd, env, cwd=ROS2_ROOT, shell=True)
    else:
        old = dict(os.environ)
        try:
            os.environ.update(env)
            argv = [os.path.expandvars(str(c)) for c in cmd]
        finally:
            os.environ.clear()
            os.environ.update(old)
        procs.start(name, argv, env, cwd=ROS2_ROOT)


def planner_env(ln: dict) -> dict:
    """Environment for the planner process: the AMCL spread (the start-noise range when start noise was applied, else
    0) and the belief pose (timing_planner reads the spread defaults from here)."""
    return {"IC_INIT_SPREAD_XY_M": repr(float(ln["spread"]["xy_m"])),
            "IC_INIT_SPREAD_YAW_DEG": repr(float(ln["spread"]["yaw_deg"])),
            "IC_INIT_BELIEF": ",".join(repr(float(v)) for v in ln["belief"])}


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _world_meta(world: dict) -> dict:
    try:
        return json.loads(world["world.json"].decode()) if "world.json" in world else {}
    except (ValueError, UnicodeDecodeError):
        return {}


def loc_noise_check(cfg: dict, res: dict) -> list[str]:
    """R-W4 reasons an evaluation episode is invalid: a twin run without start noise and clutter, or a planner whose
    belief (goal.json "init") or AMCL spread (goal.json "initial_pose".covariance_sent) differs from the episode's
    loc_noise record."""
    import math

    ln = res.get("loc_noise") or {}
    g = res.get("goal") or {}
    out = []
    if res.get("twin") or cfg.get("twin"):
        if not ln.get("start_applied"):
            out.append("twin run without R-W4 start-pose noise (cfg start_noise)")
        if not ln.get("clutter_applied"):
            out.append("twin run without R-W4 unmapped clutter (cfg clutter)")
    if ln.get("start_applied") and g:
        init = g.get("init")
        if not (isinstance(init, list) and len(init) == 3
                and all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(init, ln["belief"]))):
            out.append(f"planner init {init} is not the start_noise belief {ln.get('belief')}")
        sent = (g.get("initial_pose") or {}).get("covariance_sent") or {}
        sp = ln.get("spread") or {}
        want = {"xx": float(sp.get("xy_m", 0.0)) ** 2 / 3.0, "yy": float(sp.get("xy_m", 0.0)) ** 2 / 3.0,
                "yawyaw": math.radians(float(sp.get("yaw_deg", 0.0))) ** 2 / 3.0}
        if not all(k in sent and abs(float(sent[k]) - v) <= 1e-9 for k, v in want.items()):
            out.append(f"planner AMCL spread {sent} does not match the start noise {sp}")
    return out


def eval_check(cfg: dict, res: dict) -> dict:
    """Cheat guard for evaluation episodes (DIRECTION 5.1): the no-motion gate must have come from the camera model,
    never from the test-double stub, and the planner must have required it. cfg["eval"] marks an evaluation episode.
    R-W4: loc_noise_check."""
    g = res.get("goal") or {}
    reasons = []
    if cfg.get("eval"):
        reasons.extend(loc_noise_check(cfg, res))
        if res.get("error"):
            reasons.append(f"episode error: {res['error']}")
        if not g:
            reasons.append("no goal.json")
        if g.get("camera_stub_used"):
            reasons.append("camera_stub_used")
        if g and g.get("require_camera") is False:
            reasons.append("require_camera false")
        srcs = g.get("camera_sources")
        if g and srcs is None:
            reasons.append("goal.json has no camera_sources")
        elif g and not any("camera_model" in k for k in srcs) and g.get("condition") != "static":
            reasons.append(f"no camera_model observations ({srcs})")
        if cfg.get("sim") == "fake" and not cfg.get("allow_fake"):
            reasons.append("fake backend")
    return {"eval": bool(cfg.get("eval")), "valid": not reasons, "reasons": reasons,
            "camera_sources": g.get("camera_sources")}


def prepare_loc_noise(cfg: dict, world: dict) -> tuple[dict, dict, str | None]:
    """(world files for the sim, loc_noise record, error or None). R-W4: true start + cluttered world for the sim,
    the nominal start as the planner's belief. A malformed cfg is returned as the error (fail closed: the episode
    records it and starts nothing)."""
    nominal = [float(v) for v in cfg.get("start", [4.0, 32.0, 0.0])]
    ln: dict = {"start_applied": False, "clutter_applied": False, "belief": nominal, "true": list(nominal),
                "spread": {"xy_m": 0.0, "yaw_deg": 0.0}}
    if cfg.get("start_noise") is None and cfg.get("clutter") is None:
        return world, ln, None
    try:
        from intercycle_sim.loc_noise import apply_loc_noise

        world2, ln = apply_loc_noise(cfg, world)
        return world2, ln, None
    except Exception as e:  # noqa: BLE001 - fail closed
        return world, ln, f"loc_noise: {e!r}"


def run_episode(cfg: dict, world: dict, vol_root: str) -> dict:
    tag = cfg["tag"]
    work = f"/tmp/ic_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    twin = bool(cfg.get("twin", False)) or bool(_world_meta(world).get("twin", False))
    world, ln, ln_err = prepare_loc_noise(cfg, world)
    wdir = os.path.join(work, "world")
    os.makedirs(wdir)
    for name, data in world.items():
        with open(os.path.join(wdir, name), "wb") as f:
            f.write(data)
    out_dir = os.path.join(vol_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    logdir = os.path.join(work, "logs")
    os.makedirs(logdir)
    sock = os.path.join(work, "bridge.sock")
    status = os.path.join(work, "bridge_status.json")
    stop_file = os.path.join(work, "STOP")
    sim = cfg.get("sim", "fake")
    renv = ros_env(work)
    procs = Procs(logdir)
    t_start = time.time()
    res: dict = {"tag": tag, "cfg": cfg, "phases": {}, "twin": twin, "loc_noise": ln}

    run_cfg = dict(cfg, world_dir=wdir, sock=sock, out_dir=out_dir, stop_file=stop_file, start=list(ln["true"]))
    cfg_path = os.path.join(work, "run_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f)

    try:
        if ln_err is not None:
            raise RuntimeError(ln_err)
        procs.start("bridge", ["/usr/bin/python3", "-u", os.path.join(PKG_ROOT, "intercycle_sim", "ros_bridge_node.py"),
                               "--sock", sock, "--status-file", status], renv)
        py = "python" if sim == "isaac" else "python"
        procs.start("sim", [py, "-u", "-m", "intercycle_sim.runner", "--cfg", cfg_path], sim_env(),
                    cwd="/opt/IsaacLab" if sim == "isaac" else PKG_ROOT)
        # wait for the first lockstep frames (Isaac boot can take minutes)
        t0 = time.time()
        while time.time() - t0 < float(cfg.get("boot_timeout_s", 900)):
            st = _read_json(status)
            if st and st.get("frames", 0) > 0:
                break
            if not procs.alive("sim") or not procs.alive("bridge"):
                raise RuntimeError("sim or bridge exited during boot")
            time.sleep(1.0)
        else:
            raise RuntimeError("timeout waiting for the first bridge frame")
        res["phases"]["boot_wall_s"] = round(time.time() - t_start, 1)

        if cfg.get("mode") == "cmdvel":
            vx, vy, wz = cfg.get("cmd", [0.6, 0.0, 0.0])
            twist = f"{{linear: {{x: {vx}, y: {vy}, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: {wz}}}}}"
            procs.start("cmdvel", ["ros2", "topic", "pub", "-r", "20", "/cmd_vel", "geometry_msgs/msg/Twist", twist],
                        renv)
        if cfg.get("nav2"):
            icenv = dict(renv, IC_WORK=work, IC_WORLD=wdir, IC_OUT=out_dir, **planner_env(ln))
            if cfg.get("nav2_cmd"):                      # R2: replaces the stock bringup (e.g. a collision_monitor chain)
                _start_cmd(procs, "nav2", cfg["nav2_cmd"], icenv)
            else:
                params = cfg.get("nav2_params") or os.path.join(PKG_ROOT, "config", "nav2_gate.yaml")
                procs.start("nav2", ["ros2", "launch", "nav2_bringup", "bringup_launch.py", "use_sim_time:=True",
                                     "autostart:=True", "use_composition:=False", "slam:=False",
                                     f"map:={os.path.join(wdir, 'map.yaml')}", f"params_file:={params}"], icenv)
            for k, cmd in enumerate(cfg.get("extra_launch", [])):
                _start_cmd(procs, f"extra{k}", cmd, icenv)
            if cfg.get("goal_cmd"):                      # R2: timing_planner runs the mission, writes $IC_WORK/goal.json
                _start_cmd(procs, "goal", cfg["goal_cmd"], icenv)
            else:
                g = cfg["goal"]
                s = ln["belief"]                         # the robot's belief (= cfg start), never the true start
                procs.start("goal", ["/usr/bin/python3", "-u",
                                     os.path.join(PKG_ROOT, "intercycle_sim", "send_goal.py"),
                                     "--init", ",".join(str(v) for v in s), "--goal",
                                     ";".join(",".join(str(v) for v in q) for q in (g if isinstance(g[0], list) else [g])),
                                     "--timeout", str(cfg.get("timeout_s", 300)),
                                     "--until", str(cfg.get("patrol_until_s", 0.0)),
                                     "--out", os.path.join(work, "goal.json")], icenv)
        # wait for the sim to finish (time limit) or the goal client to finish
        limit = float(cfg.get("seconds", 60)) / max(float(cfg.get("max_rtf", 1.0)) or 1.0, 0.05) * 3 + 600
        t0 = time.time()
        while procs.alive("sim") and time.time() - t0 < limit:
            if cfg.get("nav2") and not procs.alive("goal"):
                time.sleep(2.0)                  # let the robot settle, then stop
                open(stop_file, "w").close()
                break
            time.sleep(1.0)
        t1 = time.time()
        while procs.alive("sim") and time.time() - t1 < 120:
            time.sleep(0.5)
    except Exception as e:  # noqa: BLE001 - report, then clean up
        res["error"] = repr(e)
    finally:
        procs.kill_all()
    res["phases"]["total_wall_s"] = round(time.time() - t_start, 1)
    res["sim"] = _read_json(os.path.join(out_dir, "sim_summary.json"))
    res["goal"] = _read_json(os.path.join(work, "goal.json"))
    res["bridge"] = _read_json(status)
    res["plant"] = _read_json(os.path.join(out_dir, "plant_summary.json"))
    res["eval_check"] = eval_check(cfg, res)
    for name in procs.p:
        shutil.copy(os.path.join(logdir, f"{name}.log"), os.path.join(out_dir, f"{name}.log"))
    with open(os.path.join(out_dir, "episode.json"), "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=1, default=str)
    tails = []
    for name in procs.p:
        tl = procs.tail(name, 2500 if name in ("sim", "nav2") else 1200)
        tails.append(f"== {name} ==\n{tl}")
    res["log_tail"] = "\n".join(tails)[-12000:]
    return res
