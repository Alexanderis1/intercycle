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
max_rtf, seed, movers / mover_source (see movers.py), extra_launch (list of shell commands R2/R3 add, run in the
ROS env after Nav2 starts).
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
    env["PYTHONPATH"] = _pkg_paths() + ":" + env.get("PYTHONPATH", "")
    env["INTERCYCLE_ROS2"] = ROS2_ROOT
    return env


def sim_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("ROS_") or k.startswith("AMENT") or k.startswith("COLCON") or k in ("RMW_IMPLEMENTATION",
                                                                                                   "CYCLONEDDS_URI"))}
    env["PYTHONPATH"] = _pkg_paths()
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


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_episode(cfg: dict, world: dict, vol_root: str) -> dict:
    tag = cfg["tag"]
    work = f"/tmp/ic_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
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
    res: dict = {"tag": tag, "cfg": cfg, "phases": {}}

    run_cfg = dict(cfg, world_dir=wdir, sock=sock, out_dir=out_dir, stop_file=stop_file)
    cfg_path = os.path.join(work, "run_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f)

    try:
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
            params = cfg.get("nav2_params") or os.path.join(PKG_ROOT, "config", "nav2_gate.yaml")
            procs.start("nav2", ["ros2", "launch", "nav2_bringup", "bringup_launch.py", "use_sim_time:=True",
                                 "autostart:=True", "use_composition:=False", "slam:=False",
                                 f"map:={os.path.join(wdir, 'map.yaml')}", f"params_file:={params}"], renv)
            for k, cmd in enumerate(cfg.get("extra_launch", [])):
                procs.start(f"extra{k}", cmd, dict(renv, IC_WORK=work, IC_WORLD=wdir, IC_OUT=out_dir), cwd=ROS2_ROOT,
                            shell=True)
            g = cfg["goal"]
            s = cfg.get("start", [4.0, 32.0, 0.0])
            procs.start("goal", ["/usr/bin/python3", "-u", os.path.join(PKG_ROOT, "intercycle_sim", "send_goal.py"),
                                 "--init", ",".join(str(v) for v in s), "--goal", ",".join(str(v) for v in g),
                                 "--timeout", str(cfg.get("timeout_s", 300)),
                                 "--out", os.path.join(work, "goal.json")], renv)
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
