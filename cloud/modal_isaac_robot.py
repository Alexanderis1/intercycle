"""Modal app for the legged inspection robot (DIRECTION §5.2 Branch I): Isaac Lab physics-only + ROS 2 Humble + Nav2.

    modal run cloud/modal_isaac_robot.py --action check                  # CPU: image has Isaac pkgs + Humble + Nav2
    modal run cloud/modal_isaac_robot.py --action fake --goal 20,32      # CPU: bridge + Nav2 with the kinematic TEST DOUBLE
    modal run cloud/modal_isaac_robot.py --action cmdvel --seconds 20    # L4: ANYmal-D follows a constant /cmd_vel (empty scene)
    modal run cloud/modal_isaac_robot.py --action nav --goal 20,32       # L4: ANYmal-D walks to a Nav2 goal in the factory
    modal run cloud/modal_isaac_robot.py --action pull --tag NAME        # copy /episodes/NAME from the Volume to out/robot/

One image, layered so the expensive Isaac layers cache and ROS / our code sit on top:
  1. nvidia/cuda:12.8.1-devel-ubuntu22.04 + python 3.11 (Modal), GL/Vulkan client libs
  2. torch 2.7.0 cu128 (pinned), isaacsim[all,extscache] 5.1.0, Isaac Lab v2.3.0 editable, rsl-rl-lib 3.0.1 (torch re-pinned)
  3. ROS 2 Humble (ros-base) + Nav2 + CycloneDDS by apt (system python 3.10)
  4. our code (ros2/intercycle_sim) mounted at container start (no rebuild when it changes)

Physics only: never `--enable_cameras`, never RTX sensors (they segfault on Modal's driver), no custom Vulkan ICD or
NVIDIA_DRIVER_CAPABILITIES overrides. The world (static L0 mesh, occupancy map) is built LOCALLY from
scenario/bodyshop.yaml by `ros2/intercycle_sim/intercycle_sim/world.py` and shipped as bytes, so the Isaac python needs
no extra packages.

Results go to the Volume `intercycle-robot` under /episodes/<tag>/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "ros2" / "intercycle_sim"
APP_NAME = "intercycle-isaac-robot"
VOL_NAME = "intercycle-robot"
VOL = "/vol"
REMOTE_PKG = "/root/intercycle_sim"
ISAACLAB_TAG = "v2.3.0"
CU128 = "https://download.pytorch.org/whl/cu128"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

_GL_LIBS = [
    "git", "build-essential", "cmake", "libglu1-mesa", "libgl1", "libegl1", "libxext6", "libx11-6", "libxrandr2",
    "libxinerama1", "libxcursor1", "libxi6", "libsm6", "libice6", "libxt6", "libvulkan1", "vulkan-tools",
    "libglib2.0-0", "libxkbcommon0", "ffmpeg",
]

_ROS_PKGS = [
    "ros-humble-ros-base", "ros-humble-navigation2", "ros-humble-nav2-bringup", "ros-humble-nav2-mppi-controller",
    "ros-humble-nav2-smac-planner", "ros-humble-nav2-simple-commander", "ros-humble-rmw-cyclonedds-cpp",
    "ros-humble-tf2-ros", "python3-numpy", "python3-yaml",
]

isaac_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"})
    .apt_install(*_GL_LIBS)
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url=CU128)
    .pip_install("isaacsim[all,extscache]==5.1.0", extra_index_url="https://pypi.nvidia.com")
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab",
        "pip install 'setuptools<70' wheel && pip install --no-build-isolation flatdict==4.0.1",
        "cd /opt/IsaacLab/source && pip install -e isaaclab -e isaaclab_assets -e isaaclab_tasks"
        " --extra-index-url https://pypi.nvidia.com",
        "pip install click==8.1.7 idna==3.10 h5py",
        # rsl-rl pulls tensordict, which would upgrade torch: re-pin torch in the same resolve
        f"cd /opt/IsaacLab/source && pip install -e isaaclab_rl 'rsl-rl-lib==3.0.1' 'torch==2.7.0'"
        f" 'torchvision==0.22.0' --extra-index-url {CU128}",
        "python -c 'import torch; assert torch.__version__.startswith(\"2.7.0\"), torch.__version__; print(torch.__version__)'",
    )
)

image = (
    isaac_image
    .apt_install("curl", "gnupg2", "lsb-release", "software-properties-common", "locales")
    .run_commands(
        "locale-gen en_US.UTF-8",
        "add-apt-repository -y universe",
        "curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key"
        " -o /usr/share/keyrings/ros-archive-keyring.gpg",
        "echo \"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg]"
        " http://packages.ros.org/ros2/ubuntu jammy main\" > /etc/apt/sources.list.d/ros2.list",
        "apt-get update && apt-get install -y --no-install-recommends " + " ".join(_ROS_PKGS)
        + " && rm -rf /var/lib/apt/lists/*",
        "bash -c 'source /opt/ros/humble/setup.bash && ros2 pkg list | grep -E \"^nav2_(bringup|mppi_controller|smac_planner|amcl)$\"'",
    )
    .env({"LANG": "en_US.UTF-8", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(PKG), REMOTE_PKG, ignore=["**/__pycache__", "**/*.pyc", "test/**"])
)


# --------------------------------------------------------------------------------------------- remote functions
def _episode(cfg: dict, world: dict[str, bytes]) -> dict:
    sys.path.insert(0, REMOTE_PKG)
    from intercycle_sim.episode import run_episode

    out = run_episode(cfg, world, vol_root=f"{VOL}/episodes")
    vol.commit()
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=900)
def check() -> dict:
    """CPU: report what the image contains (no GPU, Isaac not started)."""
    import subprocess

    def sh(cmd: str) -> str:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        return (r.stdout + r.stderr)[-3000:]

    return {
        "python": sh("python --version; /usr/bin/python3 --version"),
        "isaac": sh("pip show isaacsim isaaclab isaaclab_rl rsl-rl-lib torch 2>/dev/null | grep -E '^(Name|Version)'"),
        "warp": sh("python -c 'import warp; print(\"warp\", warp.__version__)'"),
        "ros": sh("source /opt/ros/humble/setup.bash && ros2 pkg list | grep -cE '^nav2_' && "
                  "ros2 pkg list | grep -E '^(nav2_mppi_controller|nav2_smac_planner|rmw_cyclonedds_cpp)$'"),
        "rclpy": sh("source /opt/ros/humble/setup.bash && /usr/bin/python3 -c 'import rclpy, nav2_simple_commander; print(\"rclpy ok\")'"),
        "nproc": sh("nproc; free -g | head -2"),
    }


@app.function(image=image, cpu=8, memory=8192, timeout=1800, volumes={VOL: vol})
def fake_episode(cfg: dict, world: dict[str, bytes]) -> dict:
    """CPU: the bridge + Nav2 driven by the kinematic TEST DOUBLE (not a result; checks wiring before GPU time)."""
    cfg = dict(cfg, sim="fake")
    return _episode(cfg, world)


@app.function(image=image, gpu="L4", cpu=8, memory=24576, timeout=3600, volumes={VOL: vol})
def isaac_episode_l4(cfg: dict, world: dict[str, bytes]) -> dict:
    """L4: Isaac Lab ANYmal-D episode (physics only) + bridge + optional Nav2."""
    return _episode(dict(cfg, sim="isaac", gpu="L4"), world)


@app.function(image=image, gpu="L40S", cpu=8, memory=24576, timeout=3600, volumes={VOL: vol})
def isaac_episode_l40s(cfg: dict, world: dict[str, bytes]) -> dict:
    """L40S fallback (only if the L4 fails or is too slow)."""
    return _episode(dict(cfg, sim="isaac", gpu="L40S"), world)


# --------------------------------------------------------------------------------------------- local side
def build_world_bytes(empty: bool = False) -> dict[str, bytes]:
    """Build the static world locally (needs the repo's .venv: numpy, scipy, pydantic, pyyaml)."""
    sys.path.insert(0, str(PKG))
    sys.path.insert(0, str(REPO))
    from intercycle_sim.world import build_world_files

    return build_world_files(empty=empty)


def pull(tag: str) -> Path:
    import subprocess

    dest = REPO / "out" / "robot"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOL_NAME, f"episodes/{tag}",
                    str(dest)], check=True)
    return dest / tag


@app.local_entrypoint()
def main(action: str = "check", seconds: float = 20.0, vx: float = 0.6, vy: float = 0.0, wz: float = 0.0,
         goal: str = "20,32", start: str = "4,32,0", tag: str = "", gpu: str = "L4", empty: bool = False,
         max_rtf: float = 1.0, timeout_s: float = 300.0):
    import time

    if action == "check":
        print(json.dumps(check.remote(), indent=1))
        return
    if action == "pull":
        print("pulled ->", pull(tag))
        return
    sx, sy, syaw = (float(v) for v in start.split(","))
    gx, gy = (float(v) for v in goal.split(","))
    tag = tag or f"{action}_{time.strftime('%m%d_%H%M%S')}"
    cfg = {"tag": tag, "start": [sx, sy, syaw], "max_rtf": max_rtf}
    if action == "cmdvel":
        cfg.update(mode="cmdvel", seconds=seconds, cmd=[vx, vy, wz], nav2=False)
        world = build_world_bytes(empty=True)
    elif action in ("nav", "fake"):
        cfg.update(mode="nav", goal=[gx, gy, 0.0], nav2=True, timeout_s=timeout_s, seconds=timeout_s + 60)
        world = build_world_bytes(empty=empty)
    else:
        raise SystemExit(f"unknown action {action!r}")
    fn = fake_episode if action == "fake" else (isaac_episode_l40s if gpu.upper() == "L40S" else isaac_episode_l4)
    t0 = time.time()
    res = fn.remote(cfg, world)
    res["client_wall_s"] = round(time.time() - t0, 1)
    out = REPO / "out" / "robot"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log_tail",)}, indent=1))
    if res.get("log_tail"):
        print("---- log tail ----\n" + res["log_tail"])
