"""observer_bridge (+ optional dev tools) for one episode condition.

    ros2 launch /root/ros2/intercycle_nav/launch/observer.launch.py condition:=learned feed:=unix://$IC_WORK/observer.sock \
        plant_t0:=36000.0 keyring_file:=/path/keyring.json [camera:=real|stub] [devfeed:=False map_npz:=...] [probe:=True]

condition: none (no observer map) | static (static keep-out of all hazard zones) | learned (InterCycle) |
           oracle (map from truth, upper bound: the ORACLE FEED is produced by the episode runner, never by this package) |
           reactive (E-W4 baseline: own-camera windows, reactive_config:=<json> world_id:=<id> reactive_state:=<path>)
camera:=real (default) starts nothing: the robot camera's no-motion check comes from the sim side (R3: R1's geometric
camera model) on /intercycle/camera/observations. camera:=stub and devfeed:=True are TEST DOUBLES for wiring runs on the
kinematic double, opt-in only, never part of a result (timing_planner records the camera source in goal.json).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _py(module: str, name: str, params: dict, env: dict | None = None):
    # nodes are plain python modules (the ros2/ tree is mounted, not colcon-installed)
    return Node(executable="/usr/bin/python3", arguments=["-m", module], name=name, output="screen",
                parameters=[params], additional_env=env or {})


def _nodes(context):
    g = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    cond = g("condition")
    t0 = float(g("plant_t0"))
    feed = g("feed")
    devfeed = g("devfeed").lower() in ("true", "1")
    # C-W5: the bridge refuses to start without keyring_file / keyring unless dev=true; the devfeed signs with the
    # PUBLIC demo key, so devfeed:=True implies dev=true (wiring runs only, never a result)
    bp = {"use_sim_time": True, "condition": cond, "feed": feed, "plant_t0": t0, "horizon_s": float(g("horizon_s")),
          "dev": devfeed or g("dev").lower() in ("true", "1")}
    bp.update({k: g(k) for k in ("keyring", "keyring_file") if g(k)})
    # reactive baseline (E-W4): frozen config JSON, world id and the world's state file (reactive.py)
    bp.update({k: g(k) for k in ("reactive_config", "world_id", "reactive_state", "reactive_state_out", "mission_tag")
               if g(k)})
    out = [_py("intercycle_nav.observer_bridge", "observer_bridge", bp)]
    if g("camera") == "stub":
        out.append(_py("intercycle_nav.camera_stub", "camera_stub", {"use_sim_time": True, "mode": g("camera_mode")}))
    if g("probe").lower() in ("true", "1"):
        out.append(_py("intercycle_nav.nav_probe", "nav_probe", {"use_sim_time": True}))
    if devfeed:
        out.append(_py("intercycle_nav.devfeed", "devfeed",
                       {"use_sim_time": True, "feed": feed, "plant_t0": t0, "map_npz": g("map_npz"),
                        "forge_every_s": float(g("forge_every_s")), "stop_fast_after_s": float(g("stop_fast_after_s"))}))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("condition", default_value="learned"),
        DeclareLaunchArgument("feed", default_value="unix:///tmp/ic_observer.sock"),
        DeclareLaunchArgument("plant_t0", default_value="36000.0"),
        DeclareLaunchArgument("horizon_s", default_value="6.0"),
        DeclareLaunchArgument("camera", default_value="real"),       # real | stub (TEST DOUBLE, opt-in)
        DeclareLaunchArgument("camera_mode", default_value="always_clear"),
        DeclareLaunchArgument("devfeed", default_value="False"),
        DeclareLaunchArgument("keyring", default_value=""),          # "key_id:hex,..." pinned verify keys
        DeclareLaunchArgument("keyring_file", default_value=""),     # site cert keyring JSON (codec/trust.py)
        DeclareLaunchArgument("dev", default_value="False"),         # PUBLIC demo key; wiring tests only
        DeclareLaunchArgument("probe", default_value="True"),       # nav_probe: read-only evidence logger
        DeclareLaunchArgument("map_npz", default_value=""),
        DeclareLaunchArgument("reactive_config", default_value=""),   # condition reactive: ReactiveConfig JSON
        DeclareLaunchArgument("world_id", default_value=""),
        DeclareLaunchArgument("reactive_state", default_value=""),    # the world's state from its previous mission
        DeclareLaunchArgument("reactive_state_out", default_value=""),
        DeclareLaunchArgument("mission_tag", default_value=""),
        DeclareLaunchArgument("forge_every_s", default_value="0.0"),
        DeclareLaunchArgument("stop_fast_after_s", default_value="0.0"),
        OpaqueFunction(function=_nodes),
    ])
