"""observer_bridge (+ optional dev tools) for one episode condition.

    ros2 launch /root/ros2/intercycle_nav/launch/observer.launch.py condition:=learned feed:=unix://$IC_WORK/observer.sock \
        plant_t0:=36000.0 [camera_stub:=False] [devfeed:=False map_npz:=...]

condition: none (no observer map) | static (static keep-out of all hazard zones) | learned (InterCycle) |
           oracle (map from truth, upper bound: the ORACLE FEED is produced by the episode runner, never by this package)
camera_stub / devfeed are TEST DOUBLES for wiring runs on the kinematic double; never part of a result.
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
    out = [_py("intercycle_nav.observer_bridge", "observer_bridge",
               {"use_sim_time": True, "condition": cond, "feed": feed, "plant_t0": t0,
                "horizon_s": float(g("horizon_s"))})]
    if g("camera_stub").lower() in ("true", "1"):
        out.append(_py("intercycle_nav.camera_stub", "camera_stub", {"use_sim_time": True, "mode": g("camera_mode")}))
    if g("devfeed").lower() in ("true", "1"):
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
        DeclareLaunchArgument("camera_stub", default_value="False"),
        DeclareLaunchArgument("camera_mode", default_value="always_clear"),
        DeclareLaunchArgument("devfeed", default_value="False"),
        DeclareLaunchArgument("map_npz", default_value=""),
        DeclareLaunchArgument("forge_every_s", default_value="0.0"),
        DeclareLaunchArgument("stop_fast_after_s", default_value="0.0"),
        OpaqueFunction(function=_nodes),
    ])
