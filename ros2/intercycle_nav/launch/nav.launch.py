"""InterCycle Nav2 Humble stack (R2): localisation + navigation + collision monitor, one lifecycle manager each.

    ros2 launch /root/ros2/intercycle_nav/launch/nav.launch.py map:=$IC_WORLD/map.yaml [params_file:=...] \
        [localization:=amcl|slam_toolbox] [collision_monitor:=True]

Same nodes as nav2_bringup (Humble) bringup_launch.py with use_composition False, except the velocity chain:
  controller_server --cmd_vel_nav--> velocity_smoother --cmd_vel_smoothed--> collision_monitor --cmd_vel--> robot
(nav2_bringup wires velocity_smoother straight to cmd_vel, leaving no place for the protective stop). The behavior
server (spin / backup / drive_on_heading) also publishes into cmd_vel_smoothed, so the protective stop covers it too.
collision_monitor:=False routes velocity_smoother straight to cmd_vel (debug only).
localization:=slam_toolbox is the documented fallback (needs ros-humble-slam-toolbox in the image; not installed now).
Paths resolve relative to this file: the ros2/ tree is mounted, not colcon-installed.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS = os.path.join(HERE, "..", "config", "nav2.yaml")
TF = [("/tf", "tf"), ("/tf_static", "tf_static")]


def _nodes(context):
    params = LaunchConfiguration("params_file").perform(context)
    map_yaml = LaunchConfiguration("map").perform(context)
    cm = LaunchConfiguration("collision_monitor").perform(context).lower() in ("true", "1")
    loc = LaunchConfiguration("localization").perform(context)
    log = LaunchConfiguration("log_level").perform(context)
    common = {"use_sim_time": True}
    args = ["--ros-args", "--log-level", log]
    out = []
    # ---- localisation
    loc_nodes = ["map_server"]
    out.append(Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
                    parameters=[params, common, {"yaml_filename": map_yaml}], remappings=TF, arguments=args))
    if loc == "amcl":
        out.append(Node(package="nav2_amcl", executable="amcl", name="amcl", output="screen", parameters=[params, common],
                        remappings=TF, arguments=args))
        loc_nodes.append("amcl")
    elif loc == "slam_toolbox":
        out.append(Node(package="slam_toolbox", executable="localization_slam_toolbox_node", name="slam_toolbox",
                        output="screen", parameters=[params, common], remappings=TF, arguments=args))
    out.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_localization",
                    output="screen", parameters=[common, {"autostart": True, "node_names": loc_nodes}], arguments=args))
    # ---- navigation
    nav = [("nav2_controller", "controller_server", [("cmd_vel", "cmd_vel_nav")]),
           ("nav2_smoother", "smoother_server", []),
           ("nav2_planner", "planner_server", []),
           ("nav2_behaviors", "behavior_server", [("cmd_vel", "cmd_vel_smoothed" if cm else "cmd_vel")]),
           ("nav2_bt_navigator", "bt_navigator", []),
           ("nav2_waypoint_follower", "waypoint_follower", []),
           ("nav2_velocity_smoother", "velocity_smoother",
            [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel_smoothed" if cm else "cmd_vel")])]
    names = []
    for pkg, exe, rm in nav:
        out.append(Node(package=pkg, executable=exe, name=exe, output="screen", parameters=[params, common],
                        remappings=TF + rm, arguments=args))
        names.append(exe)
    if cm:
        out.append(Node(package="nav2_collision_monitor", executable="collision_monitor", name="collision_monitor",
                        output="screen", parameters=[params, common], remappings=TF, arguments=args))
        names.append("collision_monitor")
    out.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
                    output="screen", parameters=[common, {"autostart": True, "node_names": names}], arguments=args))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map", description="map_server yaml (the static L0 map)"),
        DeclareLaunchArgument("params_file", default_value=DEFAULT_PARAMS),
        DeclareLaunchArgument("collision_monitor", default_value="True"),
        DeclareLaunchArgument("localization", default_value="amcl"),
        DeclareLaunchArgument("log_level", default_value="info"),
        OpaqueFunction(function=_nodes),
    ])
