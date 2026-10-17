"""Send one Nav2 goal with nav2_simple_commander and report the result as JSON (system python 3.10, Humble sourced).

    python3 send_goal.py --init 4,32,0 --goal 20,32,0 --timeout 300 --out /tmp/ic/goal.json

The initial pose is the robot's known dock pose (where every episode starts), given to AMCL once, as on a real robot.
Timeout is in SIM seconds.  Nothing here reads simulator truth.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.parameter import Parameter


def pose(nav, x, y, psi) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ix, iy, ipsi = (float(v) for v in a.init.split(","))
    g = [float(v) for v in a.goal.split(",")]
    gx, gy, gpsi = g[0], g[1], (g[2] if len(g) > 2 else 0.0)
    rclpy.init()
    nav = BasicNavigator()
    nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    out = {"goal": [gx, gy, gpsi], "init": [ix, iy, ipsi]}
    w0 = time.time()
    nav.setInitialPose(pose(nav, ix, iy, ipsi))
    nav.waitUntilNav2Active()
    out["nav2_active_wall_s"] = round(time.time() - w0, 1)
    t0 = nav.get_clock().now().nanoseconds * 1e-9
    out["t_goal_sent"] = t0
    nav.goToPose(pose(nav, gx, gy, gpsi))
    last_fb = None
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb is not None:
            last_fb = {"distance_remaining": float(fb.distance_remaining), "recoveries": int(fb.number_of_recoveries)}
        now = nav.get_clock().now().nanoseconds * 1e-9
        if now - t0 > a.timeout:
            nav.cancelTask()
            out["timeout"] = True
            break
        time.sleep(0.2)
    res = nav.getResult()
    out["result"] = {TaskResult.SUCCEEDED: "SUCCEEDED", TaskResult.CANCELED: "CANCELED",
                     TaskResult.FAILED: "FAILED"}.get(res, str(res))
    out["t_done"] = nav.get_clock().now().nanoseconds * 1e-9
    out["time_to_goal_s"] = round(out["t_done"] - t0, 2)
    out["last_feedback"] = last_fb
    out["wall_s"] = round(time.time() - w0, 1)
    print("GOAL_RESULT " + json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
