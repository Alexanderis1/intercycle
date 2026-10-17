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


def _go(nav, gx, gy, gpsi, timeout):
    t0 = nav.get_clock().now().nanoseconds * 1e-9
    nav.goToPose(pose(nav, gx, gy, gpsi))
    last_fb, timed_out = None, False
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb is not None:
            last_fb = {"distance_remaining": float(fb.distance_remaining), "recoveries": int(fb.number_of_recoveries)}
        if nav.get_clock().now().nanoseconds * 1e-9 - t0 > timeout:
            nav.cancelTask()
            timed_out = True
            break
        time.sleep(0.2)
    res = nav.getResult()
    t1 = nav.get_clock().now().nanoseconds * 1e-9
    return {"goal": [gx, gy, gpsi], "result": {TaskResult.SUCCEEDED: "SUCCEEDED", TaskResult.CANCELED: "CANCELED",
                                                TaskResult.FAILED: "FAILED"}.get(res, str(res)),
            "t_goal_sent": t0, "t_done": t1, "time_to_goal_s": round(t1 - t0, 2), "timeout": timed_out,
            "last_feedback": last_fb}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--goal", required=True, help="x,y[,psi]; several goals separated by ';' = a patrol")
    ap.add_argument("--timeout", type=float, default=300.0, help="per goal, sim seconds")
    ap.add_argument("--until", type=float, default=0.0, help="repeat the patrol until this sim time (0 = once)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ix, iy, ipsi = (float(v) for v in a.init.split(","))
    goals = []
    for g in a.goal.split(";"):
        v = [float(x) for x in g.split(",")]
        goals.append((v[0], v[1], v[2] if len(v) > 2 else 0.0))
    rclpy.init()
    nav = BasicNavigator()
    nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    w0 = time.time()
    nav.setInitialPose(pose(nav, ix, iy, ipsi))
    nav.waitUntilNav2Active()
    active_wall = round(time.time() - w0, 1)
    legs = []
    while True:
        for g in goals:
            legs.append(_go(nav, *g, a.timeout))
            print("LEG " + json.dumps(legs[-1]), flush=True)
            if a.until and nav.get_clock().now().nanoseconds * 1e-9 >= a.until:
                break
        if not a.until or nav.get_clock().now().nanoseconds * 1e-9 >= a.until:
            break
    out = dict(legs[0]) if len(legs) == 1 else {
        "goal": legs[-1]["goal"], "result": "SUCCEEDED" if all(l["result"] == "SUCCEEDED" for l in legs) else "FAILED",
        "t_goal_sent": legs[0]["t_goal_sent"], "t_done": legs[-1]["t_done"],
        "time_to_goal_s": round(legs[-1]["t_done"] - legs[0]["t_goal_sent"], 2),
        "legs_ok": sum(l["result"] == "SUCCEEDED" for l in legs), "legs": legs}
    out.update(init=[ix, iy, ipsi], nav2_active_wall_s=active_wall, wall_s=round(time.time() - w0, 1))
    print("GOAL_RESULT " + json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
