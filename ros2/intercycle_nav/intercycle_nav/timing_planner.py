"""timing_planner: SIPP timing layer that sends Nav2 goals with nav2_simple_commander (DIRECTION 5.3; see timing.py).

    python3 -m intercycle_nav.timing_planner --ros-args -p use_sim_time:=true -p condition:=learned \
        -p plant_t0:=36000.0 -p init:="4,32,0" -p targets:="HEM1,S80" -p out:=$IC_WORK/goal.json

Inputs:  /intercycle/observer/state (observer_bridge: verified idle windows), /intercycle/camera/observations
         (std_msgs/String JSON {"stamp": ros_s, "targets": [{"id", "visible", "motion_seen"}]}, the robot camera's
         no-motion check; `camera_stub` for the test double), TF map -> base_link (AMCL estimate).
Outputs: Nav2 goals; /intercycle/inspection (String JSON: dwell_start / dwell_end events for R3's inspection checks);
         /intercycle/timing/events (String JSON: every decision); the result JSON (`out`, same top-level shape as
         intercycle_sim/send_goal.py: result, time_to_goal_s, ... plus "targets": per-target records).
Per target: walk to the wait pose at the hazard boundary (facing the machine) -> wait for an open predicted idle window
with need_s left AND the camera's no-motion confirmation -> enter -> dwell -> exit to the wait pose.
"""
from __future__ import annotations

import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.parameter import Parameter
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from intercycle import scenario as S
from intercycle_nav.masks import assert_no_truth
from intercycle_nav.timing import CameraGate, TimingCore, window_open

RESULT = {TaskResult.SUCCEEDED: "SUCCEEDED", TaskResult.CANCELED: "CANCELED", TaskResult.FAILED: "FAILED"}


class Planner:
    def __init__(self) -> None:
        self.nav = nav = BasicNavigator()
        nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        dp = nav.declare_parameter
        self.condition = dp("condition", "learned").value
        self.plant_t0 = float(dp("plant_t0", 0.0).value)
        self.init = [float(v) for v in str(dp("init", "4,32,0").value).split(",")]
        tl = str(dp("targets", "").value)
        self.mission_timeout = float(dp("mission_timeout_s", 900.0).value)
        self.max_wait = float(dp("max_wait_s", 180.0).value)
        self.goal_timeout = float(dp("goal_timeout_s", 240.0).value)
        self.out = str(dp("out", os.path.join(os.environ.get("IC_WORK", "/tmp"), "goal.json")).value)
        self.require_camera = bool(dp("require_camera", True).value)
        sp = dp("scenario", "").value
        scn = S.load(sp) if sp else S.load()
        self.core = TimingCore.from_scenario(scn, speed=float(dp("speed_mps", 0.5).value),
                                             dwell_s=float(dp("dwell_s", scn.inspection.dwell_s).value),
                                             margin_s=float(dp("margin_s", scn.inspection.safety_margin_s).value))
        self.cam = CameraGate(confirm_n=int(dp("camera_confirm_n", 2).value),
                              min_span_s=float(dp("camera_span_s", scn.inspection.local_check_latency_s).value))
        by_id = {t.id: t for t in self.core.graph.targets}
        ids = [s.strip() for s in tl.split(",") if s.strip()] or [t.id for t in self.core.graph.mission() if t.kind == "machine"]
        self.targets = [by_id[i] for i in ids]
        assert_no_truth()
        self.obs: dict | None = None
        nav.create_subscription(String, "/intercycle/observer/state", self._on_state, 10)
        nav.create_subscription(String, "/intercycle/camera/observations", self._on_cam, 10)
        self.pub_ev = nav.create_publisher(String, "/intercycle/timing/events", 10)
        self.pub_insp = nav.create_publisher(String, "/intercycle/inspection", 10)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, nav)
        self.events: list[dict] = []

    # ---------------------------------------------------------------- io
    def _on_state(self, m: String) -> None:
        try:
            self.obs = json.loads(m.data)
        except ValueError:
            pass

    def _on_cam(self, m: String) -> None:
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        st = float(d.get("stamp", self.ros_t()))
        for tg in d.get("targets", []):
            self.cam.observe(st, str(tg.get("id")), bool(tg.get("visible")), bool(tg.get("motion_seen")))

    def ros_t(self) -> float:
        return self.nav.get_clock().now().nanoseconds * 1e-9

    def plant_t(self) -> float:
        return self.plant_t0 + self.ros_t()

    def windows(self) -> dict:
        if not self.obs:
            return {}
        return {mid: m.get("windows") for mid, m in self.obs.get("machines", {}).items()}

    def robot_xy(self):
        try:
            tr = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return float(tr.transform.translation.x), float(tr.transform.translation.y)
        except Exception:  # noqa: BLE001
            return None

    def event(self, kind: str, **kw) -> dict:
        e = {"event": kind, "t": round(self.plant_t(), 3), "ros_t": round(self.ros_t(), 3), **kw}
        self.events.append(e)
        m = String()
        m.data = json.dumps(e)
        (self.pub_insp if kind.startswith("dwell") else self.pub_ev).publish(m)
        self.nav.get_logger().info(m.data)
        return e

    def pose(self, x, y, psi) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = self.nav.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
        return p

    def spin_for(self, seconds: float) -> None:
        t_end = self.ros_t() + seconds
        while self.ros_t() < t_end:
            rclpy.spin_once(self.nav, timeout_sec=0.05)

    def go(self, x, y, psi) -> str:
        t0 = self.ros_t()
        if not self.nav.goToPose(self.pose(x, y, psi)):
            return "REJECTED"
        while not self.nav.isTaskComplete():
            if self.ros_t() - t0 > self.goal_timeout:
                self.nav.cancelTask()
                return "TIMEOUT"
        return RESULT.get(self.nav.getResult(), str(self.nav.getResult()))

    # ---------------------------------------------------------------- mission
    def wait_for_entry(self, tg) -> dict | None:
        """Wait at the boundary until the entry rule holds; None on timeout."""
        need = self.core.need_s(tg)
        t0 = self.ros_t()
        while self.ros_t() - t0 < self.max_wait:
            rclpy.spin_once(self.nav, timeout_sec=0.05)
            t = self.plant_t()
            win = None
            if self.condition in ("learned", "oracle"):
                win = window_open(self.windows().get(tg.machine), t, need)
                if win is None:
                    continue
            cam_ok = self.cam.clear(tg.machine, self.ros_t()) or not self.require_camera
            if cam_ok:
                return {"window": win, "need_s": round(need, 2), "waited_s": round(self.ros_t() - t0, 2),
                        "camera_confirmed": self.require_camera}
        return None

    def run(self) -> dict:
        nav = self.nav
        out = {"planner": "intercycle_nav.timing_planner (SIPP, Phillips & Likhachev 2011)", "condition": self.condition,
               "init": self.init, "targets": []}
        w0 = time.time()
        nav.setInitialPose(self.pose(*self.init))
        nav.waitUntilNav2Active()
        out["nav2_active_wall_s"] = round(time.time() - w0, 1)
        t_start = self.ros_t()
        out["t_goal_sent"] = t_start
        remaining = list(self.targets)
        no_plan_since = None
        while remaining and self.ros_t() - t_start < self.mission_timeout:
            rclpy.spin_once(nav, timeout_sec=0.05)
            xy = self.robot_xy() or (self.init[0], self.init[1])
            tp = self.core.choose(remaining, self.plant_t(), xy, self.windows(), self.condition)
            if tp is None:
                no_plan_since = no_plan_since or self.ros_t()
                if self.ros_t() - no_plan_since > self.max_wait:
                    tg = remaining.pop(0)
                    out["targets"].append(self.event("skipped_no_window", target=tg.id))
                    no_plan_since = None
                else:
                    self.spin_for(1.0)
                continue
            no_plan_since = None
            tg = tp.target
            remaining.remove(tg)
            rec = {"target": tg.id, "machine": tg.machine, "sipp": {"t_arrive_wait": round(tp.t_arrive_wait, 2),
                   "t_depart_wait": round(tp.t_depart_wait, 2), "t_goal": round(tp.t_goal, 2), "expanded": tp.expanded}}
            self.event("chosen", **rec)
            r = self.go(*tp.wait_xy, tp.face_psi)
            rec["to_wait"] = r
            rec["t_at_wait"] = round(self.plant_t(), 3)
            if r != "SUCCEEDED":
                rec["outcome"] = "nav_failed_to_wait"
                out["targets"].append(self.event("target_done", **rec))
                continue
            if tp.inside_xy is not None:                       # hazard target, conditions none / learned / oracle
                ent = self.wait_for_entry(tg)
                if ent is None:
                    rec["outcome"] = "missed_no_entry"
                    out["targets"].append(self.event("target_done", **rec))
                    continue
                rec["entry"] = ent
                rec["t_enter"] = round(self.plant_t(), 3)
                r = self.go(*tp.inside_xy, tp.face_psi)
                rec["to_inside"] = r
                if r != "SUCCEEDED":
                    rec["outcome"] = "nav_failed_inside"
                    self.go(*tp.wait_xy, tp.face_psi)
                    out["targets"].append(self.event("target_done", **rec))
                    continue
            self.event("dwell_start", target=tg.id, machine=tg.machine, inside=tp.inside_xy is not None)
            self.spin_for(self.core.dwell_s)
            self.event("dwell_end", target=tg.id, machine=tg.machine)
            if tp.inside_xy is not None:
                rec["to_exit"] = self.go(*tp.wait_xy, tp.face_psi)
                rec["t_exit"] = round(self.plant_t(), 3)
            rec["outcome"] = "inspected" if tp.inside_xy is not None else "inspected_from_boundary"
            out["targets"].append(self.event("target_done", **rec))
        done = [t for t in out["targets"] if str(t.get("outcome", "")).startswith("inspected")]
        out["result"] = "SUCCEEDED" if done and not remaining and len(done) == len(self.targets) else \
            ("PARTIAL" if done else "FAILED")
        out["timeout"] = bool(remaining)
        out["n_inspected"] = len(done)
        out["n_targets"] = len(self.targets)
        out["t_done"] = self.ros_t()
        out["time_to_goal_s"] = round(out["t_done"] - t_start, 2)
        out["wall_s"] = round(time.time() - w0, 1)
        return out


def main() -> None:
    rclpy.init()
    p = Planner()
    try:
        out = p.run()
    except KeyboardInterrupt:
        out = {"result": "INTERRUPTED", "targets": p.events}
    print("GOAL_RESULT " + json.dumps(out), flush=True)
    if p.out:
        os.makedirs(os.path.dirname(p.out) or ".", exist_ok=True)
        with open(p.out, "w") as f:
            json.dump(out, f, indent=1)
        ic_out = os.environ.get("IC_OUT")
        if ic_out:
            with open(os.path.join(ic_out, "timing.json"), "w") as f:
                json.dump(out, f, indent=1)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
