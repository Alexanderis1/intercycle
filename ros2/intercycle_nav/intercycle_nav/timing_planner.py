"""timing_planner: SIPP timing layer that sends Nav2 goals with nav2_simple_commander (DIRECTION 5.3; see timing.py).

    python3 -m intercycle_nav.timing_planner --ros-args -p use_sim_time:=true -p condition:=learned \
        -p plant_t0:=36000.0 -p init:=[4.0,32.0,0.0] -p targets:=HEM1,S80 -p out:=$IC_WORK/goal.json

Inputs:  /intercycle/observer/state (observer_bridge: verified idle windows), /intercycle/camera/observations
         (std_msgs/String JSON {"stamp": ros_s, "targets": [{"id", "visible", "motion_seen"}]}, the robot camera's
         no-motion check; `camera_stub` for the test double), TF map -> base_link (AMCL estimate).
Outputs: Nav2 goals; /intercycle/inspection (String JSON: dwell_start / dwell_end events for R3's inspection checks);
         /intercycle/timing/events (String JSON: every decision); the result JSON (`out`, same top-level shape as
         intercycle_sim/send_goal.py: result, time_to_goal_s, ... plus "targets": per-target records).
Per target: walk to the wait pose at the hazard boundary (facing the machine) -> wait for an open predicted idle window
with need_s left AND the camera's no-motion confirmation -> enter -> dwell -> exit to the wait pose.
Initial pose (R-W4 start-pose noise, audit fix 4; fix round 2): when the episode applied start noise, after Nav2 is
active the planner publishes /initialpose itself as a PoseWithCovarianceStamped with the covariance of the start-pose
draw (timing.initial_pose_covariance; parameters init_spread_xy_m, init_spread_yaw_deg), because setInitialPose sends a
zero covariance. The spread defaults to 0 (exact start: body-shop R3 runs keep AMCL on the exact pose as before); the
default is taken from $IC_INIT_SPREAD_XY_M / $IC_INIT_SPREAD_YAW_DEG, which episode.py exports (the start-noise range
when it applied start noise, 0 otherwise), and an explicit -p overrides it. It republishes every 1 s until AMCL answers
on amcl_pose with every component that was sent non-zero also non-zero (initial_pose.ack_ok; max 30 s sim time);
goal.json "initial_pose" records the covariance sent and whether AMCL acknowledged it.
"""
from __future__ import annotations

import json
import math
from collections import Counter
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from intercycle import scenario as S
from intercycle_nav.initial_pose import ack_ok, spread_components
from intercycle_nav.masks import assert_no_truth
from intercycle_nav.timing import (CameraGate, TimingCore, camera_stub_used, initial_pose_covariance, interrupted_result,
                                   window_open)

RESULT = {TaskResult.SUCCEEDED: "SUCCEEDED", TaskResult.CANCELED: "CANCELED", TaskResult.FAILED: "FAILED"}
# conditions whose entry rule needs an open idle window, (a) + (b); reactive (E-W4) gets its windows from its own camera
# through observer_bridge, with the same planner and entry logic as learned
WINDOW_CONDITIONS = ("learned", "oracle", "reactive")


class Planner:
    def __init__(self) -> None:
        self.nav = nav = BasicNavigator()
        nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        dp = nav.declare_parameter
        self.condition = dp("condition", "learned").value
        self.plant_t0 = float(dp("plant_t0", 0.0).value)
        self.init = [float(v) for v in dp("init", [4.0, 32.0, 0.0]).value]       # dock pose x, y, psi (map frame)
        tl = str(dp("targets", "").value)
        self.mission_timeout = float(dp("mission_timeout_s", 900.0).value)
        self.max_wait = float(dp("max_wait_s", 180.0).value)
        self.goal_timeout = float(dp("goal_timeout_s", 240.0).value)
        self.out = str(dp("out", os.path.join(os.environ.get("IC_WORK", "/tmp"), "goal.json")).value)
        self.require_camera = bool(dp("require_camera", True).value)
        self.backup_speed = float(dp("backup_speed", 0.35).value)
        # spread 0 unless the episode applied start noise (it exports the range; fix round 2)
        self.init_cov = initial_pose_covariance(
            float(dp("init_spread_xy_m", float(os.environ.get("IC_INIT_SPREAD_XY_M", "0") or 0.0)).value),
            float(dp("init_spread_yaw_deg", float(os.environ.get("IC_INIT_SPREAD_YAW_DEG", "0") or 0.0)).value))
        self.entry_slack = float(dp("entry_slack_s", 3.0).value)       # entry deadline = enter_s + slack        # exit speed (m/s); 0 = exit with a NavigateToPose goal
        sp = dp("scenario", "").value
        scn = S.load(sp) if sp else S.load()
        self.core = TimingCore.from_scenario(scn, speed=float(dp("speed_mps", 0.5).value),
                                             enter_s=float(dp("enter_s", 5.0).value), exit_s=float(dp("exit_s", 6.5).value),
                                             wait_offset_m=float(dp("wait_offset_m", 1.05).value),
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
        self.camera_sources: Counter[str] = Counter()
        self._init_pub_t: float | None = None
        self._amcl_ack: list[float] | None = None
        amcl_qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=1)        # = BasicNavigator's amcl_pose QoS
        nav.create_subscription(PoseWithCovarianceStamped, "amcl_pose", self._on_amcl, amcl_qos)

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
        self.camera_sources[str(d.get("source", "unlabelled"))] += 1
        for tg in d.get("targets", []):
            self.cam.observe(st, str(tg.get("id")), bool(tg.get("visible")), bool(tg.get("motion_seen")))

    def _on_amcl(self, m) -> None:
        """AMCL's answer to OUR /initialpose: received after it was sent, with a non-zero position covariance. The robot
        stands still until then, so the filter BasicNavigator initialised with zero covariance still has every particle
        on one pose (covariance 0); a scan update after our spread init may shrink the spread but not to 0."""
        if self._init_pub_t is None or self._amcl_ack is not None:
            return
        cov = [float(v) for v in m.pose.covariance]
        if self.ros_t() >= self._init_pub_t and ack_ok(self.init_cov, cov):
            self._amcl_ack = cov

    def initial_pose_msg(self) -> PoseWithCovarianceStamped:
        ps = self.pose(*self.init)
        m = PoseWithCovarianceStamped()
        m.header = ps.header
        m.pose.pose = ps.pose
        m.pose.covariance = [float(v) for v in self.init_cov]
        return m

    def spread_initial_pose(self, timeout_s: float = 30.0) -> dict:
        """Publish the believed start pose with the start-noise covariance until AMCL acknowledges it (see module doc)."""
        rec = {"pose": list(self.init), "covariance_sent": {"xx": self.init_cov[0], "yy": self.init_cov[7],
                                                              "yawyaw": self.init_cov[35]}, "acknowledged": False}
        if not spread_components(self.init_cov):
            rec["note"] = "spread 0: zero covariance (setInitialPose only)"
            return rec
        t0, last = self.ros_t(), None
        self._init_pub_t, self._amcl_ack = t0, None
        while self._amcl_ack is None and self.ros_t() - t0 < timeout_s:
            if last is None or self.ros_t() - last >= 1.0:
                m = self.initial_pose_msg()
                self.nav.initial_pose_pub.publish(m)
                last = self.ros_t()
            rclpy.spin_once(self.nav, timeout_sec=0.05)
        ack = self._amcl_ack
        rec["acknowledged"] = ack is not None
        if ack is not None:
            rec["amcl_covariance"] = {"xx": round(ack[0], 5), "yy": round(ack[7], 5), "yawyaw": round(ack[35], 5)}
        rec["wait_s"] = round(self.ros_t() - t0, 2)
        return rec

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

    def go(self, x, y, psi, timeout: float | None = None) -> str:
        t0 = self.ros_t()
        if not self.nav.goToPose(self.pose(x, y, psi)):
            return "REJECTED"
        while not self.nav.isTaskComplete():
            if self.ros_t() - t0 > (self.goal_timeout if timeout is None else timeout):
                self.nav.cancelTask()
                return "TIMEOUT"
        return RESULT.get(self.nav.getResult(), str(self.nav.getResult()))

    def wait_task(self, timeout: float) -> str:
        t0 = self.ros_t()
        while not self.nav.isTaskComplete():
            if self.ros_t() - t0 > timeout:
                self.nav.cancelTask()
                return "TIMEOUT"
        return RESULT.get(self.nav.getResult(), str(self.nav.getResult()))

    def exit_zone(self, tp) -> str:
        """Leave the zone straight back along the entry line with Nav2's stock BackUp behaviour (behavior_server, local
        costmap collision checking, through the collision monitor); fall back to a NavigateToPose goal."""
        d = math.hypot(tp.inside_xy[0] - tp.wait_xy[0], tp.inside_xy[1] - tp.wait_xy[1])
        r = "REJECTED"
        if self.backup_speed > 0 and self.nav.backup(backup_dist=d, backup_speed=self.backup_speed,
                                                     time_allowance=int(d / self.backup_speed * 3 + 5)):
            r = self.wait_task(d / self.backup_speed * 3 + 10)
        if r != "SUCCEEDED":
            r = "backup_" + r + "+" + self.go(*tp.wait_xy, tp.face_psi)
        return r

    def dwell(self, tg, inside: bool) -> dict:
        """Hold still for dwell_s; inside a hazard zone (learned / oracle / reactive) cut it short when the predicted window no
        longer leaves exit + margin."""
        t0 = self.ros_t()
        cut = False
        while self.ros_t() - t0 < self.core.dwell_s:
            rclpy.spin_once(self.nav, timeout_sec=0.05)
            if inside and self.condition in WINDOW_CONDITIONS:
                if window_open(self.windows().get(tg.machine), self.plant_t(), self.core.exit_time(tg) + self.core.margin_s) is None:
                    cut = True
                    break
        return {"dwell_s": round(self.ros_t() - t0, 2), "dwell_cut": cut}

    # ---------------------------------------------------------------- mission
    def wait_for_entry(self, tg) -> dict | None:
        """Wait at the boundary until the entry rule holds; None on timeout."""
        need = self.core.need_s(tg)
        t0 = self.ros_t()
        while self.ros_t() - t0 < self.max_wait:
            rclpy.spin_once(self.nav, timeout_sec=0.05)
            t = self.plant_t()
            win = None
            if self.condition in WINDOW_CONDITIONS:
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
        out["initial_pose"] = self.spread_initial_pose()       # audit fix 4: the start-noise covariance, not zero
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
                # entry deadline: not inside within the budgeted entry time (+ slack) -> abort and back out
                r = self.go(*tp.inside_xy, tp.face_psi, timeout=self.core.enter_time(tg) + self.entry_slack)
                rec["to_inside"] = r
                rec["t_inside"] = round(self.plant_t(), 3)
                if r != "SUCCEEDED":
                    rec["outcome"] = "nav_failed_inside"
                    rec["to_exit"] = self.exit_zone(tp)
                    rec["t_exit"] = round(self.plant_t(), 3)
                    out["targets"].append(self.event("target_done", **rec))
                    continue
            self.event("dwell_start", target=tg.id, machine=tg.machine, inside=tp.inside_xy is not None)
            rec.update(self.dwell(tg, tp.inside_xy is not None))
            self.event("dwell_end", target=tg.id, machine=tg.machine, cut=rec["dwell_cut"])
            if tp.inside_xy is not None:
                rec["to_exit"] = self.exit_zone(tp)
                rec["t_exit"] = round(self.plant_t(), 3)
                w = (rec.get("entry") or {}).get("window")
                if w:
                    rec["overstay_s"] = round(max(0.0, rec["t_exit"] - float(w[1])), 3)   # self-report; holds are R3's truth metric
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
        # cheat guard (DIRECTION 5.1): which camera fed the no-motion gate; R3's runner fails episodes run on the stub
        out["camera_sources"] = dict(self.camera_sources)
        out["camera_stub_used"] = camera_stub_used(self.camera_sources)
        out["require_camera"] = self.require_camera
        return out


def main() -> None:
    rclpy.init()
    p = Planner()
    try:
        out = p.run()
    except KeyboardInterrupt:
        out = interrupted_result(p.events, p.camera_sources, p.require_camera, p.condition, len(p.targets))
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
