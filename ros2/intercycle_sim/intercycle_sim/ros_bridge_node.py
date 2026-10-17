"""ROS 2 side of the lockstep bridge (system python 3.10, rclpy).  Run with the Humble environment sourced:

    python3 ros_bridge_node.py --sock /tmp/ic_bridge.sock [--ns ""] [--cmd-timeout 0.5] [--frames-out /tmp/n]

Listens on a Unix socket; the simulator connects and sends one frame per physics step.  Per frame it publishes
  /clock                    rosgraph_msgs/Clock      (sim time; every node runs with use_sim_time)
  /imu                      sensor_msgs/Imu          frame imu_link (no orientation: covariance[0] = -1)
  /odom                     nav_msgs/Odometry        odom -> base_link (LABELLED leg/IMU odometry stand-in, drifts)
  /tf                       odom -> base_link
  /scan                     sensor_msgs/LaserScan    frame laser (10 Hz, levelled 2D lidar)
  /tf_static                base_link -> laser, base_link -> imu_link, base_link -> camera_link
  /intercycle/contacts      std_msgs/String (JSON)   collision events reported by the sim (for R3's logger only)
  /intercycle/camera/observations  std_msgs/String (JSON)  robot-camera no-motion observations (R3 camera model:
                            {"stamp", "source", "targets": [{"id", "visible", "motion_seen"}]}, already latency-delayed)
  /intercycle/inspection/capture   std_msgs/String (JSON)  inspection capture checks (R3; logging only)
and replies with the latest /cmd_vel (geometry_msgs/Twist; zero if older than --cmd-timeout s of sim time) plus the
/intercycle/inspection messages (timing_planner dwell_start / dwell_end) received since the last frame, so the sim
side can take the inspection capture.
Nothing here is ground truth: frames carry only sensor-model outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402

from intercycle_sim.protocol import listen, recv_msg, send_msg  # noqa: E402


def _stamp(t: float) -> Time:
    s = int(math.floor(t))
    return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)


def _qz(psi: float):
    return 0.0, 0.0, math.sin(psi / 2), math.cos(psi / 2)


class Bridge(Node):
    def __init__(self, a):
        super().__init__("intercycle_sim_bridge", parameter_overrides=[])
        self.a = a
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.imu_pub = self.create_publisher(Imu, "imu", sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "scan", sensor_qos)
        self.contact_pub = self.create_publisher(String, "intercycle/contacts", 10)
        self.tfb = TransformBroadcaster(self)
        self.stfb = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.cam_pub = self.create_publisher(String, "/intercycle/camera/observations", 10)
        self.cap_pub = self.create_publisher(String, "/intercycle/inspection/capture", 10)
        self.create_subscription(String, "/intercycle/inspection", self._on_insp, 10)
        self.events: list = []
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_t = -1e9
        self.sim_t = 0.0
        self.n_frames = 0
        self.statics_sent = False

    def _on_cmd(self, m: Twist) -> None:
        with self.lock:
            self.cmd = (float(m.linear.x), float(m.linear.y), float(m.angular.z))
            self.cmd_t = self.sim_t

    def _on_insp(self, m: String) -> None:
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.events.append(d)

    def _statics(self, fr: dict) -> None:
        mounts = fr.get("mounts", {})
        out = []
        for child, xyz in (("laser", mounts.get("laser", [0.3, 0.0, 0.2])),
                           ("imu_link", mounts.get("imu", [0.0, 0.0, 0.0])),
                           ("camera_link", mounts.get("camera", [0.45, 0.0, 0.35]))):
            ts = TransformStamped()
            ts.header.stamp = _stamp(fr["t"])
            ts.header.frame_id = "base_link"
            ts.child_frame_id = child
            ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = (float(v) for v in xyz)
            ts.transform.rotation.w = 1.0
            out.append(ts)
        self.stfb.sendTransform(out)
        self.statics_sent = True

    def publish_frame(self, fr: dict) -> None:
        t = float(fr["t"])
        st = _stamp(t)
        with self.lock:
            self.sim_t = t
        self.clock_pub.publish(Clock(clock=st))
        if not self.statics_sent:
            self._statics(fr)
        im = fr.get("imu")
        if im:
            m = Imu()
            m.header.stamp, m.header.frame_id = st, "imu_link"
            m.orientation_covariance[0] = -1.0
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = im["gyro"]
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = im["acc"]
            for i in (0, 4, 8):
                m.angular_velocity_covariance[i] = im["gyro_var"]
                m.linear_acceleration_covariance[i] = im["acc_var"]
            self.imu_pub.publish(m)
        od = fr.get("odom")
        if od:
            qx, qy, qz, qw = _qz(od["psi"])
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = st, "odom", "base_link"
            o.pose.pose.position.x, o.pose.pose.position.y = od["x"], od["y"]
            o.pose.pose.orientation.x, o.pose.pose.orientation.y = qx, qy
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = od["vx"], od["vy"], od["wz"]
            for i in (0, 7):
                o.twist.covariance[i] = od["v_var"]
            o.twist.covariance[35] = od["w_var"]
            o.pose.covariance[0] = o.pose.covariance[7] = 0.05
            o.pose.covariance[35] = 0.05
            self.odom_pub.publish(o)
            ts = TransformStamped()
            ts.header.stamp, ts.header.frame_id, ts.child_frame_id = st, "odom", "base_link"
            ts.transform.translation.x, ts.transform.translation.y = od["x"], od["y"]
            ts.transform.rotation.x, ts.transform.rotation.y = qx, qy
            ts.transform.rotation.z, ts.transform.rotation.w = qz, qw
            self.tfb.sendTransform(ts)
        sc = fr.get("scan")
        if sc:
            m = LaserScan()
            m.header.stamp, m.header.frame_id = st, "laser"
            m.angle_min, m.angle_max, m.angle_increment = sc["angle_min"], sc["angle_max"], sc["angle_increment"]
            m.time_increment, m.scan_time = 0.0, sc["scan_time"]
            m.range_min, m.range_max = sc["range_min"], sc["range_max"]
            m.ranges = [float(v) for v in sc["ranges"]]
            self.scan_pub.publish(m)
        if fr.get("contacts"):
            self.contact_pub.publish(String(data=json.dumps({"t": t, "contacts": fr["contacts"]})))
        for ob in fr.get("camera_obs") or []:
            self.cam_pub.publish(String(data=json.dumps(ob)))
        for cp in fr.get("inspection_capture") or []:
            self.cap_pub.publish(String(data=json.dumps(cp)))
        self.n_frames += 1

    def reply(self) -> dict:
        with self.lock:
            age = self.sim_t - self.cmd_t
            vx, vy, wz = self.cmd if age <= self.a.cmd_timeout else (0.0, 0.0, 0.0)
            ev, self.events = self.events, []
        out = {"type": "cmd", "vx": vx, "vy": vy, "wz": wz, "age": age}
        if ev:
            out["events"] = ev
        return out


def serve(node: Bridge, a) -> None:
    srv = listen(a.sock)
    node.get_logger().info(f"bridge listening on {a.sock}")
    conn, _ = srv.accept()
    node.get_logger().info("simulator connected")
    status = a.status_file
    try:
        while rclpy.ok():
            msg = recv_msg(conn)
            if msg.get("type") == "bye":
                break
            node.publish_frame(msg)
            send_msg(conn, node.reply())
            if status and node.n_frames % 50 == 1:
                with open(status, "w") as f:
                    json.dump({"frames": node.n_frames, "t": node.sim_t}, f)
    except ConnectionError as e:
        node.get_logger().warn(f"simulator disconnected: {e}")
    finally:
        conn.close()
        srv.close()
        if status:
            with open(status, "w") as f:
                json.dump({"frames": node.n_frames, "t": node.sim_t, "closed": True}, f)
        rclpy.try_shutdown() if hasattr(rclpy, "try_shutdown") else rclpy.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sock", default="/tmp/ic_bridge.sock")
    p.add_argument("--cmd-timeout", type=float, default=0.5)
    p.add_argument("--status-file", default="")
    a, ros_args = p.parse_known_args()
    rclpy.init(args=ros_args)
    node = Bridge(a)
    th = threading.Thread(target=serve, args=(node, a), daemon=True)
    th.start()
    try:
        rclpy.spin(node)
    except Exception:  # noqa: BLE001 - shutdown from the serve thread ends spin
        pass
    th.join(timeout=2.0)


if __name__ == "__main__":
    main()
