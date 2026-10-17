"""camera_stub: TEST DOUBLE for the robot camera's no-motion check (wiring only, never a result).

Publishes /intercycle/camera/observations (std_msgs/String JSON {"stamp": ros_s, "source": "camera_stub (TEST DOUBLE)",
"targets": [{"id", "visible", "motion_seen"}]}) at `rate_hz` for every machine in the site map:
  mode "always_clear"  visible, no motion (lets the timing planner's entry gate be exercised with the kinematic double)
  mode "never_clear"   visible, motion seen (entry must never happen)
It reads nothing. The real publisher is the geometric camera model in the sim bridge (R1/R3), same message.
"""
from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from intercycle_nav.masks import Site


class CameraStub(Node):
    def __init__(self) -> None:
        super().__init__("camera_stub")
        self.mode = self.declare_parameter("mode", "always_clear").value
        rate = float(self.declare_parameter("rate_hz", 5.0).value)
        self.ids = Site.from_scenario().ids
        self.pub = self.create_publisher(String, "/intercycle/camera/observations", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().warn(f"camera_stub TEST DOUBLE mode={self.mode}")

    def _tick(self) -> None:
        clear = self.mode == "always_clear"
        m = String()
        m.data = json.dumps({"stamp": self.get_clock().now().nanoseconds * 1e-9, "source": "camera_stub (TEST DOUBLE)",
                             "targets": [{"id": i, "visible": True, "motion_seen": not clear} for i in self.ids]})
        self.pub.publish(m)


def main() -> None:
    rclpy.init()
    n = CameraStub()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
