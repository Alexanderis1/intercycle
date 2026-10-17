"""nav_probe: evidence logger. Does what Nav2 sees follow the observer masks? (read-only; no effect on the robot)

Logs at 1 Hz to $IC_OUT/nav_probe.jsonl:
  * the global costmap value (nav_msgs/OccupancyGrid translation of /global_costmap/costmap: 100 = lethal) at each
    hazard zone's inspection pose and zone centre, next to the bridge's verdict (zone open/closed, bins);
  * the latest /plan: length, number of poses inside any hazard-zone rectangle;
  * the latest /speed_limit (nav2_msgs/SpeedLimit from the SpeedFilter).
"""
from __future__ import annotations

import json
import math
import os

import rclpy
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from intercycle import scenario as S
from intercycle_nav.masks import Site
from intercycle_nav.timing import TimingCore


class NavProbe(Node):
    def __init__(self) -> None:
        super().__init__("nav_probe")
        self.site = Site.from_scenario()
        core = TimingCore.from_scenario(S.load())
        self.points = {}
        for mid, (x0, y0, x1, y1) in self.site.rects.items():
            self.points[f"{mid}:centre"] = ((x0 + x1) / 2, (y0 + y1) / 2)
            if mid in core.inside:
                self.points[f"{mid}:inside"] = core.graph.pos[core.inside[mid]]
        self.cm = self.plan = self.obs = self.speed = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", lambda m: setattr(self, "cm", m), qos)
        self.create_subscription(Path, "/plan", lambda m: setattr(self, "plan", m), 10)
        self.create_subscription(String, "/intercycle/observer/state", lambda m: setattr(self, "obs", m), 10)
        self.create_subscription(SpeedLimit, "/speed_limit", lambda m: setattr(self, "speed", m), 10)
        out = os.environ.get("IC_OUT", "/tmp")
        self.f = open(os.path.join(out, "nav_probe.jsonl"), "a")
        self.create_timer(1.0, self._tick)

    def _cost(self, x: float, y: float):
        m = self.cm
        if m is None:
            return None
        i = int((x - m.info.origin.position.x) / m.info.resolution)
        j = int((y - m.info.origin.position.y) / m.info.resolution)
        if not (0 <= i < m.info.width and 0 <= j < m.info.height):
            return None
        return int(m.data[j * m.info.width + i])

    def _tick(self) -> None:
        rec = {"ros_t": self.get_clock().now().nanoseconds * 1e-9,
               "cost": {k: self._cost(*xy) for k, xy in self.points.items()}}
        if self.plan is not None and self.plan.poses:
            pts = [(p.pose.position.x, p.pose.position.y) for p in self.plan.poses]
            rec["plan"] = {"n": len(pts), "length_m": round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), 2),
                           "goal": [round(v, 2) for v in pts[-1]],
                           "in_zone": {mid: n for mid in self.site.rects
                                       if (n := sum(self.site.zone_of(x, y) == mid for x, y in pts))}}
        if self.speed is not None:
            rec["speed_limit"] = {"percentage": bool(self.speed.percentage), "value": float(self.speed.speed_limit)}
        if self.obs is not None:
            try:
                o = json.loads(self.obs.data)
                rec["bridge"] = {"t": o.get("t"), "zones": {k: v.get("zone") for k, v in o.get("keepout", {}).get("bins", {}).items()},
                                 "keepout_cells": o.get("keepout", {}).get("keepout_cells"), "map_ok": o.get("map_ok")}
            except ValueError:
                pass
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


def main() -> None:
    rclpy.init()
    n = NavProbe()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.f.close()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
