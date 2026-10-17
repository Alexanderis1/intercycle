"""observer_bridge: signed observer broadcast -> Nav2 costmap-filter masks (DIRECTION 5.3). ROS 2 Humble, python 3.10.

    python3 -m intercycle_nav.observer_bridge --ros-args -p use_sim_time:=true -p condition:=learned \
        -p feed:=unix:///tmp/ic/observer.sock -p plant_t0:=36000.0

Publishes (all latched: reliable + transient_local, as Nav2's costmap filters expect):
  /intercycle/keepout_filter_info  nav2_msgs/CostmapFilterInfo  type 0 (keepout), mask /intercycle/keepout_mask
  /intercycle/keepout_mask         nav_msgs/OccupancyGrid       0 free / 100 keep-out, frame map
  /intercycle/speed_filter_info    nav2_msgs/CostmapFilterInfo  type 1 (speed limit, % of max), base 0, multiplier 1
  /intercycle/speed_mask           nav_msgs/OccupancyGrid       percent of max speed, 0 = no limit
  /intercycle/observer/state       std_msgs/String (JSON)       verified view: map, per-machine state/phase/idle windows,
                                                                 production state, rejection counters (for timing_planner)
Parameters: condition (none | static | learned | oracle), feed (transport url, see feed.py), plant_t0 (plant seconds at
ROS time 0), rate_hz, horizon_s, step_s, k_sigma, pad_bins, inflate_m, static_margin_m, self_clear_r, mask_origin
[x, y], mask_size [w, h] (m), mask_res, keyring ("key_id:hex,..."; default = the public DEMO key of the site's key_id),
scenario (site-map YAML path; default scenario/bodyshop.yaml), log_path (JSONL; default $IC_OUT/observer_bridge.jsonl).

The robot pose for the HELD rule is the localisation estimate (TF map -> base_link from AMCL), never simulator truth.
"learned" and "oracle" are identical here: the feed differs (episode side). No ground-truth topic is read.
"""
from __future__ import annotations

import json
import os
import time

import rclpy
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from intercycle_nav import masks as K
from intercycle_nav.feed import open_transport

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ObserverBridge(Node):
    def __init__(self) -> None:
        super().__init__("observer_bridge")
        dp = self.declare_parameter
        self.condition = dp("condition", "learned").value
        if self.condition not in K.CONDITIONS:
            raise ValueError(f"condition must be one of {K.CONDITIONS}")
        self.plant_t0 = float(dp("plant_t0", 0.0).value)
        rate = float(dp("rate_hz", 2.0).value)
        self.p = K.MaskParams(horizon_s=float(dp("horizon_s", 6.0).value), step_s=float(dp("step_s", 0.5).value),
                              k_sigma=float(dp("k_sigma", 2.0).value), pad_bins=int(dp("pad_bins", 1).value),
                              inflate_m=float(dp("inflate_m", 0.2).value),
                              static_margin_m=float(dp("static_margin_m", 0.15).value),
                              self_clear_r=float(dp("self_clear_r", 0.75).value))
        ox, oy = (float(v) for v in dp("mask_origin", [-2.0, -2.0]).value)
        w, h = (float(v) for v in dp("mask_size", [124.0, 52.0]).value)
        res = float(dp("mask_res", 0.1).value)          # = global costmap resolution (box edges snap outward by < 1 cell)
        self.grid = K.Grid(ox, oy, res, int(round(w / res)), int(round(h / res)))
        scn = dp("scenario", "").value or None
        self.site = K.Site.from_scenario(scn)
        kr = dp("keyring", "").value
        self.state = K.ObserverState(self.site, K.keyring_from_hex(kr) if kr else None)
        self.feed_url = dp("feed", "none").value
        self.transport = open_transport(self.feed_url if self.condition in ("learned", "oracle") else "none")
        out = os.environ.get("IC_OUT", "")
        log_path = dp("log_path", os.path.join(out, "observer_bridge.jsonl") if out else "").value
        self.log = open(log_path, "a") if log_path else None
        K.assert_no_truth()

        self.pub_kinfo = self.create_publisher(CostmapFilterInfo, "/intercycle/keepout_filter_info", LATCHED)
        self.pub_kmask = self.create_publisher(OccupancyGrid, "/intercycle/keepout_mask", LATCHED)
        self.pub_sinfo = self.create_publisher(CostmapFilterInfo, "/intercycle/speed_filter_info", LATCHED)
        self.pub_smask = self.create_publisher(OccupancyGrid, "/intercycle/speed_mask", LATCHED)
        self.pub_state = self.create_publisher(String, "/intercycle/observer/state", 10)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self._info_sent = False
        self._last = {"k": None, "s": None}
        self._last_pub_wall = 0.0
        self.n_pub = 0
        self.create_timer(0.02, self._poll)
        self.create_timer(1.0 / rate, self._update)
        self.get_logger().info(f"observer_bridge condition={self.condition} feed={self.feed_url} "
                               f"grid={self.grid.nx}x{self.grid.ny}@{res} H={self.p.horizon_s}s")

    # ---------------------------------------------------------------- time / pose
    def plant_t(self) -> float:
        return self.plant_t0 + self.get_clock().now().nanoseconds * 1e-9

    def robot_xy(self):
        try:
            tr = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return float(tr.transform.translation.x), float(tr.transform.translation.y)
        except Exception:  # noqa: BLE001 - no localisation yet
            return None

    # ---------------------------------------------------------------- input
    def _poll(self) -> None:
        recs = self.transport.poll()
        if recs:
            t = self.plant_t()
            for r in recs:
                self.state.ingest(t, r)

    # ---------------------------------------------------------------- output
    def _grid_msg(self, g, stamp) -> OccupancyGrid:
        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.info.resolution = float(self.grid.res)
        m.info.width, m.info.height = self.grid.nx, self.grid.ny
        m.info.origin.position.x, m.info.origin.position.y = float(self.grid.x0), float(self.grid.y0)
        m.info.origin.orientation.w = 1.0
        m.data = K.to_occupancy_data(g)
        return m

    def _send_info(self) -> None:
        for pub, typ, topic in ((self.pub_kinfo, 0, "/intercycle/keepout_mask"), (self.pub_sinfo, 1, "/intercycle/speed_mask")):
            i = CostmapFilterInfo()
            i.header.frame_id = "map"
            i.header.stamp = self.get_clock().now().to_msg()
            i.type, i.filter_mask_topic, i.base, i.multiplier = typ, topic, 0.0, 1.0
            pub.publish(i)
        self._info_sent = True

    def _update(self) -> None:
        if not self._info_sent:
            self._send_info()
        t = self.plant_t()
        xy = self.robot_xy()
        snap = view = None
        if self.condition in ("learned", "oracle"):
            snap = self.state.snapshot(t, robot_xy=xy, k_sigma=self.p.k_sigma)
            view = self.state.map_view(t)
        else:
            snap = self.state.snapshot(t, robot_xy=xy)         # empty view; keeps the held flag for the log
        kg, kinfo = K.keepout_mask(self.grid, self.condition, self.site, snap if view is not None else None, view, self.p,
                                   robot_xy=xy if self._robot_in_keepout(xy) else None)
        sg, sinfo = K.speed_mask(self.grid, self.condition, snap if view is not None else None, view, self.p)
        stamp = self.get_clock().now().to_msg()
        kb, sb = kg.tobytes(), sg.tobytes()
        now_wall = time.time()
        if kb != self._last["k"] or sb != self._last["s"] or now_wall - self._last_pub_wall > 5.0:
            self.pub_kmask.publish(self._grid_msg(kg, stamp))
            self.pub_smask.publish(self._grid_msg(sg, stamp))
            self._last = {"k": kb, "s": sb}
            self._last_pub_wall = now_wall
            self.n_pub += 1
        js = K.snapshot_json(snap, self.condition, {"keepout": kinfo, "speed": sinfo, "robot_xy": xy,
                                                    "stats": self.state.all_stats(), "n_mask_pub": self.n_pub,
                                                    "ros_t": self.get_clock().now().nanoseconds * 1e-9})
        msg = String()
        msg.data = json.dumps(js)
        self.pub_state.publish(msg)
        if self.log is not None:
            self.log.write(msg.data + "\n")
            self.log.flush()

    def _robot_in_keepout(self, xy) -> bool:
        return xy is not None and self.site.zone_of(*xy, pad=self.p.static_margin_m + 0.5) is not None


def main() -> None:
    rclpy.init()
    node = ObserverBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.transport.close()
        if node.log:
            node.log.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
