"""devfeed: DEV broadcaster for wiring tests (episode side, NOT the robot; never a result).

Signs and broadcasts, into the bridge's feed, (a) a learned LayeredMap (`map_npz`, e.g. out/layered_map.npz from
scripts/build_map.py) as the signed map layer carousel and (b) slow/fast layers from `SyntheticSchedule`: perfectly
periodic SYNTHETIC cycles (period from the map, busy fraction from the public scenario file). Paced by /clock.
Options to exercise the bridge: forge_every (also send a forged fast layer, wrong key, every N s),
stop_fast_after_s (stop the fast layer: entries must go stale -> UNKNOWN).

The real broadcasters live outside this package: the lead's LearnedObserver (condition "learned") and the episode
runner's oracle (condition "oracle"); both use `broadcast.Broadcaster` the same way.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from intercycle import scenario as S
from intercycle.codec import ble
from intercycle.codec import maplayer as ML
from intercycle.codec.layout import FAST, encode_fast
from intercycle.codec.sign import demo_key, sign
from intercycle.observer import mapping as M
from intercycle.types import DAY0_UNIX

from intercycle_nav.broadcast import Broadcaster, SyntheticSchedule
from intercycle_nav.feed import FeedWriter


class DevFeed(Node):
    def __init__(self) -> None:
        super().__init__("devfeed")
        dp = self.declare_parameter
        self.plant_t0 = float(dp("plant_t0", 36000.0).value)
        url = dp("feed", "unix:///tmp/ic_observer.sock").value
        npz = dp("map_npz", "").value
        self.forge_every = float(dp("forge_every_s", 0.0).value)
        self.stop_fast_after = float(dp("stop_fast_after_s", 0.0).value)
        offs = [float(v) for v in dp("offsets", [0.0]).value]
        scn = S.load()
        self.key = demo_key(scn.key_id)
        bundle = None
        periods = [float(m.period_s or scn.line.takt_s) if not m.on_line else float(scn.line.takt_s) for m in scn.machines]
        if npz:
            lm = M.LayeredMap.load(npz)
            bundle = ML.pack_map(lm, self.key, version=int(dp("map_version", 1).value),
                                 issue_time=DAY0_UNIX + self.plant_t0 - 60.0)
            periods = [float(p) if np.isfinite(p) else q for p, q in zip(lm.period, periods)]
        busy = [float(m.busy) for m in scn.machines]
        offsets = (offs * len(busy))[:len(busy)] if len(offs) < len(busy) else offs
        self.sched = SyntheticSchedule(scn.site_id, scn.key_id, periods, busy, offsets)
        self.w = FeedWriter(url)
        self.bc = Broadcaster(self.w, self.key, bundle, self.sched.slow, self._fast)
        self.forger = demo_key(9999)
        self.t_forge = None
        self.create_timer(0.02, self._tick)
        self.get_logger().warn(f"devfeed SYNTHETIC schedule -> {url}; map={'yes' if bundle else 'no'} "
                               f"({bundle.stats['compressed_bytes'] if bundle else 0} B), periods={np.round(periods, 1).tolist()}")

    def plant_t(self) -> float:
        return self.plant_t0 + self.get_clock().now().nanoseconds * 1e-9

    def _fast(self, t: float):
        ros = t - self.plant_t0
        if self.stop_fast_after > 0 and ros > self.stop_fast_after:
            return None
        return self.sched.fast(t)

    def _tick(self) -> None:
        if self.get_clock().now().nanoseconds == 0:
            return
        t = self.plant_t()
        self.bc.tick(t)
        if self.forge_every > 0 and (self.t_forge is None or t - self.t_forge >= self.forge_every):
            f = self.sched.fast(t)            # a forged layer: every machine "IDLE for 200 s", signed with the wrong key
            from dataclasses import replace

            from intercycle.types import State
            ms = [replace(m, state=State.IDLE, next_idle_in_s=0.0, next_idle_len_s=200.0, ttl_s=300.0) for m in f.machines]
            fake = replace(f, machines=ms, seq=(f.seq + 7) & 0xFFFF)
            self.w.frames(ble.frame(sign(encode_fast(fake), self.forger), layer=FAST))
            self.t_forge = t


def main() -> None:
    rclpy.init()
    n = DevFeed()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.w.close()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
