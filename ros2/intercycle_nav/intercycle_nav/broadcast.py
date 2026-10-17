"""Broadcaster side of the episode feed (runs OUTSIDE the robot: the episode's observer process). No ROS.

`Broadcaster` signs and frames whatever an observer produces and pushes it through a `FeedWriter`, on the broadcast
schedule: map-layer carousel at `frame_hz`, fast layer every `fast_every_s`, slow layer every `slow_every_s`.
The observer itself is the caller's business:
  condition "learned": the lead's `LearnedObserver.slow(t)` / `.fast(t)` + the learned LayeredMap (sim/observer_learned.py)
  condition "oracle":  the episode runner's oracle (truth -> SlowLayer/FastLayer), an upper bound labelled as such
  dev / wiring only:   `SyntheticSchedule` below (SYNTHETIC periodic cycles, no truth, never a result)
Time `t` is plant time (sim seconds since 00:00 of day 0); unix = DAY0_UNIX + t, as everywhere in the codec.

Key ids (WEAKNESSES C-W5): the header key_id is DERIVED from the signing key (codec.trust.key_id_of = sha256(vk)[:2]),
never the scenario's constant. `Broadcaster` writes key_id_of(its key) into every slow / fast header it signs, whatever
id the observer model carries, and refuses a map bundle whose manifest names another id; `pack_map_for_key` packs a
map under the derived id.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from nacl.signing import SigningKey

from intercycle.codec import ble
from intercycle.codec import maplayer as ML
from intercycle.codec.layout import FAST, SLOW, encode_fast, encode_slow
from intercycle.codec.sign import sign
from intercycle.codec.trust import key_id_of
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State

if TYPE_CHECKING:
    from intercycle.observer.mapping import LayeredMap


class FrameSink(Protocol):
    """Where signed frames go: `intercycle_nav.feed.FeedWriter`, or an in-memory writer in tests."""

    def frames(self, frs: list[bytes], /) -> object: ...


def pack_map_for_key(m: LayeredMap, key: SigningKey, version: int, **kw) -> ML.MapBundle:
    """ML.pack_map with the manifest's key_id = key_id_of(key) (the caller's map is not modified)."""
    m2 = copy.copy(m)
    m2.spec = dict(m.spec, key_id=key_id_of(key.verify_key))
    return ML.pack_map(m2, key, version, **kw)


class Broadcaster:
    def __init__(self, writer: FrameSink, key: SigningKey, bundle: ML.MapBundle | None = None,
                 slow_fn: Callable[[float], SlowLayer | None] | None = None,
                 fast_fn: Callable[[float], FastLayer | None] | None = None,
                 frame_hz: float = 20.0, fast_every_s: float = 0.5, slow_every_s: float = 10.0):
        self.w, self.key, self.key_id = writer, key, key_id_of(key.verify_key)
        self.slow_fn, self.fast_fn = slow_fn, fast_fn
        self.frame_hz, self.fast_every, self.slow_every = frame_hz, fast_every_s, slow_every_s
        self.bundle, self.loop, self.k = None, [], 0
        if bundle is not None:
            self.set_map(bundle)
        self.t_frames = self.t_fast = self.t_slow = None

    def set_map(self, bundle: ML.MapBundle) -> None:
        if bundle.manifest.key_id != self.key_id:
            raise ValueError(f"map bundle manifest key_id {bundle.manifest.key_id} != key_id_of(signing key) "
                             f"{self.key_id}: pack it with pack_map_for_key (C-W5)")
        self.bundle, self.loop, self.k = bundle, bundle.carousel(), 0

    def tick(self, t: float) -> None:
        if self.loop:
            if self.t_frames is None:
                self.t_frames = t
            n = int((t - self.t_frames) * self.frame_hz)
            if n > 0:
                for _ in range(min(n, 200)):
                    self.w.frames([self.loop[self.k % len(self.loop)]])
                    self.k += 1
                self.t_frames += n / self.frame_hz
        if self.slow_fn is not None and (self.t_slow is None or t - self.t_slow >= self.slow_every):
            s = self.slow_fn(t)
            if s is not None:
                s = replace(s, key_id=self.key_id)
                self.w.frames(ble.frame(sign(encode_slow(s), self.key), layer=SLOW))
            self.t_slow = t
        if self.fast_fn is not None and (self.t_fast is None or t - self.t_fast >= self.fast_every):
            f = self.fast_fn(t)
            if f is not None:
                f = replace(f, key_id=self.key_id)
                self.w.frames(ble.frame(sign(encode_fast(f), self.key), layer=FAST))
            self.t_fast = t


class SyntheticSchedule:
    """SYNTHETIC test feed (wiring / mask demos only, never a result): machine i runs perfectly periodic cycles of
    period P_i, busy fraction b_i, starting at t = offset_i (motion first, then idle). No ground truth involved."""

    def __init__(self, site_id: int, key_id: int, periods, busy, offsets=None, ttl_s: float = 4.0, conf: int = 12):
        self.site_id, self.key_id = site_id, key_id
        self.P, self.b = [float(p) for p in periods], [float(b) for b in busy]
        self.off = [0.0] * len(self.P) if offsets is None else [float(o) for o in offsets]
        self.ttl, self.conf, self.seq = ttl_s, conf, 0

    def models(self) -> list[MachineModel]:
        return [MachineModel(i, P, b, 0.005, 0.0, 0.0, self.conf) for i, (P, b) in enumerate(zip(self.P, self.b))]

    def slow(self, t: float) -> SlowLayer:
        return SlowLayer(self.site_id, self.key_id, DAY0_UNIX + t, [], self.models())

    def state_at(self, i: int, t: float) -> tuple[State, float, float]:
        P, b = self.P[i], self.b[i]
        ph = ((t - self.off[i]) / P) % 1.0
        if ph < b:
            return State.ACTIVE, (b - ph) * P, (1 - b) * P
        return State.IDLE, 0.0, (1 - ph) * P

    def fast(self, t: float) -> FastLayer:
        self.seq = (self.seq + 1) & 0xFFFF
        ms = []
        for i in range(len(self.P)):
            st, nin, nlen = self.state_at(i, t)
            ms.append(MachineNow(i, st, nin, nlen, self.ttl, self.conf, self.seq))
        return FastLayer(self.site_id, self.key_id, DAY0_UNIX + t, self.seq, ms)
