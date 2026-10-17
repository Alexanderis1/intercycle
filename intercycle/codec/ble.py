"""BLE 5 extended-advertising transport: (a) framing, (b) a deterministic channel sim, (c) the robot-side receiver.

Frame = one AD structure, Manufacturer Specific Data (0xFF), company ID 0xFFFF (reserved for testing), <= 251 B AdvData:
  len(1) 0xFF(1) company(2, LE) | msg_id 16 | layer 8 | frag_idx 4 | frag_count 4 | crc8(1) | fragment (<= 242 B)
crc8 (poly 0x07) covers the 4 fragment-header bytes + the fragment. Time arguments are sim seconds (unix = DAY0_UNIX + t).
"""
from __future__ import annotations

import math
import struct
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from intercycle.codec.layout import FAST, SLOW, CodecError, decode_fast, decode_slow, peek_header
from intercycle.codec.sign import Keyring, verify
from intercycle.scenario import Ble
from intercycle.types import DAY0_UNIX, Decoded, FastLayer, MachineNow, SlowLayer, State

AD_MFR = 0xFF
COMPANY = 0xFFFF
ADV_MAX = 251                            # AdvData bytes per extended-advertising frame
FRAG_OVERHEAD = 4 + 5                    # AD len/type/company + frag header (msg_id, layer, idx|count, crc)
FRAG_DATA = ADV_MAX - FRAG_OVERHEAD      # 242 B of signed blob per frame
MAX_FRAGS = 15
SLOW_MAX_AGE_S = 8 * 3600


def _crc8_table() -> list[int]:
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return t


_CRC8 = _crc8_table()


def crc8(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc = _CRC8[crc ^ b]
    return crc


# ---- (a) framing ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Frag:
    msg_id: int
    layer: int
    idx: int
    count: int
    data: bytes


def frame(blob: bytes, msg_id: int | None = None, layer: int | None = None) -> list[bytes]:
    """Split a signed blob into AdvData frames. msg_id defaults to crc32(blob) & 0xFFFF, layer to the blob header's."""
    if not blob:
        raise CodecError("empty blob")
    n = -(-len(blob) // FRAG_DATA)
    if n > MAX_FRAGS:
        raise CodecError(f"{len(blob)} B needs {n} frames (max {MAX_FRAGS})")
    layer = peek_header(blob).layer if layer is None else layer
    msg_id = zlib.crc32(blob) & 0xFFFF if msg_id is None else msg_id
    out = []
    for i in range(n):
        data = blob[i * FRAG_DATA:(i + 1) * FRAG_DATA]
        hdr = struct.pack(">HBB", msg_id, layer, i << 4 | n)
        body = hdr + bytes([crc8(hdr + data)]) + data
        out.append(struct.pack("<BBH", len(body) + 3, AD_MFR, COMPANY) + body)
    return out


def parse_frame(f: bytes) -> Frag | None:
    """None if the frame is not ours or fails its CRC."""
    if not FRAG_OVERHEAD <= len(f) <= ADV_MAX or f[0] != len(f) - 1 or f[1] != AD_MFR:
        return None
    if struct.unpack_from("<H", f, 2)[0] != COMPANY:
        return None
    msg_id, layer, ic = struct.unpack_from(">HBB", f, 4)
    idx, count = ic >> 4, ic & 15
    data = f[FRAG_OVERHEAD:]
    if count == 0 or idx >= count or crc8(f[4:8] + data) != f[8]:
        return None
    return Frag(msg_id, layer, idx, count, data)


class Reassembler:
    """Collects fragments per (layer, msg_id); an incomplete message is dropped timeout_s after its first fragment."""

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.dropped = 0                 # incomplete messages discarded on timeout
        self.bad_frames = 0              # not ours, or CRC failure
        self._buf: dict[tuple[int, int], tuple[float, int, dict[int, bytes]]] = {}

    def purge(self, t: float) -> None:
        for k in [k for k, (t0, _, _) in self._buf.items() if t - t0 > self.timeout_s]:
            del self._buf[k]
            self.dropped += 1

    def ingest(self, t: float, frame_: bytes) -> bytes | None:
        """Returns the reassembled blob when the last fragment arrives, else None."""
        self.purge(t)
        fr = parse_frame(frame_)
        if fr is None:
            self.bad_frames += 1
            return None
        if fr.count == 1:
            return fr.data
        key = (fr.layer, fr.msg_id)
        t0, count, parts = self._buf.get(key, (t, fr.count, {}))
        if count != fr.count:            # msg_id collision: restart
            t0, count, parts = t, fr.count, {}
        parts[fr.idx] = fr.data
        if len(parts) < count:
            self._buf[key] = (t0, count, parts)
            return None
        self._buf.pop(key, None)
        return b"".join(parts[i] for i in range(count))


# ---- (b) channel sim -----------------------------------------------------------------------------------

_M64 = (1 << 64) - 1


def _u01(*keys: int) -> float:
    """Counter-based uniform [0,1): splitmix64 chained over keys, so every draw depends only on (seed, stream, b, k)."""
    h = 0
    for k in keys:
        h = (h + (k & _M64) + 0x9E3779B97F4A7C15) & _M64
        h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _M64
        h ^= h >> 31
    return (h >> 11) / (1 << 53)


FramesSource = Mapping[int, Sequence[bytes]] | Sequence[bytes] | Callable[[int, float], Sequence[bytes]]
XY = tuple[float, float]


class BleChannel:
    """Fixed broadcasters advertise one frame per interval (round-robin over their frame list). A robot within range_m
    hears each advertisement with probability 1 - packet_loss (iid), delayed by 20..80 ms. Pure function of
    (seed, broadcaster, event index): results do not depend on how [t0, t1) is chunked across calls."""

    ADV_DELAY_S = 0.010                  # BLE advDelay: random 0..10 ms added to each event
    LATENCY_S = (0.020, 0.080)

    def __init__(self, ble: Ble, seed: int = 0) -> None:
        self.pos = [(float(x), float(z)) for x, z in ble.broadcasters]
        self.interval = ble.adv_interval_s
        self.range_m = ble.range_m
        self.loss = ble.packet_loss
        self.seed = seed
        self.phase = [_u01(seed, 0, b) * self.interval for b in range(len(self.pos))]

    def in_range(self, xy: XY) -> list[int]:
        return [b for b, (x, z) in enumerate(self.pos) if math.hypot(x - xy[0], z - xy[1]) <= self.range_m]

    def receive(self, robot_xy: XY | Callable[[float], XY], t0: float, t1: float,
                frames_by_broadcaster: FramesSource) -> list[tuple[float, bytes]]:
        """(t_rx, frame) for every advertisement from an in-range broadcaster that arrives in [t0, t1), sorted by t_rx.
        frames_by_broadcaster: {b: [frames]}, one list for all broadcasters, or f(b, t_tx) -> [frames]."""
        lo, hi = self.LATENCY_S
        out: list[tuple[float, bytes]] = []
        for b in range(len(self.pos)):
            k0 = math.floor((t0 - self.phase[b] - self.ADV_DELAY_S - hi) / self.interval)
            k1 = math.ceil((t1 - self.phase[b] - lo) / self.interval)
            for k in range(k0, k1 + 1):
                t_tx = self.phase[b] + k * self.interval + _u01(self.seed, 3, b, k) * self.ADV_DELAY_S
                t_rx = t_tx + lo + _u01(self.seed, 2, b, k) * (hi - lo)
                if not t0 <= t_rx < t1 or _u01(self.seed, 1, b, k) < self.loss:
                    continue
                xy = robot_xy(t_rx) if callable(robot_xy) else robot_xy
                if math.hypot(self.pos[b][0] - xy[0], self.pos[b][1] - xy[1]) > self.range_m:
                    continue
                if callable(frames_by_broadcaster):
                    frames = frames_by_broadcaster(b, t_tx)
                elif isinstance(frames_by_broadcaster, Mapping):
                    frames = frames_by_broadcaster.get(b, ())
                else:
                    frames = frames_by_broadcaster
                if frames:
                    out.append((t_rx, frames[k % len(frames)]))
        out.sort(key=lambda e: e[0])
        return out


# ---- (c) receiver --------------------------------------------------------------------------------------

def _newer16(seq: int, last: int) -> bool:
    return 0 < (seq - last) & 0xFFFF < 0x8000


def _expire(f: FastLayer | None, now: float) -> tuple[FastLayer | None, list[int]]:
    """Entries with issue_time + ttl < now become UNKNOWN (kept, not dropped); returns (layer, expired indices)."""
    if f is None:
        return None, []
    ms, gone = [], []
    for m in f.machines:
        if now > f.issue_time + m.ttl_s:
            ms.append(MachineNow(m.index, State.UNKNOWN, 0.0, 0.0, 0.0, 0, m.seq))
            gone.append(m.index)
        else:
            ms.append(m)
    return replace(f, machines=ms), gone


class Receiver:
    """Verifies and tracks the newest slow and fast layers heard. Rejections never touch the accepted state."""

    def __init__(self, keyring: Keyring, site_id: int, timeout_s: float = 5.0,
                 slow_max_age_s: float = SLOW_MAX_AGE_S) -> None:
        self.keyring, self.site_id, self.slow_max_age_s = keyring, site_id, slow_max_age_s
        self.reasm = Reassembler(timeout_s)
        self.stats: Counter[str] = Counter()
        self._slow: SlowLayer | None = None
        self._fast: FastLayer | None = None

    def ingest(self, t_rx: float, frame_: bytes) -> Decoded | None:
        """None while a message is incomplete (or the frame is junk); a Decoded once a whole blob has been judged."""
        blob = self.reasm.ingest(t_rx, frame_)
        return None if blob is None else self.accept(t_rx, blob)

    def accept(self, t_rx: float, blob: bytes) -> Decoded:
        """Judge one reassembled signed blob. Order: signature, parse, site, freshness (slow age / seq)."""
        ok, payload, key_id, reason = verify(blob, self.keyring)
        if not ok:
            return self._rej(reason, key_id=key_id)
        now = DAY0_UNIX + t_rx
        try:
            h = peek_header(payload)
            if h.site_id != self.site_id:
                return self._rej("wrong_site", site_id=h.site_id)
            if h.layer == SLOW:
                s = decode_slow(payload)
                age = now - s.issue_time
                if age > self.slow_max_age_s:
                    return self._rej("expired", layer="slow", age_s=age)
                if self._slow is not None and s.issue_time <= self._slow.issue_time:
                    return self._rej("stale_seq", layer="slow", duplicate=s.issue_time == self._slow.issue_time)
                self._slow = s
                self.stats["ok_slow"] += 1
                return Decoded(True, slow=s, extra={"layer": "slow", "key_id": key_id, "t_rx": t_rx})
            if h.layer == FAST:
                f = decode_fast(payload)
                if self._fast is not None and not _newer16(f.seq, self._fast.seq):
                    return self._rej("stale_seq", layer="fast", seq=f.seq, last_seq=self._fast.seq,
                                     duplicate=f.seq == self._fast.seq)
                self._fast = f
                self.stats["ok_fast"] += 1
                view, gone = _expire(f, now)
                return Decoded(True, fast=view, extra={"layer": "fast", "key_id": key_id, "t_rx": t_rx,
                                                       "expired_fast": gone})
        except CodecError as e:
            return self._rej("malformed", error=str(e))
        return self._rej("malformed", error=f"unknown layer {h.layer}")

    def latest(self, t_now: float) -> Decoded:
        """Best current view: newest verified slow (None once older than 8 h) and fast (expired entries -> UNKNOWN)."""
        now = DAY0_UNIX + t_now
        slow, extra = self._slow, {}
        if slow is not None:
            extra["slow_age_s"] = now - slow.issue_time
            if extra["slow_age_s"] > self.slow_max_age_s:
                slow, extra["slow_expired"] = None, True
        fast, gone = _expire(self._fast, now)
        if fast is not None:
            extra["fast_age_s"] = now - fast.issue_time
            extra["expired_fast"] = gone
        if slow is None and fast is None:
            return Decoded(False, "expired" if extra.get("slow_expired") else "no_data", extra=extra)
        return Decoded(True, slow=slow, fast=fast, extra=extra)

    def _rej(self, reason: str, **extra) -> Decoded:
        self.stats[reason] += 1
        return Decoded(False, reason, extra=extra)
