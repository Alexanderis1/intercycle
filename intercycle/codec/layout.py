"""Bit-exact wire layout of the two broadcast layers. Big-endian, MSB-first, no padding.

Common header (12 B): magic 0x4943 "IC" 16 | version 4 (=1) | layer 4 (0 slow, 1 fast) | site_id 24 | key_id 16 | issue_time 32 (unix s)
  slow header (14 B): + zone_count 8 | machine_count 8
  fast header (16 B): + issue_ms 16 (0..999) | seq 16
Slow body: traffic [zone][24 h] 4 bit (12 B/zone), then 8 B per machine:
  index 6 | period 17 (ms; x10 flag -> 10 ms units) | busy 6 (v/63) | jitter 4 (log) | drift 6 (signed, 0.01 %/h)
  | stops_per_h 4 (log) | confidence 4 | x10 flag 1 | reserved 16
Fast body: 8 B per machine (count = body length / 8):
  index 6 | state 2 | next_idle_in 17 (ms) | next_idle_len 11 (0.1 s) | ttl 8 (2 s, floor) | confidence 4 | reserved 16
Signature (Ed25519, 64 B) is appended by sign.py. Fast MachineNow.seq is the layer seq (not stored per machine).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from intercycle.types import FastLayer, MachineModel, MachineNow, SlowLayer, State

MAGIC = 0x4943
VERSION = 1
SLOW, FAST = 0, 1
HDR_LEN, SLOW_HDR_LEN, FAST_HDR_LEN = 12, 14, 16
REC_LEN, ZONE_LEN, HOURS = 8, 12, 24
PERIOD_MAX_MS = (1 << 17) - 1            # 131.071 s; x10 flag extends to 1310.71 s at 10 ms
IDLE_IN_MAX_MS = (1 << 17) - 1
IDLE_LEN_MAX = (1 << 11) - 1             # x 0.1 s = 204.7 s
TTL_MAX = 255                            # x 2 s = 510 s

# Worst-case |decoded - true| for in-range inputs (abs unless "rel"). Out-of-range inputs saturate (fast, drift, jitter, stops).
TOL = {
    "period_s": 0.0005,                  # 5 ms when period > 131.071 s (x10 mode)
    "busy_fraction": 1 / 126,
    "jitter_rel": 2 ** 0.25 - 1,         # log step 2^(1/2): 18.9 % relative for 0.119 %..18 %; [0.1 %, 0.119 %) reads 0.141 % (+41 %,
                                         # conservative) because code 0 is reserved for "< 0.1 %", which decodes as 0.05 %
    "jitter_abs_low": 5e-4,
    "drift_per_h": 5e-5,                 # +-0.32 %/h range
    "stops_rel": 2 ** 0.25 - 1,          # 0.25..32 /h; below 0.125 /h decodes as 0
    "stops_abs_low": 0.125,
    "slow_issue_s": 0.5,
    "fast_issue_s": 1e-6,
    "next_idle_in_s": 0.0005,
    "next_idle_len_s": 0.05,
    "ttl_s": 2.0,                        # one-sided: decoded in (true - 2 s, true] because ttl rounds down
}


class CodecError(ValueError):
    """Malformed or unrepresentable data."""


class BitWriter:
    def __init__(self) -> None:
        self._v, self._n = 0, 0

    def put(self, v: int, n: int) -> None:
        if not 0 <= v < 1 << n:
            raise CodecError(f"value {v} does not fit {n} bits")
        self._v = (self._v << n) | v
        self._n += n

    def put_signed(self, v: int, n: int) -> None:
        if not -(1 << (n - 1)) <= v < 1 << (n - 1):
            raise CodecError(f"value {v} does not fit signed {n} bits")
        self.put(v & ((1 << n) - 1), n)

    def to_bytes(self) -> bytes:
        pad = -self._n % 8
        return (self._v << pad).to_bytes((self._n + pad) // 8, "big")


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._v, self._n, self._p = int.from_bytes(data, "big"), len(data) * 8, 0

    def get(self, n: int) -> int:
        if self._p + n > self._n:
            raise CodecError("truncated")
        self._p += n
        return (self._v >> (self._n - self._p)) & ((1 << n) - 1)

    def get_signed(self, n: int) -> int:
        v = self.get(n)
        return v - (1 << n) if v >> (n - 1) else v

    @property
    def left(self) -> int:
        return self._n - self._p


# ---- field quantizers (value <-> code); the encoder, decoder and quantize_* all use these -------------

def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def period_code(s: float) -> tuple[int, int]:
    """(x10 flag, 17-bit code). ms resolution up to 131.071 s, else 10 ms resolution up to 1310.71 s."""
    if s < 0:
        raise CodecError(f"negative period {s}")
    ms = round(s * 1000)
    if ms <= PERIOD_MAX_MS:
        return 0, ms
    c = round(s * 100)
    if c > PERIOD_MAX_MS:
        raise CodecError(f"period {s} s not representable (max {PERIOD_MAX_MS / 100} s)")
    return 1, c


def period_val(flag: int, c: int) -> float:
    return c / (100 if flag else 1000)


def busy_code(f: float) -> int:
    return _clip(round(f * 63), 0, 63)


def busy_val(c: int) -> float:
    return c / 63


def jitter_code(s: float) -> int:
    """sigma = 0.1 % * 2^(v/2); v = 0 means below 0.1 % (decoded 0.05 %); saturates at v = 15 (18.1 %)."""
    return 0 if s < 1e-3 else _clip(round(2 * math.log2(s / 1e-3)), 1, 15)


def jitter_val(c: int) -> float:
    return 5e-4 if c == 0 else 1e-3 * 2 ** (c / 2)


def drift_code(d: float) -> int:
    """Signed 6-bit, 0.01 %/h steps, saturating at -0.32 .. +0.31 %/h."""
    return _clip(round(d / 1e-4), -32, 31)


def drift_val(c: int) -> float:
    return c * 1e-4


def stops_code(s: float) -> int:
    """0 = none (< 0.125 /h); else 0.25 * 2^((v-1)/2) per hour, v = 1..15 (0.25 .. 32 /h)."""
    return 0 if s < 0.125 else _clip(1 + round(2 * math.log2(s / 0.25)), 1, 15)


def stops_val(c: int) -> float:
    return 0.0 if c == 0 else 0.25 * 2 ** ((c - 1) / 2)


def idle_in_code(s: float) -> int:
    return _clip(round(s * 1000), 0, IDLE_IN_MAX_MS)


def idle_len_code(s: float) -> int:
    return _clip(round(s * 10), 0, IDLE_LEN_MAX)


def ttl_code(s: float) -> int:
    # Round DOWN: a robot must never trust an entry longer than the observer intended.
    return _clip(math.floor(s / 2 + 1e-9), 0, TTL_MAX)


def _conf(c: int) -> int:
    return _clip(int(c), 0, 15)


def _split_ms(t: float) -> tuple[int, int]:
    return divmod(round(t * 1000), 1000)


def quantize_model(m: MachineModel) -> MachineModel:
    """What the robot sees for this machine after a trip through the wire format."""
    return MachineModel(
        m.index, period_val(*period_code(m.period_s)), busy_val(busy_code(m.busy_fraction)),
        jitter_val(jitter_code(m.jitter)), drift_val(drift_code(m.drift_per_h)),
        stops_val(stops_code(m.stops_per_h)), _conf(m.confidence))


def quantize_now(m: MachineNow, seq: int | None = None) -> MachineNow:
    return MachineNow(
        m.index, State(m.state), idle_in_code(m.next_idle_in_s) / 1000, idle_len_code(m.next_idle_len_s) / 10,
        ttl_code(m.ttl_s) * 2.0, _conf(m.confidence), m.seq if seq is None else seq)


def quantize_slow(s: SlowLayer) -> SlowLayer:
    return SlowLayer(s.site_id, s.key_id, float(round(s.issue_time)), [list(r) for r in s.traffic],
                     [quantize_model(m) for m in s.machines])


def quantize_fast(f: FastLayer) -> FastLayer:
    sec, ms = _split_ms(f.issue_time)
    return FastLayer(f.site_id, f.key_id, sec + ms / 1000, f.seq, [quantize_now(m, f.seq) for m in f.machines])


# ---- sizes ---------------------------------------------------------------------------------------------

def slow_len(zones: int, machines: int) -> int:
    return SLOW_HDR_LEN + ZONE_LEN * zones + REC_LEN * machines


def fast_len(machines: int) -> int:
    return FAST_HDR_LEN + REC_LEN * machines


# ---- header --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Header:
    layer: int
    site_id: int
    key_id: int
    issue_s: int


def _put_hdr(w: BitWriter, layer: int, site_id: int, key_id: int, issue_s: int) -> None:
    w.put(MAGIC, 16)
    w.put(VERSION, 4)
    w.put(layer, 4)
    w.put(site_id, 24)
    w.put(key_id, 16)
    w.put(issue_s, 32)


def _get_hdr(r: BitReader) -> Header:
    if r.get(16) != MAGIC:
        raise CodecError("bad magic")
    if r.get(4) != VERSION:
        raise CodecError("unsupported version")
    layer = r.get(4)
    return Header(layer, r.get(24), r.get(16), r.get(32))


def peek_header(data: bytes) -> Header:
    """Parse the unauthenticated common header (key_id selects the verify key; trust nothing else yet)."""
    if len(data) < HDR_LEN:
        raise CodecError("truncated header")
    return _get_hdr(BitReader(data[:HDR_LEN]))


# ---- slow ----------------------------------------------------------------------------------------------

def encode_slow(s: SlowLayer) -> bytes:
    if len(s.traffic) > 255 or len(s.machines) > 255:
        raise CodecError("too many zones/machines")
    w = BitWriter()
    _put_hdr(w, SLOW, s.site_id, s.key_id, round(s.issue_time))
    w.put(len(s.traffic), 8)
    w.put(len(s.machines), 8)
    for row in s.traffic:
        if len(row) != HOURS:
            raise CodecError(f"traffic row needs {HOURS} hours, got {len(row)}")
        for v in row:
            w.put(v, 4)
    for m in s.machines:
        flag, pc = period_code(m.period_s)
        w.put(m.index, 6)
        w.put(pc, 17)
        w.put(busy_code(m.busy_fraction), 6)
        w.put(jitter_code(m.jitter), 4)
        w.put_signed(drift_code(m.drift_per_h), 6)
        w.put(stops_code(m.stops_per_h), 4)
        w.put(_conf(m.confidence), 4)
        w.put(flag, 1)
        w.put(0, 16)
    return w.to_bytes()


def decode_slow(b: bytes) -> SlowLayer:
    r = BitReader(b)
    h = _get_hdr(r)
    if h.layer != SLOW:
        raise CodecError("not a slow layer")
    zc, mc = r.get(8), r.get(8)
    if len(b) != slow_len(zc, mc):
        raise CodecError(f"slow length {len(b)} != {slow_len(zc, mc)}")
    traffic = [[r.get(4) for _ in range(HOURS)] for _ in range(zc)]
    machines = []
    for _ in range(mc):
        idx, pc, busy, jit = r.get(6), r.get(17), r.get(6), r.get(4)
        drift, stops, conf = r.get_signed(6), r.get(4), r.get(4)
        flag = r.get(1)
        r.get(16)
        machines.append(MachineModel(idx, period_val(flag, pc), busy_val(busy), jitter_val(jit),
                                     drift_val(drift), stops_val(stops), conf))
    return SlowLayer(h.site_id, h.key_id, float(h.issue_s), traffic, machines)


# ---- fast ----------------------------------------------------------------------------------------------

def encode_fast(f: FastLayer) -> bytes:
    sec, ms = _split_ms(f.issue_time)
    w = BitWriter()
    _put_hdr(w, FAST, f.site_id, f.key_id, sec)
    w.put(ms, 16)
    w.put(f.seq, 16)
    for m in f.machines:
        w.put(m.index, 6)
        w.put(int(State(m.state)), 2)
        w.put(idle_in_code(m.next_idle_in_s), 17)
        w.put(idle_len_code(m.next_idle_len_s), 11)
        w.put(ttl_code(m.ttl_s), 8)
        w.put(_conf(m.confidence), 4)
        w.put(0, 16)
    return w.to_bytes()


def decode_fast(b: bytes) -> FastLayer:
    r = BitReader(b)
    h = _get_hdr(r)
    if h.layer != FAST:
        raise CodecError("not a fast layer")
    ms, seq = r.get(16), r.get(16)
    if ms > 999:
        raise CodecError("issue_ms > 999")
    if len(b) < FAST_HDR_LEN or (len(b) - FAST_HDR_LEN) % REC_LEN:
        raise CodecError(f"fast length {len(b)} not {FAST_HDR_LEN} + {REC_LEN}n")
    machines = []
    for _ in range((len(b) - FAST_HDR_LEN) // REC_LEN):
        idx, st, nin, nlen, ttl, conf = r.get(6), r.get(2), r.get(17), r.get(11), r.get(8), r.get(4)
        r.get(16)
        machines.append(MachineNow(idx, State(st), nin / 1000, nlen / 10, ttl * 2.0, conf, seq))
    return FastLayer(h.site_id, h.key_id, h.issue_s + ms / 1000, seq, machines)
