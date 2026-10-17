"""Map layer (layer id 3): the observer's L1-L3 layered map as a signed, chunked BLE broadcast.

What is sent
  payload = zlib( meta | L1 | L2/L3 ), then cut into chunks of 240 B (one chunk = one 251 B extended-advertising frame).
  meta  : machine ids, local grid origin / size / cell count, learned period and jitter, site grid.
  L1    : per machine, per phase bin (16 + the STOP bin): greedy rectangle cover of the occupancy envelope
          (p >= p_min), each box 4 coordinates of `coord_bits` bits in local 0.25 m cells.
  L2/L3 : per class (vehicle, person) and production state, the 48 time-of-day slices in order, each coded as a sparse
          delta against the previous observed slice of that (class, state): changed cells as (LEB128 gap, byte);
          byte = 4-bit log-quantised probability (0 = below 0.02, 1..15 = 0.02 * 50^((c-1)/14), i.e. 2 % .. 100 %,
          max relative step 32 %) | 3-bit dominant heading (8 dirs) << 4 | heading-valid << 7.
          Hysteresis in the encoder keeps a cell unchanged when it moves by <= 1 code, so quiet slices cost ~nothing.
Authentication
  Manifest = common header (layer 3, same 12 B as the other layers) | map version | valid_s | raw/payload length |
  chunk count / size | parity group | SHA-256 of the payload | 4-byte hash of every chunk (data and parity), signed
  with Ed25519 (codec/sign.py) and sent as ordinary ble.frame() fragments (layer 3). Chunk frames carry no signature:
  they are accepted only if their hash matches the signed table, so a forged or corrupted chunk can never enter the map.
Carousel
  The broadcaster loops [manifest fragments, 16 chunks, manifest fragments, 16 chunks, ...] (periodic advertising
  train). Every `parity_group` data chunks are followed by an XOR parity chunk that repairs one loss per group.
  `MapReceiver` assembles whatever it hears, verifies, decodes and caches per (site, version); a cached version is
  never downloaded again, and partial downloads survive leaving and re-entering range.
Frame layout (chunk): len(1) 0xFF company(2, LE) | map_id 16 | layer 8 (=3) | 0x00 | chunk_idx 16 | crc8(1) | data (<= 240 B).
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from nacl.signing import SigningKey

from intercycle.codec import ble
from intercycle.codec.layout import MAGIC, VERSION, BitReader, BitWriter, CodecError, peek_header
from intercycle.codec.sign import Keyring, sign, verify
from intercycle.observer.mapping import N_BINS, N_PHASE, N_PROD, N_TOD, PERSON, VEHICLE, LayeredMap, MapView, box_cover
from intercycle.types import DAY0_UNIX

LAYER_MAP = 3
CHUNK_DATA = ble.ADV_MAX - 11          # 240 B of payload per frame
P_LO, P_RANGE = 0.02, 50.0             # code 1 = 2 %, code 15 = 100 %
CLASSES = (VEHICLE, PERSON)            # order of the L2 / L3 sections
MAGIC_PAYLOAD = b"ICMP"


# ------------------------------------------------------------------------------------------------ small helpers

def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, pos
        shift += 7


def p_to_code(p: np.ndarray) -> np.ndarray:
    """Probability -> 4-bit code (0 below 2 %, else 1..15 on a log scale up to 100 %)."""
    p = np.asarray(p, float)
    c = np.rint(1 + 14 * np.log(np.maximum(p, 1e-9) / P_LO) / math.log(P_RANGE))
    return np.where(p >= P_LO, np.clip(c, 1, 15), 0).astype(np.uint8)


def code_to_p(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    return np.where(c > 0, P_LO * P_RANGE ** ((np.maximum(c, 1) - 1) / 14.0), 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ payload encoder

def _coord_bits(m: LayeredMap) -> int:
    return max(4, int(max(max(mm["nx"], mm["nz"]) for mm in m.spec["machines"])).bit_length())


def encode_payload(m: LayeredMap, p_min: float = 0.05, max_boxes: int = 12, cover_frac: float = 0.985,
                   p_floor: float = P_LO, hyst_sigma: float = 2.5) -> tuple[bytes, dict]:
    """Uncompressed map payload and a size breakdown {section: bytes}."""
    sp = m.spec
    mm = sp["machines"]
    cb = _coord_bits(m)
    sizes = {}
    # ---- meta
    w = BitWriter()
    w.put(len(mm), 8)
    w.put(cb, 4)
    w.put(round(sp["env_res"] * 100), 8)
    s = sp["site"]
    w.put_signed(round(s["x0"] * 4), 16)
    w.put_signed(round(s["z0"] * 4), 16)
    w.put(round(s["res"] * 100), 12)
    w.put(s["nx"], 8)
    w.put(s["nz"], 8)
    for i, d in enumerate(mm):
        ident = d["id"].encode("ascii")
        w.put(len(ident), 4)
        for ch in ident:
            w.put(ch, 7)
        w.put_signed(round(d["x0"] / sp["env_res"]), 12)
        w.put_signed(round(d["z0"] / sp["env_res"]), 12)
        w.put(d["nx"], 8)
        w.put(d["nz"], 8)
        P = m.period[i]
        w.put(0 if not np.isfinite(P) else min(int(round(P * 100)), 65535), 16)
        J = m.jitter[i]
        w.put(0 if not np.isfinite(J) else min(int(round(J * 2000)), 255), 8)
    meta = w.to_bytes()
    sizes["meta"] = len(meta)
    # ---- L1: box covers
    w = BitWriter()
    for d in mm:
        for b in range(N_BINS):
            cbx = box_cover(m.envelope(d["id"], b, p_min), max_boxes, cover_frac)
            w.put(len(cbx), 5)
            for i0, j0, i1, j1 in cbx:
                for v in (i0, j0, i1, j1):
                    w.put(int(v), cb)
    l1 = w.to_bytes()
    sizes["L1"] = len(l1)
    # ---- L2 / L3
    out = bytearray()
    nx, nz = m.nx, m.nz
    cls_sizes = {}
    for c in CLASSES:
        start = len(out)
        for prod in range(N_PROD):
            prev_code = np.zeros((nx, nz), np.uint8)
            prev_head = np.full((nx, nz), -1, np.int8)
            have_prev = False
            for tod in range(N_TOD):
                if not m._observed(tod, prod, c):
                    out.append(0)
                    continue
                S = tod * N_PROD + prod
                hits, hist = m.hits[c][S], m.head[c][S]
                p, hd = m.traffic(c, tod, prod, fallback=False)
                tgt = p_to_code(np.where(p >= p_floor, p, 0.0))
                # hysteresis: a cell keeps its previous code unless the new estimate differs by more than the larger of the
                # quantisation step and `hyst_sigma` Poisson sigmas of its hit count (cells seen a few times stay put)
                tol = np.maximum(0.42, hyst_sigma / np.sqrt(np.maximum(hits, 1.0)))
                dev = np.abs(np.log(np.maximum(p, 1e-9) / np.maximum(code_to_p(prev_code), 1e-9)))
                tgt = np.where((prev_code > 0) & (tgt > 0) & (dev <= tol), prev_code, tgt)
                tgt = np.where((tgt == 0) & (prev_code > 0) & (p >= 0.6 * p_floor), prev_code, tgt)
                # heading: switch only if the new direction clearly beats the old one in the histogram
                cand = hd.astype(np.int64)
                ph = np.take_along_axis(hist, np.maximum(prev_head, 0)[None].astype(np.int64), 0)[0]
                nh = np.take_along_axis(hist, np.maximum(cand, 0)[None], 0)[0]
                adopt = (cand >= 0) & ((prev_head < 0) | ((cand != prev_head) & (nh > 1.5 * ph)))
                hgt = np.where(tgt > 0, np.where(adopt, cand, prev_head), -1).astype(np.int8)
                new = (tgt != prev_code) | (hgt != prev_head)
                if not new.any():
                    out.append(1)
                else:
                    idx = np.flatnonzero(new.reshape(-1))
                    val = (tgt.reshape(-1)[idx].astype(np.uint8)) | (np.maximum(hgt.reshape(-1)[idx], 0).astype(np.uint8) << 4) \
                        | ((hgt.reshape(-1)[idx] >= 0).astype(np.uint8) << 7)
                    gaps = np.diff(idx, prepend=-1) - 1
                    out.append(2)
                    out += _varint(len(idx))
                    for g, v in zip(gaps.tolist(), val.tolist()):
                        out += _varint(g)
                        out.append(v)
                prev_code, prev_head, have_prev = tgt, hgt, True
        cls_sizes["L2_vehicle" if c == VEHICLE else "L3_person"] = len(out) - start
    sizes.update(cls_sizes)
    payload = MAGIC_PAYLOAD + bytes([1]) + struct.pack(">HH", len(meta), len(l1)) + meta + l1 + bytes(out)
    return payload, sizes


# ------------------------------------------------------------------------------------------------ decoder

class DecodedMap(MapView):
    """Robot-side view of a decoded map. Same read API as LayeredMap (box covers instead of count grids)."""

    def __init__(self, spec: dict, period: np.ndarray, jitter: np.ndarray, boxes: dict, codes: dict, heads: dict, observed: dict):
        self.spec, self.period, self.jitter = spec, period, jitter
        self._boxes, self._codes, self._heads, self._obs = boxes, codes, heads, observed
        self.nx, self.nz = spec["site"]["nx"], spec["site"]["nz"]
        self._ids = [d["id"] for d in spec["machines"]]

    def _observed(self, tod: int, prod: int, cls: int | None = None) -> bool:
        return bool(self._obs[VEHICLE if cls is None else cls][tod * N_PROD + prod])

    def boxes_for(self, machine_id: str, bins) -> np.ndarray:
        i = self._ids.index(machine_id)
        parts = [self._boxes[(i, int(b))] for b in bins if len(self._boxes[(i, int(b))])]
        return np.vstack(parts) if parts else np.zeros((0, 4))

    def box_cover(self, machine_id: str, phase_bin, **_) -> np.ndarray:
        bins = (phase_bin,) if isinstance(phase_bin, (int, np.integer)) else tuple(phase_bin)
        return self.boxes_for(machine_id, bins)

    def traffic(self, cls, tod_bin: int, prod_state: int, fallback: bool = True):
        c = self._cls(cls)
        r = self._resolve(tod_bin, prod_state, c) if fallback else ((tod_bin, prod_state) if self._observed(tod_bin, prod_state, c) else None)
        if r is None:
            return np.zeros((self.nx, self.nz), np.float32), np.full((self.nx, self.nz), -1, np.int8)
        S = r[0] * N_PROD + r[1]
        return code_to_p(self._codes[c][S]), self._heads[c][S]


def decode_payload(payload: bytes) -> DecodedMap:
    if payload[:4] != MAGIC_PAYLOAD or payload[4] != 1:
        raise CodecError("bad map payload")
    lm, ll = struct.unpack_from(">HH", payload, 5)
    pos = 9
    r = BitReader(payload[pos:pos + lm])
    l1 = payload[pos + lm:pos + lm + ll]
    body = payload[pos + lm + ll:]
    n = r.get(8)
    cb = r.get(4)
    env_res = r.get(8) / 100
    site = dict(x0=r.get_signed(16) / 4, z0=r.get_signed(16) / 4, res=r.get(12) / 100, nx=r.get(8), nz=r.get(8))
    machines, period, jitter = [], np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        k = r.get(4)
        ident = "".join(chr(r.get(7)) for _ in range(k))
        x0, z0 = r.get_signed(12) * env_res, r.get_signed(12) * env_res
        nx, nz = r.get(8), r.get(8)
        P, J = r.get(16), r.get(8)
        period[i] = P / 100 if P else np.nan
        jitter[i] = J / 2000 if J else np.nan
        machines.append(dict(id=ident, x0=x0, z0=z0, nx=nx, nz=nz))
    spec = dict(machines=machines, site=site, env_res=env_res, n_phase=N_PHASE, n_tod=N_TOD, n_prod=N_PROD)
    # ---- L1
    r = BitReader(l1)
    boxes = {}
    for i, d in enumerate(machines):
        for b in range(N_BINS):
            k = r.get(5)
            cbx = np.array([[r.get(cb) for _ in range(4)] for _ in range(k)], float).reshape(-1, 4)
            res = env_res
            boxes[(i, b)] = np.stack([d["x0"] + cbx[:, 0] * res, d["z0"] + cbx[:, 1] * res,
                                      d["x0"] + cbx[:, 2] * res, d["z0"] + cbx[:, 3] * res], 1) if k else np.zeros((0, 4))
    # ---- L2 / L3
    S = N_TOD * N_PROD
    nx, nz = site["nx"], site["nz"]
    codes = {c: np.zeros((S, nx, nz), np.uint8) for c in CLASSES}
    heads = {c: np.full((S, nx, nz), -1, np.int8) for c in CLASSES}
    observed = {c: np.zeros(S, bool) for c in CLASSES}
    pos = 0
    for c in CLASSES:
        for prod in range(N_PROD):
            cur_c = np.zeros(nx * nz, np.uint8)
            cur_h = np.full(nx * nz, -1, np.int8)
            for tod in range(N_TOD):
                flag = body[pos]
                pos += 1
                if flag == 0:
                    continue
                if flag == 2:
                    cnt, pos = _read_varint(body, pos)
                    at = -1
                    for _ in range(cnt):
                        g, pos = _read_varint(body, pos)
                        v = body[pos]
                        pos += 1
                        at += g + 1
                        cur_c[at] = v & 15
                        cur_h[at] = (v >> 4) & 7 if v & 128 else -1
                s = tod * N_PROD + prod
                codes[c][s] = cur_c.reshape(nx, nz)
                heads[c][s] = cur_h.reshape(nx, nz)
                observed[c][s] = True
    return DecodedMap(spec, period, jitter, boxes, codes, heads, observed)


# ------------------------------------------------------------------------------------------------ manifest

@dataclass(frozen=True)
class Manifest:
    site_id: int
    key_id: int
    issue_time: int            # unix s
    version: int               # map version (monotonic per site)
    valid_s: int
    raw_len: int               # uncompressed payload
    payload_len: int           # compressed payload (what the chunks carry)
    n_chunks: int
    chunk_size: int
    parity_group: int          # 0 = no parity chunks
    sha: bytes                 # SHA-256 of the compressed payload
    chunk_hashes: tuple        # 4-byte hash of every data chunk then every parity chunk

    @property
    def n_parity(self) -> int:
        return -(-self.n_chunks // self.parity_group) if self.parity_group else 0

    @property
    def map_id(self) -> int:
        return int.from_bytes(self.sha[:2], "big")

    @property
    def n_total(self) -> int:
        return self.n_chunks + self.n_parity


def _h4(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()[:4]


def encode_manifest(m: Manifest) -> bytes:
    w = BitWriter()
    w.put(MAGIC, 16)
    w.put(VERSION, 4)
    w.put(LAYER_MAP, 4)
    w.put(m.site_id, 24)
    w.put(m.key_id, 16)
    w.put(m.issue_time, 32)
    w.put(m.version, 32)
    w.put(m.valid_s, 32)
    w.put(m.raw_len, 32)
    w.put(m.payload_len, 32)
    w.put(m.n_chunks, 16)
    w.put(m.chunk_size, 12)
    w.put(m.parity_group, 8)
    w.put(int.from_bytes(m.sha, "big"), 256)
    if len(m.chunk_hashes) != m.n_total:
        raise CodecError("chunk hash table size")
    for h in m.chunk_hashes:
        w.put(int.from_bytes(h, "big"), 32)
    return w.to_bytes()


def decode_manifest(payload: bytes) -> Manifest:
    r = BitReader(payload)
    if r.get(16) != MAGIC or r.get(4) != VERSION or r.get(4) != LAYER_MAP:
        raise CodecError("not a map manifest")
    site, key, issue = r.get(24), r.get(16), r.get(32)
    ver, valid, raw, plen, n, cs, pg = r.get(32), r.get(32), r.get(32), r.get(32), r.get(16), r.get(12), r.get(8)
    sha = r.get(256).to_bytes(32, "big")
    n_par = -(-n // pg) if pg else 0
    if r.left < 32 * (n + n_par):
        raise CodecError("truncated manifest")
    hashes = tuple(r.get(32).to_bytes(4, "big") for _ in range(n + n_par))
    if cs == 0 or n * cs < plen or (n - 1) * cs >= plen:
        raise CodecError("inconsistent manifest")
    return Manifest(site, key, issue, ver, valid, raw, plen, n, cs, pg, sha, hashes)


# ------------------------------------------------------------------------------------------------ frames

def chunk_frame(map_id: int, idx: int, data: bytes) -> bytes:
    hdr = struct.pack(">HBBH", map_id, LAYER_MAP, 0, idx)
    body = hdr + bytes([ble.crc8(hdr + data)]) + data
    return struct.pack("<BBH", len(body) + 3, ble.AD_MFR, ble.COMPANY) + body


def parse_chunk(f: bytes) -> tuple[int, int, bytes] | None:
    """(map_id, chunk_idx, data) or None if f is not a valid map chunk frame."""
    if not 11 <= len(f) <= ble.ADV_MAX or f[0] != len(f) - 1 or f[1] != ble.AD_MFR:
        return None
    if struct.unpack_from("<H", f, 2)[0] != ble.COMPANY:
        return None
    map_id, layer, zero, idx = struct.unpack_from(">HBBH", f, 4)
    if layer != LAYER_MAP or zero != 0 or ble.crc8(f[4:10] + f[11:]) != f[10]:
        return None
    return map_id, idx, f[11:]


def is_map_frame(f: bytes) -> bool:
    """True for map-layer frames (manifest fragments and chunks): lets a receiver route them away from the 3 other layers."""
    return len(f) > 6 and f[1] == ble.AD_MFR and f[6] == LAYER_MAP


# ------------------------------------------------------------------------------------------------ pack (broadcaster side)

@dataclass
class MapBundle:
    manifest: Manifest
    manifest_blob: bytes
    manifest_frames: list[bytes]
    chunks: list[bytes]                 # data chunks then parity chunks (raw bytes, no framing)
    frames: list[bytes]                 # the chunk frames, same order
    stats: dict = field(default_factory=dict)

    def carousel(self, manifest_every: int = 16) -> list[bytes]:
        """One loop of the periodic-advertising train: manifest fragments, then `manifest_every` chunk frames, repeat."""
        out: list[bytes] = []
        for i in range(0, len(self.frames), manifest_every):
            out += self.manifest_frames
            out += self.frames[i:i + manifest_every]
        return out


def pack_map(m: LayeredMap, key: SigningKey, version: int, issue_time: float | None = None, valid_s: int = 6 * 3600,
             parity_group: int = 8, p_min: float = 0.05, max_boxes: int = 12, hyst_sigma: float = 2.5) -> MapBundle:
    """Encode, compress, chunk and sign a LayeredMap. issue_time is unix seconds (default: map's last observation)."""
    payload, sizes = encode_payload(m, p_min, max_boxes, hyst_sigma=hyst_sigma)
    comp = zlib.compress(payload, 9)
    cs = CHUNK_DATA
    n = -(-len(comp) // cs)
    chunks = [comp[i * cs:(i + 1) * cs] for i in range(n)]
    par = []
    if parity_group:
        for g in range(0, n, parity_group):
            acc = np.zeros(cs, np.uint8)
            for c in chunks[g:g + parity_group]:
                acc ^= np.frombuffer(c.ljust(cs, b"\0"), np.uint8)
            par.append(acc.tobytes())
    sha = hashlib.sha256(comp).digest()
    issue = int(round(issue_time if issue_time is not None else DAY0_UNIX + m.t_last))
    man = Manifest(m.spec["site_id"], m.spec["key_id"], issue, version, valid_s, len(payload), len(comp), n, cs, parity_group, sha,
                   tuple(_h4(c) for c in chunks + par))
    blob = sign(encode_manifest(man), key)
    allc = chunks + par
    frames = [chunk_frame(man.map_id, i, c) for i, c in enumerate(allc)]
    mf = ble.frame(blob, msg_id=man.map_id, layer=LAYER_MAP)
    stats = dict(sizes, raw_bytes=len(payload), compressed_bytes=len(comp), manifest_bytes=len(blob), n_chunks=n,
                 n_parity=len(par), n_manifest_frames=len(mf), payload_on_air_bytes=sum(len(f) for f in frames) + sum(len(f) for f in mf),
                 total_bytes=len(comp) + len(blob))
    return MapBundle(man, blob, mf, allc, frames, stats)


# ------------------------------------------------------------------------------------------------ receiver (robot side)

class MapCache:
    """Maps the robot already holds, per (site_id, version), plus partial downloads keyed by manifest id."""

    def __init__(self) -> None:
        self.maps: dict[tuple[int, int], DecodedMap] = {}
        self.partial: dict[bytes, dict[int, bytes]] = {}

    def get(self, site_id: int, version: int) -> DecodedMap | None:
        return self.maps.get((site_id, version))

    def put(self, site_id: int, version: int, m: DecodedMap) -> None:
        self.maps[(site_id, version)] = m

    def newest(self, site_id: int) -> tuple[int, DecodedMap] | None:
        v = [k[1] for k in self.maps if k[0] == site_id]
        return (max(v), self.maps[(site_id, max(v))]) if v else None


class MapReceiver:
    """Assembles the map layer from whatever frames are heard (any order, any loss, junk and forgeries included).

    ingest(t_rx, frame) returns the DecodedMap the moment a version completes (and was verified), else None.
    Rejections never touch accepted state; `stats` counts every reason."""

    def __init__(self, keyring: Keyring, site_id: int, cache: MapCache | None = None, timeout_s: float = 10.0,
                 max_pending: int = 1024) -> None:
        self.keyring, self.site_id = keyring, site_id
        self.cache = cache if cache is not None else MapCache()
        self.reasm = ble.Reassembler(timeout_s)
        self.stats: Counter[str] = Counter()
        self.manifest: Manifest | None = None
        self._chunks: dict[int, bytes] = {}
        self._pending: dict[int, dict[int, bytes]] = {}
        self.max_pending = max_pending
        self.done = False
        self.t_manifest: float | None = None
        self.t_done: float | None = None

    # ------------------------------------------------------------------
    def ingest(self, t_rx: float, frame: bytes) -> DecodedMap | None:
        ch = parse_chunk(frame)
        if ch is not None:
            return self._on_chunk(t_rx, *ch)
        fr = ble.parse_frame(frame)
        if fr is None or fr.layer != LAYER_MAP:
            self.stats["not_map"] += 1
            return None
        blob = self.reasm.ingest(t_rx, frame)
        return None if blob is None else self._on_manifest(t_rx, blob)

    def _on_manifest(self, t_rx: float, blob: bytes) -> DecodedMap | None:
        ok, payload, key_id, reason = verify(blob, self.keyring)
        if not ok:
            self.stats[reason] += 1
            return None
        try:
            h = peek_header(payload)
            if h.layer != LAYER_MAP:
                self.stats["wrong_layer"] += 1
                return None
            man = decode_manifest(payload)
        except CodecError:
            self.stats["malformed"] += 1
            return None
        if man.site_id != self.site_id:
            self.stats["wrong_site"] += 1
            return None
        now = DAY0_UNIX + t_rx
        if now > man.issue_time + man.valid_s:
            self.stats["expired"] += 1
            return None
        if man.n_total > 4095 or man.chunk_size > CHUNK_DATA:
            self.stats["malformed"] += 1
            return None
        cached = self.cache.get(man.site_id, man.version)
        if cached is not None:
            self.stats["cached"] += 1
            if self.manifest is None or man.version >= self.manifest.version:
                self.manifest, self._chunks, self.done = man, {}, True
            return None
        newest = self.cache.newest(man.site_id)
        if newest is not None and man.version < newest[0]:
            self.stats["stale_version"] += 1
            return None
        if self.manifest is not None and man.sha == self.manifest.sha and man.version == self.manifest.version:
            self.stats["dup_manifest"] += 1
            return None
        if self.manifest is not None and not self.done and man.version < self.manifest.version:
            self.stats["stale_version"] += 1
            return None
        self.manifest, self.done, self.t_manifest = man, False, t_rx
        self._chunks = self.cache.partial.setdefault(man.sha, {})      # shared: a partial download survives leaving range
        self.stats["manifest_ok"] += 1
        for idx, data in self._pending.pop(man.map_id, {}).items():       # chunks heard before the manifest
            self._take(man, idx, data)
        return self._try_finish(t_rx)

    def _on_chunk(self, t_rx: float, map_id: int, idx: int, data: bytes) -> DecodedMap | None:
        man = self.manifest
        if man is None or man.map_id != map_id:           # keep a few chunks heard before their manifest
            buf = self._pending.setdefault(map_id, {})
            if len(buf) < self.max_pending and len(self._pending) <= 4:
                buf[idx] = data
            self.stats["chunk_before_manifest"] += 1
            return None
        if self.done:
            self.stats["chunk_ignored"] += 1
            return None
        if not self._take(man, idx, data):
            return None
        return self._try_finish(t_rx)

    def _take(self, man: Manifest, idx: int, data: bytes) -> bool:
        if idx >= man.n_total or _h4(data) != man.chunk_hashes[idx]:
            self.stats["bad_chunk"] += 1
            return False
        if idx in self._chunks:
            self.stats["dup_chunk"] += 1
            return False
        self._chunks[idx] = data
        self.stats["chunk_ok"] += 1
        return True

    def _repair(self, man: Manifest) -> None:
        pg, n, cs = man.parity_group, man.n_chunks, man.chunk_size
        if not pg:
            return
        for g in range(0, n, pg):
            pi = n + g // pg
            if pi not in self._chunks:
                continue
            grp = range(g, min(g + pg, n))
            miss = [i for i in grp if i not in self._chunks]
            if len(miss) != 1:
                continue
            acc = np.frombuffer(self._chunks[pi], np.uint8).copy()
            for i in grp:
                if i != miss[0]:
                    acc ^= np.frombuffer(self._chunks[i].ljust(cs, b"\0"), np.uint8)
            full = acc.tobytes()
            rec = full if miss[0] < n - 1 else full[:man.payload_len - (n - 1) * cs]
            if _h4(rec) == man.chunk_hashes[miss[0]]:
                self._chunks[miss[0]] = rec
                self.stats["repaired"] += 1

    def _try_finish(self, t_rx: float) -> DecodedMap | None:
        man = self.manifest
        if man is None or self.done:
            return None
        if any(i not in self._chunks for i in range(man.n_chunks)):
            self._repair(man)
            if any(i not in self._chunks for i in range(man.n_chunks)):
                return None
        comp = b"".join(self._chunks[i] for i in range(man.n_chunks))
        if len(comp) != man.payload_len or hashlib.sha256(comp).digest() != man.sha:
            self.stats["bad_payload"] += 1
            self._chunks.clear()
            self.cache.partial.pop(man.sha, None)
            return None
        try:
            dm = decode_payload(zlib.decompress(comp))
        except (CodecError, zlib.error, IndexError, ValueError):
            self.stats["bad_payload"] += 1
            self._chunks.clear()
            self.cache.partial.pop(man.sha, None)
            return None
        self.cache.put(man.site_id, man.version, dm)
        self.cache.partial.pop(man.sha, None)
        self.done, self.t_done = True, t_rx
        self.stats["map_ok"] += 1
        return dm

    @property
    def progress(self) -> float:
        man = self.manifest
        return 0.0 if man is None else sum(1 for i in range(man.n_chunks) if i in self._chunks) / man.n_chunks

    def latest(self, t_now: float) -> DecodedMap | None:
        """Newest cached map for this site whose manifest has not expired."""
        n = self.cache.newest(self.site_id)
        if n is None:
            return None
        man = self.manifest
        if man is not None and man.version == n[0] and DAY0_UNIX + t_now > man.issue_time + man.valid_s:
            return None
        return n[1]


# ------------------------------------------------------------------------------------------------ assembly-time simulation

def simulate_assembly(bundle: MapBundle, keyring: Keyring, site_id: int, loss: float = 0.1, fps: float = 20.0,
                      trials: int = 100, seed: int = 0, manifest_every: int = 16, max_s: float = 900.0) -> dict:
    """Monte Carlo: a robot enters range at a random point of the carousel; every frame is lost independently with
    probability `loss`; frames arrive at `fps` (periodic advertising). Returns time-to-map statistics in seconds."""
    rng = np.random.default_rng(seed)
    t_base = bundle.manifest.issue_time - DAY0_UNIX + 60.0
    loop = bundle.carousel(manifest_every)
    L = len(loop)
    times, ok = [], 0
    for _ in range(trials):
        rx = MapReceiver(keyring, site_id)
        start = int(rng.integers(L))
        k_max = int(max_s * fps)
        got = rng.random(k_max) >= loss
        t_done = math.nan
        for k in np.flatnonzero(got):
            if rx.ingest(t_base + k / fps, loop[(start + k) % L]) is not None:
                t_done = (k + 1) / fps
                break
        if not math.isnan(t_done):
            ok += 1
            times.append(t_done)
    t = np.array(times) if times else np.array([math.nan])
    return dict(loop_frames=L, loop_s=L / fps, trials=trials, completed=ok, mean_s=float(np.mean(t)), median_s=float(np.median(t)),
                p95_s=float(np.percentile(t, 95)), max_s=float(np.max(t)), loss=loss, fps=fps, times=[float(x) for x in times])
