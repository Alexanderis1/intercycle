"""Snapshot writer + live feed for the map-growing view (workstream D owns it; M calls it).

    sw = StreamWriter("week1", live=False)            # live=True also serves ws://127.0.0.1:8765
    sw.snapshot(t_sim=..., t_obs=..., prod_state="running", grid=dict(x0=0, z0=0, res=1.0, nx=120, nz=48),
                l1={"S30": [(phase_bin, x0, z0, x1, z1), ...]}, l2_occ=..., l2_head=..., l3_occ=..., l3_head=...,
                unknown=..., phase_now={"S30": 5}, metrics={"l1_iou": 0.91, "unsafe_free": 0.0})
    sw.event(t_sim=..., kind="change_detected", x=61.0, z=40.0, cells=12)
    sw.close()

Files under `out/stream/<run>/` (everything SYNTHETIC: simulated plant):
  * `snap_<t_sim:09d>.json` - one snapshot (t_sim rounded to whole seconds), <= 200 kB each.
  * `events.jsonl`          - one event per line, appended.
  * `index.json`            - manifest the static page loads first: run meta, every snapshot (file, t_sim, t_obs,
                              prod_state, metrics) and every event. Rewritten atomically after each snapshot / event.

Snapshot JSON (`"schema": "intercycle.stream/1"`, `"label": "SYNTHETIC: simulated plant"`):
  {schema, label, run, seq, t_sim, t_obs, prod_state, phase_now{mid: bin}, metrics{name: number},
   grid{x0, z0, res, nx, nz},                 # the grid the layers below are encoded on (after any downsampling)
   grid_in{...}, downsample: k,               # the grid M passed in, and the integer pooling factor used (1 = none)
   l1{mid: [[phase_bin, x0, z0, x1, z1], ...]},   # boxes in metres, site frame
   layers{l2_occ, l3_occ, l2_head, l3_head, unknown}}
Array encodings (every grid is (nz, nx), row-major: row = z index, column = x index; cell (i, j) covers
x0 + j*res .. x0 + (j+1)*res, z0 + i*res .. z0 + (i+1)*res):
  * occupancy `{"enc": "u8", "scale": 255, "b64": ...}`: round(clip(p, 0, 1) * 255) as uint8, base64. If the raw bytes
    exceed 16 kB the bytes are zlib-compressed first and `"enc": "u8z"` (browser: DecompressionStream("deflate")).
  * heading `{"enc": "u8" | "u8z", "none": 255, "b64": ...}`: 0..7 compass bins (0 = +z, clockwise towards +x),
    255 = no heading (the -1 of the input).
  * mask (UNKNOWN) `{"enc": "rle", "runs": [n0, n1, ...]}`: run lengths over the flattened mask, alternating
    False, True, False, ... starting with False (n0 may be 0). Sum of runs = nz * nx.
If a snapshot would exceed 200 kB the grids are pooled by an integer factor k (occupancy and UNKNOWN by max, i.e.
conservative; heading by the value at the pooled cell holding the max occupancy) until it fits.

Event JSON line: {schema, t_sim, kind, seq, ...fields}. Known kinds: change_detected, machine_spawned, phase_discovered,
phase_named, camera_offline (others are accepted and passed through). Fields must be JSON-able (numpy scalars and
arrays are converted).

Live mode: an in-process websocket relay (package `websockets`) on ws://127.0.0.1:<port>. Messages (JSON text):
  {"type": "hello", "index": <index.json>, "last": <last snapshot or null>}   on connect
  {"type": "snapshot", "snap": <snapshot>}  /  {"type": "event", "event": <event>}  /  {"type": "close"}
If `websockets` is missing or the port is taken, the writer warns and keeps writing files.

Must never import ground truth (tests/test_stream.py checks): metrics arrive as plain numbers.
"""
from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

SCHEMA = "intercycle.stream/1"
LABEL = "SYNTHETIC: simulated plant"
MAX_SNAP_BYTES = 200_000
ZLIB_ABOVE = 16_000
KNOWN_KINDS = ("change_detected", "change_cleared", "machine_converged", "machine_spawned", "phase_discovered", "phase_named", "camera_offline", "production_state", "note")


# ------------------------------------------------------------------------------------------------ encoding

def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _dumps(obj) -> str:
    return json.dumps(obj, default=_jsonable, separators=(",", ":"), allow_nan=False)


def _clean(x):
    """Non-finite floats -> None (JSON has no NaN)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _bytes_field(raw: bytes, **extra) -> dict:
    if len(raw) > ZLIB_ABOVE:
        return {"enc": "u8z", **extra, "b64": base64.b64encode(zlib.compress(raw, 6)).decode("ascii")}
    return {"enc": "u8", **extra, "b64": base64.b64encode(raw).decode("ascii")}


def encode_occ(p: np.ndarray) -> dict:
    q = np.rint(np.clip(np.nan_to_num(np.asarray(p, np.float64), nan=0.0), 0.0, 1.0) * 255.0).astype(np.uint8)
    return _bytes_field(np.ascontiguousarray(q).tobytes(), scale=255)


def encode_head(h: np.ndarray) -> dict:
    h = np.asarray(h).astype(np.int16)
    q = np.where((h >= 0) & (h <= 7), h, 255).astype(np.uint8)
    return _bytes_field(np.ascontiguousarray(q).tobytes(), none=255)


def encode_mask(m: np.ndarray) -> dict:
    f = np.ascontiguousarray(np.asarray(m, bool)).reshape(-1)
    if f.size == 0:
        return {"enc": "rle", "runs": [0]}
    edges = np.flatnonzero(f[1:] != f[:-1]) + 1
    bounds = np.concatenate([[0], edges, [f.size]])
    runs = np.diff(bounds).tolist()
    if f[0]:
        runs = [0] + runs
    return {"enc": "rle", "runs": runs}


def decode_field(d: dict, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of the encoders (tests, offline tools). Occupancy -> float32 0..1, heading -> int8 (-1 none), mask -> bool."""
    if d["enc"] == "rle":
        vals = np.zeros(len(d["runs"]), bool)
        vals[1::2] = True
        return np.repeat(vals, d["runs"]).reshape(shape)
    raw = base64.b64decode(d["b64"])
    if d["enc"] == "u8z":
        raw = zlib.decompress(raw)
    a = np.frombuffer(raw, np.uint8).reshape(shape)
    if "scale" in d:
        return (a.astype(np.float32) / float(d["scale"]))
    return np.where(a == d.get("none", 255), -1, a.astype(np.int16)).astype(np.int8)


def _pool(a: np.ndarray, k: int, how: str) -> np.ndarray:
    nz, nx = a.shape
    pz, px = (-nz) % k, (-nx) % k
    if how == "max":
        b = np.pad(a, ((0, pz), (0, px)), constant_values=0)
        return b.reshape(b.shape[0] // k, k, b.shape[1] // k, k).max(axis=(1, 3))
    raise ValueError(how)


def _pool_head(head: np.ndarray, occ: np.ndarray, k: int) -> np.ndarray:
    nz, nx = head.shape
    pz, px = (-nz) % k, (-nx) % k
    h = np.pad(head, ((0, pz), (0, px)), constant_values=-1)
    o = np.pad(np.where(head >= 0, occ, -1.0), ((0, pz), (0, px)), constant_values=-1.0)
    Z, X = h.shape[0] // k, h.shape[1] // k
    hb = h.reshape(Z, k, X, k).transpose(0, 2, 1, 3).reshape(Z, X, k * k)
    ob = o.reshape(Z, k, X, k).transpose(0, 2, 1, 3).reshape(Z, X, k * k)
    i = ob.argmax(-1)
    out = np.take_along_axis(hb, i[..., None], -1)[..., 0]
    return np.where(np.take_along_axis(ob, i[..., None], -1)[..., 0] >= 0, out, -1).astype(np.int8)


# ------------------------------------------------------------------------------------------------ live relay

class _Relay:
    """Tiny websocket broadcast relay running an asyncio loop in a daemon thread."""

    def __init__(self, host: str, port: int, hello):
        import asyncio

        from websockets.asyncio.server import serve  # noqa: F401  (import error -> caller warns)

        self._asyncio = asyncio
        self._hello = hello
        self._clients: set = set()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._err: BaseException | None = None
        self._stop = None
        self._thread = threading.Thread(target=self._run, args=(host, port), daemon=True, name="stream-relay")
        self._thread.start()
        self._ready.wait(10.0)
        if self._err is not None:
            raise self._err

    def _run(self, host: str, port: int) -> None:
        asyncio = self._asyncio
        from websockets.asyncio.server import serve

        async def handler(ws):
            self._clients.add(ws)
            try:
                await ws.send(self._hello())
                async for _ in ws:          # clients only listen; drain anything they send
                    pass
            except Exception:
                pass
            finally:
                self._clients.discard(ws)

        async def main():
            self._stop = asyncio.get_running_loop().create_future()
            try:
                async with serve(handler, host, port, max_size=2**22):
                    self._ready.set()
                    await self._stop
            except BaseException as e:      # port in use etc.
                self._err = e
                self._ready.set()

        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(main())

    def send(self, text: str) -> None:
        if self._err is not None or not self._loop.is_running():
            return

        def _bcast():
            from websockets.asyncio.server import broadcast
            broadcast(set(self._clients), text)

        self._loop.call_soon_threadsafe(_bcast)

    def close(self) -> None:
        if self._loop.is_running() and self._stop is not None:
            self._loop.call_soon_threadsafe(lambda: self._stop.done() or self._stop.set_result(None))
        self._thread.join(5.0)


# ------------------------------------------------------------------------------------------------ writer

class StreamWriter:
    def __init__(self, run: str, out_root: str | Path = "out/stream", live: bool = False, port: int = 8765,
                 meta: dict | None = None, host: str = "127.0.0.1"):
        """`meta` (optional, JSON-able) is stored in index.json, e.g. scenario name, horizon, notes for the page."""
        if not run or any(c in run for c in '\\/:*?"<>|'):
            raise ValueError(f"bad run name {run!r}")
        self.run = run
        self.dir = Path(out_root) / run
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.dir / "events.jsonl"
        self._events_path.write_text("", encoding="utf-8")       # a run starts fresh
        for old in self.dir.glob("snap_*.json"):
            old.unlink()
        self._snaps: list[dict] = []
        self._events: list[dict] = []
        self._seq = 0
        self._last_snap_text: str | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.index = dict(schema=SCHEMA, label=LABEL, run=run, created=time.strftime("%Y-%m-%dT%H:%M:%S"), updated=None,
                          complete=False, meta=_clean(meta or {}), snapshots=self._snaps, events=self._events,
                          events_file="events.jsonl")
        self._write_index()
        self.relay: _Relay | None = None
        if live:
            try:
                self.relay = _Relay(host, port, self._hello)
            except ImportError:
                warnings.warn("StreamWriter(live=True): package 'websockets' is not installed; writing files only")
            except OSError as e:
                warnings.warn(f"StreamWriter(live=True): cannot serve ws://{host}:{port} ({e}); writing files only")

    # ---------------------------------------------------------------- internals
    def _hello(self) -> str:
        with self._lock:
            idx = _dumps(self.index)
            last = self._last_snap_text or "null"
        return '{"type":"hello","index":' + idx + ',"last":' + last + "}"

    def _write_index(self) -> None:
        self.index["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(_dumps(self.index), encoding="utf-8")
        os.replace(tmp, self.dir / "index.json")

    @staticmethod
    def _grid2d(name: str, a: np.ndarray, nz: int, nx: int) -> np.ndarray:
        a = np.asarray(a)
        if a.shape == (nz, nx):
            return a
        if a.shape == (nx, nz):             # LayeredMap.traffic() returns [nx, nz]; accept it, the contract is (nz, nx)
            warnings.warn(f"StreamWriter: {name} has shape (nx, nz) = {a.shape}; transposing to (nz, nx)", stacklevel=3)
            return a.T
        raise ValueError(f"{name}: shape {a.shape}, expected (nz, nx) = {(nz, nx)}")

    # ---------------------------------------------------------------- API
    def snapshot(self, *, t_sim: float, t_obs: float, prod_state: str, grid: dict,
                 l1: dict[str, list[tuple[int, float, float, float, float]]],
                 l2_occ: np.ndarray, l2_head: np.ndarray, l3_occ: np.ndarray, l3_head: np.ndarray,
                 unknown: np.ndarray, phase_now: dict[str, int], metrics: dict[str, float]) -> Path:
        if self._closed:
            raise RuntimeError("StreamWriter is closed")
        nx, nz = int(grid["nx"]), int(grid["nz"])
        g_in = dict(x0=float(grid["x0"]), z0=float(grid["z0"]), res=float(grid["res"]), nx=nx, nz=nz)
        occ2 = self._grid2d("l2_occ", l2_occ, nz, nx).astype(np.float32)
        occ3 = self._grid2d("l3_occ", l3_occ, nz, nx).astype(np.float32)
        hd2 = self._grid2d("l2_head", l2_head, nz, nx).astype(np.int8)
        hd3 = self._grid2d("l3_head", l3_head, nz, nx).astype(np.int8)
        unk = self._grid2d("unknown", unknown, nz, nx).astype(bool)
        boxes = {str(mid): [[int(b[0])] + [round(float(v), 3) for v in b[1:5]] for b in bl] for mid, bl in (l1 or {}).items()}
        base = dict(schema=SCHEMA, label=LABEL, run=self.run, seq=len(self._snaps), t_sim=float(t_sim), t_obs=float(t_obs),
                    prod_state=str(prod_state), phase_now=_clean({str(k): (None if v is None else int(v)) for k, v in (phase_now or {}).items()}),
                    metrics=_clean({str(k): v for k, v in (metrics or {}).items()}), grid_in=g_in, l1=boxes)
        k = 1
        while True:
            if k == 1:
                o2, o3, h2, h3, u = occ2, occ3, hd2, hd3, unk
            else:
                o2, o3, u = _pool(occ2, k, "max"), _pool(occ3, k, "max"), _pool(unk.astype(np.uint8), k, "max").astype(bool)
                h2, h3 = _pool_head(hd2, occ2, k), _pool_head(hd3, occ3, k)
            g = dict(x0=g_in["x0"], z0=g_in["z0"], res=g_in["res"] * k, nx=int(o2.shape[1]), nz=int(o2.shape[0]))
            snap = dict(base, grid=g, downsample=k, layers=dict(l2_occ=encode_occ(o2), l3_occ=encode_occ(o3), l2_head=encode_head(h2),
                                                                 l3_head=encode_head(h3), unknown=encode_mask(u)))
            text = _dumps(snap)
            if len(text.encode()) <= MAX_SNAP_BYTES or min(o2.shape) <= 4:
                break
            k += 1
        if len(text.encode()) > MAX_SNAP_BYTES:
            warnings.warn(f"StreamWriter: snapshot is {len(text) / 1e3:.0f} kB > 200 kB (L1 boxes dominate?)")
        name = f"snap_{int(round(float(t_sim))):09d}.json"
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        with self._lock:
            # a re-written t_sim replaces its entry
            self._snaps[:] = [s for s in self._snaps if s["file"] != name]
            self._snaps.append(dict(file=name, t_sim=snap["t_sim"], t_obs=snap["t_obs"], prod_state=snap["prod_state"],
                                    metrics=snap["metrics"], bytes=len(text.encode())))
            self._snaps.sort(key=lambda s: (s["t_obs"], s["t_sim"]))
            self._last_snap_text = text
            self._write_index()
        if self.relay:
            self.relay.send('{"type":"snapshot","snap":' + text + "}")
        return path

    def event(self, *, t_sim: float, kind: str, **fields) -> None:
        if not kind or not isinstance(kind, str):
            raise ValueError("event kind must be a non-empty string")
        if kind not in KNOWN_KINDS:
            warnings.warn(f"StreamWriter.event: unknown kind {kind!r} (passed through)", stacklevel=2)
        reserved = {"schema", "seq"} & set(fields)
        if reserved:
            raise ValueError(f"reserved event fields: {sorted(reserved)}")
        with self._lock:
            ev = _clean(dict(schema=SCHEMA, t_sim=float(t_sim), kind=kind, seq=len(self._events), **fields))
            text = _dumps(ev)
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            self._events.append(ev)
            self._write_index()
        if self.relay:
            self.relay.send('{"type":"event","event":' + text + "}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self.index["complete"] = True
            self._write_index()
        if self.relay:
            self.relay.send('{"type":"close"}')
            time.sleep(0.05)
            self.relay.close()
            self.relay = None

    def __enter__(self) -> StreamWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
