"""CPU video decode (PyAV) with stride/fps subsampling and multi-camera batching.

Decoding is deliberately CPU-only (libav software H.264 via PyAV): no NVDEC,
DALI or TensorRT, so the identical code runs on NVIDIA and AMD/ROCm hosts.
``av`` is imported lazily so geometry/zone/occupancy code never needs it.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = ["VideoInfo", "Frame", "probe", "stride_for", "iter_frames", "MultiCameraBatcher", "DecodeStats"]


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass
class Frame:
    cam: str
    frame_idx: int
    t: float  # seconds since video start
    image: np.ndarray  # (H, W, 3) uint8 RGB


@dataclass
class DecodeStats:
    decode_s: float = 0.0  # summed decode-thread time (can exceed wall time with several cameras)
    frames_decoded: int = 0  # every frame the demuxer/decoder produced
    frames_emitted: int = 0  # frames converted to RGB and handed on
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, dt: float, decoded: int, emitted: int) -> None:
        with self.lock:
            self.decode_s += dt
            self.frames_decoded += decoded
            self.frames_emitted += emitted


def probe(path: str | Path) -> VideoInfo:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 30.0
        n = int(s.frames) if s.frames else int(round(float(s.duration * s.time_base) * fps)) if s.duration else 0
        return VideoInfo(str(path), fps, n, int(s.codec_context.width), int(s.codec_context.height))


def stride_for(src_fps: float, target_fps: float | None = None, stride: int | None = None) -> int:
    """Frame stride from either an explicit stride or a target sampling rate."""
    if stride is not None:
        return max(1, int(stride))
    if target_fps and target_fps > 0:
        return max(1, int(round(src_fps / target_fps)))
    return 1


def iter_frames(
    path: str | Path,
    cam: str = "cam",
    *,
    stride: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
    scale: float = 1.0,
    stats: DecodeStats | None = None,
) -> Iterator[Frame]:
    """Yield RGB frames whose global index is a multiple of ``stride``.

    Alignment on ``frame_idx % stride == 0`` (not on the start offset) keeps all
    cameras of one scene on the same frame grid, which the cross-camera fusion
    and the ground-truth comparison rely on.  ``scale`` < 1 downsizes on the CPU
    (libswscale) before the host->device copy.
    """
    import av

    t_dec = 0.0
    n_dec = n_out = 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate) if s.average_rate else 30.0
        tb = float(s.time_base)
        start_idx = int(round(start_s * fps))
        end_idx = None if duration_s is None else start_idx + int(round(duration_s * fps))
        if start_s > 0:
            c.seek(int(start_s / tb), stream=s, backward=True, any_frame=False)
        out_w = out_h = None
        if scale != 1.0:
            out_w = int(round(s.codec_context.width * scale / 2) * 2)
            out_h = int(round(s.codec_context.height * scale / 2) * 2)
        counter = 0
        it = c.decode(s)
        while True:
            t0 = time.perf_counter()
            try:
                f = next(it)
            except StopIteration:
                t_dec += time.perf_counter() - t0
                break
            idx = int(round(f.pts * tb * fps)) if f.pts is not None else counter
            counter = idx + 1
            n_dec += 1
            if idx < start_idx:
                t_dec += time.perf_counter() - t0
                continue
            if end_idx is not None and idx >= end_idx:
                t_dec += time.perf_counter() - t0
                break
            if idx % stride != 0:
                t_dec += time.perf_counter() - t0
                continue
            if out_w:
                img = f.reformat(width=out_w, height=out_h, format="rgb24").to_ndarray()
            else:
                img = f.to_ndarray(format="rgb24")
            t_dec += time.perf_counter() - t0
            n_out += 1
            yield Frame(cam, idx, idx / fps, img)
    if stats is not None:
        stats.add(t_dec, n_dec, n_out)


_END = object()


class MultiCameraBatcher:
    """Round-robin batches of frames from several cameras, decoded in background threads.

    With one camera this is plain temporal batching; with N cameras every batch
    interleaves the cameras (c0 t0, c1 t0, ..., cN t0, c0 t1, ...), i.e. the live
    "one frame per camera -> one batched forward" layout.  Per-camera order is
    always preserved, so trackers can consume results sequentially.
    """

    def __init__(self, sources: dict[str, Iterator[Frame]], batch_size: int, prefetch: int = 4):
        self.sources = sources
        self.batch_size = max(1, int(batch_size))
        self.prefetch = max(1, int(prefetch))
        self._queues: dict[str, queue.Queue] = {}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def _worker(self, cam: str, it: Iterator[Frame], q: queue.Queue) -> None:
        try:
            for fr in it:
                while not self._stop.is_set():
                    try:
                        q.put(fr, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
            q.put(_END)
        except BaseException as exc:  # propagate decode errors to the consumer
            q.put(exc)

    def __iter__(self) -> Iterator[list[Frame]]:
        for cam, it in self.sources.items():
            q: queue.Queue = queue.Queue(maxsize=self.prefetch)
            self._queues[cam] = q
            th = threading.Thread(target=self._worker, args=(cam, it, q), daemon=True, name=f"decode-{cam}")
            th.start()
            self._threads.append(th)
        active = list(self._queues)
        batch: list[Frame] = []
        try:
            while active:
                for cam in list(active):
                    item = self._queues[cam].get()
                    if item is _END:
                        active.remove(cam)
                        continue
                    if isinstance(item, BaseException):
                        raise item
                    batch.append(item)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch
        finally:
            self._stop.set()


def cameras_from_dir(video_dir: str | Path, cams: Sequence[str] | None = None) -> dict[str, Path]:
    """Map ``Camera_NNNN`` -> video path for ``videos/Camera_NNNN.mp4`` (optionally filtered)."""
    d = Path(video_dir)
    found = {p.stem: p for p in sorted(d.glob("*.mp4"))}
    if cams:
        missing = [c for c in cams if c not in found]
        if missing:
            raise FileNotFoundError(f"videos not found in {d}: {missing}")
        return {c: found[c] for c in cams}
    return found
