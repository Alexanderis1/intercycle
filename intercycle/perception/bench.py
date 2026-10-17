"""Throughput benchmark: device report + model-only fps per batch size + decode fps.

Used by ``scripts/perception_bench.py`` (local / MI300X) and ``cloud/modal_perception.py`` (L4).
Frames are decoded once up-front (excluded from the model timing) so the model numbers are
GPU-bound; CPU decode throughput is measured separately and the end-to-end pipeline fps comes
from the real pipeline run (``meta.json``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .detector import RFDetrDetector, OnnxDetector, describe_device, env_summary
from .video import iter_frames, probe

__all__ = ["run_benchmark", "format_report"]


def _load_frames(videos: list[str], n_per_video: int, stride: int, scale: float = 1.0) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for v in videos:
        k = 0
        for fr in iter_frames(v, "bench", stride=stride, scale=scale):
            frames.append(fr.image)
            k += 1
            if k >= n_per_video:
                break
    return frames


def _measure_decode(video: str, n: int = 300) -> dict:
    t0 = time.perf_counter()
    k = 0
    for _ in iter_frames(video, "bench", stride=1):
        k += 1
        if k >= n:
            break
    dt = time.perf_counter() - t0
    return {"frames": k, "seconds": dt, "fps": k / dt if dt else None}


def run_benchmark(
    videos: list[str] | None = None,
    model: str = "medium",
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    shape: tuple[int, int] | None = (800, 1408),
    precision: str = "fp16",
    device: str = "auto",
    backend: str = "torch",
    onnx_path: str | None = None,
    n_frames: int = 64,
    sample_fps: float = 5.0,
    optimize: bool = False,
    gpu_input: bool = True,
    log=print,
) -> dict:
    """Return a dict with device info and ``results[batch] = {fps, ms_per_frame, ...}``."""
    dev = describe_device(device)
    log(f"torch={dev.get('torch')} hip={dev.get('hip')} cuda={dev.get('cuda')} device={dev['device']} gpu={dev['name']}")
    hw = (1080, 1920)
    frames: list[np.ndarray]
    src = "synthetic"
    dec = None
    if videos:
        info = probe(videos[0])
        stride = max(1, int(round(info.fps / sample_fps)))
        per = max(1, int(np.ceil(n_frames / len(videos))))
        frames = _load_frames(videos, per, stride)
        hw = frames[0].shape[:2]
        src = f"{len(videos)} video(s), stride {stride}"
        dec = _measure_decode(videos[0])
        log(f"decode (CPU, PyAV, 1 thread-pool): {dec['fps']:.0f} frames/s at {info.width}x{info.height}")
    else:
        rng = np.random.default_rng(0)
        frames = [(rng.random((hw[0], hw[1], 3)) * 255).astype(np.uint8) for _ in range(8)]
    while len(frames) < n_frames:
        frames = frames + frames
    frames = frames[:n_frames]

    if backend == "onnx":
        det = OnnxDetector(onnx_path, shape=shape or (576, 576))
    else:
        det = RFDetrDetector(size=model, device=device, precision=precision, shape=shape, optimize=optimize,
                             gpu_input=gpu_input)
    results: dict[int, dict] = {}
    for bs in batch_sizes:
        det.warmup(bs, hw=hw, n=2)
        det.synchronize()
        t0 = time.perf_counter()
        n_done = 0
        for i in range(0, len(frames), bs):
            chunk = frames[i : i + bs]
            det.detect(chunk)
            n_done += len(chunk)
        det.synchronize()
        dt = time.perf_counter() - t0
        fps = n_done / dt
        results[bs] = {"fps": fps, "ms_per_frame": 1000.0 / fps, "frames": n_done, "seconds": dt}
        log(f"  model={model} batch={bs:>3}: {fps:8.1f} frames/s  ({1000.0 / fps:6.1f} ms/frame)")
    peak = None
    try:
        import torch

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    best_bs = max(results, key=lambda b: results[b]["fps"])
    best = results[best_bs]["fps"]
    out = {
        "device": dev,
        "env": env_summary(),
        "model": model if backend == "torch" else f"onnx:{onnx_path}",
        "backend": backend,
        "provider": getattr(det, "provider_used", None),
        "precision": precision,
        "gpu_input": gpu_input,
        "shape": shape,
        "frames_source": src,
        "frame_hw": list(hw),
        "decode": dec,
        "results": results,
        "best_batch": best_bs,
        "best_fps": best,
        "cameras_realtime_at_5fps": best / 5.0,
        "cameras_realtime_at_30fps": best / 30.0,
        "peak_gpu_mem_gb": peak,
    }
    log(
        f"best: {best:.1f} frames/s at batch {best_bs} -> {best / 5.0:.0f} cameras real-time at 5 fps/cam, "
        f"{best / 30.0:.1f} at 30 fps/cam (model only; decode on CPU is separate)"
    )
    return out


def format_report(res: dict) -> str:
    d = res["device"]
    lines = [
        f"device      : {d['device']} / {d['name']}",
        f"torch       : {d.get('torch')}  hip={d.get('hip')}  cuda={d.get('cuda')}  rocm={d.get('is_rocm')}",
        f"model       : {res['model']} ({res['backend']}, {res['precision']}, shape={res['shape']})",
    ]
    for bs, r in res["results"].items():
        lines.append(f"batch {int(bs):>3}   : {r['fps']:.1f} fps")
    lines.append(f"best        : {res['best_fps']:.1f} fps (batch {res['best_batch']}); "
                 f"{res['cameras_realtime_at_5fps']:.0f} cams @5fps, {res['cameras_realtime_at_30fps']:.1f} cams @30fps")
    return "\n".join(lines)


def save_report(res: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
