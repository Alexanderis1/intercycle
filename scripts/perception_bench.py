#!/usr/bin/env python
"""Perception throughput benchmark: prints torch.version.hip, the GPU name and frames/s.

Runs unchanged on an NVIDIA GPU (CUDA) and on an AMD Instinct MI300X (ROCm, where PyTorch still
calls the device "cuda"; ROCm is detected through ``torch.version.hip``).

    python scripts/perception_bench.py --videos data/.../videos/Camera_0003.mp4 --model medium
    python scripts/perception_bench.py                      # synthetic frames, no dataset needed
    python scripts/perception_bench.py --backend onnx --onnx out/onnx/rfdetr-medium.onnx --shape 800x1408
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intercycle.perception.bench import format_report, run_benchmark, save_report  # noqa: E402


def _hw(s: str):
    if s.lower() in ("none", "square"):
        return None
    h, w = s.lower().split("x")
    return int(h), int(w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", nargs="*", default=None, help="video files (default: synthetic frames)")
    ap.add_argument("--model", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--shape", type=_hw, default=(800, 1408))
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--onnx", dest="onnx_path", default=None)
    ap.add_argument("--frames", type=int, default=64, help="frames per timed pass")
    ap.add_argument("--optimize", action="store_true", help="RF-DETR .inference() (square input only)")
    ap.add_argument("--no-gpu-input", action="store_true", help="let RF-DETR copy/pin every frame itself (slower)")
    ap.add_argument("--json", default=None, help="write the full result as JSON")
    a = ap.parse_args()
    res = run_benchmark(
        a.videos, a.model, tuple(a.batch_sizes), a.shape, a.precision, a.device, a.backend, a.onnx_path,
        a.frames, optimize=a.optimize, gpu_input=not a.no_gpu_input,
    )
    print()
    print(format_report(res))
    if a.json:
        save_report(res, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
