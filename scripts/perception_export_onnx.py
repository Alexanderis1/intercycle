#!/usr/bin/env python
"""Export RF-DETR to ONNX for ONNX Runtime's MIGraphX execution provider (AMD) - NMS stays OUT of the graph.

The exported graph returns raw ``dets`` (B, 300, 4; normalised cx,cy,w,h) and ``labels`` (B, 300, C logits).
Sigmoid, top-k, thresholding and (optional) per-class NMS run afterwards in numpy
(``intercycle.perception.detector.postprocess_detr_numpy``).  MIGraphX compiles for static shapes, so the
default is a fixed batch (``--batch-size 8``) with the real camera aspect ratio (``--shape 800x1408``);
``OnnxDetector`` pads/chunks arbitrary batches to that size.

    python scripts/perception_export_onnx.py --model medium --shape 800x1408 --batch-size 8 --out out/onnx
    python scripts/perception_bench.py --backend onnx --onnx out/onnx/rfdetr-medium.onnx --shape 800x1408
    python -m intercycle.perception.pipeline run --backend onnx --onnx out/onnx/rfdetr-medium.onnx ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intercycle.perception.detector import export_onnx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--shape", default="800x1408", help="HxW, divisible by patch_size*num_windows (32 for most sizes)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--weights", default=None, help="fine-tuned checkpoint")
    ap.add_argument("--out", default="out/onnx")
    a = ap.parse_args()
    h, w = (int(x) for x in a.shape.lower().split("x"))
    p = export_onnx(a.model, a.out, (h, w), a.batch_size, a.dynamic_batch, a.weights)
    print(f"exported: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
