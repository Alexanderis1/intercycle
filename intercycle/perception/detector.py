"""Detectors: RF-DETR (PyTorch, device-agnostic) and an optional ONNX Runtime path.

Licences: RF-DETR Nano..Large (Apache-2.0). No Ultralytics / AGPL code.

* Device-agnostic: PyTorch ROCm builds expose the HIP device as ``"cuda"``, so the same
  code runs on an NVIDIA GPU (Modal L4) and an AMD Instinct MI300X.  ROCm is detected
  with ``torch.version.hip``.
* COCO-pretrained RF-DETR knows ``person`` but has no ``forklift`` / ``pallet truck``
  class.  ``class_map`` controls which model classes become which of our labels; a
  fine-tuned checkpoint (``weights=``) that names its classes ``forklift`` /
  ``pallet_truck`` is mapped automatically through ``LABEL_ALIASES``.
* ``torch`` / ``rfdetr`` / ``onnxruntime`` are imported lazily inside the classes.
"""
from __future__ import annotations

import os
import sys
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "PERSON",
    "FORKLIFT",
    "PALLET_TRUCK",
    "Dets",
    "canonical_label",
    "describe_device",
    "RFDetrDetector",
    "OnnxDetector",
    "export_onnx",
    "onnx_providers",
    "nms_xyxy",
    "postprocess_detr_numpy",
    "MODEL_SIZES",
]

PERSON = "person"
FORKLIFT = "forklift"
PALLET_TRUCK = "pallet_truck"

MODEL_SIZES = ("nano", "small", "medium", "large")  # RF-DETR sizes under Apache-2.0

LABEL_ALIASES = {
    "person": PERSON,
    "pedestrian": PERSON,
    "worker": PERSON,
    "human": PERSON,
    "forklift": FORKLIFT,
    "forkliftreachtruck": FORKLIFT,
    "pallettruck": PALLET_TRUCK,
    "palletjack": PALLET_TRUCK,
    "pallettruckmanual": PALLET_TRUCK,
}

# COCO category ids (RF-DETR pretrained ONNX 'labels' output is indexed by COCO id).
COCO_ID_TO_NAME = {1: "person", 3: "car", 8: "truck"}


def canonical_label(name: str, class_map: dict[str, str] | None = None) -> str | None:
    """Map a model class name to one of person/forklift/pallet_truck (or ``None`` = drop)."""
    if class_map and name in class_map:
        return class_map[name]
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return LABEL_ALIASES.get(key)


@dataclass
class Dets:
    xyxy: np.ndarray  # (N, 4) float32, source-image pixels
    score: np.ndarray  # (N,) float32
    label: np.ndarray  # (N,) '<U..' canonical label

    @classmethod
    def empty(cls) -> "Dets":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.array([], dtype="<U16"))

    def __len__(self) -> int:
        return int(len(self.score))

    def select(self, mask: np.ndarray) -> "Dets":
        return Dets(self.xyxy[mask], self.score[mask], self.label[mask])


def describe_device(prefer: str = "auto") -> dict:
    """Describe the compute device without assuming CUDA vs ROCm."""
    info: dict = {"torch": None, "hip": None, "cuda": None, "device": "cpu", "name": "cpu", "is_rocm": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch missing
        info["error"] = repr(exc)
        return info
    info["torch"] = str(torch.__version__)  # TorchVersion is a str subclass that pickles as a torch type
    hip = getattr(torch.version, "hip", None)
    info["hip"] = str(hip) if hip is not None else None
    cuda = getattr(torch.version, "cuda", None)
    info["cuda"] = str(cuda) if cuda is not None else None
    info["is_rocm"] = info["hip"] is not None
    want_gpu = prefer in ("auto", "cuda", "gpu", "rocm")
    if want_gpu and torch.cuda.is_available():
        info["device"] = "cuda"  # also the correct name for ROCm builds
        info["name"] = torch.cuda.get_device_name(0)
        try:
            info["gpu_count"] = torch.cuda.device_count()
            info["total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            pass
    return info


@dataclass
class RFDetrDetector:
    """Batched RF-DETR inference returning canonical person/forklift/pallet_truck detections."""

    size: str = "medium"
    device: str = "auto"
    precision: str = "fp16"  # fp16 | bf16 | fp32 (autocast on GPU; CPU always fp32)
    threshold: float = 0.25
    shape: tuple[int, int] | None = None  # (H, W) inference size; None = model default (square)
    keep: Sequence[str] = (PERSON,)
    class_map: dict[str, str] | None = None
    weights: str | None = None  # optional fine-tuned checkpoint (.pth)
    optimize: bool = False  # RF-DETR's own inference() optimisation (forces square input)
    gpu_input: bool = True  # stage the uint8 batch in one pinned buffer, one H2D copy, convert on the GPU
    _model: object = field(default=None, init=False, repr=False)
    _staging: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    dev: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self.dev = describe_device(self.device)
        self.device_name = self.dev["device"]
        size = self.size.lower()
        if size not in MODEL_SIZES:
            raise ValueError(f"size must be one of {MODEL_SIZES} (Apache-2.0 RF-DETR sizes), got {self.size!r}")
        import rfdetr

        cls = {
            "nano": rfdetr.RFDETRNano,
            "small": rfdetr.RFDETRSmall,
            "medium": rfdetr.RFDETRMedium,
            "large": rfdetr.RFDETRLarge,
        }[size]
        kwargs: dict = {"device": self.device_name}
        if self.weights:
            kwargs["pretrain_weights"] = self.weights
        self._model = cls(**kwargs)
        self.model_resolution = int(getattr(self._model.model_config, "resolution", 0))
        if self.optimize:
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.float32)
            if self.device_name == "cpu":
                dtype = torch.float32
            self._model.inference(compile=False, dtype=dtype)

    # -- helpers -----------------------------------------------------------------
    def _autocast(self):
        torch = self._torch
        if self.device_name != "cuda" or self.precision == "fp32" or self.optimize:
            return torch.autocast(device_type="cpu", enabled=False)
        dt = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dt)

    def synchronize(self) -> None:
        if self.device_name == "cuda":
            self._torch.cuda.synchronize()

    def warmup(self, batch_size: int, hw: tuple[int, int] = (1080, 1920), n: int = 2) -> None:
        img = (np.random.rand(hw[0], hw[1], 3) * 255).astype(np.uint8)
        for _ in range(n):
            self.detect([img] * batch_size)
        self.synchronize()

    # -- inference ---------------------------------------------------------------
    def _stage_on_gpu(self, images: list[np.ndarray]):
        """One pinned staging buffer + ONE host->device copy for the whole batch, then uint8 -> float on the GPU.

        ``RFDETR.predict`` given per-image uint8 arrays pins and copies every 1080p frame separately, which made
        all model sizes cost the same ~10 ms/frame on an L4 (host-bound).  Handing it float CHW tensors that are
        already on the accelerator skips that path (documented in its docstring).
        """
        torch = self._torch
        h, w = images[0].shape[:2]
        n = len(images)
        if any(im.shape[:2] != (h, w) or im.dtype != np.uint8 for im in images):
            return images
        st = self._staging
        if st is None or st.shape[0] < n or st.shape[1:3] != (h, w):
            st = torch.empty((n, h, w, 3), dtype=torch.uint8).pin_memory()
            self._staging = st
        for i, im in enumerate(images):
            st[i].copy_(torch.from_numpy(np.ascontiguousarray(im)))
        g = st[:n].to(self.device_name, non_blocking=True)
        g = g.permute(0, 3, 1, 2).float().div_(255.0)
        return list(g.unbind(0))

    def detect(self, images: list[np.ndarray]) -> list[Dets]:
        """Run one batched forward pass; ``images`` are (H, W, 3) uint8 RGB arrays."""
        if not images:
            return []
        kw = {} if self.optimize else {"shape": self.shape}
        with self._torch.inference_mode(), self._autocast():
            batch = self._stage_on_gpu(images) if (self.gpu_input and self.device_name == "cuda") else images
            res = self._model.predict(batch, threshold=self.threshold, include_source_image=False, **kw)
        if not isinstance(res, list):
            res = [res]
        return [self._to_dets(d) for d in res]

    def _to_dets(self, d) -> Dets:
        n = len(d.xyxy)
        if n == 0:
            return Dets.empty()
        names = d.data.get("class_name") if getattr(d, "data", None) else None
        if names is None:
            names = np.array([COCO_ID_TO_NAME.get(int(c), str(c)) for c in d.class_id])
        labels = np.array([canonical_label(str(nm), self.class_map) or "" for nm in names], dtype="<U16")
        mask = np.isin(labels, list(self.keep))
        conf = d.confidence if d.confidence is not None else np.ones(n, np.float32)
        return Dets(
            np.asarray(d.xyxy, np.float32)[mask], np.asarray(conf, np.float32)[mask], labels[mask]
        )


# --------------------------------------------------------------------------- ONNX path


def export_onnx(
    size: str = "medium",
    out_dir: str | Path = "onnx",
    shape: tuple[int, int] | None = None,
    batch_size: int = 8,
    dynamic_batch: bool = False,
    weights: str | None = None,
    opset: int = 17,
) -> Path:
    """Export RF-DETR to ONNX (detection head included, *no NMS in the graph*).

    The graph returns ``dets`` (B, Q, 4) normalised cx,cy,w,h and ``labels`` (B, Q, C)
    class logits.  Top-k selection, sigmoid, thresholding and any NMS happen outside the
    graph in :func:`postprocess_detr_numpy`.  MIGraphX compiles per static shape, so a
    fixed ``batch_size`` (``dynamic_batch=False``) is recommended for the MI300X.
    """
    import rfdetr

    cls = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
    }[size.lower()]
    kwargs: dict = {"device": "cpu"}
    if weights:
        kwargs["pretrain_weights"] = weights
    model = cls(**kwargs)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return Path(
        model.export(
            output_dir=str(out_dir),
            shape=shape,
            batch_size=batch_size,
            dynamic_batch=dynamic_batch,
            opset_version=opset,
            verbose=False,
        )
    )


def onnx_providers(prefer: str = "auto", fp16: bool = True) -> list:
    """Execution-provider list: MIGraphX (AMD) > CUDA > CPU, filtered by availability.

    On ROCm 10.x the MIGraphX EP ships as the ``onnxruntime_ep_migraphx`` plugin which must be
    registered with ``ort.register_execution_provider_library`` before it is listed.
    """
    import onnxruntime as ort

    try:  # AMD plugin EP (ROCm 10.x); harmless if absent
        import onnxruntime_ep_migraphx as m  # type: ignore

        for name, path in zip(m.get_ep_names(), m.get_library_paths()):
            try:
                ort.register_execution_provider_library(name, path)
            except Exception:
                pass
    except Exception:
        pass
    avail = ort.get_available_providers()
    order = {
        "auto": ["MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "migraphx": ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[prefer]
    out: list = []
    for p in order:
        if p not in avail:
            continue
        if p == "MIGraphXExecutionProvider":
            out.append((p, {"device_id": 0, "migraphx_fp16_enable": int(fp16)}))
        else:
            out.append(p)
    if not out:
        out = ["CPUExecutionProvider"]
    return out


def nms_xyxy(xyxy: np.ndarray, score: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on one class. Returns kept indices (descending score)."""
    order = np.argsort(-score, kind="stable")
    keep: list[int] = []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def postprocess_detr_numpy(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    image_hw: tuple[int, int],
    threshold: float,
    num_select: int = 300,
    nms_iou: float | None = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DETR post-processing for ONE image, outside the graph.

    ``boxes_cxcywh`` (Q, 4) normalised, ``logits`` (Q, C).  Mirrors RF-DETR's PostProcess
    (sigmoid, top-k over query x class, scale to the source image) and adds an optional
    per-class NMS.  Returns ``xyxy`` (N, 4), ``score`` (N,), ``class_index`` (N,).
    """
    q, c = logits.shape
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    flat = prob.reshape(-1)
    k = min(num_select, flat.size)
    top = np.argpartition(-flat, k - 1)[:k]
    top = top[np.argsort(-flat[top], kind="stable")]
    scores = flat[top]
    keep = scores > threshold
    top, scores = top[keep], scores[keep]
    qi, cls = top // c, top % c
    b = boxes_cxcywh[qi].astype(np.float32)
    h, w = image_hw
    xyxy = np.stack(
        [(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h, (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h],
        axis=1,
    )
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    if nms_iou is not None and len(scores):
        kept: list[int] = []
        for cid in np.unique(cls):
            idx = np.flatnonzero(cls == cid)
            kept.extend(idx[nms_xyxy(xyxy[idx], scores[idx], nms_iou)].tolist())
        kept_arr = np.array(sorted(kept, key=lambda i: -scores[i]), dtype=np.int64)
        xyxy, scores, cls = xyxy[kept_arr], scores[kept_arr], cls[kept_arr]
    return xyxy, scores, cls


@dataclass
class OnnxDetector:
    """ONNX Runtime RF-DETR (MIGraphX EP on AMD, CUDA EP on NVIDIA, CPU fallback)."""

    onnx_path: str
    shape: tuple[int, int]  # (H, W) the graph was exported with
    threshold: float = 0.25
    keep: Sequence[str] = (PERSON,)
    class_names: dict[int, str] | None = None  # class index -> name; default COCO ids
    class_map: dict[str, str] | None = None
    nms_iou: float | None = 0.7
    providers: str = "auto"
    fp16: bool = True
    means: tuple = (0.485, 0.456, 0.406)
    stds: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        import onnxruntime as ort

        self._ort = ort
        prov = onnx_providers(self.providers, self.fp16)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=prov)
        self.provider_used = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        b = inp.shape[0]
        self.static_batch = int(b) if isinstance(b, int) else None
        self.names = self.class_names or COCO_ID_TO_NAME
        self.device_name = self.provider_used

    def _preprocess(self, images: list[np.ndarray]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        h, w = self.shape
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=False)
        m = torch.tensor(self.means).view(1, 3, 1, 1)
        s = torch.tensor(self.stds).view(1, 3, 1, 1)
        return ((x - m) / s).contiguous().numpy().astype(np.float32)

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outs = self.session.run(None, {self.input_name: x})
        names = [o.name for o in self.session.get_outputs()]
        d = dict(zip(names, outs))
        return d["dets"], d["labels"]

    def synchronize(self) -> None:
        return None

    def warmup(self, batch_size: int, hw: tuple[int, int] = (1080, 1920), n: int = 2) -> None:
        img = (np.random.rand(hw[0], hw[1], 3) * 255).astype(np.uint8)
        for _ in range(n):
            self.detect([img] * batch_size)

    def detect(self, images: list[np.ndarray]) -> list[Dets]:
        if not images:
            return []
        h0, w0 = images[0].shape[:2]
        x = self._preprocess(images)
        n = len(images)
        if self.static_batch and self.static_batch != n:
            sb = self.static_batch
            pad = (-n) % sb
            if pad:
                x = np.concatenate([x, np.zeros((pad,) + x.shape[1:], x.dtype)])
            parts = [self._run(x[i : i + sb]) for i in range(0, len(x), sb)]
            boxes = np.concatenate([p[0] for p in parts])[:n]
            logits = np.concatenate([p[1] for p in parts])[:n]
        else:
            boxes, logits = self._run(x)
        out: list[Dets] = []
        for i in range(n):
            xyxy, sc, ci = postprocess_detr_numpy(boxes[i], logits[i], (h0, w0), self.threshold, nms_iou=self.nms_iou)
            labels = np.array(
                [canonical_label(self.names.get(int(c), str(int(c))), self.class_map) or "" for c in ci], dtype="<U16"
            )
            m = np.isin(labels, list(self.keep)) if len(labels) else np.zeros(0, bool)
            out.append(Dets(xyxy[m].astype(np.float32), sc[m].astype(np.float32), labels[m]))
        return out


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


def env_summary() -> dict:
    """Small reproducibility record for result files."""
    d = describe_device()
    d["python"] = sys.version.split()[0]
    for p in ("rfdetr", "trackers", "supervision", "av", "onnxruntime"):
        try:
            from importlib.metadata import version

            d[p] = version(p)
        except Exception:
            d[p] = None
    return d
