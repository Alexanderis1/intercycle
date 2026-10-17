"""Ground-truth handling for the CCTV post-process.

The GT (instance map, depth) is warped with EXACTLY the same sampling map as the
RGB image (see ``cctv_post.Geometry``), using nearest neighbour, so that boxes
computed from the warped instance mask line up with what is visible in the frame.

Conventions
-----------
* instance map: uint16, 0 = background.
* depth: float32 metres (NaN / inf are kept as they are, nearest neighbour only).
* ids: ``{instance_id: {"cls": str, "entity_id": str}}`` (JSON keys are strings).
* boxes: ``bbox = [x1, y1, x2, y2]`` in output pixels, x2/y2 EXCLUSIVE (pixel edge
  convention: ``w = x2 - x1`` is the width in pixels, COCO-compatible).

GT file layout read by :func:`load_gt` for a frame ``<stem>.png``::

    <gt_dir>/<stem>_inst.png      (also accepted: <stem>_instance.png, <stem>.inst.png)
    <gt_dir>/<stem>_depth.npy
    <gt_dir>/<stem>_ids.json      (or one shared <gt_dir>/ids.json / instances.json)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

INST_SUFFIXES = ("_inst.png", "_instance.png", ".inst.png", "_instances.png")
IDS_SHARED_NAMES = ("ids.json", "instances.json", "instance_ids.json")


@dataclass
class GtFrame:
    """Ground truth of one frame (input or output side)."""

    inst: Optional[np.ndarray] = None  # (H, W) uint16, 0 = background
    depth: Optional[np.ndarray] = None  # (H, W) float32 metres
    ids: Optional[dict] = None  # {instance_id: {"cls": ..., "entity_id": ...}}
    boxes: list = field(default_factory=list)  # filled after warping


# ----------------------------------------------------------------------------
# warping
# ----------------------------------------------------------------------------
def warp_nearest(arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Nearest-neighbour warp: ``out[i, j] = arr[rint(map_y[i, j]), rint(map_x[i, j])]``.

    Uses the same pixel convention as ``cv2.remap`` (pixel centres at integer
    coordinates). Works for any dtype (uint16 ids, float32 depth). Samples
    that fall outside are clamped to the border.
    """
    h, w = arr.shape[:2]
    ix = np.clip(np.rint(map_x), 0, w - 1).astype(np.intp)
    iy = np.clip(np.rint(map_y), 0, h - 1).astype(np.intp)
    return arr[iy, ix]


def warp_gt(gt: GtFrame, map_x: np.ndarray, map_y: np.ndarray, min_pixels: int = 24) -> GtFrame:
    """Warp a :class:`GtFrame` with the image sampling map and compute boxes."""
    inst = None if gt.inst is None else warp_nearest(gt.inst, map_x, map_y)
    depth = None if gt.depth is None else warp_nearest(gt.depth, map_x, map_y)
    out = GtFrame(inst=inst, depth=depth, ids=gt.ids)
    if inst is not None:
        out.boxes = boxes_from_instances(inst, gt.ids, min_pixels)
    return out


# ----------------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------------
def boxes_from_instances(inst: np.ndarray, ids: Optional[dict], min_pixels: int = 24) -> list:
    """Boxes from an instance map. Instances below ``min_pixels`` visible pixels are dropped."""
    ys, xs = np.nonzero(inst)
    if ys.size == 0:
        return []
    v = inst[ys, xs]
    order = np.argsort(v, kind="stable")
    v, xs, ys = v[order], xs[order], ys[order]
    uniq, start = np.unique(v, return_index=True)
    counts = np.diff(np.append(start, v.size))
    x1 = np.minimum.reduceat(xs, start)
    x2 = np.maximum.reduceat(xs, start) + 1
    y1 = np.minimum.reduceat(ys, start)
    y2 = np.maximum.reduceat(ys, start) + 1
    boxes = []
    for k in range(uniq.size):
        if counts[k] < min_pixels:
            continue
        iid = int(uniq[k])
        meta = (ids or {}).get(str(iid)) or (ids or {}).get(iid) or {}
        boxes.append(
            {
                "instance_id": iid,
                "cls": str(meta.get("cls", "unknown")),
                "entity_id": str(meta.get("entity_id", f"inst{iid}")),
                "bbox": [int(x1[k]), int(y1[k]), int(x2[k]), int(y2[k])],
                "pixels": int(counts[k]),
            }
        )
    return boxes


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def _imread_gray16(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:  # colour-coded fallback: take first channel
        a = a[..., 0]
    return a.astype(np.uint16, copy=False)


def load_gt(gt_dir, stem: str) -> Optional[GtFrame]:
    """Load GT of frame ``stem`` from ``gt_dir``; None if nothing is found."""
    gt_dir = Path(gt_dir)
    gt = GtFrame()
    for suf in INST_SUFFIXES:
        p = gt_dir / f"{stem}{suf}"
        if p.exists():
            gt.inst = _imread_gray16(p)
            break
    p = gt_dir / f"{stem}_depth.npy"
    if p.exists():
        gt.depth = np.load(p).astype(np.float32, copy=False)
    cands = [gt_dir / f"{stem}_ids.json"] + [gt_dir / n for n in IDS_SHARED_NAMES]
    for p in cands:
        if p.exists():
            gt.ids = json.loads(p.read_text(encoding="utf-8"))
            break
    if gt.inst is None and gt.depth is None:
        return None
    return gt


def save_gt(gt: GtFrame, gt_out_dir, stem: str, extra: Optional[dict] = None) -> None:
    """Write warped GT: ``<stem>_inst.png`` (16-bit), ``<stem>_depth.npy``, ``<stem>_boxes.json``."""
    d = Path(gt_out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if gt.inst is not None:
        cv2.imwrite(str(d / f"{stem}_inst.png"), gt.inst.astype(np.uint16))
    if gt.depth is not None:
        np.save(d / f"{stem}_depth.npy", gt.depth.astype(np.float32))
    if gt.inst is not None:
        h, w = gt.inst.shape
        doc = {"width": w, "height": h, "boxes": gt.boxes}
        if extra:
            doc.update(extra)
        (d / f"{stem}_boxes.json").write_text(json.dumps(doc), encoding="utf-8")


def build_coco(frames: list, categories: Optional[list] = None) -> dict:
    """COCO-style annotation dict for a sequence.

    ``frames``: list of ``{"file_name", "width", "height", "frame", "t", "boxes"}``.
    Category ids are 1..N over the sorted class names found in the boxes unless
    ``categories`` (list of class names) is given. Every annotation also carries
    ``instance_id``, ``entity_id`` (use as track id) and ``visible_pixels``.
    ``bbox`` is COCO ``[x, y, w, h]``; ``area`` is the visible pixel count.
    """
    if categories is None:
        categories = sorted({b["cls"] for f in frames for b in f["boxes"]})
    cat_id = {c: i + 1 for i, c in enumerate(categories)}
    images, anns = [], []
    for i, f in enumerate(frames):
        images.append(
            {
                "id": i + 1,
                "file_name": f["file_name"],
                "width": f["width"],
                "height": f["height"],
                "frame": f.get("frame", i),
                "t": f.get("t", 0.0),
            }
        )
        for b in f["boxes"]:
            x1, y1, x2, y2 = b["bbox"]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i + 1,
                    "category_id": cat_id[b["cls"]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": b["pixels"],
                    "iscrowd": 0,
                    "instance_id": b["instance_id"],
                    "entity_id": b["entity_id"],
                    "visible_pixels": b["pixels"],
                }
            )
    return {
        "info": {"description": "InterCycle synthetic CCTV sequence (SYNTHETIC)"},
        "images": images,
        "annotations": anns,
        "categories": [{"id": cat_id[c], "name": c} for c in categories],
    }


def write_coco(path, frames: list, categories: Optional[list] = None) -> dict:
    doc = build_coco(frames, categories)
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return doc
