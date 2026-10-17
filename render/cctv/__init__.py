"""CCTV post-process: turns clean Blender frames into CCTV-like frames with aligned ground truth.

Names are imported lazily so that ``python -m render.cctv.cctv_post`` does not import the module twice.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "cctv_post": (
        "CctvParams", "Geometry", "build_geometry", "camera_params", "compress_still", "display_cam", "draw_overlay",
        "effective_map", "frame_params", "load_rgb", "process_frame", "process_sequence", "process_still",
        "recommended_overscan", "timestamp_text",
    ),
    "gt": (
        "GtFrame", "boxes_from_instances", "build_coco", "load_gt", "save_gt", "warp_gt", "warp_nearest", "write_coco",
    ),
}
_WHERE = {name: mod for mod, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name):
    if name in _WHERE:
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
