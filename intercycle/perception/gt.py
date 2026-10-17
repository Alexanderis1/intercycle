"""Ground-truth ingestion for NVIDIA PhysicalAI-SmartSpaces ``ground_truth.json``.

The real files use space-separated keys (``"object type"``, ``"3d location"``,
``"2d bounding box visible"``) although the dataset README documents underscores; both
are accepted.  ``Warehouse_000/ground_truth.json`` is ~280 MB, so it is streamed with
``ijson`` when a C backend is available and compacted into two small parquet tables:

* ``gt_boxes``   : camera, frame, object_id, cls, x1, y1, x2, y2   (selected cameras only)
* ``gt_objects`` : frame, object_id, cls, wx, wy, wz, cams, max_h  (objects visible in a selected camera;
  ``wx, wy`` = floor position in metres, ``max_h`` = tallest visible 2D box in px)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = ["canonical_gt_class", "iter_gt_frames", "build_gt_tables", "save_gt_tables", "load_gt_tables", "gt_bounds"]

_GT_CLASS = {"person": "person", "forklift": "forklift", "pallettruck": "pallet_truck"}


def canonical_gt_class(name: str) -> str:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return _GT_CLASS.get(key, key)  # robots (NovaCarter, Transporter, ...) keep their lower-cased name


def _get(o: dict, *names: str):
    for n in names:
        if n in o:
            return o[n]
    raise KeyError(names[0])


def iter_gt_frames(path: str | Path, max_frames: int | None = None) -> Iterator[tuple[int, list[dict]]]:
    """Yield ``(frame_id, [object dict, ...])`` from a SmartSpaces ground_truth.json."""
    n = 0
    try:
        import ijson

        backend = getattr(ijson, "backend", None)
        name = getattr(backend, "backend_name", None) or getattr(ijson, "backend_name", "")
        use_ijson = str(name) != "python"
    except Exception:  # pragma: no cover
        use_ijson = False
    if use_ijson:
        with open(path, "rb") as fh:
            for key, objs in ijson.kvitems(fh, "", use_float=True):
                yield int(key), objs
                n += 1
                if max_frames is not None and n >= max_frames:
                    return
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key in sorted(data, key=int):
            yield int(key), data[key]
            n += 1
            if max_frames is not None and n >= max_frames:
                return


def build_gt_tables(
    frames: Iterable[tuple[int, list[dict]]],
    cams: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact per-camera GT boxes + per-object floor positions for ``cams``."""
    camset = set(cams)
    box_rows: list[tuple] = []
    obj_rows: list[tuple] = []
    for fr, objs in frames:
        for o in objs:
            vis = _get(o, "2d bounding box visible", "2d_bounding_box_visible")
            sel = {c: b for c, b in vis.items() if c in camset}
            if not sel:
                continue
            cls = canonical_gt_class(_get(o, "object type", "object_type"))
            oid = int(_get(o, "object id", "object_id"))
            loc = _get(o, "3d location", "3d_location")
            max_h = 0.0
            for c, b in sel.items():
                box_rows.append((c, fr, oid, cls, float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                max_h = max(max_h, float(b[3]) - float(b[1]))
            obj_rows.append((fr, oid, cls, float(loc[0]), float(loc[1]), float(loc[2]), "|".join(sorted(sel)), max_h))
    boxes = pd.DataFrame(box_rows, columns=["camera", "frame", "object_id", "cls", "x1", "y1", "x2", "y2"])
    objs = pd.DataFrame(obj_rows, columns=["frame", "object_id", "cls", "wx", "wy", "wz", "cams", "max_h"])
    return boxes, objs


def save_gt_tables(boxes: pd.DataFrame, objs: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes.to_parquet(out / "gt_boxes.parquet", index=False)
    objs.to_parquet(out / "gt_objects.parquet", index=False)


def load_gt_tables(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    return pd.read_parquet(out / "gt_boxes.parquet"), pd.read_parquet(out / "gt_objects.parquet")


def gt_bounds(objs: pd.DataFrame, cls: str | None = "person", margin: float = 1.0) -> tuple[float, float, float, float]:
    """Floor bounding box (xmin, ymin, xmax, ymax) of GT objects, padded by ``margin`` metres."""
    d = objs if cls is None else objs[objs["cls"] == cls]
    if len(d) == 0:
        d = objs
    return (
        float(d["wx"].min() - margin),
        float(d["wy"].min() - margin),
        float(d["wx"].max() + margin),
        float(d["wy"].max() + margin),
    )
