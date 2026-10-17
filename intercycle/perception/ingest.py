"""Fetch a SmartSpaces scene (selected cameras only) and compact its ground truth.

Dataset: ``nvidia/PhysicalAI-SmartSpaces`` on Hugging Face, CC-BY-4.0, **not gated** (anonymous
download works; ``HF_TOKEN`` only raises rate limits).  Verified layout::

    MTMC_Tracking_2026/{train,val,test}/Warehouse_NNN/
        videos/Camera_NNNN.mp4        1080p, 30 fps, 9000 frames (5 min) per camera
        calibration.json              2025 schema (cameraMatrix / homography per camera)
        ground_truth.json             ~280 MB for Warehouse_000 (3D + per-camera visible 2D boxes)
        map.png
        depth_maps/Camera_NNNN.h5     7-14 GB EACH - never downloaded here

Only videos of the requested cameras, ``calibration.json``, ``ground_truth.json`` and ``map.png`` are
fetched (``allow_patterns``).  Real-world test scenes Warehouse_026 / _027 have no ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .geometry import grid_zones, save_zones
from .gt import build_gt_tables, gt_bounds, iter_gt_frames, save_gt_tables

__all__ = ["REPO_ID", "DEFAULT_CAMERAS", "scene_patterns", "ingest_scene", "main"]

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
ROOT = "MTMC_Tracking_2026"
# Warehouse_000 cameras with the most visible persons in the first ~14 s of ground truth.
DEFAULT_CAMERAS = ("Camera_0003", "Camera_0010", "Camera_0015", "Camera_0001")


def scene_patterns(split: str, scene: str, cams: list[str], with_gt: bool = True) -> list[str]:
    base = f"{ROOT}/{split}/{scene}"
    pats = [f"{base}/calibration.json", f"{base}/map.png"]
    pats += [f"{base}/videos/{c}.mp4" for c in cams]
    if with_gt:
        pats.append(f"{base}/ground_truth.json")
    return pats


def ingest_scene(
    local_dir: str | Path,
    split: str = "train",
    scene: str = "Warehouse_000",
    cams: list[str] | None = None,
    with_gt: bool = True,
    zone_grid: tuple[int, int] = (3, 3),
    token: str | None = None,
    log=print,
) -> dict:
    """Download the scene subset into ``local_dir`` and write ``<scene_dir>/gt/*.parquet`` + zones.json."""
    from huggingface_hub import snapshot_download

    cams = list(cams or DEFAULT_CAMERAS)
    local_dir = Path(local_dir)
    pats = scene_patterns(split, scene, cams, with_gt)
    log(f"downloading {len(pats)} files from {REPO_ID}: {pats}")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", allow_patterns=pats, local_dir=str(local_dir), token=token, max_workers=8
    )
    scene_dir = local_dir / ROOT / split / scene
    out = {"scene_dir": str(scene_dir), "cameras": cams, "gt_dir": None}
    gt_path = scene_dir / "ground_truth.json"
    if with_gt and gt_path.exists():
        log(f"compacting {gt_path} ({gt_path.stat().st_size / 1e6:.0f} MB) for {cams}")
        boxes, objs = build_gt_tables(iter_gt_frames(gt_path), cams)
        gt_dir = scene_dir / "gt"
        save_gt_tables(boxes, objs, gt_dir)
        bounds = gt_bounds(objs, "person", margin=1.0)
        save_zones(
            grid_zones(bounds, *zone_grid),
            gt_dir / "zones.json",
            {"source": "grid over ground-truth person extent in selected cameras", "bounds": bounds, "grid": list(zone_grid)},
        )
        out.update(
            gt_dir=str(gt_dir),
            gt_boxes=int(len(boxes)),
            gt_objects=int(len(objs)),
            gt_frames=int(objs["frame"].nunique()),
            zones=str(gt_dir / "zones.json"),
            bounds=bounds,
        )
    (scene_dir / "ingest.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m intercycle.perception.ingest", description=__doc__.split("\n\n")[0])
    ap.add_argument("--local-dir", default="data/smartspaces")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="Warehouse_000")
    ap.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    ap.add_argument("--no-gt", action="store_true")
    a = ap.parse_args(argv)
    res = ingest_scene(a.local_dir, a.split, a.scene, a.cameras, with_gt=not a.no_gt)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
