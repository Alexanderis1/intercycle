"""Batched multi-camera person detection + tracking -> floor positions -> zone occupancy.

    python -m intercycle.perception.pipeline run \
        --scene-dir data/smartspaces/MTMC_Tracking_2026/train/Warehouse_000 \
        --cameras Camera_0003 Camera_0008 --fps 5 --duration-s 60 --out out/perception/demo

Stages
  1. CPU decode (PyAV) with stride / target-fps subsampling, round-robin multi-camera batches.
  2. RF-DETR batched inference (PyTorch "cuda" = NVIDIA CUDA *or* AMD ROCm, or ONNX Runtime).
  3. ByteTrack (roboflow ``trackers``) per camera and class.
  4. Foot point (bottom-centre of the box) -> floor plane through the calibrated homography.
  5. Cross-camera de-duplication of floor points, zone polygons -> counts per step and per time bin.

Outputs (``--out``): ``tracks.parquet``, ``detections.parquet``, ``occupancy_steps.parquet``,
``occupancy.parquet`` (binned), ``zones.json``, ``meta.json`` (timings, throughput, device).
``forklift`` / ``pallet_truck`` rows are only produced by a fine-tuned detector; with
``--gt-dir`` and ``--gt-classes`` they are injected from ground truth into the occupancy
(flagged ``source="gt"``) so the downstream time series is complete until a fine-tune exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import Zone, foot_points, grid_zones, load_calibration, load_zones, save_zones
from .occupancy import auto_bin_seconds, fuse_world_points, occupancy_bins, occupancy_steps
from .video import DecodeStats, MultiCameraBatcher, cameras_from_dir, iter_frames, probe, stride_for

__all__ = ["PipelineConfig", "process_cameras", "aggregate_run", "run", "build_detector", "main"]


@dataclass
class PipelineConfig:
    scene_dir: str
    out_dir: str
    cameras: list[str] | None = None
    # sampling
    fps: float = 5.0  # target sampling rate per camera (frames/s); ignored if stride is set
    stride: int | None = None
    start_s: float = 0.0
    duration_s: float | None = None
    scale: float = 1.0  # CPU downscale before the host->device copy
    # detector
    backend: str = "torch"  # torch | onnx
    model: str = "medium"  # nano | small | medium | large
    device: str = "auto"
    precision: str = "fp16"
    threshold: float = 0.25
    shape: tuple[int, int] | None = (800, 1408)  # (H, W); None = square model default
    batch_size: int = 8
    weights: str | None = None
    onnx_path: str | None = None
    classes: tuple[str, ...] = ("person",)
    class_map: dict[str, str] | None = None
    optimize: bool = False
    gpu_input: bool = True
    # tracking
    lost_track_s: float = 3.0
    min_consecutive: int = 2
    high_conf: float = 0.5
    activation: float = 0.4
    # occupancy
    zones: str | None = None
    zone_grid: tuple[int, int] = (3, 3)
    bin_seconds: float | None = None
    fuse_radius_m: float = 0.8
    min_box_h: float = 20.0  # occupancy only counts tracks at least this tall (px) in the camera view
    min_score: float = 0.0  # ... and at least this confident
    gt_dir: str | None = None
    gt_classes: tuple[str, ...] = ()
    tag: str | None = None


def build_detector(cfg: PipelineConfig):
    from .detector import OnnxDetector, RFDetrDetector

    if cfg.backend == "onnx":
        if not cfg.onnx_path:
            raise ValueError("--onnx is required with --backend onnx")
        shape = cfg.shape or (576, 576)
        return OnnxDetector(
            cfg.onnx_path, shape=shape, threshold=cfg.threshold, keep=cfg.classes, class_map=cfg.class_map
        )
    return RFDetrDetector(
        size=cfg.model,
        device=cfg.device,
        precision=cfg.precision,
        threshold=cfg.threshold,
        shape=cfg.shape,
        keep=cfg.classes,
        class_map=cfg.class_map,
        weights=cfg.weights,
        optimize=cfg.optimize,
        gpu_input=cfg.gpu_input,
    )


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------- stage 1


def process_cameras(cfg: PipelineConfig, detector=None, log=print) -> dict:
    """Detect + track the selected cameras; write ``tracks_<tag>.parquet`` etc. Returns the meta dict."""
    from .detector import env_summary
    from .tracking import CameraTracker, TrackerConfig

    scene = Path(cfg.scene_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calib = load_calibration(scene / "calibration.json")
    vids = cameras_from_dir(scene / "videos", cfg.cameras)
    infos = {c: probe(p) for c, p in vids.items()}
    src_fps = infos[next(iter(infos))].fps
    stride = stride_for(src_fps, cfg.fps, cfg.stride)
    eff_fps = src_fps / stride
    tag = cfg.tag or ("all" if cfg.cameras is None else "+".join(sorted(vids)))

    det = detector or build_detector(cfg)
    det.warmup(cfg.batch_size)
    tcfg = TrackerConfig(cfg.lost_track_s, cfg.min_consecutive, cfg.high_conf, cfg.activation)
    trackers = {c: CameraTracker(eff_fps, cfg.classes, tcfg) for c in vids}

    dstats = DecodeStats()
    sources = {
        c: iter_frames(
            p, c, stride=stride, start_s=cfg.start_s, duration_s=cfg.duration_s, scale=cfg.scale, stats=dstats
        )
        for c, p in vids.items()
    }
    batcher = MultiCameraBatcher(sources, cfg.batch_size, prefetch=max(2, cfg.batch_size))

    det_rows: list[tuple] = []
    trk_rows: list[tuple] = []
    frames_seen: set[int] = set()
    n_frames = 0
    infer_s = track_s = 0.0
    t_wall = time.perf_counter()
    for batch in batcher:
        t0 = time.perf_counter()
        results = det.detect([f.image for f in batch])
        det.synchronize()
        infer_s += time.perf_counter() - t0
        t1 = time.perf_counter()
        for fr, d in zip(batch, results):
            info = infos[fr.cam]
            sx = info.width / fr.image.shape[1]
            sy = info.height / fr.image.shape[0]
            if sx != 1.0 or sy != 1.0:
                d.xyxy = d.xyxy * np.array([sx, sy, sx, sy], np.float32)
            frames_seen.add(fr.frame_idx)
            for b, s, lab in zip(d.xyxy, d.score, d.label):
                det_rows.append((fr.cam, fr.frame_idx, lab, float(s), *map(float, b)))
            tb = trackers[fr.cam].update(d, fr.t)
            for b, s, tid, lab in zip(tb.xyxy, tb.score, tb.track_id, tb.label):
                trk_rows.append((fr.cam, fr.frame_idx, fr.t, int(tid), lab, float(s), *map(float, b)))
        track_s += time.perf_counter() - t1
        n_frames += len(batch)
    wall_s = time.perf_counter() - t_wall

    box_cols = ["x1", "y1", "x2", "y2"]
    dets_df = pd.DataFrame(det_rows, columns=["camera", "frame", "cls", "score", *box_cols])
    trk_df = pd.DataFrame(
        trk_rows, columns=["camera", "frame", "t", "track_id", "cls", "score", *box_cols]
    )
    # foot point -> floor
    trk_df["foot_u"] = np.nan
    trk_df["foot_v"] = np.nan
    trk_df["wx"] = np.nan
    trk_df["wy"] = np.nan
    for cam, idx in trk_df.groupby("camera").groups.items():
        sub = trk_df.loc[idx, box_cols].to_numpy()
        fp = foot_points(sub)
        wxy = calib[cam].image_to_floor(fp)
        trk_df.loc[idx, ["foot_u", "foot_v"]] = fp
        trk_df.loc[idx, ["wx", "wy"]] = wxy
    dets_df.to_parquet(out / f"detections_{tag}.parquet", index=False)
    trk_df.to_parquet(out / f"tracks_{tag}.parquet", index=False)

    meta = {
        "tag": tag,
        "cameras": sorted(vids),
        "src_fps": src_fps,
        "stride": stride,
        "sampled_fps": eff_fps,
        "frame_first": min(frames_seen) if frames_seen else None,
        "frame_last": max(frames_seen) if frames_seen else None,
        "n_frames": n_frames,
        "n_detections": len(dets_df),
        "n_track_rows": len(trk_df),
        "batch_size": cfg.batch_size,
        "model": cfg.model if cfg.backend == "torch" else f"onnx:{cfg.onnx_path}",
        "backend": cfg.backend,
        "precision": cfg.precision,
        "shape": cfg.shape,
        "wall_s": wall_s,
        "infer_s": infer_s,
        "track_s": track_s,
        "decode_thread_s": dstats.decode_s,
        "frames_decoded": dstats.frames_decoded,
        "fps_e2e": n_frames / wall_s if wall_s else None,
        "fps_infer_only": n_frames / infer_s if infer_s else None,
        "fps_decode_per_thread": dstats.frames_decoded / dstats.decode_s if dstats.decode_s else None,
        "cameras_realtime_e2e": (n_frames / wall_s) / eff_fps if wall_s else None,
        "config": asdict(cfg),
        "env": env_summary(),
    }
    _write_json(out / f"meta_{tag}.json", meta)
    log(
        f"[{tag}] {n_frames} frames @ {eff_fps:.1f} fps/cam in {wall_s:.1f}s -> "
        f"e2e {meta['fps_e2e']:.1f} fps, infer-only {meta['fps_infer_only']:.1f} fps "
        f"({meta['cameras_realtime_e2e']:.1f} cameras real-time at {eff_fps:.1f} fps); "
        f"{len(trk_df)} track rows, device={det.device_name}"
    )
    return meta


# --------------------------------------------------------------------------- stage 2


def _load_zones_for_run(
    cfg: PipelineConfig, out: Path, tracks: pd.DataFrame, gt_objs: pd.DataFrame | None
) -> list[Zone]:
    if cfg.zones:
        zones = load_zones(cfg.zones)
        meta = {"source": str(cfg.zones)}
    else:
        from .gt import gt_bounds

        if gt_objs is not None and len(gt_objs):
            bounds = gt_bounds(gt_objs, "person", margin=1.0)
            meta = {"source": "grid over ground-truth person extent", "bounds": bounds}
        elif len(tracks):
            lo = tracks[["wx", "wy"]].quantile(0.02)
            hi = tracks[["wx", "wy"]].quantile(0.98)
            bounds = (float(lo.wx), float(lo.wy), float(hi.wx), float(hi.wy))
            meta = {"source": "grid over 2-98 percentile of tracked floor points", "bounds": bounds}
        else:
            bounds = (0.0, 0.0, 1.0, 1.0)
            meta = {"source": "empty"}
        zones = grid_zones(bounds, cfg.zone_grid[0], cfg.zone_grid[1])
        meta["grid"] = list(cfg.zone_grid)
    save_zones(zones, out / "zones.json", meta)
    return zones


def _inject_gt_points(gt_objs: pd.DataFrame, frames: np.ndarray, classes: tuple[str, ...], min_h: float) -> pd.DataFrame:
    d = gt_objs[gt_objs["cls"].isin(list(classes)) & gt_objs["frame"].isin(frames) & (gt_objs["max_h"] >= min_h)]
    return pd.DataFrame(
        {"frame": d["frame"].to_numpy(), "cam": "gt", "cls": d["cls"].to_numpy(), "wx": d["wx"].to_numpy(),
         "wy": d["wy"].to_numpy(), "score": 1.0, "source": "gt"}
    )


def aggregate_run(cfg: PipelineConfig, log=print) -> dict:
    """Merge per-camera tracks, fuse across cameras and write the occupancy tables."""
    out = Path(cfg.out_dir)
    metas = [json.loads(p.read_text()) for p in sorted(out.glob("meta_*.json"))]
    if not metas:
        raise FileNotFoundError(f"no meta_*.json in {out}; run stage 1 first")
    parts = [pd.read_parquet(p) for p in sorted(out.glob("tracks_*.parquet"))]
    tracks = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    tracks.to_parquet(out / "tracks.parquet", index=False)
    dparts = [pd.read_parquet(p) for p in sorted(out.glob("detections_*.parquet"))]
    pd.concat(dparts, ignore_index=True).to_parquet(out / "detections.parquet", index=False) if dparts else None

    eff_fps = float(metas[0]["sampled_fps"])
    src_fps = float(metas[0]["src_fps"])
    stride = int(metas[0]["stride"])
    f0 = min(m["frame_first"] for m in metas if m["frame_first"] is not None)
    f1 = max(m["frame_last"] for m in metas if m["frame_last"] is not None)
    frames = np.arange((f0 // stride) * stride, f1 + 1, stride)
    frames = frames[frames >= f0]
    duration_s = (f1 - f0) / src_fps + 1.0 / eff_fps
    bin_s = cfg.bin_seconds or auto_bin_seconds(duration_s)

    gt_objs = None
    if cfg.gt_dir:
        from .gt import load_gt_tables

        _, gt_objs = load_gt_tables(cfg.gt_dir)
    zones = _load_zones_for_run(cfg, out, tracks, gt_objs)

    occ_tracks = tracks
    if len(tracks):
        keep = ((tracks["y2"] - tracks["y1"]) >= cfg.min_box_h) & (tracks["score"] >= cfg.min_score)
        occ_tracks = tracks[keep]
    pts = occ_tracks[["frame", "camera", "cls", "wx", "wy", "score"]].rename(columns={"camera": "cam"}) if len(occ_tracks) else pd.DataFrame(columns=["frame", "cam", "cls", "wx", "wy", "score"])
    fused = fuse_world_points(pts, cfg.fuse_radius_m)
    fused["source"] = "pred"
    classes = list(cfg.classes)
    if cfg.gt_classes and gt_objs is not None:
        g = _inject_gt_points(gt_objs, frames, cfg.gt_classes, min_h=0.0)
        fused = pd.concat([fused, g[["frame", "cls", "wx", "wy", "source"]]], ignore_index=True)
        classes += [c for c in cfg.gt_classes if c not in classes]
    steps = occupancy_steps(fused, zones, frames, classes)
    steps["source"] = np.where(steps["cls"].isin(list(cfg.gt_classes)), "gt", "pred")
    bins = occupancy_bins(steps, src_fps, bin_s)
    bins["source"] = np.where(bins["cls"].isin(list(cfg.gt_classes)), "gt", "pred")
    steps.to_parquet(out / "occupancy_steps.parquet", index=False)
    bins.to_parquet(out / "occupancy.parquet", index=False)
    fused.to_parquet(out / "world_points.parquet", index=False)
    summary = {
        "bin_seconds": bin_s,
        "n_zones": len(zones),
        "n_steps": int(len(frames)),
        "duration_s": duration_s,
        "tracks": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        "fused_points": int(len(fused)),
    }
    # merged throughput over per-camera jobs (they may have run in parallel containers)
    tot_frames = sum(m["n_frames"] for m in metas)
    summary["total_frames"] = tot_frames
    summary["sum_infer_s"] = sum(m["infer_s"] for m in metas)
    summary["sum_wall_s"] = sum(m["wall_s"] for m in metas)
    _write_json(out / "meta.json", {"summary": summary, "jobs": metas})
    log(f"aggregate: {summary}")
    return summary


def run(cfg: PipelineConfig, log=print) -> dict:
    meta = process_cameras(cfg, log=log)
    aggregate_run(cfg, log=log)
    return meta


# --------------------------------------------------------------------------- CLI


def _hw(s: str) -> tuple[int, int] | None:
    if s.lower() in ("none", "square", ""):
        return None
    h, w = s.lower().split("x")
    return int(h), int(w)


def _grid(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scene-dir", required=True, help="dir with videos/, calibration.json")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--cameras", nargs="*", default=None, help="Camera_NNNN ids (default: all videos)")
    p.add_argument("--fps", type=float, default=5.0, help="sampling rate per camera (frames/s)")
    p.add_argument("--stride", type=int, default=None, help="explicit frame stride (overrides --fps)")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=None, help="seconds of video to process")
    p.add_argument("--scale", type=float, default=1.0, help="CPU downscale factor before inference")
    p.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    p.add_argument("--model", choices=["nano", "small", "medium", "large"], default="medium")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu (cuda also selects ROCm/HIP)")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--shape", type=_hw, default=(800, 1408), help="HxW inference size or 'square'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--weights", default=None, help="fine-tuned RF-DETR checkpoint (.pth)")
    p.add_argument("--onnx", dest="onnx_path", default=None, help="ONNX file for --backend onnx")
    p.add_argument("--classes", nargs="+", default=["person"], choices=["person", "forklift", "pallet_truck"])
    p.add_argument("--map", nargs="*", default=[], metavar="COCO=LABEL", help="e.g. truck=forklift (proxy, not accurate)")
    p.add_argument("--optimize", action="store_true", help="RF-DETR .inference() (square input only)")
    p.add_argument("--no-gpu-input", action="store_true", help="disable the single-copy pinned GPU staging path")
    p.add_argument("--lost-track-s", type=float, default=3.0)
    p.add_argument("--zones", default=None, help="zones.json (floor polygons, metres)")
    p.add_argument("--zone-grid", type=_grid, default=(3, 3), help="NXxNY grid if no --zones")
    p.add_argument("--bin-seconds", type=float, default=None, help="default: 300 s (>=15 min clips) else 60 s")
    p.add_argument("--fuse-radius-m", type=float, default=0.8)
    p.add_argument("--min-box-h", type=float, default=20.0, help="occupancy ignores tracks shorter than this (px)")
    p.add_argument("--min-score", type=float, default=0.0, help="occupancy ignores tracks below this confidence")
    p.add_argument("--gt-dir", default=None, help="dir with gt_boxes/gt_objects.parquet (zone bounds, GT forklifts)")
    p.add_argument("--gt-classes", nargs="*", default=[], help="classes taken from GT, e.g. forklift pallet_truck")
    p.add_argument("--tag", default=None)


def cfg_from_args(a: argparse.Namespace) -> PipelineConfig:
    cmap = dict(kv.split("=", 1) for kv in a.map) if a.map else None
    return PipelineConfig(
        scene_dir=a.scene_dir, out_dir=a.out, cameras=a.cameras or None, fps=a.fps, stride=a.stride,
        start_s=a.start_s, duration_s=a.duration_s, scale=a.scale, backend=a.backend, model=a.model,
        device=a.device, precision=a.precision, threshold=a.threshold, shape=a.shape,
        batch_size=a.batch_size, weights=a.weights, onnx_path=a.onnx_path, classes=tuple(a.classes),
        class_map=cmap, optimize=a.optimize, gpu_input=not a.no_gpu_input, lost_track_s=a.lost_track_s, zones=a.zones,
        zone_grid=a.zone_grid, bin_seconds=a.bin_seconds, fuse_radius_m=a.fuse_radius_m, min_box_h=a.min_box_h,
        min_score=a.min_score, gt_dir=a.gt_dir,
        gt_classes=tuple(a.gt_classes), tag=a.tag,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m intercycle.perception.pipeline", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "detect", "aggregate"):
        sp = sub.add_parser(name, help={"run": "detect+track+aggregate", "detect": "stage 1 only", "aggregate": "stage 2 only"}[name])
        add_common_args(sp)
    a = ap.parse_args(argv)
    cfg = cfg_from_args(a)
    if a.cmd == "run":
        run(cfg)
    elif a.cmd == "detect":
        process_cameras(cfg)
    else:
        aggregate_run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
