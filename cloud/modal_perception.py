"""Modal app for the perception workload (NVIDIA GPUs on Modal; same code runs on AMD MI300X / ROCm).

    modal run cloud/modal_perception.py --action ingest            # CPU: SmartSpaces subset -> Volume
    modal run cloud/modal_perception.py --action run --duration-s 60   # L4 per camera (.map) + eval + pull
    modal run cloud/modal_perception.py --action bench             # L4 benchmark (device, fps, batch)
    modal run cloud/modal_perception.py --action pull --run NAME   # pull results to out/perception/
    modal run cloud/modal_perception.py --action all               # ingest + run + bench

Volume ``intercycle-data``::

    /data/smartspaces/MTMC_Tracking_2026/train/Warehouse_000/{videos,calibration.json,ground_truth.json,gt/}
    /data/perception/<run>/{tracks*.parquet, detections*.parquet, occupancy*.parquet, zones.json, meta*.json, metrics.json}

Dataset nvidia/PhysicalAI-SmartSpaces is CC-BY-4.0 and not gated (no token needed).
Licences of everything installed here: RF-DETR / trackers / huggingface_hub Apache-2.0, supervision MIT,
PyAV BSD-3, PyTorch BSD-3.  No Ultralytics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

try:  # make `intercycle` importable locally even when it is not pip-installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
except Exception:  # pragma: no cover
    pass

APP_NAME = "intercycle-perception"
VOLUME_NAME = "intercycle-data"
DATA = "/data"
SCENES = f"{DATA}/smartspaces"
SCENE_ROOT = "MTMC_Tracking_2026"
DEFAULT_CAMS = ["Camera_0003", "Camera_0010", "Camera_0015", "Camera_0001"]
LOCAL_OUT = Path(__file__).resolve().parents[1] / "out" / "perception"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _prefetch_weights() -> None:
    """Bake the RF-DETR COCO weights into the image so containers do not download ~1.4 GB at start."""
    import rfdetr

    for cls in (rfdetr.RFDETRNano, rfdetr.RFDETRSmall, rfdetr.RFDETRMedium, rfdetr.RFDETRLarge):
        cls(device="cpu")


# Versions pinned to releases that are >= 2 weeks old on 2026-10-17.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # CUDA 12.8 PyTorch wheels (work on Modal L4/A10/A100/H100 drivers)
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "rfdetr==1.11.1",  # Apache-2.0
        "trackers==2.6.1",  # Apache-2.0 (ByteTrack)
        "supervision==0.30.6",  # MIT
        "transformers==5.18.0",
        "av==19.0.0",  # PyAV, BSD-3: CPU decode
        "huggingface_hub==2.1.1",
        "ijson==3.5.1",
        "motmetrics==1.4.0",  # MIT, IDF1/MOTA cross-check
        "pandas",
        "pyarrow",
        "scipy",
        "numpy",
    )
    .run_function(_prefetch_weights)
    # `trackers` / `supervision` import cv2 (opencv-python, not headless) which needs libGL at import time.
    .apt_install("libgl1", "libglib2.0-0")
    .add_local_python_source("intercycle")
)


# --------------------------------------------------------------------------- CPU: ingest


@app.function(image=image, volumes={DATA: vol}, cpu=4, memory=16384, timeout=3600)
def ingest(scene: str = "Warehouse_000", split: str = "train", cams: list[str] | None = None, with_gt: bool = True) -> dict:
    """Download the selected cameras (allow_patterns) + calibration + ground truth into the Volume."""
    from intercycle.perception.ingest import ingest_scene

    res = ingest_scene(SCENES, split, scene, cams or DEFAULT_CAMS, with_gt=with_gt)
    vol.commit()
    return res


# --------------------------------------------------------------------------- GPU: per-camera processing


@app.function(image=image, gpu="L4", volumes={DATA: vol}, cpu=8, memory=32768, timeout=3600)
def process_camera(job: dict) -> dict:
    """Detect + track one camera (or, with job["cameras"], several interleaved in one batched stream) on an L4."""
    from intercycle.perception.pipeline import PipelineConfig, process_cameras

    vol.reload()
    cfg = PipelineConfig(
        scene_dir=job["scene_dir"],
        out_dir=job["out_dir"],
        cameras=job.get("cameras") or [job["camera"]],
        tag=job.get("tag") or job.get("camera"),
        fps=job.get("fps", 5.0),
        start_s=job.get("start_s", 0.0),
        duration_s=job.get("duration_s"),
        model=job.get("model", "medium"),
        precision=job.get("precision", "fp16"),
        shape=tuple(job["shape"]) if job.get("shape") else None,
        batch_size=job.get("batch_size", 8),
        threshold=job.get("threshold", 0.25),
        classes=tuple(job.get("classes", ["person"])),
        gpu_input=job.get("gpu_input", True),
    )
    meta = process_cameras(cfg)
    vol.commit()
    keep = ("tag", "n_frames", "n_detections", "n_track_rows", "wall_s", "infer_s", "fps_e2e", "fps_infer_only",
            "fps_decode_per_thread", "cameras_realtime_e2e", "sampled_fps", "batch_size", "model", "precision")
    out = {k: meta[k] for k in keep}
    out["env"] = meta["env"]
    return json.loads(json.dumps(out, default=str))  # plain JSON types only: the caller has no torch installed


# --------------------------------------------------------------------------- CPU: aggregate + evaluate


@app.function(image=image, volumes={DATA: vol}, cpu=4, memory=16384, timeout=1800)
def aggregate_and_eval(run_dir: str, scene_dir: str, gt_dir: str, zones: str, gt_classes: list[str] | None = None,
                       fuse_radius_m: float = 0.8, min_box_h: float = 20.0, min_score: float = 0.0) -> dict:
    """Fuse cameras, build per-zone occupancy, and score against ground truth."""
    from intercycle.perception.evaluate import evaluate_run
    from intercycle.perception.pipeline import PipelineConfig, aggregate_run

    vol.reload()
    cfg = PipelineConfig(scene_dir=scene_dir, out_dir=run_dir, zones=zones, gt_dir=gt_dir,
                         gt_classes=tuple(gt_classes or ["forklift", "pallet_truck"]),
                         fuse_radius_m=fuse_radius_m, min_box_h=min_box_h, min_score=min_score)
    summary = aggregate_run(cfg)
    metrics = evaluate_run(run_dir, gt_dir, scene_dir)
    vol.commit()
    return json.loads(json.dumps({"summary": summary, "metrics": metrics}, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# --------------------------------------------------------------------------- GPU: benchmark


@app.function(image=image, gpu="L4", volumes={DATA: vol}, cpu=8, memory=32768, timeout=3600)
def benchmark(
    scene: str = "Warehouse_000",
    split: str = "train",
    cams: list[str] | None = None,
    models: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    shape: list[int] | None = None,
    precision: str = "fp16",
    frames: int = 96,
    gpu_input: bool = True,
) -> dict:
    """Print device (torch.version.hip/cuda, GPU name), frames/s per batch size, cameras real-time."""
    from intercycle.perception.bench import format_report, run_benchmark, save_report

    vol.reload()
    vids = [f"{SCENES}/{SCENE_ROOT}/{split}/{scene}/videos/{c}.mp4" for c in (cams or DEFAULT_CAMS)]
    vids = [v for v in vids if Path(v).exists()] or None  # synthetic frames if the Volume is empty
    out: dict = {}
    for m in models or ["nano", "small", "medium", "large"]:
        res = run_benchmark(vids, m, tuple(batch_sizes or [1, 4, 8, 16, 32]), tuple(shape) if shape else (800, 1408),
                            precision, "auto", n_frames=frames, gpu_input=gpu_input)
        print(format_report(res), flush=True)
        out[m] = res
    save_report(out, f"{DATA}/perception/bench_L4{'' if gpu_input else '_nogpuinput'}.json")
    vol.commit()
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# --------------------------------------------------------------------------- local helpers + entrypoint


def _pull_dir(vol_dir: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    prefix = vol_dir.strip("/") + "/"
    for entry in vol.listdir(vol_dir.strip("/"), recursive=True):
        if str(entry.type).lower().endswith("directory"):
            continue
        rel = entry.path.lstrip("/").split(prefix, 1)[-1]
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in vol.read_file(entry.path):
                fh.write(chunk)
        n += 1
    return n


def pull(run: str, dest: Path | None = None, scene: str = "Warehouse_000", split: str = "train") -> Path:
    """Copy ``/perception/<run>`` to ``out/perception/<run>`` plus the compact GT tables and calibration
    (``out/perception/data_<scene>/``) so ``python -m intercycle.perception.evaluate`` also works offline."""
    root = dest or LOCAL_OUT
    out = root / run
    n = _pull_dir(f"perception/{run}", out)
    print(f"pulled {n} files -> {out}")
    sd = f"smartspaces/{SCENE_ROOT}/{split}/{scene}"
    data = root / f"data_{scene}"
    k = _pull_dir(f"{sd}/gt", data / "gt")
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "calibration.json", "wb") as fh:
        for chunk in vol.read_file(f"{sd}/calibration.json"):
            fh.write(chunk)
    print(f"pulled {k} gt files + calibration.json -> {data}")
    return out


@app.local_entrypoint()
def main(
    action: str = "run",
    scene: str = "Warehouse_000",
    split: str = "train",
    cameras: str = ",".join(DEFAULT_CAMS),
    fps: float = 5.0,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    model: str = "medium",
    precision: str = "fp16",
    batch_size: int = 8,
    shape: str = "800x1408",
    run: str = "",
    models: str = "",
    gpu_input: bool = True,
    threshold: float = 0.25,
    fuse_radius_m: float = 0.8,
    min_box_h: float = 20.0,
    min_score: float = 0.0,
):
    cams = [c.strip() for c in cameras.split(",") if c.strip()]
    scene_dir = f"{SCENES}/{SCENE_ROOT}/{split}/{scene}"
    run_name = run or f"{scene}_{model}_{int(fps)}fps_{int(duration_s)}s"
    hw = [int(x) for x in shape.lower().split("x")] if shape.lower() not in ("none", "square") else None

    if action in ("ingest", "all"):
        print(json.dumps(ingest.remote(scene, split, cams), indent=1))

    if action in ("run", "all"):
        run_dir = f"{DATA}/perception/{run_name}"
        jobs = [
            dict(scene_dir=scene_dir, out_dir=run_dir, camera=c, fps=fps, start_s=start_s, duration_s=duration_s,
                 model=model, precision=precision, batch_size=batch_size, shape=hw, threshold=threshold, gpu_input=gpu_input)
            for c in cams
        ]
        results = list(process_camera.map(jobs))  # one L4 container per camera, in parallel
        for r in results:
            print(f"{r['tag']}: {r['n_frames']} frames, e2e {r['fps_e2e']:.1f} fps, infer-only {r['fps_infer_only']:.1f} fps, "
                  f"{r['n_track_rows']} track rows ({r['env'].get('name')}, torch {r['env'].get('torch')})")
        agg = aggregate_and_eval.remote(run_dir, scene_dir, f"{scene_dir}/gt", f"{scene_dir}/gt/zones.json", None,
                                        fuse_radius_m, min_box_h, min_score)
        print(json.dumps(agg["summary"], indent=1))
        for k, v in agg["metrics"].items():
            if k.startswith("min_h_"):
                d, t, o = v["detection"], v["tracking"], v["occupancy"]
                print(f"[{k}] P={d['precision']:.3f} R={d['recall']:.3f} MOTA={t['mota']:.3f} IDF1={t['idf1']:.3f} "
                      f"occ step-MAE={o['step_mae']:.3f} bin-MAE={o['bin_mae']:.3f}")
        dest = pull(run_name)
        (dest / "modal_job_summaries.json").write_text(json.dumps(results, indent=1, default=str))

    if action == "eval":  # re-aggregate + re-score an existing run on the Volume (CPU only, no GPU)
        run_dir = f"{DATA}/perception/{run_name}"
        agg = aggregate_and_eval.remote(run_dir, scene_dir, f"{scene_dir}/gt", f"{scene_dir}/gt/zones.json", None,
                                        fuse_radius_m, min_box_h, min_score)
        print(json.dumps(agg["summary"], indent=1))
        for k, v in agg["metrics"].items():
            if k.startswith("min_h_"):
                d, t, o = v["detection"], v["tracking"], v["occupancy"]
                print(f"[{k}] P={d['precision']:.3f} R={d['recall']:.3f} MOTA={t['mota']:.3f} IDF1={t['idf1']:.3f} "
                      f"occ step-MAE={o['step_mae']:.3f} bin-MAE={o['bin_mae']:.3f} bin-MAE-rel={o['bin_mae_rel']:.3f}")
        pull(run_name)

    if action == "multicam":  # all selected cameras interleaved into batched forwards on ONE L4
        run_dir = f"{DATA}/perception/{run_name}_multicam{len(cams)}"
        job = dict(scene_dir=scene_dir, out_dir=run_dir, cameras=cams, tag=f"multicam{len(cams)}", fps=fps,
                   start_s=start_s, duration_s=duration_s, model=model, precision=precision, batch_size=batch_size,
                   shape=hw, threshold=threshold, gpu_input=gpu_input)
        r = process_camera.remote(job)
        print(json.dumps({k: v for k, v in r.items() if k != "env"}, indent=1))
        print(f"{len(cams)} cameras on one {r['env'].get('name')}: e2e {r['fps_e2e']:.1f} frames/s "
              f"= {r['cameras_realtime_e2e']:.1f} cameras real-time at {r['sampled_fps']:.1f} fps/camera")
        pull(f"{run_name}_multicam{len(cams)}")

    if action in ("bench", "all"):
        res = benchmark.remote(scene, split, cams, [m for m in models.split(",") if m] or None, None, hw, precision, 96, gpu_input)
        LOCAL_OUT.mkdir(parents=True, exist_ok=True)
        fn = LOCAL_OUT / f"bench_L4{'' if gpu_input else '_nogpuinput'}.json"
        fn.write_text(json.dumps(res, indent=1, default=str))
        print(f"benchmark written to {fn}")

    if action == "pull":
        pull(run or run_name)
