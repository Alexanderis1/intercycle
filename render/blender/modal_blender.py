"""Modal wrapper for the Blender/Cycles fallback (render/blender/replay_bpy.py).

    .venv\\Scripts\\modal.exe run render/blender/modal_blender.py::smoke
        cube + light, prints devices (OptiX/CUDA), returns render/out/bpy_smoke.png      (~3 min cold)
    .venv\\Scripts\\modal.exe run render/blender/modal_blender.py::replay --trace out/trace.parquet \\
        --shots overview,curtain --t0 0 --t1 60 --samples 64
        -> render/out/<shot>.mp4 (+ credits_blender.json, manifest_blender.json)

GPU: L4 by default (~$0.80/h, Ada RT cores -> OptiX works; 24 GB). Set INTERCYCLE_BLENDER_GPU=L40S for
~2x the speed at ~2.4x the price (about the same cost per frame, shorter wall-clock).
bpy 4.5.14 is the newest 4.x LTS wheel for Python 3.11 (bpy 5.1+ is cp313-only, 5.0.x is cp311 but
needs numpy>=2.2). bpy 4.5 pins numpy<2, so pandas/pyarrow are pinned to numpy-1.x-compatible builds.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPU = os.environ.get("INTERCYCLE_BLENDER_GPU", "L4")
BPY = "bpy==4.5.14"

app = modal.App("intercycle-blender")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "PYTHONUNBUFFERED": "1"})  # 'graphics' injects libnvoptix for OptiX
    .apt_install(
        "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libgl1", "libegl1", "libsm6", "libice6",
        "libxkbcommon0", "libxext6", "libx11-6", "libgomp1", "ffmpeg", "fonts-dejavu-core", "ca-certificates",
    )
    .pip_install(BPY, "pandas==2.2.3", "pyarrow==17.0.0", "pyyaml", "pillow")
    .add_local_file(str(HERE / "replay_bpy.py"), "/root/replay_bpy.py")
    .add_local_file(str(HERE.parent / "isaac" / "replay.py"), "/root/replay.py")  # shared pure-python half
)


def _sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout or "") + (r.stderr or ""), flush=True)


@app.function(image=image, gpu=GPU, timeout=1800)
def smoke_gpu() -> bytes:
    _sh("nvidia-smi")
    _sh("ldconfig -p | grep -i -E 'nvoptix|libcuda|nvidia-rtcore' | head")
    out = Path("/tmp/bpy_out")
    shutil.rmtree(out, ignore_errors=True)
    rc = subprocess.run(["python", "/root/replay_bpy.py", "--smoke", "--out", str(out)]).returncode
    png = out / "bpy_smoke.png"
    if rc != 0 or not png.exists():
        raise RuntimeError(f"bpy smoke failed rc={rc}")
    return png.read_bytes()


@app.local_entrypoint()
def smoke(out: str = ""):
    png = smoke_gpu.remote()
    dst = Path(out) if out else REPO / "render" / "out" / "bpy_smoke.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(png)
    print(f"wrote {dst} ({len(png)} bytes); check the log above for 'Cycles device: OPTIX'")


@app.function(image=image, gpu=GPU, timeout=3600)
def replay_gpu(trace: bytes, scenario_yaml: bytes, replay_args: list[str]) -> bytes:
    work = Path("/tmp/replay")
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    (work / "trace.parquet").write_bytes(trace)
    (work / "bodyshop.yaml").write_bytes(scenario_yaml)
    cmd = ["python", "/root/replay_bpy.py", "--scenario", str(work / "bodyshop.yaml"), "--trace", str(work / "trace.parquet"),
           "--out", str(work / "out"), "--cache", "/tmp/ph_cache", *replay_args]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted((work / "out").glob("*.mp4")) + sorted((work / "out").glob("*.json")):
            tf.add(p, arcname=p.name)
    if rc != 0:
        print(f"replay_bpy exited {rc}; returning whatever exists", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def replay(
    trace: str = "out/trace.parquet",
    scenario: str = "scenario/bodyshop.yaml",
    shots: str = "overview,curtain",
    t0: float = 0.0,
    t1: float = 60.0,
    fps: int = 30,
    samples: int = 64,
    res: str = "1920x1080",
    label: str = "",
    extra: str = "",
):
    args = ["--shots", shots, "--t0", str(t0), "--t1", str(t1), "--fps", str(fps), "--samples", str(samples), "--res", res]
    if label:
        args += ["--label", label]
    if extra:
        args += extra.split()
    blob = replay_gpu.remote(Path(trace).read_bytes(), Path(scenario).read_bytes(), args)
    dst = REPO / "render" / "out"
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(dst)
        print("extracted:", [m.name for m in tf.getmembers()], "->", dst)
