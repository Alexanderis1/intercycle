"""InterCycle photoreal tier: Isaac Sim 5.1 (pip) on a Modal L40S.

Modelled on github.com/TheFloatingString/isaac-lab-on-modal (same base image, same pip
extras, same "run the heavy thing in a subprocess, check a sentinel, return PNG bytes"
pattern), minus Isaac Lab.

Run from the repo root (C:\\AMD\\intercycle):

    .venv\\Scripts\\modal.exe run render/isaac/modal_app.py::smoke
        -> boots Kit, opens a stock warehouse from the public asset bucket, adds a camera,
           renders ~30 warm-up frames, saves /cache/out/smoke.png in the Modal Volume and
           writes render/out/smoke.png locally. Also prints nvidia-smi + vulkaninfo.

    .venv\\Scripts\\modal.exe run render/isaac/modal_app.py::replay --trace out/trace.parquet \\
        --shots overview,curtain --t0 0 --t1 60
        -> uploads trace + scenario, runs render/isaac/replay.py in the container, downloads
           a tar.gz of the mp4s into render/out/ (design skeleton, see replay.py).

Names: the GPU function is `smoke_gpu`; the local entrypoint is called `smoke` so that
`modal run ...::smoke` both renders and saves the PNG locally (a bare remote function would
return its bytes to nowhere).

Needs `modal setup` (token in %USERPROFILE%\\.modal.toml) before anything runs.
Cost: L40S is about $1.95/h on Modal (check https://modal.com/pricing). First run = image
build (CPU, 10-20 min, cached afterwards) + cold shader compile (5-10 min, cached in the
Volume afterwards).

EULA: setting ACCEPT_EULA=Y / OMNI_KIT_ACCEPT_EULA=YES in the image environment is the
documented non-interactive way to accept the NVIDIA Software License Agreement / Isaac Sim
Additional Software and Materials License for the pip package. The person running this
app is the licensee; see docs/RENDER.md for what that licence does and does not cover.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import textwrap
import time
from pathlib import Path

import modal

APP_NAME = "intercycle-isaac"
CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
PYTHON = "3.11"  # isaacsim 5.x wheels are cp311 only (6.x needs 3.12: bump both together)
ISAACSIM_PIP = "isaacsim[all,extscache]==5.1.0"
CACHE_VOLUME = "intercycle-isaac-cache"
CACHE_DIR = "/cache"
GPU = "L40S"
TIMEOUT_S = 3600

# Public, unauthenticated asset bucket (S3 list + GET both work anonymously; verified 2026-10-17).
ASSET_ROOT_51 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
DEFAULT_ENV = "Environments/Simple_Warehouse/warehouse_with_forklifts.usd"  # relative to <root>/Isaac/

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) >= 2 else HERE
REPLAY_PY = HERE / "replay.py"
LOCAL_OUT = REPO / "render" / "out"

app = modal.App(APP_NAME)
cache_vol = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

_VK_ICD = '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}'
_EGL_ICD = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'

image = (
    modal.Image.from_registry(CUDA_BASE, add_python=PYTHON)
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "ACCEPT_EULA": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ALLOW_ROOT": "1",  # Kit refuses to start as root otherwise
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",  # graphics+compute so Vulkan/OptiX libs are injected
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        }
    )
    .apt_install(
        # requested set
        "libvulkan1", "libegl1", "libglu1-mesa", "libxrandr2", "libxinerama1", "libxcursor1",
        "libxi6", "libsm6", "libice6", "libxt6",
        # additions: vulkaninfo for diagnostics, ffmpeg for the mp4 step, misc Kit runtime libs
        "vulkan-tools", "ffmpeg", "libglib2.0-0", "libxkbcommon0", "libgomp1", "fonts-dejavu-core",
        "ca-certificates", "curl",
    )
    .run_commands(
        # The CUDA base image has no Vulkan/EGL vendor files; point the loader at the injected driver.
        "mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        f"echo '{_VK_ICD}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
        f"echo '{_VK_ICD}' > /etc/vulkan/icd.d/nvidia_icd.json",
        f"echo '{_EGL_ICD}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .pip_install(ISAACSIM_PIP, extra_index_url="https://pypi.nvidia.com")
    # separate layer so tweaking these never re-runs the 10+ GB Isaac install.
    # PINS MATTER: isaacsim 5.1 pins numpy==1.26; pyarrow>=19/pandas 3 need numpy 2 and die at import.
    .pip_install("pillow", "pyyaml", "pandas==2.2.3", "pyarrow==17.0.0")
)
if REPLAY_PY.exists():
    image = image.add_local_file(str(REPLAY_PY), "/root/replay.py")  # must stay last (not copy=True)


# --------------------------------------------------------------------------------------
# Smoke script. Runs in a subprocess inside the container (own process => a Kit crash or
# a hang on close cannot take the Modal function down, and stdout is easy to stream).
# --------------------------------------------------------------------------------------
SMOKE_SCRIPT = textwrap.dedent(
    r'''
    import os, sys, time
    T0 = time.time()
    def log(msg):
        print("T+%6.1fs %s" % (time.time() - T0, msg), flush=True)

    W = int(os.environ.get("SMOKE_W", "1280"))
    H = int(os.environ.get("SMOKE_H", "720"))
    WARMUP = int(os.environ.get("SMOKE_WARMUP", "30"))
    ENV_REL = os.environ["SMOKE_ENV"]
    FALLBACK_ROOT = os.environ["SMOKE_ASSET_ROOT"]
    OUT = os.environ["SMOKE_OUT"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": W, "height": H})
    log("SimulationApp up")

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom, UsdLux

    root = None
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
    except Exception as e:  # noqa
        log("get_assets_root_path failed: %r" % (e,))
    if not root:
        root = FALLBACK_ROOT
    url = root.rstrip("/") + "/Isaac/" + ENV_REL
    log("assets root = %s" % root)
    log("opening %s" % url)

    ctx = omni.usd.get_context()
    ok = ctx.open_stage(url)
    log("open_stage -> %r" % (ok,))

    def loading():
        _, done, total = ctx.get_stage_loading_status()
        return bool(done or total)

    t = time.time()
    while loading() and time.time() - t < 900:
        app.update()
    log("stage loaded (%.1fs of waiting)" % (time.time() - t))
    stage = ctx.get_stage()
    if stage is None:
        log("SMOKE_FAIL no stage")
        app.close()
        sys.exit(2)

    # ---- camera framed from the stage bounds -------------------------------------------
    up_axis = UsdGeom.GetStageUpAxis(stage)
    up = Gf.Vec3d(0, 0, 1) if up_axis == UsdGeom.Tokens.z else Gf.Vec3d(0, 1, 0)
    ui = 2 if up_axis == UsdGeom.Tokens.z else 1                # index of the up component
    hi = [i for i in (0, 1, 2) if i != ui]                       # the two horizontal components
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
    lo, hi_ = box.GetMin(), box.GetMax()
    size = hi_ - lo
    center = (lo + hi_) * 0.5
    if not (0.5 < max(size) < 2000):
        lo, hi_ = Gf.Vec3d(-10, -10, 0), Gf.Vec3d(10, 10, 6)
        size, center = hi_ - lo, (lo + hi_) * 0.5
    eye = Gf.Vec3d(center)
    eye[hi[0]] += 0.30 * size[hi[0]]
    eye[hi[1]] -= 0.30 * size[hi[1]]
    eye[ui] = lo[ui] + min(0.45 * size[ui], 4.0)
    tgt = Gf.Vec3d(center)
    tgt[ui] = lo[ui] + 1.0
    log("bbox size=%s eye=%s target=%s up=%s" % (tuple(size), tuple(eye), tuple(tgt), up_axis))

    cam_path = "/SmokeCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
    view = Gf.Matrix4d().SetLookAt(eye, tgt, up)                 # world -> camera
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

    # a light in case the stock scene has none
    if not any(p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase)
               for p in stage.Traverse()):
        dome = UsdLux.DomeLight.Define(stage, "/SmokeDome")
        dome.CreateIntensityAttr(1000.0)
        log("no lights in stage -> added dome light")

    rp = rep.create.render_product(cam_path, (W, H))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    log("render product attached; warming up %d frames (first frames compile shaders)" % WARMUP)
    for i in range(WARMUP):
        app.update()
        if i % 10 == 0:
            log("  warm-up frame %d" % i)

    data = None
    for attempt in range(10):
        rep.orchestrator.step(rt_subframes=4)
        d = rgb.get_data()
        if d is not None and getattr(d, "size", 0) > 0 and np.asarray(d).std() > 0.5:
            data = np.asarray(d)
            break
        log("  empty/blank frame on attempt %d, retrying" % attempt)
        for _ in range(5):
            app.update()
    if data is None:
        log("SMOKE_FAIL no usable frame")
        app.close()
        sys.exit(3)

    from PIL import Image
    Image.fromarray(data[..., :3].astype("uint8")).save(OUT)
    log("SMOKE_STATS shape=%s mean=%.1f std=%.1f" % (data.shape, float(data.mean()), float(data.std())))
    log("SMOKE_OK %s" % OUT)
    app.close()
    '''
)


def _sh(cmd: str, timeout: int = 120) -> None:
    """Run a diagnostic shell command and echo its output (never raises)."""
    print(f"\n$ {cmd}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        print((r.stdout or "") + (r.stderr or ""), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(failed: {e!r})", flush=True)


def _link_caches() -> None:
    """Point Kit/RTX/driver caches at the Volume so shaders and ext data survive across runs."""
    home = Path.home()
    mapping = {
        home / ".cache" / "ov": Path(CACHE_DIR) / "ov",
        home / ".cache" / "nvidia": Path(CACHE_DIR) / "nvidia-cache",
        home / ".nv": Path(CACHE_DIR) / "nv",
        home / ".local" / "share" / "ov": Path(CACHE_DIR) / "ov-data",
    }
    for src, dst in mapping.items():
        dst.mkdir(parents=True, exist_ok=True)
        src.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            continue
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)
        src.symlink_to(dst, target_is_directory=True)


def _diagnostics() -> None:
    _sh("nvidia-smi")
    _sh("vulkaninfo --summary")
    _sh("ldconfig -p | grep -i -E 'libGLX_nvidia|libnvidia-rtcore|libnvoptix|libvulkan|libEGL_nvidia' | head -20")
    _sh("ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 | head -20")
    _sh("python -c \"import sys; print(sys.version)\"; pip list 2>/dev/null | grep -i -E 'isaacsim|omniverse|torch|numpy' | head -20")


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={CACHE_DIR: cache_vol})
def smoke_gpu(env_rel: str = DEFAULT_ENV, warmup: int = 30, width: int = 1280, height: int = 720) -> bytes:
    """Boot Isaac Sim headless, render one frame of a stock environment, return PNG bytes."""
    t_start = time.time()
    _link_caches()
    _diagnostics()

    script = Path("/tmp/smoke_scene.py")
    script.write_text(SMOKE_SCRIPT)
    out_png = Path(CACHE_DIR) / "out" / "smoke.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.unlink(missing_ok=True)

    env = dict(os.environ)
    env.update(
        SMOKE_ENV=env_rel,
        SMOKE_W=str(width),
        SMOKE_H=str(height),
        SMOKE_WARMUP=str(warmup),
        SMOKE_ASSET_ROOT=ASSET_ROOT_51,
        SMOKE_OUT=str(out_png),
    )
    print(f"\n=== launching Isaac Sim ({ISAACSIM_PIP}) ===", flush=True)
    proc = subprocess.Popen(
        ["python", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    ok = False
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        if "SMOKE_OK" in line:
            ok = True
    rc = proc.wait()
    cache_vol.commit()  # persist shader/kit caches + the PNG even if the check below fails
    print(f"=== isaac subprocess exit={rc}, ok={ok}, wall={time.time() - t_start:.0f}s ===", flush=True)
    if not (ok and out_png.exists()):
        raise RuntimeError(f"smoke failed (rc={rc}); see log above")
    return out_png.read_bytes()


@app.local_entrypoint()
def smoke(env: str = DEFAULT_ENV, warmup: int = 30, width: int = 1280, height: int = 720, out: str = ""):
    """modal run render/isaac/modal_app.py::smoke   (writes render/out/smoke.png)"""
    png = smoke_gpu.remote(env, warmup, width, height)
    dst = Path(out) if out else LOCAL_OUT / "smoke.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(png)
    print(f"wrote {dst} ({len(png)} bytes)")


# --------------------------------------------------------------------------------------
# Replay (design skeleton; see replay.py). Same image, same Volume.
# --------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={CACHE_DIR: cache_vol})
def replay_gpu(trace: bytes, scenario_yaml: bytes, replay_args: list[str]) -> bytes:
    """Run /root/replay.py on an uploaded trace; return a tar.gz of the resulting mp4 files."""
    _link_caches()
    work = Path("/tmp/replay")
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    (work / "trace.parquet").write_bytes(trace)
    (work / "bodyshop.yaml").write_bytes(scenario_yaml)
    cmd = [
        "python", "/root/replay.py",
        "--scenario", str(work / "bodyshop.yaml"),
        "--trace", str(work / "trace.parquet"),
        "--out", str(work / "out"),
        *replay_args,
    ]
    print("$", " ".join(cmd), flush=True)
    env = dict(os.environ)
    rc = subprocess.run(cmd, env=env).returncode
    cache_vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted((work / "out").glob("*.mp4")) + sorted((work / "out").glob("*.json")):
            tf.add(p, arcname=p.name)
    if rc != 0:
        print(f"replay.py exited {rc}; returning whatever mp4s exist", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def replay(
    trace: str = "out/trace.parquet",
    scenario: str = "scenario/bodyshop.yaml",
    shots: str = "overview,curtain",
    t0: float = 0.0,
    t1: float = 60.0,
    fps: int = 30,
    quality: str = "fast",
    extra: str = "",
):
    """modal run render/isaac/modal_app.py::replay --trace out/trace.parquet --shots overview --t0 0 --t1 30"""
    args = ["--shots", shots, "--t0", str(t0), "--t1", str(t1), "--fps", str(fps), "--quality", quality]
    if extra:
        args += extra.split()
    blob = replay_gpu.remote(Path(trace).read_bytes(), Path(scenario).read_bytes(), args)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(LOCAL_OUT)
        print("extracted:", [m.name for m in tf.getmembers()], "->", LOCAL_OUT)
