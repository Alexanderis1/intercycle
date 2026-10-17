# Blender / Cycles fallback renderer

Photoreal replay of `trace.parquet` with `pip install bpy` (Blender as a Python module) on a Modal GPU.
Same trace contract, same `bodyshop.yaml`, same shots and same camera functions as the Isaac path:
`replay_bpy.py` imports the pure-python half of `render/isaac/replay.py` (trace loading, tracks, shots,
layout, plan, ffmpeg), so the two renderers cannot drift apart.

**Status (2026-10-17, tested locally, not yet on Modal):** `bpy==4.5.14` + Python 3.11 on Windows, Cycles on
CUDA (GeForce MX250), a synthetic trace -> 5 shots -> PNG frames -> mp4 through ffmpeg -> A/B split screen.
Frames were inspected: line, 32 KUKA-proportioned arms, hazard-zone decals changing colour with machine
state, light curtains, block body-in-white on skids, people/forklift, hero AMR with mast and status beacon,
BLE packets, racks with thermal false colour. Not yet seen: OptiX (needs the Modal GPU), real sim trace.

## Which bpy

| bpy wheel | Python | numpy | verdict |
|---|---|---|---|
| **4.5.14** (Blender 4.5 LTS, 2026-09-15, manylinux_2_28 x86_64, 373 MB) | **3.11** | `>=1.26,<2` | **use this** |
| 5.0.1 (Blender 5.0, cp311) | 3.11 | `>=2.2` | works in principle, newer API, not LTS |
| 5.1.x / 5.2.x (latest 5.2.2) | **3.13 only** | 2.x | needs a Python 3.13 image |

Source: PyPI JSON metadata for `bpy` (`requires_python` per release). Gotcha found while testing: bpy 4.5
pins numpy 1.x, so `pyarrow>=19` / `pandas 3` import-fail; pin `pandas==2.2.3 pyarrow==17.0.0` (done in
`modal_blender.py`).

## Run

```powershell
# one-time: modal setup   (browser login; writes %USERPROFILE%\.modal.toml)
.venv\Scripts\python.exe render\synth_trace.py --out render\out\trace_synth.parquet      # stand-in trace
.venv\Scripts\modal.exe run render/blender/modal_blender.py::smoke                      # ~3 min, prints "Cycles device: OPTIX"
.venv\Scripts\modal.exe run render/blender/modal_blender.py::replay --trace render/out/trace_synth.parquet `
    --shots overview,curtain,ble,pov,datahall --t0 36000 --t1 36060 --samples 64
# split screen: render the same shot for two policies, then
python render/isaac/replay.py compose --left render/out/ab_cam_A.mp4 --right render/out/ab_cam_B.mp4 --out render/out/ab.mp4
```
`INTERCYCLE_BLENDER_GPU=L40S` switches from L4 (default) to L40S. Local test without Modal (any GPU or CPU):
`uv venv --python 3.11 && uv pip install bpy==4.5.14 pandas==2.2.3 pyarrow==17.0.0 pyyaml pillow imageio-ffmpeg`,
then `python render/blender/replay_bpy.py --smoke` or the same flags as above (`--res 640x360 --samples 8` for a quick look;
put an `ffmpeg` on PATH).

Useful flags: `--samples 64` (final: 128-256), `--res 1920x1080`, `--fps 30`, `--speed 1.0` (sim seconds per video second),
`--light-w 150` (ceiling area lights, tune once on the first frame), `--exposure`, `--hero robot_B`, `--label A|B`,
`--synth-bodies` (invent body motion if the trace has no `body` rows), `--keep-frames`, `--dry-run`.

## Assets: nothing is required

Everything is procedural so the repo and the video carry no third-party licence risk by default:
KUKA KR210 L150-**proportioned** arms (joint origins/axes taken from `kuka_kr210_support` in ros-industrial/kuka_experimental,
Apache-2.0; link geometry is ours), light-curtain posts + emissive plane, gantries, block body-in-white (bevel + subsurf), racks with LED
panel, CRAH boxes, containment glass, forklift/AGV/tugger, people capsules in hi-vis. Optional, fetched at run time, never committed:

| Slot | Asset | Licence | URL | How |
|---|---|---|---|---|
| Lighting / reflections | Poly Haven HDRI `autoshop_01` (also `aircraft_workshop_01`, `aerodynamics_workshop`) | CC0 | https://polyhaven.com/a/autoshop_01 | automatic (`--hdri-id`, via `api.polyhaven.com/files/<id>`) |
| Car body | Kenney Car Kit (45 low-poly cars, glTF/GLB/FBX/OBJ) | CC0 | https://kenney.nl/assets/car-kit (zip: https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip) | unzip, `--body-glb <sedan.glb>`; wheels/glass/lights are stripped and everything painted bare steel; adjust `--body-glb-scale/--body-glb-yaw/--body-glb-yup` |
| Car body (alt.) | "Car Body" by B L A V E N (1.4k tris, unfinished shell = BIW look) | CC-BY | https://sketchfab.com/3d-models/car-body-f11ff10659cd4282b3a1535288b86bfe | download glTF, credit the author |
| Car body (alt.) | Poly Haven "Covered Car" | CC0 | https://polyhaven.com/a/covered_car | not a BIW; hero prop only |
| Robot arm (real meshes) | KUKA KR210 L150 visual DAEs (`base_link`, `link_1`..`link_6`) | Apache-2.0 | https://github.com/ros-industrial/kuka_experimental/tree/melodic-devel/kuka_kr210_support/meshes/kr210l150/visual | not wired yet; import each DAE and parent it to the matching joint empty in `_kuka()` (the URDF visual origin carries a -90 deg X rotation: Y-up DAEs) |
| People | Quaternius animated characters (CC0) | CC0 | https://quaternius.com | not wired; capsules by default |
| Server rack (alt.) | "Server Rack" Anais3Dcraft (1.1k tris) / Spellkaze (744 tris) | CC-BY | https://sketchfab.com/3d-models/server-rack-1348f8c5313f4059b9a2b5b3f37c81f3 / https://sketchfab.com/3d-models/server-rack-62f6779cb7e448b19aaf58544c3c7218 | not needed: procedural racks are used |

Not used on purpose: `ros-industrial/fanuc` meshes (GitHub reports the licence as NOASSERTION; check `package.xml` before
using), Mixamo characters (Adobe terms, not CC), Sketchfab models without an explicit CC licence.
CC-BY items need a credit line in the video description/README; the script writes `credits_blender.json` listing what it used.

## Cost and speed (estimates, measure on the smoke run)

L4 $0.80/h, L40S $1.95/h (Modal, per-second billing). Expect roughly 5-8 s/frame on L4 and 2-4 s/frame on L40S at 1080p,
64 spp, OptiX denoise (about 3 600 frames for a 2 min cut -> about $4-8 either way; L40S finishes sooner). Wall-clock is cut by
launching one container per shot (`replay_gpu.map`), not by a bigger GPU. If OptiX is unavailable the script falls back to CUDA, then
CPU (samples capped at 16 and a loud warning: do not render the final on CPU).

## Known limits / next steps

* Procedural art direction, not a photoscan: good at 20+ m, plain at close range. The 5 quickest upgrades are the real KR210 meshes,
  a CC0 body shell (`--body-glb`), the Poly Haven HDRI (on by default), skid/rail detail, and a floor texture (Poly Haven `concrete_floor_02`, CC0).
* Zones/curtains are translucent emitters hidden from diffuse/glossy rays (they do not light the room). Hero beacon colour follows the hero state.
* People walk with a bob only; no skeletal animation.
* Thermal pass swaps to emissive false colour over a black world with synthetic temperatures (`rack_temp_c` in `render/isaac/replay.py`).
  It must be captioned SYNTHETIC; there is no heat simulation behind it.
