#!/usr/bin/env bash
# Run the InterCycle perception workload on an AMD Instinct MI300X (ROCm) host.
# Identical entrypoints to the Modal/NVIDIA path:
#   python -m intercycle.perception.pipeline run ...     and     scripts/perception_bench.py
#
#   cloud/run_rocm.sh build            # docker build (context = cloud/)
#   cloud/run_rocm.sh check            # prints torch.version.hip + GPU name
#   cloud/run_rocm.sh ingest           # SmartSpaces subset (no HF token needed; dataset is not gated)
#   cloud/run_rocm.sh bench            # frames/s per model & batch size  -> out/perception/bench_mi300x.json
#   cloud/run_rocm.sh run              # detect+track+occupancy+evaluate  -> out/perception/<RUN>/
#   cloud/run_rocm.sh all              # build check ingest bench run
#   cloud/run_rocm.sh shell            # interactive shell in the container
#
# Env overrides: IMAGE, CAMERAS (space separated), DURATION_S (default 300), MODEL (medium), SHAPE (800x1408),
#                FPS (5), BATCH (8), SCENE (Warehouse_000), RUN, HF_TOKEN (optional, only raises rate limits),
#                WITH_MIGRAPHX=1 at build time to include the ONNX Runtime MIGraphX EP.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-intercycle-perception:rocm}"
SCENE="${SCENE:-Warehouse_000}"
SPLIT="${SPLIT:-train}"
CAMERAS="${CAMERAS:-Camera_0003 Camera_0010 Camera_0015 Camera_0001}"
DURATION_S="${DURATION_S:-300}"
MODEL="${MODEL:-medium}"
SHAPE="${SHAPE:-800x1408}"
FPS="${FPS:-5}"
BATCH="${BATCH:-8}"
RUN="${RUN:-${SCENE}_${MODEL}_${SHAPE}_${FPS}fps_mi300x}"
DATA_DIR="${DATA_DIR:-$REPO/data}"          # dataset cache (git-ignored)
OUT_DIR="${OUT_DIR:-$REPO/out/perception}"  # results (git-ignored)
SCENE_DIR="/data/smartspaces/MTMC_Tracking_2026/${SPLIT}/${SCENE}"

mkdir -p "$DATA_DIR" "$OUT_DIR"

drun() {  # docker run with the ROCm device flags from the AMD ROCm PyTorch install docs
  docker run --rm ${TTY:--i} \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G \
    -v "$REPO":/workspace/intercycle \
    -v "$DATA_DIR":/data \
    -v "$OUT_DIR":/out \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/data/hf \
    -w /workspace/intercycle \
    "$IMAGE" "$@"
}

cmd_build() {
  docker build -f "$REPO/cloud/Dockerfile.rocm" -t "$IMAGE" \
    --build-arg WITH_MIGRAPHX="${WITH_MIGRAPHX:-0}" ${BASE_IMAGE:+--build-arg BASE_IMAGE="$BASE_IMAGE"} "$REPO/cloud"
}

cmd_check() {
  drun python3 -c "
import torch
print('torch        :', torch.__version__)
print('torch.version.hip :', torch.version.hip)
print('cuda available (HIP) :', torch.cuda.is_available())
print('GPU          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('memory (GB)  :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else None)
"
}

cmd_ingest() {
  # shellcheck disable=SC2086
  drun python3 scripts/perception_ingest.py --local-dir /data/smartspaces --split "$SPLIT" --scene "$SCENE" --cameras $CAMERAS
}

cmd_bench() {
  local vids=()
  for c in $CAMERAS; do vids+=("$SCENE_DIR/videos/$c.mp4"); done
  local avail=()
  for v in "${vids[@]}"; do [ -f "${v/\/data/$DATA_DIR}" ] && avail+=("$v"); done
  # one run per model size; falls back to synthetic frames if the dataset is not downloaded
  for m in nano small medium large; do
    drun python3 scripts/perception_bench.py ${avail[@]:+--videos "${avail[@]}"} --model "$m" \
      --shape "$SHAPE" --batch-sizes 1 4 8 16 32 64 --json "/out/bench_mi300x_${m}.json"
  done
}

cmd_run() {
  # shellcheck disable=SC2086
  drun python3 -m intercycle.perception.pipeline run \
      --scene-dir "$SCENE_DIR" --cameras $CAMERAS \
      --fps "$FPS" --duration-s "$DURATION_S" --model "$MODEL" --shape "$SHAPE" --batch-size "$BATCH" \
      --zones "$SCENE_DIR/gt/zones.json" --gt-dir "$SCENE_DIR/gt" --gt-classes forklift pallet_truck \
      --out "/out/$RUN"
  drun python3 -m intercycle.perception.evaluate --run-dir "/out/$RUN" --gt-dir "$SCENE_DIR/gt" --scene-dir "$SCENE_DIR"
}

case "${1:-}" in
  build)  cmd_build ;;
  check)  cmd_check ;;
  ingest) cmd_ingest ;;
  bench)  cmd_bench ;;
  run)    cmd_run ;;
  shell)  TTY=-it drun bash ;;
  all)    cmd_build; cmd_check; cmd_ingest; cmd_bench; cmd_run ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
