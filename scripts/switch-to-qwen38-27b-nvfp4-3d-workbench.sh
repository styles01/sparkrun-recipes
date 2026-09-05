#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cold-boot GB10 Docker CUDA guard; see scripts/lib/spark-docker-cuda-preflight.sh.
GPU_PREFLIGHT="$SCRIPT_DIR/lib/spark-docker-cuda-preflight.sh"
[ -f "$GPU_PREFLIGHT" ] || GPU_PREFLIGHT="$HOME/sparkrun-recipes/scripts/lib/spark-docker-cuda-preflight.sh"
source "$GPU_PREFLIGHT"

# 3D-modelling workbench lane: Qwen3.8-27B NVFP4, vision ON, 2 lanes,
# 450K total context (225K/lane), big-headroom posture for TRELLIS.2.
#
# Companion stack (see runbook qwen3.8-27b-nvfp4-3d-workbench.md):
#   - cad-khana (`khana` CLI) runs on the Mac/agent side: the LLM writes
#     build123d scripts, khana executes + returns diagnostics + renders.
#   - TRELLIS.2 int8 in ComfyUI on the Spark for organic image-to-3D.
#
# Posture: STAGE-ONLY by default. This script does NOT stop the live
# qwen3.8-flash-next lane unless run with `start` after an explicit Go
# (the standing rule: a model switch kills running workloads, but only
# when James says so).
#
# Recipe: @styles01/qwen3.8-27b-nvfp4-3d-workbench

IMAGE="ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813"
MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-57926baca9a82b4d6906b43f2750d55315f5b10f}"
CONTAINER="qwen38-3d-workbench"
PORT="${PORT:-8000}"
GMU="${GMU:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-225280}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
BATCH_TOKENS="${BATCH_TOKENS:-16384}"

usage() {
  cat <<'USAGE'
switch-to-qwen38-27b-nvfp4-3d-workbench.sh — 3D workbench lane

  --check    verify prerequisites (image, model dir, disk, no cgroup conflicts)
  --stage    verify + download model if missing (offline-safe)
  --start    launch the workbench lane (refuses if live service must be killed
             unless KILL_LIVE=1 — a Go has to be explicit)
  --stop     stop the workbench container

Contract (frozen 2026-09-05):
  endpoint        :8000 (canonical)
  model name      qwen3.8-27b
  vision          ON (native vision tower in checkpoint)
  lanes           2 (parent + 1 subagent)
  context         225,280 per lane (450,560 total)
  KV              bf16
  MTP/DSpark      OFF (workbench = code-heavy short tasks + vision; spec-decode
                  adds no benefit on 27B per prior arena results; keeps the
                  drowzeys image's stock, proven launch path)
  headroom        ~40+GiB free for TRELLIS.2 int8 (5.25GB) + ComfyUI
USAGE
}

verify_model() {
  if [ ! -d "$MODEL_DIR" ]; then
    echo "[3d-workbench] model dir missing: $MODEL_DIR" >&2
    echo "[3d-workbench] run with --stage to download (offline-safe)" >&2
    return 1
  fi
  if [ ! -d "$MODEL_DIR/snapshots/$MODEL_SNAPSHOT" ]; then
    echo "[3d-workbench] pinned snapshot $MODEL_SNAPSHOT not found in $MODEL_DIR" >&2
    return 1
  fi
  if [ ! -f "$MODEL_DIR/snapshots/$MODEL_SNAPSHOT/config.json" ]; then
    echo "[3d-workbench] config.json missing under pinned snapshot" >&2
    return 1
  fi
  echo "[3d-workbench] model OK: $MODEL_DIR/snapshots/$MODEL_SNAPSHOT"
}

verify_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "[3d-workbench] image missing: $IMAGE" >&2
    return 1
  }
  echo "[3d-workbench] image OK: $IMAGE"
}

verify_free() {
  local avail
  avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  [ "$avail" -ge 40 ] || { echo "[3d-workbench] need >=40G free disk, have ${avail}G" >&2; return 1; }
  echo "[3d-workbench] disk OK: ${avail}G free"
}

do_check() {
  verify_image
  verify_model
  verify_free
  echo "[3d-workbench] prerequisites OK"
}

do_stage() {
  do_check || {
    echo "[3d-workbench] staging: downloading model…"
    python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/Qwen3.8-27B-NVFP4",
    local_dir=None,  # default hub cache layout (models--…/snapshots/<sha>)
)
PY
    do_check
  }
  echo "[3d-workbench] staged."
}

do_start() {
  # Refuse to kill the live lane unless explicitly authorized.
  if [ "${KILL_LIVE:-0}" != "1" ]; then
    if docker ps --format '{{.Names}}' | grep -qE 'vllm-fn-tp1|qwen38'; then
      echo "[3d-workbench] REFUSING to start: live lane(s) running:" >&2
      docker ps --format '{{.Names}} {{.Status}}' | grep -E 'vllm-fn-tp1|qwen38' >&2
      echo "[3d-workbench] set KILL_LIVE=1 after an explicit Go to replace them." >&2
      exit 1
    fi
  fi
  do_check
  require_spark_docker_cuda "$IMAGE"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --privileged \
    --gpus all \
    --ipc host \
    --network host \
    -v "$MODEL_DIR/snapshots/$MODEL_SNAPSHOT":/models:ro \
    -v "$CACHE_DIR":/root/.cache/vllm \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    "$IMAGE" \
    vllm serve /models \
      --served-model-name qwen3.8-27b \
      --host 0.0.0.0 \
      --port "$PORT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --kv-cache-dtype bfloat16 \
      --gpu-memory-utilization "$GMU" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$BATCH_TOKENS" \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --enable-flashinfer-autotune \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
    # (detached run; watch with: docker logs -f qwen38-3d-workbench)
  echo "[3d-workbench] launched detached on :$PORT — poll http://127.0.0.1:$PORT/health"
}

do_stop() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "[3d-workbench] stopped $CONTAINER" || echo "[3d-workbench] not running"
}

case "${1:-}" in
  --check)  do_check ;;
  --stage)  do_stage ;;
  --start)  do_start ;;
  --stop)   do_stop ;;
  *)        usage; exit 1 ;;
esac