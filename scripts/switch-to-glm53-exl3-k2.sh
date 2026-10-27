#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 K2 on one DGX Spark (GB10 / SM121).
# Canonical operational launcher. It is intentionally inert without an action.
# Upstream recipe pinned at: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe@841d864694056db202e3c75f6759400af6672293
#
# Actions:
#   --check       Validate host + installed custom runtime; no model download/start.
#   --stage       Clone/pin upstream recipe and install its prebuilt custom runtime.
#   --download    Resume the 91.017-GiB K2 pack directly into ~/models/hf/.
#   --start       Exclusively switch Spark to GLM, then launch the selected lane.
#
# No Hermes/Loca config is altered. No model files are ever staged on the Mac.
set -euo pipefail

UPSTREAM_REPO="https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe.git"
UPSTREAM_REV="841d864694056db202e3c75f6759400af6672293"
RECIPE_DIR="${GLM53_RECIPE_DIR:-$HOME/src/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe}"
VENV="${GLM53_VENV:-$HOME/venvs/glm53-exl3-local}"
MODEL_DIR="${GLM53_MODEL_DIR:-$HOME/models/hf/GLM-5.3-Flash-EXL3-K2}"
PORT="${GLM53_PORT:-8000}"
LANE="${GLM53_LANE:-64k-mtp}"
LOG="${GLM53_LOG:-/tmp/glm53-exl3-${LANE}.log}"
MEMORY_MAX="${GLM53_MEMORY_MAX:-110G}"

DO_CHECK=0
DO_STAGE=0
DO_DOWNLOAD=0
DO_START=0

usage() {
  cat <<'USAGE'
Usage: switch-to-glm53-exl3-k2.sh [--check] [--stage] [--download] [--start] [--lane 64k-mtp|258k-probe]

  --check       Host/runtime preflight only; makes no Spark workload changes.
  --stage       Clone the pinned upstream repo and install prebuilt custom wheels.
  --download    Download/resume the 91.017-GiB model directly to ~/models/hf/.
  --start       Stop exclusive inference/video workloads, verify memory, then serve GLM.
  --lane        64k-mtp (default) or 258k-probe.

Recommended reversible bring-up:
  1. --check
  2. --stage
  3. --download
  4. --start --lane 64k-mtp

The 258k lane is a single-request research probe, not a concurrent agent lane.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1 ;;
    --stage) DO_STAGE=1 ;;
    --download) DO_DOWNLOAD=1 ;;
    --start) DO_START=1 ;;
    --lane) LANE="${2:?--lane requires a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if (( ! DO_CHECK && ! DO_STAGE && ! DO_DOWNLOAD && ! DO_START )); then
  usage >&2
  exit 2
fi

case "$LANE" in
  64k-mtp|258k-probe) ;;
  *) echo "Unsupported lane: $LANE (use 64k-mtp or 258k-probe)" >&2; exit 2 ;;
esac

sync_upstream() {
  mkdir -p "$(dirname "$RECIPE_DIR")"
  if [[ -d "$RECIPE_DIR/.git" ]]; then
    git -C "$RECIPE_DIR" fetch --depth 1 origin "$UPSTREAM_REV"
  else
    git clone "$UPSTREAM_REPO" "$RECIPE_DIR"
  fi
  git -C "$RECIPE_DIR" checkout --detach "$UPSTREAM_REV"
  [[ "$(git -C "$RECIPE_DIR" rev-parse HEAD)" == "$UPSTREAM_REV" ]]
}

custom_preflight() {
  [[ -d "$RECIPE_DIR" ]] || { echo "Missing upstream recipe: run --stage first" >&2; return 1; }
  export PATH="/usr/local/cuda-13.0/bin:$VENV/bin:$PATH"
  "$VENV/bin/python" "$RECIPE_DIR/scripts/preflight.py" --model-dir "$MODEL_DIR"
}

stage_runtime() {
  sync_upstream
  export VENV
  cd "$RECIPE_DIR"
  # The upstream script installs pinned patched vLLM/ExLlamaV3/EXL3 wheels.
  # Never substitute stock `pip install vllm`: it cannot serve Glm5Next + EXL3.
  bash scripts/install_prebuilt.sh
  custom_preflight
}

download_pack() {
  [[ -d "$RECIPE_DIR" ]] || sync_upstream
  export DEST="$MODEL_DIR"
  cd "$RECIPE_DIR"
  bash scripts/download_weights.sh
  "$VENV/bin/python" scripts/preflight.py --model-dir "$MODEL_DIR"
}

stop_exclusive_workloads() {
  echo "[glm53] Stopping exclusive inference/video workloads..."
  systemctl --user stop h3-sol-engine.service 2>/dev/null || true
  docker rm -f qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'ds4-server' 2>/dev/null || true
  for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | awk '/run-r/ {print $1}'); do
    systemctl --user stop "$scope" 2>/dev/null || true
  done
  sleep 5
}

require_memory_headroom() {
  local avail_kib min_kib
  avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  min_kib=$((100 * 1024 * 1024))
  if (( avail_kib < min_kib )); then
    echo "[glm53] Refusing launch: only $((avail_kib / 1024 / 1024)) GiB MemAvailable; require >=100 GiB." >&2
    exit 1
  fi
  echo "[glm53] Memory preflight passed: $((avail_kib / 1024 / 1024)) GiB available."
}

start_server() {
  [[ -x "$VENV/bin/vllm" ]] || { echo "Missing custom runtime: run --stage first" >&2; exit 1; }
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing model pack: run --download first" >&2; exit 1; }
  stop_exclusive_workloads
  require_memory_headroom

  export PATH="/usr/local/cuda-13.0/bin:$VENV/bin:$PATH"
  export VENV MODEL_DIR PORT HOST=0.0.0.0 SERVED_NAME=GLM-5.3-Flash-EXL3
  export EXL3_FUSED_MOE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export ENABLE_PREFIX_CACHING=0

  case "$LANE" in
    64k-mtp)
      export SPEC_METHOD=mtp MTP_TOKENS=2 MAX_MODEL_LEN=65536 GPU_MEM_UTIL=0.91
      unset KV_CACHE_MEMORY_BYTES
      ;;
    258k-probe)
      export SPEC_METHOD=none MAX_MODEL_LEN=262144 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=2048
      export KV_CACHE_MEMORY_BYTES=3221225472
      ;;
  esac

  cd "$RECIPE_DIR"
  "$VENV/bin/python" scripts/patch_chat_template_thinking.py "$MODEL_DIR/chat_template.jinja"
  echo "[glm53] Launching lane=$LANE; log=$LOG"
  # MemoryMax protects SSH and the host if a custom runtime misbehaves. The scope
  # itself is stopped during a future switch so unified memory is actually freed.
  setsid systemd-run --user --scope --collect \
    -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    env PATH="$PATH" VENV="$VENV" MODEL_DIR="$MODEL_DIR" PORT="$PORT" HOST="0.0.0.0" \
      EXL3_FUSED_MOE="$EXL3_FUSED_MOE" PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      ENABLE_PREFIX_CACHING="$ENABLE_PREFIX_CACHING" SPEC_METHOD="$SPEC_METHOD" \
      MTP_TOKENS="${MTP_TOKENS:-}" MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEM_UTIL="${GPU_MEM_UTIL:-}" \
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}" \
      KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}" \
      bash scripts/serve_one_spark.sh >"$LOG" 2>&1 < /dev/null &

  echo "[glm53] Waiting up to 15 minutes for the 91-GiB load + kernel/JIT initialization..."
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "[glm53] READY: $(curl -sf "http://127.0.0.1:$PORT/v1/models")"
      echo "[glm53] Log: $LOG"
      return 0
    fi
    sleep 5
  done
  echo "[glm53] Server did not become ready. Inspect $LOG" >&2
  exit 1
}

if (( DO_STAGE )); then stage_runtime; fi
if (( DO_CHECK )); then custom_preflight; fi
if (( DO_DOWNLOAD )); then download_pack; fi
if (( DO_START )); then start_server; fi
