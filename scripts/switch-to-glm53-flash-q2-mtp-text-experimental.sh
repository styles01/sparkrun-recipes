#!/usr/bin/env bash
# Source-pinned, text-only GLM-5.3-Flash UD-Q2_K_XL research lane for one GB10.
# Inert unless a named action is supplied. Never changes Hermes/Loca config.
set -euo pipefail

MODEL_REPO="unsloth/GLM-5.3-Flash-GGUF"
MODEL_REV="2975ab414d30340466d8c51533c6e91f0cca64c1"
MODEL_HOME="${GLM53_Q2_MODEL_HOME:-$HOME/models/hf/GLM-5.3-Flash-GGUF-UD-Q2_K_XL}"
MODEL_SUBDIR="UD-Q2_K_XL"
MODEL_GLOB="GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf"
RUNTIME_REPO="https://github.com/eauchs/llama.cpp.git"
RUNTIME_REV="1d0c76f3c6d030fdfc269aa27db6334ea2834cec"
RUNTIME_DIR="${GLM53_Q2_RUNTIME_DIR:-$HOME/src/llama-glm53-flash-mtp}"
PORT="${GLM53_Q2_PORT:-8000}"
LANE="${GLM53_Q2_LANE:-32k-baseline}"
SERVED_NAME="GLM-5.3-Flash-Q2-MTP"
# systemd's G suffix is decimal: 118G gives the process about 109.9 GiB,
# enough for 101.5-GiB Q2 weights plus q8 KV/scratch while retaining about
# 11.8 GiB of the 121.7-GiB pool outside the scope for OS and SSH.
MEMORY_MAX="${GLM53_Q2_MEMORY_MAX:-118G}"
LOG="${GLM53_Q2_LOG:-/tmp/glm53-flash-q2-${LANE}.log}"
DO_CHECK=0 DO_STAGE=0 DO_DOWNLOAD=0 DO_START=0

usage() {
  cat <<'USAGE'
Usage: switch-to-glm53-flash-q2-mtp-text-experimental.sh [--check] [--stage] [--download] [--start] [--lane 32k-baseline|128k-eval|256k-probe]

Actions are explicit:
  --check     validate pins, GB10, build artifact, and local model layout; no state change
  --stage     clone/build exactly-pinned llama.cpp MTP fork; no workload switch
  --download  download the exact Q2 GGUF only into ~/models/hf/
  --start     switch the exclusive Spark lane and launch the selected one-request text lane

The first permitted lane is 32k-baseline. 128K and 256K are manual research
steps after their preceding correctness, recovery, and memory gates pass.
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
(( DO_CHECK || DO_STAGE || DO_DOWNLOAD || DO_START )) || { usage >&2; exit 2; }
case "$LANE" in 32k-baseline|128k-eval|256k-probe) ;; *) echo "Unsupported lane: $LANE" >&2; exit 2;; esac

require_host() {
  [[ "$(uname -m)" == aarch64 ]] || { echo "Refusing: requires aarch64 DGX Spark." >&2; return 1; }
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA GB10' || { echo "Refusing: requires NVIDIA GB10." >&2; return 1; }
  command -v nvcc >/dev/null || { echo "Refusing: nvcc is required to build this fork." >&2; return 1; }
}

verify_runtime() {
  [[ "$(git -C "$RUNTIME_DIR" rev-parse HEAD 2>/dev/null || true)" == "$RUNTIME_REV" ]] || { echo "Pinned runtime missing/mismatched; run --stage." >&2; return 1; }
  [[ -x "$RUNTIME_DIR/build/bin/llama-server" ]] || { echo "Missing pinned llama-server build; run --stage." >&2; return 1; }
}

verify_model() {
  local model_dir="$MODEL_HOME/$MODEL_SUBDIR" i shard
  for i in 1 2 3 4; do
    printf -v shard 'GLM-5.3-Flash-UD-Q2_K_XL-%05d-of-00004.gguf' "$i"
    [[ -f "$model_dir/$shard" ]] || { echo "Missing Q2 GGUF shard: $model_dir/$shard" >&2; return 1; }
  done
  [[ "$(<"$MODEL_HOME/.oracle-model-revision")" == "$MODEL_REV" ]] || { echo "Model revision marker mismatch; refusing." >&2; return 1; }
}

stage() {
  require_host
  if [[ -d "$RUNTIME_DIR/.git" ]]; then
    git -C "$RUNTIME_DIR" fetch --depth 1 origin "$RUNTIME_REV"
  elif [[ -e "$RUNTIME_DIR" ]]; then
    echo "Refusing: $RUNTIME_DIR exists but is not a git checkout." >&2; return 1
  else
    mkdir -p "$(dirname "$RUNTIME_DIR")"
    git clone "$RUNTIME_REPO" "$RUNTIME_DIR"
    git -C "$RUNTIME_DIR" fetch --depth 1 origin "$RUNTIME_REV"
  fi
  git -C "$RUNTIME_DIR" checkout --detach "$RUNTIME_REV"
  CUDACXX="$(command -v nvcc)" cmake -S "$RUNTIME_DIR" -B "$RUNTIME_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$RUNTIME_DIR/build" --target llama-server -j "${GLM53_Q2_BUILD_JOBS:-8}"
  verify_runtime
  echo "[glm53-q2] staged exact MTP fork; no model was started."
}

download() {
  require_host
  python3 - <<PY
from huggingface_hub import model_info, snapshot_download
repo="$MODEL_REPO"; revision="$MODEL_REV"; local_dir="$MODEL_HOME"
info=model_info(repo, revision=revision)
if info.sha != revision: raise SystemExit(f"resolved {info.sha}, expected {revision}")
snapshot_download(repo_id=repo, revision=revision, local_dir=local_dir, allow_patterns=["UD-Q2_K_XL/*", "*tokenizer*", "*.json", "*.jinja"])
PY
  printf '%s\n' "$MODEL_REV" > "$MODEL_HOME/.oracle-model-revision"
  verify_model
  echo "[glm53-q2] exact Q2 GGUF staged in ~/models/hf only."
}

stop_exclusive_workloads() {
  echo "[glm53-q2] stopping the exclusive inference lane."
  systemctl --user stop h3-sol-engine.service 2>/dev/null || true
  docker rm -f vllm-fn-tp1 qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 2>/dev/null || true
  pkill -f 'llama-server' 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'ds4-server' 2>/dev/null || true
  while IFS= read -r scope; do systemctl --user stop "$scope" 2>/dev/null || true; done < <(systemctl --user list-units --type=scope --no-legend 2>/dev/null | python3 -c 'import sys; [print(x.split()[0]) for x in sys.stdin if x.startswith("run-r")]')
  sleep 5
}

require_headroom() {
  local kib; kib=$(python3 -c 'import re; print(next(int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable:")))')
  (( kib >= 100 * 1024 * 1024 )) || { echo "Refusing: only $((kib / 1024 / 1024)) GiB MemAvailable; require at least 100 GiB." >&2; return 1; }
}

start() {
  require_host; verify_runtime; verify_model
  stop_exclusive_workloads; require_headroom
  local ctx
  case "$LANE" in
    32k-baseline) ctx=32768 ;;
    128k-eval) ctx=131072 ;;
    256k-probe) ctx=262144 ;;
  esac
  echo "[glm53-q2] launching text-only lane=$LANE ctx=$ctx, one request, q8 KV; log=$LOG"
  # The selected MTP fork is text-only. Keep FA disabled because the distinct
  # vision PR documents a correctness requirement for -fa off; no vision claim
  # may be inferred from this MTP lane.
  setsid systemd-run --user --scope --collect -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    env NVIDIA_TF32_OVERRIDE=0 "$RUNTIME_DIR/build/bin/llama-server" \
      -m "$MODEL_HOME/$MODEL_SUBDIR/$MODEL_GLOB" -ngl 99 -c "$ctx" -ctk q8_0 -ctv q8_0 \
      --jinja --spec-type draft-mtp -fa off --alias "$SERVED_NAME" \
      --parallel 1 --host 0.0.0.0 --port "$PORT" >"$LOG" 2>&1 < /dev/null &
  echo "[glm53-q2] waiting up to 15 minutes for a local health endpoint."
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
      curl -fsS "http://127.0.0.1:$PORT/v1/models"; echo
      return 0
    fi
    sleep 5
  done
  echo "[glm53-q2] server did not become ready; inspect $LOG" >&2; return 1
}

if (( DO_STAGE )); then stage; fi
if (( DO_CHECK )); then require_host; verify_runtime; verify_model; fi
if (( DO_DOWNLOAD )); then download; fi
if (( DO_START )); then start; fi
