#!/usr/bin/env bash
# Isolated Qwen3.8-27B llama.cpp native-MTP candidate for one DGX Spark.
# Explicit actions only: --check, --stage, --download, --start. No default action.
# No Loca/Hermes configuration is read, written, or repointed by this script.
set -euo pipefail

UPSTREAM="https://github.com/ggml-org/llama.cpp.git"
SRC="${LLAMA_CPP_SRC:-$HOME/src/llama.cpp-qwen38-mtp-experimental}"
MODEL_ROOT="$HOME/models/hf"
MODEL_DIR="${QWEN38_MODEL_DIR:-$MODEL_ROOT/unsloth-Qwen3.8-27B-GGUF}"
MODEL_REPO="${QWEN38_MODEL_REPO:-unsloth/Qwen3.8-27B-GGUF}"
MODEL_REVISION="${QWEN38_MODEL_REVISION:-4ca720788d1e01f1bff70c033e0d0028fd02e502}"
TARGET_FILE="${QWEN38_TARGET_FILE:-Qwen3.8-27B-UD-Q4_K_M.gguf}"
MTP_FILE="${QWEN38_MTP_FILE:-MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
TARGET_GGUF="$MODEL_DIR/$TARGET_FILE"
MTP_GGUF="$MODEL_DIR/$MTP_FILE"
PORT="${QWEN38_LLAMA_PORT:-8102}"
CTX="${QWEN38_CTX:-131072}"
PARALLEL="${QWEN38_PARALLEL:-1}"
MTP_DRAFT_N="${MTP_DRAFT_N:-0}"
MEMORY_MAX="${QWEN38_MEMORY_MAX:-110G}"
LOG="${QWEN38_LLAMA_LOG:-/tmp/qwen38-llamacpp-mtp-experimental.log}"

CHECK=0; STAGE=0; DOWNLOAD=0; START=0
usage() {
  cat <<'EOF'
Usage: switch-to-qwen38-llamacpp-mtp-experimental.sh [--check] [--stage] [--download] [--start]

  --check      Inspect source/build/model staging only; does not stop or start workloads.
  --stage      Require LLAMA_CPP_REVISION, clone exact source, and build llama-server with CUDA.
  --download   Direct-download the pinned GGUF artifacts only under ~/models/hf/.
  --start      Stop known exclusive inference workloads, measure release, and start one cgrouped lane.

No arguments: inert; prints this usage and exits nonzero.
Initial lane: MTP off (MTP_DRAFT_N=0), context 131072, parallel 1.
After baseline gates only: MTP_DRAFT_N=2. Allowed comparison values are 0,2,3,4.
EOF
}
while (($#)); do
  case "$1" in
    --check) CHECK=1 ;;
    --stage) STAGE=1 ;;
    --download) DOWNLOAD=1 ;;
    --start) START=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if (( ! CHECK && ! STAGE && ! DOWNLOAD && ! START )); then usage >&2; exit 2; fi
[[ "$MODEL_DIR" == "$MODEL_ROOT"/* ]] || { echo "QWEN38_MODEL_DIR must remain under $MODEL_ROOT" >&2; exit 2; }
[[ "$CTX" =~ ^[0-9]+$ && "$CTX" -ge 131072 ]] || { echo 'QWEN38_CTX must be an integer >= 131072' >&2; exit 2; }
[[ "$PARALLEL" == 1 ]] || { echo 'initial candidate requires QWEN38_PARALLEL=1; test concurrency outside this launcher' >&2; exit 2; }
case "$MTP_DRAFT_N" in 0|2|3|4) ;; *) echo 'MTP_DRAFT_N must be 0, 2, 3, or 4' >&2; exit 2;; esac

require_revision() {
  : "${LLAMA_CPP_REVISION:?Set LLAMA_CPP_REVISION to a manually verified post-fix 40-hex llama.cpp commit. This candidate intentionally has no unverified default.}"
  [[ "$LLAMA_CPP_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo 'LLAMA_CPP_REVISION must be a 40-character lowercase hexadecimal commit SHA' >&2; exit 2; }
}
mem_available_kib() { awk '/MemAvailable:/ {print $2}' /proc/meminfo; }
print_memory() { printf '[qwen38-llamacpp-mtp] %s MemAvailable=%s KiB\n' "$1" "$(mem_available_kib)"; }

sync_source() {
  require_revision
  mkdir -p "$(dirname "$SRC")"
  if [[ -d "$SRC/.git" ]]; then
    git -C "$SRC" fetch --no-tags origin "$LLAMA_CPP_REVISION"
  elif [[ -e "$SRC" ]]; then
    echo "source path exists but is not a git checkout: $SRC" >&2; exit 1
  else
    git clone --filter=blob:none --no-checkout "$UPSTREAM" "$SRC"
    git -C "$SRC" fetch --no-tags origin "$LLAMA_CPP_REVISION"
  fi
  git -C "$SRC" checkout --detach "$LLAMA_CPP_REVISION"
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$LLAMA_CPP_REVISION" ]] || { echo 'source revision verification failed' >&2; exit 1; }
}

check() {
  require_revision
  [[ -d "$SRC/.git" ]] || { echo "[qwen38-llamacpp-mtp] source missing: run --stage" >&2; return 1; }
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$LLAMA_CPP_REVISION" ]] || { echo '[qwen38-llamacpp-mtp] staged source is not the requested revision' >&2; return 1; }
  [[ -x "$SRC/build/bin/llama-server" ]] || { echo '[qwen38-llamacpp-mtp] llama-server missing: run --stage' >&2; return 1; }
  "$SRC/build/bin/llama-server" --help >/dev/null
  if [[ -e "$MODEL_DIR" ]]; then
    [[ -f "$TARGET_GGUF" ]] || { echo "[qwen38-llamacpp-mtp] target GGUF missing: $TARGET_GGUF" >&2; return 1; }
    [[ -f "$MTP_GGUF" ]] || echo "[qwen38-llamacpp-mtp] MTP artifact not downloaded; baseline MTP-off remains usable"
  else
    echo "[qwen38-llamacpp-mtp] model not downloaded: expected under $MODEL_DIR"
  fi
  echo "[qwen38-llamacpp-mtp] staged source/build verified: $LLAMA_CPP_REVISION"
}

stage() {
  command -v cmake >/dev/null || { echo 'cmake is required' >&2; exit 1; }
  command -v nvcc >/dev/null || { echo 'nvcc is required for GGML_CUDA build' >&2; exit 1; }
  sync_source
  cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SRC/build" --target llama-server --config Release -j"$(nproc)"
  check
}

download() {
  command -v hf >/dev/null || { echo 'Hugging Face CLI command `hf` is required for direct download' >&2; exit 1; }
  mkdir -p "$MODEL_DIR"
  # local-dir is intentionally constrained to ~/models/hf; no cache/symlink location is used for weights.
  hf download "$MODEL_REPO" "$TARGET_FILE" "$MTP_FILE" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
  [[ -f "$TARGET_GGUF" && -f "$MTP_GGUF" ]] || { echo 'download did not yield both expected GGUF artifacts' >&2; exit 1; }
  echo "[qwen38-llamacpp-mtp] direct artifacts downloaded under $MODEL_DIR"
}

stop_exclusive_lanes() {
  print_memory before-exclusive-stop
  echo '[qwen38-llamacpp-mtp] stopping known exclusive inference workloads only...'
  docker rm -f qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 2>/dev/null || true
  pkill -f '(^|/)llama-server( |$)' 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'ds4-server' 2>/dev/null || true
  sleep 5
  print_memory after-exclusive-stop
}

start() {
  check
  [[ -f "$TARGET_GGUF" ]] || { echo 'target model is missing; run --download' >&2; exit 1; }
  if (( MTP_DRAFT_N > 0 )); then
    [[ -f "$MTP_GGUF" ]] || { echo 'MTP artifact is missing; run --download' >&2; exit 1; }
  fi
  stop_exclusive_lanes
  local available
  available="$(mem_available_kib)"
  (( available >= 100 * 1024 * 1024 )) || { echo '[qwen38-llamacpp-mtp] less than 100 GiB MemAvailable after release; refusing launch' >&2; exit 1; }
  local -a spec=()
  if (( MTP_DRAFT_N > 0 )); then spec=(--spec-type draft-mtp --spec-draft-n "$MTP_DRAFT_N"); fi
  echo "[qwen38-llamacpp-mtp] start: ctx=$CTX parallel=1 mtp_n=$MTP_DRAFT_N port=$PORT MemoryMax=$MEMORY_MAX"
  setsid systemd-run --user --scope --collect -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    "$SRC/build/bin/llama-server" -m "$TARGET_GGUF" -ngl 99 -c "$CTX" -np 1 \
    --flash-attn on --jinja --host 0.0.0.0 --port "$PORT" --alias qwen3.8-27b-llamacpp-mtp-experimental "${spec[@]}" \
    >"$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "[qwen38-llamacpp-mtp] endpoint ready on :$PORT; run a real completion health check before use"
      echo "[qwen38-llamacpp-mtp] log: $LOG"
      return 0
    fi
    sleep 5
  done
  echo "[qwen38-llamacpp-mtp] server did not become ready; inspect $LOG" >&2
  exit 1
}

(( STAGE )) && stage
(( DOWNLOAD )) && download
(( CHECK )) && check
(( START )) && start
