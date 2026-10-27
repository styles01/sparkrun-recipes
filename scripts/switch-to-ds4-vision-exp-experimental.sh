#!/usr/bin/env bash
# Experimental ONLY: antirez/ds4 DeepSeek-V4-Flash-Vision-Exp Q2 on one DGX Spark.
# This does NOT replace scripts/switch-to-ds4.sh or the production 0731 text lane.
# Upstream pin: antirez/ds4@110afdd8886586f18fc9b28bc5533152dd10e728
#
# Explicit actions only:
#   --check      inspect staged experimental artifacts; no service/workload change
#   --stage      clone the pinned upstream code and make cuda-spark
#   --download   direct-download Vision-Exp Q2 + matching encoder under ~/models/hf/
#   --start      exclusive, single-session experimental service on port 8101
#
# Never point Hermes/Loca at this endpoint until all vision/API/long-context gates pass.
set -euo pipefail

UPSTREAM="https://github.com/antirez/ds4.git"
REV="110afdd8886586f18fc9b28bc5533152dd10e728"
SRC="${DS4_VISION_SRC:-$HOME/src/antirez-ds4-vision-exp}"
MODEL_DIR="${DS4_VISION_MODEL_DIR:-$HOME/models/hf/DeepSeek-V4-Flash-Vision-Exp-Q2}"
LANG="$MODEL_DIR/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
ENCODER="$MODEL_DIR/DeepSeek-V4-Flash-Vision-Encoder.gguf"
DSPARK="$MODEL_DIR/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf"
PORT="${DS4_VISION_PORT:-8101}"
CTX="${DS4_VISION_CTX:-4096}"
SESSIONS="${DS4_VISION_SESSIONS:-1}"
LOG="${DS4_VISION_LOG:-/tmp/ds4-vision-exp.log}"
MEMORY_MAX="${DS4_VISION_MEMORY_MAX:-110G}"

CHECK=0; STAGE=0; DOWNLOAD=0; START=0; WITH_DSPARK=0
usage() {
  cat <<'EOF'
Usage: switch-to-ds4-vision-exp-experimental.sh [--check] [--stage] [--download] [--with-dspark] [--start]

  --check        Validate the pinned source/build and any downloaded artifacts.
  --stage        Clone antirez/ds4 at the pinned Vision-Exp commit and build cuda-spark.
  --download     Download Q2 language GGUF + matching vision encoder under ~/models/hf/.
  --with-dspark  Also download the matching Vision-Exp DSpark support GGUF. It is NOT enabled by default.
  --start        Stop the exclusive current lane, then start one experimental vision session on port 8101.

Defaults deliberately start conservatively: 4K context, one session, no DSpark.
Set DS4_VISION_CTX only after sequential memory/API/vision correctness gates pass.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --stage) STAGE=1 ;;
    --download) DOWNLOAD=1 ;;
    --with-dspark) WITH_DSPARK=1 ;;
    --start) START=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if (( ! CHECK && ! STAGE && ! DOWNLOAD && ! START )); then usage >&2; exit 2; fi
if (( CTX < 1024 || SESSIONS < 1 )); then echo 'CTX must be >=1024 and SESSIONS >=1' >&2; exit 2; fi

sync_source() {
  mkdir -p "$(dirname "$SRC")"
  if [[ -d "$SRC/.git" ]]; then git -C "$SRC" fetch --depth 1 origin "$REV"; else git clone "$UPSTREAM" "$SRC"; fi
  git -C "$SRC" checkout --detach "$REV"
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$REV" ]]
}

check() {
  [[ -d "$SRC/.git" ]] || { echo '[ds4-vision] source missing; run --stage'; return 1; }
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$REV" ]] || { echo '[ds4-vision] wrong upstream revision' >&2; return 1; }
  [[ -x "$SRC/ds4-server" ]] || { echo '[ds4-vision] ds4-server missing; run --stage' >&2; return 1; }
  "$SRC/ds4-server" --help >/dev/null
  if [[ -e "$MODEL_DIR" ]]; then
    [[ -f "$LANG" ]] || { echo "[ds4-vision] language GGUF missing: $LANG" >&2; return 1; }
    [[ -f "$ENCODER" ]] || { echo "[ds4-vision] encoder GGUF missing: $ENCODER" >&2; return 1; }
    echo "[ds4-vision] staged model artifacts found under $MODEL_DIR"
  else
    echo "[ds4-vision] model is not downloaded yet (expected under $MODEL_DIR)"
  fi
  echo "[ds4-vision] pinned source/build check passed: $REV"
}

stage() {
  sync_source
  command -v nvcc >/dev/null || { echo 'nvcc is required for make cuda-spark' >&2; exit 1; }
  make -C "$SRC" cuda-spark
  check
}

download() {
  [[ -d "$SRC/.git" ]] || sync_source
  mkdir -p "$MODEL_DIR"
  # The upstream downloader takes an absolute DS4_GGUF_DIR; no weights go to ~/gguf or the Mac.
  DS4_GGUF_DIR="$MODEL_DIR" bash "$SRC/download_model.sh" ds4f-vision-q2
  if (( WITH_DSPARK )); then
    DS4_GGUF_DIR="$MODEL_DIR" bash "$SRC/download_model.sh" ds4f-vision-dspark
  fi
  check
}

stop_exclusive_lanes() {
  echo '[ds4-vision] stopping exclusive inference/video workloads...'
  systemctl --user stop h3-sol-engine.service 2>/dev/null || true
  docker rm -f qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'ds4-server' 2>/dev/null || true
  for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | awk '/run-r/ {print $1}'); do
    systemctl --user stop "$scope" 2>/dev/null || true
  done
  sleep 5
}

start() {
  check
  stop_exclusive_lanes
  local avail_kib
  avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  (( avail_kib >= 100 * 1024 * 1024 )) || { echo '[ds4-vision] insufficient headroom after switch; refusing launch' >&2; exit 1; }

  echo "[ds4-vision] launching experimental Vision-Exp: ctx=$CTX sessions=$SESSIONS port=$PORT"
  # No --dspark in the initial lane. The matching Vision-Exp support file is optional
  # and must separately earn a measured quality/performance improvement.
  setsid systemd-run --user --scope --collect -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    "$SRC/ds4-server" --cuda -m "$LANG" --vision "$ENCODER" \
    --ctx "$CTX" --batched-session "$SESSIONS" --host 0.0.0.0 --port "$PORT" \
    >"$LOG" 2>&1 < /dev/null &

  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "[ds4-vision] READY on :$PORT (experimental; no client repoint performed)"
      echo "[ds4-vision] log: $LOG"
      return 0
    fi
    sleep 5
  done
  echo "[ds4-vision] server did not become ready; inspect $LOG" >&2
  exit 1
}

(( STAGE )) && stage
(( CHECK )) && check
(( DOWNLOAD )) && download
(( START )) && start
