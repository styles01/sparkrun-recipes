#!/usr/bin/env bash
# Qwen3.8-Flash-Next EXL3 native (MTP) on one DGX Spark (GB10 / SM121).
# CURRENT PRODUCTION DAILY DRIVER — canonical operational launcher.
#
# Engine: exllamav3 fork (vcruz305/exllamav3 @ bf9e10f) + OpenAI shim
# (scripts/exl3_serve_openai.py in this repo, v3 /health contract).
#
# Actions:
#   --check       Validate host, venv, fork and model pack; no workload changes.
#   --start       Exclusively switch the Spark to this lane, then serve.
#   --status      Print health endpoint + process state.
#   --stop        Stop the engine cleanly.
#
# No Hermes/Loca config is altered. No model files are ever staged on the Mac.
set -euo pipefail

VENV="${EXL3_VENV:-$HOME/venvs/exl3-150}"
FORK_DIR="${EXL3_FORK:-$HOME/work/exllamav3-fork}"
SHIM="${EXL3_SHIM:-$HOME/sparkrun-recipes/scripts/exl3_serve_openai.py}"
MODEL_DIR="${EXL3_MODEL_DIR:-$HOME/models/hf/turboderp/Qwen3.8-Flash-Next-exl3}"
PORT="${EXL3_PORT:-8000}"
HOST="${EXL3_HOST:-0.0.0.0}"
LOG="${EXL3_LOG:-/tmp/exl3_serve.log}"
MEMORY_MAX="${EXL3_MEMORY_MAX:-110G}"

# Tuned env (measured on GB10; see runbooks/qwen38-flash-next-exl3-daily-driver.md)
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_GR_INT8=1
export EXL3_MTP_HEAD_N=65536 EXL3_NGRAM_STREAM=0
export TORCH_CUDA_ARCH_LIST=12.1
export PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH"

# MTP/draft flags: -mtp (enable MTP draft) -ndt 5 (draft tokens) -dds (dynamic draft
# scaling) -dc 0.6 (draft confidence) -cq 8,8 (quant) -cs 262144 (max ctx)
MTP_FLAGS=(-mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144)
# CPU pin: keep 10 cores on 5-9,15-19; leave the rest for SSH/OS/dashboards.
TASKSET_CPUS="${EXL3_TASKSET:-5-9,15-19}"

DO_CHECK=0; DO_START=0; DO_STATUS=0; DO_STOP=0
usage() {
  cat <<'USAGE'
Usage: switch-to-qwen38-exl3.sh [--check] [--start] [--status] [--stop]

  --check   Preflight only: venv, fork, shim, model pack present.
  --start   Stop exclusive workloads, verify memory headroom, serve EXL3 lane.
  --status  curl /health + process listing.
  --stop    Kill the engine (pkill -f exl3_serve_openai.py).

Restart blip is ~50s (model reload); health counters RESET on restart — dashboard
acceptance bars/averages rebuild from live traffic, this is expected, not data loss.
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1 ;;
    --start) DO_START=1 ;;
    --status) DO_STATUS=1 ;;
    --stop) DO_STOP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

health() { curl -s -m 5 "http://127.0.0.1:$PORT/health" || true; }

if (( DO_STATUS )); then
  echo "[exl3] health:"; health; echo
  pgrep -af exl3_serve_openai.py || echo "[exl3] not running"
  exit 0
fi

if (( DO_STOP )); then
  # Match the shim by its SCRIPT PATH, never by interpreter name.
  pkill -f 'exl3_serve_openai\.py' || true
  sleep 2
  echo "[exl3] stopped."
  exit 0
fi

if (( DO_CHECK )); then
  [[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
  [[ -d "$FORK_DIR/examples" ]] || { echo "Missing fork: $FORK_DIR" >&2; exit 1; }
  [[ -f "$SHIM" ]] || { echo "Missing shim: $SHIM" >&2; exit 1; }
  [[ -d "$MODEL_DIR" ]] || { echo "Missing model pack: $MODEL_DIR" >&2; exit 1; }
  "$VENV/bin/python" -c 'import exllamav3, torch; print("exllamav3", exllamav3.__version__, "| torch", torch.__version__)'
  echo "[exl3] preflight OK"
  exit 0
fi

if (( DO_START )); then
  echo "[exl3] Stopping exclusive inference/video workloads..."
  docker rm -f qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'ds4-server' 2>/dev/null || true
  pkill -f 'exl3_serve_openai\.py' 2>/dev/null || true
  for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | awk '/run-r/ {print $1}'); do
    systemctl --user stop "$scope" 2>/dev/null || true
  done
  sleep 5

  avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  min_kib=$((100 * 1024 * 1024))
  if (( avail_kib < min_kib )); then
    echo "[exl3] Refusing launch: only $((avail_kib / 1024 / 1024)) GiB MemAvailable; require >=100 GiB." >&2
    exit 1
  fi
  echo "[exl3] Memory preflight passed: $((avail_kib / 1024 / 1024)) GiB available."

  echo "[exl3] Launching on port $PORT (taskset $TASKSET_CPUS); log=$LOG"
  # MemoryMax protects SSH + host (OOM must kill the engine, never SSH).
  cd "$FORK_DIR/examples"
  setsid systemd-run --user --scope --collect \
    -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    env PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH" \
      TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
      EXL3_INT8_GEMV="$EXL3_INT8_GEMV" EXL3_MOE_COOP_WIDE="$EXL3_MOE_COOP_WIDE" \
      EXL3_GR_INT8="$EXL3_GR_INT8" EXL3_MTP_HEAD_N="$EXL3_MTP_HEAD_N" \
      EXL3_NGRAM_STREAM="$EXL3_NGRAM_STREAM" \
      taskset -c "$TASKSET_CPUS" "$VENV/bin/python" "$SHIM" \
      -m "$MODEL_DIR" "${MTP_FLAGS[@]}" --host "$HOST" --port "$PORT" \
    >"$LOG" 2>&1 < /dev/null &

  echo "[exl3] Waiting up to 10 minutes for load + engine init (~50s typical)..."
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
      echo "[exl3] UP. Health counters reset to zero on every restart (expected)."
      exit 0
    fi
    sleep 5
  done
  echo "[exl3] FAILED to come up in 10 min; tail the log:" >&2
  tail -30 "$LOG" >&2
  exit 1
fi

usage