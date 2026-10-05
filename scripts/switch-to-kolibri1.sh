#!/usr/bin/env bash
# switch-to-kolibri1.sh — Aleph Alpha Kolibri-1 (vLLM 0.29 + plugin) on one DGX Spark (GB10 / SM121).
# DRAFT LANE — DO NOT RUN --start until the deploy gate in
# runbooks/kolibri1.md clears. --check/--status/--stop are safe any time.
#
# Engine: vLLM 0.29 (official arm64 image) + aleph-alpha-inference 1.0.0 plugin
# + Kolibri-1 FP8 weights (local HF cache after the gated ~78.9 GB download).
#
# Actions:
#   --check       Validate host, image, plugin, model pack; no workload changes.
#   --start       Exclusively switch the Spark to this lane, then serve.
#   --status      Print health endpoint + process state.
#   --stop        Stop the engine cleanly (container).
#
# No Hermes/Loca config is altered. No model files are ever staged on the Mac.
set -euo pipefail

IMAGE="${KOLIBRI_IMAGE:-kolibri1:v0.29.0-p1}"
MODEL_ID="${KOLIBRI_MODEL:-Aleph-Alpha/Kolibri-1}"
REVISION="${KOLIBRI_REVISION:-e52eb4627d11516b0c01de49210ab5a4e4061444}"
PORT="${KOLIBRI_PORT:-8000}"
HOST="${KOLIBRI_HOST:-0.0.0.0}"
CONTAINER="${KOLIBRI_CONTAINER:-kolibri1-service}"
HF_CACHE="${KOLIBRI_HF_CACHE:-$HOME/models/hf}"   # house bind: HF cache lives on the NVMe
LOG="${KOLIBRI_LOG:-/tmp/kolibri1_serve.log}"
MEMORY_MAX="${KOLIBRI_MEMORY_MAX:-110G}"          # OOM must kill the engine, never SSH
# 4x128k shape — safe under BOTH SWA-bounded and unbounded KV readings.
# After the KV-pool measurement (runbooks/kolibri1.md), consider 4x262k.
MAX_LEN="${KOLIBRI_MAX_LEN:-131072}"
MAX_SEQS="${KOLIBRI_MAX_SEQS:-4}"

# Lanes this script may stop when --start runs. Exact identifiers only —
# never broad interpreter patterns (a `python`-matching pkill would take down
# unrelated host services). Container lanes go by docker rm -f; native lanes
# by their exact script/binary name; CLI UIs by listening-port owner.
DOCKER_LANES="qwen-spark qwen35b-spark qwen38 puzzle-spark glm53-exl3 comfyui-spark vllm-fn-tp1"
NATIVE_LANE_PATTERNS='exl3_serve_openai\.py|ds4-server|llama-server'
COMFY_PORTS="8188 8189 8299"

DO_CHECK=0; DO_START=0; DO_STATUS=0; DO_STOP=0
usage() {
  cat <<'USAGE'
Usage: switch-to-kolibri1.sh [--check] [--start] [--status] [--stop]

  --check   Preflight only: docker image, plugin import, model pack present.
  --start   Stop exclusive workloads, verify memory headroom, serve Kolibri lane.
  --status  curl /health + container state.
  --stop    Stop + remove the kolibri1-service container.

DRAFT — read runbooks/kolibri1.md before --start. First boot compiles no
kernels (plugin is pure Python) but does allocate the KV pool; watch
nvidia-smi vs the VRAM math table the first time.
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

kill_port_owners() {
  # Kill only PIDs whose LISTEN socket owns these ports — never name patterns.
  local p pids
  for p in $1; do
    pids=$(ss -tlnp 2>/dev/null | awk -v pt=":${p}$" '$4 ~ pt' | grep -oP 'pid=\K[0-9]+' | sort -u || true)
    [ -n "${pids:-}" ] && kill $pids 2>/dev/null || true
  done
}

health() { curl -s -m 5 "http://127.0.0.1:$PORT/health" || true; }

if (( DO_STATUS )); then
  echo "[kolibri1] health:"; health; echo
  docker ps -a --filter "name=$CONTAINER" --format '{{.Names}} {{.Status}}' || true
  exit 0
fi

if (( DO_STOP )); then
  docker rm -f "$CONTAINER" 2>/dev/null || true
  sleep 3
  echo "[kolibri1] stopped. Memory reclaim: verify with free -g before restarting any lane."
  exit 0
fi

if (( DO_CHECK )); then
  [[ -d "$HF_CACHE/hub/models--Aleph-Alpha--Kolibri-1" ]] || {
    echo "Missing model pack: $HF_CACHE/hub/models--Aleph-Alpha--Kolibri-1" >&2
    echo "(DRAFT lane: weights not downloaded yet — see runbooks/kolibri1.md deploy gate)" >&2
    exit 1
  }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "Missing image: $IMAGE (build per runbooks/kolibri1.md deploy step 2)" >&2
    exit 1
  }
  docker run --rm --platform linux/arm64 "$IMAGE" python -c \
    'import aleph_alpha_inference, vllm; print("plugin", aleph_alpha_inference.__version__, "| vllm", vllm.__version__)'
  echo "[kolibri1] preflight OK (image + plugin + pack)"
  exit 0
fi

if (( DO_START )); then
  echo "[kolibri1] DRAFT gate: confirm runbooks/kolibri1.md deploy steps 1-3 done (image, weights, NVMe fix) and the lane swap is approved."
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Missing image: $IMAGE" >&2; exit 1; }
  [[ -d "$HF_CACHE/hub/models--Aleph-Alpha--Kolibri-1" ]] || { echo "Missing model pack under $HF_CACHE" >&2; exit 1; }

  echo "[kolibri1] Stopping exclusive inference/video workloads (incl. Music 3 ComfyUI)..."
  # 1) Container lanes: exact names from the house teardown list (switch-to-qwen38-exl3.sh),
  #    + the vllm daily-driver container when it is up.
  docker rm -f $DOCKER_LANES 2>/dev/null || true
  # 2) Native engine lanes: exact script/binary names, anchored — no generic python matchers.
  pkill -f "$NATIVE_LANE_PATTERNS" 2>/dev/null || true
  # 3) ComfyUI: systemd unit + listening-port owners (its cmdline is a relative
  #    ./venv/bin/python call that name patterns miss; the laya sidecar lives in
  #    ComfyUI's venv and listens on :8299).
  systemctl stop comfyui.service 2>/dev/null || true
  kill_port_owners "$COMFY_PORTS"
  # 4) Transient user scopes (e.g. prior engine runs under systemd-run --user).
  for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | awk '/run-r/ {print $1}'); do
    systemctl --user stop "$scope" 2>/dev/null || true
  done
  sleep 5

  # RAM reclaim: page cache is GPU-allocatable on GB10 (house pattern).
  sync
  [[ -x "$HOME/sparkrun-recipes/scripts/drop-model-cache.sh" ]] \
    && "$HOME/sparkrun-recipes/scripts/drop-model-cache.sh" || true
  sleep 3

  avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  min_kib=$((100 * 1024 * 1024))
  if (( avail_kib < min_kib )); then
    echo "[kolibri1] Refusing launch: only $((avail_kib / 1024 / 1024)) GiB MemAvailable; require >=100 GiB." >&2
    exit 1
  fi
  echo "[kolibri1] Memory preflight passed: $((avail_kib / 1024 / 1024)) GiB available."

  echo "[kolibri1] Launching container on port $PORT ($MAX_SEQS streams x $MAX_LEN ctx); log=$LOG"
  # Host networking (house pattern); MemoryMax scope so OOM kills the engine, not SSH.
  setsid systemd-run --user --scope --collect \
    -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 \
    docker run --rm --name "$CONTAINER" \
      --runtime nvidia --gpus all --ipc=host --network host \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 \
      -e TRANSFORMERS_OFFLINE=1 \
      "$IMAGE" \
      --served-model-name "$MODEL_ID" \
      --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
      --tensor-parallel-size 1 --kv-cache-dtype fp8 \
      --reasoning-parser kolibri1 --tool-call-parser kolibri1 \
      --enable-auto-tool-choice \
    >"$LOG" 2>&1 < /dev/null &

  echo "[kolibri1] Waiting up to 15 minutes for load + engine init (~3-5 min typical)..."
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "[kolibri1] UP. First-boot note: record nvidia-smi used-memory vs the runbook VRAM table."
      exit 0
    fi
    sleep 5
  done
  echo "[kolibri1] FAILED to come up in 15 min; tail the log:" >&2
  tail -30 "$LOG" >&2
  echo "Rollback: $0 --stop && ~/sparkrun-recipes/scripts/switch-to-qwen38-exl3.sh --start" >&2
  exit 1
fi

usage
