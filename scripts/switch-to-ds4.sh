#!/bin/bash
# switch-to-ds4.sh — Switch Spark to DeepSeek-V4-Flash on the ds4 CUDA engine
# Run ON the Spark: bash ~/switch-to-ds4.sh
# Or from Mac: ssh jaita@<spark-host> 'bash ~/switch-to-ds4.sh'
#
# Engine: ds4 CUDA (Entrpi/ds4 fork v0.6.2) — native C/CUDA binary, NOT vLLM
# Launch: via ~/.local/bin/ds4-serve wrapper
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail

PORT=8000
# Default: 256K context, 1 concurrent (pass --max-num-seqs 2 for 128K/2 concurrent)
CONTEXT="${DS4_CONTEXT:-262144}"
COALESCE="${DS4_SERVER_COALESCE_MAX:-2}"

echo "[ds4] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[ds4] 1. Killing ALL inference processes..."
# Kill Docker containers (vLLM-based flavors: Qwen 122B, 35B, etc.)
docker rm -f qwen-spark qwen35b-spark qwen38 puzzle-spark 2>/dev/null || true
# Kill any vLLM processes
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
# Kill any existing ds4-server process
pkill -f "ds4-server" 2>/dev/null || true
# Also abort any lingering systemd user scopes from previous runs
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[ds4] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[ds4] 2. Checking free memory..."
free -g | head -2

echo "[ds4] 3. Clearing FlashInfer cache (safe — re-autotunes in ~15 min)..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[ds4] 4. PRESERVING torch compile cache — CUDA graphs are durable assets per model."
echo "[ds4]    vLLM keys graphs by model+spec config. They coexist across flavors."
echo "[ds4]    Only clear if: corrupted graphs, vLLM upgrade, or spec config change."
echo "[ds4]    (ds4 CUDA engine has its own graph management — no torch compile cache)"

echo "[ds4] 5. Verifying ds4 CUDA engine..."
DS4_BIN="${DS4_SRC_DIR:-$HOME/code/ds4}/ds4-server"
if [ ! -x "$DS4_BIN" ]; then
  echo "[ds4] ❌ ds4-server not found at $DS4_BIN"
  echo "[ds4] Install: curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh | bash"
  exit 1
fi
DS4_VERSION=$(cd "${DS4_SRC_DIR:-$HOME/code/ds4}" && git describe --tags 2>/dev/null || echo "unknown")
echo "[ds4] ✅ ds4 CUDA engine: $DS4_VERSION ($DS4_BIN)"

echo "[ds4] 6. Verifying model files..."
GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
MODEL_FILE="$GGUF_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
DSPARK_FILE="$GGUF_DIR/DSpark-drafter-Q2K-Q8-0731.gguf"
if [ ! -f "$MODEL_FILE" ]; then
  echo "[ds4] ❌ Model GGUF not found: $MODEL_FILE"
  exit 1
fi
if [ ! -f "$DSPARK_FILE" ]; then
  echo "[ds4] ⚠️ DSpark drafter not found: $DSPARK_FILE (will run without DSpark)"
fi
echo "[ds4] ✅ Model: $(basename $MODEL_FILE) ($(du -h "$MODEL_FILE" | cut -f1))"
[ -f "$DSPARK_FILE" ] && echo "[ds4] ✅ DSpark: $(basename $DSPARK_FILE) ($(du -h "$DSPARK_FILE" | cut -f1))"

echo "[ds4] === LAUNCHING (ds4 CUDA engine v$DS4_VERSION, ${CONTEXT} context, coalesce=$COALESCE) ==="
# Launch via ds4-serve wrapper — it handles DSpark/MTP mode automatically
# Override context and host/port; all DS4_* env vars can be passed through
export DS4_SERVER_COALESCE_MAX="$COALESCE"
export DS4_BATCH_FIT_HEADROOM_MB="${DS4_BATCH_FIT_HEADROOM_MB:-8192}"
export DS4_CONT_DSPARK="${DS4_CONT_DSPARK:-1}"
export DS4_CONT_MTP_MODE="${DS4_CONT_MTP_MODE:-2}"
export DS4_DSPARK_MODEL="$DSPARK_FILE"
export FORK_PARTIAL="${FORK_PARTIAL:-0}"

~/.local/bin/ds4-serve -c "$CONTEXT" --host 0.0.0.0 --port "$PORT" \
  > "/tmp/ds4-${CONTEXT}.log" 2>&1 &

echo "[ds4] Server starting in background. Waiting for health check..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[ds4] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[ds4] ❌ Server not ready within 10 min. Check: tail -50 /tmp/ds4-${CONTEXT}.log"
  exit 1
}

echo "[ds4] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[ds4] ✅ Done. DS4 CUDA engine serving on port $PORT"
echo "[ds4] Log: /tmp/ds4-${CONTEXT}.log"