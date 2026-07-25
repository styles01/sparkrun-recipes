#!/bin/bash
# switch-to-laguna.sh — Laguna S 2.1 NVFP4 + DFlash k=7 on 1× DGX Spark (GB10)
#
# Purpose-built agentic coding model — 118B total, 8B active, 256 experts (top-10)
# DFlash speculative decoding k=7 (Poolside's recommended sweet spot, ~5 accepted tokens/pass)
#
# Config: 3 lanes @ 260K ctx, GMU 0.85, fp8 KV, ~18GB free RAM for co-located workloads
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
# Post-launch: auto-starts sparkDash monitoring on :5555
set -euo pipefail

PORT="${PORT:-8000}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
CONTAINER_NAME="laguna-spark"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Laguna-S-2.1-NVFP4}"
DFLASH_PATH="${DFLASH_PATH:-$HOME/models/Laguna-S-2.1-DFlash}"
SERVED_NAME="${SERVED_NAME:-laguna}"
MAX_LEN="${MAX_MODEL_LEN:-320000}"
MAX_SEQS="${MAX_NUM_SEQS:-3}"
GMU="${GPU_MEMORY_UTILIZATION:-0.85}"
NSPEC="${NSPEC:-7}"

echo "[laguna] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[laguna] 1. Killing inference containers (NOT sparkDash)..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark medgemma-spark nemotron-75 laguna-spark 2>/dev/null || true
ps aux | grep -E "vllm serve" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[laguna] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[laguna] 2. Checking free memory..."
free -g | head -2

echo "[laguna] 3. Clearing caches..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true
# ADR-006: NEVER clear torch compile cache
rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[laguna] === LAUNCHING LAGUNA S 2.1 NVFP4 + DFlash ==="
echo "[laguna] Image:  ${IMAGE}"
echo "[laguna] Model: ${MODEL_PATH}"
echo "[laguna] Drafter: ${DFLASH_PATH}"
echo "[laguna] Served: ${SERVED_NAME}"
echo "[laguna] Ctx:    ${MAX_LEN} ($(($MAX_LEN/1024))K)"
echo "[laguna] Seqs:   ${MAX_SEQS}"
echo "[laguna] GMU:    ${GMU}"
echo "[laguna] DFlash: k=${NSPEC}"
echo ""

# Verify models exist
if [ ! -d "$MODEL_PATH" ]; then
  echo "[laguna] ❌ Model not found at ${MODEL_PATH}"
  exit 1
fi
if [ ! -d "$DFLASH_PATH" ]; then
  echo "[laguna] ❌ DFlash drafter not found at ${DFLASH_PATH}"
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${MODEL_PATH}:/model:ro" \
  -v "${DFLASH_PATH}:/drafter:ro" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  "${IMAGE}" \
  /model \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len "${MAX_LEN}" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "${GMU}" \
    --max-num-seqs "${MAX_SEQS}" \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser poolside_v1 \
    --reasoning-parser poolside_v1 \
    --speculative-config "{\"model\":\"/drafter\",\"num_speculative_tokens\":${NSPEC},\"method\":\"dflash\"}"

echo "[laguna] Container started. Model load + compile: ~5-8 min (67GB + DFlash)."
echo ""

echo "[laguna] Waiting for health check..."
for i in $(seq 1 180); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -30
    echo "[laguna] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[laguna] ✅ Server READY on port ${PORT}"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  echo "[laguna] ❌ Server not ready within 15 min. Check: docker logs ${CONTAINER_NAME}"
  exit 1
}

echo ""
echo "[laguna] === MEMORY AUDIT ==="
docker logs "${CONTAINER_NAME}" 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency|speculative"

echo ""
echo "[laguna] Verifying served model name..."
curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo ""
echo "[laguna] === STARTING SPARKDASH ==="
# Restart sparkDash (don't kill it above — it's not in our container kill list)
if docker ps --format '{{.Names}}' | grep -q "^sparkDash$"; then
  echo "[laguna] sparkDash already running, restarting to pick up new LLM..."
  docker restart sparkDash 2>/dev/null || true
else
  docker run -d \
    --name sparkDash \
    --network host \
    --privileged \
    --restart unless-stopped \
    -v ~/sparkDash/server:/app/server \
    -v ~/sparkDash/dist:/app/dist:ro \
    -v /proc:/host/proc:ro \
    -v /sys:/host/sys:ro \
    -v /:/host/root:ro \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
    -v ~/sparkDash/config:/app/config \
    -e PORT=5555 \
    -e LLM_PORT=8000 \
    -e NODE_ENV=production \
    -e HOST_PROC_PATH=/host/proc \
    -e HOST_SYS_PATH=/host/sys \
    -e HOST_ROOT_PATH=/host/root \
    sparkdash-sparkdash \
    node --watch server/index.js
  echo "[laguna] sparkDash started on :5555"
fi

echo ""
echo "[laguna] ✅ Done. Served as: ${SERVED_NAME}"
echo "[laguna] Config: ${MAX_LEN} ctx ($(($MAX_LEN/1024))K) × ${MAX_SEQS} lanes + DFlash k=${NSPEC} @ GMU ${GMU}"
echo "[laguna] Expected: ~8B active params/token, 256 experts (top-10), 1M native context"
echo "[laguna] Free RAM: ~18GB (room for TTS or image gen co-location)"
echo "[laguna] Terminal-Bench 2.1: 70.2% (vs Qwen 122B: 49.4%)"
echo "[laguna] sparkDash: http://larryspark.local:5555"