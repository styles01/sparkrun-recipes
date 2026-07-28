#!/bin/bash
set -euo pipefail

# Switch to Qwen 3.6 27B FP8 on vLLM v26
# MTP k=7, fp8 KV, async scheduling, 5 lanes, 256K context, GMU 0.79
# Recipe: @styles01/qwen3.6-27b-fp8-mtp

IMAGE="vllm-v26-patched:latest"
MODEL="Qwen/Qwen3.6-27B-FP8"
CONTAINER="qwen27b"
PORT="${PORT:-8000}"
GMU="${GMU:-0.79}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-5}"
NSPEC="${NSPEC:-7}"

echo "[qwen27b] Stopping existing containers..."
docker rm -f qwen27b 2>/dev/null || true
# Also stop any sparkrun containers
for c in $(docker ps --format "{{.Names}}" | grep -i sparkrun 2>/dev/null || true); do
  docker rm -f "$c" 2>/dev/null || true
done

echo "[qwen27b] ADR-006 pre-flight: checking torch compile cache..."
# Don't clear torch compile cache (ADR-006)

echo "[qwen27b] Launching vLLM v26 container..."
docker run -d \
  --name "$CONTAINER" \
  --ipc host \
  --gpus all \
  -p "$PORT:8000" \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface" \
  --entrypoint sleep \
  "$IMAGE" \
  infinity

echo "[qwen27b] Starting vLLM serve..."
docker exec "$CONTAINER" bash -c "vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization $GMU \
  --kv-cache-dtype fp8 \
  --load-format safetensors \
  --attention-backend flashinfer \
  --async-scheduling \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --served-model-name $MODEL \
  -tp 1 -pp 1 \
  > /tmp/serve.log 2>&1 & echo \$! > /tmp/serve.pid"

echo "[qwen27b] Waiting for server (this takes ~7 minutes)..."
for i in $(seq 1 60); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[qwen27b] ✅ Ready on port $PORT"
    echo "[qwen27b] Model: $MODEL | GMU: $GMU | MTP k=$NSPEC | Lanes: $MAX_NUM_SEQS | Context: $((MAX_MODEL_LEN/1024))K"
    exit 0
  fi
  echo "[qwen27b]   check $i/60..."
done

echo "[qwen27b] ❌ Server failed to start"
docker logs "$CONTAINER" 2>&1 | tail -20
exit 1