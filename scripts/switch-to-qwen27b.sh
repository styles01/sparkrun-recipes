#!/bin/bash
set -euo pipefail

# Switch to Qwen 3.8 27B NVFP4 on the GB10 drowzeys build
# MTP n=3, fp8 KV, flashinfer autotune, 4 seqs, 256K context
# Recipe: @styles01/qwen-38-27b (GB10 canonical)
# Tuned with arena leader values: GMU 0.72, chunked-prefill 8192, max-num-batched-tokens 8192
# NOTE: DSpark drafter (Doopeworld) forces FLASH_ATTN which rejects fp8 KV on this GB10
#       build — DSpark does NOT work here. MTP n=3 is the validated spec-decode path.
# NOTE: EAGLE 3/1/4 on SGLang tested (~27 avg, 29.4 tool) — slower than drowzeys 96 aggregate.
#       SGLang recipe saved as qwen-38-27b-nvfp4-eagle-sglang.yaml for reference.

IMAGE="ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813"
MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4}"
MODEL="/models/snapshots/b0d9f9de93a9e98df9b1dd41ba444ab1139b1ab3"
CONTAINER="qwen38"
PORT="${PORT:-8000}"
GMU="${GMU:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
NSPEC="${NSPEC:-3}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"

echo "[qwen38] Stopping existing containers..."
docker rm -f "$CONTAINER" 2>/dev/null || true
for c in $(docker ps --format "{{.Names}}" | grep -i sparkrun 2>/dev/null || true); do
  docker rm -f "$c" 2>/dev/null || true
done

echo "[qwen38] Launching GB10 vLLM container (MTP n=$NSPEC + prefix caching + arena tuning)..."
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --network host \
  -v "$MODEL_DIR":/models \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name qwen38-27b \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GMU" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$BATCH_TOKENS" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" \
  > /tmp/qwen38-serve.log 2>&1

echo "[qwen38] Waiting for server (first run compiles FP4 kernels, up to ~12 min)..."
for i in $(seq 1 60); do
  sleep 15
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[qwen38] ✅ Ready on port $PORT"
    echo "[qwen38] Model: qwen38-27b | GMU: $GMU | MTP n=$NSPEC | Seqs: $MAX_NUM_SEQS | Context: $((MAX_MODEL_LEN/1024))K"
    exit 0
  fi
  echo "[qwen38]   check $i/60..."
done

echo "[qwen38] ❌ Server failed to start"
docker logs "$CONTAINER" 2>&1 | tail -20
exit 1
