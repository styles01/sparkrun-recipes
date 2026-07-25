#!/bin/bash
# switch-to-nemotron-super.sh — Nemotron-3-Super-120B NVFP4
# Merges Spark Arena recipe with our 122B production config
# Config: MTP k=3, GMU 0.83, 150K ctx, 3 lanes
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail

PORT=8000
NAME="nemotron-spark"

echo "[nemotron] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[nemotron] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark nemotron-spark 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[nemotron] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[nemotron] 2. Checking free memory..."
free -g | head -2

echo "[nemotron] 3. Clearing FlashInfer cache..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[nemotron] 4. Clearing torch compile cache (new model family)..."
rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[nemotron] === LAUNCHING (NVFP4, MTP k=3, 150K, 3 lanes) ==="
echo "[nemotron] Config: GMU 0.83, 150K ctx, 3 lanes, MTP k=3"
echo "[nemotron] Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (67GB)"
echo "[nemotron] Based on Spark Arena recipe + our 122B production tuning"
docker run --gpus all -d --name "$NAME" \
  --network host --ipc host \
  -e OMP_NUM_THREADS=4 \
  -e CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.24.0 \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name nemotron-3-super \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 1 --trust-remote-code \
  --quantization fp4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization 0.83 \
  --max-model-len 150000 \
  --max-num-seqs 3 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors

echo "[nemotron] Container started. Waiting for health check (~5-15 min first boot)..."
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker logs "$NAME" 2>&1 | tail -30
    echo "[nemotron] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[nemotron] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[nemotron] ❌ Server not ready within 20 min. Check: docker logs $NAME"
  exit 1
}

echo "[nemotron] === MEMORY AUDIT ==="
docker logs "$NAME" 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency"

echo "[nemotron] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[nemotron] ✅ Done. Served as: nemotron-3-super"
echo "[nemotron] Free RAM:"
free -h