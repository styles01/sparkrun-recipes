#!/bin/bash
# switch-to-puzzle.sh — Nemotron Puzzle 75B-A9B NVFP4 on 1× DGX Spark (GB10)
#
# Uses NGC image nvcr.io/nvidia/vllm:26.06-py3 — community-validated by joeynyc + VramJon
# This image auto-selects FLASHINFER_CUTLASS MoE backend, avoiding the FlashInfer #3738
# regression that caused our OOM crashes on v0.24/v0.25.
#
# Recipe source: https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark
# Expected: ~36 tok/s solo, ~75 tok/s 4-stream aggregate, MTP k=3 ~75% acceptance
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cold-boot GB10 Docker CUDA guard; see scripts/lib/spark-docker-cuda-preflight.sh.
GPU_PREFLIGHT="$SCRIPT_DIR/lib/spark-docker-cuda-preflight.sh"
[ -f "$GPU_PREFLIGHT" ] || GPU_PREFLIGHT="$HOME/sparkrun-recipes/scripts/lib/spark-docker-cuda-preflight.sh"
source "$GPU_PREFLIGHT"

PORT=8000
IMAGE="nvcr.io/nvidia/vllm:26.06-py3"
CONTAINER_NAME="puzzle-spark"
MODEL="nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4"
SERVED_NAME="nemotron-puzzle-75b-nvfp4"
GMU="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_LEN="${MAX_MODEL_LEN:-262144}"
MAX_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED="${MAX_NUM_BATCHED_TOKENS:-8192}"
MTP_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"

echo "[puzzle] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[puzzle] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark nemotron-75 nemotron-puzzle-75b 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[puzzle] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[puzzle] 2. Checking free memory..."
free -g | head -2

echo "[puzzle] 3. Clearing FlashInfer cache (switching model family + image)..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[puzzle] 4. Clearing torch compile cache (ADR-006: switching model family)..."
rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[puzzle] 5. Dropping OS page cache (frees ~44GB)..."
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "[puzzle] (drop_caches skipped — need sudo)"

echo "[puzzle] === LAUNCHING PUZZLE-75B ==="
echo "[puzzle] Image:  ${IMAGE}"
echo "[puzzle] Model:  ${MODEL}"
echo "[puzzle] GMU:    ${GMU}"
echo "[puzzle] Ctx:    ${MAX_LEN}"
echo "[puzzle] Seqs:   ${MAX_SEQS}"
echo "[puzzle] MTP:    k=${MTP_TOKENS}"
echo "[puzzle] Batch:  ${MAX_BATCHED}"
echo ""

# Pull image if not present
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[puzzle] Pulling NGC image (first time only, ~10GB)..."
  docker pull "${IMAGE}"
fi

require_spark_docker_cuda "${IMAGE}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --gpus all \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  "${IMAGE}" \
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --mamba-backend flashinfer \
    --async-scheduling \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --max-num-seqs "${MAX_SEQS}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GMU}"

echo "[puzzle] Container started. First boot: 6-15 min (download + load + compile)."
echo "[puzzle]   logs:   docker logs -f ${CONTAINER_NAME}"
echo "[puzzle]   health: curl -s http://127.0.0.1:${PORT}/v1/models"
echo ""

echo "[puzzle] Waiting for health check..."
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -50
    echo "[puzzle] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[puzzle] ✅ Server READY on port ${PORT}"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  echo "[puzzle] ❌ Server not ready within 20 min. Check: docker logs ${CONTAINER_NAME}"
  exit 1
}

echo "[puzzle] === MEMORY AUDIT ==="
docker logs "${CONTAINER_NAME}" 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency|NvFp4 MoE backend"

echo "[puzzle] Verifying served model name..."
curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[puzzle] ✅ Done. Served as: ${SERVED_NAME}"
echo "[puzzle] Expected: ~36 tok/s solo, ~75 tok/s 4-stream, MTP ~75% acceptance"
echo "[puzzle] If OOM: reduce GMU to 0.85, max-model-len to 131072, or max-num-seqs to 2"