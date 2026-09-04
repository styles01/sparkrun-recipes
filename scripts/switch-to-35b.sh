#!/bin/bash
# switch-to-35b.sh — Switch Spark to Qwen 3.6 35B NVFP4
# Run ON the Spark: bash ~/switch-to-35b.sh
# Or from Mac: ssh user@<spark-host> 'bash ~/switch-to-35b.sh'
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cold-boot GB10 Docker CUDA guard; see scripts/lib/spark-docker-cuda-preflight.sh.
source "$SCRIPT_DIR/lib/spark-docker-cuda-preflight.sh"

PORT=8000
NAME="qwen35b-spark"

echo "[35b] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[35b] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
# Also abort any lingering systemd user scopes from previous DS4 runs
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[35b] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[35b] 2. Checking free memory..."
free -g | head -2

echo "[35b] 3. Clearing FlashInfer cache..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[35b] 4. PRESERVING torch compile cache — CUDA graphs are durable assets per model."
echo "[35b]    vLLM keys graphs by model+spec config. They coexist across flavors."
echo "[35b]    Only clear if: corrupted graphs, vLLM upgrade, or spec config change."
# DO NOT clear torch_compile_cache. Each model's graphs persist on disk.

echo "[35b] === LAUNCHING (Media-server config — MTP k=3 + PR #48375 Mamba bugfix) ==="
echo "[35b] Config: 6 lanes × 256K ctx, GMU 0.40, MTP k=3, fp8 KV, prefix caching"
echo "[35b] Memory budget: ~22GB weights + ~21GB CUDA graphs + KV from GMU 0.40"
echo "[35b] Free for co-located workloads: ~59GB (video gen, TTS, image gen, GNOME)"
require_spark_docker_cuda "vllm/vllm-openai:v0.24.0"
docker run --privileged --gpus all -d --name "$NAME" \
  --network host --ipc host \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/patch_mamba_drop_eagle.sh:/patch_mamba_drop_eagle.sh:ro \
  --entrypoint bash \
  vllm/vllm-openai:v0.24.0 \
  -c "bash /patch_mamba_drop_eagle.sh && vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name qwen35b \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 1 --trust-remote-code \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --gpu-memory-utilization 0.40 \
    --max-model-len 262144 \
    --max-num-seqs 6 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"triton\"}' \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice"

echo "[35b] Container started. Waiting for health check (FlashInfer autotune ~5-15 min first boot)..."
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker logs "$NAME" 2>&1 | tail -30
    echo "[35b] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[35b] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[35b] ❌ Server not ready within 20 min. Check: docker logs $NAME"
  exit 1
}

echo "[35b] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[35b] ✅ Done. Served as: qwen35b"