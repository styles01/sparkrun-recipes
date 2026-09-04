#!/bin/bash
# switch-to-medgemma.sh — MedGemma 27B FP8 on 1× DGX Spark (GB10) + SparkDash
#
# Recipe source: AI-Scribe corpus formatter (docs/CORPUS.md, ADRs 0004/0005/0008)
# Tested on <spark-host> — 4,961/4,963 MTSamples notes completed
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cold-boot GB10 Docker CUDA guard; see scripts/lib/spark-docker-cuda-preflight.sh.
source "$SCRIPT_DIR/lib/spark-docker-cuda-preflight.sh"

PORT="${PORT:-8000}"
IMAGE="${IMAGE:-vllm-node:latest}"
CONTAINER_NAME="medgemma-spark"
MODEL_PATH="${MODEL_PATH:-$HOME/models/medgemma-27b-fp8}"
SERVED_NAME="${SERVED_NAME:-medgemma}"
MAX_LEN="${MAX_MODEL_LEN:-5000}"
MAX_SEQS="${MAX_NUM_SEQS:-48}"
GMU="${GPU_MEMORY_UTILIZATION:-0.65}"

echo "[medgemma] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[medgemma] 1. Killing inference containers (NOT sparkDash)..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark medgemma-spark nemotron-75 nemotron-puzzle-75b 2>/dev/null || true
ps aux | grep -E "vllm serve" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[medgemma] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[medgemma] 2. Checking free memory..."
free -g | head -2

echo "[medgemma] 3. Clearing caches..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true
rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[medgemma] === LAUNCHING MEDGEMMA 27B FP8 ==="
echo "[medgemma] Image:  ${IMAGE}"
echo "[medgemma] Model:  ${MODEL_PATH}"
echo "[medgemma] Served: ${SERVED_NAME}"
echo "[medgemma] Ctx:    ${MAX_LEN}"
echo "[medgemma] Seqs:   ${MAX_SEQS}"
echo "[medgemma] GMU:    ${GMU}"
echo ""

# Verify model exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "[medgemma] ❌ Model not found at ${MODEL_PATH}"
  exit 1
fi

require_spark_docker_cuda "${IMAGE}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --gpus all \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${MODEL_PATH}:/model:ro" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len "${MAX_LEN}" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "${GMU}" \
    --max-num-seqs "${MAX_SEQS}" \
    --attention-backend flashinfer \
    --enable-prefix-caching

echo "[medgemma] Container started. Model load + compile: ~3-5 min."
echo ""

echo "[medgemma] Waiting for health check..."
for i in $(seq 1 120); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -30
    echo "[medgemma] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[medgemma] ✅ Server READY on port ${PORT}"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  echo "[medgemma] ❌ Server not ready within 10 min. Check: docker logs ${CONTAINER_NAME}"
  exit 1
}

echo "[medgemma] === MEMORY AUDIT ==="
docker logs "${CONTAINER_NAME}" 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency"

echo "[medgemma] Verifying served model name..."
curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo ""
echo "[medgemma] === LAUNCHING SPARKDASH ==="
docker rm -f sparkDash 2>/dev/null || true
sleep 2

# Rebuild frontend before launching (ensures dist/ is current)
if [ -d "$HOME/sparkDash" ]; then
  echo "[medgemma] Rebuilding SparkDash frontend..."
  cd "$HOME/sparkDash"
  docker build --target builder -t sparkdash-builder:tmp . 2>&1 | tail -3
  # Extract fresh dist from builder image
  docker create --name sparkdash-extract sparkdash-builder:tmp 2>/dev/null
  rm -rf ./dist/*
  docker cp sparkdash-extract:/app/dist/. ./dist/ 2>/dev/null
  docker rm sparkdash-extract 2>/dev/null
  docker rmi sparkdash-builder:tmp 2>/dev/null || true
  echo "[medgemma] Frontend rebuilt."
  cd "$HOME"
fi

# Ensure LlmProbe uses 'localhost' not 'host.docker.internal' (Linux --network host fix)
if [ -f "$HOME/sparkDash/server/collectors/LlmProbe.js" ]; then
  if grep -q "host.docker.internal" "$HOME/sparkDash/server/collectors/LlmProbe.js" 2>/dev/null; then
    echo "[medgemma] Fixing LlmProbe.js: host.docker.internal → localhost (Linux --network host)"
    sed -i 's/host\.docker\.internal/localhost/g' "$HOME/sparkDash/server/collectors/LlmProbe.js"
  fi
fi

# Ensure sparks.json has llmPort (not just llmPorts array)
if [ -f "$HOME/sparkDash/config/sparks.json" ]; then
  if ! python3 -c "import json; d=json.load(open('$HOME/sparkDash/config/sparks.json')); exit(0 if 'llmPort' in d['sparks'][0] else 1)" 2>/dev/null; then
    echo "[medgemma] Fixing sparks.json: adding llmPort field..."
    cp "$HOME/sparkDash/config/sparks.json" /tmp/sparks_fix.json
    python3 -c "
import json
with open('/tmp/sparks_fix.json') as f:
    d = json.load(f)
s = d['sparks'][0]
if 'llmPort' not in s:
    ports = s.get('llmPorts', [8000])
    s['llmPort'] = ports[0] if ports else 8000
with open('/tmp/sparks_fix.json', 'w') as f:
    json.dump(d, f, indent=2)
"
    cp /tmp/sparks_fix.json "$HOME/sparkDash/config/sparks.json" 2>/dev/null || true
  fi
fi

if [ -d "$HOME/sparkDash" ]; then
  docker run -d \
    --name sparkDash \
    --network host \
    --privileged \
    --restart unless-stopped \
    -v "$HOME/sparkDash/server:/app/server" \
    -v "$HOME/sparkDash/dist:/app/dist:ro" \
    -v /proc:/host/proc:ro \
    -v /sys:/host/sys:ro \
    -v /:/host/root:ro \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
    -v "$HOME/sparkDash/config:/app/config" \
    -e PORT=5555 \
    -e LLM_PORT=8000 \
    -e NODE_ENV=production \
    -e HOST_PROC_PATH=/host/proc \
    -e HOST_SYS_PATH=/host/sys \
    -e HOST_ROOT_PATH=/host/root \
    sparkdash-sparkdash \
    node --watch server/index.js

  echo "[medgemma] ✅ SparkDash launched on :5555"
  echo "[medgemma]   dashboard: http://<spark-host>:5555"
else
  echo "[medgemma] ⚠️ ~/sparkDash not found — skipping dashboard launch"
fi

echo ""
echo "[medgemma] ✅ Done. Served as: ${SERVED_NAME}"
echo "[medgemma] Expected: ~336+ tok/s aggregate (256 seqs), ~7 tok/s per request"
echo "[medgemma] Memory-bandwidth-bound at 27GB FP8 weights — batch size is the throughput lever"
echo "[medgemma] SparkDash: http://<spark-host>:5555"