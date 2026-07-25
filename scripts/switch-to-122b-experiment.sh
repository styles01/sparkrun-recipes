#!/bin/bash
# switch-to-122b-experiment.sh — Qwen 122B with DFlash n=4 (EXPERIMENTAL)
# Goal: Test if lower DFlash n reduces CUDA graph memory enough for 40GB headroom
# Compare against current n=12 config to see if we can co-locate workloads
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail

PORT=8000

echo "[122b-exp] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[122b-exp] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[122b-exp] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[122b-exp] 2. Checking free memory..."
free -g | head -2

echo "[122b-exp] 3. Clearing FlashInfer cache..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[122b-exp] 4. Clearing torch compile cache (switching model family AND spec config)..."
rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[122b-exp] === LAUNCHING (DFlash n=4 — EXPERIMENTAL) ==="
echo "[122b-exp] TARGET: 3 lanes × 100K+ ctx, 30GB headroom, max speed"
echo "[122b-exp] Budget: 91GB for vLLM (121 - 30 headroom)"
echo "[122b-exp] Fixed: 64GB weights + 3GB overhead = 67GB"
echo "[122b-exp] Remaining for graphs + KV: 24GB"
echo "[122b-exp] If graphs >18GB → try n=2 next"
echo "[122b-exp] Config: 3 lanes × 262K ctx, GMU 0.75, DFlash n=4, fp8 KV, prefix caching"
cd ~/qwen3.5-122B-A10B-on-spark

CTX=262144 GPU_MEM=0.75 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 \
  bash install.sh --start --profile dense --nspec 4

echo "[122b-exp] Waiting for health check (~8-12 min for model load + compile)..."
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -q "^qwen-spark$"; then
    docker logs qwen-spark 2>&1 | tail -30
    echo "[122b-exp] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[122b-exp] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[122b-exp] ❌ Server not ready within 20 min. Check: docker logs qwen-spark"
  exit 1
}

echo "[122b-exp] === CAPTURING MEMORY AUDIT (the whole point of this experiment) ==="
docker logs qwen-spark 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency"

echo "[122b-exp] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[122b-exp] ✅ Done. Served as: qwen"
echo "[122b-exp] ⚠️ Check memory audit above — compare graph cost vs n=12"
echo "[122b-exp] Run speed test: curl -s http://127.0.0.1:8000/v1/chat/completions ..."
echo "[122b-exp] Run prose test to check if n=4 has better acceptance than n=12's 9%"