#!/bin/bash
# switch-to-122b.sh — Qwen 3.5 122B DFlash (Production)
# Config: n=4, GMU 0.83, 150K ctx, 3 lanes, ~16GB free for one media workload
# Per DFlash paper (arXiv:2602.06036): smaller n is more efficient under concurrent load
# Or from Mac: ssh user@<spark-host> 'bash ~/switch-to-122b.sh'
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
set -euo pipefail

PORT=8000

echo "[122b] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[122b] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[122b] Aborting orphaned scope: $scope"
  systemctl --user stop "$scope" 2>/dev/null || true
done
sleep 5

echo "[122b] 2. Checking free memory..."
free -g | head -2

echo "[122b] 3. Clearing FlashInfer cache..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[122b] 4. Preserving torch compile cache (same n=4 config — graphs reusable)"
# Only clear if switching from different n value
# rm -rf ~/.cache/huggingface/.vllm_cache/torch_compile_cache 2>/dev/null || true

echo "[122b] === LAUNCHING (Production: n=4, 150K, 3 lanes) ==="
echo "[122b] Config: DFlash n=4, GMU 0.83, 150K ctx, 3 lanes (3.30x concurrency)"
echo "[122b] Memory: 64GB weights + 0.52GB graphs + 28.5GB KV + 11GB drafter/OH = ~104GB"
echo "[122b] Free RAM: ~16GB (for image gen OR TTS, not both)"
echo "[122b] Per DFlash paper: n=4 is more efficient under concurrent load (less wasted verification)"
cd ~/qwen3.5-122B-A10B-on-spark

CTX=165000 GPU_MEM=0.83 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 SERVED_NAME=qwen122b \
  bash install.sh --start --profile dense --nspec 4

echo "[122b] Waiting for health check (~5-8 min)..."
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -q "^qwen-spark$"; then
    docker logs qwen-spark 2>&1 | tail -30
    echo "[122b] ❌ Container exited during load."
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[122b] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[122b] ❌ Server not ready within 20 min. Check: docker logs qwen-spark"
  exit 1
}

echo "[122b] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[122b] ✅ Done. Served as: qwen"