#!/bin/bash
# switch-to-ds4.sh — Switch Spark to DeepSeek-V4-Flash
# Run ON the Spark: bash ~/switch-to-ds4.sh
# Or from Mac: ssh jaita@larryspark.local 'bash ~/switch-to-ds4.sh'
#
# Pre-flight checklist (ADR-006): kills all inference, clears caches, safe launch
# NOTE: DS4 serve script must be patched with tool calling flags first.
# See: recipes/deepseek-v4-flash.md for the patch.
# DS4 has cgroup containment (MemoryMax=110G) built into serve script.
set -euo pipefail

PORT=8000

echo "[ds4] === PRE-FLIGHT CHECKLIST (ADR-006) ==="

echo "[ds4] 1. Killing ALL inference processes..."
docker rm -f qwen-spark qwen35b-spark puzzle-spark 2>/dev/null || true
ps aux | grep -E "vllm serve|comfyui|flux" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
# Also abort any lingering systemd user scopes from previous DS4 runs
for scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep "run-r" | awk '{print $1}'); do
  echo "[ds4] Aborting orphaned scope: $scope"
  systemctl --user abort "$scope" 2>/dev/null || true
done
sleep 5

echo "[ds4] 2. Checking free memory..."
free -g | head -2

echo "[ds4] 3. Clearing FlashInfer cache..."
rm -rf ~/.cache/flashinfer/* 2>/dev/null || true

echo "[ds4] 4. PRESERVING torch compile cache — CUDA graphs are durable assets per model."
echo "[ds4]    vLLM keys graphs by model+spec config. They coexist across flavors."
echo "[ds4]    Only clear if: corrupted graphs, vLLM upgrade, or spec config change."
# DO NOT clear torch_compile_cache. Each model's graphs persist on disk.
if ! grep -q "MemoryMax" ~/vllm-moet-spark/spark/serve-ds4-flash-1node.sh; then
  echo "[ds4] ⚠️ WARNING: serve script missing cgroup MemoryMax containment!"
  echo "[ds4] Without it, an OOM can freeze the entire Spark (ADR-006)."
  echo "[ds4] Aborting for safety."
  exit 1
fi
echo "[ds4] ✅ Cgroup containment verified (MemoryMax=110G)"

echo "[ds4] === LAUNCHING (256K context, 1 concurrent — pass override args for 128K/2) ==="
/usr/bin/bash ~/vllm-moet-spark/spark/serve-ds4-flash-1node.sh \
  --max-model-len 262144 --max-num-seqs 1 \
  "$@" \
  > /tmp/ds4-256k.log 2>&1 &

echo "[ds4] Server starting in background. Waiting for health check (~5 min)..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[ds4] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[ds4] ❌ Server not ready within 10 min. Check: tail -50 /tmp/ds4-256k.log"
  exit 1
}

echo "[ds4] Verifying served model name..."
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Model:', d['data'][0]['id'])"

echo "[ds4] ✅ Done. Served as: deepseek-v4-flash"