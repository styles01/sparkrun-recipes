#!/bin/bash
# switch-to-step37.sh — StepFun Step 3.7 Flash (198B MoE, 11B active)
# llama.cpp on DGX Spark, IQ4_XS quant
# Config: 190K context, 3 slots, flash attention, no auto-fit
#
# Prerequisites:
#   - llama.cpp fork at ~/llama.cpp-step37 (branch step3.7, CUDA 13.0)
#   - Model at ~/models/step3.7/IQ4_XS/ (3 shards, ~99GB)
#   - No MTP draft model (tensor mismatch bug — load without it)
#
# Built by Lara, tuned by Oracle, July 15 2026

set -euo pipefail

PORT=8000
MODEL=~/models/step3.7/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
CTX=190000
SLOTS=3
NGL=999

echo "[step37] Killing any existing servers..."
docker rm -f qwen-spark qwen35b-spark 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
sleep 3

echo "[step37] Protecting SSH from OOM..."
echo -1000 | sudo tee /proc/$(pgrep -n sshd)/oom_score_adj 2>/dev/null || true

echo "[step37] Launching Step 3.7 Flash (IQ4_XS, ${CTX} ctx, ${SLOTS} slots, -fa on, -fit off)..."
cd ~/llama.cpp-step37
nohup ./build-cuda/bin/llama-server \
  -m "$MODEL" \
  -c $CTX \
  -ngl $NGL \
  -np $SLOTS \
  -fa on \
  -fit off \
  --host 0.0.0.0 --port $PORT \
  > /tmp/step37_server.log 2>&1 &

PID=$!
echo "[step37] Launched PID $PID"
echo "[step37] Waiting for health check (~3-5 min model load)..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[step37] ✅ Server READY on port $PORT"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "[step37] ❌ Server not ready. Check: tail -30 /tmp/step37_server.log"
  exit 1
}

echo "[step37] Smoke test..."
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"step3.7","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":20}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message'].get('content','') or r['choices'][0]['message'].get('reasoning_content','')[:50])" 2>/dev/null

echo "[step37] === MEMORY ==="
free -h
echo ""
echo "[step37] ✅ Done. Served as: step3.7"
echo "[step37] Server: http://<spark-host>:8080/v1"
echo "[step37] Stop: kill $PID"
echo "[step37] Logs: tail -f /tmp/step37_server.log"