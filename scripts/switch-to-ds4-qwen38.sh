#!/bin/bash
# switch-to-ds4.sh — canonical switcher for the DwarfStar (ds4) Qwen3.8-Flash-Next lane.
# Usage: ./switch-to-ds4.sh [--build] [--stop]
# Per house rules: kill running lane, verify GPU free, occupy, keep scripts canonical.

set -euo pipefail
ENGINE_DIR="$HOME/src/cudafast-qwen38-125b-a6b-engine"
WEIGHTS="$ENGINE_DIR/reference_weights/Qwen3.8-Flash-Next-GGUF"
DRAFT_VOCAB="$HOME/src/arena-submission/draft_vocab_output_q8_0_47149.bin"
PORT=8000

if [[ "${1:-}" == "--stop" ]]; then
  pkill -f ds4-server || true
  sleep 2
  python3 -c "import ctypes; print('cuInit:', ctypes.CDLL('libcuda.so').cuInit(0))"
  exit 0
fi

# 1. Stop any running lane (vLLM or ds4) — sequential-only discipline:
pkill -f ds4-server 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# 2. Verify CUDA alive (cuInit, not nvidia-smi):
python3 -c "import ctypes; r = ctypes.CDLL('libcuda.so').cuInit(0); import sys; sys.exit(0 if r == 0 else 1)" \
  || { echo "cuInit FAILED — kernel/driver wedged; reboot required"; exit 1; }

# 3. Build if requested or missing:
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$ENGINE_DIR"
if [[ "${1:-}" == "--build" || ! -x .build/ds4/src/ds4-server ]]; then
  ./tools/ds4/build.sh
fi

# 4. Launch (setsid so it survives SSH drop; 0.0.0.0 so Loca/SparkDash reach it):
DS4_SERVER_MODEL_ID=qwen3.8-flash-next \
DS4_MTP_DRAFT_VOCAB="$DRAFT_VOCAB" \
nohup setsid .build/ds4/src/ds4-server \
  -m "$WEIGHTS/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  --mtp-model "$WEIGHTS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --mtp-draft 2 \
  --host 0.0.0.0 --port "$PORT" \
  > /tmp/ds4-server.log 2>&1 < /dev/null &

# 5. Wait for readiness:
for i in $(seq 1 40); do
  sleep 15
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { echo "ds4-server UP on :$PORT ($(date +%H:%M))"; grep -E "listening|draft vocabulary resident" /tmp/ds4-server.log | strings | head -2; exit 0; }
  grep -qiE "fatal|refus" /tmp/ds4-server.log 2>/dev/null && { echo "BOOT FAILED:"; tail -5 /tmp/ds4-server.log | strings | cut -c1-140; exit 1; }
done
echo "boot timeout after 10 min"; exit 1