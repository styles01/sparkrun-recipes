#!/usr/bin/env bash
# Kill the trimmed 9-cell run and start the FULL arena grid.
# Arena standard: pp 2048, tg exact, depth 0/8192/32768/65536/102400,
# concurrency 1/2/5/10, 3 runs  => 28 cells.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH

echo "=== $(date -u) switching to FULL arena grid ==="

# 1. stop the partial run
pkill -f "llama-benchy --base-url http://localhost:8888" 2>/dev/null && echo "partial run killed"
sleep 5
pgrep -fa llama-benchy | grep -v pgrep | head -2 || echo "no benchy running"

OUT=/home/jaita/muse-bench-FULL-$(date -u +%Y%m%d-%H%M).json
echo "output -> $OUT"

# 2. full grid, detached
nohup setsid llama-benchy \
  --base-url http://localhost:8888/v1 \
  --model ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw \
  --served-model-name Muse-Spark-Open \
  --pp 2048 \
  --tg 400 \
  --exact-tg \
  --depth 0 8192 32768 65536 102400 \
  --concurrency 1 2 5 10 \
  --runs 3 \
  --latency-mode api \
  --no-warmup \
  --save-result "$OUT" \
  --format md \
  > /home/jaita/muse-bench-FULL.log 2>&1 < /dev/null &

sleep 20
echo "fired; running:"
pgrep -fa llama-benchy | grep -v pgrep | head -2
