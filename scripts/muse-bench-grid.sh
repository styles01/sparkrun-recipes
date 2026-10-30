#!/usr/bin/env bash
# Muse bench grid against the LIVE :8888 endpoint.
# max_batch_size=1, so concurrency is 1 only — an honest single-stream grid.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
OUT=/home/jaita/muse-bench-$(date -u +%Y%m%d-%H%M).json

echo "=== $(date -u) muse bench start ==="
llama-benchy \
  --base-url http://localhost:8888/v1 \
  --model ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw \
  --served-model-name Muse-Spark-Open \
  --pp 2048 \
  --tg 128 \
  --exact-tg \
  --depth 0 8192 32768 \
  --concurrency 1 \
  --runs 3 \
  --latency-mode api \
  --no-warmup \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --save-result "$OUT" \
  --format md 2>&1

echo "=== $(date -u) bench done -> $OUT ==="
ls -la "$OUT" 2>/dev/null
