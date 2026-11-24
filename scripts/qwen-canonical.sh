#!/usr/bin/env bash
# sparkDash launcher for the qwen-fn lane.
# Brings up BOTH halves of the qwen canonical setup:
#   1. the text lane on spark1 :8000  (Mia TensorFold v1.1.4)
#   2. the Imagination/Wasserman Studio on spark2 :8000-sidecar port 7870
#      (started over ssh because the studio lives on spark2, not this node)
#
# The studio is idempotent: if it already answers, it is left alone.
set -euo pipefail

TEXT_DIR="$HOME/src/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold"
STUDIO_HOST="10.200.0.2"
STUDIO_PORT="${STUDIO_PORT:-7870}"     # the studio's own port. NOT an LLM port:
                                       # PORT LAW covers LLM serving on spark1 only,
                                       # and the studio runs on spark2.

# ---------- 1. studio first (it is quick and independent of the GPU) ----------
echo "[qwen-canonical] studio check on $STUDIO_HOST:$STUDIO_PORT"
if curl -s -o /dev/null -m 6 "http://$STUDIO_HOST:$STUDIO_PORT/" 2>/dev/null; then
  echo "[qwen-canonical] studio already up"
else
  echo "[qwen-canonical] starting studio on spark2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$STUDIO_HOST" \
    "cd ~/imagination-studio && nohup setsid bash -c 'HOST=0.0.0.0 PORT=$STUDIO_PORT bash scripts/app.sh' \
       > ~/studio-app.log 2>&1 < /dev/null & " || echo "[qwen-canonical] studio start FAILED"
  for i in $(seq 1 20); do
    curl -s -o /dev/null -m 4 "http://$STUDIO_HOST:$STUDIO_PORT/" 2>/dev/null && break
    sleep 3
  done
fi
curl -s -o /dev/null -m 6 "http://$STUDIO_HOST:$STUDIO_PORT/" \
  && echo "[qwen-canonical] studio UP" || echo "[qwen-canonical] studio NOT answering"

# ---------- 2. the text lane on spark1:8000 ----------
echo "[qwen-canonical] text lane on :8000"
cd "$TEXT_DIR" || exit 1
export PORT=8000
exec ./start.sh
