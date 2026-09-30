#!/usr/bin/env bash
# keepalive-8000.sh — mitigates TensorFold GPU-idle hang (ashhart/TensorFold#122).
# After ~60s idle the engine's FIRST generation request hangs with zero bytes
# for 38-60s; a second request recovers. Serve a cheap 1-token generation every
# 27s to keep the generation path warm so real traffic never eats the hang.
#
# Only for the tf-exl3 lane (:8000). Exits automatically if the server is gone
# (restart lane -> restart pinger).
#
# Log: /tmp/keepalive8000.log (running log on Spark /tmp — sanctioned).

ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
INTERVAL=27

while true; do
  # Lane gone? Exit quietly; respawner/restore flow restarts us.
  if ! curl -s --max-time 3 http://127.0.0.1:8000/health | grep -q '"status"'; then
    echo "$(date +%F' '%T) server down, pinger exiting" >> /tmp/keepalive8000.log
    exit 0
  fi
  t0=$(date +%s.%N)
  http=$(curl -s --max-time 8 -o /dev/null -w "%{http_code}" "$ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0}')
  t1=$(date +%s.%N)
  ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", (b-a)*1000}')
  echo "$(date +%F' '%T) http=$http ${ms}ms" >> /tmp/keepalive8000.log
  sleep "$INTERVAL"
done