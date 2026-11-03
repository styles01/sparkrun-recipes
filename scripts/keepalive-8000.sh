#!/usr/bin/env bash
# keepalive-8000.sh v2 (2026-10-23) — adaptive-model generation pinger + TP2 dead-ring detector.
# v1 kept the generation path warm but hardcoded the model id and could not catch a
# head-without-rank ring (a TP2 head answers /v1/models + /health while every
# completion hangs — see spark-lanes 10/23 dsv41 exhibit).
#
# v2 behavior every probe:
#   1. /health gate as before (exit if server gone).
#   2. Pull first served id from /v1/models — pings whatever lane owns the port.
#   3. Real 1-token generation POST; 3 consecutive non-200 => !! RING-DEAD marker lines
#      in the log (the thing that would have caught the 5h hang at 11:00Z instead of 12:00Z).
#
# Deployed at spark1:/home/jaita/keepalive-8000.sh (called by SPARK-KEEPALIVE rearm cron);
# Mac master = /Users/clawdio/sparkrun-recipes/scripts/keepalive-8000.sh.
# Log path stays /home/jaita/keepalive8000.log (rearm redirects; pinger echoes only).

ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
HEALTH="http://127.0.0.1:8000/health"
MODELS="http://127.0.0.1:8000/v1/models"
INTERVAL=27
miss=0

while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$HEALTH" 2>/dev/null)
  if [ "$code" != "200" ]; then
    echo "$(date +%F' '%T) server down (health=$code), pinger exiting" >&2
    exit 0
  fi
  model=$(curl -s --max-time 3 "$MODELS" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -z "$model" ] && model="dsv41"
  t0=$(date +%s.%N)
  http=$(curl -s --max-time 12 -o /dev/null -w "%{http_code}" "$ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}")
  t1=$(date +%s.%N)
  ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", (b-a)*1000}')
  if [ "$http" = "200" ]; then
    miss=0
    echo "$(date +%F' '%T) ping ok model=$model ${ms}ms"
  else
    miss=$((miss + 1))
    echo "$(date +%F' '%T) ping FAIL http=$http ${ms}ms model=$model miss=$miss"
    if [ "$miss" -ge 3 ]; then
      echo "$(date +%F' '%T) !! RING-DEAD $miss consecutive failed generation probes - a metadata-answering head with a dead rank still counts as DOWN; check ss -tn '( sport = :25000 )' for peer count"
    fi
  fi
  sleep "$INTERVAL"
done