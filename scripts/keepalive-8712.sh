#!/usr/bin/env bash
# keepalive-8712.sh (2026-10-24) — keep-warm pinger for the :8712 outguard deciserv.
# Same doctrine as keepalive-8710.sh; the CPU model needs a hot cache or every
# screen pays ~12s cold. Bespoke "warm_outguard" question keeps pinger traffic
# separable in /metrics by_question from outguard/outguard_send/donegate traffic.
# Exit when the daemon is gone (pinger discipline, same as 8710/8000).
# Deployed at spark1:/home/jaita/keepalive-8712.sh
# Mac master: /Users/clawdio/sparkrun-recipes/scripts/keepalive-8712.sh
# Log: /home/jaita/keepalive8712.log

ENDPOINT="http://127.0.0.1:8712"
HEALTH="$ENDPOINT/health"
DECIDE="$ENDPOINT/decide"
INTERVAL=240

PAYLOAD='{"state":"keepalive warm-up probe - no real tool output behind this screen.","questions":{"warm_outguard":{"type":"choice","instructions":"Is this a keepalive probe (always answer warm)?","criteria":{"warm_outguard":"yes it is a probe"}}}}'

while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$HEALTH" 2>/dev/null)
  if [ "$code" != "200" ]; then
    echo "$(date +%F' '%T) outguard down (health=$code), pinger exiting" >&2
    exit 0
  fi
  t0=$(date +%s.%N)
  http=$(curl -s --max-time 30 -o /dev/null -w "%{http_code}" -X POST "$DECIDE" \
    -H 'Content-Type: application/json' -d "$PAYLOAD")
  t1=$(date +%s.%N)
  ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", (b-a)*1000}')
  echo "$(date +%F' '%T) warm http=$http ${ms}ms"
  sleep "$INTERVAL"
done