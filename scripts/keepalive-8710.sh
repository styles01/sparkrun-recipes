#!/usr/bin/env bash
# keepalive-8710.sh (2026-10-23) — laya deciserv keep-warm pinger.
# Why: laya's CPU gate was measuring 22s avg lifetime (cold starts, 0 fast-path
# hits) against a 1.5s warm gate — cold latency was outcosting dsv41 detours.
# This pinger keeps the gate hot so real agent decisions land fast.
#
# Design notes:
# - The warm POST uses a bespoke "warm" question name, so pinger traffic is
#   separable in /metrics by_question (real-traffic stats stay clean).
# - Exit when the daemon is gone (same pattern as keepalive-8000); the rearm
#   cron (or relaunch-dsv41-canonical.sh's sidecar step) revives it on relight.
# Deployed at spark1:/home/jaita/keepalive-8710.sh
# Mac master: /Users/clawdio/sparkrun-recipes/scripts/keepalive-8710.sh
# Log: /home/jaita/keepalive8710.log

ENDPOINT="http://127.0.0.1:8710"
HEALTH="$ENDPOINT/health"
DECIDE="$ENDPOINT/decide"
INTERVAL=240

PAYLOAD='{"state":"keepalive warm-up probe - no real tool call behind this gate.","questions":{"warm":{"type":"choice","instructions":"Is this a keepalive probe (always answer warm)?","criteria":{"warm":"yes it is a probe"}}}}'

while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$HEALTH" 2>/dev/null)
  if [ "$code" != "200" ]; then
    echo "$(date +%F' '%T) deciserv down (health=$code), pinger exiting" >&2
    exit 0
  fi
  t0=$(date +%s.%N)
  http=$(curl -s --max-time 20 -o /dev/null -w "%{http_code}" -X POST "$DECIDE" \
    -H 'Content-Type: application/json' -d "$PAYLOAD")
  t1=$(date +%s.%N)
  ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", (b-a)*1000}')
  echo "$(date +%F' '%T) warm http=$http ${ms}ms"
  sleep "$INTERVAL"
done