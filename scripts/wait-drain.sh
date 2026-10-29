#!/usr/bin/env bash
# Wait for dsv41 to drain (poll up to 60 min), then report. Does not stop anything.
set -uo pipefail
for i in $(seq 1 360); do
  line=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_running\{' | head -1)
  run=$(echo "$line" | awk '{print $2}')
  wait=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_waiting\{' | head -1 | awk '{print $2}')
  run=${run:-0}; wait=${wait:-0}
  if [ "${run%%.*}" = "0" ] && [ "${wait%%.*}" = "0" ]; then
    echo "DRAINED run=$run wait=$wait at $(date -u)"
    exit 0
  fi
  [ $((i % 6)) -eq 0 ] && echo "$(date -u) run=$run wait=$wait"
  sleep 10
done
echo "STILL BUSY after 60min run=$run wait=$wait"
exit 1
