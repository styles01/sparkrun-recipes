#!/usr/bin/env bash
# Kill the Muse bench harness (llama-benchy against :8888) and report leftover queue.
set -uo pipefail
echo "=== $(date -u) stopping bench harness ==="

for round in 1 2 3 4; do
  mapfile -t PIDS < <(pgrep -f "llama-benchy --base-url http://localhost:8888" || true)
  if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "no benchy processes"
    break
  fi
  echo "round $round: ${PIDS[*]}"
  for p in "${PIDS[@]}"; do
    kill "$p" 2>/dev/null || true
  done
  sleep 4
  mapfile -t STILL < <(pgrep -f "llama-benchy --base-url http://localhost:8888" || true)
  for p in "${STILL[@]}"; do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 3
done

echo "--- final ---"
mapfile -t FINAL < <(pgrep -f "llama-benchy" || true)
echo "remaining llama-benchy pids: ${FINAL[*]:-none}"
curl -s -m5 localhost:8888/health | head -c 200
echo
