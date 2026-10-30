#!/usr/bin/env bash
# MUSE BENCH WINDOW — 2026-10-30 01:00 ET
# 1) drain-gate dsv41  2) stop it  3) boot Muse TP2  4) verify  5) leave serving for bench
set -uo pipefail

echo "=== $(date -u) MUSE WINDOW START ==="

# --- 1. drain gate: poll, never stop at a single non-zero sample ---
echo "--- drain gate ---"
for i in $(seq 1 180); do
  run=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_running\{' | head -1 | awk '{print $2}')
  wait=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_waiting\{' | head -1 | awk '{print $2}')
  run=${run:-0}; wait=${wait:-0}
  if [ "${run%%.*}" = "0" ] && [ "${wait%%.*}" = "0" ]; then
    echo "DRAINED run=$run wait=$wait at $(date -u)"; break
  fi
  [ $((i % 6)) -eq 0 ] && echo "  $(date -u) run=$run wait=$wait"
  sleep 10
done

# --- 2. teardown dsv41 (canonical launcher) ---
echo "--- stopping dsv41 ---"
cd /home/jaita/dsv41_recipe_ablit && HEAD=10.200.0.1 WORKER=10.200.0.2 bash scripts/launch/dsv41_tp2.sh down 2>&1 | tail -3
sleep 20
echo "port 8000 listeners after stop: $(ss -ltn | grep -c ':8000')"

# --- 3. boot Muse TP2 ---
echo "--- booting Muse (PROFILE=long) ---"
cd /home/jaita/src/muspark-2dgx || exit 1
nohup setsid bash -c './start.sh > /home/jaita/muse-bench-boot.log 2>&1' >/dev/null 2>&1 </dev/null &
echo "fired; waiting for readiness"

# --- 4. wait for the API ---
for i in $(seq 1 60); do
  if curl -s -m5 localhost:8888/health 2>/dev/null | grep -q '"warming": false'; then
    echo "MUSE READY at $(date -u)"
    curl -s -m5 localhost:8888/v1/models | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"];print([(m["id"],m.get("max_model_len")) for m in d])'
    exit 0
  fi
  sleep 15
done
echo "MUSE NOT READY after 15 min"
tail -20 /home/jaita/muse-bench-boot.log
exit 1
