#!/usr/bin/env bash
# qwen-fn SWITCH to Mia's TensorFold v1.1.4 lane, 2026-10-30.
# Normal switch shape: drain -> teardown dsv41 -> clone/prepare -> boot on :8000 -> verify.
set -uo pipefail

echo "=== $(date -u) QWEN SWITCH START ==="

# 1. drain gate (poll, do not stop at one sample)
for i in $(seq 1 180); do
  run=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_running\{' | head -1 | awk '{print $2}')
  wait=$(curl -s -m8 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:num_requests_waiting\{' | head -1 | awk '{print $2}')
  run=${run:-0}; wait=${wait:-0}
  if [ "${run%%.*}" = "0" ] && [ "${wait%%.*}" = "0" ]; then
    echo "DRAINED run=$run wait=$wait $(date -u)"; break
  fi
  [ $((i % 6)) -eq 0 ] && echo "  $(date -u) run=$run wait=$wait"
  sleep 10
done

# 2. teardown dsv41 (canonical)
echo "--- teardown dsv41 ---"
cd /home/jaita/dsv41_recipe_ablit && HEAD=10.200.0.1 WORKER=10.200.0.2 bash scripts/launch/dsv41_tp2.sh down 2>&1 | tail -3
sleep 20
echo "8000 listeners: $(ss -ltn | grep -c ':8000')"
echo "MemAvailable: $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) GiB"
echo "disk free: $(df -B1 / | awk 'NR==2 {print int($4/1e9)}') GB"

# 3. clone Mia's lane
echo "--- clone ---"
cd /home/jaita/src || exit 1
if [ ! -d Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/.git ]; then
  git clone -q https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
fi
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold || exit 1
echo "HEAD: $(git log --oneline -1)"

# 4. DRY RUN first — never fire blind
echo "--- DRY_RUN ---"
PORT=8000 DRY_RUN=1 ./start.sh 2>&1 | tail -25

echo "=== $(date -u) PREP DONE — read the DRY_RUN output before real boot ==="
