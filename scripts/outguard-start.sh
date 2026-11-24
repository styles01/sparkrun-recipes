#!/usr/bin/env bash
# outguard-start.sh (2026-11-13 rev) — canonical launcher for the :8712 outguard deciserv.
#
# Why: the adversarial review (deleg_35b8db04, B1) proved :8710's hybrid gate floors
# trip on tool-output prose and never populate by_question off the floor path —
# screens pointed at :8710 would produce garbage telemetry. This second laya-class
# process runs DECISERV_POLICY=model-only so every screen gets a real model verdict
# with separable by_question counters. :8710 (harvest gate) is NOT touched.
#
# - Clearly non-serving port (8712). Model: same laya-typed-q8_0 CPU checkpoint.
# - Isolated data: ~/outguard-data/ (decisions.jsonl + log), never ~/deciserv-data.
# - One canonical launcher law applies (no ad-hoc docker/processes).
# Deployed at spark1:/home/jaita/outguard-start.sh
# Mac master: /Users/clawdio/sparkrun-recipes/scripts/outguard-start.sh
# Log: /home/jaita/outguard-data/outguard.log
#
# 2026-11-13 HARDENING (three fixes, after :8712 went dark twice):
#   1. CUDA_VISIBLE_DEVICES="" — the decider is CPU-only. Previously a launcher chain
#      that lost --device cpu let the provider touch CUDA and die with
#      "torch.AcceleratorError: CUDA error: out of memory" at cuDevicePrimaryCtxRetain
#      while a serving lane held the GPU. Hiding all devices makes that impossible:
#      a GPU path now fails loudly at import instead of half-loading.
#   2. Readiness ASSERTION — waits for :8712, then requires /health to report
#      device=cpu. Fails fast and loudly instead of leaving a dead port.
#   3. Log truncation — the log kept a stale "decider lane ready on :8712" line from a
#      previous run sitting above a later OOM, which made 11 consecutive heartbeat
#      readings report "dark" with no visible cause. Old log is rotated, not deleted.
#
# Exit codes: 0 = ready and verified | 2 = process died | 3 = health never answered
#             4 = health answered but device was NOT cpu

set -u

PORT=8712
HEALTH_WAIT_S=${HEALTH_WAIT_S:-60}
LOG=~/outguard-data/outguard.log

# ---------------------------------------------------------------- 1. CPU-ONLY, HARD
# Hide every CUDA device from the decider and its children. --device cpu remains as
# belt; this is the braces. If anything in the chain tries the GPU, it fails at
# import with "no CUDA-capable device" rather than OOMing on context creation.
export CUDA_VISIBLE_DEVICES=""
# Belt-and-braces: keep libcuda from being probed at all by the torch loader.
export PYTORCH_NVML_BASED_CUDA_CHECK=0

# ---------------------------------------------------------------- 2. rotate log
mkdir -p ~/outguard-data
if [ -f "$LOG" ]; then
  mv "$LOG" "$LOG.prev-$(date +%Y%m%dT%H%M%S)"
fi

# ---------------------------------------------------------------- 3. stop old
SPID=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP "pid=\K[0-9]+" | head -1)
if [ -n "${SPID:-}" ]; then
  kill "$SPID" 2>/dev/null
  for _ in $(seq 1 20); do ss -ltn | grep -q ":${PORT} " || break; sleep 0.5; done
fi
# pinger relaunch (codereview W-fix): keepalive-8712 exits permanently on a missed
# health window; the canonical launcher must revive it or the cache goes cold.
for PK in $(ps -eo pid,cmd | grep '[k]eepalive-8712.sh' | awk '{print $1}'); do
  kill "$PK" 2>/dev/null
done
sleep 1

# ---------------------------------------------------------------- 4. launch
cd ~/deci-serv || { echo "FAIL: ~/deci-serv missing"; exit 2; }
DECISERV_POLICY=model-only \
CUDA_VISIBLE_DEVICES="" \
setsid nohup python3 -m deciserv.server \
  --provider laya-gguf \
  --checkpoint ~/.qwen38-spark-engine/deciders/laya-gguf/laya-typed-q8_0.gguf \
  --device cpu --port "${PORT}" \
  --log-path ~/outguard-data/decisions.jsonl \
  </dev/null >>"$LOG" 2>&1 &

LAUNCH_PID=$!

# ---------------------------------------------------------------- 5. readiness ASSERT
ready=0
for _ in $(seq 1 "$HEALTH_WAIT_S"); do
  sleep 1
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null && [ "$ready" -eq 0 ]; then
    # process may have re-exec'd under setsid; only fail if the PORT never opens
    :
  fi
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/tmp/outguard-health.$$ 2>/dev/null; then
    ready=1
    break
  fi
done

if [ "$ready" -ne 1 ]; then
  echo "FAIL: :${PORT} never answered /health within ${HEALTH_WAIT_S}s"
  echo "--- last 12 log lines:"
  tail -12 "$LOG"
  exit 3
fi

HEALTH=$(cat /tmp/outguard-health.$$ 2>/dev/null)
rm -f /tmp/outguard-health.$$

# device must be cpu. If it says cuda, the CPU-only guarantee is broken.
if ! printf '%s' "$HEALTH" | grep -q '"device": *"cpu"'; then
  echo "FAIL: :${PORT} is up but device is NOT cpu"
  echo "  health: $HEALTH"
  exit 4
fi

# ---------------------------------------------------------------- 6. pinger
setsid nohup ~/keepalive-8712.sh </dev/null >>~/keepalive8712.log 2>&1 &

# ---------------------------------------------------------------- 7. decision probe
# Health alone is not enough — the trading runner needs /decide. Prove it decides.
DECIDE=$(curl -s -m 60 -X POST "http://127.0.0.1:${PORT}/decide" \
  -H 'Content-Type: application/json' \
  -d '{"state":"launcher self-test","questions":{"t":{"type":"choice","instructions":"Is this a self-test?","criteria":["yes","no"]}}}' 2>/dev/null)

if ! printf '%s' "$DECIDE" | grep -q '"probabilities"'; then
  echo "FAIL: :${PORT} health OK but /decide did not return probabilities"
  echo "  decide: ${DECIDE:0:300}"
  exit 3
fi

echo "OK: outguard :${PORT} ready — device=cpu, /decide returning probabilities"
echo "  health: $HEALTH"
echo "  log:    $LOG"
