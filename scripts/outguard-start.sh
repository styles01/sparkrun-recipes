#!/usr/bin/env bash
# outguard-start.sh (2026-10-24) — canonical launcher for the :8712 outguard deciserv.
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

SPID=$(ss -tlnp | grep ":8712" | grep -oP "pid=\K[0-9]+" | head -1)
[ -n "$SPID" ] && kill "$SPID"; sleep 2
mkdir -p ~/outguard-data
# pinger relaunch (codereview W-fix): keepalive-8712 exits permanently on a missed
# health window; the canonical launcher must revive it or the cache goes cold.
for PK in $(ps -eo pid,cmd | grep '[k]eepalive-8712.sh' | awk '{print $1}'); do kill "$PK" 2>/dev/null; done
sleep 1
cd ~/deci-serv || exit 1
DECISERV_POLICY=model-only \
setsid nohup python3 -m deciserv.server \
  --provider laya-gguf \
  --checkpoint ~/.qwen38-spark-engine/deciders/laya-gguf/laya-typed-q8_0.gguf \
  --device cpu --port 8712 \
  --log-path ~/outguard-data/decisions.jsonl \
  </dev/null >>~/outguard-data/outguard.log 2>&1 &
setsid nohup ~/keepalive-8712.sh </dev/null >>~/keepalive8712.log 2>&1 &
echo "outguard :8712 + pinger relaunched (pid ${!}); DECISERV_POLICY=model-only"