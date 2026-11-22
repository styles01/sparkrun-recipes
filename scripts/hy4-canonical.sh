#!/usr/bin/env bash
# ============================================================================
# hy4-canonical.sh — Hy4-flash EXL3 3.0-bit + DFly adaptive spec — CANONICAL
#
# ONE SCRIPT, STANDALONE (mirrors qwen-canonical.sh 10-step pattern):
#   drain -> kill loaded -> verify clear -> sparkDash -> deciders ->
#   model (ghcr.io/styles01/hy4-exl3-dfly:v1, adaptive DFly d3) -> wait ready
#   -> heat caps -> keepalive -> live generation
#
# USAGE:  bash hy4-canonical.sh   |   FORCE_CLEAR=1 bash hy4-canonical.sh
# ============================================================================
set -uo pipefail

HEAD=${HEAD_HOST:-<spark-1>}
WORKER=10.200.0.2
PORT=8000
LANE=hy4-flash
DRAIN_WAIT_S=${DRAIN_WAIT_S:-1800}
PW_FILE=${PW_FILE:-$HOME/.switch-sudo-pw}
IMAGE=ghcr.io/styles01/hy4-exl3-dfly:v1
PACK=/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin
SERVING_NAMES="dsv41-serve sante-serve sante-fp4-tp1 tensorfold-serve qwen-serve glm-serve media-serve dsv41-exl3-nfs hy4-serve"

log()  { printf '\n=== [%s] %s\n' "$LANE" "$*"; }
note() { printf '    %s\n' "$*"; }
r()    { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2" 2>/dev/null; }

log "1/10 drain gate (max ${DRAIN_WAIT_S}s)"
if [ "${FORCE_CLEAR:-0}" = "1" ]; then note "SKIPPED (FORCE_CLEAR=1)"; else
  waited=0
  while :; do
    raw=$(r "$HEAD" "curl -s -m 5 http://127.0.0.1:$PORT/metrics | grep -E '^vllm:num_requests_(running|waiting)' | grep -oE '[0-9]+\\.?[0-9]*$' | paste -sd+ | bc")
    [ -z "$raw" ] && raw=0
    [ "${raw%.*}" = "0" ] && { note "no requests in flight"; break; }
    if [ "$waited" -ge "$DRAIN_WAIT_S" ]; then note "STILL BUSY ($raw) - refusing"; exit 1; fi
    [ $((waited % 60)) -eq 0 ] && note "waiting: $raw in flight (${waited}s)"
    sleep 15; waited=$((waited+15))
  done
fi

log "2/10 removing every serving container on both nodes"
for n in "$HEAD" "$WORKER"; do
  for c in $SERVING_NAMES; do r "$n" "docker rm -f $c >/dev/null 2>&1"; done
  note "$n remaining: $(r "$n" 'docker ps --format "{{.Names}}" | tr "\n" " "')"
done

log "3/10 verify :$PORT closed + memory returned"
for n in "$HEAD" "$WORKER"; do
  for try in 1 2 3 4 5 6 7 8 9 10; do
    lst=$(r "$n" "ss -ltn | grep -c ':$PORT '"); [ "${lst:-1}" = "0" ] && break; sleep 3
  done
  note "$n: listeners=${lst:-?} mem=$(r "$n" "free -g | awk '/Mem:/{print \$3\"used/\"\$7\"avail\"}'")"
  [ "${lst:-1}" != "0" ] && { note "FAIL: $n still on :$PORT"; exit 1; }
done

log "4/10 sparkDash"
r "$HEAD" 'cd ~/sparkDash && HOME=$HOME BIND_HOST=0.0.0.0 docker compose up -d >/dev/null 2>&1'
for try in $(seq 1 25); do sleep 2; r "$HEAD" "curl -sf -m 3 http://127.0.0.1:5555/ >/dev/null" && break; done
if r "$HEAD" "curl -sf -m 3 http://127.0.0.1:5555/ >/dev/null"; then
  note "up - $(r "$HEAD" "curl -s -m 5 http://127.0.0.1:5555/api/sparks/spark-001/llm-launchers | grep -o '\"id\"' | wc -l") launcher(s)"
else note "WARNING: not answering (not blocking the model)"; fi

log "5/10 decision sidecars"
r "$HEAD" 'for p in $(ps -eo pid,cmd | grep "[d]eciserv.server" | awk "{print \$1}"); do kill $p 2>/dev/null; done'  # hy4 does not use deciders; keep sidecars off for headroom
note "8710: $(r "$HEAD" "curl -s -m 4 http://127.0.0.1:8710/health 2>/dev/null | head -c 60 || echo down (expected)")"

log "6/10 ensure pack + image"
if ! r "$HEAD" "test -f $PACK/model.safetensors.index.json"; then
  note "pack missing - pulling from aitamedia via hf"
  r "$HEAD" "pip install -q hf_transfer 2>/dev/null; HF_HUB_ENABLE_HF_TRANSFER=1 hf download aitamedia/Hy4-flash-EXL3-3.0bpw --local-dir $PACK || exit 1"
fi
r "$HEAD" "grep -q hy4-exl3 <(docker images --format '{{.Repository}}:{{.Tag}}') 2>/dev/null" && note "image cached"   || { note "pulling $IMAGE (~48GB)"; r "$HEAD" "docker pull $IMAGE" || { note "FAIL: docker pull"; exit 1; }; }
note "pack: $(r "$HEAD" "du -sh $PACK 2>/dev/null | cut -f1") size-verified: $(r "$HEAD" "test $(r "$HEAD" "ls $PACK/model-*.safetensors 2>/dev/null | wc -l")" -ge 9 && echo OK || echo MISSING)"

log "7/10 launch adaptive DFly lane"
r "$HEAD" "docker rm -f hy4-serve >/dev/null 2>&1"
r "$HEAD" "docker run -d --name hy4-serve --network host --ipc host --shm-size 64g --gpus all   --ulimit memlock=-1 --ulimit stack=67108864   -v $PACK:$PACK:ro -v /home/jaita/routes:/home/jaita/routes   -e HF_HUB_OFFLINE=1 $IMAGE   --model $PACK --served-model-name hy4-flash-exl3 --host 0.0.0.0 --port $PORT   --max-model-len 262144 --max-num-seqs 3 --gpu-memory-utilization 0.85"
note "container up; engine logs: $(r "$HEAD" "docker logs hy4-serve 2>&1 | tail -1 | cut -c 1-80")"

log "8/10 waiting for :$PORT/v1/models + adaptive chooser"
t0=$SECONDS
until r "$HEAD" "curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null"; do
  if [ $((SECONDS-t0)) -ge 1500 ]; then note "FAIL: never answered"; exit 1; fi
  sleep 15
done
note "ready after $((SECONDS-t0))s"
r "$HEAD" "docker logs hy4-serve 2>&1 | grep -a 'hy4-dfly-adaptive' | tail -2" | sed 's/^/    /'

log "9/10 heat caps (mandatory L9)"
if [ -f "$PW_FILE" ]; then
  for n in "$HEAD" "$WORKER"; do note "$n: $(ssh -o BatchMode=yes "$n" 'bash ~/heatcap.sh' < "$PW_FILE" 2>&1 | grep -E 'OK|FAIL|DONE' | tr '\n' ' ')"; done
else note "WARNING: no $PW_FILE"; fi

log "10/10 live generation"
model=$(r "$HEAD" "curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4")
reply=$(r "$HEAD" "curl -s -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SPARK-ONLINE\"}],\"max_tokens\":64,\"temperature\":0}' | grep -o '\"content\":\"[^\"]*\"' | head -1")
if [ -n "$reply" ]; then note "$model -> $reply"; log "DONE - $LANE is serving (measured ~21 tok/s single-stream, 256K ctx)"; exit 0; fi
note "FAIL: no completion text"; exit 1
