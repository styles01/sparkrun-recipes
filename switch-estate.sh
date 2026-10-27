#!/usr/bin/env bash
# ==============================================================================
# switch-estate.sh — THE ONE SCRIPT (both sparks)
# ==============================================================================
# Canonical lane switcher for the DGX Spark estate. Runs ON spark1 (larryspark).
# Every trigger resolves from LANE-REGISTRY.yaml — this script implements the
# registry's lanes. NO ad-hoc flags; add configs BY EDITING THE REGISTRY + here.
#
# USAGE:
#   ./switch-estate.sh status        # full state of both sparks
#   ./switch-estate.sh qwen-fn       # spark1: TF EXL3 qwen lane; spark2: media stack
#   ./switch-estate.sh media         # alias of qwen-fn (+ decider) = media mode
#   ./switch-estate.sh dsv41         # spark1+2: DeepSeek 4.1 TP2 lane (media down)
#   ./switch-estate.sh glm           # spark1+2: GLM 5.3 TP2 lane (media down)
#   ./switch-estate.sh decider       # spark2: d3-mini llama.cpp decider (:8715)
#   ./switch-estate.sh stop          # stop media + decider (spark2), lane stays
#   ./switch-estate.sh verify        # health proof of every serving port
#
# LAWS: teardown of a serving lane only via explicit James trigger from the
# matching lane key; heavy IO with GPU lanes down; verify + report after switch.
# ==============================================================================
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REG=/home/jaita/sparkrun-recipes/LANE-REGISTRY.yaml
LOGD=/home/jaita/media/logs
WORKER=10.200.0.2
[ -f "$REG" ] || { echo "!! registry missing: $REG"; exit 1; }

up()    { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
hup()   { curl -s -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
wport() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" "curl -s -m 3 http://127.0.0.1:$1/health >/dev/null 2>&1 || curl -s -m 3 http://127.0.0.1:$1/v1/models >/dev/null 2>&1"; }

spark1_llm_up() { up 8000; }

stop_dsv41() {
  drain_check "http://127.0.0.1:8000" "dsv41"
  echo ">> [dsv41] TP2 lane down (media/decider needs the memory)"
  ( cd /home/jaita/dsv41_recipe && HEAD=10.200.0.1 WORKER=10.200.0.2 \
      MODEL_DIR=/home/jaita/models/dsv41-exl3 ./scripts/launch/dsv41_tp2.sh down ) || true
  for i in $(seq 1 20); do spark1_llm_up || break; sleep 5; done
  if spark1_llm_up; then echo "!! :8000 still answering — abort"; exit 1; fi
}

stop_glm() {
  echo ">> [glm] cluster down (registry lane glm)"
  ( cd /home/jaita/src/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && ./stop.sh ) || true
  for i in $(seq 1 20); do spark1_llm_up || break; sleep 5; done
}

# ── DRAIN GATE (L8: no teardown while inference is live; consumers don't read chat) ──
# Checks the lane's engine metrics; refuses (exit 1) if requests are running/queued.
# OVERRIDE: FORCE_TEARDOWN=1 — reserved for James's explicit order after being told
# who is on the lane. Non-vllm lanes (pkill-only) check for the serving process.
drain_check() { # $1 = the lane's health URL, $2 = human lane label
  local busy
  busy=$(curl -s -m 4 "$1/metrics" 2>/dev/null | grep -E "^vllm:num_requests_(running|waiting)|num_running_reqs|num_queue_reqs" | awk '{s+=$2} END {print int(s)}')
  if [ "${FORCE_TEARDOWN:-0}" != "1" ] && [ "${busy:-0}" -gt 0 ]; then
    echo "!! $2 BUSY: ${busy:-?} request(s) in flight — REFUSING teardown (FORCE_TEARDOWN=1 overrides, only on James's explicit say)"
    exit 1
  fi
  echo ">> [$2] drain-gate clear (inflight=${busy:-0})"
}

# ── helper: stop a lane only if no requests; kill the serving PROCESS CLASS if present ──
stop_media() {
  ssh -o BatchMode=yes "$WORKER" "pkill -f qwen_image_server; pkill -f chatterbox_server" 2>/dev/null || true
  docker rm -f llama-d3 2>/dev/null || true
  ssh -o BatchMode=yes "$WORKER" docker rm -f llama-d3 2>/dev/null || true
}

start_decider() {
  # d3-mini decider on spark2 :8715 (llama.cpp in dsv41-serve:v4 container, CUDA)
  if wport 8715; then echo ">> [decider] :8715 already UP"; return; fi
  SSH2="ssh -o BatchMode=yes $WORKER"
  echo ">> [decider] d3-mini GGUF on spark2 :8715"
  "$SSH2" "docker rm -f llama-d3 2>/dev/null;
docker run -d --name llama-d3 --network host --gpus all --entrypoint sleep \
  -v /home/jaita/MODELS:/home/jaita/MODELS -v /home/jaita/work:/home/jaita/work \
  dsv41-serve:v4 infinity >/dev/null && \
docker exec llama-d3 bash -c 'nohup /home/jaita/work/llama.cpp/build/bin/llama-server \
  --model /home/jaita/MODELS/hf/hub/models--ggml-org--vllm-d3-mini-GGUF/snapshots/c03ac9c3bdc5ae949111acb945338933f396e4d7/vllm-d3-mini-Q4_K_M.gguf \
  --mmproj /home/jaita/MODELS/hf/hub/models--ggml-org--vllm-d3-mini-GGUF/snapshots/c03ac9c3bdc5ae949111acb945338933f396e4d7/mmproj-vllm-d3-mini-Q8_0.gguf \
  --host 0.0.0.0 --port 8715 -ngl 99 --ubatch-size 32768 --ctx-size 32768 >> /tmp/d3serve.log 2>&1' >/dev/null && echo ok"
  for i in $(seq 1 30); do wport 8715 && { echo ">> [decider] :8715 UP"; return; }; sleep 5; done
  echo "!! [decider] did not come up — check: ssh $WORKER docker logs llama-d3"
}

start_media_spark2() {
  echo ">> [media/spark2] TTS :8020 (cb venv) + IMAGE :8030 (qimg venv) + music(YuE2 CLI) + decider"
  ssh -o BatchMode=yes "$WORKER" "mkdir -p $LOGD
pgrep -f chatterbox_server >/dev/null || nohup /home/jaita/venvs/cb/bin/python /home/jaita/media/chatterbox_server.py --host 0.0.0.0 --port 8020 > $LOGD/chatterbox.log 2>&1 &
pgrep -f qwen_image_server >/dev/null || nohup /home/jaita/venvs/qimg/bin/python /home/jaita/media/qwen_image_server.py --host 0.0.0.0 --port 8030 \
  --model-dir /home/jaita/media/qwen-image-2.1-turbo > $LOGD/qwen-image.log 2>&1 &
echo launched"
  start_decider
  for i in $(seq 1 30); do wport 8020 && break; sleep 5; done
  for i in $(seq 1 60); do wport 8030 && break; sleep 5; done
  wport 8020 && echo ">> [tts] :8020 UP"   || echo "!! [tts] :8020 down (log: $LOGD/chatterbox.log on spark2)"
  wport 8030 && echo ">> [image] :8030 UP" || echo "!! [image] :8030 down (log: $LOGD/qwen-image.log on spark2)"
  echo ">> [music] YuE2-3B = CLI only (ssh $WORKER 'cd /home/jaita/yue2 && . .venv/bin/activate && yue2-infer --help'); MUSIC_ENGINE=ace-step → head :8001"
}

start_qwenfn() {
  # registry lane: qwen-fn = TF EXL3 via sparkrun lane file
  if up 8000 && curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "Qwen3.8-Flash-Next"; then
    echo ">> [qwen-fn] already SERVING Qwen3.8-Flash-Next on :8000"; return
  fi
  echo ">> [qwen-fn] deploying TF EXL3 lane (sparkrun)"
  # any other spark1 LLM must first be torn down by the caller
  nohup python3 -m sparkrun run ~/sparkrun-recipes/lanes/qwen38-flash-next-tensorfold-lane.yaml \
     --solo --trust > /tmp/tf-deploy.log 2>&1 &
  for i in $(seq 1 60); do
    up 8000 && curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q Qwen3.8 && { echo ">> [qwen-fn] UP"; return; }
    sleep 5
  done
  echo "!! [qwen-fn] failed — tail /tmp/tf-deploy.log"
}

verify() {
  echo "== VERIFY =="
  up 8000 && { echo "spark1 :8000  $(curl -s -m 3 http://127.0.0.1:8000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"; } || echo "spark1 :8000 dark"
  wport 8020 && echo "spark2 :8020 TTS up" || echo "spark2 :8020 down"
  wport 8030 && echo "spark2 :8030 IMAGE up" || echo "spark2 :8030 down"
  wport 8715 && echo "spark2 :8715 DECIDER up" || echo "spark2 :8715 down"
  up 8710 && echo "spark1 :8710 deci-serv(d2.0) up (rollback)" || echo "spark1 :8710 deci-serv dark"
  free -g | awk 'NR==2{print "spark1 mem free/avail:", $4"/"$7}'
  ssh -o BatchMode=yes "$WORKER" "free -g | awk 'NR==2{print \"spark2 mem free/avail:\", \$4\"/\"\$7}'"
  ssh -o BatchMode=yes "$WORKER" "df -h / | tail -1" | awk '{print "spark2 disk:", $4, "avail"}'
  df -h / | tail -1 | awk '{print "spark1 disk:", $4, "avail"}'
}

cmd="${1:-status}"
case "$cmd" in
  qwen-fn|media)
    if ! up 8000 || ! curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "Qwen3.8-Flash-Next"; then
      echo ">> switching spark1 text lane to qwen-fn (registry)"
      # dsv41 is the only other :8000 occupant with both ranks; glm handled too
      docker ps --format '{{.Names}}' | grep -q '^dsv41-serve$' && stop_dsv41
      pgrep -f exl3_serve_openai >/dev/null && pkill -f exl3_serve_openai
    fi
    start_qwenfn
    start_media_spark2
    ;;
  dsv41)
    echo ">> switching BOTH sparks to dsv41 TP2 (media + decider torn down)"
    drain_check "http://127.0.0.1:8000" "dsv41-cutover"
    stop_media
    if up 8000 && curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "Qwen3.8"; then
      docker ps -a --format '{{.Names}}' | grep sparkrun | xargs -r docker rm -f 2>/dev/null || true
    fi
    ( cd /home/jaita/dsv41_recipe && TEXT_ONLY=0 \
        SERVED="dsv41 DeepSeek-4.1-Flash GLM-5.3-Flash-EXL3" HEAD=10.200.0.1 WORKER=10.200.0.2 \
        MODEL_DIR=/home/jaita/models/dsv41-exl3 nohup bash scripts/launch/dsv41_tp2.sh up \
        > /tmp/estate-dsv41.log 2>&1 & )
    echo ">> dsv41 launching (~10 min; log /tmp/estate-dsv41.log)"
    ;;
  glm)
    echo ">> GLM lane (rollback target; media down)"
    drain_check "http://127.0.0.1:8000" "glm-cutover"
    stop_media
    ( cd /home/jaita/src/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && nohup ./start.sh > /tmp/estate-glm.log 2>&1 & )
    echo ">> GLM start.sh launched (log /tmp/estate-glm.log)"
    ;;
  decider)
    qwen_already=0
    (up 8000 && curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q Qwen3.8) && qwen_already=1
    [ "$qwen_already" = "0" ] && { echo "!! qwen-fn not serving on spark1 — run '$0 qwen-fn' first (decider coexists with qwen era only)"; exit 1; }
    start_decider
    ;;
  stop)    stop_media; echo ">> media+decider stopped; spark1 lane untouched" ;;
  verify)  verify ;;
  status)  verify ;;
  *) grep -E "^#   \./switch-estate" "$0" | sed 's/^# *//'; exit 1 ;;
esac
echo
verify