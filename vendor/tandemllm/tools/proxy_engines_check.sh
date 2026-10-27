#!/bin/bash
# The L1 adapters (P2, P3) against real vLLM and SGLang servers, for the night pass on the engine's
# host. One engine at a time, a small model, beside the live engine, and only when memory allows.
#
#   bash tools/proxy_engines_check.sh [--out DIR] [--only vllm|sglang] [--engine-port 8000]
#                                     [--plugins] [--ab vllm,sglang]
#
# For each engine:
#   1 memory   MemAvailable must stay at least 30 GiB after the live engine's own headroom (its
#              memory budget's ceiling less what it holds, from :8000's system body); without that
#              figure the margin is 46 GiB. Otherwise the engine is SKIPPED, never squeezed in.
#   2 lock     the box lock (~/.qwen38-box.flock, flock -w 1800) for the whole start-check-stop:
#              never two loads at once
#   3 start    the engine's container with the model from the local Hugging Face cache (offline),
#              a small KV share and eager mode; vLLM with --enable-prompt-tokens-details,
#              --reasoning-parser qwen3 and n-gram speculation (retried without speculation when
#              the start fails), SGLang with --enable-metrics, --enable-cache-report and
#              --reasoning-parser qwen3. A watcher in this script stops the container when
#              MemAvailable falls under 8 GiB during the load or the checks
#   4 check    tools/proxy_engines_live.py (discovery, the page, the relay, the mapped page against
#              the engine's own, the window figures, the choice constraint, X-Request-Id on vLLM)
#   5 stop     the container is removed (also on any exit of this script) and the lock released
#
# P6, with --plugins: the plugin wheels are built on the host from plugins/vllm and plugins/sglang
# (pip wheel, before any lock), mounted into the container and installed there before the engine
# starts; the engine posts to the check's own proxy (TANDEM_COLLECT_URL on 127.0.0.1:18531 or
# :18532, a token made for the run). tools/proxy_engines_live.py --plugin then checks the batches
# arrive, every row is joined, and the plugin's sums equal the engine's /metrics within 1 %.
# With --ab KINDS (default with --plugins: vllm), the plugin's cost: the engine three more times
# under the same lock, without the plugin, with it (posting to a standalone proxy), without it
# again, 20 streamed requests each (tools/proxy_plugin_ab.py); the gate is decode tok/s within 1 %
# of the mean of the two runs without it, `inconclusive` when those two differ by more than 1 %.
#
# Images: VLLM_IMAGE and SGLANG_IMAGE, by default the release tags the tables were checked against
# (vllm/vllm-openai:v0.30.0 and lmsysorg/sglang:v0.5.21-cu130, both with an arm64 image on Docker
# Hub), pulled before the lock is taken when the box does not have them; set either to an image
# already on the box to skip the pull. MODEL defaults to Qwen/Qwen3.5-0.8B and must be in the local
# Hugging Face cache. The versions the servers report are in each report.json.
set -u
cd "$(dirname "$0")/.." || exit 2
PY=${PY:-python3}
OUT=""; ONLY=""; EPORT=8000; PLUGINS=0; AB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --engine-port) EPORT="$2"; shift 2 ;;
    --plugins) PLUGINS=1; shift ;;
    --ab) AB="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
OUT=${OUT:-results/proxy-engines-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.21-cu130}
LOCK=${BOX_LOCK:-$HOME/.qwen38-box.flock}
HF=${HF_HOME:-$HOME/.cache/huggingface}
NAME=""
WATCH=""
SINK=""
[ $PLUGINS = 1 ] && [ -z "$AB" ] && AB=vllm
WHEELS="$OUT/wheels"
export TANDEM_COLLECT_TOKEN=${TANDEM_COLLECT_TOKEN:-$($PY -c 'import secrets; print(secrets.token_urlsafe(24))')}

cleanup() {
  [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
  [ -n "$NAME" ] && docker rm -f "$NAME" >/dev/null 2>&1
  [ -n "$SINK" ] && kill "$SINK" 2>/dev/null
  WATCH=""; NAME=""; SINK=""
}

build_wheels() {                            # P6: the two plugin wheels, on the host
  mkdir -p "$WHEELS"
  local rc=0
  if $PY -m pip --version >/dev/null 2>&1; then
    $PY -m pip wheel --no-deps -q -w "$WHEELS" plugins/vllm plugins/sglang \
      > "$OUT/wheels.log" 2>&1 || rc=1
  else
    { uv build --wheel -o "$WHEELS" plugins/vllm && uv build --wheel -o "$WHEELS" plugins/sglang; } \
      > "$OUT/wheels.log" 2>&1 || rc=1
  fi
  rm -rf plugins/*/build plugins/*/*.egg-info
  ls "$WHEELS"/tandemllm_vllm_plugin-*.whl "$WHEELS"/tandemllm_sglang_plugin-*.whl \
    >> "$OUT/wheels.log" 2>&1 || rc=1
  return $rc
}

# run_engine <kind> <with plugin 0|1> <collect port>: the docker arguments that install the plugin
# wheel first; prints nothing, sets PLUG_ARGS and PLUG_PRE
plug_args() {
  PLUG_ARGS=(); PLUG_PRE=""
  [ "$2" = 1 ] || return 0
  PLUG_ARGS=(-v "$(cd "$WHEELS" && pwd):/tandem-wheels:ro"
             -e "TANDEM_COLLECT_URL=http://127.0.0.1:$3/v1/collect/${1}1"
             -e TANDEM_COLLECT_TOKEN)
  PLUG_PRE="(python3 -m pip install -q --no-deps /tandem-wheels/tandemllm_${1}_plugin-*.whl || uv pip install --system --no-deps /tandem-wheels/tandemllm_${1}_plugin-*.whl) && "
}
trap cleanup EXIT INT TERM

mem_ok() {                                  # prints the decision; exit 0 when the load may run
  $PY - "$EPORT" <<'EOF'
import json, sys, urllib.request
GiB = 1 << 30
avail = None
for line in open("/proc/meminfo"):
    if line.startswith("MemAvailable:"):
        avail = int(line.split()[1]) * 1024
head = None
try:
    body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/dashboard/system",
                                             timeout=10).read())
    b = ((body.get("memory") or {}).get("limits") or {}).get("budget")
    if b:
        head = max(0, int(b["ceiling"]) - int(b["reserved"]))
except Exception as exc:                                       # noqa: BLE001
    print(f"engine budget unreadable ({type(exc).__name__}); using a 16 GiB margin")
need = 30 * GiB + (head if head is not None else 16 * GiB)
print(f"MemAvailable {avail / GiB:.1f} GiB, engine headroom "
      f"{'?' if head is None else f'{head / GiB:.1f} GiB'}, need {need / GiB:.1f} GiB")
sys.exit(0 if avail is not None and avail >= need else 1)
EOF
}

watch_mem() {                               # stop the container when memory runs low
  ( while sleep 2; do
      a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
      if [ "${a:-0}" -lt 8 ]; then
        echo "MemAvailable ${a} GiB: stopping $1" >> "$OUT/memory-stop.log"
        docker rm -f "$1" >/dev/null 2>&1; exit 0
      fi
    done ) &
  WATCH=$!
}

wait_up() {                                 # wait_up <port> <seconds>
  local t=0
  until curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    [ $t -ge "$2" ] && return 1
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || return 1
  done
}

VLLM_ARGS=(--host 127.0.0.1 --port 8101 --gpu-memory-utilization 0.08 --max-model-len 4096
           --enforce-eager --enable-prefix-caching --enable-prompt-tokens-details
           --reasoning-parser qwen3)

start_vllm() {                              # start_vllm <spec 0|1> [plugin 0|1] [collect port]
  local sp=()
  [ "$1" = 1 ] && sp=(--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}')
  plug_args vllm "${2:-$PLUGINS}" "${3:-18531}"
  NAME=px-l1-vllm
  docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -v "$HF:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 "${PLUG_ARGS[@]}" \
    --entrypoint /bin/bash "$VLLM_IMAGE" -c "${PLUG_PRE}exec vllm serve \"\$@\"" vllm \
    "$MODEL" "${VLLM_ARGS[@]}" "${sp[@]}" \
    > "$OUT/vllm-container.id" 2> "$OUT/vllm-start.err"
}

run_vllm() {
  local spec=1
  start_vllm 1 && watch_mem "$NAME"
  if ! wait_up 8101 900; then
    docker logs "$NAME" > "$OUT/vllm-server-spec.log" 2>&1
    cleanup
    echo "vLLM did not start with n-gram speculation; again without it" | tee -a "$OUT/summary.txt"
    spec=0
    start_vllm 0 && watch_mem "$NAME"
    wait_up 8101 900 || { docker logs "$NAME" > "$OUT/vllm-server.log" 2>&1; cleanup; return 1; }
  fi
  local flag=(); [ $spec = 1 ] && flag=(--spec)
  [ $PLUGINS = 1 ] && flag+=(--plugin --proxy-port 18531)
  $PY tools/proxy_engines_live.py --kind vllm --url http://127.0.0.1:8101/v1 --model "$MODEL" \
    --out "$OUT/vllm" "${flag[@]}" > "$OUT/vllm-check.log" 2>&1
  local rc=$?
  docker logs "$NAME" > "$OUT/vllm-server.log" 2>&1
  cleanup
  return $rc
}

SGLANG_ARGS=(--host 127.0.0.1 --port 8102 --mem-fraction-static 0.08 --context-length 4096
             --disable-cuda-graph --enable-metrics --enable-cache-report --reasoning-parser qwen3)

start_sglang() {                            # start_sglang [plugin 0|1] [collect port]
  plug_args sglang "${1:-$PLUGINS}" "${2:-18532}"
  NAME=px-l1-sglang
  docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -v "$HF:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 "${PLUG_ARGS[@]}" \
    --entrypoint /bin/bash "$SGLANG_IMAGE" \
    -c "${PLUG_PRE}exec python3 -m sglang.launch_server \"\$@\"" sglang \
    --model-path "$MODEL" "${SGLANG_ARGS[@]}" \
    > "$OUT/sglang-container.id" 2> "$OUT/sglang-start.err"
}

run_sglang() {
  start_sglang || return 1
  watch_mem "$NAME"
  wait_up 8102 900 || { docker logs "$NAME" > "$OUT/sglang-server.log" 2>&1; cleanup; return 1; }
  local flag=(); [ $PLUGINS = 1 ] && flag=(--plugin --proxy-port 18532)
  $PY tools/proxy_engines_live.py --kind sglang --url http://127.0.0.1:8102/v1 --model "$MODEL" \
    --out "$OUT/sglang" "${flag[@]}" > "$OUT/sglang-check.log" 2>&1
  local rc=$?
  docker logs "$NAME" > "$OUT/sglang-server.log" 2>&1
  cleanup
  return $rc
}

# P6: the plugin's cost, three starts of one engine under the lock already held (off, on, off)
sink_up() {                                 # a standalone proxy the plugin posts to (port, kind)
  local d="$OUT/ab-proxy-$2" u; mkdir -p "$d"
  u=$(echo "$2" | tr a-z A-Z)
  env "TANDEM_BACKENDS=${2}1" "TANDEM_BACKEND_${u}1_KIND=$2" \
    "TANDEM_BACKEND_${u}1_URL=http://127.0.0.1:$3/v1" QSE_TEST=1 \
    $PY -m proxy --port "$1" --config-db "$d/config.sqlite3" --usage-ledger off --no-keychain \
    > "$d/proxy.log" 2>&1 &
  SINK=$!
}

run_ab() {                                  # run_ab <kind>
  local kind=$1 port=8101 cport=18541 args
  [ "$kind" = sglang ] && port=8102 && cport=18542
  args="${VLLM_ARGS[*]}"; [ "$kind" = sglang ] && args="${SGLANG_ARGS[*]}"
  local i=0 arm
  for arm in off on off; do
    i=$((i+1))
    if [ "$kind" = vllm ]; then start_vllm 0 "$([ $arm = on ] && echo 1 || echo 0)" "$cport"
    else start_sglang "$([ $arm = on ] && echo 1 || echo 0)" "$cport"; fi
    [ $arm = on ] && sink_up "$cport" "$kind" "$port"
    watch_mem "$NAME"
    if ! wait_up "$port" 900; then
      docker logs "$NAME" > "$OUT/ab-$kind-$i-server.log" 2>&1; cleanup; return 1
    fi
    $PY tools/proxy_plugin_ab.py --url "http://127.0.0.1:$port/v1" --model "$MODEL" --arm "$arm" \
      --n 20 --engine-args "$args" --out "$OUT/ab-$kind-$i-$arm.json" \
      >> "$OUT/ab-$kind.log" 2>&1 || { cleanup; return 1; }
    docker logs "$NAME" > "$OUT/ab-$kind-$i-server.log" 2>&1
    cleanup
  done
  $PY tools/proxy_plugin_ab.py --compare "$OUT/ab-$kind-1-off.json" "$OUT/ab-$kind-2-on.json" \
    "$OUT/ab-$kind-3-off.json" --out "$OUT/ab-$kind.json" >> "$OUT/ab-$kind.log" 2>&1
}

FAILS=0
: > "$OUT/summary.txt"
if [ $PLUGINS = 1 ] && ! build_wheels; then
  echo "FAIL plugins: the wheels did not build (see wheels.log); running without the plugins" \
    | tee -a "$OUT/summary.txt"
  PLUGINS=0; AB=""; FAILS=$((FAILS+1))
fi
for kind in vllm sglang; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$kind" ] && continue
  if ! mem_ok > "$OUT/$kind-memory.txt" 2>&1; then
    echo "SKIP $kind: $(tail -1 "$OUT/$kind-memory.txt")" | tee -a "$OUT/summary.txt"
    continue
  fi
  img=$VLLM_IMAGE; [ $kind = sglang ] && img=$SGLANG_IMAGE
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    docker pull "$img" > "$OUT/$kind-pull.log" 2>&1 || {
      echo "SKIP $kind: could not pull $img" | tee -a "$OUT/summary.txt"; continue; }
  fi
  t0=$(date +%s)
  exec 9>"$LOCK"
  if ! flock -w 1800 9; then
    echo "SKIP $kind: the box lock stayed held for 30 minutes" | tee -a "$OUT/summary.txt"
    exec 9>&-; continue
  fi
  # memory again under the lock: a hold that ended just now may have changed it
  if ! mem_ok >> "$OUT/$kind-memory.txt" 2>&1; then
    echo "SKIP $kind: $(tail -1 "$OUT/$kind-memory.txt")" | tee -a "$OUT/summary.txt"
    exec 9>&-; continue
  fi
  if "run_$kind"; then r=PASS; else r=FAIL; FAILS=$((FAILS+1)); fi
  echo "$r $kind $(( $(date +%s) - t0 ))s" | tee -a "$OUT/summary.txt"
  if [[ ",$AB," == *",$kind,"* ]]; then
    t0=$(date +%s)
    if mem_ok >> "$OUT/$kind-memory.txt" 2>&1 && run_ab "$kind"; then
      r="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"].upper())' \
           "$OUT/ab-$kind.json" 2>/dev/null || echo FAIL)"
    else r=FAIL; fi
    [ "$r" = FAIL ] && FAILS=$((FAILS+1))
    echo "$r ab-$kind $(( $(date +%s) - t0 ))s" | tee -a "$OUT/summary.txt"
  fi
  exec 9>&-
done
echo "failed: $FAILS (logs in $OUT)" | tee -a "$OUT/summary.txt"
exit $FAILS
