#!/bin/bash
# Start the engine on :8000. Idempotent: if a healthy one is already there, say so and stop.
#
# The port is the contract. your-host.example reaches this box's nginx, which proxies :8000, so
# whatever answers on :8000 is what Open WebUI calls the model -- and exactly one thing may.
#
#   ops/start.sh                      the configured engine
#   ops/start.sh --setup [--host A]   setup mode (docs/setup.md): no model of Qwen's, only the dashboard,
#                                     /v1/setup/* and Wren, the setup guide's assistant; on 127.0.0.1
#                                     unless --host says otherwise. While run/setup-pending exists (the
#                                     installer writes it, the setup guide's apply job removes it) a
#                                     plain start is setup mode too, with the host the marker names.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
. "$HERE/engines.sh"
SETUP=0; SETUP_HOST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --setup) SETUP=1; shift ;;
        --host) [ $# -ge 2 ] || { echo "[start] --host needs an address"; exit 2; }
                SETUP_HOST="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "[start] unknown argument $1 (--setup, --host ADDR)"; exit 2 ;;
    esac
done
PENDING="$REPO/run/setup-pending"
if [ -f "$PENDING" ]; then
    SETUP=1
    [ -n "$SETUP_HOST" ] || SETUP_HOST="$(sed -n 's/^HOST=//p' "$PENDING" | head -1)"
fi
SETUP_HOST="${SETUP_HOST:-127.0.0.1}"
case "$SETUP_HOST" in
    *[!0-9.:a-fA-F]*|"") echo "[start] --host takes an address, not '$SETUP_HOST'"; exit 2 ;;
esac
LOGS="$REPO/logs"; mkdir -p "$LOGS"
PIDFILE="$LOGS/engine.pid"
LOG="$LOGS/engine-$(date +%Y%m%d-%H%M%S).log"

# OPS-155: chatd, the Chat tab's service, once the engine answers /health: started if it is not
# running, by the checkout's ops/chatd.sh (its own session, nothing inherited above stdio, serve.env,
# local.env and secrets.env exported, the admin token held back). Not in setup mode, not for a
# hold's restart (HOLD_RESTART=1), not with QSE_CHATD=off. A chatd that does not start is a warning:
# the engine is up either way. ops/stop.sh never stops it, so an engine restart, the watchdog's
# included, never ends a chat run; `tandem stop` does.
load_secrets() {
    SECRETS="${QSE_SECRETS:-${QSE_STATE_DIR:+$QSE_STATE_DIR/secrets.env}}"
    SECRETS="${SECRETS:-$HOME/.qwen38-spark-engine/secrets.env}"
    if [ -f "$SECRETS" ]; then set -a; . "$SECRETS"; set +a; fi
}
start_chatd() {
    [ "${QSE_CHATD:-on}" = "off" ] && { echo "[start] chatd: off (QSE_CHATD=off)"; return 0; }
    [ "${HOLD_RESTART:-0}" = "1" ] && return 0
    [ -f "$REPO/ops/chatd.sh" ] || return 0
    bash "$REPO/ops/chatd.sh" status >/dev/null 2>&1 && return 0
    bash "$REPO/ops/chatd.sh" start \
        || echo "[start] WARNING: chatd did not start (above, log $REPO/logs/chatd.log); the engine" \
                "serves without the Chat tab. tandem chat start tries again"
}

# A hold owns the board: the pause file is how it says so, and this script must respect it too.
# The @reboot cron line calls this directly (not through the watchdog, which already respects the
# pause), and on 2026-09-18 21:38 it started :8000 under a running hold -- a second engine beside
# the hold's own, which contaminated a row3 (the numbers read 3.5 % slower TTFT). The hold's own
# restart passes HOLD_RESTART=1, because that start is the point of the pause.
if [ -f "$REPO/.watchdog.off" ] && [ "${HOLD_RESTART:-0}" != "1" ]; then
    # A hold cannot survive a reboot, but its pause file can, and then the @reboot line refuses to
    # start the service for up to WATCHDOG_PAUSE_MAX with nothing holding anything. Older than the
    # longest hold means nobody is holding: the watchdog already ignores such a file, and this
    # removes it. A real refusal exits non-zero, so cron and the unit can see it.
    AGE=$(( $(date +%s) - $(stat -c %Y "$REPO/.watchdog.off") ))
    if [ "$AGE" -ge "${WATCHDOG_PAUSE_MAX:-2400}" ]; then
        echo "[start] pause file is ${AGE}s old, older than the longest hold; removing it"
        rm -f "$REPO/.watchdog.off"
    else
        echo "[start] a hold owns the board (.watchdog.off is armed, ${AGE}s); refusing to start"
        exit 1
    fi
fi
if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[start] already healthy on :$PORT"
    [ "$SETUP" = "1" ] || { load_secrets; start_chatd; }
    exit 0
fi
# One engine at a time, and a LOADING one counts: the server binds its port only once the
# weights are in, so for the minutes of a load neither /health nor `ss` sees it, and this script
# used to start a second engine beside it -- from the @reboot line, the watchdog or a hand. The same
# goes for an engine on another port (a row3 or probe server a signalled hold left behind, 2026-09-25
# 03:45): the watchdog would bring :8000 up beside it three minutes after the hold gave up.
LIVE="$(engine_pids)"
if [ -n "$LIVE" ]; then
    echo "[start] REFUSING: an engine is already alive (pid $(echo $LIVE)): loading, or on another" \
         "port; one engine at a time"
    exit 1
fi
BUSY="$(ss -ltnp 2>/dev/null | grep ":$PORT " || true)"
if [ -n "$BUSY" ]; then
    echo "[start] REFUSING: something else holds :$PORT and it is not answering /health:"
    echo "$BUSY"; exit 1
fi

cd "$REPO" || exit 1
# OPS-65: the status file (ops/engine_status.py) and, when serve.env names its port, the responder
# that answers for the engine while it loads (the reverse proxy's fallback)
STATUS_PY="${STATUS_PY:-python3}"
status() { "$STATUS_PY" "$HERE/engine_status.py" set "$@" >/dev/null 2>&1 || true; }
# (OPS-173: one of another tree or release is replaced; it listens where the engine or setup mode does)
[ -n "${QSE_STATUS_PORT:-}" ] && "$STATUS_PY" "$HERE/engine_status.py" ensure --port "$QSE_STATUS_PORT" \
    --engine-port "$PORT" --host "$( [ "$SETUP" = 1 ] && echo "$SETUP_HOST" || echo "${QSE_STATUS_HOST:-$HOST}")" 2>&1 | tail -1
# The access tokens, mode 600, never in this repository: QSE_METRICS_TOKEN (Prometheus) and, only
# when an admin password is wanted, QSE_ADMIN_TOKEN (ops/make-secrets.sh [--admin]). With no admin
# token every admin route is open and the dashboard asks for nothing (docs/server.md "The admin
# rule"). QSE_DASHBOARD_LOGIN reaches the server exported from serve.env (set -a above). Setup mode
# reads them too: its /v1/setup routes run apply and reset, and a password set must guard them.
load_secrets
T_START=$(date +%s)
if [ "$SETUP" = "1" ]; then
    # Setup mode: the server with no model of Qwen's, the dashboard and Wren alone (no caches, no
    # store, no drafters, no conductor, no ledger: nothing of serve.env's engine flags reaches it)
    status starting --reason "starting the setup guide"
    export PYTHONPATH TZ=Europe/Berlin QSE_OPS_DIR="$HERE" QSE_SETUP_MODE=1
    for fd in /proc/$$/fd/*; do
        fd=${fd##*/}
        case "$fd" in 0|1|2|255|*[!0-9]*) ;; *) eval "exec $fd>&-" 2>/dev/null ;; esac
    done
    setsid nohup "$PY" -u server/app.py --setup --host "$SETUP_HOST" --port "$PORT" \
        >"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    ln -sfn "$LOG" "$LOGS/engine.log"
    echo "[start] setup mode: pid $(cat "$PIDFILE"), log $LOG"
    for i in $(seq 1 60); do
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            # no --took-s: setup mode loads no model, and its 2 s would make the next model's
            # restart screen promise "any moment now" (the typical start is the median of these)
            status up --engine-port "$PORT"
            echo "[start] setup mode answers on $SETUP_HOST:$PORT after $(( $(date +%s) - T_START )) s:" \
                 "open http://<this box>:$PORT/dashboard/#/setup"
            exit 0
        fi
        sleep 2
    done
    status failed --reason "the setup guide did not answer /health within 120 s"
    echo "[start] setup mode did not answer /health in 120 s; tail of $LOG:"; tail -20 "$LOG"; exit 1
fi
status starting --reason "$( [ "${HOLD_RESTART:-0}" = "1" ] && echo "restarting after a hold" || echo "loading the model")"
# The usage ledger is this service's alone: the server does not read QSE_USAGE_LEDGER from
# its environment, so a benchmark server started with serve.env exported stays off. The same for
# the conductor's config (QSE_CONDUCTOR_CONFIG): `TandemLLM` exists on the :8000 service only. And
# for the disk tier (QSE_DISK_CACHE, C1): a row, gate or soak server never reads or writes it, so a
# benchmark row can never be a disk hit. And for the proxy's config database (QSE_BACKENDS).
export PYTHONPATH TZ=Europe/Berlin QSE_OPS_DIR="$HERE" QWEN38_FUSE_PROJ="$FUSE_PROJ" \
       QWEN38_DF2_TREE_MODE="$QWEN38_DF2_TREE_MODE" \
       QWEN38_TREE_ALIAS_STATE="$QWEN38_TREE_ALIAS_STATE"
LATCH_FLAG=""; [ "${LEN_LATCH:-0}" = "1" ] && LATCH_FLAG="--len-latch"
DROP_FLAG=""; [ "${DROP_IDLE:-0}" = "1" ] && DROP_FLAG="--drop-idle"
# the kill switch: LIVE_ACTIVITY=off in serve.env keeps the live view at contract 1.1 with the
# activity fields null (QSE_LIVE_HZ, exported from serve.env when set, is the busy cadence; 0 = off)
LIVE_FLAG=""; [ "${LIVE_ACTIVITY:-on}" = "off" ] && LIVE_FLAG="--live-activity off"
# The anti-repetition flags are passed only when serve.env sets them; empty means the
# server's own defaults, which are the identities (off) -- a run without them is byte-identical.
PEN_FLAGS=""
[ -n "${REP_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --rep-penalty $REP_PENALTY"
[ -n "${PRESENCE_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --presence-penalty $PRESENCE_PENALTY"
[ -n "${FREQUENCY_PENALTY:-}" ] && PEN_FLAGS="$PEN_FLAGS --frequency-penalty $FREQUENCY_PENALTY"
[ -n "${PATTERN_STOP:-}" ] && PEN_FLAGS="$PEN_FLAGS --pattern-stop $PATTERN_STOP"
# the conductor's session monitor (docs/server.md): raw, the default, runs none of it
CONDUCTOR_FLAGS="--conductor ${CONDUCTOR:-raw}"
[ -n "${CONDUCTOR_POLICY:-}" ] && CONDUCTOR_FLAGS="$CONDUCTOR_FLAGS --conductor-policy $CONDUCTOR_POLICY"
# OPS-47: the memory limits and SPD-75's narrow arm; a key serve.env does not set is the server's
# own default (the same numbers)
MEM_FLAGS=""
for kv in MEM_CEILING_GIB:--mem-ceiling-gib MEM_PEAK_GIB:--mem-peak-gib \
          RESERVE_SLACK_GIB:--reserve-slack-gib RESIDENT_GB:--resident-gb \
          RESIDENT_STASH_GB:--resident-stash-gb RESIDENT_GUEST_MAX:--resident-guest-max \
          RESIDENT_PIN_GB:--resident-pin-gb RESIDENT_PARK_GB:--resident-park-gb \
          STORE_ENTRY_GB:--store-entry-gb; do
    k="${kv%%:*}"; f="${kv#*:}"; v="${!k:-}"
    [ -n "$v" ] && MEM_FLAGS="$MEM_FLAGS $f $v"
done
[ "${NARROW_ARM:-1}" = "0" ] && MEM_FLAGS="$MEM_FLAGS --narrow-arm off"
# The resident prefix's end anchor, on unless serve.env says off
[ "${END_ANCHOR:-on}" = "off" ] && MEM_FLAGS="$MEM_FLAGS --end-anchor off"
# A copy whose whitespace is not the rows' resumes at the end anchor too
[ "${END_ALIGN:-on}" = "off" ] && MEM_FLAGS="$MEM_FLAGS --end-align off"
# Live conversations to the NVMe at a planned stop and before an eviction (needs the
# disk tier, QSE_DISK_CACHE); on unless serve.env says off
[ "${SESSION_SPILL:-on}" = "off" ] && MEM_FLAGS="$MEM_FLAGS --session-spill off"
# 0.4.3-test: the spill's budget and codec, and the day's NVMe write budget (unset: the server's
# defaults, 64 GB, zstd, 200 GB a day; settings.memory.session_spill / disk_writes override live)
[ -n "${SESSION_SPILL_GB:-}" ] && MEM_FLAGS="$MEM_FLAGS --session-spill-gb $SESSION_SPILL_GB"
[ -n "${SESSION_SPILL_CODEC:-}" ] && MEM_FLAGS="$MEM_FLAGS --session-spill-codec $SESSION_SPILL_CODEC"
[ -n "${DISK_WRITE_BUDGET_GB:-}" ] && MEM_FLAGS="$MEM_FLAGS --disk-write-budget-gb $DISK_WRITE_BUDGET_GB"
# The vision tower, on unless serve.env says off, and the response cache, off unless it says on
# (the setup guide's two switches for them, docs/setup.md)
[ "${VISION:-on}" = "off" ] && MEM_FLAGS="$MEM_FLAGS --vision off"
[ "${RESPONSE_CACHE:-off}" = "on" ] && MEM_FLAGS="$MEM_FLAGS --response-cache"
SAMPLE_FLAGS=""
[ -n "${TEMPERATURE:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --temperature $TEMPERATURE"
[ -n "${TOP_P:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --top-p $TOP_P"
[ -n "${TOP_K:-}" ] && SAMPLE_FLAGS="$SAMPLE_FLAGS --top-k $TOP_K"
# a sampled request's verify. det (or 1): the greedy request's tree, walked by drawing the
# target's token at each node; mixed: the sampled chain as the tree's spine, accepted against its q;
# unset: the q-aware chain.
case "${SAMPLED_TREE:-}" in
    1|det) SAMPLE_FLAGS="$SAMPLE_FLAGS --sampled-tree=det" ;;
    mixed) SAMPLE_FLAGS="$SAMPLE_FLAGS --sampled-tree=mixed" ;;
esac
# OPS-99: a profile of another model names its own model and drafter flags (QSE_ENGINE_FLAGS; a
# Flash-Next pack: --model <pack> --drafter mtp) in place of the 27B's, with the lookup's corpus (the
# tokenizer's, shared; ENG-229); unset, the 27B's, in their places
if [ -n "${QSE_ENGINE_FLAGS:-}" ]; then
    MODEL_HEAD=($QSE_ENGINE_FLAGS --corpus "$CORPUS"); MODEL_TAIL=()
else
    MODEL_HEAD=(--drafter lenrouter --len-fixed "$LEN_FIXED" $LATCH_FLAG $DROP_FLAG --dflash2-path greedy)
    MODEL_TAIL=(--tree --budget "$BUDGET" --corpus "$CORPUS"
                --dflash2-ckpt "$CKPT8" --dflash2-ckpt16 "$CKPT16"
                --nvfp4 "$NV" --fp8-head "$HEAD")
fi
# The engine inherits stdin/stdout/stderr and nothing else. Holds run as
# `flock ~/.qwen38-box.flock bash ops/hold.sh ...`, and flock hands its lock descriptor to the
# command unless told `-o`: it came down through hold.sh and this script into the restarted engine,
# which then held the box lock for its whole life, and the next hold waited on a server that does
# not exit. 255 is bash's own handle on this script, close-on-exec already.
for fd in /proc/$$/fd/*; do
    fd=${fd##*/}
    case "$fd" in 0|1|2|255|*[!0-9]*) ;; *) eval "exec $fd>&-" 2>/dev/null ;; esac
done
setsid nohup "$PY" -u server/app.py \
    --host "$HOST" --port "$PORT" --served-model "$SERVED_MODEL" \
    --max-len "$MAX_LEN" --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    --reasoning-format "$REASONING_FORMAT" --reasoning-effort "$REASONING_EFFORT" \
    "${MODEL_HEAD[@]}" \
    $PEN_FLAGS $SAMPLE_FLAGS $LIVE_FLAG $CONDUCTOR_FLAGS \
    ${MODEL_TAIL[@]+"${MODEL_TAIL[@]}"} --cache-budget-gb "$CACHE_GB" $MEM_FLAGS \
    --request-timeout "$REQUEST_TIMEOUT" --max-queue "$MAX_QUEUE" \
    --queue-timeout "$QUEUE_TIMEOUT" --verbose \
    --usage-ledger "${QSE_USAGE_LEDGER:-off}" \
    --conductor-config "${QSE_CONDUCTOR_CONFIG:-off}" --conductor-defaults shipped \
    --backends "${QSE_BACKENDS:-off}" \
    --disk-cache "${QSE_DISK_CACHE:-off}" \
    >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
ln -sfn "$LOG" "$LOGS/engine.log"
echo "[start] pid $(cat "$PIDFILE"), log $LOG"
# every 2 s (it was every 10 s: an install's job waited up to 10 s more for an engine that answered,
# VIS-211), 600 s in all
for i in $(seq 1 300); do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[start] healthy after $(( $(date +%s) - T_START ))s"
        status up --took-s "$(( $(date +%s) - T_START ))" --engine-port "$PORT"   # OPS-67: keeps the model list
        # the weights are on the device now and nothing reads their files
        # again, but their page cache is ~52 GB of the board, counted as available and not given
        # back fast enough: an 8,192-row prefill took MemFree to 1 GB and the driver logged
        # NV_ERR_NO_MEMORY; with the cache dropped the same request left 53 GB free.
        if [ "${DROP_PAGE_CACHE:-0}" = "1" ]; then "$PY" tools/drop_page_cache.py 2>&1 | tail -1; fi
        # then the roles in ROLES_AUTOLOAD, one at a time through the residency planner (OPS-41);
        # a refusal leaves the engine serving and says why
        ROLES_SERVICE_START=1 bash "$HERE/roles.sh" up || echo "[start] a role did not load (above)"
        # then the decision models the conductor's `deciders` document loads on start (ENG-179):
        # the engine loads them itself, in the background, one at a time through the residency
        # planner; a refusal is journaled and chat is served either way. /readyz says when they are warm
        if [ -n "${QWEN38_DECIDER_MODELS:-}" ]; then
            HR=false; [ "${HOLD_RESTART:-0}" = "1" ] && HR=true
            curl -sf -m 10 -X POST -H 'Content-Type: application/json' \
                -d "{\"service_start\": true, \"hold_restart\": $HR}" \
                "http://127.0.0.1:$PORT/v1/conductor/deciders/autoload" >/dev/null \
                && echo "[start] decision models: autoload asked" \
                || echo "[start] decision models: the autoload did not answer (conductor off?)"
        fi
        # v3: the narrator's model (the live lines) loads after the decision models (one load at a
        # time), in the start's idle window, through the residency planner; the server's keeper
        # loads it again whenever it is not loaded. 0.11.3: the role is off by default, and the
        # engine answers 409 role_off without loading it
        if [ -n "${QWEN38_NARRATOR_MODEL:-}" ]; then
            HR=false; [ "${HOLD_RESTART:-0}" = "1" ] && HR=true
            NC="$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
                -d "{\"service_start\": true, \"hold_restart\": $HR}" \
                "http://127.0.0.1:$PORT/v1/conductor/narrator/load" || true)"
            case "$NC" in
                2*) echo "[start] narrator: load asked" ;;
                409) echo "[start] narrator: not loaded (the narrator role is off, or the planner refused it; the engine log says which)" ;;
                *) echo "[start] narrator: the load did not answer (conductor off?)" ;;
            esac
        fi
        start_chatd
        exit 0
    fi
    sleep 2
done
status failed --reason "the engine did not answer /health within 600 s"
echo "[start] FAILED to become healthy in 600s; tail of $LOG:"; tail -20 "$LOG"; exit 1
