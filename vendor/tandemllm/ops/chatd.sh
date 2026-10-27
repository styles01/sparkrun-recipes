#!/bin/bash
# chatd, the Chat tab's service (python -m chatd, chatd/__init__.py):
#
#   bash ops/chatd.sh start [--front 0.0.0.0:4180]   start it in the background (QSE_CHAT_URL)
#   bash ops/chatd.sh stop                          SIGTERM: open runs end `interrupted` (Continue)
#   bash ops/chatd.sh restart                       stop, then start
#   bash ops/chatd.sh status                        pid, interpreter, data, URL and GET /health
#
# ops/start.sh starts it once the engine answers /health, `tandem stop` stops it first and
# `tandem chat start|stop|restart|status|logs` calls this script (OPS-155). Its own process, never
# the engine's: ops/stop.sh, an engine restart and a hold leave it alone, and it does not touch the
# engine (ops/engines.sh does not count it). It stays in the checkout (DIR/src/ops in an install):
# the `chatd` package is found from here.
#
# The interpreter, first match: CHATD_PY; PY (serve.env's, exported by start.sh and bin/tandem);
# the install's venv beside this checkout (../venv/bin/python); ./venv/bin/python;
# $VIRTUAL_ENV/bin/python; python3, said so with its version. chatd needs only the standard library;
# the MCP servers' stored secrets need the venv's `cryptography`, and pyproject.toml asks for 3.11.
#
# The environment: run by hand in an installed tree (no serve.env in the environment, REPO unset),
# DIR/run/ops/serve.env and its local.env are read as start.sh reads them; then QSE_CHATD_ENV, else
# run/chatd.env of the install or of this checkout, when there is one. The forward key,
# QSE_CHAT_FORWARD_KEY, comes from the secrets file the engine reads (QSE_SECRETS, else
# $QSE_STATE_DIR/secrets.env, else ~/.qwen38-spark-engine/secrets.env, start.sh's order); with no
# key there chatd makes <data>/forward.key, which the engine reads too. The admin and metrics
# tokens never reach chatd. Address from QSE_CHAT_URL (http://127.0.0.1:8016), the engine from
# QSE_CHAT_ENGINE (else http://127.0.0.1:$PORT), data in QSE_CHAT_DATA (~/.tandemllm/chat, the pid
# file there too), the log in logs/chatd.log (ids and counts, never message text).
# `--front HOST:PORT` also serves the dashboard and the Chat tab from this process, for an engine
# whose server has no /v1/chat-app forward yet (chatd/front.py).
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
UP="$(cd "$HERE/.." && pwd)"                 # an install's DIR when this is DIR/src
if [ -z "${REPO:-}" ] && [ -f "$UP/.tandemllm-install" ] && [ -f "$UP/run/ops/serve.env" ]; then
    set +u; set -a; . "$UP/run/ops/serve.env"; set +a; set -u
fi
CHATD_ENV="${QSE_CHATD_ENV:-$HERE/run/chatd.env}"
[ -n "${QSE_CHATD_ENV:-}" ] || [ ! -f "$UP/.tandemllm-install" ] || CHATD_ENV="$UP/run/chatd.env"
if [ -f "$CHATD_ENV" ]; then set +u; set -a; . "$CHATD_ENV"; set +a; set -u; fi
DATA="${QSE_CHAT_DATA:-$HOME/.tandemllm/chat}"
PIDF="$DATA/chatd.pid"
LOG="$HERE/logs/chatd.log"
URL="${QSE_CHAT_URL:-http://127.0.0.1:8016}"
ADDR="${URL#*://}"; ADDR="${ADDR%%/*}"
case "$ADDR" in *:*) CHOST="${ADDR%:*}"; CPORT="${ADDR##*:}" ;; *) CHOST="$ADDR"; CPORT=8016 ;; esac
case "$CPORT" in ''|*[!0-9]*) echo "[chatd] QSE_CHAT_URL=$URL has no port number"; exit 2 ;; esac
HEALTH="http://$CHOST:$CPORT/health"
SECRETS="${QSE_SECRETS:-${QSE_STATE_DIR:+$QSE_STATE_DIR/secrets.env}}"
SECRETS="${SECRETS:-$HOME/.qwen38-spark-engine/secrets.env}"
cmd="${1:-status}"; shift || true

health() { curl -s -m "${1:-1}" "$HEALTH" 2>/dev/null || true; }
is_chatd_health() { case "$1" in *'"service": "chatd"'*|*'"service":"chatd"'*) return 0 ;; esac; return 1; }
# a pid is chatd when it lives and its command line runs `-m chatd`: a pid file whose pid was
# reused by another process is not a running chatd, and stop never signals that process
cmdline() {
    if [ -r "/proc/$1/cmdline" ]; then tr '\0' ' ' < "/proc/$1/cmdline"; else ps -o args= -p "$1" 2>/dev/null || true; fi
}
is_chatd() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null || return 1
    case " $(cmdline "$1") " in *" -m chatd "*) return 0 ;; esac
    return 1
}
pid_of() { [ -f "$PIDF" ] && tr -dc '0-9' < "$PIDF" || true; }
alive() { is_chatd "$(pid_of)"; }
version_of() { "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null; }

# PYBIN and PYVER: the interpreter that runs chatd, in the order above
pick_python() {
    local c v
    if [ -n "${CHATD_PY:-}" ]; then
        v="$(version_of "$CHATD_PY")" || { echo "[chatd] CHATD_PY=$CHATD_PY does not run"; return 1; }
        PYBIN="$CHATD_PY"; PYVER="$v"; return 0
    fi
    for c in "${PY:-}" "$UP/venv/bin/python" "$HERE/venv/bin/python" \
             "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"; do
        [ -n "$c" ] && [ -e "$c" ] || continue
        if [ -f "$c" ] && [ -x "$c" ] && v="$(version_of "$c")"; then
            PYBIN="$c"; PYVER="$v"; return 0
        fi
        echo "[chatd] $c does not run; trying the next interpreter"
    done
    PYBIN="$(command -v python3 || true)"
    [ -n "$PYBIN" ] && PYVER="$(version_of "$PYBIN")" || { echo "[chatd] no venv and no python3 found"; return 1; }
    echo "[chatd] no venv found: using the system python3 ($PYBIN, $PYVER)"
    case "$PYVER" in 3.1[1-9].*|3.[2-9][0-9].*) ;;
        *) echo "[chatd] WARNING: Python $PYVER is older than 3.11, the version TandemLLM is tested with" ;;
    esac
}

forward_key() {
    local k
    [ -f "$SECRETS" ] || return 0
    k="$(grep -E '^[[:space:]]*(export[[:space:]]+)?QSE_CHAT_FORWARD_KEY=' "$SECRETS" | tail -1 | cut -d= -f2- | tr -d '\r' || true)"
    k="${k#[\"\']}"; k="${k%[\"\']}"
    if [ -n "$k" ]; then export QSE_CHAT_FORWARD_KEY="$k"; fi
}

show() {   # the running chatd, for start and status
    local pid h exe tree
    pid="$(pid_of)"
    h="$(health 2)"
    exe="$(cmdline "$pid" | awk '{print $1}')"
    tree="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    echo "[chatd] running: pid $pid, $URL, chatd $(printf '%s' "$h" | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p')"
    echo "[chatd] python $exe ($(version_of "$exe" || echo '?')), data $DATA${tree:+, tree $tree}"
    echo "[chatd] health ${h:-(no answer)}"
}

start() {
    if alive; then echo "[chatd] already running"; show; return 0; fi
    rm -f "$PIDF"
    local h
    h="$(health 1)"
    if [ -n "$h" ]; then
        if is_chatd_health "$h"; then
            echo "[chatd] a chatd this data folder does not know answers on $URL (another install?): $h"
        else
            echo "[chatd] port $CPORT is taken by another program (QSE_CHAT_URL=$URL)"
        fi
        return 1
    fi
    mkdir -p "$DATA" "$HERE/logs" 2>/dev/null && [ -w "$DATA" ] \
        || { echo "[chatd] cannot write the data folder $DATA (QSE_CHAT_DATA)"; return 1; }
    pick_python || return 1
    forward_key
    export QSE_CHAT_ENGINE="${QSE_CHAT_ENGINE:-http://127.0.0.1:${PORT:-8000}}"
    FRONT=()
    if [ "${1:-}" = "--front" ]; then FRONT=(--front "$2" --dashboard "$HERE/dashboard/dist"); fi
    cd "$HERE"
    # as start.sh launches the engine: its own session, nothing inherited above stdio (a hold's
    # box lock came down to the engine that way once)
    for fd in /proc/$$/fd/* /dev/fd/*; do
        fd=${fd##*/}
        case "$fd" in 0|1|2|255|*[!0-9]*) ;; *) eval "exec $fd>&-" 2>/dev/null || true ;; esac
    done
    SETSID=(); command -v setsid >/dev/null 2>&1 && SETSID=(setsid)
    ${SETSID[@]+"${SETSID[@]}"} nohup env -u QSE_ADMIN_TOKEN -u QSE_METRICS_TOKEN PYTHONPATH="$HERE" \
        "$PYBIN" -u -m chatd --host "$CHOST" --port "$CPORT" --data "$DATA" ${FRONT[@]+"${FRONT[@]}"} \
        >> "$LOG" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$PIDF"
    for _ in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || break
        is_chatd_health "$(health 1)" && { echo "[chatd] started with $PYBIN ($PYVER)"; show; return 0; }
        sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "[chatd] FAILED: pid $pid did not answer $HEALTH within 10 s; stopped it"
        kill -TERM "$pid" 2>/dev/null || true
    else
        echo "[chatd] FAILED: chatd exited at start ($PYBIN, port $CPORT)"
    fi
    rm -f "$PIDF"
    echo "[chatd] last lines of $LOG:"
    tail -20 "$LOG" 2>/dev/null | sed 's/^/    /'
    return 1
}

stop() {
    local pid
    pid="$(pid_of)"
    if ! is_chatd "$pid"; then
        [ -f "$PIDF" ] && echo "[chatd] not running (a stale pid file, removed)" || echo "[chatd] not running"
        rm -f "$PIDF"; return 0
    fi
    kill -TERM "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    rm -f "$PIDF"
    echo "[chatd] stopped"
}

case "$cmd" in
    start) start "$@" ;;
    stop) stop ;;
    restart) stop; start "$@" ;;
    status)
        if alive; then show; exit 0; fi
        echo "[chatd] not running ($URL, data $DATA)"
        h="$(health 1)"
        if is_chatd_health "$h"; then echo "[chatd] a chatd of another tree answers on $URL: $h"; fi
        exit 1
        ;;
    *) echo "usage: $0 start [--front HOST:PORT] | stop | restart | status"; exit 2 ;;
esac
