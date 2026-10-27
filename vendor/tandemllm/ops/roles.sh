#!/bin/bash
# The roles beside the engine (OPS-41): load them one at a time, through the residency planner.
#
#   bash ops/roles.sh load ROLE [-- COMMAND...]   load one role (its command from serve.env if none)
#   bash ops/roles.sh up                          the roles in ROLES_AUTOLOAD, in the planner's order
#   bash ops/roles.sh down                        stop every role this script started
#   bash ops/roles.sh settle                      the seconds to wait after a stop (the planner's)
#
# The order on this board is fixed: the engine first (start.sh calls `up` once /health answers and
# the page cache is dropped), then each role, and never two loads at once -- two large loads at the
# same time wedged the box on 2026-09-11. A load is refused, with the planner's reason journaled
# in <conductor root>/residency.jsonl, when
#
#   refused_hold              a hold owns the board: its pause (.watchdog.off) is armed, or the box
#                             lock is held, and this is not the hold's own restart (HOLD_RESTART=1)
#   refused_load_in_progress  another role is loading (logs/roles/.loading), or the engine is
#                             loading (an engine process and no answer on /health)
#   refused_memory            MemAvailable (`free`) minus the engine's growth to its declared peak
#                             minus the role's claim is under floor + margin (OPS-47: the peak and
#                             what the engine holds now come from its /health; without an answer
#                             the whole MEM_PEAK_GIB of serve.env counts)
#
# A role's command, health URL and runtime come from serve.env: ROLE_<NAME>_START,
# ROLE_<NAME>_HEALTH, ROLE_<NAME>_RUNTIME (vllm or empty), ROLE_<NAME>_WAIT (s, default 300); the
# claims, floor, margin, settle time and order are the conductor's `residency` document (the
# dashboard edits it). A vLLM role runs BF16 weights with the KV cache unquantised, and its memory
# fraction is the planner's byte-math (claim / board total, rounded down), never vLLM's default
# 0.9: a command that sets --dtype other than bfloat16, --quantization, a --kv-cache-dtype other
# than auto, or its own --gpu-memory-utilization is refused. Roles never load on the Mac.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
. "$HERE/engines.sh"
RDIR="$REPO/logs/roles"; mkdir -p "$RDIR"
BOX_LOCK="${BOX_LOCK:-$HOME/.qwen38-box.flock}"
ROOT="${QSE_CONDUCTOR_ROOT:-$HOME/.qwen38-spark-engine/conductor}"

planner() { ( cd "$REPO" && PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m server.conductor.residency --root "$ROOT" "$@" ); }
# MemAvailable and MemTotal in GiB, from `free -m` ("available" is the 7th field of Mem:)
mem_gib() { free -m | LC_ALL=C awk -v c="$1" '/^Mem:/ { printf "%.2f\n", $c / 1024 }'; }
upper() { echo "$1" | tr '[:lower:]-' '[:upper:]_'; }
# the engine's declared peak and what it holds now, from /health (OPS-47): "--engine-peak-gib X
# --engine-now-gib Y", or nothing when the engine does not answer (the planner then counts the whole
# MEM_PEAK_GIB of serve.env)
engine_mem() {
    curl -sf -m 3 "http://127.0.0.1:${PORT:-8000}/health" 2>/dev/null | "$PY" -c '
import json, sys
try:
    m = json.load(sys.stdin).get("memory") or {}
except ValueError:
    sys.exit(0)
if m.get("peak_gb") is not None and m.get("engine_reserved_gb") is not None:
    print("--engine-peak-gib %s --engine-now-gib %s" % (m["peak_gb"], m["engine_reserved_gb"]))
' 2>/dev/null
}

held() {
    [ "${HOLD_RESTART:-0}" = "1" ] && return 1
    [ -f "$REPO/.watchdog.off" ] && return 0
    # start.sh's own `up`: the watchdog restarts the service holding the box lock itself
    [ "${ROLES_SERVICE_START:-0}" = "1" ] && return 1
    if [ -e "$BOX_LOCK" ] && command -v flock >/dev/null 2>&1; then
        flock -n "$BOX_LOCK" true 2>/dev/null || return 0
    fi
    return 1
}

engine_loading() {
    [ -n "$(engine_pids)" ] && ! curl -sf -m 3 "http://127.0.0.1:${PORT:-8000}/health" >/dev/null 2>&1
}

load() {
    local role="$1"; shift
    [ "${1:-}" = "--" ] && shift
    if [ "$(uname -s)" = "Darwin" ] && [ "${ROLES_ALLOW_DARWIN:-0}" != "1" ]; then
        echo "[roles] REFUSING: roles never load on the Mac"; return 4
    fi
    local U; U="$(upper "$role")"
    local start_var="ROLE_${U}_START" health_var="ROLE_${U}_HEALTH" rt_var="ROLE_${U}_RUNTIME" wait_var="ROLE_${U}_WAIT"
    local -a cmd
    if [ $# -gt 0 ]; then cmd=("$@"); else eval "cmd=(${!start_var:-})"; fi
    [ ${#cmd[@]} -gt 0 ] || { echo "[roles] no command for $role (set $start_var in serve.env)"; return 2; }
    if [ "${!rt_var:-}" = "vllm" ]; then
        local a
        for a in "${cmd[@]}"; do
            case "$a" in
                --quantization*|--gpu-memory-utilization*|--kv-cache-dtype=fp8*|--dtype=*)
                    if [ "$a" != "--dtype=bfloat16" ]; then echo "[roles] REFUSING $role: $a (BF16 weights, KV unquantised, the planner's fraction)"; return 5; fi ;;
            esac
        done
        local i
        for ((i = 0; i < ${#cmd[@]}; i++)); do
            case "${cmd[$i]}" in
                --dtype|--kv-cache-dtype)
                    local v="${cmd[$((i + 1))]:-}"
                    if { [ "${cmd[$i]}" = "--dtype" ] && [ "$v" != "bfloat16" ]; } || { [ "${cmd[$i]}" = "--kv-cache-dtype" ] && [ "$v" != "auto" ]; }; then
                        echo "[roles] REFUSING $role: ${cmd[$i]} $v (BF16 weights, KV unquantised)"; return 5
                    fi ;;
            esac
        done
    fi
    local flags=() lock_ok=0
    held && flags+=(--hold)
    if mkdir "$RDIR/.loading" 2>/dev/null; then
        lock_ok=1; echo "$$ $role" > "$RDIR/.loading/owner"
    else
        local owner; owner="$(awk '{print $1}' "$RDIR/.loading/owner" 2>/dev/null)"
        if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
            # a loader that died left its mark: take it over
            rm -rf "$RDIR/.loading"; mkdir "$RDIR/.loading" && lock_ok=1 && echo "$$ $role" > "$RDIR/.loading/owner"
        fi
    fi
    [ "$lock_ok" = 1 ] || flags+=(--loading)
    engine_loading && flags+=(--loading)
    release() { [ "$lock_ok" = 1 ] && rm -rf "$RDIR/.loading"; }
    local before; before="$(mem_gib 7)"
    local out rc
    out="$(planner check "$role" --mem-available-gib "$before" --source ops $(engine_mem) ${flags[@]+"${flags[@]}"})"; rc=$?
    echo "[roles] $out"
    if [ "$rc" != 0 ]; then release; echo "[roles] $role refused"; return 3; fi
    if [ "${!rt_var:-}" = "vllm" ]; then
        local frac; frac="$(planner fraction "$role" --total-gib "$(mem_gib 2)")" || { release; return 5; }
        cmd+=(--dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization "$frac")
    fi
    local log="$RDIR/$role-$(date +%Y%m%d-%H%M%S).log"
    echo "[roles] loading $role: ${cmd[*]}"
    ( for fd in $(seq 3 254); do eval "exec $fd>&-"; done 2>/dev/null
      exec setsid nohup "${cmd[@]}" >"$log" 2>&1 < /dev/null ) &
    local pid=$!
    echo "$pid" > "$RDIR/$role.pid"
    local url="${!health_var:-}" waited=0 limit="${!wait_var:-300}"
    if [ -n "$url" ]; then
        until curl -sf -m 3 "$url" >/dev/null 2>&1; do
            if ! kill -0 "$pid" 2>/dev/null || [ "$waited" -ge "$limit" ]; then
                echo "[roles] $role did not become healthy in ${waited}s; stopping it"; tail -5 "$log"
                stop_one "$role"; release; return 1
            fi
            sleep 1; waited=$((waited + 1))
        done
    fi
    local after; after="$(mem_gib 7)"
    planner loaded "$role" --before-gib "$before" --after-gib "$after" --source ops >/dev/null
    echo "[roles] $role up after ${waited}s (pid $pid); MemAvailable $before -> $after GiB"
    release
}

stop_one() {
    local role="$1" pid
    pid="$(cat "$RDIR/$role.pid" 2>/dev/null)" || return 0
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        echo "[roles] stopped $role (pid $pid)"
    fi
    rm -f "$RDIR/$role.pid"
}

case "${1:-}" in
    load) shift; [ $# -gt 0 ] || { echo "usage: roles.sh load ROLE [-- COMMAND...]"; exit 2; }; load "$@"; exit $? ;;
    up)
        [ -n "${ROLES_AUTOLOAD:-}" ] || { echo "[roles] none to load (ROLES_AUTOLOAD is empty)"; exit 0; }
        ORDER="$(planner plan $ROLES_AUTOLOAD | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)["order"]))')"
        RC=0
        for R in $ORDER; do load "$R" || RC=$?; done
        exit $RC ;;
    down)
        for f in "$RDIR"/*.pid; do [ -e "$f" ] || continue; R="$(basename "$f" .pid)"; stop_one "$R"; done
        exit 0 ;;
    settle)
        S="$(planner plan 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["settle_s"])' 2>/dev/null)"
        echo "${S:-5}"; exit 0 ;;
    *) echo "usage: roles.sh load ROLE [-- COMMAND...] | up | down | settle"; exit 2 ;;
esac
