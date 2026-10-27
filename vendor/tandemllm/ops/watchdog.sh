#!/bin/bash
# One health check. Meant for cron every minute, or a loop; it restarts only what it is sure about.
#
# Three states and only one of them is a restart:
#   healthy   -> nothing
#   draining  -> nothing. A graceful stop is in progress and killing it is the bug, not the fix.
#   silent    -> restart, after CONSECUTIVE failures, so one slow health check during a long
#                verify does not bounce a working server.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
. "$HERE/engines.sh"
LOGS="$REPO/logs"; mkdir -p "$LOGS"
STATE="$LOGS/watchdog.fails"; WD="$LOGS/watchdog.log"
NEED="${WATCHDOG_FAILS:-3}"
say() { echo "$(date -Is) $*" >> "$WD"; }
# OPS-155: the engine answers and chatd does not: start chatd, never the engine's restart for it. At
# most 3 starts in an hour (logs/watchdog.chatd: the hour and the count), then one "giving up"
# line until the next hour; a chatd found running clears the count. QSE_CHATD=off or setup mode
# (run/setup-pending): nothing.
chatd_check() {
    local f="$LOGS/watchdog.chatd" hour h n
    [ "${QSE_CHATD:-on}" = "off" ] && return 0
    [ -f "$REPO/ops/chatd.sh" ] && [ ! -f "$REPO/run/setup-pending" ] || return 0
    if bash "$REPO/ops/chatd.sh" status >/dev/null 2>&1; then rm -f "$f"; return 0; fi
    hour="$(date +%Y%m%d%H)"
    h=""; n=0; [ -f "$f" ] && read -r h n < "$f"
    [ "${h:-}" = "$hour" ] || n=0
    n="${n:-0}"
    if [ "$n" -ge 3 ]; then
        [ "$n" = 3 ] && { say "chatd: 3 starts this hour did not keep it running; giving up until the next hour"; echo "$hour 4" > "$f"; }
        return 0
    fi
    echo "$hour $((n + 1))" > "$f"
    if bash "$REPO/ops/chatd.sh" start >> "$WD" 2>&1; then
        say "chatd was not running; started it ($((n + 1))/3 this hour)"
    else
        say "chatd was not running and did not start ($((n + 1))/3 this hour); log: $REPO/logs/chatd.log"
    fi
}

# A fourth state, added 2026-09-18: PAUSED. A measurement takes the board lock and drains :8000 on
# purpose, and a supervisor that cannot tell a deliberate stop from a crash restarts the service
# under the measurement -- which happened, and put a second full engine on a bandwidth-bound board
# for three atlas rows. The pause file is how a hold says "this
# silence is mine"; it carries who and why, and the hold removes it in its EXIT trap so a crashed
# hold cannot leave the service unsupervised for long.
PAUSE="$REPO/.watchdog.off"
if [ -f "$PAUSE" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$PAUSE") ))
    if [ "$AGE" -lt "${WATCHDOG_PAUSE_MAX:-2400}" ]; then exit 0; fi
    say "pause file is ${AGE}s old, older than the longest hold; ignoring it"
fi

# curl prints 000 itself when nothing answers, and exits non-zero: an `|| echo 000` after it made the
# code "000000" in every log line of 2026-09-19.
CODE="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
CODE="${CODE:-000}"
if [ "$CODE" = "200" ]; then echo 0 > "$STATE"; rm -f "$LOGS/loading.since"; chatd_check; exit 0; fi
if [ "$CODE" = "503" ]; then say "draining (503), leaving it alone"; echo 0 > "$STATE"; exit 0; fi
# The box lock. Every hold runs as `flock ~/.qwen38-box.flock ops/hold.sh...`, and a held
# lock means somebody owns the board: the silence is theirs, as with the pause file, which a hold
# arms only after it has the lock and removes just before it lets go. Taken here, not before the
# health check, so a healthy minute never makes an agent's flock wait; and kept for the rest of this
# run, so a hold cannot begin between this check and the restart below (start.sh gives the engine
# nothing above stdio, so the restarted service does not inherit it --).
BOX_LOCK="${BOX_LOCK:-$HOME/.qwen38-box.flock}"
if [ -e "$BOX_LOCK" ] && command -v flock >/dev/null 2>&1; then
    exec 9<"$BOX_LOCK"
    if ! flock -n 9; then
        say "box lock held (code $CODE), leaving it alone"
        echo 0 > "$STATE"
        exit 0
    fi
fi
# A process that is LOADING answers nothing and is not absent (the 2026-09-18 entry in the ledger,
# relearned at every cold boot: the load reads 29 GB from cold disk and outruns the three-strike
# window; the old behaviour killed the loader and started another while the first one's memory was
# still draining -- a second full load is how this box wedges). So: if a server process exists,
# give it time; after LOADING_MAX minutes of consecutive failure, restart for real.
PID="$(engine_pids "$PORT" | head -1)"                 # an engine, not a look-alike
if [ -n "$PID" ]; then
    # The timestamp belongs to THIS loader, so it carries the pid it was taken for. It used to be
    # removed only on a 200: a loader that died before it ever answered left its start time behind,
    # the next one inherited an age already past LOADING_MAX, and three strikes later the watchdog
    # restarted a server in the middle of its load -- the second full load is how this box wedges.
    SINCE="$(cat "$LOGS/loading.since" 2>/dev/null || true)"
    case "$SINCE" in
        "$PID "*) ;;                                   # same loader: keep its first attempt
        *) SINCE="$PID $(date +%s)"; echo "$SINCE" > "$LOGS/loading.since";;
    esac
    AGE=$(( $(date +%s) - ${SINCE#* } ))
    if [ "$AGE" -lt "${LOADING_MAX:-900}" ]; then
        say "loading (code $CODE, ${AGE}s), leaving it alone"
        echo 0 > "$STATE"
        exit 0
    fi
    say "still not answering after ${AGE}s, treating as hung"
fi
N=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$STATE"
say "health check failed (code $CODE), $N/$NEED"
[ "$N" -lt "$NEED" ] && exit 0
say "restarting"
rm -f "$LOGS/loading.since"                            # the next loader times its own load
bash "$HERE/stop.sh" 30 >> "$WD" 2>&1
sleep "${RESTART_SETTLE_S:-5}"                          # the kernel's memory back first (OPS-41)
bash "$HERE/start.sh" >> "$WD" 2>&1 && say "restarted" || say "RESTART FAILED"
echo 0 > "$STATE"
