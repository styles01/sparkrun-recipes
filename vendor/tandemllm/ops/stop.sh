#!/bin/bash
# Graceful stop: SIGTERM makes the server drain (new requests get 503 + Retry-After, the
# generation in flight finishes), and only a server that will not go after the grace period is
# killed. A hard kill mid-verify truncates whoever is streaming.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HERE/serve.env"; set +a
. "$HERE/engines.sh"
GRACE="${1:-60}"
# ALL of them, not the first. `head -1` was here until 2026-09-18, and the day two supervisors
# raced -- cron's watchdog and a hold's own restore -- it stopped one engine, reported success, and
# left the other loading; the next thing to look at :8000 saw nothing answering and concluded the
# port was free. A stop that leaves a process behind is worse than one that fails.
# Engines only: a pattern over command lines took a lock holder that mentioned the path for
# one, and SIGTERM to that flock freed the box lock in the middle of a hold.
PIDS="$(engine_pids "$PORT")"
[ -z "$PIDS" ] && { echo "[stop] nothing on :$PORT"; exit 0; }
# OPS-65: the status file says so (a hold has written `held` already and keeps it)
status() { [ "${QSE_STATUS_KEEP:-0}" = "1" ] || "${STATUS_PY:-python3}" "$HERE/engine_status.py" set "$@" \
    >/dev/null 2>&1 || true; }
status stopping --reason "${STATUS_REASON:-a stop}"
trap 'status down --reason "${STATUS_REASON:-a stop}"' EXIT
echo "[stop] SIGTERM $(echo $PIDS | tr '\n' ' '), up to ${GRACE}s to drain"
for P in $PIDS; do kill -TERM "$P" 2>/dev/null; done
for _ in $(seq 1 "$GRACE"); do
    LEFT=""; for P in $PIDS; do kill -0 "$P" 2>/dev/null && LEFT="$LEFT $P"; done
    [ -z "$LEFT" ] && { echo "[stop] stopped"; exit 0; }
    sleep 1
done
echo "[stop] still alive after ${GRACE}s, SIGKILL:$LEFT"
for P in $LEFT; do kill -KILL "$P" 2>/dev/null; done
sleep 2
echo "[stop] killed"
