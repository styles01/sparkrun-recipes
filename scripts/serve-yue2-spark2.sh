#!/usr/bin/env bash
# ============================================================================
# serve-yue2-spark2.sh — CANONICAL launcher for the YuE2-3B music lane (:8045).
# Runs on LARRYSPARK2 (192.168.2.14). Sibling of serve-music3-spark2.sh (:8050).
#
#   ./serve-yue2-spark2.sh status     # health + queue + pid
#   ./serve-yue2-spark2.sh start      # start the HTTP job server (no-op if up)
#   ./serve-yue2-spark2.sh drain      # block until queue==0 and running==0
#   ./serve-yue2-spark2.sh restart    # refuse if busy, then swap in the .py + start
#   ./serve-yue2-spark2.sh stop       # refuse while a render runs (append: force)
#
# Lane facts (LANE-REGISTRY.yaml is truth): subprocess-per-render (zero VRAM idle),
# 14GiB VRAM budget cap, 16GiB MemAvailable gate, concurrency 1. Weights
# /home/jaita/models/yue2 (7.3GB). NEVER fire concurrently with a MiniMax Music 3
# render on :8050.
#
# WHY A PIDFILE: jobs.json/STATE_DIR are shared by every instance of this lane, so
# a second instance would run a second worker against the same queue and render two
# songs at once. Never pattern-kill by script path — kill the recorded pid only.
# ============================================================================
set -euo pipefail

YUE2_DIR="/home/jaita/yue2"
SERVER="$YUE2_DIR/yue2-server.py"
PY="/home/jaita/yue2/.venv/bin/python"
LOG="$YUE2_DIR/server.log"
PORT="${YUE2_PORT:-8045}"
PIDFILE="${YUE2_PIDFILE:-$YUE2_DIR/server.pid}"
HEALTH="http://127.0.0.1:$PORT/health"

up()    { curl -s -m 3 "$HEALTH" >/dev/null 2>&1; }
field() { curl -s -m 5 "$HEALTH" | "$PY" -c "import json,sys;print(json.load(sys.stdin).get('$1'))"; }
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

kill_recorded() {
  # TRASH-ONLY law: never rm. Mark the pidfile dead instead of deleting it.
  if alive; then kill "$(cat "$PIDFILE")" && echo dead > "$PIDFILE"; sleep 2; fi
}

case "${1:-status}" in
  status)
    echo "== yue2 lane :$PORT on $(hostname) =="
    if up; then curl -s -m 5 "$HEALTH" | "$PY" -c "
import json,sys
h=json.load(sys.stdin)
print(f\"  UP  queue={h['queue']} running={h['running']} done={h['done']} failed={h['failed']} \"
      f\"done_truncated={h.get('done_truncated')} mem={h['mem_available_gib']}G budget={h['budget_gib']}G\")
print('  last_error:', h['last_error'])
print('  knobs:', ', '.join((h.get('knobs') or {}).keys()) or 'PRE-KNOBS BUILD')"
      echo "  pidfile: $( [ -f "$PIDFILE" ] && cat "$PIDFILE" || echo none )"
    else echo "  DOWN (start: $0 start)"; fi
    ;;

  drain)
    # Wait for the lane to be idle so a code swap never kills a render in flight.
    while :; do
      up || { echo "lane is down"; exit 0; }
      q=$(field queue); r=$(field running)
      [ "$q" = "0" ] && [ "$r" = "0" ] && { echo "idle (queue=0 running=0)"; exit 0; }
      echo "busy: queue=$q running=$r — waiting 20s"; sleep 20
    done
    ;;

  start)
    up && { echo "already up on :$PORT"; exit 0; }
    "$PY" -m py_compile "$SERVER"
    cd "$YUE2_DIR"
    # exec via the interpreter: never depends on the file's exec bit
    setsid nohup "$PY" "$SERVER" >> "$LOG" 2>&1 < /dev/null & echo $! > "$PIDFILE"
    for _ in $(seq 1 20); do up && break; sleep 1; done
    up && { echo "started on :$PORT (pid $(cat "$PIDFILE"))"; exit 0; }
    echo "FAILED to bind :$PORT; tail -30 $LOG"; exit 1
    ;;

  restart)
    if up && { [ "$(field running)" != "0" ] || [ "$(field queue)" != "0" ]; }; then
      echo "refusing to restart with work in flight — run '$0 drain' first"; exit 1
    fi
    kill_recorded
    "$0" start
    ;;

  stop)
    if [ "${2:-}" != "force" ] && up && [ "$(field running)" != "0" ]; then
      echo "a render is running — refusing (use: $0 stop force)"; exit 1
    fi
    kill_recorded
    up && { echo "pid is gone but :$PORT still answers — inspect before killing anything"; exit 1; } || echo "stopped"
    ;;

  *) echo "usage: $0 {status|start|drain|restart|stop [force]}"; exit 2 ;;
esac
