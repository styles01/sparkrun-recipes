#!/usr/bin/env bash
# Restore: stop muse, relight dsv41 canon, verify, re-arm keepalive, re-lock heat cap.
set -uo pipefail
echo "=== $(date -u) restore start ==="
cd /home/jaita/src/muspark-2dgx 2>/dev/null && ./stop.sh 2>&1 | tail -5
sleep 15
echo "--- containers after muse stop ---"; docker ps --format '{{.Names}}' | head -5
ssh -o BatchMode=yes 10.200.0.2 'docker ps --format "{{.Names}}"' </dev/null | head -5

echo "--- heat cap (post-boot law) ---"
PW=$(grep -m1 '^SUDO_PASSWORD=' /home/jaita/.hermes/.env 2>/dev/null | cut -d= -f2- | tr -d '"')
[ -n "$PW" ] && PW="$PW" /home/jaita/heatcap.sh 2>&1 | tail -6 || echo "heatcap: no PW available on node (run from Mac)"

echo "--- relight dsv41 canon (detached) ---"
nohup setsid bash -c 'cd /home/jaita && bash /home/jaita/relaunch-dsv41-canonical.sh > /tmp/dsv41-restore.log 2>&1' >/dev/null 2>&1 </dev/null &
sleep 45
tail -8 /tmp/dsv41-restore.log
echo "=== fired; poll /v1/models for readiness ==="
