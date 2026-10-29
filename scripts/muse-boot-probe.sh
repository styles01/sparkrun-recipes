#!/usr/bin/env bash
# Muse boot probe: stop dsv41 via canonical launcher, boot Muse TP2 (PROFILE=long), wait for health, one generation.
set -uo pipefail
cd /home/jaita || exit 1
echo "=== $(date -u) muse boot probe start ==="

echo "--- stop dsv41 (canonical) ---"
cd /home/jaita/dsv41_recipe_ablit && HEAD=10.200.0.1 WORKER=10.200.0.2 bash scripts/launch/dsv41_tp2.sh down 2>&1 | tail -5
cd /home/jaita || exit 1
sleep 20
echo "containers after stop:"; docker ps --format '{{.Names}}' | head -5
ssh -o BatchMode=yes 10.200.0.2 'docker ps --format "{{.Names}}"' </dev/null | head -5

echo "--- muse start (PROFILE=long, detached) ---"
cd /home/jaita/src/muspark-2dgx || exit 1
nohup setsid bash -c './start.sh > /home/jaita/muse-boot.log 2>&1' >/dev/null 2>&1 </dev/null &
sleep 30
echo "--- boot log so far ---"
tail -20 /home/jaita/muse-boot.log
echo "=== probe fired; poll with: tail /home/jaita/muse-boot.log ==="
