#!/usr/bin/env bash
# Slate/Imagination Studio install on spark2 — reuse-first (discover.py links what exists).
set -uo pipefail
cd "$HOME/imagination-studio" || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "=== $(date -u) studio install start ==="
echo "free GB: $(df -B1 /home | awk 'NR==2 {print int($4/1e9)}')"

# --no-render: provision + build, do not fire a test clip yet.
# --voice: Drama3 already staged (discover found it).
# NOT included on first pass: --ref2va --h3-base --flux3 --moss --music --train --graphics --all
nohup setsid bash -c '
  cd "$HOME/imagination-studio" || exit 1
  export PATH="$HOME/.local/bin:$PATH"
  bash oneshot-setup-spark.sh --voice --no-render
' > "$HOME/studio-install.log" 2>&1 < /dev/null &

sleep 45
echo "--- first output ---"
tail -30 "$HOME/studio-install.log"
