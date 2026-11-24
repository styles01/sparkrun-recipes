#!/usr/bin/env bash
# Inventory spark2 for the media-model optimization plan.
set -uo pipefail
cd "$HOME/imagination-studio" 2>/dev/null || exit 1

echo "=== disk ==="
df -B1 /home | awk 'NR==2 {print "free GB:", int($4/1e9), "| used GB:", int($3/1e9)}'

echo
echo "=== studio model tree (what the app actually uses) ==="
du -sh "$HOME"/spark-studio-models/* 2>/dev/null | sort -h

echo
echo "=== legacy MEDIA tree (what we hold for other lanes) ==="
du -sh "$HOME"/MODELS/MEDIA/* 2>/dev/null | sort -h | tail -15

echo
echo "=== catalogue status: what the app wants ==="
python3 app/discover.py --scan 2>/dev/null | grep -E '^(found|missing)' | awk '{print $1}' | sort | uniq -c

echo
echo "=== FOUND (reusable, no download) ==="
python3 app/discover.py --scan 2>/dev/null | grep '^found' | sed 's/  */ /g' | cut -c1-120

echo
echo "=== MISSING (would need download) ==="
python3 app/discover.py --scan 2>/dev/null | grep '^missing' | sed 's/  */ /g' | cut -c1-100
