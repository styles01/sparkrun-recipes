cd ~/imagination-studio || exit 1
echo "=== settings.js: the Connections UI + what it calls ==="
grep -n "not answering\|local_agents\|Connections\|endpoint" app/static/settings.js | head -25
