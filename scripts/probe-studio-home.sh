cd ~/imagination-studio || exit 1
echo "=== how is HOME set for the server? ==="
grep -rn "agents.HOME" --include=*.py . | head -8
echo
echo "=== what is HOME at runtime (ask the live server's env) ==="
PID=$(pgrep -f "app/server.py" | head -1)
echo "server pid: $PID"
tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep -E "^HOME=|^STUDIO_" | head -6
echo
echo "=== studio settings endpoints ==="
grep -n "settings" app/server.py | head -10
