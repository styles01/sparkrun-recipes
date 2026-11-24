cd ~/imagination-studio || exit 1
echo "=== STUDIO_HOME as the live server sees it ==="
PID=$(pgrep -f "app/server.py" | head -1)
tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep -E "^STUDIO_HOME="
echo
echo "=== studio_home() definition ==="
grep -rn "def studio_home" -A 14 app/ 2>/dev/null | head -20
