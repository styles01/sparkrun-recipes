cd ~/imagination-studio || exit 1
echo "=== how does server.py set agents.HOME? ==="
grep -rn "HOME" app/server.py | grep -iE "agents|studio-agents|setattr|=" | head -10
echo
echo "=== does .studio-agents dir exist anywhere? ==="
ls -la ~/.studio-agents 2>/dev/null | head -5 || echo "no ~/.studio-agents"
echo
echo "=== app/launcher.py HOME handling ==="
grep -n "HOME" app/launcher.py | head -8
