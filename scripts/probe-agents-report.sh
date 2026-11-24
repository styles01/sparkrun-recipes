cd ~/imagination-studio || exit 1
echo "=== lines 780-840: how agents are reported to the UI ==="
sed -n '795,840p' app/agents.py
echo
echo "=== the settings route that SAVES this ==="
grep -rn "local_agents" app/server.py | head -10
