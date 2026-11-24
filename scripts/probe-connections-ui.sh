cd ~/imagination-studio || exit 1
echo "=== the Settings/Connections page in the JS bundle ==="
grep -rln "local_agents\|hermes_profile\|Connections" static/ 2>/dev/null | head -5
echo
echo "=== which js file holds the Connections UI ==="
grep -rn "local_agents" static/ 2>/dev/null | head -5
