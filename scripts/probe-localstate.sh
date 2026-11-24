cd ~/imagination-studio || exit 1
echo "=== how the studio PROBES a local endpoint ==="
grep -n "def local_state\|def probe_local\|def check_local\|models\"" app/agents.py | head -12
echo
echo "=== the probe function body ==="
sed -n '/def local_state/,/^def /p' app/agents.py | head -50
