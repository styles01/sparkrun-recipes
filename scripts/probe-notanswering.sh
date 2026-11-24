cd ~/imagination-studio || exit 1
echo "=== where 'not answering' comes from ==="
grep -rn "not answering" app/*.py static/*.js static/**/*.js 2>/dev/null | head -10
echo
echo "=== the probe that decides it ==="
grep -rn "PROBE_TIMEOUT" app/agents.py | head -5
