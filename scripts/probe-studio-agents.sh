cd ~/imagination-studio || exit 1
echo "=== folder() definition ==="
sed -n '60,80p' app/agents.py
echo
echo "=== studio data dir listing ==="
ls -la ~/ImaginationStudio/ 2>/dev/null | head -20
echo
echo "=== hermes settings keys ==="
sed -n '600,630p' app/agents.py
echo
echo "=== is hermes CLI on spark2? ==="
command -v hermes || echo "hermes NOT on spark2"
echo
echo "=== studio key/auth for agent calls ==="
grep -n "hermes_profile\|hermes_command\|HERMES" app/agents.py | head -10
