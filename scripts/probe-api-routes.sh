cd ~/imagination-studio || exit 1
KEY=$(tr -d '[:space:]' < ~/ImaginationStudio/access-key)
echo "=== the connections/agents endpoint the Settings page calls ==="
grep -nE '"/api/[^"]*"' app/server.py | head -30
