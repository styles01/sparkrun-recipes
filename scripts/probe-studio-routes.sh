cd ~/imagination-studio || exit 1
KEY=$(tr -d '[:space:]' < ~/ImaginationStudio/access-key)
echo "=== routes mentioning agents/settings ==="
grep -nE '@app\.(get|post)\("[^"]*(agent|settings|connect)[^"]*"' app/server.py | head -20
