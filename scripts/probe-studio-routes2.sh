cd ~/imagination-studio || exit 1
echo "=== all route decorators (first 40) ==="
grep -nE '@app\.(get|post|put|delete)\(' app/server.py | head -40
