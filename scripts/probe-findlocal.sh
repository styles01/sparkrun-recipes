cd ~/imagination-studio || exit 1
echo "=== slate.find_local: how it discovers the server ==="
grep -n "def find_local" -A 45 app/slate.py | head -55
