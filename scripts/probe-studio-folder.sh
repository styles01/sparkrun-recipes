cd ~/imagination-studio || exit 1
echo "=== how does folder() resolve HOME ==="
grep -n "def folder" -A 12 app/agents.py
echo
echo "=== does settings.json parse? ==="
cat ~/ImaginationStudio/settings.json
echo
echo "=== BACKENDS ==="
grep -n "^BACKENDS" -A 4 app/agents.py
