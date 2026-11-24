cd ~/imagination-studio || exit 1
KEY=$(tr -d '[:space:]' < ~/ImaginationStudio/access-key)
echo "=== keys returned by /api/state ==="
python3 - <<'PY'
import json
d = json.load(open("/tmp/state.json"))
print("top-level keys:", list(d.keys())[:40])
PY
echo
echo "=== find the local agents in the state payload (raw) ==="
grep -o '"local:qwen38"[^}]*}' /tmp/state.json | head -3
grep -o 'larryspark.local:8000[^,]*' /tmp/state.json | head -3
