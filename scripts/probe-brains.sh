cd ~/imagination-studio || exit 1
KEY=$(tr -d '[:space:]' < ~/ImaginationStudio/access-key)
echo "=== LOCAL_PRESETS (what the UI offers) ==="
grep -n "LOCAL_PRESETS" -A 6 app/static/settings.js | head -12
echo
echo "=== api/brains as the UI actually calls it ==="
curl -s -m25 -H "X-Studio-Key: $KEY" "http://127.0.0.1:7870/api/brains?endpoint=http%3A%2F%2Flarryspark.local%3A8000%2Fv1" -o /tmp/brains.json
python3 - <<'PY'
import json
d = json.load(open("/tmp/brains.json"))
for b in d.get("backends", []):
    if b.get("id") in ("local",) or b.get("local_agent"):
        print(b.get("id"), "| found:", b.get("found"), "| endpoint:", b.get("endpoint"), "| models:", b.get("models"))
PY
