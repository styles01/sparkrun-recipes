cd ~/imagination-studio || exit 1
KEY=$(tr -d '[:space:]' < ~/ImaginationStudio/access-key)
echo "=== /api/state: what the Settings page renders for agents ==="
curl -s -m20 -H "X-Studio-Key: $KEY" http://127.0.0.1:7870/api/state -o /tmp/state.json
python3 - <<'PY'
import json
d = json.load(open("/tmp/state.json"))
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in k.lower() for s in ("agent", "brain", "local", "connect", "hermes")):
                s = json.dumps(v)[:300]
                print(f"{path}/{k} = {s}")
            walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            walk(v, f"{path}[{i}]")
walk(d)
PY
