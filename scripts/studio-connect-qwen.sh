#!/usr/bin/env bash
# Connect the Imagination/Wasserman Studio (spark2) to the local Qwen lane on spark1.
# The Studio stores agent settings in <studio home>/.studio-agents/settings.json,
# NOT directly in <studio home>/settings.json. Verified against app/agents.py:211.
set -uo pipefail

STUDIO_HOME="$HOME/ImaginationStudio"
SETTINGS="$STUDIO_HOME/.studio-agents/settings.json"
ENDPOINT="http://larryspark.local:8000/v1"
MODEL="Qwen3.8-Flash-Next"

echo "=== $(date -u) wiring studio -> spark1:$MODEL ==="

CODE=$(curl -s -o /tmp/qwen-models.json -w "%{http_code}" -m 10 "$ENDPOINT/models" || echo 000)
echo "endpoint $ENDPOINT -> HTTP $CODE"
if [ "$CODE" != "200" ]; then
  echo "REFUSING to write: endpoint not answering"
  exit 1
fi

mkdir -p "$STUDIO_HOME/.studio-agents"

python3 - <<PYEOF
import json, os, shutil, datetime

path = os.path.expanduser("$SETTINGS")
endpoint = "$ENDPOINT"
model = "$MODEL"

saved = {}
if os.path.exists(path):
    shutil.copy2(path, path + ".bak-" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    try:
        saved = json.load(open(path))
    except Exception:
        saved = {}
if not isinstance(saved, dict):
    saved = {}

agents = [a for a in (saved.get("local_agents") or []) if isinstance(a, dict)]
agents = [a for a in agents if a.get("id") != "local:qwen38"]
agents.append({"id": "local:qwen38", "name": "Qwen3.8 Flash Next",
               "endpoint": endpoint, "model": model})
saved["local_agents"] = agents

json.dump(saved, open(path, "w"), indent=2)
os.chmod(path, 0o600)
print("wrote", path)
PYEOF

echo "=== verify through the studio's own code path ==="
cd "$HOME/imagination-studio" || exit 1
~/.local/opt/tensorfold-studio/venv/bin/python - <<'PYEOF'
import sys
sys.path.insert(0, "/home/jaita/imagination-studio/app")
import agents
from studio_home import studio_home
agents.HOME = studio_home()

print("folder():", agents.folder())
print("saved agent ids:", agents.local_agent_ids({}))
for a in agents.clean_local_agents(agents.read_json(agents.folder()/"settings.json", {}).get("local_agents")):
    print("  ACCEPTED:", a["id"], "|", a["name"], "|", a["endpoint"], "|", a["model"])
PYEOF
