#!/usr/bin/env python3
"""Point the sparkDash qwen tile at the combined qwen-canonical wrapper."""
import json, os, datetime

CFG = os.path.expanduser("~/sparkDash/config/llm-launchers.json")
with open(CFG) as f:
    cfg = json.load(f)

tiles = cfg["launchers"]["spark-001"]
found = False
for t in tiles:
    if t["id"] == "qwen-fn-canon":
        t["name"] = "Qwen3.8-Flash-Next + Imagination Studio (canonical)"
        t["startScript"] = "qwen-canonical.sh"
        t["port"] = 8000
        t["notes"] = (
            "CANONICAL qwen setup, both halves. spark1 :8000 = Mia TensorFold v1.1.4 "
            "text lane (NVIDIA FP4 QAD, 5 concurrent, 262144, vision+video, MTP). "
            "spark2 :7870 = Imagination/Wasserman Studio (AI film: images, video with "
            "sound, voice, music), exposed on the LAN behind its access key. "
            "PORT: the text lane is forced to 8000 (recipe defaults to 8888, PORT LAW); "
            "the studio keeps 7870 because it is not an LLM lane and lives on spark2. "
            "The wrapper is idempotent — an already-running studio is left alone."
        )
        found = True

if not found:
    raise SystemExit("qwen tile not found in config")

bak = CFG + ".bak-qwencanon-" + datetime.date.today().strftime("%Y%m%d")
with open(bak, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

with open(CFG, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print("updated tile:", [t["name"] for t in tiles if t["id"] == "qwen-fn-canon"])
print("backup:", bak)
