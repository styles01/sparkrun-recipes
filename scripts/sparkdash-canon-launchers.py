#!/usr/bin/env python3
"""Regenerate sparkDash's llm-launchers.json from LANE-REGISTRY canon.

Each tile points at a thin wrapper under ~/launchers/ whose ONLY job is to exec
the registry's `launch:` line verbatim (env vars exported), because the sparkDash
runner accepts a bare `name.sh` + a directory and passes no env.

Writes:
  ~/launchers/<lane>.sh          (wrapper per lane)
  ~/sparkDash/config/llm-launchers.json  (replacement)
Backs up the old config first.
"""
import json
import os
import stat
import subprocess
import datetime

HOME = os.path.expanduser("~")
LAUNCH_DIR = os.path.join(HOME, "launchers")
CFG = os.path.join(HOME, "sparkDash", "config", "llm-launchers.json")

# lane -> (display name, dir, wrapper body, port, notes)
LANES = {
    "dsv41": dict(
        name="DeepSeek-V4.1 1017 ABLIT (daily driver)",
        d="~/dsv41_recipe_ablit",
        port=8000,
        body='''cd "$HOME/dsv41_recipe_ablit" || exit 1
export HEAD=10.200.0.1 WORKER=10.200.0.2
export MODEL_DIR=/home/jaita/models/dsv41-ablit-tree
export TEXT_ONLY=0
export SPEC='{"method":"dspark","num_speculative_tokens":6,"draft_parallel_config":{"tensor_parallel_size":2}}'
export UTIL=0.85 MAX_SEQS=3 KV_BYTES=2820000000
export IMAGE=dsv41-serve:v52rc
export SERVED='dsv41 DeepSeek-4.1-Flash DeepSeek-4.1-Flash-Ablit GLM-5.3-Flash-EXL3 vision'
exec bash scripts/launch/dsv41_tp2.sh up
''',
        notes="CANONICAL DAILY DRIVER. ABLIT twin, TP2 both Sparks, 3x250k, DSpark k=6, v52rc image. "
              "Serves 5 names incl the vision alias and the GLM compat shim. Heat cap must be run "
              "separately (needs sudo, the runner cannot).",
    ),
    "qwen-fn": dict(
        name="Qwen3.8-Flash-Next (TensorFold, Mia)",
        d="~/src/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold",
        port=8000,
        body='''cd "$HOME/src/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold" || exit 1
export PORT=8000
exec ./start.sh
''',
        notes="Mia TensorFold v1.1.4 lane: NVIDIA FP4 QAD, 5 concurrent, 262144, vision+video, MTP drafts. "
              "PORT forced to 8000 (the recipe defaults to 8888 - PORT LAW). Needs the box free of dsv41.",
    ),
    "sante": dict(
        name="Ling-3.0-flash-Sante (FP4 TP1)",
        d="~/dsv41_recipe_ablit",
        port=8000,
        body='''cd "$HOME/dsv41_recipe_ablit" || exit 1
export HEAD=10.200.0.1
exec bash scripts/launch/sante_fp4_tp1.sh up
''',
        notes="Our own FP4 pack, TP1 single box, v032 image, ~33 tok/s. Tools live (ling3 parsers). "
              "Needs dsv41 down and ~110 GiB free on spark1.",
    ),
    "media-mode": dict(
        name="Media stack (qwen text + spark2 media)",
        d="~/dsv41_recipe_ablit",
        port=8000,
        body='''cd "$HOME/src/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold" || exit 1
export PORT=8000
exec ./start.sh
''',
        notes="Text lane on spark1 (same as qwen-fn); spark2 media lanes are started individually "
              "(switch-media.sh / serve-music3 / serve-yue2). See registry media-mode entry.",
    ),
}

ORDER = ["dsv41", "qwen-fn", "sante", "media-mode"]


def main():
    os.makedirs(LAUNCH_DIR, exist_ok=True)
    entries = []
    now = int(datetime.datetime.now().timestamp() * 1000)

    for i, lane in enumerate(ORDER):
        spec = LANES[lane]
        wrapper = os.path.join(LAUNCH_DIR, f"{lane}.sh")
        with open(wrapper, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"# sparkDash launcher for the {lane} lane.\n")
            f.write("# Body is the LANE-REGISTRY.yaml launch: line, verbatim env + command.\n")
            f.write("set -euo pipefail\n")
            f.write(spec["body"])
        os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"  wrapper: {wrapper}")

        entries.append({
            "id": f"{lane}-canon",
            "name": spec["name"],
            "dir": "~/launchers",
            "startScript": f"{lane}.sh",
            "stopScript": None,
            "port": spec["port"],
            "notes": spec["notes"],
            "createdAt": now + i,
        })

    cfg = {"version": 1, "launchers": {"spark-001": entries}}

    if os.path.exists(CFG):
        bak = CFG + ".bak-canon-" + datetime.date.today().strftime("%Y%m%d")
        with open(CFG) as f:
            old = f.read()
        with open(bak, "w") as f:
            f.write(old)
        print(f"  backup: {bak}")
        print("  REMOVING stale tiles:")
        try:
            prev = json.loads(old)["launchers"]["spark-001"]
            for t in prev:
                if "-canon" not in t["id"]:
                    print(f"    - {t['name']}  ({t['dir']}, port {t.get('port')})")
        except Exception:
            pass

    with open(CFG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote: {CFG}  ({len(entries)} tiles)")


if __name__ == "__main__":
    main()
