#!/usr/bin/env python3
"""Collect everything needed to stand the cluster up on a fresh pair of Sparks.

Pulls the real files off spark1/spark2 (not copies from memory), verifies each,
and lists them with what role they play. Prints a manifest.
"""
import os
import subprocess
import sys

OUT = os.path.expanduser("~/sparkrun-recipes/private/cluster-bundle")
os.makedirs(OUT, exist_ok=True)

# (local name, remote path, node)  node: 's1' = spark1, 's2' = via spark1
FILES = [
    # --- the launcher itself (carries the whole serve config) ---
    ("dsv41_tp2.sh", "/home/jaita/dsv41_recipe_ablit/scripts/launch/dsv41_tp2.sh", "s1"),
    ("relaunch-dsv41-canonical.sh", "/home/jaita/relaunch-dsv41-canonical.sh", "s1"),
    ("heatcap.sh", "/home/jaita/heatcap.sh", "s1"),
    # --- docs that explain the recipe ---
    ("METHOD-1017.md", "/home/jaita/dsv41_recipe_ablit/METHOD-1017.md", "s1"),
    ("RECIPE.md", "/home/jaita/dsv41_recipe_ablit/RECIPE.md", "s1"),
    ("FINDINGS.md", "/home/jaita/dsv41_recipe_ablit/FINDINGS.md", "s1"),
    ("BENCHMARKS.md", "/home/jaita/dsv41_recipe_ablit/BENCHMARKS.md", "s1"),
    ("GUIDE.md", "/home/jaita/dsv41_recipe_ablit/GUIDE.md", "s1"),
    ("NOTES.md", "/home/jaita/dsv41_recipe_ablit/NOTES.md", "s1"),
    # --- keepalive ---
    ("keepalive-8000.sh", "/home/jaita/sparkrun-recipes/scripts/keepalive-8000.sh", "s1"),
]


def fetch(name, path, node):
    dest = os.path.join(OUT, name)
    if node == "s1":
        cmd = ["ssh", "-o", "BatchMode=yes", "larryspark", f"cat {path}"]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "larryspark",
               f"ssh -o BatchMode=yes 10.200.0.2 cat {path}"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        print(f"  MISS {name}  ({path})")
        return False
    with open(dest, "wb") as f:
        f.write(r.stdout)
    print(f"  OK   {name}  {len(r.stdout)} B")
    return True


print("=== collecting ===")
ok = 0
for name, path, node in FILES:
    if fetch(name, path, node):
        ok += 1

# also grab the live config snapshots (env + docker cmd) for reference
print("\n=== live config snapshots ===")
snaps = {
    "live-docker-cmd.txt": "docker inspect dsv41-serve --format '{{json .Config.Cmd}}' | tr ',' '\n'",
    "live-env.txt": "docker inspect dsv41-serve --format '{{json .Config.Env}}' | tr ',' '\n'",
    "live-models.txt": "curl -s localhost:8000/v1/models",
    "live-nccl.txt": "ss -tn state established sport = :25000; echo ---; ibv_devinfo -d rocep1s0f0 2>/dev/null | grep -E 'hca_id|fw_ver|port_lid|link_layer|state|active_mtu' | head -20",
    "node-info.txt": "hostname; uname -a; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; ip -4 addr show enp1s0f0np0 2>/dev/null | grep inet",
}
for name, cmd in snaps.items():
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "larryspark", cmd], capture_output=True)
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(r.stdout or b"(unavailable)\n")
    print(f"  snap {name}  {len(r.stdout or b'')} B")

print(f"\ncollected {ok}/{len(FILES)} files -> {OUT}")
