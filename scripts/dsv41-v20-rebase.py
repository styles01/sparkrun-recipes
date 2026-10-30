#!/usr/bin/env python3
"""Correct rebase: our v52rc is styles01-overlay + our +98/-3, on top of upstream v1.2.

Approach that cannot silently discard our delta:
  1. Find the real common ancestor: the upstream commit styles01 forked from.
  2. diff (styles01/main -> ours)  == our ablit delta, expressed as a patch.
  3. Apply that patch to v2.0's files.
  4. Verify by NAMING the pieces we must keep (not by trusting the merge tool).
"""
import os
import subprocess
import sys

WORK = "/home/jaita/src/dsv41-v20"
V20 = os.path.join(WORK, "v20")
FILES = ["config.py", "dense.py"]
# The pieces our lane cannot boot without.
REQUIRED = [
    "def _dense_ckpt_k",
    "Exl3DequantDenseLinearMethod",
    "compressor.fused_wkv_wgate",
    "indexer.wk",
    "indexer.wq_b",
    "def _trellis_k_from_header",
    "def _is_drafter_layer",
]


def sh(cmd, cwd=WORK, check=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=isinstance(cmd, str))
    if check and r.returncode != 0:
        print("FAILED:", cmd, "\n", r.stderr[:600])
        sys.exit(1)
    return r


def main():
    # 1. merge-base between v20 HEAD and styles01/main
    mb = sh(["git", "merge-base", "HEAD", "styles01/main"], cwd=V20)
    base = mb.stdout.strip()
    print("true merge-base:", base[:10] or "NONE FOUND")
    if not base:
        print("No common ancestor — styles01 is a re-rooted history.")
        print("Fallback: treat the styles01 file as the base (delta = ours vs styles01).")

    for name in FILES:
        rel = f"patches/exl3_moe/vllm_exl3_moe/{name}"
        ours = os.path.join(WORK, f"ours-{name}")
        theirs = os.path.join(V20, rel)

        # delta = ours vs styles01's own file (that IS our contribution)
        delta = os.path.join(WORK, f"delta-{name}.patch")
        d = subprocess.run(
            ["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}",
             theirs if False else os.path.join(WORK, f"ours-{name}"), os.path.join(WORK, f"ours-{name}")],
            capture_output=True, text=True)
        # (unused — kept for clarity)

        # Build the real delta: styles01 main version  ->  ours
        styles01_ver = os.path.join(WORK, f"styles01-{name}")
        blob = sh(["git", "show", f"styles01/main:{rel}"], cwd=V20)
        with open(styles01_ver, "w") as fh:
            fh.write(blob.stdout)

        p = subprocess.run(["diff", "-u", styles01_ver, ours], capture_output=True, text=True)
        with open(delta, "w") as fh:
            fh.write(p.stdout)
        adds = sum(1 for l in p.stdout.splitlines() if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in p.stdout.splitlines() if l.startswith("-") and not l.startswith("---"))
        print(f"\n{name}: our delta vs styles01 = +{adds} -{dels} lines")

        # apply onto v2.0
        target = os.path.join(WORK, f"rebased-{name}")
        with open(target, "wb") as fh:
            fh.write(open(theirs, "rb").read())
        ap = subprocess.run(["patch", "--forward", "--fuzz=3", "-p0", target, delta],
                            capture_output=True, text=True)
        print(f"  patch apply rc={ap.returncode}")
        if ap.stdout.strip():
            print("   ", ap.stdout.strip().replace("\n", "\n    ")[:500])

        # 3. verify by naming the pieces
        txt = open(target).read()
        missing = [r for r in REQUIRED if r not in txt]
        print(f"  REQUIRED present: {len(REQUIRED)-len(missing)}/{len(REQUIRED)}")
        if missing:
            print(f"  MISSING: {missing}")


if __name__ == "__main__":
    main()
