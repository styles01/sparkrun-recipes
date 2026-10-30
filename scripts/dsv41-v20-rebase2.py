#!/usr/bin/env python3
"""Rebase our REAL ablit plugin (extracted from the live container) onto v2.0.

True lineage: upstream v2.0 (297) <- styles01 427 <- ours 533.
Delta = live/config.py vs styles01's 427-line file. That delta is what must
land on v2.0. Verify by naming required pieces, never by trusting the tool.
"""
import os
import subprocess

WORK = "/home/jaita/src/dsv41-v20"
V20 = os.path.join(WORK, "v20")
FILES = ["config.py", "dense.py"]
REQUIRED = {
    "config.py": [
        "def _dense_ckpt_k",
        "compressor.fused_wkv_wgate",
        "indexer.wk",
        "indexer.wq_b",
        "def _trellis_k_from_header",
        "def _is_drafter_layer",
    ],
    "dense.py": ["Exl3DequantDenseLinearMethod", "def _dequant_weight"],
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


for name in FILES:
    rel = f"patches/exl3_moe/vllm_exl3_moe/{name}"
    live = os.path.join(WORK, "live", name)          # 533 / 445  = OURS
    styles01 = os.path.join(WORK, f"styles01-{name}")  # 427 / 424  = base
    theirs = os.path.join(V20, rel)                  # v2.0

    d = run(["diff", "-u", styles01, live])
    delta = os.path.join(WORK, f"delta-{name}.patch")
    with open(delta, "w") as fh:
        fh.write(d.stdout)
    adds = sum(1 for l in d.stdout.splitlines() if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in d.stdout.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"\n{name}: OURS vs styles01 = +{adds} -{dels}  -> {delta}")

    target = os.path.join(WORK, f"rebased-{name}")
    with open(target, "wb") as fh:
        fh.write(open(theirs, "rb").read())
    ap = run(["patch", "--forward", "--fuzz=3", "-p0", target, delta])
    print(f"  apply rc={ap.returncode} {ap.stdout.strip()[:300]}")

    txt = open(target).read()
    miss = [r for r in REQUIRED[name] if r not in txt]
    print(f"  required: {len(REQUIRED[name]) - len(miss)}/{len(REQUIRED[name])}" +
          (f"  MISSING {miss}" if miss else "  ALL PRESENT"))
    c = run(["python3", "-m", "py_compile", target])
    print("  py_compile:", "OK" if c.returncode == 0 else c.stderr[:300])
