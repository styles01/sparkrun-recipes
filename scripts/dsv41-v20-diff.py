#!/usr/bin/env python3
"""Compare upstream v2.0 config.py against our styles01 overlay config.py.

Goal: produce an exact inventory of what our delta adds, so the rebase is
mechanical rather than guesswork.
"""
import difflib
import subprocess
import sys

V20 = "/home/jaita/src/dsv41-v20/v20/patches/exl3_moe/vllm_exl3_moe/config.py"
OURS = "/home/jaita/src/dsv41-v20/ours-config.py"  # fetched separately
DENSE_V20 = "/home/jaita/src/dsv41-v20/v20/patches/exl3_moe/vllm_exl3_moe/dense.py"
DENSE_OURS = "/home/jaita/src/dsv41-v20/ours-dense.py"


def read(p):
    try:
        with open(p) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return None


def summary(name, a, b):
    if a is None or b is None:
        print(f"  {name}: MISSING FILE (v20={a is not None}, ours={b is not None})")
        return
    sm = difflib.SequenceMatcher(None, a, b)
    adds = dels = mods = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            adds += j2 - j1
        elif tag == "delete":
            dels += i2 - i1
        elif tag == "replace":
            mods += max(i2 - i1, j2 - j1)
    print(f"  {name}: v20={len(a)} lines, ours={len(b)} lines | +{adds} -{dels} ~{mods}")

    # what identifiers exist in ours but not v20 (functions/defs/constants)
    def names(lines):
        out = set()
        for l in lines:
            s = l.strip()
            if s.startswith("def ") or s.startswith("class "):
                out.add(s.split("(")[0].split(":")[0])
            elif s and s[0].isupper() and "=" in s:
                out.add(s.split("=")[0].strip())
        return out

    only_ours = sorted(names(b) - names(a))
    only_v20 = sorted(names(a) - names(b))
    print(f"    defs/consts ONLY in ours: {only_ours}")
    print(f"    defs/consts ONLY in v20 : {only_v20}")


print("=== config.py ===")
summary("vllm_exl3_moe/config.py", read(V20), read(OURS))
print("=== dense.py ===")
summary("vllm_exl3_moe/dense.py", read(DENSE_V20), read(DENSE_OURS))
