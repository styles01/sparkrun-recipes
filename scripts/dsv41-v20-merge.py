#!/usr/bin/env python3
"""Three-way merge: upstream v2.0 <- styles01 base <- our ablit delta.

Base = styles01 main (the parent both sides share).
Left = v2.0 upstream.
Right = ours (v52rc lineage).

Emits merged files and reports conflicts; does NOT install anything.
"""
import subprocess
import sys
import os

WORK = "/home/jaita/src/dsv41-v20"
FILES = ["config.py", "dense.py"]


def git(*args, cwd=WORK):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r


def main():
    os.makedirs(WORK, exist_ok=True)
    # fetch styles01 main into the v20 repo's object store
    if not os.path.isdir(os.path.join(WORK, "v20", ".git")):
        print("v20 not cloned")
        sys.exit(1)
    v20 = os.path.join(WORK, "v20")

    if git("remote", "get-url", "styles01", cwd=v20).returncode != 0:
        git("remote", "add", "styles01",
            "https://github.com/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks.git", cwd=v20)
    f = git("fetch", "-q", "styles01", "main", cwd=v20)
    if f.returncode != 0:
        print("fetch styles01 failed:", f.stderr[:400])

    base_ref = git("rev-parse", "styles01/main", cwd=v20).stdout.strip()[:10]
    print("styles01 base:", base_ref)
    print("v20 HEAD     :", git("rev-parse", "HEAD", cwd=v20).stdout.strip()[:10])

    for name in FILES:
        left = f"patches/exl3_moe/vllm_exl3_moe/{name}"
        base = f"styles01/main:patches/exl3_moe/vllm_exl3_moe/{name}"
        right_path = os.path.join(WORK, f"ours-{name}")
        if not os.path.exists(right_path):
            print(f"  {name}: missing {right_path}")
            continue

        # write the base version out
        blob = git("show", base, cwd=v20)
        if blob.returncode != 0:
            print(f"  {name}: cannot read base from styles01/main")
            continue
        base_path = os.path.join(WORK, f"base-{name}")
        with open(base_path, "w") as fh:
            fh.write(blob.stdout)

        merged_path = os.path.join(WORK, f"merged-{name}")
        m = subprocess.run(
            ["git", "merge-file", "-p", "--diff3", right_path, base_path,
             os.path.join(v20, left)],
            capture_output=True, text=True)
        with open(merged_path, "w") as fh:
            fh.write(m.stdout)
        conflicts = m.stdout.count("<<<<<<<")
        print(f"  {name}: merged -> {merged_path} | conflicts={conflicts}")

    print("\nNext: inspect merged-*.py, resolve, py_compile.")


if __name__ == "__main__":
    main()
