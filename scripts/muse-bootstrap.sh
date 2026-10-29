#!/usr/bin/env bash
# Muse Spark bootstrap: clone muspark-2dgx, inspect, prepare local.sh. No lane touched.
set -euo pipefail
SRC=/home/jaita/src/muspark-2dgx
echo "=== $(date -u) bootstrap start ==="
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/ajensenwaud/muspark-2dgx.git "$SRC"
fi
cd "$SRC"
echo "HEAD: $(git log --oneline -1)"
echo "tag: $(git describe --tags 2>/dev/null || echo none)"
echo "--- tree ---"
ls
echo "--- docs ---"
ls docs 2>/dev/null
echo "--- patches ---"
ls patches-zig patches-exl3 2>/dev/null | head -20
echo "--- config defaults worth knowing ---"
grep -E '^(HEAD|WORKER|MASTER|MASTER|WORKER_IP|ZIG_TF_COMMIT|ZIG_TREE|BASE_IMAGE|ZIG_BUILD_IMAGE|EXL3_COMMIT|IMAGE|PACK_REPO|PACK_REV)' scripts/config.sh || true
echo "--- local.sh.example ---"
cat scripts/local.sh.example 2>/dev/null
echo "=== bootstrap done ==="
