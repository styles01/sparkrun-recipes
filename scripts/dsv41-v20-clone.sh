#!/usr/bin/env bash
# Clone upstream v2.0 and lay our styles01 delta beside it for the rebase.
set -uo pipefail
SRC=/home/jaita/src/dsv41-v20
mkdir -p "$SRC"
cd "$SRC" || exit 1

if [ ! -d v20/.git ]; then
  git clone -q https://github.com/sudoingX/dgx-spark-deepseek-v41.git v20 || { echo "CLONE FAILED"; exit 1; }
fi
cd v20 || exit 1
echo "=== upstream v2.0 ==="
git log --oneline -3
echo "--- tags ---"
git tag | tail -8
echo "--- HEAD date ---"
git log -1 --format='%H %ci'
echo
echo "=== exl3_moe plugin files (v2.0) ==="
ls patches/exl3_moe/vllm_exl3_moe/*.py | head -30
echo
echo "=== does v2.0 ship a Dockerfile / build path for the serve image? ==="
ls scripts/image/ 2>/dev/null
grep -n "FROM\|COPY\|pip install" scripts/image/Dockerfile.serve-overlay 2>/dev/null | head -15
