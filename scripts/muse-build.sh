#!/usr/bin/env bash
# Muse Spark: build the Zig engine + patched exllamav3 image. CPU/container only; does NOT touch the serving lane.
set -uo pipefail
SRC=/home/jaita/src/muspark-2dgx
LOG=/home/jaita/muse-build-$(date -u +%Y%m%d-%H%M).log
cd "$SRC" || exit 1

# host config
if [ ! -f scripts/local.sh ]; then
  cat > scripts/local.sh <<'EOF'
HEAD=larryspark
WORKER=jaita@10.200.0.2
MASTER=10.200.0.1
WORKER_IP=10.200.0.2
EOF
  echo "wrote scripts/local.sh"
fi

echo "=== build start $(date -u) ===" 
echo "--- 1. exllamav3 patched image (needed to read the muse_spark pack) ---"
bash scripts/build-image.sh 2>&1 | tail -25
echo "EXL3_RC=$?"

echo "--- 2. zig engine build (TensorFold main + patches-zig) ---"
timeout 7200 bash scripts/zig_prepare.sh 2>&1 | tail -40
echo "ZIG_RC=$?"

echo "=== build end $(date -u) ==="
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -Ei 'muse|tensorfold|exl3' | head -10
