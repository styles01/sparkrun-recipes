#!/usr/bin/env bash
# Fetch Qwen3.5-4B-3D-Coder to spark2 for later testing. No serving, no lane changes.
set -uo pipefail
DEST="$HOME/3d-coder"
mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "=== $(date -u) fetching Qwen3.5-4B-3D-Coder ==="
export PATH="$HOME/.local/bin:$PATH"

REPO="ccore/Qwen3.5-4B-3D-Coder"
REV="main"

echo "--- free before ---"
df -B1 /home | awk 'NR==2 {print int($4/1e9), "GB free"}'

# GGUF quants (the useful ones) + the vision projector + tokenizer/config
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --revision "$REV" --local-dir "$DEST/repo" \
    --include "*.gguf" "*.json" "*.jinja" "*.md"
else
  python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="ccore/Qwen3.5-4B-3D-Coder",
    local_dir=os.path.expanduser("~/3d-coder/repo"),
    allow_patterns=["*.gguf", "*.json", "*.jinja", "*.md"],
)
print("snapshot:", p)
PYEOF
fi

echo "--- what landed ---"
ls -la "$DEST/repo" 2>/dev/null | head -20
du -sh "$DEST" 2>/dev/null

echo "--- free after ---"
df -B1 /home | awk 'NR==2 {print int($4/1e9), "GB free"}'
echo "=== done $(date -u) ==="
