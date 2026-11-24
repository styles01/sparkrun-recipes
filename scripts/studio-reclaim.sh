#!/usr/bin/env bash
# Reclaim dead weight on spark2 for the Studio media-model plan.
# TRASH-ONLY: everything is MOVED to ~/trash, never deleted.
# Refuses to move anything a live process holds open.
set -uo pipefail

TRASH="$HOME/trash/media-reclaim-$(date -u +%Y%m%d)"
mkdir -p "$TRASH"

echo "=== $(date -u) reclaim start ==="
df -B1 /home | awk 'NR==2 {print "free GB before:", int($4/1e9)}'

move_one() {
  local src="$1" label="$2"
  if [ ! -e "$src" ]; then echo "  skip (absent): $label"; return; fi
  # safety: refuse if anything holds a file open under it
  if command -v lsof >/dev/null 2>&1 && lsof +D "$src" 2>/dev/null | head -2 | grep -q .; then
    echo "  REFUSED (in use): $label"; lsof +D "$src" 2>/dev/null | head -3; return
  fi
  local sz; sz=$(du -sb "$src" 2>/dev/null | cut -f1)
  mv "$src" "$TRASH/$(basename "$src")" 2>/dev/null \
    && echo "  moved: $label  ($(awk -v s="${sz:-0}" 'BEGIN{printf "%.1f", s/1e9}') GB)" \
    || echo "  FAILED: $label"
}

echo "--- 1. ComfyUI-test scratch tree (242 GB) ---"
move_one "$HOME/MODELS/MEDIA/ComfyUI-test" "ComfyUI-test"

echo "--- 2. dead GGUF FLUX copy (ComfyUI FLUX nodes need safetensors) ---"
move_one "$HOME/MODELS/MEDIA/FLUX-3-Image-Klein-GGUF" "FLUX-3 GGUF"

echo "--- 3. int8 H3 transformers the studio cannot load (wants bf16) ---"
for f in \
  "$HOME/MODELS/MEDIA/checkpoints/MiniMax-H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
  "$HOME/models/checkpoints/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
do
  [ -e "$f" ] && move_one "$f" "$(basename "$f")"
done

echo
df -B1 /home | awk 'NR==2 {print "free GB after:", int($4/1e9)}'
echo "trash dir: $TRASH"
du -sh "$TRASH" 2>/dev/null
echo "=== done $(date -u) ==="
