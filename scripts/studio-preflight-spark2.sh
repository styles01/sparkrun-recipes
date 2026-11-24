#!/usr/bin/env bash
# spark2 preflight for the Slate/Imagination Studio install.
# Installs ffmpeg + uv (the installer hard-dies without them) and clones 4.3.1.
set -uo pipefail

echo "=== $(date -u) spark2 studio preflight ==="

# 1. ffmpeg / ffprobe (needs sudo; password arrives on stdin from the caller)
if ! command -v ffmpeg >/dev/null 2>&1; then
  PWLINE=$(head -1)
  printf '%s\n' "$PWLINE" | sudo -S -p "" apt-get update -qq 2>&1 | tail -2
  printf '%s\n' "$PWLINE" | sudo -S -p "" apt-get install -y -qq ffmpeg 2>&1 | tail -3
fi
command -v ffmpeg && ffmpeg -version 2>/dev/null | head -1
command -v ffprobe && echo "ffprobe OK"

# 2. uv (user-space, no sudo)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv && uv --version

# 3. clone 4.3.1 (new repo name; old one redirects)
cd "$HOME" || exit 1
if [ ! -d imagination-studio/.git ]; then
  git clone -q https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman.git imagination-studio
fi
cd imagination-studio || exit 1
echo "HEAD: $(git log --oneline -1)"
echo "VERSION: $(cat VERSION 2>/dev/null)"

# 4. what the installer will want
echo "--- setup script presence ---"
ls -la oneshot-setup-spark.sh oneshot-setup-cuda.sh 2>/dev/null
echo "--- discover.py (the reuse mechanism) ---"
ls -la app/discover.py 2>/dev/null && grep -c "elsewhere" app/discover.py 2>/dev/null
echo "=== preflight done $(date -u) ==="
