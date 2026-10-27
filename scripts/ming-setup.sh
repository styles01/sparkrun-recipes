#!/usr/bin/env bash
# MING-3.0-tts-tiny setup for spark2 — weights already fetched + size-verified (ming-fetch.log).
# CPU-only: clone repo, uv + py3.10.15 venv, adapted cu130 pins, preflight. GPU untouched.
set -uo pipefail
OUT=/tmp/ming-setup.out
say() { echo "[$(date +%m-%d %H:%M:%S)] $*" >> "$OUT"; }
: > "$OUT"
say "=== MING SETUP START ==="
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME" || exit 1
mkdir -p src
cd src || exit 1

# 1 — inference repo (recipe EXACTLY: pinned defaults of the card)
[ -d Ming-3.0-tts ] || git clone -q --depth 1 https://github.com/inclusionAI/Ming-3.0-tts.git
say "repo: $(cd Ming-3.0-tts && git rev-parse HEAD)"

# 2 — uv + python 3.10.15 (card pin)
# 2 — uv (PEP 668 externally-managed: pip --user is refused on this box) → boot uv from a mini venv
nice -n 19 python3 -m venv /home/jaita/venvs/uvboot >> "$OUT" 2>&1 || { say "FATAL uvboot venv (python3-venv missing?)"; exit 2; }
nice -n 19 /home/jaita/venvs/uvboot/bin/pip install -q uv >> "$OUT" 2>&1 && say "uv bootstrapped via uvboot venv"
UVBIN=/home/jaita/venvs/uvboot/bin/uv
[ -x "$UVBIN" ] || { nice -n 19 curl -sSf https://astral.sh/uv/install.sh | bash >> "$OUT" 2>&1; UVBIN="$HOME/.local/bin/uv"; }
[ -x "$UVBIN" ] || { say "FATAL no uv obtainable"; exit 2; }
command -v uv >/dev/null 2>&1 && UVBIN=uv
nice -n 19 "$UVBIN" python install 3.10.15 >> "$OUT" 2>&1 || { say "FATAL uv python install"; tail -8 "$OUT"; exit 2; }
[ -x "$HOME/venvs/ming-tts/bin/python" ] || nice -n 19 "$UVBIN" venv /home/jaita/venvs/ming-tts --python 3.10.15 >> "$OUT" 2>&1
V=/home/jaita/venvs/ming-tts/bin/python
"$V" --version >> "$OUT" 2>&1 || { say "FATAL venv python"; exit 2; }
say "venv: $($V --version) at /home/jaita/venvs/ming-tts"

# 3 — adapted requirements: torch cu118->cu130 (GB10 sm_121 needs cu130; cu118 wheels top out at sm_90)
cat > /home/jaita/src/Ming-3.0-tts/requirements-tiny-adapted.txt <<'EOF'
# Ming-3.0-tts-tiny requirements-tiny.txt, GB10-ADAPTED (estate recipe, 2026-10-20)
# DEVIATION from upstream pins (license MIT, code Apache): torch/torchaudio cu118 -> cu130
#   (sm_121 GB10 unsupported by cu118 wheels) + triton UNPINNED (torch 2.10 bundles its driver;
#   fla-core uses whatever torch provides). Everything else = upstream pins exact.
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.10.0+cu130
torchaudio==2.10.0+cu130
transformers==4.46.3
tokenizers==0.20.3
safetensors==0.8.0
accelerate==1.14.0
huggingface_hub==0.36.2
packaging==26.2
numpy==1.26.4
scipy==1.15.3
soundfile==0.14.0
onnxruntime==1.23.2
einops==0.8.2
einx==0.4.3
loguru==0.7.3
x-transformers==2.7.4
fla-core==0.4.2
EOF
say "requirements adapted written"

# 4 — install (nice'd: dsv41-serve worker serves on this box; throttle by niceness only, wheels ~4GB)
nice -n 19 "$UVBIN" pip install --index-strategy unsafe-best-match --python "$V" -r /home/jaita/src/Ming-3.0-tts/requirements-tiny-adapted.txt \
  >> /tmp/ming-pip.log 2>&1 || { say "FATAL pip install (see /tmp/ming-pip.log)"; tail -20 /tmp/ming-pip.log >> "$OUT"; exit 3; }
say "pip install OK"

# 5 — sanity: torch cuda + versions
"$V" - <<'EOF' >> "$OUT" 2>&1 || { say "FATAL torch sanity"; exit 4; }
import torch, torchaudio, transformers
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__,
      "| transformers", transformers.__version__, "| cuda:", torch.cuda.is_available(),
      "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF

# 6 — official preflight (CPU-only, seconds; card's own gate)
"$V" /home/jaita/src/Ming-3.0-tts/tools/preflight_tiny.py \
  --model_path /home/jaita/MODELS/MEDIA/Ming-3.0-tts-tiny >> "$OUT" 2>&1 || { say "PREFLIGHT-FAILED"; exit 5; }

say "MING-SETUP-DONE"