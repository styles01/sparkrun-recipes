#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next Q4_K_XL via the llama.cpp qwen4exp fork (PR #27742,
# commit 035e227) with the 0xBakeer "180B-fits-in-119GB" trick:
#   -ot per_layer_token_embd=CPU   -> pin the 51B n-gram token-embd tensor to CPU (never GPU)
#   -lm mmap                       -> serve it from NVMe via mmap
# Spec decode: stock ngram-mod (default n_max=3). Do NOT raise n-min/max — it
# burns compute on rejected drafts and drops decode to ~14-18 tok/s (measured).
# Model: unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL (104GB, 4 shards).
#
# ENV:
#   MODEL_DIR      directory holding the 4 shard gguf files (default: auto-download to /models)
#   SERVED_NAME    --alias (default: qwen3.8-flash-next)
#   HOST / PORT    listen addr (default: 0.0.0.0 / 8000)
#   CTX            --ctx-size (default: 262144)
#   HF_TOKEN       optional HF read token (for gated downloads)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
# llama-server's sibling shared libraries live in the qwen4exp bin directory.
export LD_LIBRARY_PATH="/opt/qwen4exp/bin:${LD_LIBRARY_PATH:-}"

# --- model acquisition -------------------------------------------------------
SHARD1="$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
if [ ! -f "$SHARD1" ]; then
  echo ">> Model shards not present in $MODEL_DIR — downloading from HuggingFace (104GB, 4 shards)..."
  mkdir -p "$MODEL_DIR"
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL=huggingface-cli
  elif command -v hf >/dev/null 2>&1; then
    HF_DL="hf download"
  else
    python3 - <<'PY'
import os,sys,subprocess
subprocess.check_call([sys.executable,"-m","pip","install","-q","-U","huggingface_hub"])
PY
    HF_DL="huggingface-cli"
  fi
  # shellcheck disable=SC2086
  $HF_DL download unsloth/Qwen3.8-Flash-Next-GGUF \
      --include "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000?-of-00004.gguf" \
      --local-dir "$MODEL_DIR/UD-Q4_K_XL" \
      --local-dir-use-symlinks False
  SHARD1="$MODEL_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
fi

echo ">> Serving $SHARD1 (alias=$SERVED_NAME, host=$HOST:$PORT, ctx=$CTX)"
exec llama-server \
  -m "$SHARD1" \
  --alias "$SERVED_NAME" \
  -lm mmap \
  -ot per_layer_token_embd=CPU \
  --n-gpu-layers 999 \
  --ctx-size "$CTX" \
  --parallel 1 \
  --spec-type ngram-mod \
  --flash-attn on \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}' \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host "$HOST" --port "$PORT"
