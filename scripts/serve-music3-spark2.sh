#!/bin/bash
# CANONICAL LAUNCHER — MiniMax Music 3 text2music lane on SPARK-2 (larryspark2, 192.168.2.14)
# Lane card: t_034e473e (Humpty Dumpty A/B control arm). Port 8050 (8040 stays reserved for MiniMax-H3).
#
# WHY THIS ENV EXISTS (both were silent crash-loops on spark2, root-caused 2026-10-13):
#   1) PATH must carry .venv/bin  -> flashinfer JIT shells out to `ninja`; no host ninja-build,
#      but the pip `ninja` wheel lands ninja in .venv/bin. Without it:
#      FileNotFoundError: 'ninja' at stage minimax_music3_ar.
#   2) CPATH must carry the python dev headers -> python3.12-dev is NOT installed on spark2 and
#      there is no sudo. Triton's inductor launcher compiles __triton_launcher.c against Python.h,
#      so CUDA graph capture died with "fatal error: Python.h". Headers are a ~1.8MB rsync from
#      spark1 (both nodes python 3.12.3, ABI-identical). BOTH paths are required, because Debian's
#      pyconfig.h re-includes <aarch64-linux-gnu/python3.12/pyconfig.h> relative to the include
#      ROOT (mirrors /usr/include on spark1):
#        mkdir -p /home/jaita/pyinc/aarch64-linux-gnu
#        rsync -a jaita@10.200.0.1:/usr/include/python3.12/               /home/jaita/pyinc/python3.12/
#        rsync -a jaita@10.200.0.1:/usr/include/aarch64-linux-gnu/python3.12/ /home/jaita/pyinc/aarch64-linux-gnu/python3.12/
#      Verified by building a launcher-style .so: CPATH=pyinc/python3.12:pyinc gcc -shared -fPIC ... -> OK
#   3) CUDA_HOME=/usr/local/cuda (release 13.0) is spark2's host toolkit for the flashinfer JIT.
#   4) ~/.cache/flashinfer/<ver>/121a holds spark1's proven sm121 build; seeding it avoids a
#      cold JIT recompile of norm/prefill kernels.
#
# Weights (L3 blob-verified 2026-10-13, 54GB real blobs, NOT an hf stub):
#   /home/jaita/MODELS/MEDIA/MiniMax-Music3   (/home/jaita/models -> /home/jaita/MODELS symlink)
#
# Contract (unchanged from batch-2, card t_034e473e): POST /v1/audio/speech with ONLY
#   input (lyrics, section tags on their own lines), instructions (5-field caption),
#   seed, max_new_tokens (seconds x 25). Other fields rejected by the API.
set -u
LAYER=/home/jaita/sglang-omni
MODEL=/home/jaita/MODELS/MEDIA/MiniMax-Music3
PORT=${PORT:-8050}
LOG=${LOG:-/tmp/sgl-omni-music3-8050.log}

export PATH="$LAYER/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda
export CPATH=/home/jaita/pyinc/python3.12:/home/jaita/pyinc${CPATH:+:$CPATH}
export CUDA_VISIBLE_DEVICES=0
# Dump all Python thread stacks into the log on a fatal signal (kill -ABRT <pid>).
# Needed because yama ptrace_scope=1 blocks py-spy/gdb attach on this box.
export PYTHONFAULTHANDLER=1

# KV pool cap for the generation (AR) stage. sglang sizes this off
# torch.cuda.mem_get_info(), which on GB10 unified memory reports the DEVICE view
# (~77GB free at load begin) and ignores host-side use by the coexisting media lanes:
# left alone it grabbed a 159,728-token / 21.9GB KV pool, taking the lane to ~90GB of
# the 124GB box while tts/decider/yue2 neighbours still needed room. Lara's contract is
# serial single-client renders, so 32K tokens is ~14x the longest batch payload (2,250
# tokens) with real headroom left over.
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}

# fail loud rather than crash-loop into a dead lane
test -f "$HOME/pyinc/python3.12/Python.h" || { echo "MISSING Python.h in ~/pyinc/python3.12 — see header comment for the rsync"; exit 1; }
command -v ninja >/dev/null || { echo "MISSING ninja on PATH"; exit 1; }
test -f "$MODEL/flowmatching_vae.pth" || { echo "MISSING weights at $MODEL"; exit 1; }

cd "$LAYER" || exit 1
exec .venv/bin/sgl-omni serve \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  "$@" \
  >> "$LOG" 2>&1
