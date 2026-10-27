#!/bin/bash
# nemf-drama3-build.sh — spark2 overnight Drama3 lane builder (fires 3am ET via cron trigger).
# Gates: waits for DRAMA3-STAGED-CLEAN in /tmp/nemf-drama3.log (checkpoint staged after
# the HF upload). Runs: nvcc fatbins inside nvcr container → zig build → install to
# $HOME/.local/opt/tensorfold-studio/drama3 → codec_convert → guarded first takes
# (drama3 vs chatterbox pairs, same lines, RAW output) → report.
# CPU-heavy compile only (nvcc/zig never touch the GPU); takes run under run_guarded's
# lock + mem-guard. Idempotent — every step skips itself if its output exists.
exec >> /tmp/nemf-drama3-build.log 2>&1
echo $$ > /tmp/nb.pid
echo "=== drama3 BUILD chain start $(date -u) ==="
set -u

# ── gate 1: checkpoint staged ──
echo "-- gate 1: waiting for DRAMA3-STAGED-CLEAN ..."
for i in $(seq 1 60); do
  grep -q "DRAMA3-STAGED-CLEAN" /tmp/nemf-drama3.log 2>/dev/null && break
  grep -q "GATE-BLOCKED" /tmp/nemf-drama3.log 2>/dev/null && { echo "FATAL: stager gate-blocked; BUILD ABORTED"; exit 1; }
  grep -q "DRAMA3-STAGE-INCOMPLETE" /tmp/nemf-drama3.log 2>/dev/null && { echo "FATAL: stager incomplete; BUILD ABORTED"; exit 1; }
  sleep 600
done
grep -q "DRAMA3-STAGED-CLEAN" /tmp/nemf-drama3.log && echo "gate 1 passed" || { echo "FATAL: gate 1 timeout (10h)"; exit 1; }

CKPT=/home/jaita/MODELS/MEDIA/Drama3
T=~/tf-drama3
B=~/zig-port/d3build
PREFIX=$HOME/.local/opt/tensorfold-studio
mkdir -p $B/fatbin $PREFIX/drama3/bin $PREFIX/drama3/lib

# ── step 2: fatbins (nvcc INSIDE the nvcr container — spark2 host has no nvcc) ──
if [ ! -f $B/fatbin/drama3_codec.fatbin ]; then
  echo "-- step 2: nvcc fatbins in container ($(date -u))"
  HF_FLAGS="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121"
  docker run --rm -v /home/jaita:/hf --entrypoint bash nvcr.io/nvidia/pytorch:26.07-py3 -c "set -e; cd /hf/tf-drama3; \
    nvcc $HF_FLAGS -o /hf/zig-port/d3build/fatbin/drama3.fatbin zig/kernels/cuda/drama3.cu; \
    nvcc $HF_FLAGS -diag-suppress 177 -o /hf/zig-port/d3build/fatbin/drama3_fp8.fatbin zig/kernels/cuda/drama3_fp8.cu; \
    nvcc $HF_FLAGS -o /hf/zig-port/d3build/fatbin/drama3_codec.fatbin zig/kernels/cuda/drama3_codec.cu; \
    echo fatbins-ok"
  echo "fatbins rc=$? $(date -u)"
else
  echo "-- step 2: fatbins already built (skip)"
fi

# ── step 3: zig build → drama3-generate + lib (CPU-only) ──
if [ ! -x $B/zig-out/bin/drama3-generate ]; then
  echo "-- step 3: zig build ($(date -u))"
  cd $T
  export ZIG_LOCAL_CACHE_DIR=$B/cache/zig-local ZIG_GLOBAL_CACHE_DIR=$B/cache/zig-global
  nice -n 10 ~/opt/zig-0.17.0/zig build -Dfatbins=$B/fatbin -Dsm=121 -Doptimize=ReleaseFast --prefix $B/zig-out -j4
  echo "zig build rc=$? $(date -u)"
else
  echo "-- step 3: zig build exists (skip)"
fi
[ -x $B/zig-out/bin/drama3-generate ] || { echo "FATAL: no drama3-generate after build"; exit 1; }

# ── step 4: install ──
cp -a $B/zig-out/bin/drama3-generate $PREFIX/drama3/bin/ 2>/dev/null || cp -a $B/zig-out/drama3-generate $PREFIX/drama3/bin/
LIBSRC=$(ls $B/zig-out/lib/libtensorfold_drama3.so $B/zig-out/lib/*.so 2>/dev/null | head -1)
[ -n "$LIBSRC" ] && cp -a "$LIBSRC" $PREFIX/drama3/lib/libtensorfold_drama3.so
ls -la $PREFIX/drama3/bin $PREFIX/drama3/lib

# ── step 4b: engine usage dump (informs any flag diagnosis later) ──
mkdir -p $HOME/drama3
$PREFIX/drama3/bin/drama3-generate --help 2>&1 | head -40 || true

# ── step 5: codec conversion (needs torch → inside the container; GPU untouched) ──
CODEC=$PREFIX/drama3/codec-v1.safetensors
if [ ! -f "$CODEC" ]; then
  echo "-- step 5: codec convert in container ($(date -u))"
  docker run --rm -v /home/jaita:/hf --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 \
    /hf/tf-drama3/tools/drama3/codec_convert.py /hf/MODELS/MEDIA/Drama3 /hf/.local/opt/tensorfold-studio/drama3/codec-v1.safetensors
  echo "codec rc=$? $(date -u)"; du -sh "$CODEC" 2>/dev/null
else
  echo "-- step 5: codec exists (skip)"
fi

# ── step 6: guarded first takes (drama3, then chatterbox pairs for A/B) ──
TAKES=$HOME/media/drama3/takes-$(date -u +%m%d)
mkdir -p "$TAKES"
GUARD=$T/tools/drama3/run_guarded.sh
BIN=$PREFIX/drama3/bin/drama3-generate
MODEL=$CKPT

say_once() { # $1 label $2 text
  OUT=$TAKES/d3-$1.wav
  [ -s "$OUT" ] && { echo "  (skip exists: $OUT)"; return; }
  bash $GUARD $BIN --model "$MODEL" --codec "$CODEC" --text "$2" --out "$OUT" --seed 1 \
    2>>/tmp/nemf-drama3-build.log | tail -2
  echo "take $1 rc=$? size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)"
}

say_cbf() { # $1 label $2 text — chatterbox arm via its HTTP server (RAW output)
  OUT=$TAKES/cb-$1.wav
  [ -s "$OUT" ] && { echo "  (skip exists: $OUT)"; return; }
  curl -s -m 120 -X POST http://127.0.0.1:8020/v1/audio/speech -H "Content-Type: application/json" \
    -H "Accept: audio/wav" -o "$OUT" \
    -d "{\"input\":\"$2\"}" && echo "cb take $1 size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)"
}

L1="Hello there. This is Drama three, speaking locally on your Spark. No cloud was involved."
L2="The estate watched every layer tonight, and the pack landed perfectly. All four lanes report green."
L3="<|speaker:0|> [whispering] We should not wake him. He has been driving all evening.<|speaker:1|> [sighs] Then we keep our voices down, and the machine keeps its own counsel."

echo "-- step 6: takes ($TAKES) $(date -u)"
say_once hello "$L1"
say_cbf hello "$L1"
say_once status "$L2"
say_cbf status "$L2"

# 2-speaker directed take through the stdio serve mode (server pattern, then kill)
echo "-- step 6b: serve-mode multi-speaker take $(date -u)"
( bash $GUARD $BIN --model "$MODEL" --codec "$CODEC" --serve 1 \
    <<< '{"text":"'"$L3"'","out":"'"$TAKES/d3-two-voices.wav"'","seed":1}' \
    2>>/tmp/nemf-drama3-build.log | tail -1 ) || echo "serve-mode take rc=$? (non-fatal, reported)"
echo "serve take size=$(stat -c%s $TAKES/d3-two-voices.wav 2>/dev/null || echo 0)"

# ── step 7: QC echo (peak/peak_dBFS/clipped per yue2 contract definition) ──
python3 - <<'PY' 2>&1 | tail -12
import glob, struct, wave, os
d = os.path.expanduser("~/media/drama3/takes-*")
for p in sorted(glob.glob(d + "/*.wav")):
    try:
        import wave
        w = wave.open(p)
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        frames = w.readframes(n)
        peak = max(abs(s) for i in range(0, len(frames), 2*ch) for s in [int.from_bytes(frames[i:i+2], "little", signed=True)])
        clipped = sum(1 for i in range(0, len(frames), 2*ch) if abs(int.from_bytes(frames[i:i+2], "little", signed=True)) >= 32760)
        dbfs = 20 * __import__("math").log10(peak / 32768) if peak else -99.0
        print(f"{os.path.basename(p)}: {sr}Hz {ch}ch {n/sr:.2f}s peak {peak} ({dbfs:.2f} dBFS) clipped {clipped}")
    except Exception as e:
        print(os.path.basename(p), "QC-FAIL", repr(e)[:120])
PY

echo "=== drama3 BUILD chain end $(date -u) ==="
echo "BUILD-DONE marker: takes at $TAKES"