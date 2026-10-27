#!/bin/bash
# fatbin-zig-build.sh — complete the drama3 CUDA engine build on spark2.
# Builds ALL engine fatbins the zig embed expects (42 kernels from the drama3-cuda
# fork's zig/kernels/cuda/), then runs the zig build with the drama3-generate target
# and installs to $HOME/.local/opt/tensorfold-studio/drama3/. CPU-only (nvcc/zig/GPU-free).
exec >> /tmp/fatbin-zig-build.log 2>&1
echo "=== fatbin factory start $(date -u) ==="
set -u
T=/home/jaita/tf-drama3
B=/home/jaita/zig-port/d3build
HF_FLAGS="-fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121"
mkdir -p $B/fatbin $B/zig-out

docker run --rm -v /home/jaita:/hf --entrypoint bash nvcr.io/nvidia/pytorch:26.07-py3 -c '
set -u
cd /hf/tf-drama3
FLAGS="-fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121"
fails=0; ok=0
for SRC_DIR in zig/kernels/cuda zig/kernels/cuda/torch_ops; do
  for CU in "$SRC_DIR"/*.cu; do
    BN=$(basename "$CU" .cu)
    OUT=/hf/zig-port/d3build/fatbin/${BN}.fatbin
    [ -s "$OUT" ] && { ok=$((ok+1)); continue; }
    EXTRA=""
    [ "$BN" = "drama3_fp8" ] && EXTRA="-diag-suppress 177"
    if nvcc $FLAGS $EXTRA -o "$OUT" "$CU" 2>"/tmp/nvcc-${BN}.err"; then
      echo "OK   $BN (~$(du -m $OUT | cut -f1)M)"; ok=$((ok+1))
    else
      echo "FAIL $BN"; tail -3 "/tmp/nvcc-${BN}.err"; fails=$((fails+1))
    fi
  done
done
echo "fatbin loop done: ok=$ok fails=$fails"
' 
echo "kernel phase rc=$? $(date -u)"

# zig build: drama3-generate target (+ install)
cd $T
export ZIG_LOCAL_CACHE_DIR=$B/cache/zig-local ZIG_GLOBAL_CACHE_DIR=$B/cache/zig-global
nice -n 10 ~/opt/zig-0.17.0/zig build -Dfatbins=$B/fatbin -Dsm=121 -Doptimize=ReleaseFast --prefix $B/zig-out -j4 drama3-generate
echo "zig rc=$? $(date -u)"
[ -f $B/zig-out/bin/drama3-generate ] || [ -f $B/zig-out/drama3-generate ] || { echo "FATAL: drama3-generate still missing"; tail -30 $B/cache 2>/dev/null; exit 1; }
mkdir -p /home/jaita/.local/opt/tensorfold-studio/drama3/{bin,lib}
cp -a $B/zig-out/bin/drama3-generate /home/jaita/.local/opt/tensorfold-studio/drama3/bin/ 2>/dev/null || cp -a $B/zig-out/drama3-generate /home/jaita/.local/opt/tensorfold-studio/drama3/bin/
LIBSRC=$(ls $B/zig-out/lib/libtensorfold_drama3.so $B/zig-out/lib/*.so 2>/dev/null | head -1)
[ -n "$LIBSRC" ] && cp -a "$LIBSRC" /home/jaita/.local/opt/tensorfold-studio/drama3/lib/libtensorfold_drama3.so
ls -la /home/jaita/.local/opt/tensorfold-studio/drama3/bin /home/jaita/.local/opt/tensorfold-studio/drama3/lib
echo "=== fatbin factory end $(date -u) ==="