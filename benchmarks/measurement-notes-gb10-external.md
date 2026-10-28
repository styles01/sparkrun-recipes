# External GB10 measurement notes (GH #2, jschmied — absorbed 2026-10-28)

Three single-box measurements shared on the repo issues thread, absorbed into the
docs because they bear directly on our kernel-patch and bench-claims culture.
Reproduce his tooling: https://github.com/jschmied/qwen38-flash-next-gb10 (shapebench.py).

## 1. The GB10 noise floor is ~6.9% (single runs ±4.6% at 95%)

Six identical runs (c=1, i4000/o512): mean 35.68, sd 0.84, spread 6.9%. Confirmed
independently by 0xBakeer on llama.cpp with mincore(2)-verified residency — a box
property, not an engine quirk.

**Rule: nothing below ~10% is callable from a single run.** Any tuned-submission
"win" claimed off single runs (38.0 vs 36.8 class) is inside spread. This exact
trap cost him a published k-optimal claim. Run repeats (n≥4, report mean+sd) for
anything we publish; floor-round per the owner's standing rule on top of that.

## 2. BF16 hyper-connections: ~27% of decode time, latency-bound, NOT fixable by precision

Decode-only profile on Flash-Next: 26.6% of GPU kernel time in
`cutlass_80_wmma_tensorop_bf16` = the ~102,000 hyper-connection projections
(~200/forward, ~300MB rotating weights — L2 always defeated). They sit at ~78% of
roofline already: M=1 BF16 up 30.8µs vs 24.0 roofline; the down-proj is 37.0µs at
M=1 and gets SLOWER in FP8 (0.89x). All three interventions (blockwise FP8,
CUTE-DSL skinny GEMM, per-channel FP8) = null vs the noise floor, n=6 per arm.
hn7305 ships NVFP4 hyper-connection quantization disabled as net-negative.
dolf3131 reached the same null by kernel selection.

**Rule: don't spend a window quantizing hyper-connections** — the win is ~4µs/call
on `up`, a ~6–9µs/call loss on `down`, the halves cancel. The latency-bound shapes
are at roofline; only weight-streaming changes (fewer bytes moved per layer) help.

### The benchmark-integrity trap that cost him a benchmark arm

⚠️ **Patching vLLM source does NOT invalidate the torch.compile cache** — a stale
graph replays the unpatched code and every arm agrees to within 0.2%. Benchmark
kernel patches from EMPTY `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`. Config
flags ARE hashed correctly (checked). Our `docker/patch_*.py` build-time patches
are unaffected (baked into the image), but any RUNTIME patch A/B must nuke both
caches between arms.

## 3. lm_head quantization compounds with speculation — measured both ways

Same checkpoint, MTP k=2, c=1, i4000/o512:

| | BF16 head | FP8 head |
| --- | --- | --- |
| decode tok/s | 30.60 (n=4) | **36.45** (n=6, sd 1.04) |
| draft acceptance | **60.3%** | 56.6% |

FP8 head = +19% decode, −6% (relative −3.7pp) acceptance — the head speedup feeds
the verify step, but the softer logits cost draft hits. Net still wins at k=2.
Bears on `docker/patch_int8_lmhead_v3.py` + the v26 dflash-int8 recipe: the int8
or fp8 head choice is NOT free of speculation interplay — re-measure acceptance,
not just wall decode, whenever the head precision changes (and mind §1's floor:
19% > 6.9% spread is callable, smaller diffs are not).