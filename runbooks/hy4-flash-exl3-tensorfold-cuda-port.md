# Hy4-flash EXL3 on TensorFold v1.3.2 CUDA — port runbook (experimental)

**State (2026-11-21):** serving end-to-end on GB10, measured 10.7 tok/s decode, but output
degenerates into repeated-token loops after ~30 tokens → **not quality-cleared, do not use for
real work yet**. The Metal path upstream is correct; this doc records what the CUDA port is and
where the remaining bug lives.

## Why a port exists

TensorFold releases through v1.3.2 (2026-11-20) serve `hy_v4_flash` only on Metal (M5 Ultra
measured). The Hy4 CUDA model *code* exists upstream (family modules, and #4003/#4008 history:
an ExL3 4-bit load was proven on one GB10 at 85.4 GiB), but there was no CUDA `native` family
entry — the server's `cuda_families.zig` registry and the serve interface had no Hy4 implementation.
Metal-only is a policy gate upstream, not a technical one.

## What we built on top of v1.3.2

Worktree `/home/jaita/tensorfold/tf132` (from `ashhart/TensorFold v1.3.2`):

**New:**
- `zig/src/families/hy4_flash/cuda_native.zig` — the registry shim: single-sequence
  `lanes.Backend` over the family's CUDA model (512-row prefill chunks, host-side sampling like
  serve_metal.zig, 1-row verify windows, keep-rollback, warm pass over the whole window at open
  for UMA first-touch). Exports `model_type = "hy_v4_flash"`, `formats = {"exl3"}`.
- `zig/src/families/hy4_flash/cuda_family.zig` — module root exposing `native`.
- `zig/build/hy4_cuda.zig` + wiring into `zig/build/cuda_engines.zig`, `zig/build/cuda_kernels.zig`
  (hy4 engram fatbin), `zig/src/cuda/kernels.zig`.
- `zig/src/native/cuda_families.zig` — registry += the hy4 native.
- `zig/kernels/cuda/hy4_flash/engram.cu` — engram gather kernel (ported from the tfhy4
  hy4-flash-cuda branch worktree).

**Ported from the `hy4-flash-cuda` branch worktree (+ fixes):**
`cuda_model.zig` (+ Kernels.deinit / Model.deinit, + config.json EXL3 quantization_config
detection with `Hy4NotExl3` refusal), `engram_cuda.zig`, `exl3_weights.zig` (MCG codebook,
fp16 scales, requires `mul1` per its test — our pack is MCG, see caveat), `moe_exl3_cuda.zig`,
`bf16_rows.zig` (O_DIRECT engram row streaming — tables never uploaded), `mla_weights.zig`
(BF16 kv_b head split), `host_tests.zig`.

**info.zig:** EXL3 quantization detection (`exl3Quant`) + Hy4-aware memory floor that excludes
the engram tables: for our pack prints `format: exl3 (3.0-bit, mcg codebook)`, `memory floor:
63.57 GiB + 118.02 GiB engram streamed O_DIRECT`.

**cli/serve-side memory fit:** fixed to use the same floor (was pricing all 179.6 GiB resident).

## Build

```bash
cd /home/jaita/tensorfold/tf132
/home/jaita/tensorfold/zig-aarch64-linux-0.17.0/zig build \
  -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseSafe
# → zig-out/native/bin/tensorfold-native
```

## Serve (verified sequence)

```bash
/path/tensorfold-native serve /path/to/exl3-3.0bpw-dq2 --port 8508
# loads 87.52 GiB resident in ~270 s; 0.09 GiB per stream
# http://127.0.0.1:8508/v1 — OpenAI-compatible /v1/models, /v1/chat/completions
```

Pack choice: the **dq2** (BF16-dequantized dense) pack is the one that loads coherently. The
FP8-native pack loads but garbles from token 1 — the ported dense projection path reads
MLX affine-4/q8 pack conventions and misreads FP8-blockwise scales (host.words/scales shapes
come from the wrong geometry). Fixing that means adding an FP8-blockwise dense path to
`cuda_projection.zig` (dequant at load or a scaled kernel).

## The remaining quality bug

Symptom: first ~30 tokens correct, then endless single-token loops (`and, and, and, ...`),
also at temperature 0 → KV/position handling, not sampling. Debug evidence so far:
the rotary table phases step correctly per row but carry a large constant offset — position data
in the decode path looks stale/uninitialized; the probe (and the fix) is in `cuda_native.zig`'s
round loop (queue/read/verify/keep-rollback) and how it advances the kv-pool slots between rounds.
`mla_cuda` slot addressing is the next thing to compare against `serve_metal.zig`.

## Caveats

- `exl3_weights.zig` asserts `mul1` codebook; our 3.0bpw pack is MCG. The MoE path goes through
  the family's own trellis/MCG code (`gate_trellis/up_trellis` logs show it reading the pack), so
  the assert is about the alternative kernel — but record it before reuse.
- Load time ~4.5 min (NFS from spark2); local disk will be faster.
- GPU exclusivity: this lane is 87.5 GiB resident — same one-lane-at-a-time rule as vLLM.
