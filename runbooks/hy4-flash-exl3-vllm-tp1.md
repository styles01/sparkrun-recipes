# Hy4-flash EXL3 3.0-bit — vLLM TP1 on GB10 (runbook)

**Published code:** the drop-in vLLM overlay (patched `vllm_exl3` + Hy4 model tree, FP8 pack
converter chain `pack_build/`, launcher) lives at
[`styles01/hy4-exl3-vllm-serving`](https://github.com/styles01/hy4-exl3-vllm-serving) — the
companion **TensorFold CUDA port** fork is
[`styles01/tensorfold-hy4-cuda`](https://github.com/styles01/tensorfold-hy4-cuda). Base pack:
[`aitamedia/Hy4-flash-EXL3-3.0bpw`](https://huggingface.co/aitamedia/Hy4-flash-EXL3-3.0bpw).
Files referenced below by estate path (`/home/jaita/...`) are also in that repo's `overlay/`.

**Lane shape:** one DGX Spark GB10, vLLM `vllm/vllm-openai:hy4-flash` + the `vllm_exl3` plugin
(vcruz305-lineage build, `v0.1.dev25463`), Tencent Hy4-flash (model type `hy_v4_flash`) at
3.0-bit EXL3 (MCG codebook) routed experts, **FP8-native dense linears**, 262144 max context,
`--max-num-seqs 3` (c3), engram multi-head-embedding table streamed from disk.

**Verified numbers (2026-11-21, single Spark, TP1):**

| Metric | Value |
|---|---|
| Single-stream decode (no spec) | **16.1 tok/s** (16.12/16.14/15.46 across 3 runs) |
| c3 concurrency | 12.9–13.0 tok/s per stream, **34.3 tok/s aggregate** |
| TTFT | 240–420 ms short prompts |
| Coherence | 261+ token generations clean, no loops |
| Spec decode (DFly2, depth 3/5) | **net-negative** — see drafter-loader bug below |

Lane best path: 4.3 (eager, first boot) → 9.4 (graphs on, spec off) → 13.7 (IDX schedule cache
+ zz_oracle.pth) → **16.1 (FP8-native dense pack)**.

## The pack

Base: `aitamedia/Hy4-flash-EXL3-3.0bpw` (198.3 GB, 102 files). Two build stages on top:

1. **Loader root cause:** vLLM's exl3 loader skipped `weight_scale_inv` on non-expert linears
   (~2785× scale error → garble). First pack `exl3-3.0bpw` dequantized 153 dense tensors to BF16
   (dq2) to sidestep. Re-upload fixed pack: 2026-11-19T22:24Z.
2. **FP8-native dense (the 16.1 tok/s pack, `exl3-3.0bpw-fp8lin`):** dense linears stored
   FP8 with proper 128×128 `weight_scale_inv` in-pack, experts untouched EXL3. Built from the
   verified pack copy (254 GB → spark2, then converted ~9 min/shard, 383 dense tensors).

Build chain (scripts in the estate scratch, reproducible):

- `convert_pack.py` — per-shard: for every BF16 dense linear in
  `layers.N >= 1` (attn `o_proj/q_b_proj/linear_gate`, `kv_b_proj`, mlp gate/up/down,
  `hidden_correction`, `eh_proj`, shared + routed expert companions), quantize dynamic →
  FP8 e4m3 with block scales `(ceil(N/128), ceil(K/128))`, write new shard + index entries.
- **Must stay BF16** (hard-won list):
  - `vit.*` / visual tower — the vision loader has no FP8 path.
  - indexer projections (`token_key/query/key_value/key_gate/head_weight/weights_proj`) — FP32-correctness path.
  - `kv_a_proj_with_mqa` — N=576 not 128-aligned; cutlass rejects. Restore from clean copy + drop scales.
  - `q_a_proj` **and** `fused_qkv_a_proj` — the merged `fused_qkv_a_proj` layer takes ONE
    quant_config for both halves; must be unquantized. Matcher must exclude the name itself.

## Boot flags that matter (GB10 sm_121)

- `--kernel-config '{"linear_backend":"torch","linear_backend_per_quant":{"fp8_block_w8a8":"cutlass"}}'`
  — cutlass is the **only** FP8-blockwise kernel on sm_121; torch's DeepSeek-style block kernel is
  SM90-only. Pin it or a profile-run init will crash with `cutlass_gemm_caller Invalid status` /
  `Failed to find a kernel`.
- `--engram-config '{"cpu_offload":false,"disk_offload":true,"disk_threads":256}'` — engram rows
  stream from disk, never resident.
- `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` — graphs ~2× over eager.
- `VLLM_USE_DEEP_GEMM=0` — DeepGEMM JIT crashes during capture on sm_121.
- `VLLM_FLASH_FP8_NONEXPERT=1` + `VLLM_FLASH_FP8_ALL=1` — routes FP8-native linears through
  `Fp8LinearMethod` blockwise; matcher exclusions live in `vllm_exl3/exl3.py::_matches_flash_fp8_nonexpert`.
- `zz_oracle.pth` written into image `dist-packages/:` spawned EngineCore processes don't inherit
  `PYTHONPATH`; the .pth forces the overlay import everywhere.
- `--speculative-config null` until the drafter bug is fixed (below).

Full launcher: `scripts/launch/hy4-tp1.sh` in this repo (paths are estate-specific; keep the argv
continuation chain intact — one orphaned `\` silently drops later flags).

**DFly spec retune (2026-11-22):** scale-loader fix landed → acceptance 1.96. Depth sweep:
d1=12.5 · d2=10.9 · **d3=16.9 tok/s (beats no-spec 16.1)** · d4=9.2 · d7=7.5. The spec lane
launcher is `hy4-tp1-spec-d3.sh` in the overlay repo. **Default stays no-spec** because d3 only
wins marginally on this prompt class; the dsv41-style adaptive policy (per-round cost table +
early exit) is the intended next step before making spec canonical.

DFly2 = 5 trained draft layers (targets 25/29/33/37/41), KV-share + index-share, greedy drafts only.
Card's own recipe: `--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}'` — matches what
we ran. Symptom on the FP8-native pack: **Mean acceptance length 1.00, 0.00 accepted t/s, ~30 drafted
t/s** — the drafter produces tokens the target never accepts. Root cause: `vllm/models/hy_v4/nvidia/dfly.py`
skips every `weight_scale_inv` companion then raw-casts fp8→bf16 without scale application (third instance
of the same loader-bug pattern). A scale-aware dequant fix is in our patched tree (`scales` dict +
block dequant at load); acceptance didn't recover in one retest — the weights may also be routed
through a second loader path (probe prints `DFLY-WDBG` show fp8-typed tensors yielding directly into
bf16-param modules). **Until acceptance >1.0 is measured, keep spec off.**

## Pitfalls learned (each cost a boot cycle)

- Serve-side GPU memory fit: TF/vLLM must price engram tables + non-resident parts out of the fit check,
  not check raw checkpoint bytes.
- NFS on spark2 with `no_root_squash` + `fsid` unique per export; mount ro.
- One GPU lane at a time: vLLM resident leaves ~1.9 GiB — kill the resident lane before any other load.
- TRASH-ONLY cleanup — `mv` to `~/trash`, never `rm` on these boxes.
- `docker logs` are UTC, file mtimes are local (~UTC-6): mind `find -newermt` correlations.
- `VLLM_ENGRAM_ZERO=1` (engram debug: zeroed rows) is a debug env — quality requires the real table.
