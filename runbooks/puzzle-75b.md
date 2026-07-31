# Recipe: Nemotron Puzzle 75B-A9B NVFP4

**Status:** 🧪 NEW — Community-validated recipe (joeynyc + VramJon + brian322), untested by us
**Served name:** `nemotron-puzzle-75b-nvfp4`
**Stack:** Docker — `nvcr.io/nvidia/vllm:26.06-py3` (NGC image, NOT public vllm/vllm-openai)
**Source:** https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark
**Community validation:** 3 independent users on DGX Spark (joeynyc, VramJon, brian322)

> **Recipe contract:** [`recipes/puzzle-75b.yaml`](../recipes/puzzle-75b.yaml)

## Why This Recipe Works (and our previous attempts didn't)

Our previous OOM crashes used `vllm/vllm-openai:v0.24.0` / `v0.25.0` which hit the FlashInfer PR #3738 regression. The NGC image `nvcr.io/nvidia/vllm:26.06-py3` (vLLM 0.22.1) auto-selects FLASHINFER_CUTLASS MoE backend instead of TRTLLM, avoiding the 20GB autotune workspace that caused our OOM. Two independent users (VramJon, joeynyc) confirm GMU 0.88 + 262K + MTP k=3 works with zero OOM.

## Container

| Key | Value |
|---|---|
| Image | `nvcr.io/nvidia/vllm:26.06-py3` |
| Name | `puzzle-spark` |
| Port | `8000` |
| GPU | all |
| Restart | `unless-stopped` |
| Volume | `$HOME/.cache/huggingface` → `/root/.cache/huggingface` |

### Environment

```bash
HF_HOME=/root/.cache/huggingface
NVIDIA_TF32_OVERRIDE=1
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## vLLM Serve Command

```bash
vllm serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
  --served-model-name nemotron-puzzle-75b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --mamba-backend flashinfer \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.88
```

## Performance (joeynyc's benchmarks, 2026-07-08)

| Metric | Value |
|---|---|
| Solo decode (mean) | 35.9 tok/s |
| 4-stream aggregate | 75.3 tok/s |
| MTP acceptance | ~74.7% |
| Prefix cache hit rate | 70.4% |
| KV cache tokens | ~2.0M |
| Model load | ~50 GiB |
| Available KV | ~55 GiB |
| TTFT (warm) | ~0.25s |

### Before/After flag diff

| Flag | Before (stock) | After (this recipe) | Why |
|---|---|---|---|
| `max_num_seqs` | 1 | 4 | Multi-session concurrency |
| `enable-prefix-caching` | off | on | Multi-turn agent TTFT |
| `max-num-batched-tokens` | default (~2048) | 8192 | Avoid MTP schedule bottleneck |
| `gpu-memory-utilization` | 0.85 | 0.88 | Slightly more KV headroom |
| MTP | 3 | 3 | Already strong accept rate |
| `max-model-len` | 262144 | 262144 | Keep long context |

## OOM Fallbacks

If OOM at boot:
1. Drop `gpu-memory-utilization` to `0.85`
2. Drop `max-model-len` to `131072`
3. Drop `max-num-seqs` to `2`
4. Drop `max-num-batched-tokens` to `4096`

## Client Tips

```json
{
  "model": "nemotron-puzzle-75b-nvfp4",
  "chat_template_kwargs": { "enable_thinking": false },
  "temperature": 0.3,
  "max_tokens": 2048
}
```

- **Thinking off** improves structured/code MTP accept and avoids empty `content` with huge `reasoning` fields.
- For pure peak solo decode: set `max_num_seqs=1`; aggregate multi-user will regress.

## Community Configs (for reference)

### moranilt (HF discussion #3) — 32.2 tok/s
- `vllm/vllm-openai:v0.23.0`, GMU 0.73, 160K, MTP k=3, `--kv-cache-dtype fp8`, `--max-num-seqs 8`, prefix caching on
- No `moe_backend` in spec config (auto-selects Triton for BF16 drafter)
- `--mamba_ssm_cache_dtype float16`, stochastic rounding, philox rounds 5

### brian322 (NVIDIA forum) — fast agentic, Hermes work
- `--moe-backend cutlass`, GMU 0.72, 262K, MTP k=1, `--calculate-kv-scales`, `--load-format instanttensor`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}'`
- `--default-chat-template-kwargs '{"enable_thinking":true,"force_nonempty_content":true}'`

### TheAwakenOne (NVIDIA forum) — 19 tok/s
- `--moe-backend marlin`, `VLLM_NVFP4_GEMM_BACKEND=marlin`, GMU 0.85, 131K, MTP k=3, `max_num_seqs 12`

## Sources

- joeynyc's repo: https://github.com/joeynyc/Nemotron-Puzzle-75B-NVFP4-1x-DGX-Spark
- HF discussion #3: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions/3
- NVIDIA forum: https://forums.developer.nvidia.com/t/nvidia-nvidia-nemotron-labs-3-puzzle-75b-a9b-nvfp4/376095
- Our OOM research: `research/puzzle-mtp-exact-fix.md`