# Puzzle & Nemotron Super MTP Research — DGX Spark

**Source:** Subagent research (Jul 18, 2026)
**Key finding:** MTP IS possible on Spark — our OOM is from FlashInfer autotune, not drafter weights.

## Why We OOM'd

FlashInfer FP4 MoE autotune allocates huge workspace tensors on SM121 and OOMs. `--enforce-eager` doesn't help — it only disables CUDA graphs, not FlashInfer autotune.

## The Missing Fixes

1. **`drop_caches` before startup** — OS page cache eats ~44GB. `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` frees it to 100+GB.
2. **`--moe-backend marlin`** — bypasses FlashInfer MoE entirely (we had this)
3. **`moe_backend: triton` inside spec config** — drafter independently selects FlashInfer without this
4. **`--mamba-cache-mode=align`** — MTP spec decode incompatible with `all` mode
5. **`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`** — fixes allreduce on single-GPU
6. **Piecewise cudagraph mode** — reduces graph memory from 13GB to 5GB

## moranilt's Working Puzzle Config (HF Discussion #3)

- vLLM v0.23.0, MTP k=3, GMU 0.73, 160K context, 1 lane
- **32.2 tok/s, 69.4% acceptance, acceptance length 3.08**
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
- `--mamba-backend flashinfer`, `VLLM_USE_FLASHINFER_MOE_FP4=0`

## NVIDIA Official Nemotron Super SparkDeploymentGuide

- GitHub: `github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide`
- vLLM: `vllm/vllm-openai:cu130-nightly`
- MTP k=3, 4 lanes, 1M context, GMU 0.90
- Marlin backend, Triton MoE in spec path
- eugr got 26.5 tok/s with community Docker

## Memory Estimation (rmagur1203)

| cudagraph_mode | MTP k | Model | compile | CUDA graph | KV cache | Total | Status |
|---|---|---|---|---|---|---|---|
| NONE | 0 | 70GB | 0 | 0 | ~48GB | ~118GB | ✅ slow |
| PIECEWISE | 1 | 71GB | 6GB | 5GB | ~33GB | ~115GB | ✅ recommended |
| PIECEWISE | 3 | 71GB | 10GB | 8GB | ~24GB | ~113GB | ⚠️ tight but fits |
| PIECEWISE | 5 | 71GB | 15GB | 12GB | ~15GB | ~113GB | ❌ OOM |

k=3 fits at ~113GB — tight but within 121GB. The OOM at k=5 is from torch.compile activations, not drafter weights.

## Community Resources

| Resource | URL | What |
|---|---|---|
| NVIDIA SparkDeploymentGuide | github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide | Official MTP k=3 config |
| eugr/spark-vllm-docker | github.com/eugr/spark-vllm-docker | Community Docker recipes |
| rmagur1203/vllm-dgx-spark | github.com/rmagur1203/vllm-dgx-spark | Patches, 144-combo benchmarks |
| vLLM Issue #37754 | github.com/vllm-project/vllm/issues/37754 | FlashInfer+MTP+SM121 crash bug (OPEN) |

## Next Steps

1. Run `drop_caches` before startup
2. Add `--mamba-cache-mode=align` and `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`
3. Start with MTP k=2, then try k=3
4. Try piecewise cudagraph mode if available
5. Replicate moranilt's exact config for Puzzle