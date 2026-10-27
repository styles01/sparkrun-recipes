# Recipe: Laguna S 2.1 NVFP4 + DFlash

**Status:** ✅ PROD — Updated July 22, 2026 (k=7, 3 lanes, GMU 0.85)
**Served name:** `laguna`
**Stack:** Docker — `vllm/vllm-openai:v0.25.0`
**Model:** `poolside/Laguna-S-2.1-NVFP4` (67GB, NVFP4 compressed-tensors)
**Drafter:** `poolside/Laguna-S-2.1-DFlash` (2.1GB, EAGLE-style, shares embeddings + lm_head with target)
**License:** OpenMDW-1.1 (commercial use OK)

> **Recipe contract:** [`recipes/laguna-s-2.1.yaml`](../recipes/laguna-s-2.1.yaml)

## Model Specs

| Spec | Value |
|---|---|
| Total params | 118B |
| Active per token | 8B |
| Experts | 256 routed (top-10) + 1 shared |
| Layers | 48 (12 global attention, 36 SWA window=512) |
| Attention | GQA, 8 KV heads, head_dim 128, softplus gating |
| Context window | 1,048,576 (1M native) |
| Vocab | 100,352 |
| Reasoning | Native, interleaved thinking, per-request `enable_thinking` |
| Tool calling | `poolside_v1` parser |
| Reasoning parser | `poolside_v1` |

## Container Config

| Key | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.25.0` (REQUIRED — 0.24.0 doesn't support Laguna) |
| Name | `laguna-spark` |
| Port | 8000 |
| GPU | all |
| Restart | `unless-stopped` |
| Volume | `~/models/Laguna-S-2.1-NVFP4` → `/model:ro` |
| Volume | `~/models/Laguna-S-2.1-DFlash` → `/drafter:ro` |
| Volume | `~/.cache/huggingface` → `/root/.cache/huggingface` |

### ⚠️ Entrypoint Note

The `vllm/vllm-openai` image has `["vllm", "serve"]` baked in as the entrypoint. Do NOT pass `vllm serve /model` — that becomes `vllm serve serve /model` (double serve). Pass `/model --flags...` directly.

### Environment

```bash
HF_HOME=/root/.cache/huggingface
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

## vLLM Serve Command (current production)

```bash
docker run -d \
  --name laguna-spark \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/models/Laguna-S-2.1-NVFP4:/model:ro \
  -v $HOME/models/Laguna-S-2.1-DFlash:/drafter:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.25.1 \
  /model \
    --served-model-name laguna \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 300000 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 3 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser poolside_v1 \
    --reasoning-parser poolside_v1 \
    --speculative-config '{"model":"/drafter","num_speculative_tokens":7,"method":"dflash"}'
```

## Flag Rationale

| Flag | Value | Why |
|---|---|---|
| `vllm/vllm-openai:v0.25.1` | Required | 0.25.1+ has `LagunaForCausalLM` + `DFlashLagunaForCausalLM` in registry |
| `--max-model-len 300000` | 300K | 3.06× concurrency. SWA layers cache less, so KV scales slower |
| `--max-num-seqs 3` | 3 lanes | 3.19× KV headroom — fits with ~18GB free for co-located workloads |
| `--gpu-memory-utilization 0.85` | 0.85 | ~31GB KV cache. ~18GB free for OS + TTS/image gen + sparkDash |
| `--kv-cache-dtype fp8` | fp8 | Halves KV memory, enables more concurrency |
| `--attention-backend flashinfer` | Required | Triton degrades at batch ≥12 |
| `--enable-prefix-caching` | ON | Shared system prompts → high hit rate |
| `--tool-call-parser poolside_v1` | Required | Poolside's native tool call format |
| `--reasoning-parser poolside_v1` | Required | Interleaved thinking support |
| `--speculative-config dflash k=7` | DFlash | 7 draft tokens per step. Poolside's recommended sweet spot — ~5 accepted tokens/pass (34% acceptance, 2.3-3.7× speedup per Poolside benchmarks) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Required | 260K > model's `max_position_embeddings` |
| `--enable-auto-tool-choice` | ON | Auto-detect tool calls in output |

## Performance (actual measured)

| Metric | Value |
|---|---|
| Model load | 69.45 GiB (67GB weights + 2.1GB drafter + overhead), ~7.5 min |
| KV cache | 32.07 GiB → 918,419 tokens |
| Max concurrency | 3.06× at 300K (3 lanes) |
| KV per token | ~37.8 KB (SWA layers cache only 512 tokens, not full context) |
| Free RAM | ~18 GB (room for TTS or image gen co-location) |
| NVFP4 backend | FLASHINFER_CUTLASS (auto-selected) |
| DFlash k=7 acceptance | ~5 tokens/pass (Poolside benchmark data, 34% acceptance at k=15) |
| Throughput speedup | 2.3-3.7× (concurrency 1), 1.7-2.7× (concurrency 4+) |

## Benchmark Scores

| Benchmark | Laguna S 2.1 | Qwen 3.5 122B | Gap |
|---|---|---|---|
| **Terminal-Bench 2.1** | **70.2%** | 49.4% | **+20.8** |
| **SWE-bench Multilingual** | **78.5%** | 72.0% | **+6.5** |
| **SWE-Bench Pro** | **59.4%** | — | — |
| **DeepSWE** | **40.4%** | — | — |
| **Toolathlon Verified** | **49.7%** | — | — |

## Switch Script

`scripts/switch-to-laguna.sh` — one-command deploy with ADR-006 pre-flight.
Deployed to Spark at `~/switch-to-laguna.sh`.

```bash
# Default (260K, 3 lanes, k=7, GMU 0.85):
ssh user@<spark-host> 'bash ~/switch-to-laguna.sh'

# Custom config:
ssh user@<spark-host> 'MAX_MODEL_LEN=196608 MAX_NUM_SEQS=2 NSPEC=7 bash ~/switch-to-laguna.sh'
```

## OOM Fallbacks

1. Drop `--max-num-seqs` to 2 → frees ~10GB KV headroom
2. Drop `--max-model-len` to 196K (still 2.3× concurrency at 3 lanes)
3. Drop DFlash (remove `--speculative-config`) — saves ~2.5GB but loses 2.3-3.7× speedup
4. If co-located workload OOMs: drop GMU to 0.82 → ~28GB KV → still fits 3×260K

## Download

```bash
# NVFP4 model (67GB, 14 safetensors shards)
hf download poolside/Laguna-S-2.1-NVFP4 --local-dir ~/models/Laguna-S-2.1-NVFP4

# DFlash drafter (2.1GB, 5 files)
hf download poolside/Laguna-S-2.1-DFlash --local-dir ~/models/Laguna-S-2.1-DFlash
```

## Verify

```bash
curl -s http://<spark-host>:8000/v1/models | jq '.data[].id'
# expect: laguna

curl -s http://<spark-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna","messages":[{"role":"user","content":"Write a Python function to check if a string is a palindrome."}],"max_tokens":500}' \
  | jq '.choices[0].message.content'
```

## Comparison to Previous Models

| Model | Context | Lanes | Speed | Terminal-Bench | Notes |
|---|---|---|---|---|---|
| **Laguna S 2.1** | **300K** | **3** | TBD | **70.2%** | Purpose-built agentic coding, DFlash k=7 |
| Qwen 122B DFlash | 165K | 3 | 82.8 tok/s | 49.4% | Previous daily driver |
| Qwen 35B | 100K | 6 | 107 tok/s | 40.5% | Fast but hallucinates |
| MedGemma 27B | 5K | 75 | 336 tok/s | — | Corpus formatting only |
| Puzzle 75B | 300K | 3 | 11 tok/s | — | Weakest of the four |

## Sources

- Model card: https://huggingface.co/poolside/Laguna-S-2.1
- NVFP4: https://huggingface.co/poolside/Laguna-S-2.1-NVFP4
- DFlash: https://huggingface.co/poolside/Laguna-S-2.1-DFlash
- Blog: https://poolside.ai/blog/introducing-laguna-s-2-1
- License: OpenMDW-1.1 (https://openmdw.ai/)
- Trajectories: https://trajectories.poolside.ai