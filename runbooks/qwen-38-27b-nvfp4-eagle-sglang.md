# Recipe: Qwen 3.8 27B NVFP4 + MTP EAGLE on SGLang

**Status:** ✅ Experimental working (accept 2.45-3.88, ~27 tok/s with thinking ON)
**Served name:** `qwen38-27b`
**Port:** 30000
**Image:** `lmsysorg/sglang:qwen38-27b` (pinned, ships modelopt)

> **Recipe contract:** [`recipes/qwen-38-27b-nvfp4-eagle-sglang.yaml`](../recipes/qwen-38-27b-nvfp4-eagle-sglang.yaml)
> **Source:** [@calneymgp tweet](https://x.com/calneymgp/status/2088822975550071292) → `calneymgp/Qwen3.8-27B-NVFP4-lmhead4-recipe`

## Why EAGLE (not DSpark)

EAGLE drives the model's **native in-checkpoint MTP head**. It **aligns with the model** — acceptance 2.45-3.88 mean draft length on thinking-on traffic. DSpark (external drafter) collapses to ~1-2 on thinking-on prose (28% → ~8-15 tok/s). **DSpark is the wrong method for a thinking-on agent workload.**

## Why SGLang (not vLLM) for MTP

The fused GDN decode kernel requires `num_v_heads == 8*num_k_heads`. This model is 48/16=3, so vLLM's check fails and every decode step silently drops to Triton — a **net 3.6× loss**. SGLang does not have this restriction. Use SGLang.

## Setup (one-time, on Spark)

```bash
# 1. Clone the recipe
git clone https://huggingface.co/calneymgp/Qwen3.8-27B-NVFP4-lmhead4-recipe && cd "$(basename "$_")"

# 2. Download model (~22 GB, RadixArk NVFP4, resumable)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RadixArk/Qwen3.8-27B-NVFP4', local_dir='models/qwen38')"

# 3. lm_head quantization — NO-OP for RadixArk (already ships quantized lm_head). Verify:
#    config.json has lm_head.input_scale / weight_scale keys → already done.

# 4. Copy to lmhead4 dir
mkdir -p models/qwen38-lmhead4 && cp -r models/qwen38/* models/qwen38-lmhead4/
```

## Serve (adapted: 256K, 4 seqs, prefix cache ON, mem-frac 0.80)

```bash
docker run -d --name sglang-qwen38 --restart unless-stopped --gpus all --ipc host \
  -p 127.0.0.1:30000:30000 -v "$(pwd)/models/qwen38-lmhead4:/model:ro" \
  lmsysorg/sglang:qwen38-27b python3 -m sglang.launch_server \
    --model-path /model --served-model-name qwen38-27b \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --attention-backend triton --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.80 --context-length 262144 --max-running-requests 4 \
    --mamba-ssm-dtype bfloat16 --chunked-prefill-size 2048 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --trust-remote-code --host 0.0.0.0 --port 30000
```

## Verification

```bash
# Recipe's own agentic benchmark (segmented by workload + accept length)
python3 scripts/bench/agentic_bench.py --port 30000
```

Expected (measured Aug 17, thinking ON, RadixArk NVFP4): prose 22.7, code 32.0, tool_call 23.9, tool_multi 30.9, thinking 26.0, **average ~27 tok/s**, accept 2.45-3.88.

## Key settings (matched to drowzeys where we could)

| Setting | Value | Why |
|---|---|---|
| served-model-name | qwen38-27b | match Loca |
| context | 262144 | 256K, what we ran |
| max-running-requests | 4 | 4 seqs |
| mem-fraction-static | 0.80 | fp8 KV keeps pool big, safe headroom |
| radix cache | ON (flag removed) | prefix caching; drowseys measured 53% hit |
| kv-cache-dtype | fp8_e4m3 | fp8 KV |

## Caveats

- **~27 tok/s is our GB10 ceiling**, not Calney's 120-151 (that was an RTX 5090 sm_120 with different bandwidth). EAGLE acceptance is correct; the tok/s is hardware-bound.
- **256K + 4 concurrent** may exceed the KV pool (4×256K = 1M tokens). `max-model-len` stays 256K as per-request ceiling; concurrent long contexts share the pool (same as drowzeys).
- RadixArk already ships quantized lm_head — the recipe's `02-quantize-lmhead.py` is a no-op.
