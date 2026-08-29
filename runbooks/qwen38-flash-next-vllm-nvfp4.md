# Runbook: Qwen3.8-Flash-Next vLLM + NVFP4 + MTP (alternative recipe)

> **ALTERNATIVE — NOT YET DEPLOYED.** This is the 0xBakeer vLLM long-context
> recipe, documented as an alternative option. It is **complementary** to our
> llama.cpp Q3_K_XL 3-lane primary recipe:
> - **vLLM + NVFP4 wins** on chat, reasoning, and anything that reads long
>   documents — flat decode at depth, ~5× faster prefill, native MTP head, and
>   **96-109 tok/s aggregate at 16 concurrent** (SEQS=16).
> - **llama.cpp Q3 3-lane wins** on deep-context concurrent coding-agent load
>   (~57 tok/s aggregate across 3 lanes × 200K, ~19 tok/s per request).
>
> We are **waiting on deployment** — this is documentation only.

## Model
- **HF repo:** `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- **Size:** 126 GB on disk, **206 shards** (NVFP4 compute + FP8 PLE table + BF16 MTP head)
- **Arch:** Qwen4 (`Qwen4ExpForConditionalGeneration`), 125B MoE (6B active) + 51B n-gram + 4B MTP ≈ 180B

## Engine / container
- **vLLM** built from **`blazux/qwen3.8-Flash-DGX`** (Apache-2.0), **no version pin** (tracks upstream main).
- The container's contribution is a patch that serves the **51.2B n-gram/PLE table from disk**
  (`VLLM_PLE_MMAP=1`) instead of keeping it resident — the only reason a 122 GiB checkpoint
  fits next to a usable KV cache on one box. That patch stays in their repository; this recipe
  only adds the serving configuration we measured.

## Install
```bash
./setup.sh      # clones+builds blazux/qwen3.8-Flash-DGX, fetches ~126 GB checkpoint
./serve.sh      # starts on http://localhost:8000/v1
```
First start reads ~83 GiB off disk and takes **12-15 minutes**.

## Launch (exact flags from the upstream recipe's serve.sh)
```bash
docker run -d --name qwen38-flash --gpus all --ipc=host --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  qwen38-flash-dgx \
  "$SNAP_IN" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len 262144 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]' \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
> `$SNAP_IN` is the mounted HF snapshot path for `RadixArk/Qwen3.8-Flash-Next-NVFP4`
> (the upstream `serve.sh` resolves it from `$HF_CACHE`). `MTP=0` drops the
> `--speculative-config` for an A/B.

## Settings (upstream tunables)
| variable | default | what it does |
|---|---|---|
| `PORT` | `8000` | host port |
| `BIND` | `127.0.0.1` | loopback only. `0.0.0.0` exposes it to your whole network |
| `CTX` | `262144` | context length. The full native window |
| `SEQS` | `16` | concurrent sequences. **Raised from 2** (upstream default) — 2 caps the box at ~a quarter of its throughput. See *How many at once* below |
| `GPU_MEM` | `0.85` | share of the 128 GB pool for weights + KV |
| `MTP` | `3` | speculative tokens. `0` disables, for an A/B |
| `PREWARM` | `1` | stream the table at boot so the first request is not cold |

The upstream default is `GPU_MEM=0.78`, which leaves only 10.82 GiB of KV —
**227,651 tokens, less than the model's own context window**, so a single
full-length request will not fit. We raise it to 0.85, which measured
**18.13 GiB of KV = 641,601 tokens**, comfortably 2.4× the context, while still
leaving ~19 GiB of headroom.

## Measured (GB10, single-stream)
| Metric | Value |
|---|---|
| Free-form prose | **32.2 tok/s** |
| Rewriting a file with one change | 39.1 tok/s |
| Fixing a bug in a file | 35.0 tok/s |
| Adding a function | 33.6 tok/s |
| Spread across those four | **1.2×** (the edit recipe swings 3.2×) |
| Time to first token, short prompt | **~0.3 s** |
| Prefill | **~2,200-2,460 tok/s**, flat to 195k tokens |
| Decode at 1k / 32k / 128k context | 31.7 / 33.5 / 31.7 — **no falloff** |
| Concurrent requests | **16 served well** (64 possible for batch work) |
| **Aggregate decode, 16 concurrent** | **96-109 tok/s**, TTFT under 2.7 s |

## How many at once (SEQS=16)
`SEQS` was `2` (the upstream container's default) — a scheduler cap, not a memory
limit. At `GPU_MEM=0.85` the KV pool is **654,635 tokens**, while 64 concurrent
~1.3k-token requests need only ~83,000 — so `2` left ~87% of the pool unused and
capped the box at ~a quarter of its throughput. Measured (512-token prompts):

| concurrent | tok/s each | aggregate | TTFT p50 |
|---:|---:|---:|---:|
| 1 | 27.70 | 27.7 | 0.68 s |
| 2 | 20.23 | 40.5 | 1.02 s |
| 4 | 15.06 | 60.2 | 1.10 s |
| 8 | 10.68 | 85.4 | 1.27 s |
| **16** | 6.80 | **108.8** | **2.15 s** |
| 32 | 4.21 | 134.8 | 13.80 s |
| 64 | — | — | 70.42 s |

**Aggregate never plateaus** (keeps climbing to 64, diminishing returns), but **TTFT
is the cliff**: under 2.7 s to 16 concurrent, then 16 s at 32, 70 s at 64. So **16 is
the last concurrency this config serves well** (~75% of the box's ceiling). The direct
`SEQS=2` vs `SEQS=64` A/B: serve-chat-c8 (8 concurrent) 43.4 → **81.4 (1.88×)**;
serve-short-c16 (16 concurrent) 53.9 → **146.7 (2.72×)**. One-caller rows are identical
— the cap costs nothing when quiet. **Set `SEQS=64` for batch work** (extra ~35%
aggregate, no one waiting on first token).

## Why it behaves this way
The model ships a **trained MTP head** — a small predictor that guesses the next
few tokens. vLLM runs it; llama.cpp cannot, because the GGUF converter drops the
head (`supports_mtp_export = False`). Our GGUF contains **zero** MTP tensors
across all four shards; the NVFP4 checkpoint contains **all 31**.

Because the head is trained rather than copying from your prompt, it works the
same on any text — hence the flatness, and why this recipe wins on prose and
loses on file rewriting. The gain is real but modest: across engines, prose goes
27.8 → 32.2 (**~1.16×**). On a top-10-of-512 mixture-of-experts, verifying *k*
draft tokens touches the *union* of experts across those positions, so
speculation buys far less here than the 3-5× seen on dense models. Raising `MTP`
above 3 makes it worse, not better.

## Turn thinking off
Thinking is on by default and **86% of generated tokens were reasoning** in our
measurements. The same prompt answered in **15.0 s with thinking off against
55.1 s with it on** — not because tokens got faster, but because there were a
quarter as many.
```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Known issues / caveats
- **Prefix caching must stay off** (`--no-enable-prefix-caching`) — a GB10 GDN
  kernel bug. This is also why our prefill figures are honest: nothing is served
  from cache.
- **Full `torch.compile` is off** — an Inductor int64-indexing assert on sm_121.
- **The n-gram gather must stay outside CUDA graphs.** `serve.sh` declares it a
  splitting op and captures `PIECEWISE`. Do not switch to a `FULL` capture mode.
- **`VLLM_PLE_CPU_OFFLOAD=1` hangs.** The official pinned-host-RAM path registers
  a `PleOffloadLayer` and then spins a core with no disk I/O, indefinitely — it
  expects an offload worker this image does not launch. Only `VLLM_PLE_MMAP`
  works here.
- **No vLLM version pin** — the container tracks upstream main.
- **12-15 min boot** — loads ~83 GiB of weights.
- **111/121 GiB footprint** — tight against 119 GB.
- **16 sequences served well** (SEQS=16). Beyond 16, callers queue rather than all
  being served badly. Set `SEQS=64` for batch work where nothing waits on first token.

## Verifying it works
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## Source references
- 0xBakeer upstream recipe: https://github.com/0xBakeer/qwen38-flash-next-spark/tree/main/recipes/vllm-longctx
- Container: https://github.com/blazux/qwen3.8-Flash-DGX (Apache-2.0)
- Model: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4
