# Runbook: Qwen 3.6 27B FP8 — MTP k=7

**Status:** ✅ Production — vLLM v26 patched, MTP k=7 (48.7% accept), fp8 KV, async scheduling
**Served name:** `Qwen/Qwen3.6-27B-FP8`
**Stack:** Docker — `ghcr.io/styles01/vllm-v26-patched:latest`
**Tool parser:** `qwen3_coder`
**Reasoning parser:** `qwen3`

> **Recipe contract:** [`recipes/qwen-27b.yaml`](../recipes/qwen-27b.yaml)

## Model Location on Spark

```
~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/
```
FP8 quantized, ~27GB on disk.

## Config

| Parameter | Value |
|---|---|
| GPU mem util | 0.79 |
| KV cache dtype | fp8 |
| MTP speculative | k=7, 48.7% acceptance |
| Attention backend | flashinfer |
| Max model len | 262,144 (256K) |
| Max seqs | 5 |
| Max batched tokens | 32,768 |
| Load format | safetensors |
| Served name | Qwen/Qwen3.6-27B-FP8 |
| Port | 8000 |
| Tool parser | qwen3_coder |
| Reasoning parser | qwen3 |
| Async scheduling | ON |
| Prefix caching | ON |
| Chunked prefill | ON |

## Start Command

```bash
ssh jaita@larryspark.local 'bash ~/switch-to-qwen27b.sh'
```

### Full Docker launch

```bash
docker run -d \
  --name qwen27b-spark --gpus all -p 8000:8000 --user root \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  ghcr.io/styles01/vllm-v26-patched:latest \
  Qwen/Qwen3.6-27B-FP8 \
    --served-model-name Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-model-len 262144 \
    --max-num-seqs 5 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.79 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --load-format safetensors \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
```

## Stop Command

```bash
ssh jaita@larryspark.local 'docker rm -f qwen27b-spark'
```

## Performance

| Metric | Value |
|---|---|
| MTP acceptance | 48.7% at k=7 |
| Context | 256K |
| Concurrency | 5 lanes |
| Decode speed | TBD (benchmarks pending) |

## Why k=7

- 27B model has sufficient MTP head capacity for k=7
- 48.7% acceptance rate — good balance of speedup vs. draft overhead
- Higher k (8+) shows diminishing returns on 27B class models
- Async scheduling + prefix caching maximize throughput at 5 concurrent

## vLLM v26 Patches

Uses the same patched vLLM v26 container as the 122B recipe:
- `patch_inc_hybrid.py` — hybrid quantization support
- `patch_int8_lmhead_v3.py` — int8 lm-head optimization
- `patch_prefix_align.py` — prefix caching alignment fix

## Verify

```bash
curl -s http://larryspark.local:8000/v1/models | jq '.data[].id'
# expect: Qwen/Qwen3.6-27B-FP8

curl -s http://larryspark.local:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"Write a Python function to reverse a linked list."}],"max_tokens":500}' \
  | jq '.choices[0].message.content'
```

## Trade-Offs

- 27B is the sweet spot for single-user speed + quality on DGX Spark
- k=7 MTP gives ~2x throughput vs no-spec, but 48.7% acceptance means ~5 of 7 drafts accepted
- 5 concurrent at 256K — enough for agent + subagent workflows
- fp8 KV doubles KV capacity vs bf16

## Switch Script

`scripts/switch-to-qwen27b.sh` — one-command deploy.