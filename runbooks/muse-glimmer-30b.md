# Muse-Glimmer-30B on DGX Spark

**Status:** ✅ Production  
**Created:** August 10, 2026  
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)  

---

## Overview

Meta's Muse-Glimmer-30B is a multimodal reasoning model supporting text, vision, and video input. 30B parameters, 52 layers with sliding + full attention (4:1 ratio), 131K context, BF16 precision. Uses ATEM XML format for tool calling (not JSON). Includes a DFlash draft head for speculative decoding.

## Model Details

| Metric | Value |
|---|---|
| **Params** | 30B (dense) |
| **Architecture** | MuseGlimmerForConditionalGeneration |
| **Precision** | BF16 (55.5 GB) |
| **Context** | 131,072 tokens |
| **Layers** | 52 (sliding + full attention, 4:1 ratio) |
| **Vision** | 50-layer encoder, patch 14, supports video |
| **Tool calling** | ATEM XML format (not JSON) |
| **Reasoning** | Yes — reasoning separated from content via `--reasoning-parser muse_glimmer` |
| **DFlash head** | meta-models/Muse-Glimmer-30B-assistant (3B, 4.8 GB) |

## Requirements

- **vLLM 0.27.0+** — vLLM v26 does NOT have native MuseGlimmer support (falls back to Transformers wrapper which produces empty responses)
- **Docker image:** `vllm/vllm-openai:muse-glimmer`
- **`--enforce-eager`** required on GB10/SM121 (torch.compile hits CUBLAS error)
- **Do NOT use `--attention-backend flashinfer`** — warmup hangs on SM121. Let vLLM auto-select.
- **Sampling:** temp=1.0, top_p=0.95, top_k=64 — do NOT run greedy (produces empty responses)

## Launch

```bash
# Stop any existing inference
docker rm -f muse-glimmer 2>/dev/null

# Launch Muse-Glimmer-30B
docker run -d \
  --name muse-glimmer \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 8gb \
  -v ~/models/hf/Muse-Glimmer-30B:/model \
  -v /tmp:/tmp \
  vllm/vllm-openai:muse-glimmer \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 5 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.79 \
  --kv-cache-dtype fp8 \
  --load-format safetensors \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --served-model-name muse-glimmer-30b \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  -tp 1 \
  -pp 1
```

## Monitor

```bash
# Logs
docker logs -f muse-glimmer

# Smoke test
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":200}' | python3 -m json.tool

# Stop
docker rm -f muse-glimmer
```

## Benchmark

```bash
docker exec muse-glimmer pip install llama-benchy
docker exec muse-glimmer llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model muse-glimmer-30b \
  --pp 2048 4096 \
  --tg 128 256 \
  --depth 0 4096 32768 65536 \
  --concurrency 1 2 3 \
  --latency-mode generation \
  --save-result /tmp/muse-glimmer-bench.json
```

## Download

```bash
# Main model (BF16, 55.5 GB)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-models/Muse-Glimmer-30B', local_dir='~/models/hf/Muse-Glimmer-30B')"

# DFlash draft head (3B, 4.8 GB)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-models/Muse-Glimmer-30B-assistant', local_dir='~/models/hf/Muse-Glimmer-30B-assistant')"

# GGUF k-quants (for llama.cpp)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-models/Muse-Glimmer-30B-GGUF', local_dir='~/models/hf/Muse-Glimmer-30B-GGUF')"
```

## Known Issues

1. **vLLM v26 does NOT work** — no native MuseGlimmer implementation, falls back to Transformers wrapper which produces empty responses. Must use vLLM 0.27.0+ (`vllm/vllm-openai:muse-glimmer` Docker image).

2. **FlashInfer warmup hangs on SM121** — remove `--attention-backend flashinfer` and let vLLM auto-select. The model still uses FlashInfer internally but the explicit flag causes a warmup hang.

3. **torch.compile CUBLAS error on GB10** — use `--enforce-eager`. Slower inference but stable.

4. **Greedy decoding produces empty responses** — the model requires sampling (temp=1.0, top_p=0.95, top_k=64). Do NOT use temp=0.

5. **Reasoning model** — generates reasoning first, then content. The `--reasoning-parser muse_glimmer` flag separates them. Content may be empty if max_tokens is too low (reasoning consumes the budget).

## Troubleshooting

| Issue | Fix |
|---|---|
| Empty responses | Use vLLM 0.27+ with `--reasoning-parser muse_glimmer` |
| CUBLAS error | Add `--enforce-eager` |
| FlashInfer hang | Remove `--attention-backend flashinfer` |
| Immediate stop | Increase max_tokens (reasoning needs ~50-100 tokens before content) |
| Greedy empty | Use sampling: temp=1.0, top_p=0.95, top_k=64 |

## Model Files on Spark

```
~/models/hf/Muse-Glimmer-30B/                    # BF16 main model (55.5 GB)
~/models/hf/Muse-Glimmer-30B-assistant/          # DFlash draft head (4.8 GB)
```

## Recipe

See [recipe](../recipes/muse-glimmer-30b.yaml) for sparkrun-compatible YAML.