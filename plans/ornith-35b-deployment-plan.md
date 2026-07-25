# Ornith-1.0-35B Deployment Plan

**Created:** July 24, 2026
**Status:** Ready to deploy — recipe created, registered, validated

## Model Summary

- **Ornith-1.0-35B** — Qwen3.5-MoE architecture, 35B total / 3B active
- 256 experts, 8 per token, hybrid linear/full attention (30/40 layers linear)
- Multimodal (vision-capable), 262K context, MTP heads built-in
- int4 AutoRound W4A16, ~20GB
- Beats Qwen3.5-397B on Terminal-Bench (agentic coding)
- 80-91 tok/s with MTP vs 60-65 without
- Post-trained by DeepReinforce AI

## Recipe

- **Sparkrun:** `@styles01/recipe` (registered on Spark)
- **GitHub:** https://github.com/styles01/sparkrun-recipes/tree/main/recipes/ornith-1.0-35b-mtp
- **Model:** `deepreinforce-ai/Ornith-1.0-35B-FP8` (official FP8, 813K downloads)
- **Container:** `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`
- **Spec decode:** MTP k=2 (built-in heads)
- **KV dtype:** fp8
- **Attention:** flashinfer
- **GMU:** 0.8
- **Served name:** `ornith`

## Deployment Steps

1. Kill current Qwen 122B deployment:
   ```bash
   ssh jaita@larryspark.local 'docker rm -f qwen-spark'
   ```

2. Deploy Ornith using sparkrun (if logged in to arena):
   ```bash
   ssh jaita@larryspark.local '~/.local/bin/sparkrun run @styles01/recipe --hosts larryspark.local'
   ```

   OR deploy manually with Docker (no arena login needed):
   ```bash
   ssh jaita@larryspark.local 'docker run -d \
     --name ornith-spark --gpus all -p 8000:8000 \
     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
     -e HF_HOME=/root/.cache/huggingface \
     -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest \
     azampatti/Ornith-1.0-35B-int4-AutoRound-SAR \
       --served-model-name ornith --host 0.0.0.0 --port 8000 \
       --trust-remote-code --max-model-len 262144 \
       --max-num-batched-tokens 32768 \
       --gpu-memory-utilization 0.8 \
       --enable-auto-tool-choice --tool-call-parser qwen3_coder \
       --reasoning-parser qwen3 \
       --kv-cache-dtype fp8 \
       --load-format auto \
       --attention-backend flashinfer \
       --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": 2}' \
       --enable-prefix-caching \
       -tp 1 -pp 1'
   ```

3. Wait for model load (~2-5 min, only 20GB)

4. Verify it's serving:
   ```bash
   ssh jaita@larryspark.local 'curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])"'
   ```

5. Benchmark with llama-benchy:
   ```bash
   ssh jaita@larryspark.local '~/.local/bin/llama-benchy --base-url http://127.0.0.1:8000/v1 --model ornith --pp 512 --tg 128 --runs 3'
   ```

6. Update STATE.md with Ornith config + benchmark numbers

7. To submit to Spark Arena (needs login first):
   ```bash
   ssh jaita@larryspark.local '~/.local/bin/sparkrun arena login'
   # Go to https://auth.sparkrun.dev/device, enter code
   ssh jaita@larryspark.local '~/.local/bin/sparkrun arena benchmark @styles01/recipe --hosts larryspark.local'
   ```

## Rollback (back to Qwen 122B)

```bash
ssh jaita@larryspark.local 'docker rm -f ornith-spark 2>/dev/null; cd ~/qwen3.5-122B-A10B-on-spark && CTX=262144 GPU_MEM=0.85 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 SERVED_NAME=qwen bash install.sh --start --profile dense --nspec 7 --no-smoke'
```

## Expected Performance

| Metric | Estimate | Source |
|---|---|---|
| Weights | ~20GB (int4) | HF model card |
| KV cache | ~10GB (fp8, 256K ctx) | sparkrun VRAM estimate |
| Total VRAM | ~30GB | sum |
| Free RAM | ~90GB | 121GB - 30GB |
| Decode tok/s | 80-91 (MTP k=2) | HF model card claims |
| Prefill tok/s | TBD | needs benchmark |
| Load time | ~2-5 min | 20GB, fastsafetensors |
| Concurrency | high (lots of KV headroom) | 61.9GB available for KV |

## Risks

- 45% of layers have quantization warnings (int4 AutoRound)
- Concurrency-4 hangs reported in NVFP4 sibling variant
- No existing community validation on DGX Spark for int4 AutoRound variant
- MTP k=2 may need tuning (Qwen 122B needed k=7 for DFlash, but MTP is different)

## Comparison to Qwen 122B

| Metric | Qwen 122B (current) | Ornith 35B (planned) |
|---|---|---|
| Weights | 64GB | 20GB |
| Active params | 10B | 3B |
| Spec decode | DFlash k=7 | MTP k=2 (built-in) |
| Decode tok/s | 50.2 | 80-91 (claimed) |
| Terminal-Bench | 49.4% | beats 397B |
| Grounding | zero fabrications in 240 runs | unknown |
| Free RAM | 15GB | 90GB |
| Context | 256K | 256K |

## Files

- Recipe: `sparkrun/recipes/ornith-1.0-35b-mtp/recipe.yaml`
- Research: `research/ornith-35b-research.md`
- GitHub repo: https://github.com/styles01/sparkrun-recipes