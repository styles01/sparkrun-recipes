# styles01 Sparkrun Recipes

Custom inference recipes for NVIDIA DGX Spark.

## Recipes

- **qwen3.5-122b-dflash-hybrid** — Qwen 3.5 122B A10B with DFlash n=7, hybrid INT4+FP8, aeon vLLM 0.23.0

## Usage

```bash
# Add this registry to sparkrun
sparkrun registry add https://github.com/styles01/sparkrun-recipes

# List recipes
sparkrun list

# Run the recipe
sparkrun run qwen3.5-122b-dflash-hybrid --hosts larryspark.local

# Benchmark and submit to Spark Arena
sparkrun arena login
sparkrun arena benchmark qwen3.5-122b-dflash-hybrid --hosts larryspark.local
```

## Recipe Details

### qwen3.5-122b-dflash-hybrid

| Parameter | Value |
|---|---|
| Model | bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid (67GB) |
| Drafter | z-lab/Qwen3.5-122B-A10B-DFlash (1.5GB) |
| Runtime | vLLM (aeon 0.23.0 dflashfix) |
| DFlash n | 7 |
| GMU | 0.85 |
| Context | 262,144 (256K) |
| Lanes | 3 |
| KV cache | bf16, 549K tokens |
| Prefill | 827 tok/s |
| Decode | 50.2 tok/s |
| Load time | 36 seconds |

Credits: entrpi (base repo), bleysg (hybrid checkpoint), z-lab (DFlash drafter), BlackwellBoy (k=7 sweep), Poolside (acceptance data).