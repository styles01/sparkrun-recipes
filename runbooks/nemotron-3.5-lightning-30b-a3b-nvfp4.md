# Nemotron 3.5 Lightning 30B-A3B NVFP4 — DGX Spark Runbook

**Model:** [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
**Recipe:** [recipes/nemotron-3.5-lightning-30b-a3b-nvfp4.yaml](../recipes/nemotron-3.5-lightning-30b-a3b-nvfp4.yaml)
**Engine:** vLLM 0.27.1 (`vllm/vllm-openai:v0.27.1`) + **DSpark** speculative decoding
**Hardware:** DGX Spark (GB10 / SM121, 121 GB unified memory)

---

## What It Is

NVIDIA Nemotron 3.5 Lightning is a **30B-total / ~3B-active** MoE reasoning model with a
**hybrid architecture** — 52 layers = 23 Mamba-2 SSM + 23 MoE + 6 Attention. Only the 6
attention layers pay a growing K/V cost, which is what makes a 1M-token context window
affordable in ~14 GiB. NVFP4-quantized (modelopt) with FP8 KV cache.

This recipe adapts the **official NVIDIA vLLM DGX Spark cookbook** command, replacing the
SGLang recipe that Mia's AI Lab published (the same underlying model/draft/quant, served
through vLLM). It ships with NVIDIA's **DSpark draft model** for speculative decoding.

## Key Facts

| | Value |
|---|---|
| Params | 30B total / ~3B active |
| Layers | 52 (23 Mamba-2 + 23 MoE + 6 Attention) |
| Context | 1,048,576 (1M) max |
| Quant | NVFP4 (modelopt) — ~20 GiB weights |
| Draft model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` |
| Spec decode | DSpark, block size 3 |
| KV cache | FP8 e4m3 (~3 KB/token) |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `nemotron_v3` |

## Benchmark Results (Spark Arena, 2026-08-11)

Submission: `sub1786493259764` — profile `@official/spark-arena-v2` (pp 2048, tg 128, 3 runs)

| Depth | Conc 1 | Conc 2 | Conc 5 | Conc 10 |
|---|---|---|---|---|
| 0 | **118.1** | 154.7 | 179.2 | **210.2** |
| 4K | **120.4** | 105.7 | 94.3 | 103.1 |
| 8K | 118.0 | 73.8 | 71.8 | 67.3 |
| 16K | 86.8 | 77.3 | 65.9 | 62.7 |
| 32K | 96.7 | 73.0 | 59.5 | 58.9 |
| 64K | 93.8 | 67.2 | 55.6 | 54.9 |
| 100K | 85.6 | 64.0 | 48.4 | 46.3 |

Table shows **generation tok/s** (tg). Prefill peaked at **~5,775 tok/s** (depth 0).

**Headline:** 120.4 tok/s single-lane (4K ctx), 210.2 tok/s aggregate (10 concurrent).

## Launch

```bash
sparkrun run @styles01/nemotron-3.5-lightning-30b-a3b-nvfp4 --hosts larryspark.local
```

Or run the recipe file directly:

```bash
sparkrun run recipes/nemotron-3.5-lightning-30b-a3b-nvfp4.yaml --solo
```

> **IMPORTANT:** launch **detached** (no `--foreground`). The `--foreground` mode imposes a
> 60-second exec timeout, but Nemotron takes ~5 minutes to load (52 shards + DSpark draft +
> Mamba init + CUDA graph compile). Detached mode doesn't kill it.

## Model Files

Weights must live in sparkrun's HF cache mount at `~/models/hf/hub/models--<safe-name>/`
with a `refs/main` file pointing to a content-hash snapshot dir. sparkrun mounts
`~/models/hf` → `/cache/huggingface` and sets `HF_HUB_OFFLINE=1`.

## Serving Details

- **OpenAI-compatible** on port 8000, served as `nemotron-3.5-lightning`
  (also serves the full HF id `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
  so llama-benchy's HF-id requests resolve).
- **Reasoning control:** thinking is ON by default (emits reasoning tokens); disable via
  `chat_template_kwargs: {"enable_thinking": false}`.
- **Recommended sampling:** temperature 1.0, top_p 0.95 (per NVIDIA).

## Tool Calling

Uses `qwen3_coder` parser + `--enable-auto-tool-choice`. Verified working:
`get_weather("Paris")` returns a clean tool call.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `--foreground` kills launch at 60s | Launch detached (no `--foreground`) — model load takes ~5 min |
| `LocalEntryNotFoundError` / `HF_HUB_OFFLINE` | Weights not in `~/models/hf/hub/models--<id>/` — check the mount |
| llama-benchy `404 model does not exist` | `served_model_name` must include the full HF id |
| `up_proj` gate warnings | Benign — vLLM skips a gate/up fusion for Nemotron's layout |

## Credits

- Recipe & model: NVIDIA (vLLM DGX Spark cookbook, `vllm/vllm-openai:v0.27.1`)
- SGLang reference: Mia's AI Lab (`MiaAI-Lab/Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO`)
- vLLM adaptation + arena submission: styles01
