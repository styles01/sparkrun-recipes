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
| Spec decode | DSpark, block size 4 (n=4) |
| KV cache | FP8 e4m3 (~3 KB/token) |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `nemotron_v3` |

## Benchmark Results (Spark Arena, 2026-08-11)

Submission: `sub1786523649341` — profile `@official/spark-arena-v2` (pp 2048, tg 128, 3 runs)
Config: DSpark n=4, GMU 0.60, max_num_seqs 10, 1M ctx, chunked-prefill, async-scheduling, marlin

| Depth | Conc 1 | Conc 2 | Conc 5 | Conc 10 |
|---|---|---|---|---|
| 0 | 90.6 | 145.3 | 180.9 | **224.4** |
| 4K | **108.1** | 103.5 | 105.9 | 118.7 |
| 8K | 90.8 | 100.3 | 103.0 | 100.2 |
| 16K | 83.5 | 87.3 | 89.4 | 85.1 |
| 32K | 96.4 | 90.7 | 75.4 | 77.9 |
| 64K | 96.2 | 72.4 | 59.7 | 65.2 |
| 100K | 85.4 | 54.7 | 51.9 | 54.2 |

Table shows **generation tok/s** (tg). Prefill peaked at **7,665 tok/s** (depth 0, c10).

**Headline:** 108.1 tok/s single-lane (4K ctx), 224.4 tok/s aggregate (10 concurrent).

> **Tradeoff vs v1 (DSpark n=3):** n=4 + safe wins gives +49% at 8K/c10 and +7% aggregate at
> c10, but costs ~10-23% single-stream (the draft-depth tradeoff Saiyam's stock-image run
> noted). v3 is tuned for concurrent/multi-agent use; drop to n=3 if you want peak single-lane.

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
