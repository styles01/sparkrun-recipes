# Step 3.7 Flash — Research Lead

**Source:** https://flowtivity.ai/blog/step-3-7-flash-review-dgx-spark/
**Author:** Flowtivity (AI consultancy, "Project Rocketship" — 2x DGX Spark)
**Date observed:** July 14, 2026
**Model release:** May 29, 2026

## The Model

| Spec | Value |
|---|---|
| Total params | 198B (196B language + 1.8B vision encoder) |
| Active params | ~11B per token (sparse MoE) |
| Context | 256K |
| License | Apache 2.0 |
| Vision | Yes (multimodal — screenshots, diagrams, UI) |
| Reasoning | 3 levels (low/medium/high) |
| Languages | 10 |

## DGX Spark Performance (llama.cpp, GGUF)

| Quant | Size | Decode tok/s | PP @ 131K | 256K? |
|---|---|---|---|---|
| IQ4_XS | 105GB | 24-27 | 425 tok/s | Yes |
| Q4_K_S | 112GB | 25 | 450 tok/s | Yes |
| Q3_K_L | 103GB | 22 | 402 tok/s | Yes |

## Benchmark Scores

| Benchmark | Step 3.7 Flash | DS4 Flash | Gemini 3.5 Flash | GPT 5.5 | Claude 4 Opus |
|---|---|---|---|---|---|
| SWE-Bench PRO | 56.3 | 55.6 | 55.1 | 58.6 | **64.3** |
| **ClawEval-1.1** | **67.1** | 43.6 | 57.8 | 60.3 | 59.8 |
| Toolathlon | 49.5 | — | — | — | — |
| Terminal-Bench 2.1 | 59.5 | — | — | — | — |
| SimpleVQA (Search) | 79.2 | — | — | — | — |
| V* (Python) | 95.3 | — | — | — | — |

## Why It Matters

1. **ClawEval 67.1 = #1 overall** — beats Claude 4 Opus, GPT 5.5, DS4 Flash. Purpose-built for agent tool calling.
2. **100% tool call success rate** — tested with OpenClaw (Pedro's platform). Every tool call correctly formatted, every chain completed. Unprecedented.
3. **11B active params** — same as DS4-Flash, close to Qwen 122B (10B). More expert diversity from 198B total.
4. **256K context** — matches our 35B, beats our 122B production (150K).
5. **Apache 2.0** — fully open for commercial use.
6. **Multimodal** — vision encoder for screenshots/diagrams/UI. Our other models are text-only.

## The Catch

| Issue | Impact |
|---|---|
| llama.cpp only (no vLLM) | Different stack from our vLLM containers |
| 105GB on disk (IQ4_XS) | 38GB more than Qwen 122B (67GB) |
| 27 tok/s | Slower than our 122B (40 tok/s with DFlash) |
| No MTP/DFlash speculative decoding | No speed boost available |
| No vLLM optimizations | Loses CUDA graphs, prefix caching, chunked prefill |

## Comparison to Our Models

| | Qwen 122B (ours) | DS4 Flash (ours) | Step 3.7 Flash |
|---|---|---|---|
| Total params | 122B | 159B | 198B |
| Active params | 10B | 11B | 11B |
| ClawEval | — | 43.6 | **67.1** |
| SWE-Bench PRO | — | 55.6 | 56.3 |
| Context | 150K (our config) | 128K | 256K |
| Disk | 67GB | 227GB | 105GB |
| Speed (Spark) | 40 tok/s | 21-30 tok/s | 27 tok/s |
| Spec decoding | DFlash n=4 | MTP k=2 | None |
| Runtime | vLLM | vLLM | llama.cpp |
| Vision | No | No | **Yes** |
| License | Apache 2.0 | MIT | Apache 2.0 |

## Action Items

1. **Watch for vLLM support** — if Step 3.7 Flash gets vLLM + MTP, it becomes the top agent model candidate
2. **Check HuggingFace for safetensors** — GGUF is llama.cpp only; vLLM needs safetensors
3. **Consider llama.cpp as fallback** — if vLLM never supports it, 27 tok/s might be acceptable for single-agent reasoning tasks (like DS4's role)
4. **The vision encoder is unique** — no other model in our roster has multimodal. Could be valuable for UI/screenshot workflows
5. **105GB is tight** — would need GMU ~0.87 to fit weights + minimal KV. Probably 1-2 lanes only. Not a multi-agent config.