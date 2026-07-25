# Qwen3.6-27B — Research Brief

**Compiled:** 2026-07-18
**Sources:** HuggingFace model card, Qwen blog, Artificial Analysis, mr_r0b0t GitHub repo, Spark Arena leaderboard, HF model search API, z-lab DFlash repo
**Purpose:** Evaluate Qwen3.6-27B as a candidate vs. our current 35B (Qwen3.6-35B-A3B) and 122B (Qwen3.5-122B-A10B) models on DGX Spark.

---

## TL;DR

Qwen3.6-27B is a **dense** 27B-parameter multimodal model (NOT MoE) released April 2026 by Qwen Team. It is the headline-grabbing model that **surpasses the previous open-source flagship Qwen3.5-397B-A17B (397B total / 17B active MoE) on every major agentic coding benchmark** at ~1/15th the total parameter count. Architecture is a hybrid Gated DeltaNet + Gated Attention design (same "qwen3_5" family as 35B-A3B), trained with **MTP** (multi-token prediction) and compatible with **DFlash** speculative decoding. Apache 2.0 license. 262K native context, extensible to 1.01M with YaRN.

**Bottom line for us:** At 27B dense it is ~23% smaller than our 35B-A3B MoE (35B total / 3B active) but, being dense, it has **far higher active-param cost per token** (27B active vs 3B active). On raw quality it beats the 35B-A3B on coding (SWE-bench Verified 77.2 vs 73.4) and matches/beats it on most knowledge & reasoning. On DGX Spark throughput it is **much slower** than the 35B-A3B-NVFP4 (which tops the Spark Arena decode chart at 178 t/s vs ~32–46 t/s for 27B-FP8/NVFP4). The "punches above its weight class" claim is **verified** — its AA Intelligence Index of 37 is the highest in the open-weights small (4B–40B) class.

---

## 1. Architecture

| Property | Value |
|---|---|
| Type | Causal LM with Vision Encoder (multimodal: text + image + video) |
| Architecture family | `qwen3_5` (same as Qwen3.6-35B-A3B) |
| **MoE?** | **No — dense** (27B total = 27B active) |
| Total parameters | 27B (AA reports 27.8B) |
| Hidden dimension | 5120 |
| Number of layers | 64 |
| Hidden layout | **16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))** |
| Gated DeltaNet (linear attn) | 48 V heads / 16 QK heads, head_dim 128 |
| Gated Attention (full attn) | 24 Q heads / 4 KV heads, head_dim 256, RoPE dim 64 |
| FFN intermediate dim | 17408 |
| Token embedding (padded) | 248320 |
| **MTP** | Trained with multi-steps (speculative decoding supported) |
| **DFlash support** | Yes — `z-lab/Qwen3.6-27B-DFlash` drafter model (block-diffusion speculative decoding, up to 15 spec tokens) |
| Context (native) | **262,144 tokens** |
| Context (extended, YaRN) | up to **1,010,000 tokens** (factor 4.0) |
| Vision | Native multimodal — images & video in a single unified checkpoint |
| Thinking modes | Thinking (default) + Non-thinking (instruct) + **Preserve Thinking** (new in 3.6) |
| License | **Apache 2.0** |
| Release | April 2026 (HF card says April; blog post dated 2026/04/21) |

**Key architectural note:** The hybrid GDN (Gated DeltaNet) + full-attention layout means vLLM **disables prefix caching by design** for this model family (non-causal attention layers) — this is correct behavior, not a bug. Also, `/think` and `/no_think` soft switches from Qwen3 are **not** supported in 3.6; use `chat_template_kwargs: {enable_thinking: false}` instead.

---

## 2. Benchmark Scores — Language (from HF model card / Qwen blog)

Headline comparison columns vs our current models. **Bold** = Qwen3.6-27B.

### Coding Agent
| Benchmark | Qwen3.5-27B | Qwen3.5-397B-A17B | Gemma4-31B | Claude 4.5 Opus | **Qwen3.6-35B-A3B** | **Qwen3.6-27B** |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 75.0 | 76.2 | 52.0 | 80.9 | 73.4 | **77.2** |
| SWE-bench Pro | 51.2 | 50.9 | 35.7 | 57.1 | 49.5 | **53.5** |
| SWE-bench Multilingual | 69.3 | 69.3 | 51.7 | 77.5 | 67.2 | **71.3** |
| Terminal-Bench 2.0 | 41.6 | 52.5 | 42.9 | 59.3 | 51.5 | **59.3** |
| SkillsBench (Avg5) | 27.2 | 30.0 | 23.6 | 45.3 | 28.7 | **48.2** |
| QwenWebBench | 1068 | 1186 | 1197 | 1536 | 1397 | **1487** |
| NL2Repo | 27.3 | 32.2 | 15.5 | 43.2 | 29.4 | **36.2** |
| Claw-Eval (Avg) | 64.3 | 70.7 | 48.5 | 76.6 | 68.7 | **72.4** |
| Claw-Eval (Pass³) | 46.2 | 48.1 | 25.0 | 59.6 | 50.0 | **60.6** |
| QwenClawBench | 52.2 | 51.8 | 41.7 | 52.3 | 52.6 | **53.4** |

### Knowledge
| Benchmark | Qwen3.5-27B | Qwen3.5-397B-A17B | Gemma4-31B | Claude 4.5 Opus | Qwen3.6-35B-A3B | **Qwen3.6-27B** |
|---|---|---|---|---|---|---|
| MMLU-Pro | 86.1 | 87.8 | 85.2 | 89.5 | 85.2 | **86.2** |
| MMLU-Redux | 93.2 | 94.9 | 93.7 | 95.6 | 93.3 | **93.5** |
| SuperGPQA | 65.6 | 70.4 | 65.7 | 70.6 | 64.7 | **66.0** |
| C-Eval | 90.5 | 93.0 | 82.6 | 92.2 | 90.0 | **91.4** |

### STEM & Reasoning
| Benchmark | Qwen3.5-27B | Qwen3.5-397B-A17B | Gemma4-31B | Claude 4.5 Opus | Qwen3.6-35B-A3B | **Qwen3.6-27B** |
|---|---|---|---|---|---|---|
| GPQA Diamond | 85.5 | 88.4 | 84.3 | 87.0 | 86.0 | **87.8** |
| HLE | 24.3 | 28.7 | 19.5 | 30.8 | 21.4 | **24.0** |
| LiveCodeBench v6 | 80.7 | 83.6 | 80.0 | 84.8 | 80.4 | **83.9** |
| HMMT Feb 25 | 92.0 | 94.8 | 88.7 | 92.9 | 90.7 | **93.8** |
| HMMT Nov 25 | 89.8 | 92.7 | 87.5 | 93.3 | 89.1 | **90.7** |
| HMMT Feb 26 | 84.3 | 87.9 | 77.2 | 85.3 | 83.6 | **84.3** |
| IMOAnswerBench | 79.9 | 80.9 | 74.5 | 84.0 | 78.9 | **80.8** |
| AIME26 | 92.6 | 93.3 | 89.2 | 95.1 | 92.7 | **94.1** |

**Reading the table:** Qwen3.6-27B beats the 397B-A17B MoE flagship on coding (SWE-bench Verified 77.2 vs 76.2, Terminal-Bench 59.3 vs 52.5, SkillsBench 48.2 vs 30.0) and is competitive on knowledge/reasoning. It beats our 35B-A3B on every coding metric and on GPQA Diamond (87.8 vs 86.0), AIME26 (94.1 vs 92.7), HMMT Feb 25 (93.8 vs 90.7), and LiveCodeBench v6 (83.9 vs 80.4).

---

## 3. Benchmark Scores — Vision Language (from HF model card)

| Benchmark | Qwen3.5-27B | Qwen3.5-397B-A17B | Gemma4-31B | Claude 4.5 Opus | Qwen3.6-35B-A3B | **Qwen3.6-27B** |
|---|---|---|---|---|---|---|
| MMMU | 82.3 | 85.0 | 80.4 | 80.7 | 81.7 | **82.9** |
| MMMU-Pro | 75.0 | 79.0 | 76.9 | 70.6 | 75.3 | **75.8** |
| MathVista mini | 87.8 | -- | 79.3 | -- | 86.4 | **87.4** |
| DynaMath | 87.7 | 86.3 | 79.5 | 79.7 | 82.8 | **85.6** |
| VlmsAreBlind | 96.9 | -- | 87.2 | -- | 96.6 | **97.0** |
| RealWorldQA | 83.7 | 83.9 | 72.3 | 77.0 | 85.3 | **84.1** |
| MMStar | 81.0 | 83.8 | 77.3 | 73.2 | 80.7 | **81.4** |
| MMBench EN-DEV v1.1 | 92.6 | -- | 90.9 | -- | 92.8 | **92.3** |
| CharXiv (RQ) | 79.5 | 80.8 | 67.9 | 68.5 | 78.0 | **78.4** |
| CC-OCR | 81.0 | 82.0 | 75.7 | 76.9 | 81.9 | **81.2** |
| OCRBench | 89.4 | -- | 86.1 | -- | 90.0 | **89.4** |
| VideoMME (w sub.) | 87.0 | 87.5 | -- | 77.7 | 86.6 | **87.7** |
| VideoMMMU | 82.3 | 84.7 | 81.6 | 84.4 | 83.7 | **84.4** |
| MLVU | 85.9 | 86.7 | -- | 81.7 | 86.2 | **86.6** |
| V* | 93.7 | 95.8 | -- | 67.0 | 90.1 | **94.7** |
| AndroidWorld | 64.2 | -- | -- | -- | -- | **70.3** |

---

## 4. Artificial Analysis — Intelligence Index & Per-Eval Breakdown

Source: https://artificialanalysis.ai/models/qwen3-6-27b (AA Intelligence Index v4.1)

### Summary metrics
| Metric | Qwen3.6-27B (Reasoning) | Rank / 130 |
|---|---|---|
| **AA Intelligence Index** | **37** | #1 in open-weights small (4B–40B) class; #11/28 open-weights overall shown |
| Output speed | 54.6 t/s (DashScope hosted) | #57/130 |
| Input price | $0.60 / 1M tok (expensive) | #124/130 |
| Output price | $3.60 / 1M tok (expensive) | #126/130 |
| Verbosity | 140M output tokens on AA Index | #21/130 (very verbose) |
| Total params | 27.8B | — |
| Context window | 262k | — |
| License | Apache 2.0 | — |
| Cost to run AA Index | $668.08 | — |
| Cost per AA Index task | $0.27 | — |
| AA Openness Index | 39 / 100 | — |

### Per-eval scores (AA Intelligence Index v4.1 components)
| Evaluation | Qwen3.6-27B | Notes |
|---|---|---|
| GDPval-AA v2 (agentic work, Elo) | 32% | #11/28 |
| 𝜏³-Banking (agentic tool use) | 15% | #9/28 |
| Terminal-Bench v2.1 (agentic coding) | 51% | #11/28 (note: Qwen's own card reports 59.3 — different harness/protocol) |
| SciCode (coding) | 37% | #13/28 |
| Humanity's Last Exam | 14% | #16/28 |
| GPQA Diamond | 83% | #12/28 (Qwen card reports 87.8 — eval-settings diff) |
| CritPt (physics) | 1% | low across the field |
| AA-Omniscience Accuracy | 19% | #12/28 |
| AA-Omniscience Non-Hallucination | 16% | #19/28 |
| AA-LCR (long context) | 55% | #19/28 |
| AA-Briefcase (agentic knowledge, Elo) | 806 | #11/15 |
| IFBench (instruction following) | 44% | #20/28 |
| MMMU-Pro (visual reasoning) | 72% | #9/16 |

**AA commentary:** "Amongst the leading models in intelligence, but particularly expensive when comparing to other open weight models of similar size. It's also notably slow and very verbose." The expensive/slow numbers reflect DashScope API pricing/throughput, not local DGX Spark inference.

### AA Intelligence Index — open-weights small-class ranking (context)
| Rank (open-weights shown) | Model | AA Index |
|---|---|---|
| #11 | Qwen3.6 27B | **37** |
| #12 | Qwen3.6 35B A3B | 32 |
| #13 | Qwen3.6 27B (non-reasoning variant) | 30 |
| #14 | Gemma 4 31B | 29 |
| #15 | Gemma 4 26B A4B | 26 |
| #16 | Qwen3.6 35B A3B (other variant) | 24 |
| #17 | Qwen3.5 35B A3B | 24 |
| #18 | gpt-oss-120b (high) | 24 |

So at the ** Intelligence Index** level, Qwen3.6-27B (reasoning) scores **5 points higher than our Qwen3.6-35B-A3B** (37 vs 32) despite being smaller — consistent with the "punches above its weight class" claim.

---

## 5. Quantizations Available

Sourced from HuggingFace model search API. Downloads as of 2026-07-18.

### Official (Qwen)
| Repo | Format | Downloads | Likes |
|---|---|---|---|
| `Qwen/Qwen3.6-27B` | BF16 (Safetensors) | 5,395,520 | 1,991 |
| `Qwen/Qwen3.6-27B-FP8` | FP8 | 5,768,865 | 312 |

### NVIDIA
| Repo | Format | Downloads | Likes |
|---|---|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` | NVFP4 (modelopt_mixed: MLP W4A16_NVFP4 group_size=16, attn FP8, lm_head NVFP4) | 1,466,980 | 371 |

### Community (high-signal)
| Repo | Format | Downloads | Likes |
|---|---|---|---|
| `unsloth/Qwen3.6-27B-MTP-GGUF` | GGUF (MTP) | 2,900,201 | 1,122 |
| `unsloth/Qwen3.6-27B-NVFP4` | NVFP4 | 2,064,590 | 229 |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ INT4 | 1,605,897 | 93 |
| `QuantTrio/Qwen3.6-27B-AWQ` | AWQ | 1,419,821 | 21 |
| `unsloth/Qwen3.6-27B-GGUF` | GGUF | 792,343 | 878 |
| `Lorbus/Qwen3.6-27B-int4-AutoRound` | INT4 (AutoRound) | 406,473 | 123 |
| `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive` | Uncensored finetune | 386,353 | 524 |
| `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF` | GGUF (ThinkingCap) | 339,198 | 145 |
| `DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-...-GGUF` | GGUF uncensored | 143,966 | 402 |
| `z-lab/Qwen3.6-27B-DFlash` | DFlash drafter (speculative) | 68,668 | 381 |
| `michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF` | GGUF NVFP4 MTP | 88,389 | 48 |
| `bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF` | GGUF MTP tune | 97,851 | 125 |
| `Minachist/Qwen3.6-27B-INT8-AutoRound` | INT8 (AutoRound) | 26,105 | 23 |
| `CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF` | GGUF NVFP4 | 29,907 | 19 |
| `rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm` | PrismaAURA 5.5bit | 16,788 | 19 |
| `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` | FP8 (ThinkingCap) | 16,886 | 19 |
| `z-lab/Qwen3.6-27B-PARO` | PARO | 3,341 | 30 |
| `turboderp/Qwen3.6-27B-exl3` | EXL3 | 187 | 6 |
| `mlx-community/Qwen3.6-27B-OptiQ-4bit` | MLX 4bit | 13,336 | 68 |
| `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` | DFlash GGUF | 12,321 | 11 |

**Summary of formats:**
- **BF16**: official `Qwen/Qwen3.6-27B` (~54 GB)
- **FP8**: official `Qwen/Qwen3.6-27B-FP8` (~27 GB)
- **NVFP4**: official NVIDIA `nvidia/Qwen3.6-27B-NVFP4` + `unsloth` mirror (~14 GB)
- **INT4 (AWQ / AutoRound)**: `cyankiwi`, `QuantTrio`, `Lorbus` (~14–15 GB)
- **INT8 (AutoRound)**: `Minachist`
- **GGUF (all quants)**: `unsloth`, `bartowski`, many uncensored finetunes
- **DFlash drafter**: `z-lab/Qwen3.6-27B-DFlash` (must be paired with `Qwen/Qwen3.6-27B` target)
- **MLX 4bit**: `mlx-community/Qwen3.6-27B-OptiQ-4bit`
- **EXL3**: `turboderp/Qwen3.6-27B-exl3`

---

## 6. mr_r0b0t's DGX Spark NVFP4 Serving Repo

Source: https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4 (6★, 0 forks, MIT license, last commit 2 weeks ago)

### What it is
Optimized vLLM v0.24.0 runtime for `nvidia/Qwen3.6-27B-NVFP4` on **NVIDIA GB10 / SM121 (DGX Spark)**, with FP8 KV cache, MTP speculative decoding, and native NVFP4 weight quantization via FlashInfer FA2 JIT. NVFP4 KV cache support is in progress (FlashInfer PR #3684 + vLLM PR #46329).

### Current runtime config (as of 2026-07-03)
- **KV Cache Dtype**: fp8 (NVFP4 KV pending FlashInfer patch — quality regression risk if forced)
- **MTP**: ✅ active, 1 spec token, 88–93% acceptance rate
- **vLLM**: v0.24.0 source-built, `TORCH_CUDA_ARCH_LIST=12.1`
- **FlashInfer**: PR #3684 branch (`nvfp4-vosplit-rederive`), compiled from source
- **Quantization**: modelopt_mixed (MLP W4A16_NVFP4 group_size=16, attention FP8, lm_head NVFP4)
- **GPU**: NVIDIA GB10 SM121, CUDA 13.0, Torch 2.11.0+cu130
- **Image**: 15.4 GB (`sm121-vllm-v0240-nvfp4:kv-exp`)
- Prefix caching: **disabled by design** (hybrid GDN non-causal attention — do not force-enable)

### Performance on DGX Spark (FP8 KV + MTP, 256-token gen, 8K context)
| Concurrency | Output tok/s | Power (W) | Efficiency (J/1K tok) | Temp (°C) |
|---:|---:|---:|---:|---:|
| 1 | **19.15** | 34.3 | 1,793 | 60.5 |
| 4 | 69.62 | 32.3 | 465 | 60.8 |
| 8 | 102.76 | 36.9 | 359 | 62.7 |
| 16 | 144.00 | 38.9 | 270 | 64.7 |
| 32 | **248.40** | 44.2 | 178 | 67.6 |

### MTP vs KV-capacity tradeoff
| Mode | KV tokens | c1 tok/s | c32 tok/s |
|---|---|---|---|
| NVFP4 KV, 32K ctx, no MTP | 2,846,446 | 12.13 | 239.24 |
| NVFP4 KV, 8K ctx, MTP | 1,109,560 | 19.15 | 248.40 |
| FP8 KV, 32K ctx, MTP | 1,702,722 | 19.78 | 222.84 |

### GSM8K accuracy (lm-eval-harness, full 1319 samples, FP8 KV)
| Protocol | Score | Stderr |
|---|---|---|
| 0-shot, flexible-extract | **81.88%** | ±1.06% |
| 8-shot, flexible-extract | 76.80% | ±1.16% |

### Sanity suite: 5/5 passed (math, code, logic, factual, instruction-following)

### Six fixes required to build on SM121 (documented in repo)
1. OOM killer during build → `MAX_JOBS=6` (was 20)
2. Missing Python packages → bulk site-packages COPY
3. PTX version mismatch → CUDA 13.0 toolkit (not 13.2)
4. No C compiler at runtime → `build-essential`
5. No Ninja → `ninja-build`
6. Missing CUDA dev headers → `cuda-nvcc-13-0` + `cuda-libraries-dev-13-0`

---

## 7. Spark Arena Leaderboard — Qwen3.6-27B entries

Source: https://spark-arena.com/leaderboard (default test: `tg128 @ d16384`, single-stream decode, updated 2026-07-18 12:30 PM). 141 total entries across all models.

### Qwen3.6-27B entries on Spark Arena (decode test, sorted by tok/s)
| Rank | Model | Runtime | Quant | Cluster | Tok/s |
|---:|---|---|---|---|---:|
| 79 | Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP | vLLM | NVFP4 | 2 nodes | 45.78 |
| 82 | Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP | vLLM | NVFP4 | Single | 44.09 |
| 87 | Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP | vLLM | NVFP4 | Single | 41.27 |
| 93 | Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm | vLLM | NVFP4 | Single | 39.31 |
| 99 | Qwen3.6-27B-FP8 | SGLang | FP8 | Single | 36.63 |
| 109 | Qwen3.6-27B-FP8 | vLLM | FP8 | Single | 32.09 |

### Comparison vs our current models on the same test
| Model | Quant | Tok/s (decode, c1, d16384) | Notes |
|---|---|---:|---|
| **Qwen3.6-35B-A3B-NVFP4** | NVFP4 | **178.29** (top of chart, rank #4) | Our current 35B MoE — only 3B active |
| Qwen3.6-27B-FP8 (vLLM) | FP8 | 32.09 | Dense 27B active |
| Qwen3.6-27B-FP8 (SGLang) | FP8 | 36.63 | Dense 27B active, SGLang faster |
| Qwen3.6-27B NVFP4 variants | NVFP4 | 39–46 | Best 27B decode on Spark |

**Key takeaway:** On DGX Spark single-stream decode, our 35B-A3B-NVFP4 is **~4–5× faster** than Qwen3.6-27B-FP8/NVFP4 because the 35B is MoE with only 3B active params, while the 27B is dense (27B active). The 27B is quality-superior but throughput-inferior on this hardware. For throughput-sensitive serving the 35B-A3B remains the better choice; for quality-sensitive single-stream tasks (e.g. agentic coding) the 27B may justify the latency.

---

## 8. DFlash Speculative Decoding

`z-lab/Qwen3.6-27B-DFlash` is a **drafter** model that pairs with the target `Qwen/Qwen3.6-27B`. DFlash is a novel block-diffusion speculative decoding method (arXiv:2602.06036, GitHub: github.com/z-lab/dflash) that enables high-quality parallel drafting with up to **15 speculative tokens**.

- vLLM: `--speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 15}' --attention-backend flash_attn`
- SGLang: `--speculative-algorithm DFLASH` (PR #23000)
- Still under training; inference engine support may be incomplete due to causal SWA architectural changes
- Paper: https://arxiv.org/abs/2602.06036
- Blog: https://z-lab.ai/projects/dflash/

---

## 9. Community Sentiment (partial — X/Reddit blocked)

**Note:** xAI/X search was out of credits and Reddit returned a bot-block ("whoa there, pardner!") during this research session, so direct community-quote collection was limited. The signals we *did* gather:

- **HF download counts** (strong adoption signal): official BF16 has 5.4M downloads, official FP8 has 5.8M, NVIDIA NVFP4 has 1.5M, unsloth GGUF variants have 2.9M+ — this is a heavily-used model.
- **Spark Arena**: 6 community-submitted Qwen3.6-27B benchmarks exist (abliterated, AEON-Ultimate, PrismaSCOUT, FP8) — community is actively benchmarking it on DGX Spark.
- **Uncensored/abliterated ecosystem**: HauhauCS, DavidAU, huihui-ai, bottlecapai all shipped variants within weeks — strong hobbyist dev interest.
- **Qwen blog framing**: "Flagship-Level Coding in a 27B Dense Model" — Qwen themselves lean into the "punches above its weight" narrative, explicitly claiming it surpasses the 397B-A17B flagship on every major coding benchmark.
- **AA framing**: "Amongst the leading models in intelligence" but flagged as expensive/slow/verbose on the DashScope API (this is API pricing, not local-inference reality).

For a fuller community read, a follow-up pass with X/Reddit access (or a manual browse of r/LocalLLaMA, r/DGXSparc, @nousresearch threads) is recommended.

---

## 10. Comparison vs Our Current Spark Models

| Dimension | **Qwen3.6-27B** (candidate) | **Qwen3.6-35B-A3B** (current 35B) | **Qwen3.5-122B-A10B** (current 122B) |
|---|---|---|---|
| Architecture | Dense | MoE (3B active) | MoE (10B active) |
| Total params | 27B | 35B | 122B |
| Active params | 27B | 3B | 10B |
| License | Apache 2.0 | Apache 2.0 | (Qwen3.5 license) |
| Native ctx | 262K | 262K | (Qwen3.5 ctx) |
| MTP | Yes | Yes | (varies) |
| DFlash | Yes | Yes (z-lab variant exists) | No |
| AA Intelligence Index | **37** | 32 | (not in shown list) |
| SWE-bench Verified | **77.2** | 73.4 | (Qwen3.5-122B not in this table) |
| Terminal-Bench 2.0 | **59.3** | 51.5 | — |
| GPQA Diamond | **87.8** | 86.0 | — |
| AIME26 | **94.1** | 92.7 | — |
| HMMT Feb 25 | **93.8** | 90.7 | — |
| LiveCodeBench v6 | **83.9** | 80.4 | — |
| DGX Spark decode (c1, d16384) | 32–46 t/s (FP8/NVFP4) | **178 t/s** (NVFP4) | (check STATE.md) |
| DGX Spark best quant | NVFP4 (~14GB) | NVFP4 (~14GB) | INT4-FP8-hybrid |
| Multimodal | Yes (text+image+video) | Yes | Yes |

### Verdict
- **Quality:** Qwen3.6-27B ≥ Qwen3.6-35B-A3B on nearly every benchmark, especially coding & reasoning. The "punches above its weight class" claim holds — its AA Intelligence Index (37) beats the 35B-A3B (32) and it even beats the 397B-A17B flagship on coding.
- **Throughput on Spark:** Qwen3.6-27B is **much slower** than 35B-A3B-NVFP4 (32–46 vs 178 t/s) because dense 27B active >> MoE 3B active. The 35B-A3B remains the throughput champion on DGX Spark.
- **Use case fit:**
  - For **agentic coding / quality-critical single-stream** work where latency tolerance is higher → Qwen3.6-27B is a quality upgrade over 35B-A3B.
  - For **high-throughput serving / many concurrent users** → 35B-A3B-NVFP4 remains superior on this hardware.
  - For **comparison vs 122B-A10B** → 27B would need its own Spark benchmark run; 122B is heavier but also MoE (10B active), likely faster than dense 27B at c1 but slower at high concurrency. Need STATE.md data to confirm.

---

## 11. Sources

| Source | URL | What we got |
|---|---|---|
| HuggingFace model card | https://huggingface.co/Qwen/Qwen3.6-27B | Full benchmark tables (language + vision), architecture, MTP, context, license, serving commands |
| HuggingFace README (raw) | https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md | Same data in markdown/HTML for parsing |
| Qwen blog | https://qwen.ai/blog?id=qwen3.6-27b | "Flagship-Level Coding in a 27B Dense Model" framing, comparison narrative |
| Artificial Analysis | https://artificialanalysis.ai/models/qwen3-6-27b | AA Intelligence Index (37), per-eval breakdown, pricing, openness, comparison summary |
| mr_r0b0t GitHub repo | https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4 | DGX Spark NVFP4 serving recipe, throughput tables, GSM8K accuracy, six SM121 build fixes |
| Spark Arena | https://spark-arena.com/leaderboard | 6 community-submitted Qwen3.6-27B benchmarks on DGX Spark (decode test) |
| HuggingFace API search | https://huggingface.co/api/models?search=Qwen3.6-27B | Full quantization landscape (50+ repos) |
| z-lab DFlash repo | https://huggingface.co/z-lab/Qwen3.6-27B-DFlash | DFlash speculative-decoding drafter model + paper/blog links |

### Sources that were blocked / unavailable this session
- **web_search / web_extract**: Firecrawl not configured (`FIRECRAWL_API_KEY` missing). Worked around with direct `curl` + browser navigation.
- **x_search (xAI)**: Out of credits / no Grok subscription. X/Twitter community sentiment not collected.
- **Reddit (r/LocalLLaMA)**: Returned "whoa there, pardner!" bot block on both JSON API and old.reddit.com. Reddit community discussion not collected.
- **Artificial Analysis alt slugs** (`qwen-3-6-27b`, `qwen-3.6-27b`) returned 404; correct slug is `qwen3-6-27b`.

---

## 12. Recommended Next Steps

1. **Run our own Qwen3.6-27B-NVFP4 benchmark on DGX Spark** using mr_r0b0t's recipe (or our spark-vllm-docker stack) to get apples-to-apples numbers vs 35B-A3B-NVFP4 and 122B-A10B on our harness.
2. **Test DFlash speculative decoding** (`z-lab/Qwen3.6-27B-DFlash` + `Qwen/Qwen3.6-27B`) on Spark — 15 spec tokens could narrow the throughput gap vs the 35B MoE.
3. **Gather Reddit/X community sentiment** in a follow-up session with working credentials — r/LocalLLaMA, r/DGXSparc, and @drwl / @eugr / @QwenLM threads are the highest-signal channels.
4. **A/B test on Loca**: serve Qwen3.6-27B-NVFP4 alongside 35B-A3B-NVFP4 and compare coding-agent task success + latency on real workloads. The 27B's SkillsBench (48.2) and Claw-Eval Pass³ (60.6) gains over 35B-A3B (28.7 / 50.0) suggest it may materially outperform on agentic coding even at lower throughput.
5. **Update STATE.md** if/when we add a 27B recipe to the Spark rotation.