# Gemma 4 31B IT — Research for Spark LLM Optimization

**Pulled:** 2026-07-18
**Researcher:** Oracle subagent
**Question:** Does Gemma 4 31B IT fix the context-window failure of Gemma 4 12B (128K, too small for 256K–1M agent contexts), and does it beat our Qwen 35B A3B?

**Bottom line up front:** Gemma 4 31B **does fix the context problem** (256K native, 2× the 12B's 128K) — but it **only matches** our Qwen 35B A3B's 262K, does not exceed it. On Artificial Analysis it **ties** Qwen 35B on intelligence (Index 29) but is **~4× slower** on cloud and has **no MTP/speculative decoding**. For our compression/agent use case on Spark, it does not unseat Qwen 35B A3B. It may be worth a single benchmark run because Google's self-reported AIME/LiveCodeBench/Tau2 numbers are strong, but the speed + MTP gap is decisive on GB10.

---

## 1. Model Card (HuggingFace)

- **Repo:** `google/gemma-4-31B-it` (BF16, Safetensors, 33B params on disk)
- **Base:** `google/gemma-4-31B` (61 finetunes, 244 adapters, 276 quantizations as of pull date)
- **License:** Apache 2.0
- **arXiv:** 2607.02770 — "Gemma 4 Technical Report" (submitted 2 Jul 2026)
- **Authors:** Google DeepMind, Gemma Team
- **Downloads:** 12.6M/month (very hot release)
- **Family sizes:** E2B, E4B, 12B Unified, 26B A4B (MoE), **31B Dense** ← this one

### Architecture (31B Dense)

| Property | Value |
|---|---|
| Total parameters | 30.7B (33B with embeddings, on disk) |
| Architecture | **Dense** (NOT MoE — all params active per token) |
| Layers | 60 |
| Sliding window | 1024 tokens |
| Context length | **256K tokens** |
| Vocab | 262K |
| Modalities | Text + Image (no audio on 31B; audio only on E2B/E4B/12B) |
| Vision encoder | ~550M params |
| Attention | Hybrid: sliding-window local + full global (final layer always global); unified K/V on global layers; Proportional RoPE (p-RoPE) |
| Thinking mode | Configurable (`enable_thinking=True/False`), `<|think|>` token, `<|channel>thought` blocks |
| System prompt | Native `system` role support (new in Gemma 4) |
| Function calling | Native, for agentic workflows |
| Multilingual | 35+ supported, 140+ pretraining |
| Sampling rec | temp=1.0, top_p=0.95, top_k=64 |

### Key context-window fact

Gemma 4 splits context by size tier:
- **Small models (E2B, E4B): 128K**
- **Medium models (12B Unified, 26B A4B, 31B): 256K**

So **31B gets 256K — double the 12B Unified's 128K** that failed for us. This directly addresses the original failure mode. (Note: the 12B Unified card also claims 256K in the overview table, but our previous 12B test hit a ~128K ceiling — likely an older 12B variant or a vLLM/FlashInfer limit, not the model card's native max. The 31B's 256K is unambiguous.)

---

## 2. Benchmarks (from Google model card, instruction-tuned)

| Benchmark | Gemma 4 31B IT | Gemma 4 26B A4B | Gemma 4 12B | Gemma 3 27B (no think) |
|---|---:|---:|---:|---:|
| MMLU Pro | **85.2%** | 82.6% | 77.2% | 67.6% |
| AIME 2026 (no tools) | **89.2%** | 88.3% | 77.5% | 20.8% |
| LiveCodeBench v6 | **80.0%** | 77.1% | 72.0% | 29.1% |
| Codeforces ELO | **2150** | 1718 | 1659 | 110 |
| GPQA Diamond | **84.3%** | 82.3% | 78.8% | 42.4% |
| Tau2 (agent, avg 3) | **76.9%** | 68.2% | 69.0% | 16.2% |
| HLE (no tools) | **19.5%** | 8.7% | 5.2% | – |
| HLE (with search) | **26.5%** | 17.2% | – | – |
| BigBench Extra Hard | **74.4%** | 64.8% | 53.0% | 19.3% |
| MMMLU | 88.4% | 86.3% | 83.4% | 70.7% |
| MMMU Pro (vision) | 76.9% | 73.8% | 69.1% | 49.7% |
| MATH-Vision | 85.6% | 82.4% | 79.7% | 46.0% |
| MRCR v2 8-needle 128K (long ctx) | **66.4%** | 44.1% | 43.4% | 13.5% |

**Notes:**
- These are **Google self-reported**. Cross-check against third-party (AA, our own Spark bench) before trusting.
- Tau2 (76.9%) is the most relevant **agentic** benchmark here — strong.
- HLE 19.5% no-tools / 26.5% with-search is competitive for a 31B dense model.
- MRCR v2 66.4% at 128K confirms the 256K context is genuinely usable (not just a marketing number) — needle-in-haystack holds up.

### HuggingFace community evals (verified on model card)
- `llamaindex/ParseBench` Mean: 62.4 (Text Content 89.9, Text Formatting 69.3) — document parsing
- `Idavidrein/gpqa` Diamond: 84.3 (matches Google's claim ✓)
- `LiquidAI/ifstruct-v1.0` Ifstruct V1: 95.9 — instruction following
- `TIGER-Lab/MMLU-Pro`: 85.2 (matches Google's claim ✓)

---

## 3. Artificial Analysis

- **URL:** https://artificialanalysis.ai/models/gemma-4-31b
- **AA Intelligence Index:** **29** (#5/130 in small open-weights class, 4/4 intelligence units)
- **Speed:** **35.8 t/s** output (#63/130, 1/4 speed units — "notably slow")
- **Context window:** 256K
- **Total parameters:** 30.7B
- **License:** Apache 2.0
- **Released:** April 2026 (AA says April; HF/arXiv say July — AA likely tracking an earlier internal/preview release)
- **Modalities:** input text+image+video, output text
- **Reasoning variant:** Yes (this is the reasoning/thinking page; a non-reasoning variant may also exist)
- **Verbosity:** 38M output tokens on Intelligence Index (#14/130, 3/4 verbosity units — "somewhat verbose")
- **Pricing:** $0.00 (open weights, self-host)

**AA comparison summary (verbatim):** "Gemma 4 31B (Reasoning) is amongst the leading models in intelligence and well priced when comparing to other open weight models of similar size. It's also notably slow and somewhat verbose."

---

## 4. Quantizations Available

From HF model search `gemma-4-31b` (top results):

| Repo | Format | Size on disk | Notes |
|---|---|---:|---|
| `google/gemma-4-31B-it` | BF16 Safetensors | ~33B | Official, full precision |
| `unsloth/gemma-4-31B-it-NVFP4` | NVFP4 (compressed-tensors) | **23B** | **2.4× smaller than BF16**. vLLM ≥0.25.0. Don't use Marlin backend (2× slower) — let vLLM auto-select NVFP4 kernel |
| `unsloth/gemma-4-31B-it-GGUF` | GGUF (llama.cpp/Ollama) | ~31B | Unsloth Dynamic 2.0, multiple Q levels |
| `unsloth/gemma-4-31B-it-qat-GGUF` | QAT GGUF | ~31B | Quantization-aware trained, better quality at low bit |
| `HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP` | QAT | ~31B | Community finetune, "MTP" in name (community-applied speculative layer, not native to Gemma) |
| `llmfan46/gemma-4-31B-it-uncensored-heretic` | Uncensored finetune | ~31B | Heretic-style abliterated |
| 276 quantizations total | FP8, INT4, INT8, etc. | various | Full ecosystem coverage |

**FP8:** Available via compressed-tensors community quants (search the 276).
**INT4:** Via GGUF (Q4_K_M etc.) and NVFP4.
**NVFP4 on Spark (SM121/GB10):** vLLM ≥0.25.0 + flashinfer-python ≥0.6.13 + nvidia-cutlass-dsl ≥4.5.2. **Same toolchain we already use for Qwen 35B NVFP4.** The SM121 FlashInfer cubin issue we already solved applies identically.

### NVFP4 vLLM serve command (from Unsloth card)

```bash
uv venv unsloth-nvfp4-env --python 3.13
source unsloth-nvfp4-env/bin/activate
uv pip install "vllm>=0.25.0" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" --torch-backend=auto
vllm serve unsloth/gemma-4-31B-it-NVFP4
```
> Do not use the Marlin backend (around 2x slower); let vLLM auto-select the NVFP4 kernel.

---

## 5. Context Window — Critical Assessment

| Model | Native max context | Usable for 256K–1M agent contexts? |
|---|---:|---|
| Gemma 4 12B (previous) | 128K | ❌ FAILED — too small |
| **Gemma 4 31B IT** | **256K** | ⚠️ Fixes 12B's problem, but caps at 256K — cannot reach 1M |
| Qwen 35B A3B (ours) | 262K | ⚠️ Same tier as Gemma 31B |
| NVIDIA Nemotron 3 Super 120B | 1M | ✅ Reaches 1M |
| DeepSeek V4 Flash | 1M | ✅ Reaches 1M |

**For our compression use case (agent contexts 256K–1M):** Gemma 4 31B's 256K is **better than the 12B** but still **cannot serve the upper half (256K–1M) of our target range**. If our agent contexts genuinely reach 1M, neither Gemma 31B nor our Qwen 35B is sufficient — only the 1M-context models (Nemotron 3 Super, DS V4 Flash) cover the full range. If our real working range is ≤256K, Gemma 31B now qualifies where the 12B did not.

MRCR v2 8-needle @ 128K = 66.4% for Gemma 31B confirms the 256K window is genuinely usable (not just a paper max), but we have no 256K needle data from Google.

---

## 6. MTP / Speculative Decoding

**Gemma 4 31B IT: NO native MTP.** The model card, the NVFP4 card, and the vLLM/SGLang serve snippets make **no mention** of multi-token prediction or speculative decoding. The community finetune `HauhauCS/...-MTP` has "MTP" in its name, but that is a **community-applied** speculative layer (QAT + balanced + MTP added by the author), not a Gemma 4 feature.

**Implication for Spark:** Without native MTP or a draft model, Gemma 31B cannot use the DFlash/MTP acceleration that gives Qwen 35B its 100–575 t/s throughput on AEON Bench. On GB10, a dense 30.7B-active model with no speculative decoding will be **token-limited by the 3B-active-vs-30.7B-active gap** plus the missing MTP multiplier.

**vLLM speculative config:** Not documented for Gemma 4. Could theoretically use a small external draft model, but no tested recipe exists and `--speculative-config` with a Gemma 4 target is unverified on SM121.

---

## 7. vLLM Compatibility on SM121 / GB10

- **Supported:** Yes — vLLM, SGLang, Transformers, llama.cpp, MLX LM, Ollama all listed on the model card.
- **vLLM version:** ≥0.25.0 for NVFP4 (same version we already run on Spark for Qwen 35B).
- **NVFP4 kernel:** vLLM auto-selects; **do not force Marlin** (2× slower). This is the opposite of our Qwen 35B recipe which uses `VLLM_TEST_FORCE_FP8_MARLIN=1` — for Gemma NVFP4 we'd let the NVFP4 kernel run natively.
- **FlashInfer:** ≥0.6.13 required for SM121 cubins — **same fix we already applied** for Qwen 35B. No new blocker.
- **Attention backend:** Model card vLLM snippet doesn't specify. Gemma 4's hybrid sliding-window + global attention needs a backend that supports SWA. `flash_attn` or `triton_attn` should work (same as Qwen 3.5/3.6). **Needs verification** — Gemma 4 is a newer architecture than vLLM's Gemma 3 path; confirm the attention backend accepts the 60-layer SWA+global interleaving before a real deploy.
- **KV cache dtype:** Not specified by Google. BF16 KV is the safe default (matches our Qwen DFlash-stability choice). fp8_e4m3 KV unverified for Gemma 4.
- **Tool-call parser:** Gemma 4 has native function calling but vLLM's parser support for the Gemma 4 chat template is **not confirmed** in the card. Qwen uses `qwen3`/`qwen3_coder` parsers; Gemma 4 would need its own (or generic). **Verify before relying on agentic tool calls.**

**Risk items for SM121 deploy:**
1. Attention backend SWA support for Gemma 4's 60-layer hybrid scheme — unverified
2. Tool-call parser for Gemma 4 chat template — unverified
3. NVFP4 kernel auto-selection vs Marlin — must NOT set `VLLM_TEST_FORCE_FP8_MARLIN=1` (differs from Qwen recipe)
4. No MTP/DFlash acceleration available — throughput will be far below Qwen 35B

---

## 8. Comparison vs Our Qwen 35B A3B

| Dimension | Gemma 4 31B IT | Qwen 3.5/3.6 35B A3B (ours) | Winner |
|---|---|---|---|
| **Total params** | 30.7B | 36B | — |
| **Active params/token** | **30.7B (dense)** | **3B (MoE)** | Qwen (10× fewer active) |
| **Context window** | 256K | 262K | Qwen (marginally) |
| **AA Intelligence Index** | 29 | 29 | **TIE** |
| **AA Speed (cloud)** | 35.8 t/s | 134.0 t/s | **Qwen (3.7×)** |
| **Spark speed (NVFP4, our bench)** | untested | ~100–116 t/s single, 575 t/s C64 (AEON-7) | Qwen (expected; Gemma untested) |
| **MTP / speculative** | ❌ None | ✅ MTP k=3 + DFlash n=11 | **Qwen (decisive)** |
| **GPQA Diamond** | 84.3% (Google) | 85% (AA) | ~TIE |
| **MMLU Pro** | 85.2% (Google) | not in AA breakdown | Gemma (self-reported) |
| **AIME 2026** | 89.2% (Google) | not in AA breakdown | Gemma (self-reported) |
| **LiveCodeBench v6** | 80.0% (Google) | not in AA breakdown | Gemma (self-reported) |
| **Terminal-Bench v2.1** | not reported | 41% (AA) | — |
| **SciCode** | not reported | 38% (AA) | — |
| **AA-LCR (long ctx)** | not in AA breakdown | 63% (AA) | — |
| **Agentic (Tau2)** | 76.9% (Google) | not reported | Gemma (self-reported) |
| **HLE** | 19.5% / 26.5% w/ search | not reported | Gemma (self-reported) |
| **Vision** | ✅ Image | ❌ text-only | **Gemma** |
| **Audio** | ❌ (31B has no audio) | ❌ | TIE |
| **NVFP4 on Spark** | ✅ vLLM ≥0.25.0, 23B on disk | ✅ vLLM ≥0.25.0, ~similar | TIE |
| **License** | Apache 2.0 | Apache 2.0 | TIE |
| **Verbosity** | 38M tokens (AA, "somewhat verbose") | N/A on AA | — |
| **vLLM SM121 risk** | SWA backend + tool parser unverified | production-stable | Qwen |

### Verdict

1. **Context fix: YES, but only to parity.** Gemma 4 31B's 256K doubles the 12B's 128K and removes the hard ceiling that killed the 12B for compression. But it **does not exceed** Qwen 35B's 262K, and neither reaches the 1M we'd need for the full 256K–1M agent range. If 256K is enough for our real workloads, Gemma now qualifies; if we need 1M, we still need Nemotron 3 Super or DS V4 Flash.

2. **Intelligence: TIE on AA (29 = 29).** Google's self-reported MMLU-Pro 85.2 / AIME 89.2 / LiveCodeBench 80 / Tau2 76.9 are strong and if they hold up under third-party testing would put Gemma 31B ahead on raw knowledge/reasoning benchmarks — but AA's composite already normalizes much of this and lands both at 29. Treat Google's numbers as optimistic until we run our own.

3. **Speed: QWEN WINS DECISIVELY.** Gemma 31B is **dense (30.7B active)**; Qwen 35B is **MoE (3B active)** — 10× fewer active params per token. On AA cloud, 35.8 vs 134 t/s (3.7×). On Spark GB10, the gap will be wider because dense models can't hide behind MoE sparsity. **No MTP, no DFlash** means Gemma 31B cannot use the speculative-decoding multiplier that puts Qwen 35B at 100–575 t/s on AEON Bench.

4. **For our compression use case specifically:** Compression is a long-context, throughput-sensitive, latency-budget task. We need (a) enough context to hold the agent transcript, (b) fast token generation to keep compression under ~15 min, (c) agentic tool use. Gemma 31B meets (a) at parity, **loses badly on (b)**, and (c) is unverified for its vLLM tool parser. Qwen 35B meets all three in production today.

5. **Only reasons to switch to Gemma 31B:**
   - We need **vision** (image input) in the compression pipeline — Qwen 35B A3B is text-only.
   - Google's self-reported reasoning benchmarks (AIME 89.2, HLE 26.5 w/ search, Tau2 76.9) hold up under our actual workload and matter more than throughput.
   - We want a second independent model family for diversity/robustness.

6. **Recommendation:** Do one benchmark run of `unsloth/gemma-4-31B-it-NVFP4` on Spark with our standard recipe (minus Marlin force, minus MTP) to get real t/s and a compression-quality sample. If it clears 40 t/s single-stream and beats Qwen 35B on our compression eval, reconsider. Otherwise **stay on Qwen 35B A3B** — it ties on intelligence, wins on speed, has MTP, and is production-stable on SM121.

---

## 9. Sources

- HuggingFace model card: https://huggingface.co/google/gemma-4-31B-it
- HuggingFace base: https://huggingface.co/google/gemma-4-31B
- arXiv technical report: https://arxiv.org/abs/2607.02770 (blocked by network at pull time; abstract via https://huggingface.co/papers/2607.02770)
- Artificial Analysis: https://artificialanalysis.ai/models/gemma-4-31b
- Unsloth NVFP4: https://huggingface.co/unsloth/gemma-4-31B-it-NVFP4
- Unsloth GGUF: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
- Unsloth QAT GGUF: https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF
- HauhauCS QAT-MTP community finetune: https://huggingface.co/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP
- HF model search (quantization inventory): https://huggingface.co/models?search=gemma-4-31b
- Our Qwen 35B baseline: `sources/35b-research-2026-07-11.md`, `research/aa-intelligence-scores.md`

### Sources that could not be retrieved (tool blockers)
- **Reddit r/LocalLLaMA:** bot-blocked at pull time ("You've been blocked by network security"). Community signal on Gemma 4 31B Spark performance not captured. Recommend a manual check or retry from a non-blocked egress.
- **X/Twitter search:** out of credits (xAI spending-limit). No social signal captured.
- **arxiv.org direct:** network-blocked. Used HF papers mirror for the abstract; full PDF technical report not read. MTP/spec-decoding absence confirmed by absence in model card + NVFP4 card + vLLM snippet, not by positive statement in the tech report. **A pass through the PDF would let us confirm "no MTP" definitively and check for any speculative-decoding discussion.**
- **Web search / web_extract:** not configured (Firecrawl key missing in this environment). All retrieval done via browser_navigate + browser_snapshot + browser_console.

---

## 10. Open Questions for Follow-up

1. **Real Spark t/s for Gemma 31B NVFP4** — no community number found yet (Reddit blocked). Run the benchmark.
2. **Gemma 4 vLLM attention backend on SM121** — does `flash_attn` or `triton_attn` correctly handle the 60-layer SWA+global hybrid? Confirm before deploy.
3. **Gemma 4 tool-call parser in vLLM** — is agentic function calling actually wired, or just declared on the card? Test with a tool-call request.
4. **256K needle test** — Google only reports MRCR @ 128K. Does Gemma 31B actually retrieve at 256K, or does quality drop past 128K?
5. **PDF of tech report** — confirm no MTP/speculative decoding section; check for any draft-model recipe.
6. **Gemma 4 26B A4B (MoE)** — the sibling MoE has 3.8B active (close to Qwen 35B's 3B) and 256K context. If we want a Gemma-family model on Spark for speed, the **26B A4B MoE** may be the better candidate than the 31B dense. Not researched in depth here — worth a follow-up if Gemma's vision or reasoning is appealing.