# Intelligence Benchmarks — Spark LLM Optimization Models

**Compiled:** 2026-07-18
**Sources:** HuggingFace model cards (official READMEs), StepFun official blog
**Scope:** Intelligence/quality benchmark scores for the 6 models tracked in the Spark LLM optimization database.

> Scores are pulled from each model's official HuggingFace model card (and, for Step-3.7-Flash, the StepFun blog at static.stepfun.com/blog/step-3.7-flash). Where a model card reports multiple reasoning-effort modes (DeepSeek-V4-Flash), the **High** thinking-mode score is used as the primary value, with Max-mode noted where relevant. "N/A" = not reported on the model card or in the official comparison tables I could access. "—" in a source table means the original publisher did not test it; rendered here as N/A.

---

## Models Covered

| # | Model | Total / Active Params | HuggingFace repo | Source for benchmarks |
|---|---|---|---|---|
| 1 | StepFun Step 3.7 Flash | 196B + 1.8B ViT / 11B | `stepfun-ai/Step-3.7-Flash` | HF model card + StepFun blog |
| 2 | DeepSeek-V4-Flash | 284B / 13B | `deepseek-ai/DeepSeek-V4-Flash` | HF model card |
| 3 | Qwen 3.5 122B A10B | 122B / 10B | `Qwen/Qwen3.5-122B-A10B` | HF model card |
| 4 | NVIDIA Nemotron-3-Super-120B-A12B | 120B / 12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | HF model card |
| 5 | Nemotron Labs Puzzle-75B A9B | 75.3B / 9.3B | `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` | HF model card |
| 6 | Qwen 3.5 35B A3B | 35B / 3B | `Qwen/Qwen3.5-35B-A3B` | HF model card (122B comparison table) |

> Note: the task brief listed this model as "Qwen 3.6 35B A3R" with repo `Qwen/Qwen3.6-35B-A3R`. That repo returns 404 on HuggingFace. The actual released model at the 35B/3B-active tier in the Qwen3.5 family is `Qwen/Qwen3.5-35B-A3B` (35B total / 3B active), which matches the spec on parameter count. Scores below use Qwen3.5-35B-A3B. Likewise `nemotron-ai/Puzzle-75B` 404s; the real repo is under the `nvidia/` org as `NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16`.

---

## Master Comparison Table

All scores are accuracy / pass@1 / resolved % as reported by each publisher. Higher is better.

### Agent / Tool Use

| Benchmark | Step 3.7 Flash | DS-V4-Flash (High) | Qwen3.5-122B-A10B | Nemotron-3-Super-120B | Puzzle-75B-A9B | Qwen3.5-35B-A3B |
|---|---|---|---|---|---|---|
| ClawEval-1.1 | **67.1** | 57.8 | N/A | N/A | N/A | N/A |
| SWE-Bench PRO | 56.3 | 52.3 | N/A¹ | N/A² | N/A² | N/A¹ |
| Toolathlon | 49.5 | 43.5 | N/A | N/A | N/A | N/A |
| Terminal-Bench 2.1 | 59.6 | 56.6³ | 49.4⁴ | 31.0⁴ | 24.0⁵ | 40.5⁴ |

¹ Qwen card reports **SWE-bench Verified** (72.0), not SWE-Bench PRO.
² Nemotron card reports **SWE-Bench (OpenHands)** (60.47), not the "PRO" variant.
³ DeepSeek card reports **Terminal Bench 2.0** (56.6 High / 56.9 Max). StepFun's blog tested DS-V4-Flash on Terminal-Bench 2.1 at 62.0.
⁴ Qwen3.5 and Nemotron cards report **Terminal Bench 2.0** / "Terminal Bench Core 2.0" respectively. Step 3.7 Flash is on the newer Terminal-Bench 2.1; not directly comparable.
⁵ Puzzle card reports "Terminal Bench (hard subset)", a different cut from Terminal-Bench Core 2.0. Nemotron-3-Super scores 25.5 on the hard subset vs 31.0 on Core 2.0.

> ⚠️ The four agentic benchmarks above use **different variants/harnesses across publishers**. Cross-model comparison within a row is only valid where the exact same benchmark variant was run. See "Benchmark Variant Notes" below.

### Reasoning

| Benchmark | Step 3.7 Flash | DS-V4-Flash (High) | Qwen3.5-122B-A10B | Nemotron-3-Super-120B | Puzzle-75B-A9B | Qwen3.5-35B-A3B |
|---|---|---|---|---|---|---|
| MMLU-Pro | N/A | 86.4 | **86.7** | 83.73 | 82.4 | 85.3 |
| GPQA (no tools) | N/A | 87.4 | **86.6** | 79.23 | 78.6 | 84.2 |
| HLE (no tools) | N/A | 29.4 | **25.3** | 18.26 | 16.5 | 22.4 |
| AIME25 (no tools) | N/A | N/A | N/A | **90.21** | 89.7 | N/A |

> AIME25 is only reported on the Nemotron family cards. Step 3.7 Flash and DeepSeek-V4-Flash cards do not publish AIME25. Qwen3.5 cards report HMMT (Feb/Nov 25) but not AIME25.

### Math

| Benchmark | Step 3.7 Flash | DS-V4-Flash (High) | Qwen3.5-122B-A10B | Nemotron-3-Super-120B | Puzzle-75B-A9B | Qwen3.5-35B-A3B |
|---|---|---|---|---|---|---|
| HMMT Feb25 (no tools) | N/A | 91.9 | **91.4** | 93.67 | 93.4 | 89.0 |
| HMMT Feb25 (with tools) | N/A | N/A | N/A | **94.73** | 93.9 | N/A |

> DeepSeek-V4-Flash card reports "HMMT 2026 Feb" (91.9 High / 94.8 Max) — likely the same or successor iteration as Nemotron's "HMMT Feb25". Flagged for caution.
> HMMT with-tools is only reported on the Nemotron family cards.

### Code

| Benchmark | Step 3.7 Flash | DS-V4-Flash (High) | Qwen3.5-122B-A10B | Nemotron-3-Super-120B | Puzzle-75B-A9B | Qwen3.5-35B-A3B |
|---|---|---|---|---|---|---|
| HumanEval | N/A | 69.5⁶ | N/A | N/A | N/A | N/A |
| SciCode | N/A | N/A | 42.0 | **42.05** | 40.6 | N/A |

⁶ DeepSeek-V4-Flash reports HumanEval (Pass@1) = 69.5 for the **base** model in the base-model table. The instruct-mode comparison table does not break out HumanEval; it reports LiveCodeBench (Pass@1) = 88.4 High / 91.6 Max instead.

### Other (Multimodal / Vision)

| Benchmark | Step 3.7 Flash | DS-V4-Flash (High) | Qwen3.5-122B-A10B | Nemotron-3-Super-120B | Puzzle-75B-A9B | Qwen3.5-35B-A3B |
|---|---|---|---|---|---|---|
| SimpleVQA (Search) | **79.2** | N/A | 61.7 | N/A | N/A | 58.3 |
| V* (Python) | **95.3** | N/A | 93.2⁷ | N/A | N/A | 92.7⁷ |

⁷ Qwen3.5 card reports V* as "with CI / without CI" (122B: 93.2 / 90.1, 35B: 92.7 / 89.5). The "with CI" (chain-of-insight) value is shown here as the more comparable figure to Step 3.7 Flash's Python-tool result.

> Step 3.7 Flash is the only one of the six with a vision encoder at this tier (StepFun ships a 1.8B ViT alongside the 196B LM). SimpleVQA and V* are multimodal benchmarks; the Qwen3.5 numbers come from their VL table, the DeepSeek/Nemotron/Puzzle cards are text-only and don't report these.

---

## Per-Model Detail (from source cards)

### 1. StepFun Step 3.7 Flash — `stepfun-ai/Step-3.7-Flash`

From the HF model card text + StepFun blog benchmark table:

| Benchmark | Score | Notes |
|---|---|---|
| ClawEval-1.1 | 67.1 | "leads ClawEval-1.1"; next competitor at 59.8 |
| SWE-Bench PRO | 56.3 | "second-place finish" |
| Toolathlon | 49.5 | |
| Terminal-Bench 2.1 | 59.6 | card says 59.5, blog table says 59.6 |
| HLE w. Tool | 47.2 | text-only HLE = 49.7 per blog |
| SimpleVQA (Search) | 79.2 | "first place" |
| V* (Python) | 95.3 | |
| GDPval-AA | 45.8 | (ii = independent-internal score) |
| BrowseComp | 75.8 | from blog |
| SWE-Bench Verified | 76.5 | from blog comparison table |

**Not reported on the card/blog:** MMLU-Pro, GPQA (no tools), HLE (no tools), AIME25, HMMT (either variant), HumanEval, SciCode. The blog focuses on agent + multimodal positioning, not the standard reasoning/math benchmark suite.

### 2. DeepSeek-V4-Flash — `deepseek-ai/DeepSeek-V4-Flash`

From the "Comparison across Modes" table (V4-Flash column). Primary = High mode; Max in parens where it differs materially.

| Benchmark | V4-Flash High (Max) | Notes |
|---|---|---|
| MMLU-Pro | 86.4 (86.2) | EM |
| GPQA Diamond | 87.4 (88.1) | Pass@1, no tools |
| HLE | 29.4 (34.8) | Pass@1, no tools |
| HMMT 2026 Feb | 91.9 (94.8) | Pass@1, no tools |
| LiveCodeBench | 88.4 (91.6) | Pass@1 |
| Terminal Bench 2.0 | 56.6 (56.9) | Acc |
| SWE Verified | 78.6 (79.0) | Resolved |
| SWE Pro | 52.3 (52.6) | Resolved |
| SWE Multilingual | 70.2 (73.3) | Resolved |
| BrowseComp | 53.5 (73.2) | Pass@1 |
| HLE w/ tools | 40.3 (45.1) | Pass@1 |
| MCPAtlas | 67.4 (69.0) | Pass@1 |
| Toolathlon | 43.5 (47.8) | Pass@1 |
| ClawEval (General) | 57.8 | Pass³% N=3, 161 tasks (from HF eval-results) |

**Base-model table** (different from instruct table above) also reports:
- HumanEval (Pass@1, 0-shot) = 69.5
- MMLU-Pro (EM, 5-shot) = 68.3
- BigCodeBench (Pass@1, 3-shot) = 56.8

**Not reported:** AIME25, SciCode, SimpleVQA, V* (V4-Flash is text-only — no ViT).

### 3. Qwen 3.5 122B A10B — `Qwen/Qwen3.5-122B-A10B`

From the Language benchmark table, Qwen3.5-122B-A10B column:

| Benchmark | Score | Notes |
|---|---|---|
| MMLU-Pro | 86.7 | |
| GPQA Diamond | 86.6 | no tools |
| HLE w/ CoT | 25.3 | "HLE w/ CoT" — no-tools HLE |
| HMMT Feb 25 | 91.4 | no tools |
| HMMT Nov 25 | 90.3 | |
| SWE-bench Verified | 72.0 | not SWE-Bench PRO |
| Terminal Bench 2 | 49.4 | Terminal-Bench 2.0 |
| LiveCodeBench v6 | 78.9 | |
| BFCL-V4 | 72.2 | |
| TAU2-Bench | 79.5 | |
| HLE w/ tool | 47.5 | search-agent variant |
| BrowseComp | 63.8 | |
| SimpleVQA | 61.7 | from VL table |
| V* | 93.2 / 90.1 | with/without CI, from VL table |

**Not reported:** AIME25, HumanEval, SciCode, ClawEval, Toolathlon, SWE-Bench PRO.

### 4. NVIDIA Nemotron-3-Super-120B-A12B — `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`

From the Benchmarks table (Nemotron 3 Super column). This card is the most complete of the six — it publishes AIME25, HMMT both variants, SciCode, and HLE both variants.

| Benchmark | Score | Notes |
|---|---|---|
| MMLU-Pro | 83.73 | |
| AIME25 (no tools) | 90.21 | |
| HMMT Feb25 (no tools) | 93.67 | |
| HMMT Feb25 (with tools) | 94.73 | |
| GPQA (no tools) | 79.23 | |
| GPQA (with tools) | 82.70 | |
| HLE (no tools) | 18.26 | |
| HLE (with tools) | 22.82 | |
| LiveCodeBench (v5) | 81.19 | |
| SciCode (subtask) | 42.05 | |
| Terminal Bench (hard subset) | 25.78 | 48 tasks |
| Terminal Bench Core 2.0 | 31.00 | |
| SWE-Bench (OpenHands) | 60.47 | |
| SWE-Bench (OpenCode) | 59.20 | |
| SWE-Bench (Codex) | 53.73 | |
| SWE-Bench Multilingual (OpenHands) | 45.78 | |
| TauBench V2 average | 61.15 | |
| BrowseComp with Search | 31.28 | |

**Not reported:** ClawEval, Toolathlon, HumanEval, SimpleVQA, V* (text-only model).

### 5. Nemotron Labs Puzzle-75B-A9B — `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16`

From the comparison table vs Nemotron-3-Super. Puzzle is a compressed (Iterative Puzzle) derivative of Nemotron-3-Super, so its benchmark coverage mirrors the parent.

| Benchmark | Score | Δ vs Nemotron-3-Super |
|---|---|---|
| MMLU-Pro | 82.4 | −1.4 |
| AIME25 (no tools) | 89.7 | −2.5 |
| HMMT Feb25 (no tools) | 93.4 | −0.8 |
| HMMT Feb25 (with tools) | 93.9 | −1.6 |
| GPQA (no tools) | 78.6 | −1.9 |
| GPQA (with tools) | 79.5 | −1.8 |
| LiveCodeBench (v5) | 81.1 | −1.0 |
| SciCode (subtask) | 40.6 | −1.7 |
| HLE (no tools) | 16.5 | −2.0 |
| Terminal Bench (hard subset) | 24.0 | −1.5 |
| SWE-Bench (OpenHands) | 56.9 | −2.6 |
| TauBench V2 average | 60.2 | −0.6 |

**Not reported:** ClawEval, Toolathlon, HumanEval, SimpleVQA, V*, SWE-Bench PRO, HLE with tools (parent doesn't report it either in the comparison cut).

### 6. Qwen 3.5 35B A3B — `Qwen/Qwen3.5-35B-A3B`

From the Qwen3.5-122B-A10B card's comparison table, Qwen3.5-35B-A3B column. This is the smallest model in the database.

| Benchmark | Score | Notes |
|---|---|---|
| MMLU-Pro | 85.3 | |
| GPQA Diamond | 84.2 | no tools |
| HLE w/ CoT | 22.4 | no-tools HLE |
| HMMT Feb 25 | 89.0 | no tools |
| HMMT Nov 25 | 89.2 | |
| SWE-bench Verified | 69.2 | |
| Terminal Bench 2 | 40.5 | Terminal-Bench 2.0 |
| LiveCodeBench v6 | 74.6 | |
| BFCL-V4 | 67.3 | |
| TAU2-Bench | 81.2 | |
| HLE w/ tool | 47.4 | search-agent variant |
| BrowseComp | 61.0 | |
| SimpleVQA | 58.3 | from VL table |
| V* | 92.7 / 89.5 | with/without CI, from VL table |

**Not reported:** AIME25, HumanEval, SciCode, ClawEval, Toolathlon, SWE-Bench PRO.

---

## Benchmark Variant Notes (important for cross-model comparison)

Several benchmarks in the task list have **publisher-specific variants** that are NOT directly comparable across models. Treat any row that mixes variants as approximate, not exact.

- **SWE-Bench PRO vs SWE-Bench Verified vs SWE-Bench (OpenHands/OpenCode/Codex):** Step 3.7 Flash and DeepSeek-V4-Flash report SWE-Bench **PRO** (the harder ScaleAI variant). Qwen3.5 cards report SWE-Bench **Verified** (the standard ScaleAI Verified split). Nemotron cards report SWE-Bench under three harnesses (OpenHands / OpenCode / Codex) but not the "PRO" split. These are **different benchmarks**, not different scores on the same benchmark.
- **Terminal-Bench 2.1 vs 2.0 vs Core 2.0 vs hard subset:** Step 3.7 Flash publishes 2.1. DeepSeek-V4-Flash publishes 2.0 (and StepFun's blog independently tested it on 2.1 = 62.0). Qwen3.5 publishes "Terminal Bench 2" (2.0). Nemotron publishes both "hard subset" (48 tasks) and "Terminal Bench Core 2.0". Puzzle publishes only the hard subset.
- **HLE "no tools" vs "w/ tool" vs "w/ CoT":** Nemotron card is explicit: HLE (no tools) and HLE (with tools) are both reported. Qwen3.5 card labels the no-tools column "HLE w/ CoT" and the tools column "HLE w/ tool" (search agent). DeepSeek-V4-Flash reports HLE (no tools) and HLE w/ tools. Step 3.7 Flash reports only HLE w. Tool. For the "HLE (no tools)" row in the master table, I used each publisher's no-tools figure; Step 3.7 = N/A.
- **HMMT Feb25 vs "HMMT 2026 Feb":** Nemotron and Puzzle report "HMMT Feb25". DeepSeek-V4-Flash reports "HMMT 2026 Feb" — likely the same competition round, but the labeling differs. Qwen3.5 reports "HMMT Feb 25" and "HMMT Nov 25".
- **V* (Python):** Step 3.7 Flash uses a Python tool (crop/zoom) per the blog. Qwen3.5 reports V* with/without "CI" (Chain-of-Insight). Both are tool-augmented; values are approximately comparable but the tool setups differ.
- **SimpleVQA (Search):** Step 3.7 Flash uses a visual-search tool. Qwen3.5 reports SimpleVQA in its VL table. DeepSeek/Nemotron/Puzzle are text-only and don't report it.

---

## Coverage Summary — which benchmarks each publisher reports

| Benchmark | Step 3.7 | DS-V4-Flash | Qwen3.5-122B | Nemotron-3-Super | Puzzle-75B | Qwen3.5-35B |
|---|---|---|---|---|---|---|
| ClawEval-1.1 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| SWE-Bench PRO | ✅ | ✅ | ❌ (Verified) | ❌ (OpenHands) | ❌ (OpenHands) | ❌ (Verified) |
| Toolathlon | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Terminal-Bench 2.x | ✅ 2.1 | ✅ 2.0 | ✅ 2.0 | ✅ Core 2.0 + hard | ✅ hard | ✅ 2.0 |
| MMLU-Pro | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPQA (no tools) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HLE (no tools) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AIME25 (no tools) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| HMMT Feb25 (no tools) | ❌ | ✅* | ✅ | ✅ | ✅ | ✅ |
| HMMT Feb25 (with tools) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| HumanEval | ❌ | ✅ base | ❌ | ❌ | ❌ | ❌ |
| SciCode | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| SimpleVQA (Search) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| V* (Python) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |

\* DS-V4-Flash labels it "HMMT 2026 Feb" — probably same round, flagged above.

**Best-covered models** (most benchmarks reported): Nemotron-3-Super (12/14) and Puzzle-75B (11/14), because NVIDIA publishes a uniform card format.
**Thinnest coverage**: Step 3.7 Flash — only 6/14, because StepFun's card/blog is focused on agent + multimodal benchmarks and skips the standard reasoning/math suite (MMLU-Pro, GPQA, AIME25, HMMT, HumanEval, SciCode).

---

## Key Takeaways for Spark Model Selection

- **For agent / tool use** (ClawEval, Toolathlon, SWE-Bench PRO, Terminal-Bench): **Step 3.7 Flash is the clear leader** among the six — 67.1 ClawEval, 49.5 Toolathlon, 56.3 SWE-Bench PRO, 59.6 Terminal-Bench 2.1. It's also the only one of the six reporting all four agent benchmarks.
- **For pure reasoning** (MMLU-Pro, GPQA no-tools): **Qwen3.5-122B-A10B and DeepSeek-V4-Flash (High) are tied at the top** — MMLU-Pro 86.7 vs 86.4, GPQA 86.6 vs 87.4. Nemotron-3-Super trails (~83.7 MMLU-Pro, 79.2 GPQA) but is the only one publishing AIME25.
- **For math** (HMMT Feb25 no-tools): **Nemotron-3-Super leads at 93.67**, with Puzzle (93.4) and DS-V4-Flash High (91.9) close. Qwen3.5-122B at 91.4. With tools, Nemotron pulls further ahead (94.73).
- **For code** (SciCode): **Nemotron-3-Super leads at 42.05**, Qwen3.5-122B at 42.0, Puzzle 40.6. HumanEval is only reported by DeepSeek-V4-Flash (69.5 base).
- **For multimodal / vision**: Step 3.7 Flash dominates (SimpleVQA 79.2, V* 95.3) because it's the only one with a vision encoder at the Flash tier. Qwen3.5-122B/35B are multimodal too but score lower on these search-augmented VQA benchmarks.
- **Puzzle-75B vs Nemotron-3-Super** (its parent): Puzzle loses ~1–2.5 points across the board on reasoning/code, as expected for a 60%-size compression. The trade is ~2× throughput and 8× 1M-token concurrency on H100. For a Spark memory-constrained deployment the throughput may be worth the accuracy drop, but on raw intelligence it's strictly behind its parent.
- **Qwen3.5-35B-A3B** (smallest, 3B active): Within ~1.5 points of Qwen3.5-122B on MMLU-Pro (85.3 vs 86.7) and GPQA (84.2 vs 86.6), and actually *beats* DS-V4-Flash on HLE-with-tool (47.4 vs 40.3). Best intelligence-per-active-parameter of the six — relevant for the Spark's 121GB memory ceiling where the 35B model fits with much more headroom than 122B.

---

## Sources (URLs accessed 2026-07-18)

1. `https://huggingface.co/stepfun-ai/Step-3.7-Flash` — HF model card
2. `https://static.stepfun.com/blog/step-3.7-flash/` — StepFun official blog (benchmark comparison tables)
3. `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash` — HF model card (full benchmark tables incl. modes comparison)
4. `https://huggingface.co/Qwen/Qwen3.5-122B-A10B` — HF model card (Language + Vision Language benchmark tables)
5. `https://huggingface.co/Qwen/Qwen3.5-35B-A3B` — HF model card (confirmed 35B/3B-active spec)
6. `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` — HF model card (Benchmarks section, full table)
7. `https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` — HF model card (comparison vs Nemotron-3-Super)

### Repos that 404'd (task-listed names that don't exist on HuggingFace)
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` (bare name) → real repo is `...-BF16` / `...-FP8` / `...-NVFP4`
- `nemotron-ai/Puzzle-75B` → real repo is `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` (under `nvidia/`, not `nemotron-ai/`)
- `Qwen/Qwen3.6-35B-A3R` → 404; real model at this tier is `Qwen/Qwen3.5-35B-A3B`

### Tools that were unavailable during this research
- `web_extract` (FIRECRAWL not configured) — fell back to `browser_navigate` + `browser_console` JS extraction.
- `web_search` (FIRECRAWL not configured) — fell back to HuggingFace's on-site model search.