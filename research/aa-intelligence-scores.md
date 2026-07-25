# Artificial Analysis Intelligence Index Scores — Spark Models

**Source:** artificialanalysis.ai (Artificial Analysis Intelligence Index v4.1)
**Pulled:** 2026-07-18
**Methodology:** AA Intelligence Index v4.1 includes 9 evaluations:
GDPval-AA v2, τ³-Banking, Terminal-Bench v2.1, SciCode, Humanity's Last Exam,
GPQA Diamond, CritPt, AA-Omniscience, AA-LCR.

Scores are 0-100 for the composite index; individual evals are percentages
unless otherwise noted. Speed is output tokens/sec on the first-party cloud
API. "n/a" = model not tracked on Artificial Analysis.

## Composite Scores

| Model | AA Intelligence Index | Speed (t/s) | Cloud Input $/M | Cloud Output $/M | Context |
|---|---:|---:|---:|---:|---:|
| Step 3.7 Flash | **30** | 378.1 | $0.20 | $1.15 | 262k |
| DeepSeek V4 Flash (max) | **40** | 103.7 | $0.14 | $0.28 | 1M |
| Qwen3.5 122B A10B | **32** | 134.9 | $0.40 | $3.20 | 262k |
| NVIDIA Nemotron 3 Super 120B A12B | **25** | 146.1 | $0.25 | $0.775 | 1M |
| Qwen3.5 35B A3B | **29** | 134.0 | $0.25 | $2.00 | 262k |
| Puzzle-75B A9B | n/a | n/a | n/a | n/a | n/a |

**Puzzle-75B is not tracked on Artificial Analysis** (no model page exists;
verified by trying multiple URL slugs and scanning AA Intelligence Index
leaderboards — it does not appear). HuggingFace-only data remains the source
of record for this model.

## Individual Evaluation Breakdowns

Individual eval scores (in %). Higher is better. Sources: each model's AA
Intelligence Breakdown chart.

| Eval | Step 3.7 Flash | DS V4 Flash | Qwen 122B A10B | Nemotron 3 Super | Qwen 35B A3B | Puzzle-75B |
|---|---:|---:|---:|---:|---:|---:|
| GDPval-AA v2 (agentic work, Elo) | 18% | 34% | 24% | 8% | 15% | n/a |
| τ³-Banking (agentic tool use) | 11% | 23% | 14% | 10% | 5% | n/a |
| Terminal-Bench v2.1 (agentic coding) | 26% | 62% | 48% | 39% | 41% | n/a |
| SciCode (coding) | 36% | 45% | 42% | 38% | 38% | n/a |
| Humanity's Last Exam (reasoning) | 18% | 32% | 23% | 19% | 20% | n/a |
| GPQA Diamond (scientific reasoning) | 78% | 89% | 86% | 80% | 85% | n/a |
| CritPt (physics reasoning) | 2% | 7% | 1% | 3% | 1% | n/a |
| AA-Omniscience Accuracy (knowledge) | 26% | 37% | 25% | 24% | ~10% | n/a |
| AA-LCR (long context reasoning) | 64% | 63% | 67% | 60% | 63% | n/a |

## Per-Model Notes

### Step 3.7 Flash (StepFun) — Intelligence Index 30
- URL: https://artificialanalysis.ai/models/step-3-7-flash
- 198B total / 11B active MoE, Apache 2.0, released May 2026
- #37/97 in intelligence within class, #1/97 in speed
- AA class: open-weights large (>150B params)
- Standout: fastest in class at 378 t/s; very verbose (260M output tokens
  vs 92M avg). GPQA Diamond 78% is its strongest individual eval.
- Weakest: CritPt 2%, τ³-Banking 11% — agentic and physics tasks lag.

### DeepSeek V4 Flash (Reasoning, Max Effort) — Intelligence Index 40
- URL: https://artificialanalysis.ai/models/deepseek-v4-flash
- 284B total / 13B active MoE, MIT license, released April 2026
- #11/97 in intelligence within class, #16/97 in speed
- Best-in-class cache pricing ($0.003/M, -98%)
- Standout: highest AA Index among Spark models; strongest on Terminal-Bench
  (62%) and SciCode (45%) of the Spark set. GPQA Diamond 89%.
- Verbosity: 230M tokens (very verbose vs 92M avg).

### Qwen3.5 122B A10B (Alibaba) — Intelligence Index 32
- URL: https://artificialanalysis.ai/models/qwen3-5-122b-a10b
- 125B total / 10B active MoE, Apache 2.0, released Feb 2026
- #1/62 in intelligence within class (medium open-weights)
- Highest output price of the Spark set at $3.20/M (4/4 price units —
  expensive). $447 cost to run AA Intelligence Index.
- Standout: GPQA Diamond 86%, Terminal-Bench 48%. AA-LCR 67% (best of Spark
  set on long context).
- Weakest: CritPt 1%, GDPval 24%.

### NVIDIA Nemotron 3 Super 120B A12B — Intelligence Index 25
- URL: https://artificialanalysis.ai/models/nvidia-nemotron-3-super-120b-a12b
- 120.6B total / 12.7B active MoE, NVIDIA Nemotron Open Model License,
  released March 2026
- #4/62 in intelligence within class (medium open-weights)
- 1M context window — tied for largest in Spark set with DeepSeek V4 Flash
- Standout: AA-LCR 60%, GPQA Diamond 80%, Terminal-Bench 39%. Best cache
  hit price in Spark set at $0.143/M (-43%).
- Weakest: CritPt 3%, GDPval 8% — agentic work tasks lag.
- Verbosity: 110M tokens.

### Qwen3.5 35B A3B (Alibaba) — Intelligence Index 29
- URL: https://artificialanalysis.ai/models/qwen3-5-35b-a3b
- 36B total / 3B active MoE, Apache 2.0, released Feb 2026
- #7/130 in intelligence within class (small open-weights)
- Smallest model in Spark set. AA notes a newer Qwen3.6 35B A3B exists.
- Standout: GPQA Diamond 85%, AA-LCR 63%, Terminal-Bench 41%. Strong GPQA
  for its size.
- Weakest: CritPt 1%, τ³-Banking 5%, GDPval 15%.
- Verbosity: N/A (not reported on AA page).

### Puzzle-75B A9B (Nemotron Labs 3) — NOT ON ARTIFICIAL ANALYSIS
- No AA model page exists. Not listed in AA Intelligence Index leaderboard.
- Existing HuggingFace-only benchmark data remains the source of record:
  MMLU-Pro 82.4, AIME25 89.7, GPQA 78.6, HLE 16.5, SciCode 40.6
- If AA scores become necessary for this model, recommend requesting AA
  add it via their contact form, or computing equivalent scores from
  HuggingFace data using AA's published methodology.

## Cross-Model Observations

- **DeepSeek V4 Flash leads the Spark set** on the AA Intelligence Index
  (40), ahead of Qwen 122B (32), Step 3.7 Flash (30), Qwen 35B (29), and
  Nemotron 3 Super (25).
- **Speed crown: Step 3.7 Flash at 378 t/s** — nearly 4x faster than DS V4
  Flash (104 t/s). Important for latency-sensitive Spark workloads.
- **Best value (intelligence per dollar):** DeepSeek V4 Flash — Index 40 at
  $0.14/$0.28 per M tokens. Qwen 122B is the worst value (Index 32 at
  $0.40/$3.20).
- **Agentic tasks (GDPval + τ³-Banking + Terminal-Bench):** DeepSeek V4
  Flash dominates. Step 3.7 Flash and Nemotron 3 Super are weakest on
  agentic work.
- **Coding (SciCode):** DS V4 Flash 45% > Qwen 122B 42% > Nemotron 38% =
  Qwen 35B = Step 3.7 36%.
- **Long context (AA-LCR):** Qwen 122B 67% > Step 3.7 64% = Qwen 35B 63%
  = DS V4 Flash 63% > Nemotron 60%.
- **GPQA Diamond:** DS V4 Flash 89% > Qwen 122B 86% > Qwen 35B 85% >
  Nemotron 80% > Step 3.7 78%. All Spark models are competitive on
  scientific reasoning.
- **Physics (CritPt):** All Spark models score ≤7%. This eval
  strongly differentiates only top-tier proprietary models.
- **Coding Index / Agentic Index:** These are separate sub-indices on AA
  but their numeric values are not shown in the main model summary — they
  require clicking into a separate tab per model. Not extracted here.

## Files Modified

- Created: `research/aa-intelligence-scores.md` (this file)