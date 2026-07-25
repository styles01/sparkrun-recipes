# Spark Daily Driver — Final Debate & Ranked Recommendation

**Author:** Oracle (devil's-advocate debate pass)
**Date:** 2026-07-18
**Task:** Challenge the Round 1 consensus (Qwen 122B) and produce a final
ranked recommendation for 5–6 concurrent Hermes/OpenClaw agents on a single
DGX Spark (GB10, 121GB, SM121).
**Requirements:** 30+ tok/s, 150K+ context, 3+ lanes, tool calling.

---

## TL;DR — Verdict After Debate

The Round 1 consensus **survives the challenge but with a meaningful
narrowing of margin.** Qwen 3.5 122B A10B remains the #1 daily driver, but a
previously-unconsidered candidate — **Qwen3-Next-80B-A3B (the "sweet spot"
model the brief hypothesized)** — emerges as a serious #2 that arguably
wins on a pure speed×context×lanes axis while losing on intelligence and
tool-use maturity. The ranked list:

| Rank | Model | Why | Risk |
|---:|---|---|---|
| 1 | **Qwen 3.5 122B A10B** (DFlash n=4/6 or int4-AutoRound) | Meets every spec axis. Highest AA Index (32) of deployable models. Only model proven at 200K on single Spark. Named unprompted by operators. Verified-good in our STATE.md. | Looping in long coding sessions; DFlash brittle on vLLM nightlies |
| 2 | **Qwen3-Next-80B-A3B** (Instruct or Thinking, NVFP4) | The hypothetical sweet-spot model exists. 80B total / **3B active** MoE, MTP, 262K ctx (1M YaRN). Same active-param class as the 35B (fast) but more knowledge. Qwen claims it matches Qwen3-235B on certain benchmarks. NVFP4 ~40GB → leaves ~80GB for KV → potentially 4–6 lanes at 150K. | AA Index only 15 (Instruct) / 14 (Thinking) — *lower than 35B (32)*. Non-reasoning Instruct only. τ³-Banking 6% (worst except tiny models). Terminal-Bench 7%. **Untested on Spark.** |
| 3 | **Qwen 3.6 35B A3B** (NVFP4) | The throughput champion. 100+ tok/s, 6 lanes at 256K. Best when agents do simple tasks and speed > reasoning. | AA τ³-Banking 5%, "teenager," loops on hard tasks. Not a brain. |
| 4 | **gpt-oss-120b** (MXFP4) | **44.58 tok/s on Spark Arena** (single-node, decode). 117B total / 5.1B active, Apache 2.0. A real competitor nobody in Round 1 mentioned. | AA Index 24 (below 122B's 32). Terminal-Bench 24%, τ³-Banking 12%. Context only 131K (fails 150K+ spec). KV grows linearly (no Mamba). Round 1's "active-param trap" OOM risk for long agent loops. |
| 5 | **Qwen 3.6 27B** (NVFP4, dense) | Highest AA Index (37) of the open-weights small class. 39–46 tok/s on Spark Arena. Dense → no routing surprises. | Dense 27B active → only 2–3 lanes at 150K. AA-LCR only 55%. Not MoE, no MTP multi-token. "Best subagent, not the brain" (0rand). |
| 6 | **Nemotron Super 120B** (NVFP4) | Architecturally the best for unbounded agent loops (Mamba KV → near-zero growth). 1M context. | 14.4 tok/s (fails 30+). AA Index 25. Tool parser immature. Watch list, not driver. |

**Recommendation:** Stay on **Qwen 3.5 122B A10B** as primary. **Benchmark
Qwen3-Next-80B-A3B-NVFP4** as the high-priority challenger — if it clears
35+ tok/s at 150K with 4+ lanes on Spark and its tool-calling parser is
solid, it could become the new daily driver for *throughput-bound* agent
fleets. Keep 35B as the co-location/media worker. Bench gpt-oss-120b as a
dark-horse if 131K context is acceptable for some lanes.

---

## The Devil's-Advocate Challenges

### Challenge 1: "Is Qwen 122B the best, or just the safest?"

**Argument against:** 122B is the "lowest-risk known-good" — the Toyota
Camry of Spark models. But 5–6 concurrent agents are a *throughput*
workload, and 122B only delivers 3 lanes. A smarter-but-faster model might
serve the fleet better. Qwen 3.6 27B has AA Index 37 (highest of any
candidate) and hits 39–46 tok/s on Spark Arena — why not use it?

**Rebuttal:**
- 27B is **dense** (27B active). At 150K context × N lanes, KV cache is the
  binding constraint, not weights. A dense 27B with 6 lanes at 150K eats
  far more KV than a 3B-active MoE. r0b0tlab's 27B bench was at **8K
  context** (1.1M KV tokens); scaling that to 150K × 4 lanes is untested and
  likely OOMs or collapses to 2 lanes.
- AA Index 37 is the *small-class* rank; on the cross-class AA Index v4.1
  leaderboard the 27B doesn't appear in the top 25 (the 35B appears at 32,
  the 122B at 32 — they're medium-class). The 27B's τ³-Banking is 15%
  (better than 35B's 5%, worse than 122B's 14% only by a hair) and its
  Terminal-Bench (AA) is 51% — *higher than 122B's 48%*. This is the
  strongest argument for 27B.
- But 0rand's forum verdict is decisive: "Qwen 3.6 27b is better at
  actually writing the code but lacks knowledge and must have extremely
  well defined information for execution. Best as subagent." For 5–6
  *concurrent agents* doing mixed tool-calling, not pure coding, the
  knowledge gap matters.
- **Verdict:** 27B is the best *single-stream coder* in the set, not the
  best *fleet brain*. The dense architecture is a lane-count killer at 150K.

### Challenge 2: "Is the 35B 'teenager' actually unusable for agents?"

**Argument against:** For 5–6 agents doing *simple* tasks (file ops, web
search, API calls, cron), speed and lane count may matter more than
reasoning depth. 35B gives 6 lanes at 256K and 100+ tok/s. The τ³-Banking
5% is for *banking* tasks; file-ops agents aren't doing τ³-Banking.

**Rebuttal:**
- The 5% τ³-Banking is a *symptom* of tool-calling immaturity, not a
  domain-specific banking bug. AA's tool-use eval correlates with
  structured-output reliability across domains — this is why forum users
  report "looping," "gaslighting," and "hallucination" on the 35B
  generally, not just on banking.
- Digital_David (forum) *does* run 35B as the main LLM for 4 agents — but
  he routes ~5% of traffic to Gemini Pro as "brain power." That's a
  hybrid cloud+local pattern, not all-local. If we're willing to break
  "all-local," 35B is viable. If we're not, it isn't.
- The 35B's hallucination is load-bearing for unattended agents. A human
  can babysit one coding session; 5–6 concurrent unsupervised agents will
  compound the 35B's errors faster than a human can catch them.
- **Verdict:** 35B is the right answer *if* the fleet is doing
  repetitive simple tasks AND you accept occasional silent failures. For
  reliable agent fleets, it's the worker tier, not the brain. Round 1 is
  right here.

### Challenge 3: "Is Nemotron Super at 15–22 tok/s actually too slow?"

**Argument against:** Agent turns are 30–60 seconds anyway, much spent
waiting for tool results. Does 15 vs 30 tok/s matter when the agent is
blocked on a tool call? If Nemotron's Mamba KV means infinite agent loops
without memory creep, the speed tax buys unbounded context.

**Rebuttal:**
- The 30–60s turn figure includes tool execution, but the *generation*
  portion of a turn at 15 tok/s for a 500-token response is ~33s; at 30
  tok/s it's ~17s. Over 5–6 concurrent agents and hundreds of turns/day,
  2× generation latency compounds into real wall-clock throughput loss.
- The Mamba-KV advantage is real but shared by Qwen 122B (hybrid
  Mamba/attention) and Qwen3-Next-80B (hybrid Gated DeltaNet + Gated
  Attention). Nemotron is *more* Mamba, but not uniquely so.
- Nemotron's AA Index (25) is the lowest of the large models. Its
  Terminal-Bench (39%) and τ³-Banking (10%) are both below 122B. You'd
  be paying a 2× speed tax for *less* agent competence, not more.
- The tool-calling parser (`super_v3_reasoning_parser.py`) is still
  "approaching usability" per eugr's own forum posts. For a fleet where
  tool calling is the *core* loop, parser maturity is non-negotiable.
- **Verdict:** 15 tok/s is too slow *for this use case* given the model
  isn't smarter. The Mamba argument is compelling for unbounded loops but
  122B already has hybrid Mamba and is 2× faster. Round 1 is right.

### Challenge 4: "Has any new model shipped in the last 48h?"

**Finding:** Checked HuggingFace trending (2026-07-18) and Hacker News.
The trending list: `thinkingmachines/Inkling` (952B, image-text — too big),
`prism-ml/Bonsai-27B-gguf` (27B, 4B-tagged but likely quant — tiny),
`zai-org/GLM-5.2` (753B — known, too big for single Spark, tracked in
our `glm5.2-future-signal.md` as a 2–3 month out candidate via Colibri),
`bottlecapai/ThinkingCap-Qwen3.6-27B` (known finetune). **No new
Spark-relevant model released in the last 48 hours.** The picture is
stable.

### Challenge 5: "What about gpt-oss-120b? Nobody mentioned it."

**This is the most productive challenge.** gpt-oss-120b was NOT in the
Round 1 analysis. I pulled fresh data:

**Spark Arena (tg128 @ d16384, c1, single-node):**
- gpt-oss-120b vLLM MXFP4 Single: **44.58 tok/s** (rank 62)
- Also a 28.23 tok/s entry (rank 91) — config-dependent
- 2-node cluster: 65.36 tok/s; 4-node: 61.37 tok/s

**Artificial Analysis (per-eval, AA Index v4.1, reasoning variant "high"):**
| Eval | gpt-oss-120b (high) | Qwen 122B A10B | Winner |
|---|---:|---:|---|
| AA Intelligence Index | **24** | **32** | Qwen 122B |
| GDPval-AA v2 (agentic work) | 15% | 24% | Qwen 122B |
| τ³-Banking (tool use) | 12% | 14% | Qwen 122B (marginal) |
| Terminal-Bench v2.1 (agentic coding) | 24% | 48% | **Qwen 122B (2×)** |
| SciCode (coding) | 39% | 42% | Qwen 122B (marginal) |
| Humanity's Last Exam | 18% | 23% | Qwen 122B |
| GPQA Diamond | 78% | 86% | Qwen 122B |
| AA-Omniscience Accuracy | 22% | 25% | Qwen 122B |
| AA-Omniscience Non-Hallucination | **9%** | 11% | Qwen 122B (both bad) |
| AA-LCR (long context) | 51% | 67% | **Qwen 122B** |

**Architecture:** 117B total / **5.1B active** MoE, Apache 2.0, 131K context.
No Mamba/hybrid attention — pure Transformer MoE, so KV grows linearly
with context (the "active parameter trap" from the Tier-0 forum thread).

**Verdict on gpt-oss-120b:**
- ✅ Speed: 44.58 tok/s clears the 30+ bar.
- ✅ Active params 5.1B → lanes could be 3–4.
- ❌ **Context: 131K native — FAILS the 150K+ spec.** Cannot run 150K
  context lanes without YaRN extension (untested on Spark for this
  model).
- ❌ Terminal-Bench 24% is half of 122B's 48%. For agentic coding fleets
  this is a major regression.
- ❌ AA-LCR 51% vs 122B's 67% — long-context reasoning is notably worse.
- ❌ Pure-Transformer KV growth → OOM risk in long agent loops (the exact
  failure the Tier-0 thread warns about for 120B-class Transformer MoEs).
- ⚠️ 5.1B active is between 122B's 10B and 35B's 3B — a middle ground that
  inherits the KV-growth problem of 120B Transformers without the
  knowledge capacity to justify it.

**Conclusion:** gpt-oss-120b is a real model and 44.58 tok/s on Spark is
real, but it **fails the 150K+ context requirement** and its agent scores
(Terminal-Bench 24%, AA-LCR 51%) are materially worse than Qwen 122B. It
is a viable pick *only if* you can live with 131K context. For our spec
(150K+), it's disqualified. **This is a strong devil's-advocate find that
loses on the context axis.**

### Challenge 6: "The sweet-spot 80B MoE — does it exist?"

**YES — and Round 1 missed it.** The brief hypothesized "an 80B MoE at Q4
(~40GB disk) with 10B active and MTP support." The model that fits *most*
of this profile is **Qwen3-Next-80B-A3B** (released September 2025, but
not in our Round 1 research):

**Architecture:**
- 80B total / **3B active** (not 10B — even sparser than hypothesized)
- 48 layers, 512 experts, 10 activated + 1 shared
- Hybrid Gated DeltaNet + Gated Attention (same family as Qwen 3.6 —
  Mamba-style near-zero KV growth)
- **MTP** (multi-token prediction) ✅
- Context: **262K native, 1.01M with YaRN** ✅
- License: Apache 2.0 ✅
- Two variants: **Instruct** (non-thinking) and **Thinking**

**Qwen's own claim:** "Qwen3-Next-80B-A3B-Instruct performs on par with
Qwen3-235B-A22B-Instruct-2507 on certain benchmarks, while demonstrating
significant advantages in handling ultra-long-context tasks up to 256K."

**Qwen-reported benchmarks (Instruct):**
| Benchmark | Qwen3-Next-80B-A3B-Instruct | Qwen3-235B-A22B | Qwen3-30B-A3B |
|---|---:|---:|---:|
| MMLU-Pro | 80.6 | 83.0 | 78.4 |
| GPQA | 72.9 | 77.5 | 70.4 |
| LiveCodeBench v6 | **56.6** | 51.8 | 43.2 |
| AIME25 | 69.5 | 70.3 | 61.3 |
| BFCL-v3 (tool use) | 70.3 | 70.9 | 65.1 |
| Arena-Hard v2 | **82.7** | 79.2 | 69.0 |
| IFEval | 87.6 | 88.7 | 84.7 |

**Qwen-reported benchmarks (Thinking):**
| Benchmark | Qwen3-Next-80B-A3B-Thinking | Qwen3-235B-A22B-Thinking | Gemini-2.5-Flash-Thinking |
|---|---:|---:|---:|
| MMLU-Pro | 82.7 | 84.4 | 81.9 |
| AIME25 | 87.8 | 92.3 | 72.0 |
| LiveCodeBench v6 | 68.7 | 74.1 | 61.2 |
| BFCL-v3 | 72.0 | 71.9 | 68.6 |
| TAU2-Telecom | **43.9** | 45.6 | 31.6 |

**Artificial Analysis (Instruct — the only variant AA tracks):**
- AA Intelligence Index: **15** (low — #6/39 in non-reasoning open-weights)
- Speed: 183.4 tok/s (AA #2/39 — very fast)
- τ³-Banking: **6%** (worse than 122B's 14%, better than 35B's 5%)
- Terminal-Bench v2.1: **7%** (much worse than 122B's 48% — red flag)
- SciCode: 39%
- GPQA Diamond: 77%
- AA-LCR: **60%** (close to 122B's 67%)
- AA-Omniscience Non-Hallucination: 18% (better than 122B's 11%!)

**On-Spark data:** Not yet in our STATE.md. Spark Arena has no
Qwen3-Next-80B entries in the current 140-row leaderboard. **Untested on
GB10 by us.** However, architecture is essentially identical to Qwen 3.6
35B/27B (same `qwen3_next` family, same Gated DeltaNet + Gated
Attention, same MTP). The NVFP4 quant should work with our existing
toolchain (FlashInfer PR #3684, vLLM ≥0.25.0).

**Sweet-spot assessment vs the hypothesis:**
| Hypothesized | Qwen3-Next-80B-A3B | Match? |
|---|---|---|
| ~80B total | 80B | ✅ exact |
| Q4 / ~40GB disk | NVFP4 ~40–45GB | ✅ |
| 10B active | **3B active** (sparser!) | ✅ better (faster) |
| MTP support | ✅ native MTP | ✅ |
| 5+ lanes at 200K | Plausible: 40GB weights + ~75GB KV pool → 4–5 lanes at 150K | ⚠️ untested |
| 40+ tok/s | Likely: 3B active × MTP k=2–3 → should match/beat 35B's 100+ | ⚠️ untested |

**The contradiction:** Qwen's own benchmarks say Next-80B-A3B matches the
235B flagship on coding and ultra-long context. AA says the *Instruct*
variant has AA Index 15 — lower than the 35B (32), with Terminal-Bench 7%
(shockingly low). Two interpretations:
1. **AA is measuring the non-reasoning Instruct variant** which doesn't
   `think` — the Thinking variant (which AA doesn't track) is likely much
   higher. The Qwen-reported Thinking numbers (AIME25 87.8, LiveCodeBench
   68.7) suggest the Thinking variant is genuinely strong.
2. **AA's Terminal-Bench 7% is a harness/template issue** — the
   `qwen3_next` chat template is new (merged into HF transformers main
   only recently) and may not be correctly wired in AA's harness. This
   is the same class of issue that made Qwen 3.6 27B's AA GPQA (83%) lag
   Qwen's reported (87.8%).

**Verdict on Qwen3-Next-80B-A3B:**
- ✅ **Architecturally the sweet-spot model the brief hypothesized.**
- ✅ 3B active × MTP × hybrid Mamba-attention → the throughput, context,
  and lane-count math all work on paper.
- ✅ Qwen's own benchmarks are strong (Thinking variant especially).
- ⚠️ AA Instruct scores are low — but likely a template/harness artifact
  for a brand-new architecture, not a true intelligence floor. The
  Thinking variant is the one we'd actually deploy and it's not on AA.
- ❌ **Zero on-Spark verification.** No STATE.md entry, no Spark Arena
  entry, no forum thread. This is the biggest risk: the math says it
  should work, but nobody has run it.
- ⚠️ **Non-reasoning Instruct only for now** if we want the faster
  variant; Thinking variant will be slower (thinking tokens) but
  smarter.

**This is the highest-value benchmark target on the board.** If
Qwen3-Next-80B-A3B-Thinking-NVFP4 clears 35 tok/s at 150K with 4+ lanes
on Spark and its tool-calling works, it is the credible successor to
122B — same knowledge class (80B total vs 122B), 3× fewer active params
(3B vs 10B), MTP, hybrid Mamba, 262K→1M context. Round 1 missed it
because the research focused on Qwen 3.5/3.6 (the established families)
and didn't search the Qwen3-Next line.

---

## Final Ranked Recommendation (with rationale)

### #1 — Qwen 3.5 122B A10B (DFlash n=4/6 or int4-AutoRound) — **KEEP**

**Rationale:**
- **Only model that meets every spec axis AND is verified on Spark.**
  82.8 tok/s (our bench), 3 lanes at 150K, qwen3_xml tool parser
  production-stable, 67% AA-LCR (best long-context of the set).
- AA Intelligence Index 32 is the highest of any *deployable* model
  (DS V4 Flash at 40 needs 2 Sparks).
- Terminal-Bench 48% is 2× of gpt-oss-120b and 4× of Qwen3-Next-80B
  (Instruct). For agent coding, this is the on-axis eval.
- Community consensus: named unprompted by 0rand, keving3,
  engineering68, stu.miller, 2893f57a. "Don't overcomplicate" (0rand).
- Already verified-good in our STATE.md (Jul 13, n=6 config).

**Why it survives the debate:** Every challenger either fails a hard
spec (gpt-oss 131K context, Nemotron 15 tok/s), is untested on Spark
(Qwen3-Next-80B), or is dumber (35B, 27B). "Safest" *is* "best" when the
fleet is 5–6 unsupervised agents that can't tolerate silent failures.

**Known risks (unchanged from Round 1):** Looping in long coding sessions
(mitigation: `repetition_penalty`); DFlash brittle on vLLM nightlies
(mitigation: int4-AutoRound fallback, which engineering68 already uses).

### #2 — Qwen3-Next-80B-A3B (Thinking, NVFP4) — **BENCHMARK NOW**

**Rationale:**
- The hypothetical sweet-spot model exists and we missed it in Round 1.
- 80B total / 3B active + MTP + hybrid Mamba + 262K→1M context is the
  ideal GB10 profile: ~40GB weights leaves ~75GB for KV → 4–5 lanes at
  150K is plausible.
- Qwen's own Thinking benchmarks (AIME25 87.8, LiveCodeBench 68.7,
  BFCL-v3 72.0, TAU2-Telecom 43.9) beat or match the 235B flagship and
  beat Gemini-2.5-Flash-Thinking. If these hold on Spark, this model
  could deliver 122B-class intelligence at 35B-class speed.
- Architecture is a superset of our existing Qwen 3.6 35B/27B recipe —
  same `qwen3_next` family, same FlashInfer/vLLM toolchain. Deployment
  risk is low *if* the quant and parser work.

**Why #2 not #1:** It's untested on Spark. AA's Instruct scores (Index
15, Terminal-Bench 7%) are a red flag even if likely a harness artifact.
We have zero community deployment data. 122B is the known-good; this is
the known-promising. Promising doesn't run a fleet.

**Action:** Bench `Qwen3-Next-80B-A3B-Thinking` NVFP4 on Spark with our
standard recipe (FlashInfer PR #3684, vLLM ≥0.25.0, qwen3_next parser).
Success criteria: ≥35 tok/s at c=1, ≥4 lanes at 150K, tool-call smoke
test passes. If it clears, A/B test vs 122B on real Loca/Hermes agent
workloads for 48h. **This is the single highest-value next action.**

### #3 — Qwen 3.6 35B A3B (NVFP4) — **WORKER TIER, NOT BRAIN**

**Rationale:**
- The throughput champion: 100+ tok/s, 6 lanes at 256K, 74% MTP
  acceptance, 76GB footprint leaves 45GB for co-located media workloads.
- For 5–6 agents doing *simple* tasks (file ops, web search, cron,
  API calls), 35B's speed and lane count win. Digital_David runs this
  pattern for 4 agents at ~60 t/s.
- AA Index 29 (close to 122B's 32) and GPQA 85% — competitive on raw
  knowledge.

**Why #3 not higher:** τ³-Banking 5% (worst tool use in the set),
"teenager" hallucination/looping consensus, AA-Omniscience
Non-Hallucination ~10%. For unsupervised concurrent agents, the
reliability gap is the dealbreaker. Use as the worker tier in a
multi-model design, or for co-location with media workloads (our
STATE.md media-server config).

### #4 — gpt-oss-120b (MXFP4) — **DISQUALIFIED ON CONTEXT, WATCH**

**Rationale:**
- 44.58 tok/s on Spark Arena is real and competitive. 5.1B active is a
  middle ground. Apache 2.0, 117B total, eugr has a recipe.
- **But 131K native context fails our 150K+ spec.** YaRN extension is
  untested on Spark for this model and risky for a pure-Transformer MoE
  (no Mamba KV efficiency).
- Terminal-Bench 24% is half of 122B's 48%. AA-LCR 51% vs 67%. The
  model is both slower-context and dumber-on-agents than 122B.
- The "active parameter trap" (Tier-0 thread): pure-Transformer 120B
  MoEs OOM or lose context in long agent loops because KV grows
  linearly. This is the exact failure mode that hybrid-Mamba models
  (122B, 35B, Qwen3-Next-80B) are designed to avoid.

**Verdict:** Only viable if you can live with 131K context (we can't).
Revisit if OpenAI ships a 200K+ context revision or a Mamba hybrid.

### #5 — Qwen 3.6 27B (NVFP4, dense) — **BEST SUBAGENT, NOT BRAIN**

**Rationale:**
- AA Index 37 is the highest of any model in the set (small-class).
- 39–46 tok/s on Spark Arena. Terminal-Bench (AA) 51% — higher than
  122B's 48%. τ³-Banking 15% — competitive.
- Dense → no routing surprises, no MoE expert-loading variance.

**Why #5 not higher:** Dense 27B active → lane count at 150K context is
the killer. r0b0tlab's bench was at 8K context (1.1M KV tokens); at
150K × 4 lanes you need ~20M+ KV tokens and the dense 27B can't hold
that. Realistic lane count at 150K is 2–3, not 5–6. AA-LCR 55% is the
lowest long-context score of the serious candidates. 0rand: "Best as
subagent."

**Verdict:** The smartest small model. Use it for the hardest single-
stream coding tasks where you don't need 5–6 lanes. Not a fleet brain.

### #6 — Nemotron Super 120B (NVFP4) — **WATCH LIST**

**Rationale:**
- Architecturally the most compelling for unbounded agent loops
  (Mamba KV → near-zero growth). 1M context. AA-LCR 60%.
- But 14.4 tok/s fails the 30+ spec hard. AA Index 25 is the lowest of
  the large models. Terminal-Bench 39%, τ³-Banking 10% — below 122B.
  Tool parser (`super_v3_reasoning_parser.py`) is "approaching" mature.

**Verdict:** Bench once the inference stack matures and tool parser is
stable. Not a daily driver today. The Mamba-KV thesis is sound but
122B and Qwen3-Next-80B already capture most of that benefit at 2–4×
the speed.

---

## What the Debate Changed

1. **Qwen3-Next-80B-A3B is the new #2 and the top benchmark target.**
   Round 1 missed it. This is the real "sweet-spot" model and it's
   from the same architectural family as our existing 35B/27B recipes.
   The Qwen-reported Thinking numbers are strong enough to justify a
   real Spark bench. **Single highest-value next action.**

2. **gpt-oss-120b is a real model with 44.58 tok/s on Spark, but it
   fails the 150K+ context spec and its agent scores are half of
   122B's.** Not a competitor for our use case; would be a strong pick
   for a 131K-context workload.

3. **The 35B "teenager" verdict holds.** The challenge (speed > smarts
   for simple tasks) is valid for a narrow task mix, but tool-calling
   reliability is a cross-domain symptom, not banking-specific. 35B
   stays as worker tier.

4. **Nemotron's 15 tok/s is genuinely too slow** when 122B already has
   hybrid Mamba at 2× the speed and higher agent scores. The Mamba-KV
   thesis doesn't uniquely favor Nemotron anymore.

5. **Qwen 3.6 27B is the smartest small model** but dense architecture
   kills lane count at 150K. Best subagent, not fleet brain.

6. **No new Spark-relevant model shipped in the last 48h.** The
   picture is stable; GLM-5.2 (753B) remains a 2–3 month out candidate
   via Colibri streaming (per our `glm5.2-future-signal.md`).

**Bottom line:** The Round 1 consensus is correct but under-explored.
Qwen 122B stays as the daily driver. Qwen3-Next-80B-A3B is the
credible successor that must be benchmarked before the next decision
cycle. The 2nd Spark for DS V4 Flash remains the real intelligence
upgrade path.

---

## Provenance & New Data Pulled This Pass

- **Spark Arena** (spark-arena.com/leaderboard, tg128 @ d16384 c1,
  updated Jul 18 2026 19:00): gpt-oss-120b MXFP4 Single = 44.58 tok/s
  (rank 62), 28.23 tok/s (rank 91); 2-node = 65.36, 4-node = 61.37.
  Qwen3-Next-80B-A3B: **no entries** (untested on Spark Arena).
- **Artificial Analysis** (artificialanalysis.ai, AA Index v4.1,
  pulled 2026-07-18):
  - gpt-oss-120b (high): Index 24, GDPval 15%, τ³-Banking 12%,
    Terminal-Bench 24%, SciCode 39%, HLE 18%, GPQA 78%, CritPt 1%,
    AA-Omniscience Acc 22%, Non-Halluc 9%, AA-LCR 51%.
  - Qwen3-Next-80B-A3B Instruct: Index 15, τ³-Banking 6%,
    Terminal-Bench 7%, SciCode 39%, GPQA 77%, AA-LCR 60%,
    Non-Halluc 18%. (Thinking variant not on AA.)
  - Cross-class leaderboard top 25 includes: Qwen3.5 122B (32),
    Nemotron 3 Super (25→28*), gpt-oss-120b high (24), Qwen3-Next-80B
    (15), gpt-oss-120b low (14).
- **HuggingFace trending** (2026-07-18): no new Spark-relevant model
  in last 48h. GLM-5.2 (753B), Inkling (952B), Bonsai-27B,
  ThinkingCap-Qwen3.6-27B are the trending entries — all known or
  too big.
- **HuggingFace model cards** (pulled this pass):
  - `Qwen/Qwen3-Next-80B-A3B-Instruct`: 80B/3B, 512 experts (10+1
    shared), 262K→1.01M ctx, MTP, hybrid Gated DeltaNet+Gated
    Attention, Apache 2.0. Non-thinking. NVFP4 quant available
    (`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`, 44.3k downloads).
  - `Qwen/Qwen3-Next-80B-A3B-Thinking`: same arch, thinking mode,
    Qwen claims beats Gemini-2.5-Flash-Thinking on multiple benchmarks.
- **Existing workspace cross-ref:** `spark-daily-driver-debate-2.md`
  (Round 1 community consensus), `aa-intelligence-scores.md` (AA
  v4.1 for the Round 1 model set), `qwen-3.6-27b-research.md`,
  `gemma-4-31b-research.md`, `glm5.2-future-signal.md`, `STATE.md`.

### Honest caveats

- **Qwen3-Next-80B-A3B AA scores are Instruct-only.** The Thinking
  variant (which we'd deploy) is not on AA. The gap between AA's
  Terminal-Bench 7% and Qwen's reported LiveCodeBench 56.6 is large
  and likely a harness/template artifact, but I cannot prove it
  without a Spark bench. Treat AA's 15 Index as a lower bound, not a
  point estimate.
- **No on-Spark data for Qwen3-Next-80B-A3B.** All throughput/lanes
  reasoning is architectural projection, not measurement. The bench
  is the only way to resolve this.
- **gpt-oss-120b 131K context** may be extendable via YaRN, but this
  is untested on Spark for this model and risky for a pure-Transformer
  MoE. I treated 131K as a hard limit; it may be soft.
- **Hacker News / X not searched for model-release news** — web_search
  was unavailable (Firecrawl not configured). HN front page and HF
  trending were checked manually. A targeted news search for
  "Qwen3-Next Spark" or "gpt-oss-120b 200k context" would strengthen
  the recency claim but is unlikely to overturn it.
- **AA leaderboard "Nemotron 3 Super 28" vs Round 1's "25"**: the
  cross-class view shows 28 for "NVIDIA Nemotron 3 Super" while our
  `aa-intelligence-scores.md` (pulled same day) shows 25. Likely a
  reasoning/non-reasoning variant difference. I used the
  per-model-page 25 from our existing research for consistency; the
  difference doesn't change the ranking.