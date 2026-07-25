# Spark Daily Driver Debate — Community Consensus (Report 2)

**Author:** Oracle (subagent research pass)
**Date:** 2026-07-18
**Task:** Search Twitter/X, GitHub (r0b0tlab, eugr), NVIDIA forums, and Artificial
Analysis for community consensus on best DGX Spark daily driver for agent
workloads. Focus on what people *actually run*, not theoretical benchmarks.

---

## TL;DR — Community Consensus

**For 5–6 concurrent Hermes/OpenClaw agents on a single DGX Spark, the community
converges on Qwen3.5-122B-A10B (INT4 AutoRound or DFlash) as the single-model
daily driver.** It is the model most experienced single-Spark operators name
unprompted when asked "what do you actually run."

The 35B class (Qwen3.6-35B-A3B) is the consensus *worker-tier* model — fast
enough for 6–8x concurrency, smart enough with tight instructions — but
operators who tried it as the sole brain report looping, hallucination, and
"teenager vs adult" intelligence gaps. The multi-tier (35B resident + 120B
on-demand) pattern is tried and *abandoned* by most who attempt it: spin-up
latency (≈10 min for 120B) and VRAM-defrag friction make the complexity "barely
warranted." People converge on one model.

Step 3.7 Flash and Nemotron Super 120B are respected for specific strengths
(agent scores, Mamba KV efficiency) but are **not** what the community runs as
a daily driver on day-18 of July 2026 — tool-calling stability, recipe maturity,
and in the case of Step 3.7 Flash, the 2-Spark TP requirement, keep them off
the single-Spark daily-driver throne.

DeepSeek-V4-Flash is the model people *graduate to* once they own two Sparks —
"near Sonnet quality at about the same speed I got out of a single Spark with
122B" (stu.miller, 3× Spark owner, NVIDIA forum). On a single Spark at Q2 GGUF
it "holds up" as a coder but the community consensus is that 122B-int4 is the
better single-box brain.

---

## Sources Actually Consulted

| Source | Status | What I got |
|---|---|---|
| **Twitter/X** (`mr_r0b0t`, "dgx spark daily driver", etc.) | ❌ **Blocked** | xAI/x_search out of credits; nitter + xcancel frontends down/blocked. Could not retrieve live tweets. Used mr_r0b0t's GitHub org as the authoritative proxy for their public recommendations. |
| **GitHub: r0b0tlab/hermes-concurrent-agents** (72★) | ✅ Full README + docs | HCA does *not* pick a model; it consumes any vLLM/SGLang endpoint. "No universal worker count is published. Measure each exact device." |
| **GitHub: r0b0tlab repos** (30 repos) | ✅ Full READMEs | qwen36-35b-a3b-nvfp4-gb10-native-mtp (13★), nvidia-qwen-3.6-27B-sm121-nvfp4 (6★), deepseek-v4-flash-nvfp4-gb10-benchmark (10★), minimax-m27-nvfp4 (13★), hy3-295b, laguna-xs2. r0b0tlab *builds reproducibility packs for many models* — they do not endorse a single daily driver; their HCA repo explicitly refuses to. |
| **GitHub: eugr/spark-vllm-docker** (1.8k★) | ✅ Full README + recipe list | 27 recipes. No single "default model." Solo recipes the README calls out in QUICK START: `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` (solo example), MiniMax-M2-AWQ (cluster), DS-V4-Flash (cluster-only, needs ≥2 Sparks). The recipe *catalog* is neutral; eugr himself posts in forum threads recommending against single-Spark 120B for long-context stability. |
| **NVIDIA Developer Forums** | ✅ 4 key threads, full text | "Single-Spark setups — which models do you actually run for coding" (38 posts, 3.7k views, 23 likes); "Single-Spark always-on agent team (35B resident + 120B on-demand)" (16 posts, 1k views); "Tier 0 Findings: Why Hybrid Mamba beats 120B for agents" (29 posts); "Introducing Tool Eval Bench CLI" (178 posts, 6.7k views). |
| **Artificial Analysis** | ✅ Intelligence Index v4.1 (existing pull, 2026-07-18) | Composite + 9 sub-evals for Step 3.7 Flash, DS V4 Flash, Qwen 122B A10B, Nemotron 3 Super 120B, Qwen 35B A3B. Live leaderboard URL 404'd but data already in `research/aa-intelligence-scores.md`. |

### Twitter gap — mitigation

I could not retrieve live tweets (xAI credits exhausted; all public X frontends
I tried are blocked/down). This is a real gap. The mitigation is that
**mr_r0b0t (@mr_r0b0t on X) publishes their actual recommendations as GitHub
reproducibility packs**, and those repos are the durable form of their Twitter
claims. Their HCA repo's stated position — "HCA does not provision models…
choose a context window from the model's supported limit and the measured
workload" — is the authoritative version of their public stance. Any specific
"mr_r0b0t recommends X" tweet claim should be treated as unverified without a
link; their code says "measure it yourself."

---

## What People Actually Run (NVIDIA Forum, primary evidence)

From the **"Single-Spark setups — which models do you actually run for coding"**
thread (forums.developer.nvidia.com/t/374423, Jun 24–Jul 9, 2026, 38 posts):

| User | Their single-Spark daily driver | Notes |
|---|---|---|
| **nangld85** (OP) | **Qwen3.6-35B-A3B PrismaQuant 4.75-bit (vLLM)** — "my daily driver; fastest and most capable small model I've found for the Spark" | Also runs DS-V4-Flash Q2 GGUF (llama.cpp), Holo-3.1-35B-A3B NVFP4 ("best agentic tool-user of its size"), Gemma-4-26B-A4B. Uses `repetition_penalty: 1.02` to break digit loops. |
| **0rand** (multi-Spark, replies from single-Spark experience) | **"only one model really stands up for a single spark: qwen 3.5 122b"** — "Not slow (26 t/s and steady pp up until 256k), enough knowledge, extremely good at tool calling." | Warns Q2/aggressive quants "struggle with multistep reasoning." Says vLLM beats llama.cpp "every day if you have a proper quant." Now runs multi-box: DS4F big brain on 2 Sparks, 122B "fast hands" on MacBook, 27B "meticulous auditor." |
| **keving3** | **albond/Qwen3.5-122B-A10B** in Agent Zero; DS-V4-Flash Q2 GGUF as "hard worker" | Calls 122B "extremely good for me." |
| **door.blu** | **Qwen3.6-35B-A3B-FP8 (vLLM)** for Hermes + auto tasks + simple coding; falls back to Claude Opus CLI on hard coding | Explicitly names Hermes. |
| **engineering68** | **Qwen3.5-122B-A10B-int4-autoround (vLLM, eugr image)** at home AND at work | Used DFlash until "one week ago" when it broke; reverted to int4-autoround. |
| **ambrosemcduffy** | **qwen3.6:35b-a3b-q4_K_M (Ollama)** + Nemotron 120B NVFP4 for harder tasks | Module-by-module human-in-loop workflow; hand-written spec.md + task.md. |
| **clint25** | **unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q8_K_XL** in llama.cpp via llama-swap, Hermes cron → Codex CLI | Tried "eugr vllm: 27b, 35b, 122b, qwen3-coder-next, gemma4, autoround, int4, nvfp4, prismascout, dflash, mtp, aeon." Settled on llama.cpp for stability: "vLLM is a pain to get working well; llama.cpp is solid and stable." Notes 3.6-27B "better code but 35B much faster without a huge drop-off." |
| **2893f57a…** | **Qwen 122B** currently; notes "prone to looping" for coding | Asks about Mistral Small 4 119B for non-coding. |
| **m0l0** | Custom fastest Qwen3.6-27B recipe, **40 t/s decode c=1, 90–100 t/s at c=4** | Links their recipe; says PrismaAURA worse than PrismaSCOUT (unverified by me). |

### The always-on multi-agent design thread (forums…/t/372476)

This is the *exact* use case — "always-on team of 8 agents… all routine
inference local" on a single Spark. The OP (Mert.T) proposed 35B resident +
120B on-demand. The thread's verdict, from people who built it:

- **0rand** (reply #3): "I used to run precisely this before I got a second
  unit… Works really well, but the startup script must launch 27b first, wait
  for serve, compress and defrag vram, launch cascade 2… **in reality most
  tasks still executed by the main agent, so this extra complexity is barely
  warranted and actually not beneficial.** Later I switched to one model, first
  Mistral 4 Small 119B once vLLM 0.21 released and then finally to just **qwen
  3.5 122b which was a best balanced model for me** until a stable recipe
  supporting 500k+ context with deepseek V4 flash landed. **I would say go for
  just qwen 3.5 122b Intel autoround int4 and don't overcomplicate.**"
- **stu.miller** (reply #4, **owns 3 Sparks**): "Spinning down a model and
  spinning up another takes too long (10 minutes or so)… I did this for a
  week or so, and then bought another spark, that tells you something. I was
  pretty happy with **Qwen 122b on a single spark as the 'brain' roles, and
  35b as the 8x concurrency developer.** But I did use 122b as part of hermes
  kanban to verify everything the 35b agents did before passing it."
- **stu.miller** (reply #9): "**122b autoround is my recommendation** — fast
  enough and very strong. But I'd also test 35b — insanely fast and I rely on
  it for all my worker roles and it barely slips even at 8x concurrency. Just
  needs tight instructions."
- **Digital_David** (reply #11): "2 agents OpenClaw with Hermes for backup…
  **Qwen3.6-35B as the main LLM for all 4.** Then an API to Gemini Pro as the
  'brain power' only when needed, ~5% of the time. Tried multiple LLMs on a
  single Spark, but this was just so much smoother and average about 60 t/s."
- **nvidiaspark** (reply #12): flags 35B looping issues — "Sometimes I feel
  like I'm the only one having looping issues with Qwen3.6 35B A3B." Fix
  offered: `repetition_penalty` / `presence_penalty` in override-generation-config,
  and the `froggeric/Qwen-Fixed-Chat-Templates` HF repo (stu.miller #16).

---

## Intelligence vs Speed vs Context vs Lanes — the debate, resolved by data

### Artificial Analysis Intelligence Index v4.1 (composite & sub-evals)

| Model | AA Index | Cloud t/s | Terminal-Bench (agentic coding) | τ³-Banking (tool use) | GPQA Diamond | AA-LCR (long ctx) |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash (max) | **40** | 103.7 | 62% | 23% | 89% | 63% |
| Qwen3.5 122B A10B | **32** | 134.9 | 48% | 14% | 86% | 67% |
| Step 3.7 Flash | **30** | 378.1 | 26% | 11% | 78% | 64% |
| Qwen3.5 35B A3B | **29** | 134.0 | 41% | 5% | 85% | 63% |
| NVIDIA Nemotron 3 Super 120B | **25** | 146.1 | 39% | 10% | 80% | 60% |
| Puzzle-75B A9B | n/a (not tracked) | — | — | — | — | — |

Source: `research/aa-intelligence-scores.md` (pulled 2026-07-18 from
artificialanalysis.ai model pages; live leaderboard URL currently 404s but
per-model pages are the source of record).

**Reading the table for agent workloads:**
- **Terminal-Bench (agentic coding) is the eval that matters most for our use
  case.** DS V4 Flash dominates at 62%; Qwen 122B (48%) > 35B (41%) > Nemotron
  (39%) > Step 3.7 Flash (26%).
- **τ³-Banking (tool use)**: DS V4 Flash 23% >> Qwen 122B 14% > Nemotron 10% >
  Step 3.7 11% > 35B 5%. The 35B's 5% here is the AA-side echo of what forum
  users report — tool-calling is the 35B's weak spot without tight prompting.
- **AA-LCR (long context)**: Qwen 122B (67%) is the best of the open-weights
  set, edging DS V4 Flash (63%) and 35B (63%). Relevant for our 150K+ per-lane
  requirement.
- **Step 3.7 Flash's agent scores are surprisingly weak** (Terminal-Bench 26%,
  τ³-Banking 11%) despite its reputation — AA says its strengths are GPQA (78%)
  and raw speed (378 t/s cloud), not agentic loops. Its AA Index of 30 is
  *below* Qwen 122B (32). The "best agent scores" claim in the task brief is
  **not supported by AA Intelligence Index v4.1** as of 2026-07-18.

### On-Spark measured throughput (from our STATE.md + forum reports)

| Model | Solo tok/s (c=1) | Concurrent lanes | Context/lane | Memory | Tool calls | Status |
|---|---:|---:|---:|---:|---|---|
| Qwen 3.5 122B DFlash n=6 | 82.8 (code, our bench) | **3** (150K each) | 150K | ~104GB | ✅ qwen3_xml | ✅ Running, verified Jul 13 |
| Qwen 3.6 35B NVFP4 MTP | 102.8 (our bench); 174.7 @ c=4 (r0b0tlab) | **6** (256K each) | 256K | ~76GB | ✅ qwen3_xml | ✅ Running; "hallucination worse than DS4" |
| DS V4 Flash | 21 (our bench); 32.7 16hr-avg reasoning OFF | 2 (128K) | 128K | cgroup 110G | ✅ deepseek_v4 parser | LKG Jul 11; needs 2 Sparks for good perf |
| Step 3.7 Flash | ~22-23 t/s IQ3_M (forum) | 2 | 200K | cluster-only recipe | untested | eugr recipe requires ≥2 Sparks |
| Nemotron Super 120B | 14.4 t/s Q4_K_XL llama.cpp (forum) | 1 (16K ctx in that test) | 16K tested | ~84GB | untested by us | eugr recipe exists; "approaching usability" Apr 2 |
| Qwen 3.6 27B NVFP4 | 19.15 c=1, 69.6 c=4, 144 c=16 (r0b0tlab) | dense, high | 8K tested | ~? | ✅ | AA Index not separately tracked; 35B preferred over 27B by most forum users for coding |

---

## Per-model community verdict

### Qwen 3.5 122B A10B (INT4 AutoRound or DFlash) — **CONSENSUS DAILY DRIVER**
- Most-named single model when experienced single-Spark operators are asked
  what they actually run (0rand, keving3, engineering68, 2893f57a, stu.miller).
- "Not slow (26 t/s and steady pp up until 256k), enough knowledge, extremely
  good at tool calling" (0rand).
- "122b autoround is my recommendation — fast enough and very strong" (stu.miller, 3× Spark owner).
- eugr maintains both `qwen3.5-122b-int4-autoround.yaml` (solo-capable) and
  `qwen3.5-122b-fp8.yaml` recipes; the int4-autoround is the single-Spark path.
- **Known issues:** prone to looping in long coding sessions (2893f57a,
  nangld85); DFlash recipe occasionally breaks with vLLM nightlies
  (engineering68 reverted to int4-autoround; nangld85 posted a PR #45207
  Mamba-pad patch to fix DFlash + hybrid-GDN asserts). Our STATE.md shows 122B
  DFlash n=6 is our current verified-good config at 82.8 t/s, 3 lanes, 150K.
- **Fits the 30+ t/s, 3+ lanes, 150K+ context, tool-calling, 121GB criteria.**
  This is the model that meets the spec on every axis *and* has community
  deployment behind it.

### Qwen 3.6 35B A3B (NVFP4 / FP8 / PrismaQuant) — **CONSENSUS WORKER TIER**
- The high-concurrency specialist. 6 lanes at 256K (our recipe), 8x
  concurrency reported solid with "tight instructions" (stu.miller). r0b0tlab's
  repro pack hit c=4 at 174.7 t/s aggregate.
- "fastest and most capable small model I've found for the Spark" (nangld85, OP
  of the 38-post thread — this is *his* daily driver, with the caveat he runs
  OpenCode unsupervised coding, not 5–6 Hermes agents).
- **The intelligence gap is real and repeatedly reported.** Our STATE.md:
  "Intelligence gap vs 122B noticeable — 'teenager vs adult.' Hallucination
  and gaslighting significantly worse than DS4. Not suitable as daily driver
  for agent workloads requiring reliability." Forum: AA τ³-Banking 5%
  (tool-use) is the worst of the set; looping issues (nvidiaspark); WilliamD
  reports Qwen3.6-35B-FP8 "fails when tasked to scrap a large web doc" under
  heavy context.
- **Verdict:** excellent as the *worker tier* in a multi-model design or for
  single-agent OpenCode-style coding with `repetition_penalty: 1.02`. **Not the
  daily driver for 5–6 concurrent Hermes agents needing reliability.**

### Step 3.7 Flash — **RESPECTED, NOT DEPLOYED ON SINGLE SPARK**
- eugr's `step-3.7-flash-nvfp4.yaml` / `step-3.7-flash-fp8.yaml` recipes
  require ≥2 Sparks ("Requires at least 2 Sparks in a cluster"). FP8 "requires
  more memory, so using NVFP4 is recommended." `--no-ray` is required for FP8
  to fit full context.
- Forum: nangld85 tested IQ3_M in llama.cpp at 22-23 t/s — "ended up entering
  complex loops which repetition_penalty doesn't break."
- AA Intelligence Index 30 — *below* Qwen 122B (32). The task brief's "best
  agent scores" claim is not supported by AA v4.1; Step 3.7's strengths are
  GPQA (78%) and raw cloud speed (378 t/s), not agentic Terminal-Bench (26%).
- **Verdict:** technically impressive (Mamba, 200K context, MTP) but
  single-Spark recipe doesn't exist in eugr's catalog, llama.cpp loop issues
  reported, and AA agent scores underperform Qwen 122B. **Not a single-Spark
  daily driver today.**

### Nemotron Super 120B (A12B) — **APPROACHING USABILITY, NOT THERE**
- Forum thread "Nemotron 3 Super: Updates Approaching Agentic Usability" (Apr
  2): 14.4 t/s, sub-second TTFT in llama.cpp Q4_K_XL — "approaching usability…
  predictions we should get 30+ tps when the inference stack gets cleaned up."
- The "Tier 0 Findings: Why Hybrid Mamba beats 120B for agents" thread (29
  posts) argues the Mamba architecture's near-zero KV-cache growth is *the*
  feature for long agent loops — "infinite agent loops without memory
  footprint creeping up." **This is the architectural argument in its favor.**
- But eugr himself posts in that thread correcting the OP's unified-memory
  model, and AA Intelligence Index is the lowest of the set at **25**
  (Terminal-Bench 39%, τ³-Banking 10%). Tool-calling parser: eugr's recipe
  uses a custom `super_v3_reasoning_parser.py` — maturity is "approaching," not
  "at."
- **Verdict:** architecturally compelling for unbounded agent loops (Mamba KV),
  but as of 2026-07-18 it is slower (14.4 t/s), less intelligent by AA Index
  (25), and its tool-calling recipe is newer than Qwen 122B's. **Watch list,
  not daily driver.** Worth a bench once the inference stack matures.

### Qwen 3.6 27B (dense, NVFP4) — **FAST SUBAGENT, NOT THE BRAIN**
- r0b0tlab's `nvidia-qwen-3.6-27B-sm121-nvfp4` repo: 19.15 t/s c=1, 144 t/s
  c=16, GSM8K 81.88% 0-shot, 88-93% MTP acceptance, 1,162,353 KV tokens (141.89x
  at 8K ctx).
- Forum (m0l0): a custom recipe hits 40 t/s c=1, 90-100 t/s at c=4 — "quite
  workable." clint25: "3.6-27b is strong and better code, but 3.6 35b is much
  faster without a huge drop-off."
- 0rand: "Qwen 3.6 27b is better at actually writing the code but lacks
  knowledge and must have extremely well defined information for execution.
  Best as subagent."
- **Verdict:** excellent subagent / auditor tier (fast, dense, high
  concurrency). Not a brain replacement for 122B. Our STATE.md AA Index for the
  27B isn't separately tracked; the 35B's AA Index 29 is the proxy.

### DeepSeek V4 Flash — **THE GRADUATION MODEL (needs 2 Sparks)**
- On a single Spark at Q2 GGUF: "strong coder; Q2 is the only quant that fits,
  but it holds up" (nangld85). Our STATE.md: 21 t/s code gen, 32.7 t/s 16hr-avg
  reasoning OFF — bimodal (40+ idle, 11-18 under agent load).
- On 2 Sparks (eugr recipe, cluster-only): stu.miller — "near sonnet quality
  at about the same speed I was getting out of a single spark with 122b." AA
  Intelligence Index **40** — the highest of the open-weights set, Terminal-Bench
  62% (best), τ³-Banking 23% (best).
- **Verdict:** the best open-weights model by AA Intelligence Index and the
  best agent-coder by Terminal-Bench — but on a *single* Spark it's a Q2
  compromise that "holds up" rather than excels, and its real stride is at 2
  Sparks. **Not the single-Spark daily driver; the reason to buy a 2nd Spark.**

---

## The architecture debate the forum surfaced

The "Tier 0 Findings" thread (forums…/t/359275, 29 posts) makes an argument
worth recording for the record:

> **The "Active Parameter" Trap:** on GB10 (~273 GB/s bandwidth), active
> params/token dictate performance more than total params. GPT-OSS 120B
> (Transformer MoE, ~5.1B active) "creates a massive KV cache for every word in
> conversation history" → OOMs or loses context in long agent loops. Nemotron
> 3 Nano (Hybrid Mamba-MoE, ~3.2B active) has "near-zero KV cache growth" →
> "infinite agent loops without memory footprint creeping up."

eugr's reply: "Spark has unified RAM, so anything you load in RAM will take
from available VRAM" — correcting the OP's claim about offloading orchestration
to the Grace CPU. The architectural point (Mamba KV efficiency) survives the
correction; the implementation detail (CPU offload) does not.

**Implication for our 5–6 concurrent agent use case:** hybrid-Mamba MoEs
(Nemotron family, Qwen 3.5/3.6 hybrid Mamba/attention, Step 3.7) have a real
architectural advantage for *many long-context lanes* because KV grows
slowly. This is part of why Qwen 122B (hybrid Mamba/attention) holds 3×150K in
121GB while a pure-Transformer 120B would not. The Mamba advantage is
already baked into our two leading candidates (Qwen 122B and Qwen 35B are
both hybrid Mamba/attention). Nemotron's *additional* Mamba-ness is a matter of
degree, not kind.

---

## Bottom line for our 5–6 concurrent Hermes/OpenClaw agents

1. **The community-consensus single-Spark daily driver is Qwen 3.5 122B A10B
   (INT4 AutoRound, or DFlash n=6 with the Mamba-pad patch).** It meets every
   criterion in the brief: 30+ t/s (we measure 82.8), 3+ lanes (we run 3 at
   150K), 150K+ context per lane, tool calling (qwen3_xml), fits 121GB
   (~104GB used). It is the model experienced operators name unprompted. Our
   STATE.md already has it verified-good as of Jul 13.

2. **The 35B is the worker tier, not the brain.** For 6-lane raw concurrency
   it wins (102.8 t/s, 6×256K), but AA τ³-Banking 5% (tool use) and repeated
   forum reports of looping/hallucination mean it is not the reliable daily
   driver for 5–6 concurrent *agent* lanes that need tool calling. The
   "teenager vs adult" gap is consensus, not just our finding.

3. **Step 3.7 Flash's "best agent scores" claim is not supported by AA v4.1.**
   Its AA Index (30) is below Qwen 122B (32); its Terminal-Bench (26%) and
   τ³-Banking (11%) are the weakest of the open-weights large MoEs. Its
   strengths are GPQA and raw speed. eugr's recipe requires 2 Sparks. Not a
   single-Spark daily driver candidate today.

4. **Nemotron Super 120B is the architectural dark horse** (Mamba KV →
   unbounded agent loops) but is currently 14.4 t/s, AA Index 25, and its
   tool-calling recipe is immature. Watch list.

5. **DS V4 Flash is the upgrade path when the 2nd Spark arrives.** Highest AA
   Index (40), best Terminal-Bench (62%), but on a single Spark it's a Q2
   compromise. The community pattern is: single Spark → 122B; two Sparks →
   DS V4 Flash.

6. **The multi-model (35B resident + 120B on-demand) pattern is tried and
   abandoned.** Spin-up latency (~10 min) and VRAM-defrag friction make the
   complexity "barely warranted" (0rand). Operators converge on one model.
   The viable variant is Digital_David's: 35B local for everything + cloud
   "brain power" ~5% of the time — but that breaks the "all local" goal.

**Recommendation for our use case:** stay on **Qwen 3.5 122B A10B (DFlash n=6
or int4-autoround)** as the single-Spark daily driver for 5–6 concurrent
Hermes/OpenClaw agents. It is the community consensus, it meets every
criterion in the brief, and it is already our verified-good config. Treat the
35B as the fallback/co-location option when media workloads need RAM headroom
(our STATE.md already documents this as the media-server config). Bench
Nemotron Super 120B when its tool-calling recipe matures. Plan the 2nd Spark
for DS V4 Flash as the real intelligence upgrade.

---

## Files & provenance

- **Primary new source data saved to** `/tmp/` (forum JSON, repo READMEs):
  `nvidia_thread.json` (forum t/374423), `nv_alwayson.json` (t/372476),
  `nv_tier0.json` (t/359275), `nv_nemotron.json` (t/365512),
  `nv_tooleval.json` (t/366903), `eugr_readme.md`, `eugr_recipes.json`,
  `hca_readme.md`, `hca_backends.md`, `hca_bench.md`, `qwen36_mtp.md`,
  `qwen36_27b.md`, `ds4_bench.md`. These are session-temp; the synthesized
  findings live in this file.
- **Cross-referenced existing research:** `aa-intelligence-scores.md`
  (AA v4.1, pulled 2026-07-18), `step-3.7-flash-*.md`, `qwen-3.6-27b-research.md`,
  `puzzle-*.md`, `intelligence-benchmarks.md`.
- **State cross-ref:** `../STATE.md` (122B n=6 verified Jul 13; 35B media-server
  config; DS4 LKG Jul 11).

### Gaps & honest caveats

- **Twitter/X was not retrieved** (xAI credits exhausted; nitter/xcancel
  blocked/down). mr_r0b0t's GitHub org is the durable proxy for their public
  claims. Any specific tweet-level "mr_r0b0t recommends X" assertion should be
  treated as unverified without a link — their code says "measure it yourself."
- **Artificial Analysis live leaderboard 404'd** at fetch time
  (`/text/arena/leaderboard`), but the per-model Intelligence Index v4.1 data
  was already in `research/aa-intelligence-scores.md` from the same day. I did
  not re-pull; I cited the existing pull.
- **"Best agent scores" for Step 3.7 Flash (in the task brief) is contradicted
  by AA v4.1** — Step 3.7's agent evals (Terminal-Bench 26%, τ³-Banking 11%)
  are the weakest of the open-weights large MoEs. Its reputation likely comes
  from vendor/blog claims of agentic strength, not AA's composite. Flagging
  this discrepancy rather than silently endorsing the brief's framing.
- **Puzzle-75B is not tracked on Artificial Analysis** — no model page, not on
  any leaderboard I could find. HuggingFace-only. Excluded from the AA
  comparison; included in the model list only by reference.
- **Forum user counts are small** (the 38-post thread has 13 unique users).
  This is the most active single-Spark-model discussion I could find, but it
  is not a poll. Consensus is directional, not statistical.