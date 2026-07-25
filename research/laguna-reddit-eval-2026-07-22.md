# Laguna S 2.1 — Community Eval Findings & Action Items

**Source:** r/LocalLLaMA post by u/tsekdotph — private agentic eval, 160 tasks × k=3, deterministic grading
**URL:** https://www.reddit.com/r/LocalLLaMA/comments/1v2ua8g/i_ran_lagunas21_through_my_private_agentic_eval/
**Date:** July 22, 2026
**Retrieved:** via Spark SSH (Reddit blocked direct access)

## Eval Setup

- **Hardware:** Single RTX Pro 6000 Blackwell 96GB
- **Stack:** vLLM 0.25.1 (native Laguna support, no patches)
- **Model:** Official NVFP4, 262144 ctx, fp8 KV, GMU 0.90
- **Parsers:** poolside_v1 (tool calls + reasoning)
- **Sampling:** Shipped generation_config defaults (temp 1.0 / top_p 1.0 / top_k 20) — NOTE: these are WRONG, see action item #3
- **Thinking:** ON (default)
- **Comparison:** Qwen 3.5-122B (same harness, same config)

## Where Laguna Wins

| Category | Laguna S 2.1 | Qwen 122B | Notes |
|---|---|---|---|
| Tool-call arg selection | 0.89 | 0.86 | Best in field of 5 models |
| Tool chain depth | 6 levels | 4 levels | Deepest locally tested |
| Plain tok/s (single stream) | 109 | 103 | Fastest 100B+ on card |
| JSON/streaming errors | 0 | — | Zero across all probes |
| Tool validation recovery | First retry | — | Recovers from errors immediately |
| Refusal/security probes | 12/12 | — | Refused all malicious payloads (tied for only perfect score) |

## Where Laguna Loses

| Category | Laguna S 2.1 | Qwen 122B | Notes |
|---|---|---|---|
| Sports knowledge + odds math | 0.80 | 1.00 | Knowledge boundary real — 118B is coding specialization, not breadth (per Poolside blog: reuses 33B pretraining corpus) |
| **Grounding under pressure** | **0.80** | **0.97** | **DISQUALIFIER for autonomous agents**: 3 hand-confirmed fabrications — invented P&L figure, named horses not in data (pulled "Genuine Risk" from pretraining), inflated position 1000× |

### Grounding failure details

- Invented a P&L figure for a market with zero data
- Twice drafted status updates naming horses not in any data
- Named "Genuine Risk" (real 1980 Derby winner) from pretraining, position size inflated 1000×
- Qwen 122B: zero inventions across ~240 grounding runs — just says "I don't have that"

### Thinking ON vs OFF

| Mode | Fabrications | Odds math | Notes |
|---|---|---|---|
| Thinking ON | 1 confirmed | 0.80 | Thinking partially suppresses fabrications |
| Thinking OFF | 11 confirmed | 1.00 | Math perfect but fabrications explode — includes new failure class: overrode grounded data ("listed as Crochet in the feed, which looks like an error" then substituted from memory) |

**Conclusion:** Grounding failure is in the model, not the reasoning mode. Thinking is the only thing keeping facts honest, but it overthinks itself out of right answers on math. Pick your poison.

## Critical Gotchas

### 1. max_tokens must be 8K+
Laguna thinks LONG. At max_tokens 2048, it burns the entire budget thinking and returns empty output. OP's first pass scored 0.40 on schema tasks because of this. Real score at 8K is 0.79. Vendor allows 32K.

### 2. DFlash is situational
- 109 → 271 tok/s on code prompts (2.5× speedup)
- Barely moves on prose (~117 tok/s, 1.07×)
- **Net LOSS under 4 concurrent streams** (268 → 198 aggregate, rejected drafts eat the batch)
- Not bitwise-stable vs spec-off at temp 0
- Fine for single-user coding, keep it OFF for concurrent serving
- Multiple commenters report DFlash collapsing to 8 tok/s (llama.cpp, not vLLM)

### 3. Wrong sampling defaults in generation_config.json
- Shipped generation_config.json: temp 1.0, top_p 1.0, top_k 20
- Poolside's model card recommends: temp 0.7, top_p 0.95
- Multiple commenters confirmed wrong defaults cause bad outputs (black screens, shitty code)
- OP defended using shipped defaults (evals every model on shipped defaults) but acknowledged the discrepancy

### 4. Thinking loops
Multiple users report Laguna getting stuck in thinking loops:
- "Okay, I'm really going to start writing the code now. But wait, maybe I..." — repeated over and over
- Gets stuck in planning, never reaches implementation
- One user (block 53) ran it in autonomous dev loop (Issue→Plan→Code→PR→Review→Merge): never completed a single meaningful iteration. Exhausts output budget during planning or spends entire context thinking.

### 5. Tokenizer fix pushed post-release
Poolside pushed a tokenizer fix ~5h after release (`</assistant>` token marked special "to match internal serving"). If model was downloaded before this, may have broken tokenizer causing loops. Worth checking our revision.

### 6. Unsloth quants may behave differently
Multiple commenters report Unsloth Q5_K_XL and Q6_K GGUF variants behave differently — reasoning output dropped drastically, possibly chat template differences. Not our concern (we use NVFP4 + vLLM) but worth noting if anyone tries GGUF.

## Community Sentiment

- Tool calling universally praised — "blowing me away", "intelligently uses them whenever appropriate"
- Thinking loops are the most common complaint
- Grounding/fabrication concern is real but debated — acceptable for human-review-loop coding, disqualifier for autonomous agents touching real data
- One user called Poolside's benchmark claims "the most egregious lies" (block 44) — but their test was barely functional, likely config issue
- Another user's autonomous dev loop test (block 53) is the most concerning signal: strong component skills that don't compose into working autonomous system

## Action Items for Our Spark Config

### Immediate (low risk)

1. **Set max_tokens ≥ 8K in agent configs** — prevent empty-output failures from thinking budget exhaustion
2. **Use vendor sampling: temp 0.7, top_p 0.95** — add `--override-generation-config '{"temperature":0.7,"top_p":0.95}'` to serve command (Mia's recipe already does this)
3. **Verify tokenizer revision** — check if our downloaded model has the post-fix tokenizer (`</assistant>` marked special)
4. **Add AGENTS.md instruction** — "Don't overthink — start implementing after planning" to mitigate thinking loops

### Test before committing (medium risk)

5. **DFlash toggle** — test k=7 ON vs OFF at 3 concurrent streams. Reddit data says net loss at 4 streams, but 3 is the gray zone. If our real traffic is single-user coding, keep k=7. If Loca serves multiple concurrent agents, consider OFF or lower k.
6. **Thinking mode per workload** — thinking ON for coding (suppresses fabrications), thinking OFF for math/odds (perfect scores). May want per-request `enable_thinking` toggle.

### Monitor (ongoing)

7. **Watch for thinking loops** in production — if agents hang, check for repeated reasoning patterns
8. **Grounding discipline** — keep Laguna in human-review-loop lane, NOT autonomous agents touching real data
9. **Community fixes** — DFlash fix mentioned in HuggingFace discussions, Poolside may push updates. Monitor for vLLM patches.

## Our Current Config vs. Recommendations

| Setting | Our current | Recommended | Status |
|---|---|---|---|
| max_tokens | not set (agent default) | 8K+ | ⚠️ NEEDS FIX |
| temperature | not set (model default 1.0) | 0.7 | ⚠️ NEEDS FIX |
| top_p | not set (model default 1.0) | 0.95 | ⚠️ NEEDS FIX |
| DFlash k | 7 | 7 for single-user, OFF for concurrent | ✅ OK for now, test concurrent |
| Thinking | ON (default) | ON for coding, OFF for math | ✅ OK, consider per-request toggle |
| Tokenizer | unknown revision | post-fix (check `</assistant>` special) | ⚠️ NEEDS CHECK |
| Context | 300K | 300K | ✅ OK |
| Lanes | 3 | 3 | ✅ OK |
| GMU | 0.85 | 0.85 | ✅ OK |