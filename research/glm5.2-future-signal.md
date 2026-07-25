# Forward Signal: GLM-5.2 Local on Spark

**Date:** July 12, 2026
**Author:** Oracle
**Confidence:** Medium-High
**Timeline:** 2-3 months (September-October 2026)

## The Signal

Colibri (JustVugg/colibri) proves that 744B MoE models can run on consumer hardware by streaming experts from disk. Issue #76 confirms it works on the exact DGX Spark GB10 hardware we have. Currently 1-3 tok/s — too slow for agent workloads, but the architecture is sound and the project is evolving fast (78 commits, 6.9K stars, active development).

## The Pattern

This mirrors the DS4 deployment trajectory:
- DS4 barely fits in 121GB (98GB used, 23GB free)
- GLM-5.2 doesn't fit (370GB int4 on disk, 10GB dense resident)
- Both are MoE — only ~40B params active per token
- The difference is just how aggressively you tier between RAM and disk

## Memory Tiering Architecture (The Real Insight)

```
Memory Budget = 121GB
- Dense part (attention, shared experts, embeddings): ~10GB (always resident)
- KV cache pool: ~6GB (always resident)
- OS/system overhead: ~5GB
- Expert cache: ~100GB available for pinning
```

### Optimal Expert Pinning Strategy

| Tier | % of Experts | Access Pattern | Storage |
|---|---|---|---|
| Dense / shared | ~5% | Every token, every layer | RAM (always) |
| Hot experts | ~20% | ~80% of token routing (Pareto) | Pinned in RAM |
| Warm experts | ~30% | Domain-specific, occasional | OS page cache (NVMe) |
| Cold experts | ~50% | Rare routing edge cases | Disk only, LRU evicted |

### Workload-Dependent Hot Experts

Different workloads route to different experts — this is why 122B DFlash collapses on prose (15 tok/s) but flies on code (82 tok/s). The hot experts for code are cold for prose.

| Workload | Hot Expert Profile |
|---|---|
| Code generation | Syntax, logic, structure experts |
| Reasoning | Chain-of-thought, math experts |
| Prose / chat | Language, tone, narrative experts |
| Tool calling | Structured output, function experts |

### The Spark's Unified Memory Advantage

On discrete GPU setups, experts must be copied CPU→GPU over PCIe (~64GB/s). On the Spark, CPU and GPU share 121GB — zero-copy. Experts in RAM are already "in VRAM." This is why the maintainer called the Spark "dream hardware" for this architecture.

## Convergence: Three Things That Make This Real in 2-3 Months

1. **Engine maturity** — Colibri at 78 commits; vLLM likely to add expert offloading for MoE models. The streaming approach gets absorbed into mainstream engines (same as llama.cpp innovations did).

2. **NVMe bandwidth** — Gen5 at ~14GB/s = ~737 experts/sec theoretical. At ~8 experts/token across 75 layers = ~152MB/token cold. Warm cache + router-lookahead prefetch (Colibri already at 71.6% predictability) eliminates the disk bottleneck.

3. **Unified memory** — The Spark's 121GB shared pool gives zero-copy access to pinned experts. No PCIe bottleneck. Unique hardware advantage vs every discrete-GPU deployment.

## Two Paths to Local GLM-5.2

| Path | How | Timeline |
|---|---|---|
| **vLLM adds expert offloading** | `--expert-cache-size 100GB` flag, same engine we already use | Likely — vLLM roadmap, MoE is the dominant architecture now |
| **Colibri matures** | Adds CUDA batching, multi-sequence, better caching | Possible — fast-moving project, active maintainer |

Either way: GLM-5.2 becomes a fourth flavor on the Spark. Same model selector, same `larryspark` provider, different model ID.

## What to Watch

- [ ] Colibri issues for multi-sequence / batching support
- [ ] vLLM PRs mentioning expert offloading or disk streaming
- [ ] Cache hit rate improvements (current: 70-85% estimated, EPYC 430GB hit 98%)
- [ ] NVMe Gen5 adoption in Spark successor hardware
- [ ] Router predictability improvements (current: 71.6%)

## Revisit Date

September 2026 — check if either path has materialized. If vLLM adds expert offloading, deploy immediately. If Colibri adds multi-sequence + hits 5+ tok/s with good caching, evaluate as a serious option.

## Related Files

- `research/colibri-glm5.2-analysis.md` — Full technical analysis (19KB)
- `STATE.md` — Current Spark state with all 3 flavors
- `benchmarks/` — Benchmark telemetry for DS4, 122B, 35B