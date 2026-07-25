# SUCCESS CRITERIA — Spark LLM Recipes

**Agreed:** July 11, 2026 by James + Oracle
**These are the minimum bars. A recipe that doesn't meet these is NOT "working."**

## Universal Requirements (All Flavors)

0. **Spark safety first** — NEVER launch without the ADR-006 pre-flight checklist. Crashes can require physical power-cycle. Low GMU on first boot. Kill all other GPU processes. Clear caches.
1. **Agent-capable** — Hermes or Claude Code (CC) can run a full agent loop on it
2. **Tool calls** — `--enable-auto-tool-choice` + correct parser (non-negotiable, see ADR-002)
3. **Context per concurrent stream** — minimum 100K-256K tokens each
4. **Decode speed** — minimum 20 tok/s per stream

## Per-Flavor Expected Results

| Flavor | Concurrency | Min tok/s (per stream) | Min context/stream | Total context needed |
|---|---|---|---|---|
| **DS4** | 1 | 20+ | 256K (single user gets max) | 256K |
| **Qwen 122B** | 3 | 20+ | 100K-256K each | 300K-768K |
| **Qwen 35B** | 5+ | 80+ | 100K-256K each | 500K-1.28M+ |

## What "Usable" Means

| Rating | tok/s | Status |
|---|---|---|
| Unusable | <20 | ❌ Reject — agent loop too slow to be productive |
| Usable | 20-50 | ✅ Acceptable — works but not snappy |
| Good | 50-80 | ✅ Good — comfortable agent experience |
| Excellent | 80+ | ✅ Great — fast agent, multiple concurrent users |

## What "Working" Means (Gate to Lock a Recipe)

A recipe is only marked LOCKED when ALL of these pass:
1. ✅ Meets minimum tok/s for its flavor (see table above)
2. ✅ Meets minimum concurrency for its flavor
4. ✅ Each concurrent stream has ≥100K context available
4. ✅ Tool calls work (no 400 errors)
5. ✅ Loca completes a full agent loop (ADR-005)
6. ✅ No crashes during a 10-minute sustained test

## KV Cache Math (Verification)

Spark has 121GB unified memory. After model weights load, remaining memory × GPU mem util = KV cache pool. Each token's KV size depends on model architecture:

| Model | KV per token | 100K context/stream | 256K context/stream |
|---|---|---|---|
| DS4 (MLA) | ~656 bytes | ~65 MB | ~168 MB |
| Qwen 122B (GDN+mamba) | ~24 KB | ~2.3 GB | ~6.1 GB |
| Qwen 35B (standard MoE) | ~? | needs verification | needs verification |

At 3 concurrent × 256K on 122B: ~18.4 GB KV (pool is ~426K tokens ≈ ~10 GB at 24KB/token — **256K × 3 = 768K tokens exceeds the 426K pool!**)
→ 122B at 3 concurrent can do ~142K each (426K/3), or 2 concurrent @ 213K, or 3 @ 100K fits (300K < 426K).

At 5 concurrent × 256K on 35B: needs verification — 35B has much smaller model weights (~18GB), so most of 121GB is available for KV.

**This means the 122B recipe may need tuning: 3 concurrent at 100K each is realistic. 256K each only works at 1-2 concurrent.**

## Economic Decision Framework

| If 35B hits 5+ concurrent @ 80+ tok/s | → Replace Ollama $150→$20 |
|---|---|
| If 35B only hits 4 concurrent @ 60 tok/s | → Still worth it, but less headroom |
| If 122B can't sustain 3 @ 20+ tok/s | → Downgrade to 2 concurrent or increase context per stream |
| If DS4 can't hit 20 tok/s | → Re-evaluate --eager mode or reduce context to 128K |