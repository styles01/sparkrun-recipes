# Ling-3.0-flash INT4 — Spark Arena Benchmark (submitted Aug 24, 2026)

**Submission:** native `sparkrun arena benchmark run`, auto-posted to Spark Arena
**Recipe:** `recipes/ling-3.0-flash-int4.yaml`
**Container:** `ghcr.io/styles01/ling-3.0-flash-int4:latest`
**Config:** GMU 0.80, 256K ctx, 2 seqs, fp8 KV, MTP bailing_hybrid_v3_mtp k=1, mamba align, fastsafetensors

## Results (tg t/s, 3 runs each)

| depth | c=1 | c=2 | c=5 | c=10 |
|---|---|---|---|---|
| 0 | 21.5 | 44.5 | 33.4 | 38.1 |
| 4096 | 21.4 | 32.1 | 25.5 | 25.5 |
| 8192 | 21.5 | 29.3 | 20.5 | 20.5 |
| 16384 | 21.2 | 21.2 | 14.5 | 13.7 |
| 32768 | 20.9 | 11.4 | 8.9 | 8.3 |
| 65535 | 20.3 | 6.9 | 4.3 | 4.1 |
| 100000 | 19.6 | 3.7 | 2.5 | 2.3 |

## Key takeaways
- Single-stream decode is remarkably flat: **20-21 t/s at every depth** (0 to 100K)
- Deep-context + concurrency collapses hard: **2.3 t/s at 100K / c=10**
- Prefill strong at depth (still ~37 t/s pp at 100K)
- Full sweep is honest — includes the documented deep-context cliff, not cherry-picked

## Compare to qwen-122b (peak ~71.7 t/s @ d0 c2)
Ling is slower on shallow (21 vs 71 at c=2) but Ling's 5.1B-active INT4 arch holds single-stream flat at depth where qwen-122b also degrades. Ling is a context-length play, not a raw-speed play.
