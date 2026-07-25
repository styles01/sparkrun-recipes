# 122B Memory Audit — TODO

**Created:** July 13, 2026
**Status:** Not yet tested — need to boot 122B and capture startup logs

## What We Don't Know

We never checked 122B's actual vLLM startup memory breakdown. We took community numbers:
- 64GB weights
- 426K KV tokens
- 3 lanes
- GMU 0.82

But 35B taught us CUDA graphs are a massive hidden cost (21GB on a 22GB model). DFlash n=12 = 13 tokens per forward — if graphs scale linearly with tokens/forward, 122B's graph cost could be 3x what 35B's is.

## What to Capture on Next Boot

```bash
ssh jaita@larryspark.local 'docker logs qwen-spark 2>&1 | grep -E "Model loading took|Available KV cache|Estimated CUDA graph|GPU KV cache size|Maximum concurrency"'
```

## Questions to Answer

1. How much do DFlash n=12 CUDA graphs actually cost? (Could be 40-68GB)
2. Would DFlash n=4 or n=6 dramatically reduce graph memory?
3. Could we get more lanes / more KV / co-location room with lower n?
4. Is the prose collapse (9% acceptance at n=12) actually WORSE than just running n=4 with higher acceptance?
5. What's the real fixed cost floor (weights + graphs + overhead)?

## Experiment Configs to Test

| Variant | DFlash n | GMU | Expected Graphs | Expected Free for KV | Lanes |
|---|---|---|---|---|---|
| Current (community) | 12 | 0.82 | ??? | ??? | 3 |
| Conservative | 6 | 0.82 | ~half of n=12? | more | 3-4? |
| Aggressive | 4 | 0.75 | even less | even more | 4-5? |
| No DFlash | 0 (eager) | 0.82 | 0GB | max KV | 4+ but slow |

## The Real Question

If 122B's fixed cost (weights + graphs) is 64+50 = 114GB, then there's almost no room for KV or co-location. But if DFlash n=4 cuts graphs to 20GB, fixed cost = 84GB, leaving 37GB for KV — enough for 3 lanes at 100K each, with 20GB spare for co-location.

## Action

Next time we switch to 122B, capture the startup logs BEFORE doing anything else. Then try n=4 and compare.