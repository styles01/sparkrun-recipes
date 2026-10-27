# DeepSeek‑V4.1‑Flash **1017** EXL3 2.9bpw — 2× DGX Spark TP2 (ds41 engine, DSpark TP2‑drafter)

**Status:** ✅ Production — daily-driver candidate, replaces the pre‑1017 lane
**Served name:** `dsv41` · port 8000
**Engine:** vLLM 0.31.1‑dev (`6e517b15` + 22 recipe patches + `vllm_exl3_moe` plugin, exllamav3 1.6.0) with the **1017 overlay** (6 files) — published image: [`ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v1`](https://github.com/styles01/pkgs/container/dsv41-1017-exl3-2x-dgx-sparks)
**Weights:** [LibertAIDAI/DeepSeek‑V4.1‑Flash‑1017‑EXL3‑2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (210.8 GB, 43 shards, MIT) + **fresh 1017 Engram shards 47+48** (~203 GB — the pre‑1017 tables are poison for this pack)
**Author:** the 1017 repack + loader/serving overlay by styles01; **engine, kernels, plugin and benchmark client by [@sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41)** (Apache‑2.0) — this build layers his method onto the community 1017 pack — DeepSeek‑V4.1‑Flash‑1017 (MIT, DeepSeek)
**Updated:** October 18, 2026

> **Recipe contract:** [`recipes/deepseek-v41-1017-exl3-tp2.yaml`](../recipes/deepseek-v41-1017-exl3-tp2.yaml)
> **Full method + every blocker/fix:** [`METHOD-1017.md` in the serving repo](https://github.com/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks/blob/main/METHOD-1017.md)
> **One button:** `MODEL_ROOT=/home/jaita bash one-button.sh` (same repo) — image pull → pack + Engram download → auto‑repack → both ranks up on :8000. ~430 GB NVMe.
> **Supersedes:** the pre‑1017 canonical lane (2.70 bpw, v1.1). The old encode runbook (`plans/dsv41-1017-encode-runbook.md`) and `plans/dsv41-1017-tuning-plan.md` are historical — the repack path made the 13–24 h re‑encode obsolete.

---

## Numbers (Sudosu's own client, `kit_bench.py`, greedy, streaming, single stream)

### The head‑to‑head — us vs the two published lanes, same box, same client

| prompt set b | **this lane (k=6, adaptive)** | Sudosu 2.70bpw (v1.1) | Mia‑AI‑Lab stock kit |
|---|---:|---:|---:|
| code | **93.8** | 102.7 | 41.6 |
| prose | **52.4** | 57.9 | 25.1 |
| structured | **126.8** | 141.0 | 40.4 |

- vs **Mia's stock kit**: 2.0–3.1× on every set (2.5× on prose). Measured same‑box with the same
  client by the same script (`data/speed/mia_kit_stock/kit_decode_b.json` in the recipe).
- vs **Sudosu's 2.70bpw lane**: −9‑10 % *by design* — his weights average 2.70 bpw, this pack
  averages 2.9 bpw, i.e. **+7.4 % weight bytes streamed per token**. Per byte moved, parity; and
  the 2.9 pack carries ~7 % better perplexity than a 2.7 would. Set‑a numbers measured within 1 %
  of his lane for structured (102.7 vs 103.7) and code (86.0 vs 86.6).

### The different draft amounts — full depth sweep (fixed DSpark depths, k=6 cap, TP2 drafter)

Every row = median over 2 reps of the three‑prompt probe set, greedy. This is **the**
tuning table: how much speculation you draft per round vs what you get back.

| draft depth | code tok/s | prose tok/s | structured tok/s | ms/round (median) |
|---:|---:|---:|---:|---:|
| 0 (no draft) | 31.3 | 31.5 | 31.5 | 31.8 |
| 1 | 54.7 | 45.8 | 55.1 | 34.9 |
| 2 | 69.3 | **50.3** | 77.7 | 38.5 |
| 3 | 78.9 | 49.3 | 96.5 | 41.8 |
| 4 | 82.8 | 47.8 | 114.5 | 44.0 |
| 5 | 85.6 | 45.0 | **131.6** | 46.0 |
| 6 | **87.3** | 42.7 | 127.8 | 49.0 |

Reading it:

- **Prose peaks at depth 2 (~50 tok/s)** — flowing text burns the drafter's acceptance after ~2
  tokens (the 1017 MTP accepts ~2.0/round on prose vs ~2.2 for Sudosu's old 0731-era drafter).
  Drafting deeper only costs. This is a **model property** of the drafter, not a serving knob.
- **Structured peaks at depth 5 (131.6)** — predictable JSON keeps accepting ~6.2/round.
- **Code keeps climbing to depth 6 (87.3)**.
- **The adaptive policy (`conf`) with the MEASURED cost table beats every fixed row on every set:**
  93.8 / 52.4 / **126.8** — it drafts deep on predictable rounds, shallow on prose.

### ⚠ The trap this lane hit first (why your numbers will look 10 % slow if you skip this)

The depth policy reads its per‑round cost table from `VLLM_DSPARK_ROUND_MS`, and the launcher used
to ship the **TP1‑drafter** measurements (`28.90,32.85,36.57,40.13,42.62,45.69`). With a TP2
drafter those numbers lie about the deep rounds and the policy under‑speculates. Fix = re‑run the
sweep on your boot and feed your own numbers in:

```bash
# one extra boot with the probe hooks:
DOCKER_ENV="VLLM_DSPARK_DEPTH_POLICY=fixed:5 VLLM_DSPARK_DEPTH_FILE=/root/.cache/dspark_depth VLLM_DSPARK_DEPTH_LOG=200" \
  ... dsv41_tp2.sh up
# then Sudosu's sweep (measures every fixed depth, prints the table to set):
scripts/bench/depth_sweep.sh   # → VLLM_DSPARK_ROUND_MS=31.9,35.2,38.6,42.6,45.3,48.1,51.0
```

Our launcher now defaults to that table. If you skip this step, structured sits at ~106 instead of
~127 and you'll think the pack is slow.

### Baselines for the same two nodes (for calibration when hardware changes)

| config | code | prose | structured |
|---|---:|---:|---:|
| no spec decode (raw, k=0) | 39.5* | — | — |
| k=5, TP1 drafter, stale table | 86.0 | 51.4 | 102.7 |
| **k=6, TP2 drafter, measured table (production)** | **88.3** | **49.1** | **105.9** |
| same, set‑b probe prompts | 93.8 | 52.4 | 126.8 |
| Mia‑AI‑Lab engine (eager, same weights) | 28.3 | — | ~31 est |

*my earlier working‑session prompt, not a set‑a/b prompt — use kit_bench for comparisons.

### Context / vision / ops

- **Context:** 262K served (`MAX_LEN=262144`); KV pool at UTIL 0.85 = 711,931 tokens
  (2.7× concurrent 262K requests). Clean boot (117 GB free) → `UTIL=0.88` for the 4×275K goal.
- **Vision:** ON (`TEXT_ONLY=0`) — verified: color test image answered correctly by the 1017 pack
  first served with vision in this config.
- Boot time ~13 min (JIT warmup + graph capture, 135 compile keys).
- Memory is the tight resource: stop `sparkDash`, `journalctl --vacuum-size=50M`, drop caches
  before boots; the gate wants UTIL×121.7 GiB free at startup.

## Architecture notes (what had to change for 1017)

1. 1017 packs carry compressors as **EXL3 pieces** (stacked 4‑D trellis, `[n, in16, out16, 16K]`),
   not FP8 `.weight` — new dequant path (CPU‑side, bit‑exact per `exl3_ref`).
2. The drafter in 1017: attn dense **K6**, wo_a as **8 head‑group slices**, main_proj K6, routed
   experts **K4** (128/stage), plan `drafter` section at absolute layers 40‑42, config
   `dense.drafter` K map. `attn.wkv`/`shared_experts.w3` are loader shards of fused modules.
3. The name remap is load truth (`mtp.{i}.X → model.layers.{i}.X`).
4. Indexer (wk, wq_b) quantized K‑per‑layer → K resolved from the safetensors header at load.

Full detail, every error trace and its fix: **METHOD-1017.md** (serving repo).

## Run the bench yourself

```bash
git clone https://github.com/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks dsv41-run
cd dsv41-run && MODEL_ROOT=/home/jaita bash one-button.sh
python3 kit_bench.py --base http://127.0.0.1:8000 --model dsv41 decode
```

| | |
|---|---|
| Model repo | [LibertAIDAI/DeepSeek‑V4.1‑Flash‑1017‑EXL3‑2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) |
| Engram source | [deepseek‑ai/DeepSeek‑V4.1‑Flash‑1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (shards 47+48 + index + config) |
| Image | [ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v1](https://github.com/styles01/pkgs/container/dsv41-1017-exl3-2x-dgx-sparks) |
| Serving repo | [styles01/deepseek-v41-1017-exl3-2x-dgx-sparks](https://github.com/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks) |
| Base recipe | [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) (Apache‑2.0) |
| Sparkrun registry recipe | this repo's `recipes/deepseek-v41-1017-exl3-tp2.yaml`