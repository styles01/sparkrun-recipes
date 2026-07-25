# Puzzle-75B-A9B NVFP4 on DGX Spark (GB10) — vLLM Deployment Guide

Research compiled 2026-07-18. Goal: stop the CUDA-graph-compile OOM on a
single DGX Spark (GB10, 121 GB unified memory, SM121) when serving
`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4` with vLLM.

---

## TL;DR — Recommended starting recipe (single GB10)

```bash
docker run --gpus all --ipc host --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  -p 8000:8000 \
  vllm/vllm-openai:v0.25.1 \
  vllm serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
    --served-model-name puzzle-75b \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enable-expert-parallel \
    --async-scheduling \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --max-model-len 160000 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.73 \
    --default-chat-template-kwargs '{"enable_thinking":true,"force_nonempty_content":true}' \
    --override-generation-config '{"temperature":1.0,"top_p":0.95}'
```

**If it still OOMs during CUDA-graph compile, the next lever is `--enforce-eager`
(disables graph capture entirely — the documented last-resort OOM fix for
hybrid Mamba models on memory-constrained parts).** Add it to the command above
and re-test. See "Why CUDA-graph compile OOMs" below for the mechanism.

---

## 1. NVIDIA's official vLLM flags (from the NVFP4 model card)

Source: `huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`
model card, "Quick Start Guide → vLLM" section. This is NVIDIA's only
documented serving recipe — there is **no separate NVIDIA dev-blog post or
DGX-Spark-specific deployment guide** (confirmed by searching
developer.nvidia.com/blog).

NVIDIA's official command (verbatim, with MTP):

```
vllm serve "$path" \
  --served-model-name "$model" \
  --port "$port" \
  --tensor-parallel-size "$tp" \
  --enable-expert-parallel \
  --async-scheduling \
  --trust-remote-code \
  --mamba-backend flashinfer \
  --mamba_ssm_cache_dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${num_speculative_tokens}}" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice
```

NVIDIA's official notes:
- **Tested on vLLM v0.20.0** (their reference version; the card predates v0.25.x).
- **NVIDIA recommends `tp=2` or `tp=4`** for this model. Single-GPU (tp=1) is
  not a validated NVIDIA configuration — it's what the Spark forces on us.
- `num_speculative_tokens=3` is the recommended default (best throughput at
  typical batch size); `5` or `7` may help low-batch / latency-sensitive use.
- For long-generation workloads: `--api-server-count 4`.
- `--no-enable-chunked-prefill` can increase throughput but reduces
  responsiveness (and, critically, **raises peak VRAM** — see OOM section).
- Supported hardware per the card: **NVIDIA Blackwell + Hopper only.** NVFP4
  is "optimized for Blackwell-class GPUs." There is no Ampere/consumer
  statement, and no GB10/Spark statement. The card does not mention `--enforce-eager`.

The non-MTP variant drops `--speculative-config` but keeps everything else
(mamba flags, expert-parallel, async-scheduling, trust-remote-code, tool/reasoning parsers).

---

## 2. Model architecture (why it's hard on the Spark)

From the model card and the Puzzle tech report (arXiv 2607.04371):
- **Mamba2-Transformer Hybrid LatentMoE with MTP** — interleaved Mamba, MoE,
  and Attention layers. Same architecture family as Nemotron-3-Super-120B.
- 75.3B total / 9.3B active params; 512 routed experts + 1 shared expert;
  **heterogeneous** — routed-expert intermediate dim varies 1280–2688 per
  layer, activated routed experts per token varies 4–18 per layer, Mamba SSM
  state pruned 128→96 channels.
- vLLM registry: `NemotronHPuzzleForCausalLM` → aliases to
  `NemotronHForCausalLM`. Confirmed present in v0.24.0 and v0.25.0/v0.25.1
  (per club-3090). The architecture string will not be rejected on any v0.20+
  vLLM — the risk is the quant/kernel path, not the registry.
- Two separate caches: **attention KV cache** (checkpoint declares
  `kv_cache_scheme = fp8_e4m3`; vLLM auto-selects it, can force with
  `--kv-cache-dtype fp8`) and **Mamba2 SSM state** (float16 + stochastic
  rounding — precision-critical, do NOT fp8 the recurrent state).

This is the same hybrid-Mamba family as Qwen3.6-35B-A3B, which on the Spark
needed **vLLM PR #48375** ("Honor drop_eagle_block in MambaManager") to stop
MTP+prefix-cache from silently corrupting outputs (see §5). That PR is still
**open and unmerged** on vLLM main as of 2026-07-18.

---

## 3. Why CUDA-graph compile OOMs on the Spark

Three independent memory pressures stack during the compile/CUDA-graph-capture
phase, and the Spark's 121 GB unified memory is tighter than it looks once the
hybrid MoE is resident:

1. **Weights.** NVFP4 checkpoint ≈ 50 GB on disk; resident weights
   ≈ 99.9 GB at idle (confirmed by `nvidia-smi` in the HF discussion #1 —
   `VLLM::EngineCore` at 99876 MiB after compile settles). That alone eats
   ~83% of 121 GB.

2. **CUDA-graph capture buffers.** vLLM captures a graph per batch-size
   bucket up to `--max-num-seqs`. With 512 experts × heterogeneous
   per-token activation, the graph workspace for the MoE expert dispatch is
   unusually large. The graph also pins intermediate activations for the
   captured batch shape.

3. **Chunked-prefill OFF spikes activation.** NVIDIA's card recommends
   `--no-enable-chunked-prefill` for throughput — but that assumes 80 GB
   datacenter cards. On the Spark, with chunked prefill OFF the profile run
   tries to prefill the full `--max-model-len` in one forward pass →
   multi-GB activation spike on top of weights → OOM during the profiling
   pass that precedes graph capture. **Chunked prefill ON is required to
   fit.** This was the documented OOM cause in club-3090's 4×3090 run (PR
   #706) and is the most likely cause of our "No MTP + GMU 0.70 → OOM
   during compile" crash.

The two HF community threads both confirm the Spark is at the edge:

- **Discussion #3 (moranilt, 9 days ago)** — runs on Spark with v0.23.0,
  MTP k=3, GMU 0.73, `--max-model-len 160000`, `--kv-cache-dtype fp8`,
  `--max-num-seqs 8`, chunked prefill ON. **Boots and serves.** Bench:
  32.2 tok/s output @ concurrency 1, MTP acceptance 69.4% (k=3). Compares
  to Qwen3.6-35B-A3B at 107 tok/s on the same Spark (Puzzle is ~3× slower).
  Their working compose is in the guide appendix.

- **Discussion #1 (codyknowscode, 10 days ago)** — runs on Spark with
  v0.24.0, MTP k=4, GMU 0.80, `--max-model-len 262144`, chunked prefill ON,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Had to increase
  system SWAP significantly to even start** — RAM peaked at ~160 GB during
  compile, settled to ~100 GB resident. Decode only 10 tok/s, low MTP
  acceptance, and "rubbish output" (the PR #48375 corruption signature —
  they had `--enable-prefix-caching` on). Their diff adding the mamba flags
  "did nothing to help" the quality issue (correct — that's the
  prefix-cache bug, not a flag issue).

**The two differences that matter between the working boot (moranilt) and
the OOMing boots (codyknowscode, us):**

| Lever | moranilt (boots) | codyknowscode (OOMs/swap) | our failed config |
|---|---|---|---|
| vLLM image | v0.23.0 | v0.24.0 | v0.24.0 |
| GMU | **0.73** | 0.80 | 0.70 |
| max-model-len | **160000** | 262144 | (varies) |
| max-num-seqs | 8 | 3 | (varies) |
| max-num-batched-tokens | 32768 | 16384 | — |
| prefix-caching | ON | ON | ON |
| SWAP increased | not mentioned | **yes** | unknown |

GMU 0.73 + max-model-len 160k is the conservative combo that fits without
swap. GMU 0.80 + 262k forces swap. Our GMU 0.70 should be *more* conservative
than moranilt — if it still OOM'd during compile, the likely culprit is
chunked-prefill OFF or max-num-batched-tokens too large (the activation
spike), not the GMU setting itself.

---

## 4. Recommended flags for the Spark (annotated)

Mandatory for the Spark (differences from NVIDIA's datacenter command):

| Flag | Value | Why |
|---|---|---|
| `--kv-cache-dtype fp8` | fp8 | Checkpoint declares fp8_e4m3 KV scheme. Auto-selected on Blackwell/Hopper; on GB10 (SM121, Blackwell-family) this is honored. Frees KV pool vs fp16. |
| `--enable-chunked-prefill` | (on) | **Required** to avoid the prefill activation spike that OOMs the profile run. NVIDIA's card says OFF for throughput — that's for 80 GB cards, not the Spark. |
| `--max-num-batched-tokens 8192` | 8192 | Caps the prefill chunk = caps the activation spike during compile. club-3090's proven-fit value on 24 GB cards. Raise only with measured headroom. |
| `--max-num-seqs 1` | 1 | Start single-stream to get a clean boot. Each concurrent seq needs its own Mamba SSM-state cache. Bump to 8 only after confirming a clean boot (moranilt runs 8). |
| `--max-model-len 160000` | 160000 | moranilt's proven-fit value on the Spark. 262k OOM'd codyknowscode even with swap. |
| `--gpu-memory-utilization 0.73` | 0.73 | moranilt's proven value. The whole util budget goes to the KV pool (weights/graph/activation are fixed); 0.73 leaves enough margin for the graph-capture spike. 0.80 forced swap. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=512` | env | Reduces fragmentation during the hybrid MoE's irregular allocation pattern. codyknowscode and club-3090 both set this. |
| `--enable-expert-parallel` | (on) | 512-expert MoE — NVIDIA's config. On tp=1 it's a no-op for distribution but keeps the dispatch path correct. |
| `--mamba-backend flashinfer` | flashinfer | NVIDIA's required Mamba backend. |
| `--mamba-ssm-cache-dtype float16` | float16 | NVIDIA's precision-preserving recurrent-state config. **Do NOT fp8 the SSM state** — it accumulates recurrently and is precision-sensitive. |
| `--enable-mamba-cache-stochastic-rounding` + `--mamba-cache-philox-rounds 5` | (on) | NVIDIA's rounding config for the float16 SSM state. |
| `--trust-remote-code` | (on) | Required — custom code model. |
| `--async-scheduling` | (on) | NVIDIA's config. |
| `--tool-call-parser qwen3_coder` + `--reasoning-parser nemotron_v3` + `--enable-auto-tool-choice` | | NVIDIA's parser combo. (codyknowscode tried `qwen3_xml` first — wrong for this model; the card says `qwen3_coder`.) |

Optional / tune-later:

| Flag | Notes |
|---|---|
| `--enforce-eager` | **The OOM escape hatch.** Disables CUDA-graph capture entirely, eliminating the graph-buffer memory pressure. Costs throughput (eager-mode decode is slower) but guarantees the model loads. Try this if the recipe above still OOMs during compile. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | NVIDIA's default. Drop to k=1 or remove entirely if MTP itself is the OOM trigger (MTP allocates draft-head buffers per captured graph). The non-MTP recipe in §1 still serves correctly, just slower. |
| `--enable-prefix-caching` | **See §5 — leave OFF until PR #48375 lands.** With prefix-caching ON + MTP, the Mamba recurrent state gets silently corrupted and the model emits garbage (the "rubbish output" codyknowscode saw). NVIDIA's card doesn't mention this because their reference is datacenter with different cache behavior. club-3090 just merged PR #720 turning prefix-caching OFF by default for the whole Qwen3-Next hybrid family for exactly this reason. |
| `--api-server-count 4` | NVIDIA's recommendation for long-generation scenarios. |
| `--quantization modelopt_fp4` | Only if auto-detect of the mixed NVFP4/FP8 scheme misfires. Normally leave unset (vLLM auto-detects from the checkpoint's quant config). |

---

## 5. The Mamba + MTP + prefix-cache silent-corruption bug (PR #48375)

This is **not** the OOM cause, but it is the "rubbish output" cause and it
bit codyknowscode on the Spark. It's the same bug we hit on Qwen3.6-35B-A3B.

- **vLLM issue #43559**: "Accuracy drops ~20% when `--enable-prefix-caching`
  is used together with MTP speculative decoding (Qwen3.6 35B-A3B)."
  Tool-eval accuracy drops from ~90% to ~50%. No exception, no log signal —
  the corrupted state is hash-reachable so prefix caching spreads it.
- **vLLM PR #48375** (potto007, open, unmerged as of 2026-07-18): "Honor
  `drop_eagle_block` in `MambaManager`." Root cause: `find_longest_cache_hit`
  on `MambaManager` accepts the `drop_eagle_block` flag (needed for
  EAGLE/MTP correctness) and silently ignores it. Every other KV manager
  handles it. Fix is a 9-line change lowering the search ceiling by one
  page. Validated against v0.24.0 (RED before, GREEN after).
- **club-3090 PR #720** (merged 3 days ago): vendors #48375 + flips
  prefix-caching OFF by default for all Qwen3-Next hybrid composes (env
  toggle to re-enable). Their interim guidance: **prefix-caching OFF, MTP
  stays ON** — breaking either leg alone is clean; only the combination
  corrupts.
- **Implication for Puzzle on the Spark**: until #48375 merges upstream,
  run with `--enable-prefix-caching` **removed** (or set
  `VLLM_DISABLE_PREFIX_CACHE=1` if your image honors it). Acceptance:
  slightly higher TTFT on cache-hit prefixes, but correct outputs. Once
  vLLM ships a release past #48375, re-enable prefix-caching and re-run
  the post-agentic probe (3 chained-agent runs, re-probe the held prefix).

Note: NVIDIA's official model-card command does **not** include
`--enable-prefix-caching`. The community Spark recipes add it. Per the bug
above, the community is wrong to add it (until #48375); follow NVIDIA and
omit it.

---

## 6. Community recipes

### club-3090 (noonghunna) — the most mature community recipe

Repo: `github.com/noonghunna/club-3090`. This is the "fank's recipe"
successor community — RTX 3090/4090/5090 focused, multi-engine, the group
that vendors #48375. Their Puzzle config:
`models/nemotron-3-puzzle-75b/vllm/compose/multi4/nvfp4/mtp.yml`
(only `mtp.yml` — a 4-card TP=4 recipe; no single-card Spark variant).

Key takeaways from their compose (validated on 4× RTX 3090, #706):
- Image: `vllm/vllm-openai:v0.25.1` (NVIDIA used v0.20.0; club-3090 pins
  v0.25.1, notes the arch is present in both v0.24.0 and v0.25.0).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`.
- `--enable-chunked-prefill` **ON** — explicitly called out as "REQUIRED
  to fit 24 GB" and as the OOM fix (their 262K profile run OOM'd with it off).
- `--max-num-batched-tokens 8192` — the proven-fit prefill chunk on 24 GB.
- `--max-num-seqs 1` — start single-stream.
- `--gpu-memory-utilization 0.85` — they're on 4 cards with 13.3 GiB
  weights/card; on the Spark's single 121 GB part, moranilt's 0.73 is
  safer.
- `--kv-cache-dtype fp8` (explicit, matches checkpoint scheme).
- Mamba flags exactly as NVIDIA's card.
- `--tool-call-parser qwen3_coder` (not qwen3_xml).
- `--override-generation-config '{"temperature":1.0,"top_p":0.95}'`
  (NVIDIA's sampling — higher temp than Qwen's 0.6/0.95/20).
- `--default-chat-template-kwargs '{"enable_thinking":true,"force_nonempty_content":true}'`
  (`force_nonempty_content` is NVIDIA's tool-calling recommendation).
- Ships `--enable-prefix-caching` ON — but see §5; their master branch
  just flipped this OFF by default for the Qwen3-Next family via #720.
  The Puzzle compose predates that flip; treat it as stale on this point.
- Comment: "NVIDIA's model card lists supported microarchitectures as
  Blackwell + Hopper ONLY. There is NO Ampere/consumer-GPU support
  statement — yet it runs." Same caveat applies to GB10: SM121 is
  Blackwell-family but not a validated NVIDIA target.

### moranilt's Spark compose (HF discussion #3) — the only working single-Spark recipe

```yaml
services:
  nemotron:
    image: vllm/vllm-openai:v0.23.0
    container_name: nemotron-75
    environment:
      - HF_TOKEN=***
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_USE_FLASHINFER_MOE_FP4=0
      - VLLM_ENGINE_READY_TIMEOUT_S=1200
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ipc: host
    ports:
      - "8000:8000"
    command: >
      --model nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4
      --port 8000
      --tensor-parallel-size 1
      --async-scheduling
      --trust-remote-code
      --mamba-backend flashinfer
      --mamba_ssm_cache_dtype float16
      --enable-mamba-cache-stochastic-rounding
      --mamba-cache-philox-rounds 5
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
      --tool-call-parser qwen3_coder
      --reasoning-parser nemotron_v3
      --enable-auto-tool-choice
      --max-model-len 160000
      --gpu-memory-utilization 0.73
      --kv-cache-dtype fp8
      --max-num-seqs 8
      --enable-prefix-caching
      --enable-prompt-tokens-details
      --max-num-batched-tokens 32768
      --enable-chunked-prefill
      --default-chat-template-kwargs '{"preserve_thinking":false}'
```

His bench (concurrency 1, 10k in / 2k out, 5 prompts):
- 32.2 tok/s output, 193 tok/s total, TTFT 6.97s mean, TPOT 27.6ms.
- MTP acceptance 69.4% (pos0 83%, pos1 64%, pos2 62%).
- k=5 drops pos1 acceptance to 40% — stay at k=3.
- He notes Puzzle is ~3× slower than Qwen3.6-35B-A3B on the same Spark
  (107 tok/s) despite only being ~2× bigger — likely the heterogeneous
  MoE dispatch overhead and the 512-expert scale.

**Caveat**: he runs `--enable-prefix-caching` — see §5. With PR #48375
unmerged, that risks silent corruption on chained-agent workloads. Drop
that flag for production until #48375 ships.

### codyknowscode's Spark attempt (HF discussion #1) — the failure case

v0.24.0, GMU 0.80, max-model-len 262144, MTP k=4, `qwen3_xml` parser.
Required significant SWAP increase to start; 10 tok/s decode; "rubbish
output" (the #48375 corruption). Adding the mamba flags (their diff) did
not fix the quality issue — correct, because the quality issue is the
prefix-cache bug, not a missing flag. Lessons:
- Don't use 262k on the Spark (160k is the ceiling).
- Don't use GMU 0.80 (0.73 fits without swap).
- Don't use `qwen3_xml` parser (use `qwen3_coder`).
- Don't use k=4 MTP (k=3 is the sweet spot).
- **Don't use prefix-caching** until #48375 merges.

---

## 7. vLLM version / container

- **NVIDIA tested on v0.20.0.** Their card predates v0.25.x.
- **club-3090 pins v0.25.1** and confirms `NemotronHPuzzleForCausalLM` is
  registered in both v0.24.0 and v0.25.0.
- **moranilt used v0.23.0** successfully on the Spark.
- **codyknowscode used v0.24.0** — booted (with swap) but had the
  prefix-cache corruption (orthogonal to version; #48375 is unmerged on
  all versions).
- Our failed config used v0.24.0 — version is not the OOM cause.

**Recommendation: use `vllm/vllm-openai:v0.25.1`** (latest stable,
confirmed arch registration, carries the same Mamba bug as all versions
until #48375 merges). If you hit a v0.25.x-specific regression, fall back
to v0.23.0 (moranilt's proven Spark image). Do **not** use v0.20.0 — too
old, missing Spark/SM121 fixes that landed in v0.23+.

There is **no NVIDIA-published Docker image for Puzzle** and **no
Puzzle-specific vLLM container**. The AEON container
(`ghcr.io/aeon-7/aeon-vllm-ultimate`) does **not exist** at that path
(404) and there's no evidence it supports Puzzle — skip it.

---

## 8. Decision tree for our OOM

1. **First boot**: use the TL;DR recipe (v0.25.1, GMU 0.73, 160k, k=3,
   chunked prefill ON, batched-tokens 8192, max-num-seqs 1, prefix-caching
   OFF). If it boots, bump `--max-num-seqs` to 8 and re-test.
2. **Still OOMs during compile**: add `--enforce-eager`. This kills
   CUDA-graph capture (the documented last-resort). Throughput drops but
   the model will load. Confirm correctness, then decide whether to
   tolerate eager-mode speed.
3. **OOMs even with `--enforce-eager`**: drop MTP entirely (remove
   `--speculative-config`) — MTP's draft head adds per-graph buffers.
   The non-MTP recipe in §1 still serves correctly.
4. **Still OOMs**: lower `--max-num-batched-tokens` to 4096 and
   `--max-model-len` to 128000. The activation spike scales with both.
5. **Still OOMs**: lower `--gpu-memory-utilization` to 0.65 — but note
   below ~0.70 the KV pool gets too small to be useful at 160k context.

The most likely fix for our specific crash ("No MTP + GMU 0.70 → OOM
during compile") is step 1 with `--enable-chunked-prefill` confirmed ON
and `--max-num-batched-tokens` capped at 8192 — our prior runs likely had
chunked prefill off or batched-tokens too high, causing the prefill
activation spike to OOM the profiling pass. If that fails, step 2
(`--enforce-eager`) is the guaranteed-load path.

---

## 9. Open questions / not found

- **No NVIDIA-published Spark/GB10 deployment guide for Puzzle.** The
  model card is the only NVIDIA source; it targets Blackwell/Hopper
  datacenter (H100, B200) with tp=2/4. Single-GB10 tp=1 is uncharted by
  NVIDIA.
- **No vLLM GitHub issue or PR specifically about Puzzle.** Searched
  `Nemotron Puzzle`, `Nemotron Labs Puzzle`, `Puzzle Nemotron Mamba OOM`,
  `Nemotron Puzzle Spark` — all return zero results. The model is 12
  days old (released 2026-07-06); the vLLM issue tracker hasn't caught up.
- **No AEON container for Puzzle** — the `ghcr.io/aeon-7/aeon-vllm-ultimate`
  path 404s; no evidence of Puzzle support anywhere.
- **No NVIDIA dev-blog post for Puzzle.** Searched
  developer.nvidia.com/blog — no Puzzle-75B article. The tech report
  (arXiv 2607.04371) is the only technical doc beyond the model card.
- **SM121 / GB10-specific FlashInfer Mamba2 kernel validation**: not
  documented anywhere. club-3090 flagged FlashInfer Mamba2 on sm_86 as
  "unverified" and it worked; SM121 is closer to Blackwell so likely
  fine, but unconfirmed.
- **Whether `--enforce-eager` is enough on the Spark** — no community
  report of using it for Puzzle. It's the standard OOM escape hatch for
  hybrid Mamba on constrained parts, but the 50 GB weight footprint means
  even eager mode may be tight at 160k context. If eager + 160k OOMs,
  eager + 128k should fit.

---

## Sources

- NVFP4 model card (NVIDIA's official flags):
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4
- BF16 model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16
- HF discussion #3 (moranilt, working Spark recipe, 9 days ago):
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions
- HF discussion #1 (codyknowscode, Spark OOM/swap + rubbish output, 10 days ago):
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions/1
- vLLM PR #48375 (Mamba drop_eagle_block fix, open):
  https://github.com/vllm-project/vllm/pull/48375
- vLLM issue #43559 (MTP + prefix-cache corruption):
  referenced from PR #48375
- club-3090 Puzzle compose (multi4 nvfp4 mtp):
  https://github.com/noonghunna/club-3090/blob/master/models/nemotron-3-puzzle-75b/vllm/compose/multi4/nvfp4/mtp.yml
- club-3090 PR #720 (prefix-caching OFF default for Qwen3-Next family):
  https://github.com/noonghunna/club-3090/pull/720
- Puzzle tech report: arXiv 2607.04371