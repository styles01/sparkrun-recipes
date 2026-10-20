# Runbook: deepseek-v41-1017-ablit-tp2 (the abliterated 1017 lane)

Production-trial lane on two DGX Spark GB10 (sm_121) nodes since 2026-10-20. This is the
serving variant of [deepseek-v41-1017-exl3-tp2](../recipes/deepseek-v41-1017-exl3-tp2.yaml):
same Sudosu engine lineage, but loaded from the **dealignai 1017 abliteration pack** with a
loader-side fix instead of a weights-side one. Decision to keep canon or cede to this lane is
the owner's, to be made on real use, not benchmarks.

## 1. The model

- **Pack:** [dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED)
  (45 shards, ~415.9GB, EXL3 2.9bpw, abliteration of the same 1017 refresh our canon lane runs).
- **Provenance (trust bytes, not card diffs):** verify before launch — date stamps (created
  2026-10-18 vs the decoy Sep-17 pre-1017 packs), decisive-tensor diffs against our canon tree,
  blob-size equality with the HF API manifest. The name alone proves nothing.
- **The 54-module quirk:** the pack ships 54 modules left unquantized **inside an EXL3 pack** —
  `model.language_model.layers.10-36` × (`attn.wo_b` ×27 + `ffn.shared_experts.w2` ×27),
  bf16, out-major. Our prior engine (v51 / TensorFold lineage) hard-rejects mixed packs
  (plan.py rejects anything not pure-EXL3).

## 2. The engine fix — v52rc bf16-loader

`dsv41-serve:v52rc` = `ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v2` (v51) with a
plugin-only overlay, no Sudosu engine patches changed:

- `vllm_exl3_moe` gets a `get_quant_method` checkpoint probe: modules whose safetensors dtype
  is bf16 resolve to `UnquantizedLinearMethod` and stream bf16; everything else rides the EXL3
  path unchanged. Loader-side, zero repack, zero weight re-encoding. (Plugin source:
  `docker/ablit-bf16-loader/` in this repo — full v52rc `config.py` + the v51→v52rc unified
  diff; the v52rc image's engine layer is labeled with plugin commit a1040ea4.)
- GB10 build: `TORCH_CUDA_ARCH_LIST=12.1` (sm_121), torch cu130.
- `FUSED=1 VLLM_EXL3_MOE_SMOE_SHARED=1` — measured **no-op on decode** (FUSED=0 set-b
  88.8/46.3/115.3 vs FUSED=1 87.9/47.4/113.1, inside 5-6% noise); kept to match canon config.

## 3. Numbers (measured, greedy, through the drafter)

| metric | ablit twin | canon 1017 (v51) | ratio |
|---|---|---|---|
| decode set-b (code/prose/structured) | 87.9 / 47.4 / 113.1 tok/s | 93.8 / 52.4 / 126.8 | 0.94 / 0.90 / 0.89× |
| decode set-a | 68.3 / 42.1 / 97.4 tok/s | 88.3 / 49.1 / 105.9 | 0.77 / 0.86 / 0.92× |
| prefill @8K | 1,486 tok/s | — | — |
| prefill @32K | 1,613 tok/s | — | — |
| prefill @131K | 1,508 tok/s (TTFT 86–87s) | — | flat scaling |
| DSpark acceptance | 43.7% | 49.3% | — |
| DSpark tokens/verify-step | **2.62** | 2.46 | ↑ healthier |

Interpretation: the raw bf16 stream costs ~9% decode, and the k=6 drafter claws most of it
back — the published set-b numbers are **post-drafter**. The residual decode deficit is
intrinsic (FUSED on/off identical); the real cost is quality: abliterated 54 modules incl. 27
shared_experts touch general reasoning — pack card MMLU-500 −4.2pp, STEM −7.6pp.

## 4. Serving layout

- Trees: `~/models/dsv41-1017-ablit-twin/` (mirror pack) → hardlinked to
  `~/models/dsv41-ablit-tree/` (serving tree, both nodes — hardlinks, never symlinks; symlinks
  crash the tokenizer). `HF_HOME` hub cache primed the same way.
- KV pin: `KV_BYTES=4000000000` (plain bytes — the ENV parses literals, not `4e9`) → pool
  **1,060,858 tokens, 4.05× concurrency @ 262K ctx**, `MAX_SEQS=4`, `UTIL=0.85`.
- Spec: `SPEC='{"method":"dspark","num_speculative_tokens":6,"draft_parallel_config":{"tensor_parallel_size":2}}'`
  — TP2 drafter.
- SERVED (verbatim, no auto-alias): `dsv41 DeepSeek-4.1-Flash DeepSeek-4.1-Flash-Ablit GLM-5.3-Flash-EXL3 vision`.
  **Alias law:** every vision-capable lane serves the stable `vision` alias — count names at
  boot; the launcher will happily serve a short set and break aux consumers silently.
- Boot gates before declaring up: `/v1/models` name count, KV pool == pin,
  vision probe ('Blue' color test), uncensor probe substantive (≥900 chars body), Canberra
  coherence probe, 100-tok fixed probe 2.4–2.6s.

## 5. Launch / stop / rollback

```bash
# up (both nodes rank up; drain gate FIRST: /metrics running+waiting both 0)
cd /home/jaita/dsv41_recipe_ablit && \
HEAD=10.200.0.1 WORKER=10.200.0.2 MODEL_DIR=/home/jaita/models/dsv41-ablit-tree TEXT_ONLY=0 \
SPEC='{"method":"dspark","num_speculative_tokens":6,"draft_parallel_config":{"tensor_parallel_size":2}}' \
UTIL=0.85 MAX_SEQS=4 KV_BYTES=4000000000 \
DOCKER_ENV='VLLM_EXL3_MOE_SMOE_SHARED=1 VLLM_EXL3_MOE_FUSED=1' \
IMAGE=dsv41-serve:v52rc \
SERVED='dsv41 DeepSeek-4.1-Flash DeepSeek-4.1-Flash-Ablit GLM-5.3-Flash-EXL3 vision' \
bash scripts/launch/dsv41_tp2.sh up

# stop — only after the drain gate read 0/0. Never force.
bash stop-dsv41.sh   # effect-verify: port + containers gone on BOTH nodes

# rollback to canon (instant, no build):
# LANE-REGISTRY.yaml → retired: dsv41-1017-ours-canon
cd /home/jaita/dsv41_recipe && \
MODEL_DIR=/home/jaita/models/dsv41-1017-ours IMAGE=ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v2 \
SERVED='dsv41 DeepSeek-4.1-Flash GLM-5.3-Flash-EXL3 vision' bash scripts/launch/dsv41_tp2.sh up
```

## 6. Windows discipline

- Heavy I/O (fetch, rsync of model trees) while the lane serves: only with the measured-throttle
  kit (nice 19, ionice idle, 8MB chunks, POSIX_FADV_DONTNEED, Range-resume) and a fixed-workload
  lane probe (100-tok ignore_eos, every cycle) — prove the lane didn't move, otherwise stop.
- Watchdogs must pgrep remotely (`ssh <host> pgrep -f …`) or you get phantom exits; pkill
  patterns need bracket-regex against self-match.
- Teardowns before GPU work require the drain gate AND effect-verification (container gone on
  both nodes; spark2's rank shows 0 available while TP2 serves — that's expected).

## 7. Lineage & lessons

- Decoy packs with matching names exist (Sep-17 "pre-1017 ablit" trees that lack the tensor
  changes). Date + 7/7 decisive-tensor diffs + HF createdAt were the discriminators.
- The cron-window pattern (write the runbook first, arm a one-shot with it, gates in
  numbers-only form) is what survived the night; the runbook itself refused to relight while
  the lane was up — let it, don't `pkill` first.
- Determinism: outputs are not bit-stable run-to-run (fp32 atomics in split-k + MoE epilogue);
  do not build byte-compare gates on this engine.