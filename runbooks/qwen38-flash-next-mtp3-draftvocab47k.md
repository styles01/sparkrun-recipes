# Qwen 3.8 Flash-Next — MTP3 + Draft Vocab 47K (vLLM) — CURRENT PRODUCTION DAILY DRIVER

> **Status: the lane we run.** If you are reading this repo and want the lane that
> handles real agent traffic on one GB10 today, this is it. Bench v27 passed all
> 28 cells (Arena submission `sub1789171536939`); this exact config serves Hermes/Loca/Lara
> in production.

- **Model:** [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) @ `925d7be6`
- **Runtime:** vLLM (fork `v0.1.dev20073+g8e685d198`), **MTP k=3** speculative decoding with `use_local_argmax_reduction`
- **Draft vocab:** custom 47,149-token code-oriented draft vocabulary (FR-Spec-style reduced draft head; `files/build_draft_vocab.py`; exact — target rejection sampling keeps outputs bit-identical)
- **Container:** [`ghcr.io/styles01/qwen38-flash-next-mtp3-draftvocab47k:v7`](https://github.com/users/styles01/packages/container/qwen38-flash-next-mtp3-draftvocab47k) (public; PLE table + draft vocab baked)
- **Recipe:** [`recipes/qwen38-flash-next-mtp3-draftvocab47k.yaml`](../recipes/qwen38-flash-next-mtp3-draftvocab47k.yaml)
- **Arena evidence:** bench v27 — 28/28 cells passed, zero errors. Prefill 1.9k tok/s short-context, 500–700 tok/s at 4k–8k; decode 33–57 tok/s aggregate at conc 1–10; 100k-context cells healthy (28.5+ tok/s decode at conc 1–2)

## Serve shape (production)

```
vllm serve Mia-AiLab/Qwen3.8-Flash-Next-NVFP4
  --served-model-name qwen3.8-flash-next --tensor-parallel-size 1
  --gpu-memory-utilization 0.78 --max-num-seqs 8 --max-num-batched-tokens 2048
  --max-model-len 262144 --kv-cache-dtype fp8
  --load-format safetensors --safetensors-load-strategy lazy --enable-chunked-prefill
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
```

- **Tool calling:** `--tool-call-parser qwen3_coder` (OpenAI-compatible `tool_calls`; validated with live agent traffic)
- **Reasoning:** `--reasoning-parser qwen3`; client sets `reasoning_effort` per request
- **Context:** 262,144 tokens max; FP8 KV keeps the KV budget sane at depth

## Switch sequence

1. Stop the current lane (kill the ds4-server / `docker stop <vllm container>`)
2. **Verify memory is actually free** — `nvidia-smi` is NOT enough; check no process holds GPU memory
3. `docker start vllm-fn-tp1` (canonical container; config frozen in the container)
4. Wait for `/v1/models` 200 (~4–7 min shard load from NVMe; lazy strategy)
5. Smoke: `GET /v1/models` → `owned_by: "vllm"`, model `qwen3.8-flash-next`, `max_model_len: 262144`; one tool-call test

## Known behavior / gotchas

- **Tool-call args can come back empty (`{}`) under very long agent contexts (110k+).**
  Hermes rejects them and the agent retries; vLLM itself logs nothing. Watch for
  `Rejected invalid terminal command value: NoneType` in the client's log.
- The PLE n-gram table is **pre-baked into the image** (`--build-context ple-host=...`).
  Rebuilding it in-container costs ~40GB transient RAM and OOMs the box — never run
  an unbaked build on a 121G machine.
- Production container runs with a **111G cgroup cap** so an OOM kills the container,
  not the box. Do not run the bench without the equivalent cap (`--executor-args
  "--memory=104g --memory-swap=111g ..."` under `SPARKRUN_ADVANCED=1`).
- Concurrency is client-queued; do not inflate `--max-num-seqs` to match concurrency.
- `VLLM_MTP_DRAFT_VOCAB` / `VLLM_PLE_PACKED_TABLE_DIR` warnings at startup are normal
  on the fork (envs are read internally).

## Alternate lane: DwarfStar (ds4 C engine)

Same model, different engine: [`recipes/qwen38-flash-next-ds4.yaml`](../recipes/qwen38-flash-next-ds4.yaml)
+ [`runbooks/qwen38-flash-next-ds4.md`](qwen38-flash-next-ds4.md). Faster at short-context
single-stream decode (~36.7 vs ~33 tok/s) but single-lane, weaker deep prefill, 32k serve window.
Metrics emitter + Qwen-native tool-call parsing were added server-side on 2026-09-12
(see that runbook's notes).

## Artifacts

- Draft vocab: 47,149-token id list + gathered Q8_0 output blob — [`recipes/qwen38-flash-next-draft-vocab-list.txt`](../recipes/qwen38-flash-next-draft-vocab-list.txt), [`recipes/qwen38-flash-next-draft-vocab-output-q8_0_47149.bin`](../recipes/qwen38-flash-next-draft-vocab-output-q8_0_47149.bin) (LFS)
- Dated bench evidence: `benchmarks/` (bench v27 = this lane, submission `sub1789171536939`)
