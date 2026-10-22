# Sage-A1.5 Arena Bench Runbook — 2026-10-22 night (1am window)

**Model:** InternScience/Sage-A1.5 — 35.9B-total MoE agentic VLM (Qwen3.6-35B-A3B base + Agents-A1), Apache-2.0, bf16, 66.97GB weights, 30 files, rev ba7d931dbbc5dc77901e285edcd3bd9555e199ca, MTP module **in-pack** (model-mtp.safetensors).
**Engine:** vLLM (single-node TP1, GB10 solo). Image pinned: `vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5` (arm64-cu130, on controller).
**Submission:** sub1792649874991 — official vendor config, bench id bench_d92f1a007527, 28/28 cells in ~1h.

## Recipe (final as-run)
Registered `@styles01/sage-a15` (GH: styles01/sparkrun-recipes/recipes/sage-a15.yaml). Defaults: gmu 0.80 / 262,144 max_model_len / max_num_seqs 3 / kv bf16 / served_model_name = FULL HF id / prefix+chunked caching on / qwen3_xml tool parser.
`executor_config: entrypoint:"", user: root, shm_size: 16gb` — REQUIRED (see pitfalls).

## Results (llama-benchy 0.4.0, official spark-arena-v2 matrix)
- **Decode c1 (daily-driver bar 80): 22.9–29.0 tok/s — FAILS the bar** (d0 29.0, d65535 24.8, d8192 28.4, d16384 27.9, d32768 26.9, d100000 23.0)
- Aggregate decode: 51.9 @c5, 50.2–50.5 @c2, ~40 @c10 deep
- Prefill: 2,849 @d8192c5 (pp512), 4,878 @8K-c1 (d8192 cell), 1.2K–4.9K across depths; 1,933 @100K-c5
- Owner's verdict: NOT daily-driver material raw (prefill specialist shape, like ornith). Phase-2 lever: in-pack MTP as its own tuned submission (only if k-sweep beats ornith's k1 net-negative result).

## Pitfalls learned (all four costless once known)
1. `vllm/vllm-openai` images: the ENTRYPOINT consumes sparkrun's launcher → `executor_config.entrypoint:""` MUST be in the recipe BEFORE firing (the runner aborts with the advisory otherwise).
2. The empty entrypoint alone → a sleeper container w/o serve, then uid-2000 exec = `chdir /home/vllm` PermissionError in 4s (silent, no docker logs) → `user: root` + `shm_size: 16gb` also REQUIRED.
3. The serve exec logs = preserved INSIDE the mounted HF cache as `serve_log_preserved.txt` — read there, not `docker logs` (which = only the sleeper's stdout = empty).
4. The recipe edits between fires = legal pre-run; the submitted recipe metadata = frozen post-run (the GH copy = the as-run final).

## Lane bookkeeping
- Window ran on the freed :8000 (the dsv41 canon torn down under the 1am bench-window law, restored after the upload: `/home/jaita/relaunch-dsv41-canonical.sh`, boot ~380s, DSpark k=6).
- Weights + build remnants TRASHED after the upload per James: `trash/sage-a15-hubcache-20261022` (67GB) + the runtime-cache + the local v26-sage image (rmi, 25GB) — the recipe/registration persist.
- Keep for later: the vllm-v26-patched image (Qwen3_5MoeMTP + Qwen3DSparkModel classes) = the Phase-2 MTP lever candidate.