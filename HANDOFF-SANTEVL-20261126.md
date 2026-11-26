# HANDOFF — LingSante-3.1-air-VL (sante VL) — 2026-11-25/26
# Read this FIRST if you are the next model and James says anything about sante / ling / air-VL / VL.

## THE STATE, plainest possible
- **Goal:** serve LingSante-3.1-air-VL (124B vision+text MoE, the newest Ling sante) as the canonical sante lane on spark1, replacing the old Ling-3.0-flash-Sante lane.
- **Owner just switched models mid-task. This file is the entire context.**

## WHAT LIVES WHERE (verified this session)
- **Pack:** official `inclusionAI/LingSante-3.1-air-VL-MXFP4` — 69.58 GB, 35 shards, MIXED quant:
  - experts: MXFP4 (E2M1 g32, E8M0 scales, 2 values/byte)
  - attention/dense/shared: FP8 E4M3, 128×128 blocks, ue8m0 scales stored as `float8_e8m0fnu` dtype
  - BF16: vision tower, projector, embeddings, lm_head, router, norms
  - FP32: KDA `A_log`, `dt_bias`, router `expert_bias`
  - `text_config.max_position_embeddings` = 131072 (NOT 262144 — booting with 262144 crashes; that was one of my errors)
- **Location:** spark1 `/home/jaita/models/lingsante31-airvl-fp4` (LOCAL, correct), plus NFS mirror from spark2.
- **Architecture:** `bailing_moe_v3_vl` → `BailingMoeV3VLForConditionalGeneration` — has BOTH a vision tower (Qwen3-style ViT) AND 35 KDA/hybrid mamba layers inside `text_config`. Text-only Ling-3.0-Sante has neither.

## THE HUNT — WHAT'S BEEN TRIED AND EXONERATED
ALL boots produce IDENTICAL multilingual soup (~28 tok/s, fluent-garble from token one). Each of these was tested cleanly, one at a time:
1. Base boot (dsv41-serve:lingv3 image) — garbage.
2. + PR #59026 patch (quantization_config must fall back from `text_config` (absent) to top-level `hf_config`) — confirmed landed, confirmed the `ue8m0` log line that ONLY appears with the fix — **still garbage**.
3. + forced cutlass pin via kernel-config — **crashed**: this fork's KernelConfig has no `linear_backend_per_quant` (that was a hy4-image extension).
4. + `--mamba-cache-mode align` + `--enable-prefix-caching` (the two official-card vLLM flags we'd missed) — **still garbage**, unchanged signature.
5. Offline numeric isolation ladder on the pack: scales read sanely through ALL paths (native float = old patch = new patch, 0.000488–0.00195), dequanted weights sane (abs-max 0.5625, std 0.039). FP8 scale decode fully exonerated. The earlier "cutlass ue8m0 decode is broken" hunch was WRONG — torch's `float8_e8m0fnu.float()` handles it; my byte-level probe misread it.

**Exonerated:** pack data, FP8 scales, Marlin MoE selection, graphs, sampling, config scope.
**Remaining lead (unprobed):** the KDA/mamba hybrid mixer and/or TRITON_MLA attention on sm_121. Text-only Ling-3.0-FP4 works on the SAME engine only because it has NO mamba and NO vision tower. This is the next probe: per-layer numeric isolation — zero out mamba layers (env-gated debug in the model code), see if text cleans up.

## THE PUBLIC EVIDENCE (what's actually working out there)
- vLLM + GB10 + this exact pack: **nobody has it working publicly.** inclusionAI's own GB10 recipes only publish INT4/GGUF for the VL family.
- Working stacks ARE: (a) **vcruz305's own SAGE-REXL3 quant** on his exllamav3 fork, 48 tok/s 1-spark (owner has REJECTED this quant: "I don't want another quant" + "I don't want the exl3 sage one either" — do NOT re-propose); (b) **SGLang flashinfer_mxfp4 on 4×H20-3e datacenter HW** (no sm_121 cubins exist for that backend, so it cannot run on Spark); (c) text-only sibling Ling-3.0-flash on vLLM 0.32 (OUR pack, works, 34.7 tok/s); (d) r0b0tlab's VL variant as **NVFP4** (different quant) on the same fork family, 43.5 tok/s GB10-verified.
- Open bug matching our symptom: **inclusionAI/Ling issue #26** — "Ling-3.0-flash-fp4 spurious tokens with W4A16+Marlin on GB10" — unresolved upstream.

## IMAGES (renamed this session — NAMING LAW now stands)
Permanent rule (in my memory): **name images by content, not ancestry.**
- `spark-vllm:0.32.0` — general vLLM 0.32.0 estate build (was dsv41-serve:v032)
- `spark-vllm:0.32.0-ling3` — + inclusionAI vllm-ling-v3 fork, Marlin MoE 29-arg ABI rebuild
- `spark-vllm:0.32.0-ling3-fix` — + PR #59026 fallback + (harmless but useless) "ue8m0 native decode" edit. **This is the image to boot the air-VL pack with.**
- Also ghcr public: `ghcr.io/styles01/spark-vllm:0.32.0` + `:0.32.0-ling3-fix` pushed/pushing.
- Registry/launchers/runbooks on Mac+spark1 all updated to the new names; pushed GitHub commit 7efa680.
- Truly DS4.1-only (`dsv41-serve:v52rc`) keeps its name — it's not misleading.

## CANONICAL BOOT (the tested recipe that boots+garbles, in scratch/airvl_official.sh on Mac):
```
docker run -d --name lingsante31airvl --network host --ipc host --shm-size 64g --gpus all \
  --memory=110g --memory-swap=110g \
  -v /home/jaita/models/lingsante31-airvl-fp4:...:ro \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e TORCH_CUDA_ARCH_LIST=12.1a \
  --entrypoint /w/ling-env/bin/vllm spark-vllm:0.32.0-ling3-fix serve \
  /home/jaita/models/lingsante31-airvl-fp4 \
  --served-model-name sante ling-sante-3-1-air-vl vision \
  --host 0.0.0.0 --port 8000 --max-model-len 131072 --max-num-seqs 3 \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --enable-prefix-caching --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser ling3 --reasoning-parser ling3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}'
```
Probe: `python3 /tmp/probe_airvl.py` on spark1 (Mac copy in scratch).

## ESTATE STATE WHEN I LEFT
- Both sparks DARK (only sparkDash on spark1, lingbuild inert container on spark2). dsv41 was drained+torn down for the last airVL boot and NOT relit. Owner may want it back — the canonical TP2 relight line is in the registry (launcher verified 11-23/24).
- Trash EMPTIED on both sparks (owner said "empty the trash": spark1 230G, spark2 160G reclaimed).
- spark1 disk 66G free; spark2 175G free.
- A background GHCR push of spark-vllm:0.32.0-ling3-fix may still be running on spark1 — check `pgrep -af "docker push"` before heavy I/O there.

## THE NEXT STEP (if James approves continuing)
Per-layer mamba/mixer numeric isolation on the fixed image: env-gate a debug flag in `/w/ling/vllm/model_executor/models/bailing_moe_v3_vl.py` (host tree at spark1:/home/jaita/lingv3-extract/w/ling, patched copy) to zero the KDA mamba layer outputs, boot, see if text renders. If yes → mamba is the poison; if no → TRITON_MLA attention. Each boot is ~4 min + probe. Then the fix depends entirely on which fires.
