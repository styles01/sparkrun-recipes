# Spark ComfyUI I2V: Path Forward Brief
**Date:** 2026-07-08 | **Spark:** larryspark.local (DGX GB10, 121GB VRAM) | **M4Max:** James-M4Max-5.local (48GB, Draw Things proven)

## Situation
Spark runs ComfyUI + Kijai WanVideoWrapper with `Wan2_2-I2V-A14B-HIGH_bf16.safetensors` (27GB). T2I works. I2V fails with:
```
RuntimeError: weight [5120,36,1,2,2] expected input[1,68,21,30,52] to have 36 channels, got 68
```
The model expects 36 input channels (16 latent + 16 image cond + 4 mask = standard Wan 2.2 I2V), but receives 68. This is a **workflow/VAE configuration mismatch**, not a fundamental model incompatibility. The workflow JSON shows `fun_or_fl2v_model=True` on `WanVideoImageToVideoEncode`, which alters latent concatenation logic.

M4Max already runs Wan 2.2 14B I2V successfully via Draw Things gRPC (proven: actual motion confirmed, 5-step smoke test passed). SSH to M4Max currently fails (permission denied). FAL_KEY exists on Mini.

---

## Option Ranking

### 1. Fix Kijai Nodes on Spark — **GO (Priority 1)**
- **Effort:** 2–4 hrs
- **Risk:** Medium
- **Action:** The 68-channel input suggests the `WanVideoImageToVideoEncode` node is outputting a latent with extra channels. Most likely culprit: `fun_or_fl2v_model=True` in the workflow JSON while the loaded model is NOT a Fun/FLF2V variant. Fix: edit workflow → set `fun_or_fl2v_model=False`. Alternative: verify VAE output channels match model expectation; ensure `Wan2_2_VAE_bf16` is correct for I2V (not T2V VAE). Check `nodes_model_loading.py` line 1623 — latent format falls back to `Wan21` for 14B because `dim==3072` check is only for 5B; this may cause latent shape mismatches if ComfyUI’s internal preview/format logic leaks into sampler dimensions.
- **Verdict:** This is the **correct fix** because Spark has 121GB RAM and should run full-precision I2V natively. Do NOT abandon this without trying the workflow flag fix first.

### 2. Convert M4Max Draw Things ckpt for Spark — **NO-GO**
- **Effort:** Days–weeks (likely impossible)
- **Risk:** Extreme
- **Reason:** Draw Things uses Apple-specific quantized formats (SVDQuant Q6P/Q8P) with custom DTTensor packing and FlatBuffer configs. There is no public converter to standard safetensors. You would need the original full-precision weights from HuggingFace, not the Draw Things download.

### 3. Run I2V on M4Max via Draw Things API — **GO for prototyping, NO-GO for production**
- **Effort:** 1–2 hrs (pipeline exists)
- **Risk:** Medium
- **Capacity:** 14B I2V uses ~22GB RAM (Activity Monitor confirmed). M4Max has 48GB. Juggernaut XL T2I uses ~8–10GB. **Both cannot run simultaneously** because Draw Things loads one model at a time into VRAM/DRAM. Switching models requires UI interaction or gRPC model switch (slow).
- **Speed:** ~280s/step at 832×448, 81 frames. 15 steps ≈ 70 min. 30 steps ≈ 2.3 hrs.
- **Verdict:** Keep as **backup/prototyping** path. Do NOT make it the primary pipeline because it blocks Aurore T2I production and is an order of magnitude slower than Spark should be.

### 4. FAL Cloud API for I2V — **GO for immediate relief**
- **Effort:** 2–3 hrs to integrate
- **Risk:** Low technical, medium operational
- **Details:** FAL.ai runs Wan video models on fast cloud GPUs. With FAL_KEY on Mini, we can call their I2V endpoint from the Spark/Mini pipeline. Spark keeps doing local T2I/stills; cloud handles I2V.
- **Caveats:** Verify FAL supports Wan 2.2 14B I2V specifically (not just T2V). Cost per video (~$0.10–$0.50 depending on resolution/frames). Network latency for upload/download.
- **Verdict:** Best **short-term bridge** while fixing Spark local I2V. Use this to unblock Aurore production today.

### 5. Abandon Wan 2.2 for CogVideoX/HunyuanVideo — **NO-GO for now**
- **Effort:** 4–6 hrs setup + re-tuning
- **Risk:** High
- **Reason:** This abandons the existing M4Max Wan 2.2 pipeline that Aurore already relies on. HunyuanVideo has ComfyUI nodes, but its I2V quality differs from Wan 2.2. Re-tuning prompts, LoRAs, and the segment-and-stitch pipeline would take days. Only pursue if Option 1 definitively fails after 2–3 debugging attempts.

---

## Recommended Execution Plan
1. **Immediate (today):** Set up FAL cloud I2V as a bridge. This unblocks production while Spark is debugged.
2. **Parallel (today–tomorrow):** Fix Spark workflow — toggle `fun_or_fl2v_model=False` in `aurore_i2v_22_A14B.json` and re-run. If that fails, check VAE compatibility (try `Wan2_1_VAE_bf16` as fallback, or verify `Wan2_2_VAE` output channels).
3. **Contingency:** If Spark fix fails after 2 days, escalate to Option 5 (evaluate HunyuanVideo I2V nodes) while keeping FAL as primary.
4. **M4Max:** Keep as Draw Things backup. Do NOT attempt ckpt conversion. Fix SSH when convenient (not blocking).

---

## Effort Summary Table
| Option | Effort | Risk | Verdict |
|--------|--------|------|---------|
| 1. Fix Kijai workflow | 2–4 hrs | Medium | **GO — primary** |
| 2. Convert Draw Things ckpt | Days+ | Extreme | **NO-GO** |
| 3. I2V on M4Max | 1–2 hrs | Medium | **GO — backup only** |
| 4. FAL cloud I2V | 2–3 hrs | Low | **GO — bridge** |
| 5. Switch model (Hunyuan) | 4–6 hrs+ | High | **NO-GO — last resort** |
