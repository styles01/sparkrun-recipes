# Spark ComfyUI I2V — Model Mismatch Brief
**Date:** 2026-07-08 | **Target:** larryspark (DGX Spark GB10)

## Executive Summary
The `Wan2_2-I2V-A14B-HIGH_bf16.safetensors` (27GB) you downloaded **IS a valid I2V model** (inspected: `patch_embedding.weight` shape `[5120,36,1,2,2]`, `model_type=i2v`, `dim=5120` = 14B). The runtime error — `patch_embedding expects 36 channels but gets 68` — is **not** a fundamental model incompatibility. It is a **VAE + node pipeline mismatch**. The example workflows reference a `_KJ` fp8_scaled variant that Kijai never uploaded publicly; the equivalent files exist on the Comfy-Org repo.

## Root Cause Analysis

| Component | What You Have | What the Workflow Expects |
|---|---|---|
| Diffusion model | `Wan2_2-I2V-A14B-HIGH_bf16.safetensors` | `Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors` |
| VAE | `Wan2_2_VAE_bf16.safetensors` (z_dim=48) | `wan_2.1_vae.safetensors` or `Wan2_1_VAE_bf16.safetensors` (z_dim=16) |
| I2V node | Likely ComfyUI core `WanImageToVideo` | Kijai `WanVideoImageToVideoEncode` |

**The 68-channel number:** If ComfyUI core `WanImageToVideo` is used with the Wan 2.2 VAE (48 z_dim), the native node may concatenate image conditioning (≈20 ch) → 48+20=68 channels fed to `patch_embedding`, which expects 36 (16 latent + 20 conditioning). **Wan 2.2 I2V models require the Wan 2.1 VAE (16 z_dim), NOT the Wan 2.2 VAE.**

## Three Paths Forward (Ranked)

### PATH A — RECOMMENDED: Download fp8_scaled models from Comfy-Org (~14GB each)
These are the exact models Kijai’s example workflows expect, just under Comfy-Org naming.

**Files to download:**
```bash
# High-noise (HNE) base
wget -O ~/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

# Low-noise (LNE) refiner
wget -O ~/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# Optional: 4-step Lightning LoRAs
wget -O ~/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
wget -O ~/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# Text encoder (if you want fp8_scaled; your existing umt5_xxl_fp16.safetensors works too)
wget -O ~/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
```

**Workflow settings:**
- `WanVideoModelLoader`: model = `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, base_precision=`fp16_fast`, quantization=`fp8_e4m3fn_scaled`
- `WanVideoVAELoader`: model_name = `Wan2_1_VAE_bf16.safetensors` (your existing file is correct)
- Use Kijai’s `WanVideoImageToVideoEncode` node, **not** ComfyUI core `WanImageToVideo`
- Connect start_image → `WanVideoImageToVideoEncode.start_image` → `WanVideoSampler.image_embeds`

**Why this is best:**
- fp8_scaled = ~14GB vs bf16 ~27GB → faster load, less VRAM
- Matches example workflows exactly
- Kijai’s loader auto-detects fp8_scaled quantization (verified in `nodes_model_loading.py:1193`)

### PATH B — Use existing bf16 model (no new download)
You can make the 27GB bf16 file work today with two fixes:

1. **VAE fix:** Use `Wan2_1_VAE_bf16.safetensors` (z_dim=16), **never** `Wan2_2_VAE_bf16.safetensors` (z_dim=48) for I2V.
2. **Node fix:** Use Kijai’s `WanVideoImageToVideoEncode` node. In `WanVideoModelLoader`, set:
   - model = `Wan2_2-I2V-A14B-HIGH_bf16.safetensors`
   - base_precision = `bf16`
   - quantization = `disabled`

**Caveat:** The example workflow JSON hardcodes the fp8_scaled_KJ filenames. You must either edit the JSON or rebuild the workflow manually in the UI.

### PATH C — Abandon Kijai for native ComfyUI or another pack
**Not recommended.** Kijai’s `ComfyUI-WanVideoWrapper` is the only mature node pack supporting Wan 2.2 I2V on ComfyUI today. Native ComfyUI has basic WanVideo support but lacks the 2.2 I2V high/low noise expert system, Lightning LoRAs, and TeaCache optimizations. On Spark (121GB VRAM), Kijai nodes are the right choice.

## Draw Things Comparison (M4Max)
Aurore’s working Draw Things model (`wan_v2.2_a14b_lne_i2v_q6p_svd.ckpt`) is a completely different format:
- Draw Things uses Apple-Silicon-optimized `.ckpt` with SVDQuant Q6P
- It runs through Draw Things’ native pipeline, not ComfyUI nodes
- That model CANNOT be used in ComfyUI — different quantization, different weight format
- The success on M4Max proves the **Wan 2.2 14B I2V architecture works**, but does not provide a cross-platform model file

## Concrete Next Steps
1. **Download the two fp8_scaled diffusion models from Comfy-Org** (~28GB total, half the size of your current bf16 file)
2. **Keep** `Wan2_1_VAE_bf16.safetensors` and `umt5_xxl_fp16.safetensors` (they are correct)
3. **Load** `wanvideo_2_2_I2V_A14B_example_WIP.json` from Kijai’s example workflows, update the model dropdowns to the newly downloaded filenames
4. **Restart ComfyUI** after model downloads (ensure no vLLM models loaded in VRAM first: `nvidia-smi`)
5. **If you keep the bf16 file**, manually wire Kijai nodes — do not use ComfyUI core `WanImageToVideo`

## File Inventory on Spark (Verified)
```
~/ComfyUI/models/diffusion_models/
  Wan2_2-I2V-A14B-HIGH_bf16.safetensors   (27GB — valid but suboptimal)

~/ComfyUI/models/vae/
  Wan2_1_VAE_bf16.safetensors             (253MB — CORRECT for I2V)
  Wan2_2_VAE_bf16.safetensors             (1.4GB — WRONG for I2V, keep for T2V)

~/ComfyUI/models/text_encoders/
  umt5_xxl_fp16.safetensors               (works)
  umt5-xxl-enc-bf16.safetensors           (works)

~/ComfyUI/models/loras/
  (empty — download Lightning LoRAs if you want 4-step)
```
