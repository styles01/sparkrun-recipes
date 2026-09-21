# PinkCherry H3 Aurore Runbook (NSFW Video Gen — MiniMax H3 fine-tune)

**Recipe ID:** `pinkcherry-h3-aurore`
**Status:** ✅ Verified end-to-end 2026-09-07 — validation clip rendered (1.6s, 39 frames, 832×1216, h264+aac)
**Author:** Oracle (verified on James's Spark, GB10)

---

## What This Is

PinkCherry H3 v0.5-alpha is a NSFW-capable fine-tune of MiniMax H3 (first-last-frame/audio-video DiT).
It is a **drop-in replacement** for the base H3 diffusion model — same architecture, same VAEs,
same text encoder, same native ComfyUI graph. Everything else in the H3 stack is stock.

## Files on Spark

| Component | File | Size |
|---|---|---|
| Diffusion model | `~/ComfyUI/models/diffusion_models/PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors` | 21 GB |
| Text encoder | `~/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~17 GB (NVFP4 AWQ) |
| Video VAE | `~/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors` | ~1 GB |
| Audio VAE | `~/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors` | ~1 GB |
| Turbo LoRA (optional) | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | — |
| Base model (for A/B) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21 GB |

## Memory Posture (HARD RULE — exclusive occupancy)

- H3 video gen needs **~35-45 GiB** on the unified pool
- The live qwen3.8-flash-next lane holds ~101 GiB
- **These DO NOT coexist.** Same class as the TRELLIS.2 bake OOM (2026-09-06).
- Procedure (verified): check `free -g` first → stop lane via upstream `./stop.sh`
  (never raw pkill on the vLLM container) → run ComfyUI → render → stop ComfyUI
  (`kill` the `main.py --port 8189` PID; ComfyUI has no upstream stop script) →
  wait for MemAvailable to return → relaunch lane → verify HTTP 200.
- Observed: lane stop → 105 GiB free; ComfyUI stop → 101 GiB free. Both directions clean.

## Workflow Graph (API format)

Published: `workflows/pinkcherry-aurore-i2v-api.json` — extracted from ComfyUI v0.34's
native `video_minimax_h3_i2v` template subgraph (NOT hand-guessed), converted to API
format, with three changes:

1. `UNETLoader` → `PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors`
2. `LoadImage` → `aurore_juggernaut.png` (the approved Aurore portrait, already in ComfyUI input)
3. Prompt → Aurore character description (below)

Node chain: LoadImage → MiniMaxH3ImageToVideo (clip+vae+prompt+first_frame) →
[conditioning, latent] → RandomNoise + KSamplerSelect(res_multistep) +
BasicScheduler(simple, 20 steps) + BasicGuider(model, conditioning) →
SamplerCustomAdvanced → output + denoised latents → VAEDecode (video VAE) +
VAEDecodeAudio (audio VAE) → CreateVideo(24 fps) → SaveVideo.

## Aurore Character (approved content, from Venus's pipeline docs)

> Photorealistic woman, wavy light brown hair, blue-grey eyes, natural skin with visible
> pores, white cream blouse with subtle floral embroidery. Soft diffused studio lighting.

Approved still (Juggernaut XL v9, 20 steps, CFG 5.0, DPM++ 2M Karras, 832×1216):
`aurore_juggernaut_00001_.png` — used as the I2V first frame.

## Verified Validation Clip

- `prompt_id dcf408c9-597e-4937-a1f9-6a4b05ef5037` — status success, zero node errors
- Output: `aurore/pinkcherry_test_00001_.mp4` — 1.625 s, 39 frames @ 24 fps, 832×1216, h264 + AAC audio, 681 KB
- First-run generation took ~3 min wall (model load + 20 steps × 39 frames)

## Full-Run Parameters (from the native template, H3 "max" settings)

- Duration: `length` frames at 24 fps on the 17k+5 grid (124 ≈ 5 s; trained range 124-362)
- 448×640 for fast tests; 832×1216 for portrait; the Aurore reference used 448×640 × 49 frames
- Turbo LoRA (8-step) is available in the template via a switch node — halves step count at some quality cost
- Seeds: `RandomNoise.noise_seed` — change per run for variation

## Gotchas Learned

- ComfyUI templates are **UI format with subgraphs** — the API graph must be extracted from
  `definitions.subgraphs[0]` (node ids, links, widget values), then converted. The
  `api_minimax_h3_max_turbo_i2v.json` template is the CLOUD-API variant — wrong path for local.
- The H3 latent is audio-video: `SamplerCustomAdvanced` output slot 0 = video frames, slot 1 = audio
  → decode each with its own VAE → `CreateVideo` merges them.
- `MiniMaxH3ImageToVideo.length` snaps to the 17k+5 grid (39 = 2×17+5 ✓).
- ComfyUI holds ~3 GiB idle; the render peak pushed total used to ~45 GiB — comfortable
  solo but never with the lane up.

## Launch Sequence (copy-paste)

```bash
# 1. CHECK FIRST (hard rule)
ssh jaita@192.168.2.185 'free -g; docker ps --format "{{.Names}}"'

# 2. Stop the lane (upstream script — never raw pkill)
ssh jaita@192.168.2.185 'cd ~/src/qwen38-flash-next-mia-262k && ./stop.sh'

# 3. Start ComfyUI
ssh jaita@192.168.2.185 'cd ~/ComfyUI && setsid nohup ./venv/bin/python main.py --port 8189 --listen 0.0.0.0 > /tmp/comfyui.log 2>&1 < /dev/null &'
# wait for :8189 to return 200

# 4. Queue the workflow
scp workflows/pinkcherry-aurore-i2v-api.json jaita@192.168.2.185:/tmp/
# POST to http://192.168.2.185:8189/prompt with {"prompt": <workflow>, "client_id": "oracle"}
# poll /history/<prompt_id> for completion, fetch via /view

# 5. Restore the lane
ssh jaita@192.168.2.185 'kill $(pgrep -f "main.py --port 8189")'   # stop ComfyUI
ssh jaita@192.168.2.185 'nohup bash /tmp/switch-to-qwen38-flash-next-mia-nvfp4-220k.sh --start &'
# verify :8000/health == 200
```

## License Note

PinkCherry H3 v0.5-alpha is a community fine-tune; license terms NOT verified.
Local personal use only until terms are checked. Alpha quality — expect drift vs base H3.

---

## EXTENSION: Multi-take Motion-Context chaining (planned, NOT yet verified)

**Source:** smfworks/h3-longform-capture (MIT, Gannotti's team) — docs archived in
`runbooks/gannotti-h3-chaining/` (IMAGE-STILLS, FRAMEWORK, HOW-TO, SOURCES, REVIEW).
Their results: 262.846 s / 6285 f / 1344×768 music video from chained H3 takes (2026-09-16).

### Core concepts (from their verified docs)

- **Sheet ≠ plate.** Sheet = character/prop bible (hero view + locked keywords + forbidden
  list). Plate = the ACTUAL first frame of a hop-1/cut/fadeblack (same identity, this
  location, this grade, this camera size). Never use a stretched sheet as a plate.
- **Three joins:** `continue` (Motion-Context hop, trim 22, concat `-c copy`) /
  `cut` (new plate, hard cut, no hold) / `fadeblack` (new take + 8-frame dip).
- **Hop-1 = 243 f (10.125 s) @ 24 fps, 1344×768; hop 2+ after trim = 221 f / 9.209 s.**
- **Never re-feed stills on hop 2+** — that fights the Motion-Context latent. Hop-1 must
  `MiniMaxH3MotionContextSaveLatent` with a unique filename; hop 2+ loads that exact latent.
- **Camera = one English verb** (H3 wants type+amplitude+speed as one action). Long chains
  accumulate texture; cap extend-takes.
- **One heavy engine per GB10:** never co-locate Qwen-Image-2.1 stills and H3 clips on one
  box. We have ONE Spark → stills first (lane down or on the other lane), then clips.

### Adaptation for our single-Spark setup

Gannotti runs 2 Sparks (stills on one, clips on the other). We have 1:
1. Stop LLM lane → run ComfyUI → generate ALL sheets + plates first (Qwen-Image-2.1 INT8
   ConvRot: euler/simple, cfg 1, AuraFlow shift 3.1, 25 steps, 1344×768 = 24 s each)
2. Generate all hop-1 clips (I2VA with plates) + SaveLatent per take
3. Watch identity at each planned cut BEFORE hopping
4. Generate hop 2+ from saved latents, trim 22 frames, concat `-c copy` within takes
5. Stop ComfyUI → restore LLM lane
- Their abort threshold: ≥85-86°C die temp. Our Qwen-Image-2.1 smoke test comes first
  (single image) before any chaining attempt.

### Our gap vs their setup

- We use PinkCherry H3 (fine-tune, drop-in) — chaining should work identically (same graph)
- Qwen-Image-2.1 NOT yet tested on our Spark (smoke test queued — Gannotti's INT8 ConvRot
  pins: ComfyUI 0.36.0, DiT 7.26GB + TE 9.35GB, euler/simple cfg1 shift 3.1)
- Motion-Context custom nodes NOT yet installed (ComfyUI-H3-Motion-Context by NikoDemon80)
