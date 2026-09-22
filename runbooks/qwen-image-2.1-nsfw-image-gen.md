# Qwen-Image-2.1 NSFW Image Lane (ComfyUI, GB10)

**Status:** WEIGHTS INSTALLED + NODES VERIFIED. First generation pending lane-down window (one-heavy-engine rule).
**Released:** 2026-09-20 (QwenLM). License: **Qwen Research License — NON-COMMERCIAL.** Local personal use OK; do NOT use outputs commercially.

## Why this model (James: "its native nsfw — you know what that means")
No prompt sanitizer / no built-in refusal in the encoder — unrestricted prompt content,
like PinkCherry H3 Aurore for video. Native 2K, RGBA, 10-image editing, 7B DiT +
Qwen3-VL 8B text encoder.

## Installed (2026-09-22, ~/ComfyUI/models/)
| File | Size | Dir |
|---|---|---|
| qwen_image_2.1_int8_convrot.safetensors | 7.3GB | diffusion_models/ |
| qwen3vl_8b_int8_convrot.safetensors | 9.4GB | text_encoders/ |
| qwen_image_2.1_vae_bf16.safetensors | 0.7GB | vae/ |

int8_convrot = the format proven on GB10 (Gannotti/SMF pins, 23/23 tests).
bf16 DiT (14.2GB) + bf16 TE (17.5GB) = ~33GB peak, available for later if lane down.
Alternative TE: qwen3.5_9b_pe_i2i/t2i int8_convrot (9.5GB each) for the PE/i2i variants.
w4a8 TE (6.3GB) = lowest-VRAM option.

## ComfyUI (port 8189, canonical launch)
```bash
ssh jaita@192.168.2.185 'cd ~/ComfyUI && setsid nohup ./venv/bin/python main.py --port 8189 --listen 0.0.0.0 > /tmp/comfyui.log 2>&1 < /dev/null &'
```
- Updated 2026-09-22 to b33e2b55 (Sep 22) — has `TextEncodeQwenImage21` + `QwenImage21Cache`.
- After ANY `git pull`: `./venv/bin/pip install -r requirements.txt` (new dep comfy_aimdo
  broke startup until this was run).
- Server start (no model load) is safe while the LLM lane is up. RUNNING a gen is NOT.

## Workflow (API format)
`workflows/qwen-image-2.1-t2i-api.json` — 8 nodes:
UNETLoader(int8) → KSampler(euler/simple/25/cfg 1.0 — template default, distilled) ←
TextEncodeQwenImage21(pos+neg in ONE node) ← CLIPLoader(type=qwen_image) ←
EmptyLatentImage(1024²) → VAEDecode ← VAELoader(2.1 VAE) → SaveImage.
Official template: Comfy-Org/workflow_templates `image_qwen_image_2_1_t2i.json`.

## Queue a gen
```bash
scp workflows/qwen-image-2.1-t2i-api.json jaita@192.168.2.185:/tmp/
curl -X POST http://192.168.2.185:8189/prompt -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(python3 -c "import json;print(json.dumps(json.load(open('/tmp/qwen-image-2.1-t2i-api.json'))))"), \"client_id\": \"oracle\"}"
# poll /history/<prompt_id>, fetch via /view
```

## HARD CONSTRAINTS
- **One heavy engine per GB10.** QI-2.1 int8 peak ≈ 20-25GB (DiT 7.3 + TE 9.4 + VAE 0.7
  + activations). EXL3 lane holds ~94GB → 27GB available = DOES NOT FIT concurrently.
  Kill the LLM lane first (see daily-driver runbook), run gens, then restore lane.
- Downloads (disk I/O) are fine while the lane runs — only model LOADS conflict.
- Memory math: 121GB total; EXL3 ~94GB; QI-2.1 int8 ~20-25GB peak → never together.

## Open items
- [ ] First gen test (needs lane-down window)
- [ ] Timing measurement (expect ~20-40s/image at 1024² on GB10 from community numbers)
- [ ] Image-edit variant (10-image refs) + qwen3.5_9b PE encoder test
- [ ] NSFW verification of unrestricted prompting (v1 Qwen-Image had no refusal; 2.1
      reports same — verify with an actual prompt)
- [ ] SparkDash: ComfyUI panel? (GPU panel already covers util)