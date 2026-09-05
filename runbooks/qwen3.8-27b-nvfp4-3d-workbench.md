# Qwen3.8-27B NVFP4 — 3D-Modelling Workbench Lane

**Status:** staged (recipe + launcher published; NOT live). The live lane remains `qwen3.8-flash-next` (Mia NVFP4) on `:8000` until an explicit Go replaces it.

## Purpose

A dedicated 3D-modelling workbench: Qwen3.8-27B NVFP4 with **vision ON**, **2 lanes**, **450K total context**, and a deliberately large memory headroom so the 3D companion stack (TRELLIS.2 in ComfyUI) runs **on the same Spark at the same time**.

Frozen posture (2026-09-05):

| Setting | Value |
|---|---|
| Model | `unsloth/Qwen3.8-27B-NVFP4` @ `57926baca9a82b4d6906b43f2750d55315f5b10f` |
| Runtime | `vllm` in `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` (proven GB10 image) |
| Served name | `qwen3.8-27b` |
| Port | `:8000` (canonical Spark route) |
| Lanes | **2** (parent + 1 subagent) |
| Context | **225,280 / lane → 450,560 total** |
| KV | bf16 (DSpark would force FLASH_ATTN; kept off) |
| MTP/DSpark | OFF (code-heavy workbench tasks; no spec-decode benefit proven on 27B) |
| Vision | ON — native vision tower in checkpoint; enables render-feedback loops |
| GMU | 0.55 → leaves ~40+ GiB for TRELLIS.2 + ComfyUI |

## The two-route 3D pipeline (why this lane exists)

Research verdict (2026-09-05, 3 subagents, ~170 primary sources): general LLMs are bottom-tier at from-scratch 3D geometry (Qwen3-VL-2B: 0.008/1.0 BenchCAD; 92% invalid at Text2CAD L3), but strong at *editing* parametric code (0.56–0.87) when given a deterministic executor + diagnostics feedback. So:

### Route 1 — functional/parametric parts (the LLM's job)
```
Qwen 27B (this lane, vision ON) writes build123d script
  → khana build/check executes (deterministic OpenCascade kernel)
  → diagnostics.json back into context:
      interferences, clearances, MIN WALL mm, OVERHANG angles,
      volume, bbox, STL/STEP exported
  → LLM reads diagnostics, edits parameters/code, iterates
  → khana render → PNG views → vision-capable LLM reads them
    for shape questions numbers can't answer
```
cad-khana (`uv tool install cad-khana`, Apache-2.0) is installed and **verified working** on the Mac (2026-09-05: L-bracket test part → STL + STEP + mechanism.json + printability.json, min wall 4.0mm, printability ok). Verified script contract: scripts must call `check(assembly, out=...)` and/or `inspect(part, method=FDM(...), out=...)` — a bare `result =` does nothing (silent no-op).

### Route 2 — organic/artistic meshes (TRELLIS.2's job)
```
reference image → TRELLIS.2-4B (ComfyUI on this Spark, int8 5.25GB)
  → textured PBR GLB
  → scale + watertight repair (Blender 3D-Print Toolbox / trimesh)
  → print
```
ComfyUI on the Spark is at v0.33.0 — TRELLIS.2 native nodes need **v0.34.0+**, so ComfyUI must be updated (pull) before use. TRELLIS.2 checkpoints to download (Comfy-Org pack, 18.85GB total): `diffusion_models/trellis_2_int8_convrot.safetensors` (5.25GB), `vae/trellis_2_shape_vae_bf16.safetensors` (1.10GB), `vae/trellis_2_texture_vae_bf16.safetensors` (0.95GB), `clip_vision/dino_v3_vit_l.safetensors` (1.21GB).

**Division of labor:** the LLM never writes triangle soup; TRELLIS.2 never does dimensions. The LLM can also drive ComfyUI via its HTTP workflow API when the agent layer wants image-to-3D in a pipeline.

## Memory budget (121.7GiB unified)

```
vLLM 27B @ GMU 0.55          ~66-70 GB  (weights 22.6 + KV ~15 + overhead)
TRELLIS.2 int8 + ComfyUI    ~10-12 GB
host/SSH/watchdog           ~5 GB
free headroom                ~40+ GB
```

## Launch (explicit Go required)

```bash
# on the Spark, after stopping the live lane (KILL_LIVE=1 = the Go):
bash scripts/switch-to-qwen38-27b-nvfp4-3d-workbench.sh --start
# poll: curl http://127.0.0.1:8000/health
```

The launcher refuses to start if `vllm-fn-tp1`/`qwen38` containers are up unless `KILL_LIVE=1` is set — replacing the healthy live service always requires an explicit Go.

## Gates before promotion (per runbook standards)

1. Health 200 + 2-lane concurrency verified
2. Vision round-trip: image in → correct description (verifies the vision tower actually engages in this container)
3. 100K+ context request completes
4. cad-khana loop end-to-end via this lane: prompt → script → diagnostics → edited script converges
5. TRELLIS.2 GLB renders from a reference image, opens in Blender
6. Both coexist: memory floor ≥ 8GiB during combined load