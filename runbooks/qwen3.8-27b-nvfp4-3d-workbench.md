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

**VERIFIED END-TO-END on the Spark, 2026-09-05** — the full pipeline produced a real textured GLB (`ComfyUI_00001.glb`, 73.6 MB, 502K vertices, 618K faces, PBR + UVs) from a single input image in ~4 min, on a Spark also running the 122B-class live lane. Details below.

```
reference image → TRELLIS.2-4B int8 (ComfyUI :8189)
  → textured PBR GLB → scale + watertight repair (Blender/trimesh) → print
```

**Stack as staged (all pinned):**
- ComfyUI v0.34.0 (commit `8a43c6bd`, master) at `~/ComfyUI`, venv torch 2.12.1+cu130 (sm_120 in arch list), native `nodes_trellis2.py` in core
- Checkpoints (all in `~/ComfyUI/models/`): `diffusion_models/trellis_2_int8_convrot.safetensors` (5.25 GB), `vae/trellis_2_shape_vae_bf16.safetensors` (1.10 GB), `vae/trellis_2_texture_vae_bf16.safetensors` (0.95 GB), `clip_vision/dino_v3_L_naf_fp32.safetensors` (1.21 GB), `geometry_estimation/moge_2_vitl_normal_fp16.safetensors`, `background_removal/birefnet.safetensors`
- venv pins upgraded for v0.34.0: `comfy-aimdo==0.5.2`, `comfyui-frontend-package==1.52.6`, `comfyui-workflow-templates==0.11.55`, `comfyui-workflow-templates-json==0.1.68`, `comfyui-embedded-docs==0.5.11`
- Proven workflow file: `docker/qwen38-flash-next/trellis2-image-to-3d-api-workflow.json` — validated against live `object_info`, fixed for the API's integer output indices, PreviewImage pass-through rewiring, `RemeshMesh` DynamicCombo flat-key format (`sign_mode` + `sign_mode.qef` etc.), and `UnwrapMesh weld_distance=0.001` (unwelded remesh output otherwise produces empty UV chunks)

**Launch (on the Spark):**
```bash
cd ~/ComfyUI && setsid nohup ./venv/bin/python main.py --port 8189 --listen 0.0.0.0 > /tmp/comfy-8189.log 2>&1 < /dev/null &
# queue via POST /prompt with the workflow JSON (client_id oracle-3d-workbench)
```

**⚠️ COEXISTENCE LIMIT — measured:** the TRELLIS.2 texture/UV/bake stage (4K atlas, 8M-vertex remesh) peaks high enough to OOM-kill the vLLM container (observed exit 137 while the 122B-class lane was up). The Mia lane's own watchdog + cgroup did NOT protect it — the GPU/host memory pressure came from ComfyUI outside the container's scope. **Two safe profiles:**
1. **Sequential (default, proven):** run TRELLIS.2 with the LLM lane stopped (stop the LLM via its own launcher `./stop.sh`, run the 3D batch, restart the lane). Recovery cost ~8-10 min weight reload.
2. **Co-resident (experimental):** LLM up + TRELLIS.2 at reduced settings (texture 2048, remesh resolution 512, decimate 300K). Must be memory-tested before trusting. NOT yet validated.

The 27B workbench lane (GMU 0.55) is the intended co-resident partner for profile 2 — that's the headroom it was designed with.

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