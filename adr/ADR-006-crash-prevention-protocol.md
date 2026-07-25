# ADR-006: Spark Crash Prevention Protocol

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle
**Priority:** CRITICAL — violating this can require physical power-cycle of the Spark

## Context

The DGX Spark has 121GB unified memory (LPDDR5X, shared between CPU and GPU). There is no separate GPU VRAM — it's all one pool. When GPU memory utilisation is set too high, or when multiple GPU-consuming processes run simultaneously, the kernel OOM killer can fail to contain the situation and the entire box freezes, requiring a physical power-cycle.

### Documented Crash History

**Puzzle-75B MTP — 9 crashes (July 7, 2026)**
- Causes: broken DFlash entrypoint, OOM at 0.85 GMU + MTP, EngineCore silent death with `--enforce-eager`
- Required: physical power-cycle multiple times
- Fix: fank's recipe — no `--enforce-eager`, headless (GNOME stopped), let FlashInfer autotuner cache (15 min first boot), 0.85 GMU only after headless

**FlashInfer autotune OOM with ComfyUI co-location**
- FlashInfer's first-boot autotune spike + ComfyUI/Flux (~17GB) = OOM
- Fix: kill Flux before first boot, use `--enforce-eager` with 0.4 GMU for first boot, warm the cache, then relaunch at 0.65 GMU and restart Flux

**DS4 memory pressure**
- DS4 uses ~98GB of 121GB at 256K context, leaving ~2GB free
- Without cgroup containment, an overrun kills the whole box
- Fix: DS4 serve script uses `systemd-run --user --scope -p MemoryMax=110G -p MemorySwapMax=0` — kills ONLY the engine on overrun, not the OS

### Root Causes (Pattern)

| Root Cause | Mechanism | Prevention |
|---|---|---|
| GMU too high + MTP | KV cache + MTP drafter exceeds memory → kernel OOM → freeze | Start low (0.4-0.65), work up |
| `--enforce-eager` + MTP | EngineCore silent death, no error, just hangs | Don't use enforce-eager with MTP |
| FlashInfer autotune spike | First-boot JIT compilation spikes memory | Kill other GPU processes before first boot |
| GNOME desktop + high GMU | Desktop session + inference = double-dip memory | Go headless (stop GDM) for max GMU |
| No cgroup containment | Kernel OOM killer takes whole box, not just engine | Use systemd-run MemoryMax scope |
| ComfyUI co-location | Steady-state OK but first-boot autotune OOMs | Kill ComfyUI before first boot of any new model |

## Decision

### Mandatory Pre-Flight Checklist (Before ANY Model Launch)

1. **Check what's running** — `ps aux | grep -E "vllm|ollama|comfyui|flux" | grep -v grep`
   - Kill ALL existing inference processes first
   - Kill ComfyUI if doing a first-boot of a new model

2. **Check GPU memory** — `nvidia-smi` (if available) or `free -g`
   - Confirm >90% of 121GB is free before launching

3. **Check FlashInfer cache** — if switching model types, clear it:
   - `rm -rf ~/.cache/flashinfer/*`
   - If first boot of a new model, expect 5-15 min autotune

4. **Verify headless if using high GMU (>0.75)**
   - `sudo systemctl stop gdm && sudo systemctl mask gdm`
   - Desktop session + high GMU = OOM risk

5. **Verify cgroup containment for venv-based launches (DS4)**
   - DS4 script already has `systemd-run --user --scope -p MemoryMax=110G`
   - Docker containers have built-in containment (container OOM kills container, not host)

6. **NEVER clear torch compile cache during model switches**
   - CUDA graphs are durable assets — each model's graphs persist on disk
   - vLLM keys graphs by model + spec config internally — they coexist, no conflict
   - Clearing destroys compiled graphs → DFlash acceptance collapses 80%→8%, speed 80→15 tok/s
   - Switch scripts NEVER run `rm -rf torch_compile_cache`
   - Only valid reasons to clear: corrupted graphs, vLLM version upgrade, spec config change on same model
   - FlashInfer cache IS safe to clear (re-autotunes in ~15 min, doesn't affect DFlash)

### Safe GMU Ranges

| Scenario | Max safe GMU | Notes |
|---|---|---|
| Headless (GNOME stopped) | 0.85 | Maximum, validated by fank |
| Headless + ComfyUI running | 0.65 | Leave room for ComfyUI ~17GB |
| Desktop (GNOME running) | 0.40-0.50 | Desktop + inference sharing memory |
| First boot of new model (autotune) | 0.40 (first), then restart higher | Autotune spikes memory |
| DS4 at 256K | 0.78 | Script default, with cgroup MemoryMax=110G |

### Launch Order for First Boot of Any New Model

1. Kill everything (vllm, ollama, comfyui, flux)
2. Stop GDM if going headless
3. Clear FlashInfer cache
4. Clear torch compile cache if switching spec mode
5. Launch with LOW GMU (0.40-0.50) + `--enforce-eager` for first boot
6. Wait for health check + autotune to complete
7. Stop the server
8. Relaunch with target GMU (0.65-0.85 depending on headless/desktop)
9. Wait for health check
10. Run smoke test
11. Update STATE.md

### Docker vs Venv Containment

| Stack | Containment | Risk to host |
|---|---|---|
| Docker (122B, 35B) | Container OOM kills container only | ✅ Low — host survives |
| Venv (DS4) | `systemd-run --user --scope -p MemoryMax=110G` | ✅ Low — engine killed, host survives |
| Venv without cgroup | None — kernel OOM killer | ❌ HIGH — can freeze entire box |

### SSH OOM Protection (KNOWN GAP — needs manual sudo)

sshd child processes have `oom_score_adj=0` (not protected). If kernel OOM killer fires despite cgroup containment, it could kill SSH sessions, locking us out remotely.

**Fix requires root on Spark** (terminal guard blocks `sudo -S`):
```bash
# Run on Spark as jaita with sudo:
for p in $(pidof sshd); do echo -1000 > /proc/$p/oom_score_adj; done

# Or permanent via systemd:
sudo systemctl edit ssh
# Add:
[Service]
OOMScoreAdjust=-1000
```

**Primary defense is cgroup containment** (MemoryMax=110G kills engine before host OOM). SSH protection is belt-and-suspenders. This must be done manually by James until passwordless sudo is configured.

### Orphaned Scope Cleanup

When DS4 is killed via `kill -9`, the systemd user scope may persist holding memory. The switch scripts should use `systemctl --user abort` on the scope, not just `kill -9`. See ADR-006 update below.

**Symptom:** `free -g` shows 99GB used after kill, no vllm process visible, but `systemctl --user list-units --type=scope` shows a `run-r*.scope` still active.

**Fix:**
```bash
systemctl --user abort run-r<hash>.scope
# Wait 5-10 seconds for memory release
free -g  # confirm memory freed
```

## Consequences

- No model is ever launched without the pre-flight checklist
- First boots always use low GMU + enforce-eager, then restart higher
- FlashInfer cache is cleared when switching model families
- GNOME is stopped before any high-GMU launch
- ComfyUI is killed before any first-boot
- DS4's cgroup containment is never removed
- STATE.md records any crash with full diagnosis