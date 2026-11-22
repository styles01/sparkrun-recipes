# Hy4-flash EXL3 3.0-bit — DFly adaptive spec lane (production runbook)

**Status: ✅ production (best measured Hy4 decode on this estate), 2026-11-22**
**Engine:** vLLM `vllm/vllm-openai:hy4-flash` + `vllm_exl3` overlay (this repo's `overlay/`)
**Pack:** [`aitamedia/Hy4-flash-EXL3-3.0bpw`](https://huggingface.co/aitamedia/Hy4-flash-EXL3-3.0bpw)
→ local `exl3-3.0bpw-fp8lin` (FP8-native dense build; recipe `recipes/hy4-flash-exl3-vllm-tp1.yaml`)
**Served:** `:8000`, model `hy4-flash-exl3`, 262K ctx, c3 concurrency, engram streamed from disk

## Numbers (GB10, single Spark, TP1, 2026-11-22, 3× per prompt, T=0)

| lane | pipeline | shortcode | JSON | mean |
|---|---:|---:|---:|---:|
| no-spec (0 depth) | 15.0 | 15.5 | 15.3 | **15.3 t/s** |
| DFly fixed d3 | 19.6 | 22.0 | 22.5 | **21.4 t/s** |
| **DFly adaptive (shipped)** | 19.3 | 21.9 | 22.3 | **21.2–21.4 t/s (+38.5% vs no-spec)** |

Aggregate c3 with spec is queued for measurement; single-stream is the headline.

## What made spec work (the three fixes, in order)

1. **Drafter scale-loader bug (blocked everything).** Hy4's draft linears are fp8-native with
   compressed-tensors-style params; the overlay's `dfly.py::load_weights` swallowed the
   checkpoint's `.weight_scale_inv` keys (drafted with unit scales → acceptance exactly 1.00).
   Fix: let fp8-typed draft linears receive the checkpoint block-scale params as runtime params
   (39 tensors land; `overlay/vllm/models/hy_v4/nvidia/dfly.py`).
2. **Depth is a property of the model, not a knob.** Measured sweep (d1..d7, ms/round, acceptance):
   d1 131.7/1.65 · d2 173.8/1.90 · **d3 116.0/1.96** · d4 244.0/2.24 · d7 305.6/2.30.
   Only d3 beats the no-spec wall (62.1 ms). The card's default (5) is the *worst* productive depth
   here — do not serve the card config on GB10.
3. **Adaptive depth launcher (dsv41 pattern).** `dfly_adaptive_launcher.py` (overlay root) wraps
   the vLLM entrypoint, reads `VLLM_HY4_DFLY_ROUND_MS` (JSON `[depth, ms/round, acceptance]` rows),
   picks the depth maximizing acceptance/ms at boot, rewrites argv before engine spawn. Falls back
   to the measured spark1 table if env unset. Log line at boot: `[hy4-dfly-adaptive] choose depth=3`.

## Serve (production)

```bash
ssh jaita@192.168.2.185
docker rm -f hy4-serve 2>/dev/null
bash /home/jaita/hy4_tp1_spec_adaptive.sh   # adaptive DFly lane on :8000 (~9-10 min load)
# verify:
docker logs hy4-serve 2>&1 | grep 'hy4-dfly-adaptive'   # expect "choose depth=3"
curl -s http://localhost:8000/v1/models | head -c 40
python3 /tmp/bench3.py   # expect ~21 t/s mean across 3 prompts
```

Rollback to no-spec: `bash /home/jaita/hy4_tp1.sh` (15.3 t/s but zero spec risk).

## Pitfalls (each cost a boot once)

- Launcher argv continuation chain: one orphaned `\` silently drops later flags (`cat -A` after edits).
- `hy4_tp1_spec2.sh` boots **depth 5** not 3 — use the `_d3`/`_adaptive` launchers.
- Spawned EngineCore ignores PYTHONPATH; the overlay is force-loaded by a `.pth` written into image
  dist-packages (launcher does this every boot).
- FP8-blockwise kernel on sm_121 = cutlass only; kernel-config must pin it.
- Never pack conversions in-place on the NFS source; build on spark2, serve ro.

## Progress history (compressed)

| milestone | t/s | how |
|---|---:|---|
| first TP1 boot | 4.5 | eager, BF16-dense pack, garble bug unsolved |
| graphs (FULL_AND_PIECEWISE) | 9.4 | +IDX scheduler cache, zz_oracle.pth overlay force-load |
| loader root-cause fix #2 (~2785× scale error) | 13.7 | dq2 dequant pack |
| **FP8-native dense pack** | **16.1** | convert_pack chain (+18%) |
| **DFly scale fix + d3 + adaptive launcher** | **21.4** | this runbook |

## Next (to 40/60)

The round wall at d3 is ~116 ms; the target forward dominates it. Levers queued, in order:
MoE coop GEMM tuning (`VLLM_EXL3_MOE_TEMP_ROWS`, `VLLM_EXL3_FAT_THRESHOLD`, `VLLM_EXL3_MOE_KERNEL=native`
explicit), static-shape CUDA graph over the expert gather (kills host dispatch), and the indexer
FP32→bf16 slice. Each ~1-2 boots. TP2 (two Sparks) is the structural escape hatch if the single-Spark
wall bottoms out — the dsv41 lane's drafter is TP2 for exactly this reason.

## Artifacts

- `overlay/` — full vLLM overlay incl. `dfly.py` fix, `dfly_adaptive.py`, `dfly_adaptive_launcher.py`
- `scripts/launch/hy4-tp1.sh` (no-spec), `hy4-tp1-spec-d3.sh`, `hy4_tp1_spec_adaptive.sh` (estate path)
- Depth-sweep JSON: estate launcher env or the README table above
- Companion TF CUDA port (experimental, separate runbook in sparkrun-recipes)
