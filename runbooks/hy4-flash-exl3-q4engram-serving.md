# Hy4-Flash EXL3 3.0bpw — Q4-Engram Serving Runbook (spark1, 2026-11-24)

**Status:** SERVING on spark1 :8000 (hy4-flash-exl3), RAM ~97/121G (80%), Q4 engram table on
local SSD with streaming row reads. Output currently garbled — debugging (see Open item).
Measured: end-to-end chat round-trip 7 s.

## The boot that works (the whole recipe)

Container: `ghcr.io/styles01/hy4-exl3-dfly:v1`, heat caps (`bash /home/jaita/heatcap.sh`)
applied BEFORE boot, `echo 3 > /proc/sys/vm/drop_caches` on the HOST before boot.

```
docker run -d --name hy4-serve --network host --ipc host --shm-size 64g --gpus all \
  --memory=115g --memory-swap=115g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/jaita/models/Hy4-flash/exl3-3.0bpw-fp8lin-q4:/models/pack:ro \
  -v /home/jaita/hy4serve/overlay-q4/engram.py:/home/jaita/hy4serve/site-packages/vllm/models/hy_v4_flash/nvidia/engram.py:ro \
  -v /home/jaita/hy4serve/site-packages/vllm/models/hy_v4_flash/nvidia/model.py:/home/jaita/hy4serve/site-packages/vllm/models/hy_v4_flash/nvidia/model.py:ro \
  -v /home/jaita/cudainc:/usr/local/cuda/include:ro \
  -v /tmp/serve.sh:/tmp/serve.sh:ro \
  -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TORCH_CUDA_ARCH_LIST='12.1' \
  --entrypoint bash ghcr.io/styles01/hy4-exl3-dfly:v1 /tmp/serve.sh
```

`/tmp/serve.sh`:
```
exec vllm serve /models/pack \
  --served-model-name hy4-flash-exl3 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser ling3 --reasoning-parser ling3 \
  --engram-config '{"cpu_offload":false,"disk_offload":true,"disk_threads":256}' \
  --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.80 \
  --kv-cache-memory-bytes 6500000000 --enforce-eager
```

Pack: `/home/jaita/models/Hy4-flash/exl3-3.0bpw-fp8lin-q4/` — the fp8lin EXL3 pack with
the 4 engram-table shards replaced by `*-q4engram.safetensors` (54 B/row ggml q4_0,
126.7 GB → 35.6 GB, cosine 0.9963), `config.json` + `engram_q4_embedding: true`,
index pointing the 4 engram tensors at the q4engram files. Also: real tokenizer /
processor files COPIED in (broken symlink = silent tokenizer failure).

## The RAM math, measured (the whole reason this kept OOMing)

| component | host RAM |
|---|---|
| weights mmap'd (63.6 GB pack) | 63.6 GB |
| BF16 engram table reads through page cache | measured 79→118 GB in 45 s |
| KV + graphs + drafter | ~11 GB |
| **Q4-table boot (this recipe)** | **peak ~102G (84%) — never killed** |

Root cause of every OOM: `disk_offload` streams rows through the page cache; on GB10
unified memory that IS the 121 GB pool. Fix = clear-after-use: the Q4 row store
(`overlay-q4/engram.py`) does `preadv` + `posix_fadvise(DONTNEED)` per row (smoke test:
4000 rows → 64 MB cache delta, bounded). NFS reads cache on the SERVER side, so the
table must live on LOCAL NVMe for this to work. RAM guard that saved the box tonight:
single 15 s poll of `free -g`, `docker rm -f hy4-serve` if used > 96% of total.

## Bugs fixed to get here (do not regress these)

1. **Duplicate gate_up (`Duplicate Flash checkpoint weight or shard`)**: prod model.py
   FP8 stash dequant dropped `.shard_id` on dequantized tensors AND keyed the stash by
   name only (gate's dequant used up's scale). Fixed in place
   (`model.py.bak-dupfix` backup; patch script `~/workspace/patch_dupfix.py`).
2. **`Unknown Flash checkpoint weight: ...weight_scale_inv`** under q4 flag: overlay-q4
   model.py was an OLD fork missing the FP8 stash path. Solution: use prod model.py +
   overlay-q4 ENGRAM.PY ONLY (q4 support lives fully in engram.py).
3. **`EXL3 mixed-K load incomplete`**: same fork-mismatch; prod model.py fixes it.
4. **exllamav3 JIT**: (a) missing cusparse.h → bind `/home/jaita/cudainc`
   (copied, not symlinked: container must resolve its own paths) to
   `/usr/local/cuda/include`; (b) ptxas `compute_75` rejection → `-e
   TORCH_CUDA_ARCH_LIST='12.1'`.
5. **`Cannot copy between CPU and CUDA tensors during CUDA graph capture`**: the Q4 row
   streaming is not graph-capture-safe → `--enforce-eager` (graphs = next lever; pinned
   staging buffer is the proper fix).
6. **Crashing at exactly 82%/88% shard load twice**: was the engram BF16 load path, not
   the stated error. Watch `docker logs` for the REAL error; `MemAvailable` alone lies on
   GB10 (use `free -g` used% + driver residency).
7. **Engine ignores host-side file edits** unless each file is bind-mounted into the
   container: the ghcr image runs its INTERNAL site-packages. Every patched
   file MUST be `-v` bound or the fix is a no-op (burned several cycles on this).

## Open item

Output garbled (mixed-language token soup, `temperature=0`). Suspects, in order:
(1) the dupfix dequant applying a wrong block-shape scale to gate_up vs down tensors —
verify one dequantized dense tensor GPU-vs-CPU-reference from shard scale; (2) Q4
54 B/row decode in _DiskRowStore feeding wrong rows/values; (3) engram seed/namespace.
Known-good discriminator: BF16-table NFS boot (same dense weights) produced sane output
last week, so dense weights are probably fine → check the Q4 row decode first.

## Ceilings / targets

- Measured best (BF16 NFS boot, spec-off, prior config): 16.1 t/s single, c3 34.3.
- filicroval community recipe (equivalent hardware): 40 t/s decode @ 256k via
  DirectIO loading + everything-except-table on GPU. DirectIO loader is the next
  big lever after graphs.
- 40/60 target needs: graphs back (pinned staging) → DirectIO loader → spec d3.

## RAM-guard one-liner (run any time a boot is live)

```
while :; do sleep 15; U=$(free -g | awk '/Mem:/{print $3}'); T=$(free -g | awk '/Mem:/{print $2}');
[ $U -gt $((T*96/100)) ] && { docker rm -f hy4-serve; echo KILLED-at-${U}G; break; }; done
```

## TandemEngine (0xbakeer) — reference implementation, adopted/ported (11-24)

Repo cloned: /Volumes/1TBSSDClawd/TandemEngine (AGPL-3.0, Python, GB10-native, updated daily).
He serves OUR EXL3 format (exllamav3 1.6.7 + convert/ patch) on ONE Spark with the exact
architecture we converged on. His hy4-flash numbers: weights stage 58.52 GiB @ 262k ctx,
4 engram tables = 40.92 GB (5-bit values vs our Q4 35.6 GB), 8 lanes full-window turning,
DFly2 drafting integrated WITH the engram lookup, up to 262k ctx. Packs private for now.

Three engine learnings to port into our overlay (confirmed in engine/disk_cache.py):
1. **O_DIRECT row reads with page-aligned buffers** (fs fallback = fadvise DONTNEED) —
   `_open_direct()` pattern; better than our preadv+fadvise-per-row (same effect, less work).
2. **Pinned staging buffer allocated once** for the device copies — likely also the fix
   for our CUDA-graph capture failure (CPU->CUDA copy needs pinned during capture).
3. **2 GiB hot-row LRU cache** in front of the SSD table reads (his model.yaml: "the
   Engram tables, read past the page cache through a 2 GiB hot-row cache") — big SSD
   traffic cut; our row store currently reads every row from NVMe.

Also available: TandemEngine as the alternate/back-to-back engine later — same port 8000
behind an OpenAI-compatible API; our pack may need his convert/ layout (4-table 5-bit
engram files) rather than the vLLM-style in-shard tables.
