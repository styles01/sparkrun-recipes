#!/usr/bin/env python3
"""spark-prefix-align: let vLLM prefix caching coexist with the DFlash drafter on the
hybrid GDN+mamba+MoE target (DGX Spark, AEON 0.23.0+aeon.sm121a.dflash).

v0.25.1 port: path changed from site-packages to dist-packages.
NOTE: 0.25.1 may have this fixed natively. Only apply if prefix caching + DFlash crashes.

Root cause (validated on GB10, 2026-06-28): the DFlash drafter's attention layers carry
a ~2x larger KV page than the target, so page-size unification scales the TARGET's mamba
+ attention blocks UP (2240 -> 4480) to match the drafter's page, while the drafter group
stays at 2240. `resolve_kv_cache_block_sizes` then sees a MambaSpec whose block_size
(4480) != cache_config.block_size (2240) and takes its back-off branch, forcing
hash_block_size = LCM = 4480. The drafter group (block 2240) is not divisible by 4480, so
HybridKVCacheCoordinator.__init__ aborts with "block_size must be divisible by
hash_block_size".

But the GCD (2240) divides EVERY group (4480 % 2240 == 0, 2240 % 2240 == 0) and is the
correct finest-common hash granularity. The back-off only exists to disable fine hashing
for *non-align* mamba (its own comment says "mamba_cache_mode != align"); the
`block_size != cache_config.block_size` test is a buggy proxy that also fires when an
ALIGN-mode mamba block was merely scaled up by unification. This patch fixes the proxy:
back off only when mamba is genuinely non-align. In align mode we fall through to the GCD
path, which is exactly what hash_block_size (finer than block_size, merged up per group)
was designed for (vLLM #29143).
"""
import pathlib, sys

P = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
src = P.read_text()

if "spark-prefix-align" in src:
    print("[patch_prefix_align] already patched", flush=True)
    sys.exit(0)

OLD = '''    # Mamba groups with block_size != cache_config.block_size
    # (mamba_cache_mode != "align") break divisibility; back off to the
    # scheduler block size.
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size'''

NEW = '''    # spark-prefix-align: back off to the scheduler block size ONLY when mamba is
    # genuinely non-align. A mamba group whose block_size != cache_config.block_size
    # while mamba_cache_mode == "align" just means page-size unification scaled the
    # block up (e.g. to match a larger DFlash drafter page); the GCD below still
    # divides every group, so fine hashing remains valid. (Upstream's proxy test
    # `block_size != cache_config.block_size` wrongly tripped this branch for the
    # hybrid+DFlash+prefix-caching case, forcing hash_block_size=LCM and breaking
    # HybridKVCacheCoordinator's divisibility assert.)
    if getattr(cache_config, "mamba_cache_mode", "none") != "align" and any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size'''

if OLD not in src:
    print("[patch_prefix_align] ERROR: back-off anchor not found — vLLM source differs", flush=True)
    sys.exit(1)
src = src.replace(OLD, NEW, 1)

# Confirmation log right before the final return so the serve log proves the fix engaged.
RET_OLD = '''    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid hash_block_size={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by hash_block_size. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size'''
RET_NEW = '''    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid hash_block_size={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by hash_block_size. "
            f"Got group block sizes={group_block_sizes}."
        )
    print("SPARK_PREFIX_ALIGN resolved scheduler=%r hash=%r (mode=%r blocks=%r)" % (
        scheduler_block_size, hash_block_size,
        getattr(cache_config, "mamba_cache_mode", "none"), group_block_sizes),
        flush=True)
    return scheduler_block_size, hash_block_size'''
if RET_OLD in src:
    src = src.replace(RET_OLD, RET_NEW, 1)
else:
    print("[patch_prefix_align] WARN: confirmation-log anchor not found (non-fatal)", flush=True)

P.write_text(src)
print("[patch_prefix_align] patched OK (align-aware back-off)", flush=True)