#!/usr/bin/env python3
"""spark-dflash-unify (CORRECTED): let the DFlash drafter's attention KV spec unify
with the hybrid GDN+mamba target's *padded* page size WITHOUT corrupting the drafter.

v0.25.1 port: path changed from site-packages to dist-packages.
NOTE: 0.25.1 may have this fixed natively (page_size_padded + indexes_kv_by_block_stride).
Only apply if the assert fires.

Root cause (AEON 0.23.0+aeon.sm121a.dflash): with --mamba-block-size set, vLLM's own
hybrid alignment makes target mamba page == target attention page by PADDING the mamba
page (e.g. "+0.54%"). That padded value becomes max_page_size. The drafter's attention
page is smaller; unify_kv_cache_spec_page_size scales its block_size by
ratio = max_page_size // layer_page_size, but because max_page_size is a *padded* (non
block-size-linear) number, the scaled page lands just under it and
`assert new_spec.page_size_bytes == max_page_size` fires.

This corrected patch mirrors what get_kv_cache_groups already does for
HiddenStateCacheSpec layers: keep the SCALED block_size AND pad the <1% remainder.
Strides stay correct (block_size is properly scaled); only the page tail is padded.
"""
import pathlib, sys

P = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
src = P.read_text()

if "spark-dflash-unify" in src:
    print("[patch_unify2] already patched", flush=True)
    sys.exit(0)

OLD = '''            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec'''

NEW = '''            new_spec = replace(layer_spec, block_size=new_block_size)
            if new_spec.page_size_bytes != max_page_size:
                # spark-dflash-unify: max_page_size is a *padded* hybrid page; the
                # scaled attention page lands just under it. Pad the remainder while
                # KEEPING the scaled block_size, exactly as get_kv_cache_groups does
                # for HiddenStateCacheSpec. (The old patch dropped the scaling ->
                # mis-strided the DFlash drafter -> accept len stuck ~1.47.)
                new_spec = replace(
                    layer_spec,
                    block_size=new_block_size,
                    page_size_padded=max_page_size,
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec'''

if OLD not in src:
    print("[patch_unify2] ERROR: target block not found — vLLM source differs", flush=True)
    sys.exit(1)

P.write_text(src.replace(OLD, NEW))
print("[patch_unify2] patched OK (scaled-block + pad-remainder)", flush=True)