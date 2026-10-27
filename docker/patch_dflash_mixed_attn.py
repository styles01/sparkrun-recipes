#!/usr/bin/env python3
"""spark-dflash-mixed-attn: allow DFlash drafters with mixed sliding/full
attention layer_types to load on vLLM 0.25.1.

v0.25.1 added a NotImplementedError in _resolve_layer_attention when
layer_types contains a mix of "sliding_attention" and "full_attention".
This blocks the z-lab/Qwen3.5-122B-A10B-DFlash drafter which has 5 SWA + 1 full.

The DFlash drafter only generates draft tokens — the target model does the
real attention. So we can safely downgrade mixed layer types to all-full
(non-causal) attention for the drafter. The drafter's output quality is
already approximate by design.

This patch replaces the NotImplementedError with a warning + fallback to
treating all layers as full_attention (non-causal).
"""
import pathlib, sys

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py"
)
src = P.read_text()

if "spark-dflash-mixed-attn" in src:
    print("[patch_dflash_mixed_attn] already patched", flush=True)
    sys.exit(0)

OLD = '''    if any_sliding and not all_sliding:
            # Mixed sliding/full attention needs per-layer causal metadata and
            # multiple KV-cache groups, which DFlash does not yet support.
            raise NotImplementedError(
                "DFlash does not yet support mixed sliding/full attention via "
                "layer_types; see "
                "https://github.com/vllm-project/vllm/issues/40898."
            )'''

NEW = '''    if any_sliding and not all_sliding:
            # spark-dflash-mixed-attn: DFlash drafters with mixed sliding/full
            # attention (e.g. z-lab/Qwen3.5-122B-A10B-DFlash: 5 SWA + 1 full)
            # are blocked by upstream's NotImplementedError. The drafter only
            # generates draft tokens — the target model does the real attention.
            # Downgrade to all-full (non-causal) for the drafter.
            import logging as _logging
            _logging.getLogger("vllm").warning(
                "spark-dflash-mixed-attn: DFlash drafter has mixed layer_types "
                "(%d sliding, %d full) — downgrading to all full_attention "
                "(non-causal). Draft quality may be slightly affected.",
                num_sliding, len(layer_types) - num_sliding,
            )
            layer_types = None
            any_sliding = False
            all_sliding = False'''

if OLD not in src:
    print("[patch_dflash_mixed_attn] ERROR: target block not found", flush=True)
    sys.exit(1)

P.write_text(src.replace(OLD, NEW))
print("[patch_dflash_mixed_attn] patched OK (mixed SWA+full -> all full)", flush=True)