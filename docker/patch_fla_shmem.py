#!/usr/bin/env python3
"""spark-fla-shmem: let the FLA (flash-linear-attention / GDN) Triton kernels use their
BIG tiles on sm121 (GB10 / DGX Spark).

v0.25.1 port: path changed from site-packages to dist-packages.

The FLA Backend gate (vllm/model_executor/layers/fla/ops/utils.py, identical in SGLang)
picks big tiles only if device_max_shared_mem >= Backend.DEFAULT (102400 = 100 KiB):
    cumsum.py : BS_LIST  = [32,64] if check_shared_mem() else [16,32]
    chunk_o.py: BKV_LIST = [64,128] if check_shared_mem() else [32,64]
sm121 reports max_shared_mem = 101376 (99 KiB), JUST below 102400 -> check returns False
-> small tiles -> slower GDN/linear-attention (the per-token hot path). But 101376 is
EXACTLY Backend.ADA, and RTX 4090 (ADA, same 99 KiB) runs the big tiles fine, so they
provably fit in 99 KiB. Lower DEFAULT to 101376 so sm121 (and ADA) pass the gate.
"""
import pathlib, sys

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/utils.py"
)
src = P.read_text()
if "spark-fla-shmem" in src:
    print("[patch_fla_shmem] already patched", flush=True)
    sys.exit(0)

OLD = "    DEFAULT = 102400  # Default"
NEW = "    DEFAULT = 101376  # spark-fla-shmem: 102400->101376 so sm121 GB10 (99 KiB) uses big GDN tiles (already fit on ADA's identical 99 KiB)"

if OLD not in src:
    print("[patch_fla_shmem] ERROR: target line not found — FLA source differs", flush=True)
    sys.exit(1)

P.write_text(src.replace(OLD, NEW))
print("[patch_fla_shmem] patched OK (DEFAULT 102400 -> 101376; big GDN tiles on sm121)", flush=True)