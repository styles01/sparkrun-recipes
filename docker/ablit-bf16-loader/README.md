# ablit-bf16-loader — v52rc plugin delta

The one-file engine change that lets the Sudosu dsv41 line serve EXL3 packs with
**mixed bf16/EXL3 modules** — i.e. the [dealignai 1017-UNCENSORED abliteration pack](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED),
which leaves 54 modules (layers 10–36 × `attn.wo_b` ×27 + `ffn.shared_experts.w2` ×27) as
plain bf16 `.weight` tensors inside an otherwise-EXL3 trellis pack.

- `config.py` — the **v52rc** version of `vllm_exl3_moe/config.py` (533 lines, drop-in copy of
  the file inside `dsv41-serve:v52rc` at
  `/usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/config.py`).
- `config-v52rc.patch` — unified diff vs the v51 image
  (`ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v2`): +87 lines. **This is the entire
  v51→v52rc delta** — no other file in the engine changed (plan.py/envs.py byte-identical
  mod timestamps).

## How it works

Per-module-instance probe, no config fields, no name lists:

1. Read the checkpoint's safetensors index (shard for the module's key).
2. Read that shard's header (raw `os.pread`, 1MB window — buffered-open broke under the
   plugin import context with "read of closed file") for dtype + shape.
3. If the module instance is a plain 2-D `.weight` in bf16/fp16/f32 with **no trellis of the
   module left**, route it to vLLM's stock unquantized path (`UnquantizedLinearMethod`);
   every other module keeps trellis EXL3.

Serving a bf16 module as bf16 is exact — the tensor is the truth. Row-parallel TP slicing of
the out-major layout is natively the checkpoint layout, so TP2 needs no repack. Nothing in the
weights tree is modified.

## Rebuild from v51

```dockerfile
FROM ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v2
COPY config.py /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/config.py
```

(`dsv41-serve:v52rc` = exactly this, built for sm_121/torch-cu130 — the base is already
GB10-built; the config.py copy needs no rebuild of any CUDA extension.)