"""Where a prefill pass spends its time, at the three lengths the bench row cares about.

Decode is a weight read and nothing else; prefill is not. One pass over 8,192 tokens reads the same
19.4 GB it reads for one token -- 71 ms at the board's peak -- and does 2 x 27e9 x 8192 = 442 TFLOP
of arithmetic on top. So the decode intuition does not transfer, and neither does the decode tiling:
`pick_config` has a separate prefill bucket for exactly this reason.

The tool times, per group, at M = the prefill length:

    one MLP            gate/up/down at M rows
    one GDN mixer      the chunked delta rule over the whole sequence, with its convolution
    one attention      projections, rope, SDPA, the output gate
    lm_head            one row, because prefill asks for `last_only`

and reports the end-to-end pass beside the sum, so the difference is launch overhead and the
residual stream. `--gdn-chunk` sweeps the chunked form's blocking, `--attn` compares the
materialised mask against `is_causal`.

    python tools/profile_prefill.py --lens 256,2048,8192
    python tools/profile_prefill.py --lens 8192 --gdn-chunk 64,128,256
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine, rms_norm  # noqa: E402

PEAK_GBPS = 273.0


def timed(fn, n: int = 3, warm: int = 1) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lens", default="256,2048,8192")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--gdn-chunk", default=None, help="sweep the chunked form's blocking")
    ap.add_argument("--fused-gdn", default=None,
                    help="sweep the fused prefill kernels: off,on. One process, one set of weights, "
                         "one allocation history, and the order reversed per length -- the phase-4 "
                         "gdn-mm sweep is the reason this is not two runs of the tool")
    ap.add_argument("--gdn-mm", default=None,
                    help="sweep the chunked form's matmul precision: fp32,tf32,bf16. One process, "
                         "one set of weights, one allocation history -- the comparison this "
                         "question needs")
    a = ap.parse_args()

    lens = [int(x) for x in a.lens.split(",")]
    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {time.time() - t0:.1f}s  {w.report()}")
    b = w.decode_step_bytes(cfg.num_hidden_layers)
    print(f"[bytes] {b['total_GB']:.3f} GB read once per pass, whatever the length "
          f"= {b['total_GB'] / PEAK_GBPS * 1e3:.1f} ms at {PEAK_GBPS:.0f} GB/s")
    eng = Qwen38Engine(cfg, w, max_len=max(lens) + 64, device=a.device)

    for T in lens:
        ids = torch.randint(1000, 100000, (T,), device=a.device)

        def one():
            eng.reset()
            return eng.forward(ids, start=0, last_only=True)

        sec = timed(one, n=a.reps)
        print(f"\n[prefill] {T:5d} tokens  {sec * 1e3:8.1f} ms  {T / sec:7.1f} tok/s")

        if a.fused_gdn:
            import engine.model as M
            keep = M.FUSED["gdnprefill"]
            modes = a.fused_gdn.split(",")
            for mode in (modes if lens.index(T) % 2 == 0 else modes[::-1]):
                M.FUSED["gdnprefill"] = (mode == "on")
                s = timed(one, n=a.reps)
                print(f"    fused gdn {mode:>3}   {s * 1e3:8.1f} ms  {T / s:7.1f} tok/s")
            M.FUSED["gdnprefill"] = keep

        if a.gdn_chunk:
            import engine.gdn as gdnmod
            orig = gdnmod.chunk_gated_delta_rule
            for cs in [int(x) for x in a.gdn_chunk.split(",")]:
                def patched(*args, chunk_size=64, _cs=cs, **kw):
                    return orig(*args, chunk_size=_cs, **kw)
                gdnmod.chunk_gated_delta_rule = patched
                import engine.model as mm
                mm.gdn.chunk_gated_delta_rule = patched
                s = timed(one, n=a.reps)
                print(f"    gdn chunk {cs:4d}   {s * 1e3:8.1f} ms  {T / s:7.1f} tok/s")
            gdnmod.chunk_gated_delta_rule = orig
            mm.gdn.chunk_gated_delta_rule = orig

        if a.gdn_mm:
            import engine.gdn as gdnmod
            keep = gdnmod.PREFILL_MM
            for mode in a.gdn_mm.split(","):
                gdnmod.PREFILL_MM = mode
                s_mm = timed(one, n=a.reps)
                eng.state.primed = False
                per = timed(lambda: eng.linear_attention(
                    torch.randn(1, T, cfg.hidden_size, device=a.device,
                                dtype=torch.bfloat16) * 0.02,
                    f"layers.{cfg.linear_layers[0]}", cfg.linear_layers[0], False), n=a.reps)
                print(f"    gdn mm {mode:>5}   {s_mm * 1e3:8.1f} ms  {T / s_mm:7.1f} tok/s"
                      f"   one GDN mixer {per * 1e3:8.3f} ms")
            gdnmod.PREFILL_MM = keep

        if not a.breakdown:
            continue
        eng.reset()
        with torch.no_grad():
            eng.forward(ids, start=0, last_only=True)
        h = torch.randn(1, T, cfg.hidden_size, device=a.device, dtype=torch.bfloat16) * 0.02
        gdn_l = cfg.linear_layers[0]
        att_l = cfg.attention_layers[0]
        pos = torch.arange(0, T, device=a.device)
        groups = {}
        with torch.no_grad():
            groups["one MLP"] = timed(lambda: eng.mlp(h, f"layers.{gdn_l}"), n=a.reps)
            eng.state.primed = False
            groups["one GDN mixer"] = timed(
                lambda: eng.linear_attention(h, f"layers.{gdn_l}", gdn_l, False), n=a.reps)
            groups["one attention"] = timed(
                lambda: eng.attention(h, f"layers.{att_l}", att_l, 0, pos), n=a.reps)
            groups["one rms_norm"] = timed(
                lambda: rms_norm(h, w.norm(f"layers.{gdn_l}.input_layernorm.weight"),
                                 cfg.rms_norm_eps), n=a.reps)
        n_gdn, n_att = len(cfg.linear_layers), len(cfg.attention_layers)
        total = 0.0
        for name, per, count in (("GDN mixers", groups["one GDN mixer"], n_gdn),
                                 ("attention mixers", groups["one attention"], n_att),
                                 ("MLPs", groups["one MLP"], n_gdn + n_att),
                                 ("rms norms", groups["one rms_norm"],
                                  2 * cfg.num_hidden_layers + 1)):
            ms = per * count * 1e3
            total += ms
            print(f"    {name:20s} {per * 1e3:8.3f} ms x {count:3d} = {ms:9.2f} ms")
        print(f"    {'sum':20s} {'':8} {'':5} {total:9.2f} ms   "
              f"(measured pass {sec * 1e3:.1f} ms)")


if __name__ == "__main__":
    main()
