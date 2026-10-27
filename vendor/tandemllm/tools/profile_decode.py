"""Step time and effective bandwidth for the autoregressive decode step.

The step reads every weight once, so the only number that matters is how close the measured step
comes to (bytes read) / (board bandwidth). Everything else the engine does -- the recurrent state,
the KV cache, the norms, the sampling -- is small by construction, and this tool says by how much.

Reports, per step: wall time, the implied effective bandwidth against the weight bytes alone, and
the same for the three groups the step decomposes into.
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
from engine.model import Qwen38Engine  # noqa: E402

PEAK_GBPS = 273.0


STEP_MEDIAN_TOKENS = 30          # `step_median`'s warm-up plus its timed steps


def room_for(a) -> int:
    """The KV the run writes: the prompt, the baseline steps, and every sweep setting's steps.

    Both sweeps advance the same position. The two-stream sweep was left out of this sum until
    2026-09-23, and its third setting ran off the end of the buffer (guard caught it).
    """
    n = len([x for x in a.sweep_two_stream.split(",") if x.strip()]) if a.sweep_two_stream else 0
    return (a.prompt_len + a.steps + a.warmup + 64 + (6 * 40 if a.sweep_fused else 0)
            + n * STEP_MEDIAN_TOKENS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--breakdown", action="store_true", help="per-group timing with cuda events")
    ap.add_argument("--sweep-fused", action="store_true",
                    help="step time for each combination of the fused kernels, one process")
    ap.add_argument("--sweep-two-stream", default="",
                    help="step time at these settings of QWEN38_TWO_STREAM in one process, in this "
                         "order: e.g. off,on,off")
    a = ap.parse_args()

    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {time.time() - t0:.1f}s  {w.report()}")
    b = w.decode_step_bytes(cfg.num_hidden_layers)
    print(f"[bytes] layers {b['layers_GB']:.3f} GB + lm_head {b['lm_head_GB']:.3f} GB "
          f"= {b['total_GB']:.3f} GB per step")
    print(f"[floor] {b['total_GB'] / PEAK_GBPS * 1e3:.1f} ms/step "
          f"= {PEAK_GBPS / b['total_GB']:.2f} tok/s at {PEAK_GBPS:.0f} GB/s")

    room = room_for(a)
    eng = Qwen38Engine(cfg, w, max_len=room, device=a.device)
    ids = torch.randint(1000, 100000, (a.prompt_len,), device=a.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        logits = eng.forward(ids, start=0, last_only=True)
    torch.cuda.synchronize()
    pf = time.time() - t0
    print(f"[prefill] {a.prompt_len} tokens in {pf * 1e3:.0f} ms = {a.prompt_len / pf:.0f} tok/s")

    pos = a.prompt_len
    tok = logits[0, -1].argmax()
    with torch.no_grad():
        for _ in range(a.warmup):
            logits = eng.forward(tok.view(1), start=pos, last_only=True)
            tok = logits[0, -1].argmax()
            pos += 1
        torch.cuda.synchronize()
        times = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            logits = eng.forward(tok.view(1), start=pos, last_only=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            tok = logits[0, -1].argmax()
            pos += 1
    times.sort()
    med = times[len(times) // 2]
    base_med = med
    print(f"[decode] median {med * 1e3:.2f} ms/step  ({1 / med:.2f} tok/s)  "
          f"min {times[0] * 1e3:.2f}  max {times[-1] * 1e3:.2f}")
    print(f"[decode] effective {b['total_GB'] / med:.1f} GB/s "
          f"= {b['total_GB'] / med / PEAK_GBPS * 100:.1f} % of {PEAK_GBPS:.0f} GB/s")
    print(f"[decode] overhead above the byte floor: "
          f"{(med - b['total_GB'] / PEAK_GBPS) * 1e3:.1f} ms")

    if a.sweep_fused or a.sweep_two_stream:
        # One process, one set of weights, the flags flipped between runs. A kernel measured in its
        # own process is not a budget -- the 13:40 entry has that mistake in it -- so the only
        # number reported here is the engine's own median step.
        from engine.model import FUSED

        def step_median(n=STEP_MEDIAN_TOKENS - 6, warm=6):
            nonlocal pos, tok
            with torch.no_grad():
                for _ in range(warm):
                    lg = eng.forward(tok.view(1), start=pos, last_only=True)
                    tok = lg[0, -1].argmax()
                    pos += 1
                torch.cuda.synchronize()
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    lg = eng.forward(tok.view(1), start=pos, last_only=True)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                    tok = lg[0, -1].argmax()
                    pos += 1
            ts.sort()
            return ts[len(ts) // 2]

        if a.sweep_two_stream:
            import engine.model as M
            print("\n[two-stream] step time and effective bandwidth, same process, same weights")
            for setting in a.sweep_two_stream.split(","):
                M.TWO_STREAM = setting.strip() == "on"
                ms = step_median() * 1e3
                print(f"    two streams {setting.strip():3s}  {ms:7.2f} ms  "
                      f"({1000 / ms:5.2f} tok/s)  {b['total_GB'] / (ms / 1e3):6.1f} GB/s  "
                      f"{(base_med * 1e3 - ms):+6.2f} ms vs the first reading of this process")
            M.TWO_STREAM = False

        names = ["norm", "gdn", "head", "attn", "gdnpre"]
        # `gdnpre` needs `gdn`: it hands the recurrence kernel a sixteen-head key side and the
        # reference path cannot read that, so the two are swept as a pair rather than alone.
        combos = (([[]] + [[n] for n in names if n != "gdnpre"]
                   + [["gdn", "gdnpre"], [n for n in names if n != "gdnpre"],
                      names]) if a.sweep_fused else [])
        if combos:
            print("\n[fused] step time by kernel set, same process, same weights")
        for combo in combos:
            for n in names:
                FUSED[n] = n in combo
            try:
                ms = step_median() * 1e3
            except Exception as exc:
                print(f"    {'+'.join(combo) or 'none':28s}  failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            print(f"    {'+'.join(combo) or 'none':28s}  {ms:7.2f} ms  "
                  f"({1000 / ms:5.2f} tok/s)  {(base_med * 1e3 - ms):+6.2f} ms vs reference")
        if combos:
            for n in names:
                FUSED[n] = False

    if a.breakdown:
        import torch.nn.functional as F
        from engine.model import rms_norm
        h = torch.randn(1, 1, cfg.hidden_size, device=a.device, dtype=torch.bfloat16)
        groups = {}

        def timeit(fn, n=20):
            for _ in range(4):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n

        gdn_l = cfg.linear_layers[0]
        att_l = cfg.attention_layers[0]
        eng.state.primed = True
        with torch.no_grad():
            groups["one GDN layer mixer"] = timeit(
                lambda: eng.linear_attention(h, f"layers.{gdn_l}", gdn_l, True))
            p = torch.arange(pos, pos + 1, device=a.device)
            groups["one attention layer mixer"] = timeit(
                lambda: eng.attention(h, f"layers.{att_l}", att_l, pos, p))
            groups["one MLP"] = timeit(lambda: eng.mlp(h, f"layers.{gdn_l}"))
            from engine.model import head_logits
            head = w.norm("lm_head.weight")
            groups["lm_head"] = timeit(lambda: head_logits(h, head))
            groups["one rms_norm"] = timeit(
                lambda: rms_norm(h, w.norm(f"layers.{gdn_l}.input_layernorm.weight"),
                                 cfg.rms_norm_eps))
        n_gdn, n_att = len(cfg.linear_layers), len(cfg.attention_layers)
        print("\n[breakdown] projected contribution to one step")
        total = 0.0
        for name, per, count in (("GDN mixers", groups["one GDN layer mixer"], n_gdn),
                                 ("attention mixers", groups["one attention layer mixer"], n_att),
                                 ("MLPs", groups["one MLP"], n_gdn + n_att),
                                 ("rms norms", groups["one rms_norm"], 2 * cfg.num_hidden_layers + 1),
                                 ("lm_head", groups["lm_head"], 1)):
            ms = per * count * 1e3
            total += ms
            print(f"    {name:20s} {per * 1e3:7.3f} ms x {count:3d} = {ms:8.2f} ms")
        print(f"    {'sum':20s} {'':7} {'':5} {total:8.2f} ms   "
              f"(measured step {med * 1e3:.2f} ms)")


if __name__ == "__main__":
    main()
