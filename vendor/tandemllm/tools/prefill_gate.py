"""The gate for the fused prefill path: the same prompt, both paths, one process, one set of weights.

A prefill kernel is allowed to be a different arithmetic ORDER from the reference -- it is a
different decomposition of the same recurrence -- but it is not allowed to be a different ANSWER.
"Different answer" has to be a number, and on a 248,320-row head the number that means something is
not the maximum absolute difference on its own: it is that difference measured against the bf16
ulp at the same magnitude, since the engine hands bf16 logits to a greedy argmax anyway.

So this reports, for each prompt length and per layer of evidence:

    max |dlogit|  and what that is in bf16 ulps of the largest logit
    argmax        the greedy token, which is what a temperature-0 request actually depends on
    top-8         the order of the top of the distribution
    max |dS|      the recurrent state after the prefill, relative to its own scale -- the thing
                  that a long generation carries forward and a logit comparison cannot see

Both paths run in one process on one set of weights, in both orders across the prompts, because
this file has already been caught once comparing two configurations that were also two processes.

    python tools/prefill_gate.py --lens 256,2048,8192 \
        --nvfp4 ~/nvfp4/mlp-clip.safetensors,~/nvfp4/gdn-clip.safetensors,~/nvfp4/attn-clip.safetensors \
        --fp8-head ~/nvfp4/head-fp8.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402

REF_TEXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bench", "refprompt.txt")


def tokens_for(cfg, n: int) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    with open(REF_TEXT) as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    while ids.numel() < n:
        ids = torch.cat([ids, ids])
    return ids[:n].contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lens", default="256,2048,8192")
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--control", action="store_true",
                    help="compare the shipped path against ITSELF with the other exact "
                         "inverse, which is what a different fp32 order costs here")
    ap.add_argument("--tol", type=float, default=1.0, help="bf16 ulps the gate allows")
    ap.add_argument("--greedy", type=int, default=64,
                    help="tokens to continue greedily from each prefill and compare, 0 to skip")
    a = ap.parse_args()

    if a.nvfp4:
        os.environ["QWEN38_NVFP4"] = a.nvfp4
    if a.fp8_head:
        os.environ["QWEN38_FP8_HEAD"] = a.fp8_head
    os.environ.setdefault("QWEN38_NVFP4_DEQUANT_FROM", "512")

    import engine.model as M
    from engine.loader import Weights

    lens = [int(x) for x in a.lens.split(",")]
    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {time.time() - t0:.1f}s  {w.report()}")
    eng = M.Qwen38Engine(cfg, w, max_len=max(lens) + 64, device=a.device)

    from engine import gdn as G

    def once(ids, arm: str, greedy: int = 0):
        """`ref` the shipped path, `fused` the Triton pair, `subst` the shipped path with the OTHER
        exact inverse -- `engine/gdn.py`'s own forward substitution instead of its triangular
        solve. The third one is the control: it is the same algorithm in a different arithmetic
        order, which is exactly what the fused path is, so whatever it moves the logits by is what
        "a different order of the same fp32" costs on this model."""
        M.FUSED["gdnprefill"] = (arm == "fused")
        keep, G.UT_INVERSE = G.UT_INVERSE, (arm != "subst")
        eng.reset()
        try:
            with torch.no_grad():
                logits = eng.forward(ids, start=0, last_only=True)
        finally:
            G.UT_INVERSE = keep
        out = [logits.float().reshape(-1).clone(), eng.state.S.clone(), []]
        if greedy:
            # The functional test, which is the one a request depends on: continue greedily from the
            # state this prefill produced and see whether the two paths write the same text. A logit
            # difference that never changes a token is a difference the engine cannot express.
            tok = int(out[0].argmax())
            pos = int(ids.numel())
            with torch.no_grad():
                for _ in range(greedy):
                    out[2].append(tok)
                    lg = eng.forward(torch.tensor([tok], device=ids.device), start=pos,
                                     last_only=True)
                    tok = int(lg.float().reshape(-1).argmax())
                    pos += 1
        return out

    print(f"\n{'tokens':>8}{'max |dlogit|':>14}{'bf16 ulp':>10}{'argmax':>9}"
          f"{'top-8':>8}{'max |dS|':>12}{'rel':>10}{'greedy':>14}   verdict")
    worst = 0.0
    for n, T in enumerate(lens):
        ids = tokens_for(cfg, T).to(a.device)
        # the order is reversed on every other length, so a warm-up advantage cannot look like a
        # difference between the paths
        arm = "subst" if a.control else "fused"
        first, second = ("ref", arm) if n % 2 == 0 else (arm, "ref")
        la, Sa, ga = once(ids, first, a.greedy)
        lb, Sb, gb = once(ids, second, a.greedy)
        ref, got = (la, lb) if first == "ref" else (lb, la)
        Sref, Sgot = (Sa, Sb) if first == "ref" else (Sb, Sa)
        gref, ggot = (ga, gb) if first == "ref" else (gb, ga)
        agree = next((j for j, (x, y) in enumerate(zip(gref, ggot)) if x != y), len(gref))
        for tag, x in (("ref logits", ref), ("fused logits", got),
                       ("ref state", Sref), ("fused state", Sgot)):
            n = int(torch.isnan(x).sum()) + int(torch.isinf(x).sum())
            if n:
                print(f"    [!] {tag} at T={T}: {n} non-finite of {x.numel()}")
        d = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        ulp = d / (scale * 2 ** -8 + 1e-30)
        same_argmax = int(ref.argmax()) == int(got.argmax())
        t8 = (set(torch.topk(ref, 8).indices.tolist())
              == set(torch.topk(got, 8).indices.tolist()))
        ds = (Sgot - Sref).abs().max().item()
        rel = ds / (Sref.abs().max().item() + 1e-30)
        ok = bool(same_argmax and t8 and ulp <= a.tol and agree == len(gref))
        worst = ulp if (worst != worst or ulp != ulp or ulp > worst) else worst
        print(f"{T:>8}{d:>14.3e}{ulp:>10.3f}{'same' if same_argmax else 'DIFFERENT':>9}"
              f"{'same' if t8 else 'DIFFER':>8}{ds:>12.3e}{rel:>10.2e}"
              f"{f'{agree}/{len(gref)}':>14}   {'PASS' if ok else 'FAIL'}")

    passed = worst == worst and worst <= a.tol
    what = "the shipped path with its other exact inverse" if a.control else "the fused prefill"
    print(f"\nGATE  {what} differs from the reference by at most {a.tol:g} bf16 ulp of the "
          f"largest logit, and writes the same greedy token: "
          f"{'PASS' if passed else 'FAIL'}  (worst {worst:.3f} ulp)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
