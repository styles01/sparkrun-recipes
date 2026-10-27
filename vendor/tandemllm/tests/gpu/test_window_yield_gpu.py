"""On the board (SRV-321): a layer-major prefill window that hands the GPU to another slot between
its layers is bit-identical, and so is the other slot's work.

One process, one copy of the weights, two engines: A prefills a prompt of random tokens through
engine/cache.py `prefill` (chunks of 1,024, the layer-major windows); B stands for the decoding
slots and runs plain forwards of its own (8 and 512 rows, a decode's shape and a prefill chunk's,
which takes the EXL3 slab path). The run with `engine.model.WINDOW_YIELD` set calls one of B's
forwards at every layer boundary of A's windows, as the server's turn hand-over does; the
reference runs A alone and then B alone. Checked with `torch.equal`: A's last-row logits, every
KV row, S and the convolution tails; every logit of every B forward and B's final S and conv. And
the yield ran between layers (one call a layer boundary of each window).

Run on the box, the box alone (stop the engine first), with the served environment:
    set -a; . ~/tandem-live/ops/serve.env; . ~/tandem-live/ops/profiles/exl3-4.5.env; set +a
    python tests/gpu/test_window_yield_gpu.py [--len 24576]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import engine.model as M  # noqa: E402
from engine import cache  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402


def _S(eng):
    f = getattr(eng, "committed_state", None)
    return f() if f is not None else eng.state.S


def _eq(name, x, y, bad):
    ok = torch.equal(x, y)
    if not ok:
        d = (x.float() - y.float()).abs().max().item()
        bad.append(f"{name}: max|d| {d:.3e}")
    return ok


class Other:
    """B's work, one forward a call: 8 rows, then 512, alternating, from row 0 on."""

    def __init__(self, eng, steps: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.eng, self.pos, self.out = eng, 0, []
        self.toks = [torch.randint(1000, 100000, (8 if k % 2 == 0 else 512,), generator=g).cuda()
                     for k in range(steps)]

    def step(self) -> None:
        if len(self.out) >= len(self.toks):
            return
        t = self.toks[len(self.out)]
        with torch.no_grad():
            lg = self.eng.forward(t, start=self.pos)
        self.out.append(lg.float().clone())
        self.pos += t.numel()

    def rest(self) -> None:
        while len(self.out) < len(self.toks):
            self.step()

    def state(self):
        return _S(self.eng).clone(), self.eng.state.conv.clone()


def prefill_A(A, ids):
    A.reset()
    with torch.no_grad():
        lg = cache.prefill(A, None, ids, "cuda", chunk=1024)[0].float().clone()
    torch.cuda.synchronize()
    L = A.kv.length
    return lg, A.kv.k[:, :, :, :L].clone(), A.kv.v[:, :, :, :L].clone(), _S(A).clone(), A.state.conv.clone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=24576)
    a = ap.parse_args()
    n = a.len
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=os.environ.get("NV") or None,
                fp8_head=os.environ.get("HEAD") or None)
    A = Qwen38Engine(cfg, w, max_len=n + 4096, paged=False)
    steps = 2 * cfg.num_hidden_layers + 8
    B = Qwen38Engine(cfg, w, max_len=steps * 512 + 1024, paged=False)
    g = torch.Generator().manual_seed(n + 1)
    ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
    bad: list = []
    # the reference: A alone, then B alone
    t0 = time.time()
    ref = prefill_A(A, ids)
    t_ref = time.time() - t0
    B.reset()
    ob = Other(B, steps, seed=7)
    ob.rest()
    ref_b, ref_bs = ob.out, ob.state()
    # interleaved: one of B's forwards at every layer boundary of A's windows
    B.reset()
    ob = Other(B, steps, seed=7)
    calls = [0]

    def yield_():
        calls[0] += 1
        ob.step()
    M.WINDOW_YIELD = yield_
    try:
        t0 = time.time()
        got = prefill_A(A, ids)
        t_got = time.time() - t0
    finally:
        M.WINDOW_YIELD = None
    during = len(ob.out)
    ob.rest()
    for j, nm in enumerate(("logits", "k", "v", "S", "conv")):
        _eq(f"A {n} {nm}", ref[j], got[j], bad)
    for k, (x, y) in enumerate(zip(ref_b, ob.out)):
        _eq(f"B forward {k} ({ob.toks[k].numel()} rows) logits", x, y, bad)
    bs = ob.state()
    _eq("B S", ref_bs[0], bs[0], bad)
    _eq("B conv", ref_bs[1], bs[1], bad)
    print(f"[yield] A {n:,} tokens: alone {t_ref:.1f} s, with B between layers {t_got:.1f} s; "
          f"{calls[0]} yields, {during} of B's {steps} forwards ran inside A's windows", flush=True)
    if calls[0] < cfg.num_hidden_layers - 1:
        bad.append(f"only {calls[0]} yields: the window did not offer the turn between its layers")
    if bad:
        print("IDENTITY FAIL:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("IDENTITY PASS: A's logits, KV rows, S and conv and every B logit, S and conv bit-identical")


if __name__ == "__main__":
    main()
