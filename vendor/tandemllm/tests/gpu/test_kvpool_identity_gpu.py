"""On the board (SRV-307): the shared KV pool is bit-identical to the static KV buffers.

One process, one copy of the weights, two engines: A with the static `torch.zeros` KV, B with its
KV (and its drafters' context KV) reserved as addresses and committed page by page from the pool.
For prompts of 8k, 32k and 128k random tokens:

  * the prefill (engine/cache.py `prefill`, the served grid and windows): last-row logits, every
    KV row written, the recurrent state S and the convolution tails -- `torch.equal`;
  * then the decode shapes at that position: chain verifies of 2..16 rows with rollbacks, and
    branching tree verifies of 3..16 nodes with path commits, through the verify graphs when the
    environment turns them on (`QWEN38_VERIFY_GRAPH=1`, the served config) -- every logit and the
    final state, `torch.equal`;
  * B's pool commits only what was written (committed rows within one page and the margin of the
    KV length) and gives it back on `trim(0)`.

And with the served drafter stack (the length router's two DFlash2 arms over each engine): a greedy
tree decode of 256 tokens on a real 8k text prompt yields the same tokens and the same block count
on A and B.

Run on the box, the box alone, with the served environment:
    set -a; . ~/tandem-live/ops/serve.env; . ~/tandem-live/ops/profiles/exl3-4.5.env; set +a
    python tests/gpu/test_kvpool_identity_gpu.py [--lens 8192,32768,131072]
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import cache, kvpool  # noqa: E402
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


def _trees(seed):
    rng = random.Random(seed)
    out = []
    for n in range(3, 17):
        parents, path = [-1], [0]
        for i in range(1, n):
            path = path[:rng.randint(1, len(path))]
            parents.append(path[-1])
            path.append(i)
        if parents != [-1] + list(range(n - 1)):
            out.append(parents)
    return out


def identity_at(A, B, n: int, seed: int, bad: list) -> None:
    from engine.tree import DraftTree
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
    with torch.no_grad():
        outs = []
        for eng in (A, B):
            eng.reset()
            t0 = time.time()
            lg = cache.prefill(eng, None, ids, "cuda", chunk=1024)[0]
            torch.cuda.synchronize()
            outs.append((lg.float().clone(), time.time() - t0))
        _eq(f"{n} prefill logits", outs[0][0], outs[1][0], bad)
        L = A.kv.length
        assert L == B.kv.length == n, (L, B.kv.length, n)
        for t in ("k", "v"):
            _eq(f"{n} kv.{t} rows", getattr(A.kv, t)[:, :, :, :L], getattr(B.kv, t)[:, :, :, :L], bad)
        _eq(f"{n} S", _S(A), _S(B), bad)
        _eq(f"{n} conv", A.state.conv, B.state.conv, bad)
        print(f"  {n:>7} prefill A {outs[0][1]:.1f}s B {outs[1][1]:.1f}s; B committed rows "
              f"{B.kv.pages.rows:,} ({B.kv.pages.committed_bytes() / kvpool.GiB:.2f} GiB)", flush=True)
        assert B.kv.pages.rows <= n + kvpool.POOL.margin_rows + 4096, B.kv.pages.rows
        # the decode shapes
        pos = n
        for T in range(2, 17):
            blk = torch.randint(1000, 100000, (T,), generator=g).cuda()
            keep = 1 + (3 * T + 5) % T
            la = A.forward_block(blk, start=pos).float().clone()
            A.rollback_to(keep)
            lb = B.forward_block(blk, start=pos).float().clone()
            B.rollback_to(keep)
            _eq(f"{n} chain{T}", la, lb, bad)
            pos += keep
        for j, parents in enumerate(_trees(seed)):
            toks = torch.randint(1000, 100000, (len(parents),), generator=g).tolist()
            path = DraftTree(tokens=toks, parents=parents).path(len(parents) - 1)
            res = []
            for eng in (A, B):
                lg = eng.forward_tree(torch.tensor(toks).cuda(), parents, start=pos)
                res.append(lg.float().clone())
                eng.commit_tree(path)
            _eq(f"{n} tree{j}", res[0], res[1], bad)
            pos += len(path)
        _eq(f"{n} S after decode", _S(A), _S(B), bad)
        L = A.kv.length
        for t in ("k", "v"):
            _eq(f"{n} kv.{t} rows after decode", getattr(A.kv, t)[:, :, :, :L],
                getattr(B.kv, t)[:, :, :, :L], bad)


def spec_identity(A, B, cfg, bad: list, new: int = 256) -> None:
    from engine.spec import generate_spec_tree
    from tools.verify_spec import tree_router
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "engine", "*.py"))))
    msgs = [{"role": "user", "content": text[:28000] + "\n\nList the five most important functions "
                                                      "above and what each does."}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False)
    ids = torch.tensor(tok(rendered, add_special_tokens=False).input_ids, device="cuda")
    rp = os.path.expanduser("~/tandem-live")
    ns = types.SimpleNamespace(dflash2_ckpt=os.path.join(rp, "train/ft-b8-v2"), dflash2_path="greedy",
                               max_len=A.max_len, lenrouter=os.path.join(rp, "train/ft-b16"),
                               corpus="", drop_idle=True,
                               deep=int(os.environ.get("QWEN38_DEEP", "0") or 0),
                               deep_after=int(os.environ.get("QWEN38_DEEP_AFTER", "2") or 2),
                               len_fixed=16)
    got = []
    for eng in (A, B):
        lr = tree_router(eng, ns, narrow=False)
        eng.reset()
        t0 = time.time()
        with torch.no_grad():
            toks, st = generate_spec_tree(eng, ids, new, lr, 15)
        got.append((toks, st.blocks, time.time() - t0))
        for d in (getattr(lr, "small", None), getattr(lr, "big", None), lr):
            if d is not None and hasattr(d, "detach"):
                try:
                    d.detach()
                except Exception:                                  # noqa: BLE001
                    pass
    (ta, ba, sa), (tb, bb, sb) = got
    print(f"  spec tree 8k: A {len(ta)} tokens in {ba} blocks ({sa:.1f}s), B {len(tb)} in {bb} "
          f"({sb:.1f}s); draft KV paged: "
          f"{sum(1 for b in B.kv.pages.bufs if b.name.startswith('draft'))} buffers", flush=True)
    if ta != tb:
        i = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        bad.append(f"spec tree tokens differ at {i}")
    if ba != bb:
        bad.append(f"spec tree blocks {ba} vs {bb} (the drafts differ)")


def window_cap_identity(A, n: int, bad: list) -> None:
    """SRV-309: a prefill with the layer-major window capped at 2,048 rows (what a slot gets while
    another decodes) equals the uncapped one bit for bit."""
    g = torch.Generator().manual_seed(n + 1)
    ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
    outs = []
    with torch.no_grad():
        for cap in (None, 2048, 1024):
            cache.WINDOW_CAP = (lambda c=cap: c) if cap is not None else None
            A.reset()
            t0 = time.time()
            lg = cache.prefill(A, None, ids, "cuda", chunk=1024)[0].float().clone()
            torch.cuda.synchronize()
            L = A.kv.length
            outs.append((cap, lg, A.kv.k[:, :, :, :L].clone(), A.kv.v[:, :, :, :L].clone(),
                         _S(A).clone(), A.state.conv.clone(), time.time() - t0))
    cache.WINDOW_CAP = None
    ref = outs[0]
    for o in outs[1:]:
        for j, nm in ((1, "logits"), (2, "k"), (3, "v"), (4, "S"), (5, "conv")):
            _eq(f"window cap {o[0]} {n} {nm}", ref[j], o[j], bad)
    print("  window cap prefill s: " + ", ".join(f"{o[0]}: {o[6]:.1f}" for o in outs), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="8192,32768,131072")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--window-cap", type=int, default=0,
                    help="also check the SRV-309 window cap at this prompt length (0: skip)")
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",") if x]
    M = max(lens) + 4096
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=os.environ.get("NV") or None,
                fp8_head=os.environ.get("HEAD") or None)
    A = Qwen38Engine(cfg, w, max_len=M, paged=False)
    kvpool.POOL = kvpool.KVPool(hard_floor=10 * kvpool.GiB, margin_rows=256)
    kvpool.OWNER = "B"
    B = Qwen38Engine(cfg, w, max_len=M, paged=True)
    assert A.kv.pages is None and B.kv.pages is not None
    assert A.kv.k.shape == B.kv.k.shape and A.kv.k.stride() == B.kv.k.stride()
    import engine.model as M_
    print(f"[identity] window {M:,}; verify graphs {'on' if M_.VERIFY_GRAPH else 'off'}; "
          f"page {B.kv.pages.bufs[0].page >> 20} MiB", flush=True)
    bad: list = []
    if a.window_cap:
        window_cap_identity(A, a.window_cap, bad)
    for n in lens:
        identity_at(A, B, n, seed=n, bad=bad)
    if not a.no_spec:
        spec_identity(A, B, cfg, bad)
    got = B.kv.pages.trim(0)
    print(f"  trim(0) gave back {got / kvpool.GiB:.2f} GiB; pool committed now "
          f"{kvpool.POOL.committed / kvpool.GiB:.3f} GiB", flush=True)
    assert kvpool.POOL.committed == 0
    print("pool", {k: v for k, v in kvpool.POOL.report().items() if k != "by_owner_gib"})
    if bad:
        print("IDENTITY FAIL:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("IDENTITY PASS: logits, KV rows, S and conv bit-identical; spec tokens and blocks equal")


if __name__ == "__main__":
    main()
