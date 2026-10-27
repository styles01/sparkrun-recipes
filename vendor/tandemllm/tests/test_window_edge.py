"""The KV window edge: a verify block must never run off the end.

The 2026-09-18 crashes (16 rows into 12, 8 into 6, `engine/model.py:180`) happened because the
decode loops clamp on OUTPUT tokens while the KV write counts ROWS, and a drafter is free to
return more than the count it was handed -- the block drafter proposes its whole block. The fix
clamps on rows in the loops and puts a named guard in `KVCache.append`.

Three claims:

  1. `KVCache.append` raises a NAMED error, not a torch shape mismatch, when a write would cross
     `max_len` -- a future path that misses the clamp fails legibly;
  2. `DraftTree.truncate` keeps a valid tree (DFS pre-order prefixes are ancestor-closed);
  3. end to end, a drafter that ignores the count it is given cannot crash the loops: with a tiny
     window, `generate_spec` and `generate_spec_tree` run to the edge and stop cleanly, and the
     KV never holds more than `max_len` rows.
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

if not torch.cuda.is_available():
    # engine/spec.py's loops time their phases with `torch.cuda.synchronize()`, which raises "No
    # CUDA GPUs are available" on the box's CPU run (CUDA_VISIBLE_DEVICES=""). There is nothing to
    # wait for on a CPU, so the timing calls become no-ops here and the loops run as written
    #.
    torch.cuda.synchronize = lambda *a, **k: None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import TextConfig  # noqa: E402
from engine.model import KVCache, Qwen38Engine  # noqa: E402
from engine.spec import generate_spec, generate_spec_tree  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

# the harness from the tree tests: a random 4-layer model with the real layer structure, on a CPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_forward_tree import FakeWeights, tiny_config  # noqa: E402

DEV = "cpu"


def test_append_guard_names_the_overrun():
    cfg = tiny_config()
    kv = KVCache(cfg, 16, DEV, dtype=torch.float32)
    k = torch.zeros(1, cfg.num_key_value_heads, 4, cfg.head_dim)
    v = torch.zeros_like(k)
    layer = cfg.attention_layers[0]
    kv.append(layer, k[:, :, :2], v[:, :, :2], start=14)      # rows [14, 16) fit exactly
    try:
        kv.append(layer, k, v, start=14)                       # rows [14, 18) do not
        raise AssertionError("an overrun must raise")
    except RuntimeError as e:
        assert "overruns the KV window" in str(e), f"the guard must name the bug, got: {e}"


def test_truncate_keeps_a_valid_tree():
    tree = DraftTree.from_sequences(0, [([5, 6, 7], 0.9), ([5, 8], 0.5), ([9, 6], 0.3)])
    tree.check()
    for n in range(1, len(tree) + 1):
        t = tree.truncate(n)
        t.check()
        assert len(t) == n
        assert t.tokens[0] == 0, "the anchor is node 0 and is always kept"
    assert tree.truncate(len(tree) + 5) is tree, "a tree that fits is returned as-is"


class FixedDrafter:
    """Proposes a full block/tree every step, ignoring the count it is handed.

    This is not a strawman: it is exactly the observed failure -- the clamps in the loops bound
    output tokens, and a drafter that returns its whole block regardless crosses the KV window.
    The loops must survive it.
    """

    def __init__(self, vocab: int, width: int, tree_mode: bool, seed: int = 0):
        self.vocab, self.width, self.tree_mode = vocab, width, tree_mode
        g = torch.Generator().manual_seed(seed)
        self.pool = torch.randint(0, vocab, (256,), generator=g).tolist()

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def propose(self, ctx, count):
        return [self.pool[(i * 7 + len(ctx)) % 256] for i in range(self.width)]

    def propose_tree(self, ctx, count):
        anchor = ctx[-1]
        toks = [self.pool[(i * 11 + len(ctx)) % 256] for i in range(self.width)]
        return DraftTree.chain(anchor, toks)


def _engine(max_len: int):
    cfg = tiny_config()
    eng = Qwen38Engine(cfg, FakeWeights(cfg), max_len=max_len, device=DEV)
    # the CPU reference path is fp32 end to end; the KV buffer defaults to bf16, which the GPU
    # kernels tolerate and CPU SDPA does not. The tree tests' harness casts the same way.
    eng.kv.k = eng.kv.k.to(torch.float32)
    eng.kv.v = eng.kv.v.to(torch.float32)
    eng.state.conv = eng.state.conv.to(torch.float32)
    return eng


def test_chain_loop_survives_a_drafter_that_ignores_the_count():
    eng = _engine(max_len=192)
    prompt = torch.arange(64)
    dr = FixedDrafter(eng.cfg.vocab_size, width=15, tree_mode=False)
    out, st = generate_spec(eng, prompt, 10000, dr, 15)        # max_new the window cannot hold
    assert len(out) - 1 <= 192 - 64 - 1, "the window, not max_new, is the bound"
    assert eng.kv.length <= eng.max_len, "the KV must never cross max_len"


def test_tree_loop_survives_a_drafter_that_ignores_the_count():
    eng = _engine(max_len=192)
    prompt = torch.arange(64)
    dr = FixedDrafter(eng.cfg.vocab_size, width=15, tree_mode=True)
    out, st = generate_spec_tree(eng, prompt, 10000, dr, 15)
    assert len(out) - 1 <= 192 - 64 - 1
    assert eng.kv.length <= eng.max_len


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:48s} ok")
            passed += 1
    print(f"{passed} passed")
