"""ENG-E8 step 0: the shared walk and the 27B's cold adapter verbs.

  * engine/methods/walk.py `accept_tree` returns what `Qwen38Engine.accept_tree` returns on random
    trees and picks (the server's loop calls the shared one for every model);
  * the 27B's `cost()` (engine/models/qwen3_5/adapter.py) gives the dashboard the bytes
    `cache_stats` computed inline before, and `discard_pending()` clears the fields the caches reset.
"""

from __future__ import annotations

import os
import random
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import walk  # noqa: E402
from engine.models.qwen3_5 import adapter  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402
from engine.tree import DraftTree  # noqa: E402


def _random_tree(rng: random.Random, n: int) -> DraftTree:
    toks, parents = [rng.randrange(6)], [-1]
    for i in range(1, n):
        parents.append(rng.randrange(i))
        toks.append(rng.randrange(6))
    return DraftTree(tokens=toks, parents=parents)


def test_shared_walk_is_the_27b_walk():
    rng = random.Random(5)
    deep = 0
    for _ in range(3000):
        t = _random_tree(rng, rng.randrange(1, 12))
        picks = [rng.randrange(6) for _ in t.tokens]
        got = walk.accept_tree(t, picks)
        assert got == Qwen38Engine.accept_tree(None, t, picks), (t.tokens, t.parents, picks)
        deep += len(got[0]) > 2
    return f"3,000 random trees equal ({deep} walks deeper than one draft)"


def test_paths_through_first_and_later_children():
    t = DraftTree(tokens=[5, 10, 11, 20, 21, 30, 12, 22], parents=[-1, 0, 1, 1, 3, 4, 0, 6])
    assert walk.accept_tree(t, [12, 0, 0, 0, 0, 0, 22, 9]) == ([0, 6, 7], [12, 22, 9])
    assert walk.accept_tree(t, [10, 20, 0, 21, 30, 1, 0, 0]) == ([0, 1, 3, 4, 5], [10, 20, 21, 30, 1])
    assert walk.accept_tree(t, [99] * 8) == ([0], [99])
    return "a later child, a deep path, the empty path (Flash-Next's walk cases)"


def _fake_27b():
    cfg = types.SimpleNamespace(attention_layers=[3, 7, 11], num_key_value_heads=4, head_dim=256)
    S, conv = torch.zeros(48, 16, 128, 128), torch.zeros(48, 3, 10240, dtype=torch.bfloat16)
    state = types.SimpleNamespace(S=S, conv=conv, nbytes=S.numel() * 4 + conv.numel() * 2)
    return types.SimpleNamespace(cfg=cfg, state=state, _pending_walk=True, _pend=("trace", None, [0]))


def test_27b_cost_is_the_dashboards_formula():
    eng = _fake_27b()
    sc = adapter.cost(eng).snapshot_cost()
    # the inline formula of server/app.py `cache_stats` before the adapter
    want = {"recurrent_bytes": eng.state.S.numel() * 4,
            "conv_bytes": eng.state.conv.numel() * eng.state.conv.element_size(),
            "kv_bytes_per_token": len(eng.cfg.attention_layers) * eng.cfg.num_key_value_heads * eng.cfg.head_dim * 2 * 2}
    assert sc == want, (sc, want)
    assert adapter.cost(eng).section_bytes == want["recurrent_bytes"] + want["conv_bytes"]
    adapter.discard_pending(eng)
    assert eng._pending_walk is False and eng._pend is None
    assert Qwen38Engine.cost is adapter.cost and Qwen38Engine.discard_pending is adapter.discard_pending
    assert Qwen38Engine.precapture is adapter.precapture and Qwen38Engine.manifest is adapter.manifest
    assert Qwen38Engine.caps.caches and Qwen38Engine.caps.side_slots
    return f"snapshot_cost {sc}; installed on Qwen38Engine"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<50} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<50} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
