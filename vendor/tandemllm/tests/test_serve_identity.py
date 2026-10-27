"""The served loop without the conductor: token for token what the release wrote.

`server/app.py`'s `generate_stream` grew the conductor's safe points (server/conductor/monitor/),
and with `--conductor raw` -- no supervisor -- every one of them is skipped. This pins that: a
fixed set of generations on the random 4-layer model, recorded ONCE from the release commit
(`vis-40-live-rates` a45af52), must come out of this code unchanged, on every path the loop has:

  * the chain, the tree and the single-token step, each to its token budget;
  * the same three with a client reasoning budget that closes the block (the streamed phrase);
  * the chain and the tree with a drafter that is always right (every block accepted in full),
    and with one that is right but for its last token (every block rolled back by one);
  * the chain and the single-token step stopping at an end token;
  * the KV length and the published context at the end of each (since SRV-530 the KV length stops
    at the committed tokens where the release's ran past the end token).

The outputs depend on the torch build and the CPU's kernels, so the fixture is keyed by platform
and torch version, as tests/test_refactor_gate.py does. A host without a recorded key prints what
it got and skips the comparison. To record a key, run this file ON THE RELEASE CODE:

    QSE_IDENTITY_RECORD=1 python tests/test_serve_identity.py

This file imports nothing of the conductor, so it runs on the release tree as it is.

Run: python tests/test_serve_identity.py
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.spec import ThinkBudget  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import OPEN, FakeTok, serve  # noqa: E402
from test_window_edge import FixedDrafter  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "serve_identity.json")


def host_key() -> str:
    return f"{platform.machine()}-{platform.system().lower()}-torch{torch.__version__}"


def _think(budget: int) -> ThinkBudget:
    """A budget with no stall detection, on the release's ThinkBudget and on this one."""
    if "stall" in inspect.signature(ThinkBudget).parameters:
        return ThinkBudget(FakeTok(), budget=budget, stall=False)      # the release
    return ThinkBudget(FakeTok(), budget=budget)


class Oracle:
    """A drafter that proposes the reference continuation, so every block is accepted whole."""

    def __init__(self, ref: list[int], start: int, width: int, tree_mode: bool,
                 miss_last: bool = False):
        self.ref, self.start, self.width, self.tree_mode = ref, start, width, tree_mode
        self.miss_last = miss_last

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def _next(self, ctx):
        i = len(ctx) - self.start
        got = self.ref[i:i + self.width]
        if self.miss_last and got:
            got = got[:-1] + [(got[-1] + 1) % 97]
        return got

    def propose(self, ctx, count):
        return self._next(ctx)

    def propose_tree(self, ctx, count):
        toks = self._next(ctx)
        return DraftTree.chain(ctx[-1], toks) if toks else None


def _run(drafter, tree, prompt, max_new, eos=(), budget=0):
    eng = serve(drafter, tree=tree)
    think = _think(budget) if budget else None
    out = list(app.generate_stream(torch.tensor(prompt), max_new, set(eos), think))
    return {"out": out, "kv": int(eng.kv.length), "ctx": len(app.STATE["last_ctx"])}


def cases() -> dict:
    """Every case's result on the code that is importable now."""
    plain = [5, 6, 7, 8, 9]
    thinking = [5, 6, 7, 8, OPEN]
    got = {}
    for name, dr, tree in (("chain", FixedDrafter(97, 5, tree_mode=False), False),
                           ("tree", FixedDrafter(97, 5, tree_mode=True), True),
                           ("single", None, False)):
        got[name] = _run(dr, tree, plain, 40)
        got[name + "_budget"] = _run(dr, tree, thinking, 40, budget=3)
    ref = got["single"]["out"]
    got["chain_oracle"] = _run(Oracle(ref, len(plain), 5, False), False, plain, 40)
    got["tree_oracle"] = _run(Oracle(ref, len(plain), 5, True), True, plain, 40)
    got["chain_partial"] = _run(Oracle(ref, len(plain), 5, False, miss_last=True), False,
                                plain, 40)
    got["tree_partial"] = _run(Oracle(ref, len(plain), 5, True, miss_last=True), True,
                               plain, 40)
    # an end token: the first token the plain run wrote after its tenth
    stop = next(t for t in ref[10:] if t not in ref[:10])
    got["chain_eos"] = _run(Oracle(ref, len(plain), 5, False), False, plain, 40, eos=[stop])
    got["single_eos"] = _run(None, False, plain, 40, eos=[stop])
    got["stop_token"] = stop
    return got


def test_the_raw_loop_writes_what_the_release_wrote():
    got = cases()
    fixture = json.load(open(FIXTURE)) if os.path.exists(FIXTURE) else {}
    key = host_key()
    if os.environ.get("QSE_IDENTITY_RECORD") == "1":
        fixture[key] = got
        with open(FIXTURE, "w") as f:
            json.dump(fixture, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"  recorded {key}")
        return
    want = fixture.get(key)
    if want is None:
        print(f"  no recorded key for {key}; this host's outputs: {json.dumps(got)[:300]}...")
        return
    bad = {}
    for k, v in want.items():
        if isinstance(v, dict) and v.get("kv", 0) > v.get("ctx", 0):
            # SRV-530: the rows a block verified past the end token are not committed any more;
            # the engine stops at the committed tokens, the text is the release's
            v = dict(v, kv=v["ctx"])
        if got.get(k) != v:
            bad[k] = (got.get(k), v)
    assert not bad, f"the raw loop no longer writes what the release wrote: {bad}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
