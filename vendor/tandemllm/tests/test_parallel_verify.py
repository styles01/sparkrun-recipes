"""Parallel requests on the CPU with the random 4-layer model: the joint verify and the lockstep
rounds of the served loop, held to what each request writes alone.

  1. `forward_tree_multi` over two engines (two sequence slots over ONE set of weights) gives each
     engine the logits, the recurrent and convolution state and the KV rows its own
     `forward_tree` gives it, before and after `commit_tree`, over several rounds and on
     branching trees of different sizes;
  2. a chain-shaped block is not batchable while the chain kernel takes chains (it verifies alone);
  3. two requests decoding at once on slot 0 and a side slot, in `server/app.py`'s own loop with
     the GPU turn and the barrier, each write the tokens they write alone, and the log of the
     barrier shows joint rounds (the proof that they verified together);
  4. the same with three slots;
  5. without side slots nothing of this exists: `generate_stream` is the loop itself, `STATE` reads
     and writes like the plain dict it was, and the engine lock has no turn.

The CPU's matrix products are row-invariant on the box (aarch64, OpenBLAS); on a platform where
they are not, (1) and (3) fall back to an argmax check with a tolerance on the logits.

Run: python tests/test_parallel_verify.py
"""

from __future__ import annotations

import os
import sys
import threading

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.model import forward_tree_multi  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402  (the served class: its cold verbs)
from engine.tree import DraftTree  # noqa: E402
from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax  # noqa: E402
from server import app, slots  # noqa: E402
from test_forward_tree import FakeWeights, tiny_config  # noqa: E402
from test_app_loop import FakeTok  # noqa: E402

DT = torch.float32
EXACT = True                      # set False by the first comparison that is not bit-exact


def _engine(cfg, w, max_len=256):
    eng = Qwen38Engine(cfg, w, max_len=max_len, device="cpu")
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    return eng


def _same(a, b, what):
    global EXACT
    if torch.equal(a, b):
        return
    EXACT = False
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"{what}: max diff {(a - b).abs().max()}"


def _tree(seed, n_main, n_side, anchor):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(1, 97, (n_main + n_side,), generator=g).tolist()
    tokens, parents = [anchor], [-1]
    for i in range(n_main):                      # the spine 0 -> 1 -> ... -> n_main
        tokens.append(toks[i])
        parents.append(len(tokens) - 2)
    for j in range(n_side):                      # a branch off the anchor
        tokens.append(toks[n_main + j])
        parents.append(0 if j == 0 else len(tokens) - 2)
    return DraftTree(tokens, parents)


def _prefill(eng, seed, n):
    torch.manual_seed(seed)
    ids = torch.randint(1, 97, (n,))
    with torch.no_grad():
        eng.forward(ids, start=0, last_only=True)
    return ids.tolist()


def _state(eng):
    return (eng.state.S.clone(), eng.state.conv.clone(), eng.kv.k[..., :eng.kv.length, :].clone(),
            eng.kv.v[..., :eng.kv.length, :].clone(), eng.kv.length)


def test_forward_tree_multi_equals_each_engines_own_verify():
    _joint_rounds()


def test_forward_tree_multi_keeps_each_engines_rotary_offset():
    """SRV-629: a request with images on a side slot verifies beside a text request on slot 0.
    Every row after its prompt reads the rotary at index + `pos_delta` (engine/vision.py: 8
    screenshots, about -1,080), slot 0's at its index: the joint round gives each engine the
    bits its own verify gives it, and the offset is not lost (its logits are not the text
    rows')."""
    _joint_rounds(delta1=-1080, rows1=1100)


def _joint_rounds(delta1: int = 0, rows1: int = 9):
    cfg = tiny_config()
    w = FakeWeights(cfg, seed=3)
    L = 256 if rows1 + 64 <= 256 else rows1 + 128
    alone = [_engine(cfg, w, L), _engine(cfg, w, L)]
    joint = [_engine(cfg, w, L), _engine(cfg, w, L)]
    n0 = [_prefill(e, 11, 14) for e in (alone[0], joint[0])][0]
    n1 = [_prefill(e, 12, rows1) for e in (alone[1], joint[1])][0]
    pos = [len(n0), len(n1)]
    if delta1:
        # not vacuous: the same block after the same rows, with and without the offset
        flat, off = _engine(cfg, w, L), _engine(cfg, w, L)
        for e in (flat, off):
            _prefill(e, 12, len(n1))
        off.pos_delta = delta1
        t = _tree(99, 4, 2, 7)
        with torch.no_grad():
            a = flat.forward_tree(torch.tensor(t.tokens), t.parents, start=pos[1])
            b = off.forward_tree(torch.tensor(t.tokens), t.parents, start=pos[1])
        assert not torch.equal(a, b), "the offset changed no logit"
        alone[1].pos_delta = joint[1].pos_delta = delta1
    shapes = [((5, 3), (3, 2)), ((7, 0), (2, 4)), ((1, 1), (6, 3))]
    for rnd, ((m0, s0), (m1, s1)) in enumerate(shapes):
        trees = [_tree(100 + rnd, m0, s0, 5), _tree(200 + rnd, m1, s1, 7)]
        blocks = [torch.tensor(t.tokens) for t in trees]
        with torch.no_grad():
            lone = [alone[i].forward_tree(blocks[i], trees[i].parents, start=pos[i])
                    for i in (0, 1)]
            multi = forward_tree_multi([(joint[i], blocks[i], trees[i].parents, pos[i])
                                        for i in (0, 1)])
            for i in (0, 1):
                _same(multi[i], lone[i], f"round {rnd} slot {i} logits")
                assert multi[i].shape == lone[i].shape
                _same(joint[i].hidden_post_norm, alone[i].hidden_post_norm,
                      f"round {rnd} slot {i} hidden rows")
            # commit a path that leaves the spine for the branch where there is one
            for i in (0, 1):
                t = trees[i]
                path = [0, 1, 2] if len(t.tokens) > 2 and t.parents[2] == 1 else [0, 1]
                alone[i].commit_tree(path)
                joint[i].commit_tree(path)
                for a, b, what in zip(_state(alone[i]), _state(joint[i]),
                                      ("S", "conv", "K", "V", "length")):
                    if what == "length":
                        assert a == b
                    else:
                        _same(a, b, f"round {rnd} slot {i} committed {what}")
                pos[i] += len(path)


def test_a_chain_block_is_not_batchable_while_chains_take_their_own_kernel():
    cfg = tiny_config()
    eng = _engine(cfg, FakeWeights(cfg))
    chain = DraftTree.chain(3, [4, 5, 6])
    import engine.model as m
    saved = m.TREE_CHAIN_DELEGATE
    try:
        m.TREE_CHAIN_DELEGATE = True
        assert not eng.tree_batchable(chain.parents)
        assert eng.tree_batchable(_tree(1, 3, 2, 3).parents)
        m.TREE_CHAIN_DELEGATE = False
        assert eng.tree_batchable(chain.parents)
    finally:
        m.TREE_CHAIN_DELEGATE = saved
    assert not eng.tree_batchable([-1]), "an anchor alone is no block"


# ------------------------------------------------------------------ the served loop
class OracleTree:
    """A tree drafter that knows the request's own greedy continuation (`full`): the spine is the
    next tokens of it with one wrong token at a position that moves with the context, and a wrong
    branch hangs off the anchor. So rounds accept 0 to several tokens, through the spine or not."""

    def __init__(self, full, width=6):
        self.full, self.width = list(full), width

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def propose_tree(self, ctx, count):
        i = len(ctx)
        spine = self.full[i:i + min(self.width, max(1, count))]
        if not spine:
            spine = [1]
        bad = (i * 7) % (len(spine) + 2)
        spine = [(t + 13) % 97 if j == bad else t for j, t in enumerate(spine)]
        tokens, parents = [ctx[-1]], [-1]
        for t in spine:
            tokens.append(int(t))
            parents.append(len(tokens) - 2)
        for j, t in enumerate(((i * 5) % 97, (i * 3 + 1) % 97)):
            tokens.append(int(t))
            parents.append(0 if j == 0 else len(tokens) - 2)
        return DraftTree(tokens, parents)

    def propose(self, ctx, count):
        return []


def _setup(engines, drafters, max_len=256, cut="share", max_rows=32):
    app.STATE.clear()
    app.STATE.update(engine=engines[0], drafter=drafters[0], k=7, tree=True, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                     prefix_chunk=0, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=max_len, default_max_tokens=64, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None)
    ss = [slots.Slot(0, max_len, values=None, lock=app.LOCK)]
    for i in range(1, len(engines)):
        v = {k: None for k in slots.OVERLAY}
        v.update(engine=engines[i], drafter=drafters[i], max_len=max_len)
        ss.append(slots.Slot(i, max_len, values=v))
    turn = slots.Turn(quantum_s=0.0)
    meet = slots.Meeting(max_rows, 0.002, cut=cut)
    meet.timing = True                 # the round's parts in the stats (QSE_PARALLEL_TIMING=1)
    pool = slots.Pool(ss, turn=turn, meeting=meet)
    app.STATE["pool"] = pool
    app.LOCK.turn = turn
    return pool


def _teardown():
    app.LOCK.turn = None
    app.STATE.pop("pool", None)


def _greedy(cfg, w, prompt, n):
    """Plain greedy decoding of `prompt`, one token a forward, no drafter: the reference."""
    eng = _engine(cfg, w)
    out = []
    with torch.no_grad():
        lg = eng.forward(torch.tensor(prompt), start=0, last_only=True)
        tok = int(lg[0, -1].argmax())
        out.append(tok)
        pos = len(prompt)
        while len(out) < n:
            lg = eng.forward(torch.tensor([tok]), start=pos, last_only=True)
            pos += 1
            tok = int(lg[0, -1].argmax())
            out.append(tok)
    return out


def _alone(cfg, w, prompt, full, n):
    eng = _engine(cfg, w)
    _setup([eng], [OracleTree(full)])
    _teardown()                     # one slot: no pool, the loop as it always ran
    with torch.no_grad():
        return list(app.generate_stream(torch.tensor(prompt), n, set()))


def _together(cfg, w, prompts, fulls, n, cut="share", max_rows=32):
    engines = [_engine(cfg, w) for _ in prompts]
    pool = _setup(engines, [OracleTree(f) for f in fulls], cut=cut, max_rows=max_rows)
    outs: dict = {}
    errs: list = []
    go = threading.Barrier(len(prompts))

    def run(i):
        try:
            go.wait()
            if i == 0:
                with app.LOCK:
                    with torch.no_grad():
                        outs[i] = list(app.generate_stream(torch.tensor(prompts[i]), n, set()))
            else:
                s = pool.slots[i]
                slots.bind(s)
                pool.turn.reset(i)
                pool.turn.acquire(i)
                try:
                    with torch.no_grad():
                        outs[i] = list(app.generate_stream(torch.tensor(prompts[i]), n, set()))
                    ctx = app.STATE["last_ctx"]
                    assert ctx[-len(outs[i]):] == outs[i], "the slot's own published context"
                finally:
                    if pool.turn.held_by(i):
                        pool.turn.release(i)
                    slots.bind(None)
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    ths = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for th in ths:
        th.start()
    for th in ths:
        th.join(120)
    stats = dict(pool.meeting.stats)
    phases = [s.phase for s in pool.slots]
    _teardown()
    if errs:
        raise errs[0]
    return [outs[i] for i in range(len(prompts))], stats, phases


def _check_parallel(k):
    cfg = tiny_config()
    w = FakeWeights(cfg, seed=5)
    n = 40
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(k)]
    refs = [_greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]
    alone = [_alone(cfg, w, p, f, n) for p, f in zip(prompts, fulls)]
    together, stats, phases = _together(cfg, w, prompts, fulls, n)
    for i in range(k):
        assert together[i] == alone[i], (f"slot {i}: joint rounds wrote other tokens than alone\n"
                                         f"alone    {alone[i]}\ntogether {together[i]}")
        if alone[i] != refs[i]:
            # the tree verify and the one-token forward sum in different orders (docs/
            # exactness.md); what this file holds is joint == alone
            print(f"    note: slot {i}'s tree loop parts from plain greedy at "
                  f"{next(j for j, (a, b) in enumerate(zip(alone[i], refs[i])) if a != b)}")
    assert stats["joint_rounds"] > 0, f"no joint round: the requests never verified together {stats}"
    assert stats["joint_hist"].get(k, 0) > 0 or k > 2, stats
    assert all(p == "" for p in phases), f"every slot out of the barrier at the end: {phases}"
    # the round's parts the bench reads: a draft's time by slots decoding, tokens a joint round
    assert stats["draft_ms"] and any(k.startswith("joint") for k in stats["accept"]), stats
    assert stats["post_ms"] and stats["verify_wait_ms"], stats
    assert any(int(k) >= 2 for k in stats["round_ms"]), stats["round_ms"]
    return stats


def test_two_requests_in_lockstep_write_what_they_write_alone():
    stats = _check_parallel(2)
    print(f"    joint rounds {stats['joint_rounds']}, rows {stats['rows_hist']}, "
          f"timeouts {stats['timeouts']}")


def test_three_requests_in_lockstep_write_what_they_write_alone():
    stats = _check_parallel(3)
    print(f"    joint rounds {stats['joint_rounds']}, blocks {stats['joint_hist']}")


# ------------------------------------------------------------------ flag off
def test_without_side_slots_nothing_changes():
    cfg = tiny_config()
    eng = _engine(cfg, FakeWeights(cfg))
    _setup([eng], [None])
    _teardown()
    assert app.LOCK.turn is None
    gen = app.generate_stream(torch.tensor([1, 2, 3]), 3, set())
    assert gen.gi_code.co_name == "_generate_stream", "no wrapper without side slots"
    gen.close()
    s = app.STATE
    s["x"] = 1
    assert s["x"] == 1 and s.get("x") == 1 and "x" in s and s.pop("x") == 1 and "x" not in s
    assert s.get("missing", 7) == 7 and s.setdefault("y", 2) == 2
    s.pop("y")
    with app.LOCK:
        assert app.LOCK.locked()
    assert not app.LOCK.locked()
    assert app.LOCK.acquire(blocking=False)
    app.LOCK.release()


def test_a_side_slot_sees_its_own_values_and_slot_0_sees_state():
    _setup([object(), object()], [None, None])
    pool = app.STATE["pool"]
    side = pool.slots[1]
    try:
        eng0 = app.STATE["engine"]
        slots.bind(side)
        assert app.STATE["engine"] is side.values["engine"] is not eng0
        app.STATE["last_ctx"] = [1, 2]
        assert app.STATE.get("resident") is None and app.STATE["max_len"] == 256
        assert app.STATE.pop("last_ctx") == [1, 2] and app.STATE.get("last_ctx") is None
        app.STATE["blocks"] = "side"
        slots.bind(None)
        assert app.STATE["engine"] is eng0
        assert app.STATE.get("blocks") is None, "a side slot's value never reaches slot 0"
        assert side.values["blocks"] == "side"
    finally:
        slots.bind(None)
        _teardown()


def test_a_joint_cut_shares_one_row_budget_and_keeps_the_text():
    """`--parallel-cut joint`: the round cuts both slots' whole trees to one budget (12 rows here,
    under the two trees' 18), best nodes first. Every pass stays in the budget, each slot keeps at
    least its anchor and one node, and on this model the texts are each request's alone text."""
    from engine.tree import DraftTree
    t1 = DraftTree([5, 6, 7, 8, 9], [-1, 0, 1, 2, 0], [1.0, .9, .8, .1, .05])
    t2 = DraftTree([3, 4, 5], [-1, 0, 1], [1.0, .2, .1])
    k1, k2 = slots.joint_cut([t1, t2], 5)
    assert len(k1) + len(k2) <= 5 and {0, 1} <= k1 and {0, 1} <= k2, (k1, k2)
    assert 2 in k1 and 3 not in k1, "best first: node 2 (0.8) before node 3 (0.1)"
    for k, t in ((k1, t1), (k2, t2)):
        assert all(t.parents[i] in k for i in k if i), "ancestor-closed"
    cfg = tiny_config()
    w = FakeWeights(cfg, seed=5)
    n = 40
    prompts = [[(3 + 7 * i + 11 * j) % 96 + 1 for j in range(10 + 3 * i)] for i in range(2)]
    refs = [_greedy(cfg, w, p, n) for p in prompts]
    fulls = [p + r for p, r in zip(prompts, refs)]
    alone = [_alone(cfg, w, p, f, n) for p, f in zip(prompts, fulls)]
    together, stats, _ = _together(cfg, w, prompts, fulls, n, cut="joint", max_rows=12)
    assert stats["joint_rounds"] > 0 and stats["trimmed"] > 0, stats
    assert max(int(r) for r in stats["rows_hist"]) <= 12, stats["rows_hist"]
    for i in range(2):
        assert together[i] == alone[i], (i, alone[i], together[i])


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed (bit-exact: {EXACT})")
