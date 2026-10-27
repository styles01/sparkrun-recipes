"""The host off the round's critical path, on a CPU, with the random 4-layer model.

Two flags, both off by default:

  * `QWEN38_LAUNCH_FIRST` (`server.app.LAUNCH_FIRST`): the tree loop starts the next proposal --
    up to its draft launch -- before it streams the block it just accepted, and collects the
    draft after the stream. The proposal is a `*_steps` generator that stops once after its
    launch; the plain `propose_tree` drives the same generator straight through.
  * `QWEN38_HOST_ASYNC` (`engine.model.HOST_ASYNC`): the loop's host-to-device copies through
    pinned memory (`h2d`), the draft's walk and log-probabilities in one read-back, the verify
    graph's own argmax.

What is tested here is what a CPU can show: the tokens are the same with the flags on and off
(greedy, an eos inside a block, the token cap inside a block, the repetition guard, the forced
reasoning close, a seeded sampled tree); the next draft is launched before the last block's
tokens are yielded and collected after them; a reader that stops mid-block leaves the published
context exactly as before; and no pageable host-to-device copy is left in the served tree loop.
The synchronisations themselves are counted on the board (tools/loop_sync.py).

Run: python tests/test_launch_first.py
"""

from __future__ import annotations

import os
import sys
import time

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine.model as M  # noqa: E402
from engine.drafters import run_steps, tree_steps  # noqa: E402
from engine.spec import ThinkBudget  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import CLOSING, OPEN, FakeTok, StopOn, serve  # noqa: E402

PROMPT = [5, 6, 7, 8, 9, 10, 11]
MAX_NEW = 60


class Log(list):
    pass


class StepDrafter:
    """A tree drafter shaped like the served one: `propose_tree_steps` stops once after its
    "launch". It proposes the true greedy continuation (from a reference run) to a depth that
    varies with the position, spoils the deepest node on every third call, and puts a decoy
    sibling under the anchor -- so blocks accept anything from zero drafted tokens to all."""

    def __init__(self, ref: list[int], log: Log):
        self.ref, self.log, self.calls = ref, log, 0

    def reset(self):
        self.calls = 0

    def observe(self, tokens):
        self.log.append(("observe", tuple(tokens)))

    def propose_tree(self, ctx, k):
        return run_steps(self.propose_tree_steps(ctx, k))

    def propose_tree_steps(self, ctx, k):
        self.log.append(("launch", len(ctx), k))
        yield
        self.log.append(("collect", len(ctx)))
        self.calls += 1
        n = len(ctx)
        depth = min(k, 1 + n % 5, len(self.ref) - n)
        if depth <= 0:
            return None
        true = list(self.ref[n:n + depth])
        if self.calls % 4 == 0:
            true[0] = (true[0] + 1) % 90            # nothing drafted is accepted
        elif self.calls % 3 == 0:
            true[-1] = (true[-1] + 1) % 90
        decoy = (self.ref[n] + 3) % 90 if n < len(self.ref) else 1
        tokens = [ctx[-1], decoy] + true
        parents = [-1, 0, 0] + list(range(2, 2 + depth - 1))
        return DraftTree(tokens=tokens, parents=parents)


def reference(max_new=MAX_NEW, prompt=PROMPT, **kw) -> list[int]:
    """The drafter-less greedy output: what every drafted run must reproduce."""
    serve(None)
    return list(app.generate_stream(torch.tensor(prompt), max_new, set(), **kw))


def run(launch_first: bool, host_async: bool = False, *, eos=(), max_new=MAX_NEW, take=None,
        think=None, pstop=None, sampler=None, ref=None, prompt=PROMPT):
    """One request through the served loop with a StepDrafter. Returns (tokens, log, ctx, kv)."""
    ref = reference() if ref is None else ref
    log = Log()
    d = StepDrafter(prompt + ref, log)
    eng = serve(d, tree=True, k=15)
    app.STATE["sampled_tree"] = sampler is not None
    app.LAUNCH_FIRST, M.HOST_ASYNC = launch_first, host_async
    try:
        gen = app.generate_stream(torch.tensor(prompt), max_new, set(eos), think, pstop=pstop,
                                  sampler=sampler)
        out = []
        for t in gen:
            out.append(t)
            log.append(("yield", t))
            if take is not None and len(out) >= take:
                gen.close()
                break
    finally:
        app.LAUNCH_FIRST, M.HOST_ASYNC = False, False
    return out, log, list(app.STATE["last_ctx"]), eng.kv.length


def calls(log):
    return [e for e in log if e[0] in ("launch", "observe")]


# ------------------------------------------------------------------ the tokens do not change

def test_greedy_tokens_are_the_same_with_the_flags_on_and_off():
    ref = reference()
    runs = {(lf, ha): run(lf, ha, ref=ref) for lf in (False, True) for ha in (False, True)}
    for key, (out, log, ctx, kv) in runs.items():
        assert out == ref[:MAX_NEW], (key, out, ref)
    base = runs[(False, False)]
    for key, (out, log, ctx, kv) in runs.items():
        assert ctx == base[2] and kv == base[3], key
        # the drafter is asked the same questions with the same arguments, in the same order
        assert calls(log) == calls(base[1]), key
    # and the test means something: blocks accepted several tokens, and some accepted none
    obs = [len(e[1]) for e in base[1] if e[0] == "observe"][1:]
    assert max(obs) >= 3 and min(obs) == 1, obs


def test_an_eos_or_the_cap_inside_a_block_ends_it_the_same_way():
    ref = reference()
    for eos, max_new in (((ref[20],), MAX_NEW), ((), 23), ((), 24), ((ref[5], ref[31]), 40)):
        a = run(False, eos=eos, max_new=max_new, ref=ref)
        b = run(True, True, eos=eos, max_new=max_new, ref=ref)
        assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3], (eos, max_new, a[0], b[0])
        assert calls(a[1]) == calls(b[1]), (eos, max_new)


def test_the_repetition_guard_ends_the_stream_at_the_same_token():
    ref = reference()
    block = None
    for e in run(False, ref=ref)[1]:
        if e[0] == "observe" and len(e[1]) >= 2:
            block = list(e[1])                      # the first block of two or more
            break
    assert block is not None
    a = run(False, pstop=StopOn(block), ref=ref)
    b = run(True, True, pstop=StopOn(block), ref=ref)
    assert a[0] == b[0] and a[2] == b[2], (a[0], b[0])
    assert len(a[0]) < MAX_NEW


def test_the_forced_reasoning_close_is_the_same():
    """The think budget looks at every block before the next proposal in both orders; when it
    fires, the close is forced and nothing was launched ahead of it."""
    prompt = PROMPT + [OPEN]
    ref = reference(prompt=prompt, think=ThinkBudget(FakeTok(), budget=9).start(prompt))
    outs = []
    for lf in (False, True):
        think = ThinkBudget(FakeTok(), budget=9)
        outs.append(run(lf, lf, think=think, ref=ref, prompt=prompt))
        assert think.done, "the budget closed the block"
    a, b = outs
    assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3], (a[0], b[0])
    assert calls(a[1]) == calls(b[1])
    assert CLOSING[0] in a[0]


def test_a_seeded_sampled_tree_draws_the_same_tokens():
    ref = reference()
    outs = [run(lf, lf, sampler=app.Sampler(temperature=0.9, top_p=0.95, seed=7), ref=ref)[0]
            for lf in (False, True)]
    assert outs[0] == outs[1], outs


# ------------------------------------------------------------------ the order is the point

def test_the_next_draft_is_launched_before_the_last_block_is_streamed():
    ref = reference()
    _, log, _, _ = run(True, ref=ref)
    # after each block's observe: the launch, then that block's yields, then the collect
    seen = 0
    for i, e in enumerate(log):
        if e[0] != "observe" or i == 0:
            continue
        rest = log[i + 1:]
        kinds = [x[0] for x in rest]
        if "launch" not in kinds:
            continue                                # the last block: nothing after it
        n = len(e[1])
        assert kinds[0] == "launch", (i, rest[:4])
        assert kinds[1:1 + n] == ["yield"] * n, (i, rest[:n + 2])
        assert kinds[1 + n] == "collect", (i, rest[:n + 3])
        seen += 1
    assert seen >= 5, seen


def test_without_the_flag_the_draft_follows_the_stream_as_before():
    ref = reference()
    _, log, _, _ = run(False, ref=ref)
    for i, e in enumerate(log):
        if e[0] == "launch":
            assert log[i + 1][0] == "collect", log[i:i + 3]


def test_a_reader_that_stops_mid_block_leaves_the_context_as_before():
    """`_remember` publishes `STATE["last_ctx"]`; the old order appended a token and then yielded
    it, so a reader that stopped left the list ending at the last token it was handed."""
    ref = reference()
    for take in range(2, 30):
        a = run(False, take=take, ref=ref)
        b = run(True, True, take=take, ref=ref)
        assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3], (take, a[2][-5:], b[2][-5:])


# ------------------------------------------------------------------ no pageable copy left

def test_no_pageable_host_to_device_copy_in_the_served_tree_loop():
    """Every list that becomes a device tensor in the tree loop -- the block's tokens, the
    accepted path (app, commit, drafter sync), a new tree shape's tables -- goes through `h2d`.
    On the board `torch.tensor(list, device=cuda)` is a copy from pageable memory followed by a
    stream synchronisation; it was made five or six times a round."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    watched = (os.path.join(here, "server"), os.path.join(here, "engine"))
    hits, armed = [], [False]
    orig = {"tensor": torch.tensor, "as_tensor": torch.as_tensor}

    def spy(name):
        def f(*a, **kw):
            if armed[0] and kw.get("device") is not None:
                fr = sys._getframe(1)
                if fr.f_code.co_filename.startswith(watched):
                    hits.append((name, os.path.relpath(fr.f_code.co_filename, here), fr.f_lineno))
            return orig[name](*a, **kw)
        return f

    ref = reference()
    log = Log()
    serve(StepDrafter(PROMPT + ref, log), tree=True, k=15)
    torch.tensor, torch.as_tensor = spy("tensor"), spy("as_tensor")
    app.LAUNCH_FIRST, M.HOST_ASYNC = True, True
    try:
        M.TreeCtx._cache.clear()
        gen = app.generate_stream(torch.tensor(PROMPT), MAX_NEW, set())
        next(gen)                                   # the prefill (once a request) is behind us
        armed[0] = True
        out = [t for t in gen]
    finally:
        torch.tensor, torch.as_tensor = orig["tensor"], orig["as_tensor"]
        app.LAUNCH_FIRST, M.HOST_ASYNC = False, False
    assert len(out) == MAX_NEW - 1
    rounds = sum(1 for e in log if e[0] == "observe") - 1
    assert rounds >= 10, rounds
    assert not hits, f"{len(hits)} pageable copies in {rounds} rounds: {sorted(set(hits))}"


def test_h2d_takes_pinned_memory_and_does_not_wait_on_a_gpu():
    seen = []
    real_pin = torch.Tensor.pin_memory
    real_to = torch.Tensor.to

    def pin(self, *a, **kw):
        seen.append("pin")
        return self

    def to(self, *a, **kw):
        seen.append(("to", str(a[0]) if a else None, kw.get("non_blocking", False)))
        return real_to(self, "cpu")

    torch.Tensor.pin_memory, torch.Tensor.to = pin, to
    try:
        M.HOST_ASYNC = True
        t = M.h2d([3, 1, 2], torch.long, "cuda")
        assert seen == ["pin", ("to", "cuda", True)], seen
        seen.clear()
        M.HOST_ASYNC = False
        M.h2d([3, 1, 2], torch.long, "cuda")
        assert seen == [("to", "cuda", False)], seen        # flag off: the old copy
    finally:
        torch.Tensor.pin_memory, torch.Tensor.to = real_pin, real_to
        M.HOST_ASYNC = False
    assert t.tolist() == [3, 1, 2] and t.dtype == torch.long
    assert M.h2d([[True, False]], torch.bool, "cpu").tolist() == [[True, False]]


# ------------------------------------------------------------------ the drafters' split

def test_walk_host_logp_reads_the_numbers_walk_host_and_log_softmax_read():
    from engine.drafters.dflash2 import DFlash2Module
    g = torch.Generator().manual_seed(49)
    for L, k in ((7, 16), (15, 16), (3, 4)):
        cand = torch.randint(0, 5000, (L, k), generator=g)
        scores = torch.randn(L, k, k, generator=g)
        lp = torch.log_softmax(scores.float() / 0.7, dim=-1)
        toks, table, logp = DFlash2Module.walk_host_logp(cand, scores, lp)
        assert (toks, table) == DFlash2Module.walk_host(cand, scores)
        assert logp == torch.log_softmax(scores.float() / 0.7, dim=-1).tolist()


class _Head:
    """A block drafter with a steps proposal: the tree it returns depends only on the context."""
    cfg = type("C", (), {"block_size": 16})()
    tree_temp = 1.0

    def __init__(self, pause_s=0.0):
        self.pause_s, self.launched = pause_s, 0

    def propose_tree_steps(self, ctx, budget):
        self.launched += 1
        yield
        n = len(ctx)
        return DraftTree(tokens=[ctx[-1]] + [(n + i) % 50 for i in range(3)],
                         parents=[-1, 0, 1, 2], scores=[1.0, 0.9, 0.5, 0.2])

    def propose_tree(self, ctx, budget):
        return run_steps(self.propose_tree_steps(ctx, budget))


def test_tree_steps_drives_a_plain_drafter_and_a_steps_one():
    class Plain:
        def propose_tree(self, ctx, k):
            return ("plain", len(ctx), k)
    steps = tree_steps(Plain(), [1, 2, 3], 5)
    try:
        next(steps)
        raise AssertionError("a drafter without steps must not stop")
    except StopIteration as done:
        assert done.value == ("plain", 3, 5)
    h = _Head()
    steps = tree_steps(h, [1, 2, 3], 5)
    assert next(steps) is None and h.launched == 1     # stopped at the launch
    assert run_steps(steps).tokens == [3, 3, 4, 5]


def test_the_length_router_prices_the_draft_without_the_time_it_was_stopped():
    """The router learns what a draft call costs from wall time. Stopped, the loop streams; that
    time is not the draft's and must not enter the estimate."""
    from engine.lenrouter import LengthRouter
    steps = LengthRouter._timed_steps(_Head(), [1, 2, 3, 4], 7)
    next(steps)
    time.sleep(0.05)
    try:
        next(steps)
        raise AssertionError("one stop only")
    except StopIteration as done:
        tree, ms = done.value
    assert tree.tokens == [4, 4, 5, 6]
    assert ms < 20.0, ms


if __name__ == "__main__":
    import traceback
    fails = passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {fails} failed")
    sys.exit(1 if fails else 0)
