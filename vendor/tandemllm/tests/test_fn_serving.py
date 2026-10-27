"""SRV-87: the host side of Flash-Next in the server loop (engine/models/flashnext/serving.py and the
shared MTP tree drafter, engine/methods/mtp_drafter.py), no GPU.

  * the drafter pairs every synced row with the token after it -- the next synced row's token, or for
    the last row the anchor the proposal names -- across prompt chunks, a forced phrase, verified paths
    and single decode steps; the backlog past depth + 1 rows is primed in one call, in position order,
    and the last rows ride the draft's own feed;
  * the walk down a block (the shared engine/methods/walk.py) follows the target's picks and adds the
    target's token at the stop;
  * `kv.length` reads the state's length after a pending chain commit; zero resets; another length
    keeps the rows by length and clamps the pooled keys;
  * a forward that does not start at the state's length is refused;
  * the adapter's cold verbs: `cost()` from the state's sections, `settle()` lands a pending chain and
    the parked GDN rows, `discard_pending()` drops them, `trim_pending(n)` cuts a pending chain and
    refuses a cut inside rows the state absorbed.
"""

from __future__ import annotations

import os
import sys

import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import walk  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.methods.mtp_drafter import MTPTreeDrafter  # noqa: E402
from engine.methods.staircut import PriceModel, StairCut  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing  # noqa: E402
from engine.tree import DraftTree  # noqa: E402


class _Cfg:
    attention_layers = (3,)
    num_key_value_heads, head_dim, indexer_head_dim, has_indexer = 2, 8, 4, True
    ple_layers = {1: 0}
    eos_token_id, hc_count, hidden_size = 3, 1, 3


class _FN:
    """What the adapters read of FlashNextEngine, on the host."""

    device, dtype = "cpu", torch.float32

    def __init__(self):
        self.cfg = _Cfg()
        self.hash = {1: types.SimpleNamespace(context_len=2)}
        self.pools: list = []
        self.primed: list = []
        self.resets = 0
        self.commits: list = []
        self._pend_owner = None
        self._pend_n = torch.ones(2, dtype=torch.int32)
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        self._pend_n.zero_()
        self._pend_owner = None

    def new_state(self, cap=None, paged=False):
        cap = cap or 64
        st = FNState(cap=cap)
        st.S[0] = torch.zeros(1, 2, 4, 4)
        st.conv[0] = torch.zeros(1, 3, 3)
        st.ple_conv[1] = torch.zeros(1, 3, 9)
        st.k[3], st.v[3], st.ik[3] = torch.zeros(2, cap, 8), torch.zeros(2, cap, 8), torch.zeros(cap, 4)
        return st

    def new_mtp_state(self, cap=None, pages=None):
        cap = cap or 64
        st = FNState(cap=cap)
        st.k[-1], st.v[-1], st.ik[-1] = torch.zeros(2, cap, 8), torch.zeros(2, cap, 8), torch.zeros(cap, 4)
        return st

    def build_pool(self, st, layers=None):
        self.pools.append((st.length, layers))

    def mtp_prime(self, stack, nexts, st_m):
        self.primed.append((stack[:, 0].tolist(), list(nexts), st_m.length))
        st_m.length += len(nexts)

    def reset(self, st):
        self.resets += 1
        st.reset_inplace()

    def commit(self, st, path, graphs=None):
        self.commits.append(list(path))
        st.length += len(path)
        return torch.full((len(path), 3), float(st.length))


def _rows(pos, n):
    """Streams whose first value is the row's position (so a test reads where a row went)."""
    return torch.arange(pos, pos + n, dtype=torch.float32)[:, None].repeat(1, 3)


def test_drafter_pairs_rows_with_their_next_token():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    d = MTPTreeDrafter(s, s.add_mtp_head(2), StairCut(PriceModel()))
    assert s.draft_heads()["mtp"] is d.head
    prompt = list(range(100, 110))
    d.sync(prompt[:6], _rows(0, 6), 0)                 # a prompt in two chunks
    d.sync(prompt[6:], _rows(6, 4), 6)
    assert d._feed(500)                                # the first token after the prompt
    # 10 rows: 7 primed at once in position order, the last 3 (depth + 1) for the draft's feed
    assert fn.primed == [([0.0, 1, 2, 3, 4, 5, 6], prompt[1:8], 0)]
    rows, nexts = d.pending
    assert rows[:, 0].tolist() == [7, 8, 9] and nexts == prompt[8:] + [500]
    d.pending = None
    d.head.st.length += 3                               # what the draft's own feed does
    # a verified path (anchor 500 and two drafts), then a forced phrase, then one decode step
    d.sync([500, 501, 502], _rows(10, 3), 10)
    d.sync([503, 504], _rows(13, 2), 13)
    d.sync([505], _rows(15, 1), 15)
    assert d._feed(506)
    assert fn.primed[-1] == ([10.0, 11, 12], [501, 502, 503], 10)
    rows, nexts = d.pending
    assert rows[:, 0].tolist() == [13, 14, 15] and nexts == [504, 505, 506]
    d.reset()
    assert d.head.st.length == 0 and d.held is None and d.queue == [] and not d._feed(1)
    return "prompt chunks, forced phrase, decode steps: each row with its next token; backlog primed in order"


class _StubHead:
    """An MTP head that always drafts tokens 10 + k (p 0.5) and 20 + k (p 0.1) at depth k."""

    depth = 3

    def __init__(self):
        self.st = types.SimpleNamespace(length=0)

    def reset(self):
        self.st.length = 0

    def prime(self, rows, nexts):
        self.st.length += len(nexts)

    def launch(self, rows, nexts):
        self.st.length += len(nexts)

    def read(self):
        return [[10 + k, 20 + k] for k in range(self.depth)], [[0.5, 0.1]] * self.depth


def test_the_lookup_joins_the_mtp_round():
    """ENG-229: with a lookup the round's tree carries the head's chain and the lookup's copy of the
    request's own text (one verify); the lookup is primed with the prompt, follows what is committed
    and is reset with the drafter; without a match the round is the head's alone."""
    from engine.drafters.ngram import NgramDrafter
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16)
    d = MTPTreeDrafter(eng, _StubHead(), StairCut(PriceModel(), siblings=False), ng)
    text = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 600, 601]
    prompt = text + [700] + text[:4]                    # the request repeats its own text
    d.reset()
    d.prime(prompt)
    d.sync(prompt, _rows(0, len(prompt)), 0)
    tree = d.propose_tree(prompt, 32)
    tree.check()
    copy = text[4:] + [700] + text[:4]                 # what followed the last match, the lookup's draft
    assert tree.tokens[1:1 + len(copy)] == copy and tree.parents[1:1 + len(copy)] == list(range(len(copy))), tree.tokens
    assert tree.tokens[1 + len(copy):] == [10, 11, 12] and tree.parents[1 + len(copy)] == 0  # the head's chain
    assert d.stats["looked"] == 1
    d.observe([504, 505])                               # committed: the lookup's index follows
    assert ng.local.lookup(prompt + [504, 505], 3)[0] >= 3
    d.reset()
    assert ng.local.lookup(prompt, 3) == (0, [])
    d.prime([1, 2, 3, 4])
    d.sync([1, 2, 3, 4], _rows(0, 4), 0)
    tree = d.propose_tree([1, 2, 3, 4], 32)
    assert tree.tokens == [4, 10, 11, 12] and d.stats["looked"] == 1, tree.tokens
    return "the copy and the head's chain in one tree; primed, observed, reset; no match: the head alone"


def test_counting_the_state_issues_no_copy():
    """2026-10-12: /health and the dashboard (other threads) count the state's bytes through `cost()`;
    inside a graph capture a copy from another thread invalidates the capture, so counting reads the
    buffers as they are, and only `sections()` (the caches, on the engine's thread) brings the PLE
    context up to date."""
    fn = _FN()
    s = FlashNextServing(fn, 64)
    s.st.ple_ctx = {1: torch.tensor([7, 8])}
    before = s._ctx[1].clone()
    s.cost()
    assert torch.equal(s._ctx[1], before)
    assert s.state.sections()["ple.ctx.1"].tolist() == [[7, 8]]
    return "cost() leaves the buffers alone; sections() refreshes the PLE context"


def test_a_class_where_drafts_do_not_pay_takes_the_plain_step():
    """P0 (2026-10-12): the drafter reads the engine's `price_class`, prices its rounds by wall time (a
    plain step's span, a drafted round's span a token), and where drafting costs more a token than the
    plain step it returns no draft, keeping the rows for the next round that drafts."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0, price_class=lambda n: 262_144 if n > 100 else 0)
    head = _StubHead()
    d = MTPTreeDrafter(eng, head, StairCut(PriceModel(), siblings=False))
    long = list(range(200))
    d.reset()
    d.sync(long, _rows(0, len(long)), 0)
    assert d.propose_tree(long, 8).n_draft > 0                    # no prices for the modes: draft
    for _ in range(30):
        d.cut.observe_plain(25.0, 262_144)
        d.cut.observe_spec(30.0, 262_144)
    d.sync([500], _rows(len(long), 1), len(long))
    assert d.propose_tree(long + [500], 8) is None and d.stats["skipped"] == 1
    assert d.queue and d.held is None                              # the skipped round's rows wait
    d.on_plain(0.0)
    d.observe([500])
    d.sync([501], _rows(len(long) + 1, 1), len(long) + 1)
    d.propose_tree(long + [500, 501], 8)
    assert 262_144 in d.cut.plain and d.stats["plain"] == 1       # the plain step priced at the next start
    sp = d.req["spans"]                                            # the report's spans: what priced the time
    assert sp["plain"][0] == 1 and sp["unpriced"][0] == 1 and "spans=" in d.report(), sp
    short = list(range(50))
    d.reset()
    d.sync(short, _rows(0, 50), 0)
    assert d.propose_tree(short, 8).n_draft > 0 and d.cls == 0     # another class drafts as before
    return "plain where drafting costs more a token, rows kept; the plain step priced; class 0 untouched"


def test_a_draft_cut_to_the_anchor_is_a_drafted_round():
    """P0 live check (2026-10-12): a round that drafted but whose cut kept the anchor alone takes the
    plain step after paying for the draft; priced as a plain step it put the plain step at 25.7 ms
    against 21.6 in a plain turn. It is a drafted round of one token: `spec` learns it, `plain` not."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0, price_class=lambda n: 0)
    head = _StubHead()
    head.read = lambda: ([[10 + k, 20 + k] for k in range(3)], [[0.02, 0.01]] * 3)
    d = MTPTreeDrafter(eng, head, StairCut(PriceModel({"0": {"chain": {"1": 20.0, "2": 40.0}}}), siblings=False))
    prompt = list(range(50))
    d.reset()
    d.sync(prompt, _rows(0, 50), 0)
    assert d.propose_tree(prompt, 8).n_draft == 0                  # drafted, cut to the anchor
    d.on_plain(21.0)
    d.observe([100])
    d.sync([100], _rows(50, 1), 50)
    d.propose_tree(prompt + [100], 8)
    assert 0 in d.cut.spec and 0 not in d.cut.plain and d.req["spans"]["anchor"][:1] == [1], d.req["spans"]
    # the server's idle captures ask whether drafting can pay at a length (`wants_graphs`)
    d.cut = StairCut(PriceModel(), siblings=False)                # (the host clock's rounds above took ~0 ms)
    assert d.wants_graphs(50)
    d.cut.observe_plain(20.0, 0)
    for _ in range(40):
        d.cut.observe_spec(30.0, 0, 1)
    assert not d.wants_graphs(50)
    return "a draft cut to the anchor: a drafted round of one token, not a plain step; wants_graphs"


def test_a_turns_backlog_prime_is_not_its_first_rounds_price():
    """P0 (2026-10-12): the first drafted round of a turn at 140k read the turn's backlog into the head
    (772 rows, 200 ms) and was priced at 93 ms; the turn weighs the prime (`StairCut.turn`), the round's
    price is its wall time less the prime."""
    import time as _t
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0, price_class=lambda n: 0)
    head = _StubHead()
    head.prime = lambda rows, nexts: (_t.sleep(0.08), setattr(head.st, "length", head.st.length + len(nexts)))
    d = MTPTreeDrafter(eng, head, StairCut(PriceModel(), siblings=False))
    prompt = list(range(200))
    d.reset()
    d.sync(prompt, _rows(0, 200), 0)
    tree = d.propose_tree(prompt, 8)
    assert tree.n_draft > 0 and d._prime_ms >= 80.0
    d.on_verify(tree.n_draft + 1, 1.0)
    d.observe([10, 11])
    d.sync([10, 11], _rows(200, 2), 200)
    d.propose_tree(prompt + [10, 11], 8)
    assert d.cut.spec[0] < 40.0 and d._prime_ms == 0.0, (d.cut.spec, d._prime_ms)
    return f"a turn's 80 ms prime left out of its first round's price ({d.cut.spec[0]:.1f} ms a token)"


def test_the_warm_up_rounds_price_nothing():
    """P0 (2026-10-12): the first request after a start drafted its 1k rounds at 22.4 ms a token
    against 17.8 later (graph captures, first kernel calls) and 1k took the plain step after it. The
    server's warm-up now decodes `WARM_TOKENS` through the rounds inside `warming`: the cut learns
    nothing, keeps no turn or probe state, and the usual answer length stays."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0, price_class=lambda n: 0)
    d = MTPTreeDrafter(eng, _StubHead(), StairCut(PriceModel(), siblings=False))
    prompt = list(range(50))
    with d.warming():
        assert d.cut.frozen
        d.reset()
        d.sync(prompt, _rows(0, 50), 0)
        for i in range(4):
            d.propose_tree(prompt + [100 + j for j in range(i)], 8)
            d.on_verify(3, 500.0)
            d.observe([100 + i])
            d.sync([100 + i], _rows(50 + i, 1), 50 + i)
            d.cut.observe_plain(400.0, 0)
            d.cut.observe_spec(900.0, 0, 1)
            d.cut.skip(0)
    assert not d.cut.frozen and d.cut.plain == {} and d.cut.spec == {} and d.cut.prices.V == {}
    assert d.cut.skipped == {} and d.cut.plain_turns == {} and d.answer == 256.0 and d.req["tokens"] == 0
    assert d.cut.draft == {}
    return f"{MTPTreeDrafter.WARM_TOKENS} warm-up tokens through the rounds: nothing priced, no state kept"


def test_a_plain_turn_reads_nothing_and_the_next_resumes_past_it():
    """P0: a turn whose saving does not pay for its backlog takes the plain step throughout, drops the
    backlog and reads no row; the next turn that drafts puts the head at the first row it has not read
    (rows it skipped stay unread: they change a draft, never the text)."""
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0, price_class=lambda n: 262_144)
    head = _StubHead()
    resumed = []
    head.resume = lambda n: (resumed.append(n), setattr(head.st, "length", n))
    d = MTPTreeDrafter(eng, head, StairCut(PriceModel({"262144": {"chain": {"1": 60.0, "2": 70.0},
                                                                    "plain": {"1": 24.7}, "spec": {"1": 25.4},
                                                                    "prime": {"1": 0.28}}}), siblings=False))
    prompt = list(range(5000))
    d.reset()
    d.sync(prompt, _rows(0, 5000), 0)
    assert d.propose_tree(prompt, 8) is None and d.req["turn"] is False and d.plain_turn
    assert d.queue == [] and d.held is None and head.st.length == 0      # the backlog dropped, nothing read
    d.sync([9], _rows(5000, 1), 5000)
    assert d.held is None                                               # rows of a plain turn are not kept
    assert d.propose_tree(prompt + [9], 8) is None and d.stats["skipped"] == 2
    # the next request: drafting pays now (learned), the head resumes past the rows it skipped
    for _ in range(40):
        d.cut.observe_spec(10.0, 262_144)
    d.reset()
    nxt = prompt + [9, 10, 11]
    d.sync([10, 11], _rows(5001, 2), 5001)
    tree = d.propose_tree(nxt, 8)
    assert resumed == [len(nxt) - 1 - 2] and tree is not None and d.req["turn"] is True, (resumed, tree)
    return "a plain turn drops and skips; the next drafting turn resumes the head at 5,001"


def test_the_price_classes():
    """Dense contexts are class 0; past that the classes run 4,096, 6,144, 8,192, 12,288 ... 262,144 (a
    verify's cost grows with the context inside the sparse attention's own classes)."""
    s = types.SimpleNamespace(fn=types.SimpleNamespace(qsa_class=lambda n: 0 if n <= 2048 else 1024))
    pc = lambda n: FlashNextServing.price_class(s, n)
    got = [pc(n) for n in (1000, 2049, 5000, 32_000, 40_000, 140_000, 200_000, 260_800)]
    assert got == [0, 4096, 6144, 32_768, 49_152, 196_608, 262_144, 262_144], got
    return "0, then 4,096 x 2^k and 6,144 x 2^k up to 262,144"


def test_idle_captures_take_the_shapes_the_rounds_use():
    """The capture gap (2026-10-12): a class's verify, commit and MTP graphs, the shapes the rounds so
    far used (95 % of them), captured one a step while the server is idle, a moved commit over a
    tree whose path leaves the chain; until then a drafted turn there is charged for them."""
    from engine.models.flashnext import serving as S
    fn = _FN()
    fn.qsa_class = lambda n: 0 if n < 100 else 1024
    staged, made = [], []

    class VG:
        def __init__(self, fn_, st, M, cls):
            self.key = (M, cls)

        def capture(self):
            made.append(("v",) + self.key)

        def replay(self):
            made.append(("replay",) + self.key)

    class CG:
        def __init__(self, fn_, st, key):
            self.key = key

        def capture(self):
            made.append(("c",) + self.key)

    fn.stage_rows = lambda toks, parents, st: staged.append(("rows", list(parents)))
    fn.stage_commit = lambda st, path: staged.append(("commit", list(path)))
    head = types.SimpleNamespace(depth=1, graphs={(1, 0): 1, (2, 0): 1})
    head.capture_class = lambda n, cls: head.graphs.__setitem__((n, cls), 1)
    s = FlashNextServing(fn, 64)
    s.heads["mtp"] = head
    assert s.capture_cost(500) > 0 and s.capture_cost(0) > 0            # the prior's shapes owed
    s.uses = {2: 500, 3: 300, 5: 1, (2, 1, False): 300, (2, 2, False): 200, (3, 2, True): 250, (3, 3, False): 50}
    s.vgraphs = {(2, 0): 1, (3, 0): 1, (5, 0): 1}
    s.cgraphs = {k + (0,): 1 for k in s.uses if isinstance(k, tuple)}
    s.st.length = 500
    assert s._top(False) == [2, 3]                                       # width 5: 0.1 % of the rounds
    assert s.capture_cost(50) == 0.0                                      # class 0 has them
    assert s.capture_cost(500) == 8 * S.CAPTURE_MS                       # 2 widths, 4 commits, 2 feeds
    real, S.VerifyGraph, S.CommitGraph = (S.VerifyGraph, S.CommitGraph), VG, CG
    sync, torch.cuda.synchronize = torch.cuda.synchronize, lambda: None
    try:
        n = sum(1 for _ in s.idle_captures())
    finally:
        S.VerifyGraph, S.CommitGraph = real
        torch.cuda.synchronize = sync
    assert n == 8 and made[:2] == [("v", 2, 1024), ("v", 3, 1024)], made
    assert ("c", 3, 2, True, 1024) in made and ("rows", [-1, 0, 0]) in staged and ("commit", [0, 2]) in staged
    assert (1, 1024) in head.graphs and (2, 1024) in head.graphs
    assert s.idle_captures() is None and s.capture_cost(500) == 0.0 and s.capture_ms is not None
    assert s.st.length == 500
    # a graph captured ahead uploads at its first replay: that round is not priced, the next one is
    fn._vcls, s._width = 1024, 2
    s.commit_tree([0])
    assert s.captured and (2, 1, False, 1024) not in s._fresh
    s.captured = False
    s.st.length = 500
    s.commit_tree([0])
    assert not s.captured
    # after the server's warm-up: the state starts over and class 0's graphs are taken at once
    s.st.length = 500
    s.idle_captures = lambda: iter([1, 2, 3]) if s.st.length == 0 else None
    assert s.precapture(None) is None and s.precapture(object()) == 3 and s.st.length == 0
    return ("8 graphs of the 1,024 class (2 widths, 4 commits, a moved one over a tree, 2 feeds); then nothing owed; "
            "a first replay unpriced; class 0 right after the warm-up")


def test_a_capture_saves_only_what_its_warm_up_writes():
    """ENG-241: a capture's warm-up writes the recurrent sections and rows from the length on; it saved
    every buffer whole (8 GB at a 262,144-token window), and with the KV pool rows past the committed
    ones are not mapped. Now the sections whole, `CAPTURE_ROWS` rows from the length, the pooled-key
    blocks from the length's block; a restore puts back exactly what was there."""
    from engine.models.flashnext import engine as E
    fn = _FN()
    fn.cfg.indexer_compress_ratio = 4
    st = fn.new_state(400)
    st.pool[3] = torch.zeros(400 // 4 + 9, 4)
    for t in [st.S[0], st.conv[0], st.ple_conv[1], st.k[3], st.v[3], st.ik[3], st.pool[3]]:
        t.copy_(torch.randn_like(t))
    st.length = 100
    before = {k: v.clone() for k, v in E._state_tensors(st).items()}
    saved = E._state_saved(fn, st)
    rows = {f"k.3": 64, f"v.3": 64, f"ik.3": 64}
    for t, ax, lo, n, x in saved:
        name = next(k for k, v in E._state_tensors(st).items() if v is t)
        assert n == rows.get(name, n) and (name not in rows or lo == 100), (name, lo, n)
    st.S[0].add_(1.0)
    st.k[3][:, 100:164].add_(1.0)
    st.ik[3][120].add_(1.0)
    st.pool[3][25:30].add_(1.0)
    E._state_restore(saved)
    assert all(torch.equal(before[k], v) for k, v in E._state_tensors(st).items())
    kv = sum(x.numel() for t, ax, lo, n, x in saved if t is st.k[3])
    return f"the K rows saved: {kv} values of {st.k[3].numel()}; restored exactly"


def test_the_pool_commits_rows_before_a_write():
    """ENG-241: with the KV pool the state's rows are committed before a forward or a verify block
    writes them (rows plus the pool's margin, one compare when they are)."""
    fn = _FN()
    s = FlashNextServing(fn, 64)
    asked = []
    s.st.pages = types.SimpleNamespace(rows=0, margin=lambda: 8, on_device=lambda: False,
                                       ensure=lambda r: (asked.append(r), setattr(s.st.pages, "rows", r + 8)))
    assert s.kv.pages is s.st.pages
    s._ensure(10)
    s._ensure(9)                                    # committed already (10 rows and the margin): no call
    s._ensure(20)
    assert asked == [10, 20], asked
    return "rows 10 then 20 committed; 9 already"


def test_a_side_slot_shares_the_weights_and_has_its_own_state():
    """ENG-241 step 2: a side slot over the same engine (`fn`, the weights once) with its own state,
    graphs and MTP head; its drafter's StairCut from the same prices (its own learning) and its lookup
    over slot 0's corpus; its MTP head drafts into its own buffers."""
    fn = _FN()
    s0 = FlashNextServing(fn, 64)
    assert s0.caps.side_slots and s0.caps.joint_verify
    ng0 = types.SimpleNamespace(corpus=object())
    d0 = MTPTreeDrafter(s0, s0.add_mtp_head(3), StairCut(PriceModel({"0": {"chain": {"1": 20.0}}}, 2.0),
                                                         siblings=False), ng0)
    d0.cut.observe_spec(30.0, 0)
    real = MTPTreeDrafter.capture_all
    MTPTreeDrafter.capture_all = lambda self: None
    try:
        e2, d2 = s0.side_slot(32, d0)
    finally:
        MTPTreeDrafter.capture_all = real
    assert e2.fn is fn and e2.st is not s0.st and e2.max_len == 32 and e2.vgraphs is not s0.vgraphs
    assert d2.s is e2 and d2.head is e2.heads["mtp"] and d2.head is not d0.head and d2.depth == 3
    assert (d0.head.key, d2.head.key) == (0, 1)                    # its own draft buffers
    assert d2.cut is not d0.cut and d2.cut.prices.table == d0.cut.prices.table and d2.cut.spec == {}
    assert d2.ngram is not None and d2.ngram.corpus is ng0.corpus
    per_token, const = s0.side_slot_bytes(32, paged=False)
    assert per_token > s0.cost().view_bytes_per_token and const > s0.cost().section_bytes
    assert s0.side_slot_bytes(32, paged=True)[0] == 0
    return "own state, graphs, head and cut; the same weights, prices and corpus; bytes by window"


def test_a_joint_pass_leaves_each_slot_as_its_lone_verify_would():
    """ENG-241 step 3: `forward_tree_multi` hands every slot's block to the engine's one pass
    (`verify_multi`) and leaves each slot's logits, picks and streams where its own `forward_tree`
    leaves them; the commit is each slot's own. Blocks over 32 rows are not batchable."""
    fn = _FN()
    seen = []

    def verify_multi(jobs):
        seen.append([(t, p, st) for t, p, st in jobs])
        return [(torch.full((len(t), 4), float(i)), torch.arange(len(t)) + 10 * i, torch.zeros(len(t), 3) + i)
                for i, (t, p, st) in enumerate(jobs)]
    fn.verify_multi = verify_multi
    a, b = FlashNextServing(fn, 64), FlashNextServing(fn, 64)
    a.st.length, b.st.length = 5, 9
    real = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: types.SimpleNamespace(synchronize=lambda: None)
    try:
        out = FlashNextServing.forward_tree_multi([(a, torch.tensor([7, 8]), [-1, 0], 5),
                                                    (b, torch.tensor([1, 2, 3]), [-1, 0, 0], 9)])
    finally:
        torch.cuda.current_stream = real
    assert seen == [[([7, 8], [-1, 0], a.st), ([1, 2, 3], [-1, 0, 0], b.st)]]
    assert out[0].shape == (2, 4) and float(out[1][0, 0]) == 1.0
    assert a.picks.tolist() == [0, 1] and b.picks.tolist() == [10, 11, 12] and b._hidden.shape == (1, 3, 3)
    assert a._width == 2 and b._width == 3 and a.stats["verify_blocks"] == 1 and not b.captured
    assert a.tree_batchable(list(range(-1, 31))) and not a.tree_batchable(list(range(-1, 32)))
    return "one pass for both slots; each slot's logits, picks and streams its own; 33 rows not batchable"


def test_walk_and_kv_length():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    tree = DraftTree(tokens=[7, 8, 9, 4, 5], parents=[-1, 0, 1, 0, 3])
    assert walk.accept_tree(tree, [8, 9, 1, 0, 0]) == ([0, 1, 2], [8, 9, 1])
    assert walk.accept_tree(tree, [4, 0, 0, 5, 6]) == ([0, 3, 4], [4, 5, 6])
    assert walk.accept_tree(tree, [2, 0, 0, 0, 0]) == ([0], [2])
    s._chain = 3                                       # a chain block whose commit waits
    assert s.kv.length == 3 and fn.commits == [[0, 1, 2]] and s._chain is None
    s.st.pool_n = {3: 5}
    s.kv.length = 2
    assert s.st.length == 2 and s.st.pool_n[3] == 0
    s.st.k[3][:, 1] = 5.0                              # a row in place (a resident conversation's)
    s.kv.length = 0
    # a new sequence in place: the length and the recurrent sections, never the rows (a guest runs from
    # row 0 over a resident conversation whose rows its stash may not hold, engine/cache.py)
    assert fn.resets == 0 and s.st.length == 0 and float(s.st.k[3][0, 1, 0]) == 5.0
    s.st.S[0].fill_(1.0)
    s.reset()
    # ENG-157: a reset is a new sequence too (the cache's cold prefill of a guest calls it), as the
    # 27B's: the recurrent sections cleared, the rows kept
    assert fn.resets == 0 and float(s.st.S[0].abs().max()) == 0.0 and float(s.st.k[3][0, 1, 0]) == 5.0
    try:
        s.forward(torch.tensor([1]), start=5)
        raise AssertionError("a forward off the state's length must be refused")
    except ValueError:
        pass
    cost = s.cost()
    sc = cost.snapshot_cost()
    assert sc["recurrent_bytes"] == 2 * 4 * 4 * 4 and sc["kv_bytes_per_token"] == (2 * 8 * 2 + 4) * 2
    # the GDN conv tails and the PLE conv state; the sections add the PLE context and the MTP input
    assert sc["conv_bytes"] == (9 + 27) * 4
    assert cost.section_bytes == sc["recurrent_bytes"] + sc["conv_bytes"] + 2 * 8 + 3 * 4
    return "walk with stops at the root and deep; pending chain committed on read; reset; pool clamp; refusal; cost"


def test_pending_verbs():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    s.st.length = 10
    s._chain = 4                                       # a chain block at 10..13 whose commit waits
    assert s.trim_pending(12) and fn.commits == [[0, 1]] and s.st.length == 12 and s._chain is None
    s._chain = 3
    assert s.trim_pending(12) and fn.commits == [[0, 1]] and s.st.length == 12   # none kept: dropped
    assert s.trim_pending(12) and not s.trim_pending(11) and s.st.length == 12   # absorbed: refused
    s._chain = 2
    assert not s.trim_pending(20) and s.st.length == 12 and s._chain == 2       # past the chain: no
    fn._pend_owner = s.st                              # parked GDN rows of this state
    s.settle()
    assert fn.commits[-1] == [0, 1] and fn.flushes == 1 and fn._pend_owner is None and s.st.length == 14
    fn._pend_owner, fn._pend_n[:] = s.st, 3
    s._chain, s.picks = 2, [1, 2]
    s.discard_pending()
    assert s._chain is None and s.picks is None and fn._pend_owner is None and int(fn._pend_n.sum()) == 0
    other = FNState(cap=8)
    fn._pend_owner = other                             # another state's parked rows are not ours
    s.discard_pending()
    s.settle()
    assert fn._pend_owner is other and fn.flushes == 1
    return "trim inside a pending chain, drop it, refuse absorbed rows; settle lands chain and parked rows; discard"


def test_the_cache_state_and_a_resume():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    sec = s.cache_state().sections()
    assert list(sec) == ["gdn.S.0", "gdn.conv.0", "ple.conv.1", "ple.ctx.1", "mtp.h_last"], list(sec)
    assert sec["ple.ctx.1"].tolist() == [[3, 3]]                     # no context yet: the end-of-text id
    assert [n for n, _, _ in s.cache_state().views()] == ["k.3", "v.3", "ik.3"]
    s.st.ple_ctx[1] = torch.tensor([[5, 6]])
    s.st.length = 4
    s.commit_tree([0, 1])                                             # h_last: the path's last row
    assert s.last_streams().tolist() == [[6.0, 6.0, 6.0]]
    kept = tuple(t.clone() for t in s.cache_state().sections().values())
    assert kept[3].tolist() == [[5, 6]]
    s.st.ple_ctx[1] = torch.tensor([[9, 9]])
    s.st.S[0].fill_(7.0)
    for t, x in zip(s.cache_state().sections().values(), kept):      # what `state_load` does
        t.copy_(x)
    s.resume_at(6)
    assert s.st.ple_ctx[1].tolist() == [[5, 6]] and float(s.st.S[0].abs().sum()) == 0.0
    assert s.st.length == 6 and fn.pools[-1] == (6, None)
    assert s.cost().section_bytes == sum(t.numel() * t.element_size() for t in kept)
    return "5 sections (the PLE context and the MTP input among them), 3 views; a resume restores them"


def test_the_drafter_resumes_and_snapshots_its_head():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    d = MTPTreeDrafter(s, s.add_mtp_head(2), StairCut(PriceModel()))
    assert [t.shape for t, _ in d.kv_views()] == [(2, 64, 8), (2, 64, 8), (64, 4)]
    d.sync(list(range(100, 110)), _rows(0, 10), 0)
    d.sync([110, 111], _rows(10, 2), 10)                              # rows 0..9 now know their next token
    snap = d.state_snapshot()
    assert snap["len"] == 10 and fn.primed[-1][2] == 0 and len(fn.primed[-1][1]) == 10
    assert d.held[0] == [110, 111] and d.queue == []
    s.st.length = 12
    s._h_last.fill_(11.0)
    d.state_resume(12)                                                # a resume in place at 12
    assert d.head.st.length == 11 and fn.pools[-1] == (11, [-1])
    assert d.held[1].tolist() == [[11.0, 11.0, 11.0]] and d.pending is None and d.queue == []
    d.state_restore(snap)
    assert d.head.st.length == 10 and fn.pools[-1] == (10, [-1])
    d.state_resume(0)
    assert d.held is None and d.head.st.length == 0
    return "the head's views; a snapshot feeds the queue first; resume and restore set its length and the waiting row"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
