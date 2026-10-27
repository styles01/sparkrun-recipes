"""SRV-530 (QA SRV-531): a block's rows past the end token are not committed, so a turn that ends
with a tool call keeps its end anchor.

THE MEASUREMENT (2026-10-16, Super VL EXL3 4.0, `--parallel 4`, engine-20261016-223733.log): 31 of
54 turns ended with `session put declined, the engine is ahead of the committed tokens` and `end
anchor not kept (ahead)`, `kv.length - ctx` +1 on 30 and +2 on one, every one a `tool_calls`
finish. The lookup drafts what followed the same tool call earlier in the prompt (`</tool_call>`,
`<|im_end|>`, `\\n`, `<|im_start|>`) and the target accepts the newline after its end token: the
walk's path ran one or two rows past the end, the loop committed the whole path and returned at the
end token. NemotronH's state had walked those rows (`trim_pending` cannot cut a committed row
back), so no end anchor was kept, and the next turn was routed as a new conversation (SRV-528).

THE FIX (server/app.py `_end_cut`): the commit stops at the last kept token's row -- the end token,
or the last token the budget allows -- in the tree and the chain loops alike. That is the length
Qwen's pending commit was cut back to (`trim_pending`), so nothing changes where the cut back
worked; the text is the same, as the rows past the end were never sent.

  1  Qwen (CPU, the random model): a drafter that runs 1 or 2 rows past the end token, tree and
     chain: the end anchor is kept at the committed tokens; the engine's state, the rows and the
     next turn are bit for bit those of a drafter that never drafted past the end; the text is the
     plain greedy text cut at the end token; unfolded commits (the CPU default) were refused before
  2  the budget: a generation that reaches max_tokens inside a block keeps its end anchor too
  3  memory (SRV-655): refused new buffers, the end anchor takes an anchor's and the next turn
     resumes at the end; with no anchor to take the refusal stays a refusal (`memory`,
     `end_refused`), said with the policy's figures
  4  NemotronH (GPU, the tiny board engine through `NemotronHServing`): the incident's +1 and +2;
     before the fix `ahead`, now kept, the state bit for bit the never-past-the-end run's, the next
     turn resumed at the end with the same tokens and logits as that run, and the same greedy tokens
     as the turn read without the end anchor (the end anchor's gate)

Run: python tests/test_end_cut.py
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, _state, folded  # noqa: E402
from test_resident import same_state  # noqa: E402

PROMPT = tokens(100, seed=60)


class Past:
    """Drafts the plain greedy text `width` tokens a round, every one accepted, past the end token
    unless `stop_at` cuts the draft there (the run that never drafted past it)."""

    def __init__(self, seq, width, stop_at=None):
        self.seq, self.width, self.stop_at = list(seq), width, stop_at

    def reset(self):
        pass

    def observe(self, toks):
        pass

    def _draft(self, ctx):
        toks = self.seq[len(ctx):len(ctx) + self.width]
        if self.stop_at is not None and self.stop_at in toks:
            toks = toks[:toks.index(self.stop_at) + 1]
        return toks

    def propose(self, ctx, count):
        return self._draft(ctx)

    def propose_tree(self, ctx, count):
        toks = self._draft(ctx)
        return DraftTree.chain(ctx[-1], toks) if toks else None


def run(ids, n, eos, session="ses"):
    """One request as the handler runs it, with an end token: the stream, then `_remember`."""
    log = io.StringIO()
    with redirect_stdout(log):
        out = list(app.generate_stream(torch.tensor(ids), n, set(eos), None, session))
        ctx = list(app.STATE["last_ctx"])
        pre = dict(app.STATE["last_prefill"])
        kv = int(app.STATE["engine"].kv.length)
        app._remember(list(ids), out, session)
    return out, pre, ctx, kv, log.getvalue()


def _greedy(n=80):
    """The plain greedy text, and an end token that first appears at least 12 tokens in."""
    _state()
    out = list(app.generate_stream(torch.tensor(PROMPT), n, set(), None, "ses"))
    i = next(i for i in range(12, n) if out[i] not in out[:i])
    return out, i


def _width_for(i: int, past: int) -> int:
    """A draft width whose block holds the end token (out[i]) as a drafted row with `past` drafted
    rows after it: the first token comes from the prefill, then every block adds width + 1."""
    for w in range(2, 12):
        off = (i - 1) % (w + 1)
        if off <= w - 1 and w - 1 - off == past:
            return w
    raise AssertionError(f"no width puts {past} rows past token {i}")


class no_cut:
    """The 0.11.2 loop: every row of the accepted path committed (`_end_cut` keeps them all)."""

    def __enter__(self):
        self.real = app._end_cut
        app._end_cut = lambda new, eos, room: len(new)

    def __exit__(self, *exc):
        app._end_cut = self.real
        return False


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _case(tree: bool, past: int, labels=("cut", "base")):
    """The same turn, the drafter `past` rows past the end token: with the cut, and as 0.11.2 ran it
    (`base`) or with a drafter that stops at the end token (`ref`)."""
    plain, i = _greedy()
    E = plain[i]
    seq = PROMPT + plain
    w = _width_for(i, past)
    got = {}
    for label in labels:
        dr = Past(seq, w, stop_at=E if label == "ref" else None)
        res = _state(drafter=dr, tree=tree, k=w)
        with (no_cut() if label == "base" else _nothing()):
            out, pre, ctx, kv, log = run(PROMPT, 60, {E})
        got[label] = (res, app.STATE["engine"], out, ctx, kv, log)
    return plain, i, E, got


def _next(res, eng, ids, n=16):
    app.STATE.update(engine=eng, resident=res, drafter=None)
    out, pre, _, _, _ = run(ids, n, set())
    return out, pre


# ------------------------------------------------------------------ 1. Qwen, CPU
def _check_case(tree: bool, past: int):
    """Unfolded commits (the CPU default): 0.11.2 refused the end anchor (`ahead`); the cut keeps it
    at the committed tokens, the text the same, the next turn resumed at the end and writing the
    greedy tokens of the turn read without it (the end anchor's gate)."""
    plain, i, E, got = _case(tree, past)
    res, eng, out, ctx, kv, log = got["cut"]
    bres, beng, bout, bctx, bkv, blog = got["base"]
    assert out == bout == plain[:i + 1], "the text: the greedy text through the end token"
    assert bkv == len(bctx) + past and bres.end is None and bres.stats["end_ahead"] == 1, \
        (bkv, len(bctx), bres.stats)
    assert kv == len(ctx) and res.end is not None and res.end["at"] == len(ctx), (kv, res.end)
    assert res.stats["end_cut"] == 1 and res.stats["end_ahead"] == 0, res.stats
    nxt = PROMPT + out + TOOL
    out2, pre2 = _next(res, eng, nxt)
    bout2, bpre2 = _next(bres, beng, nxt)
    assert pre2["kind"] == "end" and pre2["reused"] == len(ctx), pre2
    assert pre2["forwarded"] == len(TOOL), pre2
    assert bpre2["kind"] == "resident" and out2 == bout2, (bpre2, out2, bout2)
    return (f"{past} row(s) past the end at token {i}: 0.11.2 ahead by {bkv - len(bctx)}, now kept "
            f"at {len(ctx)}; next turn forwards {pre2['forwarded']} instead of {bpre2['forwarded']}")


def test_a_tree_past_the_end_by_one_keeps_the_end_anchor():
    return _check_case(tree=True, past=1)


def test_a_tree_past_the_end_by_two_keeps_the_end_anchor():
    return _check_case(tree=True, past=2)


def test_a_chain_past_the_end_keeps_the_end_anchor():
    return _check_case(tree=False, past=1)


def test_the_folded_commit_is_bit_for_bit_what_its_cut_back_was():
    """The served Qwen (`QWEN38_COMMIT_IN_VERIFY`): 0.11.2 cut a tree's pending commit back to the
    committed tokens (`trim_pending`) and kept the end anchor; the shorter commit leaves the same
    state, rows and end anchor, and the next turn the same tokens and state."""
    out = []
    for tree in (True,):
        for past in (1, 2):
            with folded():
                plain, i, E, got = _case(tree, past)
                res, eng, o, ctx, kv, _ = got["cut"]
                bres, beng, bo, bctx, bkv, _ = got["base"]
                assert o == bo and res.end["at"] == bres.end["at"] == len(ctx), (res.end, bres.end)
                assert bres.stats["end_ahead"] == 0 and res.stats["end_cut"] == 1
                eng.settle()
                beng.settle()
                same_state(eng, beng, len(ctx))
                assert all(torch.equal(a, b) for a, b in zip(res._end_bufs, bres._end_bufs))
                nxt = PROMPT + o + TOOL
                o2, pre2 = _next(res, eng, nxt)
                bo2, bpre2 = _next(bres, beng, nxt)
                assert pre2 == {**bpre2, "ms": pre2["ms"]} and o2 == bo2, (pre2, bpre2)
                same_state(eng, beng, eng.kv.length)
            out.append(f"tree +{past}")
    # the chain's pending commit was not cut back (0.11.2 refused it as `ahead`): kept now
    with folded():
        plain, i, E, got = _case(False, 2)
    assert got["base"][0].stats["end_ahead"] == 1 and got["cut"][0].end["at"] == len(got["cut"][3])
    return ", ".join(out) + ": bit for bit; chain +2: refused before, kept now"


def test_the_log_says_the_rows_were_not_committed():
    plain, i = _greedy()
    w = _width_for(i, 1)
    res = _state(drafter=Past(PROMPT + plain, w), tree=True, k=w)
    app.STATE["verbose"] = True
    try:
        out, pre, ctx, kv, log = run(PROMPT, 60, {plain[i]})
    finally:
        app.STATE["verbose"] = False
    assert res.stats["end_cut"] == 1 and res.end is not None
    assert "[spec] the accepted path ran 1 row(s) past the end: not committed" in log, log
    assert "end anchor not kept" not in log and "ahead of the committed" not in log, log
    return "one [spec] line, no `ahead` line"


# ------------------------------------------------------------------ 2. the budget
def test_a_budget_reached_inside_a_block_keeps_the_end_anchor():
    plain, _ = _greedy()
    seq = PROMPT + plain
    for n in (13, 14, 15):                    # the budget ends inside a block of 5 rows
        res = _state(drafter=Past(seq, 4), tree=True, k=4)
        out, pre, ctx, kv, log = run(PROMPT, n, set())
        assert out == plain[:n] and kv <= len(ctx), (n, kv, len(ctx))
        assert res.end is not None and res.end["at"] == kv, (n, res.end, res.stats)
    return "max_tokens 13, 14, 15: kept"


# ------------------------------------------------------------------ 3. memory
class Refuse(cache.Budget):
    """A memory policy with room for everything but the end anchor's buffers."""

    def __init__(self):
        super().__init__(10 * BIG, 12 * BIG, 0, probe=lambda: (0, 0), empty=lambda: None)
        self.asked = []

    def admit(self, level, nbytes):
        self.asked.append(level)
        return level != "end" and super().admit(level, nbytes)


class RefuseAnchors(Refuse):
    """No room for an anchor's buffers either: nothing to take the end's from."""

    def admit(self, level, nbytes):
        self.asked.append(level)
        return level not in ("end", "anchors") and cache.Budget.admit(self, level, nbytes)


def test_the_memory_refusal_takes_an_anchors_buffers_for_the_end():
    """SRV-655: the end anchor is worth more than an anchor below it (the next turn resumes at
    the end and reads only its new tail). Refused new buffers for it, the end takes the buffers
    of the anchor whose loss leaves the smallest gap; the next turn resumes at the end."""
    plain, i, E, _ = _case(True, 1, labels=())
    mem = Refuse()
    res = _state(drafter=Past(PROMPT + plain, _width_for(i, 1)), tree=True, k=_width_for(i, 1),
                 mem=mem)
    out, pre, ctx, kv, log = run(PROMPT, 60, {E})
    assert "end" in mem.asked, mem.asked
    assert res.end is not None and res.end["at"] == kv, res.stats
    assert res.stats["end_from_anchor"] == 1 and res.stats["end_refused"] == 0, res.stats
    assert len(res.anchors) == len(PROMPT) // CH - 1, sorted(res.anchors)
    assert res._allocated == len(res.anchors) + len(res._free), res._allocated
    _, pre2, _, _, _ = run(PROMPT + out + TOOL, 4, set())
    assert pre2["kind"] == "end" and pre2["reused"] == kv, pre2
    return f"memory: the end from an anchor's buffers, the next turn resumed at {kv}"


def test_the_memory_refusal_stays_a_refusal_with_no_anchor_to_take():
    plain, i, E, _ = _case(True, 1, labels=())
    mem = RefuseAnchors()
    res = _state(drafter=Past(PROMPT + plain, _width_for(i, 1)), tree=True, k=_width_for(i, 1),
                 mem=mem)
    app.STATE["verbose"] = True
    out, pre, ctx, kv, log = run(PROMPT, 60, {E})
    assert "end" in mem.asked and res.end is None and not res.anchors, res.stats
    assert res.stats["end_refused"] == 1 and res.stats["end_ahead"] == 0, res.stats
    assert "end anchor not kept (memory, allocated " in log, log
    app.STATE["verbose"] = False
    _, pre2, _, _, _ = run(PROMPT + out + TOOL, 4, set())
    assert pre2["reused"] == 0, pre2
    return "memory: refused, counted, said with the policy's figures; the next turn from 0"


# ------------------------------------------------------------------ 4. NemotronH, GPU
def _nh_skip():
    if not torch.cuda.is_available():
        return "skipped: NemotronHServing's decode and verify run from CUDA graphs (the GPU)"
    from engine.models.nemotron_h import kernels as K
    return None if K.HAVE_TRITON else "skipped: no Triton"


def _nh_state(eng, drafter, k, end=True):
    from engine.models.nemotron_h.serving import NemotronHServing
    res = _state(end=end, drafter=drafter, tree=True, k=k)
    srv = NemotronHServing(eng, 256)
    srv.kv.length = 0
    app.STATE.update(engine=srv, device="cuda")
    return res, srv


def _nh_sections(srv):
    return {k: v.clone() for k, v in srv.state.sections().items()}


def test_nemotron_h_keeps_the_end_anchor_after_a_tool_call_turn():
    """The incident's shape on the board engine: the path 1 or 2 rows past the end token. 0.11.2:
    `ahead` (the state walked them; `trim_pending` cannot cut a committed row back). Now: kept at
    the committed tokens; its state, the end anchor and the next turn (resumed at the end) bit for
    bit those of a drafter that never drafted past the end (the row kernels make a verified row the
    decode step's at any block width); the next turn's greedy tokens equal the turn read without the
    end anchor."""
    skip = _nh_skip()
    if skip:
        return skip
    from test_nemotron_h_engine import tiny
    _, eng, _ = tiny()
    prompt = [5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88, 101, 7, 9, 200, 31] * 3
    _nh_state(eng, None, 4)
    out = list(app.generate_stream(torch.tensor(prompt), 60, set(), None, "ses"))
    i = next(i for i in range(12, 60) if out[i] not in out[:i])
    E, seq = out[i], prompt + out
    tool = [77, 78, 79, 80, 81]
    lines = []
    for past in (1, 2):
        w = _width_for(i, past)
        runs = {}
        for label in ("cut", "base", "ref"):
            res, srv = _nh_state(eng, Past(seq, w, stop_at=E if label == "ref" else None), w)
            with (no_cut() if label == "base" else _nothing()):
                o, pre, ctx, kv, log = run(prompt, 60, {E})
            runs[label] = (res, srv, o, ctx, kv)
        res, srv, o, ctx, kv = runs["cut"]
        bres, bsrv, bo, bctx, bkv = runs["base"]
        rres, rsrv, ro, rctx, rkv = runs["ref"]
        assert o == bo == ro == out[:i + 1], "the text"
        assert bkv == len(bctx) + past and bres.end is None and bres.stats["end_ahead"] == 1, \
            (bkv, len(bctx), bres.stats)
        assert kv == len(ctx) == rkv and res.end["at"] == rres.end["at"] == len(ctx), (kv, res.stats)
        assert res.stats["end_cut"] == 1 and res.stats["end_ahead"] == 0, res.stats
        srv.settle()
        rsrv.settle()
        sec, rsec = _nh_sections(srv), _nh_sections(rsrv)
        assert all(torch.equal(sec[k], rsec[k]) for k in sec), "the state at the end: bit for bit"
        assert all(torch.equal(a, b) for a, b in zip(res._end_bufs, rres._end_bufs))
        nxt = prompt + o + tool
        o2, pre2 = _next(res, srv, nxt, 12)
        ro2, rpre2 = _next(rres, rsrv, nxt, 12)
        bo2, bpre2 = _next(bres, bsrv, nxt, 12)
        assert pre2["kind"] == rpre2["kind"] == "end" and pre2["reused"] == len(ctx), pre2
        assert pre2["forwarded"] == len(tool) and o2 == ro2, (pre2, o2, ro2)
        srv.settle()
        rsrv.settle()
        sec, rsec = _nh_sections(srv), _nh_sections(rsrv)
        assert all(torch.equal(sec[k], rsec[k]) for k in sec), "the next turn: bit for bit"
        assert bpre2["kind"] != "end" and bo2 == o2, (bpre2, bo2, o2)
        lines.append(f"+{past}: 0.11.2 ahead, now kept at {len(ctx)}, next turn forwards "
                     f"{pre2['forwarded']} (0.11.2: {bpre2['forwarded']})")
    return "; ".join(lines) + "; bit for bit the never-past-the-end run, greedy equal to the turn without it"


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
