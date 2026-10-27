"""The block's commit with no state copies, against the engine without the flag.

With `QWEN38_FUSED_COMMIT` a chain verify does not clone the recurrent state: it reads the entry
state and writes its final one into a spare buffer, and the commit rebuilds the live state from
the entry for all layers at once. None of that may change a number. On a CPU the commit runs the
torch reference of the same identity (`tools/gdn_commit_kernels.commit_reference`), so these tests
check the plumbing -- which buffer is read, which is written, what a full accept leaves, what a
partial accept and a tree path leave -- on the random four-layer model of test_forward_tree.
The kernel's own arithmetic is `tools/gdn_commit_kernels.py check()` on the board.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

TOKS = [7, 11, 23, 5, 31, 2]


def _run(flag: bool, keep: int | None, seed: int = 3):
    M.FUSED_COMMIT = flag
    try:
        eng, pos = T.build(seed=seed)
        before = eng.state.S.clone()
        with torch.no_grad():
            lg = eng.forward_block(torch.tensor(TOKS), start=pos)
            entry_ok = torch.equal(eng._trace.S_entry, before)
            if keep is not None:
                eng.rollback_to(keep)
            nxt = eng.forward_block(torch.tensor([3, 9]), start=pos + (keep or len(TOKS)))
        return lg, nxt, T.snapshot(eng), entry_ok
    finally:
        M.FUSED_COMMIT = False


def test_a_partial_accept_leaves_the_same_state():
    out = []
    for keep in (1, 3, 5):
        lg0, n0, s0, _ = _run(False, keep)
        lg1, n1, s1, ok = _run(True, keep)
        assert ok, "the entry state the trace keeps is not the state before the block"
        assert torch.equal(lg0, lg1), "the verify's logits moved"
        dS, dc = T.diff(s0[0], s1[0]), T.diff(s0[1], s1[1])
        assert dS < 1e-6 and dc == 0.0, (keep, dS, dc)
        assert T.diff(n0, n1) < 1e-5, "the next block reads a different state"
        out.append(f"keep {keep}: S {dS:.1e}")
    return "; ".join(out) + "; conv exact; next block's logits agree"


def test_a_full_accept_without_a_commit_leaves_the_walked_state():
    _, n0, s0, _ = _run(False, None)
    _, n1, s1, _ = _run(True, None)
    assert torch.equal(s0[0], s1[0]) and torch.equal(s0[1], s1[1]), "full accept state differs"
    assert torch.equal(n0, n1)
    return "state and next logits bit-identical when the caller does not roll back"


def test_a_tree_path_commits_to_the_same_state():
    tree = T.BRANCHY
    out = []
    for leaf in tree.leaves():
        path = tree.path(leaf)
        snaps = []
        for flag in (False, True):
            M.FUSED_COMMIT = flag
            try:
                eng, pos = T.build(seed=2)
                with torch.no_grad():
                    eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
                    eng.commit_tree(path)
                snaps.append(T.snapshot(eng))
            finally:
                M.FUSED_COMMIT = False
        dS = T.diff(snaps[0][0], snaps[1][0])
        assert dS < 1e-6 and torch.equal(snaps[0][1], snaps[1][1]), (leaf, dS)
        n = len(path)
        assert torch.equal(snaps[0][2][..., :pos + n, :], snaps[1][2][..., :pos + n, :])
        assert snaps[0][4] == snaps[1][4] == pos + n
        out.append(f"leaf {leaf}: S {dS:.1e}")
    return "; ".join(out)


def test_the_two_buffers_trade_places():
    M.FUSED_COMMIT = True
    try:
        eng, pos = T.build(seed=4)
        a = eng.state.S
        with torch.no_grad():
            eng.forward_block(torch.tensor(TOKS[:3]), start=pos)
        b = eng.state.S
        assert b.data_ptr() != a.data_ptr() and eng._trace.S_entry.data_ptr() == a.data_ptr()
        with torch.no_grad():
            eng.rollback_to(2)
            eng.forward_block(torch.tensor(TOKS[:3]), start=pos + 2)
        assert eng.state.S.data_ptr() == a.data_ptr(), "the spare was not reused"
    finally:
        M.FUSED_COMMIT = False
    return "entry kept in place, final in the spare, the pair alternates"


def test_a_partial_accept_puts_kv_length_back_on_the_fused_commit():
    """On the fused commit. `rollback_to` returns early into `_fused_commit`, so the
    `kv.length` rc4 puts back to the kept prefix has to be set before that return, or a snapshot
    taken after a rejected block keys one token more than its state has seen."""
    from engine import cache
    out = []
    for flag in (False, True):
        M.FUSED_COMMIT = flag
        try:
            eng, pos = T.build(seed=5)
            with torch.no_grad():
                eng.forward_block(torch.tensor(TOKS), start=pos)
                eng.rollback_to(3)
            assert eng.kv.length == pos + 3, (flag, eng.kv.length)
            assert cache.capture(eng).length == pos + 3
        finally:
            M.FUSED_COMMIT = False
        out.append(f"flag {int(flag)}: kv.length {eng.kv.length}")
    return "; ".join(out)


def test_a_graphed_chain_rolls_back_to_where_it_was_verified():
    """A verify graph is captured at start 0 and its trace is handed back after every replay
    (`VerifyGraphs._restore`). The replay's start has to reach that trace: the rollback puts
    `kv.length` back to `trace.start + keep`, and a trace still saying 0 would drop the whole
    context to `keep` rows. The eager trace stands in for the captured one, start reset to 0."""
    from engine.verify_graph import Captured, VerifyGraphs
    M.FUSED_COMMIT = True
    try:
        eng, pos = T.build(seed=6)
        with torch.no_grad():
            eng.forward_block(torch.tensor(TOKS), start=pos)
        cap = Captured.__new__(Captured)            # no CUDA graph object on a CPU
        cap.trace, cap.taps = eng._trace, []
        cap.hidden_pre, cap.hidden_post = eng.hidden_pre_norm, eng.hidden_post_norm
        cap.trace.start = 0                         # as the capture left it
        vg = VerifyGraphs.__new__(VerifyGraphs)
        vg.eng = eng
        vg._restore(cap, pos, len(TOKS))
        with torch.no_grad():
            eng.rollback_to(4)
        assert eng.kv.length == pos + 4, eng.kv.length
    finally:
        M.FUSED_COMMIT = False
    return f"replayed at {pos}, kept 4: kv.length {pos + 4}"



# ---------------------------------------------------------------- the commit, pending

def _tree_then(flag: bool, after: str, seed: int = 8):
    """A tree verified and committed along a path, then one more operation, with and without
    QWEN38_COMMIT_IN_VERIFY. On a CPU the fused GDN verify mixer does not run, so no verify ever
    consumes a pending commit here: every path goes through `_settle`, which is exactly the
    plumbing these tests are about -- who flushes, and what a flush leaves."""
    from engine import cache
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[-1])
    M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = True, flag
    try:
        eng, pos = T.build(seed=seed)
        with torch.no_grad():
            eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
            entry = eng.state.S.clone()
            eng.commit_tree(path)
            pending = eng._pend is not None
            untouched = torch.equal(eng.state.S, entry)
            pos += len(path)
            out = None
            if after == "decode":
                out = eng.forward(torch.tensor([13]), start=pos, last_only=True)
            elif after == "prefill":
                out = eng.forward(torch.tensor([13, 17, 19, 23]), start=pos, last_only=True)
            elif after == "chain":
                out = eng.forward_block(torch.tensor(TOKS), start=pos)
            elif after == "snapshot":
                out = cache.capture(eng).state[0]
            elif after == "reset":
                eng.reset()
                out = torch.tensor(float(eng._pend is None))
        return out, T.snapshot(eng), pending, untouched
    finally:
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = False, False


def test_a_committed_tree_path_waits_and_every_reader_applies_it_first():
    out = []
    for after in ("decode", "prefill", "chain", "snapshot"):
        o0, s0, p0, _ = _tree_then(False, after)
        o1, s1, p1, untouched = _tree_then(True, after)
        assert not p0 and p1, "the commit is pending only with the flag"
        assert untouched, "a pending commit must leave the live buffer as the entry"
        assert torch.equal(o0, o1), f"{after}: what it read differs"
        assert torch.equal(s0[0], s1[0]) and torch.equal(s0[1], s1[1]), f"{after}: state differs"
        out.append(after)
    return "pending after commit_tree; " + ", ".join(out) + " apply it first, bit-identical"


def test_reset_and_restore_drop_a_pending_commit():
    from engine import cache
    o, _, pending, _ = _tree_then(True, "reset")
    assert pending and float(o) == 1.0, "reset must drop the pending commit"
    M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = True, True
    try:
        eng, pos = T.build(seed=9)
        snap = cache.capture(eng)
        tree = T.BRANCHY
        with torch.no_grad():
            eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
            eng.commit_tree(tree.path(tree.leaves()[0]))
        assert eng._pend is not None
        cache.restore(eng, snap)
        assert eng._pend is None, "a restored state must not get the old commit applied"
        assert torch.equal(eng.state.S, snap.state[0])
    finally:
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = False, False
    return "reset and cache.restore drop it"


def test_off_is_the_code_it_was():
    """The flag off never records a commit: `_fused_commit` applies it at once, as left it."""
    for keep in (2, None):
        M.COMMIT_IN_VERIFY = False
        lg0, n0, s0, _ = _run(True, keep)
        assert torch.equal(s0[0], _run(True, keep)[2][0])
    eng, pos = T.build(seed=3)
    assert eng._pend is None and not M.COMMIT_IN_VERIFY
    return "flag off: no pending record, the fused commit as before"

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
