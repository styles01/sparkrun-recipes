"""The session snapshot after a generation that stopped part way through an accepted path.

On the 2026-09-30 night 13 of the 20 session puts small enough to store were declined, and
2,753 of 3,100 were 1 to 4 tokens ahead: the loop commits a block's whole accepted path and
yields its tokens one at a time, and a generation that stops at the end token, a stop or its
budget returns before yielding the rest. `kv.length` is then ahead of the committed tokens and
the snapshot has no key.

With the commit folded into the next verify (QWEN38_COMMIT_IN_VERIFY) the recurrent state has
not absorbed that last block when the generation ends, so `Qwen38Engine.trim_pending` can cut
the pending commit to the committed rows. The state it leaves must be the one committing the
shorter path leaves, bit for bit; on this CPU the commit runs the torch reference of the kernel's
identity (tests/test_fused_commit.py). And the server must say why a put was declined.

Run: python tests/test_session_trim.py
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from engine import cache  # noqa: E402


class folded:
    """The served commit: fused (its torch reference here) and folded into the next verify."""

    def __enter__(self):
        self.old = (M.FUSED_COMMIT, M.COMMIT_IN_VERIFY)
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = True, True

    def __exit__(self, *exc):
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = self.old
        return False


def _tree_block(path, seed=2):
    eng, pos = T.build(seed=seed)
    tree = T.BRANCHY
    with torch.no_grad():
        eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
        eng.commit_tree(path)
    return eng, pos


def _settled(eng):
    eng._settle()
    return T.snapshot(eng)


def test_a_pending_commit_cut_short_is_the_shorter_commit():
    tree = T.BRANCHY
    n = 0
    with folded():
        for leaf in tree.leaves():
            path = tree.path(leaf)
            for m in range(0, len(path)):
                a, pos = _tree_block(path)
                assert a._pend is not None, "the commit should be pending"
                assert a.trim_pending(pos + m)
                sa = _settled(a)
                if m:
                    b, _ = _tree_block(path[:m])
                    sb = _settled(b)
                else:
                    b, _ = T.build(seed=2)
                    sb = T.snapshot(b)
                assert sa[4] == sb[4] == pos + m
                assert torch.equal(sa[0], sb[0]), (leaf, m, T.diff(sa[0], sb[0]))
                assert torch.equal(sa[1], sb[1]), (leaf, m)
                L = pos + m
                assert torch.equal(sa[2][..., :L, :], sb[2][..., :L, :])
                # and the next forward from there reads the same state
                with torch.no_grad():
                    la = a.forward(torch.tensor([5, 9]), start=L, last_only=True)
                    lb = b.forward(torch.tensor([5, 9]), start=L, last_only=True)
                assert torch.equal(la, lb), (leaf, m)
                n += 1
    return f"{n} (path, cut) pairs bit-equal to committing the shorter path"


def test_nothing_to_trim_without_a_pending_commit():
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[0])
    eng, pos = _tree_block(path)                   # the flags off: committed at once
    assert eng._pend is None
    before = T.snapshot(eng)
    assert not eng.trim_pending(pos + 1)
    after = T.snapshot(eng)
    assert after[4] == before[4] and torch.equal(after[0], before[0])
    assert eng.trim_pending(eng.kv.length)         # already there


def test_a_settled_commit_cannot_be_trimmed():
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[0])
    with folded():
        eng, pos = _tree_block(path)
        eng._settle()
        assert not eng.trim_pending(pos + 1)


# ------------------------------------------------------------------ the server's put

def _remember_with(ahead: int, trim: bool, max_entry: int = 0, said=None, conv=None, mem=None):
    from server import app
    tree = T.BRANCHY
    path = tree.path(tree.leaves()[0])
    with folded():
        eng, pos = _tree_block(path)
        n_c = eng.kv.length - ahead
        store = cache.StateStore(1 << 30, chunk=16, max_entry_bytes=max_entry)
        saved = {k: app.STATE.get(k) for k in ("engine", "drafter", "state_store", "session_cache",
                                               "verbose", "last_ctx", "session_trim",
                                               "suffix_store", "store_big_said", "mem")}
        app.STATE.update(engine=eng, drafter=None, state_store=store, session_cache=True,
                         verbose=True, last_ctx=list(range(1, n_c + 1)), session_trim=trim,
                         suffix_store=None, store_big_said=said, mem=mem)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                app._remember(list(range(1, pos + 1)), list(range(pos + 1, n_c + 1)), conv)
        finally:
            app.STATE.update(saved)
    return store, out.getvalue(), eng, n_c


def test_the_put_names_the_engine_being_ahead():
    store, log, _, _ = _remember_with(ahead=2, trim=False)
    assert store.stats["puts"] == 0 and store.stats.get("declined_short") == 1
    assert "ahead of the committed tokens" in log, log


def test_the_put_names_the_size_cap():
    store, log, _, _ = _remember_with(ahead=0, trim=False, max_entry=1)
    assert store.stats["puts"] == 0 and store.stats.get("declined_big") == 1
    # FX10: over the cap is said as what it is, and who keeps the conversation instead
    assert "over the store's per-entry cap" in log and "resident prefix keeps" in log, log


def test_over_the_cap_is_said_once_per_conversation():
    """FX10: 165 of 245 requests on a 0.4.0 test build printed the decline, every turn of every
    conversation past about 7,400 tokens; the counter still counts each one."""
    store, log, _, _ = _remember_with(ahead=0, trim=False, max_entry=1, said="a", conv="a")
    assert store.stats.get("declined_big") == 1 and "per-entry cap" not in log, log
    store, log, _, _ = _remember_with(ahead=0, trim=False, max_entry=1, said="a", conv="b")
    assert "over the store's per-entry cap" in log, log


class _Full(cache.Budget):
    """A memory policy with no room left: 10 GiB allocated against a 10 GiB ceiling."""

    def __init__(self):
        g = 1 << 30
        super().__init__(10 * g, 12 * g, 1 * g, probe=lambda: (10 * g, 10 * g), empty=lambda: None)


def test_the_put_names_the_memory_policy_not_the_size_cap():
    """SRV-655 (Ornith, 2026-10-20): every decline read "too big for the per-entry cap (1024
    MiB)" -- at kv.length 18 too -- while the memory policy had refused a snapshot well under
    the cap. The log and the counters now say which."""
    store, log, _, _ = _remember_with(ahead=0, trim=False, max_entry=1 << 30, mem=_Full())
    assert store.stats["puts"] == 0 and store.stats.get("declined_mem") == 1, store.stats
    assert not store.stats.get("declined_big"), store.stats
    assert "per-entry cap" not in log and "the memory policy refused its " in log and \
        "allocated 10.00 GiB of a 10.00 GiB ceiling less 1.00 GiB of slack" in log, log


def test_with_the_trim_on_the_put_succeeds_at_the_committed_length():
    store, log, eng, n_c = _remember_with(ahead=2, trim=True)
    assert store.stats["puts"] == 1 and store.stats.get("session_trimmed") == 1, store.stats
    assert store.report()["boundaries"] == [n_c] and eng.kv.length == n_c
    assert "declined" not in log


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:65s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
