"""Tools/row_drift.py on a CPU: the rows it pairs are the rows the two runs chose each token from.

On the random 4-layer model in fp32 a verify block's row and a single step's row agree to float
precision, so a pairing that is off by one position shows up as a large difference at once.

Run: python tests/test_row_drift.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.spec import generate_greedy, generate_spec  # noqa: E402
from test_window_edge import _engine  # noqa: E402
from tools.row_drift import RawRows, rows_by_position  # noqa: E402


class Wrong:
    """Right for `prefix` tokens of `truth`, then wrong (the tiny model's vocabulary is 97)."""

    def __init__(self, truth, prefix):
        self.truth, self.prefix, self.n = truth or [], prefix, 0

    def reset(self):
        pass

    def observe(self, tokens):
        self.n += len(tokens)

    def propose(self, ctx, k):
        out = []
        for i in range(k):
            j = self.n + i
            t = self.truth[j] if j < len(self.truth) else 0
            out.append(t if i < self.prefix else (t + 1) % 97)
        return out


def test_the_pairing_is_position_for_position():
    eng = _engine(256)
    ids = torch.arange(5, 40)
    rg, rs = RawRows(), RawRows()
    base, _ = generate_greedy(eng, ids, 30, pen=rg)
    for truth, prefix in ((None, 0), (base, 3)):          # always wrong; right for 3 then wrong
        rs.seed([])
        got, st = generate_spec(eng, ids, 30, Wrong(truth or base, prefix), 6, pen=rs)
        assert got[:len(base)] == base, "(a last block may commit past max_new)"
        Rg, Rs = rows_by_position(rg, []), rows_by_position(rs, st.per_block)
        assert len(Rg) == len(base) and len(Rs) == len(got)
        for p, (g, s) in enumerate(zip(Rg, Rs)):
            assert int(g.argmax()) == base[p] == int(s.argmax()), p
            assert float((g - s).abs().max()) < 1e-3, (p, float((g - s).abs().max()))
    return "30 positions, always-wrong and half-right drafts: every pair is the same row"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:48s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
