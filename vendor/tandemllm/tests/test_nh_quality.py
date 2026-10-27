"""ENG-339: the Bolt quality harness's arithmetic (tools/nh_quality.py) on the CPU: identical logits give KL 0
and full agreement, a constant shift changes nothing, a known two-token case gives the analytic KL, a
disagreement at a near tie is counted as one, the NLL is the set's, and a bundle written the way exllamav3's
model_diff writes it reads back row by row."""

from __future__ import annotations

import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nh_quality import bundle, logits_file, metrics  # noqa: E402


def test_identical_and_shifted_logits():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 50, generator=g)
    for y in (x.clone(), x + 3.0, x.to(torch.bfloat16)):
        m = metrics(x, y)
        assert m["kl"] < 1e-3 and m["top1"] == 16 and m["differ"] == 0, m


def test_two_token_kl_and_ties():
    p, q = torch.tensor([0.9, 0.1]), torch.tensor([0.6, 0.4])
    m = metrics(p.log().view(1, 2), q.log().view(1, 2))
    want = 0.9 * math.log(0.9 / 0.6) + 0.1 * math.log(0.1 / 0.4)
    assert abs(m["kl"] - want) < 1e-6 and m["top1"] == 1
    # the reference nearly tied (margin 0.1 nat) and the set picks the other token: a near tie
    r = torch.tensor([[1.0, 0.9, -5.0]])
    s = torch.tensor([[0.9, 1.0, -5.0]])
    m = metrics(r, s)
    assert m["differ"] == 1 and m["near_ties"] == 1
    # a clear reference choice flipped: a difference, not a near tie
    m = metrics(torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([[0.0, 3.0, 0.0]]))
    assert m["differ"] == 1 and m["near_ties"] == 0


def test_nll_and_vocab_width():
    lg = torch.log(torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]))
    m = metrics(lg, lg, torch.tensor([1, 1]))
    assert abs(m["nll"] - (-(math.log(0.25) + math.log(0.8)) / 2)) < 1e-6
    # a padded vocabulary on one side is compared over the narrower one
    m = metrics(lg, torch.cat([lg, torch.full((3, 5), -1e4)], -1))
    assert m["kl"] < 1e-6 and m["top1"] == 3


def test_bundle_reads_model_diff_files():
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as d:
        ids = [torch.randint(0, 100, (1, 8)) for _ in range(3)]
        lg = [torch.randn(1, 8, 100) for _ in range(3)]
        save_file({f"tensor.{i}": t for i, t in enumerate(ids)}, os.path.join(d, "input_ids.safetensors"))
        save_file({f"tensor.{i}": t for i, t in enumerate(lg)}, os.path.join(d, "logits_bf16.safetensors"))
        r_ids, r_lg = bundle(d)
        assert len(r_ids) == 3 and all(torch.equal(a.view(-1), b) for a, b in zip(ids, r_ids))
        assert all(torch.equal(a[0], b) for a, b in zip(lg, r_lg))
        assert torch.equal(logits_file(os.path.join(d, "logits_bf16.safetensors"))[2], lg[2][0])


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"  {name:<45} ok")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:<45} FAIL {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} failed" if fails else ""))
    sys.exit(1 if fails else 0)
