"""The relaxed accept rule: what it accepts, and that it is off unless it is asked for.

The rule is the only thing in the engine that can change what the model writes, so the test that
matters most is the one that says the default does nothing.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.spec import Relax  # noqa: E402

passed = failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}")


# Logits whose softmax is roughly [0.60, 0.22, 0.08, 0.03, 0.01, ...]
row = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, -6.0])
argmax = 0
probs = torch.softmax(row, dim=0)

off = Relax()
check("default rule is off", not off.on)
check("default accepts the argmax", off.accepts(row, argmax, argmax))
for token in range(1, 6):
    check(f"default rejects token {token}", not off.accepts(row, token, argmax))

# tau: p(t) >= tau * p(argmax). p1/p2 = e, so tau just below 1/e must take the runner-up.
ratio = float(probs[1] / probs[0])
check("tau above the ratio rejects the runner-up",
      not Relax(tau=ratio * 1.01).accepts(row, 1, argmax))
check("tau below the ratio accepts the runner-up",
      Relax(tau=ratio * 0.99).accepts(row, 1, argmax))
check("tau is on when below one", Relax(tau=0.5).on)
check("tau one is off", not Relax(tau=1.0).on)

# The rule is stated on probabilities and evaluated on logits; those must agree.
for tau in (0.9, 0.5, 0.3, 0.1, 0.02):
    rule = Relax(tau=tau)
    for token in range(6):
        by_prob = float(probs[token]) >= tau * float(probs[argmax])
        check(f"logit form agrees with probability form, tau={tau} t={token}",
              rule.accepts(row, token, argmax) == by_prob)

# rank
check("rank 1 is off", not Relax(rank=1).on)
check("rank 3 accepts the third", Relax(rank=3).accepts(row, 2, argmax))
check("rank 3 rejects the fourth", not Relax(rank=3).accepts(row, 3, argmax))
check("rank 6 accepts the last", Relax(rank=6).accepts(row, 5, argmax))

# The two knobs are a union, not an intersection: either one may accept.
both = Relax(tau=0.02, rank=2)
check("union accepts what only tau would", both.accepts(row, 3, argmax) ==
      (float(probs[3]) >= 0.02 * float(probs[0])))
check("union accepts what only rank would", both.accepts(row, 1, argmax))

# A degenerate row: every token identical. Everything is the argmax's equal, so tau accepts all.
flat = torch.zeros(6)
check("flat row, tau 1 still only the argmax", not Relax().accepts(flat, 3, 0))
check("flat row, tau 0.99 accepts everything", Relax(tau=0.99).accepts(flat, 3, 0))

# -inf logits, which a trimmed head produces, must never be accepted by tau.
masked = torch.tensor([2.0, -float("inf"), 0.0])
check("masked token is never accepted", not Relax(tau=1e-9).accepts(masked, 1, 0))
check("log of tau is finite for the smallest sensible tau", math.isfinite(math.log(0.001)))

# --- the typical rule -----------------------------------------------------------------------
# Off unless asked for, and then a bar that moves with the target's own entropy rather than with
# its argmax. The two properties worth pinning are the ones the rule exists for: on a PEAKED row
# it must be stricter than a tau rule of the same nominal size, and on a FLAT row looser.
check("typical is off by default", not Relax().on)
check("typical on makes the rule on", Relax(typical=True).on)
check("typical still accepts the argmax", Relax(typical=True).accepts(row, argmax, argmax))

peaked = torch.tensor([12.0, 2.0, 1.9, 1.8, 0.0, -3.0])       # entropy near zero
flat6 = torch.zeros(6)                                        # entropy = log 6 = 1.79 nats


def threshold(r, lg):
    lp = lg.float().log_softmax(-1)
    h = float(-(lp.exp() * lp).sum())
    return min(r.eps, r.delta * math.exp(-h))


rule = Relax(typical=True, eps=0.09, delta=0.3)
check("peaked row: the bar is eps, because delta*exp(-H) exceeds it",
      abs(threshold(rule, peaked) - 0.09) < 1e-9)
check("flat row: the bar is delta*exp(-H), well under eps",
      threshold(rule, flat6) < 0.09)
check("peaked row rejects the runner-up", not rule.accepts(peaked, 1, 0))
check("flat row accepts a non-argmax", rule.accepts(flat6, 3, 0))
check("a masked -inf token is never typical", not rule.accepts(
    torch.tensor([2.0, -float("inf"), 0.0]), 1, 0))
# delta below eps/1 can never let eps bind: min() picks delta*exp(-H) at every entropy.
check("eps only binds above ln(delta/eps) nats",
      threshold(Relax(typical=True, eps=0.3, delta=0.3), flat6)
      == threshold(Relax(typical=True, eps=0.09, delta=0.3), flat6))
# The knobs stay a union: tau, rank and typical each may accept on their own.
check("typical joins the union rather than replacing it",
      Relax(tau=0.02, typical=True).accepts(row, 3, argmax))

print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
sys.exit(1 if failed else 0)
