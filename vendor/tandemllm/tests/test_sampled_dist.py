"""CPU tests for tools/sampled_dist.py (distribution check on the served stack).

Run: python tests/test_sampled_dist.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sampled_dist import PROMPTS, analyse, chi2_sf, greedy_same, homogeneity, pick, prefix  # noqa: E402


def test_the_chi_square_tail_matches_the_tables():
    for x, dof, want in ((3.841459, 1, 0.05), (9.210340, 2, 0.01), (0.0, 3, 1.0), (30.0, 20, 0.06985),
                         (100.0, 10, 5.1e-17), (1.0, 5, 0.96257)):
        got = chi2_sf(x, dof)
        assert abs(got - want) <= 1e-4 * max(want, 1e-12) + 1e-18, (x, dof, got, want)
    return "chi2 tail at six table points, both branches of the incomplete gamma"


def test_prefix_is_a_function_of_the_answer():
    assert prefix("Cat, dog, horse", 2) == "Cat, dog,"
    assert prefix("Cat", 3) == "Cat <end>"
    assert prefix("Cat", 3) != prefix("Cat dog", 3)
    return "w words, a short answer marked, so two answers that differ never share a prefix of their own length"


def _draw(rng, dist, n):
    words, weights = zip(*dist.items())
    return [" ".join(rng.choices(words, weights)[0] for _ in range(4)) for _ in range(n)]


def test_an_exact_sampler_passes_and_a_biased_one_is_flagged():
    """Four configurations drawn from one distribution, one biased toward the mode, the control cooler: the
    exact ones pass, the biased one and the control are flagged, and the verdict is PASS only with the bias out."""
    rng = random.Random(5)
    p = {"cat": 0.40, "dog": 0.30, "fox": 0.15, "owl": 0.10, "yak": 0.05}
    biased = {"cat": 0.55, "dog": 0.25, "fox": 0.10, "owl": 0.07, "yak": 0.03}
    cool = {"cat": 0.60, "dog": 0.28, "fox": 0.08, "owl": 0.03, "yak": 0.01}
    n = 300
    ans = {c: {"a": _draw(rng, d, n), "b": _draw(rng, d, n)}
           for c, d in (("nospec", p), ("chain", p), ("det", p), ("mixed", biased), ("cool", cool))}
    res = analyse(ans)
    assert not res["configs"]["chain"]["flagged"] and not res["configs"]["det"]["flagged"], res["configs"]
    assert res["configs"]["mixed"]["flagged"] and res["control_seen"] and not res["pass"]
    ans["mixed"] = {"a": _draw(rng, p, n), "b": _draw(rng, p, n)}
    res = analyse(ans)
    assert res["pass"], {k: v["p_min"] for k, v in res["configs"].items()}
    ans["cool"] = {"a": _draw(rng, p, n), "b": _draw(rng, p, n)}
    assert not analyse(ans)["pass"]                                    # a control it cannot see voids the check
    return (f"exact configs min p {min(res['configs'][c]['p_min'] for c in ('chain', 'det', 'mixed')):.3f}; "
            "the biased config and the control flagged; no control seen -> FAIL")


def test_rare_prefixes_are_pooled_and_identical_samples_do_not_differ():
    a = ["x"] * 50 + [f"r{i}" for i in range(20)]
    stat, dof, p = homogeneity(a, list(a))
    assert stat == 0.0 and p == 1.0 and dof == 1                       # "x" and one pooled <rare> cell
    return "20 singletons -> one <rare> cell; the same sample twice: chi2 0, p 1"


def test_greedy_answers_must_match_text_and_tokens():
    one = {"prose": {"text": "a b", "tokens": 2}, "code": {"text": "x", "tokens": 1}}
    assert greedy_same({"chain": one, "det": dict(one), "mixed": dict(one)})["pass"]
    other = dict(one, code={"text": "x", "tokens": 2})
    g = greedy_same({"chain": one, "det": other})
    assert not g["pass"] and g["differ"]["det"] == ["code"]
    assert greedy_same({})["pass"] is None
    return "same text + count passes; a token-count difference alone fails and names the workload"


def test_prompts_are_picked_by_name_and_unknown_names_refused():
    assert pick("") == PROMPTS
    assert list(pick("story")) == ["story"] and list(pick("story,animals")) == ["animals", "story"]
    try:
        pick("stroy")
    except SystemExit as e:
        assert "stroy" in str(e)
    else:
        raise AssertionError("an unknown prompt name must be refused")
    return "empty -> all; names -> those, in PROMPTS order; a typo refused"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:62s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
