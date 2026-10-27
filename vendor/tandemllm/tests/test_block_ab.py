"""`tools/block_ab.py`, the alternated in-engine block A/B, without a board.

The rule is the row's (tools/row3.py): a difference counts only when it is bigger than the spread
and the run ranges do not overlap. These tests feed recorded-shape runs to the resolver and the
stub: a null pair reads not resolved, a clear gain reads resolved better per workload and pooled,
a token difference fails and names the workload, the state, the pair and the token.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(pair, state, workload, ms, out=(1, 2, 3), tok_blk=4.0):
    import hashlib
    import json
    return {"pair": pair, "state": state, "workload": workload, "block_ms": ms, "blocks": 64,
            "tokens": 256, "tok_blk": tok_blk, "tok_s": 1e3 * tok_blk / ms, "out": list(out),
            "sha": hashlib.sha256(json.dumps(list(out)).encode()).hexdigest()}


def _result(runs, states=("base", "cand"), label="t"):
    from tools.block_ab import parse_mix, DEFAULT_MIX
    return {"label": label, "runs": runs, "states": list(states), "base": states[0],
            "state_specs": [f"{s}=" for s in states], "pairs": 3,
            "workloads": list(dict.fromkeys(r["workload"] for r in runs)), "max_new": 256,
            "mix": parse_mix(DEFAULT_MIX), "precapture": 32, "discard": 1, "learn_cost": False,
            "proc": False,
            "code": "0123456789abcdef0123"}


# phase5b hold 3's base, prose/chat/code, three runs each (ms a block, loose)
BASE = {"prose": [89.97, 90.60, 90.78], "chat": [98.90, 98.98, 99.54], "code": [99.69, 99.65, 99.90]}


def _pairs(base, cand):
    runs = []
    for p in range(3):
        for w in base:
            runs.append(_run(p, "base", w, base[w][p]))
            runs.append(_run(p, "cand", w, cand[w][p]))
    return runs


def test_parse_state_reads_modules_values_and_env():
    from tools.block_ab import parse_state
    s = parse_state("kr1=tools.nvfp4_skinny:WIDE_KR=1;PF=2+engine.model:VERIFY_GRAPH=0")
    assert s["name"] == "kr1" and s["env"] == {}
    assert s["mods"] == [("tools.nvfp4_skinny", {"WIDE_KR": "1", "PF": "2"}),
                         ("engine.model", {"VERIFY_GRAPH": "0"})]
    e = parse_state("srun0=env:QWEN38_SKINNY_SRUN=0;QWEN38_X=1")
    assert e["env"] == {"QWEN38_SKINNY_SRUN": "0", "QWEN38_X": "1"} and e["mods"] == []
    b = parse_state("base=")
    assert b == {"name": "base", "mods": [], "env": {}}
    for bad in ("nope", "=x:y=1", "x=mod", "x=mod:attr"):
        try:
            parse_state(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    return "module attributes, several per module, env, the empty base, four malformed specs refused"


def test_switch_restores_the_modules_own_values_between_states():
    from tools.block_ab import Switch, parse_state
    mod = types.ModuleType("fake_ab_mod")
    mod.FLAG, mod.N, mod.NAME, mod.OVR = False, 16, "x", None
    sys.modules["fake_ab_mod"] = mod
    try:
        sw = Switch([parse_state("base="), parse_state("a=fake_ab_mod:FLAG=1;N=24"),
                     parse_state("b=fake_ab_mod:NAME=y")])
        sw.apply("a")
        assert (mod.FLAG, mod.N, mod.NAME) == (True, 24, "x")
        sw.apply("b")
        assert (mod.FLAG, mod.N, mod.NAME) == (False, 16, "y"), "a's values leaked into b"
        sw.apply("base")
        assert (mod.FLAG, mod.N, mod.NAME) == (False, 16, "x")
        ov = Switch([parse_state("base="), parse_state("o=fake_ab_mod:OVR=1")])
        ov.apply("o")
        assert mod.OVR == 1 and isinstance(mod.OVR, int), "an override that defaults to None is a number"
        ov.apply("base")
        assert mod.OVR is None
        sw.apply("a")
        sw.restore()
        assert (mod.FLAG, mod.N, mod.NAME) == (False, 16, "x")
        try:
            Switch([parse_state("c=fake_ab_mod:MISSING=1")])
        except SystemExit:
            pass
        else:
            raise AssertionError("an attribute the module lacks was accepted")
    finally:
        del sys.modules["fake_ab_mod"]
    return "values cast to the module's types (bool, int, str); each state starts from the originals"


def test_the_order_alternates_and_never_repeats_a_state():
    from tools.block_ab import order
    o = order(["base", "cand"], 3)
    assert [n for _, n in o] == ["base", "cand"] * 3
    assert [p for p, _ in o] == [0, 0, 1, 1, 2, 2]
    o3 = [n for _, n in order(["base", "c1", "c2"], 2)]
    assert all(x != y for x, y in zip(o3, o3[1:]))
    return "A B A B A B; with two candidates base c1 c2 base c1 c2"


def test_a_null_pair_reads_not_resolved():
    from tools.block_ab import judge
    # the same distribution on both sides, interleaved: what two identical states produce
    cand = {"prose": [90.41, 90.12, 90.66], "chat": [99.12, 98.95, 99.31], "code": [99.80, 99.62, 99.75]}
    rc, lines = judge(_result(_pairs(BASE, cand)), "better")
    v = _result(_pairs(BASE, cand))
    judge(v, "better")
    for w, x in v["verdicts"]["cand"]["ms_blk"].items():
        assert x["verdict"] == "not resolved", (w, x)
    assert rc == 2, lines
    rc2, _ = judge(_result(_pairs(BASE, cand)), "noworse")
    assert rc2 == 0
    return "every workload and the pool not resolved; rule better -> rc 2, noworse -> rc 0"


def test_a_known_gain_reads_resolved_better_everywhere():
    from tools.block_ab import judge
    # the SRUN size of gain, -2.4 ms a block on every run
    cand = {w: [x - 2.4 for x in xs] for w, xs in BASE.items()}
    r = _result(_pairs(BASE, cand))
    rc, lines = judge(r, "better")
    assert rc == 0, lines
    vs = r["verdicts"]["cand"]["ms_blk"]
    for w in ("prose", "chat", "code", "pooled"):
        assert vs[w]["verdict"] == "RESOLVED better", (w, vs[w])
        assert vs[w]["signs"] == "---", vs[w]
    assert abs(vs["pooled"]["delta"] + 2.4) < 1e-9
    assert "PASS" in lines[-1]
    return f"-2.4 ms on every run: resolved better per workload and pooled ({vs['pooled']['delta']:+.2f} ms)"


def test_a_workload_that_resolves_worse_fails():
    from tools.block_ab import judge
    cand = {w: [x - 2.4 for x in xs] for w, xs in BASE.items()}
    cand["prose"] = [x + 2.0 for x in BASE["prose"]]
    r = _result(_pairs(BASE, cand))
    rc, lines = judge(r, "better")
    assert rc == 1, lines
    assert r["verdicts"]["cand"]["ms_blk"]["prose"]["worse"]
    return "prose +2 ms under a pooled gain: FAIL"


def test_the_pool_weights_the_arm_mix():
    from tools.block_ab import pooled, parse_mix
    mix = parse_mix("prose=0.5,chat=0.25,code=0.25")
    got = pooled({"prose": [90.0, 92.0], "chat": [100.0, 100.0], "code": [110.0, 102.0]}, mix)
    assert got == [0.5 * 90 + 0.25 * 100 + 0.25 * 110, 0.5 * 92 + 0.25 * 100 + 0.25 * 102]
    # a workload the mix does not name is left out; one that did not run renormalises the rest
    assert pooled({"prose": [90.0], "quote": [10.0]}, mix) == [90.0]
    assert pooled({"chat": [100.0], "code": [110.0]}, mix) == [105.0]
    return "per pair, weighted by the mix, renormalised over what ran"


def test_a_token_difference_fails_and_names_where():
    from tools.block_ab import judge
    cand = {w: [x - 2.4 for x in xs] for w, xs in BASE.items()}
    runs = _pairs(BASE, cand)
    for r in runs:
        if r["state"] == "cand" and r["workload"] == "chat" and r["pair"] == 1:
            r["out"] = [1, 9, 3]
    res = _result(runs)
    rc, lines = judge(res, "better")
    assert rc == 1, lines
    assert res["tokens"]["chat"] == {"state": "cand", "pair": 1, "at": 1}, res["tokens"]
    assert res["tokens"]["prose"] is None
    assert any("chat" in x and "DIFFER" in x and "pair 2" in x and "token 1" in x for x in lines), lines
    return "a -2.4 ms gain with chat's tokens changed in pair 2: FAIL, named"


def test_a_base_that_changes_its_own_tokens_is_caught_too():
    from tools.block_ab import tokens_check
    runs = _pairs(BASE, BASE)
    runs[-6]["out"] = [1, 2, 4]            # pair 3, base, prose
    assert runs[-6]["state"] == "base" and runs[-6]["workload"] == "prose"
    got = tokens_check(runs, "base")
    assert got["prose"] == {"state": "base", "pair": 2, "at": 2}, got
    return "the base against its own first run: a history-dependent base is a finding, not noise"


def test_the_stub_carries_runs_tokens_verdicts_and_the_code():
    from tools.block_ab import judge, stub
    cand = {w: [x - 2.4 for x in xs] for w, xs in BASE.items()}
    r = _result(_pairs(BASE, cand), label="srun-cal")
    rc, lines = judge(r, "better")
    md = stub(r, lines)
    assert md.startswith("## ") and "block A/B srun-cal" in md
    assert "0123456789abcdef" in md
    assert md.count("\n") > 18
    for w in ("prose", "chat", "code"):
        assert sum(1 for x in md.splitlines() if f" {w} " in x and ("base" in x or "cand" in x)) >= 6
    assert "tokens prose" in md and "pooled" in md and "RESOLVED better" in md
    assert "[block-ab] srun-cal (better): PASS" in md
    assert "1 pair(s) discarded first" in md
    return "18 runs, the tokens lines, the per-workload and pooled verdicts, the code hash"


def test_the_saved_result_rejudges_the_same():
    import json
    import subprocess
    import tempfile
    from tools.block_ab import judge
    cand = {w: [x - 2.4 for x in xs] for w, xs in BASE.items()}
    r = _result(_pairs(BASE, cand))
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(r, f)
    try:
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "block_ab.py"), "--judge", path],
            capture_output=True, text=True)
    finally:
        os.remove(path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PASS" in p.stdout.splitlines()[-1]
    rc, _ = judge(r, "better")
    assert rc == 0
    return "--judge on a saved JSON: the same verdict, no board"


def test_env_states_run_one_process_each_alternated_with_their_own_environment():
    """`name=env:K=V` states run as child processes, base then candidate, pair after pair, each
    child with its state's environment and only its own --state, and the runs come back labelled
    with the parent's pair index."""
    import argparse
    import json
    import subprocess
    import tempfile
    from tools import block_ab as B
    out = tempfile.mkdtemp()
    a = argparse.Namespace(out=out, label="srun", pairs=2, state=["base=env:QWEN38_SKINNY_SRUN=0",
                                                                    "srun1=env:QWEN38_SKINNY_SRUN=1"],
                           model=None, nvfp4="/nv", fp8_head="/head", ckpt8="/c8", ckpt16="/c16",
                           corpus="/corpus", max_len=4096, max_new=64, warm=8, workloads="prose",
                           precapture=32, fixed=0, learn_cost=False, greedy_check=True, discard=1)
    calls = []

    def fake_call(argv, env):
        calls.append((argv, env))
        child = argv[argv.index("--child") + 1]
        name = argv[argv.index("--state") + 1].split("=", 1)[0]
        ms = 90.0 if name == "base" else 88.0
        json.dump({"runs": [_run(0, name, "prose", ms)],
                   "greedy": {f"{name}/prose": "identical"} if "--greedy-check" in argv else {}},
                  open(child, "w"))
        return 0

    orig = subprocess.call
    subprocess.call = fake_call
    try:
        got = B.run_proc(a, [B.parse_state(x) for x in a.state])
    finally:
        subprocess.call = orig
    names = [c[0][c[0].index("--state") + 1].split("=", 1)[0] for c in calls]
    assert names == ["base", "srun1", "base", "srun1"], names
    assert [c[1]["QWEN38_SKINNY_SRUN"] for c in calls] == ["0", "1", "0", "1"]
    assert all(c[0].count("--state") == 1 for c in calls)
    assert ["--greedy-check" in c[0] for c in calls] == [True, True, False, False]
    assert [(r["pair"], r["state"]) for r in got["runs"]] == [(0, "base"), (0, "srun1"),
                                                             (1, "base"), (1, "srun1")]
    assert got["greedy"] == {"base/prose": "identical", "srun1/prose": "identical"}
    for flag in ("--ckpt8", "--nvfp4", "--fp8-head", "--corpus", "--max-new", "--precapture",
                 "--discard"):
        assert flag in calls[0][0], flag
    return "base srun1 base srun1, each its own env, greedy only in the first pair"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
