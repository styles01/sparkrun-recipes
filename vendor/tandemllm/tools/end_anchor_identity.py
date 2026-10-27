"""The end anchor off is the resident prefix as it was: two trees, one scenario, bit for bit (CPU).

    python tools/end_anchor_identity.py --base ~/tandem-cond-4f47239 --cand ~/tandem-cond-<sha>

Each tree runs the same conversation in a process of its own, on the random 4-layer model and
through its own `server/app.py` loop (`generate_stream`, then `_remember`, as the handler runs a
request) over its own `engine/cache.py`: an opencode-like session of three turns, a title call
between them (a guest), another session that shares the head (the park, and the way back), a
compaction that shares nothing (the pin, and the way back), with no drafter, a wrong chain and a
wrong tree. Every turn records its output tokens, where its prefill started and from what, and
sha256s of the engine's recurrent state, convolution tails and KV rows after it, and of the
resident prefix's rows, anchors, pin and park.

  * the candidate with `--end-anchor off` must equal the base on every record;
  * the candidate with it on must differ (the negative control: the check sees the end anchor),
    and its turns that continue the last generation must resume at its end;
  * with `--on-base` (a base that has the end anchor itself, a commit such as 7954241):
    the candidate with it on must equal the base with it on, every record.

Exit 0 when both hold. Nothing here touches a GPU, a server or a file outside a temporary
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

SCENARIO = r'''
import hashlib, inspect, json, os, sys
for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")
sys.path.insert(0, os.path.join(TREE, "tests"))
sys.path.insert(0, TREE)
import torch
from engine import cache
from engine.penalty import PenaltySpec
from engine.spec import Relax
from server import app
from test_app_loop import FakeTok
from test_cache import fresh, tokens
from test_window_edge import FixedDrafter

CH, BIG = 16, 1 << 30


def h(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def run(drafter, tree):
    kw = dict(stash_bytes=BIG, pin_bytes=BIG, park_bytes=BIG)
    if "end" in inspect.signature(cache.ResidentPrefix).parameters:
        kw["end"] = END
    res = cache.ResidentPrefix(BIG, CH, **kw)
    app.STATE.clear()
    app.STATE.update(engine=fresh(), drafter=drafter, k=4, tree=tree, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=True,
                     prefix_chunk=CH, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=512, default_max_tokens=32, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None, resident=res, disk_cache=None)
    recs = []

    def turn(ids, n, session):
        out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session))
        pre = app.STATE["last_prefill"]
        app._remember(list(ids), out, session)
        eng = app.STATE["engine"]
        L = eng.kv.length
        r = res.report()
        recs.append({"out": out, "reused": pre["reused"], "forwarded": pre["forwarded"],
                     "kind": pre["kind"], "L": L, "S": h(eng.state.S), "conv": h(eng.state.conv),
                     "k": h(eng.kv.k[:, :, :, :L]), "v": h(eng.kv.v[:, :, :, :L]),
                     "res": {k: r[k] for k in ("valid", "anchors", "pin_rows", "pin_anchor_list",
                                               "park_rows", "park_base", "park_anchor_list",
                                               "owner", "park_owner")},
                     "tokens": hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16]})
        return out

    head = tokens(64, seed=1)
    a1 = head + tokens(60, seed=2)
    o1 = turn(a1, 40, "A")
    a2 = a1 + o1 + tokens(25, seed=3)
    o2 = turn(a2, 30, "A")
    turn(tokens(20, seed=4), 8, "A")                       # the title call: a guest
    a3 = a2 + o2 + tokens(18, seed=5)
    o3 = turn(a3, 20, "A")
    turn(head + tokens(70, seed=6), 12, "B")                # another session: A is parked
    a4 = a3 + o3 + tokens(9, seed=7)
    o4 = turn(a4, 16, "A")                                  # A comes back
    turn(tokens(200, seed=8), 6, "A")                       # a compaction: the head is pinned
    turn(a4 + o4 + tokens(11, seed=9), 8, "A")              # the next turn after it
    return recs


out = {}
for label, dr, tree in (("none", None, False), ("chain", FixedDrafter(97, 5, tree_mode=False), False),
                        ("tree", FixedDrafter(97, 5, tree_mode=True), True)):
    out[label] = run(dr, tree)
print(json.dumps(out))
'''


def scenario(tree: str, end: bool) -> dict:
    code = f"TREE = {os.path.abspath(os.path.expanduser(tree))!r}\nEND = {end!r}\n" + SCENARIO
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", ""),
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "TZ": "Europe/Berlin",
           "PYTHONPATH": os.path.abspath(os.path.expanduser(tree))}
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       cwd="/tmp", timeout=900)
    if p.returncode != 0:
        raise SystemExit(f"[endid] {tree} (end={end}) failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="the tree before the end anchor (4f47239)")
    ap.add_argument("--cand", required=True, help="the tree with it")
    ap.add_argument("--on-base", action="store_true",
                    help="the base has the end anchor too: the candidate on must equal it on")
    a = ap.parse_args()
    base = scenario(a.base, False)
    off = scenario(a.cand, False)
    on = scenario(a.cand, True)
    n = sum(len(v) for v in base.values())
    bad = []
    for label in base:
        for i, (x, y) in enumerate(zip(base[label], off[label])):
            if x != y:
                bad.append(f"{label} turn {i}: " + ", ".join(k for k in x if x[k] != y.get(k)))
        if len(base[label]) != len(off[label]):
            bad.append(f"{label}: {len(base[label])} turns against {len(off[label])}")
    for b in bad[:10]:
        print(f"[endid] DIFF {b}")
    differ = sum(x != y for label in base for x, y in zip(base[label], on[label]))
    ends = [(label, i, r["reused"]) for label in on for i, r in enumerate(on[label])
            if r["kind"] == "end"]
    saved = sum(r["reused"] for label in on for r in on[label]) - sum(
        r["reused"] for label in base for r in base[label])
    digest = hashlib.sha256(json.dumps(off, sort_keys=True).encode()).hexdigest()[:12]
    print(f"[endid] off: {n - len(bad)} of {n} turns equal to the base (tokens, resume, state, "
          f"rows, anchors, pin, park; digest {digest})")
    print(f"[endid] on (the control): {differ} of {n} turns differ, {len(ends)} resumed at the "
          f"end of the last generation, {saved} more tokens reused")
    ok = not bad and differ > 0 and len(ends) > 0
    if a.on_base:
        base_on = scenario(a.base, True)
        bad_on = [f"{label} turn {i}: " + ", ".join(k for k in x if x[k] != y.get(k))
                  for label in base_on for i, (x, y) in enumerate(zip(base_on[label], on[label]))
                  if x != y]
        bad_on += [f"{label}: {len(base_on[label])} turns against {len(on[label])}"
                   for label in base_on if len(base_on[label]) != len(on[label])]
        for b in bad_on[:10]:
            print(f"[endid] DIFF on {b}")
        print(f"[endid] on against the base on: {n - len(bad_on)} of {n} turns equal (digest "
              f"{hashlib.sha256(json.dumps(on, sort_keys=True).encode()).hexdigest()[:12]})")
        ok = ok and not bad_on
    print(f"[endid] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
