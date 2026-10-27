"""The session spill against the engine it came from, bit for bit, on the real engine (a hold step).

    python -u tools/session_spill_check.py --phase write --dir DIR --out A.json -- <server flags>
    python -u tools/session_spill_check.py --phase read  --dir DIR --ref A.json --out B.json -- <server flags>

THE QUESTION. engine/spill.py writes a live conversation to the NVMe (at a planned
stop, before another client's request evicts it) and reads it back. Its claim is not a cold
prefill's equality: the bytes read back are the bytes that were in the engine, so the next turn
resumes where it would have resumed had they never left. This checks that on the served engine,
loaded in this process by the server's own `_load` from the served flags with `--disk-cache` at
DIR (a directory of its own; the production tier and its sessions are never read or written), so
the grid, the drafter stack and the kernels are the served ones. tests/test_session_spill.py makes
the same checks on the random model.

  write (process 1):
    1. turn 1   an opencode-shaped prompt (`--head-tokens`, a nonce nothing on the box holds),
                through the served loop (`generate_stream`, greedy, `--decode` tokens) and
                `_remember`: the end anchor; every row of every buffer over [0, end) and the end
                anchor's state and the saved anchors by a 64-bit fingerprint
    2. save     the conversation written as a planned stop writes it (`drain`)
    3. turn 2   its next turn (the copy, then a new user message) from the end anchor in RAM:
                the reference tokens
    4. restore  every row clobbered, the resident prefix emptied, the entry read back: rows, end
                anchor and anchors against step 1's fingerprints
    5. turn 2   again, from the restored rows: kind `end`, the same tokens as step 3
    6. evict    the conversation (now turn 2's), then a `--other-tokens` request of another
                session: the spill's write before it (ms, bytes); then the conversation's turn 3:
                read back (`spill`), the restore's ms and bytes, against turn 3 without the other
                request in between (its tokens)
  read (process 2, after a restart of the process): steps 4 and 5 against process 1's files,
     fingerprints and tokens -- the planned stop's path across a restart.

Exit 0 when every comparison is equal. Prints one `[ssc]` line a step; the JSON keeps the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402


def say(msg: str) -> None:
    print(f"[ssc] {msg}", flush=True)


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("usage: session_spill_check.py [options] -- <server/app.py flags>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("write", "read"), required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-tokens", type=int, default=16000)
    ap.add_argument("--other-tokens", type=int, default=10000)
    ap.add_argument("--decode", type=int, default=48)
    ap.add_argument("--nonce", default=None)
    opt = ap.parse_args(sys.argv[1:cut])
    from server import app, diskcache
    from engine import cache
    import disk_cache_live as dcl
    from disk_restore_check import fingerprint, digest
    flags = sys.argv[cut + 1:]
    # the RAM park off and evictions from 4,096 tokens: a conversation of `--head-tokens` is the
    # spill's here (served: longer than the park holds, 35k rows above the shared head)
    a = app.parser().parse_args(flags + ["--disk-cache", os.path.expanduser(opt.dir),
                                         "--session-spill", "on", "--resident-park-gb", "0",
                                         "--session-spill-min-tokens", "4096"])
    t0 = time.time()
    app._load(a)
    S = app.STATE
    S["disk_cache"] = diskcache.start(S, a, None)
    S["spill"] = diskcache.start_spill(S, a, S["disk_cache"], None)
    say(f"engine loaded in {time.time() - t0:.0f} s ({opt.phase})")
    eng, drafter, tok = S["engine"], S["drafter"], S["tok"]
    res, sp = S["resident"], S["spill"]
    assert res is not None and sp is not None, "the resident prefix and the spill"
    ref = json.load(open(opt.ref)) if opt.ref else None
    nonce = opt.nonce or (ref or {}).get("nonce") or time.strftime("%Y%m%d%H%M%S")
    system, tools = dcl.head(opt.head_tokens, f"ssc-{nonce}")
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "Add a --verbose flag to the command line "
                          "tool and print each file it reads."}], "tools": tools}
    prompt = app.build_prompt(body)[0].tolist()
    eos = app.eos_ids(body)
    conv = f"ssc-{nonce}-a"
    rep = {"phase": opt.phase, "nonce": nonce, "prompt": len(prompt), "flags": flags, "steps": {}}
    ok = True

    def ids_of(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False).input_ids)

    def turn(ids: list[int], n: int, session: str) -> tuple[list, dict, float]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = [int(x) for x in app.generate_stream(torch.tensor(ids, device=S.get("device",
                                                                                "cuda")),
                                                   n, eos, None, session)]
        ttft = (time.perf_counter() - t) * 1e3
        pre = dict(S.get("last_prefill") or {})
        app._remember(list(ids), out, session)
        return out, pre, ttft

    def state_fp(e: int) -> dict:
        views = cache._kv_views(eng, drafter)
        return {"rows": [fingerprint(t, ax, e) for t, ax in views],
                "end": [digest(x) for x in res._end_bufs] if res.end is not None else None,
                "anchors": {str(b): [digest(s), digest(c)] for b, (s, c) in
                            sorted(res.anchors.items())},
                "tokens": digest(torch.tensor(res.tokens[:e]))}

    def clobber(e: int) -> None:
        for t, ax in cache._kv_views(eng, drafter):
            t.narrow(ax, 0, min(t.shape[ax], e + 2048)).fill_(7)
        res.clear()
        eng.reset()

    def restore(eid_owner: str) -> dict | None:
        ent = next((x for x in sp.entries.values() if x.man.get("owner") == eid_owner), None)
        if ent is None:
            return None
        torch.cuda.synchronize()
        with torch.no_grad():
            info = sp.restore(res, eng, drafter, ent)
        torch.cuda.synchronize()
        return info

    tool_msg = ids_of("<|im_end|>\n<|im_start|>user\nNow add a test for the flag.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n")
    if opt.phase == "write":
        # 1. turn 1
        out1, pre1, ttft1 = turn(prompt, opt.decode, conv)
        e = res.end["at"] if res.end is not None else None
        assert e is not None, "no end anchor after turn 1"
        fp1 = state_fp(e)
        rep["steps"]["turn1"] = {"out": out1, "prefill": pre1, "end": e, "ms": round(ttft1, 1)}
        say(f"turn 1: {len(prompt)} tokens, {len(out1)} out, end anchor at {e}, "
            f"{len(fp1['anchors'])} anchors")
        # 2. the planned stop's write, under an owner of its own (the eviction below is turn 2's)
        own = res.owner
        res.owner = f"ssc-{nonce}-drain"
        with torch.no_grad():
            saved = sp.save_resident(res, eng, drafter, "drain")
        res.owner = own
        rep["steps"]["save"] = saved
        say(f"save: {saved}")
        # 3. turn 2 from the end anchor in RAM
        ids2 = prompt + out1 + tool_msg
        out2, pre2, ttft2 = turn(ids2, 32, conv)
        rep["steps"]["turn2_ref"] = {"out": out2, "prefill": pre2, "ms": round(ttft2, 1)}
        say(f"turn 2 (RAM): prefill {pre2}")
        rep["ref_fp"], rep["ids2"], rep["out2"] = fp1, ids2, out2
        # 6a. turn 3 reference: the conversation's next turn with nothing in between
        ids3 = ids2 + out2 + tool_msg
        out3_ref, pre3_ref, ttft3_ref = turn(ids3, 16, conv)
        rep["steps"]["turn3_ref"] = {"out": out3_ref, "prefill": pre3_ref, "ms": round(ttft3_ref, 1)}
    # 4. restore into clobbered buffers
    e = (rep["steps"].get("turn1") or ref["steps"]["turn1"])["end"]
    fp_ref = rep.get("ref_fp") or ref["ref_fp"]
    clobber(e)
    info = restore(f"ssc-{nonce}-drain")
    if info is None:
        say("NO drained entry to read back")
        rep["pass"] = False
        json.dump(rep, open(opt.out, "w"), indent=1)
        os._exit(1)
    got = state_fp(e)
    same = {k: got[k] == fp_ref[k] for k in ("rows", "end", "tokens")}
    # the entry keeps the newest 4 grid anchors of turn 1's 10: each one it brought back must be
    # the very state turn 1 had at that line
    same["anchors"] = bool(got["anchors"]) and all(fp_ref["anchors"].get(b) == v
                                                   for b, v in got["anchors"].items())
    same["anchors_restored"] = sorted(int(b) for b in got["anchors"])
    rep["steps"]["restore"] = {"info": info, "same": same}
    ok &= all(v for k, v in same.items() if k != "anchors_restored") and res.valid == e
    say(f"restore: {info['tokens']} tokens, {info['bytes'] / 1e9:.2f} GB in {info['ms']} ms "
        f"({info['blocks_read']} blocks read, {info['blocks_in_place']} in place); against "
        f"turn 1's state: {same}")
    # 5. turn 2 from the restored rows
    ids2, want2 = rep.get("ids2") or ref["ids2"], rep.get("out2") or ref["out2"]
    res.owner = conv
    out2b, pre2b, ttft2b = turn(ids2, 32, conv)
    k = next((i for i, (x, y) in enumerate(zip(out2b, want2)) if x != y), None)
    rep["steps"]["turn2_restored"] = {"out": out2b, "prefill": pre2b, "ms": round(ttft2b, 1),
                                      "same": out2b == want2, "first_diff": k}
    ok &= out2b == want2 and pre2b.get("kind") == "end" and pre2b.get("reused") == e
    say(f"turn 2 (restored): prefill {pre2b}; tokens equal to turn 2 in RAM: {out2b == want2}"
        + (f" (first difference at {k})" if k is not None else ""))
    if opt.phase == "write":
        # 6. an eviction by another session, then the conversation's turn 3
        other_sys, other_tools = dcl.head(opt.other_tokens, f"ssc-{nonce}-other")
        ob = {"messages": [{"role": "system", "content": other_sys},
                           {"role": "user", "content": "Summarise the rules."}],
              "tools": other_tools}
        other = app.build_prompt(ob)[0].tolist()
        saves = sp.stats["saves"]
        _, pre_o, ttft_o = turn(other, 8, f"ssc-{nonce}-other")
        ev = sp.stats["last_save"] if sp.stats["saves"] > saves else None
        rep["steps"]["evict"] = {"other": len(other), "save": ev, "prefill": pre_o,
                                 "ms": round(ttft_o, 1)}
        say(f"evict: another session's {len(other)} tokens; the spill wrote {ev}")
        ids3 = ids2 + out2b + tool_msg
        out3, pre3, ttft3 = turn(ids3, 16, conv)
        r3 = rep["steps"]["turn3_ref"]
        rep["steps"]["turn3_spill"] = {"out": out3, "prefill": pre3, "ms": round(ttft3, 1),
                                       "restore": sp.stats["last_restore"],
                                       "same": out3 == r3["out"]}
        ok &= ev is not None and pre3.get("kind") == "spill" and out3 == r3["out"]
        say(f"turn 3 after the eviction: prefill {pre3}, {ttft3:.0f} ms to the end of 16 tokens "
            f"(in RAM: {r3['ms']:.0f} ms); restore {sp.stats['last_restore']}; tokens equal: "
            f"{out3 == r3['out']}")
    rep["pass"] = bool(ok)
    with open(opt.out, "w") as f:
        json.dump(rep, f, indent=1)
    say(f"{'PASS' if ok else 'FAIL'} ({opt.out})")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
