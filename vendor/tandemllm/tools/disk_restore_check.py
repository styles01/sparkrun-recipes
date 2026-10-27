"""The disk tier's restore against a cold prefill on the real engine, bit for bit (a hold step).

    python -u tools/disk_restore_check.py --phase write --dir DIR --out A.json -- <server flags>
    python -u tools/disk_restore_check.py --phase read  --dir DIR --ref A.json --out B.json -- <server flags>

THE QUESTION (2026-10-02 22:08). tools/disk_cache_live.py found the disk hit working
(15,360 tokens restored, 1.79 GB in 334 ms) and its `same_text` gate false. Two explanations: the
restored state is not the cold prefill's (rows, recurrent states, convolution tails or the drafters'
rows written or read wrong, a layout or dtype that differs between the write and the read, Triton's
autotune choosing another prefill configuration after a restart, an anchor off the cold prefill's
1,024 grid), or the text differs where nothing in the engine does. This answers the first one on the
served engine, with no HTTP: the engine is loaded in this process by the server's own `_load` from
the served flags (`--disk-cache` pointed at DIR, a directory of its own that the write phase finds
empty; the production tier is never read or written), so the grid, the drafter stack and the kernels are the served ones.

  write (process 1):
    1. cold   an opencode-shaped prompt (a synthetic head of `--head-tokens`, nothing on the box
              holds it, then one user message), prefilled cold through `cache.prefill` with the
              resident prefix, whose anchors are the cold prefill's own state at every 1,024
              line; the last row's logits, the recurrent state, the convolution tails, the target
              KV rows and the drafters' rows by a 64-bit fingerprint, the state at every line
    2. text   the same prompt through the served loop (`generate_stream`, greedy, `--decode`
              tokens), as the session's first long request: the tier writes its head anchors and
              the chunks below them behind the request; the writer is waited for
    3. direct every row and state buffer clobbered, the tier's longest anchor read back
              (`DiskTier.find`, `restore`): its recurrent state and tails against the cold
              prefill's at that line (`torch.equal`); the rows it wrote are step 4's
    4. resume the prefill resumed from the disk (`cache.prefill` with only the tier): logits,
              state and every row against step 1
    5. text   the served loop again, the caches in RAM emptied: its prefill must come from the
              disk, and its tokens must equal step 2's
  read (process 2, after a restart of the process -- a fresh Triton autotune): step 1 again,
     against process 1's numbers (the cold prefill itself across a restart), then 3, 4 and 5
     against process 1's files and numbers.

Exit 0 when every comparison is equal. Prints one `[drc]` line a step; the JSON keeps the numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402


def say(msg: str) -> None:
    print(f"[drc] {msg}", flush=True)


def fingerprint(t: torch.Tensor, axis: int, n: int, rows: int = 2048) -> int:
    """tools/resident_check.py's: a 64-bit position-weighted sum of the raw 16-bit words of rows
    [0, n) along `axis`; nothing is copied off the device."""
    acc = 0
    for s in range(0, n, rows):
        e = min(n, s + rows)
        x = t.narrow(axis, s, e - s).contiguous()
        w = x.view(torch.int16).to(torch.int64).flatten()
        idx = torch.arange(w.numel(), device=w.device, dtype=torch.int64)
        wt = (idx * 2654435761 + s * 40503) % 2147483647 + 1
        acc = (acc * 1000003 + int((w * wt).sum().item())) & ((1 << 64) - 1)
        del x, w, idx, wt
    return acc


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                          ).hexdigest()[:24]


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("usage: disk_restore_check.py [options] -- <server/app.py flags>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("write", "read"), required=True)
    ap.add_argument("--dir", required=True, help="the tier's directory for this check")
    ap.add_argument("--ref", help="the write phase's JSON (read phase)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-tokens", type=int, default=16000)
    ap.add_argument("--decode", type=int, default=48)
    ap.add_argument("--nonce", default=None)
    opt = ap.parse_args(sys.argv[1:cut])
    from server import app, diskcache
    from engine import cache
    import disk_cache_live as dcl
    flags = sys.argv[cut + 1:]
    a = app.parser().parse_args(flags + ["--disk-cache", os.path.expanduser(opt.dir)])
    t0 = time.time()
    app._load(a)
    S = app.STATE
    S["disk_cache"] = diskcache.start(S, a, None)      # as `_serve` starts it, with no journal
    say(f"engine loaded in {time.time() - t0:.0f} s ({opt.phase})")
    eng, drafter, dev, tok = S["engine"], S["drafter"], S.get("device", "cuda"), S["tok"]
    res, store, tier, C = S.get("resident"), S.get("state_store"), S.get("disk_cache"), \
        int(S.get("prefix_chunk") or 0)
    assert res is not None and tier is not None and tier.on and C > 0, "resident, tier, grid"
    ref = json.load(open(opt.ref)) if opt.ref else None
    nonce = opt.nonce or (ref or {}).get("nonce") or time.strftime("%Y%m%d%H%M%S")
    system, tools = dcl.head(opt.head_tokens, f"drc-{nonce}")
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "Add a --verbose flag to the command line "
                          "tool and print each file it reads."}], "tools": tools}
    prompt = app.build_prompt(body)[0]
    ids = prompt.tolist()
    n = len(ids)
    tag = diskcache.request_tag(None, "opencode", ids, tok)
    rep = {"phase": opt.phase, "nonce": nonce, "prompt": n, "head_end": tag["head_end"],
           "flags": flags, "steps": {}}
    say(f"prompt {n} tokens, the user message starts at {tag['head_end']}")
    views = cache._kv_views(eng, drafter)

    def rows_fp(m: int) -> list:
        return [fingerprint(t, ax, m) for t, ax in views]

    def empty_ram():
        res.clear()
        if store is not None:
            store.clear()

    def prefill(**kw):
        if drafter is not None:
            drafter.reset()
            if hasattr(drafter, "prime"):
                drafter.prime(list(ids))
        info = {}
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            lg, reused, fwd = cache.prefill(eng, drafter, list(ids), dev, chunk=C, info=info,
                                            **kw)
        torch.cuda.synchronize()
        return lg[0, -1].float().clone(), reused, info.get("kind"), (time.perf_counter() - t) * 1e3

    def text(conv: str) -> tuple[list, dict]:
        empty_ram()
        out = list(app.generate_stream(prompt, opt.decode, app.eos_ids(body), None, conv,
                                       disk_tag=tag))
        return [int(x) for x in out], dict(S.get("last_prefill") or {})

    def wait_writer(limit: float = 300.0) -> dict:
        t = time.time()
        while time.time() - t < limit:
            r = tier.report()
            if not r.get("open_jobs") and tier._q.empty():
                return r
            time.sleep(0.5)
        return tier.report()

    # 1. the cold prefill, its state at every line from the resident prefix's own anchors
    empty_ram()
    eng.reset()
    lg, _, _, ms = prefill(resident=res)
    cold = {"logits": digest(lg), "argmax": int(lg.argmax()), "S": digest(eng.state.S),
            "conv": digest(eng.state.conv), "rows": rows_fp(n), "ms": round(ms, 1),
            "lines": {str(b): [digest(s), digest(c)] for b, (s, c) in sorted(res.anchors.items())}}
    cold_lg, cold_S, cold_conv = lg, eng.state.S.clone(), eng.state.conv.clone()
    cold_anchors = {b: (s.clone(), c.clone()) for b, (s, c) in res.anchors.items()}
    rep["steps"]["cold"] = cold
    ok = True
    if ref is not None:
        same = {k: cold[k] == ref["steps"]["cold"][k] for k in ("logits", "S", "conv", "rows",
                                                                 "lines")}
        rep["steps"]["cold"]["same_as_write"] = same
        ok &= all(same.values())
        say(f"cold prefill {ms / 1e3:.1f} s; against the write process's cold prefill: {same}")
    else:
        say(f"cold prefill {ms / 1e3:.1f} s, {len(cold['lines'])} lines")

    # 2. the served loop, the session's first long request: the tier writes its head
    if opt.phase == "write":
        toks, pre = text(f"drc-{nonce}-a")
        w = wait_writer()
        rep["steps"]["text_cold"] = {"tokens": toks, "prefill": pre}
        rep["entries"] = [e for e in w.get("top") or []]
        say(f"text, cold: {len(toks)} tokens, prefill {pre}; the tier: {w.get('entries')} "
            f"entries, open jobs {w.get('open_jobs')}, events "
            f"{[e.get('kind') for e in (w.get('events') or [])][-6:]}")
    else:
        rep["steps"]["text_cold"] = ref["steps"]["text_cold"]

    # 3. the longest anchor read back into clobbered buffers
    empty_ram()
    eng.reset()
    for t, ax in views:
        t.narrow(ax, 0, n).fill_(7)
    keys = list(ids)
    hashes = cache._hash_memo(keys)
    hit = tier.find(keys, hashes, n - 1, 0, 0)
    if hit is None:
        say("NO disk anchor for the prompt: nothing to compare")
        rep["pass"] = False
        json.dump(rep, open(opt.out, "w"), indent=1)
        sys.stdout.flush()
        os._exit(1)
    L = hit.L
    if drafter is not None:
        drafter.reset()
    t = time.perf_counter()
    done = tier.restore(eng, drafter, hit)
    torch.cuda.synchronize()
    direct = {"line": L, "restored": bool(done), "ms": round((time.perf_counter() - t) * 1e3, 1),
              "on_grid": L % C == 0, "below_user": tag["head_end"] is None or L <= tag["head_end"]}
    if done:
        cs, cc = cold_anchors.get(L, (None, None))
        direct["S_equal"] = cs is not None and bool(torch.equal(eng.state.S, cs))
        direct["conv_equal"] = cc is not None and bool(torch.equal(eng.state.conv, cc))
        if ref is not None:
            direct["S_equal_write"] = digest(eng.state.S) == ref["steps"]["cold"]["lines"][str(L)][0]
            direct["conv_equal_write"] = (digest(eng.state.conv)
                                          == ref["steps"]["cold"]["lines"][str(L)][1])
    rep["steps"]["direct"] = direct

    # 4. the prefill resumed from the disk
    eng.reset()
    lg, reused, kind, ms = prefill(disk=tier, disk_tag=tag)
    resume = {"kind": kind, "reused": reused, "ms": round(ms, 1),
              "logits_equal": bool(torch.equal(lg, cold_lg)),
              "argmax": [int(lg.argmax()), int(cold_lg.argmax())],
              "S_equal": bool(torch.equal(eng.state.S, cold_S)),
              "conv_equal": bool(torch.equal(eng.state.conv, cold_conv))}
    rows = rows_fp(n)
    resume["rows_equal"] = rows == cold["rows"]
    resume["rows_differ"] = [i for i, (x, y) in enumerate(zip(rows, cold["rows"])) if x != y]
    if not resume["logits_equal"]:
        top = torch.topk(cold_lg, 2)
        resume["max_abs_diff"] = float((lg - cold_lg).abs().max())
        resume["cold_top2_gap"] = float(top.values[0] - top.values[1])
    rep["steps"]["resume"] = resume
    direct_ok = bool(done and direct.get("S_equal") and direct.get("conv_equal")
                     and direct["on_grid"])
    resume_ok = (kind == "disk" and resume["logits_equal"] and resume["S_equal"]
                 and resume["conv_equal"] and resume["rows_equal"])
    ok &= direct_ok and resume_ok
    say(f"direct: line {L} (on the grid {direct['on_grid']}, below the user message "
        f"{direct['below_user']}), restored {done} in {direct['ms']} ms; state at the line equal "
        f"to the cold prefill's: S {direct.get('S_equal')} conv {direct.get('conv_equal')}"
        + (f"; to the write process's: S {direct.get('S_equal_write')} conv "
           f"{direct.get('conv_equal_write')}" if ref is not None else ""))
    say(f"resume: {kind}, reused {reused}, {ms / 1e3:.2f} s; against the cold prefill: logits "
        f"{resume['logits_equal']} S {resume['S_equal']} conv {resume['conv_equal']} rows "
        f"{resume['rows_equal']} (differ: {resume['rows_differ']} of {len(views)} buffers)")

    # 5. the served loop from the disk
    toks, pre = text(f"drc-{nonce}-b")
    want = rep["steps"]["text_cold"]["tokens"]
    k = next((i for i, (x, y) in enumerate(zip(toks, want)) if x != y), None)
    rep["steps"]["text_disk"] = {"tokens": toks, "prefill": pre, "same": toks == want,
                                 "first_diff": k}
    ok &= toks == want and pre.get("kind") == "disk"
    say(f"text from the disk: prefill {pre}; tokens equal to the cold text: {toks == want}"
        + (f" (first difference at {k})" if k is not None else ""))
    rep["pass"] = bool(ok)
    with open(opt.out, "w") as f:
        json.dump(rep, f, indent=1)
    say(f"{'PASS' if ok else 'FAIL'} ({opt.out})")
    tier.close()
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
