"""On the real engine: an opencode-shaped conversation replayed turn by turn, the resident
prefix against a cold prefill of the same prompt, bit for bit, and the time a turn costs.

    flock ~/.qwen38-box.flock ops/hold.sh 30 -- \\
        bash -c 'set -a; . ops/serve.env; set +a; cd ~/qwen38-spark-engine-stopfix && \\
                 python -u tools/resident_check.py --out /tmp/resident.json -- <server flags>'

The server flags are `server/app.py`'s own (the ones ops/start.sh passes); the engine is loaded
in this process by the server's own `_load`, so the caches, the drafter stack and the chunk grid are
the served ones. No HTTP: the check needs the engine's state, which no endpoint shows.

The conversation is what an agent client sends: a system turn with tool schemas, a user request,
then per turn the previous prompt + a scripted assistant turn (reasoning, one tool call) + a tool
result holding a file of this repository (text that is public and not anybody's data). It starts at
`--start-tokens` and grows by about `--step-tokens` a turn to `--end-tokens`. Between two turns
the real decode loop writes `--decode` tokens past the prompt, as a turn's answer does, and once a
short unrelated request (a title call) runs in between, which is the stash's case.

At the checkpoint turns (`--checks`, token counts; the nearest turn at or above each) the resumed
prefill is compared with a cold prefill of the same prompt: the last row's logits, the recurrent
state and the convolution tail with `torch.equal`, the target KV rows and both draft-KV arms by a
64-bit position-weighted fingerprint over their raw bytes. The cold prefill's time is the "before"
number: what the turn cost when nothing past 13,312 tokens was cached.

A guard thread reads MemAvailable every second and exits the process (code 3) below
`--mem-floor-gib`; the hold's own guard is the second line.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

MEM = {"min_gib": None}


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    return 0.0


def guard(floor: float) -> None:
    while True:
        m = mem_available_gib()
        MEM["min_gib"] = m if MEM["min_gib"] is None else min(MEM["min_gib"], m)
        if m < floor:
            print(f"[guard] MemAvailable {m:.1f} GiB < {floor} GiB: exiting", flush=True)
            os._exit(3)
        time.sleep(1.0)


def fingerprint(t: torch.Tensor, axis: int, n: int, rows: int = 2048) -> int:
    """A 64-bit fingerprint of rows [0, n) along `axis`: every raw 16-bit word times a weight
    that depends on its position, summed with wrap-around. Two buffers that differ in one bit
    differ here with overwhelming probability; nothing is copied off the device."""
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


def repo_files() -> list[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for pat in ("engine/*.py", "engine/drafters/*.py", "server/*.py", "docs/*.md", "tools/*.py"):
        out += sorted(glob.glob(os.path.join(here, pat)))
    return out


TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": list(props)[:1]}}}
    for name, desc, props in (
        ("read", "Read a file from the working tree. Returns its contents with line numbers. "
                 "Use it before editing a file, and read whole files rather than guessing.",
         {"filePath": {"type": "string", "description": "absolute path"},
          "offset": {"type": "number"}, "limit": {"type": "number"}}),
        ("bash", "Run a shell command in the working tree and return its output. Commands run "
                 "non-interactively; long-running commands need a timeout in milliseconds.",
         {"command": {"type": "string"}, "timeout": {"type": "number"},
          "description": {"type": "string"}}),
        ("edit", "Replace an exact string in a file. The old string must be unique.",
         {"filePath": {"type": "string"}, "oldString": {"type": "string"},
          "newString": {"type": "string"}}),
        ("grep", "Search file contents with a regular expression.",
         {"pattern": {"type": "string"}, "path": {"type": "string"}}),
        ("glob", "List files matching a glob pattern.", {"pattern": {"type": "string"}}),
    )]


def agent_tools() -> list:
    """Opencode 1.18's own nine tool schemas (tests/fixtures/opencode_tools_described.json, ~5k tokens), the
    tool block an agent turn actually carries; the short list above if the fixture is missing."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(here, "tests", "fixtures", "opencode_tools_described.json")) as f:
            return json.load(f)
    except OSError:
        return TOOLS


class Conversation:
    """The message list an agent client keeps, and the prompt the server renders from it."""

    def __init__(self, app, files, piece_chars: int = 40000):
        self.app = app
        self.tools = agent_tools()
        # a file longer than `piece_chars` is read in pieces, one a turn, as an agent pages
        self.files = []
        for path in files:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for i in range(0, max(1, len(text)), piece_chars):
                self.files.append((path, text[i:i + piece_chars]))
        self.messages = [
            {"role": "system", "content": "You are a coding agent working in this repository. "
                                          "Read before you edit, keep changes small, and explain "
                                          "what you changed in one short paragraph."},
            {"role": "user", "content": "Read the engine's source files one by one and then "
                                        "summarise how a request flows through the server."}]
        self.turn = 0

    def prompt(self) -> list[int]:
        ids, _, _ = self.app.build_prompt({"messages": self.messages, "tools": self.tools})
        return ids.tolist()

    def add_turn(self, max_chars: int = 0) -> None:
        path, text = self.files.pop(0)
        if max_chars:
            text = text[:max_chars]
        rel = os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cid = f"call_{self.turn:04d}"
        self.messages.append({
            "role": "assistant", "content": "",
            "reasoning_content": f"I should read {rel} next to follow the request path.",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "read",
                                         "arguments": json.dumps({"filePath": "/repo/" + rel})}}]})
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": text})
        self.turn += 1

    def grow_to(self, target: int) -> list[int]:
        ids = self.prompt()
        while len(ids) < target and self.files:
            self.add_turn()
            ids = self.prompt()
        return ids


def _post(port: int, body: dict):
    """A raw HTTP/1.1 POST on a socket this code owns, so it can be closed mid-response."""
    import socket
    raw = json.dumps(body).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=900)
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Content-Type: application/json\r\nContent-Length: " + str(len(raw)).encode()
                 + b"\r\n\r\n" + raw)
    return sock, sock.makefile("rb")


def http_abandon(app, port: int, target: int) -> dict:
    """On the real engine over a real socket: a streamed request whose client leaves
    during the prefill. Reads until the first `: prefill` comment, closes, and times how long the
    engine keeps the lock; then sends the same request again and reads how much it reused."""
    import socket
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", port), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    conv = Conversation(app, list(reversed(repo_files())))
    conv.messages[1]["content"] = "Go through the tools and docs and list what each one measures."
    conv.grow_to(target)
    body = {"model": "t", "messages": conv.messages, "tools": conv.tools, "stream": True,
            "max_tokens": 8, "reasoning_format": "reasoning_content"}
    out = {"prompt": len(conv.prompt())}
    before = dict(app.INFLIGHT)
    t0 = time.perf_counter()
    sock, f = _post(port, body)
    out["status_line"] = f.readline().decode().strip()
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith(b": prefill"):
            out["first_comment_s"] = round(time.perf_counter() - t0, 1)
            out["first_comment"] = line.decode().strip()
            break
        if line.startswith(b"data:"):
            out["data_before_comment"] = True
            break
    time.sleep(2.0)
    t_close = time.perf_counter()
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    f.close()
    sock.close()
    while app.LOCK.locked() and time.perf_counter() - t_close < 600:
        time.sleep(0.05)
    out["lock_free_after_close_s"] = round(time.perf_counter() - t_close, 2)
    out["abandoned_delta"] = app.INFLIGHT["abandoned"] - before["abandoned"]
    res = app.STATE.get("resident")
    out["resident_valid_after_abandon"] = res.valid if res is not None else None
    # the retry: the same request, read to the end
    t1 = time.perf_counter()
    sock, f = _post(port, body)
    raw = f.read().decode(errors="replace")
    f.close()
    sock.close()
    out["retry_s"] = round(time.perf_counter() - t1, 1)
    out["retry_comments"] = sum(1 for ln in raw.splitlines() if ln.startswith(":"))
    out["retry_done"] = "data: [DONE]" in raw
    lp = app.STATE.get("last_prefill") or {}
    out["retry_reused"], out["retry_kind"] = lp.get("reused"), lp.get("kind")
    out["retry_prefill_ms"] = round(lp.get("ms", 0.0), 1)
    httpd.shutdown()
    return out


def compaction_check(app, conv, prefill, decode, state_prints, res) -> dict:
    """The compaction and the turn after it (SRV-97's case), on the real engine: the
    compaction request shares nothing with the conversation, the next one its tool block and
    system text; that one must resume from the pinned head and equal a cold prefill."""
    history = "\n\n".join(f"{m['role']}: {m.get('content') or ''}" for m in conv.messages[1:])
    comp = app.build_prompt({"messages": [{"role": "user", "content":
                                           "Summarise this for a new session.\n\n"
                                           + history[:120000]}]})[0].tolist()
    lg, reused, fwd, ms, kind = prefill(comp, warm=True)
    decode(comp, 16)
    out = {"compaction_tokens": len(comp), "compaction_reused": reused,
           "compaction_ms": round(ms, 1), "pins": res.stats.get("pins"),
           "pin_rows": res.pin_rows}
    nxt = app.build_prompt({"messages": [conv.messages[0], {
        "role": "user", "content": "Continue from this summary: " + history[:6000]}],
        "tools": conv.tools})[0].tolist()
    lg, reused, fwd, ms, kind = prefill(nxt, warm=True)
    n = len(nxt)
    warm_lg = lg[0, -1].clone()
    warm_S, warm_conv = app.STATE["engine"].state.S.clone(), app.STATE["engine"].state.conv.clone()
    warm_fp = state_prints(n)
    clg, _, _, cms, _ = prefill(nxt, warm=False)
    eng = app.STATE["engine"]
    cold_fp = state_prints(n)
    out.update({"next_tokens": n, "next_reused": reused, "next_kind": kind,
                "next_warm_ms": round(ms, 1), "next_cold_ms": round(cms, 1),
                "pin_hits": res.stats.get("pin_hits"),
                "logits_equal": bool(torch.equal(warm_lg, clg[0, -1])),
                "S_equal": bool(torch.equal(warm_S, eng.state.S)),
                "conv_equal": bool(torch.equal(warm_conv, eng.state.conv)),
                "kv_equal": warm_fp["kv_k"] == cold_fp["kv_k"] and warm_fp["kv_v"] == cold_fp["kv_v"],
                "draft_kv_equal": warm_fp["draft"] == cold_fp["draft"]})
    return out


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: resident_check.py [options] -- <server/app.py flags>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-tokens", type=int, default=55000)
    ap.add_argument("--step-tokens", type=int, default=12000)
    ap.add_argument("--end-tokens", type=int, default=190000)
    ap.add_argument("--checks", default="65000,120000,185000")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--small-turns", type=int, default=3,
                    help="turns at the end that add a short tool result, as most agent turns do")
    ap.add_argument("--small-chars", type=int, default=2400)
    ap.add_argument("--guest-at", type=int, default=3, help="turn index before which a guest runs")
    ap.add_argument("--mem-floor-gib", type=float, default=10.0)
    ap.add_argument("--no-cold", action="store_true")
    ap.add_argument("--abandon-port", type=int, default=8011,
                    help="check over a real socket after the replay; 0 = skip")
    ap.add_argument("--abandon-tokens", type=int, default=40000)
    ap.add_argument("--only-abandon", action="store_true", help="skip the replay")
    ap.add_argument("--check-small", action="store_true",
                    help="also check the first of the short turns at the end, the ordinary "
                         "agent turn")
    ap.add_argument("--compaction", action="store_true",
                    help="after the replay: a compaction-shaped request (no tools, no system "
                         "text, the history as one user message), then the next turn with the "
                         "same tool block and system text, checked against a cold prefill")
    opt = ap.parse_args(sys.argv[1:cut])
    threading.Thread(target=guard, args=(opt.mem_floor_gib,), daemon=True).start()

    from server import app
    from engine import cache
    a = app.parser().parse_args(sys.argv[cut + 1:])
    t0 = time.time()
    app._load(a)
    print(f"[check] engine loaded in {time.time() - t0:.0f} s", flush=True)
    subprocess.run([sys.executable, "tools/drop_page_cache.py"], check=False)
    S = app.STATE
    eng, drafter, dev = S["engine"], S["drafter"], S.get("device", "cuda")
    res, store, chunk = S.get("resident"), S.get("state_store"), S.get("prefix_chunk", 0)
    assert res is not None, "the resident prefix is off in these flags"
    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "flags": sys.argv[cut + 1:],
              "turns": [], "checks": [], "guest": None}
    checks = sorted(int(x) for x in opt.checks.split(",") if x)
    conv = Conversation(app, repo_files())

    def prefill(ids, *, warm):
        if drafter is not None:
            drafter.reset()
            if hasattr(drafter, "prime"):
                drafter.prime(list(ids))
        torch.cuda.synchronize()
        t = time.perf_counter()
        info = {}
        with torch.no_grad():
            lg, reused, fwd = cache.prefill(
                eng, drafter, list(ids), dev, store=store if warm else None, chunk=chunk,
                checkpoint=bool(S.get("prefix_cache")) and warm,
                resident=res if warm else None, info=info)
        torch.cuda.synchronize()
        return lg, reused, fwd, (time.perf_counter() - t) * 1e3, info.get("kind")

    def decode(ids, n):
        out = list(app.generate_stream(torch.tensor([ids], device=dev)[0], n, set()))
        return out

    def state_prints(n):
        arms = []
        fn = getattr(drafter, "kv_views", None)
        if fn is not None:
            for t, ax in fn():
                arms.append(fingerprint(t, ax, n))
        return {"kv_k": fingerprint(eng.kv.k, 3, n), "kv_v": fingerprint(eng.kv.v, 3, n),
                "draft": arms}

    prev = None
    target = opt.start_tokens if not opt.only_abandon else 0
    turn = 0
    small_left = opt.small_turns
    while not opt.only_abandon:
        ids = conv.grow_to(target)
        if prev is not None:
            common = next((i for i, (x, y) in enumerate(zip(prev, ids)) if x != y), len(prev))
        else:
            common = 0
        if turn == opt.guest_at:
            title = app.build_prompt({"messages": [
                {"role": "user", "content": "Generate a short title for this conversation: "
                                            "reading the engine's source files."}]})[0].tolist()
            lg, reused, fwd, ms, kind = prefill(title, warm=True)
            decode(title, 24)
            report["guest"] = {"tokens": len(title), "ms": round(ms, 1),
                               "stash": res.report()["stash_rows"]}
            print(f"[check] guest {len(title)} tokens, stash {res.report()['stash_rows']} rows",
                  flush=True)
        lg, reused, fwd, ms, kind = prefill(ids, warm=True)
        row = {"turn": turn, "prompt": len(ids), "prefix_of_previous": common == len(prev or []),
               "common_with_previous": common, "reused": reused, "forwarded": fwd,
               "warm_ms": round(ms, 1), "kind": kind,
               "mem_available_gib": round(mem_available_gib(), 1)}
        print(f"[check] turn {turn}: prompt {len(ids)} reused {reused} ({kind}) "
              f"forwarded {fwd} in {ms / 1e3:.1f} s", flush=True)
        small_check = opt.check_small and small_left == opt.small_turns - 1 and target == 0
        if ((checks and len(ids) >= checks[0]) or small_check) and not opt.no_cold:
            if checks and len(ids) >= checks[0]:
                checks.pop(0)
            n = len(ids)
            warm_lg = lg[0, -1].clone()
            warm_S, warm_conv = eng.state.S.clone(), eng.state.conv.clone()
            warm_fp = state_prints(n)
            clg, _, _, cms, _ = prefill(ids, warm=False)
            cold_fp = state_prints(n)
            chk = {"turn": turn, "prompt": n, "warm_ms": round(ms, 1), "cold_ms": round(cms, 1),
                   "reused": reused,
                   "logits_equal": bool(torch.equal(warm_lg, clg[0, -1])),
                   "argmax": [int(warm_lg.argmax()), int(clg[0, -1].argmax())],
                   "S_equal": bool(torch.equal(warm_S, eng.state.S)),
                   "conv_equal": bool(torch.equal(warm_conv, eng.state.conv)),
                   "kv_equal": warm_fp["kv_k"] == cold_fp["kv_k"]
                   and warm_fp["kv_v"] == cold_fp["kv_v"],
                   "draft_kv_equal": warm_fp["draft"] == cold_fp["draft"],
                   "draft_arms": len(warm_fp["draft"])}
            del warm_S, warm_conv
            report["checks"].append(chk)
            print(f"[check] CHECK turn {turn} ({n} tokens): warm {ms / 1e3:.1f} s vs cold "
                  f"{cms / 1e3:.1f} s; logits {chk['logits_equal']} S {chk['S_equal']} "
                  f"conv {chk['conv_equal']} kv {chk['kv_equal']} draft {chk['draft_kv_equal']}",
                  flush=True)
        report["turns"].append(row)
        # the answer: the real loop, which writes rows past the prompt the next turn must not use
        decode(ids, opt.decode)
        prev = ids
        turn += 1
        if len(ids) >= opt.end_tokens or not conv.files:
            if small_left <= 0 or not conv.files:
                break
            small_left -= 1
            conv.add_turn(max_chars=opt.small_chars)
            target = 0
            continue
        target = len(ids) + opt.step_tokens
    if opt.compaction and not opt.only_abandon:
        report["compaction"] = compaction_check(app, conv, prefill, decode, state_prints, res)
        print(f"[check] compaction: {report['compaction']}", flush=True)
    with open(opt.out, "w") as f:                       # the replay, whatever happens next
        json.dump(report, f, indent=1)
    if opt.abandon_port:
        try:
            report["abandon"] = http_abandon(app, opt.abandon_port, opt.abandon_tokens)
        except Exception as exc:                        # noqa: BLE001
            report["abandon"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[check] abandon: {report['abandon']}", flush=True)
    report["resident"] = res.report()
    report["mem_available_min_gib"] = round(MEM["min_gib"] or 0.0, 1)
    report["all_equal"] = all(c["logits_equal"] and c["S_equal"] and c["conv_equal"]
                              and c["kv_equal"] and c["draft_kv_equal"]
                              for c in report["checks"] + ([report["compaction"]]
                                                           if report.get("compaction") else []))
    with open(opt.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[check] wrote {opt.out}: all_equal={report['all_equal']} "
          f"min MemAvailable {report['mem_available_min_gib']} GiB", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
