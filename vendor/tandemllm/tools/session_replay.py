"""An agent session over HTTP, shaped like an opencode night, against a running server.

    python tools/session_replay.py --base http://127.0.0.1:8011/v1 --out cand.json
    python tools/session_replay.py --compare base.json cand.json

What an agent client sends, in order, with synthetic content only (this repository's own files as
tool results, its docs as the system text):

  1. a conversation that grows turn by turn: one long tool block and system text, a user
     request, then per turn a scripted assistant turn (reasoning, one tool call) and a tool
     result, each turn a few hundred tokens longer than the last;
  2. a compaction request: no tools, no system text, the whole history rendered as text in one
     user message (what opencode sends when the prompt nears the window);
  3. the next turn after it: the same tool block and system text, then the summary;
  4. more turns of the new conversation, one of them with the date line in the system text
     changed (a session that crosses midnight);
  5. a short chat, many turns under the session cache's size cap, with tool calls, so its next
     turn can resume from a session snapshot.

Every request is greedy with a small token budget. Per request the file records what the
server reported (`timings`: forwarded and cached tokens, prompt_ms, the cache it resumed from)
and the answer's text; at the end `/v1/cache/stats`. `--compare` holds two files request by
request: the texts, and the forwarded tokens and prompt time of each phase.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# `--x-tandem raw`: a server with the conductor serves one model, `TandemLLM`, and its raw path is
# a header (`X-Tandem: raw`), not a model name
HEADERS = {"Content-Type": "application/json"}


def _post(base: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=dict(HEADERS))
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_client_ms"] = (time.perf_counter() - t) * 1e3
    return out


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _pieces(pattern_list, size: int) -> list[str]:
    out = []
    for pat in pattern_list:
        for path in sorted(glob.glob(os.path.join(HERE, pat))):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            rel = os.path.relpath(path, HERE)
            for i in range(0, len(text), size):
                out.append((rel, text[i:i + size]))
    return out


def _tools() -> list:
    with open(os.path.join(HERE, "tests", "fixtures", "opencode_tools_described.json")) as f:
        return json.load(f)


def _system(chars: int, date: str) -> str:
    """A long system text (the docs), with the date line last, as opencode writes it."""
    text = ""
    for path in sorted(glob.glob(os.path.join(HERE, "docs", "*.md"))):
        with open(path, encoding="utf-8") as f:
            text += f.read() + "\n"
        if len(text) >= chars:
            break
    return text[:chars] + f"\n\nToday's date: {date}\n"


class Session:
    def __init__(self, system: str, request: str, files):
        self.tools = _tools()
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": request}]
        self.files = files
        self.turn = 0

    def add_turn(self, chars: int) -> None:
        path, text = self.files.pop(0)
        cid = f"call_{self.turn:04d}"
        self.messages.append({
            "role": "assistant", "content": "",
            "reasoning_content": f"I should read {path} next.",
            "tool_calls": [{"id": cid, "type": "function", "function": {
                "name": "read", "arguments": json.dumps({"filePath": "/repo/" + path})}}]})
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": text[:chars]})
        self.turn += 1

    def body(self, max_tokens: int) -> dict:
        return {"model": "replay", "messages": self.messages, "tools": self.tools,
                "max_tokens": max_tokens, "temperature": 0, "stream": False}

    def as_text(self) -> str:
        """The history as one block of text, the way a compaction request carries it."""
        out = []
        for m in self.messages[1:]:
            if m["role"] == "assistant":
                calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})"
                                  for c in m.get("tool_calls", []))
                out.append(f"assistant: {m.get('content', '')} {calls}")
            else:
                out.append(f"{m['role']}: {m['content']}")
        return "\n\n".join(out)


def _record(kind: str, out: dict) -> dict:
    t = out.get("timings") or {}
    u = out.get("usage") or {}
    msg = (out.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "") + "\x00" + \
        json.dumps([c.get("function") for c in (msg.get("tool_calls") or [])], sort_keys=True)
    return {"kind": kind, "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "forwarded": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "ttft_ms": t.get("ttft_ms"), "cache_source": t.get("cache_source"),
            "completion": u.get("completion_tokens"), "client_ms": round(out["_client_ms"], 1),
            "finish": (out.get("choices") or [{}])[0].get("finish_reason"),
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}


def run(a) -> dict:
    files = _pieces(("engine/*.py", "server/*.py", "tools/*.py"), a.piece_chars)
    rep = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "base": a.base, "args": vars(a),
           "requests": []}

    def send(kind, body):
        out = _post(a.base, body)
        r = _record(kind, out)
        rep["requests"].append(r)
        print(f"[replay] {len(rep['requests']):3d} {kind:10s} prompt {r['prompt']} cached "
              f"{r['cached']} forwarded {r['forwarded']} prompt_ms {r['prompt_ms']} "
              f"({r['cache_source']})", flush=True)
        return out

    s = Session(_system(a.system_chars, "Tue Sep 29 2026"),
                "Read the engine's source files and summarise how a request flows.", files)
    for _ in range(a.first_files):
        s.add_turn(a.piece_chars)
    send("first", s.body(a.max_tokens))
    for _ in range(a.turns):
        s.add_turn(a.turn_chars)
        send("turn", s.body(a.max_tokens))
    comp = {"model": "replay", "stream": False, "temperature": 0, "max_tokens": a.summary_tokens,
            "messages": [{"role": "user", "content":
                          "Summarise the conversation below for a new session.\n\n" + s.as_text()}]}
    out = send("compaction", comp)
    summary = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    s2 = Session(_system(a.system_chars, "Tue Sep 29 2026"),
                 "Continue from this summary of the work so far:\n\n" + summary
                 + "\n\n" + s.as_text()[:a.summary_chars], files)
    send("post", s2.body(a.max_tokens))
    for i in range(a.after_turns):
        s2.add_turn(a.turn_chars)
        if i == a.after_turns // 2:
            s2.messages[0]["content"] = _system(a.system_chars, "Wed Sep 30 2026")
            send("date", s2.body(a.max_tokens))
        else:
            send("after", s2.body(a.max_tokens))
    chat = Session("You are a helpful assistant.", "Help me read a few files.", files)
    for _ in range(a.chat_turns):
        chat.add_turn(a.chat_chars)
        send("chat", chat.body(a.max_tokens))
    root = a.base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        rep["cache_stats"] = _get(root + "/v1/cache/stats")
    except Exception as exc:                                        # noqa: BLE001
        rep["cache_stats"] = {"error": str(exc)}
    return rep


def compare(paths: list[str]) -> None:
    a, b = (json.load(open(p)) for p in paths)
    ra, rb = a["requests"], b["requests"]
    n = min(len(ra), len(rb))
    same = sum(1 for x, y in zip(ra, rb) if x["text"] == y["text"])
    diffs = [i for i in range(n) if ra[i]["text"] != rb[i]["text"]]
    print(f"texts: {same} of {n} identical" + (f"; differ at {diffs}" if diffs else ""))
    kinds = []
    for r in ra:
        if r["kind"] not in kinds:
            kinds.append(r["kind"])
    print(f"{'phase':12s}{'n':>4s}{'fwd A':>10s}{'fwd B':>10s}{'prompt_ms A':>14s}"
          f"{'prompt_ms B':>14s}")
    for k in kinds:
        xa = [r for r in ra if r["kind"] == k]
        xb = [r for r in rb if r["kind"] == k]
        fa = sum(r["forwarded"] or 0 for r in xa)
        fb = sum(r["forwarded"] or 0 for r in xb)
        pa = sum(r["prompt_ms"] or 0 for r in xa)
        pb = sum(r["prompt_ms"] or 0 for r in xb)
        print(f"{k:12s}{len(xa):>4d}{fa:>10d}{fb:>10d}{pa:>14.0f}{pb:>14.0f}")
    for name, rep in (("A", a), ("B", b)):
        cs = rep.get("cache_stats") or {}
        st, sx, res = cs.get("state_store") or {}, cs.get("suffix_store") or {}, cs.get("resident") or {}
        print(f"{name}: store puts {st.get('puts')} evictions {st.get('evictions')} hits_session "
              f"{st.get('hits_session')} declined_short {st.get('declined_short')} declined_big "
              f"{st.get('declined_big')}; suffix appended {sx.get('appended')} rebuilds "
              f"{sx.get('rebuilds')} rebuild_ms_total {sx.get('rebuild_ms_total')}; resident hits "
              f"{res.get('hits')} tail_hits {res.get('tail_hits')} pins {res.get('pins')} "
              f"pin_hits {res.get('pin_hits')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--system-chars", type=int, default=60000,
                    help="the system text's length; with the tool block it is the shared head")
    ap.add_argument("--first-files", type=int, default=3)
    ap.add_argument("--piece-chars", type=int, default=12000)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--turn-chars", type=int, default=2400)
    ap.add_argument("--after-turns", type=int, default=6)
    ap.add_argument("--summary-tokens", type=int, default=64)
    ap.add_argument("--summary-chars", type=int, default=6000)
    ap.add_argument("--chat-turns", type=int, default=16)
    ap.add_argument("--chat-chars", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--x-tandem", default="", choices=("", "raw", "shadow", "on"),
                    help="send X-Tandem with every request (raw: the raw path of a conductor server)")
    a = ap.parse_args()
    if a.x_tandem:
        HEADERS["X-Tandem"] = a.x_tandem
    if a.compare:
        compare(a.compare)
        return
    rep = run(a)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[replay] wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
