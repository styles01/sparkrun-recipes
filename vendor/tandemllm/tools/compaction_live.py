"""A real opencode compaction replayed on the live service: the session's last turn,
then its compaction request, timed.

    python3 tools/compaction_live.py --turn T.json --compaction C.json --session ID --phase continue
    python3 tools/compaction_live.py ... --phase prepared   (waits for the summary written ahead)
    python3 tools/compaction_live.py --compaction C.json --session OTHER --phase client

T.json and C.json are captured requests (`{"headers": ..., "body": ...}`, opencode 1.18.34's own
bodies, recorded by a stand-in provider); they hold a user's conversation, so they stay on the box
and out of the repository, and this tool prints counts, times and the summary's section headings,
never its text. Every request carries bearer `d5-test` (the idle check does not count it) and the
given `x-session-id`.

  continue  the turn (its answer capped at --turn-max tokens), then the compaction right away: the
            continuation of the conversation the turn left resident
  prepared  the turn, then wait until the server has written the summary ahead
            (/v1/cache/stats `compaction.prepared` moves), then the compaction
  client    the compaction alone, under a session the server holds nothing of: served as the
            client sent it (its text history read cold), the before-number on the same build
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request


def say(msg: str) -> None:
    print(f"[compaction-live] {msg}", flush=True)


def stats(base: str) -> dict:
    with urllib.request.urlopen(base + "/v1/cache/stats", timeout=30) as r:
        return json.loads(r.read())


def send(base: str, cap: dict, session: str, max_tokens: int | None) -> dict:
    body = dict(cap["body"], stream=True, stream_options={"include_usage": True})
    if max_tokens:
        body["max_tokens"] = max_tokens
    ua = next((v for k, v in cap["headers"].items() if k.lower() == "user-agent"), "opencode")
    hdr = {"Content-Type": "application/json", "User-Agent": ua, "x-session-id": session,
           "Authorization": "Bearer d5-test"}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=hdr)
    t0 = time.monotonic()
    first, usage, text, n_ev = None, None, [], 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    first = first if first is not None else time.monotonic()
                    text.append(piece)
                    n_ev += 1
    done = time.monotonic()
    out = "".join(text)
    u = usage or {}
    return {"ttft_s": round((first or done) - t0, 2), "total_s": round(done - t0, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"), "chars": len(out),
            "sections": re.findall(r"(?m)^#{2,3} [^\n]{1,40}", out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--turn")
    ap.add_argument("--compaction", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--phase", choices=("continue", "prepared", "client"), required=True)
    ap.add_argument("--turn-max", type=int, default=200)
    ap.add_argument("--wait-s", type=float, default=600.0)
    a = ap.parse_args()
    comp = json.load(open(a.compaction))
    if a.phase != "client":
        turn = json.load(open(a.turn))
        r = send(a.base, turn, a.session, a.turn_max)
        say(f"turn: {r['prompt_tokens']} prompt, {r['cached_tokens']} cached, "
            f"{r['completion_tokens']} out, TTFT {r['ttft_s']} s, total {r['total_s']} s")
    if a.phase == "prepared":
        before = (stats(a.base).get("compaction") or {}).get("prepared", 0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.wait_s:
            c = stats(a.base).get("compaction") or {}
            if c.get("prepared", 0) > before:
                say(f"prepared after {time.monotonic() - t0:.0f} s: {c.get('last')}")
                break
            time.sleep(3)
        else:
            say(f"nothing prepared in {a.wait_s:.0f} s: {stats(a.base).get('compaction')}")
    r = send(a.base, comp, a.session, None)
    say(f"compaction ({a.phase}): {r['prompt_tokens']} prompt, {r['cached_tokens']} cached, "
        f"{r['completion_tokens']} out, {r['chars']} chars, TTFT {r['ttft_s']} s, "
        f"total {r['total_s']} s; sections {r['sections']}")
    c = stats(a.base).get("compaction") or {}
    say(f"server: {json.dumps({k: c.get(k) for k in ('continued', 'served', 'client', 'guard', 'no_record', 'stale', 'stopped')})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
