"""The session spill on the live service, from the box: an eviction, then a restart.

    python3 tools/spill_live.py --phase evict  --state S.json [--head-tokens 70000]
    (restart :8000 through ops/stop.sh and ops/start.sh: the drain writes, the start reads back)
    python3 tools/spill_live.py --phase after  --state S.json
    python3 tools/spill_live.py --phase endcheck --state S.json   (instead of `after`)

One conversation of its own (a synthetic opencode-shaped head nothing else holds, about 45,000 tokens with no tool block: longer than the RAM park holds, `X-Tandem: raw`
so no note moves the prompt, thinking off, greedy, bearer `d4-test` so the idle check does not
count it): `evict` sends turn 1, then a `--other-tokens` request of another session (bearer
`d4-test-other`, over the guest size: it replaces the resident prompt), then turn 2, and reports
what turn 2 reused, where from (`spill`), its TTFT, and the spill's write before the other request
and its read for turn 2 (GET /v1/cache/stats); then turn 3, so the conversation is resident at the
restart. `after` sends turn 4 and reports what it reused and its TTFT. Prints `[spill-live]` lines;
the state file keeps the conversation and the numbers. Run it only inside an idle window: the other
request evicts whatever is resident.

`endcheck` (`evict` before the restart with `--other-tokens` between the guest size
and the spill's 16,384, so the other conversation is never on the spill): the other session's
prompt again under a new session (cold: it evicts the conversation, and its second cold prefill
admits its head to the disk tier), turn 4 (read back from the spill), the same prompt under a third
session (now from the store or the disk tier, the request that dropped the end anchor before the
eviction wrote it on 0.3.23), the spill's save before it (`end` must be true), then turn 5, which
must reuse every token up to the end of turn 4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disk_cache_live as dcl  # noqa: E402

UA = dcl.OPENCODE_UA
USERS = ["Add a --verbose flag to the command line tool and print each file it reads.",
         "Now add a test for the flag, in the style of the existing tests.",
         "Rename the function parse_line to parse_record everywhere.",
         "Summarise what changed in three bullet points."]


def say(msg: str) -> None:
    print(f"[spill-live] {msg}", flush=True)


def chat(server: str, messages: list, tools: list, session: str, bearer: str,
         max_tokens: int = 64) -> dict:
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": messages, **({"tools": tools} if tools else {}), "temperature": 0,
            "seed": 7,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA,
                                          "x-session-id": session, "X-Tandem": "raw",
                                          "Authorization": f"Bearer {bearer}"})
    t0 = time.monotonic()
    first, text, usage, timings = None, [], None, None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                if piece:
                    first = first if first is not None else time.monotonic()
                    text.append(piece)
    t = timings or {}
    return {"text": "".join(text), "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "reused": int(t.get("cache_n") or 0), "cache_source": t.get("cache_source"),
            "ttft_ms_client": round((first - t0) * 1e3, 1) if first is not None else None,
            "ttft_ms_server": t.get("ttft_ms")}


def stats(server: str, token: str | None) -> dict:
    try:
        return dcl.call(server, "GET", "/v1/cache/stats", token).get("spill") or {}
    except Exception as exc:                                       # noqa: BLE001
        return {"error": str(exc)[:200]}


def line(label: str, r: dict) -> str:
    return (f"{label}: {r['prompt_tokens']} prompt tokens, reused {r['reused']} "
            f"({r['cache_source']}), TTFT {r['ttft_ms_server']} ms server / "
            f"{r['ttft_ms_client']} ms client")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("evict", "after", "endcheck"), required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--server", default=dcl.BASE)
    ap.add_argument("--head-tokens", type=int, default=70000)
    ap.add_argument("--other-tokens", type=int, default=20000)
    ap.add_argument("--token-file", default="~/tandem-staircut/ops/secrets.env")
    a = ap.parse_args()
    server = a.server.rstrip("/")
    token = dcl.admin_token(a.token_file)
    if a.phase == "evict":
        nonce = time.strftime("%Y%m%d%H%M%S")
        system, tools = dcl.head(a.head_tokens, f"spill-live-{nonce}")
        # no tools: the model answers in text, so the client's copy of each answer is the text
        # the engine wrote and the next turn can resume at the end anchor
        tools = []
        st = {"nonce": nonce, "system": system, "tools": tools, "session": f"spill-live-{nonce}",
              "messages": [{"role": "system", "content": system}], "runs": {}}
    else:
        st = json.load(open(a.state))
    msgs, tools, ses = st["messages"], st["tools"], st["session"]

    def turn(k: int, label: str) -> dict:
        msgs.append({"role": "user", "content": USERS[k]})
        r = chat(server, msgs, tools, ses, "d4-test")
        msgs.append({"role": "assistant", "content": r["text"]})
        st["runs"][label] = r
        say(line(label, r))
        return r

    if a.phase == "evict":
        turn(0, "turn1")
        before = stats(server, token)
        osys, _ = dcl.head(a.other_tokens, f"spill-live-{st['nonce']}-other")
        otools = []
        ro = chat(server, [{"role": "system", "content": osys},
                           {"role": "user", "content": "List the rules by number."}], otools,
                  f"spill-live-{st['nonce']}-other", "d4-test-other", max_tokens=16)
        mid = stats(server, token)
        st["runs"]["other"] = ro
        say(line("another session", ro))
        st["spill_save"] = mid.get("last_save") if mid.get("saves", 0) > before.get("saves", 0) \
            else None
        say(f"the spill's write before it: {st['spill_save']}")
        r2 = turn(1, "turn2")
        after = stats(server, token)
        st["spill_restore"] = after.get("last_restore") \
            if after.get("restores", 0) > mid.get("restores", 0) else None
        say(f"the spill's read for turn 2: {st['spill_restore']}")
        st["pass_evict"] = bool(st["spill_save"] and r2["cache_source"] == "spill"
                                and r2["reused"] >= st["runs"]["turn1"]["prompt_tokens"])
        turn(2, "turn3")
        say(f"evict: {'PASS' if st['pass_evict'] else 'FAIL'}; the conversation is resident at "
            f"{st['runs']['turn3']['prompt_tokens']} prompt tokens")
    elif a.phase == "endcheck":
        osys, _ = dcl.head(a.other_tokens, f"spill-live-{st['nonce']}-other")

        def other(k: int) -> dict:
            r = chat(server, [{"role": "system", "content": osys},
                              {"role": "user", "content": "List the rules by number."}], [],
                     f"spill-live-{st['nonce']}-other{k}", "d4-test-other", max_tokens=16)
            st["runs"][f"other{k}"] = r
            say(line(f"another session ({k})", r))
            return r
        other(2)
        r4 = turn(3, "turn4")
        before = stats(server, token)
        r3 = other(3)
        mid = stats(server, token)
        save = mid.get("last_save") if mid.get("saves", 0) > before.get("saves", 0) else None
        st["spill_save_end"] = save
        say(f"the spill's write before the third: {save}")
        msgs.append({"role": "user", "content": "And one more time, shorter."})
        r5 = chat(server, msgs, tools, ses, "d4-test")
        msgs.append({"role": "assistant", "content": r5["text"]})
        st["runs"]["turn5"] = r5
        say(line("turn5", r5))
        # turn 4's end: its prompt plus what it wrote, which turn 5's prompt starts with
        st["pass_end"] = bool(save and save.get("end") and r5["cache_source"] == "spill"
                              and r5["reused"] > r4["prompt_tokens"])
        say(f"endcheck: {'PASS' if st['pass_end'] else 'FAIL'} (third request from "
            f"{r3['cache_source']}, save end {save and save.get('end')}, turn 5 reused "
            f"{r5['reused']} of {r5['prompt_tokens']}, turn 4's prompt {r4['prompt_tokens']})")
    else:
        r4 = turn(3, "turn4")
        prev = st["runs"]["turn3"]["prompt_tokens"]
        st["pass_restart"] = r4["reused"] >= prev
        say(f"after the restart: {'PASS' if st['pass_restart'] else 'FAIL'} (reused {r4['reused']} "
            f"of a prompt that continues turn 3's {prev})")
    with open(a.state, "w") as f:
        json.dump(st, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
