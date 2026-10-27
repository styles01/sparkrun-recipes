"""ENG-391: one greedy request a kind through a running server, for an A/B of drafters on the Qwen3.5 MoE trunk:
tools/fn_lookup_ab.py's five asks (prose, code, chat, edit, copy; 512 new tokens, thinking off) and a long code answer
(a mobile platformer game in one HTML file, 2,048 new tokens). `--compare` checks every kind the same text in
every run and prints tok/s and tokens a block side by side.

    python tools/q35m_ab.py --url http://127.0.0.1:8000 --label dflash2 --out run.json
    python tools/q35m_ab.py --compare plain.json dflash2.json ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fn_lookup_ab import cases  # noqa: E402

GAME = ("Write a Mario-style platformer game in one HTML file that works on a phone (touch buttons for left, right "
        "and jump), with a scrolling level, coins and enemies. Give the whole file in one code block.")


#: the long code answer through the raw model (no conductor: the agent persona answered the game ask in 134 tokens)
LONG = ("Write a complete Mario-style platformer game as one HTML file for a phone: canvas rendering, touch buttons "
        "for left, right and jump, a scrolling level with platforms, coins, enemies, score and lives. Reply with the "
        "whole file in one code block and nothing else.")


def ask(url: str, text: str, n: int, model: str = "TandemLLM") -> dict:
    body = {"model": model, "max_tokens": n, "temperature": 0, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    t = d.get("timings") or {}
    return {"text": d["choices"][0]["message"].get("content"), "tok_s": t.get("predicted_per_second"),
            "per_block": t.get("tokens_per_block"), "n": t.get("predicted_n"), "prompt": t.get("prompt_n")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--long", action="store_true", help="only the long code answer (2,048 tokens, the raw model)")
    a = ap.parse_args()
    if a.compare:
        runs = [json.load(open(p)) for p in a.compare]
        ok = True
        for c in runs[0]["answers"]:
            same = len({r["answers"][c]["text"] for r in runs if c in r["answers"]}) == 1
            ok = ok and same
            print(f"{c:6} same={same} " + "  ".join(
                f"{r['label']}: {r['answers'][c]['tok_s']} ({r['answers'][c]['per_block']})" for r in runs if c in r["answers"]))
        print("all same:", ok)
        sys.exit(0 if ok else 1)
    health = json.loads(urllib.request.urlopen(a.url + "/health", timeout=10).read())
    out = {"label": a.label, "version": health.get("version"), "drafters": (health.get("weights") or {}).get("drafters"),
           "answers": {}}
    asks = [(k, v, 512) for k, v in cases().items()] + [("game", GAME, 2048)]
    if a.long:
        asks = [("long", LONG, 2048)]
    for name, text, n in asks:
        out["answers"][name] = r = ask(a.url, text, n, "qwen38-spark-engine" if a.long else "TandemLLM")
        print(f"[ab] {a.label} {name}: prompt {r['prompt']}, {r['n']} tokens, {r['tok_s']} tok/s, {r['per_block']} a block",
              flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
