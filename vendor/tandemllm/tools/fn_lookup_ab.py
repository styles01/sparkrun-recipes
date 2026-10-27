"""ENG-229 on the box: one greedy request a class through a running server, for an A/B of drafters.

    python tools/fn_lookup_ab.py --url http://127.0.0.1:8011 --label mtp --out run-mtp.json
    python tools/fn_lookup_ab.py --compare run-none.json run-mtp.json run-look.json

Classes (512 greedy tokens, thinking off): prose, code and chat (the asks of tools/exl3_fastrun.py, no
context), edit (a 4,000-character piece of bench/heldout_code.txt with one name renamed throughout, the
whole file back) and copy (a 3,000-character piece of bench/heldout_prose.txt back unchanged in a code
block). Each answer keeps its text and the server's `timings` (tok/s, tokens a block).
`--compare` checks every class the same text in every run (greedy speculation is lossless) and prints
tok/s and tokens a block side by side, one run each: no sweep, no repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cases() -> dict:
    sys.path.insert(0, ROOT)
    from tools.exl3_fastrun import CHAT, CODE, PROSE
    code = open(os.path.join(ROOT, "bench", "heldout_code.txt")).read()[:4000]
    prose = open(os.path.join(ROOT, "bench", "heldout_prose.txt")).read()[:3000]
    return {
        "prose": PROSE, "code": CODE, "chat": CHAT,
        "edit": "Here is a file:\n\n```\n" + code + "\n```\n\nRename every `self` to `this` and give me the "
                "whole file back in one code block, nothing else.",
        "copy": "Here is a text:\n\n" + prose + "\n\nGive me this text back unchanged in one code block. "
                "Nothing else.",
    }


def ask(url: str, text: str) -> dict:
    body = {"model": "TandemLLM", "max_tokens": 512, "temperature": 0, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    t = d.get("timings") or {}
    return {"text": d["choices"][0]["message"].get("content"), "tok_s": t.get("predicted_per_second"),
            "per_block": t.get("tokens_per_block"), "n": t.get("predicted_n"), "prompt": t.get("prompt_n")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        runs = [json.load(open(p)) for p in a.compare]
        ok = True
        for c in runs[0]["answers"]:
            same = len({r["answers"][c]["text"] for r in runs}) == 1
            ok = ok and same
            print(f"{c:6} same={same} " + "  ".join(
                f"{r['label']}: {r['answers'][c]['tok_s']} tok/s, {r['answers'][c]['per_block']} a block"
                for r in runs))
        print("all same:", ok)
        sys.exit(0 if ok else 1)
    health = json.loads(urllib.request.urlopen(a.url + "/health", timeout=10).read())
    out = {"label": a.label, "version": health.get("version"), "answers": {}}
    for name, text in cases().items():
        out["answers"][name] = r = ask(a.url, text)
        print(f"[ab] {a.label} {name}: prompt {r['prompt']}, {r['n']} tokens, {r['tok_s']} tok/s, "
              f"{r['per_block']} a block", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
