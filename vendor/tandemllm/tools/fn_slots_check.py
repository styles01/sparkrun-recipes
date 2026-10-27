"""N requests at once on a server with side slots against each one alone (ENG-241 step 2).

    python tools/fn_slots_check.py --url http://127.0.0.1:8011 --pack ~/flashnext-packs/3.05 \
        --tokens 1000,3000,6000 --new 256 --json out.json

Each prompt (held-out text of that many tokens, X-Tandem: raw, greedy) is sent alone first -- its
answer is the single-stream truth -- and then all of them at once, one thread each (`--mode` runs
one of the two; a like-for-like comparison runs each mode on a fresh server, the same requests, and
`--compare` checks the answers against the other run). Each mode reports its wall time from the first
send to the last answer, the total tokens a second over it, and each request's time to first token. While they run,
the server's own report (`/health` `parallel`: slots busy, slots decoding) is read every 50 ms: the
largest number of slots decoding at the same moment is the concurrency the server really ran, not
what it was configured for. Recorded: each answer against its single-stream one, each request's
decode rate alone and together, the slots the server reports and its joint verify rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fn_fit_run import prompt  # noqa: E402


def ask(url: str, text: str, new: int) -> dict:
    body = {"model": "TandemLLM", "max_tokens": new, "temperature": 0, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Tandem": "raw"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    t = r.get("timings") or {}
    return {"answer": r["choices"][0]["message"].get("content"), "wall_s": round(time.time() - t0, 2),
            "ttft_ms": t.get("ttft_ms"), "tok_s": t.get("predicted_per_second"), "n": t.get("predicted_n"),
            "draft_n": t.get("draft_n"), "t_send": t0, "t_end": time.time()}


def health(url: str) -> dict:
    try:
        return json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
    except Exception:                                              # noqa: BLE001
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="1000,3000,6000")
    ap.add_argument("--kinds", default="", help="prose,code,chat: one held-out request of each kind, all "
                    "--ctx tokens (tools/fn_generate.py), in place of --tokens: prompts that share nothing")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--mode", default="both", choices=("both", "alone", "together"),
                    help="alone: one after the other; together: all at once; both: alone, then together")
    ap.add_argument("--compare", help="another run's JSON whose answers each answer must equal")
    ap.add_argument("--json")
    a = ap.parse_args()
    pack = os.path.expanduser(a.pack)
    if a.kinds:
        from transformers import AutoTokenizer
        from tools.fn_generate import class_text
        tok = AutoTokenizer.from_pretrained(pack)
        names = a.kinds.split(",")
        texts = [class_text(tok, k, a.ctx) for k in names]
    else:
        names = a.tokens.split(",")
        texts = [prompt(pack, int(n))[0] for n in names]
    par = health(a.url).get("parallel") or {}
    print(f"[slots] server reports {par.get('n') or 1} slots: "
          f"{[(s.get('index'), s.get('window')) for s in par.get('slots') or []]}", flush=True)
    out = {"slots": par.get("n") or 1}
    if a.mode in ("both", "alone"):
        alone = [ask(a.url, t, a.new) for t in texts]
        out["alone"] = summary(names, alone)
    if a.mode in ("both", "together"):
        together: list = [None] * len(texts)

        def one(i):
            together[i] = ask(a.url, texts[i], a.new)
        seen = {"decoding": 0, "running": 0}
        done = threading.Event()

        def watch():
            while not done.is_set():
                p = health(a.url).get("parallel") or {}
                dec = p.get("decoding") or []
                seen["decoding"] = max(seen["decoding"], len(dec) if isinstance(dec, list) else int(dec))
                seen["running"] = max(seen["running"], int(p.get("running") or 0))
                time.sleep(0.05)
        w = threading.Thread(target=watch)
        w.start()
        th = [threading.Thread(target=one, args=(i,)) for i in range(len(texts))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        done.set()
        w.join()
        meet = (health(a.url).get("parallel") or {}).get("meeting") or {}
        out["together"] = {**summary(names, together), "max_decoding_at_once": seen["decoding"],
                           "max_running_at_once": seen["running"], "joint_rounds": meet.get("joint_rounds"),
                           "rounds": meet.get("rounds"), "joint_hist": meet.get("joint_hist")}
    ref = json.load(open(a.compare)) if a.compare else None
    for mode in ("alone", "together"):
        if mode in out and ref is not None:
            truth = {r["prompt"]: r["answer"] for r in (ref.get("alone") or ref.get("together"))["requests"]}
            for r in out[mode]["requests"]:
                r["equal_to_ref"] = r["answer"] == truth.get(r["prompt"])
    if "alone" in out and "together" in out:
        for x, y in zip(out["alone"]["requests"], out["together"]["requests"]):
            y["equal_to_alone"] = x["answer"] == y["answer"]
    brief = {m: {k: v for k, v in out[m].items() if k != "requests"} | {"requests": [
        {k: v for k, v in r.items() if k not in ("answer", "t_send", "t_end")} for r in out[m]["requests"]]}
        for m in ("alone", "together") if m in out}
    print(f"[slots] {json.dumps(brief)}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def summary(names: list, rs: list) -> dict:
    """One mode's run: the wall time from the first send to the last answer, the tokens a second over
    it, and each request as it went."""
    wall = max(r["t_end"] for r in rs) - min(r["t_send"] for r in rs)
    return {"wall_s": round(wall, 2), "total_tok_s": round(sum(r["n"] or 0 for r in rs) / wall, 1),
            "requests": [{"prompt": n, **r} for n, r in zip(names, rs)]}


if __name__ == "__main__":
    main()
