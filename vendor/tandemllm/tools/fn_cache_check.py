"""ENG-157 on the box: Flash-Next's caches through the server, checked against cold answers.

    python tools/fn_cache_check.py --url http://127.0.0.1:8011 --phase warm --out warm.json
    python tools/fn_cache_check.py --url http://127.0.0.1:8011 --phase cold --out cold.json
    python tools/fn_cache_check.py --compare warm.json cold.json

`warm` runs a sequence on a server with its caches on: two conversations, A and B, of about 6,000
and 5,000 tokens (two sessions), each continued by a second turn, in the order A1, A2 (the resident
prefix in place), B1 (A is parked), A3 (A comes back from the park), B2 (B from the park in turn);
then a 10,000-token document twice (`D1`, `D2`), and with `--after-restart` only `D3` (the document
and another question, after a restart: the session spill's or the disk tier's restore); `--names`
sends only the named requests (the disk tier's sequence: D1, a hard stop, D1, a hard stop, D3). Each request records its
answer, its logprobs and the server's `last_prefill` (where the prefill started and how much it
reused). `cold` sends each final request alone to a server with its caches off. `--compare`: every
answer of `warm` equals the cold one token for token (greedy), the logprobs within bf16 noise, and
says which cache each one came from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

WORDS = ("river stone lantern harbor meadow copper violet signal orchard engine silver window "
         "thunder cedar quartz ember canyon falcon marble willow anchor beacon cobalt prairie").split()


def doc(seed: int, n_words: int) -> str:
    """A deterministic text of numbered sentences (no tokenizer needed: ~1.3 tokens a word)."""
    out, x = [], seed
    for i in range(n_words // 12):
        words = []
        for _ in range(12):
            x = (x * 1103515245 + 12345) % (1 << 31)
            words.append(WORDS[x % len(WORDS)])
        out.append(f"Line {i}: " + " ".join(words) + ".")
    return " ".join(out)


A_DOC, B_DOC, D_DOC = doc(1, 4400), doc(2, 3700), doc(3, 7400)
Q = {"A1": "Summarise the first ten lines of the text above in two sentences.",
     "A2": "Which words appear most often in lines 20 to 30? Answer briefly.",
     "A3": "Now list three words that never appear in line 5.",
     "B1": "What is the last line of the text above about? One sentence.",
     "B2": "Count the lines that mention copper. Answer with a number and one sentence.",
     "D": "Quote line 400 of the text above exactly.",
     "D3": "Quote line 401 of the text above exactly, then line 402."}


def call(url: str, body: dict, session: str | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    if session:
        h["x-session-id"] = session
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["_s"] = round(time.time() - t0, 2)
    with urllib.request.urlopen(url + "/health", timeout=60) as r:
        out["_prefill"] = (json.loads(r.read()).get("cache") or {}).get("last_prefill")
    return out


def body(messages: list) -> dict:
    return {"model": "TandemLLM", "messages": messages, "max_tokens": 48, "temperature": 0,
            "logprobs": True, "top_logprobs": 1, "chat_template_kwargs": {"enable_thinking": False}}


def text(r: dict) -> str:
    return r["choices"][0]["message"].get("content") or ""


def conversations(prev: dict) -> dict:
    """The final message list of every request, given the answers so far (`prev`, name -> text)."""
    a0 = [{"role": "user", "content": A_DOC + "\n\n" + Q["A1"]}]
    b0 = [{"role": "user", "content": B_DOC + "\n\n" + Q["B1"]}]
    out = {"A1": a0, "B1": b0}
    if "A1" in prev:
        out["A2"] = a0 + [{"role": "assistant", "content": prev["A1"]}, {"role": "user", "content": Q["A2"]}]
    if "A2" in prev:
        out["A3"] = out["A2"] + [{"role": "assistant", "content": prev["A2"]}, {"role": "user", "content": Q["A3"]}]
    if "B1" in prev:
        out["B2"] = b0 + [{"role": "assistant", "content": prev["B1"]}, {"role": "user", "content": Q["B2"]}]
    d = [{"role": "user", "content": D_DOC + "\n\n" + Q["D"]}]
    out["D1"] = out["D2"] = d
    out["D3"] = [{"role": "user", "content": D_DOC + "\n\n" + Q["D3"]}]
    return out


ORDER = (("A1", "A"), ("A2", "A"), ("B1", "B"), ("A3", "A"), ("B2", "B"), ("D1", "D"), ("D2", "D"))


def warm(url: str, after_restart: bool, names: str = "") -> dict:
    res, prev = {}, {}
    order = (("D3", "D"),) if after_restart else ORDER
    if names:
        order = tuple((n, n[0]) for n in names.split(","))
    for name, sess in order:
        msgs = conversations(prev)[name]
        r = call(url, body(msgs), session=sess)
        prev[name] = text(r)
        res[name] = {"messages": msgs, "answer": r, "prefill": r["_prefill"]}
        p = r["_prefill"] or {}
        print(f"[warm] {name}: {r['_s']} s, prompt {r['usage']['prompt_tokens']}, prefill {p.get('kind')} "
              f"reused {p.get('reused')} forwarded {p.get('forwarded')}; {prev[name][:60]!r}", flush=True)
    return res


def cold(url: str, warm_doc: dict) -> dict:
    res = {}
    for name, w in warm_doc.items():
        r = call(url, body(w["messages"]))
        res[name] = {"answer": r, "prefill": r["_prefill"]}
        print(f"[cold] {name}: {r['_s']} s; {text(r)[:60]!r}", flush=True)
    return res


def compare(w: dict, c: dict) -> dict:
    out, ok = {}, True
    for name, x in w.items():
        a, b = x["answer"], c[name]["answer"]
        la = [t["logprob"] for t in a["choices"][0]["logprobs"]["content"]]
        lb = [t["logprob"] for t in b["choices"][0]["logprobs"]["content"]]
        same = text(a) == text(b)
        p = x["prefill"] or {}
        out[name] = {"same": same, "via": p.get("kind"), "reused": p.get("reused"),
                     "logprob_max_diff": max((abs(i - j) for i, j in zip(la, lb)), default=0.0),
                     "tokens": [len(la), len(lb)]}
        ok = ok and same
    return {"all_same": ok, "requests": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--phase", choices=("warm", "cold"))
    ap.add_argument("--after-restart", action="store_true")
    ap.add_argument("--names", default="", help="only these requests, in this order (e.g. D1 or D3)")
    ap.add_argument("--warm", help="the warm JSON the cold phase repeats")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        r = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["all_same"] else 1)
    out = warm(a.url, a.after_restart, a.names) if a.phase == "warm" else cold(a.url, json.load(open(a.warm)))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
