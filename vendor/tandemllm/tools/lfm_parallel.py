"""SPD-107: parallel requests on a running server, with the proof that they decoded together (stdlib only).

    python tools/lfm_parallel.py --url http://127.0.0.1:8017 --levels 1,2,4 --tokens 256 --json out.json

For each level c: c greedy streamed requests sent at once (thinking off, `max_tokens` fixed, distinct
prompts), each request's decode tok/s ((completion tokens - 1) / (last token - first token)) and the
group's total (all completion tokens over the span from the first first token to the last end), and
each request's text against its own run at c = 1. Meanwhile /health is read every 0.2 s: the most slots
the barrier counted as decoding at once, and the joint rounds by size that the level added. A level
whose samples never show c slots decoding and no joint round of c blocks is marked INVALID (its
requests queued). The server's slot count and the parallel knobs it reports go into the output.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Describe a walk through Kyoto in autumn.",
    "Write a letter to a friend about learning to cook.",
    "Explain why the moon has phases.",
    "Describe how bread is made, from the field to the table.",
    "Write a dialogue between a cat and a dog about the weather.",
    "Explain the rules of chess to a beginner.",
]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _stream(url: str, model: str, prompt: str, tokens: int, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    parts, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    parts.append(piece)
    n = (usage or {}).get("completion_tokens") or len(parts)
    out.update(text="".join(parts), tokens=n, ttft_s=round((first or t0) - t0, 3), first=first, last=last,
               tok_s=round((n - 1) / (last - first), 2) if first and last and last > first else None)


def _poll(url: str, stop: threading.Event, seen: dict) -> None:
    while not stop.is_set():
        try:
            p = _get(url + "/health").get("parallel") or {}
            d = p.get("decoding")
            n = len(d) if isinstance(d, list) else int(d or 0)
            seen["max_decoding"] = max(seen.get("max_decoding", 0), n)
            seen["last"] = p
        except Exception:                                   # noqa: BLE001
            pass
        stop.wait(0.2)


def _joint(p: dict) -> dict:
    m = (p or {}).get("meeting") or p or {}
    return {str(k): int(v) for k, v in (m.get("joint_hist") or {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args()
    health = _get(a.url + "/health")
    model = (_get(a.url + "/v1/models").get("data") or [{}])[0].get("id")
    res = {"url": a.url, "model": model, "tokens": a.tokens, "server_parallel": health.get("parallel"),
           "levels": {}}
    alone: dict = {}
    for c in [int(x) for x in a.levels.split(",")]:
        seen: dict = {}
        before = _joint((_get(a.url + "/health").get("parallel") or {}))
        stop = threading.Event()
        th = threading.Thread(target=_poll, args=(a.url, stop, seen), daemon=True)
        th.start()
        outs = [dict() for _ in range(c)]
        ts = [threading.Thread(target=_stream, args=(a.url, model, PROMPTS[i], a.tokens, outs[i])) for i in range(c)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        stop.set()
        th.join()
        after = _joint((_get(a.url + "/health").get("parallel") or {}))
        added = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
        firsts = [o["first"] for o in outs if o.get("first")]
        lasts = [o["last"] for o in outs if o.get("last")]
        total = sum(o["tokens"] for o in outs) / (max(lasts) - min(firsts)) if firsts and lasts else None
        for i, o in enumerate(outs):
            if c == 1:
                alone[i] = o["text"]
        same = [outs[i]["text"] == alone.get(i) for i in range(c) if i in alone]
        valid = c == 1 or seen.get("max_decoding", 0) >= c or added.get(str(c), 0) > 0
        lv = {"per_request_tok_s": [o["tok_s"] for o in outs], "ttft_s": [o["ttft_s"] for o in outs],
              "total_tok_s": round(total, 2) if total else None, "max_decoding_seen": seen.get("max_decoding", 0),
              "joint_rounds_added": added, "texts_equal_alone": same, "valid": valid}
        res["levels"][c] = lv
        print(json.dumps({"c": c, **lv}), flush=True)
        if c == 1:
            # the other prompts alone too, for the agreement of the larger levels
            for i in range(1, max(int(x) for x in a.levels.split(","))):
                o: dict = {}
                _stream(a.url, model, PROMPTS[i], a.tokens, o)
                alone[i] = o["text"]
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
