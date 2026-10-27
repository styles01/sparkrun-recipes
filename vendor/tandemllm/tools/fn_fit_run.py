"""SPD-99 on the box: one confirming run of a pack at its largest window, through a running server.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_fit_run.py --url http://127.0.0.1:8011 \
        --pack ~/flashnext-packs/3.05 --tokens 261500 --json fit-3.05.json

One greedy request of `--tokens` prompt tokens (bench/ held-out text and calib.txt, repeated, counted with
the pack's tokenizer through its chat template) and 256 new tokens, thinking off. Meanwhile MemAvailable
is sampled at 10 Hz and the n-gram table's resident bytes (/health `cache.resident_files`) once a
second, so the counted MemAvailable (less the warm table, as every floor check reads it, SRV-335) is
known at each sample. Recorded: the lowest raw and counted MemAvailable and when, the server's memory
after the request (allocated, reserved, the allocator's peak, the split), the request's timings
(prefill and decode rates, tokens a block) and every limit knob the server reports (max_len, the memory
limits, the prefill chunk, the table policy). No sweep: one request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = float(1 << 30)


def get(url: str, path: str) -> dict:
    return json.loads(urllib.request.urlopen(url + path, timeout=30).read())


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return -1


def prompt(pack: str, n: int) -> tuple[str, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pack)
    names = ("heldout_prose.txt", "heldout_code.txt", "calib.txt")
    fill = "\n\n".join(open(os.path.join(ROOT, "bench", x)).read() for x in names)
    ask = "\n\nIn three sentences, what is the text above about?"
    ids = tok(fill, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids

    def count(m: int) -> int:
        chat = tok.apply_chat_template([{"role": "user", "content": tok.decode(ids[:m]) + ask}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return len(tok(chat, add_special_tokens=False)["input_ids"])

    lo, hi = 0, n
    while lo < hi:                                  # the longest fill that keeps the prompt <= n
        mid = (lo + hi + 1) // 2
        if count(mid) <= n:
            lo = mid
        else:
            hi = mid - 1
    return tok.decode(ids[:lo]) + ask, count(lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", type=int, default=261_500)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--prewarm-table", action="store_true",
                    help="read the pack's n-gram table through the page cache first: the warm policy's worst "
                         "case, the whole table resident, which every floor check counts as used")
    ap.add_argument("--again", action="store_true",
                    help="send the request a second time after the run: its prompt comes back from the caches, "
                         "so its decode rate is the window's own, without the first round's graph captures")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds between the run and --again (the server's idle window: its graph captures)")
    ap.add_argument("--raw", action="store_true", help="X-Tandem: raw (the conductor adds nothing, so the "
                    "second request's prompt is the first's)")
    ap.add_argument("--json")
    a = ap.parse_args()
    headers = {"Content-Type": "application/json", **({"X-Tandem": "raw"} if a.raw else {})}
    text, n = prompt(os.path.expanduser(a.pack), a.tokens)
    if a.prewarm_table:
        with open(os.path.join(os.path.expanduser(a.pack), "ngram_embedding.safetensors"), "rb") as f:
            while f.read(1 << 26):
                pass
    health0, sys0 = get(a.url, "/health"), get(a.url, "/v1/dashboard/system")
    res = {"prompt_tokens_template": n, "prewarm_table": a.prewarm_table, "health_before": {k: health0.get(k) for k in ("version", "max_len", "weights")},
           "mem_available_before_gib": round(mem_available() / GIB, 2)}
    samples, table = [], [0]
    done = threading.Event()

    def resident():
        while not done.is_set():
            try:
                h = get(a.url, "/health")
                table[0] = sum(int(f.get("resident_bytes") or 0) for f in (h.get("cache") or {}).get("resident_files") or []
                               if f.get("policy") == "warm")
            except Exception:                                      # noqa: BLE001
                pass
            done.wait(1.0)

    def sample():
        t0 = time.time()
        while not done.is_set():
            m = mem_available()
            samples.append((round(time.time() - t0, 2), m, m - table[0]))
            done.wait(0.1)

    threads = [threading.Thread(target=f, daemon=True) for f in (resident, sample)]
    for t in threads:
        t.start()
    body = {"model": "TandemLLM", "max_tokens": a.new, "temperature": 0, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        res["answer"] = r["choices"][0]["message"].get("content")
        res["timings"], res["usage"] = r.get("timings"), r.get("usage")
    except urllib.error.HTTPError as exc:
        res["error"] = {"status": exc.code, "body": exc.read().decode()[:2000]}
    res["wall_s"] = round(time.time() - t0, 1)
    if a.again and "error" not in res:
        time.sleep(a.gap)
        t1 = time.time()
        r2 = json.loads(urllib.request.urlopen(urllib.request.Request(
            a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers), timeout=3600).read())
        res["again"] = {"wall_s": round(time.time() - t1, 1), "timings": r2.get("timings"),
                        "answer": r2["choices"][0]["message"].get("content"),
                        "same_answer": r2["choices"][0]["message"].get("content") == res.get("answer")}
    done.set()
    for t in threads:
        t.join()
    low = min(samples, key=lambda s: s[1])
    lowc = min(samples, key=lambda s: s[2])
    res["mem_available_min_gib"] = {"raw": round(low[1] / GIB, 2), "at_s": low[0],
                                    "counted": round(lowc[2] / GIB, 2), "counted_at_s": lowc[0]}
    res["samples"] = len(samples)
    sys1 = get(a.url, "/v1/dashboard/system")
    mem = sys1.get("memory") or {}
    res["memory_after"] = {k: mem.get(k) for k in ("gpu_allocated_bytes", "gpu_reserved_bytes", "gpu_max_allocated_bytes",
                                                   "unified_available_bytes")}
    res["split_after"] = {k: (mem.get("split") or {}).get(k) for k in ("parts", "other", "reserve", "in_use", "pinned_cache",
                                                                       "max_len")}
    res["knobs"] = {"max_len": health0.get("max_len"), "flags": sys0.get("flags"),
                    "limits": (mem.get("limits") or {}).get("rows"),
                    "env": {k: os.environ.get(k) for k in ("QWEN38_FN_TABLE", "QWEN38_FN_TABLE_CACHE_GB",
                                                           "QWEN38_FN_PREFILL_CHUNK", "QWEN38_FN_LOOKUP")}}
    t = res.get("timings") or {}
    print(f"[fit] prompt {n} tokens ({(res.get('usage') or {}).get('prompt_tokens')} served), wall {res['wall_s']} s, "
          f"prefill {t.get('prompt_per_second')} tok/s, decode {t.get('predicted_per_second')} tok/s, "
          f"{t.get('tokens_per_block')} a block; MemAvailable low {res['mem_available_min_gib']}; "
          f"peak allocated {round((mem.get('gpu_max_allocated_bytes') or 0) / GIB, 2)} GiB, reserved "
          f"{round((mem.get('gpu_reserved_bytes') or 0) / GIB, 2)} GiB; error {res.get('error')}", flush=True)
    if res.get("again"):
        t2 = res["again"]["timings"] or {}
        print(f"[fit] again: cached {t2.get('cache_n')}, decode {t2.get('predicted_per_second')} tok/s, "
              f"{t2.get('tokens_per_block')} a block, same answer {res['again']['same_answer']}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
