"""Decode speed at long context: one request per length through the served stack. A probe, not a row.

The row is 256 tokens in and 256 out, and it says nothing about what the engine does once a prompt
is an agent's 30k-token conversation, where every verify also reads the whole KV of sixteen
attention layers. This starts a test server in the served configuration (caches off, as the row
runs it), sends ONE request per length -- a real document from `tools/longprompts.py`'s set,
thinking off, greedy, 256 tokens out, "continue the text" -- and reports time to first token and the
decode rate by the row's own formula, (completion - 1) / (e2e - ttft).

    # inside ops/hold.sh: it starts an engine
    python tools/longctx_probe.py --lens 8192,32768 --label before
    python tools/longctx_probe.py --lens 8192,32768 --label kvfp8 --env QWEN38_KV_FP8=1

One request a length decides nothing by itself; before and after on the same prompt, in the same
hold, is the comparison, and a difference inside a few per cent is not one.

The memory curve. The 131k probe of 2026-09-23 wedged the board, so every
request is also a memory measurement: MemAvailable and MemFree polled every 0.1 s while it runs
(their minimum is the request's peak on this unified pool), the server's allocator numbers after it
(`/health` memory), and the kernel log's NVRM lines since the probe started. A length whose minimum
falls below `--stop-below-gb`, or any NVRM out-of-memory line, ends the probe before the next,
longer length; row3's MemGuard still kills the server at `--mem-floor-gb`. `--served-caches` runs
the prefix and session caches as :8000 does (1,024-row prefill chunks) instead of the row's caches
off (chunks of `--max-prefill-rows`, 8,192). A length the prompt set does not have is the first
N tokens of the same domain's next longer prompt, and the report says so.

    python tools/longctx_probe.py --lens 8192,16384,32768,65536 --label curve-off
    python tools/longctx_probe.py --lens 8192,16384,32768,65536 --label curve-served --served-caches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import row3  # noqa: E402


def meminfo_gb(text: str | None = None) -> dict:
    """MemAvailable and MemFree from /proc/meminfo, in GB (the GPU allocates from this pool)."""
    text = text if text is not None else open("/proc/meminfo").read()
    out = {}
    for line in text.splitlines():
        key = line.split(":", 1)[0]
        if key in ("MemAvailable", "MemFree"):
            out[key] = int(line.split()[1]) * 1024 / 1e9
    return out


class MemSampler:
    """The minimum MemAvailable and MemFree over a window, polled every `period` seconds."""

    def __init__(self, period: float = 0.1, read=meminfo_gb):
        self.period, self.read = period, read
        self.min_avail = self.min_free = float("inf")
        self.samples = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def poll(self) -> None:
        m = self.read()
        self.min_avail = min(self.min_avail, m.get("MemAvailable", float("inf")))
        self.min_free = min(self.min_free, m.get("MemFree", float("inf")))
        self.samples += 1

    def _run(self) -> None:
        self.poll()
        while not self._stop.wait(self.period):
            self.poll()

    def __enter__(self) -> "MemSampler":
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self.poll()

    def report(self) -> dict:
        return {"min_available_gb": round(self.min_avail, 2), "min_free_gb": round(self.min_free, 2),
                "samples": self.samples}


def nvrm_lines(since_epoch: float, run=subprocess.run) -> list[str] | None:
    """The kernel log's NVRM lines since `since_epoch` (None when the log cannot be read). The
    wedge of 2026-09-23 announced itself there as `NVRM: ... Out of memory [NV_ERR_NO_MEMORY]`."""
    try:
        r = run(["journalctl", "-k", "--no-pager", "-q", "--since", f"@{int(since_epoch)}"],
                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if "NVRM" in ln]


def stop_reason(mem: dict, nvrm: list[str] | None, stop_below_gb: float) -> str | None:
    """Why the probe must not go on to a longer length, or None."""
    if nvrm and any("out of memory" in ln.lower() or "NO_MEMORY" in ln for ln in nvrm):
        return f"NVRM out-of-memory in the kernel log ({len(nvrm)} NVRM lines)"
    if mem["min_available_gb"] < stop_below_gb:
        return f"MemAvailable fell to {mem['min_available_gb']:.1f} GB (< {stop_below_gb:.0f})"
    return None


def projected_stop(mem: dict, peak_gb: float, loaded_gb: float, length: int, next_length: int,
                   stop_below_gb: float) -> str | None:
    """Refuse a longer length whose PROJECTED minimum MemFree falls below the floor.

    Hold 2 (2026-09-25) found a prefill's transient growing with the context (~1 GB per 1k
    tokens at an 8,192-row chunk), so a length that passed says little about twice that length.
    The projection scales this request's transient (allocator peak over the loaded engine) by the
    ratio of the lengths and takes it from this request's minimum MemFree."""
    growth = max(0.0, peak_gb - loaded_gb)
    extra = growth * (next_length / length - 1.0)
    projected = mem["min_free_gb"] - extra
    if projected < stop_below_gb:
        return (f"the next length ({next_length}) projects MemFree to {projected:.1f} GB "
                f"(transient {growth:.1f} GB at {length}, < {stop_below_gb:.0f})")
    return None


def load_prompt(data: str, length: int, domain: str, man: dict) -> tuple[list[int], str]:
    """The domain's prompt of exactly `length` tokens, or the first `length` tokens of its next
    longer prompt when the set has no such length (and where it came from)."""
    have = sorted(int(k) for k in man["prompts"])
    for L in [length] + [x for x in have if x > length]:
        if str(L) not in man["prompts"]:
            continue
        i = next((j for j, m in enumerate(man["prompts"][str(L)]) if m["domain"] == domain), None)
        if i is None:
            continue
        ids = np.load(os.path.join(data, f"ids-{L}.npy"))[i].tolist()
        return ids[:length], (f"ids-{L}.npy[{i}]" if L == length
                              else f"ids-{L}.npy[{i}][:{length}]")
    raise SystemExit(f"[longctx] no {domain} prompt of {length} tokens or longer in {data}")


def health_memory(port: int) -> dict | None:
    """The server's allocator numbers (`/health` answers trusted-local callers in full)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as fh:
            return json.loads(fh.read()).get("memory")
    except Exception:
        return None


def stream(port: int, text: str, max_tokens: int) -> dict:
    body = {"model": "x", "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage, pieces = None, {}, []
    with urllib.request.urlopen(req, timeout=3600) as fh:
        for raw in fh:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    pieces.append(piece)
    e2e = time.perf_counter() - t0
    n = int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
            "ttft_s": ttft, "e2e_s": e2e,
            "tok_s": (n - 1) / (e2e - ttft) if ttft is not None and n > 1 and e2e > ttft else 0.0,
            "head": "".join(pieces)[:120], "text": "".join(pieces),
            "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest()}


def repeat_report(runs: dict) -> dict:
    """Each repeat (`8192-r2`, ...) against the first request of its length: the same text or not,
    and -- where both texts were kept (`--save-text`) -- the first character where they part. The
    served path's repeat of an 8k prompt once wrote a different text from the cold one (phase4,
    hold 1 and 3: b332b4fa vs 1a6844e6) and only the hashes were kept."""
    out = {}
    for key, r in runs.items():
        base = key.split("-r")[0]
        if key == base or base not in runs or not r.get("text_sha256"):
            continue
        a, b = runs[base].get("text"), r.get("text")
        rep = {"same": r["text_sha256"] == runs[base].get("text_sha256"), "first_diff_char": None}
        if not rep["same"] and a is not None and b is not None:
            rep["first_diff_char"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                          min(len(a), len(b)))
        out[key] = rep
    return out


def last_request(log_text: str) -> dict | None:
    """The server's own record of the last request: its `[req]` line's block count, committed
    tokens and decode time (`tools/rowlog.py`), so a length's rate splits into tokens a block and
    milliseconds a block, and the `[drafter]` line printed after it."""
    from tools.rowlog import parse_requests
    reqs = parse_requests(log_text)
    if not reqs:
        return None
    r = dict(reqs[-1])
    r.pop("accept", None)
    if r.get("blocks"):
        r["tok_blk"] = r["committed"] / r["blocks"]
        r["ms_blk"] = r["decode_ms"] / r["blocks"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lens", default="8192,32768")
    ap.add_argument("--allow-long", action="store_true",
                    help="permit lengths above 65,536. The 131k probe of 2026-09-23 wedged the "
                         "board; the server now chunks the prefill and row3's MemGuard "
                         "kills a server below 10 GB available, and neither has been proven at 131k")
    ap.add_argument("--served-caches", action="store_true",
                    help="the prefix and session caches on, as :8000 runs them (1,024-row prefill "
                         "chunks), instead of the row's caches off")
    ap.add_argument("--cache-gb", type=float, default=8.0,
                    help="the state cache's budget with --served-caches (serve.env CACHE_GB)")
    ap.add_argument("--mem-floor-gb", type=float, default=10.0,
                    help="row3's MemGuard kills the server below this much MemAvailable")
    ap.add_argument("--stop-below-gb", type=float, default=20.0,
                    help="no longer length after one whose minimum MemAvailable fell below this")
    ap.add_argument("--repeat", type=int, default=1,
                    help="send each length this many times; the later ones (`<len>-r2`, ...) decode "
                         "with every graph of the length's context class already captured (a first "
                         "request at a new class pays the captures inside its decode), and with "
                         "--served-caches they resume the prompt from the prefix cache")
    ap.add_argument("--drop-page-cache", default="no", choices=("no", "start", "after-first"),
                    help="fadvise(DONTNEED) the weight files' page cache (tools/drop_page_cache.py) "
                         "once the server is loaded ('start') or after the first measured request "
                         "('after-first': the same server and prompt before and after, as an A/B)")
    ap.add_argument("--data", default="bench/longprompts")
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--save-text", action="store_true",
                    help="keep each answer's whole text in the report (a continuation of a bench "
                         "document), so a repeat that differs says where; off keeps the hash and "
                         "the first 120 characters")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default="results/longctx")
    a = ap.parse_args()

    lens = [int(x) for x in a.lens.split(",")]
    if max(lens) > 65536 and not a.allow_long:
        raise SystemExit(f"[longctx] REFUSING {max(lens)} tokens without --allow-long")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    prompts, sources = {}, {}
    for length in lens:
        ids, sources[length] = load_prompt(a.data, length, a.domain, man)
        prompts[length] = (tok.decode(ids) + "\n\nContinue the text above from where "
                           "it stops, in the same style.")

    ns = argparse.Namespace(port=a.port, python=row3.DEFAULT_PY, pythonpath=row3.HOME / "pylibs",
                            repo=row3.REPO, max_len=a.max_len, len_fixed=0, budget=16,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD,
                            server_arg=["--drop-idle"] + a.server_arg, len_latch=True,
                            start_timeout=600, mem_floor_gb=a.mem_floor_gb,
                            served_caches=a.served_caches, cache_gb=a.cache_gb)
    env = dict(kv.split("=", 1) for kv in a.env)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    mem_before_load = meminfo_gb()
    proc = row3.start_server(ns, env, out_dir / f"{a.label}-server.log")
    res = {"label": a.label, "env": env, "server_arg": a.server_arg, "max_len": a.max_len,
           "domain": a.domain, "args": {k: v for k, v in sorted(vars(a).items())},
           "server_cmd": row3.server_cmd(ns), "mem_before_load_gb": mem_before_load,
           "runs": {}}
    try:
        stream(a.port, "Say hello.", 16)                  # the autotuning, outside the numbers
        res["mem_loaded_gb"] = meminfo_gb()
        res["health_memory_loaded"] = health_memory(a.port)
        print(f"[longctx] {a.label} loaded: MemAvailable {res['mem_loaded_gb']['MemAvailable']:.1f} "
              f"GB, allocator {res['health_memory_loaded']}", flush=True)
        plan = [(length, text, f"{length}" + (f"-r{i + 1}" if i else ""))
                for length, text in prompts.items() for i in range(max(1, a.repeat))]
        from tools.drop_page_cache import cached_gb, drop

        def drop_cache(when: str) -> None:
            before = cached_gb()
            r = drop()
            res.setdefault("page_cache_drops", []).append(
                {"when": when, **r, "cached_before_gb": before, "cached_after_gb": cached_gb(),
                 "mem_after_gb": meminfo_gb()})
            print(f"[longctx] {a.label} page cache dropped ({when}): {r['gb']:.1f} GB of files, "
                  f"Cached {before:.1f} -> {cached_gb():.1f} GB, MemFree "
                  f"{meminfo_gb()['MemFree']:.1f} GB", flush=True)
        if a.drop_page_cache == "start":
            drop_cache("start")
        for n_done, (length, text, key) in enumerate(plan):
            if a.drop_page_cache == "after-first" and n_done == 1:
                drop_cache("after the first request")
            t_req = time.time()
            with MemSampler() as ms:
                try:
                    r = stream(a.port, text, a.max_tokens)
                except Exception as e:                # the server died under it (MemGuard, OOM)
                    r = {"error": repr(e)[:300], "ttft_s": None, "tok_s": 0.0,
                         "prompt_tokens": None, "completion_tokens": 0, "head": ""}
            if not a.save_text:
                r.pop("text", None)
            r["source"] = sources[length]
            r["server"] = last_request((out_dir / f"{a.label}-server.log").read_text(errors="replace"))
            r["mem"] = ms.report()
            r["health_memory"] = health_memory(a.port)
            r["nvrm"] = nvrm_lines(t_req)                 # this request's own, not the load's
            res["runs"][key] = r
            print(f"[longctx] {a.label} {key:>10}  prompt {r['prompt_tokens']}  "
                  f"ttft {r['ttft_s'] or 0:.1f} s  decode {r['tok_s']:.2f} tok/s  "
                  f"({r['completion_tokens']} tok)  {r['head'][:60]!r}", flush=True)
            sv = r["server"] or {}
            if sv.get("blocks"):
                print(f"[longctx] {a.label} {key:>10}  blocks {sv['blocks']}  "
                      f"tok/blk {sv['tok_blk']:.2f}  ms/blk {sv['ms_blk']:.1f}", flush=True)
            print(f"[longctx] {a.label} {key:>10}  min MemAvailable {r['mem']['min_available_gb']:.1f} GB  "
                  f"min MemFree {r['mem']['min_free_gb']:.1f} GB  allocator {r['health_memory']}  "
                  f"NVRM lines {None if r['nvrm'] is None else len(r['nvrm'])}", flush=True)
            why = (f"the request failed: {r['error']}" if "error" in r
                   else stop_reason(r["mem"], r["nvrm"], a.stop_below_gb))
            nxt = next((L for L, _, _ in plan[n_done + 1:] if L > length), None)
            hm, hl = r.get("health_memory") or {}, res.get("health_memory_loaded") or {}
            if not why and nxt and hm.get("max_allocated_gb") and hl.get("allocated_gb"):
                why = projected_stop(r["mem"], hm["max_allocated_gb"], hl["allocated_gb"], length, nxt,
                                     a.stop_below_gb)
            if why and a.drop_page_cache == "after-first" and n_done == 0 and "NVRM" in why:
                # the A/B's control: the same request is sent again once the cache is dropped
                print(f"[longctx] {a.label} control request: {why}; dropping the cache and going on",
                      flush=True)
                why = None
            if why:
                res["stopped"] = f"after {key}: {why}"
                print(f"[longctx] STOP {res['stopped']}", flush=True)
                break
    finally:
        guard = getattr(proc, "memguard", None)
        if guard is not None and guard.tripped is not None:
            res["memguard_tripped_gb"] = guard.tripped
        row3.stop_server(proc)
    res["repeats"] = repeat_report(res["runs"])
    for key, rep in res["repeats"].items():
        where = ("" if rep["same"] or rep["first_diff_char"] is None
                 else f" from character {rep['first_diff_char']}")
        print(f"[longctx] {a.label} {key} against the first request: "
              f"{'the same text' if rep['same'] else 'a different text' + where}", flush=True)
    path = out_dir / f"{a.label}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"[longctx] {path}")


if __name__ == "__main__":
    main()
