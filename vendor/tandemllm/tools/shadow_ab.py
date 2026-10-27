"""The shadow reader's cost to Qwen, on a candidate server (SR1, 2026-10-04): decode rate and TTFT.

Runs inside a hold (row3's `start_server` refuses while the service answers): a candidate server
on :8011 with the conductor, the served engine flags from the environment, and the shadow reader
(`QWEN38_SHADOW_MODEL`, plus `QWEN38_NARRATOR_MODEL` of the same name so the narrator arm shares
the slot). Three header-only profiles copied from `coding`, state lines on:

  sr-off    no shadow block (the profile as shipped)
  sr-on     shadow on, the router's question in shadow mode from 0 tokens (every turn asks)
  sr-narr   sr-on plus `ui.narrator: thoughts` read from the shadow (`shadow.narrator: on`)

One fixed prompt of about `--prompt-tokens` tokens of the repository's own docs (no personal
text) and a question, greedy, `--max-tokens`, a new session each run (so the shadow has the whole
prompt to read each time), `--pause-s` between two requests (an idle gap in which the shadow loop
reads; the next request may arrive in the middle of a burst, which shows in its TTFT). The arms
alternate (off, on, narr / narr, on, off ...) after a warm-up each.

Per request: the server's decode rate (`timings.predicted_per_second`), the client's (completion
tokens after the first over the first to the last token that is not a state line), TTFT on the
client, the end delay. The gate (Qwen's decode): sr-on's median server rate within 3 % of
sr-off's, with the off runs' own spread beside it. The shadow's own counts (rows read, questions,
agreement) from /v1/conductor/shadow/status after the runs.

    python tools/shadow_ab.py --runs 6 --out ~/sr1/ab.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

GATE = 0.03
ARMS = ("sr-off", "sr-on", "sr-narr")
FRAME = re.compile(r"\n› ([^\n]*)\n\n")
QUESTION = ("\n\nThat is the documentation of an inference engine. Think about which three of its "
            "design decisions matter most for speed and why, then answer in a few short "
            "paragraphs.")


def prompt_text(tokens: int) -> str:
    docs = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
    text = ""
    for p in docs:
        with open(p) as f:
            text += f.read() + "\n\n"
        if len(text) > tokens * 4:
            break
    return text[:tokens * 4] + QUESTION


def profiles(root: str, min_tokens: int) -> None:
    from server.conductor import schema
    from server.conductor.store import Store
    s = Store(root)
    base = schema.seed("profiles", "coding")
    for name in ARMS:
        doc = json.loads(json.dumps(base))
        doc.update(description=f"shadow_ab: coding, {name}; reachable by header only", clients=[])
        doc["ui"] = dict(doc.get("ui") or {}, states="on",
                         narrator="thoughts" if name == "sr-narr" else "off")
        if name != "sr-off":
            doc["shadow"] = {"mode": "on", "router": "shadow", "router_min_tokens": min_tokens,
                             "narrator": "on" if name == "sr-narr" else "off"}
        s.put("profiles", name, doc, note=f"SR1 measure: {name}", author="sr1-measure")


def one(port: int, arm: str, run: int, text: str, max_tokens: int) -> dict:
    body = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": text}], "temperature": 0, "seed": 7,
            "max_tokens": max_tokens, "reasoning_format": "reasoning_content"}
    hdr = {"Content-Type": "application/json", "X-Tandem-Profile": arm,
           "User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5",
           "x-session-id": f"sr1-{arm}-{run}-{int(time.time() * 1000)}"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr)
    t0 = time.monotonic()
    first = last = None
    pieces, usage, timings = [], None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: {"):
                continue
            ev = json.loads(ln[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if not piece:
                    continue
                pieces.append(piece)
                if FRAME.fullmatch(piece):
                    continue
                now = time.monotonic()
                first = first if first is not None else now
                last = now
    done = time.monotonic()
    n = int((usage or {}).get("completion_tokens") or 0)
    span = (last - first) if first is not None and last is not None else 0.0
    lines = FRAME.findall("".join(pieces))
    return {"arm": arm, "run": run, "completion_tokens": n,
            "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "ttft_ms": round((first - t0) * 1e3, 1) if first is not None else None,
            "client_tps": round((n - 1) / span, 3) if span > 0 and n > 1 else None,
            "server_tps": (timings or {}).get("predicted_per_second"),
            "end_ms": round((done - last) * 1e3, 1) if last is not None else None,
            "thought_lines": sum(1 for x in lines if x.startswith("Thought"))}


def get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    import row3
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--prompt-tokens", type=int, default=16000)
    ap.add_argument("--pause-s", type=float, default=1.5)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pythonpath", default=ROOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    root = tempfile.mkdtemp(prefix="sr1-cond-")
    profiles(root, 0)
    ns = argparse.Namespace(
        python=Path(a.python), port=a.port, max_len=65536, len_fixed=0, budget=16,
        nvfp4=os.environ.get("NV") or row3.DEFAULT_NV, head=os.environ.get("HEAD") or row3.DEFAULT_HEAD,
        repo=Path(ROOT), pythonpath=Path(a.pythonpath), start_timeout=900, len_latch=True,
        store="off", server_arg=["--conductor=on", f"--conductor-config={root}",
                                 "--reasoning-format=reasoning_content", "--drop-idle",
                                 "--narrow-arm=on", "--no-learn-cost"],
        mem_floor_gb=10.0, tree=True, with_suffix_store=False)
    mdir = os.path.expanduser(f"~/.qwen38-spark-engine/models/{a.model}")
    env = {"QWEN38_SHADOW_MODEL": a.model, "QWEN38_SHADOW_DIR": mdir,
           "QWEN38_NARRATOR_MODEL": a.model, "QWEN38_NARRATOR_DIR": mdir,
           "QWEN38_SHADOW_DISK": "", "BOX_LOCK": os.path.join(root, "no-box-lock")}
    log = Path(tempfile.mkdtemp(prefix="sr1-server-")) / "server.log"
    text = prompt_text(a.prompt_tokens)
    proc = row3.start_server(ns, env, log)
    rows, out = [], {"server_log": str(log), "args": vars(a)}
    try:
        # the model loads with the first turn of a shadow profile; a warm-up of each arm
        for arm in arms:
            w = one(a.port, arm, -1, text, a.max_tokens)
            print(f"[sr1] warm-up {arm}: {w}", flush=True)
            time.sleep(a.pause_s)
        for _ in range(120):
            st = get(a.port, "/v1/conductor/shadow/status")
            if st.get("slot", {}).get("state") == "loaded":
                break
            time.sleep(2)
        out["slot"] = {k: st.get("slot", {}).get(k) for k in ("state", "precision", "error",
                                                              "refusal", "load_s", "took_gib")}
        print(f"[sr1] slot: {out['slot']}", flush=True)
        if out["slot"]["state"] != "loaded":
            print("[sr1] NOT LOADED: no A/B", flush=True)
            out["loaded"] = False
        else:
            out["loaded"] = True
            # one more warm-up of the shadow arms with the model loaded
            for arm in arms:
                if arm != "sr-off":
                    one(a.port, arm, -2, text, a.max_tokens)
                    time.sleep(a.pause_s)
            for i in range(a.runs):
                order = arms if i % 2 == 0 else list(reversed(arms))
                for arm in order:
                    r = one(a.port, arm, i, text, a.max_tokens)
                    rows.append(r)
                    print(f"[sr1] {arm:8s} run {i}: {r}", flush=True)
                    time.sleep(a.pause_s)
            time.sleep(5)
            out["shadow"] = get(a.port, "/v1/conductor/shadow/status")
            out["shadow"].pop("recent", None)
    finally:
        row3.stop_server(proc)
    out["rows"] = rows

    def med(arm, k):
        xs = [r[k] for r in rows if r["arm"] == arm and r[k] is not None]
        return round(statistics.median(xs), 3) if xs else None

    summ = {arm: {k: med(arm, k) for k in ("server_tps", "client_tps", "ttft_ms", "end_ms")}
            for arm in arms}
    offs = [r["server_tps"] for r in rows if r["arm"] == "sr-off" and r["server_tps"]]
    out["summary"] = summ
    out["off_spread"] = round(max(offs) / min(offs) - 1.0, 4) if len(offs) > 1 else None
    off, on = summ.get("sr-off", {}).get("server_tps"), summ.get("sr-on", {}).get("server_tps")
    out["delta_on"] = round(on / off - 1.0, 4) if off and on else None
    nr = summ.get("sr-narr", {}).get("client_tps")
    offc = summ.get("sr-off", {}).get("client_tps")
    out["delta_narr_client"] = round(nr / offc - 1.0, 4) if nr and offc else None
    out["gate"] = out["delta_on"] is not None and out["delta_on"] >= -GATE
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("summary", "off_spread", "delta_on",
                                          "delta_narr_client", "gate")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
