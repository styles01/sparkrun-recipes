"""The proxy's background memory writer against a REAL backend (the open item of
0.3.35's release note): a vLLM or SGLang server on this host, or one cloud API.

    python tools/proxy_writer_live.py --kind vllm --url http://127.0.0.1:8101/v1 \
        --model Qwen/Qwen3.5-0.8B --out DIR
    python tools/proxy_writer_live.py --kind deepseek --model deepseek-flash \
        --key-file ~/.tandem-night/deepseek.key --price 0.30,0.006,1.2 --usd-cap 0.05 --out DIR

The standalone proxy runs in this process (a temp config, ledger, conductor and notes root), the
backend registered with its harness on and the coding profile's `memory` group on; the writer's
thread is off and the script calls its look (`tick()`) itself, so each step is one look.

A local backend (vllm, sglang), four steps:
    on        a session of two turns with a fact worth keeping, made idle; one look: the writer's
              request goes to the backend (the session's messages as sent, its answer, the
              writer's turn; tool_choice none; thinking off), the answer through the engine
              writer's parser; notes written; a ledger row of harness:writer
    shadow    the same in a second session with the settings' writer mode shadow: the request
              runs, nothing is stored, the would-be blocks in the user's shadow log
    busy      a client's long request in flight on the backend: the look starts nothing (`busy`)
    yield     the writer streams (a long answer asked for) and a client request arrives: the
              writer's stream is closed, the session stays due (`yielded`); when the writer is
              done before the client's request lands, the step says so (not a failure: the
              window is a race on a fast small model)
A cloud backend (deepseek, openai): the `on` step only, with the writer switched on, a price and a
daily cap; the exact cost of every request it made is read back from the ledger.

Each step prints one line and the report goes to DIR/report.json. Nothing of a note's text or
the key goes to the report beyond the titles of the notes the test session itself asked for.
Exit 0 when every step that must hold held.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from proxy_util import Proxy  # noqa: E402

from proxy import app as proxy_app  # noqa: E402

UA = {"User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3"}
SYSTEM = {"role": "system", "content": "You are a coding agent. Working directory: /tmp/tandem-writer-live"}
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a shell command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}},
                                               "required": ["command"]}}}]
FACT_1 = ("Remember this for the whole project: the test proxy always listens on port 18700, and "
          "the end-to-end suite must run with --workers=1 because two workers crash the Mac. "
          "Now list the files in the working directory.")
FACT_2 = "Good. Also: never run vite on the Mac; builds run on the Spark only. Thanks, that is all."


PREFILL = ""


def setup(px, mode: str, max_tokens: int):
    px.sync()
    cond = proxy_app.STATE["conductor"]
    v, prof = cond.store.get("profiles", "coding")
    cond.store.put("profiles", "coding", dict(prof, memory="on"), note="live", author="n1",
                   base=v)
    set_mode(cond, mode, max_tokens)
    px.svc.harness.writer.auto = False
    _, _, body = px.request("POST", "/v1/config/users", {"label": "live"})
    return cond, json.loads(body)["key"]


def set_mode(cond, mode: str, max_tokens: int) -> None:
    v, st = cond.store.get("settings")
    st.setdefault("memory", {})["notes"] = {"writer": {
        "mode": mode, "gate": False, "idle_s": 30.0, "due_idle_s": 0.0, "min_new_tokens": 1,
        "max_tokens": max_tokens, "after_compaction": "idle", "prefill": PREFILL}}
    cond.store.put("settings", None, st, note="live", author="n1", base=v)


def turn(px, hdr, model, history, text, max_tokens=200):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "tools": TOOLS,
            "messages": [SYSTEM, *history, {"role": "user", "content": text}]}
    st, _, out = px.post("/v1/chat/completions", body, headers=hdr, timeout=300)
    text_out = []
    for ev in out.split(b"\n\n"):
        if ev.startswith(b"data: {"):
            d = json.loads(ev[6:])
            for c in d.get("choices") or []:
                text_out.append((c.get("delta") or {}).get("content") or "")
    time.sleep(0.3)                     # the turn's bookkeeping runs after the answer
    return st, "".join(text_out)


def make_idle(px, key: str) -> None:
    s = px.svc.harness.writer.sessions.get(key)
    if s is not None:
        s.t = time.monotonic() - 1000.0


def session(px, hdr, model) -> tuple[int, str]:
    st, a1 = turn(px, hdr, model, [], FACT_1)
    hist = [{"role": "user", "content": FACT_1}, {"role": "assistant", "content": a1 or "ok"}]
    st2, _ = turn(px, hdr, model, hist, FACT_2)
    return max(st, st2), a1


def last_run(cond) -> dict:
    r = dict(cond.memory.writer.runs[0]) if cond.memory.writer.runs else {}
    return {k: r.get(k) for k in ("trigger", "mode", "how", "via", "backend", "model", "answer",
                                  "none", "cut", "ops", "tokens", "ms", "skipped", "yielded",
                                  "cost_micros", "shadow_log", "finish", "trimmed", "refused")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("vllm", "sglang", "generic", "deepseek",
                                                       "openai"))
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--price", default=None, help="input,cached,output USD per million tokens")
    ap.add_argument("--usd-cap", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=400, help="the writer's answer cap")
    ap.add_argument("--prefill", default="", help="the writer's answer start (settings prefill)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    global PREFILL
    PREFILL = a.prefill
    os.makedirs(a.out, exist_ok=True)
    cloud = a.kind in ("deepseek", "openai")
    bid = "b1"
    env = {"TANDEM_BACKENDS": bid, f"TANDEM_BACKEND_{bid.upper()}_KIND": a.kind,
           f"TANDEM_BACKEND_{bid.upper()}_HARNESS": "on"}
    if a.url:
        env[f"TANDEM_BACKEND_{bid.upper()}_URL"] = a.url
    if a.key_file:
        env[f"TANDEM_BACKEND_{bid.upper()}_KEY"] = open(os.path.expanduser(a.key_file)).read().strip()
    if cloud:
        env[f"TANDEM_BACKEND_{bid.upper()}_WRITER"] = "on"
        env[f"TANDEM_BACKEND_{bid.upper()}_WRITER_USD_DAY"] = str(a.usd_cap)
        env[f"TANDEM_BACKEND_{bid.upper()}_WRITER_RUNS_DAY"] = "1"
        if a.price:
            env["TANDEM_PRICE_1"] = f"{bid}/{a.model}={a.price}"
    report = {"kind": a.kind, "model": a.model, "url": a.url, "prefill": a.prefill, "steps": {},
              "at": time.time()}
    ok = True
    with Proxy(env=env, conductor=True) as px:
        cond, key = setup(px, "on", a.max_tokens)
        listed = [m["id"] for m in json.loads(px.get("/v1/models")[2])["data"]]
        model = a.model if a.model in listed else f"{bid}/{a.model}"
        hx = px.svc.harness
        base = dict(UA, Authorization=f"Bearer {key}")

        # ---- on
        hdr = dict(base, **{"x-session-id": "live-on"})
        t0 = time.time()
        st, first = session(px, hdr, model)
        make_idle(px, "px:c:live-on")
        got = hx.writer.tick()
        run = last_run(cond)
        notes = [{"title": n.title, "kind": n.kind, "scope": n.scope, "writer": n.writer}
                 for n in cond.notes.notes("live")]
        rows = [r for r in px.rows() if r["client_id"] == "harness:writer"]
        step = {"turns_status": st, "first_answer_chars": len(first), "tick": got, "run": run,
                "notes": notes, "writer_rows": [{k: r.get(k) for k in (
                    "status", "backend", "prompt_tokens", "completion_tokens", "cached_tokens",
                    "cost_micros", "decode_tps", "ttft_ms")} for r in rows],
                "seconds": round(time.time() - t0, 1)}
        hold = st == 200 and got == "wrote" and len(rows) == 1 and rows[0]["status"] == 200 \
            and run.get("via") == "proxy"
        step["pass"] = hold
        ok &= hold
        report["steps"]["on"] = step
        print(f"[writer-live] on: tick={got} answer={run.get('answer')} ops={len(run.get('ops') or [])} "
              f"notes={len(notes)} tokens={run.get('tokens')} ms={run.get('ms')} "
              f"rows={len(rows)} {'PASS' if hold else 'FAIL'}", flush=True)
        if cloud:
            all_rows = px.rows()
            report["cost"] = {
                "requests": len(all_rows),
                "cost_micros_total": sum(int(r.get("cost_micros") or 0) for r in all_rows),
                "by_client": {c: sum(int(r.get("cost_micros") or 0) for r in all_rows
                                     if r["client_id"] == c)
                              for c in sorted({r["client_id"] for r in all_rows})},
                "tokens": [{k: r.get(k) for k in ("client_id", "prompt_tokens", "cached_tokens",
                                                  "completion_tokens", "cost_micros")}
                           for r in all_rows]}
            print(f"[writer-live] cost: {report['cost']['cost_micros_total']} micro-USD over "
                  f"{len(all_rows)} requests", flush=True)
        else:
            # ---- shadow
            set_mode(cond, "shadow", a.max_tokens)
            hdr = dict(base, **{"x-session-id": "live-shadow"})
            n_before = len(cond.notes.notes("live"))
            st, _ = session(px, hdr, model)
            make_idle(px, "px:c:live-shadow")
            got = hx.writer.tick()
            run = last_run(cond)
            d = os.path.join(cond.notes.user_dir("live"), "shadow")
            shadow = []
            if os.path.isdir(d):
                for f in os.listdir(d):
                    p = os.path.join(d, f)
                    shadow.append({"file": f, "mode": oct(os.stat(p).st_mode & 0o777),
                                   "lines": sum(1 for _ in open(p))})
            stored = len(cond.notes.notes("live")) - n_before
            hold = got == "wrote" and run.get("mode") == "shadow" and stored == 0 and (
                run.get("none") or bool(shadow))
            ok &= hold
            report["steps"]["shadow"] = {"tick": got, "run": run, "stored": stored,
                                         "shadow_log": shadow, "pass": hold}
            print(f"[writer-live] shadow: tick={got} answer={run.get('answer')} stored={stored} "
                  f"shadow_log={shadow} {'PASS' if hold else 'FAIL'}", flush=True)

            # ---- busy: a client's request in flight, the look starts nothing
            set_mode(cond, "on", a.max_tokens)
            hdr = dict(base, **{"x-session-id": "live-busy"})
            session(px, hdr, model)
            make_idle(px, "px:c:live-busy")
            long_body = {"model": model, "stream": True, "max_tokens": 1500, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": "Write a long story about a "
                                       "lighthouse keeper, at least 1000 words."}]}
            th = threading.Thread(target=lambda: px.post("/v1/chat/completions", long_body,
                                                         headers=UA, timeout=300))
            th.start()
            t_end = time.time() + 30
            while px.svc.relay.inflight.count(bid) == 0 and time.time() < t_end:
                time.sleep(0.02)
            inflight = px.svc.relay.inflight.count(bid)
            got = hx.writer.tick()
            th.join(300)
            hold = inflight > 0 and got == "busy"
            ok &= hold
            report["steps"]["busy"] = {"inflight": inflight, "tick": got, "pass": hold}
            print(f"[writer-live] busy: inflight={inflight} tick={got} "
                  f"{'PASS' if hold else 'FAIL'}", flush=True)
            # the same session once the backend is idle again
            got2 = hx.writer.tick()
            report["steps"]["busy"]["after"] = got2
            print(f"[writer-live] busy, after: tick={got2}", flush=True)

            # ---- yield: a client request while the writer streams
            set_mode(cond, "on", 1500)
            hdr = dict(base, **{"x-session-id": "live-yield"})
            session(px, hdr, model)
            make_idle(px, "px:c:live-yield")
            res = {}
            wt = threading.Thread(target=lambda: res.update(r=hx.writer.tick()))
            wt.start()
            # the writer's own request is not a relayed one (relay.inflight does not count it):
            # wait for the look to be busy, then give the request a moment to stream
            t_end = time.time() + 30
            while not hx.writer.busy and wt.is_alive() and time.time() < t_end:
                time.sleep(0.005)
            time.sleep(0.8)
            t_writer = time.time()
            alive_at_send = wt.is_alive()
            short = {"model": model, "stream": True, "max_tokens": 8, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": "Say hi."}]}
            stc, _, _ = px.post("/v1/chat/completions", short, headers=UA, timeout=120)
            wt.join(300)
            run = last_run(cond)
            due = cond.memory.writer.state("px:c:live-yield")
            got = res.get("r")
            yielded = got == "yielded"
            hold = stc == 200 and (yielded and (due or {}).get("done_at") == 0 or got == "wrote")
            ok &= hold
            report["steps"]["yield"] = {"tick": got, "client_status": stc, "run": run,
                                        "still_due": (due or {}).get("done_at") == 0,
                                        "writer_streaming_when_sent": alive_at_send,
                                        "client_after_writer_s": round(time.time() - t_writer, 2),
                                        "observed": "yielded" if yielded else
                                        "the writer finished before the client's request",
                                        "pass": hold}
            print(f"[writer-live] yield: tick={got} skipped={run.get('skipped')} client={stc} "
                  f"still_due={(due or {}).get('done_at') == 0} {'PASS' if hold else 'FAIL'}",
                  flush=True)
        cond.journal.flush()
        report["log_lines"] = [m for m in px.logs if "writer" in m][-20:]
    report["pass"] = ok
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[writer-live] {a.kind} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
