"""A fake decider for the conductor's tests: the Jev HTTP API with no model, stdlib only.

    python server/conductor/fake_decider.py --port 8710 [--model other-decider --context 256
        --question 96] [--delay-ms 0]

It answers the Jev contract the decider uses (`GET /readyz`, `GET /v1/models` with `limits`,
`POST /v1/systemone` with `choice` questions) and, for the Qwen stage, `POST /v1/chat/completions`
with `structured_outputs.choice` and log-probabilities, so the e2e can point both classifier roles
at it. It is a test double: words in the text steer it, which is exactly what production routing
must never do.

    FAKE_CLASS=<id> [FAKE_CONF=0.xx]    systemone picks <id> at that probability (default 0.92)
    FAKE_QWEN=<id> [FAKE_QCONF=0.xx]    chat picks <id> (default 0.9)
    FAKE_PKG=<name>                     a package question picks <name>
    otherwise                           an option id that appears in the text as a word, at 0.91;
                                        else the first option at 0.2 (below every threshold)

`GET /__calls` counts the requests per endpoint, and `POST /__calls` resets them; `--model` and
the limits check that the conductor works with any decision model (the decider's contract). The
states it received are kept (`GET /__states`) so a test can check what a classifier saw.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS = {"systemone": 0, "chat": 0, "models": 0, "readyz": 0}
STATES: list[str] = []
LOCK = threading.Lock()


def _pick(text: str, options: list[str], key: str, conf_key: str, default: float):
    m = re.search(rf"{key}=([a-z0-9_-]+)", text)
    if m and m.group(1) in options:
        c = re.search(rf"{conf_key}=([0-9.]+)", text)
        return m.group(1), float(c.group(1)) if c else default
    for o in options:
        if o != "none" and re.search(rf"(?<![A-Za-z0-9_-]){re.escape(o)}(?![A-Za-z0-9_-])", text):
            return o, 0.91
    return options[0], 0.2


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/readyz":
            CALLS["readyz"] += 1
            return self._send(200, {"status": "ready", "engine": "fake", "models": ["fake"],
                                    "device": "cpu", "version": "0.0.0"})
        if path == "/healthz":
            return self._send(200, {"status": "ok"})
        if path == "/v1/models":
            CALLS["models"] += 1
            lim = {"context_tokens": self.cfg["context"], "question_tokens": self.cfg["question"],
                   "max_options": 20}
            return self._send(200, {"object": "list", "data": [
                {"id": self.cfg["model"], "object": "model", "limits": lim, "aliases": []},
                {"id": "jev-latest", "object": "model", "limits": lim, "aliases": ["auto"]},
                {"id": "qwen38-spark-engine", "object": "model"}]})
        if path == "/__calls":
            return self._send(200, dict(CALLS))
        if path == "/__states":
            return self._send(200, {"states": STATES[-50:]})
        return self._send(404, {"error": {"message": "no route"}})

    def do_POST(self):
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if path == "/__calls":
            with LOCK:
                for k in CALLS:
                    CALLS[k] = 0
                STATES.clear()
            return self._send(200, {})
        if self.cfg["delay_ms"]:
            time.sleep(self.cfg["delay_ms"] / 1000)
        if path == "/v1/systemone":
            with LOCK:
                CALLS["systemone"] += 1
                STATES.append(body.get("state", ""))
            state = str(body.get("state", ""))
            answers = {}
            for qid, q in (body.get("questions") or {}).items():
                opts = list((q.get("criteria") or {}).keys())
                if "none" in opts:
                    label, conf = _pick(state, opts, "FAKE_PKG", "FAKE_PCONF", 0.9)
                else:
                    label, conf = _pick(state, opts, "FAKE_CLASS", "FAKE_CONF", 0.92)
                rest = (1.0 - conf) / max(1, len(opts) - 1)
                probs = {o: (conf if o == label else rest) for o in opts}
                answers[qid] = {"type": "choice", "choice": label, "probabilities": probs,
                                "confidence": conf}
            return self._send(200, {"model": self.cfg["model"], "answers": answers,
                                    "usage": {"input_tokens": len(state) // 4,
                                              "output_tokens": 0},
                                    "latency_ms": self.cfg["delay_ms"] or 1.0})
        if path == "/v1/chat/completions":
            with LOCK:
                CALLS["chat"] += 1
            msgs = body.get("messages") or []
            text = str((msgs[-1] if msgs else {}).get("content") or "")
            STATES.append(text)
            opts = list(((body.get("structured_outputs") or {}).get("choice")) or [])
            label, conf = _pick(text, opts, "FAKE_QWEN", "FAKE_QCONF", 0.9)
            return self._send(200, {
                "id": "chatcmpl-fake", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": label},
                             "logprobs": {"content": [{"token": label,
                                                       "logprob": math.log(conf),
                                                       "top_logprobs": []}]}}]})
        return self._send(404, {"error": {"message": "no route"}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--model", default="laya-english")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--question", type=int, default=192)
    ap.add_argument("--delay-ms", type=int, default=0)
    a = ap.parse_args()
    H.cfg = {"model": a.model, "context": a.context, "question": a.question,
             "delay_ms": a.delay_ms}
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[fake-decider] on 127.0.0.1:{a.port} model {a.model}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
