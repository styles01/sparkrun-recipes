#!/usr/bin/env python3
"""sparkdash-llm-gateway — attribution proxy in front of the vLLM lane.

Agent name  <- Authorization: Bearer <name>
Session id  <- X-Hermes-Session (Hermes session_affinity_header)
Profile     <- X-Hermes-Profile (Hermes extra_headers)

Permissive by design: any token is accepted, the token IS the signin.
Fail-open: if upstream is down we return 502 but never crash.

Listens :8001  ->  forwards to UPSTREAM (default http://127.0.0.1:8000)
Logs one JSON line per request to LOG_PATH.
"""
import json, os, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request, urllib.error
from urllib.parse import urlsplit

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
LISTEN = os.environ.get("LISTEN", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8001"))
LOG_PATH = os.environ.get("LOG_PATH", "/home/jaita/sparkgateway/requests.jsonl")
UPSTREAM_KEY = os.environ.get("UPSTREAM_API_KEY", "")  # optional real vLLM key

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
       "authorization"}


def parse_auth(hdr):
    """Bearer token, or raw token. Returns (name, session_from_token)."""
    if not hdr:
        return "", ""
    h = hdr.strip()
    if h.lower().startswith("bearer "):
        h = h[7:].strip()
    if ":" in h:
        name, _, sess = h.partition(":")
        return name.strip(), sess.strip()
    return h, ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # we write structured records instead

    def _record(self, rec):
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def _proxy(self, method):
        rid = uuid.uuid4().hex[:16]
        t0 = time.time()
        name, tok_session = parse_auth(self.headers.get("Authorization"))
        session = self.headers.get("X-Hermes-Session") or tok_session or ""
        profile = self.headers.get("X-Hermes-Profile") or name

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        # pass everything except hop-by-hop and the caller's bearer
        fwd = {}
        for k, v in self.headers.items():
            if k.lower() not in HOP:
                fwd[k] = v
        fwd["X-Spark-Agent"] = name or "anonymous"
        fwd["X-Spark-Session"] = session
        fwd["X-Spark-Rid"] = rid
        if UPSTREAM_KEY:
            fwd["Authorization"] = f"Bearer {UPSTREAM_KEY}"

        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=body, headers=fwd, method=method)

        model = ""
        if body:
            try:
                model = json.loads(body).get("model", "")
            except Exception:
                pass

        try:
            resp = urllib.request.urlopen(req, timeout=1800)
            status = resp.status
            data = resp.read()
        except urllib.error.HTTPError as e:
            status = e.code
            data = e.read()
        except Exception as e:
            status = 502
            data = json.dumps({"error": {"message": f"gateway upstream failure: {e}"}}).encode()

        usage = {}
        if body and self.path.rstrip("/").endswith("chat/completions") and status == 200:
            try:
                usage = json.loads(data).get("usage", {}) or {}
            except Exception:
                pass

        self._record({
            "ts": round(time.time(), 3),
            "rid": rid,
            "agent": name or "anonymous",
            "profile": profile,
            "session": session,
            "method": method,
            "path": urlsplit(self.path).path,
            "model": model,
            "status": status,
            "ms": round((time.time() - t0) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })

        self.send_response(status)
        ct = "application/json"
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass


if __name__ == "__main__":
    print(f"sparkdash-llm-gateway listening {LISTEN}:{PORT} -> {UPSTREAM} (log {LOG_PATH})", flush=True)
    ThreadingHTTPServer((LISTEN, PORT), Handler).serve_forever()
