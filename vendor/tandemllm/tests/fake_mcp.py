"""A reference MCP server for chatd's tests, the standard library only, stdio or streamable HTTP.

    python tests/fake_mcp.py                     stdio (one JSON message a line)
    srv, port = fake_mcp.start_http()            streamable HTTP on 127.0.0.1:<port>/mcp

Tools: `echo {text}` (the text back), `add {a, b}`, `slow {seconds}` (progress notifications every
0.2 s, then a result; a `notifications/cancelled` for it ends it with no answer), `grow` (adds the
tool `extra` and sends `notifications/tools/list_changed`), `fail` (isError), `secret_env` (says
whether FAKE_MCP_TOKEN is set, never its value). tools/list pages two tools at a time (nextCursor).
One prompt (`review {file}`) and one resource (`file:///notes/todo.txt`).
`FAKE_MCP_HEADER` (HTTP): the bearer the server wants; a request without it gets 401.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = [
    {"name": "echo", "description": "Say the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "add", "description": "Add two numbers.",
     "inputSchema": {"type": "object", "properties": {"a": {"type": "number"},
                                                      "b": {"type": "number"}}}},
    {"name": "slow", "description": "Takes a while, with progress.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "grow", "description": "Adds a tool.", "inputSchema": {"type": "object"}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}},
    {"name": "secret_env", "description": "Whether the token is set.",
     "inputSchema": {"type": "object"}},
]


class Server:
    def __init__(self, send):
        self.send = send
        self.tools = [dict(t) for t in TOOLS]
        self.cancelled: set = set()
        self.calls = []

    def handle(self, m: dict):
        """Returns the response (or None for a notification); progress goes through send."""
        method, mid, p = m.get("method"), m.get("id"), m.get("params") or {}
        if method == "notifications/cancelled":
            self.cancelled.add(p.get("requestId"))
            return None
        if mid is None:
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": True},
                                                                  "prompts": {}, "resources": {}},
                "serverInfo": {"name": "fake-mcp", "version": "1"}}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            start = int(p.get("cursor") or 0)
            page = self.tools[start:start + 2]
            res = {"tools": page}
            if start + 2 < len(self.tools):
                res["nextCursor"] = str(start + 2)
            return {"jsonrpc": "2.0", "id": mid, "result": res}
        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"prompts": [
                {"name": "review", "description": "Review a file.",
                 "arguments": [{"name": "file", "required": True}]}]}}
        if method == "prompts/get":
            f = (p.get("arguments") or {}).get("file", "?")
            return {"jsonrpc": "2.0", "id": mid, "result": {"messages": [
                {"role": "user", "content": {"type": "text", "text": f"Review {f} for bugs."}}]}}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"resources": [
                {"uri": "file:///notes/todo.txt", "name": "todo", "mimeType": "text/plain"}]}}
        if method == "resources/read":
            return {"jsonrpc": "2.0", "id": mid, "result": {"contents": [
                {"uri": p.get("uri"), "mimeType": "text/plain", "text": "buy milk"}]}}
        if method == "tools/call":
            name, args = p.get("name"), p.get("arguments") or {}
            self.calls.append((name, args))
            token = (p.get("_meta") or {}).get("progressToken")
            if name == "echo":
                return _ok(mid, str(args.get("text")))
            if name == "add":
                return _ok(mid, str(float(args.get("a", 0)) + float(args.get("b", 0))))
            if name == "slow":
                n = round(float(args.get("seconds", 1)) / 0.2)
                for i in range(n):
                    if mid in self.cancelled:
                        return None
                    if token is not None:
                        self.send({"jsonrpc": "2.0", "method": "notifications/progress",
                                   "params": {"progressToken": token, "progress": i + 1,
                                              "total": n}})
                    time.sleep(0.2)
                return None if mid in self.cancelled else _ok(mid, "slow done")
            if name == "grow":
                if not any(t["name"] == "extra" for t in self.tools):
                    self.tools.append({"name": "extra", "description": "Added later.",
                                       "inputSchema": {"type": "object"}})
                self.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
                return _ok(mid, "grown")
            if name == "fail":
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "it failed"}], "isError": True}}
            if name == "secret_env":
                return _ok(mid, "token set" if os.environ.get("FAKE_MCP_TOKEN") else "no token")
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602,
                                                          "message": f"no tool {name}"}}
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no method"}}


def _ok(mid, text):
    return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}


def stdio() -> None:
    lock = threading.Lock()

    def send(m):
        with lock:
            sys.stdout.write(json.dumps(m) + "\n")
            sys.stdout.flush()
    srv = Server(send)

    def work(m):
        r = srv.handle(m)
        if r is not None:
            send(r)
    for line in sys.stdin:
        if line.strip():
            threading.Thread(target=work, args=(json.loads(line),), daemon=True).start()


def start_http(header: str | None = None):
    """(server, port): streamable HTTP; a tools/call answers as an SSE stream (progress, then the
    result), everything else as JSON."""
    state = {"srv": None, "sessions": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            if header and self.headers.get("Authorization") != f"Bearer {header}":
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length") or 0)
            m = json.loads(self.rfile.read(n))
            if m.get("method") == "tools/call":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def send(x):
                    self.wfile.write(f"data: {json.dumps(x)}\n\n".encode())
                    self.wfile.flush()
                state["srv"].send = send
                r = state["srv"].handle(m)
                if r is not None:
                    send(r)
                return
            r = state["srv"].handle(m)
            if r is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            raw = json.dumps(r).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if m.get("method") == "initialize":
                state["sessions"] += 1
                self.send_header("Mcp-Session-Id", f"s{state['sessions']}")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    state["srv"] = Server(lambda m: None)
    httpd.fake = state["srv"]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


if __name__ == "__main__":
    stdio()
