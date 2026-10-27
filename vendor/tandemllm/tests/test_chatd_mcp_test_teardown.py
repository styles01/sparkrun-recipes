"""A failed draft test must close the transport it started (deep audit 0.11.3).

test() spawns a real stdio server into the chat's sandbox before the handshake; on
McpError it used to return without closing, so the pipe's reader stayed up holding
the sandbox's busy-pin (its socket has no timeout) and the sandbox never idle-stopped.
session() has closed on the same failure all along (SRV-359); test() had not.

Run: python -m pytest -q tests/test_chatd_mcp_test_teardown.py
"""

from __future__ import annotations

import threading
import time

from chatd_util import Chat


def test_a_failed_draft_test_closes_the_transport():
    closed = threading.Event()

    def opener(row, cid):
        def open_pipe(on_line):
            def reader():
                time.sleep(0.15)          # a draft server that dies without answering
                on_line(None)
            threading.Thread(target=reader, daemon=True).start()
            return (lambda b: None), (lambda: None), closed.set
        return open_pipe

    c = Chat()
    c.svc.mcp.stdio_pipe = opener
    cid = c.new()
    r = c.ok("POST", "/mcp/servers/test", {"name": "silent", "transport": "stdio",
                                           "command": "x", "conversation_id": cid})
    assert r["ok"] is False and "closed the connection" in r["message"], r
    assert closed.wait(2), "the failed draft test never closed its transport"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
