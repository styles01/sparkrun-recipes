"""ops/sandbox/relay.py: a refused round trip closes its socket (deep audit 0.11.3).

connect(SOCK) failing -- chatd's proxy.sock absent, a sandbox's every proxied
request before chatd answers -- used to answer 502, close the client and return
WITHOUT closing the unix socket it had just created: one leaked fd per request,
until accept() itself raises EMFILE at the fd ceiling -- and an accept() that
escapes the bare serve() loop ends the container's PID-1, the sandbox's whole
network path.

Run: python -m pytest -q tests/test_sandbox_relay_fd.py
"""

from __future__ import annotations

import importlib.util
import os
import socket
import threading
import time


def _load():
    spec = importlib.util.spec_from_file_location(
        "sb_relay", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "ops", "sandbox", "relay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _SocketSpy:
    """Forwards the socket module to relay.py, counting AF_UNIX sockets opened vs closed."""

    def __init__(self, real):
        self.real = real
        self.opened = 0
        self.closed = 0

    def socket(self, family=-1, *a, **k):
        s = self.real.socket(family, *a, **k)
        if family == self.real.AF_UNIX:
            self.opened += 1
            return _SpySock(s, self)
        return s

    def __getattr__(self, name):
        return getattr(self.real, name)


class _SpySock:
    """A real socket instance cannot get attributes, so close() is counted through a
    wrapper; every other call passes through untouched."""

    def __init__(self, s, spy):
        self._s = s
        self._spy = spy

    def close(self):
        self._spy.closed += 1
        self._s.close()

    def __getattr__(self, name):
        return getattr(self._s, name)


def test_a_refused_round_trip_closes_its_upstream_socket():
    relay = _load()
    relay.SOCK = "/no/such/dir/proxy-9s7d.sock"     # chatd socket absent: connect -> ENOENT
    relay.LOG = "/dev/null"
    spy = _SocketSpy(socket)
    relay.socket = spy
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def loop() -> None:
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=relay.handle, args=(c,), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    for _ in range(20):
        c = socket.create_connection(("127.0.0.1", port), timeout=2)
        c.sendall(b"GET /v1/chat/completions HTTP/1.1\r\nHost: q\r\nContent-Length: 0\r\n\r\n")
        assert b"502" in c.recv(200)
        c.close()
    deadline = time.time() + 2.0
    while spy.closed < spy.opened and time.time() < deadline:   # a handler mid-flight is not
        time.sleep(0.1)                                         # a leak; a held socket is
    assert spy.opened == 20 and spy.closed == 20, \
        f"upstream sockets opened {spy.opened}, closed {spy.closed}"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
