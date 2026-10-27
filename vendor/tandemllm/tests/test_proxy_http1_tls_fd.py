"""A failed TLS handshake closes the connected socket: no fd leak, no walk to EMFILE.

`Pool.get` reaches the handshake with a plain socket it created. The rebound `sock` only
exists once `wrap_socket` RETURNS, so every raise out of the handshake -- TLS against a
plain port, a refused version, an alert -- escaped with the fd owned by nobody: neither
caller nor pool could close it. One fd per failed request to a misconfigured backend, and
fd exhaustion is a proxy-wide outage. The handshake failure now closes the socket and
re-raises.

Run: python tests/test_proxy_http1_tls_fd.py
"""

from __future__ import annotations

import os
import socket
import ssl
import sys

try:
    import pytest

    def _needs_proc(fn):
        return pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts fds through /proc")(fn)
except ModuleNotFoundError:                    # a standalone run without pytest: the body skips
    def _needs_proc(fn):
        return fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy.http1 import Pool, UpstreamError  # noqa: E402


class _Boom:
    """A context whose wrap_socket always fails, like any handshake against a plain port."""

    def __init__(self):
        self.got = []                          # the raw sockets handed to wrap_socket

    def wrap_socket(self, sock, server_hostname=None):
        self.got.append(sock)
        raise ssl.SSLError(f"handshake failure against {server_hostname}")


def _listener():
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", 0))
    ls.listen(64)                              # the refused handshakes stay in the backlog
    return ls


def _fds():
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def _fail(p):
    try:
        p.get()
        raise AssertionError("the handshake was rigged to fail")
    except UpstreamError as exc:
        assert "handshake failure" in str(exc), exc


def test_a_handshake_failure_closes_the_socket_it_was_given():
    ls = _listener()
    try:
        p = Pool(f"https://127.0.0.1:{ls.getsockname()[1]}")
        boom = _Boom()
        p._ctx = boom  # type: ignore[misc]  (a stand-in context)
        for _ in range(5):
            _fail(p)
        assert len(boom.got) == 5
        for s in boom.got:
            try:                                # a closed socket: a send now raises
                s.send(b"x")
                raise AssertionError("the socket the failed handshake held is still open")
            except OSError:
                pass
    finally:
        ls.close()


@_needs_proc
def test_twenty_failures_leak_no_fd():
    if not os.path.isdir("/proc/self/fd"):    # a pytest-less standalone run skips itself
        return
    # the real symptom: a misconfigured https backend drains the fd table a request at a
    # time. The count may breathe a little (the loop's own temporaries); it must not walk.
    ls = _listener()
    try:
        p = Pool(f"https://127.0.0.1:{ls.getsockname()[1]}")
        p._ctx = _Boom()
        for _ in range(3):
            _fail(p)                            # settle: first connect warms caches
        before = _fds()
        for _ in range(20):
            _fail(p)
        assert _fds() <= before + 1, f"{before} -> {_fds()} fds over 20 failed handshakes"
    finally:
        ls.close()


class _BoomValueError:
    """A context whose wrap_socket raises a non-OSError: a bad server_hostname is a
    ValueError. The fd must still close and the exit still convert to UpstreamError."""

    def __init__(self):
        self.got = []

    def wrap_socket(self, sock, server_hostname=None):
        self.got.append(sock)
        raise ValueError("illegal character or position in hostname")


def test_a_non_oserror_from_the_handshake_closes_the_socket_and_converts():
    ls = _listener()
    try:
        p = Pool(f"https://127.0.0.1:{ls.getsockname()[1]}")
        boom = _BoomValueError()
        p._ctx = boom  # type: ignore[misc]  (a stand-in context)
        try:
            p.get()
            raise AssertionError("wrap_socket was rigged to raise a ValueError")
        except UpstreamError as exc:
            assert "ValueError" in str(exc), exc
        assert len(boom.got) == 1
        try:                                # the same close invariant as the OSError door
            boom.got[0].send(b"x")
            raise AssertionError("the ValueError exit left the socket open")
        except OSError:
            pass
    finally:
        ls.close()


def test_an_empty_addr_list_names_the_host_as_upstream_error():
    ls = _listener()
    try:
        p = Pool(f"https://127.0.0.1:{ls.getsockname()[1]}")
        p._addrs = []                         # a resolver that answers with nothing
        p._addrs_at = 1e18                    # fresh, so get() keeps the empty list
        try:
            p.get()
            raise AssertionError("the addr list was rigged empty")
        except UpstreamError as exc:
            assert "no addresses" in str(exc), exc      # not a raw AttributeError
    finally:
        ls.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
