"""The standalone proxy's request timeout (SRV-605, the proxy's part of GitHub issue #9).

A client that connects and sends nothing, stops halfway through a request, or keeps a connection
idle after an answer gives its thread back after the timeout; the answer has none: a streamed
answer whose upstream goes quiet for longer than the timeout arrives whole, and every answer's
head goes out with the timeout lifted. The option refuses a negative or non-numeric value as a
usage error. The chat forward's side is in tests/test_chat_forward.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import Proxy, fake, run  # noqa: E402

from proxy import __main__ as proxy_main  # noqa: E402
from proxy import app as proxy_app  # noqa: E402

T = 0.4


@contextlib.contextmanager
def _args(**kw):
    """Proxy() builds with these options (tests/proxy_util.py passes none)."""
    real = proxy_app.build
    proxy_app.build = lambda a, env=None: real(argparse.Namespace(**vars(a), **kw), env=env)
    try:
        yield
    finally:
        proxy_app.build = real
        proxy_app.Handler.timeout = proxy_app.IDLE_TIMEOUT_S


def _idle_timeout(t):
    return _args(idle_timeout=t)


def _closed_within(port: int, send: bytes, s: float) -> float:
    cl = socket.create_connection(("127.0.0.1", port), timeout=s + 5)
    t0 = time.monotonic()
    if send:
        cl.sendall(send)
    try:
        while cl.recv(4096):
            pass
    except ConnectionResetError:
        pass
    took = time.monotonic() - t0
    cl.close()
    assert took < s, f"the connection stayed open {took:.1f} s"
    return took


def test_a_silent_client_is_let_go_and_a_quiet_stream_is_not_cut():
    srv, port = fake(models=["fake-a"])
    env = {"TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "generic",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{port}"}
    try:
        with _idle_timeout(T), Proxy(env) as px:
            px.sync()
            assert _closed_within(px.port, b"", 3.0) >= T * 0.8
            _closed_within(px.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                                    b"Content-Length: 100\r\n\r\n{", 3.0)
            # idle after an answer: the timeout is armed again for the next request
            _closed_within(px.port, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n", 3.0)
            heads = []
            real = proxy_app.Handler.end_headers

            def end_headers(self):
                heads.append(self.connection.gettimeout())
                real(self)
            proxy_app.Handler.end_headers = end_headers
            try:
                # the upstream is quiet for 1.5 s after the third token: past the timeout
                st, _, got = px.post("/v1/chat/completions", {
                    "model": "fake-a", "stream": True,
                    "messages": [{"role": "user", "content": "x FAKE_STALL 1.5"}]})
                assert st == 200 and got.endswith(b"data: [DONE]\n\n"), got[-200:]
                assert b"timeout" not in got
                assert px.get("/health")[0] == 200
            finally:
                proxy_app.Handler.end_headers = real
            assert heads and all(t is None for t in heads), heads
    finally:
        srv.shutdown()


def test_the_knob():
    old = os.environ.pop("TANDEM_IDLE_TIMEOUT_S", None)
    try:
        assert proxy_main.parser().parse_args([]).idle_timeout == 60.0
        os.environ["TANDEM_IDLE_TIMEOUT_S"] = "15"
        assert proxy_main.parser().parse_args([]).idle_timeout == 15.0
        assert proxy_main.parser().parse_args(["--idle-timeout", "0"]).idle_timeout == 0.0
    finally:
        os.environ.pop("TANDEM_IDLE_TIMEOUT_S", None)
        if old is not None:
            os.environ["TANDEM_IDLE_TIMEOUT_S"] = old
    # build() puts it on the handler: the default without the argument, none for 0
    with Proxy():
        assert proxy_app.Handler.timeout == proxy_app.IDLE_TIMEOUT_S
    with _idle_timeout(0), Proxy():
        assert proxy_app.Handler.timeout is None


def _usage_error(argv, env=None) -> str:
    saved = os.environ.pop("TANDEM_IDLE_TIMEOUT_S", None)
    os.environ.update(env or {})
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            proxy_main.parser().parse_args(argv)
    except SystemExit as exc:
        assert exc.code == 2
        return err.getvalue()
    finally:
        os.environ.pop("TANDEM_IDLE_TIMEOUT_S", None)
        if saved is not None:
            os.environ["TANDEM_IDLE_TIMEOUT_S"] = saved
    raise AssertionError(f"{argv} {env} was accepted")


def test_a_bad_value_is_a_usage_error():
    for argv, env, var in ((["--idle-timeout", "-1"], None, "TANDEM_IDLE_TIMEOUT_S"),
                           (["--idle-timeout", "soon"], None, "TANDEM_IDLE_TIMEOUT_S"),
                           ([], {"TANDEM_IDLE_TIMEOUT_S": "abc"}, "TANDEM_IDLE_TIMEOUT_S"),
                           ([], {"TANDEM_IDLE_TIMEOUT_S": "nan"}, "TANDEM_IDLE_TIMEOUT_S")):
        msg = _usage_error(argv, env)
        assert var in msg and ">= 0" in msg, msg


if __name__ == "__main__":
    run(globals())
