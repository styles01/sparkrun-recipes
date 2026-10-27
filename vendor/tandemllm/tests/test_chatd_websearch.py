"""chatd's web_search tool (SRV-469, QA SRV-470): the DuckDuckGo and SearXNG providers on recorded
answers, the egress host list, the settings (validation, the secret, a chat's override) and the
tool in a chat. CPU only; the network is a recorder or a stub server on 127.0.0.1.

Run: python -m pytest -q tests/test_chatd_websearch.py
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chatd_util import Chat, run_all

from chatd import egress, tools as T, websearch as W
from chatd.plans import PLAN_TOOLS

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "websearch")


def fx(name: str) -> bytes:
    with open(os.path.join(FX, name), "rb") as fh:
        return fh.read()


def calls(*c) -> str:
    return "FAKE_CALLS " + json.dumps([{"name": n, "arguments": a} for n, a in c])


class Recorder:
    """The transport: records each request, answers (status, type, body)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen: list[dict] = []

    def __call__(self, url, **kw):
        self.seen.append({"url": url, **kw})
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        return a


DDG = {**W.DEFAULTS, "provider": "duckduckgo"}
SX = {**W.DEFAULTS, "provider": "searxng", "searxng_url": "http://search.example:8888",
      "searxng_auth_header": "X-API-Key", "searxng_auth_value": "k1"}


# ---------------------------------------------------------------------- DuckDuckGo
def test_duckduckgo_answers_a_citable_list_without_ads_or_duplicates():
    rec = Recorder((200, "text/html", fx("ddg_results.html")))
    cfg = {**DDG, "language": "de-DE", "safe_search": "strict", "time_range": "month"}
    res = W.search(cfg, "rust ownership", 5, http=rec)
    req = rec.seen[0]
    assert req["url"] == "https://html.duckduckgo.com/html/" and req["method"] == "POST"
    assert req["hosts"] == frozenset({"html.duckduckgo.com"}) and not req.get("private_ok")
    form = dict(urllib.parse.parse_qsl(req["body"].decode()))
    assert form == {"q": "rust ownership", "kl": "de-de", "kp": "1", "df": "m"}
    assert req["headers"]["User-Agent"].startswith("Mozilla/5.0")
    rs = res["results"]
    assert len(rs) == 5 and res["provider"] == "duckduckgo"
    assert all(r["title"] and r["url"].startswith("https://") and r["snippet"] for r in rs)
    urls = [r["url"] for r in rs]
    assert not any("duckduckgo.com" in u for u in urls)                 # the ad is gone
    assert "https://www.w3schools.com/rust/rust_ownership.php" in urls   # uddg decoded
    assert len({W._key(u) for u in urls}) == len(urls)                   # no duplicate


def test_duckduckgo_keeps_dates_and_all_results_up_to_the_limit():
    rec = Recorder((200, "text/html", fx("ddg_results.html")))
    rs = W.search(DDG, "rust ownership", 20, http=rec)["results"]
    assert len(rs) == 10
    assert [r.get("date") for r in rs if r.get("date")] == ["2026-03-18", "2026-06-07"]


def test_duckduckgo_regions_and_safe_search_words():
    assert W.ddg_region("") == "wt-wt" and W.ddg_region("de") == "wt-wt"
    assert W.ddg_region("en-GB") == "uk-en" and W.ddg_region("en-US") == "us-en"
    rec = Recorder((200, "text/html", b"<html></html>"))
    W.search({**DDG, "safe_search": "off"}, "x", http=rec)
    form = dict(urllib.parse.parse_qsl(rec.seen[0]["body"].decode()))
    assert form["kp"] == "-2" and form["kl"] == "wt-wt" and "df" not in form


def test_duckduckgo_bot_check_is_a_clear_error():
    for status in (202, 200):
        rec = Recorder((status, "text/html", fx("ddg_anomaly.html")))
        try:
            W.search(DDG, "rust ownership", http=rec)
        except W.SearchError as exc:
            msg = str(exc)
        else:
            raise AssertionError("no error")
        assert "DuckDuckGo refused the search" in msg and "SearXNG" in msg


def test_duckduckgo_unreachable_and_other_statuses():
    for answer, words in ((egress.Refused("no answer from html.duckduckgo.com within 10 s"),
                           "cannot reach DuckDuckGo"), ((500, "text/html", b"oops"), "HTTP 500")):
        try:
            W.search(DDG, "x", http=Recorder(answer))
        except W.SearchError as exc:
            assert words in str(exc)
        else:
            raise AssertionError("no error")


def test_nothing_found_is_no_results_not_an_error():
    res = W.search(DDG, "zzqx", http=Recorder((200, "text/html", b"<div class=no-results/>")))
    assert res["results"] == []
    assert W.as_text(res) == 'Web search for "zzqx" (DuckDuckGo): no results.'


# ---------------------------------------------------------------------- SearXNG
def test_searxng_answers_json_with_engine_and_date():
    rec = Recorder((200, "application/json", fx("searxng_results.json")))
    res = W.search({**SX, "language": "de-DE", "safe_search": "strict", "time_range": "week"},
                   "lit element", http=rec)
    req = rec.seen[0]
    u = urllib.parse.urlsplit(req["url"])
    assert (u.scheme, u.netloc, u.path) == ("http", "search.example:8888", "/search")
    q = dict(urllib.parse.parse_qsl(u.query))
    assert q == {"q": "lit element", "format": "json", "categories": "general", "pageno": "1",
                 "language": "de-DE", "safesearch": "2", "time_range": "week"}
    assert req["headers"]["X-API-Key"] == "k1"
    assert req["hosts"] == frozenset({"search.example:8888"}) and req["private_ok"] is True
    rs = res["results"]
    assert [r["url"] for r in rs] == ["https://lit.dev/docs/components/overview/",
                                      "https://github.com/lit/lit",
                                      "https://developer.mozilla.org/en-US/docs/Web/API/"
                                      "Web_components"]
    assert rs[0]["engine"] == "bing" and rs[1]["date"] == "2026-09-30" and "date" not in rs[0]
    text = W.as_text(res)
    assert "1. Components overview – Lit" in text and "2026-09-30 · brave" in text
    assert "web_fetch" in text


def test_searxng_without_auth_sends_no_header_and_language_all():
    rec = Recorder((200, "application/json", fx("searxng_results.json")))
    W.search({**SX, "searxng_auth_header": "", "searxng_auth_value": ""}, "x", http=rec)
    assert "X-API-Key" not in rec.seen[0]["headers"]
    assert dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(rec.seen[0]["url"]).query))[
        "language"] == "all"


def test_searxng_errors_name_the_cause():
    cases = [((403, "text/html", b"Forbidden"), "search.formats"),
             ((401, "text/plain", b""), "auth header"),
             ((429, "text/plain", b""), "limiter"),
             ((502, "text/html", b""), "HTTP 502"),
             ((200, "text/html", b"<html>search page</html>"), "did not answer JSON (text/html)"),
             ((200, "application/json", b'{"query": "x"}'), "without results"),
             ((200, "application/json", fx("searxng_unresponsive.json")),
              "engines did not answer: google, bing"),
             (egress.Refused("cannot connect to search.example:8888 (ConnectionRefusedError)"),
              "cannot reach the SearXNG instance at search.example:8888")]
    for answer, words in cases:
        try:
            W.search(SX, "x", http=Recorder(answer))
        except W.SearchError as exc:
            assert words in str(exc), (words, str(exc))
        else:
            raise AssertionError(f"no error for {words}")


def test_searxng_empty_answer_is_no_results():
    res = W.search(SX, "zzqx", http=Recorder((200, "application/json", fx("searxng_empty.json"))))
    assert res["results"] == [] and "no results" in W.as_text(res)


def test_a_query_is_checked():
    for q, words in (("  ", "query is empty"), ("x" * 501, "longer than 500")):
        try:
            W.search(DDG, q, http=Recorder((200, "text/html", b"")))
        except W.SearchError as exc:
            assert words in str(exc)
        else:
            raise AssertionError("no error")


# ---------------------------------------------------------------------- stub servers
class _Stub(BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.seen.append({"path": self.path, "headers": dict(self.headers)})
        path = self.path.split("?")[0]
        status, headers, body = self.routes.get(path, (404, {}, b""))
        if status == "hang":
            time.sleep(5)
            return
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def stub(routes: dict):
    handler = type("H", (_Stub,), {"routes": routes, "seen": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler


def test_searxng_that_does_not_answer_in_time_fails_within_the_timeout():
    srv, _ = stub({"/search": ("hang", {}, b"")})
    try:
        port = srv.server_address[1]
        cfg = {**SX, "searxng_url": f"http://127.0.0.1:{port}", "timeout_s": 2}
        t0 = time.monotonic()
        try:
            W.search(cfg, "x")
        except W.SearchError as exc:
            msg = str(exc)
        else:
            raise AssertionError("no error")
        assert time.monotonic() - t0 < 4.5
        assert f"cannot reach the SearXNG instance at 127.0.0.1:{port}" in msg
        assert "no answer" in msg and "2 s" in msg
    finally:
        srv.shutdown()


def test_egress_reaches_only_the_listed_host_redirects_included():
    srv, h = stub({})
    port = srv.server_address[1]
    h.routes.update({"/search": (302, {"Location": f"http://localhost:{port}/search"}, b""),
                     "/ok": (200, {"Content-Type": "application/json"}, b"{}")})
    try:
        hosts = frozenset({f"127.0.0.1:{port}"})
        st, ctype, body = egress.request(f"http://127.0.0.1:{port}/ok", hosts=hosts,
                                         private_ok=True)
        assert (st, ctype, body) == (200, "application/json", b"{}")
        for url in (f"http://127.0.0.1:{port}/search", f"http://localhost:{port}/ok",
                    f"http://127.0.0.1:{port + 1}/ok"):
            try:
                egress.request(url, hosts=hosts, private_ok=True)
            except egress.Refused as exc:
                assert "not a host this request may reach" in str(exc), str(exc)
            else:
                raise AssertionError(url)
        # without a host list a private address stays refused, whatever private_ok says
        try:
            egress.request(f"http://127.0.0.1:{port}/ok", private_ok=True)
        except egress.Refused as exc:
            assert "local network address" in str(exc)
        else:
            raise AssertionError("private address reached")
        assert len(h.seen) == 2                    # /ok once, /search once (its redirect refused)
    finally:
        srv.shutdown()


def test_a_reset_connection_is_a_tool_error_not_a_crash():
    import struct
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def reset():
        for _ in range(2):
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.recv(1024)
            c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            c.close()                                   # RST instead of an answer
    threading.Thread(target=reset, daemon=True).start()
    try:
        cfg = {**SX, "searxng_url": f"http://127.0.0.1:{port}", "timeout_s": 3}
        try:
            W.search(cfg, "x")
        except W.SearchError as exc:
            assert f"cannot reach the SearXNG instance at 127.0.0.1:{port}" in str(exc)
            assert "connection to 127.0.0.1 failed" in str(exc), str(exc)
        else:
            raise AssertionError("no error")
    finally:
        srv.close()


def test_duckduckgo_on_a_private_address_is_refused_before_connecting():
    real_gai, real_socket = socket.getaddrinfo, socket.socket
    opened = []

    def fake(host, port, *a, **kw):
        if host == "html.duckduckgo.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))]
        return real_gai(host, port, *a, **kw)

    class NoSocket(real_socket):
        def connect(self, sa):
            opened.append(sa)
            raise OSError("must not connect")

    socket.getaddrinfo, socket.socket = fake, NoSocket
    try:
        W.search(DDG, "x")
    except W.SearchError as exc:
        assert "local network address" in str(exc)
    else:
        raise AssertionError("no error")
    finally:
        socket.getaddrinfo, socket.socket = real_gai, real_socket
    assert opened == []


# ---------------------------------------------------------------------- settings
def test_settings_validation_names_the_field():
    cur = dict(W.DEFAULTS)
    bad = [({"provider": "bing"}, "provider"), ({"max_results": 0}, "max_results"),
           ({"max_results": 21}, "max_results"), ({"max_results": True}, "max_results"),
           ({"timeout_s": 1}, "timeout_s"), ({"safe_search": "on"}, "safe_search"),
           ({"time_range": "hour"}, "time_range"), ({"language": "german"}, "language"),
           ({"searxng_url": "ftp://x"}, "searxng_url"),
           ({"searxng_url": "http://u:p@x"}, "searxng_url"),
           ({"searxng_url": "http://x/search?q=1"}, "searxng_url"),
           ({"provider": "searxng"}, "searxng_url"),
           ({"searxng_auth_header": "Bad Header"}, "searxng_auth_header"),
           ({"searxng_auth_header": "Host", "searxng_auth_value": "x"}, "searxng_auth_header"),
           ({"searxng_auth_value": "x"}, "searxng_auth_header"),
           ({"searxng_auth_header": "X-Key", "searxng_auth_value": "a\nb"}, "searxng_auth_value")]
    for patch, field in bad:
        try:
            W.clean(patch, cur)
        except W.Invalid as exc:
            assert exc.field == field, (patch, exc.field)
        else:
            raise AssertionError(patch)
    ok = W.clean({"provider": "searxng", "searxng_url": "https://s.example/searx/",
                  "language": "de-DE", "unknown": 1}, cur)
    assert ok["searxng_url"] == "https://s.example/searx" and "unknown" not in ok


def test_a_chat_override_and_its_validation():
    assert W.clean_override(None) is None and W.clean_override({}) is None
    assert W.clean_override({"provider": "off", "max_results": "3"}) == {"provider": "off",
                                                                         "max_results": 3}
    for v in ({"searxng_url": "http://x"}, {"provider": "x"}, "on"):
        try:
            W.clean_override(v)
        except W.Invalid:
            pass
        else:
            raise AssertionError(v)
    eff = W.effective({**W.DEFAULTS}, {"provider": "searxng"})
    assert eff["provider"] == "off"                                    # SearXNG without a URL


def test_the_routes_store_validate_and_never_answer_the_secret():
    c = Chat()
    try:
        view = c.ok("GET", "/web-search")
        assert view["settings"]["provider"] == "duckduckgo"
        assert view["settings"]["max_results"] == 8 and view["providers"] == list(W.PROVIDERS)
        st, obj, _ = c.call("PUT", "/web-search", {"max_results": 50})
        assert st == 400 and obj["field"] == "max_results"
        put = c.ok("PUT", "/web-search", {"provider": "searxng",
                                          "searxng_url": "http://127.0.0.1:9/",
                                          "searxng_auth_header": "Authorization",
                                          "searxng_auth_value": "Bearer s3cr3t"})
        for got in (put, c.ok("GET", "/web-search"), c.ok("GET", "/capabilities")):
            assert "s3cr3t" not in json.dumps(got)
        assert put["settings"]["searxng_auth_set"] is True
        raw = c.svc.store.one("SELECT value FROM meta WHERE key = 'web_search'")["value"]
        assert "s3cr3t" not in raw and "searxng_auth_sealed" in raw           # sealed at rest
        assert put["settings"]["searxng_url"] == "http://127.0.0.1:9"
        assert c.ok("GET", "/capabilities")["web_search"]["provider"] == "searxng"
        # a partial update keeps the secret; null clears it
        c.ok("PUT", "/web-search", {"max_results": 5})
        assert c.svc.web_search_global()["searxng_auth_value"] == "Bearer s3cr3t"
        cleared = c.ok("PUT", "/web-search", {"searxng_auth_value": None})
        assert cleared["settings"]["searxng_auth_set"] is False
        assert not any("s3cr3t" in line for line in c.lines)
        c.ok("PUT", "/web-search", {"provider": "duckduckgo"})
    finally:
        c.close()


def test_the_test_route_searches_with_the_draft_and_stores_nothing():
    srv, h = stub({"/search": (200, {"Content-Type": "application/json"},
                               fx("searxng_results.json"))})
    c = Chat()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        r = c.ok("POST", "/web-search/test", {"settings": {
            "provider": "searxng", "searxng_url": url, "searxng_auth_header": "X-API-Key",
            "searxng_auth_value": "k9"}})
        assert r["ok"] is True and r["count"] == 3 and r["provider"] == "searxng"
        assert r["message"].startswith("SearXNG answered 3 results in")
        assert h.seen[0]["headers"]["X-API-Key"] == "k9"
        assert c.svc.web_search_global()["provider"] == "duckduckgo"       # not stored
        h.routes["/search"] = (403, {}, b"")
        r = c.ok("POST", "/web-search/test", {"settings": {"provider": "searxng",
                                                           "searxng_url": url}})
        assert r["ok"] is False and "search.formats" in r["message"]
        st, obj, _ = c.call("POST", "/web-search/test", {"settings": {"provider": "nope"}})
        assert st == 400 and obj["field"] == "provider"
        r = c.ok("POST", "/web-search/test", {"settings": {"provider": "off"}})
        assert r["ok"] is False and r["message"] == "web search is off"
    finally:
        c.close()
        srv.shutdown()


# ---------------------------------------------------------------------- the tool in a chat
def tool_names(body) -> list[str]:
    return [t["function"]["name"] for t in body.get("tools") or []]


def test_the_tool_in_a_chat_searches_and_the_ui_gets_the_list():
    srv, h = stub({"/search": (200, {"Content-Type": "application/json"},
                               fx("searxng_results.json"))})
    c = Chat()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        c.ok("PUT", "/web-search", {"provider": "searxng", "searxng_url": url,
                                    "max_results": 2})
        caps = c.ok("GET", "/capabilities")
        assert "web_search" in caps["defaults"]["tools"]
        cid = c.new()
        c.send(cid, calls(("web_search", {"query": "lit element"})))
        s = c.wait_idle(cid, 20)
        call = next(p for m in s["messages"] for p in m["parts"] if p["type"] == "tool_call")
        assert call["name"] == "web_search" and call["status"] == "done", call
        data = call["output"]["data"]
        assert data["provider"] == "searxng" and data["query"] == "lit element"
        assert [r["url"] for r in data["results"]] == ["https://lit.dev/docs/components/overview/",
                                                       "https://github.com/lit/lit"]
        assert 'Web search for "lit element" (SearXNG, 2 results)' in call["output"]["content"]
        assert call["approved"] is None                       # default policy: always
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(h.seen[0]["path"]).query))
        assert q["q"] == "lit element" and q["format"] == "json"
        assert "web_search" in tool_names(c.last_body())
        assert any("[search] searxng: 2 results" in line for line in c.lines)
        assert not any("lit element" in line for line in c.lines)        # no query in the log
    finally:
        c.close()
        srv.shutdown()


def test_a_chat_override_switches_its_provider_off_and_back():
    c = Chat()
    try:
        cid = c.new()
        st, obj, _ = c.call("PATCH", f"/conversations/{cid}/settings",
                            {"web_search": {"searxng_url": "http://x"}})
        assert st == 400 and obj["field"] == "web_search"
        s = c.ok("PATCH", f"/conversations/{cid}/settings", {"web_search": {"provider": "off"}})
        assert s["web_search"] == {"provider": "off"}
        c.send(cid, calls(("web_search", {"query": "x"})))
        snap = c.wait_idle(cid, 20)
        call = next(p for m in snap["messages"] for p in m["parts"] if p["type"] == "tool_call")
        assert call["status"] == "failed" and "not available in this chat" in \
            call["output"]["content"]
        assert "web_search" not in tool_names(c.last_body())
        other = c.new()
        info = c.svc.model_info(None)
        conv = c.snap(other)
        assert "web_search" in [t.name for t in c.svc.tools_for(conv["settings"], info, other)]
        s = c.ok("PATCH", f"/conversations/{cid}/settings", {"web_search": None})
        assert s["web_search"] is None
        assert "web_search" in [t.name for t in c.svc.tools_for(s, info, cid)]
    finally:
        c.close()


def test_global_off_removes_the_tool_and_the_words_for_it():
    c = Chat()
    try:
        c.ok("PUT", "/web-search", {"provider": "off"})
        cid = c.new()
        conv = c.snap(cid)
        assert "web_search" not in [t.name for t in c.svc.tools_for(conv["settings"],
                                                                    c.svc.model_info(None), cid)]
    finally:
        c.close()
    tool = T.BY_NAME["web_search"]
    assert tool.risk == "network" and tool.default == "allow" and not tool.sandbox
    assert tool.schema["required"] == ["query"]
    assert T.summary("web_search", {"query": "rust"}) == 'Search the web for "rust"'
    assert "web_search" in PLAN_TOOLS


if __name__ == "__main__":
    run_all(globals())
