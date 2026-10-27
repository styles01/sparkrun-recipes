"""The preview's target and what loads in it (SRV-511, chatd/preview.py): the port proxy's CORS and
shim, a page's own directory on the view route, the target a `port:N` or a bundler project's
index.html resolves to, the dev server the pane starts, and what `present` tells the model. The
port proxy streams an answer as it arrives and holds nothing unbounded (SRV-633, #86)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time

from chatd_util import KEY, Chat, run_all
from test_chatd_agent import RelayBoxes, app_server, ws_open, ws_recv, ws_send

from chatd import preview as PV, sandbox as sb_mod, terminals as TM, tools as T
from chatd.terminals import Dechunk, port_prefix, port_target
from chatd.workspace import Workspace

_spec = importlib.util.spec_from_file_location("relay", sb_mod.RELAY)
relay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay)


class PortBoxes(RelayBoxes):
    """RelayBoxes, and `relay.py ports` answers `self.listen`, `relay.py probe` runs on the host."""

    def __init__(self, data):
        super().__init__(data)
        self.listen: list[dict] = []

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        if cmd[:2] == ["python3", "-c"] and len(cmd) > 3 and cmd[3] in ("ports", "probe"):
            if cmd[3] == "ports":
                rows = [dict(r) for r in self.listen]
                for r in rows:
                    r.setdefault("http", True)
                out = json.dumps(rows).encode()
            else:
                out = json.dumps(relay.probe(int(cmd[4]), cmd[5] if len(cmd) > 5 else "/")).encode()
            a, b = socket.socketpair()
            b.sendall(bytes([1, 0, 0, 0]) + len(out).to_bytes(4, "big") + out)
            b.close()
            return "exec-ports", a, b""
        return super().open_exec(cid, cmd, tty=tty, stdin=stdin, env=env, workdir=workdir)


def chat():
    c = Chat()
    c.svc.sandboxes = PortBoxes(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    return c


def server(pages: dict):
    """A dev server: path -> (status, type, body, extra headers); records each request's headers."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            st, ctype, body, extra = pages.get(self.path.split("?")[0],
                                               (404, "text/plain", b"no", {}))
            self.send_response(st)
            self.send_header("Content-Type", ctype)
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


VITE_INDEX = (b'<!DOCTYPE html><html><head>\n<script type="module" src="/@vite/client"></script>'
              b'<link rel="stylesheet" href="style.css"></head><body><canvas></canvas>'
              b'<script type="module" src="/src/main.js"></script></body></html>')
VITE_CLIENT = (b'const base$1 = "/" || "/";\nconst socketHost = `${null || importMetaUrl.hostname}'
               b':${hmrPort || importMetaUrl.port}${"/"}`;\nconst base = "/" || "/";\n'
               b'import "/node_modules/vite/dist/client/env.mjs";')


# ---------------------------------------------------------------------- the port proxy
def test_the_port_proxy_lets_the_opaque_preview_read_its_modules_and_adds_the_shim():
    """Live (2026-10-16, WebKit behind a box's https name): "Origin null is not allowed by
    Access-Control-Allow-Origin" for /src/main.js and /@vite/client (a dark pane) and for
    dist/assets/*.css and *.js under `vite preview` (a white one). Each answer now allows the
    opaque origin; the server sees a local client; a page gets the shim before Vite's client; the
    client's base is the port's prefix (hot updates import `base + path`)."""
    c = chat()
    app, seen = server({
        "/": (200, "text/html", VITE_INDEX, {"Access-Control-Allow-Origin": "http://localhost",
                                              "X-Frame-Options": "DENY"}),
        "/@vite/client": (200, "text/javascript", VITE_CLIENT, {}),
        "/style.css": (200, "text/css", b"body{background:url(/bg.png)}", {}),
    })
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        h_null = {"Origin": "null", "Referer": "https://tandem.example/x",
                  "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "script"}
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/", headers=h_null)
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null" and "X-Frame-Options" not in h
        # WebKit matches no 'self' for a fetch or socket from the opaque origin (plain http port)
        assert "connect-src 'self' https: wss: http: ws:;" in h["Content-Security-Policy"]
        page = body if isinstance(body, bytes) else body.encode()
        head = page.index(b"<head>") + len(b"<head>")
        assert page[head:head + 8] == b"<script>", "the shim comes first in the head"
        assert page.index(b"tandemPreview") < page.index(f'src="{base}/@vite/client"'.encode())
        assert f'"{base}"'.encode() in page[:page.index(b"</script>")] and b",L=%d," % port in page
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/@vite/client",
                             headers=h_null)
        js = body if isinstance(body, bytes) else body.encode()
        assert h["Access-Control-Allow-Origin"] == "null"
        assert f'const base$1 = "{base}/" || "/";'.encode() in js
        assert f'const base = "{base}/" || "/";'.encode() in js
        assert f'import "{base}/node_modules/vite/dist/client/env.mjs"'.encode() in js
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/style.css")
        assert "Access-Control-Allow-Origin" not in h, "only the opaque origin is allowed"
        for path, hdrs in seen:
            low = {k.lower() for k in hdrs}
            assert not low & {"origin", "referer", "sec-fetch-site", "sec-fetch-mode"}, (path, low)
        assert any("sec-fetch-dest" in {k.lower() for k in hd} for _, hd in seen)
    finally:
        app.shutdown()
        c.close()


def test_the_port_proxy_forwards_the_path_as_the_client_sent_it():
    """A dev server answers /docs with a redirect to /docs/ (a loop when the slash is cut), and a
    decoded %20 or %0d%0a in the request line breaks it or adds a header."""
    c = chat()
    app, seen = server({"/docs/": (200, "text/plain", b"docs", {})})
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        for rest in ("/docs/?q=1", "/my%20file.png", "/a%0d%0aX-Evil:%201"):
            c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}{rest}")
        assert [p for p, _ in seen] == ["/docs/?q=1", "/my%20file.png",
                                        "/a%0d%0aX-Evil:%201"], seen
        assert not any("x-evil" in {k.lower() for k in hd} for _, hd in seen), seen
        assert port_target(f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}") == "/"
        assert port_target(f"/conversations/{cid}/sandbox/ports/{port}/a b") is None
        assert port_target(f"/conversations/{cid}/sandbox/ports/{port}/a\r\nX: 1") is None
    finally:
        app.shutdown()
        c.close()


def test_the_port_proxy_keeps_an_apps_redirects_inside_the_preview():
    """SRV-631: an app's `Location: /login` sent the browser to /login on chatd's origin, out of
    the preview, and one built from the Host the proxy sends (127.0.0.1:<port>) was unreachable.
    Both now go to the prefix the client used; an external URL, another port and a relative value
    pass as the app sent them."""
    c = chat()
    pages: dict = {}
    app, seen = server(pages)
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        cases = [
            (302, "Location", "/login", f"{base}/login"),
            (302, "Location", f"http://127.0.0.1:{port}/login?next=%2Fa#top",
             f"{base}/login?next=%2Fa#top"),
            (302, "location", f"http://localhost:{port}/x", f"{base}/x"),
            (302, "Location", f"http://[::1]:{port}/x", f"{base}/x"),
            (302, "Location", f"//127.0.0.1:{port}", f"{base}/"),
            (201, "Content-Location", "/items/7", f"{base}/items/7"),
            (302, "Location", "https://example.org/cb", "https://example.org/cb"),
            (302, "Location", "http://127.0.0.1:5173/x", "http://127.0.0.1:5173/x"),
            (302, "Location", "login", "login"),
            (302, "Location", "../x", "../x"),
            (302, "Location", f"{base}/login", f"{base}/login"),
        ] + [(st, "Location", "/a", f"{base}/a") for st in (301, 303, 307, 308)]
        for i, (st, name, value, want) in enumerate(cases):
            pages[f"/r{i}"] = (st, "text/plain", b"", {name: value})
        pages["/plain"] = (200, "text/plain", b"ok", {})
        for i, (st, name, value, want) in enumerate(cases):
            got, _, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/r{i}")
            hl = {k.lower(): v for k, v in h.items()}
            assert got == st and hl[name.lower()] == want, (value, got, h)
        got, _, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/plain")
        assert got == 200 and not {"location", "content-location"} & {k.lower() for k in h}
        # the prefix is the one the client sent, without /v1/chat-app when it left it out
        short = f"/conversations/{cid}/sandbox/ports/{port}"
        assert port_prefix(short + "/a/b?q=/sandbox/ports/1") == short
        assert port_prefix(base + "?x=1") == base and port_prefix(f"/conversations/{cid}") is None
    finally:
        app.shutdown()
        c.close()


def test_the_location_rewrite_alone():
    pre, port = "/v1/chat-app/conversations/c1/sandbox/ports/3000", 3000
    fix = lambda v: PV.rewrite_location(v, pre, port)               # noqa: E731
    assert fix(" /login ") == pre + "/login"
    assert fix("/") == pre + "/"
    assert fix("/\\evil.example/x") == pre + "/\\evil.example/x", "a backslash host stays a path"
    assert fix("HTTP://LOCALHOST:3000/x") == pre + "/x"
    assert fix("http://127.0.0.1:3000?q=1") == pre + "/?q=1"
    assert fix(pre) == pre and fix(pre + "?q=1") == pre + "?q=1"
    assert fix("/v1/chat-app/conversations/c1/sandbox/ports/30001") == pre + \
        "/v1/chat-app/conversations/c1/sandbox/ports/30001", "another port's prefix is a path"
    for same in ("", "//example.org/x", "http://127.0.0.1/x", "http://127.0.0.1:abc/x",
                 "https://localhost:3000/x", "http://localhost.example:3000/x", "#top", "?a=1",
                 "mailto:a@b"):
        assert fix(same) == same, same
    assert PV.rewrite_location("http://localhost/x", "/p", 80) == "/p/x", "the default port"


def test_the_port_proxys_websocket_is_untouched_by_the_rewrite():
    """SRV-631 changed the HTTP answer's headers only: an upgrade under the prefix still reaches
    the app's socket and echoes."""
    c = chat()
    app = app_server()
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        sock = ws_open(c.port, f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}/hmr?t=1")
        ws_send(sock, b"ping", op=1)
        assert ws_recv(sock) == b"echo:ping"
        sock.close()
        c.wait(lambda: c.svc.sandboxes.relays == 0, 10)
    finally:
        app.shutdown()
        c.close()


def test_the_view_route_serves_a_built_page_from_its_own_directory():
    """dist/index.html of `vite build` asks for /assets/index-x.js: on the view route that is the
    page's own directory, and its module may be read by the opaque preview."""
    c = chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("app/dist/index.html", b'<html><head><script type="module" crossorigin '
                                        b'src="/assets/index-x.js"></script><link rel="stylesheet" '
                                        b'crossorigin href="/assets/a.css"></head></html>')
        ws.write("app/dist/assets/index-x.js", b"document.body.dataset.ok = 1")
        view = f"/v1/chat-app/conversations/{cid}/sandbox/view/app/dist"
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/view/app/dist/index.html",
                             headers={"Origin": "null"})
        page = body if isinstance(body, bytes) else body.encode()
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null"
        assert f'src="{view}/assets/index-x.js"'.encode() in page
        assert f'href="{view}/assets/a.css"'.encode() in page
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/view/app/dist/assets/index-x.js",
                             headers={"Origin": "null"})
        assert st == 200 and h["Access-Control-Allow-Origin"] == "null"
        # the artifact of a built page is the page itself: it runs as a file
        a = c.ok("GET", f"/conversations/{cid}/artifacts/file%3A%2Fworkspace%2Fapp%2Fdist%2Findex.html")
        assert a["url"].endswith("/sandbox/view/app/dist/index.html") and "waiting" not in a
    finally:
        c.close()


# ---------------------------------------------------------------------- the target
def ws_with_vite_project() -> Workspace:
    root = tempfile.mkdtemp()
    ws = Workspace(root)
    ws.ensure()
    ws.write("pagoda-garden/package.json", json.dumps({
        "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
        "devDependencies": {"vite": "^5.4.0", "three": "^0.170.0"}}).encode())
    ws.write("pagoda-garden/index.html", b'<html><head><link rel="stylesheet" href="style.css">'
                                         b'</head><body><canvas class="webgl"></canvas>'
                                         b'<script type="module" src="/src/main.js"></script>'
                                         b'</body></html>')
    ws.write("pagoda-garden/src/main.js", b"import * as THREE from 'three';\n"
                                          b"import { CONFIG } from './config.js';\n")
    ws.write("pagoda-garden/dist/index.html", b'<script type="module" crossorigin '
                                              b'src="/assets/index-B.js"></script>')
    ws.write("pagoda-garden/dist/assets/index-B.js", b'const a=1;export{a};')
    ws.write("plain/index.html", b'<script type="module" src="app.js"></script>')
    ws.write("plain/app.js", b'import { x } from "./x.js";')
    ws.write("mapped/index.html", b'<script type="importmap">{"imports":{"three":"https://c/t.js"}}'
                                  b'</script><script type="module">import * as T from "three"'
                                  b'</script>')
    ws.write("ts/index.html", b'<script type="module" src="/src/main.ts"></script>')
    return ws


def test_a_bundler_projects_source_page_needs_its_dev_server_and_a_built_or_plain_page_does_not():
    ws = ws_with_vite_project()
    try:
        need = PV.needs_server(ws, "/workspace/pagoda-garden/index.html")
        assert need == {"reason": 'its script /src/main.js imports the package "three" by name',
                        "dir": "/workspace/pagoda-garden", "command": "npm run dev",
                        "bundler": "vite"}
        assert PV.needs_server(ws, "/workspace/pagoda-garden/dist/index.html") is None
        assert PV.needs_server(ws, "/workspace/plain/index.html") is None
        assert PV.needs_server(ws, "/workspace/mapped/index.html") is None, "an import map resolves it"
        ts = PV.needs_server(ws, "/workspace/ts/index.html")
        assert ts["reason"] == "its script /src/main.ts needs compiling" and ts["command"] is None
        assert PV.project_of(ws, "/workspace/pagoda-garden/src/main.js")["dir"] == \
            "/workspace/pagoda-garden"
        assert PV.project_of(ws, "/workspace/plain/index.html") is None
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_the_preview_picks_the_port_that_listens_or_the_projects_newest_server():
    vite = {"port": 5173, "cwd": "/workspace/pagoda-garden", "started": 200, "command": "vite"}
    other = {"port": 8000, "cwd": "/workspace/api", "started": 300, "command": "python3 -m http.server"}
    db = {"port": 5432, "cwd": "/", "started": 400, "http": False}
    assert PV.pick([vite, other], want=5173) is vite
    assert PV.pick([vite, other, db], want=4173) is other, "the newest server that answers HTTP"
    assert PV.pick([vite, other], want=4173, project="/workspace/pagoda-garden") is vite
    assert PV.pick([other], project="/workspace/pagoda-garden") is None
    assert PV.pick([], want=4173) is None and PV.pick(None) is None


def test_a_port_or_page_artifact_resolves_to_what_listens_or_waits():
    c = chat()
    try:
        cid = c.new()
        boxes = c.svc.sandboxes
        ws = c.svc.workspace(cid)
        src = ws_with_vite_project()
        for rel in ("pagoda-garden/package.json", "pagoda-garden/index.html",
                    "pagoda-garden/src/main.js"):
            ws.write(rel, src.read(rel))
        shutil.rmtree(src.root, ignore_errors=True)
        page = "file%3A%2Fworkspace%2Fpagoda-garden%2Findex.html"
        # the sandbox stopped: the pane waits and names the command
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A4173")
        assert "url" not in a and "the sandbox is stopped" in a["waiting"]["message"]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/{page}")
        assert a["waiting"]["command"] == "npm run dev" and \
            a["waiting"]["dir"] == "/workspace/pagoda-garden" and \
            "the source of a Vite project" in a["waiting"]["message"]
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        boxes.listen = [{"port": 5173, "addresses": ["::1"], "cwd": "/workspace/pagoda-garden",
                         "started": 9, "command": "node /workspace/pagoda-garden/node_modules/.bin/vite"}]
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A4173")
        assert a["url"].endswith("/sandbox/ports/5173/") and a["port"] == 5173
        assert a["note"] == ("Nothing listens on port 4173; this is port 5173 (node vite), the server "
                             "that runs in the sandbox.")
        a = c.ok("GET", f"/conversations/{cid}/artifacts/port%3A5173")
        assert a["url"].endswith("/sandbox/ports/5173/") and "note" not in a
        a = c.ok("GET", f"/conversations/{cid}/artifacts/{page}")
        assert a["url"].endswith("/sandbox/ports/5173/") and "it runs from its dev server" in a["note"]
        got = c.ok("GET", f"/conversations/{cid}/sandbox/ports")
        assert [p["port"] for p in got["ports"]] == [5173] and got["ports"][0]["http"] is True
    finally:
        c.close()


def test_the_pane_starts_the_projects_dev_server_from_its_package_json():
    c = chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("game/package.json", json.dumps({"scripts": {"dev": "vite --host"},
                                                  "devDependencies": {"vite": "5"}}).encode())
        ws.write("loose/a.txt", b"x")
        got = c.ok("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/workspace/game"})
        assert got == {"dir": "/workspace/game", "command": "npm run dev",
                       "log": "/workspace/.tandem/outputs/dev-server.log"}
        line = c.svc.sandboxes.ran[-1][-1]
        assert "cd /workspace/game && { [ -d node_modules ] || npm install; } && exec npm run dev" \
            in line and "nohup" in line and line.endswith("& echo started")
        st, body, _ = c.call("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/workspace/loose"})
        assert st == 400 and body["error"]["type"] == "no_dev_server"
        st, body, _ = c.call("POST", f"/conversations/{cid}/sandbox/serve", {"dir": "/etc"})
        assert st == 400
    finally:
        c.close()


# ---------------------------------------------------------------------- what present says
class Ctx:
    def __init__(self, svc, cid):
        self.svc, self.cid = svc, cid
        self.ws = svc.workspace(cid)
        self.panes = []
        self.run_id = "run-1"

    def present(self, pane):
        self.panes.append(pane)


def test_present_reports_the_server_and_what_the_users_pane_loaded():
    c = chat()
    app, _ = server({"/": (200, "text/html", VITE_INDEX, {}),
                     "/@vite/client": (200, "text/javascript", b"1", {})})
    port = app.server_address[1]
    try:
        cid = c.new()
        ctx = Ctx(c.svc, cid)
        # nothing listens: the pane opens and waits; the model hears it shows nothing
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert res.is_error and ctx.panes[-1]["ref"] == "port:4173"
        assert "shows nothing yet" in res.content and "do not tell the user it does" in res.content
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        c.svc.sandboxes.listen = [{"port": port, "cwd": "/workspace/app", "started": 5,
                                   "command": "node vite"}]
        # no browser has the chat open: the server's answer, and that no pane has loaded it
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert res.content.startswith(f"opened the preview pane on port {port} (node vite).")
        assert f"Nothing listens on port 4173; this is port {port}" in res.content
        assert "answered GET / 200 text/html; 1 of 3 scripts and style sheets answered, failing: " \
               "style.css 404" in res.content and res.is_error
        assert "No browser has this chat open" in res.content

        # a browser has it open: the pane's own report comes back
        conv = c.svc.conv(cid)

        class Sub:
            def send(self, *a):
                pass
        conv.fan.add(Sub())

        def report(errors, failed):
            time.sleep(0.3)
            c.ok("POST", f"/conversations/{cid}/preview/report", {
                "ref": "port:4173", "url": f"https://x/v1/chat-app/conversations/{cid}/sandbox/"
                                           f"ports/{port}/", "title": "Pagoda Garden",
                "errors": errors, "failed": failed, "elements": 3, "text": 0, "canvas": 1})
        threading.Thread(target=report, args=([], ["script https://x/src/main.js"]),
                         daemon=True).start()
        res = T._present(ctx, {"kind": "preview", "port": 4173})
        assert f'The user\'s pane loaded ports/{port}/ (title "Pagoda Garden"); 1 resource(s) ' \
               "failed: script https://x/src/main.js." in res.content and res.is_error
        assert "fix this before telling the user it does" in res.content
        threading.Thread(target=report, args=([], []), daemon=True).start()
        rep = c.svc.wait_report(cid, "port:4173", time.time(), 5)
        assert PV.report_words(rep, 1) == (
            f'The user\'s pane loaded ports/{port}/ (title "Pagoda Garden"); no errors, 3 elements, '
            "1 canvas.", False)
        assert PV.report_words(None, 0)[0].startswith("No browser has this chat open")
        assert PV.report_words(dict(rep, elements=0, canvas=0), 1)[1], "an empty page is a problem"
    finally:
        app.shutdown()
        c.close()


def test_present_of_a_static_page_needs_no_server():
    c = chat()
    try:
        cid = c.new()
        ctx = Ctx(c.svc, cid)
        ctx.ws.write("site/index.html", b"<h1>hi</h1>")
        res = T._present(ctx, {"kind": "preview", "path": "site/index.html"})
        assert not res.is_error and ctx.panes[-1] == {"kind": "preview",
                                                       "ref": "file:/workspace/site/index.html",
                                                       "title": None}
        assert res.content.startswith("opened the preview pane on /workspace/site/index.html.")
    finally:
        c.close()


# ---------------------------------------------------------------------- the shim
def test_the_shim_sends_runtime_urls_and_the_hot_reload_socket_to_the_ports_prefix():
    """The shim as a browser runs it (node, with the few globals it touches): fetch, XHR, a
    WebSocket and an image's src that start with `/` or name localhost:PORT go to the prefix;
    another host and an already prefixed URL stay."""
    node = shutil.which("node")
    if not node:
        print("no node here: the shim's run is skipped")
        return
    prefix = "/v1/chat-app/conversations/C1/sandbox/ports/5173"
    shim = PV.page_shim(prefix, 5173).decode()[len("<script>"):-len("</script>")]
    harness = r"""
const got = [];
globalThis.location = new URL('https://tandem.example/v1/chat-app/conversations/C1/sandbox/ports/5173/');
globalThis.window = globalThis;
globalThis.fetch = (u) => { got.push(['fetch', String(u)]); return Promise.resolve(); };
globalThis.Request = class { constructor(u) { this.url = u; } };
globalThis.XMLHttpRequest = class {}; XMLHttpRequest.prototype.open = function (m, u) { got.push(['xhr', u]); };
globalThis.WebSocket = class { constructor(u, p) { got.push(['ws', u, p]); } };
class El { setAttribute(n, v) { got.push(['attr', n, v]); } }
globalThis.Element = El;
globalThis.HTMLAnchorElement = class extends El {};
globalThis.HTMLImageElement = class extends El {};
Object.defineProperty(HTMLImageElement.prototype, 'src', { configurable: true, get() { return this._s; }, set(v) { this._s = v; got.push(['img', v]); } });
globalThis.addEventListener = () => {};
globalThis.document = { title: '', body: null, getElementsByTagName: () => [] };
globalThis.parent = { postMessage() {} };
""" + shim + r"""
fetch('/textures/a.jpg'); fetch('https://cdn.example/x.js'); fetch(location.href + 'data.json');
new XMLHttpRequest().open('GET', '/api/x?y=1');
new WebSocket('wss://tandem.example:/?token=T', 'vite-hmr');
new WebSocket('ws://localhost:5173/?token=T', 'vite-hmr');
new WebSocket('ws://localhost:9999/');
const img = new HTMLImageElement(); img.src = '/textures/b.png';
img.setAttribute('src', '/c.png'); new HTMLAnchorElement().setAttribute('href', '/about');
console.log(JSON.stringify(got));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
    try:
        out = subprocess.run([node, fh.name], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(fh.name)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    p = "https://tandem.example" + prefix
    assert got == [
        ["fetch", p + "/textures/a.jpg"], ["fetch", "https://cdn.example/x.js"],
        ["fetch", p + "/data.json"], ["xhr", p + "/api/x?y=1"],
        ["ws", "wss://tandem.example" + prefix + "/?token=T", "vite-hmr"],
        ["ws", "wss://tandem.example" + prefix + "/?token=T", "vite-hmr"],
        ["ws", "ws://localhost:9999/", None],
        ["img", p + "/textures/b.png"], ["attr", "src", p + "/c.png"], ["attr", "href", "/about"]]


# ---------------------------------------------------------------------- streaming (SRV-633, #86)
class LiveRelays(PortBoxes):
    """PortBoxes whose `relay.py connect` stand-ins run until the app closes or `end_exec` ends
    them, each with its own id, so a test sees which relay chatd ended and which still runs."""

    def __init__(self, data):
        super().__init__(data)
        self.live: dict[str, tuple] = {}
        self.ended: list[str] = []
        self.n = 0

    def open_exec(self, cid, cmd, *, tty=False, stdin=False, env=None, workdir="/workspace"):
        if cmd[-2:-1] != ["connect"]:
            return super().open_exec(cid, cmd, tty=tty, stdin=stdin, env=env, workdir=workdir)
        a, b = socket.socketpair()
        srv = socket.create_connection(("127.0.0.1", int(cmd[-1])))
        self.n += 1
        eid = f"exec-relay-{self.n}"
        self.live[eid] = (b, srv)

        def up():
            try:
                while True:
                    d = b.recv(65536)
                    if not d:
                        break
                    srv.sendall(d)
            except OSError:
                pass

        def down():
            try:
                while True:
                    d = srv.recv(65536)
                    if not d:
                        break
                    b.sendall(bytes([1, 0, 0, 0]) + len(d).to_bytes(4, "big") + d)
            except OSError:
                pass
            self._end(eid)
        threading.Thread(target=up, daemon=True).start()
        threading.Thread(target=down, daemon=True).start()
        return eid, a, b""

    def _end(self, eid):
        socks = self.live.pop(eid, None)
        for s in socks or ():
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def exit_code(self, exec_id):
        return exec_id in self.live, 0, None

    def end_exec(self, exec_id):
        if exec_id in self.live:
            self.ended.append(exec_id)
        self._end(exec_id)


def stream_chat():
    c = Chat()
    c.svc.sandboxes = LiveRelays(c.data)
    c.svc.sandboxes.on_change = c.svc._sandbox_changed
    c.svc.sandboxes.network_mode = c.svc._network_mode
    return c


def stream_app():
    """A dev server with the answers a preview app streams or holds: /events (SSE, one event every
    0.3 s; ?n=N ends after N, else until the client goes), /poll?s=S (a long-poll held S seconds),
    /page?kb=K (an HTML page of K KiB with an absolute script URL), /chunked.js (a module sent in
    chunks), /hang (never answers), /stall (Content-Length 1000, 10 bytes, then nothing).
    `state` counts the event streams still open."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit
    state = {"open": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                state["open"] += 1
                try:
                    i = 0
                    while "n" not in q or i < int(q["n"]):
                        self.wfile.write(f"data: {i} {time.monotonic():.3f}\n\n".encode())
                        self.wfile.flush()
                        i += 1
                        time.sleep(0.3)
                except OSError:
                    pass
                finally:
                    state["open"] -= 1
                self.close_connection = True
            elif u.path == "/poll":
                time.sleep(float(q.get("s", 1)))
                body = b'{"event":"done"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/page":
                body = (b'<html><head><title>t</title></head><body><script src="/app.js"></script>'
                        + b"x" * (int(q.get("kb", 1)) * 1024) + b"</body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i in range(0, len(body), 3000):
                    part = body[i:i + 3000]
                    self.wfile.write(b"%x;ext=1\r\n" % len(part) + part + b"\r\n")
                self.wfile.write(b"0\r\nX-Trailer: 1\r\n\r\n")
            elif u.path == "/chunked.js":
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in (b'import a fr', b'om "/src/a.js";\n', b"export default a;"):
                    self.wfile.write(b"%x\r\n" % len(part) + part + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
                self.wfile.write(b"0\r\n\r\n")
            elif u.path == "/hang":
                time.sleep(30)
            elif u.path == "/stall":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b"0123456789")
                self.wfile.flush()
                time.sleep(30)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def raw_get(c, path, timeout=10.0):
    """A GET to chatd on a raw socket, as a browser's EventSource sends it: (socket, file)."""
    s = socket.create_connection(("127.0.0.1", c.port), timeout=timeout)
    s.sendall((f"GET /v1/chat-app{path} HTTP/1.1\r\nHost: x\r\nX-Tandem-Forward: {KEY}\r\n"
               "Accept: text/event-stream\r\nOrigin: null\r\n\r\n").encode())
    return s, s.makefile("rb")


def read_head(f) -> tuple[int, dict]:
    status = int(f.readline().split()[1])
    h = {}
    while True:
        line = f.readline().decode("latin-1").strip()
        if not line:
            return status, h
        k, _, v = line.partition(":")
        h[k.strip().lower()] = v.strip()


def until(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_the_port_proxy_streams_server_sent_events_as_they_come():
    """#86: the proxy sent nothing until the app closed; an EventSource in the preview never saw
    an event. Each event now reaches the client when the app writes it."""
    c = stream_chat()
    app, _ = stream_app()
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        t0 = time.monotonic()
        s, f = raw_get(c, f"/conversations/{cid}/sandbox/ports/{port}/events?n=5")
        st, h = read_head(f)
        assert st == 200 and h["content-type"] == "text/event-stream", (st, h)
        assert "content-length" not in h and h["connection"] == "close"
        assert "connect-src" in h["content-security-policy"]
        assert h["access-control-allow-origin"] == "null"
        got = []
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                got.append(time.monotonic() - t0)
        s.close()
        assert len(got) == 5, got
        assert got[0] < 0.25, f"the first event came at {got[0]:.2f} s, not at once"
        assert got[-1] - got[0] > 0.9, f"the events came together: {got}"
        assert until(lambda: c.svc.sandboxes.box(cid).busy == 0)
    finally:
        app.shutdown()
        c.close()


def test_a_client_that_leaves_mid_stream_ends_the_relay_and_frees_the_sandbox():
    """#86: a browser that left was not noticed; the thread held the relay and box.busy forever,
    so the guard's idle stop never picked the sandbox."""
    c = stream_chat()
    app, state = stream_app()
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        boxes = c.svc.sandboxes
        s, f = raw_get(c, f"/conversations/{cid}/sandbox/ports/{port}/events")
        assert read_head(f)[0] == 200
        n = 0
        while n < 2:
            n += f.readline().startswith(b"data: ")
        assert boxes.box(cid).busy == 1 and len(boxes.live) == 1
        s.close()
        f.close()
        assert until(lambda: boxes.box(cid).busy == 0, 3), "busy stays above 0"
        assert boxes.ended == ["exec-relay-1"] and not boxes.live, (boxes.ended, boxes.live)
        assert until(lambda: state["open"] == 0, 3), "the app still streams to nobody"
    finally:
        app.shutdown()
        c.close()


def test_a_long_poll_answers_when_the_app_does_and_a_client_that_leaves_ends_it():
    c = stream_chat()
    app, _ = stream_app()
    port = app.server_address[1]
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        boxes = c.svc.sandboxes
        t0 = time.monotonic()
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/poll?s=1.2")
        assert st == 200 and body == {"event": "done"} and h["Content-Length"] == "16"
        assert 1.1 < time.monotonic() - t0 < 5
        # the client leaves while the app holds the request: noticed before the app answers
        s, f = raw_get(c, f"/conversations/{cid}/sandbox/ports/{port}/poll?s=20")
        assert until(lambda: boxes.box(cid).busy == 1)
        time.sleep(0.3)
        s.close()
        f.close()
        assert until(lambda: boxes.box(cid).busy == 0, 3)
        assert boxes.ended == ["exec-relay-2"], boxes.ended
    finally:
        app.shutdown()
        c.close()


def test_a_page_is_rewritten_under_the_cap_and_passes_unrewritten_over_it():
    """A page, script or style sheet is read whole for its rewrite up to REWRITE_LIMIT; a bigger
    one passes as the app sent it, so chatd never holds more than the cap."""
    c = stream_chat()
    app, _ = stream_app()
    port = app.server_address[1]
    cap = TM.REWRITE_LIMIT
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        base = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/page?kb=64")
        assert st == 200 and "X-Tandem-Rewrite" not in h
        assert body.startswith(b"<html><head><script>") and f'src="{base}/app.js"'.encode() in body
        assert int(h["Content-Length"]) == len(body) and body.endswith(b"x</body></html>")
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/chunked.js")
        assert body == f'import a from "{base}/src/a.js";\nexport default a;'.encode(), body
        TM.REWRITE_LIMIT = 16 * 1024
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/page?kb=64")
        assert st == 200 and h["X-Tandem-Rewrite"].startswith("skipped")
        assert body.startswith(b'<html><head><title>t</title></head><body><script src="/app.js">')
        page = (b'<html><head><title>t</title></head><body><script src="/app.js"></script>'
                + b"x" * 64 * 1024 + b"</body></html>")
        assert body == page and "Content-Length" not in h, "byte for byte as the app sent it"
        assert until(lambda: c.svc.sandboxes.box(cid).busy == 0)
    finally:
        TM.REWRITE_LIMIT = cap
        app.shutdown()
        c.close()


def test_an_app_that_stalls_gets_a_504_or_a_short_body_and_the_relay_ends():
    c = stream_chat()
    app, _ = stream_app()
    port = app.server_address[1]
    wait, idle = TM.HEAD_WAIT_S, TM.READ_IDLE_S
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        boxes = c.svc.sandboxes
        TM.HEAD_WAIT_S, TM.READ_IDLE_S = 0.6, 0.6
        t0 = time.monotonic()
        st, body, h = c.call("GET", f"/conversations/{cid}/sandbox/ports/{port}/hang")
        assert st == 504 and b"did not answer within 0.6 s" in body, (st, body)
        assert time.monotonic() - t0 < 3 and h["Content-Security-Policy"]
        assert until(lambda: boxes.box(cid).busy == 0) and boxes.ended == ["exec-relay-1"]
        s, f = raw_get(c, f"/conversations/{cid}/sandbox/ports/{port}/stall")
        st, h = read_head(f)
        assert st == 200 and h["content-length"] == "1000"
        assert f.read() == b"0123456789", "a sized body that stalls closes short"
        s.close()
        assert until(lambda: boxes.box(cid).busy == 0) and boxes.ended[-1] == "exec-relay-2"
    finally:
        TM.HEAD_WAIT_S, TM.READ_IDLE_S = wait, idle
        app.shutdown()
        c.close()


def test_a_port_lookup_that_does_not_end_is_ended():
    """`relay.py ports` and `probe` have a deadline (RELAY_OUT_S): past it the exec ends and the
    lookup answers None, as an unreadable one does."""
    c = chat()
    held = []

    def open_exec(cid, cmd, **kw):
        a, b = socket.socketpair()
        held.append(b)
        return "exec-ports-hang", a, b""
    ended = []
    c.svc.sandboxes.open_exec = open_exec
    c.svc.sandboxes.end_exec = ended.append
    limit = PV.RELAY_OUT_S
    try:
        cid = c.new()
        c.ok("POST", f"/conversations/{cid}/sandbox/start")
        PV.RELAY_OUT_S = 0.3
        assert c.svc.listening(cid) is None
        assert ended == ["exec-ports-hang"] and c.svc.sandboxes.box(cid).busy == 0
    finally:
        PV.RELAY_OUT_S = limit
        c.close()


def test_a_chunked_body_is_decoded_as_it_arrives():
    raw = b"4;x=y\r\nWiki\r\n5\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0\r\nT: 1\r\n\r\n"
    d, out = Dechunk(), b""
    for i in range(len(raw)):
        out += d.feed(raw[i:i + 1])
    assert out == b"Wikipedia in\r\n\r\nchunks." and d.done
    d = Dechunk()
    assert d.feed(raw) == b"Wikipedia in\r\n\r\nchunks." and d.done
    for bad in (b"-1\r\n", b"0x4\r\n", b"zz\r\n", b"1" * 5000):
        try:
            Dechunk().feed(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


if __name__ == "__main__":
    run_all(globals())
