"""What the preview pane shows, and what loads in it (SRV-511). Mixed into `chatd.service.Service`.

    GET  /conversations/{id}/sandbox/ports           {ports: [{port, addresses, command, cwd,
                                                     started, http}]}: what listens in the sandbox
    POST /conversations/{id}/sandbox/serve {dir}     start the dev server of the project in `dir`
                                                     (its package.json says which); 202
    POST /conversations/{id}/preview/report {ref,..} what the user's preview pane loaded (the page's
                                                     own report, passed on by the dashboard)

THE TARGET. A preview of `port:N` shows port N when something listens there; when nothing does, the
server that listens (the newest one) with a note that says so; when nothing listens at all, the pane
waits and opens the server once one listens. A page that cannot run as a file (a bundler project's
source index.html: `<script type="module" src="/src/main.js">` with imports of packages by name,
or a .ts/.jsx/.vue entry) shows the project's running dev server, or waits for it and names the
command that starts it. Live (2026-10-16) the model presented port 4173 after `vite preview` had
gone, and the user's index.html preview of a Vite project was a blank page.

THE PAGE. The preview's iframe has an opaque origin (sandbox without allow-same-origin, CONTRACT 5),
so its module scripts and `crossorigin` style sheets are CORS requests from `Origin: null`: every
answer of the port proxy and the view route says `Access-Control-Allow-Origin: null` to them (live
2026-10-16 WebKit: "Origin null is not allowed by Access-Control-Allow-Origin" for /src/main.js,
/@vite/client and dist/assets/*.css, a white or dark pane). A page also gets a small script first in
its head (`page_shim`): URLs the app builds at run time (fetch, XHR, WebSocket, EventSource, an
image's src) that start with `/` or name localhost:PORT go to the port's prefix, so a dev server's
hot-reload socket and a texture at "/textures/a.jpg" load; and it posts what loaded (errors, failed
resources, the title) to the dashboard, which passes it here for `present`.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from chatd.workspace import PathError

#: the dev servers a package.json names (its devDependencies or dependencies)
BUNDLERS = ("vite", "next", "react-scripts", "@sveltejs/kit", "astro", "nuxt", "parcel",
            "webpack-dev-server", "@angular/cli", "@vue/cli-service", "snowpack")
SCRIPT_EXTS = (".ts", ".tsx", ".jsx", ".vue", ".svelte", ".mts")
REPORT_WAIT_S = 15.0
PAGE_LIMIT = 512 * 1024
#: a port lookup or probe (relay.py ports / probe): its longest run and its largest output
RELAY_OUT_S = 60.0
RELAY_OUT_LIMIT = 4 * 1024 * 1024

_MODULE = re.compile(rb"<script\b([^>]*)>(.*?)</script>", re.I | re.S)
_SRC = re.compile(rb"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)
_TYPE_MODULE = re.compile(rb"""\btype\s*=\s*["']?module\b""", re.I)
_IMPORTMAP = re.compile(rb"""<script\b[^>]*\btype\s*=\s*["']?importmap""", re.I)
#: an import of a package by its name (`from "three"`, `import "lil-gui"`): only a bundler or an
#: import map resolves it
_BARE = re.compile(rb"""(?:^|[;\s}])(?:import\s*(?:[\w*{}\s,$]+\s*from\s*)?|export\s*[\w*{}\s,$]+"""
                   rb"""\s*from\s*)["'](?![./]|https?:|data:|blob:)([^"']+)["']""", re.M)
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_HEAD = re.compile(rb"<head\b[^>]*>", re.I)
_HTML = re.compile(rb"<html\b[^>]*>", re.I)
_ATTR_ABS = re.compile(rb"""(\s(?:src|href|action|poster))=(["'])/(?![/\\])""", re.I)
#: a module's absolute imports (`from "/src/a.js"`, `import "/@vite/client"`, `import("/x")`): a
#: dev server that serves ES modules (Vite) writes them so, and the browser asked the dashboard's
#: origin for them; live (2026-10-16) a Vite app's index came through and its 18 modules were 404
_JS_ABS = re.compile(rb"""((?:\bfrom|\bimport)\s*\(?\s*)(["'])/(?![/*])""")
_CSS_ABS = re.compile(rb"""(url\(\s*["']?)/(?![/*])""")
#: Vite's client builds hot updates as `base + path` (`import(base + "src/a.js?t=1")`)
_VITE_BASE = re.compile(rb"""(const base(?:\$1)? = )"/"( \|\|)""")


# ---------------------------------------------------------------------- the page
def cors(origin: str | None) -> tuple:
    """The CORS headers of a preview answer: the preview's opaque origin may read it."""
    return (("Access-Control-Allow-Origin", "null"), ("Vary", "Origin")) \
        if (origin or "").strip() == "null" else (("Vary", "Origin"),)


def page_shim(prefix: str, port: int = 0) -> bytes:
    """The script that goes first into a previewed page's head (see the module's doc). `prefix`:
    the port's or the page directory's address without the trailing slash; `port`: the server's
    port in the sandbox (0 for a file)."""
    js = ("(function(){var P=%s,L=%d,H=location.host;"
          "function fix(u){if(u==null)return u;var x;try{x=new URL(String(u),location.href)}"
          "catch(e){return u}var ws=x.protocol==='ws:'||x.protocol==='wss:';"
          "var dp=x.protocol==='https:'||x.protocol==='wss:'?443:80;"
          "var loc=L&&/^(localhost|127\\.0\\.0\\.1|\\[::1\\])$/.test(x.hostname)&&+(x.port||dp)===L;"
          "if(!loc&&x.host!==H)return u;"
          "if(!loc&&(x.pathname===P||x.pathname.indexOf(P+'/')===0))return u;"
          "if(x.pathname!==P&&x.pathname.indexOf(P+'/')!==0)x.pathname=P+x.pathname;"
          "var h=location.protocol==='https:';x.protocol=ws?(h?'wss:':'ws:'):location.protocol;"
          "x.hostname=location.hostname;x.port=location.port;return x.href}"
          "var F=window.fetch;if(F)window.fetch=function(i,o){if(typeof i==='string'||i instanceof URL)"
          "i=fix(i);else if(i&&i.url){var n=fix(i.url);if(n!==i.url)i=new Request(n,i)}"
          "return F.call(this,i,o)};"
          "var O=XMLHttpRequest.prototype.open;XMLHttpRequest.prototype.open=function(m,u){"
          "var a=[].slice.call(arguments);a[1]=fix(u);return O.apply(this,a)};"
          "['WebSocket','EventSource','Worker','SharedWorker'].forEach(function(n){var C=window[n];"
          "if(C)window[n]=new Proxy(C,{construct:function(t,a,nt){a=[].slice.call(a);a[0]=fix(a[0]);"
          "return Reflect.construct(t,a,nt)}})});"
          "[['HTMLImageElement','src'],['HTMLScriptElement','src'],['HTMLLinkElement','href'],"
          "['HTMLMediaElement','src'],['HTMLSourceElement','src'],['HTMLIFrameElement','src']]"
          ".forEach(function(p){var C=window[p[0]];if(!C)return;"
          "var d=Object.getOwnPropertyDescriptor(C.prototype,p[1]);if(!d||!d.set)return;"
          "Object.defineProperty(C.prototype,p[1],{configurable:true,enumerable:d.enumerable,"
          "get:d.get,set:function(v){d.set.call(this,fix(v))}})});"
          "var SA=Element.prototype.setAttribute;Element.prototype.setAttribute=function(n,v){"
          "if((n==='src'||n==='href')&&typeof v==='string'&&v.charAt(0)==='/'&&"
          "!(this instanceof HTMLAnchorElement))v=fix(v);return SA.call(this,n,v)};"
          "var E=[],B=[],sent=0,t=0;function q(){if(sent&&!t)t=setTimeout(send,1000)}"
          "function add(a,v){if(a.length<20)a.push(String(v).slice(0,300));q()}"
          "addEventListener('error',function(e){var el=e.target;if(el&&el!==window&&(el.src||el.href))"
          "add(B,(el.tagName||'').toLowerCase()+' '+(el.src||el.href));else add(E,(e.message||'error')"
          "+(e.filename?' ('+e.filename.split('/').pop()+':'+e.lineno+')':''))},true);"
          "addEventListener('unhandledrejection',function(e){var r=e.reason;"
          "add(E,'unhandled rejection: '+(r&&r.message||r))});"
          "var CE=console.error;console.error=function(){try{add(E,[].map.call(arguments,"
          "function(a){return a&&a.message||String(a)}).join(' '))}catch(x){}"
          "return CE.apply(this,arguments)};"
          "function send(){t=0;var b=document.body;try{parent.postMessage({tandemPreview:1,"
          "url:location.href,title:document.title,errors:E,failed:B,"
          "elements:b?b.getElementsByTagName('*').length:0,text:b?(b.innerText||'').trim().length:0,"
          "canvas:document.getElementsByTagName('canvas').length},'*')}catch(x){}}"
          "addEventListener('load',function(){setTimeout(function(){sent=1;send()},1500)})"
          "})();") % (json.dumps(prefix), port)
    return b"<script>" + js.encode() + b"</script>"


def rewrite_page(payload: bytes, prefix: str, port: int = 0) -> bytes:
    """A page's absolute `/` URLs (src, href, action, poster; inline imports and url(/...)) to
    `prefix`, and the shim first in its head."""
    pre = prefix.rstrip("/").encode() + b"/"
    payload = _ATTR_ABS.sub(lambda m: m.group(1) + b"=" + m.group(2) + pre, payload)
    payload = _JS_ABS.sub(lambda m: m.group(1) + m.group(2) + pre, payload)
    payload = _CSS_ABS.sub(lambda m: m.group(1) + pre, payload)
    shim = page_shim(prefix.rstrip("/"), port)
    m = _HEAD.search(payload) or _HTML.search(payload)
    return payload[:m.end()] + shim + payload[m.end():] if m else shim + payload


def rewrite_script(payload: bytes, prefix: str, path: str = "") -> bytes:
    pre = prefix.rstrip("/").encode() + b"/"
    payload = _JS_ABS.sub(lambda m: m.group(1) + m.group(2) + pre, payload)
    if path.split("?")[0] == "/@vite/client":
        payload = _VITE_BASE.sub(lambda m: m.group(1) + json.dumps(pre.decode()).encode()
                                 + m.group(2), payload)
    return payload


def rewrite_css(payload: bytes, prefix: str) -> bytes:
    pre = prefix.rstrip("/").encode() + b"/"
    return _CSS_ABS.sub(lambda m: m.group(1) + pre, payload)


def rewrite_location(value: str, prefix: str, port: int) -> str:
    """A `Location` or `Content-Location` from the app on `port` (SRV-631): an absolute path
    (`/login`) or a URL on the app's own address (`http://127.0.0.1:<port>/login`, localhost, [::1],
    or `//` before one of them) becomes `prefix` + its path, query and fragment. A browser follows
    `/login` to chatd's origin, out of the preview, and cannot reach 127.0.0.1 at all. An external
    URL, another port, a relative value (it resolves inside the prefix) and one already under
    `prefix` stay as they are."""
    v = value.strip()
    pre = prefix.rstrip("/")
    if v.startswith("/") and not v.startswith("//"):
        rest = v
    else:
        try:
            u = urlsplit(v)
            own = (u.scheme in ("http", "") and bool(u.netloc) and u.hostname in _LOOPBACK
                   and (u.port or 80) == port)
        except ValueError:                                     # a port that is not a number
            return value
        if not own:
            return value
        rest = urlunsplit(("", "", u.path or "/", u.query, u.fragment))
    if rest == pre or rest.startswith((pre + "/", pre + "?", pre + "#")):
        return rest
    return pre + rest


# ---------------------------------------------------------------------- the target
def project_of(ws, path: str) -> dict | None:
    """The nearest package.json at or above `path`'s directory, inside /workspace:
    {dir, command, bundler}. `command` starts its dev server (`npm run dev`, `npm start`, or the
    bundler's own command); None when it names none."""
    d = posixpath.dirname(ws.show(ws.resolve(path)))
    while True:
        try:
            pkg = json.loads(ws.read(posixpath.join(d, "package.json"), limit=PAGE_LIMIT) or b"{}")
        except (OSError, ValueError, PathError):
            pkg = None
        if isinstance(pkg, dict):
            deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
            bundler = next((b for b in BUNDLERS if b in deps), None)
            scripts = pkg.get("scripts") or {}
            pm = "pnpm" if _exists(ws, posixpath.join(d, "pnpm-lock.yaml")) else \
                "yarn" if _exists(ws, posixpath.join(d, "yarn.lock")) else "npm"
            command = (f"{pm} run dev" if "dev" in scripts else
                       f"{pm} start" if "start" in scripts else
                       "npx vite" if bundler == "vite" else None)
            return {"dir": d, "command": command, "bundler": bundler}
        if d in ("/workspace", "/") or not d.startswith("/workspace"):
            return None
        d = posixpath.dirname(d)


def _exists(ws, path: str) -> bool:
    try:
        ws.stat(path)
        return True
    except (OSError, PathError):
        return False


#: a relative import inside an inline module (`from "./scene.js"`, `import("./a.js")`)
_REL_IMPORT = re.compile(rb"""(?:\bfrom\s*|\bimport\s*\(?\s*)["']\.{1,2}/""")


def file_blocks_modules(ws, path: str) -> bool:
    """Whether the page at `path` loads a module from its own folder (a `<script type="module"
    src>` with a relative or root path, or an inline module that imports one): Chromium refuses
    module scripts from file:// (origin null, CORS), so the page runs only over HTTP."""
    try:
        doc = ws.read(path, limit=PAGE_LIMIT)
    except (OSError, PathError):
        return False
    for attrs, body in _MODULE.findall(doc):
        if not _TYPE_MODULE.search(attrs):
            continue
        m = _SRC.search(attrs)
        if m:
            src = (m.group(1) or m.group(2) or m.group(3) or b"").decode("utf-8", "replace")
            if not re.match(r"^[a-z]+:|^//", src):
                return True
        elif _REL_IMPORT.search(body):
            return True
    return False


def server_for(ports: list[dict] | None, path: str) -> tuple[dict, str] | None:
    """The newest HTTP server in the sandbox that serves `path`: its working directory is the
    page's folder or above it. `(row, the page's URL path)`; None without one."""
    best = None
    for p in ports or []:
        cwd = (p.get("cwd") or "").rstrip("/")
        if not p.get("http", True) or not cwd or not path.startswith(cwd + "/"):
            continue
        if best is None or (p.get("started") or 0, p.get("port") or 0) > \
                (best.get("started") or 0, best.get("port") or 0):
            best = p
    return (best, path[len((best.get("cwd") or "").rstrip("/")) + 1:]) if best else None


def needs_server(ws, path: str) -> dict | None:
    """Why the page at `path` cannot run as a file, with its project: {reason, dir, command,
    bundler}; None when it can. It cannot when a module script is a .ts/.jsx/.vue/... file or
    imports a package by name without an import map: only a dev server (a bundler) resolves it."""
    try:
        doc = ws.read(path, limit=PAGE_LIMIT)
    except (OSError, PathError):
        return None
    if _IMPORTMAP.search(doc):
        return None
    page_dir = posixpath.dirname(ws.show(ws.resolve(path)))
    proj = project_of(ws, path)
    root = proj["dir"] if proj and proj.get("bundler") else page_dir
    reason = None
    for attrs, body in _MODULE.findall(doc):
        if not _TYPE_MODULE.search(attrs):
            continue
        m = _SRC.search(attrs)
        if not m:
            bare = _BARE.search(body)
            if bare:
                reason = f'an inline module imports the package "{bare.group(1).decode()}" by name'
                break
            continue
        src = (m.group(1) or m.group(2) or m.group(3) or b"").decode("utf-8", "replace")
        if re.match(r"^[a-z]+:|^//", src):
            continue
        clean = src.split("?")[0].split("#")[0]
        if clean.endswith(SCRIPT_EXTS):
            reason = f"its script {src} needs compiling"
            break
        target = posixpath.normpath(posixpath.join(root if clean.startswith("/") else page_dir,
                                                   clean.lstrip("/")))
        try:
            code = ws.read(target, limit=PAGE_LIMIT)
        except (OSError, PathError):
            continue
        bare = _BARE.search(code)
        if bare:
            reason = f'its script {src} imports the package "{bare.group(1).decode()}" by name'
            break
    if reason is None:
        return None
    return {"reason": reason, "dir": proj["dir"] if proj else page_dir,
            "command": proj["command"] if proj else None,
            "bundler": proj["bundler"] if proj else None}


def pick(ports: list[dict] | None, want: int | None = None, project: str | None = None
         ) -> dict | None:
    """The server a preview shows: port `want` when it listens; else the newest server started in
    `project` (its directory or below); else, without a project, the newest server that answers
    HTTP. None when there is none."""
    rows = [p for p in ports or [] if p.get("http", True)]
    if want:
        hit = next((p for p in ports or [] if p.get("port") == want), None)
        if hit:
            return hit
    if project:
        rows = [p for p in rows if (p.get("cwd") or "") == project or
                (p.get("cwd") or "").startswith(project.rstrip("/") + "/")]
    return max(rows, key=lambda p: (p.get("started") or 0, p.get("port") or 0), default=None)


def server_words(p: dict) -> str:
    """`port 5173 (node vite)`: the command with its paths cut to their last part."""
    cmd = " ".join(posixpath.basename(w) if "/" in w else w
                   for w in (p.get("command") or "").split())[:80]
    return f"port {p['port']}" + (f" ({cmd})" if cmd else "")


class PreviewMixin:
    def _pv(self) -> dict:
        if not hasattr(self, "_preview"):
            self._preview = {"lock": threading.Condition(), "reports": {}}
        return self._preview

    # ------------------------------------------------------------------ what listens
    def _relay_out(self, cid: str, *args: str) -> bytes:
        """What `relay.py <args>` (ports, probe) writes, within RELAY_OUT_S and RELAY_OUT_LIMIT
        (#86); past either the exec is ended and OSError says why."""
        from chatd.sandbox import ExecReader, relay_cmd
        box = self.sandboxes.box(cid)
        exec_id, sock, first = self.sandboxes.open_exec(cid, relay_cmd(*args))
        with box.lock:
            box.busy += 1
        reader = ExecReader(sock, first)
        try:
            out = bytearray()
            deadline = time.monotonic() + RELAY_OUT_S
            while True:
                d = reader.read(max(0.0, deadline - time.monotonic()))
                if not d:
                    return bytes(out)
                out += d
                if len(out) > RELAY_OUT_LIMIT:
                    raise OSError(f"relay.py {args[0]} wrote over {RELAY_OUT_LIMIT >> 20} MiB")
        finally:
            if not reader.ended:
                self.sandboxes.end_exec(exec_id)
            with box.lock:
                box.busy -= 1
            try:
                sock.close()
            except OSError:
                pass

    def listening(self, cid: str, http: bool = False) -> list[dict] | None:
        """What listens in the chat's sandbox ([] when it is not running; None when it could not
        be read). `http`: each row says whether the server answered an HTTP request."""
        from chatd.sandbox import Unavailable
        if self.sandboxes.box(cid).state != "running":
            return []
        try:
            raw = self._relay_out(cid, "ports", *(["--http"] if http else []))
            got = json.loads(raw.strip().splitlines()[-1]) if raw.strip() else None
        except (Unavailable, OSError, ValueError, IndexError):
            return None
        return got if isinstance(got, list) else None

    def probe(self, cid: str, port: int, path: str = "/") -> dict | None:
        from chatd.sandbox import Unavailable
        try:
            raw = self._relay_out(cid, "probe", str(port), path or "/")
            got = json.loads(raw.strip().splitlines()[-1])
        except (Unavailable, OSError, ValueError, IndexError):
            return None
        return got if isinstance(got, dict) else None

    def sandbox_ports(self, cid: str) -> dict:
        self.conv(cid)
        return {"ports": self.listening(cid, http=True) or []}

    # ------------------------------------------------------------------ the target
    def port_url(self, cid: str, port: int) -> str:
        return f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}/"

    def preview_of_port(self, cid: str, port: int, ports=None) -> dict:
        """The artifact of `port:N` (see the module's doc)."""
        ports = self.listening(cid, http=True) if ports is None else ports
        row = pick(ports, want=port)
        base = {"kind": "preview", "content": "", "mime": "text/html", "title": f"port {port}"}
        if ports is None:                    # what listens could not be read: the port as asked
            return {**base, "url": self.port_url(cid, port), "port": port}
        if row and row["port"] == port:
            return {**base, "url": self.port_url(cid, port), "port": port,
                    "server": row.get("command") or ""}
        if row:
            return {**base, "url": self.port_url(cid, row["port"]), "port": row["port"],
                    "title": f"port {row['port']}", "server": row.get("command") or "",
                    "note": f"Nothing listens on port {port}; this is {server_words(row)}, the "
                            f"server that runs in the sandbox."}
        stopped = self.sandboxes.box(cid).state != "running"
        return {**base, "waiting": {
            "port": port, "dir": None, "command": None,
            "message": (f"Nothing listens on port {port}" +
                        (": the sandbox is stopped." if stopped else " in the sandbox.") +
                        " Start the server; this pane opens it when it listens.")}}

    def preview_of_page(self, cid: str, path: str, need: dict) -> dict:
        """The artifact of a page that needs its dev server: the project's running server, or a
        pane that waits for it."""
        title = posixpath.basename(path)
        row = pick(self.listening(cid, http=True), project=need["dir"])
        what = f"the source of a {need['bundler'].capitalize()} project" if need.get("bundler") \
            else "a page that does not run as a file"
        if row:
            return {"kind": "preview", "content": "", "mime": "text/html", "title": title,
                    "url": self.port_url(cid, row["port"]), "port": row["port"],
                    "server": row.get("command") or "",
                    "note": f"{title} is {what}; it runs from its dev server: this is "
                            f"{server_words(row)}."}
        cmd = need.get("command")
        return {"kind": "preview", "content": "", "mime": "text/html", "title": title,
                "waiting": {"port": None, "dir": need["dir"], "command": cmd,
                            "message": f"{title} is {what} ({need['reason']}): it runs only from "
                                       f"its dev server." + (f" Start it with `{cmd}` in "
                                                            f"{need['dir']}; " if cmd else " ") +
                                       "this pane opens it when it listens."}}

    def serve(self, cid: str, body: dict) -> dict:
        """Start the dev server of the project in `dir`, in the background (its output in
        /workspace/.tandem/outputs/dev-server.log), installing its packages first when
        node_modules is missing. The command comes from its package.json, never from the client."""
        from chatd.sandbox import Unavailable
        self.conv(cid)
        ws = self.workspace(cid)
        try:
            d = ws.show(ws.resolve(str(body.get("dir") or ""), must_exist=True))
            proj = project_of(ws, posixpath.join(d, "package.json"))
        except PathError as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None
        if not proj or not proj.get("command") or proj["dir"] != d:
            raise self._api_error(400, "no_dev_server",
                                  f"{d} has no package.json with a dev or start script")
        if self.sandboxes.box(cid).state != "running":
            why = self.sandboxes.admit()
            if why:
                raise self._api_error(503, "waiting_memory", why)
        log = "/workspace/.tandem/outputs/dev-server.log"
        inner = f"cd {shlex.quote(d)} && {{ [ -d node_modules ] || npm install; }} && " \
                f"exec {proj['command']}"
        cmd = (f"mkdir -p /workspace/.tandem/outputs && nohup bash -lc {shlex.quote(inner)} "
               f"> {log} 2>&1 & echo started")
        try:
            self.sandboxes.ensure(cid)
            code, _ = self.sandboxes.run(cid, ["bash", "-lc", cmd], timeout=30)
        except Unavailable as exc:
            raise self._api_error(503, "sandbox_unavailable", str(exc)) from None
        if code:
            raise self._api_error(500, "serve_failed", f"the command could not start (exit {code})")
        self.log(f"[preview] {cid}: started {proj['command']} in {d}")
        return {"dir": d, "command": proj["command"], "log": log}

    # ------------------------------------------------------------------ the pane's report
    def preview_report(self, cid: str, body: dict) -> None:
        self.conv(cid)

        def strs(v, n=20):
            return [str(x)[:300] for x in v[:n]] if isinstance(v, list) else []
        rep = {"at": time.time(), "ref": str(body.get("ref") or "")[:300],
               "url": str(body.get("url") or "")[:500], "title": str(body.get("title") or "")[:200],
               "errors": strs(body.get("errors")), "failed": strs(body.get("failed")),
               "elements": int(body.get("elements") or 0), "text": int(body.get("text") or 0),
               "canvas": int(body.get("canvas") or 0),
               "viewer": str(body.get("viewer") or "")[:80]}
        pv = self._pv()
        with pv["lock"]:
            pv["reports"][cid] = rep
            pv["lock"].notify_all()

    def wait_report(self, cid: str, ref: str, since: float, timeout: float) -> dict | None:
        """The first report for `ref` that came after `since`, waiting up to `timeout` s."""
        pv = self._pv()
        end = time.monotonic() + timeout
        with pv["lock"]:
            while True:
                rep = pv["reports"].get(cid)
                if rep and rep["at"] >= since and rep["ref"] == ref:
                    return rep
                left = end - time.monotonic()
                if left <= 0:
                    return None
                pv["lock"].wait(left)


def report_words(rep: dict | None, viewers: int) -> tuple[str, bool]:
    """(what the user's pane showed, whether it showed a problem), for `present`'s result."""
    if rep is None:
        if not viewers:
            return ("No browser has this chat open, so no pane has loaded it yet; render_page shows "
                    "the page and its console in the sandbox.", False)
        return ("The user's pane sent no report within "
                f"{REPORT_WAIT_S:g} s (it may still be loading, or be closed).", False)
    bad = rep["errors"] or rep["failed"]
    words = f"The user's pane loaded {rep['url'].split('/sandbox/')[-1] or 'the page'}"
    words += f' (title "{rep["title"]}")' if rep["title"] else ""
    if rep["failed"]:
        words += f"; {len(rep['failed'])} resource(s) failed: " + "; ".join(rep["failed"][:6])
    if rep["errors"]:
        words += f"; {len(rep['errors'])} console error(s): " + " | ".join(rep["errors"][:6])
    if not bad:
        empty = rep["elements"] == 0 and rep["text"] == 0 and rep["canvas"] == 0
        words += ("; the page is EMPTY (no elements, no text)" if empty else
                  f"; no errors, {rep['elements']} element{'' if rep['elements'] == 1 else 's'}" +
                  (f", {rep['canvas']} canvas" if rep["canvas"] else ""))
        bad = empty
    return words + ".", bool(bad)


def probe_words(pr: dict | None) -> tuple[str, bool]:
    if pr is None:
        return "The server could not be probed.", False
    if not pr.get("status"):
        return f"The server did not answer GET {pr.get('path')}: {pr.get('type')}.", True
    words = f"The server answered GET {pr['path']} {pr['status']} {pr.get('type') or ''}".rstrip()
    words += f', title "{pr["title"]}"' if pr.get("title") else ""
    assets = pr.get("assets") or []
    broken = [a for a in assets if not 200 <= int(a.get("status") or 0) < 400]
    if assets:
        words += (f"; {len(assets) - len(broken)} of {len(assets)} scripts and style sheets "
                  f"answered" + ("" if not broken else ", failing: " + ", ".join(
                      f"{a['url']} {a['status'] or a['type']}" for a in broken[:6])))
    return words + ".", pr["status"] >= 400 or bool(broken)
