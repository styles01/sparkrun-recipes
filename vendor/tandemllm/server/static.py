"""The dashboard's files at `/dashboard/`, served by the engine itself.

`npm run build` in `dashboard/` on the Mac writes `dashboard/dist/` -- one `index.html`, hashed
JS and CSS -- and that directory is committed, so the rsync deploy carries it and the box needs
no Node. The shell holds no data: every number in it comes from `/v1/dashboard/*`, which needs
the admin token or the session, so the files themselves are public.

    GET /dashboard            301 to /dashboard/
    GET /dashboard/           index.html, Cache-Control: no-cache
    GET /dashboard/assets/*   the file; immutable for a hashed name
    GET /dashboard/<route>    index.html again (the app routes on the client)

Fixed content types, `X-Content-Type-Options: nosniff`, the CSP of the dashboard design
(`default-src 'self'; connect-src 'self'; img-src 'self' data:`; with QSE_STATUS_PORT the status
responder's port on any host joins connect-src, OPS-175), and no path outside the
directory: `..`, an absolute path or a symlink out of it is a 404. With no `dist/` at all, a
one-paragraph placeholder says how to build it.
"""

from __future__ import annotations

import os
import re

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".map": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon",
         ".webp": "image/webp", ".woff2": "font/woff2", ".woff": "font/woff",
         ".txt": "text/plain; charset=utf-8", ".webmanifest": "application/manifest+json"}
CSP = "default-src 'self'; connect-src 'self'; img-src 'self' data:"
# `index.html` may carry small inline scripts (the dashboard sets its theme before the first paint);
# each is allowed by its own sha256, never by 'unsafe-inline'. Styles: Lit writes `style=`
# attributes from templates, which a style-src without 'unsafe-inline' would silently drop.
STYLE_SRC = "style-src 'self' 'unsafe-inline'"
_INLINE = re.compile(rb"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)
# a Vite asset name: `index-3f9a1c0e.js`, `app-Bx7_kQ2d.css`
HASHED = re.compile(r"-[A-Za-z0-9_-]{8,}\.[a-z0-9]+$")

PLACEHOLDER = b"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Engine dashboard</title>
</head><body style="font-family: system-ui, sans-serif; background: #0b0f17; color: #cfd6e4;
padding: 2rem; line-height: 1.5"><h1 style="font-size: 1.2rem">The dashboard is not built on this
server</h1><p>Build it on the Mac (<code>cd dashboard &amp;&amp; npm run build</code>), commit
<code>dashboard/dist/</code> and deploy. The API it reads is already here:
<code>/v1/dashboard/summary</code>, with the admin token.</p></body></html>
"""


def default_dir() -> str:
    """`dashboard/dist` in a checkout; `proxy/dashboard` beside this package in an installed
    wheel (`pip install tandemllm` puts the built page there)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.join(root, "dashboard", "dist")
    installed = os.path.join(root, "proxy", "dashboard")
    if not os.path.isdir(repo) and os.path.isfile(os.path.join(installed, "index.html")):
        return installed
    return repo


def resolve(root: str, rel: str) -> tuple[str | None, bool]:
    """`(file, fallback)`: the file under `root` for the URL path below /dashboard/, or index.html
    for a client-side route (`fallback`), or `(None, False)` for anything that leaves the root or
    does not exist."""
    from urllib.parse import unquote
    rel = unquote(rel.split("?", 1)[0].split("#", 1)[0])
    if "\x00" in rel or "\\" in rel:
        return None, False
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None, False
    real_root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(real_root, *parts)) if parts else real_root
    if path != real_root and not path.startswith(real_root + os.sep):
        return None, False
    if os.path.isdir(path):
        path = os.path.join(path, "index.html")
    if os.path.isfile(path):
        return path, False
    last = parts[-1] if parts else ""
    if "." not in last:                                  # a client route, not a missing asset
        index = os.path.join(real_root, "index.html")
        if os.path.isfile(index):
            return index, True
    return None, False


def serve(handler, raw_path: str, root: str | None = None) -> None:
    """Answer one GET under /dashboard on a `BaseHTTPRequestHandler`."""
    root = root or default_dir()
    path = raw_path.split("?", 1)[0]
    if path == "/dashboard":
        return _send(handler, 301, b"", "text/plain", extra=(("Location", "/dashboard/"),))
    if not os.path.isdir(root):
        return _send(handler, 200, PLACEHOLDER, TYPES[".html"], cache="no-cache",
                     csp=csp_for(PLACEHOLDER))
    file, _ = resolve(root, path[len("/dashboard/"):])
    if file is None:
        return _send(handler, 404, b"not found\n", TYPES[".txt"], cache="no-cache")
    ext = os.path.splitext(file)[1].lower()
    ctype = TYPES.get(ext, "application/octet-stream")
    if ext == ".html":
        cache = "no-cache"
    elif HASHED.search(os.path.basename(file)):
        cache = "public, max-age=31536000, immutable"
    else:
        cache = "no-cache"
    with open(file, "rb") as f:
        raw = f.read()
    return _send(handler, 200, raw, ctype, cache=cache,
                 csp=csp_for(raw) if ext == ".html" else CSP)


def csp_for(html: bytes) -> str:
    """The CSP for one HTML page: the design's policy, its inline scripts by hash, styles inline."""
    import base64
    import hashlib
    hashes = [f"'sha256-{base64.b64encode(hashlib.sha256(m).digest()).decode()}'"
              for m in _INLINE.findall(html)]
    return f"{policy()}; script-src 'self'{''.join(' ' + h for h in hashes)}; {STYLE_SRC}"


def policy() -> str:
    """The design's policy; with a status responder (QSE_STATUS_PORT, OPS-175) the page may also ask
    it, on any host at that port, for the restart while this server is away (VIS-209)."""
    port = os.environ.get("QSE_STATUS_PORT") or ""
    if not port.isdigit():
        return CSP
    return CSP.replace("connect-src 'self'", f"connect-src 'self' http://*:{port}")


def _send(handler, code: int, raw: bytes, ctype: str, cache: str | None = None,
          extra: tuple = (), csp: str = CSP) -> None:
    try:
        handler.send_response(code)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(raw)))
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Content-Security-Policy", csp)
        if cache:
            handler.send_header("Cache-Control", cache)
        for k, v in extra:
            handler.send_header(k, v)
        handler.end_headers()
        handler.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True
