"""The workspace API the panes read (SRV-357, Chat 5): the sandbox's state, the files, a run's
diff and its undo, artifacts and the preview. Mixed into `chatd.service.Service`.

    GET    /conversations/{id}/sandbox                    SandboxStatus
    POST   /conversations/{id}/sandbox/{start|stop|reset} SandboxStatus (reset: a new container,
                                                          the files stay)
    GET    /conversations/{id}/sandbox/files?path=        SandboxFile[] (one directory)
    GET    /conversations/{id}/sandbox/file?path=         {content, mime, bytes, etag}; ?raw=1 bytes
    PUT    /conversations/{id}/sandbox/file?path=         If-Match: etag, 412 when it moved
    DELETE /conversations/{id}/sandbox/file?path=
    POST   /conversations/{id}/sandbox/upload?dir=        multipart files, 50 MB each
    GET    /conversations/{id}/sandbox/diff?run_id=       {files: [{path, status, additions,
                                                          deletions, patch}]}
    POST   /conversations/{id}/sandbox/undo {run_id}      the run's files back as they were
    GET    /conversations/{id}/artifacts/{ref}            what a pane shows: `file:<path>`,
                                                          `diff:<run>`, `port:<n>` (a preview's
                                                          target: chatd/preview.py)
    GET    /conversations/{id}/sandbox/view/<path>        a workspace file for the preview iframe
                                                          (the preview CSP, CONTRACT 5) or a media
                                                          player: streamed, with byte ranges; a
                                                          page whole, its absolute URLs to its own
                                                          directory, with the preview's shim

Paths are resolved by chatd/workspace.py (no `..`, nothing outside /workspace, no symlink on the
way). Files are read on the host; nothing here needs the container running.
"""

from __future__ import annotations

import mimetypes
import os
import re
import stat
import urllib.parse

from chatd.ids import ulid
from chatd.workspace import ChangedError, PathError

# connect-src names http: and ws: too: in the preview's opaque origin WebKit does not match 'self'
# for a fetch or a socket (scripts and images it does), so on the engine's plain http port a page's
# fetch("data.json") and a dev server's hot-reload socket were refused (2026-10-16, SRV-511)
PREVIEW_CSP = ("default-src 'self' data: blob: https:; script-src 'self' 'unsafe-inline' "
               "'unsafe-eval' https: blob:; style-src 'self' 'unsafe-inline' https:; "
               "connect-src 'self' https: wss: http: ws:; frame-ancestors 'self'")
# VIS-159: video, audio, PDFs and other binaries play or download from the view route; their
# answer never runs a script. A PDF keeps `object-src 'self'` for the browser's own viewer
MEDIA_CSP = ("default-src 'none'; media-src 'self'; img-src 'self' data:; "
             "style-src 'unsafe-inline'; frame-ancestors 'self'")
PDF_CSP = "script-src 'none'; object-src 'self'; frame-ancestors 'self'"
TEXT_LIMIT = 2 * 1024 * 1024
UNSATISFIABLE = "unsatisfiable"
_RANGE = re.compile(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*")


# the system's table differs by host: `.ts` is video/mp2t on one and a Qt file on another, `.m4a`
# audio/mp4a-latm, which no browser plays by that name
_FIXED = {".ts": "text/plain", ".m4a": "audio/mp4", ".wav": "audio/wav", ".flac": "audio/flac",
          ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
          ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
          ".opus": "audio/ogg", ".pdf": "application/pdf"}


def mime_of(path: str) -> str:
    fixed = _FIXED.get(os.path.splitext(path)[1].lower())
    if fixed:
        return fixed
    m, _ = mimetypes.guess_type(path)
    if m:
        return m
    if path.endswith((".ts", ".tsx", ".py", ".rs", ".go", ".sh", ".toml", ".yaml", ".yml",
                      ".md", ".txt", ".log", ".cfg", ".ini", ".lock")):
        return "text/plain"
    return "application/octet-stream"


def is_text(mime: str) -> bool:
    return mime.startswith("text/") or mime in ("application/json", "application/javascript",
                                                "application/xml", "image/svg+xml")


def media_kind(mime: str) -> str | None:
    """The pane kind of a file the player or viewer shows: video, audio, pdf; None for others."""
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime == "application/pdf":
        return "pdf"
    return None


def kind_of(mime: str, path: str) -> str:
    """The pane kind for a file: image, preview (HTML), markdown, video, audio, pdf, file (text)
    or binary (a size and Download)."""
    if mime.startswith("image/"):
        return "image"
    if mime == "text/html":
        return "preview"
    if mime == "text/markdown" or path.endswith(".md"):
        return "markdown"
    return media_kind(mime) or ("file" if is_text(mime) else "binary")


def view_csp(mime: str, path: str = "") -> str:
    """The view route's CSP: the preview's for pages, images and text, else one without scripts."""
    k = kind_of(mime, path)
    if k == "pdf":
        return PDF_CSP
    return MEDIA_CSP if k in ("video", "audio", "binary") else PREVIEW_CSP


def byte_range(header: str | None, size: int):
    """A `Range: bytes=` header as (first, last) for a file of `size` bytes; None to answer the
    whole file (no header, another unit, several ranges, a malformed or reversed one), or
    UNSATISFIABLE (a 416): a first byte at or past the end, a suffix of 0, an empty file."""
    m = _RANGE.fullmatch(header or "")
    if not m or not (m.group(1) or m.group(2)):
        return None
    a, b = m.groups()
    if not a:
        n = int(b)
        if n == 0 or size == 0:
            return UNSATISFIABLE
        return max(0, size - n), size - 1
    first = int(a)
    if first >= size:
        return UNSATISFIABLE
    last = min(int(b), size - 1) if b else size - 1
    if last < first:
        return None
    return first, last


def stream_etag(st) -> str:
    """A streamed file's validator: inode, size and mtime (a write replaces the inode), so a file
    still growing between two range requests gets a new one without a read of its bytes."""
    return f'"{st.st_ino:x}-{st.st_size:x}-{st.st_mtime_ns:x}"'


class PanesMixin:
    def _api_error(self, status, kind, msg):
        from chatd.service import ApiError
        return ApiError(status, kind, msg)

    def _ws_for(self, cid: str):
        self.conv(cid)                       # 404 for an unknown chat
        return self.workspace(cid)

    # ------------------------------------------------------------------ the sandbox
    def sandbox_status(self, cid: str) -> dict:
        self.conv(cid)
        return self.sandboxes.status(cid)

    def sandbox_action(self, cid: str, action: str) -> dict:
        from chatd.sandbox import Unavailable
        self.conv(cid)
        if not self.sandboxes.available():
            raise self._api_error(503, "sandbox_unavailable",
                                  "the sandbox image is not installed on this box")
        try:
            if action == "start":
                why = self.sandboxes.admit() if self.sandboxes.box(cid).state != "running" else None
                if why:
                    raise self._api_error(503, "waiting_memory", why)
                self.sandboxes.ensure(cid)
            elif action == "stop":
                self.sandboxes.stop(cid, "user")
            elif action == "reset":
                self.sandboxes.remove(cid, files=False)
        except Unavailable as exc:
            raise self._api_error(503, "sandbox_unavailable", str(exc)) from None
        return self.sandboxes.status(cid)

    # ------------------------------------------------------------------ files
    def files(self, cid: str, path: str) -> list[dict]:
        ws = self._ws_for(cid)
        try:
            return ws.list(path or "/workspace")
        except FileNotFoundError:
            raise self._api_error(404, "not_found", "no such directory") from None
        except (PathError, NotADirectoryError) as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def file(self, cid: str, path: str) -> tuple[dict, bytes]:
        """({content?, mime, bytes, etag}, raw bytes)"""
        from chatd.workspace import etag
        ws = self._ws_for(cid)
        try:
            st = ws.stat(path)
            if stat.S_ISDIR(st.st_mode):
                raise self._api_error(400, "bad_path", "a directory")
            # the JSON view holds text up to TEXT_LIMIT; bigger files stream from the view route
            data = ws.read(path, limit=TEXT_LIMIT + 1) if st.st_size <= TEXT_LIMIT else b""
        except PathError as exc:
            raise self._api_error(400 if "no such" not in str(exc) else 404, "bad_path",
                                  str(exc)) from None
        except FileNotFoundError:
            raise self._api_error(404, "not_found", "no such file") from None
        mime = mime_of(path)
        meta = {"mime": mime, "bytes": int(st.st_size), "etag": etag(data)
                if st.st_size <= TEXT_LIMIT else None, "kind": kind_of(mime, path),
                "url": self.view_url(cid, path)}
        if is_text(mime) and st.st_size <= TEXT_LIMIT and b"\0" not in data[:4096]:
            meta["content"] = data.decode("utf-8", "replace")
        elif meta["kind"] == "file":
            meta["kind"] = "binary"          # text by its name, binary by its bytes or size
        return meta, data

    def view_url(self, cid: str, path: str) -> str:
        rel = self.workspace(cid).rel(path)
        return (f"/v1/chat-app/conversations/{cid}/sandbox/view/"
                f"{urllib.parse.quote(rel, safe='/')}")

    def view_page(self, cid: str, path: str, data: bytes) -> bytes:
        """A page on the view route: absolute `/` URLs to the page's own directory (a built site's
        `/assets/...`), and the preview's shim (chatd/preview.py)."""
        from chatd.preview import rewrite_page
        url = self.view_url(cid, path)
        return rewrite_page(data, url.rsplit("/", 1)[0])

    def open_file(self, cid: str, path: str):
        """(file object, {mime, bytes, etag, mtime}) for the view and download routes, which
        stream it; the caller closes the file."""
        ws = self._ws_for(cid)
        try:
            fh, st = ws.open(path)
        except PathError as exc:
            raise self._api_error(400 if "no such" not in str(exc) else 404, "bad_path",
                                  str(exc)) from None
        except IsADirectoryError:
            raise self._api_error(400, "bad_path", "a directory") from None
        except FileNotFoundError:
            raise self._api_error(404, "not_found", "no such file") from None
        return fh, {"mime": mime_of(path), "bytes": int(st.st_size), "etag": stream_etag(st),
                    "mtime": st.st_mtime}

    def put_file(self, cid: str, path: str, data: bytes, if_match: str | None) -> None:
        ws = self._ws_for(cid)
        try:
            ws.write(path, data, if_match=if_match)
        except ChangedError:
            raise self._api_error(412, "changed", "the file changed on the box; reload it") \
                from None
        except PathError as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def delete_file(self, cid: str, path: str) -> None:
        ws = self._ws_for(cid)
        try:
            ws.delete(path)
        except (PathError, FileNotFoundError) as exc:
            raise self._api_error(400, "bad_path", str(exc)) from None

    def upload_files(self, cid: str, folder: str, content_type: str, body: bytes) -> list[dict]:
        import email.parser
        import email.policy
        ws = self._ws_for(cid)
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        out = []
        for part in msg.iter_parts() if msg.is_multipart() else []:
            name = os.path.basename(part.get_filename() or "")
            if not name:
                continue
            data = part.get_payload(decode=True) or b""
            if len(data) > 50 * 1024 * 1024:
                raise self._api_error(413, "too_large", f"{name} is over 50 MB")
            target = (folder.rstrip("/") or "/workspace") + "/" + name
            try:
                ws.write(target, data)
            except PathError as exc:
                raise self._api_error(400, "bad_path", str(exc)) from None
            full = ws.resolve(target)
            out.append({"path": ws.show(full), "kind": "file", "bytes": len(data)})
        return out

    # ------------------------------------------------------------------ diff and undo
    def diff(self, cid: str, run_id: str | None) -> dict:
        self.conv(cid)
        if not run_id:
            r = self.store.one("SELECT id FROM runs WHERE conversation_id = ? ORDER BY started "
                               "DESC LIMIT 1", (cid,))
            run_id = r["id"] if r else ""
        return {"run_id": run_id, **self.snaps.diff(cid, run_id)}

    def undo(self, cid: str, run_id: str) -> dict:
        conv = self.conv(cid)
        if conv.run is not None and conv.run.active:
            raise self._api_error(409, "run_active", "undo when the run has ended")
        try:
            n = self.snaps.undo(cid, self.workspace(cid), run_id, "undo-" + ulid())
        except ValueError as exc:
            raise self._api_error(409, "nothing_to_undo", str(exc)) from None
        self.log(f"[undo] {cid} run {run_id}: {n} file(s)")
        return {"files": n}

    # ------------------------------------------------------------------ artifacts
    def artifact(self, cid: str, ref: str) -> dict:
        """What a pane shows for `ref`; the preview loads `url` in a sandboxed iframe."""
        if ref.startswith("diff:"):
            run = ref[5:]
            d = self.diff(cid, run)
            text = "".join(f["patch"] for f in d["files"])
            return {"kind": "diff", "content": text or "No file changed in this run.\n",
                    "mime": "text/x-diff", "title": "changes", "files": d["files"]}
        if ref.startswith("port:"):
            return self.preview_of_port(cid, int(ref[5:]))
        path = ref[5:] if ref.startswith("file:") else ref
        meta, data = self.file(cid, path)
        mime = meta["mime"]
        rel = self.workspace(cid).rel(path)
        title = os.path.basename(rel) or "workspace"
        url = meta["url"]
        if meta["kind"] in ("video", "audio", "pdf", "binary"):
            return {"kind": meta["kind"], "content": "", "mime": mime, "title": title, "url": url,
                    "bytes": meta["bytes"]}
        if mime.startswith("image/"):
            return {"kind": "image", "content": url, "mime": mime, "title": title, "url": url}
        if mime == "text/html":
            from chatd.preview import needs_server
            need = needs_server(self.workspace(cid), path)
            if need:
                return self.preview_of_page(cid, path, need)
            return {"kind": "preview", "content": meta.get("content") or "", "mime": mime,
                    "title": title, "url": url}
        if mime in ("text/markdown",) or rel.endswith(".md"):
            return {"kind": "markdown", "content": meta.get("content") or "", "mime": mime,
                    "title": title}
        return {"kind": "file", "content": meta.get("content") or "", "mime": mime,
                "title": title, "url": url}
