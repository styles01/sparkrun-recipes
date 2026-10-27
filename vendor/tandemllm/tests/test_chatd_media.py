"""VIS-159 (QA VIS-160): media files in the Chat tab. The view route streams a workspace file with
byte ranges (206, 416, Content-Range, Accept-Ranges, ETag) and no 2 MiB cap; the file and
artifact routes name the pane kind (video, audio, pdf, binary) and the view URL; `present` on a
media file opens the player. CPU only, a chatd on a free port, no sandbox."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.parse

from chatd_util import KEY, Chat, run_all

from chatd import panes as P

MiB = 1024 * 1024


def _blob(n: int) -> bytes:
    """n bytes that differ every 64 KiB, so an offset error shows in a compare"""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(i.to_bytes(4, "big")).digest() * 2048
        i += 1
    return bytes(out[:n])


def _get(c: Chat, path: str, headers: dict | None = None) -> tuple[int, bytes, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", c.port, timeout=60)
    conn.request("GET", "/v1/chat-app" + path, headers={"X-Tandem-Forward": KEY,
                                                        **(headers or {})})
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body, {k.lower(): v for k, v in r.getheaders()}


# ---------------------------------------------------------------------- Range parsing
def test_byte_range_forms():
    assert P.byte_range("bytes=0-99", 1000) == (0, 99)
    assert P.byte_range("bytes=1000000-", 40 * MiB) == (1000000, 40 * MiB - 1)
    assert P.byte_range("bytes=-500", 1000) == (500, 999)               # suffix
    assert P.byte_range("bytes=-5000", 1000) == (0, 999)                # suffix over the size
    assert P.byte_range("bytes=900-5000", 1000) == (900, 999)           # last byte clamped
    assert P.byte_range(" bytes = 3 - 4 ", 10) == (3, 4)


def test_byte_range_unsatisfiable_and_ignored():
    assert P.byte_range("bytes=1000-", 1000) is P.UNSATISFIABLE
    assert P.byte_range("bytes=-0", 1000) is P.UNSATISFIABLE
    assert P.byte_range("bytes=0-", 0) is P.UNSATISFIABLE                # an empty file
    assert P.byte_range("bytes=-10", 0) is P.UNSATISFIABLE
    for h in (None, "", "bytes=", "bytes=-", "bytes=5-2", "bytes=0-1,4-5", "items=0-1",
              "bytes=a-b"):
        assert P.byte_range(h, 1000) is None, h                          # answer the whole file


# ---------------------------------------------------------------------- kinds by mime
def test_pane_kind_by_file():
    want = {"clip.mp4": "video", "clip.webm": "video", "clip.MOV": "video", "a.wav": "audio",
            "a.mp3": "audio", "a.m4a": "audio", "doc.pdf": "pdf", "blob.bin": "binary",
            "x.zip": "binary", "page.html": "preview", "notes.md": "markdown",
            "shot.png": "image", "main.py": "file", "app.ts": "file", "data.json": "file"}
    for name, kind in want.items():
        assert P.kind_of(P.mime_of(name), name) == kind, name
    assert P.mime_of("a.m4a") == "audio/mp4" and P.mime_of("a.ts") == "text/plain"


def test_view_csp_blocks_scripts_for_media():
    for name in ("clip.mp4", "a.wav", "blob.bin"):
        csp = P.view_csp(P.mime_of(name), name)
        assert "media-src 'self'" in csp and "default-src 'none'" in csp
        assert "script-src" not in csp                       # default-src 'none' covers scripts
    assert "script-src 'none'" in P.view_csp("application/pdf", "d.pdf")
    assert P.view_csp("text/html", "i.html") == P.PREVIEW_CSP             # pages keep theirs


# ---------------------------------------------------------------------- the view route
def test_a_file_over_the_text_limit_is_served_whole():
    c = Chat()
    try:
        cid = c.new()
        data = _blob(12 * MiB)
        c.svc.workspace(cid).write("out/clip.mp4", data)
        st, body, h = _get(c, f"/conversations/{cid}/sandbox/view/out/clip.mp4")
        assert st == 200 and int(h["content-length"]) == len(data) == len(body)
        assert hashlib.sha256(body).digest() == hashlib.sha256(data).digest()
        assert h["content-type"] == "video/mp4" and h["accept-ranges"] == "bytes"
        assert h["etag"].startswith('"') and h["last-modified"].endswith("GMT")
        assert h["x-content-type-options"] == "nosniff"
        assert "media-src 'self'" in h["content-security-policy"]
        # the download is whole too
        st, body, h = _get(c, f"/conversations/{cid}/sandbox/file?path=out/clip.mp4&raw=1")
        assert st == 200 and len(body) == len(data) and h["content-disposition"] == "attachment"
    finally:
        c.close()


def test_seeking_answers_ranges():
    c = Chat()
    try:
        cid = c.new()
        data = _blob(40 * MiB)
        c.svc.workspace(cid).write("v.mp4", data)
        url = f"/conversations/{cid}/sandbox/view/v.mp4"
        st, body, h = _get(c, url, {"Range": "bytes=1000000-"})
        assert st == 206 and h["accept-ranges"] == "bytes"
        assert h["content-range"] == f"bytes 1000000-{len(data) - 1}/{len(data)}"
        assert int(h["content-length"]) == len(data) - 1000000 and body == data[1000000:]
        st, body, h = _get(c, url, {"Range": "bytes=100-199"})
        assert st == 206 and body == data[100:200] and h["content-range"] == \
            f"bytes 100-199/{len(data)}"
        st, body, h = _get(c, url, {"Range": "bytes=-4096"})
        assert st == 206 and body == data[-4096:]
        st, body, h = _get(c, url, {"Range": f"bytes={len(data)}-"})
        assert st == 416 and h["content-range"] == f"bytes */{len(data)}" and body == b""
        # Safari's first probe
        st, body, h = _get(c, url, {"Range": "bytes=0-1"})
        assert st == 206 and body == data[:2]
    finally:
        c.close()


def test_validators_if_range_and_if_none_match():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("v.webm", b"a" * 5000)
        url = f"/conversations/{cid}/sandbox/view/v.webm"
        _, _, h = _get(c, url)
        tag = h["etag"]
        st, body, _ = _get(c, url, {"Range": "bytes=10-19", "If-Range": tag})
        assert st == 206 and len(body) == 10
        st, body, _ = _get(c, url, {"If-None-Match": tag})
        assert st == 304 and body == b""
        ws.write("v.webm", b"b" * 7000)                     # rewritten while the player had it
        st, body, h2 = _get(c, url, {"Range": "bytes=10-19", "If-Range": tag})
        assert st == 200 and len(body) == 7000 and h2["etag"] != tag
    finally:
        c.close()


def test_the_view_route_keeps_its_path_checks_and_page_csp():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        ws.write("site/index.html", b"<h1>hi</h1>")
        st, body, h = _get(c, f"/conversations/{cid}/sandbox/view/site/index.html")
        # a page comes whole, with the preview's shim first (chatd/preview.py)
        assert st == 200 and body.startswith(b"<script>") and body.endswith(b"</script><h1>hi</h1>")
        assert h["content-security-policy"] == P.PREVIEW_CSP
        os.symlink("/etc/passwd", os.path.join(ws.root, "pw.mp4"))
        st, _, _ = _get(c, f"/conversations/{cid}/sandbox/view/pw.mp4")
        assert st == 400
        st, _, _ = _get(c, f"/conversations/{cid}/sandbox/view/..%2F..%2Fetc%2Fpasswd")
        assert st == 400
        st, _, _ = _get(c, f"/conversations/{cid}/sandbox/view/missing.mp4")
        assert st == 404
        st, _, _ = _get(c, f"/conversations/{cid}/sandbox/view/site")
        assert st == 400
        st, _, _ = _get(c, f"/conversations/{cid}/sandbox/view/site/index.html")
        assert st == 200
    finally:
        c.close()


def test_an_empty_file_is_served_empty():
    c = Chat()
    try:
        cid = c.new()
        c.svc.workspace(cid).write("e.mp3", b"")
        st, body, h = _get(c, f"/conversations/{cid}/sandbox/view/e.mp3")
        assert st == 200 and body == b"" and h["content-length"] == "0"
        st, _, h = _get(c, f"/conversations/{cid}/sandbox/view/e.mp3", {"Range": "bytes=0-"})
        assert st == 416 and h["content-range"] == "bytes */0"
    finally:
        c.close()


# ---------------------------------------------------------------------- what the panes read
def test_file_and_artifact_name_the_kind_and_url():
    c = Chat()
    try:
        cid = c.new()
        ws = c.svc.workspace(cid)
        files = {"clips/a b.mp4": b"\x00\x00\x00\x18ftypmp42", "a.wav": b"RIFF....WAVE",
                 "d.pdf": b"%PDF-1.4\n", "x.bin": b"\x00\x01\x02", "big.log": b"x" * (3 * MiB),
                 "main.py": b"print(1)\n"}
        for p, b in files.items():
            ws.write(p, b)
        kinds = {"clips/a b.mp4": "video", "a.wav": "audio", "d.pdf": "pdf", "x.bin": "binary",
                 "big.log": "binary", "main.py": "file"}
        for p, kind in kinds.items():
            meta = c.ok("GET", f"/conversations/{cid}/sandbox/file?path="
                               f"{urllib.parse.quote('/workspace/' + p)}")
            assert meta["kind"] == kind, (p, meta)
            assert meta["bytes"] == len(files[p])
            assert meta["url"].endswith("/sandbox/view/" + p.replace(" ", "%20"))
            assert (meta["content"] is not None) == (kind == "file")
        a = c.ok("GET", f"/conversations/{cid}/artifacts/" + "file%3A%2Fworkspace%2Fclips%2Fa%20b.mp4")
        assert a["kind"] == "video" and a["mime"] == "video/mp4" and a["bytes"] == 12
        assert a["url"] == f"/v1/chat-app/conversations/{cid}/sandbox/view/clips/a%20b.mp4"
        st, body, _ = _get(c, a["url"][len("/v1/chat-app"):])
        assert st == 200 and body == files["clips/a b.mp4"]
        for p, kind in (("a.wav", "audio"), ("d.pdf", "pdf"), ("x.bin", "binary")):
            a = c.ok("GET", f"/conversations/{cid}/artifacts/file%3A%2Fworkspace%2F{p}")
            assert a["kind"] == kind and a["url"].endswith("/sandbox/view/" + p)
    finally:
        c.close()


def test_present_on_a_video_opens_the_player():
    c = Chat()
    try:
        cid = c.new()
        c.svc.workspace(cid).write("lfm/explainer.mp4", b"\x00\x00\x00\x18ftypmp42")
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, "FAKE_CALLS " + json.dumps([{"name": "present", "arguments": {
            "kind": "preview", "path": "lfm/explainer.mp4"}}]))
        ev = json.loads(sse.until(lambda f: f[1] == "pane.open")[-1][2])
        assert ev["pane"] == {"kind": "video", "ref": "file:/workspace/lfm/explainer.mp4",
                              "title": None}
        sse.close()
        tools = {t["name"]: t for t in c.ok("GET", "/tools")}
        assert "video" in json.dumps(tools["present"])
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
