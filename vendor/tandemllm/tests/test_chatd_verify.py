"""chatd's verify phase (SRV-532): a run that built or changed a web page or app checks it in a
browser before it ends, the model reads the check, a bounded number of rounds, and the answer's
"Checked / Not checked" block. CPU only: the model is tests/fake_backend.py, the sandbox a fake
whose kit run answers with a page check report and writes the screenshot.

Run: python -m pytest -q tests/test_chatd_verify.py
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib

from chatd_util import Chat, run_all
from test_chatd_agent import FakeBoxes, calls, parts_of, requests

from chatd import sandbox as sb_mod, tools as T, verify as V
from chatd.workspace import Workspace

PAGE = "<!doctype html><title>Cube</title><canvas></canvas><script>1</script>"
PNG = b"\x89PNG\r\n\x1a\nfake"


def report(ok=True, errors=(), text="Day Pause", look=None, state=None, overflow=False):
    rep = {"url": "file:///workspace/index.html", "ok": ok,
           "verdict": "PASS: the page loads, no console errors" if ok else
           f"FAIL: {len(errors)} console error(s)", "errors": list(errors), "steps": [],
           "controls": [{"tag": "button", "label": "Day"}],
           "widths": [{"width": 1280, "scroll_width": 1280, "text": text, "overflow_x": False,
                       "look": look or {"brightness": 62, "top_colour": "#88ccff", "top_share": 40,
                                        "colours": 9, "saturation": 35}},
                      {"width": 390, "scroll_width": 512 if overflow else 390, "text": text,
                       "overflow_x": overflow, "look": None}]}
    if state is not None:
        rep["state"] = state
    return rep


class KitBoxes(FakeBoxes):
    """The kit's page check answers `rep` (a dict, or a callable of the call count), writes the
    1280 px screenshot where the command names it, and runs `after` (a fix the model would make)."""

    def __init__(self, data, svc):
        super().__init__(data)
        self.svc, self.rep, self.code, self.after, self.checks = svc, report(), 0, None, []

    def run(self, cid, cmd, *, timeout=600, on_output=None, cancel=None):
        line = cmd[-1]
        if "laya_drive.py page" not in line:
            return super().run(cid, cmd, timeout=timeout, on_output=on_output, cancel=cancel)
        self.ran.append(cmd)
        self.checks.append(line)
        shots = line.split("--shots ")[1].split()[0]
        ws = self.svc.workspace(cid)
        ws.write(f"{shots}/page-1280.png".replace("/workspace/", ""), PNG)
        if self.after:
            self.after(ws, len(self.checks))
        rep = self.rep(len(self.checks)) if callable(self.rep) else self.rep
        on_output and on_output(json.dumps(rep, indent=1) + "\n")
        return self.code, False


def chat_with_kit(vision=False):
    c = Chat()
    boxes = KitBoxes(c.data, c.svc)
    c.svc.sandboxes = boxes
    boxes.on_change = c.svc._sandbox_changed
    boxes.network_mode = c.svc._network_mode
    c.svc._caps = None
    if vision:
        info = c.svc.model_info
        c.svc.model_info = lambda mid: {**info(mid), "vision": True}
    return c, boxes


def verify_parts(s):
    return [p for p in parts_of(s) if p["type"] == "verify"]


def checks(s):
    return [p for p in parts_of(s) if p["type"] == "tool_call" and p.get("harness") == "verify"]


def write_page(path="index.html", content=PAGE):
    return calls(("write_file", {"path": path, "content": content}))


# ---------------------------------------------------------------------- the phase in a run
def test_a_run_that_writes_a_page_checks_it_and_the_model_reads_the_check():
    c, boxes = chat_with_kit()
    try:
        cid = c.new()
        ev = c.events(cid)
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        frames = ev.until(lambda f: f[1] == "run.done", 30)
        ev.close()
        a = s["messages"][-1]
        assert a["status"] == "done" and a["stats"]["steps"] == 3, a["stats"]
        chk = checks(s)
        assert len(chk) == 1 and chk[0]["name"] == "render_page" and chk[0]["status"] == "done"
        assert chk[0]["source"] == {"kind": "harness"}
        assert json.loads(chk[0]["input"]) == {"path": "/workspace/index.html"}
        assert len(boxes.checks) == 1 and "page file:///workspace/index.html --no-laya" in \
            boxes.checks[0] and "--widths 1280,390" in boxes.checks[0]
        ws = c.svc.workspace(cid)
        assert ws.read(".tandem/laya_drive.py").startswith(b"#!/usr/bin/env python3")
        phases = [json.loads(f[2]) for f in frames if f[1] == "run.phase"]
        assert any(p["phase"] == "verifying" and p["now"].startswith(
            "Verifying the app: loading /workspace/index.html in a browser (check 1 of 2)")
                   for p in phases), [p["phase"] for p in phases]
        tool = requests(c)[-1]["messages"][-1]
        assert tool["role"] == "tool" and tool["content"].startswith(
            "The harness checked the app before the run ends (check 1 of 2)")
        assert "Result: PASS" in tool["content"] and "brightness 62 %" in tool["content"]
        assert "You read no images here" in tool["content"]
        # the check sits in the step of the answer it followed: the answer, then the call
        first = requests(c)[-1]["messages"][-2]
        assert first["role"] == "assistant" and first["tool_calls"][0]["id"] == chk[0]["id"]
        vp = verify_parts(s)
        assert len(vp) == 1 and vp[0]["state"] == "passed", vp
        assert vp[0]["text"].startswith("Verified by the harness.\nChecked: /workspace/index.html "
                                        "in headless Chromium at 1280 and 390 px (harness check 1)"
                                        ": no console errors, ")
        assert vp[0]["checked"][0].startswith("/workspace/index.html in headless Chromium")
        assert vp[0]["not_checked"][-2:] == ["input and interactions", "sound and motion over time"]
        assert "Not checked: how it looks: the model read no screenshot" in vp[0]["text"]
        assert "input and interactions; sound and motion over time." in vp[0]["text"]
    finally:
        c.close()


def test_a_failed_check_goes_back_to_the_model_and_the_summary_keeps_it():
    c, boxes = chat_with_kit()
    try:
        boxes.rep = report(ok=False, errors=["exception: TypeError: x is undefined"],
                           overflow=True)
        boxes.code = 1
        cid = c.new()
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        chk = checks(s)
        assert len(chk) == 1 and chk[0]["status"] == "failed" and chk[0]["output"]["is_error"]
        tool = requests(c)[-1]["messages"][-1]["content"]
        assert "Console errors and failed loads: exception: TypeError: x is undefined" in tool
        assert "At 390 px the page is wider than the screen (scroll width 512 px)." in tool
        assert "fix it now" in tool
        vp = verify_parts(s)[0]
        assert vp["state"] == "problems"
        assert vp["text"].startswith("The last check found problems the run did not fix.")
        assert "problems: 1 console error (exception: TypeError" in vp["text"]
        assert "wider than the screen at 390 px" in vp["text"]
    finally:
        c.close()


def test_checks_are_capped_and_changes_after_the_last_are_named():
    c, boxes = chat_with_kit()
    try:
        boxes.after = lambda ws, n: ws.write("index.html", (PAGE + f"<!-- fix {n} -->").encode())
        cid = c.new()
        c.send(cid, write_page())
        s = c.wait_idle(cid, 40)
        assert len(boxes.checks) == V.ROUNDS == 2
        assert len(checks(s)) == 2 and s["messages"][-1]["stats"]["steps"] == 4
        assert "(check 2 of 2)" in json.dumps(requests(c)[-1]["messages"][-1])
        vp = verify_parts(s)[0]
        assert vp["state"] == "stale"
        assert "Not checked: 1 file changed after the last check (index.html): the limit is 2 " \
               "checks a run" in vp["text"]
    finally:
        c.close()


def test_a_second_check_only_when_web_files_changed_after_the_first():
    c, boxes = chat_with_kit()
    try:
        cid = c.new()
        c.send(cid, write_page())
        c.wait_idle(cid, 30)
        assert len(boxes.checks) == 1
        # a run that changes no web file: no check, no block
        c.send(cid, calls(("write_file", {"path": "notes.txt", "content": "hi"})))
        s = c.wait_idle(cid, 30)
        assert len(boxes.checks) == 1
        assert not [p for p in s["messages"][-1]["parts"] if p["type"] == "verify"]
        # a script is no app: nothing to load
        c.send(cid, calls(("write_file", {"path": "tool/run.js", "content": "console.log(1)"})))
        s = c.wait_idle(cid, 30)
        assert len(boxes.checks) == 1
        assert not [p for p in s["messages"][-1]["parts"] if p["type"] == "verify"]
    finally:
        c.close()


def test_without_a_sandbox_the_answer_says_it_was_not_verified():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        assert s["messages"][-1]["stats"]["steps"] == 2, "no extra model call"
        assert not checks(s)
        vp = verify_parts(s)[0]
        assert vp["state"] == "not_run"
        assert vp["text"] == ("Not verified.\nNot checked: /workspace/index.html in a browser: "
                              "this box has no sandbox to run a browser in; input and "
                              "interactions; sound and motion over time.")
    finally:
        c.close()


def test_the_chat_setting_turns_it_off():
    c, boxes = chat_with_kit()
    try:
        cid = c.new()
        assert c.snap(cid)["settings"]["verify"] is True
        c.ok("PATCH", f"/conversations/{cid}/settings", {"verify": 0})
        assert c.snap(cid)["settings"]["verify"] is False
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        assert boxes.checks == [] and not checks(s) and not verify_parts(s)
        assert s["messages"][-1]["stats"]["steps"] == 2
    finally:
        c.close()


def test_a_page_that_needs_its_dev_server_names_the_command():
    c, boxes = chat_with_kit()
    try:
        pkg = json.dumps({"scripts": {"dev": "vite"}, "devDependencies": {"vite": "^5"}})
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "app/package.json", "content": pkg}),
                          ("write_file", {"path": "app/index.html", "content":
                                          '<script type="module" src="/src/main.ts"></script>'})))
        s = c.wait_idle(cid, 30)
        assert boxes.checks == [], "nothing to load"
        tool = requests(c)[-1]["messages"][-1]["content"]
        assert "index.html runs only from its dev server (its script /src/main.ts needs " \
               "compiling), and none runs in the sandbox. The command is `npm run dev` in " \
               "/workspace/app." in tool
        assert checks(s)[0]["status"] == "failed"
        assert verify_parts(s)[0]["state"] == "problems"
    finally:
        c.close()


MODULE_PAGE = ('<!doctype html><script type="importmap">{"imports":{"three":"https://unpkg.com/'
               'three@0.170.0/build/three.module.js"}}</script><canvas></canvas>'
               '<script type="module" src="js/main.js"></script>')


def test_a_module_page_is_checked_through_the_server_that_serves_its_folder():
    """Live (2026-10-17, the pagoda replay): a 15-file Three.js app with
    `<script type="module" src="js/main.js">` and an import map, served by `python3 -m
    http.server` from its folder. Loaded as file://, Chromium refuses the module (origin null,
    CORS) and the page stays blank: the check must load it through the server."""
    c, boxes = chat_with_kit()
    try:
        c.svc.listening = lambda cid, http=False: [
            {"port": 8124, "addresses": ["0.0.0.0"], "command": "python3 -m http.server 8124",
             "cwd": "/workspace/garden", "started": 2, "http": True},
            {"port": 9000, "addresses": ["0.0.0.0"], "command": "python3 -m http.server 9000",
             "cwd": "/workspace/other", "started": 3, "http": True}]
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "garden/index.html", "content": MODULE_PAGE}),
                          ("write_file", {"path": "garden/js/main.js",
                                          "content": "import * as THREE from 'three';"})))
        s = c.wait_idle(cid, 30)
        assert json.loads(checks(s)[0]["input"]) == {"port": 8124, "path": "index.html"}
        assert "page http://localhost:8124/index.html --no-laya" in boxes.checks[0]
        assert verify_parts(s)[0]["text"].startswith(
            "Verified by the harness.\nChecked: /workspace/garden/index.html on port 8124 in "
            "headless Chromium")
    finally:
        c.close()


def test_a_module_page_without_a_server_is_not_loaded_as_a_file():
    c, boxes = chat_with_kit()
    try:
        c.svc.listening = lambda cid, http=False: []
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "garden/index.html", "content": MODULE_PAGE}),
                          ("write_file", {"path": "garden/js/main.js", "content": "let a = 1;"})))
        s = c.wait_idle(cid, 30)
        assert boxes.checks == [], "a file:// load would only show the CORS refusal"
        tool = requests(c)[-1]["messages"][-1]["content"]
        assert "index.html loads its module scripts from its folder, which a browser refuses " \
               "from file:// (CORS), and no HTTP server runs from /workspace/garden in the " \
               "sandbox. Serve the folder, for example `python3 -m http.server 8000` there." in tool
        assert checks(s)[0]["status"] == "failed"
    finally:
        c.close()


def test_which_pages_need_http_and_which_server_serves_them():
    from chatd import preview as pv
    ws = Workspace(tempfile.mkdtemp())
    ws.write("a/index.html", MODULE_PAGE.encode())
    ws.write("b/index.html", b'<script type="module">import { x } from "./x.js";</script>')
    ws.write("c/index.html", b'<script type="module">import * as T from "https://unpkg.com/'
                             b'three@0.170.0/build/three.module.js";</script>')
    ws.write("d/index.html", PAGE.encode())
    got = {d: pv.file_blocks_modules(ws, f"/workspace/{d}/index.html") for d in "abcd"}
    assert got == {"a": True, "b": True, "c": False, "d": False}, got
    rows = [{"port": 1, "cwd": "/workspace", "started": 1},
            {"port": 2, "cwd": "/workspace/a", "started": 5},
            {"port": 3, "cwd": "/workspace/ab", "started": 9},
            {"port": 4, "cwd": "/workspace/a", "started": 7, "http": False}]
    assert pv.server_for(rows, "/workspace/a/index.html") == (rows[1], "index.html")
    assert pv.server_for(rows[:1], "/workspace/a/x/index.html") == (rows[0], "a/x/index.html")
    assert pv.server_for([], "/workspace/a/index.html") is None


def test_a_presented_port_is_the_target():
    c, boxes = chat_with_kit()
    try:
        c.svc.artifact = lambda cid, ref: {"kind": "preview", "port": 5173, "server": "vite"}
        c.svc.probe = lambda cid, port, path="/": {"path": "/", "status": 200, "type": "text/html"}
        cid = c.new()
        c.send(cid, calls(("write_file", {"path": "web/main.js", "content": "let a = 1"}),
                          ("present", {"kind": "preview", "port": 5173})))
        s = c.wait_idle(cid, 30)
        assert json.loads(checks(s)[0]["input"]) == {"port": 5173}
        assert "page http://localhost:5173/ --no-laya" in boxes.checks[0]
        vp = verify_parts(s)[0]
        assert "the preview pane loaded without a problem" in vp["text"]
    finally:
        c.close()


def test_a_model_that_reads_images_gets_the_screenshot():
    c, boxes = chat_with_kit(vision=True)
    try:
        boxes.rep = report(state={"frames": 120})
        cid = c.new()
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        # a backend's model: the tool's images in a user message right after the results
        tool, imgs = requests(c)[-1]["messages"][-2:]
        assert tool["role"] == "tool" and imgs["role"] == "user"
        assert imgs["content"][1]["type"] == "image_url"
        assert "The screenshot at 1280 px is attached" in tool["content"]
        assert 'The page\'s state hook: {"frames": 120}' in tool["content"]
        vp = verify_parts(s)[0]
        assert "; the screenshot, by the model; the page's state hook answered." in vp["text"]
        assert "how it looks" not in vp["text"]
    finally:
        c.close()


def test_a_check_the_sandbox_cannot_run_ends_the_run_with_the_reason():
    c, boxes = chat_with_kit()
    try:
        def gone(cid, cancel=None, on_wait=None):
            raise sb_mod.Unavailable("the sandbox image is not installed on this box",
                                     {"sandbox": {"reason": "no_image"}})
        cid = c.new()
        c.send(cid, write_page())
        boxes.ensure = gone
        s = c.wait_idle(cid, 30)
        assert s["messages"][-1]["stats"]["steps"] == 2, "nothing for the model to fix"
        assert checks(s)[0]["status"] == "failed"
        vp = verify_parts(s)[0]
        assert vp["state"] == "not_run" and "the check could not run (the sandbox image is not " \
                                               "installed on this box)" in vp["text"]
    finally:
        c.close()


def test_a_kit_that_finds_no_browser_is_no_round():
    c, boxes = chat_with_kit()
    try:
        boxes.rep = {"url": "x", "ok": False, "verdict": "NO BROWSER: no Chrome or Chromium on "
                     "this machine."}
        boxes.code = 2
        cid = c.new()
        c.send(cid, write_page())
        s = c.wait_idle(cid, 30)
        assert s["messages"][-1]["stats"]["steps"] == 2
        assert verify_parts(s)[0]["state"] == "not_run"
        assert "NO BROWSER" in verify_parts(s)[0]["text"]
    finally:
        c.close()


# ---------------------------------------------------------------------- the words
def test_changed_web_files_and_pages_first():
    before = {"index.html": "a", "notes.txt": "a", "src/app.js": "a"}
    now = {"index.html": "b", "notes.txt": "b", "src/app.js": "a", "about/index.html": "n",
           "x.html": "n", "style.css": "n"}
    assert V.changed_web(before, now) == ["about/index.html", "index.html", "style.css", "x.html"]
    assert V._pages_first(["/w/x.html", "/w/about/index.html", "/w/index.html", "/w/a.css"]) == \
        ["/w/index.html", "/w/about/index.html", "/w/x.html"]


def test_the_look_in_words():
    """Numbers, no verdict: live (2026-10-17) "95 % one colour (dark)" made the model take a red
    cube (the second colour, 4 %) for an empty page and rework a page that was right."""
    assert V.look_words({"brightness": 7, "top_colour": "#111111", "top_share": 95,
                         "top": [["#111111", 95], ["#cc2222", 4]], "colours": 2,
                         "saturation": 3}) == \
        "brightness 7 %, colours #111111 95 %, #cc2222 4 %, saturation 3 %"
    assert V.look_words({"brightness": 18, "top_colour": "#445577", "top_share": 72,
                         "saturation": 6}) == "brightness 18 %, colours #445577 72 %, saturation 6 %"
    assert V.look_words(None) == ""


def test_the_playtest_line_names_the_package_when_it_is_in_the_task():
    t = {"url": "file:///workspace/index.html", "label": "/workspace/index.html"}
    assert "laya-playtest package is in this task" in V.check_words(report(), t, 1, False, True)
    assert "laya-playtest" not in V.check_words(report(), t, 1, False, False)


def test_own_checks_come_from_the_call_outcomes():
    parts = [{"type": "tool_call", "name": "render_page", "status": "done",
              "output": {"content": "rendered ...\nconsole:\nUncaught x"}},
             {"type": "tool_call", "name": "render_page", "status": "done",
              "output": {"content": "rendered ...\nno console errors"}},
             {"type": "tool_call", "name": "bash", "status": "done",
              "input": json.dumps({"command": "python3 .tandem/laya_drive.py browser http://x"}),
              "output": {"content": "", "exit_code": 0}},
             {"type": "tool_call", "name": "render_page", "harness": "verify", "status": "failed",
              "output": {"content": "x"}}]
    assert V.own_checks(parts) == ["render_page 2 times by the model (the last one rendered)",
                                   "a Laya playtest of its own (exit 0)"]
    st = {**V.new_state(), "target": {"label": "/workspace/index.html"},
          "checks": [{"n": 1, "label": "/workspace/index.html", "state": "passed",
                      "facts": "it loaded", "hook": False, "seen": True}]}
    text = V.summary(st, parts)["text"]
    assert "input and interactions" not in text, "a playtest ran"


def test_a_library_text_that_downloads_the_kit_gets_it_in_the_sandbox():
    ws = Workspace(tempfile.mkdtemp())

    class Ctx:
        def __init__(self):
            self.ws = ws
            self.engine = self

        def get_json(self, path, timeout=3.0):
            return 200, {"name": "laya-playtest", "body": "1. Kit: `curl -sf http://127.0.0.1:8000"
                         "/s/abc/v1/laya/kit/laya_drive.py -o x`"}
    res = T._library_read(Ctx(), {"name": "laya-playtest"})
    assert "In this chat the kit is in the sandbox already: /workspace/.tandem/laya_drive.py" in \
        res.content
    assert ws.read(".tandem/laya_drive.py").startswith(b"#!/usr/bin/env python3")
    assert ws.read(".tandem/laya-drive.js")
    assert V.kit_note("# web-layout\n\nno kit here", ws) == ""


def test_the_kit_reads_a_screenshot_in_numbers():
    """laya_drive.py `look_of`: the PNG decoder handles every row filter Chrome writes."""
    import sys
    sys.path.insert(0, V.KIT_DIR)
    import laya_drive as L

    def pixel(x, y):
        return (200, 30, 30, 255) if x < 4 else (60, 70, 90, 255)
    w, h, bpp = 8, 5, 4
    rows = [bytes(v for x in range(w) for v in pixel(x, y)) for y in range(h)]
    out, prev = b"", bytes(w * bpp)
    for y, (f, row) in enumerate(zip((0, 1, 2, 3, 4), rows)):
        enc = bytearray()
        for i, v in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b, cc = prev[i], prev[i - bpp] if i >= bpp else 0
            p = a + b - cc
            pr = a if abs(p - a) <= abs(p - b) and abs(p - a) <= abs(p - cc) else \
                b if abs(p - b) <= abs(p - cc) else cc
            enc.append((v - (0, a, b, (a + b) // 2, pr)[f]) & 255)
        out += bytes([f]) + bytes(enc)
        prev = row

    def chunk(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(out)) + chunk(b"IEND", b""))
    assert L.png_pixels(png)[2][4][0] == (200, 30, 30) and L.png_pixels(png)[2][4][7] == \
        (60, 70, 90)
    assert L.look_of(png) == {"brightness": 27, "top_colour": "#cc1111", "top_share": 50,
                              "top": [["#cc1111", 50], ["#334455", 50]], "colours": 2,
                              "saturation": 39}
    assert L.look_of(b"\x89PNG\r\n\x1a\n") is None


if __name__ == "__main__":
    run_all(globals())
