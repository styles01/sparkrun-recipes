"""The sandbox on a real box (SRV-355, SRV-357, OPS-103): one real container, CPU only. Skipped
unless QSE_SANDBOX_BOX=1, and it refuses to start a container where the admission would not (the
limit plus the floor, chatd/sandbox.py). One container at a time; it is removed at the end.

    QSE_SANDBOX_BOX=1 python3 tests/test_sandbox_box.py

Checks: the lockdown (uid 1000, no capabilities, read-only root, no network, the limits, no
devices); the LAN, this box and the engine unreachable while the public internet works through
chatd's proxy; a 2 GiB allocation dies at the 1 GiB limit and the cgroup counts the kill; the pids limit stops a fork storm;
render_page's Chromium writes a screenshot and reports a console error; a PTY; the port relay;
a second runtime adopts the running container; stop. Then every built-in tool through chatd's loop
(the fake model asks, the real sandbox and workspace answer; SRV-353), and an attachment sent
after the sandbox linked uploads/ out of the workspace (SEC-1).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the tests' 1 GiB limit (tests/chatd_util.py, tests/conftest.py), before chatd/sandbox.py reads
# it: a direct run imports neither
os.environ.setdefault("QSE_SANDBOX_MEM_GB", "1")

from chatd import sandbox as sb_mod  # noqa: E402
from chatd.sandbox import Sandboxes, demux  # noqa: E402

ON = os.environ.get("QSE_SANDBOX_BOX") == "1"
CID = "BOXTEST" + str(int(time.time()))


def _run(sb, cmd, timeout=120):
    out = []
    code, timed = sb.run(CID, ["bash", "-lc", cmd], timeout=timeout, on_output=out.append)
    return code, "".join(out)


def test_the_sandbox_on_this_box():
    if not ON:
        print("skipped: set QSE_SANDBOX_BOX=1 on the box")
        return
    avail = sb_mod.meminfo()["MemAvailable"] / sb_mod.GiB
    need = sb_mod.FLOOR_GB + sb_mod.LIMIT_BYTES / sb_mod.GiB
    assert avail >= need, f"MemAvailable {avail:.1f} GiB: wait for {need:g}"
    d = tempfile.mkdtemp(prefix="sbx-box-")
    sb = Sandboxes(d, log=print)
    sb.max_running = 1
    try:
        t0 = time.monotonic()
        ctr = sb.ensure(CID)
        print(f"started in {time.monotonic() - t0:.1f} s")
        st, info = sb.engine.call("GET", f"/containers/{ctr}/json")
        hc = info["HostConfig"]
        assert info["Config"]["User"] == "1000:1000"
        assert hc["CapDrop"] == ["ALL"] and hc["ReadonlyRootfs"] is True
        assert hc["NetworkMode"] == "none" and hc["Memory"] == sb_mod.LIMIT_BYTES == sb_mod.GiB
        assert hc["MemorySwap"] == sb_mod.GiB and hc["PidsLimit"] == 512
        assert not hc.get("Devices") and not hc.get("DeviceRequests")
        assert "no-new-privileges" in hc["SecurityOpt"]

        code, out = _run(sb, "id -u; touch /etc/x 2>&1 || echo ro-ok; echo hi > /workspace/a.txt")
        assert out.splitlines()[0] == "1000" and "ro-ok" in out
        assert open(os.path.join(d, "workspaces", CID, "a.txt")).read() == "hi\n"
        code, out = _run(sb, "ls /dev | tr '\\n' ' '; nvidia-smi 2>&1 | head -1")
        assert "nvidiactl" not in out and "nvidia0" not in out, out      # no GPU device

        # the network: nothing local, the internet through the proxy
        for target in ("http://192.168.1.1/", "http://192.168.1.131:8000/v1/models",
                       "http://10.0.0.1/", "http://169.254.169.254/"):
            code, out = _run(sb, f"curl -sS -m 10 -o /dev/null -w '%{{http_code}}' {target}")
            assert out.strip().endswith("403") or code != 0, (target, code, out)
        code, out = _run(sb, "curl -sS -m 5 --noproxy '*' http://192.168.1.131:8000/ 2>&1 "
                             "| head -c 200")
        assert "Couldn't connect" in out or "unreachable" in out, out   # no route at all
        code, out = _run(sb, "curl -sS -m 30 -o /dev/null -w '%{http_code}' "
                             "https://pypi.org/simple/pip/")
        assert out.strip() == "200", (code, out)
        print("network: LAN, box and engine refused; https://pypi.org 200 through the proxy")

        # the memory limit and the pids limit
        before = sb.oom_kills(CID)
        code, out = _run(sb, "python3 -c 'x = bytearray(2 * 1024 ** 3)'; echo exit=$?")
        assert "exit=137" in out, out
        assert before is not None and sb.oom_kills(CID) == before + 1, (before, sb.oom_kills(CID))
        _run(sb, "nohup bash -c 'for i in $(seq 700); do sleep 15 & done; wait' >/dev/null "
                 "2>&1 & sleep 3", timeout=60)
        st, stats = sb.engine.call("GET", f"/containers/{ctr}/stats?stream=false")
        n = int(stats["pids_stats"]["current"])
        assert 400 < n <= 512, n
        time.sleep(16)
        code, out = _run(sb, "echo alive")
        assert out.strip() == "alive", out
        print(f"limits: 2 GiB allocation killed (137, oom_kill counted), fork storm stopped at {n} processes")

        # render_page's Chromium
        page = ("<html><body><h1>box test</h1><script>console.error('boom from the page')"
                "</script></body></html>")
        os.makedirs(os.path.join(d, "workspaces", CID, "site"), exist_ok=True)
        with open(os.path.join(d, "workspaces", CID, "site", "index.html"), "w") as fh:
            fh.write(page)
        code, out = _run(sb, "mkdir -p /workspace/.tandem/shots && timeout 60 chromium "
                             "--headless=new --no-sandbox --disable-gpu --enable-logging=stderr "
                             "--v=0 --window-size=800,600 --virtual-time-budget=2000 "
                             "--screenshot=/workspace/.tandem/shots/t.png "
                             "file:///workspace/site/index.html 2>&1 | grep -E 'CONSOLE|boom' "
                             "| head -5", timeout=90)
        png = os.path.join(d, "workspaces", CID, ".tandem", "shots", "t.png")
        assert os.path.getsize(png) > 1000 and open(png, "rb").read(4) == b"\x89PNG"
        assert "boom from the page" in out, out
        print("render_page: PNG", os.path.getsize(png), "bytes; console:", out.strip()[:120])

        # a PTY
        ex, sock, first = sb.open_exec(CID, ["bash", "-l"], tty=True, stdin=True)
        sock.sendall(b"echo $((6*7))\nexit\n")
        buf = first
        sock.settimeout(10)
        while b"42" not in buf:
            d2 = sock.recv(4096)
            if not d2:
                break
            buf += d2
        assert b"42" in buf
        sock.close()

        # the port relay to a server inside
        _run(sb, "cd /workspace/site && nohup python3 -m http.server 8123 --bind 127.0.0.1 "
                 "> /tmp/srv.log 2>&1 & sleep 1.5")
        ex, sock, first = sb.open_exec(CID, ["python3", "/opt/tandem/relay.py", "connect",
                                              "8123"], stdin=True)
        sock.sendall(b"GET /index.html HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        got = []
        demux(sock, first, lambda s, b: got.append(b) if s == 1 else None)
        resp = b"".join(got)
        assert resp.startswith(b"HTTP/1.0 200") and b"box test" in resp, resp[:200]
        print("port relay: 200 from 127.0.0.1:8123 inside the sandbox")

        # a second runtime adopts the running container
        sb2 = Sandboxes(d)
        assert sb2.adopt() >= 1 and sb2.box(CID).state == "running"
        sb2.box(CID).proxy and sb2.box(CID).proxy.close()

        sb.stop(CID, "user")
        st, info = sb.engine.call("GET", f"/containers/{ctr}/json")
        assert info["State"]["Running"] is False
        print("stopped")
    finally:
        sb.remove(CID, files=True)
        sb.stopping.set()
        shutil.rmtree(d, ignore_errors=True)


def test_every_builtin_through_chatd_in_a_real_sandbox():
    if not ON:
        print("skipped: set QSE_SANDBOX_BOX=1 on the box")
        return
    avail = sb_mod.meminfo()["MemAvailable"] / sb_mod.GiB
    need = sb_mod.FLOOR_GB + sb_mod.LIMIT_BYTES / sb_mod.GiB
    assert avail >= need, f"MemAvailable {avail:.1f} GiB: wait for {need:g}"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chatd_util import Chat

    def calls(*c):
        return "FAKE_CALLS " + json.dumps([{"name": n, "arguments": a} for n, a in c])

    c = Chat()
    sb = c.svc.sandboxes
    sb.max_running = 1
    cid = None
    try:
        assert sb.available(), "no sandbox runtime on this box"
        names = [t["id"] for t in c.ok("GET", "/tools")]
        for n in names:
            c.svc.add_rule(n, None, "allow", None)
        cid = c.new()
        page = ("<html><body><h1 id=t>box</h1><script>console.error('box-console-check');"
                "</script></body></html>")
        c.send(cid, calls(("write_file", {"path": "site/index.html", "content": page}),
                          ("write_file", {"path": "notes.md", "content": "alpha\nbeta\n"}),
                          ("bash", {"command": "id -u && ls /workspace/site && uname -m"})))
        first = c.wait_idle(cid, 180)["messages"][-1]
        c.send(cid, calls(("read_file", {"path": "notes.md"}),
                          ("glob", {"pattern": "**/*.html"}),
                          ("grep", {"pattern": "beta"}),
                          ("edit_file", {"path": "notes.md", "old": "beta", "new": "gamma"}),
                          ("web_fetch", {"url": "https://example.com/"}),
                          ("render_page", {"path": "site/index.html"}),
                          ("present", {"kind": "preview", "path": "site/index.html"})))
        second = c.wait_idle(cid, 240)["messages"][-1]
        got = {}
        for m in (first, second):
            for p in m["parts"]:
                if p["type"] == "tool_call":
                    got[p["name"]] = p
        for n, p in got.items():
            print(f"  {n:12s} {p['status']:7s} {(p.get('output') or {}).get('content', '')[:90]!r}")
        out = {n: (p.get("output") or {}).get("content", "") for n, p in got.items()}
        assert all(p["status"] == "done" for p in got.values()), {n: p["status"]
                                                                  for n, p in got.items()}
        assert "1000" in out["bash"] and "index.html" in out["bash"] and "aarch64" in out["bash"]
        assert "alpha" in out["read_file"] and "site/index.html" in out["glob"]
        assert "notes.md" in out["grep"] and "documentation examples" in out["web_fetch"]
        assert "box-console-check" in out["render_page"]
        f = c.ok("GET", f"/conversations/{cid}/sandbox/file?path=/workspace/notes.md")
        assert "gamma" in json.dumps(f) and "beta" not in json.dumps(f)
        lib = [n for n in ("library_search", "library_read") if n in names]
        print(f"every built-in ran; library tools offered: {lib or 'none (no /v1/library)'}")
    finally:
        if cid:
            c.call("POST", f"/conversations/{cid}/sandbox/stop")
            sb.remove(cid, files=True)
        c.close()


HUMANIZE = '''"""
title: Big numbers
requirements: humanize
"""
from pydantic import BaseModel


class Tools:
    class Valves(BaseModel):
        SEP: str = ","

    def __init__(self):
        self.valves = self.Valves()

    async def commas(self, n: int, __event_emitter__=None) -> str:
        """Write a number with thousands separators."""
        import humanize
        return humanize.intcomma(n).replace(",", self.valves.SEP)
'''


def test_an_open_webui_tool_runs_in_a_real_sandbox():
    """SRV-361: an imported Tools file's method in the chat's sandbox, its requirement installed
    with pip through chatd's proxy, pydantic there, a valve applied."""
    if not ON:
        print("skipped: set QSE_SANDBOX_BOX=1 on the box")
        return
    avail = sb_mod.meminfo()["MemAvailable"] / sb_mod.GiB
    need = sb_mod.FLOOR_GB + sb_mod.LIMIT_BYTES / sb_mod.GiB
    assert avail >= need, f"MemAvailable {avail:.1f} GiB: wait for {need:g}"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chatd_util import Chat

    c = Chat()
    sb = c.svc.sandboxes
    sb.max_running = 1
    cid = None
    try:
        kit = c.svc.owui.add("big.py", HUMANIZE)
        c.svc.owui.patch(kit["id"], {"valves": {"SEP": "."}})
        c.svc.add_rule("owui__big_numbers__*", None, "allow", None)
        cid = c.new()
        c.send(cid, "FAKE_CALLS " + json.dumps([{"name": "owui__big_numbers__commas",
                                                  "arguments": {"n": 1234567}}]))
        m = c.wait_idle(cid, 300)["messages"][-1]
        call = next(p for p in m["parts"] if p["type"] == "tool_call")
        print("  owui call:", call["status"], repr((call.get("output") or {}).get("content"))[:200])
        assert call["status"] == "done" and call["output"]["content"] == "1.234.567"
    finally:
        if cid:
            c.call("POST", f"/conversations/{cid}/sandbox/stop")
            sb.remove(cid, files=True)
        c.close()


def test_an_upload_does_not_follow_a_symlink_the_sandbox_made():
    """SEC-1: in a real sandbox, bash links /workspace/uploads to a host folder (by the path the
    host sees), then a message with an attachment is sent: the host folder stays empty, the message
    is sent and the attachment says why it is not in the workspace. Then a dangling link at the
    file's name: the file takes the next free name, the link's target is not created."""
    if not ON:
        print("skipped: set QSE_SANDBOX_BOX=1 on the box")
        return
    avail = sb_mod.meminfo()["MemAvailable"] / sb_mod.GiB
    need = sb_mod.FLOOR_GB + sb_mod.LIMIT_BYTES / sb_mod.GiB
    assert avail >= need, f"MemAvailable {avail:.1f} GiB: wait for {need:g}"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chatd_util import Chat
    from test_chatd_attach import upload

    def bash(command):
        return "FAKE_CALLS " + json.dumps([{"name": "bash", "arguments": {"command": command}}])

    def send_file(cid, n):
        att = upload(c, "notes.txt", "text/plain", b"the user's notes\n")
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": f"sec1-{n}", "text": "read it", "attachments": [att["id"]]})
        s = c.wait_idle(cid, 180)
        return [m for m in s["messages"] if m["role"] == "user"][-1]["attachments"][0]

    c = Chat()
    sb = c.svc.sandboxes
    sb.max_running = 1
    outside = tempfile.mkdtemp(prefix="sec1-outside-")
    cid = None
    try:
        assert sb.available(), "no sandbox runtime on this box"
        c.svc.add_rule("bash", None, "allow", None)
        cid = c.new()
        c.send(cid, bash(f"ln -s {outside} /workspace/uploads && ls -l /workspace"))
        c.wait_idle(cid, 180)
        a = send_file(cid, 1)
        print(f"  uploads/ a symlink: {a.get('unplaced')!r}; host folder {os.listdir(outside)}")
        assert os.listdir(outside) == [] and "path" not in a and "symlink" in a["unplaced"], a
        c.send(cid, bash(f"rm /workspace/uploads && mkdir /workspace/uploads && "
                         f"ln -s {outside}/planted.txt /workspace/uploads/notes.txt"))
        c.wait_idle(cid, 180)
        a = send_file(cid, 2)
        c.send(cid, bash("cat /workspace/uploads/notes-2.txt"))
        out = [p for p in c.wait_idle(cid, 180)["messages"][-1]["parts"]
               if p["type"] == "tool_call"][0]["output"]["content"]
        print(f"  a dangling link at the name: {a.get('path')}; host folder {os.listdir(outside)}")
        assert os.listdir(outside) == [] and a["path"] == "/workspace/uploads/notes-2.txt", a
        assert "the user's notes" in out, out
    finally:
        if cid:
            c.call("POST", f"/conversations/{cid}/sandbox/stop")
            sb.remove(cid, files=True)
        c.close()
        shutil.rmtree(outside, ignore_errors=True)


if __name__ == "__main__":
    test_the_sandbox_on_this_box()
    test_every_builtin_through_chatd_in_a_real_sandbox()
    test_an_open_webui_tool_runs_in_a_real_sandbox()
    test_an_upload_does_not_follow_a_symlink_the_sandbox_made()
    print("ok")
