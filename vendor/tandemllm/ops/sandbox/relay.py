"""The sandbox's one network helper, baked into tandem-sandbox:1 at /opt/tandem/relay.py.

The container runs with no network but its loopback. Two jobs:

    relay.py serve          (the container's main process) listen on 127.0.0.1:3128 and pass every
                            connection to chatd's proxy for this chat, /run/tandem/proxy.sock
                            (HTTP_PROXY and HTTPS_PROXY point here; chatd/egress.py checks where
                            each one goes: the internet yes, the LAN and the host no)
    relay.py connect PORT   (an exec per request) connect to localhost:PORT and pipe stdin to it
                            and its answer to stdout: chatd's way to a dev server in the sandbox
                            for the preview pane. ::1, then 127.0.0.1, by address (not /etc/hosts).
                            Seen: Vite 5 on `localhost` listened on [::1]:5173 only
                            (Node 17+ takes the resolver's order) and a dial to 127.0.0.1 alone
                            was refused while curl in the sandbox answered 200
    relay.py ports [--http] the TCP ports something listens on in the sandbox, as JSON: [{port,
                            addresses, command, cwd, started}] (render_page's check and its
                            error, the preview's choice of server); --http adds `http`: whether
                            the server answered an HTTP request within 1.5 s
    relay.py probe PORT PATH  GET PATH from localhost:PORT and, for a page, its scripts and style
                            sheets, as JSON: {status, type, title, assets: [{url, status, type}]}
                            (what `present` tells the model the server answered)

chatd sends this file with each exec (`python3 -c <this file> connect PORT`, chatd/sandbox.py
`relay_cmd`), so a container made from an older image dials the same way.
"""

import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse

SOCK = "/run/tandem/proxy.sock"


def pump(src, dst):
    try:
        while True:
            d = src.recv(65536)
            if not d:
                break
            dst.sendall(d)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def handle(c):
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        u.connect(SOCK)
    except OSError:
        msg = b"chatd's proxy is not there; the network is off for this sandbox\n"
        try:
            c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nContent-Length: " +
                      str(len(msg)).encode() + b"\r\nConnection: close\r\n\r\n" + msg)
        except OSError:
            pass
        u.close()          # connect failed, but the socket object still owns an fd
        c.close()
        return
    t = threading.Thread(target=pump, args=(u, c), daemon=True)
    t.start()
    pump(c, u)
    t.join(600)
    c.close()
    u.close()


def serve():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 3128))
    s.listen(128)
    while True:
        # a connect that fails at the fd ceiling leaks one fd per request (before this,
        # handle returned without closing u); at the ceiling accept() itself then raises
        # EMFILE -- and an error escaping this loop ends PID-1, the sandbox's whole
        # network path, until the container is recreated
        try:
            c, _ = s.accept()
        except OSError:
            time.sleep(0.05)          # at the fd ceiling this repeats; a bare continue spins
            continue
        try:
            threading.Thread(target=handle, args=(c,), daemon=True).start()
        except RuntimeError:          # can't spawn: don't leak the client fd
            try:
                c.close()
            except OSError:
                pass


def _addr(hexaddr):
    b = bytes.fromhex(hexaddr)
    if len(b) == 4:
        return socket.inet_ntop(socket.AF_INET, b[::-1])
    return socket.inet_ntop(socket.AF_INET6, b"".join(b[i:i + 4][::-1] for i in range(0, 16, 4)))


def listening(proc="/proc"):
    """[{port, addresses, command, cwd, started}] by port: the listening TCP sockets of
    /proc/net/tcp and tcp6, each with the command line, working directory and start (clock ticks
    since boot, /proc/PID/stat) of the process that holds it (when it is ours to read). The relay's
    own proxy port is left out."""
    socks = {}
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc, "net", name)) as fh:
                rows = fh.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) > 9 and f[3] == "0A":
                a, p = f[1].split(":")
                socks[f[9]] = (int(p, 16), _addr(a))
    owners = {}
    try:
        pids = os.listdir(proc)
    except OSError:
        pids = []
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(os.path.join(proc, pid, "fd"))
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(os.path.join(proc, pid, "fd", fd))
            except OSError:
                continue
            ino = link[8:-1] if link.startswith("socket:[") else None
            if ino in socks and ino not in owners:
                owners[ino] = _owner(os.path.join(proc, pid))
    out = {}
    for ino, (port, addr) in socks.items():
        cmd, cwd, started = owners.get(ino, ("", "", 0))
        if "relay.py serve" in cmd:
            continue
        row = out.setdefault(port, {"port": port, "addresses": [], "command": "", "cwd": "",
                                    "started": 0})
        if addr not in row["addresses"]:
            row["addresses"].append(addr)
        row["command"] = row["command"] or cmd[:120]
        row["cwd"] = row["cwd"] or cwd
        row["started"] = row["started"] or started
    return [out[p] for p in sorted(out)]


def _owner(d):
    """(command line, working directory, start in clock ticks) of the process at /proc/PID `d`;
    what cannot be read stays empty."""
    cmd, cwd, started = "", "", 0
    try:
        with open(os.path.join(d, "cmdline"), "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        pass
    try:
        cwd = os.readlink(os.path.join(d, "cwd"))
    except OSError:
        pass
    try:
        with open(os.path.join(d, "stat")) as fh:
            started = int(fh.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        pass
    return cmd, cwd, started


def answers_http(port):
    try:
        s = dial(port)
    except OSError:
        return False
    try:
        s.settimeout(1.5)
        s.sendall(f"GET / HTTP/1.0\r\nHost: localhost:{port}\r\n\r\n".encode())
        return s.recv(5) == b"HTTP/"
    except OSError:
        return False
    finally:
        s.close()


def ports_words(ports):
    if not ports:
        return "nothing listens on any port in the sandbox"
    return "listening now: " + ", ".join(
        f"{r['port']} on {' and '.join(r['addresses'])}" + (f" ({r['command']})" if r["command"]
                                                              else "") for r in ports)


def dial(port):
    """::1, then 127.0.0.1, by address: /etc/hosts may map localhost to 127.0.0.1 only."""
    for host in ("::1", "127.0.0.1"):
        try:
            return socket.create_connection((host, port), timeout=10)
        except OSError:
            pass
    raise OSError(f"nothing listens on port {port}")


def connect(port):
    try:
        s = dial(port)
    except OSError:
        msg = (f"Nothing listens on port {port} in the sandbox (dialed localhost: ::1 and "
               f"127.0.0.1); {ports_words(listening())}.\n").encode()
        os.write(1, b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nContent-Length: " +
                 str(len(msg)).encode() + b"\r\n\r\n" + msg)
        return
    s.settimeout(None)

    def up():
        try:
            while True:
                d = os.read(0, 65536)
                if not d:
                    break
                s.sendall(d)
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    threading.Thread(target=up, daemon=True).start()
    try:
        while True:
            d = s.recv(65536)
            if not d:
                break
            os.write(1, d)
    except OSError:
        pass


class _Local(http.client.HTTPConnection):
    """An HTTP connection to localhost:PORT that dials as `dial` does."""

    def connect(self):
        self.sock = dial(self.port)
        self.sock.settimeout(self.timeout)


_ASSET = re.compile(r"""<(script|link)\b([^>]*)>""", re.I)
_ATTR = re.compile(r"""([a-zA-Z-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def fetch(port, path, limit=2 * 1024 * 1024):
    """(status, content type, body up to `limit`) of GET `path`; status 0 and the reason as the
    type when the server did not answer."""
    c = _Local("localhost", port, timeout=10)
    try:
        c.request("GET", path, headers={"Host": f"localhost:{port}", "Accept": "*/*"})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read(limit)
    except (OSError, http.client.HTTPException) as exc:
        return 0, f"{type(exc).__name__}: {exc}"[:160], b""
    finally:
        c.close()


def probe(port, path="/"):
    """What the server at `port` answers for `path`, and for a page each of its scripts and style
    sheets on the same server (at most 20): {status, type, title, assets: [{url, status, type}]}."""
    status, ctype, body = fetch(port, path or "/")
    out = {"port": port, "path": path or "/", "status": status, "type": ctype, "title": "",
           "assets": []}
    if not ctype.startswith("text/html"):
        return out
    doc = body.decode("utf-8", "replace")
    m = _TITLE.search(doc)
    out["title"] = " ".join(m.group(1).split())[:120] if m else ""
    for tag, attrs in _ASSET.findall(doc)[:40]:
        a = {k.lower(): (v1 or v2 or v3) for k, v1, v2, v3 in _ATTR.findall(attrs)}
        url = a.get("src") if tag.lower() == "script" else (
            a.get("href") if "stylesheet" in (a.get("rel") or "").lower() else None)
        if not url:
            continue
        u = urllib.parse.urlsplit(urllib.parse.urljoin(f"http://localhost:{port}{path or '/'}",
                                                       url))
        if u.hostname not in ("localhost", "127.0.0.1", "::1") or (u.port or 80) != port:
            continue
        st, ct, _ = fetch(port, (u.path or "/") + (f"?{u.query}" if u.query else ""), 0)
        out["assets"].append({"url": url, "status": st, "type": ct})
        if len(out["assets"]) >= 20:
            break
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        serve()
    elif sys.argv[1:2] == ["connect"]:
        connect(int(sys.argv[2]))
    elif sys.argv[1:2] == ["ports"]:
        rows = listening()
        if "--http" in sys.argv[2:]:
            for r in rows:
                r["http"] = answers_http(r["port"])
        print(json.dumps(rows))
    elif sys.argv[1:2] == ["probe"]:
        print(json.dumps(probe(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "/")))
    else:
        sys.exit("relay.py serve | connect PORT | ports | probe PORT [PATH]")
