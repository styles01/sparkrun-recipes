"""A throwaway Chat tab for the browser tests: the fake engine, a chatd on its own data directory
with its own sandbox label, and the front serving a dashboard build. Nothing of a running chatd is
read or touched: its chats, its sandboxes (another label) and its forward key stay as they are.

    python3 tests/chat_e2e_stack.py --front 127.0.0.1:4188 --dashboard dashboard/dist

The chats run in real sandbox containers when the box has the sandbox image (the terminal and the
preview need one); the model is tests/fake_backend.py, steered by the message text (FAKE_CALLS).
The data directory is removed on exit unless --keep.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import fake_backend  # noqa: E402

import chatd.sandbox as sb  # noqa: E402

sb.LABEL = "tandem.chat.e2e"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--front", default="127.0.0.1:4188")
    ap.add_argument("--dashboard", default=os.path.join(ROOT, "dashboard", "dist"))
    ap.add_argument("--port", type=int, default=8096, help="chatd's own port")
    ap.add_argument("--data", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    fake, fport = fake_backend.start(models=["fake-a"], tps=400.0)
    data = a.data or tempfile.mkdtemp(prefix="chat-e2e-")
    os.environ.pop("QSE_CHAT_FORWARD_KEY", None)
    from chatd.__main__ import main as chatd_main
    try:
        return chatd_main(["--port", str(a.port), "--data", data, "--engine",
                           f"http://127.0.0.1:{fport}", "--no-titles", "--front", a.front,
                           "--dashboard", a.dashboard])
    finally:
        fake.shutdown()
        remove_boxes()
        if not a.keep and not a.data:
            shutil.rmtree(data, ignore_errors=True)


def remove_boxes() -> None:
    """The containers of this label (the tests' chats), never another chatd's."""
    import json
    import urllib.parse
    eng = sb.Engine()
    try:
        st, rows = eng.call("GET", "/containers/json?all=1&filters=" +
                            urllib.parse.quote(json.dumps({"label": [sb.LABEL]})))
        for r in rows if st == 200 and isinstance(rows, list) else []:
            if (r.get("Labels") or {}).get(sb.LABEL):
                eng.call("DELETE", f"/containers/{r['Id']}?force=1", timeout=30)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
