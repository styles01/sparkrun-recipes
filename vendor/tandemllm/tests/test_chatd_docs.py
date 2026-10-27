"""The docs for chatd's lifecycle and the Chat tab's down state (OPS-157, QA OPS-158; GitHub issue #3).

  1  docs/chat.md: chatd starts with tandem start, stops with tandem stop, tandem restart leaves it
     running, tandem chat start|stop|restart|status|logs, QSE_CHATD=off     test_the_normal_path
  2  docs/chat.md names the down states and the command for each, and that the mock loads only for
     ?chat=mock                                                              test_the_down_state
  3  no doc names a qwen38-spark-engine path for chatd; chatd.sh lives in ~/TandemLLM/src/ops
                                                                             test_paths_match_the_installer
  4  the manual's chapter and docs/operations.md say the same              test_the_manual_follows
  5  the CHANGELOG's newest entry: the section, the upgrade line, the credit, no aiohttp, no manual
     chatd restart                                                           test_the_patch_entry

Run: python tests/test_chatd_docs.py
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p: str) -> str:
    return open(os.path.join(ROOT, *p)).read()


def _section(text: str, start: str) -> str:
    i = text.index(start)
    j = text.find("\n## ", i + len(start))
    return text[i:j if j > 0 else None]


def test_the_normal_path():
    s = _section(_read("docs", "chat.md"), "## Starting and stopping")
    assert "chatd starts with the engine and stops with `tandem stop`" in s
    assert "`tandem\nrestart`" in s or "`tandem restart`" in s
    assert "leave it running" in s
    assert "tandem chat start | stop | restart" in s and "tandem chat status" in s and "tandem chat logs" in s
    assert "QSE_CHATD=off" in s


def test_the_down_state():
    s = _section(_read("docs", "chat.md"), "## When the Chat tab says the chat service is not running")
    for state, cmd in (("The chat service is not running.", "tandem chat start"),
                       ("does not accept this server's key.", "tandem chat restart && tandem restart"),
                       ("has no Chat tab server.", "run the installer again"),
                       ("Cannot reach the engine server.", "tandem status")):
        row = next(ln for ln in s.splitlines() if state in ln)
        assert cmd in row, row
    assert "`?chat=mock`" in s and "never falls back to an in-browser mock" in s


def test_paths_match_the_installer():
    for p in (("docs", "chat.md"), ("docs", "operations.md"), ("docs", "setup.md"), ("manual", "chapters", "chat.tex"),
              ("ops", "chatd.service")):
        text = _read(*p)
        for ln in text.splitlines():
            if "chatd" in ln:
                assert "qwen38-spark-engine" not in ln, (p, ln)
    assert "~/TandemLLM/src/ops/chatd.sh" in _read("docs", "chat.md")
    assert "bash ops/chatd.sh start             #" not in _read("docs", "chat.md")


def test_the_manual_follows():
    tex = _read("manual", "chapters", "chat.tex")
    assert r"chatd starts with the engine (\code{tandem start}" in tex and r"\code{tandem stop}" in tex
    assert r"\code{tandem restart} and every engine restart leave it running" in tex
    assert "tandem chat start | stop | restart | logs" in tex and "QSE_CHATD=off" in tex
    ops = _read("docs", "operations.md")
    assert "starts a dead chatd (at most 3 times an hour)" in ops
    assert "never restarts the engine for it" in ops and "`tandem stop` stops it" in ops
    assert "Setup mode does not start chatd" in _read("docs", "setup.md")


def test_the_patch_entry():
    log = _read("CHANGELOG.md")
    head = re.search(r"\n## (0\.11\.2[^\n]*)\n(.*?)\n## ", log, re.S)
    assert head, "no 0.11.2 entry on top"
    entry = head.group(2)
    assert log.index("## 0.11.2") < log.index("## 0.11.1")
    assert "### Chat tab and install" in entry
    assert "Upgrade: run the installer again; chatd then starts with the engine" in entry
    assert "`tandem chat restart`" in entry and "Reported by Aron (issue #3)." in entry
    assert "aiohttp" not in entry.lower()
    assert "chatd.sh stop" not in entry


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:50s} ok")
            n += 1
    print(f"{n} passed")
