"""grep on one file and on a folder with large files (SRV-649).

grep given one file's path walked it as a folder and answered "is not a directory", and a folder
walk left files over 2 MiB out with "no matches". One file is now searched whole, line by line,
and a folder's result names the files it did not search. Each test fails on 0.12.2.

Run: python -m pytest tests/test_chatd_grep_file.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import tools as T                                        # noqa: E402
from chatd.workspace import MAX_READ, Workspace                     # noqa: E402


class _Ctx:
    def __init__(self, ws, agents=False, cap=None):
        self.ws, self.agents, self.cap = ws, agents, cap

    def changed(self, path):
        pass


def _ws() -> Workspace:
    return Workspace(tempfile.mkdtemp(prefix="large-"))


def _big(ws: Workspace, name: str = "big.log", lines: int = 90_000) -> list[str]:
    """About 6.4 MiB in lines of 70-odd bytes, each naming its own number."""
    text = [f"row {i:06d} " + "x" * (60 + i % 7) for i in range(1, lines + 1)]
    ws.write(name, ("\n".join(text) + "\n").encode())
    return text


def test_grep_searches_one_file_named_by_its_path():
    ws = _ws()
    ws.write("notes/todo.md", b"alpha\nbeta\nalpha two\n")
    out = T._grep(_Ctx(ws), {"pattern": "alpha", "path": "/workspace/notes/todo.md"}).content
    assert out == ("/workspace/notes/todo.md:1: alpha\n"
                   "/workspace/notes/todo.md:3: alpha two"), out
    assert T._grep(_Ctx(ws), {"pattern": "gamma", "path": "notes/todo.md"}).content == \
        "no matches"


def test_grep_reads_a_large_file_named_by_its_path_whole():
    ws = _ws()
    text = _big(ws)
    assert ws.stat("big.log").st_size > MAX_READ
    out = T._grep(_Ctx(ws), {"pattern": "row 089999 ", "path": "big.log"}).content
    assert out == f"/workspace/big.log:89999: {text[89_998]}", out[:200]


def test_a_folder_grep_names_the_large_files_it_did_not_search():
    ws = _ws()
    _big(ws)
    ws.write("small.txt", b"nothing here\n")
    out = T._grep(_Ctx(ws), {"pattern": "row 089999 "}).content
    assert out.startswith("no matches\n[not searched: 1 file(s) over 2 MiB: /workspace/big.log ("), out
    assert out.endswith("; grep one of them by its path to search it whole]"), out


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
