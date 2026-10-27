"""read_file on a file over its 4 MiB window (SRV-651).

read_file decoded the first 4 MiB of any file and paged inside them as if the file ended there:
the last page of the window said nothing more, and an offset past the window came back empty. A
large file is now read 4 MiB at a time from the line `offset` names, numbered as in the file, and
the marker names the bytes left. Each test but the small-file one fails on 0.12.2.

Run: python -m pytest tests/test_chatd_large_read.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatd import tools as T                                        # noqa: E402
from chatd.workspace import Workspace                                # noqa: E402


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


_ROW = re.compile(r"^\s*(\d+)\t(.*)$")


def test_a_large_file_pages_to_its_end_with_the_files_line_numbers():
    ws = _ws()
    text = _big(ws)
    size = ws.stat("big.log").st_size
    assert size > T.READ_WINDOW
    ctx = _Ctx(ws)
    first = T._read(ctx, {"path": "big.log"}).content
    assert first.startswith("     1\trow 000001 "), first[:40]
    marker = first.rsplit("\n", 1)[-1]
    m = re.fullmatch(r"\[truncated at 64,000 characters: lines 1-(\d+) shown, ([\d,]+) more "
                     r"bytes of ([\d,]+); read on with offset=(\d+)\]", marker)
    assert m, marker
    assert int(m.group(4)) == int(m.group(1)) + 1 and m.group(3) == f"{size:,}"
    # paging by the offsets the markers name rebuilds the file line for line, past 4 MiB
    seen, off, reads = [], 1, 0
    while True:
        out = T._read(ctx, {"path": "big.log", "offset": off}).content
        reads += 1
        rows = [ln for ln in out.split("\n") if _ROW.match(ln)]
        for ln in rows:
            no, body = _ROW.match(ln).groups()
            assert int(no) == len(seen) + 1, (no, len(seen))
            seen.append(body)
        tail = out.rsplit("\n", 1)[-1]
        got = re.search(r"read on with offset=(\d+)\]$", tail)
        if not got:
            break
        off = int(got.group(1))
    assert seen == text, (len(seen), len(text))
    assert reads > 60
    # past the end: nothing, and no promise of more
    assert T._read(ctx, {"path": "big.log", "offset": len(text) + 1}).content == ""
    return f"{size:,} bytes in {reads} reads, every line once, numbered as in the file"


def test_a_line_past_the_first_4_mib_reads_by_offset():
    ws = _ws()
    text = _big(ws)
    out = T._read(_Ctx(ws), {"path": "big.log", "offset": 80_000, "limit": 3}).content
    rows = out.split("\n")
    assert rows[0] == f" 80000\t{text[79_999]}", rows[0]
    assert rows[2] == f" 80002\t{text[80_001]}"
    assert re.fullmatch(r"\[lines 80000-80002 shown of a [\d,]+-byte file, [\d,]+ more bytes "
                        r"after them; read on with offset=80003\]", rows[3]), rows[3]


def test_agents_mode_names_the_bytes_after_the_page():
    ws = _ws()
    _big(ws)
    out = T._read(_Ctx(ws, agents=True), {"path": "big.log", "offset": 70_000}).content
    assert out.startswith(" 70000\trow 070000 ")
    assert re.search(r"Showing lines 70000-\d+ of a [\d,]+-byte file, [\d,]+ bytes after them\. "
                     r"Read a range with offset and limit, or give the reading to a worker\.$",
                     out), out[-200:]


def test_one_line_longer_than_the_window_says_so():
    ws = _ws()
    ws.write("one.txt", b"y" * (T.READ_WINDOW + 500_000))
    out = T._read(_Ctx(ws), {"path": "one.txt"}).content
    assert out.startswith("     1\tyyy")
    assert "goes on past the 4 MiB read_file reads at once" in out.rsplit("\n", 1)[-1], out[-300:]


def test_a_small_file_reads_as_before():
    ws = _ws()
    ws.write("s.txt", b"a\nb\nc\n")
    assert T._read(_Ctx(ws), {"path": "s.txt"}).content == "     1\ta\n     2\tb\n     3\tc"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
