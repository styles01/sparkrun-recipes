"""The model's write channel to its memory: one hidden block in its answer (2026-10-05).

The design: Memo "Memory for TandemLLM: design advice" (2026-10-05), section 2. The model writes,
in its answer and usually after its text:

    <tandem_memory op="add" scope="project" kind="gotcha">
    title: the summer scene differs because offerings.js recolours by season
    tags: pagoda, checker

    Gate (a) wants the summer scene within 1.0 of night 1; ...
    </tandem_memory>

`op` is add, update (with `id:`), archive (with `id:` and `reason:`) or propose (P-M3: a change
to a package, with `name:`). An attribute in the opener and a `key: value` line at the head of
the block say the same; the body is what follows the first blank line. The engine holds the block
back the way it holds a `<tool_call>` (`MemFilter`, beside server/toolcall.py's buffer), so the
client never sees it: the stream is the answer alone, plus one state line `› remembered: ...`.

A CLOSED BLOCK IS NEVER SHOWN (SRV-277, 2026-10-08). Every closed block is held back from the
client, whether it was written or not: one over 2,000 characters (`too_long`), the fifth of a turn
(`too_many`), one the schema or the store refused (`schema`, `cap`, `secret`, ...). Before, a
refused block went out as the text it was, so an Open WebUI chat (no working directory, so no
project scope) showed the model's `<tandem_memory scope="project">` markup to the user. The journal
and the log say why a block was not written; `hidden` lists every block kept from the client (its
tokens are hidden spans, as a written one's). Only text that is not a closed block goes out as
written: one that never closed (`unclosed`), one still open past `HOLD_CHARS` (`too_long`: the
model is not writing a note any more), `<tandem_memoryx`. Only the answer writes: a block inside
the reasoning is a thought about remembering, as a call inside the reasoning is a thought about
calling, and stays as the model wrote it.

THE CACHE. The rows hold the block; the client's copy of the answer does not. `token_spans()`
finds the block's tokens in the generation, and the server records them as hidden spans, which
the end anchor's replay puts back on the next turn (engine/cache.py `ResidentPrefix.replay`), as
it does for the monitor's hidden close. A block whose edges fall inside a token that also holds
visible text is not recorded (`unaligned`): the next turn then prefills from there, correct and
a little slower.

Standard library only.
"""

from __future__ import annotations

import re

OPEN = "<tandem_memory"
CLOSE = "</tandem_memory>"
MAX_BLOCKS = 4
MAX_CHARS = 2000
HOLD_CHARS = 2 * MAX_CHARS       # how long an open block is held back before it is text after all
OPS = ("add", "update", "archive", "propose")
FIELDS = ("op", "scope", "kind", "title", "tags", "id", "reason", "verify_by", "name", "mode")
_ATTR = re.compile(r"""([a-z_]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))""")
_HEAD = re.compile(r"^(" + "|".join(FIELDS) + r")\s*:\s*(.*)$")
BLOCK = re.compile(re.escape(OPEN) + r"(?=[\s>])[\s\S]*?" + re.escape(CLOSE))


class BlockError(ValueError):
    def __init__(self, why: str, message: str):
        super().__init__(message)
        self.why = why


def parse(raw: str) -> dict:
    """One closed block's op: `{op, scope, kind, title, tags, id, reason, verify_by, name,
    body}`; `BlockError` when it is not one."""
    if not raw.startswith(OPEN) or not raw.endswith(CLOSE):
        raise BlockError("schema", "not a closed <tandem_memory> block")
    gt = raw.find(">", len(OPEN))
    if gt < 0 or gt > len(raw) - len(CLOSE):
        raise BlockError("schema", "the opener has no '>'")
    op: dict = {}
    for m in _ATTR.finditer(raw[len(OPEN):gt]):
        k = m.group(1)
        if k in FIELDS:
            op[k] = next(g for g in m.groups()[1:] if g is not None)
    inner = raw[gt + 1:len(raw) - len(CLOSE)].strip("\n")
    lines = inner.split("\n")
    i = 0
    while i < len(lines):
        m = _HEAD.match(lines[i].strip())
        if not m:
            break
        op.setdefault(m.group(1), m.group(2).strip())
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    op["body"] = "\n".join(lines[i:]).strip()
    op["op"] = (op.get("op") or "add").strip()
    if op["op"] not in OPS:
        raise BlockError("schema", f"op: one of {', '.join(OPS)}")
    if op["op"] in ("update", "archive") and not op.get("id"):
        raise BlockError("schema", f"{op['op']} needs id: the note's id")
    if op["op"] == "propose" and not op.get("name"):
        raise BlockError("schema", "propose needs name: the package or skill it changes")
    return op


def strip_blocks(text: str) -> tuple[str, list[str]]:
    """`text` without its closed blocks, and the blocks (the non-streamed answer)."""
    found = BLOCK.findall(text or "")
    return (BLOCK.sub("", text) if found else text), found


def _partial_open(s: str) -> int:
    """The length of the longest end of `s` that the opener could still begin with."""
    for n in range(min(len(s), len(OPEN)), 0, -1):
        if OPEN.startswith(s[-n:]):
            return n
    return 0


class MemFilter:
    """The answer's text with the closed blocks taken out, one piece at a time.

    `on_block(raw)` is called once a block closes; it returns None when the block was taken
    (written), else the reason it was not. Either way a closed block is hidden (SRV-277).
    `feed()` returns the text the client gets now; `finish()` the rest (an unclosed block as
    text)."""

    def __init__(self, on_block):
        self.on_block = on_block
        self.pending = ""
        self.inside = False
        self.blocks = 0                  # blocks seen closed this turn
        self.taken: list[str] = []       # the raw text of the blocks that were written
        self.hidden: list[str] = []      # every closed block kept from the client (taken or not)
        self.failed: list[str] = []      # why each block that was not written was not
        self.shown: list[str] = []       # why each piece that looked like a block went out as text

    def feed(self, piece: str) -> str:
        if not piece and not self.pending:
            return ""
        self.pending += piece
        out = []
        while self.pending:
            if not self.inside:
                i = self.pending.find(OPEN)
                if i < 0:
                    keep = _partial_open(self.pending)
                    out.append(self.pending[:len(self.pending) - keep])
                    self.pending = self.pending[len(self.pending) - keep:]
                    break
                after = self.pending[i + len(OPEN):i + len(OPEN) + 1]
                if not after:
                    out.append(self.pending[:i])
                    self.pending = self.pending[i:]
                    break                       # the character after the opener decides
                if not (after.isspace() or after == ">"):
                    out.append(self.pending[:i + 1])
                    self.pending = self.pending[i + 1:]
                    continue                    # `<tandem_memoryx`: not ours
                out.append(self.pending[:i])
                self.pending = self.pending[i:]
                self.inside = True
            j = self.pending.find(CLOSE)
            if j < 0:
                if len(self.pending) > HOLD_CHARS + len(CLOSE):
                    self.failed.append("too_long")
                    self.shown.append("too_long")
                    out.append(self.pending)
                    self.pending, self.inside = "", False
                break
            raw = self.pending[:j + len(CLOSE)]
            self.pending = self.pending[j + len(CLOSE):]
            self.inside = False
            self.blocks += 1
            why = ("too_long" if len(raw) > MAX_CHARS else
                   "too_many" if self.blocks > MAX_BLOCKS else self.on_block(raw))
            self.hidden.append(raw)
            if why is None:
                self.taken.append(raw)
            else:
                self.failed.append(why)
        return "".join(out)

    def finish(self) -> str:
        rest, self.pending = self.pending, ""
        if self.inside:
            self.failed.append("unclosed")
            self.shown.append("unclosed")
        self.inside = False
        return rest


def token_spans(ids: list[int], decode, raws: list[str]) -> tuple[list[tuple[int, int]], int]:
    """The token ranges `[k0, k1)` of `ids` that hold exactly the taken blocks (with whitespace
    around them at most), in order; and how many blocks could not be aligned to whole tokens.
    `decode(ids)` is the tokenizer's decode without special tokens."""
    if not raws or not ids:
        return [], 0
    text = decode(ids)
    lens: dict[int, int] = {}

    def at(k: int) -> int:
        if k not in lens:
            lens[k] = len(decode(ids[:k]))
        return lens[k]

    def first_k(c: int, lo: int) -> int:
        """The smallest k >= lo whose prefix decodes to at least `c` characters."""
        hi = len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if at(mid) >= c:
                hi = mid
            else:
                lo = mid + 1
        return lo

    out, bad, pos, k_lo = [], 0, 0, 0
    for raw in raws:
        c0 = text.find(raw, pos)
        if c0 < 0:
            bad += 1
            continue
        c1 = c0 + len(raw)
        pos = c1
        k0 = first_k(c0 + 1, k_lo) - 1            # the token the block's first character is in
        k1 = first_k(c1, max(k0, k_lo))
        if k0 < 0 or k1 > len(ids) or k1 <= k0:
            bad += 1
            continue
        got = decode(ids[k0:k1])
        if got.strip() != raw.strip() or not _only_space_around(got, raw):
            bad += 1
            continue
        out.append((k0, k1))
        k_lo = k1
    return out, bad


def _only_space_around(got: str, raw: str) -> bool:
    i = got.find(raw)
    return i >= 0 and not got[:i].strip() and not got[i + len(raw):].strip()
