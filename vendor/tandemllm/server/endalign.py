"""The end anchor for a copy whose whitespace is not the rows'.

THE MEASUREMENT. Within an opencode session some tool turns resumed at the previous prompt's last
1,024 line instead of the end of the last generation (2026-10-03 09:33: 73,374 prompt tokens,
68,608 from the resident rows, 4,766 re-read; 09:39: 88,300 / 83,968 / 4,332; TTFT 4 to 7 s). In
the box's suffix store (the last 4,000 documents, token ids only) the follow-up's copy of the
previous output held it token for token in 2,723 turns and parted inside it in 673; of the turns
that part inside a generation, the place is a tool call's parameter (220) or the `</think>` line
(34), and in 221 of those 254 the copy differs from what the engine wrote in whitespace alone:

  * a parameter value: the model writes `<parameter=oldString>\\n      x = 1\\n\\n</parameter>`;
    server/toolcall.py hands the client the value trimmed (`x = 1`), and the template renders it
    back as `<parameter=oldString>\\nx = 1\\n</parameter>` -- the indentation and one newline
    are gone from the copy;
  * the reasoning: the model writes `... done</think>` without a newline; the template writes
    `reasoning|trim + "\\n</think>"` -- one newline more in the copy.

The rest (33) were a boolean the parser typed (`True` -> `true`) and calls of one turn sent back in
another order: not whitespace, served as sent.

THE FIX. `EndAlign.replay` comes after `ResidentPrefix.replay` (the hidden spans): when
the client's copy parts from the rows inside a generation the engine wrote -- the last one, or an
earlier one this alignment already served -- and the two are the same text once every ASCII
whitespace byte is taken out of both, the prompt is served with the rows' tokens up to the end
anchor and the client's after it, and resumes there. That is D2's rule for the whitespace next to
a hidden span, widened to the whole of the model's own output: what differs is whitespace our own
parser and the template took out of the model's tokens, never a character the client wrote. Every
non-whitespace byte must be the rows' own, in order, and everything outside a generation must be
the rows' token for token; anything else is served as the client sent it. `usage.prompt_tokens`
stays the client's count; the journal row says `extra.end_aligned` (the tokens served from the
rows in the regions).

A generation's region is `[its prompt's length, the end anchor)` in the rows; the regions travel
with the end anchor (`kept`, from `_remember`): the regions of the served prompt (still where they
were, the prompt starting with the old rows) and the new generation's. A generation the copy
re-renders the same way every turn is aligned on every turn after it, as the hidden spans are
replayed.

THE REASONING (SRV-239, 2026-10-07). Six of six Open WebUI chats whose first answer thought
re-read that answer on their first follow-up (cached 6,144 of 6,882 to 8,621 tokens, TTFT 2.5 to
5.0 s), while the chats whose first answer did not think resumed at its end. Open WebUI does not
send an answer's reasoning back, and the Qwen3.8 template (`preserve_thinking` on by default)
renders such an assistant message as `<think>\n\n</think>\n\n` + the answer: the copy holds an
empty block where the rows hold the reasoning, it parts from them right after `<think>`, and the
resume falls back to the grid line below (the head's 6,144). A region therefore remembers where
its reasoning ends (its first `</think>`, `_think_end`), and when the copy does not align with
the reasoning in it, a second pass aligns it with every region's reasoning taken out (the
whitespace before it included): every other non-whitespace byte must still be the rows' own, in
order. The prompt is then served with the rows' tokens up to the end anchor, the reasoning the
engine wrote included -- the template's own default, which renders an earlier answer's reasoning
when the client sends it -- and the turn resumes at the end anchor. A client that sends the
reasoning back (opencode) aligns on the first pass as before. `QWEN38_END_ALIGN_REASONING=off`:
the first pass alone.

Exactness: what is served up to the end anchor is the context the engine had, the session cache's
resume, held to the greedy gate as every end-anchor resume is (docs/caches.md). The alignment
reads token ids and the tokenizer's bytes; it writes nothing the client sees. Off (`--end-align
off`): `ResidentPrefix.replay` alone, as before.
"""

from __future__ import annotations

WS = b" \t\n\r"
THINK_END = b"</think>"
#: engine/vision.py's KEY_BASE: ids at or above it are image rows' content keys
KEY_BASE = 1 << 40


def _byte_decoder() -> dict:
    """GPT-2's byte-level alphabet backwards: the character a token string uses for each byte."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok):
    """`id -> bytes` for a byte-level BPE tokenizer (Qwen's): the token's own bytes, a special
    token's text; memoised. A tokenizer without `convert_ids_to_tokens` decodes one id."""
    dec, memo = _byte_decoder(), {}
    conv = getattr(tok, "convert_ids_to_tokens", None)

    def get(i: int) -> bytes:
        b = memo.get(i)
        if b is None and int(i) >= KEY_BASE:
            # SRV-546: an image row's key id (engine/vision.py `key_id`, past the vocabulary):
            # content, never whitespace, and equal only to the same image's rows
            b = memo[i] = b"\x00img" + int(i).to_bytes(8, "little")
        if b is None:
            s = conv(int(i)) if conv is not None else None
            if isinstance(s, str) and s and all(ch in dec for ch in s):
                b = bytes(dec[ch] for ch in s)
            elif isinstance(s, str):
                b = s.encode()
            else:
                b = tok.decode([int(i)], skip_special_tokens=False).encode()
            memo[i] = b
        return b
    return get


def _bare(b: bytes) -> bytes:
    return b.translate(None, WS)


def _common(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i, step = 0, 4096
    while i < n:
        j = min(n, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return n


class EndAlign:
    """The whitespace alignment of a copy against the end anchor's rows, and the regions it may
    use. One per server; called under the engine lock, as the replay is."""

    def __init__(self, tbytes, on: bool = True, reasoning: bool = True, moved: bool = True):
        self.tbytes = tbytes
        self.on = bool(on)
        self.reasoning = bool(reasoning)  # SRV-239: the second pass, without the reasoning
        self.moved = bool(moved)          # FX16: the third pass, a client's block that moved
        # the chat template's marks (`<|im_start|>`, the ids of `user`, `<|im_end|>`), for the
        # third pass; None (the server sets them) leaves it out
        self.marks: tuple | None = None
        self._for = None             # the end anchor (the `ResidentPrefix.end` dict) of `_regions`
        # (start, end, the position of the region's first `</think>` or -1): rows positions
        self._regions: list[tuple[int, int, int]] = []
        self._carry: list[tuple[int, int, int]] = []    # the served prompt's, for the next `kept`
        self.stats = {"aligned": 0, "aligned_tokens": 0, "refused": 0, "regions": 0,
                      "aligned_reasoning": 0, "reasoning_tokens": 0,
                      "aligned_moved": 0, "moved_tokens": 0, "partial": 0, "partial_tokens": 0}
        # SRV-657: the rows ranges the first pass found token for token in the copy, (rows start,
        # rows end, copy start), for a partial alignment (`_partial`); and the rows line the last
        # `replay` served a partial alignment up to (None: a whole one, or none)
        self._segs: list[tuple[int, int, int]] = []
        self.last_partial: int | None = None
        # the rows ranges the last `replay` served that the client's copy left out (SRV-233)
        self.last_omitted: list[tuple[int, int]] = []
        # FX16: the rows ranges of the client's own block the last `replay` served from the rows
        # (the copy had moved them to its last message), and why the last copy was refused
        # (`why`, the position it parted at, the end), for the log line of a re-read
        self.last_moved: list[tuple[int, int]] = []
        self.last_refusal: dict | None = None

    def regions(self, res) -> list[tuple[int, int, int]]:
        """The end anchor's generation regions (rows positions), when they are its."""
        end = getattr(res, "end", None)
        return list(self._regions) if end is not None and end is self._for else []

    def _think_end(self, rows: list, a: int, b: int) -> int:
        """The position of the first `</think>` in rows [a, b), -1 when the region has none (a turn
        that did not think: its prompt closed the block)."""
        tb = self.tbytes
        for p in range(a, min(b, len(rows))):
            if tb(rows[p]) == THINK_END:
                return p
        return -1

    def replay(self, res, ids: list[int]) -> tuple[list[int], list, int]:
        """`(ids, hidden spans put back, tokens aligned)`: `res.replay`'s answer when it has one
        (or when off), else the copy aligned to the rows where only whitespace differs."""
        out, spans = res.replay(ids)
        aligned = 0
        self.last_omitted, self.last_partial = [], None
        if not spans and self.on and res.end is not None:
            got = self._align(res, ids)
            if got is not None:
                out, aligned = got
                spans = list(res.end["hidden"])
            else:
                part = self._partial(res, ids)
                if part is not None:
                    out, aligned, x = part
                    spans = [(p, h) for p, h in res.end["hidden"] if p + len(h) <= x]
                    self.last_partial = x
        self._carry = self._carried(res, out)
        return out, spans, aligned

    def _carried(self, res, served: list[int]) -> list[tuple[int, int]]:
        regs = self.regions(res)
        if not regs or res.end is None:
            return []
        e = res.end["at"]
        if self.last_partial is not None:
            # a partial alignment served the rows up to its line: the regions below it stay
            x = self.last_partial
            return [r for r in regs if r[1] <= x] if served[:x] == res.tokens[:x] else []
        return regs if len(served) >= e and served[:e] == res.tokens[:e] else []

    def _partial(self, res, copy: list[int]):
        """SRV-657: when no pass aligns the whole copy, the longest part that the first pass did:
        the rows up to the highest resident anchor inside a run the copy holds token for token
        (`_segs`), the copy after the matching position. A generation whose tool call the parser
        re-wrote (a boolean typed, keys in another order: not whitespace) refused the whole
        alignment, and the turn resumed at the grid line below the FIRST place the copy parts
        from the rows -- an early answer's whitespace that every turn had aligned -- and read the
        rest again: 132,635 of 144,923 tokens on Ornith (2026-10-21 02:26), 36,316 of 48,604 on
        the owner's run. `(ids, tokens aligned below the line, the line)` or None when no anchor
        beats the plain shared head's."""
        anchors = sorted(getattr(res, "anchors", None) or {})
        if not anchors or not self._segs:
            return None
        e = int(res.end["at"])
        rows = res.tokens
        c = _common(copy, rows[:e])
        plain = max((b for b in anchors if b <= c), default=0)
        best = None
        for i, a, j in self._segs:
            for x in anchors:
                if i <= x <= a and x > plain and (best is None or x > best[0]):
                    best = (x, j + (x - i))
        if best is None:
            return None
        x, y = best
        if y >= len(copy):
            return None                    # the copy must go on past the line: its own turn
        self.stats["partial"] += 1
        self.stats["partial_tokens"] += x - c
        return rows[:x] + copy[y:], max(0, x - c), x

    def kept(self, res, n_prompt: int, why: str) -> None:
        """After `ResidentPrefix.keep_end`: the new end anchor's regions are the served prompt's
        and its own generation's. Anything but `kept` leaves the old end anchor and its regions."""
        if why != "kept" or res.end is None:
            self._carry = []
            return
        e = int(res.end["at"])
        regs = [r for r in self._carry if r[1] <= int(n_prompt)]
        if self.on and e > int(n_prompt):
            regs.append((int(n_prompt), e, self._think_end(res.tokens, int(n_prompt), e)))
        self._for, self._regions, self._carry = res.end, regs, []
        self.stats["regions"] = len(regs)

    def adopt(self, res, regions) -> None:
        """A conversation read back from the NVMe (engine/spill.py): its end anchor's regions, as
        they were when it was written."""
        if res.end is None:
            return
        e = int(res.end["at"])
        self._for = res.end
        regs = []
        for r in regions:
            a, b = int(r[0]), int(r[1])
            if b <= e:
                # an entry written before SRV-239 has no reasoning end: found again in the rows
                t = int(r[2]) if len(r) > 2 else self._think_end(res.tokens, a, b)
                regs.append((a, b, t))
        self._regions = regs
        self._carry = []
        self.stats["regions"] = len(self._regions)

    def offers(self, tokens: list, end: dict, regions, copy: list[int]) -> bool:
        """SRV-269: whether `copy` would resume at the end anchor of a conversation that is not in
        place (an NVMe entry, engine/spill.py `Entry.offer`): its tokens, its end anchor and its
        regions as written. A probe of its own: this alignment's regions, stats and
        `last_omitted` stay the resident conversation's."""
        if not self.on or end is None:
            return False
        e = int(end["at"])
        if len(tokens) < e:
            return False
        rows = _Rows(list(tokens[:e]),
                     {"at": e, "hidden": [(int(p), list(h)) for p, h in (end.get("hidden") or ())],
                      "mark": end.get("mark"),
                      # FX16: the notes the client never sent, for the moved-block pass
                      "served_hidden": [(int(a), int(b)) for a, b in (end.get("served_hidden") or ())]})
        probe = EndAlign(self.tbytes, on=True, reasoning=self.reasoning, moved=self.moved)
        probe.marks = self.marks
        probe.adopt(rows, regions or [])
        return probe._align(rows, list(copy)) is not None

    def moved_common(self, view: list[int], copy: list[int]) -> int:
        """FX16: how many leading tokens `copy` shares with `view` (the last prompt served, as the
        client rendered it) when a block the client moved to its last message is allowed for:
        the shared length counted as if the moved runs were still in place, so a copy that holds
        all of the view but those runs counts as holding the view. The plain shared head when
        nothing moved or the marks are unknown. For the slot choice (server/app.py), where the
        12:14:30 copy counted as a new conversation and took the other slot."""
        c = _common(copy, view)
        if not self.moved or self.marks is None or c >= len(view):
            return c
        try:
            spans = self._moved_spans(_Rows(list(view), {"at": len(view), "hidden": []}), copy)
        except Exception:                                          # noqa: BLE001
            return c
        if not spans:
            return c
        keep = []
        i = 0
        for a, b in spans:
            keep.extend(view[i:a])
            i = b
        keep.extend(view[i:])
        k = _common(copy, keep)
        return k + sum(b - a for a, b in spans) if k >= len(keep) - 1 else c

    # ------------------------------------------------------------------ the walk
    def _align(self, res, copy: list[int]):
        """The copy aligned to the rows: first as the rows are, then (SRV-239) with every
        region's reasoning left out of what the copy must hold."""
        refused = self.stats["refused"]
        self.last_omitted, self.last_moved, self.last_refusal = [], [], None
        self._segs = []
        got = self._walk(res, copy, False, segs=self._segs)
        thinks = self.reasoning and any(r[2] > r[0] for r in self.regions(res))
        if got is None and thinks:
            got = self._walk(res, copy, True)
            if got is not None:
                self.stats["aligned_reasoning"] += 1
                self.stats["reasoning_tokens"] += got[2]
                self.last_omitted = [(a, t) for a, _, t in self.regions(res) if t > a]
        if got is None and self.moved and self.marks is not None:
            # FX16: a block the client put into an earlier message and moved to its last one
            # FX17: first with the router's notes in the copy (the router keeps them live by
            # their message), then without them (the notes dropped: served as hidden)
            moved, absent = self._moved_spans(res, copy), False
            if not moved:
                moved, absent = self._moved_spans(res, copy, notes_absent=True), True
            if moved:
                got = self._walk(res, copy, False, moved=moved, notes_absent=absent)
                if got is None and thinks:
                    got = self._walk(res, copy, True, moved=moved, notes_absent=absent)
                    if got is not None:
                        self.stats["aligned_reasoning"] += 1
                        self.stats["reasoning_tokens"] += got[2]
                        self.last_omitted = [(a, t) for a, _, t in self.regions(res) if t > a]
                if got is not None:
                    self.stats["aligned_moved"] += 1
                    self.stats["moved_tokens"] += sum(b - a for a, b in moved)
                    self.last_moved = list(moved)
        # one refusal a copy, whichever pass said it
        self.stats["refused"] = refused + (0 if got is not None else
                                           int(self.stats["refused"] > refused))
        if got is not None:
            self.last_refusal = None
        return None if got is None else got[:2]

    def _refuse(self, why: str, c: int, e: int, at: int | None = None):
        """A refusal: counted, and remembered for the log line of the re-read it causes."""
        self.stats["refused"] += 1
        # the last pass tried is the most permissive one: its reason is the one kept
        self.last_refusal = {"why": why, "c": int(c), "e": int(e),
                             "at": int(at) if at is not None else None}
        return None

    def _moved_spans(self, res, copy: list[int], notes_absent: bool = False) -> list[tuple[int, int]]:
        """FX16 (2026-10-08 12:14:30, live): Open WebUI puts its retrieval block ("### Task:
        Respond to the user query using the provided context ... <context>...</context>") into the
        LAST user message of every request. In a tool loop the user message is the first one, so
        the rows hold the block there; the next user message takes the block with it, the first
        message is sent bare, and the copy parts from the rows 12,000 tokens in: an 83,160-token
        chat with its end anchor on the NVMe was read again cold (60,814 tokens, 91 s), and the
        router's notes, hashed on the client's prefix, went with it.

        The rows ranges of such blocks: in each user message of the rows (the template's marks),
        the copy's message is the rows' message with one contiguous run of bytes taken out, and
        that run is in the copy's last user message. Token-aligned, on the bare bytes; empty for
        anything else (a message the user edited is not a moved block: its bytes are not in the
        last message). Served, the model sees the message as it saw it when it answered, and the
        block again where the client put it now."""
        im_s, user, im_e = self.marks
        e, rows = int(res.end["at"]), res.tokens[:int(res.end["at"])]
        tb, u = self.tbytes, len(user)

        def messages(ids, upto):
            out, i = [], 0
            while True:
                try:
                    i = ids.index(im_s, i)
                except ValueError:
                    return out
                if i + 1 + u <= len(ids) and list(ids[i + 1:i + 1 + u]) == user:
                    try:
                        j = ids.index(im_e, i + 1 + u)
                    except ValueError:
                        j = len(ids)
                    if i >= upto:
                        return out
                    out.append((i, j))
                    i = j
                else:
                    i += 1

        rm = messages(rows, e)
        cm = messages(copy, len(copy))
        if not rm or len(cm) < len(rm) + 1:
            return []
        last = b"".join(_bare(tb(t)) for t in copy[cm[-1][0]:cm[-1][1]])
        # the rows the client never sent (the router's notes, SRV-233): left out of the rows'
        # message here, and served as hidden by the walk (`_served_hidden`). With the block
        # gone the notes' hash no longer matches and the router does not put them back, so
        # the copy's message lacks both
        served = self._served_hidden(res) if notes_absent else []
        spans: list[tuple[int, int]] = []
        for (ra, rb), (ca, cb) in zip(rm, cm[:len(rm)]):
            rt, ct = rows[ra:rb], copy[ca:cb]
            if rt == ct:
                continue
            # token by token on the bare bytes, whitespace-only tokens left out (the copy's
            # newlines next to the run are not the rows'): the copy's message is the rows'
            # with the run [a, b) taken out, whole tokens, nothing else changed
            rb_ = [(k, _bare(tb(t))) for k, t in enumerate(rt)]
            ri = [(k, p) for k, p in rb_ if p and not any(x <= ra + k < y for x, y in served)]
            ci = [p for p in (_bare(tb(t)) for t in ct) if p]
            if len(ci) >= len(ri):
                return []
            x = 0
            while x < len(ci) and ri[x][1] == ci[x]:
                x += 1
            y = 0
            while y < len(ci) - x and ri[len(ri) - 1 - y][1] == ci[len(ci) - 1 - y]:
                y += 1
            if x + y != len(ci):
                return []
            run = ri[x:len(ri) - y]
            d = b"".join(p for _, p in run)
            if not run or not d or d not in last:
                return []
            a, b = run[0][0], run[-1][0] + 1
            while b < len(rt) and not rb_[b][1]:
                b += 1                         # the rows' whitespace after it goes with it
            spans.append((ra + a, ra + b))
        return spans

    @staticmethod
    def _served_hidden(res) -> list[tuple[int, int]]:
        """The rows ranges of the served prompt its client did not send (the router's notes,
        `end["served_hidden"]`, SRV-233), below the end anchor."""
        end = getattr(res, "end", None) or {}
        e = int(end.get("at") or 0)
        out = []
        for a, b in (end.get("served_hidden") or ()):
            a, b = int(a), int(b)
            if 0 <= a < b <= e:
                out.append((a, b))
        return out

    def _walk(self, res, copy: list[int], omit: bool, moved=(), notes_absent: bool = False,
              segs: list | None = None):
        end = res.end
        e, rows = int(end["at"]), res.tokens
        regs = self.regions(res)
        if not regs or len(rows) < e:
            return self._refuse("no_regions", 0, e)
        c = _common(copy, rows[:e])
        if c >= e:
            return self._refuse("copy_holds_rows", c, e)   # nothing to align
        hidden = [(int(p), list(h)) for p, h in end["hidden"]]
        hid = set()
        zones = [(r[0], r[1]) for r in regs]
        left = 0
        # the moved runs, and with them the notes the client never sent, which the router
        # does not put back once the prompt changed before them
        extra = list(moved) + (self._served_hidden(res) if moved and notes_absent else [])
        for a, b in extra:
            hid.update(range(a, b))
            while a > 0 and not _bare(self.tbytes(rows[a - 1])):
                a -= 1
            while b < e and not _bare(self.tbytes(rows[b])):
                b += 1
            zones.append((a, b))
        if omit:
            # each region's reasoning (from its start to its `</think>`) is not in the copy; the
            # whitespace right before it (the prompt's `\n` after `<think>`, which the copy
            # writes as one `\n\n` token with the empty block's) may differ too
            for a, b, t in regs:
                if t <= a:
                    continue
                hid.update(range(a, t))
                left += t - a
                z = a
                while z > 0 and not _bare(self.tbytes(rows[z - 1])):
                    z -= 1
                zones.append((z, b))
        for p, h in hidden:
            hid.update(range(p, p + len(h)))
            a, b = p, p + len(h)
            while a > 0 and not _bare(self.tbytes(rows[a - 1])):
                a -= 1
            while b < e and not _bare(self.tbytes(rows[b])):
                b += 1
            zones.append((a, b))
        zones = _merge(sorted(zones))
        first = next((z for z in zones if z[0] <= c < z[1] or c == z[1]), None)
        if first is None:
            # the copy parts from the rows outside every generation and every hidden span: in
            # the prompt itself (a message changed, a block moved)
            return self._refuse("in_prompt", c, e)
        i = j = first[0]
        aligned = 0
        tb = self.tbytes
        if segs is not None:
            segs.append((0, i, 0))         # the shared head, token for token
        for a, b in zones:
            if b <= i:
                continue
            if copy[j:j + a - i] != rows[i:a]:
                if segs is not None:
                    segs.append((i, i + _common(copy[j:], rows[i:a]), j))
                return self._refuse("between_zones", c, e, at=i)
            if segs is not None:
                segs.append((i, a, j))
            j += a - i
            want = b"".join(_bare(tb(rows[p])) for p in range(a, b) if p not in hid)
            pos = 0
            while pos < len(want):
                piece = _bare(tb(copy[j])) if j < len(copy) else None
                if piece is None or want[pos:pos + len(piece)] != piece:
                    return self._refuse("in_zone", c, e, at=a)
                pos += len(piece)
                j += 1
            i = b
            aligned += b - a
            # whitespace the copy has where the rows go on with the next token
            while i < e and j < len(copy) and copy[j] != rows[i] and not _bare(tb(copy[j])):
                j += 1
        if len(copy) - j <= e - i or copy[j:j + e - i] != rows[i:e]:
            if segs is not None:
                segs.append((i, i + _common(copy[j:], rows[i:e]), j))
            return self._refuse("tail", c, e, at=i)
        self.stats["aligned"] += 1
        self.stats["aligned_tokens"] += aligned
        return rows[:e] + copy[j + e - i:], aligned, left

    def report(self) -> dict:
        return {"on": self.on, "reasoning": self.reasoning, **self.stats}


class _Rows:
    """What the walk reads of a conversation: its end anchor and its rows' tokens."""

    def __init__(self, tokens: list, end: dict):
        self.tokens, self.end = tokens, end


def _merge(zones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in zones:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
