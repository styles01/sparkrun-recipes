"""Structured outputs: regular constraints as byte-level DFAs, and token masks over them.

A constraint is a regular language over the text of the answer. Every form the server accepts is
compiled to one: a raw `regex`, a `choice` list (the literals, alternated), a JSON schema (the
subset below, compiled to a regex the way Outlines does), and `json_object` (any JSON object, to a
fixed nesting depth). The regex is compiled to a DFA over BYTES, not characters, because the
tokenizer is byte-level: a token can end half way through a UTF-8 character, and a byte automaton
accepts exactly the byte strings of the language, halves included.

The one operation the decoder needs is "which tokens may come next in DFA state s": a token is
allowed when walking its bytes from s never reaches the dead state. That is computed for the whole
vocabulary at once -- the tokens as a padded byte matrix, one table gather per byte position,
shortest tokens retired first -- and cached per state, so a state costs a few milliseconds once and
nothing after. The end token is allowed exactly in accepting states; every other special token
never is.

Why a regular language and not a grammar: a mask must be a deterministic function of the decoded
prefix, which any automaton gives; a DFA gives it with a table lookup per byte, which is what makes
masking every row of a sixteen-row verify cheap. The price is bounded nesting for `json_object`
and schemas with recursion (`$ref` cycles) refused.

Regex syntax: literals, `.`, escapes (`\\d \\w \\s \\D \\W \\S`, `\\n \\t \\r`, `\\xHH`, `\\uXXXX`, escaped
metacharacters), classes `[...]` with ranges and negation, groups `(...)` / `(?:...)`, `|`, and the
quantifiers `* + ? {m} {m,} {m,n}`. The whole answer must match (implicit anchors). No
backreferences, lookaround or lazy quantifiers -- none has a DFA.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import OrderedDict

import numpy as np

MAX_CP = 0x10FFFF
DEAD = -1


class GrammarError(ValueError):
    """A constraint this engine cannot compile, or a state with no legal token: a 400 to the client."""


# ------------------------------------------------------------------------------------ regex -> AST

class _Parser:
    """Recursive descent over the pattern. Nodes: ("set", ranges), ("cat", [..]), ("alt", [..]),
    ("rep", node, lo, hi) with hi None for unbounded."""

    def __init__(self, pattern: str):
        self.p, self.i = pattern, 0

    def parse(self):
        node = self._alt()
        if self.i != len(self.p):
            raise GrammarError(f"regex: unexpected {self.p[self.i]!r} at {self.i}")
        return node

    def _peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def _alt(self):
        parts = [self._cat()]
        while self._peek() == "|":
            self.i += 1
            parts.append(self._cat())
        return parts[0] if len(parts) == 1 else ("alt", parts)

    def _cat(self):
        items = []
        while self._peek() not in (None, "|", ")"):
            items.append(self._quant(self._atom()))
        return ("cat", items)

    def _quant(self, node):
        while True:
            c = self._peek()
            if c == "*":
                self.i += 1
                node = ("rep", node, 0, None)
            elif c == "+":
                self.i += 1
                node = ("rep", node, 1, None)
            elif c == "?":
                self.i += 1
                node = ("rep", node, 0, 1)
            elif c == "{" and re.match(r"\{\d+(,\d*)?\}", self.p[self.i:]):
                m = re.match(r"\{(\d+)(,(\d*))?\}", self.p[self.i:])
                self.i += m.end()
                lo = int(m.group(1))
                hi = lo if m.group(2) is None else (int(m.group(3)) if m.group(3) else None)
                if hi is not None and hi < lo:
                    raise GrammarError("regex: {m,n} with n < m")
                node = ("rep", node, lo, hi)
            else:
                return node
            if self._peek() == "?":
                raise GrammarError("regex: lazy quantifiers have no DFA")

    def _atom(self):
        c = self._peek()
        if c is None:
            raise GrammarError("regex: unexpected end")
        if c == "(":
            self.i += 1
            if self.p.startswith("?:", self.i):
                self.i += 2
            elif self._peek() == "?":
                raise GrammarError("regex: lookaround and named groups are not supported")
            node = self._alt()
            if self._peek() != ")":
                raise GrammarError("regex: missing )")
            self.i += 1
            return node
        if c == "[":
            return ("set", self._class())
        if c == ".":
            self.i += 1
            return ("set", _negate([(10, 10)]))
        if c == "\\":
            return ("set", self._escape(in_class=False))
        if c in "^$":
            raise GrammarError("regex: anchors are implicit (the whole answer matches)")
        if c in "*+?{":
            raise GrammarError(f"regex: nothing to repeat at {self.i}")
        self.i += 1
        return ("set", [(ord(c), ord(c))])

    def _escape(self, in_class: bool) -> list[tuple[int, int]]:
        self.i += 1
        c = self._peek()
        if c is None:
            raise GrammarError("regex: trailing backslash")
        self.i += 1
        table = {"d": [(48, 57)], "w": [(48, 57), (65, 90), (95, 95), (97, 122)],
                 "s": [(9, 13), (32, 32)]}
        if c in table:
            return table[c]
        if c in "DWS":
            return _negate(table[c.lower()])
        if c in "ux":
            n = 4 if c == "u" else 2
            h = self.p[self.i:self.i + n]
            if not re.fullmatch(f"[0-9a-fA-F]{{{n}}}", h):
                raise GrammarError(f"regex: \\{c} needs {n} hex digits")
            self.i += n
            return [(int(h, 16), int(h, 16))]
        simple = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "0": 0}
        if c in simple:
            return [(simple[c], simple[c])]
        if c.isalnum():
            raise GrammarError(f"regex: unsupported escape \\{c}")
        return [(ord(c), ord(c))]

    def _class(self) -> list[tuple[int, int]]:
        self.i += 1
        neg = self._peek() == "^"
        if neg:
            self.i += 1
        ranges: list[tuple[int, int]] = []
        first = True
        while True:
            c = self._peek()
            if c is None:
                raise GrammarError("regex: missing ]")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "\\":
                lo_set = self._escape(in_class=True)
            else:
                self.i += 1
                lo_set = [(ord(c), ord(c))]
            if (len(lo_set) == 1 and lo_set[0][0] == lo_set[0][1] and self._peek() == "-"
                    and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]"):
                self.i += 1
                d = self._peek()
                if d == "\\":
                    hi_set = self._escape(in_class=True)
                else:
                    self.i += 1
                    hi_set = [(ord(d), ord(d))]
                if len(hi_set) != 1 or hi_set[0][0] != hi_set[0][1] or hi_set[0][0] < lo_set[0][0]:
                    raise GrammarError("regex: bad class range")
                ranges.append((lo_set[0][0], hi_set[0][0]))
            else:
                ranges.extend(lo_set)
        ranges = _merge(ranges)
        return _negate(ranges) if neg else ranges


def _merge(ranges):
    out = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _negate(ranges):
    out, nxt = [], 0
    for lo, hi in _merge(ranges):
        if lo > nxt:
            out.append((nxt, lo - 1))
        nxt = hi + 1
    if nxt <= MAX_CP:
        out.append((nxt, MAX_CP))
    # surrogates are not characters and have no UTF-8 encoding
    return [r for r in _subtract(out, (0xD800, 0xDFFF))]


def _subtract(ranges, cut):
    for lo, hi in ranges:
        if hi < cut[0] or lo > cut[1]:
            yield (lo, hi)
            continue
        if lo < cut[0]:
            yield (lo, cut[0] - 1)
        if hi > cut[1]:
            yield (cut[1] + 1, hi)


# ------------------------------------------------------------------------ code points -> UTF-8

_BOUNDS = ((0, 0x7F), (0x80, 0x7FF), (0x800, 0xFFFF), (0x10000, MAX_CP))


def utf8_sequences(lo: int, hi: int) -> list[list[tuple[int, int]]]:
    """The code points [lo, hi] as a list of byte-range sequences, each sequence one UTF-8 length
    and each position a contiguous byte range (the standard split: a range whose leading bytes
    differ is cut where a continuation byte wraps)."""
    out = []
    for blo, bhi in _BOUNDS:
        a, b = max(lo, blo), min(hi, bhi)
        if a <= b:
            out.extend(_split(a, b))
    return out


def _split(lo: int, hi: int) -> list[list[tuple[int, int]]]:
    el, eh = chr(lo).encode(), chr(hi).encode()
    if len(el) == 1:
        return [[(el[0], eh[0])]]
    n = len(el)
    # cut the range so that below the first differing position every byte runs over its full
    # continuation span 0x80..0xBF; then each piece is one sequence of byte ranges
    for i in range(n - 1, 0, -1):
        m = (1 << (6 * i)) - 1                  # the code points one continuation step covers
        if lo & ~m != hi & ~m:
            if lo & m != 0:
                return _split(lo, lo | m) + _split((lo | m) + 1, hi)
            if hi & m != m:
                return _split(lo, (hi & ~m) - 1) + _split(hi & ~m, hi)
    return [[(a, b) for a, b in zip(el, eh)]]


# --------------------------------------------------------------------------------- AST -> NFA

class _NFA:
    """States with byte-range edges and epsilon edges (Thompson)."""

    def __init__(self):
        self.edges: list[list[tuple[int, int, int]]] = []
        self.eps: list[list[int]] = []

    def state(self) -> int:
        self.edges.append([])
        self.eps.append([])
        return len(self.edges) - 1

    def build(self, node, depth: int = 0) -> tuple[int, int]:
        if depth > 400:
            raise GrammarError("constraint too deeply nested")
        kind = node[0]
        if kind == "set":
            s, e = self.state(), self.state()
            for lo, hi in node[1]:
                for seq in utf8_sequences(lo, hi):
                    cur = s
                    for k, (a, b) in enumerate(seq):
                        nxt = e if k == len(seq) - 1 else self.state()
                        self.edges[cur].append((a, b, nxt))
                        cur = nxt
            return s, e
        if kind == "cat":
            s = e = self.state()
            for item in node[1]:
                a, b = self.build(item, depth + 1)
                self.eps[e].append(a)
                e = b
            return s, e
        if kind == "alt":
            s, e = self.state(), self.state()
            for item in node[1]:
                a, b = self.build(item, depth + 1)
                self.eps[s].append(a)
                self.eps[b].append(e)
            return s, e
        if kind == "rep":
            _, inner, lo, hi = node
            if (hi if hi is not None else lo) > 1000:
                raise GrammarError("a repetition count above 1000")
            s = e = self.state()
            if hi is None:
                # SRV-619: `X{lo,}` as lo - 1 copies and one loop (`X+` is one copy of X, not
                # two: a large X -- the tool-call language -- doubled the automaton)
                for _ in range(max(lo - 1, 0)):
                    a, b = self.build(inner, depth + 1)
                    self.eps[e].append(a)
                    e = b
                a, b = self.build(inner, depth + 1)
                self.eps[e].append(a)
                self.eps[b].append(a)
                end = self.state()
                if lo == 0:
                    self.eps[e].append(end)
                self.eps[b].append(end)
                return s, end
            for _ in range(lo):
                a, b = self.build(inner, depth + 1)
                self.eps[e].append(a)
                e = b
            end = self.state()
            for _ in range(hi - lo):
                self.eps[e].append(end)
                a, b = self.build(inner, depth + 1)
                self.eps[e].append(a)
                e = b
            self.eps[e].append(end)
            return s, end
        raise GrammarError(f"internal: node {kind}")


class DFA:
    """A dense byte DFA: `table[s, byte]` is the next state, DEAD mapped to `dead` (absorbing)."""

    def __init__(self, table: np.ndarray, accept: np.ndarray, start: int = 0):
        self.table, self.accept, self.start = table, accept, start
        self.dead = table.shape[0] - 1
        self.rows = table.tolist()                  # scalar walks are faster on lists

    @property
    def states(self) -> int:
        return self.table.shape[0] - 1

    def walk(self, state: int, data: bytes) -> int:
        rows = self.rows
        for b in data:
            state = rows[state][b]
        return state


def compile_regex(pattern: str, max_states: int = 20000) -> DFA:
    """`pattern` -> a DFA over the UTF-8 bytes of its language (subset construction over byte
    classes: bytes no edge of the NFA tells apart share one column of work)."""
    ast = _Parser(pattern).parse()
    nfa = _NFA()
    start, final = nfa.build(ast)
    # byte classes: the cut points of every edge range
    cuts = {0, 256}
    for edges in nfa.edges:
        for a, b, _ in edges:
            cuts.add(a)
            cuts.add(b + 1)
    cuts = sorted(cuts)
    reps = cuts[:-1]                                # one representative byte a class

    def closure(states):
        stack, seen = list(states), set(states)
        while stack:
            s = stack.pop()
            for t in nfa.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    first = closure([start])
    index = {first: 0}
    order = [first]
    rows: list[list[int]] = []
    i = 0
    while i < len(order):
        cur = order[i]
        row = []
        for rep in reps:
            nxt = set()
            for s in cur:
                for a, b, t in nfa.edges[s]:
                    if a <= rep <= b:
                        nxt.add(t)
            if not nxt:
                row.append(DEAD)
                continue
            key = closure(nxt)
            j = index.get(key)
            if j is None:
                j = index[key] = len(order)
                order.append(key)
                if len(order) > max_states:
                    raise GrammarError(f"the constraint needs more than {max_states} states")
            row.append(j)
        rows.append(row)
        i += 1
    n = len(order)
    width = np.diff(np.array(cuts))
    dense = np.repeat(np.array(rows, dtype=np.int32), width, axis=1)       # [n, 256]
    dense[dense == DEAD] = n
    table = np.vstack([dense, np.full((1, 256), n, dtype=np.int32)])       # the dead row
    accept = np.array([final in s for s in order] + [False])
    return DFA(table, accept)


#: SRV-619: a product state's side: both languages alive, only the first (the calls), only the
#: second (the document), dead
BOTH, FIRST, SECOND, NEITHER = 0, 1, 2, 3


def product_dfa(a: DFA, b: DFA, max_states: int = 40000) -> tuple[DFA, np.ndarray]:
    """SRV-619: the union of two languages as the product of their automata, and each state's
    side (`BOTH`, `FIRST`, `SECOND`; the dead state `NEITHER`): the union's states that still
    hold both, and which one a state committed to."""
    da, db = a.dead, b.dead
    index = {(a.start, b.start): 0}
    order = [(a.start, b.start)]
    rows = []
    i = 0
    while i < len(order):
        sa, sb = order[i]
        key = a.table[sa].astype(np.int64) * (db + 1) + b.table[sb]
        uniq, inv = np.unique(key, return_inverse=True)
        ids = np.empty(len(uniq), dtype=np.int32)
        for k, u in enumerate(uniq.tolist()):
            pair = divmod(u, db + 1)
            if pair == (da, db):
                ids[k] = -1
                continue
            j = index.get(pair)
            if j is None:
                j = index[pair] = len(order)
                order.append(pair)
                if len(order) > max_states:
                    raise GrammarError(f"the constraint needs more than {max_states} states")
            ids[k] = j
        rows.append(ids[inv])
        i += 1
    n = len(order)
    dense = np.array(rows, dtype=np.int32).reshape(n, 256)
    dense[dense == -1] = n
    table = np.vstack([dense, np.full((1, 256), n, dtype=np.int32)])
    accept = np.array([bool(a.accept[x] or b.accept[y]) for x, y in order] + [False])
    side = np.array([BOTH if x != da and y != db else FIRST if x != da else SECOND
                     for x, y in order] + [NEITHER], dtype=np.int8)
    return DFA(table, accept), side


# ------------------------------------------------------------------------- JSON schema -> regex

WS = r"[ \t\n\r]*"
STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
STRING = f'"{STRING_CHAR}*"'
INTEGER = r"-?(?:0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_META = re.compile(r"([\\.^$|?*+()\[\]{}])")


def literal(text: str) -> str:
    """A regex matching exactly `text`."""
    return _META.sub(r"\\\1", text).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def any_json(depth: int) -> str:
    """Any JSON value, nested at most `depth` deep."""
    scalars = f"{STRING}|{NUMBER}|true|false|null"
    if depth <= 0:
        return f"(?:{scalars})"
    inner = any_json(depth - 1)
    arr = rf"\[{WS}(?:{inner}(?:{WS},{WS}{inner})*)?{WS}\]"
    return f"(?:{scalars}|{_object(inner)}|{arr})"


def json_object_regex(depth: int = 3) -> str:
    """`response_format: json_object`: one JSON object (values nested up to `depth`)."""
    return WS + _object(any_json(depth - 1)) + WS


def schema_regex(schema: dict, defs: dict | None = None, depth: int = 0) -> str:
    """The JSON-schema subset: type object (properties in their declared order, `required`;
    others optional and in order; no additional properties), array (`items`, `minItems`,
    `maxItems`), string (`enum`, `const`, `pattern`, `minLength`, `maxLength`), number, integer,
    boolean, null, `enum`/`const` of any JSON, `anyOf`/`oneOf`, local `$ref` without cycles."""
    if depth > 32:
        raise GrammarError("schema: nested too deep, or a $ref cycle")
    if not isinstance(schema, dict):
        raise GrammarError("schema: a schema is an object")
    defs = defs if defs is not None else {**schema.get("$defs", {}), **schema.get("definitions", {})}
    if "$ref" in schema:
        ref = schema["$ref"]
        m = re.fullmatch(r"#/(?:\$defs|definitions)/(.+)", ref)
        if not m or m.group(1) not in defs:
            raise GrammarError(f"schema: unsupported $ref {ref!r}")
        return schema_regex(defs[m.group(1)], defs, depth + 1)
    if "const" in schema:
        return literal(json.dumps(schema["const"]))
    if "enum" in schema:
        return "(?:" + "|".join(literal(json.dumps(v)) for v in schema["enum"]) + ")"
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return "(?:" + "|".join(schema_regex(s, defs, depth + 1) for s in schema[key]) + ")"
    t = schema.get("type")
    if isinstance(t, list):
        return "(?:" + "|".join(schema_regex({**schema, "type": x}, defs, depth + 1)
                                for x in t) + ")"
    if t == "string":
        if "pattern" in schema:
            return f'"(?:{schema["pattern"]})"'
        lo, hi = schema.get("minLength", 0), schema.get("maxLength")
        rep = f"{{{lo},{'' if hi is None else hi}}}" if (lo or hi is not None) else "*"
        return f'"{STRING_CHAR}{rep}"'
    if t == "integer":
        return INTEGER
    if t == "number":
        return NUMBER
    if t == "boolean":
        return "(?:true|false)"
    if t == "null":
        return "null"
    if t == "array":
        item = schema_regex(schema.get("items", {}), defs, depth + 1) if schema.get("items") \
            else any_json(2)
        lo, hi = int(schema.get("minItems", 0)), schema.get("maxItems")
        more = f"(?:{WS},{WS}{item})"
        if hi is not None and hi == 0:
            return rf"\[{WS}\]"
        tail = f"{more}{{{max(lo - 1, 0)},{'' if hi is None else hi - 1}}}"
        body = f"{item}{tail}"
        return rf"\[{WS}{body}{WS}\]" if lo >= 1 else rf"\[{WS}(?:{body})?{WS}\]"
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        if not props:
            return _object(any_json(1))
        required = set(schema.get("required", []))
        keys = list(props)
        pieces = [rf'"{literal(k)}"{WS}:{WS}{schema_regex(props[k], defs, depth + 1)}'
                  for k in keys]

        def members(i: int, emitted: bool) -> str:
            """Members i.. in declared order; `emitted`: one is already written (a comma first)."""
            if i == len(keys):
                return ""
            item = (f"{WS},{WS}" if emitted else "") + pieces[i]
            if keys[i] in required:
                return item + members(i + 1, True)
            if emitted:
                return f"(?:{item})?" + members(i + 1, True)
            return f"(?:{item}{members(i + 1, True)}|{members(i + 1, False)})"

        return rf"\{{{WS}{members(0, False)}{WS}\}}"
    if not schema or t is None:
        return any_json(2)
    raise GrammarError(f"schema: unsupported type {t!r}")


def _object(value: str) -> str:
    return rf"\{{{WS}(?:{STRING}{WS}:{WS}{value}(?:{WS},{WS}{STRING}{WS}:{WS}{value})*)?{WS}\}}"


# ------------------------------------------------------------------------------- the vocabulary

class Vocab:
    """Every token id as its bytes, for the masks: a padded matrix sorted by length.

    `special` ids (the tokenizer's added and control tokens: `<|im_end|>`, `<think>`, ...) have no
    text in the answer and are never allowed by a constraint; the end token is added back in
    accepting states by the caller.
    """

    def __init__(self, token_bytes: list[bytes | None], size: int):
        self.size = int(size)                     # the logits' width (may exceed the tokenizer's)
        self.bytes = [b if b else None for b in token_bytes]
        ids = [i for i, b in enumerate(self.bytes) if b]
        ids.sort(key=lambda i: len(self.bytes[i]))
        self.ids = np.array(ids, dtype=np.int64)
        lens = np.array([len(self.bytes[i]) for i in ids], dtype=np.int64)
        width = int(lens.max()) if len(lens) else 0
        mat = np.zeros((len(ids), max(width, 1)), dtype=np.uint8)
        for r, i in enumerate(ids):
            b = self.bytes[i]
            mat[r, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        self.mat = mat
        # tokens still being walked at byte position j are rows [first_live[j], n): sorted by length
        self.first_live = np.searchsorted(lens, np.arange(width + 1), side="right")

    @classmethod
    def from_tokenizer(cls, tok, size: int, markers: tuple[str, ...] = ()) -> "Vocab":
        """`markers`: added tokens that are text in the answer (not the tokenizer's special ids)
        and keep their text, so a constraint allows the model's own id for them
        (SRV-619: `<tool_call>` is one id; spelled out byte by byte it is off the
        model's distribution)."""
        from server.logprobs import _DECODER
        special = set(getattr(tok, "all_special_ids", []) or [])
        hard = set(special)
        added = getattr(tok, "added_tokens_decoder", None) or {}
        special |= {int(i) for i in added}
        n = len(tok)
        pieces = tok.convert_ids_to_tokens(list(range(n)))
        out: list[bytes | None] = []
        for i, piece in enumerate(pieces):
            if i in special or not isinstance(piece, str) or not piece:
                out.append(None)
            elif all(c in _DECODER for c in piece):
                out.append(bytes(_DECODER[c] for c in piece))
            else:
                out.append(tok.decode([i]).encode("utf-8") or None)
        for m in markers:
            i = tok.convert_tokens_to_ids(m)
            if (not isinstance(i, int) or not 0 <= i < size or i in hard
                    or i == getattr(tok, "unk_token_id", None)
                    or tok.decode([i], skip_special_tokens=True) != m):
                continue
            out.extend([None] * (i + 1 - len(out)))
            out[i] = m.encode("utf-8")
        return cls(out, size)

    def ends(self, dfa: DFA, state: int) -> np.ndarray:
        """Int [len(ids)]: the state each token's bytes walk to from `state` (`ids` order)."""
        n = len(self.ids)
        cur = np.full(n, state, dtype=np.int32)
        table = dfa.table
        for j in range(self.mat.shape[1]):
            lo = self.first_live[j]
            if lo >= n:
                break
            cur[lo:] = table[cur[lo:], self.mat[lo:, j]]
        return cur

    def mask(self, dfa: DFA, state: int) -> np.ndarray:
        """Bool [size]: the tokens whose bytes, walked from `state`, never reach the dead state."""
        out = np.zeros(self.size, dtype=bool)
        out[self.ids] = self.ends(dfa, state) != dfa.dead
        return out


# ------------------------------------------------------------------ the constraint in the decoder

# Device masks of every grammar, least recently used first. A mask is one byte a token (248 KB on
# this model) and a json_object grammar has about 3,000 states, so the cache has a bound, not the
# grammars: 1,024 masks is 254 MB, and an evicted state costs one recomputation if it comes back.
MASK_CACHE = 1024
_MASKS: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
# A grammar's masks are keyed by a serial number of its own, never by `id()`: CPython gives a freed
# object's id to the next one, and a new grammar keyed by a dead one's id read the dead one's masks
# (a 97-token mask on a 99-token vocabulary in the tests, a wrong mask on a box)
_SERIAL = itertools.count()


class Grammar:
    """A compiled constraint; its per-state masks live in the shared, bounded `_MASKS`.

    `parts` (SRV-619): (the calls, the document), compiled apart and joined as their product
    (`pattern` is their union, for the keys). Where both are still alive the model picks the
    branch: the row's own argmax, before the mask, decides -- a token that keeps both (whitespace)
    leaves the choice open, one that starts a call commits to the calls, anything else (the
    document's first character, a fence, prose, the end token) to the document. The token taken
    then fixes the side (`side`), so the mask stays a function of the prefix and the target's row.
    """

    def __init__(self, pattern: str, vocab: Vocab, parts: tuple[str, str] | None = None):
        self.pattern, self.vocab = pattern, vocab
        if parts is None:
            self.dfa, self.side = compile_regex(pattern), None
        else:
            self.dfa, self.side = product_dfa(compile_regex(parts[0]), compile_regex(parts[1]))
        self._sides: dict[int, np.ndarray] = {}
        self.serial = next(_SERIAL)

    def cached(self, state: int, device) -> bool:
        return (self.serial, state, str(device), "all") in _MASKS

    def forget(self) -> None:
        """Drop this grammar's masks from the shared cache (it left the grammar cache)."""
        for k in [k for k in _MASKS if k[0] == self.serial]:
            del _MASKS[k]

    def open(self, state: int) -> bool:
        """SRV-619: both branches still alive here (the row decides)."""
        return self.side is not None and self.side[state] == BOTH

    def _token_sides(self, state: int) -> np.ndarray:
        """Int8 [size]: the side each token's walk from `state` ends on (`NEITHER`: not legal)."""
        got = self._sides.get(state)
        if got is None:
            got = np.full(self.vocab.size, NEITHER, dtype=np.int8)
            got[self.vocab.ids] = self.side[self.vocab.ends(self.dfa, state)]
            self._sides[state] = got
        return got

    def pick(self, state: int, top: int) -> str:
        """SRV-619: the mask variant of an open state for a row whose argmax is `top`."""
        sides = self._token_sides(state)
        k = int(sides[top]) if 0 <= top < len(sides) else NEITHER
        return "all" if k == BOTH else "calls" if k == FIRST else "document"

    def mask(self, state: int, eos: frozenset, device, row=None) -> "torch.Tensor":
        """Bool [vocab] on `device`: the tokens legal in `state`, the end tokens iff it accepts.
        In an open state (`parts`) the row's argmax picks the branch (`pick`)."""
        import torch
        variant = "all"
        if row is not None and self.open(state):
            variant = self.pick(state, int(row.argmax()))
        key = (self.serial, state, str(device), variant)
        m = _MASKS.get(key)
        if m is not None:
            _MASKS.move_to_end(key)
        else:
            if variant == "all":
                arr = self.vocab.mask(self.dfa, state)
            else:
                arr = self._token_sides(state) == (FIRST if variant == "calls" else SECOND)
                if not arr.any():
                    arr = self.vocab.mask(self.dfa, state)
            ok = bool(self.dfa.accept[state])
            for e in eos:
                if 0 <= e < arr.shape[0]:
                    arr[e] = ok
            if not arr.any():
                raise GrammarError(f"the constraint reached a state with no legal token "
                                   f"(state {state} of {self.dfa.states})")
            m = _MASKS[key] = torch.from_numpy(arr).to(device)
            while len(_MASKS) > MASK_CACHE:
                _MASKS.popitem(last=False)
        return m

    def branch(self, ids, think_end: int | None = None, in_think: bool = False) -> str | None:
        """SRV-619: the branch an answer took -- "calls", "document", or "open" (nothing
        committed yet); None for a grammar without `parts`."""
        if self.side is None:
            return None
        state, thinking = self.dfa.start, bool(in_think)
        for t in ids:
            t = int(t)
            if thinking:
                thinking = t != think_end
                continue
            b = self.vocab.bytes[t] if 0 <= t < len(self.vocab.bytes) else None
            if b:
                state = self.dfa.walk(state, b)
        return {FIRST: "calls", SECOND: "document", BOTH: "open"}.get(int(self.side[state]), "none")


_GRAMMARS: dict = {}


def grammar_for(pattern: str, vocab: Vocab, keep: int = 16,
                parts: tuple[str, str] | None = None) -> Grammar:
    """The compiled grammar for `pattern`, from a small cache (a client sends one schema many times)."""
    gk = pattern if parts is None else ("parts", pattern)
    g = _GRAMMARS.pop(gk, None)
    if g is None or g.vocab is not vocab:
        if g is not None:
            g.forget()                      # the same pattern on another vocabulary
        g = Grammar(pattern, vocab, parts)
    _GRAMMARS[gk] = g
    while len(_GRAMMARS) > keep:
        _GRAMMARS.pop(next(iter(_GRAMMARS))).forget()
    return g


class Constraint:
    """A structured-output constraint as a logit processor, with PenaltyState's interface.

    The mask is applied at the penalties' decision sites -- the prefill's first token, a single
    step, every row of a chain and of a tree -- and it is a deterministic function of the decoded
    prefix: row i of a chain gets the DFA state after the committed text plus `draft[:i]`, a tree
    node the state after its ancestor path. So a draft is accepted exactly when it is the target's
    argmax under the constraint, and speculation writes what the unspeculated loop writes. A row
    after a draft token the constraint forbids is left alone: the walk can never reach it.

    Reasoning is not constrained: while the prompt's `<think>` is open nothing is masked, and the
    closing token starts the constraint from its first state (the answer may open with whitespace).
    """

    def __init__(self, grammar: Grammar, eos, device, think_end: int | None = None,
                 in_think: bool = False):
        self.g, self.eos, self.device = grammar, frozenset(int(e) for e in eos), device
        self.think_end, self.in_think = think_end, bool(in_think)
        self.state, self.thinking = grammar.dfa.start, self.in_think
        self.mask = True                                # PenaltyState's flag; unused here

    def key(self) -> tuple:
        return ("grammar", self.g.pattern)

    # --- the history ---------------------------------------------------------------------------
    def seed(self, ids) -> None:
        self.state, self.thinking = self.g.dfa.start, self.in_think

    def _step(self, state: int, thinking: bool, tok: int) -> tuple[int, bool]:
        if thinking:
            return state, tok != self.think_end
        if tok in self.eos:
            return state, False
        b = self.g.vocab.bytes[tok] if 0 <= tok < len(self.g.vocab.bytes) else None
        return (self.g.dfa.walk(state, b) if b else self.g.dfa.dead), False

    def commit(self, ids) -> None:
        for t in ids:
            self.state, self.thinking = self._step(self.state, self.thinking, int(t))

    # --- the rows -------------------------------------------------------------------------------
    def _row_mask(self, state: int, thinking: bool, row=None):
        if thinking or state == self.g.dfa.dead:
            return None
        return self.g.mask(state, self.eos, self.device, row)

    def _apply(self, rows, masks) -> None:
        import torch
        live = [i for i, m in enumerate(masks) if m is not None]
        if not live:
            return
        if len(live) == rows.shape[0]:
            rows.masked_fill_(~torch.stack(masks), float("-inf"))
            return
        for i in live:
            rows[i].masked_fill_(~masks[i], float("-inf"))

    def apply_single(self, row) -> None:
        m = self._row_mask(self.state, self.thinking, row)
        if m is not None:
            row.masked_fill_(~m, float("-inf"))

    def apply_chain(self, lg, draft) -> None:
        state, thinking = self.state, self.thinking
        masks = []
        for i in range(lg.shape[0]):
            masks.append(self._row_mask(state, thinking, lg[i]))
            if i < len(draft):
                state, thinking = self._step(state, thinking, int(draft[i]))
        self._apply(lg, masks)

    def apply_tree(self, lg, tree) -> None:
        # node 0 is the anchor, already committed; node j's row is the state after its path
        at = [(self.state, self.thinking)]
        for j in range(1, lg.shape[0]):
            at.append(self._step(*at[tree.parents[j]], int(tree.tokens[j])))
        self._apply(lg, [self._row_mask(s, th, lg[j]) for j, (s, th) in enumerate(at)])


class LogitChain:
    """Several logit processors called as one, in order (the penalties, then a constraint)."""

    def __init__(self, procs):
        self.procs = list(procs)

    @property
    def mask(self):
        return self.procs[0].mask

    @mask.setter
    def mask(self, v):
        for p in self.procs:
            p.mask = v

    def seed(self, ids):
        for p in self.procs:
            p.seed(ids)

    def commit(self, ids):
        for p in self.procs:
            p.commit(ids)

    def apply_single(self, row):
        for p in self.procs:
            p.apply_single(row)

    def apply_chain(self, lg, draft):
        for p in self.procs:
            p.apply_chain(lg, draft)

    def apply_tree(self, lg, tree):
        for p in self.procs:
            p.apply_tree(lg, tree)


# ------------------------------------------------------------ tool_choice as a constraint

def avoiding(word: str) -> str:
    """A regex for the strings that do not contain `word` -- for a word whose first character
    occurs nowhere else in it (true of `</parameter>`). The loop reads either a character that
    cannot start the word, or a run of partial matches ended by a character that breaks the last
    one; the string may end inside a partial match."""
    c0 = word[0]
    if c0 in word[1:]:
        raise ValueError("avoiding() needs a word whose first character does not recur")
    n = len(word)
    lit = [literal(c) for c in word]
    cls = [literal(c) if c not in "]\\^-" else "\\" + c for c in word]
    partial = lit[n - 2] if n > 2 else ""
    for k in range(n - 3, 0, -1):
        partial = f"{lit[k]}(?:{partial})?"
    p = f"{lit[0]}(?:{partial})?" if partial else lit[0]
    breaking = f"[^{cls[0]}{cls[n - 1]}]"
    for k in range(n - 2, 0, -1):
        breaking = f"(?:[^{cls[0]}{cls[k]}]|{lit[k]}{breaking})"
    q = f"{lit[0]}{breaking}"
    return f"(?:[^{cls[0]}]|(?:{p})*{q})*(?:{p})*"


PARAM_TEXT = avoiding("</parameter>")
# Whitespace between the tags, bounded: a model forced to call when it wanted to write words picks
# whitespace or another call over the end token (measured: 37 calls to list_dir for "say hello in
# French"), and eight bytes leave it nothing but the end token once the calls are written.
_S = r"[ \t\n\r]{0,8}"


def _xml_value(schema: dict) -> tuple[str, bool]:
    """(the value's regex inside `<parameter=K>...</parameter>`, whether it is a JSON literal).
    A string is free text without the closer (or one of its enum values); anything else is the
    JSON the template writes for it."""
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(schema, dict) and "enum" in schema and all(isinstance(v, str)
                                                             for v in schema["enum"]):
        return "(?:" + "|".join(literal(v) for v in schema["enum"]) + ")", False
    if t == "string" or not isinstance(schema, dict) or not (t or "enum" in schema
                                                              or "anyOf" in schema
                                                              or "oneOf" in schema
                                                              or "const" in schema):
        if isinstance(schema, dict) and t == "string" and "pattern" in schema:
            return f"(?:{schema['pattern']})", False
        return PARAM_TEXT, False
    return schema_regex(schema), True


#: SRV-619: the call markers a tool-call branch writes as the model's own ids
CALL_MARKERS = ("<tool_call>", "</tool_call>")


def calls_or_document(calls: str, document: str) -> str:
    """SRV-619: the answer as tool calls (`tool_call_regex`) or the document alone (a
    schema, `json_object`, a regex): one automaton, the branch fixed by the first character
    after the whitespace."""
    return f"(?:{calls})|(?:{document})"


def tool_call_regex(tools: list[dict], allowed: list[str], many: bool = True
                    ) -> tuple[str, dict[str, set[str]]]:
    """The answer as tool calls only, in the model's own XML format: `<tool_call>`, a function of
    `allowed`, the schema's parameters in declared order (the required ones present, no others),
    `</function></tool_call>`; `many`: further calls may follow (parallel calls). Whitespace between
    the tags is bounded (`_S`). Returns the regex and, per function, the parameters whose values
    are JSON literals (to be returned typed)."""
    by_name = {}
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            by_name[fn["name"]] = fn.get("parameters") or {}
    calls, typed = [], {}
    for name in allowed:
        params = by_name.get(name, {})
        props = params.get("properties") if isinstance(params, dict) else None
        if not props:
            body = rf"(?:{_S}<parameter=[^>\n]+>{PARAM_TEXT}</parameter>)*"
        else:
            required = set(params.get("required", []))
            keys = list(props)
            members = []
            for k in keys:
                value, is_json = _xml_value(props[k])
                if is_json:
                    typed.setdefault(name, set()).add(k)
                m = rf"{_S}<parameter={literal(k)}>{_S}{value}{_S}</parameter>"
                members.append(m if k in required else f"(?:{m})?")
            body = "".join(members)
        calls.append(rf"<tool_call>{_S}<function={literal(name)}>{body}{_S}</function>{_S}</tool_call>")
    one = "(?:" + "|".join(calls) + ")"
    # SRV-619: one copy of the call language for `many` (the automaton of `one (_S one)*` holds
    # it twice); the same strings
    return (f"{_S}(?:{one}{_S})+" if many else f"{_S}{one}{_S}"), typed
