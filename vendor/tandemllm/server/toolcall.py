"""Qwen XML tool calls -> the OpenAI `tool_calls` shape.

The model writes tool calls as text:

    <tool_call>
    <function=write_file>
    <parameter=path>
    /app/out.html
    </parameter>
    <parameter=content>
    <!DOCTYPE html> ...
    </parameter>
    </function>
    </tool_call>

Without a parser the block is literal content: the client receives no `tool_calls`, cannot act,
the user reads raw XML, and the payload inside the parameter eats the token budget (chat
efa916ed: `write_file` was attempted, the file arrived as a parameter, and the 16,384 cap cut it
mid-code).

Two pieces:

* `parse_tool_calls(text)` finds every COMPLETE block in a string and returns
  `(content_without_blocks, calls)` -- the non-streaming path, and the tests;
* `ToolCallBuffer` is the streaming counterpart: pieces are fed as they are detokenized, content
  is released immediately, and the text from an opener to its closer is held back until the block
  completes. A block that never closes or does not parse is released AS CONTENT -- an honest
  fallback: showing the text beats losing it.

Inside an opener, the buffer streams the call as OpenAI argument deltas WHILE the model writes it
(`drain_deltas()`): as soon as `<function=NAME>` is complete, one delta carries the id and the
name; each parameter's value then goes out as JSON-escaped fragments as it arrives, with the same
whitespace trimming the collector applies. A whole-file `write_file` argument is thousands of
tokens, and holding it silently made the client show NOTHING for minutes and then the entire call
at once (measured 2026-09-19, the rolex_svg case: a request looked hung for minutes, then every
chunk landed at the end). Calls streamed this way are listed in `streamed_ids` and are skipped by
the end-of-stream sweep; a block the machine cannot recognise is never partially streamed -- it
waits for the closer and follows the fallback above.

Two things a stream cannot do are what the rest of the machinery is about. It cannot take a delta
back, so a function whose name repeats the call before it -- the echo `_add` drops -- is not
streamed at all and goes out at closure if it survives the drop. And it cannot be re-read, so a
call is credited to `streamed_ids` only while what went out IS what `_parse_one` makes of the
closed block; anything else is corrected by the sweep. `finish()` is the end of a generation: its
content goes to the wire as it is, never back through `feed()`.

Nothing here is content-specific: any tool name and any parameters work; the arguments are the
JSON object of the parameter map, which is what OpenAI clients expect.

**Tiers.** Real traffic does not always write the strict form, so a block is read in
three tiers, and the stream machine follows the first two:

1. strict XML -- the form above;
2. lenient XML -- whitespace around `=` and inside the tags, the attribute style
   `<function name="f">` / `<parameter name="k">v</parameter>`, several functions in one block, a
   function that the next `<function` or the block's closer ends without `</function>`, and a block
   whose `</tool_call>` never came because the generation ENDED (EOS): it is a call if it parses,
   and content if it does not. A block cut by the token limit or a stop string stays content;
3. JSON -- `{"name": ..., "arguments": {...}}` (or a list of them, or the OpenAI
   `{"type": "function", "function": {...}}` shape) inside a `<tool_call>` block is a call. The
   same object as the WHOLE answer, with no tags, is converted only behind a gate, because an
   ordinary JSON answer looks the same: the request carried tools, the object (or a ```json fence
   around it) is everything the answer contains, its keys are only the call's, and its name is one
   of the request's tools. A stream holds an answer that opens with `{`, `[` or a fence until it is
   decided, and releases it as content the moment it cannot be a call.

**The boundary rule.** The format is XML-ish, not XML: nothing is escaped, and the FIRST
`</parameter>` after a parameter opener ends its value (the first `</function>` ends a function).
A value that has to contain the literal text `</parameter>` cannot be written in this format; a
client that needs one encodes it (base64, or its own escape) through its tool schema. The parser
never guesses where a value "really" ends.

Reasoning is not read here: the server hands this module the ANSWER only, under every reasoning
format -- a block the model writes inside `<think>` is a thought about a call. The one exception
is the session monitor's (SRV-199, E1 case C): a turn that ENDS with complete calls inside its
reasoning and has no answer gets those calls, the model's own tokens, forwarded as its answer, so
they come through this parser like any other.

**Hold.** With `hold=True` (the conductor's tool-call checks, server/conductor/monitor/) nothing
streams live: a call that closes waits in `held` until the server `release()`s it -- one whole
delta, its id made then and its index the next free one, so indexes stay contiguous from 0 and
a discarded call never leaves an id behind -- or `discard()`s it. A block that does not parse is
held too, as `{"bad": inner, "closed": ...}`, instead of falling back to content at once: the
checks may repair it, have the model write it again, or let it out as content after all
(`fallback()`). `cut()` drops everything held when the model is asked to write the calls of an
answer again, and a block the token limit cut is kept in `cut_block`, not in the content.
`answer_items()` reads a whole answer into the same items, for the checks at the end token.
"""

from __future__ import annotations

import json
import re
import time
import uuid

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_BLOCK = re.compile(re.escape(OPEN) + r"(.*?)" + re.escape(CLOSE), re.S)
# `<function=NAME>` with whitespace allowed around `=`, or `<function name="NAME">` (tier 2).
_FUNC_OPEN = re.compile(r'<function(?:\s*=\s*([^>\s]+)|\s+name\s*=\s*"([^"]*)")\s*>')
_PARAM_OPEN = re.compile(r'<parameter(?:\s*=\s*([^>\s]+)|\s+name\s*=\s*"([^"]*)")\s*>')
FUNC_CLOSE = "</function>"
PARAM_CLOSE = "</parameter>"
_TRIM = re.compile(r"\s*(.*?)\s*\Z", re.S)
# The keys a call object may carry, the whole-answer JSON gate's (tier 3) and its early exit.
_CALL_KEYS = frozenset(("name", "arguments", "parameters", "type", "id", "function"))
_FIRST_KEY = re.compile(r'[\[\s]*\{\s*"((?:[^"\\]|\\.)*)"')


def _call(name: str, params: dict) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(params)}}


def _add(calls: list[dict], new: dict) -> None:
    """Append unless it repeats the previous call exactly.

    The model sometimes echoes a tool call twice in one answer (measured 2026-09-19: the same
    write_file call emitted twice); a client would execute both. Consecutive exact duplicates are
    artefacts and are dropped.
    """
    if calls and calls[-1]["function"] == new["function"]:
        return
    calls.append(new)


def _tag_name(m: re.Match) -> str:
    return (m.group(1) if m.group(1) is not None else m.group(2)).strip().strip("'\"")


def _functions(inner: str) -> list[tuple[str, dict]]:
    """`(name, parameters)` of every function in a block, tiers 1 and 2.

    Sequential, as the regexes it replaced were: a parameter's value runs to the first
    `</parameter>`, a function to the first `</function>` -- or, without one, to the next
    `<function` or the end of the block. A parameter whose closer is missing (or lies past the
    function's) is not a parameter, exactly as before.
    """
    out = []
    pos = 0
    while True:
        m = _FUNC_OPEN.search(inner, pos)
        if m is None:
            return out
        name, params, p = _tag_name(m), {}, m.end()
        closed = dangling = False
        while True:
            pm = _PARAM_OPEN.search(inner, p)
            fc = inner.find(FUNC_CLOSE, p)
            nf = _FUNC_OPEN.search(inner, p)
            ends = [x for x in (fc, nf.start() if nf else -1) if x >= 0]
            end = min(ends) if ends else len(inner)
            if pm is None or pm.start() >= end:
                closed = end == fc and fc >= 0
                p = fc + len(FUNC_CLOSE) if closed else end
                break
            vc = inner.find(PARAM_CLOSE, pm.end())
            if vc < 0 or (fc >= 0 and vc > fc):
                dangling = True
                p = pm.start() + 1                 # not a parameter; look for the next opener
                continue
            params[_tag_name(pm)] = _TRIM.match(inner, pm.end(), vc).group(1)
            p = vc + len(PARAM_CLOSE)
        if name and (closed or not dangling):
            # Without its `</function>`, a function whose parameter never closed is a cut-off
            # write, not a call with one parameter fewer. A function with no parameters IS a
            # call: `get_time` takes none, and refusing it left the block in the content as raw
            # XML while the streamer had already sent half of one.
            out.append((name, params))
        pos = p


def _json_calls(text: str, names: set | None = None, whole: bool = False,
                types: dict | None = None) -> list[dict] | None:
    """Tier 3: a JSON call object (or a list of them) -> calls, or None if `text` is not one.

    `whole` is the gate for an answer with no tags: the text may be fenced as ```json, every
    object must carry `arguments` or `parameters` and no key a call does not have, and every name
    must be one of `names`, the request's tools. Inside a `<tool_call>` block the tags already say
    it is a call, so none of that is asked.
    """
    s = text.strip()
    if whole and s.startswith("```"):
        nl = s.find("\n")
        if nl < 0 or s[3:nl].strip() not in ("", "json") or not s.endswith("```"):
            return None
        s = s[nl + 1:-3].strip()
    if not s or s[0] not in "{[":
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    items = obj if isinstance(obj, list) else [obj]
    calls: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            return None
        fn = it.get("function") if isinstance(it.get("function"), dict) else it
        if whole and (set(it) - _CALL_KEYS or set(fn) - _CALL_KEYS):
            return None
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters"))
        if not isinstance(name, str) or not name.strip():
            return None
        if whole and (args is None or names is None or name not in names):
            return None
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        _add(calls, _call(name.strip(), _typed(name.strip(), args, types)))
    return calls or None


# the JSON types a parameter's schema allows, when a string is not one of them. The model
# writes every value as text (`<parameter=offset>150</parameter>`); a client validates the arguments
# against the tool's schema, so `"offset": "150"` is refused where `150` is taken (opencode, 2026-09-26:
# read offset/limit, bash timeout, todowrite todos, every MCP pageId -- 19 refused calls in 3 sessions).
_JSON_KINDS = frozenset(("integer", "number", "boolean", "null", "object", "array"))
# a tool_choice constraint wrote the value as a JSON literal: it parses as whatever it is
JSON_ANY = frozenset(("json",))


def _kinds(schema) -> frozenset | None:
    """The JSON types `schema` allows, or None when it allows a string or says nothing (the value
    then stays the text the model wrote)."""
    if not isinstance(schema, dict):
        return None
    kinds: set = set()
    t = schema.get("type")
    for x in (t if isinstance(t, list) else [t] if t is not None else []):
        if not isinstance(x, str):
            return None
        kinds.add(x)
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or ():
            k = _kinds(sub)
            if k is None:
                return None
            kinds |= k
    for v in list(schema.get("enum") or ()) + ([schema["const"]] if "const" in schema else []):
        kinds.add("string" if isinstance(v, str) else "boolean" if isinstance(v, bool)
                  else "integer" if isinstance(v, int) else "number" if isinstance(v, float)
                  else "null" if v is None else "array" if isinstance(v, list) else "object")
    if not kinds or not kinds <= _JSON_KINDS:
        return None                                        # a string, or a type this does not know
    return frozenset(kinds)


def schema_types(tools) -> dict:
    """Per function, per parameter, the JSON types its schema allows -- only the parameters whose
    schema does not allow a string. Tools that are not function tools, parameters without a
    schema and string parameters are absent: their values stay the text the model wrote."""
    out: dict = {}
    for t in tools or ():
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        if not isinstance(props, dict):
            continue
        typed = {k: kinds for k, kinds in ((k, _kinds(v)) for k, v in props.items()) if kinds}
        if typed:
            out[fn["name"]] = typed
    return out


def _kinds_of(types: dict | None, name: str | None) -> dict:
    """{parameter: kinds} of one function. the shape -- a set of parameter names the
    constraint wrote as JSON literals -- reads as JSON_ANY for each."""
    got = (types or {}).get(name) if name is not None else None
    if not got:
        return {}
    if isinstance(got, dict):
        return got
    return {k: JSON_ANY for k in got}


def _fits(v, kinds: frozenset) -> bool:
    if isinstance(v, bool):
        return "boolean" in kinds
    if isinstance(v, int):
        return "integer" in kinds or "number" in kinds
    if isinstance(v, float):
        return "number" in kinds or ("integer" in kinds and v.is_integer())
    if v is None:
        return "null" in kinds
    if isinstance(v, list):
        return "array" in kinds
    if isinstance(v, dict):
        return "object" in kinds
    return False


def convert(text: str, kinds: frozenset):
    """The value the model wrote as text, as the JSON value its schema asks for -- or the text
    unchanged when it is not one (the client's validator then says so, as it did before; nothing
    is guessed). `150` -> 150, `true` -> True, `[{"a": 1}]` -> a list; for an integer `150.0` ->
    150; `True`/`False` (the chat template renders a Python bool that way) -> a bool; an object or
    array written as a Python literal (single quotes) is read as one."""
    if "json" in kinds:
        try:
            return json.loads(text)
        except ValueError:
            return text
    try:
        v = json.loads(text)
    except ValueError:
        v = text
        low = text.strip().lower()
        if "boolean" in kinds and low in ("true", "false"):
            return low == "true"
        if kinds & {"object", "array"} and text.strip()[:1] in ("{", "["):
            try:
                import ast
                v = ast.literal_eval(text.strip())
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return text
            if not _fits(v, kinds):
                return text
            return v
        return text
    if not _fits(v, kinds):
        return text
    if isinstance(v, float) and "number" not in kinds:
        return int(v)                                     # an integer written as `150.0`
    return v


def _typed(name: str, params: dict, types: dict | None) -> dict:
    """The parameters whose schema does not allow a string, or that a tool_choice
    constraint wrote as JSON literals, as their values (`{"count": 3}`, not
    `{"count": "3"}`); a value that is not one keeps its text."""
    for k, kinds in _kinds_of(types, name).items():
        if k in params and isinstance(params[k], str):
            params[k] = convert(params[k], kinds)
    return params


def _parse_one(inner: str, types: dict | None = None) -> list[dict] | None:
    if inner.lstrip()[:1] in ("{", "["):
        return _json_calls(inner, types=types)
    calls: list[dict] = []
    for name, params in _functions(inner):
        _add(calls, _call(name, _typed(name, params, types)))
    return calls or None


def answer_items(text: str, names=None, eos: bool = False,
                 types: dict | None = None) -> list[dict]:
    """Every tool-call block of an answer, in order, as a holding `ToolCallBuffer` keeps them:
    the calls a block parses into, `{"bad": inner, "closed": bool}` for a block that does not
    parse, and `{"cut": inner}` for a block left open without the end token (the token limit or a
    stop string cut it). With no block at all, the whole-answer JSON gate (tier 3) as
    `parse_tool_calls` has it."""
    out: list[dict] = []
    i = 0

    def add(parsed, inner):
        # a call read without a parameter whose closer never came carries its block (`_block`),
        # so the checks can see what was dropped; the key never leaves the server
        for c in parsed:
            out.append(dict(c, _block=inner) if _dangling(inner) else c)

    for m in _BLOCK.finditer(text):
        parsed = _parse_one(m.group(1), types)
        if parsed:
            add(parsed, m.group(1))
        else:
            out.append({"bad": m.group(1), "closed": True})
        i = m.end()
    tail = text[i:]
    k = tail.find(OPEN)
    if k >= 0:
        inner = tail[k + len(OPEN):]
        parsed = _parse_one(inner, types) if eos else None
        if parsed:
            add(parsed, inner)
        else:
            out.append({"bad": inner, "closed": False} if eos else {"cut": inner})
    if not out and names:
        out = _json_calls(text, set(names), whole=True, types=types) or []
    return out


def _dangling(inner: str) -> bool:
    """Does a block hold a parameter `_functions` dropped, its `</parameter>` missing or past
    its function's end? (A cut-off write is not read as a call with one parameter fewer -- but
    a function that closed is, and the parameter is gone.)"""
    if inner.lstrip()[:1] in ("{", "["):
        return False
    for pm in _PARAM_OPEN.finditer(inner):
        vc = inner.find(PARAM_CLOSE, pm.end())
        fc = inner.find(FUNC_CLOSE, pm.end())
        if vc < 0 or (fc >= 0 and vc > fc):
            return True
    return False


def strip_blocks(text: str) -> str:
    """`text` without its tool-call blocks, a trailing open one included: what is left of an
    answer the model was asked to write again."""
    text = _BLOCK.sub("", text)
    k = text.find(OPEN)
    return text[:k] if k >= 0 else text


def diagnose(inner: str) -> str:
    """Why a block does not parse, in one line a model can act on (the hidden reminder quotes
    it). Only called for a block `_parse_one` refused."""
    s = inner.strip()
    if not s:
        return "the tool call block is empty"
    if s[0] in "{[":
        try:
            json.loads(s)
        except ValueError as exc:
            return f"the JSON in the tool call does not parse ({exc})"
        return "the JSON in the tool call is not a call: it needs a name and its arguments"
    m = _FUNC_OPEN.search(inner)
    if m is None:
        if "<function" in inner:
            return "the <function=NAME> opener is malformed"
        return "the block has no <function=NAME> opener"
    if not _tag_name(m):
        return "the <function=...> opener has no name"
    rest = inner[m.end():]
    for pm in _PARAM_OPEN.finditer(rest):
        if rest.find(PARAM_CLOSE, pm.end()) < 0:
            return f"parameter '{_tag_name(pm)}' has no </parameter>"
    return "the tool call does not parse"


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_UNQUOTED = re.compile(r"<(function|parameter)\s+name\s*=\s*([^\"'\s>][^\s>]*)\s*>")


def repair(inner: str, types: dict | None = None) -> tuple[list[dict], list[str]] | None:
    """A deterministic fix for a block that does not parse: `(calls, what was fixed)`, or None
    when there is none. Only fixes whose result is not a guess: a JSON object with trailing
    commas or written as a Python literal; `<function name=x>` without quotes; a parameter
    whose `</parameter>` is missing, closed where the next parameter, its function or the
    block begins. Values are typed by the schemas as every call is (`_typed`)."""
    s = inner.strip()
    fixed: list[str] = []
    if s[:1] in ("{", "["):
        cand = _TRAILING_COMMA.sub(r"\1", s)
        if cand != s:
            fixed.append("trailing comma")
        try:
            json.loads(cand)
        except ValueError:
            try:
                import ast
                cand = json.dumps(ast.literal_eval(cand))
                fixed.append("python literal")
            except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
                return None
        calls = _json_calls(cand, types=types)
        return (calls, fixed) if calls and fixed else None
    text = _UNQUOTED.sub(lambda m: f"<{m.group(1)}={m.group(2)}>", inner)
    if text != inner:
        fixed.append("unquoted name")
    out, pos = [], 0
    while True:
        pm = _PARAM_OPEN.search(text, pos)
        if pm is None:
            break
        nxt = [x for x in (text.find(PARAM_CLOSE, pm.end()), _next(_PARAM_OPEN, text, pm.end()),
                           text.find(FUNC_CLOSE, pm.end()), _next(_FUNC_OPEN, text, pm.end()))
               if x >= 0]
        end = min(nxt) if nxt else len(text)
        if text.startswith(PARAM_CLOSE, end):
            out.append(text[pos:end + len(PARAM_CLOSE)])
            pos = end + len(PARAM_CLOSE)
            continue
        out.append(text[pos:end].rstrip() + "\n" + PARAM_CLOSE + "\n")
        fixed.append(f"closed parameter '{_tag_name(pm)}'")
        pos = end
    text = "".join(out) + text[pos:]
    if not fixed:
        return None
    calls = _parse_one(text, types)
    return (calls, fixed) if calls else None


def _next(rx: re.Pattern, text: str, pos: int) -> int:
    m = rx.search(text, pos)
    return m.start() if m else -1


def _same_args(streamed: str, parsed: str) -> bool:
    """Do the fragments that went out spell the argument object the parser read?"""
    try:
        return json.loads(streamed) == json.loads(parsed)
    except ValueError:
        return False


def parse_tool_calls(text: str, names=None, eos: bool = False,
                     types: dict | None = None) -> tuple[str, list[dict]]:
    """Split every call out of an answer; returns (content, calls).

    `eos`: the generation ended on its end token, so a trailing block whose closer never came is a
    call if it parses (tier 2). `names`: the request's tool names, which turn on the whole-answer
    JSON gate (tier 3); None -- no tools, or `tool_choice: none` -- leaves it off. `types`: per
    function, the parameters a tool_choice constraint wrote as JSON literals.
    """
    calls: list[dict] = []
    out, i = [], 0
    for m in _BLOCK.finditer(text):
        parsed = _parse_one(m.group(1), types)
        if parsed is None:
            continue                         # leave it in the content: honest fallback
        out.append(text[i:m.start()])
        i = m.end()
        for call in parsed:
            _add(calls, call)
    tail = text[i:]
    k = tail.rfind(OPEN)
    if eos and k >= 0 and CLOSE not in tail[k:]:
        parsed = _parse_one(tail[k + len(OPEN):], types)
        if parsed is not None:
            out.append(tail[:k])
            tail = ""
            for call in parsed:
                _add(calls, call)
    out.append(tail)
    content = "".join(out)
    if not calls and names:
        found = _json_calls(content, set(names), whole=True, types=types)
        if found:
            return "", found
    return content, calls


_WS = " \t\r\n"


def _esc(s: str) -> str:
    """JSON string escaping, fragment-safe: every character escapes independently."""
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _is_prefix_of(text: str, tag: str) -> bool:
    return bool(text) and len(text) < len(tag) and tag.startswith(text)


def _open_tag(s: str, rx: re.Pattern, head: str) -> tuple[str, int] | None:
    """A function or parameter opener at the start of `s`, tiers 1 and 2: `(name, length)` when it
    is complete, `("", 0)` while it may still become one, None when it cannot."""
    if len(s) < len(head):
        return ("", 0) if head.startswith(s) else None
    if not s.startswith(head):
        return None
    k = s.find(">")
    if k < 0:
        return ("", 0)
    m = rx.match(s, 0, k + 1)
    if m is None or m.end() != k + 1:
        return None
    return _tag_name(m), k + 1


class _JsonEnd:
    """Where a JSON value that starts at index 0 ends, read incrementally: a long call object is
    scanned once, not once a piece."""

    def __init__(self):
        self.pos, self.depth, self.in_str, self.esc, self.end = 0, 0, False, False, None

    def scan(self, s: str) -> int | None:
        while self.end is None and self.pos < len(s):
            ch = s[self.pos]
            self.pos += 1
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
            elif ch == '"':
                self.in_str = True
            elif ch in "{[":
                self.depth += 1
            elif ch in "}]":
                self.depth -= 1
                if self.depth == 0:
                    self.end = self.pos
        return self.end


class ToolCallBuffer:
    """Streaming: feed detokenized pieces, get content to emit; tool calls collect at closure.

    `drain_deltas()` additionally yields the OpenAI `tool_calls` delta fragments vended live while
    a recognised parameter-XML call is being written.
    """

    def __init__(self, names=None, max_calls: int | None = None, types: dict | None = None,
                 hold: bool = False, live: bool = True):
        self.calls: list[dict] = []
        self.hold = bool(hold)
        # `live=False`: no call streams its arguments as they are written; each goes out whole
        # when its block closes (the end-of-stream sweep). The server passes it for a request
        # with stop strings: a stop string can end the generation INSIDE a block, and arguments
        # already streamed cannot be taken back -- the client would hold a call whose JSON never
        # closes, while the same request without `stream` keeps the block as text. Held, a cut
        # block goes out as the text it is, as the JSON answer has it, and no call goes out.
        self.live = bool(live)
        self.held: list[dict] = []
        self.discarded = 0
        self.cut_block: str | None = None   # holding: a block the token limit left open
        self.cut_live = False               # the generation ended inside a live-streamed block
        self.streamed_ids: set[str] = set()
        # per function, the parameters a tool_choice constraint writes as JSON literals --
        # streamed unquoted, as the values they are
        # per function, the parameters whose schema does not allow a string (schema_types) --
        # held to the parameter's closer and sent as the value they are
        self.types = types or {}
        self._kinds: frozenset | None = None
        self._tval = ""
        # `parallel_tool_calls: false`: at most this many calls leave; the rest are counted
        self.max_calls = max_calls
        self.dropped = 0
        # Tier 3's whole-answer gate: only a request with tools has one. `_jmode` is
        # "undecided" until the answer's first characters say whether it can be a JSON call,
        # "hold" while it still can, "off" for good once it cannot.
        self.names = set(names) if names else None
        self._jmode = "undecided" if self.names else "off"
        self._jbuf = ""
        self._jstart: int | None = None
        self._jfence = False
        self._jkey = False
        self._jend = _JsonEnd()
        self._buf = ""
        self._open = False
        self._deltas: list[dict] = []
        self._next_index = 0
        self._pos = 0              # how far into the open block the stream machine has read
        self._block_streamed = False
        self._streamed: list[dict] = []   # {id, name, args} per function streamed in this block
        self._hold = False         # stop streaming this block; it goes out at closure instead
        self._stream_name: str | None = None
        self._stream_index: int | None = None
        self._in_param = False
        self._value_started = False
        self._pending_ws = ""
        self._first_param = True
        self._done_params = False
        # `(perf_counter, kind, name)` at a block's open, a function's name and a block's
        # close -- once per call, never per piece -- for the live view; the newest 16 are kept
        self.events: list[tuple] = []

    def _event(self, kind: str, name) -> None:
        self.events.append((time.perf_counter(), kind, name))
        if len(self.events) > 16:
            del self.events[0]

    def drain_deltas(self) -> list[dict]:
        d, self._deltas = self._deltas, []
        return d

    def take_index(self) -> int:
        i = self._next_index
        self._next_index += 1
        return i

    def _collect(self, call: dict) -> None:
        """A parsed call: to the calls, or, holding, to the calls waiting for release."""
        if self.hold:
            self.held.append(call)
        else:
            _add(self.calls, call)

    def take_held(self) -> list[dict]:
        h, self.held = self.held, []
        return h

    def release(self, call: dict) -> dict | None:
        """Emit a held call whole: a fresh id, the next index, the whole arguments in one delta.
        None when it goes nowhere: the echo `_add` drops, or over `parallel_tool_calls: false`."""
        if self.calls and self.calls[-1]["function"] == call["function"]:
            return None
        if self.max_calls is not None and self._next_index >= self.max_calls:
            self.dropped += 1
            return None
        call = dict(call, id="call_" + uuid.uuid4().hex[:24])
        self.calls.append(call)
        self.streamed_ids.add(call["id"])
        self._deltas.append({"index": self.take_index(), "id": call["id"], "type": "function",
                             "function": {"name": call["function"]["name"],
                                          "arguments": call["function"]["arguments"]}})
        return call

    def discard(self, call: dict) -> None:
        """A held call that is not emitted: no id, no index, nothing on the wire."""
        self.discarded += 1

    @staticmethod
    def fallback(item: dict) -> str:
        """A held block that does not parse, as the content it is without the checks."""
        return OPEN + item["bad"] + (CLOSE if item.get("closed", True) else "")

    def cut(self) -> int:
        """The model writes this answer's calls again: everything held is dropped, an open block
        and a held JSON answer included. Content already sent stays sent. Returns the count."""
        n = len(self.held) + (1 if self._open else 0)
        self.discarded += n
        self.held, self._buf, self._open, self.cut_block = [], "", False, None
        if self._jmode == "hold":
            self._jmode, self._jbuf, self._jstart = "undecided", "", None
            self._jfence = self._jkey = False
            self._jend = _JsonEnd()
        return n

    def open_text(self) -> str | None:
        """The open block's text so far (after `<tool_call>`), None when no block is open: what
        the tool-argument loop guard reads (server/argloop.py)."""
        return self._buf if self._open else None

    def drop_open(self) -> bool:
        """The open block is dropped: nothing more of it goes out (the loop guard stopped it), and
        `finish` does not send its text as content. Calls held before it stay. A block that
        streamed live has sent its name and part of its arguments already; the client keeps a call
        whose arguments never close (opencode: an `invalid` tool result the model reads)."""
        if not self._open:
            return False
        self.discarded += 1
        self._buf, self._open = "", False
        self._end_function()
        self._hold = False
        return True

    def _cut_live(self) -> None:
        """The generation ended inside a block that streamed live (the token limit, the deadline,
        or an end token inside a value): the client holds the block's calls as they were sent.
        The block's text is NOT sent again as content (2026-10-04 16:18: a 65,536-token `write`
        went out once as arguments and once as text, and the next prompt was 157,806 tokens).
        A function the stream closed is a call the client has whole: it joins the calls under
        its streamed id, so the sweep does not send it again. The one left open stays as the
        client got it, arguments that never close: opencode answers it with the parse error, as
        it does for the loop guard's stop (`drop_open`)."""
        for rec in self._streamed:
            if self._stream_name is not None and rec is self._streamed[-1]:
                break                                     # the function the cut left open
            try:
                json.loads(rec["args"])
            except ValueError:
                break
            call = {"id": rec["id"], "type": "function",
                    "function": {"name": rec["name"], "arguments": rec["args"]}}
            self.calls.append(call)
            self.streamed_ids.add(rec["id"])
        self.discarded += 1
        self.cut_live = True
        self._buf, self._open = "", False
        self._end_function()
        self._hold = False

    @property
    def open_streamed(self) -> bool:
        """The open block has sent argument deltas to the client."""
        return self._open and self._block_streamed

    def _delta_args(self, fragment: str) -> None:
        self._streamed[-1]["args"] += fragment
        self._deltas.append({"index": self._stream_index,
                             "function": {"arguments": fragment}})

    def _previous_name(self) -> str | None:
        """The name of the call before this one -- in this block, or the last one collected."""
        if self._streamed:
            return self._streamed[-1]["name"]
        return self.calls[-1]["function"]["name"] if self.calls else None

    def _close_function(self) -> None:
        """The streamed function's arguments are complete: close the object."""
        # `{}` when the function took no parameters: `}` on its own is not JSON.
        self._delta_args("}" if not self._first_param else "{}")
        self._end_function()

    def _block_ends(self) -> None:
        """The block ends here (its closer, or EOS): a function still open without `</function>`
        ends with it (tier 2), so what went out is a complete argument object."""
        if (self._stream_name is not None and not self._in_param and not self._hold
                and not self._done_params):
            self._close_function()

    def _end_function(self) -> None:
        """`</function>`: the next `<function=` in the same block starts a new call."""
        self._stream_name = None
        self._in_param = False
        self._value_started = False
        self._pending_ws = ""
        self._first_param = True
        self._kinds, self._tval = None, ""

    def _stream_open(self, limit: int) -> None:
        """Consume the open block as far as it is safe, vending argument deltas."""
        while True:
            if self._hold:
                return
            rest = self._buf[self._pos:limit]
            if self._stream_name is None:
                if self._done_params:
                    return
                stripped = rest.lstrip(_WS)
                lead = len(rest) - len(stripped)
                if not stripped:
                    self._pos += lead
                    return
                tag = _open_tag(stripped, _FUNC_OPEN, "<function")
                if tag is None:
                    self._done_params = True                     # unknown shape: fallback
                    return
                name, k = tag
                if not k:
                    return                                       # hold the partial tag
                self._pos += lead + k
                if not name:
                    self._done_params = True
                    return
                if self.max_calls is not None and self._next_index >= self.max_calls:
                    self._hold = True                            # over the cap: never streamed
                    return
                if name == self._previous_name():
                    # The model echoes a call twice (measured 2026-09-19) and `_add` drops the
                    # repeat -- but a repeat already streamed cannot be taken back, so streaming
                    # delivered two and JSON one. A function whose name repeats the call before it
                    # is not streamed live: it goes out at closure, through the end-of-stream
                    # sweep, if `_add` keeps it at all.
                    self._hold = True
                    return
                self._stream_name = name
                self._event("name", name)
                self._stream_index = self.take_index()
                self._block_streamed = True
                self._streamed.append({"id": "call_" + uuid.uuid4().hex[:24],
                                       "name": name, "args": ""})
                self._deltas.append({"index": self._stream_index,
                                     "id": self._streamed[-1]["id"], "type": "function",
                                     "function": {"name": name, "arguments": ""}})
                continue
            if self._done_params:
                return
            if not self._in_param:
                stripped = rest.lstrip(_WS)
                lead = len(rest) - len(stripped)
                if not stripped:
                    self._pos += lead
                    return
                tag = _open_tag(stripped, _PARAM_OPEN, "<parameter")
                if tag is not None and not tag[1]:
                    return                                       # the tag is not complete yet
                if tag is not None:
                    key, k = tag
                    self._pos += lead + k
                    self._in_param = True
                    self._value_started = False
                    self._pending_ws = ""
                    # A typed value is short (a number, a flag, a small list) and must be read whole
                    # before it can be written as JSON: it is held to its closer. A string value
                    # streams as it is written (a whole-file `content` is thousands of tokens).
                    self._kinds = _kinds_of(self.types, self._stream_name).get(key)
                    self._tval = ""
                    lead_sep = "{" if self._first_param else ", "
                    self._first_param = False
                    self._delta_args(lead_sep + json.dumps(key)
                                     + (": " if self._kinds is not None else ': "'))
                    continue
                if stripped.startswith("</function>"):
                    self._pos += lead + len("</function>")
                    self._close_function()
                    continue
                if stripped[0] == "<":
                    if _is_prefix_of(stripped, "</function>"):
                        return
                    tag = _open_tag(stripped, _FUNC_OPEN, "<function")
                    if tag is not None and not tag[1]:
                        return
                    if tag is not None:
                        # The next function, and this one never wrote `</function>` (tier 2): it
                        # ends here, and the loop reads the opener as the start of the next.
                        self._close_function()
                        continue
                    k = stripped.find(">")
                    if k < 0:
                        return                                   # the tag is not complete yet
                    # An unknown tag between parameters. `_parse_one` ignores it, so the stream
                    # does too: stopping here truncated the arguments to invalid JSON while the
                    # parser went on to read the whole block.
                    self._pos += lead + k + 1
                    continue
                self._pos += lead + 1                            # stray text between parameters
                continue
            lt = rest.find("<")
            if lt < 0:
                self._pos += len(rest)
                self._value(rest)
                return
            tail = rest[lt:]
            if "</parameter>".startswith(tail):
                self._pos += lt
                self._value(rest[:lt])
                return
            if tail.startswith("</parameter>"):
                self._value(rest[:lt])
                self._pending_ws = ""
                if self._kinds is not None:
                    # the collector's trim, then the same conversion `_typed` applies
                    self._delta_args(json.dumps(convert(_TRIM.match(self._tval).group(1),
                                                        self._kinds)))
                    self._kinds, self._tval = None, ""
                else:
                    self._delta_args('"')
                self._in_param = False
                self._pos += lt + len("</parameter>")
                continue
            self._pos += lt + 1
            self._value(rest[:lt + 1])

    def _value(self, text: str) -> None:
        """A piece of the open parameter's value: held whole when typed, streamed when a string."""
        if self._kinds is not None:
            self._tval += text
        else:
            self._emit_value(text)

    def _emit_value(self, text: str) -> None:
        if not text:
            return
        out = []
        for ch in text:
            if not self._value_started:
                if ch in _WS:
                    continue                                  # leading whitespace is trimmed
                self._value_started = True
            if ch in _WS:
                self._pending_ws += ch
                continue
            if self._pending_ws:
                out.append(_esc(self._pending_ws))
                self._pending_ws = ""
            out.append(_esc(ch))
        if out:
            self._delta_args("".join(out))

    def _json_verdict(self) -> str:
        """Can the answer held so far still be one JSON call? "wait", "hold" or "no".

        Incremental: the body's start is found once, its first key read once, and after that only
        the characters that arrived since the last piece are scanned -- a whole-file argument must
        not be re-read once a piece.
        """
        buf = self._jbuf
        if self._jstart is None:
            lead = len(buf) - len(buf.lstrip())
            t = buf[lead:]
            if not t:
                return "wait"
            if t[0] in "{[":
                self._jstart = lead
            else:
                if not (t.startswith("```") or "```".startswith(t)):
                    return "no"
                nl = t.find("\n")
                if nl < 0:
                    return "wait" if len(t) <= len("```json ") else "no"
                if t[3:nl].strip() not in ("", "json"):
                    return "no"
                body = t[nl + 1:]
                if not body.strip():
                    return "wait"
                self._jfence = True
                self._jstart = lead + nl + 1 + len(body) - len(body.lstrip())
                if buf[self._jstart] not in "{[":
                    return "no"
            self._jend.pos = self._jstart
        if not self._jkey:
            # the first object's first key: `{"name"`, or `[{"name"` for a list of calls
            i = self._jstart
            while i < len(buf) and buf[i] in "[ \t\r\n":
                i += 1
            if i < len(buf) and buf[i] != "{":
                return "no"
            i += 1
            while i < len(buf) and buf[i] in " \t\r\n":
                i += 1
            if i < len(buf) and buf[i] != '"':
                return "no"
            m = _FIRST_KEY.match(buf, self._jstart)
            if m is not None:
                if m.group(1) not in _CALL_KEYS:
                    return "no"                  # an ordinary JSON answer: release it now
                self._jkey = True
        end = self._jend.scan(buf)
        if end is None:
            return "hold"
        rest = buf[end:].strip()
        if self._jfence:
            return "hold" if "```".startswith(rest) else "no"
        return "hold" if not rest else "no"

    def feed(self, piece: str) -> list[str]:
        """Content pieces to emit now. Text inside an opener..closer is withheld."""
        if self._jmode != "off":
            self._jbuf += piece
            verdict = self._json_verdict()
            if verdict != "no":
                self._jmode = "hold" if verdict == "hold" else "undecided"
                return []
            # It cannot be a JSON call: everything held goes the ordinary way, now and from here on.
            self._jmode, piece, self._jbuf = "off", self._jbuf, ""
        out: list[str] = []
        self._buf += piece
        while True:
            if not self._open:
                i = self._buf.find(OPEN)
                if i < 0:
                    # Hold back ONLY the longest tail that could be the start of an opener --
                    # never a fixed window. Holding 9 chars unconditionally delayed the first
                    # content delta and moved the row's TTFT by +14 % (measured 2026-09-19, the
                    # certification run that caught it).
                    keep = 0
                    for k in range(min(len(self._buf), len(OPEN) - 1), 0, -1):
                        if OPEN.startswith(self._buf[-k:]):
                            keep = k
                            break
                    if len(self._buf) > keep:
                        out.append(self._buf[:len(self._buf) - keep])
                        self._buf = self._buf[-keep:] if keep else ""
                    return out
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(OPEN):]
                self._open = True
                self._event("open", None)
                self._pos = 0
                self._block_streamed = False
                self._streamed = []
                self._hold = False
                self._stream_name = None
                self._stream_index = None
                self._in_param = False
                self._value_started = False
                self._pending_ws = ""
                self._first_param = True
                self._done_params = False
                self._kinds, self._tval = None, ""
                if not self.live:
                    self._hold = True        # this block goes out at closure, whole, or as text
            j = self._buf.find(CLOSE)
            if not self.hold:
                self._stream_open(j if j >= 0 else len(self._buf))
            if j < 0:
                return out
            if not self.hold:
                self._block_ends()
            inner = self._buf[:j]
            self._buf = self._buf[j + len(CLOSE):]
            self._open = False
            parsed = _parse_one(inner, self.types)
            self._event("close", [c["function"]["name"] for c in parsed] if parsed else None)
            if parsed:
                self._credit_streamed(parsed)
                for call in parsed:
                    self._collect(call)
            elif self.hold:
                self.held.append({"bad": inner, "closed": True})   # the checks decide
            elif not self._block_streamed:
                out.append(OPEN + inner + CLOSE)     # honest fallback

    def _credit_streamed(self, parsed: list[dict]) -> None:
        """Give the calls that WERE streamed live their streamed id, so the end-of-stream sweep
        does not send them a second time.

        The n-th parsed call is the n-th function of the block -- a function is streamed only up to
        the first repeated name, which is the only place `_add` can drop one -- so the two lists
        line up from the front. A call is credited only while what went out over the wire IS what
        the parser read: if a tag the machine skipped ate a parameter, the client is holding half
        an argument object, and then the complete call has to be sent by the sweep rather than
        suppressed as "already streamed".
        """
        for call, rec in zip(parsed, self._streamed):
            fn = call["function"]
            if rec["name"] != fn["name"] or not _same_args(rec["args"], fn["arguments"]):
                return
            call["id"] = rec["id"]
            self.streamed_ids.add(rec["id"])

    def flush(self) -> str:
        """At the end of a generation: anything still held goes out as content."""
        s = (OPEN + self._buf) if self._open else self._buf
        self._buf, self._open = "", False
        return s

    def finish(self, eos: bool = False) -> tuple[str, list[dict]]:
        """End of the generation: (content still held, the deltas for calls not streamed live).

        The content goes to the wire AS IS. Feeding it back through `feed()` -- which is what
        routing it through the server's own `send()` did -- re-opens the block it still contains,
        streams its arguments a second time at a new index, and swallows the fallback text; with
        nothing but a partial opener held ("hello <tool") it swallowed all of it.

        `eos`: the generation ended on its end token. A block still open then is a call if it
        parses (tier 2); cut by the token limit or a stop string, it is content as before.
        Holding, the first that does not parse is held as a bad block and the second goes to
        `cut_block`: the checks decide what the client gets of either.
        """
        if self._jmode == "hold":
            found = _json_calls(self._jbuf, self.names, whole=True, types=self.types)
            if found:
                self._jbuf = ""
                for call in found:
                    self._collect(call)
        left = self._jbuf
        self._jbuf, self._jmode = "", "off"
        if eos and self._open:
            if not self.hold:
                self._stream_open(len(self._buf))
                self._block_ends()
            parsed = _parse_one(self._buf, self.types)
            if parsed:
                self._buf, self._open = "", False
                self._event("close", [c["function"]["name"] for c in parsed])
                self._credit_streamed(parsed)
                for call in parsed:
                    self._collect(call)
            elif self.hold:
                self.held.append({"bad": self._buf, "closed": False})
                self._buf, self._open = "", False
        elif self.hold and self._open:
            # cut by the token limit or a stop string: never half a call, and never content
            # before the checks said so
            self.cut_block, self._buf, self._open = self._buf, "", False
        if self._open and self._block_streamed:
            self._cut_live()
        left += self.flush()
        deltas, kept = [], []
        for call in self.calls:
            if call["id"] in self.streamed_ids:
                kept.append(call)
                continue
            if self.max_calls is not None and self._next_index >= self.max_calls:
                self.dropped += 1
                continue
            kept.append(call)
            deltas.append({"index": self.take_index(), "id": call["id"], "type": "function",
                           "function": {"name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"]}})
        self.calls = kept
        return left, deltas


# ------------------------------------------------------------------------- Pythonic tool calls (SRV-425)
# LiquidAI's LFM2 models write their calls as a Python list of calls between two tokens of their own:
#
#     <|tool_call_start|>[get_weather(city='Berlin', days=3), search(q="a, b")]<|tool_call_end|>
#
# `PythonicCalls` turns each such block into this module's own JSON tier, `<tool_call>[{"name": ...,
# "arguments": {...}}, ...]</tool_call>`, as the text arrives, so everything above (the stream machine,
# the holds of the conductor's checks, `answer_items`, `parse_tool_calls`) reads it as it reads any call.
# A value is a Python literal (`ast.literal_eval`: strings, numbers, lists, dicts, tuples, True / False /
# None, and the JSON spellings true / false / null), never evaluated code; a block that is not a list of
# calls with keyword arguments only goes out as the text it was (the honest fallback, as for XML).
PY_OPEN = "<|tool_call_start|>"
PY_CLOSE = "<|tool_call_end|>"
_PY_NAMES = {"true": True, "false": False, "null": None, "none": None}


def _py_value(node):
    import ast
    if isinstance(node, ast.Name) and node.id.lower() in _PY_NAMES:
        return _PY_NAMES[node.id.lower()]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_py_value(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("a ** inside a dict")
        return {_py_value(k): _py_value(v) for k, v in zip(node.keys, node.values)}
    v = ast.literal_eval(node)
    if isinstance(v, (set, bytes, complex)):
        raise ValueError(f"{type(v).__name__} is not a JSON value")
    return v


def _py_name(node) -> str:
    import ast
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _py_name(node.value) + "." + node.attr
    raise ValueError("a call's name must be a name")


def pythonic_calls(inner: str) -> list[dict] | None:
    """The calls of one block's inner text, [{"name", "arguments"}], or None if it is not a list of
    calls (or one call) with keyword arguments only."""
    import ast
    s = inner.strip()
    if not s:
        return None
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError:
        return None
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    out = []
    try:
        for it in items:
            if not isinstance(it, ast.Call) or it.args or any(k.arg is None for k in it.keywords):
                return None
            out.append({"name": _py_name(it.func), "arguments": {k.arg: _py_value(k.value) for k in it.keywords}})
    except (ValueError, TypeError, SyntaxError):
        return None
    return out or None


def pythonic_block(inner: str) -> str | None:
    """One Pythonic block as this module's JSON tier, or None when it does not parse."""
    calls = pythonic_calls(inner)
    if calls is None:
        return None
    return OPEN + json.dumps(calls, ensure_ascii=False) + CLOSE


class PythonicCalls:
    """The stream's rewrite of Pythonic blocks (above): `feed(piece)` returns the text to pass on, a
    block's opener and what follows it held until its closer; `finish(eos)` the rest -- an open block
    that parses is a call when the generation ended (EOS), and text otherwise (a block the token limit
    or a stop string cut stays content, as for XML)."""

    def __init__(self):
        self._buf = ""
        self._open = False
        self.blocks = 0          # blocks rewritten
        self.bad = 0             # blocks released as text

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = ""
        while True:
            if not self._open:
                i = self._buf.find(PY_OPEN)
                if i < 0:
                    k = len(self._buf)
                    # keep a tail that could still become the opener
                    for n in range(min(len(PY_OPEN) - 1, len(self._buf)), 0, -1):
                        if PY_OPEN.startswith(self._buf[-n:]):
                            k = len(self._buf) - n
                            break
                    out += self._buf[:k]
                    self._buf = self._buf[k:]
                    return out
                out += self._buf[:i]
                self._buf = self._buf[i + len(PY_OPEN):]
                self._open = True
            j = self._buf.find(PY_CLOSE)
            if j < 0:
                return out
            out += self._close(self._buf[:j], closed=True)
            self._buf = self._buf[j + len(PY_CLOSE):]
            self._open = False

    def _close(self, inner: str, closed: bool) -> str:
        got = pythonic_block(inner)
        if got is None:
            self.bad += 1
            return PY_OPEN + inner + (PY_CLOSE if closed else "")
        self.blocks += 1
        return got

    def finish(self, eos: bool = False) -> str:
        buf, self._buf = self._buf, ""
        if not self._open:
            return buf
        self._open = False
        if eos:
            return self._close(buf, closed=False)
        self.bad += 1
        return PY_OPEN + buf


def rewrite_pythonic(text: str, eos: bool = False) -> str:
    """A whole answer with its Pythonic blocks rewritten (the non-streamed path)."""
    p = PythonicCalls()
    return p.feed(text) + p.finish(eos=eos)
