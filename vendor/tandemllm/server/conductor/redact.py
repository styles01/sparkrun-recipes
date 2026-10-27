"""The secret-pattern refusal (SRV-137; the note rules of SRV-141 use the same list).

Nothing the runtime writes to disk from a session may hold a secret: a transcript block passes
through `redact()` before it is written, and every span that matches one of the shapes below is
replaced by `[redacted:<kind>]`. The caller journals the counts per kind, never the value.

The shapes are the ones that turn up in a coding session: private-key blocks, bearer and basic
authorisation headers, the token formats of the services a project talks to, JWTs, credentials in
a URL or a connection string, and a password, secret or token assigned to a quoted string or an
upper-case environment name. A shape errs on the side of redacting: an identifier that only looks
like a token loses its value in the transcript, which costs a reader a little; a real key in a file
on the box would cost much more. Plain code (`token = tok(text)`, `password: str`) does not match:
the assignment shapes want a quoted string or an environment name.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import threading

# (kind, pattern, the group that is the secret: 0 for the whole match)
PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
                               r"[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"), 0),
    # a key block whose end was cut off still holds key material
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*\Z"), 0),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/]{16,}=*)"), 1),
    ("basic_auth", re.compile(r"(?i)\bauthorization:\s*basic\s+([A-Za-z0-9+/]{8,}=*)"), 1),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"), 0),
    ("aws_secret", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})"), 1),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"), 0),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"), 0),
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{16,}"), 0),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), 0),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), 0),
    ("ntfy_token", re.compile(r"\btk_[a-z0-9]{29}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@\"'<>]+:([^\s@/\"'<>]+)@"), 1),
    ("connection_string", re.compile(r"(?i)\b(?:AccountKey|SharedAccessKey|Password|Pwd)=([^;\s\"']{6,})"), 1),
    ("password", re.compile(r"(?i)(?:password|passwd|passphrase|secret|api[_-]?key|access[_-]?token|"
                            r"auth[_-]?token|client[_-]?secret|private[_-]?key)[\"']?\s*[:=]\s*"
                            r"[\"']([^\"'\s]{6,})[\"']"), 1),
    ("password", re.compile(r"(?i)--(?:password|passwd|token|api-key|secret)(?:=|\s+)"
                            r"(?![-$])([^\s\"']{6,})"), 1),
    ("env_secret", re.compile(r"\b[A-Z][A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|"
                              r"PRIVATE_KEY)[A-Z0-9_]*\s*[=:]\s*[\"']?([^\s\"'$][^\s\"']{5,})"), 1),
]
KINDS = tuple(dict.fromkeys(k for k, _, _ in PATTERNS))


def redact(text: str) -> tuple[str, dict[str, int]]:
    """`text` with every secret span replaced by `[redacted:<kind>]`, and the count per kind.
    The shapes run in order on what the earlier ones left, so a key inside a URL inside an
    assignment is counted once."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    for kind, rx, group in PATTERNS:
        def sub(m: re.Match, kind=kind, group=group) -> str:
            if m.group(group).startswith("[redacted:"):
                return m.group(0)                  # an earlier shape took it already
            counts[kind] = counts.get(kind, 0) + 1
            if group == 0:
                return f"[redacted:{kind}]"
            s, e = m.span(group)
            return m.group(0)[:s - m.start()] + f"[redacted:{kind}]" + m.group(0)[e - m.start():]
        text = rx.sub(sub, text)
    return text, counts


def merge(into: dict[str, int], more: dict[str, int]) -> dict[str, int]:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v
    return into


# ---------------------------------------------------------------- SRV-625: the answer side
# A credential a tool result showed the model is never reproduced in the answer the client sees.
# The tool results of the request's conversation are read with PATTERNS and the short forms below;
# every secret span found there is masked where the answer repeats it verbatim. The answer itself
# is never read for shapes: a span that did not come from a tool result is never masked, which
# is what keeps the short forms (a six-character key, a PIN) from firing on ordinary text.

# The short forms: key prefixes with a shorter body than PATTERNS asks, and unquoted assignments.
# A span is one token-like run with a letter in it; a short form's (and an unquoted environment
# or connection-string value's) needs a digit too, a PIN four to twelve digits. So prose (`bearer
# tokens`), code (`token = tokenizer(text)`, `X_TOKEN = int(_S.get(...`) and plain numbers
# (`MAX_TOKENS=128000`) are no span.
SHORT_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{6,}"), 0),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{6,}"), 0),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/]{6,}=*)"), 1),
    ("password", re.compile(r"(?i)\b(?:password|passwd|pwd|passphrase|secret|token|api[_-]?key|"
                            r"access[_-]?token|auth[_-]?token|client[_-]?secret)[\"']?\s*[:=]\s*"
                            r"[\"']?([A-Za-z0-9_\-+/=.~!@#$%^&*]{6,})(?=[\s\"'`,;]|\Z)"), 1),
    ("pin", re.compile(r"(?i)\bpin(?:[ _-]?code)?[\"']?\s*[:=]\s*[\"']?(\d{4,12})\b"), 1),
]
# the part of a span the mask keeps: the kind's own prefix, never key material
_PREFIX = re.compile(r"^(?:sk-(?:ant-|proj-)?|(?:sk|rk|pk)_(?:live|test)_|gh[pousr]_|github_pat_|"
                     r"glpat-|xox[abposr]-|AKIA|ASIA|ABIA|ACCA|AIza|hf_|npm_|tk_|eyJ)")
MASK = "[redacted]"


# key material is one run of these characters (a private key's block aside)
_TOKENLIKE = re.compile(r"[A-Za-z0-9_\-+/=.~:@!#$%^&*|?]+")
# PATTERNS' unquoted shapes, where code sits as well
_NEEDS_DIGIT = frozenset(("env_secret", "connection_string"))


def _span_ok(kind: str, span: str, short: bool) -> bool:
    if kind == "pin":
        return span.isdigit() and 4 <= len(span) <= 12
    if len(span) < 6 or not any(c.isalpha() for c in span):
        return False
    if kind == "private_key":
        return True
    if not _TOKENLIKE.fullmatch(span):
        return False
    if (short or kind in _NEEDS_DIGIT) and not any(c.isdigit() for c in span):
        return False
    return True


def _leaves(obj, out: list[str], depth: int = 0) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif depth < 16 and isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, out, depth + 1)
    elif depth < 16 and isinstance(obj, list):
        for v in obj:
            _leaves(v, out, depth + 1)


def _texts(text: str) -> list[str]:
    """A tool result's text, and, when it is JSON (most tool results are), its string values
    decoded, so an escape (`\\n`) never hides a span's start."""
    out = [text]
    if text.lstrip()[:1] in ("{", "[", "\""):
        try:
            _leaves(json.loads(text), out)
        except ValueError:
            pass
    return out


_SHAPES = ([(k, rx, g, True) for k, rx, g in SHORT_PATTERNS[:2]]      # the key prefixes first,
           + [(k, rx, g, False) for k, rx, g in PATTERNS]               # so `API_KEY=sk-...` is
           + [(k, rx, g, True) for k, rx, g in SHORT_PATTERNS[2:]])     # counted as the key
# one tool result's spans by the digest of its text: a conversation's results are read once, not
# on every turn that carries them again (0.45 s a megabyte on the box, 1 ms a megabyte cached)
_CACHE: "collections.OrderedDict[bytes, list[tuple[str, str]]]" = collections.OrderedDict()
_CACHE_MAX = 4096
_LOCK = threading.Lock()


def _result_spans(text: str) -> list[tuple[str, str]]:
    key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
    with _LOCK:
        got = _CACHE.get(key)
        if got is not None:
            _CACHE.move_to_end(key)
            return got
    out: list[tuple[str, str]] = []
    for t in _texts(text):
        for kind, rx, group, short in _SHAPES:
            for mt in rx.finditer(t):
                span = mt.group(group)
                if not span or not _span_ok(kind, span, short):
                    continue
                out.append((kind, span))
                if kind == "private_key":
                    out += [("private_key", ln.strip()) for ln in span.splitlines()
                            if len(ln.strip()) >= 20 and not ln.strip().startswith("-----")]
    with _LOCK:
        _CACHE[key] = out
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return out


def tool_spans(messages) -> dict[str, str]:
    """The secret spans of the tool results in `messages` (role `tool`, or `function`): span ->
    kind. The key prefixes, then PATTERNS, then the other short forms; the first shape to find a
    span names its kind. A private key's body lines are spans too, so an answer that quotes part
    of the block loses them as well."""
    spans: dict[str, str] = {}
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") not in ("tool", "function"):
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts = [c]
        elif isinstance(c, list):
            parts = [p["text"] for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)]
        else:
            parts = []
        for text in parts:
            for kind, span in _result_spans(text):
                spans.setdefault(span, kind)
    return spans


def masked(span: str) -> str:
    """`sk-abc123` -> `sk-[redacted]`: the kind's prefix stays, the rest goes."""
    m = _PREFIX.match(span)
    return (m.group(0) if m else "") + MASK


def _word(c: str) -> bool:
    return c.isalnum() or c == "_"


class _Channel:
    def __init__(self):
        self.buf = ""
        self.prev = ""


class AnswerMask:
    """The answer's text with every span of `spans` (tool_spans) masked where it appears
    verbatim, as a whole word (the characters around it are no letter, digit or underscore where
    the span's own end is one). One state per output field (`content`, `reasoning_content`).

    Streaming: only a tail that is still the start of a span is held back (a span that ends at the
    piece's end waits for one more character, the boundary); every other character goes out with
    the piece it came in. `finish(field)` ends that field: what is held is masked or sent.
    `replace` False (the profile's `shadow`): spans are counted and sent unchanged. `counts`: per
    kind, never the value."""

    def __init__(self, spans: dict[str, str], replace: bool = True):
        self.spans = spans
        self.replace = replace
        self.counts: dict[str, int] = {}
        self._first: dict[str, list[str]] = {}
        for s in sorted(spans, key=len, reverse=True):
            self._first.setdefault(s[0], []).append(s)
        self._ch: dict[str, _Channel] = {}

    def held(self, field: str) -> bool:
        ch = self._ch.get(field)
        return ch is not None and bool(ch.buf)

    def push(self, piece: str, field: str = "content") -> str:
        if not piece:
            return piece
        ch = self._ch.setdefault(field, _Channel())
        ch.buf += piece
        return self._run(ch, final=False)

    def finish(self, field: str = "content") -> str:
        ch = self._ch.get(field)
        if ch is None or not ch.buf:
            return ""
        return self._run(ch, final=True)

    def text(self, text: str | None, field: str = "content") -> str | None:
        """A whole answer (the non-streamed path)."""
        if not text:
            return text
        return self.push(text, field) + self.finish(field)

    def _run(self, ch: _Channel, final: bool) -> str:
        buf, out, i, n = ch.buf, [], 0, len(ch.buf)
        while i < n:
            c = buf[i]
            hit = None
            for s in self._first.get(c, ()):
                end = i + len(s)
                if end <= n and buf.startswith(s, i):
                    if ch.prev and _word(ch.prev) and _word(s[0]):
                        continue                         # inside a longer word
                    if end == n and not final and _word(s[-1]):
                        hit = "hold"                     # the boundary is the next character
                        break
                    if end < n and _word(buf[end]) and _word(s[-1]):
                        continue
                    hit = s
                    break
                if end > n and not final and s.startswith(buf[i:]):
                    if not (ch.prev and _word(ch.prev) and _word(s[0])):
                        hit = "hold"                     # it may still become the span
                        break
            if hit == "hold":
                break
            if hit is not None:
                kind = self.spans[hit]
                self.counts[kind] = self.counts.get(kind, 0) + 1
                out.append(masked(hit) if self.replace else hit)
                ch.prev = hit[-1]
                i += len(hit)
                continue
            out.append(c)
            ch.prev = c
            i += 1
        ch.buf = buf[i:]
        return "".join(out)
