"""The text of a skill or package, for the router to inject (VIS-E6 packages, inside the harness).

A package in the `packages` document is either `inline` (its `body` is the text, edited in the
dashboard) or points at a file on the box: `skill:<name>` is `<dir>/<name>/SKILL.md` in the first
of `skill_dirs` that has it (opencode's `~/.config/opencode/skills` layout), `file:<path>` is one
file. A file is read when first needed and again when its mtime moves. The YAML frontmatter
(`---` ... `---`) is the client's metadata, not instructions, so it is stripped from the text and
its `description` is offered to the dashboard's import.

A library skill (R1, `library: true`, server/conductor/libsearch.py) is a SKILL.md whose text before
its `## Reference` heading is what goes in; the reference is read only on request
(`GET /v1/library/{name}`).

THE SHIPPED TEXTS. The repository carries the packages a new box starts with
(`packages/` at its root: `inline/` the inline bodies the seed document holds, `skills/` and
`library/` the SKILL.md files): a `skill_dirs` entry may start with `{repo}`, the directory of the
running code, so the seed finds them without an install step. A text that tells the model to call
TandemLLM from its own scripts (Laya, the library) names the provider `{tandem_url}`; the router
puts in the address the request came in on (`Library.at`, from its Host header), so a box answers
with its own address however the client reaches it. A text without the token is served as stored.

SRV-163 (UT1, 2026-10-06). The box's saved packages had a literal LAN address (`https://ai-api.example.lan`) where
the token belongs: on the Spark, whose trust store lacks the private root CA, every Laya call failed
TLS and the model spent up to 42 % of a session debugging certificates. Now: `QWEN38_TANDEM_URL`
sets the address the token becomes (empty: the request's own address, as before; an opencode on the
box that talks to http://127.0.0.1:8000 gets that), and a package whose text calls TandemLLM's own
routes at a literal address is refused at save (`literal_api_hosts`), so the token is the only way.

SRV-171 (the same self-test). `{tandem_scratch}` is where the testing packages put the logs, the Laya
helper and the playtest kit: `.tandem` in the project by default, which the packages have the model
add to `.git/info/exclude` and delete before it reports done, so the deliverable never holds it.
A directory outside the project was tried first and does not work under opencode: its sandbox treats
`/tmp/...` as an external directory, asks for permission, and `opencode run` rejects it (the FX1 live
check, 2026-10-06). `QWEN38_TANDEM_SCRATCH` names another place (`{token}` in it is the session
token, e.g. `/tmp/tandem-{token}`) for clients without such a sandbox.
"""

from __future__ import annotations

import os
import re
import threading

CHARS_PER_TOKEN = 3.5
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def shipped_root(root: str = REPO) -> str:
    """The directory whose `packages/` holds the shipped texts: the checkout's root, else the
    installed wheel's `proxy/` (pyproject.toml ships `packages/` as `proxy/packages`)."""
    if os.path.isfile(os.path.join(root, "packages", "seed.json")):
        return root
    wheel = os.path.join(root, "proxy")
    if os.path.isfile(os.path.join(wheel, "packages", "seed.json")):
        return wheel
    return root


SHIPPED_ROOT = shipped_root()
# `{repo}` in a `skill_dirs` entry: the directory whose `packages/` holds the shipped texts
REPO_TOKEN = "{repo}"
URL_TOKEN = "{tandem_url}"
# SRV-615: the results a package can name as its `deliverable`, the kinds signals.tool_effects
# reads off a tool's declaration (`post` has no schema rule; the keep-out question decides it)
DELIVERABLES = ("send", "schedule", "post")
# SRV-171: the session's scratch directory, outside the user's project
SCRATCH_TOKEN = "{tandem_scratch}"
SCRATCH_DEFAULT = ".tandem"
#: FX15: the default, one folder per conversation under `.tandem`
SCRATCH_SESSION = ".tandem/{token}"
SCRATCH_ENV = "QWEN38_TANDEM_SCRATCH"
# where a request without a Host header (a test, a tool on the box) reaches the server
URL_DEFAULT = "http://127.0.0.1:8000"
# SRV-163: the address `{tandem_url}` becomes, when the operator sets one (else the request's own)
URL_ENV = "QWEN38_TANDEM_URL"
# SRV-163: TandemLLM's own routes, which a package names through `{tandem_url}` only
_OWN_ROUTE = re.compile(r"\bhttps?://[^\s\"'`<>(){}]+?"
                        r"(?=\}?[\"']?\s*\)?\s*\+?\s*[\"']?/(?:s/[A-Za-z0-9_-]+/)?v1/(?:laya|library|memory)\b)")
# what ends a text cut to the package's max_tokens (the dashboard's catalog looks for it)
CUT_NOTE = "\n[... cut to the package's max_tokens]"
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_REFERENCE = re.compile(r"(?m)^## Reference\b")


def body_packages(body: dict, known) -> tuple[list[str], set[str], list[str]]:
    """SRV-361: a client's own choice of packages for this request,
    `tandem: {"packages": {"pin": [...], "block": [...]}}` (chatd sends a chat's pinned and
    blocked skills). Returns (pinned and known, in order; blocked; pinned but unknown). A blocked
    package never goes in; a pinned one goes in whatever the class and the package question say.
    Without the field: nothing pinned, nothing blocked."""
    t = body.get("tandem") if isinstance(body, dict) else None
    p = t.get("packages") if isinstance(t, dict) else None
    if not isinstance(p, dict):
        return [], set(), []
    names = lambda v: [x for x in v if isinstance(x, str) and x][:32] if isinstance(v, list) \
        else []                                                   # noqa: E731
    block = set(names(p.get("block")))
    pin, unknown = [], []
    for n in names(p.get("pin")):
        if n in block or n in pin:
            continue
        (pin if n in known else unknown).append(n)
    return pin, block, unknown


def split_frontmatter(text: str) -> tuple[dict, str]:
    """(`{name, description}` from the frontmatter, the body after it)."""
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() in ("name", "description"):
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


def request_url(headers) -> str | None:
    """The address a request reached this server at: the proxy's forwarded host and scheme when it
    names them, else the Host header; None without either."""
    def get(name):
        try:
            return (headers.get(name) or "").split(",")[0].strip()
        except Exception:                                          # noqa: BLE001
            return ""
    from engine.settings import SETTINGS as _S
    fixed = _S.get("TANDEM_URL").strip().rstrip("/")
    if fixed:
        return fixed                       # SRV-163: the operator's address for the notes
    host = get("X-Forwarded-Host") or get("Host")
    if not host or any(c in host for c in " /\\\"'<>"):
        return None
    return f"{get('X-Forwarded-Proto') or 'http'}://{host}"


def scratch_for(url: str | None) -> str:
    """SRV-171: the scratch directory the notes name: `QWEN38_TANDEM_SCRATCH` (its `{token}` the
    session token from the address, `.../s/<token>`), else `.tandem/<token>` in the project
    (FX15: the conversation's own folder, so a `.tandem` the user or another conversation left is
    never the one the testing package creates, marks and deletes; `.tandem/session` without a
    token)."""
    from engine.settings import SETTINGS as _S
    want = _S.get("TANDEM_SCRATCH").strip() or SCRATCH_SESSION
    if "{token}" in want:
        m = re.search(r"/s/([A-Za-z0-9_-]{8,32})/?$", url or "")
        want = want.replace("{token}", m.group(1) if m else "session")
    return want.rstrip("/") or SCRATCH_DEFAULT


def with_url(text: str, url: str | None) -> str:
    """`text` with `{tandem_url}` replaced by `url` (URL_DEFAULT when None) and
    `{tandem_scratch}` by the session's scratch directory."""
    if URL_TOKEN in text:
        text = text.replace(URL_TOKEN, url or URL_DEFAULT)
    if SCRATCH_TOKEN in text:
        text = text.replace(SCRATCH_TOKEN, scratch_for(url))
    return text


def literal_api_hosts(text: str) -> list[str]:
    """SRV-163: the literal addresses in `text` that call TandemLLM's own routes (`/v1/laya`,
    `/v1/library`, `/v1/memory`, with or without a session token) instead of `{tandem_url}`."""
    return sorted({m.group(0) for m in _OWN_ROUTE.finditer(text or "")})


def split_reference(body: str) -> tuple[str, str]:
    """(the text before the `## Reference` heading, the reference from it on); the whole text and
    "" when there is none."""
    m = _REFERENCE.search(body)
    return (body[:m.start()], body[m.start():]) if m else (body, "")


class Library:
    """Resolves package names to text. One per server; reads files lazily."""

    def __init__(self):
        self._files: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._req = threading.local()          # the request's address (`at`), per thread

    def at(self, url: str | None) -> None:
        """The address of the request this thread plans now: `{tandem_url}` in a text."""
        self._req.url = url

    def _dirs(self, doc: dict) -> list[str]:
        return [os.path.expanduser(d.replace(REPO_TOKEN, SHIPPED_ROOT, 1) if d.startswith(REPO_TOKEN)
                                   else d)
                for d in doc.get("skill_dirs") or []]

    def path(self, doc: dict, name: str) -> str | None:
        p = (doc.get("packages") or {}).get(name)
        if p is None:
            return None
        src = p.get("source", "inline")
        if src.startswith("file:"):
            return os.path.expanduser(src[5:])
        if src.startswith("skill:"):
            skill = src[6:]
            for d in self._dirs(doc):
                cand = os.path.join(d, skill, "SKILL.md")
                if os.path.isfile(cand):
                    return cand
            return os.path.join(self._dirs(doc)[0], skill, "SKILL.md") if self._dirs(doc) else None
        return None

    def file_text(self, path: str) -> str | None:
        """A file's text, read again when its mtime moves (the library's index reads with it)."""
        return self._read(path)

    def _read(self, path: str) -> str | None:
        try:
            mt = os.stat(path).st_mtime
        except OSError:
            return None
        with self._lock:
            got = self._files.get(path)
            if got is not None and got[0] == mt:
                return got[1]
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                return None
            self._files[path] = (mt, text)
            return text

    def _body(self, doc: dict, name: str) -> tuple[str | None, str | None]:
        """(the package's text as stored, before `{tandem_url}` is filled in, None) or (None, why
        it is unavailable)."""
        p = (doc.get("packages") or {}).get(name)
        if p is None:
            return None, "package_missing"
        if not p.get("enabled", True):
            return None, "package_disabled"
        if p.get("source", "inline") == "inline":
            body = p.get("body") or ""
        else:
            path = self.path(doc, name)
            raw = self._read(path) if path else None
            if raw is None:
                return None, "package_file_missing"
            body = split_frontmatter(raw)[1]
        if p.get("library"):
            body = split_reference(body)[0]
        return body, None

    def requires(self, doc: dict, name: str) -> tuple[str, ...]:
        """SRV-243: the tool capabilities the package's steps need (signals.CAPS): its own
        `requires_tools` when it names them, else a shell when its text calls TandemLLM's own
        routes (`{tandem_url}`: Laya, the library, the memory are reached with curl from a shell,
        and a client's fetch tool refuses the address), else none."""
        p = (doc.get("packages") or {}).get(name) or {}
        want = p.get("requires_tools")
        if isinstance(want, list):
            return tuple(str(w) for w in want)
        body, _ = self._body(doc, name)
        return ("shell",) if body and URL_TOKEN in body else ()

    @staticmethod
    def deliverable(doc: dict, name: str) -> str | None:
        """SRV-615: what the package's result is for when a tool can make it (`send`, `schedule`,
        `post`), or None: a write-message draft is a message to send."""
        d = ((doc.get("packages") or {}).get(name) or {}).get("deliverable")
        return d if isinstance(d, str) and d in DELIVERABLES else None

    @staticmethod
    def fallback(doc: dict, name: str) -> str | None:
        """SRV-243: the package that stands in for `name` when the turn lacks its tools."""
        fb = ((doc.get("packages") or {}).get(name) or {}).get("fallback")
        return fb if isinstance(fb, str) and fb else None

    def text(self, doc: dict, name: str) -> tuple[str | None, str | None]:
        """(the package's instruction text, None) or (None, why it is unavailable)."""
        p = (doc.get("packages") or {}).get(name)
        body, err = self._body(doc, name)
        if body is None:
            return None, err
        body = with_url(body, getattr(self._req, "url", None)).strip()
        if not body:
            return None, "package_empty"
        cap = int(p.get("max_tokens") or 4000) * CHARS_PER_TOKEN
        if len(body) > cap:
            cut = body[:int(cap)]
            body = cut[:cut.rfind("\n")] if "\n" in cut else cut
            body += CUT_NOTE
        return body, None

    def status(self, doc: dict) -> dict:
        """Per package: where its text comes from, whether it is there, its size."""
        out = {}
        for name, p in (doc.get("packages") or {}).items():
            text, err = self.text(doc, name)
            out[name] = {"available": text is not None, "error": err,
                         "path": self.path(doc, name), "chars": len(text or ""),
                         "tokens_est": int(len(text or "") / CHARS_PER_TOKEN)}
        return out

    def scan(self, doc: dict) -> list[dict]:
        """Skills found in `skill_dirs`, for the dashboard's import: name, description, path."""
        seen, out = set(), []
        for d in self._dirs(doc):
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for n in names:
                path = os.path.join(d, n, "SKILL.md")
                if n in seen or not os.path.isfile(path):
                    continue
                raw = self._read(path) or ""
                meta, body = split_frontmatter(raw)
                seen.add(n)
                out.append({"name": n, "description": meta.get("description", "")[:600],
                            "path": path, "chars": len(body),
                            "registered": n in (doc.get("packages") or {})})
        return out
