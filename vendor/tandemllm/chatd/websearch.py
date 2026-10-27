"""Web search for the chat's model: the `web_search` tool's providers and settings (SRV-469).

PROVIDERS. `duckduckgo`: POST https://html.duckduckgo.com/html/ (`q`, region `kl`, safe search
`kp`, time `df`), parsed with the standard library's HTML parser; ads are left out, redirect links
(`//duckduckgo.com/l/?uddg=`) decoded, URLs de-duplicated. An HTTP 202 or the `anomaly` bot check
is the rate limit: a tool error, no retry. `searxng`: GET <base>/search?format=json on an instance
the user configured, with an optional auth header; 403 means the instance has the JSON format
off (`search.formats` in its settings.yml). Both run in chatd through `egress.request` with a host
list: DuckDuckGo's host on public addresses only, the SearXNG instance's host and port (a private
address allowed for that host, since the user named it). A redirect elsewhere is refused.

SETTINGS. Global, in chatd's `meta` table under `web_search` (JSON): provider, the SearXNG URL and
auth header, max_results, safe_search, language, time_range, timeout_s. A chat's `web_search`
setting (null: the global ones) overrides provider, max_results, safe_search, language and
time_range. The auth header's value is stored sealed with the secrets vault the MCP headers use
(proxy/secrets.py, AES-256-GCM); answers carry `searxng_auth_set`, logs carry counts.

RESULTS. For the model: a numbered list (title, URL, snippet, date and engine when given). For the
UI: `output.data` = {provider, query, results: [{title, url, snippet, date?, engine?}]}.
"""

from __future__ import annotations

import html.parser
import json
import re
import time
import urllib.parse

from chatd import egress

PROVIDERS = ("off", "duckduckgo", "searxng")
PROVIDER_WORDS = {"off": "off", "duckduckgo": "DuckDuckGo", "searxng": "SearXNG"}
SAFE = ("off", "moderate", "strict")
TIMES = ("", "day", "week", "month", "year")
DEFAULTS = {"provider": "duckduckgo", "searxng_url": "", "searxng_auth_header": "",
            "searxng_auth_value": "", "max_results": 8, "safe_search": "moderate",
            "language": "", "time_range": "", "timeout_s": 10}
CHAT_KEYS = ("provider", "max_results", "safe_search", "language", "time_range")
MAX_RESULTS = 20
MAX_QUERY = 500
SNIPPET_CHARS = 300
TITLE_CHARS = 200
TEST_QUERY = "open source large language model"
SECRET_ID = "web_search:searxng"     # the associated data of the sealed auth value

DDG_URL = "https://html.duckduckgo.com/html/"
DDG_HOSTS = frozenset({"html.duckduckgo.com"})
# a browser's; the HTML endpoint resets connections from a bare library User-Agent
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
_DDG_SAFE = {"strict": "1", "moderate": "-1", "off": "-2"}
_DDG_TIME = {"day": "d", "week": "w", "month": "m", "year": "y"}
_DDG_COUNTRY = {"gb": "uk"}
_SX_SAFE = {"off": "0", "moderate": "1", "strict": "2"}

_LANG = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_HEADER = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_BAD_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "te", "upgrade",
                "cookie"}
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class SearchError(Exception):
    """A search that did not run as asked: the message is the tool result."""


class Invalid(ValueError):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


# ---------------------------------------------------------------------- settings
def _int(v, field: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise Invalid(field, f"{field} must be a number from {lo} to {hi}")
    try:
        n = int(v)
    except ValueError:
        raise Invalid(field, f"{field} must be a number from {lo} to {hi}") from None
    if n < lo or n > hi:
        raise Invalid(field, f"{field} must be a number from {lo} to {hi}")
    return n


def _choice(v, field: str, allowed: tuple) -> str:
    if v is None:
        v = ""
    if v not in allowed:
        shown = ", ".join(a or '"" (any)' for a in allowed)
        raise Invalid(field, f"{field} must be one of {shown}")
    return v


def _language(v) -> str:
    v = "" if v is None else str(v).strip()
    if v and not _LANG.match(v):
        raise Invalid("language", 'language must be "" (any) or a code such as de or de-DE')
    return v


def base_url(v) -> str:
    v = "" if v is None else str(v).strip()
    if not v:
        return ""
    u = urllib.parse.urlsplit(v)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise Invalid("searxng_url", "searxng_url must be an http or https URL")
    if u.username or u.password:
        raise Invalid("searxng_url", "searxng_url must not carry a user or password; use the "
                                     "auth header")
    if u.query or u.fragment or len(v) > 512:
        raise Invalid("searxng_url", "searxng_url is the instance's base URL, without a query")
    try:
        u.port
    except ValueError:
        raise Invalid("searxng_url", "searxng_url has a bad port") from None
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.rstrip("/"), "", ""))


def clean(patch: dict, current: dict) -> dict:
    """The global settings after a PUT: `current` with the patch's known keys, every field
    checked. `searxng_auth_value`: a string sets it, null clears it, absent keeps it."""
    if not isinstance(patch, dict):
        raise Invalid("settings", "the web search settings must be an object")
    out = {**DEFAULTS, **current}
    for k, v in patch.items():
        if k not in DEFAULTS:
            continue
        out[k] = "" if v is None and k in ("searxng_auth_value", "searxng_auth_header",
                                           "searxng_url", "language", "time_range") else v
    out["provider"] = _choice(out["provider"], "provider", PROVIDERS)
    out["max_results"] = _int(out["max_results"], "max_results", 1, MAX_RESULTS)
    out["timeout_s"] = _int(out["timeout_s"], "timeout_s", 2, 30)
    out["safe_search"] = _choice(out["safe_search"], "safe_search", SAFE)
    out["time_range"] = _choice(out["time_range"], "time_range", TIMES)
    out["language"] = _language(out["language"])
    out["searxng_url"] = base_url(out["searxng_url"])
    name = str(out["searxng_auth_header"] or "").strip()
    value = out["searxng_auth_value"]
    if not isinstance(value, str) or "\r" in value or "\n" in value or len(value) > 4096:
        raise Invalid("searxng_auth_value", "searxng_auth_value must be one line of text")
    if name and (not _HEADER.match(name) or name.lower() in _BAD_HEADERS):
        raise Invalid("searxng_auth_header", "searxng_auth_header must be a header name such as "
                                             "Authorization or X-API-Key")
    if value and not name:
        raise Invalid("searxng_auth_header", "name the header the auth value goes in")
    out["searxng_auth_header"] = name
    if out["provider"] == "searxng" and not out["searxng_url"]:
        raise Invalid("searxng_url", "SearXNG needs the instance's URL (searxng_url)")
    return out


def clean_override(v) -> dict | None:
    """A chat's `web_search` setting: null, or some of CHAT_KEYS (unknown keys are refused)."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise Invalid("web_search", "web_search must be null or an object")
    extra = sorted(set(v) - set(CHAT_KEYS))
    if extra:
        raise Invalid("web_search", f"a chat sets only {', '.join(CHAT_KEYS)} for web search "
                                    f"(not {', '.join(extra)})")
    out = {}
    if "provider" in v:
        out["provider"] = _choice(v["provider"], "provider", PROVIDERS)
    if "max_results" in v:
        out["max_results"] = _int(v["max_results"], "max_results", 1, MAX_RESULTS)
    if "safe_search" in v:
        out["safe_search"] = _choice(v["safe_search"], "safe_search", SAFE)
    if "language" in v:
        out["language"] = _language(v["language"])
    if "time_range" in v:
        out["time_range"] = _choice(v["time_range"], "time_range", TIMES)
    return out or None


def effective(glob: dict, override: dict | None) -> dict:
    """A chat's settings: the global ones with its override; SearXNG without a URL is off."""
    out = {**DEFAULTS, **glob, **(override or {})}
    if out["provider"] == "searxng" and not out.get("searxng_url"):
        out["provider"] = "off"
    return out


def public(s: dict) -> dict:
    """The settings as an answer: the auth value replaced by whether one is stored."""
    out = {k: v for k, v in {**DEFAULTS, **s}.items() if k != "searxng_auth_value"}
    out["searxng_auth_set"] = bool(s.get("searxng_auth_value"))
    return out


# ---------------------------------------------------------------------- DuckDuckGo
class _DdgParser(html.parser.HTMLParser):
    """The results of html.duckduckgo.com: a `div.result` each (`result--ad` for an ad), its link
    `a.result__a`, its snippet `.result__snippet`, a date in a `span` of `.result__extras__url`."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self.cur: dict | None = None
        self.field: str | None = None
        self.depth = 0                  # tags open inside the field being read
        self.in_extras = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div" and "result" in cls and ("results_links" in cls or "web-result" in cls
                                                 or "result--ad" in cls):
            self.cur = {"title": "", "url": "", "snippet": "", "date": "",
                        "ad": "result--ad" in cls}
            self.results.append(self.cur)
            self.field, self.in_extras = None, False
            return
        if self.cur is None:
            return
        if self.field is not None:
            self.depth += 1
            return
        if tag == "a" and "result__a" in cls:
            self.cur["url"] = a.get("href") or ""
            self.field, self.depth = "title", 0
        elif "result__snippet" in cls:
            self.field, self.depth = "snippet", 0
            if not self.cur["url"] and a.get("href"):
                self.cur["url"] = a["href"]
        elif tag == "div" and "result__extras__url" in cls:
            self.in_extras = True
        elif tag == "span" and self.in_extras and "result__icon" not in cls:
            self.field, self.depth = "date", 0

    def handle_endtag(self, tag):
        if self.field is None:
            if tag == "div" and self.in_extras:
                self.in_extras = False
            return
        if self.depth:
            self.depth -= 1
        else:
            self.field = None

    def handle_data(self, data):
        if self.cur is not None and self.field:
            self.cur[self.field] += data


def ddg_target(href: str) -> str:
    """The result's own URL: a `/l/?uddg=` redirect decoded, a scheme-relative link made https."""
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    u = urllib.parse.urlsplit(href)
    if (u.hostname or "").endswith("duckduckgo.com") or href.startswith("/l/"):
        q = urllib.parse.parse_qs(u.query)
        if q.get("uddg"):
            return q["uddg"][0]
        return ""
    return href


def ddg_region(language: str) -> str:
    """`de-DE` -> `de-de`, `en-GB` -> `uk-en` (DuckDuckGo's country-language); none: `wt-wt`."""
    if not language or "-" not in language:
        return "wt-wt"
    lang, country = language.split("-", 1)
    c = country.lower()
    return f"{_DDG_COUNTRY.get(c, c)}-{lang.lower()}"


def parse_ddg(page: str) -> list[dict]:
    p = _DdgParser()
    try:
        p.feed(page)
        p.close()
    except Exception:                                               # noqa: BLE001
        pass
    out = []
    for r in p.results:
        if r["ad"]:
            continue
        url = ddg_target(r["url"])
        if not url.startswith(("http://", "https://")):
            continue
        d = _DATE.search(r["date"])
        out.append(_result(r["title"], url, r["snippet"], d.group(1) if d else None, None))
    return out


def ddg_refused(status: int, page: str) -> bool:
    return status == 202 or "anomaly-modal" in page or "anomaly.js" in page


def search_ddg(cfg: dict, query: str, n: int, http=egress.request) -> list[dict]:
    form = {"q": query, "kl": ddg_region(cfg.get("language") or ""),
            "kp": _DDG_SAFE[cfg.get("safe_search") or "moderate"]}
    if cfg.get("time_range"):
        form["df"] = _DDG_TIME[cfg["time_range"]]
    try:
        st, _, body = http(DDG_URL, method="POST", body=urllib.parse.urlencode(form).encode(),
                           headers={"User-Agent": UA, "Accept": "text/html",
                                    "Content-Type": "application/x-www-form-urlencoded",
                                    "Referer": "https://html.duckduckgo.com/"},
                           hosts=DDG_HOSTS, timeout=float(cfg.get("timeout_s") or 10))
    except egress.Refused as exc:
        raise SearchError(f"cannot reach DuckDuckGo: {exc}") from None
    page = body.decode("utf-8", "replace")
    if ddg_refused(st, page):
        raise SearchError("DuckDuckGo refused the search: its rate limit or bot check (HTTP "
                          f"{st}). Wait a minute and search again, or set the provider to "
                          "SearXNG in the chat settings.")
    if st != 200:
        raise SearchError(f"DuckDuckGo answered HTTP {st}")
    return parse_ddg(page)[:n]


# ---------------------------------------------------------------------- SearXNG
def searxng_hosts(base: str) -> frozenset:
    u = urllib.parse.urlsplit(base)
    host = (u.hostname or "").lower()
    return frozenset({f"{host}:{u.port or (443 if u.scheme == 'https' else 80)}"})


def search_searxng(cfg: dict, query: str, n: int, http=egress.request) -> list[dict]:
    base = cfg.get("searxng_url") or ""
    if not base:
        raise SearchError("SearXNG has no instance URL; set it in the chat settings")
    params = {"q": query, "format": "json", "categories": "general", "pageno": "1",
              "language": cfg.get("language") or "all",
              "safesearch": _SX_SAFE[cfg.get("safe_search") or "moderate"]}
    if cfg.get("time_range"):
        params["time_range"] = cfg["time_range"]
    headers = {"User-Agent": "TandemLLM-chat/0.1 (web_search)", "Accept": "application/json"}
    if cfg.get("searxng_auth_header") and cfg.get("searxng_auth_value"):
        headers[cfg["searxng_auth_header"]] = cfg["searxng_auth_value"]
    u = urllib.parse.urlsplit(base)
    where = u.netloc
    timeout = float(cfg.get("timeout_s") or 10)
    try:
        st, ctype, body = http(base + "/search?" + urllib.parse.urlencode(params),
                               headers=headers, hosts=searxng_hosts(base), private_ok=True,
                               timeout=timeout)
    except egress.Refused as exc:
        raise SearchError(f"cannot reach the SearXNG instance at {where}: {exc}") from None
    if st == 403:
        raise SearchError(f"the SearXNG instance at {where} refuses format=json (HTTP 403): add "
                          "json to search.formats in its settings.yml")
    if st in (401, 407):
        raise SearchError(f"the SearXNG instance at {where} wants credentials (HTTP {st}): check "
                          "the auth header in the chat settings")
    if st == 429:
        raise SearchError(f"the SearXNG instance at {where} limits this client (HTTP 429): its "
                          "limiter is on; allow this host or turn the limiter off")
    if st != 200:
        raise SearchError(f"the SearXNG instance at {where} answered HTTP {st}")
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        shown = ctype or "no content type"
        raise SearchError(f"the SearXNG instance at {where} did not answer JSON ({shown})") \
            from None
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        raise SearchError(f"the SearXNG instance at {where} answered JSON without results")
    out = []
    for r in obj["results"]:
        if not isinstance(r, dict) or not str(r.get("url") or "").startswith(("http://",
                                                                              "https://")):
            continue
        engine = r.get("engine") or (r.get("engines") or [None])[0]
        d = _DATE.search(str(r.get("publishedDate") or r.get("pubdate") or ""))
        out.append(_result(r.get("title"), r["url"], r.get("content"),
                           d.group(1) if d else None, engine))
    return out[:n] if out else _unresponsive(obj)


def _unresponsive(obj: dict) -> list[dict]:
    gone = [e[0] if isinstance(e, list) and e else str(e)
            for e in obj.get("unresponsive_engines") or []]
    if gone:
        raise SearchError("no results; the SearXNG instance's engines did not answer: " +
                          ", ".join(str(g) for g in gone[:6]))
    return []


# ---------------------------------------------------------------------- the search
def _clean_text(s, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def _result(title, url, snippet, date, engine) -> dict:
    r = {"title": _clean_text(title, TITLE_CHARS) or url, "url": str(url).strip(),
         "snippet": _clean_text(snippet, SNIPPET_CHARS)}
    if date:
        r["date"] = date
    if engine:
        r["engine"] = str(engine)[:40]
    return r


def _key(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    return f"{(u.hostname or '').lower().removeprefix('www.')}{u.path.rstrip('/')}?{u.query}"


def dedupe(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in results:
        k = _key(r["url"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def search(cfg: dict, query: str, max_results=None, http=egress.request) -> dict:
    """{provider, query, results, ms} or SearchError."""
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        raise SearchError("query is empty")
    if len(query) > MAX_QUERY:
        raise SearchError(f"query is longer than {MAX_QUERY} characters; search for less")
    n = int(cfg.get("max_results") or DEFAULTS["max_results"])
    if max_results not in (None, ""):
        try:
            n = max(1, min(MAX_RESULTS, int(max_results)))
        except (TypeError, ValueError):
            raise SearchError("max_results must be a number") from None
    provider = cfg.get("provider")
    t0 = time.monotonic()
    if provider == "duckduckgo":
        res = search_ddg(cfg, query, MAX_RESULTS, http)
    elif provider == "searxng":
        res = search_searxng(cfg, query, MAX_RESULTS, http)
    else:
        raise SearchError("web search is off; turn it on in the chat settings")
    return {"provider": provider, "query": query, "results": dedupe(res)[:n],
            "ms": int((time.monotonic() - t0) * 1000)}


def as_text(res: dict) -> str:
    """The tool result the model reads."""
    word = PROVIDER_WORDS.get(res["provider"], res["provider"])
    rs = res["results"]
    if not rs:
        return f'Web search for "{res["query"]}" ({word}): no results.'
    lines = [f'Web search for "{res["query"]}" ({word}, {len(rs)} result'
             f'{"s" if len(rs) != 1 else ""}):', ""]
    for i, r in enumerate(rs, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        meta = " · ".join(x for x in (r.get("date"), r.get("engine")) if x)
        if meta:
            lines.append(f"   {meta}")
    lines += ["", "Cite the URLs you use. Read a page in full with web_fetch."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- the service's side
class WebSearchMixin:
    """Mixed into `chatd.service.Service`: the stored settings, the routes, a chat's search."""

    def web_search_global(self) -> dict:
        """The stored settings, the auth value decrypted into memory (an unreadable one: none)."""
        r = self.store.one("SELECT value FROM meta WHERE key = 'web_search'")
        try:
            got = json.loads(r["value"]) if r else {}
        except ValueError:
            got = {}
        got = got if isinstance(got, dict) else {}
        out = {**DEFAULTS, **{k: v for k, v in got.items() if k in DEFAULTS}}
        sealed = got.get("searxng_auth_sealed")
        if sealed:
            try:
                out["searxng_auth_value"] = self.mcp._open(SECRET_ID, sealed).get("value") or ""
            except Exception:                                       # noqa: BLE001
                out["searxng_auth_value"] = ""
        return out

    def web_search_settings(self, settings: dict) -> dict:
        """A chat's effective settings (its `web_search` over the global ones)."""
        return effective(self.web_search_global(), settings.get("web_search"))

    def web_search_view(self) -> dict:
        """GET /web-search."""
        return {"settings": public(self.web_search_global()), "providers": list(PROVIDERS),
                "safe_search": list(SAFE), "time_ranges": list(TIMES),
                "max_results_limit": MAX_RESULTS}

    def set_web_search(self, body: dict) -> dict:
        """PUT /web-search: a partial update, checked as a whole."""
        from chatd.service import ApiError
        try:
            new = clean(body, self.web_search_global())
        except Invalid as exc:
            raise ApiError(400, "bad_request", str(exc), {"field": exc.field}) from None
        value = new.pop("searxng_auth_value")
        if value:
            try:
                new["searxng_auth_sealed"] = self.mcp._seal(SECRET_ID, {"value": value}).decode()
            except Exception as exc:                                # noqa: BLE001
                raise ApiError(400, "bad_request", str(exc), {"field": "searxng_auth_value"}) \
                    from None
        self.store.run("INSERT OR REPLACE INTO meta VALUES ('web_search', ?)", (json.dumps(new),))
        new["searxng_auth_value"] = value
        self.emit_global({"type": "web_search.changed", "provider": new["provider"]})
        self.log(f"[search] settings saved: provider {new['provider']}")
        return self.web_search_view()

    def test_web_search(self, body: dict) -> dict:
        """POST /web-search/test {settings?, query?}: one search with the draft over the stored
        settings; never stored."""
        from chatd.service import ApiError
        body = body if isinstance(body, dict) else {}
        try:
            cfg = clean(body.get("settings") or {}, self.web_search_global())
        except Invalid as exc:
            raise ApiError(400, "bad_request", str(exc), {"field": exc.field}) from None
        word = PROVIDER_WORDS[cfg["provider"]]
        if cfg["provider"] == "off":
            return {"ok": False, "provider": "off", "message": "web search is off", "count": 0,
                    "ms": 0, "results": []}
        try:
            res = search(cfg, str(body.get("query") or TEST_QUERY), min(3, cfg["max_results"]),
                         http=self._search_http)
        except SearchError as exc:
            return {"ok": False, "provider": cfg["provider"], "message": str(exc), "count": 0,
                    "ms": 0, "results": []}
        n = len(res["results"])
        self.log(f"[search] test {cfg['provider']}: {n} results in {res['ms']} ms")
        return {"ok": n > 0, "provider": cfg["provider"], "count": n, "ms": res["ms"],
                "message": (f"{word} answered {n} result{'s' if n != 1 else ''} in "
                            f"{res['ms']:,} ms" if n else f"{word} answered, with no results"),
                "results": res["results"]}

    def run_web_search(self, cid: str, query, max_results=None) -> dict:
        """The tool's search with the chat's effective settings."""
        from chatd.service import DEFAULT_SETTINGS
        row = self._row(cid)
        settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
        cfg = self.web_search_settings(settings)
        res = search(cfg, query, max_results, http=self._search_http)
        self.log(f"[search] {cfg['provider']}: {len(res['results'])} results in {res['ms']} ms")
        return res

    # the transport; tests put a recorder here
    _search_http = staticmethod(egress.request)
