"""The Packages and Skills screen's reads (G2, 2026-10-03): one list of every package with what it
is, the text that goes in and its size on the engine's tokenizer, one package's texts, and what
the journal says each one was used for.

The owner, 2026-10-03: "build the needed GUI for packages/skills in the dashboard". Everything here
reads: the `packages`, `router` and profile documents, the files on the box and the journal.
Nothing is on a request's path and nothing changes what the router injects; the edits stay the
documents' own versioned PUTs (server/conductor/routes.py).

    GET  /v1/conductor/packages/catalog          every package: its category, where its text
                                                 comes from, the tokens of the text that goes in
                                                 (the engine's tokenizer; 3.5 characters a token
                                                 without one), its limit and whether it is over,
                                                 the classes that list it (the router's and each
                                                 profile's), the library's class fit (INDEX.md)
                                                 and each profile's library switch
    GET  /v1/conductor/packages/item/{name}      one package: the frontmatter, the body, the
                                                 reference, the text that goes in, their tokens,
                                                 and for a library skill its BM25 neighbours
    POST /v1/conductor/packages/tokens {texts}   the tokens of up to 20 texts (the editor's counter)
    GET  /v1/conductor/packages/usage?name=&limit=
                                                 the journal's last 7 days: per package its
                                                 injections, its shadow notes and its library picks
                                                 in the last day and week; one package's newest
                                                 rows (`name`); the library's newest retrieval rows

Categories: `library` (a library skill), `card` (the harness card, a client's pitfalls card, the
library card), `test_skill` (a package a low-quality or stuck outcome may retrieve, the router's
`outcomes.retrieve.packages`), `task_skill` (any other skill) and `package` (any other package).
The limit: a library skill 350 tokens and the library card 200 (the house rule
tools/library_install.py checks), every other package its `max_tokens` (where the server cuts
it). A package is over when its whole text, before the cut, has more tokens than its limit.
"""

from __future__ import annotations

import hashlib
import time

from server.conductor import schema
from server.conductor.packages import _FRONT, CHARS_PER_TOKEN, CUT_NOTE, split_reference

HOUSE = {"library": 350, "library_card": 200}       # tools/library_install.py BODY_CAP, CARD_CAP
MAX_TEXTS, MAX_CHARS = 20, 200_000
NEIGHBOURS = 6
DAY_MS = 86_400_000
WEEK_DAYS = 7
LIBRARY_ROWS = 50
HOWS = ("injected", "shadow", "pick")
_COUNTS: dict[str, int] = {}
_COUNTS_MAX = 4096


def count(state: dict, texts) -> tuple[list[int], str]:
    """(the tokens of each text, `engine` or `estimate`): the engine's tokenizer, each text
    counted once (by its hash); without one, 3.5 characters a token as the server's cut."""
    tok = state.get("tok")
    if tok is None:
        return [int(len(t) / CHARS_PER_TOKEN) for t in texts], "estimate"
    out = []
    for t in texts:
        key = f"{id(tok)}:{hashlib.sha256(t.encode()).hexdigest()[:24]}"
        n = _COUNTS.get(key)
        if n is None:
            n = len(tok(t, add_special_tokens=False).input_ids) if t else 0
            if len(_COUNTS) >= _COUNTS_MAX:
                _COUNTS.clear()
            _COUNTS[key] = n
        out.append(n)
    return out, "engine"


def categories(doc: dict, router: dict) -> dict[str, str]:
    """name -> library, card, test_skill, task_skill or package (the module's header)."""
    oc, lib = schema.outcomes(router), schema.library(router)
    cards = {oc["cards"]["harness"], *oc["cards"]["pitfalls"].values(), lib.get("card")}
    tests = set(oc["retrieve"]["packages"])
    return {n: ("library" if p.get("library") else "card" if n in cards else
                "test_skill" if n in tests else
                "task_skill" if p.get("kind") == "skill" else "package")
            for n, p in doc["packages"].items()}


def texts(library, doc: dict, name: str) -> dict:
    """One package's texts: `front` (a file's frontmatter lines), `body` (the text after it; a
    library skill's part before `## Reference`), `reference` (a library skill's), `injected`
    (what goes in, `Library.text` with the package switched on), `error` (why nothing would)."""
    p = doc["packages"][name]
    path = library.path(doc, name)
    front, raw = "", None
    if p.get("source", "inline") == "inline":
        raw = p.get("body") or ""
    elif path:
        raw = library.file_text(path)
        m = _FRONT.match(raw) if raw is not None else None
        if m:
            front, raw = m.group(1), raw[m.end():]
    body, ref = split_reference(raw or "") if p.get("library") else (raw or "", "")
    on = dict(doc, packages=dict(doc["packages"], **{name: dict(p, enabled=True)}))
    injected, err = library.text(on, name)
    return {"path": path, "found": raw is not None, "front": front, "body": body.strip(),
            "reference": ref.strip(), "injected": injected, "error": err}


def _limit(name: str, p: dict, cat: str, lib: dict) -> tuple[int, str]:
    house = HOUSE["library"] if cat == "library" else \
        HOUSE["library_card"] if name == lib.get("card") else None
    cap = int(p.get("max_tokens") or 4000)
    return (house, "house") if house is not None and house <= cap else (cap, "max_tokens")


def catalog(cond) -> dict:
    st = cond.store
    v, doc = st.peek("packages")
    rv, router = st.peek("router")
    lib = schema.library(router)
    cats = categories(doc, router)
    fits: dict = {}
    if any(p.get("library") for p in doc["packages"].values()):
        idx = cond.router.index
        idx.ensure(v, doc, lib)
        fits = {e.name: e.classes for e in idx.entries}
    profiles = {n: st.peek("profiles", n)[1] for n in st.names()}
    tx = {n: texts(cond.library, doc, n) for n in doc["packages"]}
    flat = [s for t in tx.values() for s in (t["injected"] or "", t["body"], t["reference"])]
    nums, how = count(cond.state, flat)
    out = {}
    for i, (n, p) in enumerate(doc["packages"].items()):
        t, cat = tx[n], cats[n]
        k_inj, k_body, k_ref = nums[3 * i:3 * i + 3]
        limit, source = _limit(n, p, cat, lib)
        inj = t["injected"] or ""
        cut = inj.endswith(CUT_NOTE)
        out[n] = {"category": cat, "source": p["source"], "path": t["path"],
                  "available": t["injected"] is not None, "error": t["error"],
                  "tokens": k_inj, "body_tokens": k_body, "reference_tokens": k_ref,
                  "chars": len(inj), "cut": cut, "limit": limit, "limit_source": source,
                  "over": cut or k_body > limit,
                  "classes": [c["id"] for c in router["classes"]
                              if n in (c["policy"].get("packages") or ())],
                  "profiles": {pn: cl for pn, pd in profiles.items()
                               if (cl := [cid for cid, pol in (pd.get("routing") or {}).items()
                                          if n in (pol.get("packages") or ())])},
                  "fits": fits.get(n) if p.get("library") else None}
    return {"version": v, "router_version": rv, "tokenizer": how, "limits": dict(HOUSE),
            "library": {"card": lib.get("card"),
                        "profiles": {pn: schema.profile_library(pd)
                                     for pn, pd in profiles.items()}},
            "packages": out}


def item(cond, name: str) -> dict | None:
    """One package's texts and tokens; a library skill's neighbours. None: no such package."""
    v, doc = cond.store.peek("packages")
    p = doc["packages"].get(name)
    if p is None:
        return None
    _, router = cond.store.peek("router")
    t = texts(cond.library, doc, name)
    (k_inj, k_body, k_ref), how = count(cond.state, [t["injected"] or "", t["body"],
                                                     t["reference"]])
    front = {}
    for line in t["front"].splitlines():
        k, sep, val = line.partition(":")
        if sep and k.strip() and not line[:1].isspace():
            front[k.strip()] = val.strip().strip('"').strip("'")
    neighbours = None
    if p.get("library"):
        idx = cond.router.index
        idx.ensure(v, doc, schema.library(router))
        query = " ".join((name.replace("-", " "), p.get("description") or "", t["body"][:2000]))
        neighbours = [{"name": e.name, "description": e.description, "score": round(s, 3)}
                      for e, s in idx.search(query, NEIGHBOURS, exclude={name},
                                             enabled_only=False)]
    return {"name": name, "category": categories(doc, router)[name], "path": t["path"],
            "found": t["found"], "frontmatter": front, "frontmatter_raw": t["front"],
            "body": t["body"], "reference": t["reference"], "injected": t["injected"],
            "error": t["error"], "tokenizer": how,
            "tokens": {"injected": k_inj, "body": k_body, "reference": k_ref},
            "neighbours": neighbours}


# ------------------------------------------------------------------------ the journal
def _card(name, row: dict, cards: dict) -> str | None:
    if name == "harness":
        return cards["harness"]
    if name == "tools":
        pit = cards["pitfalls"]
        return pit.get(row.get("client_kind") or "other") or pit.get("other")
    return None


def uses(row: dict, cards: dict):
    """(package, how) for each package one router row names: `injected` (a note part that went
    in: the class's package, a library skill, the library card, a card, an outcome's retrieved
    package), `shadow` (what would have gone in), `pick` (the library's retrieval chose it,
    shadow or on)."""
    extra = row.get("extra") or {}
    onote = (extra.get("outcome") or {}).get("note") or {}
    for note in row.get("injected") or ():
        for part in note.get("parts") or ():
            k, n = part.get("kind"), part.get("name")
            n = (n if k in ("package", "library", "library_card") else
                 _card(n, row, cards) if k == "card" else
                 onote.get("package") if k == "outcome" else None)
            if n:
                yield n, "injected"
    sh = row.get("shadow") or {}
    for part in sh.get("notes") or ():
        if part.get("kind") == "package" and part.get("name"):
            yield part["name"], "shadow"
    for c in sh.get("cards") or ():
        n = _card(c.get("name"), row, cards)
        if n:
            yield n, "shadow"
    if isinstance(sh.get("library_card"), dict) and sh["library_card"].get("name"):
        yield sh["library_card"]["name"], "shadow"
    if onote.get("package") and not onote.get("applied"):
        yield onote["package"], "shadow"
    for n in (extra.get("library") or {}).get("chosen") or ():
        yield n, "pick"


def usage(cond, name: str | None = None, limit: int = 20, now_ms: int | None = None) -> dict:
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    since, day = now - WEEK_DAYS * DAY_MS, now - DAY_MS
    _, router = cond.store.peek("router")
    cards = schema.outcomes(router)["cards"]
    rows = cond.journal.rows(since_ms=since, limit=100000)
    counts: dict[str, dict] = {}
    recent, lib_rows = [], []
    lib = {"day": 0, "week": 0}
    picked = {"day": 0, "week": 0}
    top: dict[str, int] = {}
    for r in rows:                                       # newest first
        if r["ts_ms"] > now:
            continue
        today = r["ts_ms"] >= day
        for n, how in uses(r, cards):
            c = counts.setdefault(n, {h: {"day": 0, "week": 0} for h in HOWS})
            c[how]["week"] += 1
            c[how]["day"] += int(today)
            if n == name and len(recent) < limit:
                recent.append({"ts_ms": r["ts_ms"], "how": how, "id": r["id"],
                               "request_id": r.get("request_id"),
                               "session_key": r.get("session_key"),
                               "class_id": r.get("class_id"), "profile": r.get("profile"),
                               "client_kind": r.get("client_kind")})
        li = (r.get("extra") or {}).get("library")
        if not isinstance(li, dict):
            continue
        chosen = list(li.get("chosen") or ())
        lib["week"] += 1
        lib["day"] += int(today)
        if chosen:
            picked["week"] += 1
            picked["day"] += int(today)
        for n in chosen:
            top[n] = top.get(n, 0) + 1
        if len(lib_rows) < LIBRARY_ROWS:
            lib_rows.append({"id": r["id"], "ts_ms": r["ts_ms"],
                             "session_key": r.get("session_key"), "class_id": r.get("class_id"),
                             "profile": r.get("profile"), "client_kind": r.get("client_kind"),
                             "mode": li.get("mode"), "trigger": li.get("trigger"),
                             "query_chars": li.get("query_chars"),
                             "candidates": [{"name": c.get("name"), "score": c.get("score")}
                                            for c in li.get("candidates") or ()],
                             "chosen": chosen, "how": li.get("how"),
                             "applied": li.get("applied"), "cached": bool(li.get("cached")),
                             "live": list(li.get("live") or ())})
    return {"since_ms": since, "now_ms": now, "rows": len(rows), "packages": counts,
            "name": name, "recent": recent,
            "library": {"searched": lib, "picked": picked,
                        "top": [{"name": n, "picks": k} for n, k in
                                sorted(top.items(), key=lambda x: (-x[1], x[0]))[:10]],
                        "rows": lib_rows}}
