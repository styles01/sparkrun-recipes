"""`/v1/conductor/*`: the dashboard's API over the conductor (SRV-119, SRV-129, VIS-42).

Reads follow the dashboard's rule (open with the login off, the admin token or session with it
on). Writes (PUT, restore, mode, label) need the admin bearer token or a dashboard session, or a
tool on the box itself: auth rule `conductor_write` (server/auth.py). Every write is a new
version with a note; nothing edits a version in place.

    GET  /v1/conductor/config                    loaded versions, load errors, journal, roles
    GET  /v1/conductor/meta                      enums, the task catalogue, the schemas
    GET  /v1/conductor/profiles                  tiles
    GET  /v1/conductor/{kind}[/{name}]           settings, router, roles, packages; profiles/{name}
    PUT  /v1/conductor/{kind}[/{name}]           {doc, note, base_version?}: a new version
    GET  .../versions   .../versions/{v}   .../diff?from=&to=
    POST .../restore {version, note?}            a new version equal to an old one
    POST /v1/conductor/profiles/{name}/mode      {mode, note?}: the three-way switch
    GET  /v1/conductor/roles                     the document, health, and what each role's
                                                 tokenizer hash allows it to share (SRV-133)
    POST /v1/conductor/roles/{role}/test         one question, its latency and answer
    GET  /v1/conductor/packages/scan             skills found in skill_dirs on the box
    GET  /v1/conductor/packages/catalog          every package: category, tokens on the engine's
                                                 tokenizer, limit, classes, library switches (G2,
                                                 server/conductor/catalog.py)
    GET  /v1/conductor/packages/item/{name}      one package's texts, tokens, BM25 neighbours
    POST /v1/conductor/packages/tokens {texts}   the tokens of up to 20 texts (a read)
    GET  /v1/conductor/packages/usage?name=&limit=   injections, shadow notes and library picks
                                                 per package in the last day and week (journal)
    POST /v1/conductor/classify                  {text, profile?, tools?}: the test box
    GET  /v1/conductor/decisions[/{id}]          the live feed (router rows, each with the
                                                 monitor's `events` of its request) and one row
    POST /v1/conductor/decisions/{id}/label      {label: right|wrong|clear, note?} (any row)
    GET  /v1/conductor/events?task=&mode=&acted=&session=&request=&since=&limit=
                                                 the session monitor's decisions (SRV-115)
    GET  /v1/conductor/events/{id}               one of them
    GET  /v1/conductor/stats?hours=24            the Overview's tiles and ribbon, `monitor` too
    GET  /v1/conductor/sessions[/{key}[/export?format=json|md]]
                                                 sessions, one timeline, the honest record:
                                                 every hidden text of router and monitor alike
    GET  /v1/conductor/evals                     classifier and suite reports on the box
    POST /v1/conductor/evals                     {report}: a suite report (tools/tandem_eval.py)
    GET  /v1/conductor/residency/plan            the residency planner (OPS-41): limits, what
                                                 fits now, the engine's split, the last refusal
    GET  /v1/conductor/watch                     the session tasks' state per session (S1, S2,
                                                 S4: gates, compaction, progress, idle client)
    GET  /v1/conductor/deciders/status           the decision models this engine serves (SRV-131):
                                                 states, limits, numbers, the planner's answer
                                                 for a load now, the model behind each question
    POST /v1/conductor/deciders/models/{name}/load|unload   through the residency planner (write)
    POST /v1/conductor/deciders/autoload         {service_start?, hold_restart?}: the on_start
                                                 models, one after the other (ops/start.sh; write)
    POST /v1/conductor/deciders/test             {state, questions, model?}: the test box
    GET|PUT /v1/conductor/deciders               the `deciders` document, like the others
    GET  /v1/conductor/narrator/status           the narrator's slot (P1): its model, state,
                                                 contract, calls and latency
    POST /v1/conductor/narrator/try              {events: [{kind, facts, extra?}], budget_ms?,
                                                 max_tokens?}: each event's template and the
                                                 narrator's line, in the idle window (write)
    POST /v1/conductor/narrator/load|unload      through the residency planner (write)
    GET  /v1/conductor/shadow/status             the shadow reader (SR1): its slot, the sessions'
                                                 states (read, behind, bytes), the router's
                                                 agreement, the last answers
    POST /v1/conductor/shadow/ask                {session, question, choices: {id: text},
                                                 budget_ms?}: a typed decision of the small model
                                                 over the session's whole context, in the idle
                                                 window (write)
    GET  /v1/conductor/memory/sessions           the session transcripts on the box (VIS-48):
                                                 one row per session, the writer's counters and
                                                 settings
    GET  /v1/conductor/memory/sessions/{key}?page=N   its blocks, 200 a page (default: the
                                                 newest page), to the last complete block
    GET  .../memory/sessions/{key}/export?format=md|json   the transcript itself
    PATCH .../memory/sessions/{key}/keep         {keep}: meta.json's keep, which the prune honours
    GET  /v1/conductor/memory/disk               the disk tier (C1): running or not, its settings
                                                 and their version, bytes, budget, free space,
                                                 hit rate, the most-hit entries (a preview for
                                                 opencode heads and packages only), its events
    POST /v1/conductor/memory/disk/clear         every entry and count gone (write)
    GET  /v1/conductor/memory/notes?user=        the memory notes (2026-10-05): the users, the
                                                 tree (scopes with counts), every live note's
                                                 frontmatter with its uses in 7 days, the
                                                 archived ones, the duplicates, the store's counts
    GET  .../memory/notes/search?user=&q=&k=     the runtime's search over every scope, with the
                                                 scores
    GET  .../memory/notes/injected?user=&limit=  the last requests' injected note ids (journal)
    GET  .../memory/notes/writes?limit=          the last writes and refusals (journal)
    GET  .../memory/notes/export?user=           a zip of the user's directory
    GET  .../memory/notes/{id}?user=             one note: frontmatter, body, changelog, the
                                                 revisions, its uses by day
    PUT  .../memory/notes/{id}?user=             {title, body, kind, tags, verify_by}: a person's
                                                 edit (a revision; ratified, writer human)
    PATCH .../memory/notes/{id}?user=            {status?, pinned?}: ratify, back to provisional,
                                                 pin a ratified rule
    POST .../memory/notes/{id}/archive?user=     {reason}
    DELETE .../memory/notes/{id}?user=           the note, its revisions and archive copy, at once
    DELETE .../memory/notes/users/{user}         {confirm: <user>}: the whole directory
    Every memory route needs the admin token, a dashboard session or the box itself: the
    transcripts are the user's own sessions (auth rule `conductor_write`, reads included).

`monitor` is a document like the others: the session monitor's policy (every task, rule
parameter, text and template; ops/supervisor.yaml is its version 0), which the profiles then
set per task (mode, thresholds, caps, the tool-gate packs).
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlsplit

from server.conductor import decisions as dec_mod
from server.conductor import schema, store as store_mod
from server.conductor.store import ConfigError

#: VIS-68: the speed-guard report a release tree ships (tools/speed_guard.py ship)
SHIPPED_GUARD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "release", "speed-guard.json")

CONTRACT = "1.0"
DOCS = ("settings", "router", "roles", "packages", "monitor", "residency", "deciders")
DECIDER_ROUTES = ("status", "models", "autoload", "test")
CATALOG_ROUTES = ("catalog", "item", "tokens", "usage")


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind = status, kind

    def body(self) -> dict:
        return {"error": {"type": self.kind, "message": str(self)}}


def _author(handler, a) -> str:
    """Who wrote a version: how they were let in, and from where (never a token)."""
    h = handler.headers
    if a.admin_bearer(h):
        how = "admin token"
    elif a.session(h) is not None:
        how = "dashboard session"
    elif not a.password_set and not a.trusted_local(handler._peer(), h):
        how = "open"                    # AU1: no admin password set, a write from the network
    else:
        how = "box"
    who = (h.get("X-Real-IP") or handler._peer() or "")[:64]
    return f"{how} {who}".strip()


def _profile_tile(cond, name: str) -> dict:
    v, doc = cond.store.get("profiles", name)
    meta = cond.store.meta("profiles", name)
    tasks = [t["mode"] for t in doc["tasks"].values()]
    decs = list(doc["decisions"].values())
    return {"name": name, "version": v, "mode": doc["mode"], "description": doc["description"],
            "clients": doc["clients"], "default_class": doc["default_class"],
            "locked": schema.LOCKED.get(name),
            "decisions": doc["decisions"],
            "counts": {"tasks": {m: tasks.count(m) for m in schema.TASK_MODES},
                       "decisions": {m: decs.count(m) for m in schema.MODES3}},
            "last_change": {k: meta.get(k) for k in ("version", "note", "author", "created_at")},
            "error": meta.get("error")}


def handle(handler, cond, method: str, body: dict | None, auth) -> None:
    """Answer one `/v1/conductor/*` request on `handler` (server/app.py's Handler). Reads, the
    test box, a dry validation and a role test follow the dashboard's rule; writes need
    `conductor_write`."""
    split = urlsplit(handler.path)
    parts = [unquote(p) for p in split.path.split("/")[3:] if p]
    q = {k: v[-1] for k, v in parse_qs(split.query).items()}
    reads = parts[:1] == ["classify"] or parts[-1:] in (["check"], ["test"]) or \
        parts[:2] == ["packages", "tokens"]
    write = method in ("PUT", "POST", "PATCH", "DELETE") and not reads
    private = parts[:1] == ["memory"]
    if not handler._allowed("conductor_write" if write or private else "dashboard"):
        return
    body = body if isinstance(body, dict) else {}
    try:
        status, payload, raw = _route(handler, cond, method, parts, q, body, auth)
    except ConfigError as exc:
        return handler._json(exc.status, exc.body())
    except ApiError as exc:
        return handler._json(exc.status, exc.body())
    if raw is not None:
        ctype, data = raw
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True
        return
    payload.setdefault("contract_version", CONTRACT)
    return handler._json(status, payload, extra_headers=(("Cache-Control", "no-store"),))


def eval_errors(rep) -> list[str]:
    """What a suite report (tools/tandem_eval.py, OPS-43) lacks for the Compare screen; [] when
    it can be kept. The full schema is docs/contract/dashboard-v1/tandem-eval.schema.json."""
    if not isinstance(rep, dict):
        return ["send {report: {...}}"]
    errs = []
    if rep.get("kind") != "suite":
        errs.append("report.kind: must be 'suite'")
    # a night may carry a suffix (`2026-10-01-r2`, eval night 2 of that date)
    if not re.match(r"^\d{4}-\d{2}-\d{2}(-[a-z0-9]{1,16})?$", str(rep.get("night") or "")):
        errs.append("report.night: YYYY-MM-DD, or YYYY-MM-DD-<suffix> (a-z, 0-9)")
    rows = rep.get("rows")
    if not isinstance(rows, list):
        errs.append("report.rows: a list")
    else:
        for i, r in enumerate(rows):
            if not isinstance(r, dict) or not isinstance(r.get("suite"), str) or \
                    r.get("arm") not in ("raw", "tandem"):
                errs.append(f"report.rows[{i}]: needs suite and arm (raw or tandem)")
    return errs


def _doc_ref(parts: list[str]) -> tuple[str, str | None, list[str]]:
    """(kind, name, the rest of the path) for a document route."""
    kind = parts[0]
    if kind == "profiles":
        if len(parts) < 2:
            raise ApiError(404, "not_found", "which profile?")
        return kind, parts[1], parts[2:]
    return kind, None, parts[1:]


def _base(body: dict) -> int | None:
    b = body.get("base_version")
    if b is None:
        return None
    try:
        return int(b)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", "base_version is a version number")


def _q_int(q: dict, name: str, default: int) -> int:
    """A query parameter as a number; one that is not a number is a 400 naming it. An
    uncaught int() here raises out of the route: the connection dies with no answer and
    every retry of the bad URL spams a traceback (the versions/diff, transcripts, memory
    search and packages/usage routes have answered 400 like this from the first)."""
    v = q.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ApiError(400, "bad_request", f"{name} is a number")


def _route(handler, cond, method, parts, q, body, auth):
    st = cond.store
    if not parts:
        raise ApiError(404, "not_found", "no route /v1/conductor")
    head = parts[0]

    if head == "config" and method == "GET":
        return 200, {"server": {"composite": "TandemLLM", "raw_model": cond.state.get("model"),
                                "reasoning_effort": cond.state.get("reasoning_effort"),
                                "global_mode": cond.state.get("conductor_mode"),
                                # the session monitor's `--conductor` mode (raw: it does not run)
                                "monitor_mode": (cond.state["monitor"].mode
                                                 if cond.state.get("monitor") is not None
                                                 else "raw"),
                                "self_url": cond.self_url()},
                     "store": st.status(), "journal": cond.journal.info(),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, None
    if head == "meta" and method == "GET":
        return 200, schema.meta(), None
    if head == "profiles" and len(parts) == 1 and method == "GET":
        _, settings = st.get("settings")
        return 200, {"default_profile": settings["default_profile"],
                     "profiles": [_profile_tile(cond, n) for n in st.names()]}, None
    if head == "classify" and method == "POST":
        text = body.get("text")
        msgs = body.get("messages")
        if not (isinstance(text, str) and text.strip()) and not isinstance(msgs, list):
            raise ApiError(400, "bad_request", "send text (the prompt to test) or messages")
        prof = body.get("profile")
        if prof is not None and not st.exists("profiles", str(prof)):
            raise ApiError(400, "bad_request", f"no profile {prof!r}")
        return 200, cond.router.explain(text or "", prof, tools=bool(body.get("tools", True)),
                                        messages=msgs if isinstance(msgs, list) else None), None
    if head == "events" and len(parts) == 2 and method == "GET":
        e = cond.journal.get(parts[1])
        if e is None or e.get("kind") != "monitor":
            raise ApiError(404, "not_found", f"no monitor event {parts[1]}")
        return 200, {"event": e}, None
    if head == "events" and len(parts) == 1 and method == "GET":
        evs = cond.journal.rows(since_ms=_q_int(q, "since", 0), kind="monitor",
                                limit=max(1, min(2000, _q_int(q, "limit", 100))),
                                session=q.get("session") or None,
                                before_ms=_q_int(q, "before", 0) if q.get("before") else None,
                                # VIS-44: one request's events, the row drawer's list
                                request_ids=[q["request"]] if q.get("request") else None)
        if q.get("task"):
            evs = [e for e in evs if e.get("task") == q["task"]]
        if q.get("mode"):
            evs = [e for e in evs if e.get("mode") == q["mode"]]
        if q.get("acted") == "1":
            evs = [e for e in evs if dec_mod.is_act(e)]
        if q.get("decision"):
            evs = [e for e in evs if e.get("decision") == q["decision"]]
        return 200, {"events": evs, "monitor": cond.state.get("monitor") is not None}, None
    if head == "watch" and len(parts) == 1 and method == "GET":
        mon = cond.state.get("monitor")
        return 200, {"monitor": mon is not None,
                     "sessions": mon.watching() if mon is not None else []}, None
    if head == "decisions":
        if len(parts) == 1 and method == "GET":
            rows = cond.journal.rows(since_ms=_q_int(q, "since", 0),
                                     limit=max(1, min(500, _q_int(q, "limit", 100))),
                                     session=q.get("session") or None,
                                     before_ms=_q_int(q, "before", 0) if q.get("before") else None,
                                     kind="router")
            return 200, {"decisions": cond.journal.with_events(rows)}, None
        if len(parts) == 2 and method == "GET":
            r = cond.journal.get(parts[1])
            if r is None:
                raise ApiError(404, "not_found", f"no decision {parts[1]}")
            return 200, {"decision": r}, None
        if len(parts) == 3 and parts[2] == "label" and method == "POST":
            if cond.journal.get(parts[1]) is None:
                raise ApiError(404, "not_found", f"no decision {parts[1]}")
            label = body.get("label")
            if label not in ("right", "wrong", "clear"):
                raise ApiError(400, "bad_request", "label is right, wrong or clear")
            note = str(body.get("note") or "")[:400]
            cond.journal.update(parts[1], label=None if label == "clear" else label,
                                label_note=note or None,
                                labelled_at=int(time.time() * 1000) if label != "clear" else None)
            return 200, {"id": parts[1], "label": None if label == "clear" else label}, None
    if head == "stats" and method == "GET":
        hours = max(1, min(168, _q_int(q, "hours", 24)))
        since = int((time.time() - hours * 3600) * 1000)
        rows = cond.journal.rows(since_ms=since, limit=100000)
        # the monitor block is counts: no tile of it ever shows a hidden text, so a
        # week of monitor rows is not read with every one of them attached
        evs = (cond.journal.rows(since_ms=since, limit=200000, kind="monitor",
                                 texts=False)
               if cond.state.get("monitor") is not None else None)
        out = dec_mod.stats(rows, hours=hours, events=evs)
        out["roles"] = cond.roles_health()
        return 200, out, None
    if head == "sessions" and method == "GET":
        if len(parts) == 1:
            since = int((time.time() - 7 * 86400) * 1000)
            # Two reads, one page of 200 lines. The fetch used to be two global
            # reads of the whole week — up to 100k of the router's rows with
            # their JSON and 200k of the monitor's with every hidden text,
            # 300k dicts per request (the 2026-10-16 audit) — for a page whose
            # lines each count one session. First group the keys by newest,
            # then read only those, and never a text a line never says.
            keys = cond.journal.recent_sessions(since_ms=since, limit=200)
            rows = (cond.journal.rows(since_ms=since, limit=100000, sessions=keys)
                    if keys else [])
            evs = (cond.journal.rows(since_ms=since, limit=200000, kind="monitor",
                                     sessions=keys, texts=False) if keys else [])
            return 200, {"sessions": dec_mod.sessions(rows, evs)[:200],
                         "live": cond.router.sessions_live()}, None
        key = parts[1]
        rows = cond.journal.rows(session=key, limit=2000)
        evs = cond.journal.rows(session=key, limit=20000, kind="monitor")
        if len(parts) == 3 and parts[2] == "export":
            fmt = q.get("format") or "json"
            if fmt not in ("json", "md"):
                raise ApiError(400, "bad_request", "format is json or md")
            if not rows and not evs:
                raise ApiError(404, "not_found", f"no session {key}")
            if fmt == "md":
                return 200, {}, ("text/markdown; charset=utf-8",
                                 dec_mod.export_md(key, rows, evs).encode())
            out = dec_mod.export(key, rows, evs)
            # VIS-48: where the session's transcript is (its text needs the admin token)
            d = cond.transcripts.find(key)
            out["transcript"] = ({"blocks": (summary_counts(d) or {}).get("blocks"),
                                  "export": f"/v1/conductor/memory/sessions/{key}/export"}
                                 if d else None)
            return 200, out, None
        if len(parts) == 2:
            live = next((s for s in cond.router.sessions_live() if s["key"] == key), None)
            if not rows and not evs and live is None:
                raise ApiError(404, "not_found", f"no session {key}")
            return 200, {"session": key, "live": live,
                         "turns": sorted(cond.journal.with_events(rows, evs),
                                         key=lambda r: r["ts_ms"]),
                         "events": sorted(evs, key=lambda e: e["ts_ms"])}, None
    if head == "evals" and len(parts) == 1 and method == "POST":
        rep = body.get("report")
        errs = eval_errors(rep)
        if errs:
            raise ApiError(422, "invalid_report", "; ".join(errs[:5]))
        d = os.path.join(cond.root, "evals")
        os.makedirs(d, exist_ok=True)
        name = f"suite-{rep['night']}-{time.strftime('%H%M%S', time.gmtime())}.json"
        tmp = os.path.join(d, "." + name)
        with open(tmp, "w") as fh:
            json.dump(rep, fh, indent=1)
        os.replace(tmp, os.path.join(d, name))
        return 201, {"file": name}, None
    if head == "evals" and method == "GET":
        reports = []
        for f in sorted(glob.glob(os.path.join(cond.root, "evals", "*.json")), reverse=True)[:50]:
            try:
                with open(f) as fh:
                    rep = json.load(fh)
                rep["file"] = os.path.basename(f)
                reports.append(rep)
            except (OSError, ValueError):
                continue
        # VIS-68: a release tree ships its own speed-guard report (release/speed-guard.json, written in
        # the fold by tools/speed_guard.py ship), so a release's build line reads its result out of
        # the box; a report of the box's own for the same build comes first
        try:
            with open(SHIPPED_GUARD) as fh:
                rep = json.load(fh)
            if isinstance(rep, dict) and rep.get("kind") == "speed_guard":
                rep["file"], rep["shipped"] = "release/speed-guard.json", True
                reports.append(rep)
        except (OSError, ValueError):
            pass
        # a report another one supersedes says so, also when its own file was never marked
        # (the kept files are not rewritten)
        names = {r["file"]: r for r in reports}
        for r in reports:
            old = (r.get("supersedes") or {}).get("file") if isinstance(r.get("supersedes"),
                                                                        dict) else None
            if old in names and not names[old].get("superseded_by"):
                names[old]["superseded_by"] = {"file": r["file"],
                                               "why": r["supersedes"].get("why")}
        return 200, {"reports": reports,
                     "engine": {"version": cond.state.get("version"),
                                "code_sha": cond.state.get("code_sha")}}, None
    if head == "roles" and len(parts) == 3 and parts[2] == "test" and method == "POST":
        return 200, cond.role_test(parts[1]), None
    if head == "roles" and len(parts) == 1 and method == "GET":
        v, doc = st.get("roles")
        return 200, {"version": v, "doc": doc, "meta": st.meta("roles"),
                     "health": cond.roles_health(force=q.get("fresh") == "1"),
                     "contract": cond.roles_contract()}, None
    if head == "packages" and len(parts) == 2 and parts[1] == "scan" and method == "GET":
        _, doc = st.get("packages")
        return 200, {"skills": cond.library.scan(doc)}, None
    if head == "packages" and len(parts) >= 2 and parts[1] in CATALOG_ROUTES:
        return _catalog(cond, method, parts[1:], q, body)
    if head == "profiles" and len(parts) == 3 and parts[2] == "mode" and method == "POST":
        name = parts[1]
        mode = body.get("mode")
        if mode not in schema.PROFILE_MODES:
            raise ApiError(400, "bad_request", f"mode is one of {', '.join(schema.PROFILE_MODES)}")
        if not st.exists("profiles", name):
            raise ApiError(404, "not_found", f"no profile {name}")
        v, doc = st.get("profiles", name)
        doc = dict(doc, mode=mode)
        nv = st.put("profiles", name, doc, note=body.get("note") or f"mode to {mode}",
                    author=_author(handler, auth), base=_base(body))
        return 200, {"name": name, "version": nv, "mode": mode,
                     "loaded_at": st.meta("profiles", name).get("loaded_at")}, None

    if head == "memory":
        return _memory(cond, method, parts[1:], q, body, _author(handler, auth) if auth else
                       "dashboard")
    if head == "deciders" and parts[1:2] and parts[1] in DECIDER_ROUTES:
        return _deciders(cond, method, parts[1:], body)
    if head == "narrator" and len(parts) == 2:
        return _narrator(cond, method, parts[1], body)
    if head == "shadow" and len(parts) == 2:
        return _shadow(cond, method, parts[1], body)
    if head == "residency" and parts[1:] == ["plan"] and method == "GET":
        # OPS-41: the planner's view -- the limits, what fits now, the engine's own split, the
        # journal's last refusal -- for the System view
        from server import memsplit
        from server.conductor import residency as res_mod
        v, doc = st.get("residency")
        _, roles = st.get("roles")
        jr = res_mod.Journal(cond.root)
        try:
            split = memsplit.split(cond.state)
        except Exception:                                          # noqa: BLE001
            split = None
        # OPS-47: the planner counts the engine's growth to its declared peak, and the panel
        # shows every memory limit with what it holds now
        from server import memlimits
        peak, now = memlimits.engine_gib(cond.state)
        try:
            limits = memlimits.view(cond.state)
        except Exception:                                          # noqa: BLE001
            limits = None
        return 200, {"version": v, "doc": doc,
                     "plan": res_mod.plan(res_mod._mem_gib(), doc, roles, engine_peak_gib=peak,
                                          engine_now_gib=now),
                     "engine": split, "limits": limits, "last_refusal": jr.last_refusal(),
                     "journal": jr.tail(20)}, None
    if head in DOCS or head == "profiles":
        kind, name, rest = _doc_ref(parts)
        if not st.exists(kind, name) and not (method == "PUT" and kind == "profiles"):
            raise ApiError(404, "not_found", f"no {kind} {name or ''}".strip())
        if method == "GET" and not rest:
            v, doc = st.get(kind, name)
            out = {"kind": kind, "name": name, "version": v, "doc": doc,
                   "meta": st.meta(kind, name)}
            if kind == "profiles":
                out["locked"] = schema.LOCKED.get(name)
            if kind == "packages":
                out["status"] = cond.library.status(doc)
            return 200, out, None
        if method == "PUT" and not rest:
            doc = body.get("doc")
            if not isinstance(doc, dict):
                raise ApiError(400, "bad_request", "send {doc, note}")
            nv = st.put(kind, name, doc, note=str(body.get("note") or ""),
                        author=_author(handler, auth), base=_base(body))
            return 200, {"kind": kind, "name": name, "version": nv,
                         "loaded_at": st.meta(kind, name).get("loaded_at")}, None
        if method == "GET" and rest == ["versions"]:
            return 200, {"kind": kind, "name": name, "versions": st.versions(kind, name)}, None
        if method == "GET" and len(rest) == 2 and rest[0] == "versions":
            try:
                v = int(rest[1])
            except ValueError:
                raise ApiError(400, "bad_request", "a version is a number")
            return 200, {"kind": kind, "name": name, "version": v,
                         "doc": st.version(kind, name, v)}, None
        if method == "GET" and rest == ["diff"]:
            cur = st.get(kind, name)[0]
            try:
                a = int(q.get("from") if q.get("from") is not None else cur)
                b = int(q.get("to") if q.get("to") is not None else cur)
            except ValueError:
                raise ApiError(400, "bad_request", "from and to are version numbers")
            return 200, {"kind": kind, "name": name, "from": a, "to": b,
                         "changes": store_mod.diff(st.version(kind, name, a),
                                                   st.version(kind, name, b))}, None
        if method == "POST" and rest == ["check"]:
            doc = body.get("doc")
            errs = st.check(kind, name, doc) if isinstance(doc, dict) else ["$: send {doc}"]
            cur_v, cur = st.get(kind, name)
            return 200, {"valid": not errs, "errors": errs[:50],
                         "changes": store_mod.diff(cur, doc) if isinstance(doc, dict) else [],
                         "base_version": cur_v}, None
        if method == "POST" and rest == ["restore"]:
            try:
                v = int(body.get("version"))
            except (TypeError, ValueError):
                raise ApiError(400, "bad_request", "send {version}")
            nv = st.restore(kind, name, v, author=_author(handler, auth), note=body.get("note"))
            return 200, {"kind": kind, "name": name, "version": nv, "restored_from": v,
                         "loaded_at": st.meta(kind, name).get("loaded_at")}, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/{'/'.join(parts)}")


def summary_counts(d: str) -> dict | None:
    from server.conductor import transcripts
    return (transcripts.read_meta(d) or {}).get("counts")


def _memory(cond, method: str, rest: list[str], q: dict, body: dict,
            handler_author: str = "dashboard"):
    """The Memory screen's routes over the session transcripts (VIS-48, SRV-137) and the disk
    tier (C1)."""
    if rest[:1] == ["disk"]:
        from server import diskcache
        if rest == ["disk"] and method == "GET":
            return 200, diskcache.view(cond.state), None
        if rest == ["disk", "clear"] and method == "POST":
            got = diskcache.clear(cond.state)
            if not got.get("cleared"):
                raise ApiError(409, "not_running", got["why"])
            return 200, {**got, "disk": diskcache.view(cond.state)}, None
    if rest[:1] == ["notes"]:
        return _notes(cond, method, rest[1:], q, body)
    if rest[:1] == ["proposals"]:
        return _proposals(cond, method, rest[1:], body, handler_author)
    tr = cond.transcripts
    if rest == ["sessions"] and method == "GET":
        return 200, {"sessions": tr.sessions(), "writer": tr.info()}, None
    if len(rest) >= 2 and rest[0] == "sessions":
        key = rest[1]
        if len(rest) == 2 and method == "GET":
            try:
                page = int(q["page"]) if q.get("page") else None
            except ValueError:
                raise ApiError(400, "bad_request", "page is a number")
            got = tr.read(key, page)
            if got is None:
                raise ApiError(404, "not_found", f"no transcript for session {key}")
            return 200, got, None
        if len(rest) == 3 and rest[2] == "keep" and method in ("PATCH", "POST"):
            if not isinstance(body.get("keep"), bool):
                raise ApiError(400, "bad_request", "send {keep: true|false}")
            got = tr.set_keep(key, body["keep"])
            if got is None:
                raise ApiError(404, "not_found", f"no transcript for session {key}")
            return 200, got, None
        if len(rest) == 3 and rest[2] == "export" and method == "GET":
            fmt = q.get("format") or "md"
            if fmt not in ("json", "md"):
                raise ApiError(400, "bad_request", "format is json or md")
            got = tr.export(key)
            if got is None:
                raise ApiError(404, "not_found", f"no transcript for session {key}")
            meta, text = got
            if fmt == "md":
                return 200, {}, ("text/markdown; charset=utf-8", text.encode())
            page = tr.read(key, 1, per_page=10 ** 9)
            return 200, {"session": meta, "blocks": page["blocks"] if page else []}, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/memory/{'/'.join(rest)}")


def _notes(cond, method: str, rest: list[str], q: dict, body: dict):
    """The Memory screen's Notes tab (2026-10-05, server/conductor/notes.py)."""
    from server.conductor import notes as notes_mod
    st = cond.notes
    user = q.get("user") or "owner"
    if not notes_mod.valid_user(user):
        raise ApiError(400, "bad_request", "user: letters, digits, dot, dash, underscore")
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if rest == [] and method == "GET":
        live = st.notes(user)
        rows = []
        for n in live:
            m = n.meta()
            m.update(node=st.node_of(n), stale=n.stale(today),
                     uses_7d=sum(st.uses(user, n.id).values()))
            rows.append(m)
        rows.sort(key=lambda r: r.get("updated") or "", reverse=True)
        users = st.users()
        return 200, {"user": user, "users": users if user in users else users + [user],
                     "tree": st.tree(user)["nodes"], "notes": rows,
                     "archived": [a.meta() for a in st.archived(user)[:200]],
                     "duplicates": [{"a": a, "b": b, "similarity": x}
                                    for a, b, x in st.duplicates(user)],
                     "kinds": {k: notes_mod.KIND_WORDS[k] for k in notes_mod.KINDS},
                     "store": st.info(), "memory": dict(cond.memory.stats),
                     "settings": schema.notes(cond.store.peek("settings")[1])}, None
    if rest == ["search"] and method == "GET":
        text = q.get("q") or ""
        if not text.strip():
            raise ApiError(400, "bad_request", "q: the words to search for")
        try:
            k = max(1, min(50, int(q.get("k") or 10)))
        except ValueError:
            raise ApiError(400, "bad_request", "k is a number")
        hits = st.search(user, text, k=k, all_scopes=True)
        return 200, {"q_words": len(text.split()), "hits": [
            dict(n.meta(), score=round(sc, 3), node=st.node_of(n)) for n, sc in hits]}, None
    if rest == ["injected"] and method == "GET":
        limit = _int(q.get("limit"), 50, 1, 500)
        rows = [r for r in cond.journal.rows(kind="memory_notes", limit=limit * 4)
                if r.get("task") == "injected" and
                ((r.get("detail") or {}).get("user") in (None, user))][:limit]
        return 200, {"rows": [{"id": r.get("id"), "ts_ms": r.get("ts_ms"),
                               "request_id": r.get("request_id"),
                               "session_key": r.get("session_key"),
                               "ids": (r.get("detail") or {}).get("ids") or [],
                               "index": (r.get("detail") or {}).get("index", False)}
                              for r in rows]}, None
    if rest == ["writes"] and method == "GET":
        limit = _int(q.get("limit"), 50, 1, 500)
        rows = [r for r in cond.journal.rows(kind="memory_notes", limit=limit * 4)
                if r.get("task") in ("write", "api_write", "nightly", "writer")][:limit]
        return 200, {"rows": [{"id": r.get("id"), "ts_ms": r.get("ts_ms"), "task": r.get("task"),
                               "decision": r.get("decision"), "detail": r.get("detail"),
                               "request_id": r.get("request_id"),
                               "session_key": r.get("session_key")} for r in rows],
                     "writer": dict(cond.memory.writer.stats)}, None
    if rest == ["export"] and method == "GET":
        if user not in st.users():
            raise ApiError(404, "not_found", f"no memory for {user}")
        return 200, {}, ("application/zip", st.export(user))
    if len(rest) == 2 and rest[0] == "users" and method == "DELETE":
        who = rest[1]
        if not notes_mod.valid_user(who) or body.get("confirm") != who:
            raise ApiError(400, "bad_request", "send {confirm: <the user's name>} to delete "
                                               "every note of the user")
        if not st.delete_user(who):
            raise ApiError(404, "not_found", f"no memory for {who}")
        cond.journal.note("memory_notes", "delete_user", decision="deleted",
                          detail={"user": who})
        return 200, {"deleted": who}, None
    if not rest:
        raise ApiError(404, "not_found", f"no route {method} /v1/conductor/memory/notes")
    nid = rest[0]
    try:
        if len(rest) == 1 and method == "GET":
            n = st.get(user, nid) or next((a for a in st.archived(user) if a.id == nid), None)
            if n is None:
                raise ApiError(404, "not_found", f"no note {nid}")
            return 200, {"note": dict(n.full(), node=st.node_of(n), stale=n.stale(today)),
                         "revisions": st.revisions(user, nid),
                         "uses": st.uses(user, nid, days=30)}, None
        if len(rest) == 1 and method == "PUT":
            n = st.edit(user, nid, {k: body[k] for k in ("title", "body", "kind", "tags",
                                                         "verify_by") if k in body})
            _noted(cond, "edit", nid, user)
            return 200, {"note": n.full()}, None
        if len(rest) == 1 and method == "PATCH":
            if "status" not in body and "pinned" not in body:
                raise ApiError(400, "bad_request", "send {status} or {pinned}")
            n = st.set_status(user, nid, body.get("status"),
                              body["pinned"] if "pinned" in body else None)
            _noted(cond, "status", nid, user)
            return 200, {"note": n.full()}, None
        if len(rest) == 2 and rest[1] == "archive" and method == "POST":
            got = st.archive(user, nid, str(body.get("reason") or "archived in the dashboard"))
            _noted(cond, "archive", nid, user)
            return 200, got, None
        if len(rest) == 1 and method == "DELETE":
            if not st.delete(user, nid):
                raise ApiError(404, "not_found", f"no note {nid}")
            _noted(cond, "delete", nid, user)
            return 200, {"deleted": nid}, None
    except notes_mod.Refused as exc:
        raise ApiError(404 if exc.why == "unknown_id" else 422, exc.why, str(exc))
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/memory/notes/"
                                     f"{'/'.join(rest)}")


def _proposals(cond, method: str, rest: list[str], body: dict, author: str):
    """P-M3: the model's package proposals (server/conductor/proposals.py): the list, one with
    the live text it would replace, Apply (the packages document's versioned PUT) and Reject."""
    from server.conductor import notes as notes_mod
    pr = cond.proposals
    if rest == [] and method == "GET":
        return 200, {"proposals": [{k: v for k, v in p.items() if k not in ("proposed", "body")}
                                   for p in pr.all()]}, None
    if len(rest) < 2 or not rest[1].isdigit():
        raise ApiError(404, "not_found", f"no route {method} /v1/conductor/memory/proposals/"
                                         f"{'/'.join(rest)}")
    name, n = rest[0], int(rest[1])
    try:
        if len(rest) == 2 and method == "GET":
            p = pr.get(name, n)
            if p is None:
                raise ApiError(404, "not_found", f"no proposal {name}#{n}")
            return 200, {"proposal": pr.view(p)}, None
        if len(rest) == 3 and rest[2] == "apply" and method == "POST":
            return 200, pr.apply(name, n, author), None
        if len(rest) == 3 and rest[2] == "reject" and method == "POST":
            return 200, pr.reject(name, n, str(body.get("reason") or "rejected in the "
                                                                   "dashboard")), None
    except notes_mod.Refused as exc:
        raise ApiError({"unknown_id": 404, "unknown_package": 404, "stale": 409,
                        "closed": 409}.get(exc.why, 422), exc.why, str(exc))
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/memory/proposals/"
                                     f"{'/'.join(rest)}")


def _noted(cond, what: str, nid: str, user: str) -> None:
    cond.journal.note("memory_notes", "human", decision=what, detail={"id": nid, "user": user})


def _int(v, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v))) if v else default
    except ValueError:
        return default


def _catalog(cond, method: str, rest: list[str], q: dict, body: dict):
    """The Packages and Skills screen's reads (G2, server/conductor/catalog.py); none writes."""
    from server.conductor import catalog
    if rest == ["catalog"] and method == "GET":
        return 200, catalog.catalog(cond), None
    if len(rest) == 2 and rest[0] == "item" and method == "GET":
        got = catalog.item(cond, rest[1])
        if got is None:
            raise ApiError(404, "not_found", f"no package {rest[1]!r}")
        return 200, got, None
    if rest == ["tokens"] and method == "POST":
        texts = body.get("texts")
        if not isinstance(texts, list) or not 1 <= len(texts) <= catalog.MAX_TEXTS or \
                not all(isinstance(t, str) and len(t) <= catalog.MAX_CHARS for t in texts):
            raise ApiError(400, "bad_request", f"texts is a list of 1 to {catalog.MAX_TEXTS} "
                                               f"strings of at most {catalog.MAX_CHARS:,} "
                                               f"characters")
        nums, how = catalog.count(cond.state, texts)
        return 200, {"tokens": nums, "tokenizer": how}, None
    if rest == ["usage"] and method == "GET":
        try:
            limit = int(q.get("limit") or 20)
        except ValueError:
            raise ApiError(400, "bad_request", "limit is a number")
        if not 1 <= limit <= 200:
            raise ApiError(400, "bad_request", "limit is 1 to 200")
        return 200, catalog.usage(cond, q.get("name") or None, limit), None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/packages/{'/'.join(rest)}")


def _narrator(cond, method: str, what: str, body: dict):
    """The narrator's routes over `STATE["narrator"]` (None: no narrator, and only `status`
    answers, saying so)."""
    from server.conductor import narrator as narr_mod
    n = cond.state.get("narrator")
    if what == "status" and method == "GET":
        return 200, narr_mod.status(cond.state), None
    if n is None:
        raise ApiError(404, "not_found", narr_mod.OFF_WHY)
    if what == "try" and method == "POST":
        events = body.get("events")
        if not isinstance(events, list) or not events or len(events) > narr_mod.TRY_MAX_EVENTS \
                or not all(isinstance(e, dict) for e in events):
            raise ApiError(400, "bad_request", f"events is a list of 1 to "
                                               f"{narr_mod.TRY_MAX_EVENTS} objects")
        budget, cap = body.get("budget_ms"), body.get("max_tokens")
        for k, v, top in (("budget_ms", budget, narr_mod.TRY_MAX_BUDGET_MS),
                          ("max_tokens", cap, narr_mod.TRY_MAX_TOKENS)):
            if v is not None and (not isinstance(v, int) or isinstance(v, bool)
                                  or not 1 <= v <= top):
                raise ApiError(400, "bad_request", f"{k} is an integer from 1 to {top}")
        lines = n.try_events(events, budget_ms=budget, max_tokens=cap)
        return 200, {"lines": lines, "slot": n.slot.status()}, None
    if what in ("load", "unload") and method == "POST":
        if what == "load":
            # v3: ops/start.sh asks at the start (`service_start`); from then on the server's
            # keeper loads it again whenever it is not loaded (app.narrator_once)
            ss, hr = body.get("service_start"), body.get("hold_restart")
            for k, v in (("service_start", ss), ("hold_restart", hr)):
                if v is not None and not isinstance(v, bool):
                    raise ApiError(400, "bad_request", f"{k} is a boolean")
            if n.settings()["mode"] == "off":
                # 0.11.3: the role is off by default; ops/start.sh asks whenever the model is set
                print("[narrator] not loaded: the narrator role is off (roles document, "
                      "narrator.mode)", flush=True)
                return 409, {"state": n.slot.state, "error": {
                    "type": "role_off", "message": "the narrator role is off (the roles "
                                                   "document's narrator.mode); nothing loads"}}, None
            cond.state["narrator_keep"] = True
            from server import decider_api
            if n.slot.state not in ("loaded", "loading") and \
                    decider_api._other_loading(cond.state, "narrator"):
                # one load at a time: the keeper loads it once the decision models are in
                return 202, {"state": n.slot.state, "queued": True}, None
            d = n.slot.load(service_start=bool(ss), hold_restart=bool(hr))
            if not d.get("allowed"):
                return 409, {"state": n.slot.state, "decision": d,
                             "error": {"type": d.get("reason") or "refused",
                                       "message": f"the residency planner refused the load: "
                                                  f"{d.get('reason')}"}}, None
            return 202, {"state": n.slot.state, "decision": d}, None
        cond.state["narrator_keep"] = False        # unloaded by hand: the keeper leaves it
        if not n.slot.unload():
            raise ApiError(409, "not_loaded", f"the narrator is {n.slot.state}, not loaded")
        return 200, {"state": n.slot.state}, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/narrator/{what}")


def _shadow(cond, method: str, what: str, body: dict):
    """The shadow reader's routes over `STATE["shadow"]` (SR1)."""
    from server.conductor import shadow as sh_mod
    svc = cond.state.get("shadow")
    if what == "status" and method == "GET":
        return 200, sh_mod.status(cond.state), None
    if svc is None:
        raise ApiError(404, "not_found", cond.state.get("shadow_refused") or sh_mod.OFF_WHY)
    if what == "ask" and method == "POST":
        key, question, choices = body.get("session"), body.get("question"), body.get("choices")
        if not isinstance(key, str) or not key:
            raise ApiError(400, "bad_request", "session is the router's session key")
        if not isinstance(question, str) or not 4 <= len(question) <= 4000:
            raise ApiError(400, "bad_request", "question is a string of 4 to 4,000 characters")
        if (not isinstance(choices, dict) or not 2 <= len(choices) <= 26
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in choices.items())):
            raise ApiError(400, "bad_request", "choices is an object of 2 to 26 id: text pairs")
        budget = body.get("budget_ms", 5000)
        if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 60000:
            raise ApiError(400, "bad_request", "budget_ms is an integer from 1 to 60000")
        gate = getattr(svc.slot, "gate", None)
        if gate is not None and not gate.acquire(timeout=30):
            raise ApiError(409, "busy", "the engine stayed busy for 30 s")
        try:
            out = svc.ask(key, question, choices, budget_ms=float(budget))
        finally:
            if gate is not None:
                gate.release()
        if out.get("why"):
            return 409, {"error": {"type": out["why"], "message": out.get("error") or out["why"]}}, \
                None
        return 200, out, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/shadow/{what}")


def _deciders(cond, method: str, rest: list[str], body: dict):
    """The Deciders screen's routes over the registry in `STATE["deciders"]` (None: the engine
    serves no decision model, and only `status` answers, saying so)."""
    from server import decider_api
    reg = cond.state.get("deciders")
    if rest == ["status"] and method == "GET":
        return 200, decider_api.status(reg, cond.state), None
    if reg is None:
        raise ApiError(404, "not_found", decider_api.OFF_WHY)
    if rest == ["test"] and method == "POST":
        status, out, _ = decider_api.ask(reg, body, source="test")
        if status != 200:
            raise ApiError(status, out["error"]["type"], out["error"]["message"])
        return 200, out, None
    if rest == ["autoload"] and method == "POST":
        names = reg.autoload(service_start=bool(body.get("service_start")),
                             hold_restart=bool(body.get("hold_restart")))
        return 202, {"queued": names}, None
    if len(rest) == 3 and rest[0] == "models" and rest[2] in ("load", "unload") and \
            method == "POST":
        name = rest[1]
        if name not in reg.models:
            raise ApiError(404, "not_found", f"this engine does not serve {name!r}; it serves "
                                             f"{', '.join(reg.order)}")
        if rest[2] == "load":
            d = reg.load(name)
            if not d.get("allowed"):
                return 409, {"name": name, "state": reg.state[name], "decision": d,
                             "error": {"type": d.get("reason") or "refused",
                                       "message": f"the residency planner refused the load: "
                                                  f"{d.get('reason')}"}}, None
            return 202, {"name": name, "state": reg.state[name], "decision": d}, None
        if not reg.unload(name):
            raise ApiError(409, "not_loaded", f"{name} is {reg.state[name]}, not loaded")
        return 200, {"name": name, "state": reg.state[name]}, None
    raise ApiError(404, "not_found", f"no route {method} /v1/conductor/deciders/{'/'.join(rest)}")
