"""The HTTP reference of the manual: every route, header and error, checked against the code.

The routes, headers and errors are written here by hand, because a handler's purpose and the
shape of its body are not in the code in a form a script can read. What keeps them true is the
check: `scan()` reads the code and `problems()` names

  - a path the request handlers test that no route row covers (a route was added),
  - a route row whose path the handlers never test (a route was removed or renamed),
  - a header name the code reads or writes that no header row names, and a row the code never uses,
  - an error type or code the code puts in an error body that no error row names, and a row
    whose types the code never uses.

`gen_reference.py` writes the tables and fails (`--check` too) while any problem is open, so a
change to the API cannot ship without the book.
"""

from __future__ import annotations

import ast
import os
import re

# --------------------------------------------------------------------------- who may call a route
# The labels of the `who` column; ref-http.tex explains them.
WHO = {
    "open": "anyone who reaches the port",
    "admin": "the admin rule: anyone while no admin password is set; with one, the admin token, a dashboard session, or a tool on the machine itself",
    "admin-bearer": "as admin, but the token only as a bearer header, never the session cookie",
    "dash": "the dashboard rule: anyone unless a password is set and the dashboard login is on",
    "metrics": "the metrics token when one is set, else as admin",
    "session": "only under a session address, /s/<token>/...",
    "key": "anyone until the first user key exists; then a user key, except from the machine itself",
    "collect": "the collect token, from every address",
}

# --------------------------------------------------------------------------- the engine's routes
# (method, path, who, what, body, answer). `path` placeholders are {name}; a query is ?a=&b=.
ENGINE_ROUTES = [
    ("POST", "/v1/chat/completions", "open",
     "A chat completion, streamed as server-sent events or not. With --backends a model of another backend is relayed there.",
     "OpenAI's chat body: model, messages, tools, stream, max_tokens and the other fields of Section api-fields",
     "chat.completion, or chat.completion.chunk events and [DONE]; usage carries prompt_tokens_details.hidden_tokens for the conductor's notes"),
    ("POST", "/v1/completions", "open",
     "A plain text completion.",
     "prompt, max_tokens and the sampling fields",
     "text_completion, or its chunks and [DONE]"),
    ("POST", "/v1/embeddings", "key",
     "Relayed to the backend that serves the model; only with --backends.",
     "OpenAI's embeddings body", "the backend's answer, unchanged"),
    ("POST", "/v1/responses", "key",
     "OpenAI's Responses API, relayed to an openai backend; only with --backends.",
     "the client's body, sent on with the backend's model id", "the backend's answer, unchanged"),
    ("GET", "/v1/models", "open",
     "The model list: TandemLLM alone with the conductor on, else the served name; with --backends every backend's models after it, behind the users' keys.",
     "", "{object: list, data: [...]}, and decision_models beside data when decision models are served"),
    ("GET", "/v1/models/{id}", "open",
     "One model card: the engine's, a decision model's, or a backend's.",
     "", "the card, or 404 model_not_found"),
    ("GET", "/health", "open",
     "Whether the server takes work. The full body follows the admin rule; anyone else gets the status alone. Also /healthz and /v1/health.",
     "", "{status: ok|draining}; the full body adds model, uptime, inflight, the queue, limits, sampling, cache, memory and parallel; 503 while draining"),
    ("GET", "/metrics", "metrics",
     "The Prometheus families of Chapter ref-metrics.", "", "Prometheus text format"),
    ("GET", "/metrics/up", "open",
     "One gauge, qse_up, so a scraper can tell an engine that is down from a wrong token.", "", "qse_up 1, or 0 while draining"),
    ("GET", "/v1/cache/stats", "admin",
     "Every cache's counters.", "", "the cache report /health carries"),
    ("POST", "/v1/cache/peek", "admin",
     "Where a chat's rows are before it is sent: the body is rendered as a request would be, the session's router notes are replayed, and the cache tiers are asked without running the model (SRV-349).",
     "a chat body (messages, tools, the effort fields) and session", "{tier: resident|ram_park|ram_snapshot|nvme_spill|disk_tier|none, cached_tokens, reread_tokens, reread_eta_ms, tokens, restore_ms, session}; 409 while a request holds the engine"),
    ("POST", "/v1/cache/prewarm", "admin",
     "Reads a conversation the peek finds on the NVMe back into the engine while nothing runs or waits, so its next request resumes at the end anchor.",
     "as /v1/cache/peek", "{prewarmed, why?, restore?} and the peek's fields; 409 while busy"),
    ("POST", "/v1/tokenize", "open",
     "Exact token counts with the model's own chat template, split by part; nothing runs on the GPU (SRV-347). 240 a minute from one address.",
     "{messages, tools?} or {text}", "{tokens, parts: {system, tools, messages}}"),
    ("POST", "/v1/cache/clear?tier=ram|disk|all", "admin-bearer",
     "Empties the caches in memory (ram, the default), the disk tier and the session spill (disk), or both (all). Nothing else is reset.",
     "", "the cache report after the clear; 400 for another tier"),
    ("POST", "/v1/cache/release", "admin-bearer",
     "Gives back memory the engine holds idle, cheapest first: free allocator blocks and caches while slot 0 is idle, the small models, idle side conversations, slot 0's idle conversation (to the NVMe first). chatd asks it before a sandbox waits (SRV-483).",
     "{bytes}", "what each step gave back and the total"),
    ("GET", "/dashboard/", "open",
     "The dashboard's static files. They hold no data.", "", "the page"),
    ("GET", "/v1/dashboard/summary?from=&to=&tz=&model=&client=&backend=&user=&internal=", "dash",
     "The Usage view's totals for a range.", "", "summary.schema.json in docs/contract/dashboard-v1"),
    ("GET", "/v1/dashboard/usage?bucket=hour|day&from=&to=&tz=&model=&client=&backend=&user=&internal=", "dash",
     "Tokens, requests and cost per hour or day.", "", "usage.schema.json"),
    ("GET", "/v1/dashboard/requests?limit=&before=&finish=&model=&client=&backend=&user=&internal=", "dash",
     "Recent request rows of the usage ledger, newest first.", "", "requests.schema.json"),
    ("GET", "/v1/dashboard/system", "dash",
     "The machine, the engine, the memory plan, the slots and the proxy's mode.", "", "system.schema.json"),
    ("GET", "/v1/dashboard/live", "dash",
     "What each request is doing now, as server-sent events.", "", "live.schema.json events"),
    ("GET", "/v1/dashboard/logs?level=&grep=&backlog=&since=&follow=", "dash",
     "The log: a backlog of up to 5,000 lines, then live lines as server-sent events; Last-Event-ID resumes; follow=0 answers JSON once.",
     "", "logs-line.schema.json events, or logs-json.schema.json; 429 past four streams at once"),
    ("GET", "/v1/dashboard/metrics", "dash",
     "The /metrics page under the dashboard's rule, for the Performance view.", "", "Prometheus text format"),
    ("GET", "/v1/dashboard/parallel?since=", "dash",
     "Every request in flight, the engine's and the relayed ones.", "", "parallel.schema.json"),
    ("GET", "/v1/dashboard/session", "open",
     "Whether a sign-in exists and is needed.", "", "{authenticated, login, password_set, expires_at}"),
    ("POST", "/v1/dashboard/session", "open",
     "The sign-in, with a password set and the login on: sets the qse_dash cookie. Five failures a minute from one address and the sixth gets a 429.",
     "{token}", "the session; 401 for a wrong token"),
    ("DELETE", "/v1/dashboard/session", "open",
     "Signs this session out.", "", "the session, signed out"),
    ("*", "/v1/conductor/*", "admin",
     "The conductor's documents, journal, sessions and memory (Table tab:ref-conductor-routes). Reads follow the dashboard rule, writes and every memory route the admin rule. 404 without --conductor-config.",
     "", ""),
    ("*", "/v1/setup/*", "admin",
     "The setup guide (docs/setup.md): the model registry, the box, the memory estimate, the review plan, the apply and reset jobs and Wren. Every POST must also be same-origin JSON: 415 for another content type, 403 for another origin. In setup mode (--setup) these, the dashboard and /health are the only routes that answer.",
     "", ""),
    ("*", "/v1/config/*", "admin",
     "The Backends and Setup screens (Table tab:ref-config-routes). Reads follow the dashboard rule, writes the admin rule. 404 without --backends.",
     "", ""),
    ("POST", "/v1/collect/{backend}", "collect",
     "A batch from the vLLM or SGLang plugin, joined to the ledger rows; only with --backends.",
     "the plugin's batch: plugin, engine, requests, steps", "{accepted: {requests, steps}, joined}"),
    ("POST", "/v1/systemone", "admin",
     "Typed decisions from the decision models, in the engine's idle window. /v1/predict is the same route. Only with QWEN38_DECIDER_MODELS.",
     "{state, model?, questions: {id: {type, instructions, criteria?}}, task?, lang?}",
     "{model, answers, usage, routing, latency_ms, path}"),
    ("POST", "/v1/predict", "admin", "The same as /v1/systemone.", "as /v1/systemone", "as /v1/systemone"),
    ("GET", "/readyz", "open",
     "Whether a decision model is loaded and warm.", "", "200, or 503 with the reason"),
    ("GET", "/v1/laya", "open",
     "What the Laya route takes and answers, its limits, and whether it is on.", "", "a description of the route"),
    ("POST", "/v1/laya/ask", "open",
     "A typed answer from Laya on one text or many, for the model's own scripts; rate-limited.",
     "{question, choices | scale | neither, text | texts, threshold?, model?, deadline_ms?}",
     "{label, p, sure, probs, model, ms, ...}, or {answers: [...]} for texts"),
    ("POST", "/v1/laya/drive/end", "open",
     "Ends a playtest driver session and writes its journal row.", "the session's summary", "the row"),
    ("GET", "/v1/laya/kit/{file}", "open",
     "A file of the playtest kit the model copies into a project.", "", "the file"),
    ("OPTIONS", "/v1/laya/*", "open",
     "A browser's CORS preflight for the Laya routes. Every other path answers OPTIONS with 501.", "", "204 with the CORS headers"),
    ("GET", "/v1/library", "open",
     "The package library: every enabled skill's name, description and classes.", "", "the list and the two routes below"),
    ("GET", "/v1/library/search?q=&k=&class=", "open",
     "BM25 search over the skills, best first (k from 1 to 20, default 5).", "", "[{name, description, score}]"),
    ("GET", "/v1/library/{name}", "open",
     "One skill's text and its reference.", "", "{name, description, classes, body, reference}"),
    ("GET", "/s/{token}/v1/memory", "session",
     "The memory index this session sees. The /s/{token}/ prefix also serves the library and Laya routes, filed to the session.",
     "", "the notes' index lines"),
    ("GET", "/s/{token}/v1/memory/search?q=&k=", "session",
     "Memory notes that match, best first (k from 1 to 10).", "", "[{id, kind, scope, status, title, score}]"),
    ("GET", "/s/{token}/v1/memory/{id}", "session",
     "One note.", "", "{id, title, kind, scope, status, body, ...}"),
    ("POST", "/s/{token}/v1/memory", "session",
     "Writes, changes or archives a note, with the fields of the hidden memory block.",
     "{op, scope, kind, title, tags, body, id, reason}", "{op, id, title, ...}, or 422 with the reason as the type"),
]

# --------------------------------------------------------------------------- the standalone proxy's routes
PROXY_ROUTES = [
    ("POST", "/v1/chat/completions", "key",
     "Relayed to the backend the model id routes to (Section sec:routing); the answer's bytes go back unchanged. The same for /v1/completions, /v1/embeddings and /v1/responses.",
     "the client's body", "the backend's answer, with X-Tandem-Backend"),
    ("POST", "/v1/completions", "key", "As /v1/chat/completions.", "", ""),
    ("POST", "/v1/embeddings", "key", "As /v1/chat/completions; not for deepseek backends.", "", ""),
    ("POST", "/v1/responses", "key", "As /v1/chat/completions; openai backends only.", "", ""),
    ("GET", "/v1/models", "key",
     "Every backend's models under their real ids, owned_by the backend.", "", "{object: list, data: [... with a tandem block]}"),
    ("GET", "/v1/models/{id}", "key", "One model's card.", "", "the card, or 404 model_not_found"),
    ("GET", "/", "open", "Redirects to the dashboard.", "", "302 to /dashboard/"),
    ("GET", "/dashboard/", "open", "The dashboard's static files.", "", "the page"),
    ("GET", "/v1/dashboard/{view}", "dash",
     "summary, usage, requests, system, live, logs, metrics, parallel and session, as on the engine; the engine-only panels answer 404.",
     "", "as on the engine"),
    ("POST", "/v1/dashboard/session", "open", "As on the engine.", "{token}", "the session"),
    ("*", "/v1/config/*", "admin", "The Backends and Setup screens (Table tab:ref-config-routes).", "", ""),
    ("POST", "/v1/collect/{backend}", "collect", "A plugin's batch (as on the engine).", "", ""),
    ("GET", "/metrics", "metrics", "The proxy's own Prometheus families.", "", "Prometheus text format"),
    ("GET", "/health", "open",
     "The proxy's health, for a container's check; the full body for the admin or the machine. Also /healthz and /v1/health.",
     "", "{status}"),
    ("*", "/v1/conductor/*", "admin",
     "The conductor in front of the backends, with --conductor-config; the engine-only screens answer 404.", "", ""),
    ("GET", "/v1/library", "open", "The package library, as on the engine, also under /v1/library/*.", "", ""),
    ("GET", "/v1/laya", "open", "Laya, when the decider role names an endpoint.", "", ""),
    ("POST", "/v1/laya/ask", "open", "As on the engine, relayed to the decider role's endpoint.", "", ""),
]

# The two small servers beside them.
SIDE_ROUTES = [
    ("tandem-dash hostcollect", "GET", "/host?watch=", "the machine's sample as JSON (the bearer token when --token is set)"),
    ("tandem-dash hostcollect", "GET", "/metrics", "the same sample in the Prometheus text format"),
    ("tandem-dash hostcollect", "GET", "/health", "{status: ok}, no token"),
    ("ops/engine_status.py", "GET", "/v1/models", "200 with the engine's last model list and its state, while the engine is away"),
    ("ops/engine_status.py", "GET", "/dashboard/", "503 with a page that says why the engine is away and opens the dashboard when it is back"),
    ("ops/engine_status.py", "*", "any other path", "503 provider_unavailable with Retry-After and the state as engine_<state>"),
]

# --------------------------------------------------------------------------- headers
# (name, direction, where, meaning). direction: in (the server reads it), out (it writes it),
# both. where: engine, proxy, both.
HEADERS = [
    ("X-Tandem", "both", "both",
     "raw, shadow or on: lowers the conductor's mode for one request; the response says which mode ran"),
    ("X-Tandem-Profile", "both", "both", "picks a conductor profile; the response names the profile that ran; an unknown name is a 400"),
    ("X-Tandem-Memory", "in", "engine", "off: no memory note is read or written; read: reads only, a block the model writes is hidden and dropped"),
    ("X-Tandem-Probe", "in", "engine", "off takes the request out of the probe snapshots"),
    ("X-Tandem-User", "in", "engine", "a proxy's user name for the memory notes, honoured from loopback or QWEN38_MEMORY_PROXIES"),
    ("X-Tandem-Relay-Id", "in", "engine", "a proxy's id for a relayed request, for the live counts; a relayed request without X-Tandem-User touches no memory"),
    ("X-Tandem-Parent", "in", "engine", "the request a call of TandemLLM to itself serves; the row becomes client kind internal"),
    ("X-Tandem-Worker", "in", "engine", "1 on an agent's worker request: its session comes back, so it is never a guest, and its rows go to the NVMe rather than away while it waits"),
    ("X-Tandem-Parent-Session", "in", "engine", "a worker's parent session (the chat's): for 600 s after the worker's last request the parent's rows go to the NVMe rather than away, and its return tries slot 0 first"),
    ("X-Tandem-Regenerate", "in", "engine", "1 serves the request as a regenerate (sampled, a fresh seed); 0 turns the detection off for it"),
    ("X-Tandem-Images-Left-Out", "out", "engine", "how many earlier images were left out of the prompt"),
    ("X-Tandem-Forward", "out", "engine", "to chatd on a /v1/chat-app request: the forward key, the only proof chatd accepts"),
    ("X-Tandem-Principal", "out", "engine", "to chatd: who the request is for (owner)"),
    ("X-Tandem-Password-Set", "out", "engine", "to chatd: 1 when an admin password is set"),
    ("X-Tandem-Signed-In", "out", "engine", "to chatd: 1 when the request passed the dashboard's rule"),
    ("X-Tandem-Wait", "out", "engine", "memory, on a response that waited for memory before its prefill; its stream says so every 15 s"),
    ("X-Tandem-Backend", "out", "proxy", "the id of the backend that answered a relayed request"),
    ("X-Tandem-Cost", "in", "proxy", "1 on a streamed relayed request: one SSE comment before data: [DONE] carries the request's cost (: tandem-cost {\"cost_micros\": N, \"backend\": id}); without it the stream is the backend's"),
    ("X-Engine-Stop", "out", "engine", "pattern-stop(<label>) on a response without a stream that a stop pattern ended"),
    ("x-session-id", "in", "both", "opencode's session id: the router's session and the caches' session key"),
    ("X-Conversation-Id", "in", "both", "a conversation id for clients that send no session field; read after the body's own ids"),
    ("X-OpenWebUI-Chat-Id", "in", "both", "Open WebUI's chat id (a connection header with {{CHAT_ID}}): the router's session for that chat"),
    ("X-Request-Id", "both", "proxy", "sent to a vllm backend as px-<hex>, which the plugin reports back for the join; a client's own id is kept"),
    ("X-QSE-Client", "in", "engine", "dashboard marks the dashboard's own requests (client kind dashboard)"),
    ("X-Real-IP", "in", "both", "set by a reverse proxy; a request that carries it is never trusted as local; the sign-in's rate limit counts per address"),
    ("X-Forwarded-For", "in", "both", "as X-Real-IP: a request that carries it is never trusted as local"),
    ("X-Forwarded-Proto", "in", "both", "https makes the session cookie Secure, and the notes' {tandem_url} https"),
    ("X-Forwarded-Host", "in", "engine", "the host the notes' {tandem_url} names, unless QWEN38_TANDEM_URL sets it"),
    ("X-Decider-Deadline-Ms", "in", "engine", "how long a decision request waits for the idle window (default 30,000, at most 120,000)"),
    ("X-Decider-Window", "in", "engine", "in_decode is refused: decisions run in the idle window only"),
    ("X-Accel-Buffering", "out", "both", "no, on every stream, so nginx sends each event at once"),
    ("X-Content-Type-Options", "out", "both", "nosniff on the dashboard's static files"),
    ("x-ratelimit-*", "out", "proxy", "a cloud backend's rate-limit headers, passed on to the client"),
    ("OpenAI-Beta", "both", "proxy", "a client's OpenAI-Beta header, passed on to an openai backend"),
    ("Authorization", "in", "both", "Bearer: the admin token, the metrics token, the collect token or a proxy user key, by route; the engine's OpenAI routes ignore it and file the request under a hash of it"),
    ("User-Agent", "in", "both", "the client kind in the ledger: open-webui, opencode, openai-sdk, curl, other (a Hermes-shaped body: hermes)"),
    ("Last-Event-ID", "in", "both", "resumes the log stream after that line"),
    ("Retry-After", "out", "both", "with a 503 (queue full, memory gate, draining, a full backend), a 429, or a decision model that is loading"),
    ("WWW-Authenticate", "out", "both", "Bearer, with a 401"),
    ("Set-Cookie", "out", "both", "qse_dash, the dashboard session: HttpOnly, SameSite=Strict, Secure behind https"),
    ("Access-Control-Allow-Origin", "out", "engine", "* on the library and Laya routes, so a page the model builds can call them; with Access-Control-Allow-Methods, -Headers and -Max-Age on the preflight"),
    ("Access-Control-Allow-Methods", "out", "engine", "on the Laya preflight"),
    ("Access-Control-Allow-Headers", "out", "engine", "on the Laya preflight"),
    ("Access-Control-Max-Age", "out", "engine", "on the Laya preflight"),
]

# --------------------------------------------------------------------------- errors
# (status, types, when, retry_after). `types` are the error.type (and error.code) values.
ERRORS = [
    ("400", ["invalid_request_error"],
     "a request the engine will not serve as asked: error.param names the field (messages, a role, max_tokens, tools, an image part, a prompt too long, a body that is not JSON or not an object)", False),
    ("400", ["bad_request"], "a dashboard or conductor query that does not parse (a level, a range, a number)", False),
    ("400", ["confirm_required"], "a setup reset without the confirmation its level asks for (true, or the typed word)", False),
    ("401", ["unauthorized"], "a guarded route without the token or the session it needs, once an admin password is set; a wrong collect token", False),
    ("401", ["invalid_api_key"], "the proxy, once a user key exists: no key, or a key it does not know (code invalid_api_key)", False),
    ("403", ["forbidden"], "a memory route outside a known session address", False),
    ("404", ["not_found"], "no such route, or a route whose part is off on this server (the conductor, the proxy, the decision models)", False),
    ("404", ["unsupported_route"], "the proxy: a route the backend's kind does not serve, such as embeddings on deepseek (code unsupported_route)", False),
    ("404", ["model_not_found"], "a model id no backend serves, or one two backends serve (the message names the prefixed forms) (code model_not_found)", False),
    ("409", ["busy", "not_loaded", "not_running", "closed", "stale", "role_off"],
     "a conductor action that needs a role or a decision model in another state, or a role the roles document switches off (role_off); a package proposal already closed, or whose base text changed since", False),
    ("422", ["invalid_request_error", "invalid_report"],
     "a value the schema refuses: a config field (error.field), a field an environment variable locks (error.variable), a Laya, library or memory call's fields, an eval report", False),
    ("422", ["schema", "secret", "cap", "user", "unknown_id", "unknown_package", "rule_from_tool", "no_session", "no_user",
             "refused_root", "no_proposals", "propose"],
     "a memory note or a package proposal refused, with the reason as the type: a field (schema), a secret, a cap, an unknown note or package", False),
    ("429", ["server_busy"], "a request without a stream that waited --queue-timeout for the engine", True),
    ("429", ["too_many"], "a fifth log or live stream at once; a sixth failed sign-in in a minute", True),
    ("429", ["rate_limited"], "the library, Laya or memory routes past their calls a minute", True),
    ("500", ["internal_error", "error"], "a bug: the message is the exception's; the request is in the log", False),
    ("501", [], "OPTIONS on any path but the Laya routes", False),
    ("502", ["connect_error", "upstream_error", "key_unreadable"], "the proxy could not reach the backend or read its stored key, after its retries", False),
    ("200", ["upstream_closed", "upstream_error"], "a stream that already began: the backend failed or closed early; the client gets an error event, then [DONE]", False),
    ("503", ["server_busy"], "the queue is full, the server is draining, or the memory gate or the admission by bytes cannot make room", True),
    ("503", ["memory_wait_timeout"], "a request without a stream that waited past --queue-timeout for memory to free (type server_busy, Retry-After: 30)", True),
    ("503", ["setup_mode"], "any route but the dashboard, /health, /v1/models and /v1/setup/* on a box in setup mode (Retry-After: 60)", True),
    ("503", ["max_inflight"], "every backend for the model is at the max_inflight you set (Retry-After: 1)", True),
    ("503", ["warming", "loading", "refused", "failed", "not_loaded", "busy_decoding"], "a decision request while the model loads, was refused memory, failed, or the engine kept decoding past the deadline", True),
    ("503", ["provider_unavailable"], "the status responder while the engine is stopped, starting or held (code engine_<state>)", True),
    ("504", ["upstream_timeout", "timeout"], "a backend that sent no first byte in time; a Laya call past its deadline", False),
    ("529", ["overloaded_error"], "a decision request the runtime could not take", False),
]

# --------------------------------------------------------------------------- the scan
HANDLERS = {
    "server/app.py": ("do_GET", "do_POST", "do_PUT", "do_PATCH", "do_DELETE", "do_OPTIONS",
                      "_dashboard_get", "_conductor_write", "_strip_session"),
    "proxy/app.py": ("do_GET", "do_POST", "do_PUT", "do_PATCH", "do_DELETE", "do_OPTIONS",
                     "_dashboard_get", "_strip_session"),
}
HEADER_DIRS = ("server", "proxy")
_HEADER = re.compile(r"^(?:[Xx]-[A-Za-z0-9-]+|Retry-After|WWW-Authenticate|Set-Cookie|Last-Event-ID"
                     r"|OpenAI-[A-Za-z-]+|Access-Control-[A-Za-z-]+)$")
# code that is not part of the served API: test doubles, and the kit the model copies into a project
_SKIP = ("fake", "laya_kit")


def _parse(root: str, rel: str):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            return ast.parse(fh.read())
    except (OSError, SyntaxError):
        return None


def handler_paths(root: str) -> dict:
    """{rel: set of the path literals the request handlers test}."""
    out = {}
    for rel, names in HANDLERS.items():
        tree = _parse(root, rel)
        found = set()
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    # what tests a path: a comparison (== or in), a startswith(), a dispatch dict's key
                    tested = []
                    for c in ast.walk(node):
                        if isinstance(c, ast.Compare):
                            tested += [c.left, *c.comparators]
                        elif isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "startswith":
                            tested += c.args
                        elif isinstance(c, ast.Dict):
                            tested += [k for k in c.keys if k is not None]
                    for t in tested:
                        for c in ast.walk(t):
                            if (isinstance(c, ast.Constant) and isinstance(c.value, str)
                                    and c.value.startswith("/") and " " not in c.value):
                                found.add(c.value.rstrip("/") or "/")
        out[rel] = found
    # the proxy's completion routes are a constant of proxy/service.py
    tree = _parse(root, "proxy/service.py")
    if tree is not None:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROXY_PATHS" for t in node.targets)
                    and isinstance(node.value, ast.Tuple)):
                out["proxy/app.py"] |= {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return out


def _py_files(root: str, dirs: tuple):
    for d in dirs:
        for dp, _, fs in os.walk(os.path.join(root, d)):
            if any(s in os.path.relpath(dp, root) for s in _SKIP):
                continue
            for f in sorted(fs):
                if f.endswith(".py") and not f.startswith(_SKIP):
                    yield os.path.relpath(os.path.join(dp, f), root)


def header_names(root: str) -> set:
    """Lower-cased header names the server and the proxy read or write."""
    out = set()
    for rel in _py_files(root, HEADER_DIRS):
        tree = _parse(root, rel)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and _HEADER.match(n.value):
                out.add(n.value.lower())
    return out


def error_types(root: str) -> set:
    """The error.type and error.code values the server and the proxy put in an error body."""
    out = set()
    for rel in _py_files(root, HEADER_DIRS):
        tree = _parse(root, rel)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(k, ast.Constant) and k.value == "error" and isinstance(v, ast.Dict):
                        for k2, v2 in zip(v.keys, v.values):
                            if (isinstance(k2, ast.Constant) and k2.value in ("type", "code")
                                    and isinstance(v2, ast.Constant) and isinstance(v2.value, str)):
                                out.add(v2.value)
            elif isinstance(n, ast.Call):
                fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                args = n.args
                if fn == "_error_body":
                    out |= {a.value for a in args[1:3] if isinstance(a, ast.Constant) and isinstance(a.value, str)}
                elif (fn == "Refused" and len(args) >= 2 and isinstance(args[0], ast.Constant)
                      and isinstance(args[0].value, str) and re.fullmatch(r"[a-z_]+", args[0].value)):
                    out.add(args[0].value)          # a memory note's or a proposal's reason (422)
                elif (fn in ("Refused", "ApiError") and len(args) >= 2
                      and isinstance(args[0], ast.Constant) and isinstance(args[0].value, int)
                      and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)):
                    out.add(args[1].value)
                elif fn == "_busy":
                    out.add("server_busy")
    return out


def _static(path: str) -> str:
    """A documented path up to its first placeholder or query: what the code must test."""
    p = re.split(r"[{<*?]", path, maxsplit=1)[0]
    return p.rstrip("/") or "/"


def _unsession(path: str) -> str:
    """/s/{token}/v1/memory... is the handler's /v1/memory... after the session prefix is cut."""
    return re.sub(r"^/s/\{token\}", "", path)


def _covers(doc_paths: list, lit: str) -> bool:
    for d in map(_unsession, doc_paths):
        base, st = d.split("?")[0], _static(d)
        if base == lit or base.startswith(lit + "/") or st == lit:
            return True
        if re.search(r"[{*]", base) and lit.startswith(st + "/"):
            return True                   # a placeholder or a wildcard row covers what is under it
    return False


ALIASES = {"/healthz": "/health", "/v1/health": "/health", "/dashboard": "/dashboard/"}


def problems(root: str) -> list:
    out = []
    lits = handler_paths(root)
    for rel, rows in (("server/app.py", ENGINE_ROUTES), ("proxy/app.py", PROXY_ROUTES)):
        docs = [r[1] for r in rows]
        found = lits.get(rel, set())
        for lit in sorted(found):
            if lit.startswith("/s/"):
                continue
            target = ALIASES.get(lit, lit)
            if not _covers(docs, target) and not _covers(docs, lit):
                out.append(f"undocumented route in {rel}: {lit}")
        for d in docs:
            s = _static(_unsession(d))
            if s == "/":
                ok = "/" in found
            else:
                ok = any(l == s or l.startswith(s + "/") or s.startswith(l + "/") or s.startswith(l)
                         for l in found if l != "/")
            if not ok:
                out.append(f"documented route not in {rel}: {d}")
    code_h = header_names(root)
    doc_h = {h[0].lower() for h in HEADERS}
    for h in sorted(code_h):
        if h not in doc_h and not any(d.endswith("*") and h.startswith(d[:-1]) for d in doc_h):
            out.append(f"undocumented header: {h}")
    for d in sorted(doc_h):
        if d in ("authorization", "user-agent"):
            continue                      # read through helpers, not as bare literals everywhere
        if d.endswith("*"):
            if not any(h.startswith(d[:-1]) for h in code_h):
                out.append(f"documented header not in the code: {d}")
        elif d not in code_h:
            out.append(f"documented header not in the code: {d}")
    code_e = error_types(root)
    doc_e = {t for row in ERRORS for t in row[1]}
    for t in sorted(code_e - doc_e):
        out.append(f"undocumented error type: {t}")
    for t in sorted(doc_e - code_e):
        if t in ("warming", "loading", "refused", "failed", "not_loaded", "busy_decoding",
                 "provider_unavailable", "overloaded_error", "invalid_api_key"):
            continue                      # built from a variable (the runtime's state, an f-string)
        out.append(f"documented error type not in the code: {t}")
    return out


# --------------------------------------------------------------------------- rendering
_REF = re.compile(r"\b(Section|Chapter|Table) ((?:sec:|ch:|tab:)?[a-z][a-z0-9:-]*[a-z0-9])")
_CODEISH = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[DONE\]|(?<![\w])--[a-z][a-z-]*|"
                      r"(?<![\w/])/[\w{}.<>/*?=&|-]*|\b[A-Za-z][\w.-]*\.(?:schema\.)?json\b|"
                      r"\b[a-z]+(?:_[a-z0-9]+)+(?:\.[a-z_]+)*\b|\b[A-Z]+(?:_[A-Z0-9]+)+\b|\bqse_\w+)")


def _ref(m: re.Match) -> str:
    kind, label = m.group(1), m.group(2)
    if ":" not in label:
        label = {"Section": "sec:", "Chapter": "ch:", "Table": "tab:"}[kind] + label
    return "\x00" + kind + "~\\ref{" + label + "}\x00"


def fmt(text: str, code, tex) -> str:
    """Prose with identifiers, paths, flags and {json} set in monospace, and Section x as a \\ref."""
    if not text:
        return ""
    parts = re.split("\x00", _REF.sub(_ref, text))
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)              # a \ref, already TeX
            continue
        for j, piece in enumerate(_CODEISH.split(part)):
            out.append(code(piece) if j % 2 else tex(piece))
    return "".join(out)


def _route_rows(rows, code, tex) -> list:
    out = []
    for method, path, who, what, body, answer in rows:
        cell = fmt(what, code, tex)
        if body:
            cell += r" \emph{Body:} " + fmt(body, code, tex) + "."
        if answer:
            cell += r" \emph{Answer:} " + fmt(answer, code, tex) + "."
        out.append(code(method) + r"\newline " + code(path) + " & " + tex(who) + " & " + cell + r" \\")
    return out


def gen_who_tex(header: str, code, tex) -> str:
    out = [header, r"\begin{reftabletwo}{Who may call a route: the labels of the routes' \hd{who} column}{tab:ref-http-who}{\hd{label} & \hd{who may call}}"]
    for k, v in WHO.items():
        out.append(code(k) + " & " + fmt(v, code, tex) + r" \\")
    out.append(r"\end{reftabletwo}")
    return "\n".join(out) + "\n"


def gen_engine_routes_tex(header: str, code, tex) -> str:
    out = [header, r"\begin{httptable}{The engine server's routes}{tab:ref-engine-routes}"]
    out += _route_rows(ENGINE_ROUTES, code, tex)
    out.append(r"\end{httptable}")
    return "\n".join(out) + "\n"


def gen_proxy_routes_tex(header: str, code, tex) -> str:
    out = [header, r"\begin{httptable}{The standalone proxy's routes (\texttt{tandem-dash})}{tab:ref-proxy-routes}"]
    out += _route_rows(PROXY_ROUTES, code, tex)
    out.append(r"\end{httptable}")
    out.append(r"\begin{reftable}{The two small servers beside them}{tab:ref-side-routes}{\hd{server} & \hd{route} & \hd{answer}}")
    for server, method, path, what in SIDE_ROUTES:
        out.append(code(server) + " & " + code(method) + " " + code(path) + " & " + fmt(what, code, tex) + r" \\")
    out.append(r"\end{reftable}")
    return "\n".join(out) + "\n"


def gen_headers_tex(header: str, code, tex) -> str:
    out = [header, r"\begin{reftable}{Every header the engine and the proxy read or write}{tab:ref-headers}{\hd{header} & \hd{direction, where} & \hd{meaning}}"]
    for name, direction, where, meaning in HEADERS:
        out.append(code(name) + " & " + tex(direction + ", " + where) + " & " + fmt(meaning, code, tex) + r" \\")
    out.append(r"\end{reftable}")
    return "\n".join(out) + "\n"


def gen_errors_tex(header: str, code, tex) -> str:
    out = [header, r"\begin{reftable}{Every error status and type}{tab:ref-errors}{\hd{status, Retry-After} & \hd{error.type or code} & \hd{when}}"]
    for status, types, when, retry in ERRORS:
        out.append(code(status) + (r"\newline with \texttt{Retry-After}" if retry else "") + " & "
                   + (", ".join(code(t) for t in types) if types else "none") + " & "
                   + fmt(when, code, tex) + r" \\")
    out.append(r"\end{reftable}")
    return "\n".join(out) + "\n"
