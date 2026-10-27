"""Typed decisions on the engine port (SRV-131): the Jev contract, and the runtime's wiring.

    POST /v1/systemone, /v1/predict   {state, model?, questions{id: {type, instructions,
                                       criteria?}}, task?, lang?} -> {model, answers, usage,
                                       routing, latency_ms, path}
    GET  /v1/models                   `decision_models` beside `data` (which stays the chat model)
    GET  /v1/models/{id}              a decision model's card too
    GET  /readyz                      200 once a decision model is loaded and warm, else 503

The request and response shapes are arbiter's (its server/app.py, the code source of this file),
which are TypeSafe's Jev API plus `routing`, `latency_ms`, `confidence` and `action`: a client
written against arbiter or Jev works against the engine by changing the base URL and the key.
The errors keep arbiter's body, `{"type": "error", "error": {"type", "message"}}`: 422
`invalid_request_error`, 529 `overloaded_error`, and 503 with `Retry-After` and the runtime's
reason as the type (`warming`, `loading`, `refused`, `failed`, `not_loaded`, `busy_decoding`).

Auth (the owner, 2026-10-01 13:00: on :8000 from stage A, LAN only, behind the engine's bearer
auth): the `decider` rule of server/auth.py. With no admin password set every caller is let in
(VIS-56, AU1); with one set, the admin token, the dashboard session or a tool on the box itself.

Stage A asks only in the idle window: a caller that arrives while a completion holds the engine
waits for it up to its deadline (`X-Decider-Deadline-Ms`, default 30 s, at most 120 s), then gets
503 `busy_decoding`. `X-Decider-Window: in_decode` is refused (stage C).

The rest of the file is the wiring `server/app.py` calls: `start()` builds the registry from
`QWEN38_DECIDER_*` (None, and nothing of engine/decision imported, when `QWEN38_DECIDER_MODELS` is
empty), the residency planner's check for every load, and the conductor's in-process client
(`endpoint: self` or `engine:<family>`, server/conductor/classify.py). `status()` and friends feed
`/v1/conductor/deciders/*` (server/conductor/routes.py) for the dashboard's Deciders screen.
"""

from __future__ import annotations

import fcntl
import os
import time

from engine.settings import SETTINGS as _S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QTYPES_ALLOWED = ("noul", "choice", "score")
MAX_CHOICES = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
MAX_QUESTIONS = 64
DEADLINE_MS, MAX_DEADLINE_MS = 30_000, 120_000
# a forward that started before the deadline still finishes; this is how long a caller waits
# for one past its deadline before it gives up on the answer
FORWARD_GRACE_S = 30.0
OFF_WHY = ("this engine serves no decision model: QWEN38_DECIDER_MODELS is empty (set it in "
           "ops/serve.env, e.g. laya-english,laya-multilingual,laya-typed-decisions, and restart)")


class ValidationError(ValueError):
    """A request that does not satisfy the Jev contract: 422."""


def error_body(kind: str, message: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


# ---------------------------------------------------------------------- the runtime's wiring
def configured() -> list[str]:
    return [n.strip() for n in (_S.get("DECIDER_MODELS") or "").split(",") if n.strip()]


def start(state: dict, gate, *, fake: bool = False, log=print):
    """The registry for this server, or None when the flag is off. Registers the conductor's
    in-process decider backends."""
    names = configured()
    if not names:
        return None
    from engine.decision import catalogue, routers
    from engine.decision.registry import Registry
    models = catalogue(names, _S.get("DECIDER_DIR"), allow_fake=fake)
    device = None
    if not fake:
        from engine.decision.device import Device
        device = Device(_S.get("DECIDER_DEVICE"))
    reg = Registry(models, gate=gate, planner=planner(state, fake=fake), device=device,
                   max_batch=int(_S.get("DECIDER_MAX_BATCH")),
                   wait_ms=float(_S.get("DECIDER_WAIT_MS")),
                   max_queue=int(_S.get("DECIDER_MAX_QUEUE")),
                   collate=_S.get("DECIDER_COLLATE"),
                   policy=lambda: load_policy(state), routers=routers(models),
                   log=lambda s: log(s, flush=True), contract=_contract(state))
    from server.conductor import classify

    def factory(model, timeout_ms):
        return classify.InProcess(reg, model, timeout_ms)

    classify.register_backend("self", factory)
    for fam in reg.families:
        classify.register_backend(fam, factory)
    log(f"[deciders] serving {', '.join(reg.order)} on {device or 'no device (fake)'}; idle window "
        f"only (stage A); loads through the residency planner; /v1/systemone on this port",
        flush=True)
    return reg


def _contract(state: dict):
    """The role contract the registry checks each model against: the conductor's (SRV-133),
    with the engine's tokenizer hash for a model that would share token ids."""
    from server.conductor import schema

    def check(fields: dict, name: str) -> list[str]:
        return schema.role_errors(fields, name, state.get("tokenizer_sha"))

    return check


def _deciders_doc(state: dict) -> dict:
    from server.conductor import schema
    cond = state.get("conductor")
    if cond is not None:
        return cond.store.peek("deciders")[1]
    return schema.seed("deciders")


def load_policy(state: dict) -> dict[str, str]:
    """model -> on_start | on_demand | off, from the conductor's `deciders` document."""
    return {m["name"]: m["load"] for m in _deciders_doc(state)["models"]}


def held(*, service_start: bool = False, hold_restart: bool = False, repo: str = ROOT,
         lock_path: str | None = None) -> bool:
    """A hold owns the board, read as ops/roles.sh reads it: the hold's pause file in the repo,
    or the box lock held by someone -- except for the hold's own restart, and the box lock except
    at the service's own start (the watchdog restarts holding the lock itself)."""
    if hold_restart:
        return False
    if os.path.exists(os.path.join(repo, ".watchdog.off")):
        return True
    if service_start:
        return False
    path = os.path.expanduser(lock_path or os.environ.get("BOX_LOCK") or "~/.qwen38-box.flock")
    if not os.path.exists(path):
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _mem_available_gib(fake: bool) -> float | None:
    if fake:
        # the fake engine is a test run: its MemAvailable is what the test says (default: room)
        return float(os.environ.get("FAKE_MEM_AVAILABLE_GIB", "64"))
    from server.conductor import residency
    return residency._mem_gib()


def planner(state: dict, *, fake: bool = False, role: str = "decider"):
    """The registry's planner: OPS-41's `check(role, ...)` on the residency document (the role's
    claim, floor and margin), the hold read above, a load in progress (the registry's own, or
    the other in-process role's: the decision models and the narrator's slot never load at
    once); every check journaled in `<conductor root>/residency.jsonl`. `role`: `decider` for the
    registry, `narrator` for the small-model slot (P1)."""

    def check(name, *, loading, service_start, hold_restart):
        from server.conductor import residency, schema
        cond = state.get("conductor")
        doc = cond.store.peek("residency")[1] if cond is not None else schema.seed("residency")
        roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
        hold = held(service_start=service_start, hold_restart=hold_restart)
        from server import memlimits
        peak, now = memlimits.engine_gib(state)
        d = residency.check(role, _mem_available_gib(fake), doc, roles, hold=hold,
                            loading=loading or _other_loading(state, role),
                            engine_peak_gib=peak, engine_now_gib=now)
        if cond is not None:
            residency.Journal(cond.root).write(dict(d, kind="check", source=f"engine:{name}"))
        return d

    return check


def _other_loading(state: dict, role: str) -> bool:
    """Whether the other in-process role is loading a model right now."""
    if role != "decider":
        reg = state.get("deciders")
        return bool(reg is not None and reg.loading())
    n = state.get("narrator")
    return bool(n is not None and n.slot.state == "loading")


# ---------------------------------------------------------------------- the Jev contract
def parse(body) -> dict:
    """The request, validated as arbiter validates it, before any model runs."""
    if not isinstance(body, dict):
        raise ValidationError("request body must be a JSON object")
    state = body.get("state")
    if not isinstance(state, (str, dict, list)):
        raise ValidationError("state must be a string, an object or an array")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ValidationError("questions must contain at least one question")
    if len(qs) > MAX_QUESTIONS:
        raise ValidationError(f"at most {MAX_QUESTIONS} questions in one request, got {len(qs)}")
    for k in ("model", "task", "lang"):
        if body.get(k) is not None and not isinstance(body[k], str):
            raise ValidationError(f"{k} must be a string")
    out = {}
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise ValidationError(f"question {qid!r} must be an object")
        t, ins, crit = q.get("type"), q.get("instructions"), q.get("criteria")
        if t not in QTYPES_ALLOWED:
            raise ValidationError(f"question {qid!r} has unknown type {t!r}; expected one of "
                                  f"{list(QTYPES_ALLOWED)}")
        if ins is None or (isinstance(ins, str) and not ins.strip()) or \
                not isinstance(ins, (str, dict, list)):
            raise ValidationError(f"question {qid!r} has empty instructions")
        if crit is not None and not isinstance(crit, (dict, list)):
            raise ValidationError(f"question {qid!r}: criteria must be an object or an array")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise ValidationError(f"question {qid!r}: noul criteria must be an object with "
                                      f"'true' and/or 'false' keys")
            if isinstance(crit, dict):
                extra = sorted(set(crit) - {"true", "false"})
                if extra:
                    raise ValidationError(f"question {qid!r}: noul criteria may only contain "
                                          f"'true' and 'false'; found {extra}")
        elif t == "choice":
            if crit is None:
                raise ValidationError(f"question {qid!r}: choice requires criteria")
            if not 2 <= len(crit) <= MAX_CHOICES:
                raise ValidationError(f"question {qid!r}: choice needs 2 to {MAX_CHOICES} options, "
                                      f"got {len(crit)}")
        else:
            if not isinstance(crit, list):
                raise ValidationError(f"question {qid!r}: score criteria must be an array of "
                                      f"level descriptions")
            if not MIN_SCORE_LEVELS <= len(crit) <= MAX_SCORE_LEVELS:
                raise ValidationError(f"question {qid!r}: score needs between {MIN_SCORE_LEVELS} "
                                      f"and {MAX_SCORE_LEVELS} levels, got {len(crit)}")
        out[qid] = {"type": t, "instructions": ins, "criteria": crit}
    return {"state": state, "questions": out, "model": body.get("model"),
            "task": body.get("task"), "lang": body.get("lang")}


def ask(reg, body, *, deadline_ms: float = DEADLINE_MS, window: str = "idle",
        source: str = "http") -> tuple[int, dict, int | None]:
    """(status, body, Retry-After seconds or None) for one systemone request."""
    from engine.decision.base import DeadlineMissed, DecisionError
    from engine.decision.registry import UnknownModel
    t0 = time.perf_counter()
    try:
        req = parse(body)
        fut = reg.ask(req["state"], req["questions"], model=req["model"], window=window,
                      deadline_ms=deadline_ms, task=req["task"], lang=req["lang"], source=source)
        out = fut.result(timeout=deadline_ms / 1000.0 + FORWARD_GRACE_S)
    except (ValidationError, UnknownModel) as exc:
        return 422, error_body("invalid_request_error", str(exc)), None
    except DeadlineMissed as exc:
        return 503, error_body("busy_decoding", f"{exc}; ask again when the engine is idle, or "
                                                f"send a longer X-Decider-Deadline-Ms"), exc.retry
    except DecisionError as exc:
        kind = {422: "invalid_request_error", 529: "overloaded_error"}.get(exc.status, exc.why)
        body = error_body(kind, str(exc))
        if exc.detail.get("decision"):
            body["error"]["decision"] = exc.detail["decision"]
        return exc.status, body, exc.retry
    except TimeoutError:
        return 504, error_body("timeout", "the forward did not finish in time"), 5
    except ValueError as exc:                        # a router's own refusal (an unknown task)
        return 422, error_body("invalid_request_error", str(exc)), None
    return 200, {"model": out["model"], "answers": out["answers"], "usage": out["usage"],
                 "routing": out["routing"], "path": out["path"],
                 "latency_ms": round((time.perf_counter() - t0) * 1e3, 2)}, None


def deadline_of(headers) -> float:
    """`X-Decider-Deadline-Ms`, clamped to (0, 120 s]; the default 30 s."""
    try:
        v = float(headers.get("X-Decider-Deadline-Ms") or DEADLINE_MS)
    except (TypeError, ValueError):
        raise ValidationError("X-Decider-Deadline-Ms is a number of milliseconds")
    return min(max(v, 1.0), float(MAX_DEADLINE_MS))


def handle(handler, reg, body) -> None:
    """`POST /v1/systemone` and `/v1/predict` on server/app.py's Handler."""
    if not handler._allowed("decider"):
        return
    try:
        deadline = deadline_of(handler.headers)
    except ValidationError as exc:
        return handler._json(422, error_body("invalid_request_error", str(exc)))
    window = (handler.headers.get("X-Decider-Window") or "idle").strip().lower()
    status, out, retry = ask(reg, body, deadline_ms=deadline, window=window)
    handler._json(status, out, extra_headers=(("Retry-After", str(retry)),) if retry else ())


# ---------------------------------------------------------------------- listings
def decision_models(reg, created: int) -> list[dict]:
    """`/v1/models` `decision_models`: every served model, then `jev-latest` (the router) with
    the tightest limits of the loaded set."""
    out = [_card(s, created) for s in reg.status()]
    out.append({"id": "jev-latest", "object": "decision_model", "created": created,
                "owned_by": "tandemllm", "aliases": ["auto", "jev", "default", *reg.families],
                "contract": "systemone", "loaded": reg.ready(),
                "state": "loaded" if reg.ready() else "not_loaded",
                "description": "automatic routing across the loaded decision models",
                "limits": reg.limits("auto")})
    return out


def _card(s: dict, created: int) -> dict:
    return {"id": s["name"], "object": "decision_model", "created": created,
            "owned_by": s["owned_by"], "family": s["family"], "aliases": s["aliases"],
            "contract": s["contract"], "limits": s["limits"], "loaded": s["state"] == "loaded",
            "state": s["state"]}


def card(reg, name: str, created: int) -> dict | None:
    """`/v1/models/{id}` for a decision model or `jev-latest`; None when it is neither."""
    for c in decision_models(reg, created):
        if c["id"] == name or name.lower() in [a.lower() for a in c["aliases"]]:
            return c
    return None


def readyz(reg, state: dict) -> tuple[int, dict]:
    ready = reg.ready()
    held = bool(getattr(reg, "memory_hold", None))
    st = "ready" if ready else ("warming" if reg.loading() else
                                "unloaded_for_memory" if held else "not_loaded")
    # out for memory (server/rolemem.py) is a planned state, not a broken one: the router falls
    # back to its next stage and the governor loads the model again once memory came back
    return (200 if ready or held else 503), {
        "status": st, "engine": "tandemllm", "models": reg.loaded(),
        "device": str(reg.device.device) if reg.device is not None else None,
        "version": state.get("version", ""),
        "deciders": {"ready": ready, "models": {s["name"]: s["state"] for s in reg.status()}}}


# ---------------------------------------------------------------------- the Deciders screen
def status(reg, state: dict) -> dict:
    """`GET /v1/conductor/deciders/status`: what the engine serves, each model's state and
    numbers, the planner's answer for a load now, and the model behind each question id."""
    from server.conductor import residency, schema
    cond = state.get("conductor")
    doc = _deciders_doc(state)
    if reg is None:
        return {"enabled": False, "why": OFF_WHY, "models": [], "planner": None,
                "last_refusal": None, "questions": [], "families": [], "device": None,
                "window": "idle", "ready": False}
    rdoc = cond.store.peek("residency")[1] if cond is not None else schema.seed("residency")
    roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
    from server import memlimits
    peak, now = memlimits.engine_gib(state)
    plan = residency.check("decider", _mem_available_gib(reg.device is None), rdoc, roles,
                           hold=held(), loading=reg._loading is not None,
                           engine_peak_gib=peak, engine_now_gib=now)
    last = None
    if cond is not None:
        last = next((r for r in residency.Journal(cond.root).tail(200)
                     if r.get("kind") == "check" and not r.get("allowed")
                     and str(r.get("source", "")).startswith("engine:")), None)
    picks = reg.picks_by_source()
    questions = [{"id": qid, "model": m, "picks": picks.get(qid, {})}
                 for qid, m in sorted(doc["questions"].items())]
    from server import laya_api
    return {"enabled": True, "why": None, "models": reg.status(), "planner": plan,
            "last_refusal": last, "questions": questions, "families": reg.families,
            "device": str(reg.device.device) if reg.device is not None else None,
            "window": "idle", "ready": reg.ready(),
            "unserved": [m["name"] for m in doc["models"] if m["name"] not in reg.models],
            # L1: Laya for the model's scripts (`/v1/laya/ask`), its settings and counts
            "api": {**schema.decider_api(doc), "stats": laya_api.stats()}}
