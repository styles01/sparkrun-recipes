"""`/v1/config/*`: the Backends and Setup screens' API over the config database.

Reads follow the dashboard's rule (open with the login off, the admin token or session with it on).
Writes follow auth rule `conductor_write` (server/auth.py): with no admin password set, every caller
that reaches the port may change the configuration (VIS-56, AU1); with one set, the admin bearer
token, a dashboard session or a tool on the machine itself. No answer ever holds a key: a backend shows `key_set`, the last 4
characters and a 6-hex-digit hash prefix. A user's key is in the answer to its creation, once.

    GET    /v1/config                                 everything the screens show, in one answer
    POST   /v1/config/test                            {kind, url, key?}: test a backend not saved
    POST   /v1/config/backends                        {id, kind, url, key?, ...}: add
    GET    /v1/config/backends/{id}                   one, with its models
    PATCH  /v1/config/backends/{id}                   change fields (`key: null` removes the key)
    DELETE /v1/config/backends/{id}
    POST   /v1/config/backends/{id}/test              test the saved backend, refresh its models
    PUT    /v1/config/backends/{id}/models/{model}    public id, enabled, price, max_output
                                                      (null clears), fallback, features
                                                      (creates a manual model)
    DELETE /v1/config/backends/{id}/models/{model}
    PUT    /v1/config/rules                           {rules: [{id?, pattern, backend, enabled}]}
    POST   /v1/config/users                           {label}: the answer holds the key, once
    PATCH  /v1/config/users/{id}                      {enabled}
    DELETE /v1/config/users/{id}
    GET    /v1/config/prices/seed                     the public list prices shipped with this
                                                      release, applied only by a click
    GET    /v1/config/audit                           the last 50 changes (field names only)

A field the environment sets answers 422 naming the variable (`error.variable`); a value the schema
refuses answers 422 naming the field (`error.field`).
"""

from __future__ import annotations

import json
import os
from urllib.parse import unquote

from proxy import config as config_mod
from proxy import health as health_mod

CONTRACT = "1.0"
SEED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.seed.json")


def _who(handler, svc) -> str:
    a = svc.auth
    try:
        if a is not None and a.admin_any(handler.headers):
            return "admin"
    except Exception:                                              # noqa: BLE001
        pass
    return "local"


def backend_view(svc, b: dict, with_models: bool = False, cost: dict | None = None) -> dict:
    out = dict(b)
    out["cost_today_micros"] = (svc.cost_today() if cost is None else cost).get(b["id"])
    out["key"] = svc.cfg.key_hint(b["id"])
    out["locks"] = svc.cfg.backend_locks(b["id"])
    out["health"] = svc.health.view(b["id"])
    out["in_flight"] = svc.relay.inflight.count(b["id"])
    out["capabilities"] = svc.adapter(b).capabilities()
    out["routes"] = list(svc.adapter(b).routes())
    out["engine"] = svc.engines.view(b["id"], b["kind"])
    from proxy import harness as harness_mod
    out["degradations"] = harness_mod.degradations(b, out["capabilities"])
    from proxy import memwriter
    wcfg = None
    hx = getattr(svc, "harness", None)
    if hx is not None and hx.writer is not None:
        try:
            wcfg = hx.writer.cfg()
        except Exception:                                          # noqa: BLE001
            wcfg = None
    if with_models:
        from proxy import route as route_mod
        cls = route_mod.KINDS.get(b["kind"], route_mod.Generic)
        rows = svc.cfg.models(b["id"])
        costs = {}
        if getattr(svc, "harness", None) is not None:
            import time as _time
            costs = svc.harness.feature_costs(b["id"], int((_time.time() - 86400) * 1000))
        for m in rows:
            # P4: what each switch resolves to for this model, and what the harness's extra
            # requests cost it over the last 24 hours (from the ledger and the journal)
            m["effective"] = harness_mod.switches(b, m)
            m["harness_cost_day"] = costs.get(m["model"], {})
            # VIS-227: what the card carries while the operator's output limit is empty
            m["max_output_default"] = route_mod.default_max_output(cls, m)
            # M3: the background memory writer's mode for this model and why
            if wcfg is not None:
                wm, why = memwriter.resolve(b, svc.price_of(b["id"], m["model"]), wcfg["mode"])
                m["writer"] = {"mode": wm, "why": why}
        out["model_rows"] = rows
    h = getattr(svc, "harness", None)
    if h is not None and h.writer is not None:
        out["writer"] = {"config": memwriter.backend_cfg(b), "today": h.writer.today(b["id"]),
                         # 0.3.36: what the Backends screen shows beside the switches: the
                         # settings document's mode (the effective mode is the lower of the two)
                         # and where a shadow run's would-be blocks go
                         "settings_mode": (wcfg or {}).get("mode"),
                         "shadow_log": memwriter.shadow_log_hint(h.writer.memory)}
    return out


def overview(svc) -> dict:
    cfg = svc.cfg
    cost = svc.cost_today()
    backends = [backend_view(svc, b, with_models=True, cost=cost) for b in cfg.backends()]
    return {"contract_version": CONTRACT,
            "setup_needed": cfg.empty(),
            "key_source": cfg.key_source(),
            "config_path": "~/" + os.path.relpath(cfg.path, os.path.expanduser("~"))
            if cfg.path.startswith(os.path.expanduser("~")) else cfg.path,
            "kinds": {k: {"cloud": v["cloud"], "url": v["url"],
                          "inject_usage": v["inject_usage"],
                          "timeouts": config_mod.TIMEOUTS[v["cloud"]]}
                      for k, v in config_mod.KINDS.items()},
            "feature_groups": list(config_mod.FEATURE_GROUPS),
            # P4: whether a conductor runs the harness here, and the defaults by kind
            "harness": {"conductor": getattr(svc, "harness", None) is not None,
                        "profiles": _profiles(svc),
                        "defaults": {"local": _harness_defaults(False),
                                     "cloud": _harness_defaults(True)}},
            "backends": backends,
            "rules": cfg.rules(),
            "users": cfg.users(),
            "users_active": cfg.users_active(),
            "engine": svc.engine_info() if hasattr(svc, "engine_info") else None,
            "generation": cfg.generation}


def _profiles(svc) -> list:
    h = getattr(svc, "harness", None)
    if h is None or h.cond is None:
        return []
    try:
        return list(h.cond.store.names())
    except Exception:                                              # noqa: BLE001
        return []


def _harness_defaults(cloud: bool) -> dict:
    from proxy import harness as harness_mod
    return dict(harness_mod.DEFAULTS[cloud])


def _send(handler, code: int, payload) -> None:
    handler._json(code, payload, extra_headers=(("Cache-Control", "no-store"),))


def handle(handler, svc, method: str, path: str, body) -> None:
    """Route one /v1/config request on a handler with `_json` and `_allowed`."""
    write = method != "GET"
    if not handler._allowed("conductor_write" if write else "dashboard"):
        return
    if write and body is not None and not isinstance(body, (dict, list)):
        return _send(handler, 400, {"error": {"type": "bad_request",
                                              "message": "the body must be a JSON object"}})
    who = _who(handler, svc)
    parts = [unquote(p) for p in path[len("/v1/config"):].split("/") if p]
    cfg = svc.cfg
    try:
        if not parts:
            if method == "GET":
                return _send(handler, 200, overview(svc))
        elif parts == ["test"] and method == "POST":
            return _send(handler, 200, _test_unsaved(svc, body or {}))
        elif parts == ["audit"] and method == "GET":
            return _send(handler, 200, {"audit": cfg.audit()})
        elif parts == ["prices", "seed"] and method == "GET":
            with open(SEED) as f:
                return _send(handler, 200, json.load(f))
        elif parts[0] == "backends":
            return _backends(handler, svc, method, parts[1:], body or {}, who)
        elif parts == ["rules"]:
            if method == "GET":
                return _send(handler, 200, {"rules": cfg.rules()})
            if method == "PUT":
                rules = (body or {}).get("rules")
                if not isinstance(rules, list):
                    raise config_mod.ConfigError("rules must be a list", field="rules")
                return _send(handler, 200, {"rules": cfg.set_rules(rules, who)})
        elif parts[0] == "users":
            return _users(handler, svc, method, parts[1:], body or {}, who)
        return _send(handler, 404, {"error": {"type": "not_found",
                                              "message": f"no route {method} {path}"}})
    except config_mod.ConfigError as exc:
        return _send(handler, exc.status, exc.body())


def _backends(handler, svc, method, parts, body, who):
    cfg = svc.cfg
    if not parts:
        if method == "GET":
            return _send(handler, 200, {"backends": [backend_view(svc, b)
                                                     for b in cfg.backends()]})
        if method == "POST":
            if body.get("id") in ("engine",):
                raise config_mod.ConfigError("the id 'engine' is the engine's own", field="id")
            b = cfg.create_backend(body, who)
            svc.health.kick(b["id"])
            svc.engines.kick(b["id"])
            r = svc.health.check(b)
            out = backend_view(svc, cfg.backend(b["id"]), with_models=True)
            out["test"] = {"state": r["state"], "latency_ms": r["latency_ms"],
                           "error": r["last_error"]}
            return _send(handler, 201, out)
    bid = parts[0]
    b = cfg.backend(bid)
    if b is None:
        raise config_mod.ConfigError(f"no backend '{bid}'", status=404)
    rest = parts[1:]
    if not rest:
        if method == "GET":
            return _send(handler, 200, backend_view(svc, b, with_models=True))
        if method == "PATCH":
            nb = cfg.update_backend(bid, body, who)
            svc.relay.drop_pool(bid)
            svc.health.kick(bid)
            svc.engines.kick(bid)
            return _send(handler, 200, backend_view(svc, nb, with_models=True))
        if method == "DELETE":
            cfg.delete_backend(bid, who)
            svc.relay.drop_pool(bid)
            handler._send_empty(204)
            return
    if rest == ["test"] and method == "POST":
        r = svc.health.check(b)
        models = cfg.models(bid)
        return _send(handler, 200, {"ok": r["state"] == "up", "latency_ms": r["latency_ms"],
                                    "error": r["last_error"],
                                    "models": [m["model"] for m in models if m["present"]]})
    if rest and rest[0] == "models" and len(rest) >= 2:
        mid = "/".join(rest[1:])
        if method == "PUT":
            return _send(handler, 200, cfg.update_model(bid, mid, body, who, create=True))
        if method == "DELETE":
            cfg.delete_model(bid, mid, who)
            handler._send_empty(204)
            return
    if rest == ["models"] and method == "GET":
        return _send(handler, 200, {"models": cfg.models(bid)})
    return _send(handler, 404, {"error": {"type": "not_found", "message": "no such route"}})


def _users(handler, svc, method, parts, body, who):
    cfg = svc.cfg
    if not parts:
        if method == "GET":
            return _send(handler, 200, {"users": cfg.users(), "active": cfg.users_active()})
        if method == "POST":
            user, key = cfg.create_user(body, who)
            return _send(handler, 201, {"user": user, "key": key,
                                        "note": "the key is shown this once; it is stored as "
                                                "its sha256 only"})
    try:
        uid = int(parts[0])
    except ValueError:
        raise config_mod.ConfigError(f"no user '{parts[0]}'", status=404)
    if method == "PATCH":
        return _send(handler, 200, cfg.update_user(uid, body, who))
    if method == "DELETE":
        cfg.delete_user(uid, who)
        handler._send_empty(204)
        return
    return _send(handler, 404, {"error": {"type": "not_found", "message": "no such route"}})


def _test_unsaved(svc, body: dict) -> dict:
    """Test connection before saving: the probe with the given kind, URL and key."""
    kind = body.get("kind") or "generic"
    if kind not in config_mod.KINDS:
        raise config_mod.ConfigError(f"kind must be one of {', '.join(config_mod.KINDS)}",
                                     field="kind")
    url = body.get("url") or config_mod.KINDS[kind]["url"]
    doc = {"id": body.get("id") or "test", "kind": kind, "url": url or ""}
    errs = [e for e in config_mod.schema.validate(doc, "backend") if e[0] != "id"]
    if errs:
        raise config_mod.ConfigError(f"{errs[0][0]} {errs[0][1]}", field=errs[0][0])
    key = body.get("key")
    if not key and body.get("id"):
        try:
            key = svc.cfg.key_for(body["id"])
        except Exception:                                          # noqa: BLE001
            key = None
    cloud = config_mod.KINDS[kind]["cloud"]
    b = {"id": doc["id"], "kind": kind, "url": config_mod.norm_url(url),
         "timeouts": dict(config_mod.TIMEOUTS[cloud]), "inject_usage": True}
    r = health_mod.probe(b, key, log=svc.log)
    return {"ok": r["ok"], "status": r["status"], "latency_ms": r["latency_ms"],
            "error": r["error"], "models": [m["id"] for m in (r["models"] or [])]}
