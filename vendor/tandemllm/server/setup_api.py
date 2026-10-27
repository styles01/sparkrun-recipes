"""`/v1/setup/*`: the setup guide's routes (docs/setup.md, contract 1.0; the shapes are
dashboard/src/api/types.ts `Setup*`).

    GET  /v1/setup/models                 a tile per models/<id>/model.yaml, and the files that did not load
    GET  /v1/setup/models/{id}            the model.yaml as JSON, with `wizard_fields` (the step schema)
    GET  /v1/setup/models/{id}/steps      the step schema and the tool definitions Wren is given
    GET  /v1/setup/models/{id}/current    what the box runs now, as a selection (the wizard's seed)
    GET  /v1/setup/device                 the box, read only (memory, GPU, disk, profiles on disk)
    GET  /v1/setup/weights                the weights on the box with their sizes, the running one marked,
                                          the shared assets apart (SRV-560)
    POST /v1/setup/remove {profiles, confirm, dry_run?}   those profiles' weights removed, never the
                                          running one's; run/weights-removed.txt says how to get them back
    GET  /v1/setup/status                 the engine, the running (or newest) setup job and its step n of
                                          N; the status responder answers it while the engine
                                          restarts, with the load's progress (OPS-175)
    GET  /v1/setup/community              the community packs of every model and the sync state
    POST /v1/setup/community/sync {}      fetch the community index now (the page opened, Refresh;
                                          server/setup/community.py), then the same answer
    POST /v1/setup/estimate {selection}   the memory estimate (server/setup/estimate.py)
    POST /v1/setup/plan {selection}       the review step's facts (server/setup/apply.py)
    POST /v1/setup/apply {selection, dry_run?}   202 and a job (dry_run: the plan); 409 while
                                          requests are in flight or a job runs, 409/412 when the
                                          plan has errors (over_budget, missing_token, ...)
    POST /v1/setup/jobs/{id}/retry {}     a failed or rolled back job again, from its selection: 202
                                          and the new job (`retry_of`), the finished download kept
    GET  /v1/setup/jobs/{id}/diagnostics  the job as text for a bug report (versions, steps, tandem
                                          doctor, the log's tail), nothing secret (VIS-213)
    GET  /v1/setup/jobs/{id}              SSE: the log lines on disk, the job as it is now, then its
                                          events as they come (`id:` the sequence, Last-Event-ID
                                          resumes); ?format=json the job
    GET  /v1/setup/reset                  the levels 2 to 5, what each removes and how big it is
    POST /v1/setup/reset {level, confirm, dry_run?}   one level (server/setup/reset.py); level 2
                                          starts a job; 409 while requests are in flight or a job runs
    GET  /v1/setup/assistant/status       Wren (server/setup/wren.py); a poll keeps it loaded
    POST /v1/setup/assistant/load         idempotent; a memory refusal is state `refused`
    POST /v1/setup/assistant/unload       the memory back
    POST /v1/setup/assistant/chat         {messages, page} -> SSE of SetupChatEvent (`data: {...}`);
                                          /v1/setup/chat is the same route

Every route is an admin route (server/auth.py, kind `setup`): open while no admin password is set,
the admin token, a dashboard session or a tool on the box itself once one is. Every POST also must
be a same-origin JSON request (`same_origin_json`): a page on another site can send a text/plain
POST without a preflight, and an open box or a tunnel's loopback needs no cookie to be abused. Nothing here runs on a
chat request's path: server/app.py imports this module the first time a /v1/setup route is called.
"""

from __future__ import annotations

import os
import threading
from urllib.parse import unquote

from server.setup import device as device_mod
from server.setup import estimate as estimate_mod
from server.setup import registry, tools

CONTRACT = "1.0"
NO_STORE = (("Cache-Control", "no-store"),)
# the largest POST body a setup route takes (Wren's chat with its page state is the largest)
MAX_BODY = 512 * 1024
# one setup job at a time: the check for a running job and the new job's folder under one lock
_JOB_LOCK = threading.Lock()


def _err(h, status: int, code: str, message: str, **extra) -> None:
    h._json(status, {"error": dict({"type": code, "code": code, "message": message}, **extra)},
            extra_headers=NO_STORE)


def busy(app) -> bool:
    """Requests are in flight or waiting (what apply and reset refuse to run beside)."""
    inf = app.INFLIGHT
    return bool(inf.get("running") or inf.get("waiting") or inf.get("lane"))


def wren_of(app):
    from server.setup import wren
    return wren.get(app, device_facts=lambda: device_facts(app),
                    setup_mode=bool(app.STATE.get("setup_mode")))


def assistant_status(app) -> dict:
    return wren_of(app).status()


def device_facts(app) -> dict:
    """What Wren reads of the box each turn: the device answer without Wren's own block."""
    d = device_mod.read(app.STATE, inflight=app.INFLIGHT, busy=busy(app), assistant={},
                        gpu=(app.STATE.get("gpu_sampler").sample()
                             if app.STATE.get("gpu_sampler") is not None else None))
    return {k: d.get(k) for k in ("mode", "preset", "mem_total_gb", "mem_available_gb", "gpu",
                                  "disk", "active_profile", "profiles_on_disk", "hf_token_present",
                                  "password_set")}


def device(app) -> dict:
    sampler = app.STATE.get("gpu_sampler")
    if sampler is None:
        from server import dashboard_api
        sampler = app.STATE.setdefault("gpu_sampler", dashboard_api.GpuSampler())
    return device_mod.read(app.STATE, inflight=app.INFLIGHT, busy=busy(app),
                           assistant=assistant_status(app), gpu=sampler.sample())


def _model(h, mid: str) -> dict | None:
    m = device_mod.windowed(registry.get(mid))
    if m is None:
        _err(h, 404, "unknown_model", f"no model {mid!r} in models/")
    return m


def same_origin_json(headers) -> tuple[int, str, str] | None:
    """None for a POST the page itself sent, else (status, code, message): the body must be
    declared JSON (a form or text/plain POST needs no CORS preflight), a browser's
    `Sec-Fetch-Site: cross-site` is refused, and an `Origin` must name this server's host (the
    `Host` the browser sent, which nginx passes on as $host, or `X-Forwarded-Host`). A tool
    without a browser sends no Origin and no Sec-Fetch-Site and passes on its JSON alone."""
    from urllib.parse import urlsplit
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        return 415, "json_only", "the setup routes take a JSON body (Content-Type: application/json)"
    if (headers.get("Sec-Fetch-Site") or "").strip().lower() == "cross-site":
        return 403, "cross_site", "a request from another site cannot change the setup"
    origin = headers.get("Origin")
    if origin is not None:
        o = urlsplit(origin.strip()) if origin.strip() != "null" else None
        hosts = {x.strip().lower() for x in (headers.get("Host"), headers.get("X-Forwarded-Host"))
                 if x}
        if o is None or not o.netloc or o.netloc.lower() not in hosts:
            return 403, "cross_origin", "a request from another origin cannot change the setup"
    return None


def handle(h, app, method: str, path: str, body) -> None:
    if not h._allowed("setup"):
        return
    if method == "POST":
        bad = same_origin_json(h.headers)
        if bad is not None:
            return _err(h, *bad)
        try:
            n = int(h.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY:
            return _err(h, 413, "too_large", f"a setup request body is at most "
                                              f"{MAX_BODY // 1024} KiB")
    sub = path[len("/v1/setup"):].strip("/")
    parts = [unquote(x) for x in sub.split("/")] if sub else []
    try:
        if method == "GET":
            return _get(h, app, parts)
        return _post(h, app, parts, body)
    except estimate_mod.SelectionError as exc:
        return h._json(exc.status, exc.body(), extra_headers=NO_STORE)


def _get(h, app, parts: list[str]) -> None:
    if parts == ["models"]:
        models, errors = registry.load_all()
        return h._json(200, {"contract_version": CONTRACT,
                             "models": [dict(registry.tile(m), community_count=len(m["community"]))
                                        for m in models.values()],
                             "errors": errors}, extra_headers=NO_STORE)
    if len(parts) == 2 and parts[0] == "models":
        m = _model(h, parts[1])
        if m is None:
            return None
        from server.setup import community
        return h._json(200, dict(m, contract_version=CONTRACT, community_sync=community.state(m["id"]),
                                 wizard_fields=registry.wizard_fields(m)), extra_headers=NO_STORE)
    if parts == ["community"]:
        return h._json(200, community_body(), extra_headers=NO_STORE)
    if len(parts) == 3 and parts[0] == "models" and parts[2] == "current":
        m = _model(h, parts[1])
        if m is None:
            return None
        from server.setup import apply
        return h._json(200, apply.current_selection(m), extra_headers=NO_STORE)
    if len(parts) == 3 and parts[0] == "models" and parts[2] == "steps":
        m = _model(h, parts[1])
        if m is None:
            return None
        return h._json(200, {"contract_version": CONTRACT, "model": m["id"],
                             "steps": registry.wizard_fields(m), "tools": tools.definitions(m)},
                       extra_headers=NO_STORE)
    if parts == ["device"]:
        return h._json(200, device(app), extra_headers=NO_STORE)
    if len(parts) == 2 and parts[0] == "jobs":
        return job_route(h, parts[1])
    if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "diagnostics":
        return diagnostics_route(h, parts[1])
    if parts == ["status"]:
        return h._json(200, setup_status(app), extra_headers=NO_STORE)
    if parts == ["weights"]:
        from server.setup import remove
        return h._json(200, dict(remove.installed(setup_mode=bool(app.STATE.get("setup_mode"))),
                                 contract_version=CONTRACT), extra_headers=NO_STORE)
    if parts == ["reset"]:
        from server.setup import reset
        return h._json(200, reset.levels(app, busy(app)), extra_headers=NO_STORE)
    if parts == ["assistant", "status"]:
        w = wren_of(app)
        w.touch()
        return h._json(200, w.status(), extra_headers=NO_STORE)
    return _err(h, 404, "not_found", f"no route /v1/setup/{'/'.join(parts)}")


def status_module():
    """ops/engine_status.py of the ops folder this server was started with (stdlib only): the status
    responder's own reading of the setup job and the load, so the page gets one shape from both."""
    import importlib.util
    from server.setup import device as dev
    p = os.path.join(dev.ops_dir(), "engine_status.py")
    if not os.path.isfile(p):
        p = os.path.join(registry.ROOT, "ops", "engine_status.py")
    spec = importlib.util.spec_from_file_location("tandem_engine_status", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_status(app) -> dict:
    """GET /v1/setup/status (OPS-175): what the status responder answers while the engine restarts,
    from the engine itself while it runs: the engine up, the running or newest setup job, no load."""
    from server.setup import apply
    es = status_module()
    up = {"state": "up", "words": "up", "since": int(app.STATE.get("started") or 0) or None,
          "reason": None, "left_s": None, "version": app.STATE.get("version"),
          "profile": os.environ.get("QSE_WEIGHTS") or None,
          "mode": "setup" if app.STATE.get("setup_mode") else "serving"}
    out = es.setup_status(t=apply.ROOT, engine=up)
    out["contract_version"] = CONTRACT
    return out


def community_body() -> dict:
    """GET /v1/setup/community and POST /v1/setup/community/sync: the sync state and every
    model's community packs from the index in use."""
    from server.setup import community
    models, _ = registry.load_all()
    return {"contract_version": CONTRACT, "sync": community.state(),
            "models": {mid: m["community"] for mid, m in models.items()}}


def selection_model(body) -> dict:
    """The model a selection names, or SelectionError (400 unknown_model)."""
    mid = body.get("model") if isinstance(body, dict) else None
    m = device_mod.windowed(registry.get(mid)) if isinstance(mid, str) else None
    if m is None:
        raise estimate_mod.SelectionError(400, "unknown_model", f"no model {mid!r} in models/")
    return m


def _post(h, app, parts: list[str], body) -> None:
    if parts == ["estimate"]:
        sel = body.get("selection", body) if isinstance(body, dict) else body
        m = selection_model(sel)
        mem = device_mod.meminfo()
        out = estimate_mod.estimate(m, estimate_mod.check(m, sel),
                                    available_gb=round(mem["MemAvailable"], 2)
                                    if "MemAvailable" in mem else None)
        return h._json(200, out, extra_headers=NO_STORE)
    if parts == ["community", "sync"]:
        from server.setup import community
        community.sync()
        return h._json(200, community_body(), extra_headers=NO_STORE)
    if parts in (["plan"], ["apply"]):
        return apply_route(h, app, parts[0], body)
    if parts == ["reset"]:
        return reset_route(h, app, body)
    if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "retry":
        return retry_route(h, app, parts[1])
    if parts == ["remove"]:
        return remove_route(h, app, body)
    if parts == ["assistant", "load"]:
        return h._json(200, wren_of(app).load(), extra_headers=NO_STORE)
    if parts == ["assistant", "unload"]:
        return h._json(200, wren_of(app).unload("asked"), extra_headers=NO_STORE)
    if parts in (["assistant", "chat"], ["chat"]):
        return sse(h, wren_of(app).chat(body))
    return _err(h, 404, "not_found", f"no route POST /v1/setup/{'/'.join(parts)}")


def sse(h, events) -> None:
    """An event stream of `data: {json}` frames; a reader that goes away closes the generator,
    which stops what it was producing."""
    import json
    try:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("X-Accel-Buffering", "no")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        for ev in events:
            h.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        h.close_connection = True
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


# 412 for what the box lacks, 409 for the rest
PLAN_STATUS = {"missing_token": 412, "missing_assets": 412, "disk_full": 412,
               "no_overlay_hook": 412}


def apply_route(h, app, which: str, body) -> None:
    from server.setup import apply
    if not isinstance(body, dict):
        return _err(h, 422, "invalid_selection", "the body must be a JSON object")
    if which == "apply":
        extra = sorted(k for k in body if k not in ("selection", "dry_run", "remove_after"))
        if extra:
            return _err(h, 422, "unknown_field", f"unknown field {extra[0]!r}; apply takes "
                        f"selection, dry_run and remove_after", errors=[{"field": k, "reason": "unknown"}
                                                                        for k in extra])
        if not isinstance(body.get("dry_run", False), bool):
            return _err(h, 422, "invalid_selection", "dry_run: true or false")
        ra = body.get("remove_after", [])
        if not isinstance(ra, list) or not all(isinstance(x, str) and apply.profile_ok(x) for x in ra):
            return _err(h, 422, "invalid_selection", "remove_after: a list of this box's profile ids")
    sel_body = body.get("selection", body if which == "plan" else None)
    m = selection_model(sel_body)
    sel = estimate_mod.check(m, sel_body)
    mem = device_mod.meminfo()
    auth = app.STATE.get("auth")
    setup_mode = bool(app.STATE.get("setup_mode"))
    p = apply.plan(m, sel, setup_mode=setup_mode,
                   password_set=bool(getattr(auth, "password_set", False)),
                   available_gb=round(mem["MemAvailable"], 2) if "MemAvailable" in mem else None,
                   mem_total_gb=mem.get("MemTotal"))
    if which == "plan" or body.get("dry_run"):
        return h._json(200, dict(p, dry_run=which == "apply"), extra_headers=NO_STORE)
    return _launch(h, app, m, sel, p, setup_mode=setup_mode,
                   remove_after=[x for x in body.get("remove_after", []) if x != sel["profile"]])


def retry_route(h, app, jid: str) -> None:
    """POST /v1/setup/jobs/{id}/retry (VIS-213): a new job with the failed one's selection (or its
    reset), `retry_of` the old id. The check and tandem doctor run again; a download that finished
    is not repeated (the plan finds the weights), one cut short goes on."""
    from server.setup import apply
    old = apply.read_job(jid) if apply.JOB_ID.match(jid or "") else None
    spec = apply.job_spec(jid) if old is not None else None
    if old is None or spec is None:
        return _err(h, 404, "not_found", f"no job {jid!r}")
    if old["state"] not in ("failed", "rolled_back"):
        return _err(h, 409, "not_retryable", f"job {jid} is {old['state']}: only a failed or rolled "
                                             f"back job is tried again")
    m = device_mod.windowed(registry.get(spec.get("model")))
    if m is None:
        return _err(h, 404, "unknown_model", f"no model {spec.get('model')!r} in models/")
    setup_mode = bool(app.STATE.get("setup_mode"))
    if spec.get("kind") == "reset":
        p = apply.reset_plan(m, setup_mode=setup_mode)
        sel, kind = p["selection"], "reset"
    else:
        sel, kind = estimate_mod.check(m, spec["selection"]), "apply"
        mem = device_mod.meminfo()
        p = apply.plan(m, sel, setup_mode=setup_mode,
                       password_set=bool(getattr(app.STATE.get("auth"), "password_set", False)),
                       available_gb=round(mem["MemAvailable"], 2) if "MemAvailable" in mem else None,
                       mem_total_gb=mem.get("MemTotal"))
    return _launch(h, app, m, sel, p, setup_mode=setup_mode, kind=kind, retry_of=jid,
                   remove_after=spec.get("remove_after") or [])


def remove_route(h, app, body) -> None:
    """POST /v1/setup/remove {profiles, confirm: true, dry_run?} (SRV-560): the chosen profiles'
    weights, never the running one; refused while a setup job runs."""
    from server.setup import apply, remove
    if not isinstance(body, dict) or sorted(set(body) - {"profiles", "confirm", "dry_run"}):
        return _err(h, 422, "invalid_request", "remove takes profiles, confirm and dry_run")
    dry = body.get("dry_run") is True
    if not dry and body.get("confirm") is not True:
        return _err(h, 422, "confirm_required", "confirm: true removes the weights (dry_run: true shows "
                                                "what would go)")
    try:
        with _JOB_LOCK:     # checked under the lock: an apply that starts first is still refused
            run = apply.running_job()
            if run is not None:
                return _err(h, 409, "job_running", f"job {run['id']} runs; remove when it is done",
                            job=run["id"])
            out = remove.remove(body.get("profiles"), setup_mode=bool(app.STATE.get("setup_mode")),
                                root=apply.ROOT, dry_run=dry)
    except remove.RemoveError as exc:
        return _err(h, exc.status, exc.code, str(exc))
    if not dry:
        print(f"[setup] removed {', '.join(g['profile'] for g in out['removed'])}: {out['freed_gb']:g} GB",
              flush=True)
    return h._json(200, dict(out, contract_version=CONTRACT), extra_headers=NO_STORE)


def diagnostics_route(h, jid: str) -> None:
    """GET /v1/setup/jobs/{id}/diagnostics (VIS-213): the job as text for a bug report, no secret."""
    from server.setup import apply
    text = apply.diagnostics(jid) if apply.JOB_ID.match(jid or "") else None
    if text is None:
        return _err(h, 404, "not_found", f"no job {jid!r}")
    raw = text.encode()
    h.send_response(200)
    h.send_header("Content-Type", "text/plain; charset=utf-8")
    h.send_header("Content-Length", str(len(raw)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(raw)
    return None


def _launch(h, app, m: dict, sel: dict, p: dict, *, setup_mode: bool, kind: str = "apply",
            retry_of: str | None = None, remove_after: list[str] | None = None) -> None:
    """A job from a checked plan: refused while requests run (409 busy), when the plan has an
    error, or while another job runs (409 job_running); else Wren unloaded, the runner spawned,
    202 and the job."""
    from server.setup import apply
    if busy(app) and not setup_mode:
        inf = app.INFLIGHT
        return _err(h, 409, "busy", f"{inf.get('running', 0)} request(s) run and "
                    f"{inf.get('waiting', 0)} wait; apply when the engine is idle",
                    running=inf.get("running", 0), waiting=inf.get("waiting", 0))
    if not p["ok"]:
        code = p["error_codes"][0]
        return h._json(PLAN_STATUS.get(code, 409),
                       {"error": {"type": code, "code": code, "message": " ".join(p["errors"])},
                        "plan": p}, extra_headers=NO_STORE)
    with _JOB_LOCK:
        run = apply.running_job()
        if run is not None:
            return _err(h, 409, "job_running", f"job {run['id']} runs; one setup job at a time",
                        job=run["id"])
        wren_of(app).unload("apply")
        job = apply.new_job(m, sel, p, setup_mode=setup_mode, kind=kind, retry_of=retry_of,
                            remove_after=remove_after)
        apply.spawn(job["id"])
    print(f"[setup] {kind} job {job['id']}: {sel['profile']}, {len(p['changes'])} settings, "
          f"started" + (f" (a retry of {retry_of})" if retry_of else ""), flush=True)
    return h._json(202, {"contract_version": CONTRACT, "job": job}, extra_headers=NO_STORE)


def job_route(h, jid: str) -> None:
    from urllib.parse import parse_qs, urlsplit
    import json
    import time
    from server.setup import apply
    if not apply.JOB_ID.match(jid):
        return _err(h, 404, "not_found", f"no job {jid!r}")
    job = apply.read_job(jid)
    if job is None:
        return _err(h, 404, "not_found", f"no job {jid!r}")
    q = {k: v[-1] for k, v in parse_qs(urlsplit(h.path).query).items()}
    if q.get("format") == "json" or "text/event-stream" not in (h.headers.get("Accept") or
                                                                 "text/event-stream"):
        return h._json(200, job, extra_headers=NO_STORE)
    try:
        after = int(h.headers.get("Last-Event-ID") or q.get("after") or 0)
    except ValueError:
        after = 0
    try:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("X-Accel-Buffering", "no")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        after, offset, ended = _catch_up(h, jid, after)
        if ended:
            h.wfile.flush()
            return None
        quiet = time.monotonic()
        while True:
            recs, offset = apply.events_from(jid, offset, after)
            for rec in recs:
                after = rec["seq"]
                h.wfile.write(f"id: {after}\ndata: {json.dumps(rec['event'])}\n\n".encode())
                quiet = time.monotonic()
                if rec["event"].get("type") == "done":
                    h.wfile.flush()
                    return None
            j = apply.read_job(jid)
            if j is None or j["state"] not in ("queued", "running"):
                # a runner that died says so once
                if j is not None:
                    done = {"type": "done", "state": j["state"], "message": j["message"]}
                    h.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                h.wfile.flush()
                return None
            if time.monotonic() - quiet > 15:
                h.wfile.write(b": ping\n\n")
                quiet = time.monotonic()
            h.wfile.flush()
            time.sleep(0.5)
    except (BrokenPipeError, ConnectionResetError, OSError):
        h.close_connection = True
    return None


def _catch_up(h, jid: str, after: int) -> tuple[int, int, bool]:
    """VIS-149: the events after `after` that are on disk now, as the log lines among them and
    then the job as it is now (`id:` the last of them), so a page never gets an older state after
    a newer one: the steps and the job's first state (seq 1) are in that snapshot, with the id of
    the last of them unless that is a log line's (VIS-211). A final job ends with its `done` event. (the sequence followed, the byte offset, the stream ended)"""
    import json
    from server.setup import apply
    recs, offset = apply.events_from(jid, 0, after)
    done = None
    last = last_log = after
    for rec in recs:
        ev = rec["event"]
        if ev.get("type") == "done":
            done = rec
            continue
        last = rec["seq"]
        if ev.get("type") == "log":
            last_log = last
            h.wfile.write(f"id: {last}\ndata: {json.dumps(ev)}\n\n".encode())
    # read after the events: the snapshot is as new as the last of them (state.json is saved
    # before its event is written) or newer, and a later step event equals it or follows it
    job = apply.read_job(jid)
    if job is None:
        return last, offset, True
    # VIS-211: the snapshot takes the id of the step events it stands for, never a log line's: a
    # page that counted that line's id dropped the snapshot as a repeat, held no job, and showed
    # "Starting the install" over a finished one
    head = f"id: {last}\n" if last > last_log else ""
    h.wfile.write(f"{head}data: {json.dumps({'type': 'job', 'job': job})}\n\n".encode())
    if done is not None:
        h.wfile.write(f"id: {done['seq']}\ndata: {json.dumps(done['event'])}\n\n".encode())
        return done["seq"], offset, True
    if job["state"] not in ("queued", "running"):
        # a runner that died says so once
        ev = {"type": "done", "state": job["state"], "message": job["message"]}
        h.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        return last, offset, True
    return last, offset, False


def clear_live(app, what: str) -> int:
    """Reset level 3's caches the running engine holds, through its own functions (what
    POST /v1/cache/clear?tier=all does): the disk tier and the spill, the lookup store's log
    (truncated under its lock), or one RAM cache."""
    st = app.STATE
    if what == "disk":
        from server import diskcache as diskcache_mod
        diskcache_mod.clear(st)
        if st.get("spill") is not None:
            with app.LOCK:
                st["spill"].clear()
        return 0
    obj = st.get(what)
    if obj is None or not hasattr(obj, "clear"):
        raise ValueError(f"no {what} on this server")
    if what == "suffix_store":
        # its own lock: appends take it, and a rebuild that read the old log is dropped
        return int(obj.clear() or 0)
    with app.LOCK:
        obj.clear()
        if what == "resident":
            obj.shed_park()
    return 0


def reset_route(h, app, body) -> None:
    from server.setup import apply, reset
    if not isinstance(body, dict):
        return _err(h, 422, "invalid_request", "the body must be a JSON object")
    extra = sorted(k for k in body if k not in ("level", "confirm", "dry_run"))
    if extra:
        return _err(h, 422, "unknown_field", f"unknown field {extra[0]!r}; reset takes level, "
                    f"confirm and dry_run")
    n = body.get("level")
    if n == 1:
        return _err(h, 400, "browser_only", "level 1 (the setup progress and Wren's chat) is the "
                    "browser's own: the page clears it")
    if isinstance(n, bool) or n not in (2, 3, 4, 5):
        return _err(h, 422, "invalid_level", "level is 2, 3, 4 or 5")
    lv = reset.level(n, app)
    if body.get("dry_run"):
        reset.audit({"level": n, "dry_run": True, "freed_gb": 0, "result": "plan"})
        return h._json(200, dict(lv, contract_version=CONTRACT, dry_run=True),
                       extra_headers=NO_STORE)
    run = apply.running_job()
    if run is not None:
        reset.audit({"level": n, "dry_run": False, "freed_gb": 0, "result": "job_running"})
        return _err(h, 409, "job_running", f"job {run['id']} runs; one setup job at a time",
                    job=run["id"])
    if busy(app):
        inf = app.INFLIGHT
        reset.audit({"level": n, "dry_run": False, "freed_gb": 0, "result": "busy"})
        return _err(h, 409, "busy", f"{inf.get('running', 0)} request(s) run and "
                    f"{inf.get('waiting', 0)} wait; reset when the engine is idle",
                    running=inf.get("running", 0), waiting=inf.get("waiting", 0))
    if n == 4 and device_mod.active_profile() is None:
        reset.audit({"level": 4, "dry_run": False, "freed_gb": 0, "result": "active_unknown"})
        return _err(h, 409, "active_unknown", reset.ACTIVE_UNKNOWN[0].upper() +
                    reset.ACTIVE_UNKNOWN[1:] + ".")
    word, confirm = reset.WORDS[n], body.get("confirm")
    ok = confirm is True if word is None else (
        isinstance(confirm, str) and confirm.strip().lower() == word)
    if not ok:
        reset.audit({"level": n, "dry_run": False, "freed_gb": 0, "result": "confirm_required"})
        return h._json(400, {"error": {"type": "confirm_required", "code": "confirm_required",
                                       "message": "confirm: true" if word is None else
                                       f"type {word!r} to confirm"},
                             "level": dict(lv, contract_version=CONTRACT)},
                       extra_headers=NO_STORE)
    if n == 2:
        models, _ = registry.load_all()
        m = next(iter(models.values()))
        setup_mode = bool(app.STATE.get("setup_mode"))
        p = apply.reset_plan(m, setup_mode=setup_mode)
        if not p["ok"]:
            code = p["error_codes"][0]
            reset.audit({"level": 2, "dry_run": False, "freed_gb": 0, "result": code})
            return h._json(PLAN_STATUS.get(code, 409),
                           {"error": {"type": code, "code": code,
                                      "message": " ".join(p["errors"])}, "plan": p},
                           extra_headers=NO_STORE)
        with _JOB_LOCK:
            if apply.running_job() is not None:
                return _err(h, 409, "job_running", "a setup job runs; one at a time")
            wren_of(app).unload("reset")
            job = apply.new_job(m, p["selection"], p, setup_mode=setup_mode, kind="reset")
            apply.spawn(job["id"])
        reset.audit({"level": 2, "dry_run": False, "freed_gb": 0, "result": "job",
                     "job": job["id"]})
        return h._json(202, {"contract_version": CONTRACT, "level": 2, "removed": [],
                             "failed": [], "freed_gb": 0.0, "restart_needed": True,
                             "message": "The engine restarts with the shipped configuration.",
                             "ok": True, "job_id": job["id"], "job": job},
                       extra_headers=NO_STORE)
    # 2026-10-17: check-and-act is the job lock. The two checks above only hold until the
    # next request is admitted: an apply that passed its own 409-gate (295, under this same
    # lock) before we got here, and a level 3-5 reset, meant that a download job installed
    # into directories being deleted -- ENOENT mid-install, half-deleted weight folders,
    # a freed_gb read over files the job had just written. apply-vs-apply is serialized by
    # _JOB_LOCK already (setup_api.py:50); only this synchronous path walked around it. The
    # re-check runs INSIDE the lock, so a job that appeared after the fast check above is
    # still refused, and an apply arriving while we delete waits at its gate until we are
    # done. reset.execute takes no lock and spawns no job, so holding this one across the
    # deleting is safe (the level 2 path above is the same pattern, released before its 202).
    with _JOB_LOCK:
        run = apply.running_job()
        if run is not None:
            reset.audit({"level": n, "dry_run": False, "freed_gb": 0, "result": "job_running"})
            return _err(h, 409, "job_running", f"job {run['id']} runs; one setup job at a time",
                        job=run["id"])
        out = reset.execute(n, app, clear_live=lambda what: clear_live(app, what))
    reset.audit({"level": n, "dry_run": False, "freed_gb": out["freed_gb"],
                 "result": "ok" if out["ok"] else f"{len(out['failed'])} failed"})
    print(f"[setup] reset level {n}: {out['freed_gb']:.2f} GiB freed, {len(out['failed'])} "
          f"item(s) not removed", flush=True)
    return h._json(200, out, extra_headers=NO_STORE)
