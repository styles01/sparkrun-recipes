#!/usr/bin/env python3
"""What the engine is doing when it cannot say so itself (OPS-65, the UT1 self-test, 2026-10-06).

During a deploy hold (07:29 to 07:51) every URL of the service, the dashboard included, answered the
reverse proxy's fallback: a JSON 503 saying "vLLM cold start takes ~5 min", on a white page, for
22 minutes. Two parts fix that, both stdlib, neither touching the engine:

  THE STATUS FILE. ops/start.sh, ops/stop.sh and ops/hold.sh write what they do to
  `logs/engine-status.json` (`QSE_STATUS_FILE`): `{state, since, reason, expected_s, version,
  profile, starts: [{at, s}]}` (`profile`: the weight profile the scripts start, QSE_WEIGHTS from
  serve.env, so a page during a load says which model is coming, OPS-99). `state` is `stopping`, `held` (a hold runs a command with the service
  stopped; its reason is the hold file's first line), `starting`, `up`, `failed` or `down`. A start
  records how long it took; the last five are the estimate for the next one.

  THE MODEL LIST (OPS-67, the UT2 self-test). Open WebUI refreshes its model list from
  `/v1/models`; a 503 there during a restart dropped TandemLLM from it, and a chat answered only
  "Model not found", never the sentence above. So `set up` also keeps the engine's own model list
  in the status file (`models`, from `/v1/models` on `QSE_ENGINE_PORT`, default 8000; every
  later state keeps the last one), and the responder answers `GET /v1/models` and
  `GET /v1/models/{id}` with 200 and that list (a fixed TandemLLM entry when none was recorded),
  the state under `tandem_engine`. Every other route keeps its 503.

  THE RESPONDER (`serve`, opt-in: `QSE_STATUS_PORT` in serve.env, 127.0.0.1 only). The reverse
  proxy sends a request here when the engine does not answer (nginx `error_page 502 503 504`, see
  docs/operations.md). A browser's page request (`/dashboard/...` or `Accept: text/html`) gets a
  small page that says what is going on, since when and about how long, checks `/health` every
  5 s and opens the dashboard when the engine answers. Everything else gets the OpenAI error shape,
  status 503, `Retry-After`, and the same facts under `engine`. The facts come from the file and
  from the engine itself: a file that says `up` while the engine does not answer reads
  "not answering" from the moment the responder first saw it so.

  THE TREE (OPS-173, 2026-10-17). A responder belongs to one tree: serve.env's REPO (exported by the
  scripts; `QSE_STATUS_TREE` names it for the child), whose `run/setup-jobs` it reads, and one status
  file. `GET /_tandem/status` says which (tree, status file, version, pid). `ensure` asks a responder
  that answers on the port and replaces one of another tree, status file, version or engine port; an
  older one without the route is found by its listening socket and replaced when its command line runs
  this script's `serve`; anything else on the port is reported and left alone. A stale responder of
  another tree read that tree's two-day-old file during an install's restart ("switching to Nemotron
  3.5 Bolt") and answered the job's route with 404. A reason ages out: a stopped, failed or down
  state older than REASON_MAX_S shows none, and a start or a stop older than twice the typical start
  plus 10 min reads "not answering".

  THE RESTART (OPS-175). While an install restarts the engine, the page asks the responder:
  `GET /v1/setup/jobs/{id}` as SSE (the server's own events: the log, the job, then each new event,
  pings, the done event) or JSON, from the tree's run/setup-jobs; `GET /v1/setup/status`: the
  engine's facts, the running (or newest) setup job with its step n of N, and the load's progress
  read from the engine log (weights, drafters, warm-up, verify graphs, listening). Its read-only
  routes allow any origin, so a page on the engine's port reaches it on its own port when no reverse
  proxy sends it there; it listens on QSE_STATUS_HOST, else the engine's HOST.

  python3 ops/engine_status.py set starting|up|held|stopping|failed|down [--reason R]
                                   [--expected-s N] [--took-s N]
  python3 ops/engine_status.py show
  python3 ops/engine_status.py serve --port 8009 [--host H]
  python3 ops/engine_status.py ensure --port 8009      # ours in the background, a foreign one replaced
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STARTED = time.time()
#: OPS-173: a stopped, failed or down state's reason older than this is not shown any more
REASON_MAX_S = 6 * 3600
STATES = ("starting", "up", "held", "stopping", "failed", "down")
#: the model list when the status file has none yet (OPS-67)
FALLBACK_MODEL = {"id": "TandemLLM", "object": "model", "owned_by": "tandemllm",
                  "description": "TandemLLM (the engine is restarting or stopped for maintenance)"}
KEEP_STARTS = 5
#: a start takes about this long when none was measured yet (the 27B engine: 3 to 5 minutes)
DEFAULT_START_S = 300
WORDS = {"starting": "starting", "held": "stopped for maintenance", "stopping": "stopping",
         "failed": "failed to start", "down": "stopped", "up": "up",
         "not_answering": "not answering"}


def path() -> str:
    return os.environ.get("QSE_STATUS_FILE") or os.path.join(REPO, "logs", "engine-status.json")


def tree() -> str:
    """The checkout this responder speaks for: QSE_STATUS_TREE (ensure's child), serve.env's REPO
    (the scripts export it; on an install this file runs from DIR/run/ops and REPO is DIR/src),
    else this file's own tree."""
    return os.path.realpath(os.environ.get("QSE_STATUS_TREE") or os.environ.get("REPO") or REPO)


def tree_version(t: str | None = None) -> str | None:
    for d in (t or tree(), REPO):
        try:
            with open(os.path.join(d, "VERSION")) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def read(p: str | None = None) -> dict:
    try:
        with open(p or path()) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def engine_models(port: int | None = None, timeout: float = 3.0) -> list | None:
    """The engine's own `/v1/models` list (OPS-67); None when it does not answer one."""
    port = port or int(os.environ.get("QSE_ENGINE_PORT") or os.environ.get("PORT") or 8000)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            data = json.loads(r.read()).get("data")
    except Exception:                                              # noqa: BLE001
        return None
    keep = ("id", "object", "created", "owned_by", "description")
    out = [{k: m[k] for k in keep if k in m} for m in data or [] if isinstance(m, dict)
           and m.get("id")]
    return out or None


def write(state: str, *, reason: str | None = None, expected_s: float | None = None,
          took_s: float | None = None, p: str | None = None, now: float | None = None,
          models: list | None = None) -> dict:
    """The new state, written atomically; a start's duration joins `starts`. `up` records the
    engine's model list (`models`, else asked from the engine); every state keeps the last one."""
    if state not in STATES:
        raise SystemExit(f"state must be one of {', '.join(STATES)}")
    p = p or path()
    now = time.time() if now is None else now
    d = read(p)
    keep_models = d.get("models") if isinstance(d.get("models"), list) else None
    starts = [s for s in d.get("starts") or [] if isinstance(s, dict)]
    if took_s is not None:
        starts = (starts + [{"at": int(now), "s": round(float(took_s), 1)}])[-KEEP_STARTS:]
    # the version of the tree the status file belongs to (logs/ sits in it), not this script's
    tree = os.path.dirname(os.path.dirname(os.path.abspath(p)))
    version = None
    for d in (tree, REPO):
        try:
            with open(os.path.join(d, "VERSION")) as f:
                version = f.read().strip()
            break
        except OSError:
            continue
    if state == "up":
        models = models if models is not None else engine_models()
    out = {"state": state, "since": int(now), "reason": (reason or "")[:200] or None,
           "expected_s": expected_s, "version": version, "profile": os.environ.get("QSE_WEIGHTS") or None,
           "starts": starts, "models": models or keep_models}
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, p)
    return out


def typical_start(d: dict) -> float:
    s = [float(x["s"]) for x in d.get("starts") or [] if isinstance(x, dict) and x.get("s")]
    return float(statistics.median(s)) if s else float(DEFAULT_START_S)


def facts(d: dict, now: float, answering: bool | None = None, seen_down: float | None = None) -> dict:
    """What the page and the JSON say: the state in words, since when, about how long is left
    (a hold: its expected length; a start: the median of the recent starts), Retry-After."""
    state = d.get("state") or "down"
    since = d.get("since")
    reason = d.get("reason")
    if state == "up" and answering is False:
        state, since = "not_answering", int(seen_down or now)
    # OPS-173: an old file says nothing it no longer knows: no reason after REASON_MAX_S, and a start
    # or a stop that should long be over is a silence
    if since and state in ("stopping", "starting") and \
            now - since > 2 * typical_start(d) + 600:
        state, reason = "not_answering", None
    elif since and state in ("down", "failed", "stopping", "not_answering") and \
            now - since > REASON_MAX_S:
        reason = None
    left = None
    if state == "held" and d.get("expected_s") and since:
        left = max(0.0, since + float(d["expected_s"]) + typical_start(d) - now)
    elif state in ("starting", "stopping") and since:
        left = max(0.0, since + typical_start(d) - now)
    elif state == "not_answering":
        left = typical_start(d)
    retry = int(min(300, max(5, (left or 60) / 4)))
    return {"state": state, "words": WORDS.get(state, state), "since": since,
            "since_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since)) if since else None,
            "reason": reason, "left_s": round(left) if left is not None else None,
            "typical_start_s": round(typical_start(d)), "version": d.get("version"),
            "profile": d.get("profile"), "retry_after_s": retry}


def _mins(s: float) -> str:
    return f"{max(1, round(s / 60))} min" if s >= 90 else f"{max(1, round(s))} s"


def message(f: dict) -> str:
    """One sentence for the API error: what, since when, about how long."""
    out = f"TandemLLM is {f['words']}"
    if f.get("since_iso"):
        out += f" (since {f['since_iso']})"
    if f.get("reason"):
        out += f": {f['reason']}"
    if f.get("left_s") is not None:
        out += f"; back in about {_mins(f['left_s'])}" if f["left_s"] > 0 else \
            "; back any moment"
    return out + ". Retry after the Retry-After seconds."


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TandemLLM is {words}</title>
<style>
:root{{--bg:#0e1116;--fg:#e6e9ef;--muted:#9aa3b2;--accent:#ff8a3d;--card:#161b22;--line:#2a313c}}
@media (prefers-color-scheme:light){{:root{{--bg:#f6f7f9;--fg:#14181f;--muted:#5b6472;--card:#fff;--line:#dfe3ea}}}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}}
main{{max-width:520px;width:100%;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}}p{{margin:8px 0}}.muted{{color:var(--muted);font-size:14px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);margin-right:8px;
animation:p 1.4s ease-in-out infinite}}@keyframes p{{50%{{opacity:.3}}}}
dl{{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:16px 0 0;font-size:14px}}dt{{color:var(--muted)}}dd{{margin:0}}
</style></head><body><main data-engine-status="{state}">
<h1><span class="dot"></span><span id="what">TandemLLM is {words}</span></h1>
<p id="why">{why}</p>
<dl><dt>since</dt><dd id="since" data-ts="{since}">{since_text}</dd>
<dt>back in</dt><dd id="left">{left}</dd><dt>version</dt><dd>{version}</dd>
<dt>weights</dt><dd id="profile">{profile}</dd></dl>
<p class="muted" id="poll">This page checks every 5 s and opens the dashboard when the engine answers.</p>
</main><script>
const loc=(s)=>s?new Date(s*1000).toLocaleTimeString([], {{hour:'2-digit',minute:'2-digit'}}):'—';
const sinceEl=document.getElementById('since');sinceEl.textContent=loc(+sinceEl.dataset.ts||0);
async function poll(){{try{{const r=await fetch('/health',{{cache:'no-store'}});
if(r.ok){{location.reload();return}}const j=await r.json();const e=j.engine||{{}};
if(e.words)document.getElementById('what').textContent='TandemLLM is '+e.words;
if(e.reason!==undefined)document.getElementById('why').textContent=e.reason||'';
if(e.since)sinceEl.textContent=loc(e.since);
document.getElementById('left').textContent=e.left_s==null?'unknown':(e.left_s>=90?Math.round(e.left_s/60)+' min':Math.max(1,Math.round(e.left_s))+' s');
}}catch(_){{}}document.getElementById('poll').textContent='Last checked '+new Date().toLocaleTimeString()+'; checks every 5 s.'}}
setInterval(poll,5000);
</script></body></html>"""


def page(f: dict) -> str:
    left = "unknown" if f.get("left_s") is None else _mins(f["left_s"]) if f["left_s"] > 0 \
        else "any moment"
    since_text = f.get("since_iso") or "—"
    return PAGE.format(state=html.escape(f["state"]), words=html.escape(f["words"]),
                       why=html.escape(f.get("reason") or ""), since=int(f.get("since") or 0),
                       since_text=html.escape(since_text), left=html.escape(left),
                       version=html.escape(str(f.get("version") or "—")),
                       profile=html.escape(str(f.get("profile") or "—")))


def engine_answers(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:                                              # noqa: BLE001
        return False


# ---------------------------------------------------------------------- the restart (OPS-175)
JOB_ID = re.compile(r"^[a-f0-9]{16}$")
FINAL = ("done", "failed", "rolled_back")
#: the engine log's load phases, in order: a phase is done once one of its patterns appears (the
#: 27B's, Flash-Next's, NemotronH's and LFM2.5's start lines; tools/: the fake engine writes the same)
PHASES = (("weights", "Loading the weights",
           (r"^\[server\] \d+ quantised projections", r"^\[nh-load\] /", r"^\[server\] NemotronH .*: load ",
            r"^\[server\] drafter=", r"^\[server\] .* loaded in")),
          ("drafters", "Loading the drafters", (r"^\[server\] drafter=.* loaded in", r"^\[server\] NemotronH .*: load ",
                                               r"^\[server\] .*drafter.*loaded")),
          ("warmup", "Warming up", (r"^\[server\] warm-up prefill",)),
          ("graphs", "Capturing the verify graphs", (r"^\[server\] verify graphs",)),
          ("ready", "Opening the port", (r"^\[server\] listening on",)))
#: a line that says how far the weights are (shown under the phase)
WEIGHTS_DETAIL = re.compile(r"^\[(nh-load|exl3|fake-load)\] (experts through layer \d+|\d+ tensors from .*|.*GB of .*)")


def jobs_dir(t: str | None = None) -> str:
    return os.path.join(t or tree(), "run", "setup-jobs")


def job_state(jid: str, t: str | None = None) -> dict | None:
    if not JOB_ID.match(jid or ""):
        return None
    try:
        with open(os.path.join(jobs_dir(t), jid, "state.json")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def job_events(jid: str, offset: int = 0, after: int = 0, t: str | None = None) -> tuple[list, int]:
    """server/setup/apply.py events_from: the events from byte `offset` after sequence `after`, and
    the offset after the last whole line."""
    try:
        with open(os.path.join(jobs_dir(t), jid, "events.jsonl"), "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1
    out = []
    for line in data[:end].splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("seq", 0) > after:
            out.append(rec)
    return out, offset + end


def active_job(t: str | None = None, newest_s: float = 1800) -> dict | None:
    """The running setup job of the tree, else the newest one that ended in the last `newest_s`."""
    try:
        ids = [i for i in os.listdir(jobs_dir(t)) if JOB_ID.match(i)]
    except OSError:
        return None
    best = None
    for jid in ids:
        j = job_state(jid, t)
        if not j:
            continue
        if j.get("state") in ("queued", "running"):
            return j
        if (j.get("ended_at") or 0) > time.time() - newest_s and \
                (best is None or (j.get("ended_at") or 0) > (best.get("ended_at") or 0)):
            best = j
    return best


def job_brief(j: dict) -> dict:
    """The job as the restart card shows it: its step n of N (the running one, else the last that
    moved), its label and state."""
    steps = [s for s in j.get("steps") or [] if isinstance(s, dict)]
    shown = [s for s in steps if s.get("id") != "rollback" or s.get("state") not in ("pending", "skipped")]
    cur = next((s for s in shown if s.get("state") in ("running", "waiting", "failed")), None) or \
        next((s for s in reversed(shown) if s.get("state") == "done"), None)
    i = shown.index(cur) + 1 if cur in shown else 0
    return {"id": j.get("id"), "kind": j.get("kind"), "state": j.get("state"), "model": j.get("model"),
            "profile": j.get("profile"), "summary": j.get("summary"), "message": j.get("message"),
            "created_at": j.get("created_at"), "ended_at": j.get("ended_at"),
            "step": {"index": i, "total": len(shown), "id": cur.get("id"), "label": cur.get("label"),
                     "state": cur.get("state"), "detail": cur.get("detail"),
                     "started_at": cur.get("started_at")} if cur else None}


def load_progress(t: str | None = None, since: float | None = None, max_bytes: int = 400_000) -> dict | None:
    """The engine's load as far as its log says: each phase done or not, the current one, the last
    line about the weights. None without a log, or with one older than `since` (the last start)."""
    log = os.path.join(t or tree(), "logs", "engine.log")
    try:
        st = os.stat(log)
        if since and st.st_mtime < since - 5:
            return None
        with open(log, "rb") as f:
            text = f.read(max_bytes).decode("utf-8", "replace")
    except OSError:
        return None
    lines = text.splitlines()
    out, current, detail = [], None, None
    for pid, label, pats in PHASES:
        done = any(re.search(p, ln) for ln in lines for p in pats)
        out.append({"id": pid, "label": label, "done": done})
        if not done and current is None:
            current = pid
    for ln in lines:
        m = WEIGHTS_DETAIL.match(ln)
        if m:
            detail = m.group(2)
    if current is not None:
        # a later phase done means the earlier ones are too (a log that names them in other words)
        last = max((i for i, p in enumerate(out) if p["done"]), default=0)
        for p in out[:last]:
            p["done"] = True
        current = next((p["id"] for p in out if not p["done"]), None)
    return {"phases": out, "current": current, "weights_detail": detail}


def setup_status(t: str | None = None, engine: dict | None = None) -> dict:
    """GET /v1/setup/status: the engine's facts, the setup job and the load (OPS-175)."""
    t = t or tree()
    j = active_job(t)
    out = {"service": "tandem-status", "engine": engine, "job": job_brief(j) if j else None}
    since = (engine or {}).get("since") if (engine or {}).get("state") == "starting" else None
    out["load"] = load_progress(t, since) if (engine or {}).get("state") in ("starting", "not_answering") else None
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "tandem-status"
    status_file: str | None = None
    #: the tree's version when this responder started (a release later, ensure replaces it)
    version: str | None = None
    engine_health = "http://127.0.0.1:8000/health"
    seen_down: list = [None]

    def _facts(self) -> dict:
        now = time.time()
        d = read(self.status_file)
        answering = engine_answers(self.engine_health) if d.get("state") == "up" else None
        if answering is False and self.seen_down[0] is None:
            self.seen_down[0] = now
        if answering:
            self.seen_down[0] = None
        return facts(d, now, answering, self.seen_down[0])

    def _send(self, code: int, body: bytes, ctype: str, retry: int, cors: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(retry))
        self.send_header("Cache-Control", "no-store")
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _models(self, f: dict, path: str) -> None:
        """OPS-67: the model list stays (a client's list keeps TandemLLM through a restart); the
        state goes with it. A chat request itself still gets the 503."""
        models = read(self.status_file).get("models") or [dict(FALLBACK_MODEL)]
        engine = {"state": f["state"], "message": message(f), "retry_after_s": f["retry_after_s"]}
        if path.rstrip("/") in ("/v1/models", "/models"):
            body = {"object": "list", "data": models, "tandem_engine": engine}
            return self._send(200, json.dumps(body).encode(), "application/json",
                              f["retry_after_s"])
        want = urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])
        card = next((m for m in models if str(m.get("id", "")).lower() == want.lower()), None)
        if card is None:
            body = {"error": {"message": f"The model {want!r} does not exist. {message(f)}",
                              "type": "invalid_request_error", "code": "model_not_found"},
                    "engine": f}
            return self._send(404, json.dumps(body).encode(), "application/json",
                              f["retry_after_s"])
        return self._send(200, json.dumps(dict(card, tandem_engine=engine)).encode(),
                          "application/json", f["retry_after_s"])

    def _setup_job(self, path: str) -> None:
        """The setup guide's job while the engine restarts (server/setup/apply.py writes it), from
        this tree's run/setup-jobs, read only: JSON (the state), or SSE as the server sends it."""
        jid = path.rstrip("/").rsplit("/", 1)[-1]
        j = job_state(jid)
        if j is None:
            return self._send(404, json.dumps({"error": {"type": "not_found",
                                                         "message": f"no job {jid!r}"}}).encode(),
                              "application/json", 5, cors=True)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if "text/event-stream" not in (self.headers.get("Accept") or "") or \
                q.get("format", [""])[-1] == "json" or self.command == "HEAD":
            return self._send(200, json.dumps(j).encode(), "application/json", 5, cors=True)
        try:
            after = int(self.headers.get("Last-Event-ID") or q.get("after", ["0"])[-1] or 0)
        except ValueError:
            after = 0
        self._job_stream(jid, after)

    def _job_stream(self, jid: str, after: int, limit_s: float = 1800) -> None:
        """setup_api.job_route's stream: the log after `after`, the job as it is now (the id of
        the last state event, never a log line's: VIS-211), its done event, then each new event."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        w = self.wfile
        try:
            recs, offset = job_events(jid, 0, after)
            last = last_log = after
            done = None
            for rec in recs:
                ev = rec.get("event") or {}
                if ev.get("type") == "done":
                    done = rec
                    continue
                last = rec["seq"]
                if ev.get("type") == "log":
                    last_log = last
                    w.write(f"id: {last}\ndata: {json.dumps(ev)}\n\n".encode())
            j = job_state(jid)
            head = f"id: {last}\n" if last > last_log else ""
            w.write(f"{head}data: {json.dumps({'type': 'job', 'job': j})}\n\n".encode())
            if done is not None:
                w.write(f"id: {done['seq']}\ndata: {json.dumps(done['event'])}\n\n".encode())
                w.flush()
                return
            t_end, quiet = time.monotonic() + limit_s, time.monotonic()
            while time.monotonic() < t_end:
                recs, offset = job_events(jid, offset, last)
                for rec in recs:
                    last = rec["seq"]
                    w.write(f"id: {last}\ndata: {json.dumps(rec['event'])}\n\n".encode())
                    quiet = time.monotonic()
                    if (rec.get("event") or {}).get("type") == "done":
                        w.flush()
                        return
                j = job_state(jid) or {}
                if j.get("state") in FINAL and not recs:
                    w.write(f"data: {json.dumps({'type': 'done', 'state': j['state'], 'message': j.get('message')})}\n\n".encode())
                    w.flush()
                    return
                if time.monotonic() - quiet > 15:
                    w.write(b": ping\n\n")
                    quiet = time.monotonic()
                w.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _identity(self) -> None:
        """OPS-173: which tree and status file this responder speaks for."""
        body = {"service": "tandem-status", "tree": tree(), "status_file": os.path.realpath(self.status_file or path()),
                "version": self.version, "engine_port": int(self.engine_health.rsplit(":", 1)[1].split("/")[0]),
                "host": self.server.server_address[0], "pid": os.getpid(), "started": int(STARTED),
                "script": os.path.realpath(__file__)}
        self._send(200, json.dumps(body).encode(), "application/json", 5, cors=True)

    def _answer(self) -> None:
        path = self.path.split("?")[0]
        if self.command in ("GET", "HEAD") and path == "/_tandem/status":
            return self._identity()
        if self.command in ("GET", "HEAD") and path.startswith("/v1/setup/jobs/") and path.count("/") == 4:
            return self._setup_job(path)
        f = self._facts()
        if self.command in ("GET", "HEAD") and path == "/v1/setup/status":
            return self._send(200, json.dumps(setup_status(engine=f)).encode(), "application/json",
                              f["retry_after_s"], cors=True)
        if self.command in ("GET", "HEAD") and (path.rstrip("/") in ("/v1/models", "/models")
                                                or path.startswith("/v1/models/")):
            return self._models(f, path)
        wants_page = (self.command in ("GET", "HEAD")
                      and (path == "/" or path.startswith("/dashboard")
                           or "text/html" in (self.headers.get("Accept") or "")))
        if wants_page:
            return self._send(503, page(f).encode(), "text/html; charset=utf-8",
                              f["retry_after_s"])
        body = {"error": {"message": message(f), "type": "provider_unavailable",
                          "code": "engine_" + f["state"]}, "engine": f}
        if path == "/health":
            body["status"] = f["state"]
        self._send(503, json.dumps(body).encode(), "application/json", f["retry_after_s"],
                   cors=path == "/health")

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _answer

    def do_OPTIONS(self):
        # OPS-175: a page on the engine's port asks the read-only routes from another origin; a
        # browser may check first (a preflight)
        path = self.path.split("?")[0]
        if path in ("/health", "/_tandem/status", "/v1/setup/status") or path.startswith("/v1/setup/jobs/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._answer()

    def log_message(self, *a):
        pass


def serve(port: int, host: str = "127.0.0.1", engine_port: int = 8000) -> None:
    Handler.status_file = path()
    Handler.version = tree_version()
    Handler.engine_health = f"http://127.0.0.1:{engine_port}/health"
    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"[status] serving {Handler.status_file} for {tree()} on {host}:{port}", flush=True)
    srv.serve_forever()


def identity(port: int, timeout: float = 2.0) -> dict | None:
    """A responder's answer to /_tandem/status (OPS-173); None: none, or an older one."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_tandem/status", timeout=timeout) as r:
            d = json.loads(r.read())
        return d if isinstance(d, dict) and d.get("service") == "tandem-status" else None
    except Exception:                                              # noqa: BLE001
        return None


def listener_pid(port: int, proc: str = "/proc") -> int | None:
    """The pid that listens on 127.0.0.1/0.0.0.0:`port` (TCP, IPv4 or IPv6): the socket's inode in
    /proc/net/tcp*, then the process that holds it. None when it cannot be told (not Linux, another
    user's process)."""
    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc, "net", name)) as f:
                next(f)
                for ln in f:
                    parts = ln.split()
                    if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                        inodes.add(parts[9])
        except (OSError, ValueError, StopIteration):
            continue
    if not inodes:
        return None
    want = {f"socket:[{i}]" for i in inodes}
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            for fd in os.listdir(os.path.join(proc, d, "fd")):
                if os.readlink(os.path.join(proc, d, "fd", fd)) in want:
                    return int(d)
        except OSError:
            continue
    return None


def is_responder(pid: int, proc: str = "/proc") -> bool:
    """`pid` runs this script's `serve` (of any tree): the only process ensure may stop."""
    try:
        with open(os.path.join(proc, str(pid), "cmdline"), "rb") as f:
            argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return False
    return any(a.endswith("engine_status.py") for a in argv) and "serve" in argv


def _stop(pid: int, port: int) -> bool:
    try:
        os.kill(pid, 15)
    except OSError:
        return False
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                pass
        except OSError:
            return True
        time.sleep(0.1)
    return False


def _start(port: int, engine_port: int, host: str) -> None:
    log = os.path.join(os.path.dirname(path()), "engine-status-server.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", "--port", str(port),
                          "--engine-port", str(engine_port), "--host", host], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, start_new_session=True, close_fds=True,
                         env=dict(os.environ, QSE_STATUS_FILE=path(), QSE_STATUS_TREE=tree()))


def ensure(port: int, engine_port: int = 8000, host: str | None = None) -> str:
    """The responder of this tree in the background (its own session, stdio to a log). One that
    answers already is kept when it speaks for this tree, status file, version and engine port;
    another responder (another tree, or an older release without /_tandem/status) is stopped and
    replaced; any other program on the port is left alone (OPS-173)."""
    host = host or os.environ.get("QSE_STATUS_HOST") or os.environ.get("HOST") or "127.0.0.1"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except OSError:
        _start(port, engine_port, host)
        return "started"
    me = {"tree": tree(), "status_file": os.path.realpath(path()), "version": tree_version(),
          "engine_port": engine_port, "host": host}
    them = identity(port)
    if them is not None and all(them.get(k) == v for k, v in me.items()):
        return "running"
    pid = them.get("pid") if them else listener_pid(port)
    if not isinstance(pid, int) or not is_responder(pid):
        return "held by another program (the status page is off)"
    if not _stop(pid, port):
        return f"an old responder (pid {pid}) did not stop"
    _start(port, engine_port, host)
    was = them.get("tree") if them else "an older release"
    return f"replaced (it served {was})"


def stop(port: int) -> str:
    """Stop the responder on `port` when it speaks for this tree (an uninstall, a test's end)."""
    them = identity(port)
    if them is None:
        return "none of ours"
    if them.get("tree") != tree() or not is_responder(them.get("pid") or -1):
        return f"kept (it serves {them.get('tree')})"
    return "stopped" if _stop(them["pid"], port) else "did not stop"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the engine's status file and responder (OPS-65)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("state", choices=STATES)
    s.add_argument("--reason")
    s.add_argument("--expected-s", type=float)
    s.add_argument("--took-s", type=float)
    s.add_argument("--engine-port", type=int, help="where `up` asks the engine its model list")
    sub.add_parser("show")
    sub.add_parser("stop").add_argument("--port", type=int, required=True)
    for name in ("serve", "ensure"):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--engine-port", type=int, default=8000)
        p.add_argument("--host", help="the address to listen on (QSE_STATUS_HOST, else HOST, else 127.0.0.1)")
    a = ap.parse_args(argv)
    if a.cmd == "set":
        models = engine_models(a.engine_port) if a.state == "up" and a.engine_port else None
        write(a.state, reason=a.reason, expected_s=a.expected_s, took_s=a.took_s, models=models)
    elif a.cmd == "show":
        print(json.dumps(facts(read(), time.time()), indent=1))
    elif a.cmd == "stop":
        print(f"[status] responder on :{a.port}: {stop(a.port)}")
    elif a.cmd == "serve":
        serve(a.port, host=a.host or "127.0.0.1", engine_port=a.engine_port)
    else:
        print(f"[status] responder {ensure(a.port, a.engine_port, a.host)} on :{a.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
