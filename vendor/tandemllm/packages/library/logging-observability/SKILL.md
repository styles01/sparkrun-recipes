---
name: logging-observability
description: Adding useful logs, metrics and a state endpoint to a service or script - structured log lines, levels, request ids, timing, log files the runtime can tail, and counters that answer "what happened".
---
# Logging and observability

Rules:
- One logger per module (`logging.getLogger(__name__)`, `pino`/`console` with a prefix in Node). Configure once at the entry point: level from `LOG_LEVEL`, UTC timestamps, the module name.
- One line per event with key=value or JSON fields: `event=order_created user_id=7 ms=12`. No multi-line prose, no secrets, no full bodies (clip to 200 chars).
- Levels mean something: DEBUG for loops and payload shapes, INFO for state changes, WARNING for recoverable surprises, ERROR with the traceback once, at the place that handles it.
- Request id: generate per request, carry it through log lines and the response header.

What goes wrong:
- The same error logged at every layer; three tracebacks for one failure.
- `print` sprinkled in library code; it breaks CLIs and cannot be turned off.
- Logs with passwords, tokens or full card numbers; redact before logging.

Check: run the program, `tail -n 40 {tandem_scratch}/run.log`, confirm a start line, one line per request or job with the id and ms, and the counters at `/__state` match the log count.

Done when:
- [ ] Log tail in the result with timestamps, levels, event names and a request or job id.
- [ ] One forced error produces exactly one ERROR line with the traceback (shown).
- [ ] `/__state` or `--dump-state` counters agree with the log (numbers side by side).

## Reference

Python setup:
```python
import logging, os, sys, time, uuid
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
logging.Formatter.converter = time.gmtime
log = logging.getLogger(__name__)
COUNTERS = {"requests": 0, "errors": 0}
```
FastAPI middleware with request id and timing:
```python
@app.middleware("http")
async def observe(request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12]); t0 = time.perf_counter()
    try:
        resp = await call_next(request); return resp
    except Exception:
        COUNTERS["errors"] += 1; log.exception("event=unhandled rid=%s path=%s", rid, request.url.path); raise
    finally:
        COUNTERS["requests"] += 1
        log.info("event=request rid=%s method=%s path=%s ms=%.1f", rid, request.method, request.url.path, (time.perf_counter()-t0)*1000)
```
`/__state`: `return {"tandem_state": 1, **COUNTERS, "uptime_s": int(time.time() - START)}`.
Node: `const log = (level, event, f = {}) => console.log(JSON.stringify({ t: new Date().toISOString(), level, event, ...f }));` and an Express middleware doing the same with `res.on('finish', ...)`.
Redact: `def safe(s): return s[:3] + "***" if s else s`; never log `Authorization` headers.
Tail while running: `tail -n 60 {tandem_scratch}/run.log | grep -E 'ERROR|WARNING' | head`.

More rules:
- Time what matters: wrap the hot path and log `ms=`; keep counters (requests, errors, jobs done) and expose them at `/__state` or `--dump-state`.
- Write to stdout and let the runner redirect to `{tandem_scratch}/run.log`; the runtime tails that file.

More pitfalls:
- A log line per item in a 1M loop; log a summary every N or at the end.
