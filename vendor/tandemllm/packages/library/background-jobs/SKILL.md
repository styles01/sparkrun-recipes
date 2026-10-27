---
name: background-jobs
description: Running work outside the request - queues, workers, scheduled tasks and retries in Python or Node, with idempotent jobs, timeouts and a way to see what ran.
---
# Background jobs

Rules:
- Anything over 1 s or that calls an outside service leaves the request: enqueue a job with an id, return 202 and the id, let a worker run it.
- Jobs are idempotent: running twice gives the same result (check "already done" by job id or a natural key before writing).
- Every job has a timeout, a retry count (3) with backoff, and a dead-letter state after the last failure.
- Record status per job (queued, running, done, failed, with timestamps and the error text) in a table or a file the API can read.

What goes wrong:
- The job holds a database connection open for the whole run; the pool runs dry.
- Two workers take the same row; claim it with `UPDATE ... WHERE status = 'queued' RETURNING`.
- A crash mid-job leaves it "running" forever; stale `running` older than the timeout goes back to queued.

Check: enqueue through the API, read the job status until done (2 s polling, 30 s max), read the worker log line for that id, run the same job twice and show the second was a no-op.

Done when:
- [ ] One job's status transitions read back: queued, running, done, with times.
- [ ] A forced failure retried 3 times then marked failed, visible in the log tail.
- [ ] A duplicate submission did not repeat the side effect (count read before and after).

## Reference

Table-backed queue (SQLite/Postgres):
```sql
CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
  attempts INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
```
Worker loop (Python):
```python
import sqlite3, json, time, traceback, datetime as dt
HANDLERS = {"email": send_email}
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def claim(con):
    with con:
        row = con.execute("SELECT id, kind, payload FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row: return None
        con.execute("UPDATE jobs SET status='running', started_at=?, attempts=attempts+1 WHERE id=? AND status='queued'", (now(), row[0]))
        return row
def run_forever(path="app.db"):
    con = sqlite3.connect(path)
    while True:
        job = claim(con)
        if not job: time.sleep(2); continue
        jid, kind, payload = job
        try:
            HANDLERS[kind](json.loads(payload)); con.execute("UPDATE jobs SET status='done', finished_at=? WHERE id=?", (now(), jid))
        except Exception:
            err = traceback.format_exc()[-800:]
            att = con.execute("SELECT attempts FROM jobs WHERE id=?", (jid,)).fetchone()[0]
            con.execute("UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?", ('queued' if att < 3 else 'failed', err, now(), jid))
        con.commit(); print(now(), jid, kind, flush=True)
```
Run: `(python worker.py > {tandem_scratch}/worker.log 2>&1 &)`; status route: `GET /jobs/{id}` returns the row.
Node: `setInterval` poll with the same table, or BullMQ (`new Worker('q', fn, { connection })`) when Redis exists.

More rules:
- Start small: an in-process queue (`asyncio.Queue`, a thread, or a `jobs` table polled every 2 s) before Redis, Celery or BullMQ. Pick the library only if the project already has it.
- Scheduled work: one scheduler process, never a cron per container; log each run.

More pitfalls:
- Secrets or large payloads stored in the queue; store a reference.
