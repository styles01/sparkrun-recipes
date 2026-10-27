---
name: processes-ports
description: Starting servers and long-running commands from an agent shell - background with logs, waiting for a port, health checks, finding and killing stale processes, timeouts, and cleaning up at the end.
---
# Processes and ports

Rules:
- A server never runs in the foreground of the tool shell. Start it with `(cmd > {tandem_scratch}/run.log 2>&1 &)`, note the PID, then wait for the port.
- Wait with a loop (up to 20 s): `curl -sf localhost:PORT/healthz` or a port check; read the log tail when it fails.
- Before starting, check the port is free: `lsof -i :PORT`. Kill only the process you own or the stale one from your last run, by PID, and say so.
- Use absolute paths and set the working directory in the command (`cd /abs/path && ...`), since each tool call may start fresh.

What goes wrong:
- The tool call times out because the server held the foreground; nothing after it ran.
- "Address already in use" from the previous attempt; find it with `lsof`, do not pick a random new port silently.
- The server started but on `127.0.0.1` while the test calls another host, or on a different port than the one curl uses.

Check: port check before, start line with PID, the wait loop's result, one health call with its status code, the log tail, and the stop with a second port check.

Done when:
- [ ] `lsof -i :PORT` before and after in the result (free, then your PID, then free).
- [ ] A health call with status 200 and the first 20 log lines.
- [ ] The stop command and its result (process gone).

## Reference

```bash
PORT=8000; LOG={tandem_scratch}/run.log; mkdir -p {tandem_scratch}
lsof -i :$PORT -sTCP:LISTEN -t && { echo "port busy"; lsof -i :$PORT -sTCP:LISTEN; }   # decide, then kill by PID
cd /abs/project && (PYTHONUNBUFFERED=1 uvicorn app.main:app --port $PORT > $LOG 2>&1 & echo $! > {tandem_scratch}/server.pid)
for i in $(seq 1 20); do curl -sf localhost:$PORT/healthz >/dev/null && break; sleep 1; done
curl -s -o /dev/null -w 'health %{http_code}\n' localhost:$PORT/healthz; tail -n 20 $LOG
# ... work ...
kill "$(cat {tandem_scratch}/server.pid)" && sleep 1; kill -0 "$(cat {tandem_scratch}/server.pid)" 2>/dev/null && echo "still running" || echo "stopped"
```
Node: `(node src/server.js > {tandem_scratch}/run.log 2>&1 & echo $! > {tandem_scratch}/server.pid)`; Vite: `(npm run dev -- --port 5173 > {tandem_scratch}/dev.log 2>&1 & echo $! > {tandem_scratch}/dev.pid)`.
Find what listens: `lsof -nP -iTCP -sTCP:LISTEN | grep -E ':(3000|5173|8000)'`; kill a tree: `pkill -P PID; kill PID`.
Long job: `(pytest -q > {tandem_scratch}/test.log 2>&1 & echo $! > {tandem_scratch}/test.pid)`, then `kill -0 $(cat {tandem_scratch}/test.pid) && echo running || tail -n 5 {tandem_scratch}/test.log`.
Timeout a command: `timeout 120 cmd` (Linux) or `gtimeout` (macOS with coreutils); otherwise background plus a poll.

More rules:
- Long jobs (builds, tests over 60 s) go to the background with a log; poll with `tail -n 20` and check `kill -0 PID`.
- At the end of the task, stop what you started and show it is stopped.

More pitfalls:
- The log is empty because output is buffered; `PYTHONUNBUFFERED=1`, `--log-level info`, or `stdbuf -oL`.
- A zombie from a nohup without a PID; always capture `$!`.
