---
name: python-async
description: Using asyncio in Python - async functions, gather with limits, timeouts, cancellation, running blocking code in a thread, async HTTP with httpx, and avoiding the event loop stalls.
---
# Python asyncio

Rules:
- Use async only for I/O-bound concurrency (many HTTP calls, sockets, subprocesses). CPU work goes to a process pool; blocking libraries go to `asyncio.to_thread`.
- One entry: `asyncio.run(main())`. No nested `asyncio.run`, no loop created by hand.
- Limit concurrency with a `Semaphore` (10 to 50) and `gather` with `return_exceptions=True`; inspect the exceptions, do not drop them.
- Every await on the network has a timeout: `asyncio.wait_for(..., 10)` or the client's own timeout.

What goes wrong:
- A coroutine called without `await`; nothing runs and a warning appears. Read the warnings.
- `gather` on 10,000 tasks at once; rate limits and memory. Semaphore.
- A sync library inside `async def`; everything stalls. Measure with a timer around the await.

Check: run the program on real input with a timer, print the count of completed and failed tasks and the total time, and show that the time is near the slowest call, not the sum.

Done when:
- [ ] Total time, completed count and failed count printed for a real run.
- [ ] Wall time is clearly below the serial sum (both numbers in the result).
- [ ] A forced timeout or failure in one task is reported and does not stop the others (log line shown).

## Reference

```python
import asyncio, time, httpx
async def fetch(client, url, sem):
    async with sem:
        r = await client.get(url, timeout=10); r.raise_for_status(); return len(r.content)
async def main(urls):
    sem = asyncio.Semaphore(20); t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch(client, u, sem) for u in urls), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    bad = [(u, repr(e)) for u, e in zip(urls, results) if isinstance(e, Exception)]
    print(f"done {len(ok)} failed {len(bad)} in {time.perf_counter()-t0:.2f}s"); print(bad[:5])
    return ok
if __name__ == "__main__":
    asyncio.run(main(["https://example.com"] * 50))
```
Blocking call in a thread: `data = await asyncio.to_thread(read_big_file, path)`.
TaskGroup (3.11+):
```python
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(job_a()); t2 = tg.create_task(job_b())
print(t1.result(), t2.result())
```
Subprocess: `p = await asyncio.create_subprocess_exec("cmd", "arg", stdout=asyncio.subprocess.PIPE); out, _ = await p.communicate()`.
Detect loop stalls: `asyncio.run(main(), debug=True)` logs callbacks slower than 100 ms.
Periodic task: `while True: await work(); await asyncio.sleep(5)` inside a stored task; cancel it on shutdown.

More rules:
- Cancellation: let `CancelledError` propagate; clean up in `finally`. Use `asyncio.TaskGroup` (3.11+) so a failure cancels siblings.
- Never block the loop: no `time.sleep`, no `requests`, no big file reads without a thread.

More pitfalls:
- Tasks created with `create_task` and not kept; garbage collected mid-run. Store the task.
