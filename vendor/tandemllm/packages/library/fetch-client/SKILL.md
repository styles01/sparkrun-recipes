---
name: fetch-client
description: Calling a REST or JSON API from browser or Node JavaScript with fetch - status handling, JSON errors, timeouts with AbortController, retries, auth headers, loading and error state in the UI.
---
# fetch and REST clients

Rules:
- One `api()` helper: builds the URL, sets headers, checks `res.ok`, parses JSON, throws an error with status and body. Call sites never call `fetch` directly.
- `fetch` does not throw on 404 or 500. Check `res.ok` every time.
- Timeout every request: `AbortController` with 10 s. Cancel the previous request when a new one of the same kind starts (search boxes).
- Retry only idempotent calls (GET, PUT) and only on network errors or 5xx, at most 3 times with backoff 0.5, 1, 2 s.

What goes wrong:
- CORS blocked: the server must send `Access-Control-Allow-Origin`; the error shows in the console, not in the catch message.
- `await res.json()` on an empty 204 body throws; check `res.status === 204` or `content-length`.
- Relative URL resolves against the page, not the API; use an absolute base.

Check: call the helper against the real endpoint for a 200, a 404 and an aborted request; read the three results in the console or test output; read the network log for the headers sent.

Done when:
- [ ] A real call logged with status, duration and the parsed shape (keys) for one success and one error path.
- [ ] The UI screenshot shows the loading state and the error state once each (or the test asserts both).
- [ ] No secret string appears in client bundle or page source (grep the build output).

## Reference

```js
const BASE = location.origin.includes('localhost') ? 'http://localhost:8000' : '/api';
async function api(path, { method = 'GET', body, signal, headers = {} } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new Error('timeout 10s')), 10000);
  signal?.addEventListener('abort', () => ctl.abort(signal.reason));
  try {
    const res = await fetch(BASE + path, {
      method, signal: ctl.signal, credentials: 'include',
      headers: { Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}), ...headers },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) throw Object.assign(new Error(data?.error ?? `HTTP ${res.status}`), { status: res.status, data });
    return data;
  } finally { clearTimeout(timer); }
}
async function withRetry(fn, tries = 3) {
  for (let i = 0; ; i++) {
    try { return await fn(); }
    catch (e) { if (i >= tries - 1 || (e.status && e.status < 500)) throw e; await new Promise(r => setTimeout(r, 500 * 2 ** i)); }
  }
}
```
Latest-wins search:
```js
let current;
async function search(q) { current?.abort(); current = new AbortController(); return api(`/items?q=${encodeURIComponent(q)}`, { signal: current.signal }); }
```
Node 18+ has global `fetch`; the same helper works with `BASE` from `process.env.API_BASE`.

More rules:
- Show three states in the UI: loading, error with the message, data. Never a blank page.
- Secrets never in frontend code. The browser sends cookies or a short-lived token from a login endpoint.

More pitfalls:
- Stale response overwrites new data because the earlier request finished later; abort or compare a sequence number.
- Errors swallowed by an empty catch; always log and show.
