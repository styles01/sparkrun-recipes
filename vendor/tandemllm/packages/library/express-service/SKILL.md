---
name: express-service
description: Writing a Node.js HTTP API with Express - routers, JSON bodies, validation, error middleware, status codes, async handlers, environment config, and probing routes with curl.
---
# Node/Express service

Rules:
- `"type": "module"`, Node 18 or newer. Layout: `src/app.js` (app, middleware, routers; no `listen`), `src/server.js` (listen), `src/routes/*.js`. The app without `listen` is what tests import.
- `express.json({ limit: '1mb' })` first. Validate every body (a schema library or hand checks) and answer 400 with `{ error }`.
- Async handlers wrapped so a thrown error reaches the error middleware (Express 5 does this; Express 4 needs a wrapper).
- One error middleware at the end: logs, maps known errors to 4xx, everything else 500 with a generic body.

What goes wrong:
- A handler that throws in an `async` function (Express 4) hangs the request; wrap it.
- `res.send` called twice (after `return` missing); "headers already sent".
- Body undefined: `express.json()` missing or the client did not send `Content-Type: application/json`.

Check: start in the background with the log in `{tandem_scratch}/run.log`, curl valid, invalid and not-found for each route, read statuses and bodies, tail the log, stop the server.

Done when:
- [ ] Log line "listening on 3000"; `/healthz` 200 in the result.
- [ ] Status and body for valid, invalid and not-found per route in the result.
- [ ] Create then read-back agree; the started server is stopped (`pkill` output shown).

## Reference

```js
// src/app.js
import express from 'express';
export const app = express();
app.use(express.json({ limit: '1mb' }));
const db = new Map();
const wrap = fn => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
app.get('/healthz', (req, res) => res.json({ ok: true }));
app.get('/__state', (req, res) => res.json({ tandem_state: 1, items: db.size, uptime: process.uptime() }));
app.post('/items', wrap(async (req, res) => {
  const text = req.body?.text;
  if (typeof text !== 'string' || !text.trim() || text.length > 200) return res.status(400).json({ error: 'text: 1-200 chars' });
  const item = { id: db.size + 1, text }; db.set(item.id, item); res.status(201).json(item);
}));
app.get('/items/:id', (req, res) => {
  const item = db.get(Number(req.params.id));
  item ? res.json(item) : res.status(404).json({ error: 'item not found' });
});
app.use((err, req, res, next) => { console.error(err); res.status(err.status ?? 500).json({ error: err.status ? err.message : 'internal error' }); });
```
```js
// src/server.js
import { app } from './app.js';
const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => console.log(`listening on ${port}`));
```
Run and probe:
```bash
npm init -y >/dev/null && npm i express && (node src/server.js > {tandem_scratch}/run.log 2>&1 &) ; sleep 1
curl -s -w ' %{http_code}\n' localhost:3000/healthz
curl -s -w ' %{http_code}\n' -X POST localhost:3000/items -H 'Content-Type: application/json' -d '{"text":"a"}'
curl -s -w ' %{http_code}\n' localhost:3000/items/99
tail -n 20 {tandem_scratch}/run.log; pkill -f 'node src/server.js'
```

More rules:
- Status codes on purpose: 201 create, 204 delete, 404 unknown id. Return JSON always, including errors.
- Config from `process.env` read in one file; `.env` ignored by git. `/healthz` and `/__state` routes.

More pitfalls:
- Port in use from the last run; `lsof -i :3000` then kill.
- CORS missing for a browser client; add the `cors` middleware with the origin list.
