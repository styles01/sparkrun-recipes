---
name: websockets
description: Realtime two-way messaging between browser and server with WebSockets (FastAPI or Node ws) - message schema, reconnect, heartbeats, broadcast, and testing with a scripted client.
---
# WebSockets

Rules:
- One message shape: JSON `{ "type": "...", "data": {...} }`. The server validates `type` and ignores unknown ones with a logged warning.
- The server keeps a set of open connections; broadcast iterates a copy and drops dead ones.
- Heartbeat: server pings every 30 s; a client that misses 2 is closed. The client reconnects with backoff 1, 2, 4, 8 s (max 30) and re-sends its subscription.
- The browser side wraps the socket in a small class with `send`, `onMessage`, `status`; UI shows the connection state.

What goes wrong:
- Sending before `open`; queue messages until the socket is open.
- A broadcast loop that awaits each send; one slow client blocks all. Send concurrently with a timeout.
- The dev server reloads and every client stays disconnected; reconnect logic missing.

Check: start the server in the background, connect two scripted clients, send from one, read the message on the other with a timestamp, kill one client and show the server's connection count drops; browser client reconnects after a server restart.

Done when:
- [ ] Two clients: a message from A arrives at B (both logs in the result).
- [ ] Connection count from `/__state` before and after a client disconnect.
- [ ] A reconnect after a server restart shown in the browser console or state.

## Reference

FastAPI:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, json
app = FastAPI(); clients: set[WebSocket] = set()
@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); clients.add(sock)
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("type") == "chat":
                await broadcast({"type": "chat", "data": msg["data"]})
    except WebSocketDisconnect: pass
    finally: clients.discard(sock)
async def broadcast(obj):
    text = json.dumps(obj); dead = []
    async def one(c):
        try: await asyncio.wait_for(c.send_text(text), 2)
        except Exception: dead.append(c)
    await asyncio.gather(*(one(c) for c in list(clients)))
    for c in dead: clients.discard(c)
@app.get("/__state")
def state(): return {"tandem_state": 1, "clients": len(clients)}
```
Browser client with reconnect:
```js
class Live {
  constructor(url) { this.url = url; this.q = []; this.delay = 1000; this.status = 'closed'; this.open(); }
  open() { this.ws = new WebSocket(this.url); this.status = 'connecting';
    this.ws.onopen = () => { this.status = 'open'; this.delay = 1000; this.q.splice(0).forEach(m => this.ws.send(m)); };
    this.ws.onmessage = e => this.onMessage?.(JSON.parse(e.data));
    this.ws.onclose = () => { this.status = 'closed'; setTimeout(() => this.open(), this.delay); this.delay = Math.min(this.delay * 2, 30000); }; }
  send(obj) { const m = JSON.stringify(obj); this.status === 'open' ? this.ws.send(m) : this.q.push(m); }
}
```
Scripted client: `python -c "import asyncio,websockets,json; ..."` or `websocat ws://localhost:8000/ws`.
Node server: `import { WebSocketServer } from 'ws'; const wss = new WebSocketServer({ port: 8080 }); wss.on('connection', s => s.on('message', m => wss.clients.forEach(c => c.readyState === 1 && c.send(m))));`

More rules:
- Prefer server-sent events for one-way updates; WebSockets only when the client sends often.
- Test with a scripted client (`websocat`, Python `websockets`, Node `ws`), not only the browser.

More pitfalls:
- Messages interleave across connections because state is global; keep per-connection state in the handler.
