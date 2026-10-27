---
name: fastapi-service
description: Writing a Python HTTP API with FastAPI (or Flask) - pydantic models, routers, validation errors, status codes, dependencies, settings from the environment, running with uvicorn and checking routes with curl.
---
# FastAPI service

Rules:
- Layout: `app/main.py` (app + routers), `app/models.py` (pydantic in/out), `app/db.py`, `app/routes/*.py`. One router per resource.
- Every request body and response is a pydantic model. Return `response_model=` so the output is validated and documented.
- Status codes on purpose: 201 on create, 204 on delete, 404 with `HTTPException(404, "item not found")`, 422 left to pydantic.
- Settings from the environment (`pydantic-settings`), never hard-coded; `.env` ignored by git.
- Run: `uvicorn app.main:app --port 8000 > {tandem_scratch}/run.log 2>&1 &`, then curl `/healthz`. Add `/healthz` and `/__state`.

What goes wrong:
- `async def` route calling a sync database driver; the whole server stalls.
- 500 on a known error path; map it to 400/404/409 with a JSON body.
- Mutable default or module-level state shared across requests.

Check: curl each route: a valid call, an invalid body (422), a missing id (404); read status and body each time; read the log tail; stop the server at the end.

Done when:
- [ ] Log line shows uvicorn listening; `/healthz` returns 200 (curl output in the result).
- [ ] For every route the status and body for valid, invalid and not-found calls are in the result.
- [ ] Create then read-back agrees; the server you started is stopped.

## Reference

```python
# app/main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
app = FastAPI()
class ItemIn(BaseModel):
    text: str = Field(min_length=1, max_length=200)
class Item(ItemIn):
    id: int
DB: dict[int, Item] = {}
@app.get("/healthz")
def healthz(): return {"ok": True}
@app.get("/__state")
def state(): return {"tandem_state": 1, "items": len(DB), "routes": [r.path for r in app.routes]}
@app.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
def create(body: ItemIn):
    item = Item(id=len(DB) + 1, **body.model_dump()); DB[item.id] = item; return item
@app.get("/items/{item_id}", response_model=Item)
def get(item_id: int):
    if item_id not in DB: raise HTTPException(404, "item not found")
    return DB[item_id]
```
Run and probe:
```bash
python -m venv .venv && . .venv/bin/activate && pip install fastapi uvicorn
(uvicorn app.main:app --port 8000 > {tandem_scratch}/run.log 2>&1 &) ; sleep 1
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/healthz
curl -s -X POST localhost:8000/items -H 'Content-Type: application/json' -d '{"text":"a"}'
curl -s -w ' %{http_code}\n' -X POST localhost:8000/items -H 'Content-Type: application/json' -d '{"text":""}'
curl -s -w ' %{http_code}\n' localhost:8000/items/99
tail -n 20 {tandem_scratch}/run.log; pkill -f 'uvicorn app.main'
```
Settings: `class Settings(BaseSettings): db_url: str = "sqlite:///app.db"` with `model_config = SettingsConfigDict(env_file=".env")`.

More rules:
- Blocking work (files, CPU) in `def` routes or a threadpool; `async def` only with async libraries.

More pitfalls:
- Port in use from a previous run; check `lsof -i :8000` and kill the old process.
- Flask alternative: the same rules; validate by hand or with marshmallow, run with `flask --app app run --port 8000`.
