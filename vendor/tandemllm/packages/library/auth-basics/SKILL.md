---
name: auth-basics
description: Adding login and authorization to a web app or API - password hashing, sessions or short-lived tokens, cookies with the right flags, protecting routes, and keeping secrets out of code and git.
---
# Authentication basics

Rules:
- Passwords: hash with argon2id or bcrypt (cost 12). Never store, log or echo one. Compare with the library's verify function.
- Session or token: browser apps use a session cookie (`HttpOnly; Secure; SameSite=Lax`); APIs for scripts use a bearer token that expires (15 to 60 min) with a refresh path or re-login.
- Secrets (signing keys, API keys) come from the environment; `.env` in `.gitignore`; generate with `openssl rand -hex 32`; grep the repo before any commit.
- One middleware or dependency loads the user or answers 401. Authorization (may this user do this?) is a second check per resource, 403 when it fails.

What goes wrong:
- A JWT with no expiry, or verified with `algorithms` unspecified (`none` accepted).
- Token in `localStorage` read by any injected script; prefer the HttpOnly cookie.
- Authorization skipped on the detail route: user A reads `/items/7` of user B.

Check: curl without credentials (401), with a wrong token (401), with a valid login (200), another user's resource (403), after logout (401); grep the repo for the secret value (0 hits).

Done when:
- [ ] The five status codes above in the result from real requests.
- [ ] `git grep` (or grep -r) for the key value and for `password=` returns 0 hits outside `.env`.
- [ ] Cookie flags or token expiry read back from the response headers.

## Reference

FastAPI dependency:
```python
from fastapi import Depends, HTTPException, Header
import hmac, os, time, secrets
SESSIONS: dict[str, dict] = {}
def current_user(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    s = SESSIONS.get(token)
    if not s or s["exp"] < time.time(): raise HTTPException(401, "not authenticated")
    return s["user"]
@app.post("/login")
def login(body: Login):
    user = USERS.get(body.email)
    if not user or not ph.verify(user["hash"], body.password): raise HTTPException(401, "invalid credentials")
    token = secrets.token_urlsafe(32); SESSIONS[token] = {"user": user["id"], "exp": time.time() + 3600}
    return {"token": token, "expires_in": 3600}
```
Password hashing: `from argon2 import PasswordHasher; ph = PasswordHasher(); h = ph.hash(pw); ph.verify(h, pw)` (raises on mismatch).
Express cookie session: `res.cookie('sid', id, { httpOnly: true, secure: true, sameSite: 'lax', maxAge: 3600_000 })`.
Probe:
```bash
curl -s -w ' %{http_code}\n' localhost:8000/items
T=$(curl -s -X POST localhost:8000/login -H 'Content-Type: application/json' -d '{"email":"a@x.io","password":"pw"}' | jq -r .token)
curl -s -w ' %{http_code}\n' -H "Authorization: Bearer $T" localhost:8000/items
grep -rn "$T" --exclude-dir=.venv --exclude-dir=node_modules . | wc -l
```

More rules:
- Rate-limit login (5 tries a minute per IP and account). Same error message for wrong user and wrong password.
- Log in and log out must both be tested with real requests; the token or cookie must stop working after logout or expiry.

More pitfalls:
- A secret committed once; it is in history forever. Rotate it.
- Comparing tokens with `==`; use a constant-time compare.
