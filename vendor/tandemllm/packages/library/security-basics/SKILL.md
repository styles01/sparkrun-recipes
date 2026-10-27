---
name: security-basics
description: Basic security for web apps, APIs and scripts - input validation, SQL and command injection, XSS, path traversal, secrets handling, dependency checks, and the quick probes that prove a hole is closed.
---
# Security basics

Rules:
- Validate at the boundary: type, length, range, allowed set, for every field from a request, a file or an argument. Reject, do not fix up.
- SQL with parameters only. Shell commands as argument lists (`subprocess.run([...])`, `execFile`), never a string with user input. Paths resolved and kept under the allowed root.
- Output encoding: `textContent` in the DOM, the template engine's auto-escape on, `innerHTML` only with your own markup.
- Secrets from the environment, in `.gitignore`, rotated if they were ever printed or committed. No secret in a log, URL or error message.

What goes wrong:
- `f"SELECT ... WHERE id = {id}"` or `shell=True` with a user string.
- A download route that takes `../../etc/passwd`.
- An error page that prints the stack trace and environment to the user.

Check: send the probes (a quote in a text field, `../` in a path, `<script>` in a name, an oversized body, a missing auth header) and read the status codes; grep the code for `shell=True`, string-built SQL and `innerHTML`.

Done when:
- [ ] Five probes with their status codes in the result (all rejected with 4xx and no stack trace).
- [ ] `grep` results for `shell=True`, f-string SQL and `innerHTML` reviewed (0 hits or each justified).
- [ ] Audit tool output (`pip-audit` / `npm audit`) in the result with no critical.

## Reference

Safe path:
```python
from pathlib import Path
ROOT = Path("/app/data").resolve()
def safe_path(name: str) -> Path:
    p = (ROOT / name).resolve()
    if ROOT not in p.parents and p != ROOT: raise ValueError("path outside root")
    return p
```
Safe subprocess: `subprocess.run(["convert", str(src), str(dst)], check=True, timeout=30, capture_output=True)`.
Validation with pydantic: `class Q(BaseModel): q: str = Field(min_length=1, max_length=100, pattern=r"^[\w\s-]+$")`.
Node: `execFile('convert', [src, dst])`; `path.resolve(root, name).startsWith(root + path.sep)`.
Probes:
```bash
curl -s -w ' %{http_code}\n' "localhost:8000/items?q=%27%20OR%201%3D1--"
curl -s -w ' %{http_code}\n' "localhost:8000/files/..%2F..%2Fetc%2Fpasswd"
curl -s -w ' %{http_code}\n' -X POST localhost:8000/items -H 'Content-Type: application/json' -d '{"text":"<script>alert(1)</script>"}'
head -c 3000000 /dev/zero | curl -s -w ' %{http_code}\n' -X POST localhost:8000/items --data-binary @- -H 'Content-Type: application/json'
curl -s -w ' %{http_code}\n' localhost:8000/admin
grep -rnE 'shell=True|innerHTML|f"SELECT|f"INSERT|\$\{.*\}.*(SELECT|INSERT)' --include='*.py' --include='*.js' --include='*.ts' . | grep -v node_modules
```
Headers worth adding: `Content-Security-Policy: default-src 'self'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
Audit: `pip-audit` (or `pip install pip-audit`), `npm audit --omit=dev`.

More rules:
- Least privilege: the database user for the app cannot drop tables; files are written to one directory; the server runs as a normal user.
- Dependencies: `pip-audit`, `npm audit`, pinned versions; read the report, fix criticals.

More pitfalls:
- A JWT accepted with `alg: none` or no expiry; a cookie without `HttpOnly`.
- Validation on the client only.
