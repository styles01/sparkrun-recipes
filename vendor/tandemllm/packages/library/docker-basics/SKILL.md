---
name: docker-basics
description: Writing a Dockerfile and compose file for a service, reading container logs, and running an app in a container - layers, caching, non-root user, ports, volumes, health checks, and the project rule on where images get built.
---
# Docker basics

Rules:
- Read the project rules first. Some projects forbid local image builds (CI builds them): then edit the Dockerfile, lint it, and let the pipeline build.
- Layer order for cache: pinned base image, system deps, dependency files only (`requirements.txt`, `package.json`), install, then the source.
- Non-root user, one process per container, `EXPOSE` the port, `CMD` in exec form, bind to `0.0.0.0` inside.
- `.dockerignore` with `.git`, `.venv`, `node_modules`, `.env`. Secrets injected at run time, never baked in.
- A `HEALTHCHECK` or compose `healthcheck` on `/healthz`.

What goes wrong:
- The server listens on `127.0.0.1` inside the container; the port mapping shows nothing.
- `COPY . .` before `pip install`; every edit rebuilds everything.
- Writes to a path that is not a volume; data gone on restart.
- The container exits at once: read `docker logs <name>` before changing anything.

Check (when running is allowed): `docker compose up -d`, `ps` healthy, `curl localhost:PORT/healthz` 200, `logs --tail 50`, then `down`.

Done when:
- [ ] Dockerfile lint passes (`hadolint` if present); the project's build rule (local or pipeline) stated.
- [ ] If run: `ps` healthy and a `/healthz` 200 in the result; logs tail attached.
- [ ] No secret or `.env` content in the image (`.dockerignore` shown).

## Reference

Python service:
```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd -m app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app . .
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Node: `FROM node:22-slim`, `COPY package*.json ./`, `RUN npm ci --omit=dev`, `COPY . .`, `USER node`, `CMD ["node", "src/server.js"]`.
compose:
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes: ["./data:/app/data"]
    healthcheck: { test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"], interval: 10s, retries: 5 }
  db:
    image: postgres:16
    environment: { POSTGRES_PASSWORD: dev }
    volumes: ["pg:/var/lib/postgresql/data"]
volumes: { pg: {} }
```
Commands: `docker compose up -d --build`, `docker compose ps`, `docker compose logs -f --tail 100 api`, `docker compose exec api sh`, `docker compose down -v` (also deletes volumes).
Lint: `hadolint Dockerfile`. Image size: `docker images | head`.
