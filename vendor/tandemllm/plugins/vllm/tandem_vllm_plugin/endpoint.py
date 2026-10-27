"""`GET /tandem/stats` on vLLM's own server, for a proxy that pulls (the `vllm.endpoint_plugins`
entry point, vLLM v0.30.0 `vllm/plugins/endpoint_plugins/interface.py`).

The route reads the same ring the stat logger fills (both live in vLLM's API server process):

    GET /tandem/stats?after_requests=<seq>&after_steps=<seq>
        -> {"plugin", "plugin_version", "engine", "engine_version", "instance", "pid",
            "dropped", "requests": [...], "steps": [...]}

With `TANDEM_COLLECT_TOKEN` set the route needs it as a bearer and answers 401 otherwise. vLLM
loads an endpoint plugin only when `VLLM_PLUGINS` names it (`tandem_stats_endpoint`).
"""

from __future__ import annotations

import hmac
import os

ROUTE = "/tandem/stats"


class TandemStatsEndpoint:
    name = "tandem_stats_endpoint"
    required_tasks = None

    def attach_router(self, app) -> None:
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse

        router = APIRouter()

        @router.get(ROUTE)
        async def tandem_stats(request: Request, after_requests: int = 0, after_steps: int = 0):
            status, body = answer(request.headers.get("authorization"), after_requests,
                                  after_steps)
            return JSONResponse(body, status_code=status)

        app.include_router(router)

    async def init_state(self, engine_client, state, args) -> None:
        return None


def answer(authorization: str | None, after_requests: int, after_steps: int,
           env=None) -> tuple[int, dict]:
    """(status, body) of the route; the route itself only wraps this."""
    env = os.environ if env is None else env
    token = env.get("TANDEM_COLLECT_TOKEN") or ""
    if token:
        got = (authorization or "")
        got = got[7:].strip() if got[:7].lower() == "bearer " else ""
        if not hmac.compare_digest(got.encode(), token.encode()):
            return 401, {"error": {"message": "a bearer token is needed (TANDEM_COLLECT_TOKEN)",
                                   "type": "unauthorized"}}
    from tandem_vllm_plugin.logger import pusher
    return 200, pusher().batch(max(0, int(after_requests)), max(0, int(after_steps)))
