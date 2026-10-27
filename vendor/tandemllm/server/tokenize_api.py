"""`POST /v1/tokenize`: exact token counts with the model's own chat template (SRV-347).

    {"model"?, "messages"?, "tools"?, "text"?}  ->  {"tokens": n, "parts": {"system", "tools",
                                                    "messages"}}

The Context panel splits a chat's context by part; chatd counts what it adds itself (the system
prompt, the tool schemas, attachments) here, so the split adds up to what the engine reads. The
count is the prompt a real request with the same messages and tools would get, rendered the same
way (server/app.py `build_prompt(body, count=True)`: the host tokenizer, nothing on the GPU), so
`tokens` equals that request's `prompt_tokens`. The parts are differences of renders: `tools` is
the prompt with the tools less the prompt without, `system` the system messages over an empty user
turn less the empty turn alone, `messages` the rest. `text` counts a bare string.

Open like `/v1/models` (it reads nothing but what it is sent), and rate-limited like the library:
240 a minute from one address.
"""

from __future__ import annotations

import collections
import threading
import time

RATE_PER_MIN = 240
MAX_MESSAGES = 4096
_hits: dict = collections.defaultdict(collections.deque)
_lock = threading.Lock()


def allowed(who: str, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    with _lock:
        q = _hits[who]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return False
        q.append(now)
        return True


def count(body: dict, count_fn, encode_text) -> dict:
    """`count_fn(body) -> int`: the prompt length of a chat body; `encode_text(str) -> int`."""
    if isinstance(body.get("text"), str) and "messages" not in body:
        n = int(encode_text(body["text"]))
        return {"tokens": n, "parts": {"system": 0, "tools": 0, "messages": n}}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise ValueError("messages must be a non-empty list (or send `text`)")
    tools = body.get("tools") or None
    base = {k: body[k] for k in ("chat_template_kwargs",) if k in body}
    total = int(count_fn({**base, "messages": messages, **({"tools": tools} if tools else {})}))
    bare = int(count_fn({**base, "messages": messages})) if tools else total
    system = [m for m in messages if isinstance(m, dict) and m.get("role") in ("system",
                                                                                 "developer")]
    sys_n = 0
    if system:
        empty = [{"role": "user", "content": ""}]
        sys_n = max(0, int(count_fn({**base, "messages": system + empty}))
                    - int(count_fn({**base, "messages": empty})))
    tools_n = max(0, total - bare)
    return {"tokens": total,
            "parts": {"system": sys_n, "tools": tools_n,
                      "messages": max(0, total - tools_n - sys_n)}}


def handle(handler, body, count_fn, encode_text) -> None:
    who = handler.headers.get("X-Real-IP") or (handler.client_address or ("",))[0]
    if not allowed(who):
        return handler._json(429, {"error": {"type": "rate_limited",
                                             "message": f"at most {RATE_PER_MIN} a minute"}},
                             extra_headers=(("Retry-After", "60"),))
    if not isinstance(body, dict):
        return handler._json(400, {"error": {"type": "invalid_request_error",
                                             "message": "the body must be a JSON object"}})
    try:
        out = count(body, count_fn, encode_text)
    except Exception as exc:                                       # noqa: BLE001
        # a body the template cannot render (a jinja error, a bad role) is the client's
        return handler._json(400, {"error": {"type": "invalid_request_error",
                                             "message": f"{type(exc).__name__}: "
                                                        f"{str(exc)[:300]}"}})
    return handler._json(200, out)
