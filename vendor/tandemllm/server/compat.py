"""The OpenAI request surface, one disposition per field, and the checks that go with it.

Silently ignoring a field a caller depends on is the worst answer a server can give: the client
cannot tell an ignored `n` or `logit_bias` from a model that happened to answer that way. So every
OpenAI field has exactly one of three dispositions here, and `tests/test_compat.py` pins the table,
so a field cannot be added -- or quietly dropped -- without the test saying so:

  * HONOURED -- the field works;
  * NEUTRAL  -- accepted, and by its own definition it does not change the output (`store`, a
    latency hint such as `prediction`, a label such as `metadata`); docs/server.md says so;
  * REFUSED  -- a 400 naming the field (for the values the engine cannot serve).

Fields that are not OpenAI's (the engine's own -- `top_k`, `min_p`, `repetition_penalty`,
`chat_template_kwargs`, ... -- and whatever else a client adds) are not refused: clients send
extensions for other servers, and a 400 for those would break them without helping anyone.

`tool_choice` lives here too: auto / none / required / a named function, checked against
the request's `tools`; since, required and a named function are also a constraint on the
answer (`tool_constraint`).
"""

from __future__ import annotations

from typing import NamedTuple

HONOURED, NEUTRAL, REFUSED = "honoured", "neutral", "refused"

CHAT_FIELDS = {
    "messages": HONOURED, "model": HONOURED, "stream": HONOURED, "stream_options": HONOURED,
    "max_tokens": HONOURED, "max_completion_tokens": HONOURED, "stop": HONOURED,
    "temperature": HONOURED, "top_p": HONOURED, "seed": HONOURED,
    "presence_penalty": HONOURED, "frequency_penalty": HONOURED, "logit_bias": HONOURED,
    "logprobs": HONOURED, "top_logprobs": HONOURED, "n": HONOURED,
    "tools": HONOURED, "tool_choice": HONOURED, "parallel_tool_calls": HONOURED,
    "reasoning_effort": HONOURED, "response_format": HONOURED,
    # a conversation label for the state cache (server/app.py::conversation_id); no identity
    "user": NEUTRAL, "metadata": NEUTRAL,
    "store": NEUTRAL, "service_tier": NEUTRAL, "prompt_cache_key": NEUTRAL,
    "safety_identifier": NEUTRAL,
    # Predicted outputs change latency, never the text: honouring it is a speed question
    "prediction": NEUTRAL,
    # text is the only modality; `["text"]` is accepted, anything else refused
    "modalities": HONOURED,
    "audio": REFUSED, "functions": REFUSED, "function_call": REFUSED,
    "web_search_options": REFUSED, "verbosity": REFUSED,
}

COMPLETION_FIELDS = {
    "prompt": HONOURED, "model": HONOURED, "stream": HONOURED, "stream_options": HONOURED,
    "max_tokens": HONOURED, "stop": HONOURED, "temperature": HONOURED, "top_p": HONOURED,
    "seed": HONOURED, "presence_penalty": HONOURED, "frequency_penalty": HONOURED,
    "logit_bias": HONOURED, "logprobs": HONOURED, "n": HONOURED,
    "user": NEUTRAL,
    # `best_of: 1` and `echo: false` are the defaults and accepted; anything else is refused
    "best_of": HONOURED, "echo": HONOURED, "suffix": REFUSED,
}

MAX_N = 16
MAX_TOP_LOGPROBS = 20
MAX_COMPLETION_LOGPROBS = 5


class Refusal(ValueError):
    """A request the engine will not serve as asked: a 400 that names the field."""

    def __init__(self, param: str, message: str):
        super().__init__(message)
        self.param = param

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": "invalid_request_error",
                          "param": self.param}}


def _int(body: dict, name: str, lo: int, hi: int, default: int) -> int:
    v = body.get(name)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise Refusal(name, f"{name} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


#: the roles a chat message may have; `developer` (OpenAI's newer name) is read as `system`
ROLES = ("system", "developer", "user", "assistant", "tool")


def _is_tool_response(content) -> bool:
    t = content.strip() if isinstance(content, str) else ""
    return t.startswith("<tool_response>") and t.endswith("</tool_response>")


def messages(body: dict) -> list:
    """SRV-167 (UT1, 2026-10-06): the message list as the chat template can render it, or a 400
    that names it. Empty or missing messages, an unknown role, a list without a user message (only
    a system prompt) and a system message after the first used to reach the template, which
    raised, and the client got `500 internal_error` with the exception's text (and retried, as
    OpenAI clients do on a 5xx). Returns the list with `developer` read as `system` and a leading
    run of system messages joined into one (the template renders one system message, first)."""
    msgs = body.get("messages")
    if msgs is None:
        raise Refusal("messages", "messages is required")
    if not isinstance(msgs, list) or not msgs:
        raise Refusal("messages", "messages must be a non-empty list of messages")
    out: list = []
    user = False
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise Refusal(f"messages[{i}]", f"messages[{i}] must be an object with a role")
        role = m.get("role")
        if role not in ROLES:
            raise Refusal(f"messages[{i}].role",
                          f"messages[{i}].role must be one of {', '.join(ROLES)}, got {role!r}")
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise Refusal(f"messages[{i}].content",
                          f"messages[{i}].content must be a string or a list of parts")
        if content is None and role not in ("assistant",):
            raise Refusal(f"messages[{i}].content", f"messages[{i}].content is required")
        if role in ("system", "developer"):
            if any(o.get("role") != "system" for o in out):
                raise Refusal(f"messages[{i}]",
                              f"messages[{i}]: a {role} message must come before the "
                              f"conversation (this model reads one system prompt, first)")
            if out:                        # system + developer: one system prompt, in order
                prev = out[-1]
                out[-1] = dict(prev, content=_join(prev.get("content"), content))
                continue
            m = dict(m, role="system") if role == "developer" else m
        elif role == "user" and not _is_tool_response(content):
            user = True
        out.append(m)
    if not user:
        raise Refusal("messages", "messages must contain a user message (a system prompt alone, "
                                  "or only tool results, has nothing to answer)")
    return out


def _join(a, b):
    """Two system prompts' contents as one: text joined by a blank line; parts concatenated."""
    if isinstance(a, str) and isinstance(b, str):
        return f"{a}\n\n{b}" if a and b else a or b
    parts = lambda c: c if isinstance(c, list) else [{"type": "text", "text": c or ""}]  # noqa: E731
    return parts(a) + parts(b)


def max_tokens(body: dict) -> None:
    """SRV-167: `max_tokens` and `max_completion_tokens` are positive integers (null: the server's
    default). A negative value used to be served as a one-token answer."""
    for name in ("max_tokens", "max_completion_tokens"):
        v = body.get(name)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise Refusal(name, f"{name} must be a positive integer, got {v!r}")


def prompt(body: dict) -> None:
    """SRV-167, the completion endpoint: `prompt` is a non-empty string (or a list whose first
    item is one, which is the item served)."""
    p = body.get("prompt")
    if isinstance(p, list) and p:
        p = p[0]
    if not isinstance(p, str) or not p:
        raise Refusal("prompt", "prompt must be a non-empty string")


def request_shape(body: dict, chat: bool) -> None:
    """The request's own shape, before anything reads it: the messages (rewritten in place:
    `developer` as `system`, the leading system prompts as one) or the prompt, and the output
    limit. A Refusal names the field."""
    if not isinstance(body, dict):
        raise Refusal("body", "the request body must be a JSON object")
    if chat:
        body["messages"] = messages(body)
    else:
        prompt(body)
    max_tokens(body)


def check(body: dict, chat: bool, structured: bool = False) -> None:
    """Refuse what the engine cannot serve, naming the field. Everything else passes.
    `structured`: the server compiles constraints, so the JSON response formats work."""
    table = CHAT_FIELDS if chat else COMPLETION_FIELDS
    for name, disp in table.items():
        if disp == REFUSED and body.get(name) is not None:
            hint = {"functions": "; use tools", "function_call": "; use tool_choice"}.get(name, "")
            raise Refusal(name, f"{name} is not supported by this engine{hint}")
    n = _int(body, "n", 1, MAX_N, 1)
    if n > 1 and body.get("stream"):
        raise Refusal("n", "n > 1 is served for non-streamed requests only")
    rf = body.get("response_format")
    if rf is not None:
        kind = rf.get("type") if isinstance(rf, dict) else None
        ok = ("text", "json_object", "json_schema") if structured else ("text",)
        if kind not in ok:
            raise Refusal("response_format", f"response_format type {kind!r} is not supported "
                                             f"({', '.join(ok)})")
    if chat:
        mods = body.get("modalities")
        if mods is not None and list(mods) != ["text"]:
            raise Refusal("modalities", "text is the only modality this engine serves")
        lp = body.get("logprobs")
        if lp is not None and not isinstance(lp, bool):
            raise Refusal("logprobs", "logprobs must be true or false")
        _int(body, "top_logprobs", 0, MAX_TOP_LOGPROBS, 0)
        if body.get("top_logprobs") and not lp:
            raise Refusal("top_logprobs", "top_logprobs needs logprobs: true")
        ptc = body.get("parallel_tool_calls")
        if ptc is not None and not isinstance(ptc, bool):
            raise Refusal("parallel_tool_calls", "parallel_tool_calls must be true or false")
        tools_shape(body)
        tool_choice(body)
    else:
        _int(body, "logprobs", 0, MAX_COMPLETION_LOGPROBS, 0)
        if body.get("echo"):
            raise Refusal("echo", "echo is not supported by this engine")
        if _int(body, "best_of", 1, MAX_N, 1) not in (1, n):
            raise Refusal("best_of", "best_of must equal n: the engine returns every choice")


def json_answer(body: dict) -> bool:
    """SRV-169: the request asks for a JSON answer (`response_format` json_object or
    json_schema), the one shape whose content a client parses whole."""
    rf = body.get("response_format")
    return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")


def client_formats(spec: str | None = None) -> dict:
    """`QWEN38_CLIENT_REASONING` (`kind=format,...`) as {client kind: format}; an unknown format
    is left out (the server starts, and that client keeps the server's format)."""
    if spec is None:
        from engine.settings import SETTINGS as _S
        spec = _S.get("CLIENT_REASONING")
    out = {}
    for part in (spec or "").split(","):
        kind, _, fmt = part.partition("=")
        if kind.strip() and fmt.strip() in ("tags", "reasoning_content", "both"):
            out[kind.strip()] = fmt.strip()
    return out


def answer_format(body: dict, fmt: str, json_constrained: bool, client_kind: str | None = None,
                  per_client: dict | None = None) -> str:
    """SRV-169 (UT1, 2026-10-06): where the reasoning goes for this request. A JSON answer's
    content is parsed whole (`json.loads(message.content)`, the SDKs' structured-output helpers),
    and with the `tags` default it opened with `<think>...</think>`; so a constrained JSON answer
    whose client did not name a `reasoning_format` gets `reasoning_content` (the thinking in its
    own field, the content the JSON alone). A client's own `reasoning_format` always wins.

    SRV-193 (FX2's live check, 2026-10-06): a client kind can get its own format when it names
    none and the server's is `tags` (`per_client`, `QWEN38_CLIENT_REASONING`, `*` for every other
    kind; default since SRV-245 every kind -> `reasoning_content`). Open WebUI 0.11.3 folds `<think>` blocks after the fact: the piece
    that carries `</think>` reaches its live view raw, as the text of a new message item, and stays
    there until the round's stream ends (15 of 70 page scans of a long agent chat showed a literal
    `</think>`). Its `reasoning_content` path has no tag to show."""
    if json_constrained and not body.get("reasoning_format") and fmt in ("tags", "both"):
        return "reasoning_content"
    if not body.get("reasoning_format") and fmt == "tags" and client_kind:
        per = client_formats() if per_client is None else per_client
        return per.get(client_kind, per.get("*", fmt))
    return fmt


#: SRV-245: the client kinds that are plain API callers (server/ledger.py `client_of`): no
#: reasoning display of their own, so a state line in `content` becomes part of their answer
PLAIN_CLIENTS = ("openai-sdk", "curl", "other")


def top_logprobs(body: dict, chat: bool) -> int | None:
    """None when the request did not ask for log-probabilities, else how many alternatives."""
    if chat:
        return int(body.get("top_logprobs") or 0) if body.get("logprobs") else None
    lp = body.get("logprobs")
    return None if lp is None else int(lp)


def tools_shape(body: dict) -> None:
    """`tools` is a list of tool objects: `{"type": "function", "function": {"name": ...,
    "parameters": {...}}}`, or the flat `{"name": ..., "parameters": {...}}` the chat template also
    reads. Anything else used to reach the template and come back as an error frame with a
    traceback in the log (NP1's night pass, 2026-10-04: a fixture file's whole document sent as
    `tools`); now it is a 400 that names the field and the item."""
    tools = body.get("tools")
    if tools is None:
        return
    if not isinstance(tools, list):
        raise Refusal("tools", f"tools must be a list of tool objects, got {type(tools).__name__}")
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise Refusal("tools", f"tools[{i}] must be an object, got {type(t).__name__}")
        fn = t.get("function", t) if "function" in t else t
        where = f"tools[{i}].function" if "function" in t else f"tools[{i}]"
        if not isinstance(fn, dict):
            raise Refusal("tools", f"{where} must be an object, got {type(fn).__name__}")
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise Refusal("tools", f"{where}.name must be a non-empty string")
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise Refusal("tools", f"{where}.parameters must be a JSON Schema object, "
                                   f"got {type(params).__name__}")


def tool_names(body: dict) -> list[str]:
    out = []
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append(fn["name"])
    return out


def tool_choice(body: dict) -> tuple[str, str | None]:
    """`(mode, name)`: auto / none / required / named, validated against the request's tools."""
    tc = body.get("tool_choice")
    if tc is None or tc == "auto":
        return "auto", None
    if tc == "none":
        return "none", None
    names = tool_names(body)
    if tc == "required":
        if not names:
            raise Refusal("tool_choice", "tool_choice required needs tools")
        return "required", None
    if isinstance(tc, dict) and tc.get("type") == "function" and isinstance(tc.get("function"),
                                                                              dict):
        name = tc["function"].get("name")
        if not isinstance(name, str) or name not in names:
            raise Refusal("tool_choice", f"tool_choice names {name!r}, which is not in tools")
        return "named", name
    raise Refusal("tool_choice", "tool_choice must be auto, none, required or "
                                 '{"type": "function", "function": {"name": ...}}')


def directive(mode: str, name: str | None) -> str | None:
    """The prompt-level enforcement: the sentence the system turn ends with, or None."""
    if mode == "required":
        return "You must call at least one of the functions above in this reply."
    if mode == "named":
        return f"You must call the function {name} in this reply."
    return None


def logit_bias(body: dict, vocab: int) -> dict[int, float] | None:
    """`{token id: bias}` from OpenAI's `{"id": value}`, ids in the vocabulary, values in
    [-100, 100]; None when absent or empty."""
    raw = body.get("logit_bias")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise Refusal("logit_bias", "logit_bias must map token ids to numbers")
    out: dict[int, float] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            raise Refusal("logit_bias", f"logit_bias key {k!r} is not a token id") from None
        if not 0 <= tid < vocab:
            raise Refusal("logit_bias", f"logit_bias token id {tid} is outside the vocabulary")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not -100 <= v <= 100:
            raise Refusal("logit_bias", f"logit_bias value for {tid} must be in [-100, 100]")
        if v:
            out[tid] = float(v)
    return out or None


def structured_pattern(body: dict) -> str | None:
    """The regex a request constrains its answer to, or None.

    `response_format` `json_object` (any object, nested three deep) or `json_schema`; or the
    engine's own `structured_outputs` (vLLM's field): `{"regex": ...}`, `{"choice": [...]}`,
    `{"json": <schema>}`. One constraint a request. The document alone: with tools offered,
    `answer_constraint` joins it with the calls (SRV-619).
    """
    from engine import grammar
    rf = body.get("response_format")
    kind = rf.get("type") if isinstance(rf, dict) else None
    so = body.get("structured_outputs")
    if so is not None and kind in ("json_object", "json_schema"):
        raise Refusal("structured_outputs", "one constraint a request: response_format or "
                                            "structured_outputs")
    try:
        if kind == "json_object":
            pattern = grammar.json_object_regex(3)
        elif kind == "json_schema":
            js = rf.get("json_schema")
            schema = js.get("schema") if isinstance(js, dict) else None
            if not isinstance(schema, dict):
                raise Refusal("response_format", "json_schema needs json_schema.schema")
            pattern = grammar.WS + grammar.schema_regex(schema) + grammar.WS
        elif so is not None:
            if not isinstance(so, dict) or len(so) != 1:
                raise Refusal("structured_outputs", 'structured_outputs is one of {"regex": ...}, '
                                                    '{"choice": [...]}, {"json": <schema>}')
            (k, v), = so.items()
            if k == "regex" and isinstance(v, str):
                pattern = v
            elif k == "choice" and isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                pattern = "(?:" + "|".join(grammar.literal(x) for x in v) + ")"
            elif k == "json" and isinstance(v, dict):
                pattern = grammar.WS + grammar.schema_regex(v) + grammar.WS
            else:
                raise Refusal("structured_outputs", f"structured_outputs.{k} is not supported")
        else:
            return None
    except grammar.GrammarError as exc:
        raise Refusal("structured_outputs" if so is not None else "response_format", str(exc))
    return pattern


def constraint_field(body: dict) -> str:
    """The field a request's constraint came in, for the 400 that names it."""
    return "structured_outputs" if body.get("structured_outputs") is not None else "response_format"


def tool_constraint(body: dict) -> tuple[str, dict] | None:
    """`tool_choice` required or a named function, enforced -- the answer (after the
    reasoning) as tool calls only, in the model's own format: an allowed function (the named one,
    or any tool for required), the schema's parameters with the required ones present, and typed
    values. One call, unless the request says `parallel_tool_calls: true`: a model forced to call
    when it meant to answer in words otherwise opens call after call (37 in the run).
    Returns (regex, the parameters written as JSON literals per function), or None for auto and
    none."""
    from engine import grammar
    mode, name = tool_choice(body)
    if mode not in ("required", "named") or not body.get("tools"):
        return None
    allowed = [name] if mode == "named" else tool_names(body)
    try:
        return grammar.tool_call_regex(body["tools"], allowed,
                                       many=body.get("parallel_tool_calls") is True)
    except grammar.GrammarError as exc:
        raise Refusal("tools", f"tool_choice {mode}: {exc}")


class Answer(NamedTuple):
    """The answer's constraint: its regex, the field it came from (`rec.constrained`), the
    parameters written as JSON literals per function, and whether a tool-call branch sits beside
    a document (the vocabulary then keeps the call markers' own ids; `parts`, the two apart: the
    model picks the branch, `engine/grammar.Grammar`)."""
    pattern: str
    kind: str                       # "response_format" | "tool_choice"
    types: dict | None
    calls: bool
    parts: tuple[str, str] | None = None


def answer_constraint(body: dict, chat: bool = True, xml_calls: bool = True) -> Answer | None:
    """The constraint on the answer (after the reasoning), or None.

    `tool_choice` required or named: the calls (`tool_constraint`), a document asked for beside
    them checked and left out (the turn calls). No tools, or `tool_choice` none: the document
    (`structured_pattern`). SRV-619: tools under auto with a document: either, as one
    automaton -- the calls in the model's format (any offered function, more than one unless
    `parallel_tool_calls` is false) or the document alone, the branch the model's own first answer
    token picks (a call's start: the calls; anything else: the document), so an agent loop that asks for its
    final answer as JSON keeps its tools. `xml_calls` false (a model whose calls are not the XML
    format, `--tool-format pythonic`): the 400 as before.
    """
    from engine import grammar
    doc = structured_pattern(body)
    forced = tool_constraint(body) if chat else None
    if forced is not None:
        return Answer(forced[0], "tool_choice", forced[1] or None, False)
    if doc is None:
        return None
    if not chat or not body.get("tools") or tool_choice(body)[0] == "none":
        return Answer(doc, "response_format", None, False)
    if not xml_calls:
        raise Refusal("tools", "structured outputs with tools need the XML tool-call format; "
                               "send tool_choice none or no tools")
    # the flat `{"name": ..., "parameters": ...}` form read as a function tool
    fns = [t if "function" in t else {"type": "function", "function": t} for t in body["tools"]]
    names = [t["function"]["name"] for t in fns]
    try:
        calls, typed = grammar.tool_call_regex(fns, names,
                                               many=body.get("parallel_tool_calls") is not False)
    except grammar.GrammarError as exc:
        raise Refusal("tools", f"structured outputs with tools: {exc}")
    return Answer(grammar.calls_or_document(calls, doc), "response_format", typed or None, True,
                  (calls, doc))

