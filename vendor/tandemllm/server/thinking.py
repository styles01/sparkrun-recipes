"""A client's thinking budget, from every request shape it arrives in, read in one place.

Clients ask for a reasoning budget in many shapes, and the server used to read two of them
(`max_reasoning_tokens`, `thinking_budget`) plus `reasoning_effort` as a template word. A client
whose shape was not read got no cap, and a turn whose reasoning ran into `max_tokens` ended inside
the block: the client got reasoning and no answer (the Hermes agent, 2026-10-02: its effort word,
the whole output limit spent thinking, twice). Every shape is read here into one `Client`;
server/app.py and the router read that, never the body's fields.

The shapes, with the kind each one is:

  * a number (`number`): `max_reasoning_tokens`, `thinking_budget` (DashScope's), then the same in
    `chat_template_kwargs` (`thinking_budget`, `max_thinking_tokens`, `reasoning_budget`,
    `thinking_token_budget`), then OpenRouter's `reasoning.max_tokens`, then Anthropic's
    `thinking.budget_tokens`, then the other servers' top-level names (`reasoning_budget`,
    `max_thinking_tokens`, `thinking_token_budget`, `thinking_budget_tokens`). 0 is no cap, as it
    always was for `max_reasoning_tokens`;
  * an effort word (`effort`): `chat_template_kwargs.reasoning_effort`, then `reasoning_effort`
    (OpenAI's), then `reasoning.effort` (OpenRouter's). none, minimal, low, medium, high, xhigh
    and max map to the effort levels (none is off, minimal is low, max is xhigh) and a level to its
    cap: low 1,024, medium 8,192, high 16,384, xhigh none (the router document's levels,
    server/conductor/schema.py, SRV-159);
  * a switch (`switch`): `reasoning.enabled: true` alone is medium (OpenRouter's reading), and
    `thinking.type` `enabled` or `adaptive` without a number is thinking on, no cap;
  * thinking off: `chat_template_kwargs.enable_thinking: false`, `enable_thinking: false` (top
    level, DashScope's), `reasoning.enabled: false`, effort `none`, `thinking.type: "disabled"`.

Precedence. Whether the model thinks is decided first: `chat_template_kwargs.enable_thinking`
when the client set it (the template's own switch, honoured before this module existed), else the
top-level `enable_thinking`, else any off form, else on. The cap is then the first number in the
order above, else the first effort word's cap, else a switch's. A number is the most specific a
client can be, so it beats a word, which beats a switch.

`enable_thinking` alone asks for no budget, and a request with none of these fields reads as
`Client()`: nothing asked, served exactly as before. `reasoning.exclude` is read and not honoured
(the reasoning goes out in the request's `reasoning_format`).

The answer reserve: a client that asked for a budget in any shape gets its reasoning closed early
enough to leave `reserve()` tokens of the output limit for the answer after the closing phrase, so
the limit never ends the turn inside the block.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.compat import Refusal
from server.conductor import schema

# the effort words clients send, and the effort level each one is (server/conductor/schema.py)
WORDS = {"none": "off", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
         "xhigh": "xhigh", "max": "xhigh"}
# the levels the chat template renders a sentence for; it raises on any other word, so `high`
# leaves the template at the server's default and is the cap alone
TEMPLATE_WORDS = ("low", "medium", "xhigh")

# the numbers, in precedence order: (where, key); `where` None is the top level
NUMBERS = ((None, "max_reasoning_tokens"), (None, "thinking_budget"),
           ("chat_template_kwargs", "thinking_budget"),
           ("chat_template_kwargs", "max_thinking_tokens"),
           ("chat_template_kwargs", "reasoning_budget"),
           ("chat_template_kwargs", "thinking_token_budget"),
           ("reasoning", "max_tokens"), ("thinking", "budget_tokens"),
           (None, "reasoning_budget"), (None, "max_thinking_tokens"),
           (None, "thinking_token_budget"), (None, "thinking_budget_tokens"))
EFFORTS = (("chat_template_kwargs", "reasoning_effort"), (None, "reasoning_effort"),
           ("reasoning", "effort"))
MAX_BUDGET = 1 << 20

# the reserve's default: min(RESERVE_MAX, a RESERVE_SHARE of the output limit)
AUTO = -1
RESERVE_MAX = 2048
RESERVE_SHARE = 4


@dataclass(frozen=True)
class Client:
    budget: int | None = None        # the cap asked for in reasoning tokens; 0 no cap; None none
    kind: str | None = None          # number | effort | switch: what the cap came from
    source: str | None = None        # the field it came from, e.g. "reasoning.max_tokens"
    effort: str | None = None        # the effort level a word asked for (off, low, ..., xhigh)
    thinking: bool | None = None     # an explicit switch: False off, True on, None not said
    exclude: bool = False            # OpenRouter's `reasoning.exclude` (read, not honoured)

    @property
    def asked(self) -> bool:
        """The client asked for a budget in some shape (thinking on)."""
        return self.budget is not None

    @property
    def off(self) -> bool:
        return self.thinking is False

    @property
    def template_effort(self) -> str | None:
        """The word for the chat template's `reasoning_effort`, None to leave the default."""
        return self.effort if self.effort in TEMPLATE_WORDS else None


def caps(levels: dict | None = None) -> dict:
    """{effort level: its cap, 0 no cap} from the router document's levels (else the seed's)."""
    levels = levels or schema.ROUTER["effort"]["levels"]
    return {k: int(v.get("budget") or 0) for k, v in levels.items() if v.get("thinking")}


def _field(body: dict, where: str | None, key: str):
    obj = body if where is None else body.get(where)
    return obj.get(key) if isinstance(obj, dict) else None


def _name(where: str | None, key: str) -> str:
    return key if where is None else f"{where}.{key}"


def _number(v, name: str) -> int:
    if isinstance(v, bool):
        raise Refusal(name, f"{name} must be a number of tokens, got {v!r}")
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise Refusal(name, f"{name} must be a number of tokens, got {v!r}") from None
    if not 0 <= n <= MAX_BUDGET:
        raise Refusal(name, f"{name} must be in [0, {MAX_BUDGET}], got {v!r}")
    return n


def _word(v, name: str) -> str:
    w = v.strip().lower() if isinstance(v, str) else None
    if w not in WORDS:
        raise Refusal(name, f"{name} must be one of {', '.join(WORDS)}, got {v!r}")
    return WORDS[w]


def _switch(v, name: str) -> bool:
    if not isinstance(v, bool):
        raise Refusal(name, f"{name} must be true or false, got {v!r}")
    return v


def said(body: dict) -> bool:
    """The request says anything about thinking: a switch, an effort word or a budget, in any of
    the shapes `read` takes (an empty value counts as not said)."""
    if not isinstance(body, dict):
        return False
    keys = [(None, "enable_thinking"), ("chat_template_kwargs", "enable_thinking"),
            ("reasoning", "enabled"), *EFFORTS, *NUMBERS]
    if any(_field(body, w, k) not in (None, "") for w, k in keys):
        return True
    t = body.get("thinking")
    return isinstance(t, bool) or (isinstance(t, dict) and t.get("type") is not None)


def server_default_off(body: dict) -> dict:
    """SRV-477 (`--reasoning-effort none|off`): the body with thinking off when it says nothing
    about thinking, as the template's own switch (`chat_template_kwargs.enable_thinking: false`);
    a body that says anything is returned as it is."""
    if said(body):
        return body
    out = dict(body)
    out["chat_template_kwargs"] = dict(body.get("chat_template_kwargs") or {},
                                       enable_thinking=False)
    return out


def read(body: dict, levels: dict | None = None, strict: bool = True) -> Client:
    """The client's thinking request. `levels`: the effort levels' caps (`caps()`). `strict`:
    a value of the wrong type or out of range is a `Refusal` naming the field; the router reads
    with strict off, after the handler has refused such a request."""
    cap_of = levels if levels is not None else caps()

    def get(fn, where, key):
        v = _field(body, where, key)
        if v is None or v == "":
            return None
        try:
            return fn(v, _name(where, key))
        except Refusal:
            if strict:
                raise
            return None

    for where in ("reasoning", "thinking", "chat_template_kwargs"):
        v = body.get(where)
        if v is not None and not isinstance(v, dict) and not (where == "thinking"
                                                               and isinstance(v, bool)):
            if strict:
                raise Refusal(where, f"{where} must be an object, got {type(v).__name__}")
    thinking = body.get("thinking")
    t_type = None
    if isinstance(thinking, bool):
        t_type = "enabled" if thinking else "disabled"
    elif isinstance(thinking, dict) and thinking.get("type") is not None:
        t_type = thinking.get("type")
        if t_type not in ("enabled", "disabled", "adaptive"):
            if strict:
                raise Refusal("thinking.type", f"thinking.type must be enabled, disabled or "
                                               f"adaptive, got {t_type!r}")
            t_type = None
    r_enabled = get(_switch, "reasoning", "enabled")
    words = [(_name(w, k), get(_word, w, k)) for w, k in EFFORTS]
    words = [(n, lv) for n, lv in words if lv is not None]
    exclude = _field(body, "reasoning", "exclude") is True

    # on or off: the template's switch, the top-level one, then any off form
    on = get(_switch, "chat_template_kwargs", "enable_thinking")
    if on is None:
        on = get(_switch, None, "enable_thinking")
    if on is None and (r_enabled is False or t_type == "disabled"
                       or any(lv == "off" for _, lv in words)):
        on = False
    if on is None and (r_enabled is True or t_type in ("enabled", "adaptive")):
        on = True
    effort = next((lv for _, lv in words), None)
    if on is False:
        return Client(thinking=False, effort=effort, exclude=exclude,
                      source=next((n for n, lv in words if lv == "off"), None))

    for where, key in NUMBERS:
        n = get(_number, where, key)
        if n is not None:
            return Client(n, "number", _name(where, key), effort, on, exclude)
    for name, lv in words:
        if lv != "off":
            return Client(int(cap_of.get(lv) or 0), "effort", name, lv, on, exclude)
    if r_enabled is True:
        return Client(int(cap_of.get("medium") or 0), "switch", "reasoning.enabled", "medium",
                      on, exclude)
    if t_type in ("enabled", "adaptive"):
        return Client(0, "switch", "thinking.type", effort, on, exclude)
    return Client(thinking=on, effort=effort, exclude=exclude)


def parse_reserve(text) -> int:
    """`--answer-reserve`: `auto` (AUTO) or a number of tokens, 0 for none."""
    if str(text).strip().lower() == "auto":
        return AUTO
    n = int(text)
    if n < 0:
        raise ValueError(f"--answer-reserve must be auto or a number >= 0, got {text!r}")
    return n


def reserve(max_new: int, setting: int = AUTO) -> int:
    """The tokens of the output limit `max_new` kept for the answer: `setting`, or with AUTO
    min(2,048, a quarter of the limit); 0 keeps none."""
    if setting is None or setting < 0:
        return min(RESERVE_MAX, max_new // RESERVE_SHARE)
    return int(setting)


def with_reserve(budget: int, max_new: int, setting: int = AUTO,
                 closer: int = 0) -> tuple[int, int]:
    """(the cap, the reserve): `budget` lowered so the reasoning closes with `reserve()` tokens
    of `max_new` left for the answer after the close's own `closer` tokens (the vendor's phrase,
    which streams where no monitor hides it), and never below one token. The cap equals `budget`
    when it already leaves that much; a `budget` of 0 (no cap) becomes the limit less both."""
    r = reserve(max_new, setting)
    if r <= 0:
        return budget, 0
    room = max(1, max_new - r - closer)
    return (budget if budget and budget <= room else room), r
