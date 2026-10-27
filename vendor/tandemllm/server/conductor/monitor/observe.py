"""What the rules see: one event per safe point, built from committed tokens and the request.

An event is plain data. It never carries logits, the KV, a drafter or router state, and a rule
cannot change the request through it; a rule reads the event, the request's `RequestEvent` and
the session state, and returns an action or nothing.

    SP4  RequestEvent   the prompt is rendered, nothing is prefilled yet
    SP1  BlockEvent     a verify block's tokens are committed and yielded
    SP2  FinishEvent    an end token is about to end the stream
    SP3  ToolCallEvent  the answer ended and its calls are held: one event a call (or a block
                        that does not parse), checked before any of them is released
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestEvent:
    """SP4, and the request's context at every later safe point (`Supervisor.req`)."""
    request_id: str
    messages: list = field(default_factory=list)   # the body's messages, read-only
    tools: list = field(default_factory=list)      # the tools the model sees ([] with none)
    tool_choice: str = "auto"                      # auto | none | required | function
    n: int = 1
    stream: bool = False
    prompt_tokens: int = 0                         # as the client sent them
    in_think: bool = False                         # the prompt leaves the model thinking
    max_tokens: int = 0                            # the request's output cap, after the clamp
    client_budget: int | None = None               # the client's cap (server/thinking.py)
    client: str = ""                               # ledger client_kind: opencode, openwebui, ...
    model: str = ""
    constrained: bool = False                      # tool_choice required/named or a schema:
                                                   # the grammar writes the calls, SP3 never acts
    profile: str = ""                              # the conductor's profile, when one resolved
    cls: str = ""                                  # the router's class for the turn (E2 reads it)

    @property
    def tools_offered(self) -> bool:
        return bool(self.tools) and self.tool_choice != "none"


@dataclass
class BlockEvent:
    """SP1. `stall` is the ThinkBudget arm that fired (`loop(..)`, `repeat16(..)`,
    `novelty(..)`), `share` the history arm's last repeat share."""
    reasoning_tokens: int
    inside: bool
    stall: str | None
    share: float | None
    output_tokens: int
    pos: int = 0                                   # tokens in the context (prompt included)


@dataclass
class FinishEvent:
    """SP2: the model wrote its end token. `inside`: the reasoning block was still open.
    `answer`: the text after the block, as written so far (the whole output with thinking off).
    `calls` (SRV-199): with tools offered and an empty answer, the complete `<tool_call>` blocks
    the output ends with (inside the reasoning, nothing but whitespace and the closer after them),
    as text; "" otherwise. The supervisor holds their token ids (`Supervisor.lift_ids`)."""
    kind: str
    inside: bool
    answer: str
    reasoning_tokens: int
    output_tokens: int
    pos: int = 0                                   # where the end token would go
    calls: str = ""


@dataclass
class ToolCallEvent:
    """SP3: one held call. `arguments` is the JSON string a client would receive. A block that
    does not parse is an event too, with `error` (why, from `toolcall.diagnose`), `block` (its
    text) and an empty name; `cut`: the answer ended with it open (the token limit, or with
    `stopped` a stop string the client sent). `schema`: the tool's
    `parameters` from the request, None when it has none. `final`: the answer ended without its
    end token, so nothing can be written again -- a check that would regenerate gives up."""
    name: str
    arguments: str
    index: int
    block: str = ""
    error: str | None = None
    cut: bool = False
    stopped: bool = False
    schema: dict | None = None
    final: bool = False
