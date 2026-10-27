"""The session monitor's policy: modes, the action set, the safe points, and the policy file.

Every rule, threshold, text and mode lives in the policy file (`ops/supervisor.yaml`), so all of
it can be managed without a code change. The file is validated against a JSON Schema built from
the rule registry (`tier0.RULES`): the top level is fixed here, and each task's parameters are the
schema its rule declares. `ops/supervisor.schema.json` is that schema written out, for the config
API and the dashboard; `python tools/conductor_policy.py --schema` regenerates it and
`tests/test_supervisor_frame.py` fails when the two differ.

Modes, from least to most:

    raw     nothing runs; the served output is the raw engine's (`--conductor off` means the same)
    shadow  every rule runs and every decision is journaled; nothing is executed
    on      decisions are executed, each task in its own mode (off, shadow or act)

A request's `X-Tandem` header can lower the server's mode and never raise it.

No torch here: events in, actions out.
"""

from __future__ import annotations

import os

MODES = ("raw", "shadow", "on")
ALIASES = {"off": "raw"}
TASK_MODES = ("off", "shadow", "act")

# The fixed action set. Rules return one of these; only the executor carries them out, and only
# at the safe point that allows it (SAFE_POINTS).
PASS = "PASS"
CLOSE_REASONING = "CLOSE_REASONING"
CONTINUE_HIDDEN = "CONTINUE_HIDDEN"
REPAIR_CALL = "REPAIR_CALL"
REGENERATE_CALL = "REGENERATE_CALL"
PROMPT_TAIL_NOTE = "PROMPT_TAIL_NOTE"
SET_THINK_BUDGET = "SET_THINK_BUDGET"
NOTIFY = "NOTIFY"
GIVE_UP = "GIVE_UP"
ACTIONS = (PASS, CLOSE_REASONING, CONTINUE_HIDDEN, REPAIR_CALL, REGENERATE_CALL,
           PROMPT_TAIL_NOTE, SET_THINK_BUDGET, NOTIFY, GIVE_UP)

# The safe points, all on the request's own thread, and what may happen at each. The TandemLLM
# notes call the request start SP0; the tickets call it SP4.
SAFE_POINTS = {
    "SP1": ("block", (CLOSE_REASONING, GIVE_UP)),
    "SP2": ("finish", (CONTINUE_HIDDEN, GIVE_UP)),
    "SP3": ("tool_call", (REPAIR_CALL, REGENERATE_CALL, GIVE_UP)),
    "SP4": ("request", (PROMPT_TAIL_NOTE, SET_THINK_BUDGET)),
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_PATH = os.path.join(ROOT, "ops", "supervisor.yaml")
SCHEMA_PATH = os.path.join(ROOT, "ops", "supervisor.schema.json")


class PolicyError(ValueError):
    """The policy file does not load or does not validate; the message names the field."""


class Action:
    """One decision: what to do, which task decided it and why.

    `text` is the hidden text the executor forwards (a close phrase, a nudge, a note) or, for
    GIVE_UP, the visible sentence; it always comes from the policy file. `budget` is
    SET_THINK_BUDGET's reasoning cap (0 = thinking off). `detail` is journaled as it is.

    The supervisor fills the token ids before the executor sees the action: `ids` are forwarded
    through the engine, and `visible` -- a suffix of `ids` -- is what the client receives of
    them (the reasoning closer of a hidden close, nothing for a hidden turn)."""

    __slots__ = ("kind", "task", "reason", "text", "budget", "score", "detail", "ids", "visible")

    def __init__(self, kind: str, reason: str = "", text: str = "", budget: int | None = None,
                 score: float | None = None, detail: dict | None = None, task: str = ""):
        if kind not in ACTIONS:
            raise ValueError(f"unknown action {kind!r}")
        self.kind, self.reason, self.text, self.budget = kind, reason, text, budget
        self.score, self.detail, self.task = score, detail or {}, task
        self.ids: list[int] = []
        self.visible: list[int] = []

    @property
    def hidden(self) -> int:
        """Tokens the engine forwards that the client never receives."""
        return len(self.ids) - len(self.visible)

    def __repr__(self):
        return f"Action({self.kind}, task={self.task!r}, reason={self.reason!r})"


def parse_mode(value, default: str | None = None) -> str | None:
    """`raw`, `shadow` or `on` (`off` is `raw`); `default` for anything else."""
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    v = ALIASES.get(v, v)
    return v if v in MODES else default


def lower(server: str, requested) -> str:
    """The request's mode: the server's, lowered by the header, never raised."""
    want = parse_mode(requested)
    if want is None:
        return server
    return MODES[min(MODES.index(server), MODES.index(want))]


class Task:
    """One task of the policy: an id, the rule that decides it, its mode and its parameters."""

    __slots__ = ("id", "rule", "mode", "params")

    def __init__(self, tid: str, cfg: dict):
        self.id, self.rule, self.mode = tid, cfg["rule"], cfg["mode"]
        self.params = {k: v for k, v in cfg.items() if k not in ("rule", "mode")}

    def __getitem__(self, key):
        return self.params[key]

    def get(self, key, default=None):
        return self.params.get(key, default)


class Policy:
    """A validated policy. Read-only once built; a reload builds a new one, and a request keeps
    the one it started with."""

    def __init__(self, data: dict, path: str | None = None):
        validate(data, schema())
        self.data = data
        self.path = path
        self.version = data["version"]
        self.caps = data["caps"]
        self.session = data["session"]
        self.texts = data["texts"]
        self.templates = data["templates"]
        self.journal = data["journal"]
        self.tasks = {tid: Task(tid, cfg) for tid, cfg in data["tasks"].items()}
        from server.conductor.monitor import tier0
        for t in self.tasks.values():
            sp = tier0.RULES[t.rule].safe_point
            if sp not in SAFE_POINTS:
                raise PolicyError(f"tasks.{t.id}: rule {t.rule} has no safe point")

    def at(self, safe_point: str) -> list[Task]:
        """The tasks that run at a safe point and are not off, in file order."""
        from server.conductor.monitor import tier0
        return [t for t in self.tasks.values()
                if t.mode != "off" and tier0.RULES[t.rule].safe_point == safe_point]


# the policy file as JSON, shipped inside the package (an installed proxy has neither the
# repository's ops/ nor PyYAML); `tools/conductor_policy.py --json` writes it and
# tests/test_conductor_config.py fails when it differs from ops/supervisor.yaml
DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.default.json")


def load_default(path: str | None = None) -> Policy:
    """`path`, else ops/supervisor.yaml when PyYAML and the file are there (a checkout), else the
    JSON copy shipped in the package."""
    if path is not None:
        return load(path)
    try:
        import yaml  # noqa: F401
        if os.path.exists(DEFAULT_PATH):
            return load(DEFAULT_PATH)
    except ImportError:
        pass
    import json
    try:
        with open(DEFAULT_JSON) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise PolicyError(f"{DEFAULT_JSON}: {exc}") from exc
    return Policy(data, DEFAULT_JSON)


def load(path: str | None = None) -> Policy:
    """The policy in `path` (default `ops/supervisor.yaml`), validated."""
    import yaml
    path = path or DEFAULT_PATH
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"{path}: {exc}") from exc
    return Policy(data, path)


# ------------------------------------------------------------------ the schema

def schema() -> dict:
    """The JSON Schema of the policy file: the fixed top level, and each task as the schema its
    rule declares (one `oneOf` branch per registered rule)."""
    from server.conductor.monitor import tier0
    rules = []
    for name in sorted(tier0.RULES):
        r = tier0.RULES[name]
        rules.append({
            "type": "object",
            "description": f"{r.safe_point}: {r.doc}",
            "properties": dict({"rule": {"const": name},
                                "mode": {"enum": list(TASK_MODES)}}, **r.params),
            "required": ["rule", "mode"] + list(r.required),
            "additionalProperties": False,
        })
    text = {"type": "string"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TandemLLM session monitor policy",
        "type": "object",
        "properties": {
            "version": {"type": "integer", "minimum": 1,
                        "description": "bumped by every change, recorded with every decision"},
            "caps": {
                "type": "object",
                "properties": {
                    "acts_per_request": {"type": "integer", "minimum": 0},
                    "acts_per_session_hour": {"type": "integer", "minimum": 0},
                    "same_action_in_a_row": {
                        "type": "boolean",
                        "description": "false: the same hidden action (kind and text) twice in a "
                                       "row on one turn is refused and ends in GIVE_UP"},
                },
                "required": ["acts_per_request", "acts_per_session_hour", "same_action_in_a_row"],
                "additionalProperties": False,
            },
            "session": {
                "type": "object",
                "properties": {
                    "lineage_tokens": {"type": "integer", "minimum": 1,
                                       "description": "prompt tokens hashed into the session key "
                                                      "when the client sends no conversation id"},
                    "join_window_s": {"type": "number", "minimum": 0},
                    "max_sessions": {"type": "integer", "minimum": 1},
                },
                "required": ["lineage_tokens", "join_window_s", "max_sessions"],
                "additionalProperties": False,
            },
            "texts": {
                "type": "object",
                "properties": {"give_up": text},
                "required": ["give_up"],
                "additionalProperties": text,
            },
            "templates": {
                "type": "object",
                "properties": {
                    "hidden_turn": dict(text, description="CONTINUE_HIDDEN after an empty answer: "
                                                          "{text} is the nudge"),
                    "tail_note": dict(text, description="PROMPT_TAIL_NOTE, appended to the last "
                                                        "message: {text} is the note"),
                    "think_open": dict(text, description="the generation prompt's tail with "
                                                         "thinking on"),
                    "think_off": dict(text, description="the same tail with thinking off"),
                    "think_close": dict(text, description="the reasoning block's closer; what a "
                                                          "hidden close lets the client see "
                                                          "starts here"),
                    "turn_start": dict(text, description="the token that opens a chat turn"),
                    "turn_end": dict(text, description="the token that ends a chat turn"),
                    "regen_turn": dict(text, description="REGENERATE_CALL after the answer's "
                                                         "calls failed a check: {text} is the "
                                                         "reminder, in the shape of a tool "
                                                         "result"),
                },
                "required": ["hidden_turn", "tail_note", "think_open", "think_off",
                             "think_close", "turn_start", "turn_end", "regen_turn"],
                "additionalProperties": False,
            },
            "journal": {
                "type": "object",
                "properties": {
                    "tool_passes": {"type": "boolean",
                                    "description": "one journal row for every answer whose calls "
                                                   "passed every check (a finding is always "
                                                   "journaled; passes are counted either way)"},
                },
                "required": ["tool_passes"],
                "additionalProperties": False,
            },
            "tasks": {"type": "object", "additionalProperties": {"oneOf": rules}},
        },
        "required": ["version", "caps", "session", "texts", "templates", "journal", "tasks"],
        "additionalProperties": False,
    }


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(value, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[kind])


def _check(value, s: dict, where: str) -> str | None:
    """The first violation of `s` by `value` as a message, or None. The subset of JSON Schema
    `schema()` uses: type, enum, const, minimum, maximum, exclusiveMinimum, minLength, minItems,
    items, properties, required, additionalProperties, oneOf."""
    if "oneOf" in s:
        errs = [_check(value, sub, where) for sub in s["oneOf"]]
        ok = [e for e in errs if e is None]
        if len(ok) == 1:
            return None
        if not ok and isinstance(value, dict) and "rule" in value:
            # name the rule's own error, not every branch's
            for sub, e in zip(s["oneOf"], errs):
                if sub.get("properties", {}).get("rule", {}).get("const") == value["rule"]:
                    return e
            return f"{where}.rule: unknown rule {value['rule']!r}"
        return f"{where}: matches {len(ok)} of the allowed shapes, not exactly one"
    if "type" in s:
        kinds = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_is(value, k) for k in kinds):
            return f"{where}: expected {'/'.join(kinds)}, got {type(value).__name__}"
    if "const" in s and value != s["const"]:
        return f"{where}: must be {s['const']!r}"
    if "enum" in s and value not in s["enum"]:
        return f"{where}: must be one of {s['enum']}, not {value!r}"
    if _is(value, "number"):
        if "minimum" in s and value < s["minimum"]:
            return f"{where}: {value} is below {s['minimum']}"
        if "maximum" in s and value > s["maximum"]:
            return f"{where}: {value} is above {s['maximum']}"
        if "exclusiveMinimum" in s and value <= s["exclusiveMinimum"]:
            return f"{where}: {value} must be above {s['exclusiveMinimum']}"
    if isinstance(value, str) and len(value) < s.get("minLength", 0):
        return f"{where}: shorter than {s['minLength']}"
    if isinstance(value, list):
        if len(value) < s.get("minItems", 0):
            return f"{where}: fewer than {s['minItems']} items"
        if "items" in s:
            for i, v in enumerate(value):
                e = _check(v, s["items"], f"{where}[{i}]")
                if e:
                    return e
    if isinstance(value, dict):
        for k in s.get("required", ()):
            if k not in value:
                return f"{where}.{k}: missing"
        props = s.get("properties", {})
        extra = s.get("additionalProperties", True)
        for k, v in value.items():
            if not isinstance(k, str):
                return f"{where}: key {k!r} is not a string"
            if k in props:
                e = _check(v, props[k], f"{where}.{k}")
            elif extra is False:
                e = f"{where}.{k}: not a known field"
            elif isinstance(extra, dict):
                e = _check(v, extra, f"{where}.{k}")
            else:
                e = None
            if e:
                return e
    return None


def validate(data, s: dict | None = None) -> None:
    """Raise `PolicyError` naming the first field of `data` that `s` refuses."""
    err = _check(data, s or schema(), "policy")
    if err:
        raise PolicyError(err)

