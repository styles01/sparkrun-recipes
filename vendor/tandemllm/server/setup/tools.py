"""Wren's tools (SRV-325): what the assistant may ask the setup page to do, and the check of a call.

The browser executes every call; the server never applies one. The server checks a call against the
model's step schema (`registry.wizard_fields`) before it streams it, and the browser checks it again
before it acts. There is no tool to finish, apply, reset, delete, load or unload anything: those are
the user's buttons.

    set_field(step, field, value)   a field of a step (the step's id, or its index 0..7); the model
                                    may leave the step out (the field names are unique), and the
                                    call that reaches the browser always names it
    toggle(component, on)           a tool beside the model on or off
    select_profile(id)              the weight profile
    go_to_step(n)                   step n, 0..7
    next(), previous()              one step on or back
    explain(field)                  point at a field, a tool or a profile while Wren explains it;
                                    a step is not a field (VIS-91: the 2B called explain("overview")
                                    after every answer, the page could not point at a step, and the
                                    browser asked again four times)
"""

from __future__ import annotations

import json
import re

from server.setup import registry

NAMES = ("set_field", "toggle", "select_profile", "go_to_step", "next", "previous", "explain")
PARALLEL = ("slots", "window", "side_window", "kv_pool", "floor_gb")


def explain_ids(m: dict) -> list[str]:
    """What the page can highlight, the same set as planToolCall's explain in tools.ts: a tool, a
    method or advanced field, a weight profile, a field of the parallelism step."""
    return sorted({c["id"] for c in m["components"]} | {f["id"] for f in m["methods"] + m["advanced"]}
                  | {p["id"] for p in m["profiles"]} | set(PARALLEL))


def step_ids(m: dict, step) -> list[str]:
    """The ids a call may name on step `step`: the profiles on weights, the tools on components,
    the fields on parallelism and methods; none on the others."""
    if step == "weights":
        return [p["id"] for p in registry.profiles(m)]
    for s in registry.wizard_fields(m):
        if s["id"] == step and s["id"] != "weights":
            return [f["field"] for f in s["fields"]]
    return []


def definitions(m: dict) -> list[dict]:
    """The OpenAI tool definitions Wren is given, with the model's own ids in the enums."""
    steps = registry.wizard_fields(m)
    step_ids = [s["id"] for s in steps if s["fields"]]
    comps = [c["id"] for c in m["components"] if not c["required"]]

    def fn(name, description, props=None, required=()):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props or {},
                           "required": list(required), "additionalProperties": False}}}

    fields = "; ".join(f"{s['id']}: {', '.join(f['field'] for f in s['fields'])}"
                       for s in steps if s["fields"] and s["id"] != "components")
    return [
        fn("set_field", "Set a field to a value. The fields, by step: "
           f"{fields}. For a tool use toggle, for the weights select_profile.",
           {"field": {"type": "string"},
            "value": {"type": ["string", "number", "boolean"]},
            "step": {"type": "string", "enum": step_ids}}, ("field", "value")),
        fn("toggle", "Switch a tool beside the model on or off. The tools: "
           f"{', '.join(comps)}. Slots, windows and the KV pool are fields: set_field.",
           {"component": {"type": "string", "enum": comps}, "on": {"type": "boolean"}},
           ("component", "on")),
        fn("select_profile", "Pick the weight profile.",
           {"id": {"type": "string", "enum": [p["id"] for p in registry.profiles(m)]}}, ("id",)),
        fn("go_to_step", "Open a step by its number: "
           + ", ".join(f"{s['index']} {s['label'].lower()}" for s in steps) + ".",
           {"n": {"type": "integer", "minimum": 0, "maximum": len(steps) - 1}}, ("n",)),
        fn("next", "Go to the next step."),
        fn("previous", "Go back one step."),
        fn("explain", "Highlight a field, a tool or a weight profile when the user asks where "
           "it is. It changes nothing and looks nothing up, and a step is not a field.",
           {"field": {"type": "string", "enum": explain_ids(m)}}, ("field",)),
    ]


def _value_error(f: dict, value) -> str | None:
    spec = dict(f, options=[{"value": o} for o in f.get("options") or []])
    return registry.field_error(spec, value)


def _coerce(f: dict, value):
    """A value the model wrote as text, as the field's type: "4" for an int field, "true" for a
    bool one (the XML tool-call form carries every parameter as text)."""
    if not isinstance(value, str):
        return value
    v = value.strip()
    if f["type"] == "bool" and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if f["type"] == "int" and re.fullmatch(r"-?\d+", v):
        return int(v)
    if f["type"] == "number" and re.fullmatch(r"-?\d+(\.\d+)?", v):
        return float(v)
    return value


def check(m: dict, name: str, args) -> tuple[dict | None, str | None]:
    """(the call's arguments, None) when `name(args)` is a valid call on model `m`; else
    (None, why) in a sentence the model can act on. `args` may be the JSON text of a tool call."""
    if name not in NAMES:
        return None, f"there is no tool {name!r}; the tools are {', '.join(NAMES)}"
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None, "the arguments are not valid JSON"
    if not isinstance(args, dict):
        return None, "the arguments must be a JSON object"
    want = {"set_field": {"step", "field", "value"}, "toggle": {"component", "on"},
            "select_profile": {"id"}, "go_to_step": {"n"}, "next": set(), "previous": set(),
            "explain": {"field"}}[name]
    if name == "set_field" and "step" not in args and isinstance(args.get("field"), str):
        owner = [s["id"] for s in registry.wizard_fields(m)
                 if any(f["field"] == args["field"] for f in s["fields"])]
        if len(owner) != 1:
            return None, (f"there is no field {args['field']!r} on the page; a tool is toggle, the "
                          f"weights are select_profile")
        args = dict(args, step=owner[0])
    if set(args) != want:
        return None, (f"{name} takes {', '.join(sorted(want)) or 'no arguments'}, "
                      f"not {', '.join(sorted(args)) or 'none'}")
    steps = registry.wizard_fields(m)
    if name == "set_field":
        step = args["step"]
        if isinstance(step, str) and re.fullmatch(r"\d", step):
            step = int(step)
        f = registry.step_field(m, step, args["field"]) if isinstance(args["field"], str) else None
        if f is None:
            return None, f"step {args['step']!r} has no field {args['field']!r}"
        value = _coerce(f, args["value"])
        why = _value_error(f, value)
        if why:
            return None, f"{args['field']}: {why}"
        return dict(args, value=value), None
    if name == "toggle":
        c = registry.component(m, args["component"]) if isinstance(args["component"], str) else None
        if c is None:
            return None, f"there is no tool {args['component']!r}"
        if not isinstance(args["on"], bool):
            return None, "on must be true or false"
        if c["required"] and not args["on"]:
            return None, f"{c['label']} cannot be switched off"
        return args, None
    if name == "select_profile":
        if not isinstance(args["id"], str) or registry.profile(m, args["id"]) is None:
            return None, (f"there is no weight profile {args['id']!r}; they are "
                          f"{', '.join(p['id'] for p in m['profiles'])}")
        return args, None
    if name == "go_to_step":
        n = args["n"]
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < len(steps):
            return None, f"n is a step number from 0 to {len(steps) - 1}"
        return args, None
    if name == "explain":
        if not isinstance(args["field"], str) or args["field"] not in explain_ids(m):
            return None, (f"there is nothing called {args['field']!r} on the page to highlight; a "
                          f"step is not a field, and a question needs no tool")
        return args, None
    return args, None                                   # next, previous
