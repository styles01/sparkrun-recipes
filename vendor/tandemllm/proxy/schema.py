"""The JSON Schema subset `proxy/config.schema.json` uses, in the standard library.

`type` (one or a list; `integer` excludes booleans), `properties`, `required`,
`additionalProperties: false`, `items`, `enum`, `minimum`, `maximum`, `minLength`, `maxLength`,
`pattern`, `maxItems` and `$ref` to `#/$defs/...`. `validate` returns the violations as
`(path, message)`; the path is the field's dotted name (`timeouts.connect`), which the env overlay
turns into the variable's name and the API into a 422 that names the field.
"""

from __future__ import annotations

import json
import os
import re

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.schema.json")

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

_DOC: dict | None = None


def doc() -> dict:
    global _DOC
    if _DOC is None:
        with open(PATH) as f:
            _DOC = json.load(f)
    return _DOC


def validate(value, schema: dict | str, path: str = "") -> list[tuple[str, str]]:
    if isinstance(schema, str):
        schema = doc()["$defs"][schema]
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return validate(value, doc()["$defs"][name], path)
    out: list[tuple[str, str]] = []
    where = path or "(body)"
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        if not any(_TYPES[t](value) for t in types):
            return [(where, f"must be {' or '.join(types)}")]
    if "enum" in schema and value not in schema["enum"]:
        return [(where, f"must be one of {', '.join(map(str, schema['enum']))}")]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append((where, f"must be at least {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            out.append((where, f"must be at most {schema['maximum']}"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append((where, f"must have at least {schema['minLength']} characters"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append((where, f"must have at most {schema['maxLength']} characters"))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append((where, f"does not match {schema['pattern']}"))
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append((where, f"has more than {schema['maxItems']} items"))
        if "items" in schema:
            for i, item in enumerate(value):
                out += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                out.append((f"{path}.{name}" if path else name, "is required"))
        for name, v in value.items():
            sub = f"{path}.{name}" if path else name
            if name in props:
                out += validate(v, props[name], sub)
            elif schema.get("additionalProperties") is False:
                out.append((sub, "is not a known field"))
    return out
