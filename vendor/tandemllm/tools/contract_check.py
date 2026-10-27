"""Validate a dashboard API response against contract v1's JSON Schemas.

    curl -s -H "Authorization: Bearer $QSE_ADMIN_TOKEN" localhost:8000/v1/dashboard/summary \\
        | python tools/contract_check.py summary
    python tools/contract_check.py usage response.json

The schemas are `docs/contract/dashboard-v1/<name>.schema.json`, the single source of truth the
dashboard's mocks and this server are both held to. The validator is the subset of JSON Schema
those files use -- `type` (one or a list, `integer` excluding booleans), `properties`, `required`,
`additionalProperties: false`, `items`, `enum`, `const`, and `$ref` within a file or to a sibling
file -- in the standard library, because the box has no pip. Exit 1 and one line per violation,
with its JSON path, when the document does not validate.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "contract", "dashboard-v1")

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def load(name: str, root: str = ROOT) -> dict:
    with open(os.path.join(root, name if name.endswith(".json") else f"{name}.schema.json")) as f:
        return json.load(f)


def _resolve(ref: str, doc: dict, root: str) -> tuple[dict, dict]:
    file, _, frag = ref.partition("#")
    base = load(file, root) if file else doc
    node = base
    for part in [p for p in frag.split("/") if p]:
        node = node[part]
    return node, base


def validate(value, schema: dict, *, root: str = ROOT, doc: dict | None = None,
             path: str = "$") -> list[str]:
    doc = doc if doc is not None else schema
    if "$ref" in schema:
        node, base = _resolve(schema["$ref"], doc, root)
        return validate(value, node, root=root, doc=base, path=path)
    errs = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](value) for x in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__} {value!r:.60}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: expected {schema['const']!r}, got {value!r:.60}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r:.60} is not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], root=root, doc=doc, path=f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected {k!r}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], root=root, doc=doc, path=f"{path}[{i}]")
    return errs


def check(value, name: str, root: str = ROOT) -> list[str]:
    return validate(value, load(name, root), root=root)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    raw = open(sys.argv[2]).read() if len(sys.argv) > 2 else sys.stdin.read()
    errs = check(json.loads(raw), name)
    for e in errs:
        print(e)
    print(f"{name}: {'valid' if not errs else f'{len(errs)} violations'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
