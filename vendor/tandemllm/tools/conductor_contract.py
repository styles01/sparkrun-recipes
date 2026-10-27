"""Write or check the conductor's config schemas in docs/contract/conductor-v1/.

    python tools/conductor_contract.py --write     regenerate <kind>.config.schema.json, seeds.json
    python tools/conductor_contract.py             exit 1 when a file differs from the code

The schemas live in server/conductor/schema.py; the files are what the dashboard's contract test
(ajv) and anyone editing a version by hand read; seeds.json is the built-in version 0 of every
document, which the dashboard's mock engine serves. The API's response schemas in the same
directory are written by hand, like dashboard-v1's.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import schema  # noqa: E402

DIR = os.path.join(ROOT, "docs", "contract", "conductor-v1")


def rendered(kind: str) -> str:
    doc = {"$schema": "https://json-schema.org/draft/2020-12/schema",
           "title": f"conductor {kind} document", **schema.SCHEMAS[kind]}
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def seeds() -> str:
    """The built-in documents (version 0) and the meta, for the dashboard's mock engine."""
    out = {kind: schema.seed(kind) for kind in ("settings", "router", "roles", "packages",
                                                "monitor", "residency", "deciders")}
    out["profiles"] = {n: schema.seed("profiles", n) for n in schema.PROFILES}
    out["meta"] = schema.meta()
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    write = "--write" in sys.argv
    os.makedirs(DIR, exist_ok=True)
    stale = 0
    files = [(f"{kind}.config.schema.json", rendered(kind)) for kind in schema.SCHEMAS]
    files.append(("seeds.json", seeds()))
    for name, text in files:
        path = os.path.join(DIR, name)
        cur = open(path).read() if os.path.exists(path) else None
        if cur == text:
            continue
        if write:
            with open(path, "w") as f:
                f.write(text)
            print(f"wrote {os.path.relpath(path, ROOT)}")
        else:
            print(f"stale: {os.path.relpath(path, ROOT)}")
            stale += 1
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
