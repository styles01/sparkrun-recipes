"""Check the session monitor's policy file, or write its JSON Schema.

    python tools/conductor_policy.py                     # validate ops/supervisor.yaml
    python tools/conductor_policy.py path/to/policy.yaml
    python tools/conductor_policy.py --schema > ops/supervisor.schema.json
    python tools/conductor_policy.py --json              # the package's JSON copy of the file
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor.monitor import policy  # noqa: E402


def main(argv: list[str]) -> int:
    if argv == ["--schema"]:
        print(json.dumps(policy.schema(), indent=2))
        return 0
    if argv == ["--json"]:
        # the copy an installed proxy reads (server/conductor/monitor/policy.default.json)
        with open(policy.DEFAULT_JSON, "w") as f:
            json.dump(policy.load().data, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {policy.DEFAULT_JSON}")
        return 0
    try:
        p = policy.load(argv[0] if argv else None)
    except policy.PolicyError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    modes = ", ".join(f"{t.id}={t.mode}" for t in p.tasks.values())
    print(f"{p.path}: version {p.version}, valid; tasks {modes}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
