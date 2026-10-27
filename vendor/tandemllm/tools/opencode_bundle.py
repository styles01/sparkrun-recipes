"""opencode's own chat-completions parser, out of its binary, for node (X1, 2026-10-03).

    python tools/opencode_bundle.py [OPENCODE_BINARY] [OUT_DIR]

opencode ships as one Bun binary; its modules sit in it as `\\0/$bunfs/root/<name>\\0<source>`. A
custom provider without an `npm` field gets `@ai-sdk/openai-compatible`, which opencode imports from
`/$bunfs/root/chunk-<hash>.js` (`createOpenAICompatible`). This writes that module and every
module it imports to OUT_DIR as `.mjs` files with relative imports, so node runs the exact code
opencode parses our stream with (tools/opencode_parts.mjs). Nothing is changed but the import
paths. Prints the entry file. Stdlib only; reads the binary, writes OUT_DIR.
"""

from __future__ import annotations

import os
import re
import sys

ENTRY = re.compile(rb'"@ai-sdk/openai-compatible":\(\)=>import\("(/\$bunfs/root/[A-Za-z0-9_.\-]+)"\)')
MODULE = re.compile(rb"\x00(/\$bunfs/root/[A-Za-z0-9_.\-]+)\x00")
IMPORT = re.compile(r"""["'](/\$bunfs/root/[A-Za-z0-9_.\-]+)["']""")


def extract(binary: str, out: str) -> str:
    """The provider module and its imports under `out`; the entry's path."""
    b = open(binary, "rb").read()
    m = ENTRY.search(b)
    if m is None:
        raise SystemExit(f"{binary}: no @ai-sdk/openai-compatible import found")
    entry = m.group(1).decode()
    starts = {mm.group(1).decode(): mm.end() for mm in MODULE.finditer(b)}
    os.makedirs(out, exist_ok=True)
    need, seen = [entry], set()
    while need:
        name = need.pop()
        if name in seen:
            continue
        if name not in starts:
            raise SystemExit(f"{binary}: module {name} not found")
        seen.add(name)
        s = starts[name]
        code = b[s:b.find(b"\x00", s)].decode("utf-8")
        need.extend(IMPORT.findall(code))
        code = re.sub(r"""(["'])/\$bunfs/root/([A-Za-z0-9_.\-]+)\.js\1""", r"\1./\2.mjs\1", code)
        with open(os.path.join(out, os.path.basename(name)[:-3] + ".mjs"), "w") as f:
            f.write(code)
    return os.path.join(out, os.path.basename(entry)[:-3] + ".mjs")


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.opencode/bin/opencode")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), ".opencode-provider")
    print(extract(binary, out))
