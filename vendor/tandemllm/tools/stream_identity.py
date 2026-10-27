"""Byte identity of TandemLLM's streams between two trees, with a release's new switches off.

    python tools/stream_identity.py BASE_TREE CANDIDATE_TREE [--mode off|defaults]

Each tree runs in its own process on the fake engine (server/fake_engine.py: the real handler,
streaming, conductor, monitor and journal; no model, no GPU) with the conductor's seed documents and
the same requests: opencode's shape with a tool call the monitor repairs, a cap close, a long
thinking, a long prefill, Hermes' `tags`, a non-streamed answer, `X-Tandem: raw`, and opencode's
title request, each with the profile's state lines on (in the reasoning) and off. The streams are
compared byte for byte after the request ids, timestamps, call ids, `timings` and `metrics` (and with
them a JSON answer's Content-Length) are normalised.

`--mode off` (the default) switches S1's additions off where the candidate's schema knows them:
`structural.side_requests: off`, `ui.states_channel: reasoning`, `ui.progress.mode: off`; every
stream must be identical. `--mode defaults` leaves the candidate's defaults as a router and a
profile stored before S1 get them (side requests on, the lines in the reasoning, progress on):
only the title request may differ, and the tool says which ones did.

Stdlib only in this process; each tree's own interpreter imports its own code. Exit 0 when every
stream that must match does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

RUNNER = r'''
import contextlib, io, json, os, re, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
from server import app
from server.conductor import schema
import test_ui_states as tu
MODE = sys.argv[1]
TITLE = open(sys.argv[2]).read()
st = schema.SCHEMAS["router"]["properties"]["structural"]["properties"]
ui_props = schema.SCHEMAS["profiles"]["properties"]["ui"]["properties"]

def serve(states):
    cond, _ = tu.serve(None)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        ui = {"states": states}
        if MODE == "off" and "states_channel" in ui_props:
            ui.update(states_channel="reasoning", progress={"mode": "off"})
        doc["ui"] = ui
        cond.store.put("profiles", name, doc, note="identity", author="tool")
    if MODE == "off" and "side_requests" in st:
        _, r = cond.store.get("router")
        r["structural"]["side_requests"] = "off"
        cond.store.put("router", None, r, note="identity", author="tool")

def norm(raw):
    raw = re.sub(r'"id": "chatcmpl-[0-9a-f]+"', '"id": "x"', raw)
    raw = re.sub(r'"created": \d+', '"created": 0', raw)
    raw = re.sub(r'"(timings|metrics)": \{[^{}]*(\{[^{}]*\}[^{}]*)*\}', '"\\1": {}', raw)
    raw = re.sub(r'"id": "call_[0-9a-zA-Z]+"', '"id": "c"', raw)
    raw = re.sub(r"^Date: .*$", "Date: x", raw, flags=re.M)
    raw = re.sub(r"^Content-Length: \d+", "Content-Length: x", raw, flags=re.M)
    return raw

CALL = "FAKE_DROP_CLOSER FAKE_CALL_JSON " + json.dumps({"name": "bash", "arguments": {"command": "ls -la"}})
title = {"model": "TandemLLM", "stream": True, "stream_options": {"include_usage": True},
         "temperature": 0.5, "max_tokens": 32000, "reasoning_effort": "low",
         "messages": [{"role": "system", "content": TITLE},
                      {"role": "user", "content": "Generate a title for this conversation:\n"},
                      {"role": "user", "content": "Build a voxel pagoda viewer in three.js."}]}
cases = {
    "opencode_call": lambda: tu.opencode(CALL),
    "opencode_cap": lambda: tu.opencode("Explain the lexer.", max_reasoning_tokens=12),
    "opencode_loop": lambda: tu.opencode("FAKE_LOOP think on"),
    "opencode_prefill": lambda: tu.opencode("FAKE_SLOW_PREFILL read it all"),
    "opencode_json": lambda: tu.opencode("Write the lexer.", stream=False),
    "hermes": lambda: tu.hermes(CALL),
    "raw": lambda: tu._go({"model": "TandemLLM", "stream": True, "max_tokens": 200,
                           "reasoning_format": "reasoning_content",
                           "messages": [{"role": "user", "content": "Write the lexer."}]},
                          {"User-Agent": tu.OPENCODE_UA, "X-Tandem": "raw"}),
    "opencode_title": lambda: tu._go(title, {"User-Agent": tu.OPENCODE_UA, "x-session-id": "s"}),
}
out = {}
for states in ("on", "off"):
    for name, fn in cases.items():
        serve(states)
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn()
        out[f"{name}/states={states}"] = norm(r["head"]) + "\n\n" + norm(r["raw"])
with open(sys.argv[3], "w") as f:
    json.dump(out, f)
'''


def run(tree: str, mode: str, title: str, py: str) -> dict:
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""),
           "PYTHONPATH": tree + (":" + os.environ["PYLIBS"] if os.environ.get("PYLIBS") else ""),
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "TZ": "Europe/Berlin"}
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        p = subprocess.run([py, "-c", RUNNER, mode, title, out.name], cwd=tree, env=env,
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            sys.exit(f"{tree}: the runner failed\n{p.stderr[-3000:]}")
        with open(out.name) as f:
            return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--mode", choices=("off", "defaults"), default="off")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    title = os.path.join(here, "tests", "fixtures", "side", "opencode-1.18.34-title-system.txt")
    base = run(os.path.abspath(a.base), "off", title, a.python)
    cand = run(os.path.abspath(a.cand), a.mode, title, a.python)
    bad = 0
    for k in sorted(base):
        same = base[k] == cand.get(k)
        may_differ = a.mode == "defaults" and k.startswith("opencode_title/")
        print(f"[identity] {'SAME' if same else 'DIFF'} {k}"
              + (" (allowed: side requests are on by default)" if not same and may_differ else ""))
        if not same and not may_differ:
            bad += 1
            b, c = base[k], cand.get(k) or ""
            i = next((i for i, (x, y) in enumerate(zip(b, c)) if x != y), min(len(b), len(c)))
            print(f"  first difference at {i}:\n  base {b[max(0, i - 80):i + 120]!r}\n"
                  f"  cand {c[max(0, i - 80):i + 120]!r}")
    print(f"[identity] {len(base) - bad} of {len(base)} streams as they must be, mode {a.mode}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
