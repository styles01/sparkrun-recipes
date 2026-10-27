"""Where a resident turn's time goes before its first token, on the real engine.

    ops/hold.sh 20 -- python tools/turn_profile.py --out prof.json -- <server/app.py flags>

Loads the engine with the server's own `_load` and the given flags, grows an agent conversation
(tools/resident_check.py's, this repository's files as tool results) to each `--at` size, runs
that turn, then the next one -- a short tool result more, the ordinary agent turn -- under
cProfile: the prompt built from the messages, the drafters primed, the prefill, the first
token. Reports the wall time of each step, the functions with the most time, and the prefill's
forwarded tokens. No HTTP; the handler's own work outside these calls is not in it.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: turn_profile.py [options] -- <server/app.py flags>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", default="130000,190000")
    ap.add_argument("--small-chars", type=int, default=2400)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--content-root", default=os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    help="the checkout whose files make the conversation, so two builds "
                         "profiled from two checkouts read the same text")
    opt = ap.parse_args(sys.argv[1:cut])

    from server import app
    import glob
    from tools.resident_check import Conversation
    a = app.parser().parse_args(sys.argv[cut + 1:])
    t0 = time.time()
    app._load(a)
    print(f"[profile] engine loaded in {time.time() - t0:.0f} s", flush=True)
    S = app.STATE
    dev = S.get("device", "cuda")
    files = []
    for pat in ("engine/*.py", "engine/drafters/*.py", "server/*.py", "docs/*.md", "tools/*.py"):
        files += sorted(glob.glob(os.path.join(opt.content_root, pat)))
    conv = Conversation(app, files * 2)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "flags": sys.argv[cut + 1:],
           "turns": []}

    def turn(profile: bool) -> dict:
        body = {"messages": conv.messages, "tools": conv.tools}
        prof = cProfile.Profile() if profile else None
        torch.cuda.synchronize()
        t = time.perf_counter()
        if prof:
            prof.enable()
        prompt, _, _ = app.build_prompt(body)
        t_bp = time.perf_counter()
        gen = app.generate_stream(prompt, 1, set())
        next(gen)
        torch.cuda.synchronize()
        t_first = time.perf_counter()
        if prof:
            prof.disable()
        for _ in gen:
            pass
        lp = S.get("last_prefill") or {}
        row = {"prompt": int(prompt.numel()), "build_prompt_ms": round((t_bp - t) * 1e3, 1),
               "to_first_token_ms": round((t_first - t) * 1e3, 1),
               "prefill_ms": round(lp.get("ms", 0.0), 1), "reused": lp.get("reused"),
               "forwarded": lp.get("forwarded"), "kind": lp.get("kind")}
        row["outside_prefill_ms"] = round(row["to_first_token_ms"] - row["prefill_ms"], 1)
        if prof:
            for key in ("cumulative", "tottime"):
                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(opt.top)
                row[f"top_{key}"] = buf.getvalue().splitlines()
        return row

    for target in (int(x) for x in opt.at.split(",") if x):
        conv.grow_to(target)
        warm = turn(False)                             # this size's turn: it fills the caches
        conv.add_turn(max_chars=opt.small_chars)
        row = turn(True)
        row["previous"] = warm
        out["turns"].append(row)
        print(f"[profile] {row['prompt']} tokens: to first token {row['to_first_token_ms']} ms, "
              f"prefill {row['prefill_ms']} ms (forwarded {row['forwarded']}, {row['kind']}), "
              f"outside the prefill {row['outside_prefill_ms']} ms, build_prompt "
              f"{row['build_prompt_ms']} ms", flush=True)
        for line in row["top_cumulative"][:40]:
            print("    " + line)
    with open(opt.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[profile] wrote {opt.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
