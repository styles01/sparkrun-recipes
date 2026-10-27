"""The narrator's forward against transformers on the real checkpoint (P3's gate, run for P1).

    python tools/narrator_reference.py [--dir ~/.qwen38-spark-engine/models/minicpm5-2b]
                                       [--device cpu] [--tokens 24] [--no-reference]

For each sample state event: the narrator's prompt (server/conductor/narrator.py), greedy text
from engine/generative/llama.py, and greedy text from transformers' own LlamaForCausalLM on the
same weights and dtype; the token ids must be equal. One model is in memory at a time. On the box
CPU (`--device cpu`, the default; no GPU) it takes a few minutes at nice 19. It also prints the
lines, the first reading of what MiniCPM5-2B writes; the time per line on the CPU says nothing
about the GPU's.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# kind, facts, extra: one of every kind the conductor says most
EVENTS = [
    ("route", {"cls": "code_edit", "effort": "high effort"}, {}),
    ("route", {"cls": "three_d", "effort": "xhigh effort"}, {}),
    ("route", {"cls": "summarize", "effort": "thinking off"}, {}),
    ("package", {"pkind": "skill", "name": "photo-to-3d"}, {}),
    ("compaction", {}, {}),
    ("cap", {"cap": "16,384"}, {"set by": "the router, for the task's effort"}),
    ("check", {}, {"tools": ["edit"], "files": ["src/lexer.ts"]}),
    ("checks", {"n": 3}, {"tools": ["read", "read", "bash"], "files": ["app/main.py"]}),
    ("repair", {}, {"tool": "bash", "check": "malformed call"}),
    ("loop", {}, {"check": "reasoning loop"}),
    ("steer", {"label": "gate progress"}, {}),
    ("continue", {}, {}),
]


def entries():
    from server.conductor.states import TEXTS
    for kind, facts, extra in EVENTS:
        yield {"kind": kind, "text": TEXTS[kind].format(**facts), "facts": {**facts, **extra}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="~/.qwen38-spark-engine/models/minicpm5-2b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--no-reference", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from engine.generative.model import LlamaModel
    from server.conductor import narrator, schema

    d = os.path.expanduser(a.dir)
    m = LlamaModel("minicpm5-2b", d)
    where = None
    if a.device != "cpu":
        from engine.decision.device import Device, Placement
        where = Placement(Device(a.device))
    t0 = time.time()
    m.load(where)
    print(f"[ref] engine/generative loaded in {time.time() - t0:.1f}s: {m.precision()}, "
          f"{m.device_bytes() / 2**30:.2f} GiB, eos {sorted(m.eos)}", flush=True)
    labels = {c["id"]: c["label"] for c in schema.seed("router")["classes"]}
    ours = []
    for e in entries():
        ids = m.encode(narrator.messages(e, labels))
        t0 = time.time()
        r = m.generate(ids, a.tokens, None)
        text = m.decode(r["ids"])
        line, why = narrator.clean(text) if r["finish"] == "stop" else (None, r["finish"])
        ours.append((ids, r["ids"]))
        print(f"[ref] {e['kind']:10s} {len(ids):4d}+{len(r['ids']):2d} tok "
              f"{time.time() - t0:5.1f}s {r['finish']:6s} | {e['text']}\n"
              f"      -> {line if line else f'(template: {why}) {text!r}'}", flush=True)
    m.unload()
    gc.collect()
    if a.no_reference:
        return 0
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to(a.device).eval()
    same = 0
    for (ids, got), e in zip(ours, entries()):
        with torch.inference_mode():
            out = ref.generate(torch.tensor([ids], device=a.device), max_new_tokens=a.tokens,
                               do_sample=False, eos_token_id=sorted(m.eos),
                               pad_token_id=sorted(m.eos)[0])
        want = [t for t in out[0, len(ids):].tolist()]
        while want and want[-1] in m.eos:
            want.pop()
        ok = want == got
        same += ok
        print(f"[ref] {e['kind']:10s} {'SAME' if ok else 'DIFF'} ours {len(got)} ref {len(want)}"
              + ("" if ok else f"\n      ours {got}\n      ref  {want}"), flush=True)
    print(f"[ref] greedy ids equal to transformers on {same} of {len(ours)} events")
    return 0 if same == len(ours) else 1


if __name__ == "__main__":
    sys.exit(main())
