"""Record what the model actually generates, so drafters can be judged without the board.

A greedy speculative decoder accepts exactly the prefix of a draft that matches what the target
would have produced on its own. That makes offline evaluation *exact* rather than approximate: if
the target's greedy continuation of a prompt is known, any drafter can be replayed against it on a
laptop and its acceptance is the same number the board would have measured. The only thing the
board is needed for is producing the continuation once.

So this tool spends one GPU session and writes a trace per prompt:

    prompt_ids      the tokenised prompt, exactly as the bench feeds it
    output_ids      the greedy continuation, `--new` tokens of it
    mtp_proposals   what the checkpoint's prediction head proposed, and where

The proposals are logged for free by wrapping the drafter -- the generation is running anyway --
and they let the simulator compare a lookup drafter against the neural one, and mix them, without
either of them needing weights a second time.

Generation uses the speculative path because it is lossless: greedy output does not depend on the
drafter (the M2 gate), so speculating here changes only how long the recording takes.

The prompts are generic, public-domain-style tasks in five classes. Nothing personal goes in a
trace; the traces land in `results/`, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters import Drafter  # noqa: E402
from engine.drafters.mtp import MTPDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec  # noqa: E402

SNIPPET = '''
def load_shard(path, device):
    """Read one safetensors shard and return its tensors, quantised weights kept as codes."""
    out = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            if ".visual." in key:
                continue
            out[key] = f.get_tensor(key)
    return out


def step_bytes(weights, n_layers):
    """Bytes one autoregressive step has to read, by group."""
    per_layer = {}
    for name, tensor in weights.items():
        if not name.startswith("layers."):
            continue
        idx = int(name.split(".")[1])
        per_layer[idx] = per_layer.get(idx, 0) + tensor.numel() * tensor.element_size()
    layers = sum(v for k, v in per_layer.items() if k < n_layers)
    head = weights["lm_head.weight"]
    return layers + head.numel() * head.element_size()
'''

CONFIG_SNIPPET = '''
server:
  host: 0.0.0.0
  port: 8080
  workers: 4
  timeout_seconds: 30
logging:
  level: info
  format: json
  destination: stdout
storage:
  backend: local
  path: /var/lib/service/data
  retention_days: 14
'''

# Five classes, two prompts each. Class names match tools/bench_decode.py where they overlap.
PROMPTS: dict[str, str] = {
    "prose-1": ("Write three paragraphs about why the memory system, and not the arithmetic units, "
                "sets the speed of a language model that generates one token at a time."),
    "prose-2": ("Explain, in four paragraphs and without equations, why compression and prediction "
                "are the same problem, and what that implies for how a language model should be "
                "evaluated."),
    "chat-1": ("I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
               "Explain in plain language what that means for running a 27-billion-parameter "
               "model."),
    "chat-2": ("My colleague says we should rewrite our batch pipeline as a streaming one. What "
               "questions should I ask before agreeing, and what would change operationally?"),
    "code-1": ("Write a Python function that reads a safetensors file header and prints every "
               "tensor name, dtype and shape, sorted by the number of bytes it occupies."),
    "code-2": ("Write a Go function that walks a directory tree concurrently with a bounded worker "
               "pool, hashes every regular file with SHA-256, and returns a map from path to hex "
               "digest. Handle errors properly and include a short example of calling it."),
    "edit-1": ("Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
               "Rewrite it with type hints on every function and parameter. Change nothing else: "
               "keep the same function names, the same docstrings, the same logic and the same "
               "order. Output the complete module."),
    "edit-2": ("Here is a configuration file:\n\n```yaml" + CONFIG_SNIPPET + "```\n\n"
               "Produce the same file with the port changed to 9090, the log level changed to "
               "debug, and a new `metrics` section with `enabled: true` and `port: 9091`. Keep "
               "every other key, value, comment and the ordering exactly as they are."),
    "de-1": ("Schreibe drei Absaetze darueber, warum die Speicherbandbreite und nicht die "
             "Rechenleistung bestimmt, wie schnell ein Sprachmodell Token fuer Token schreibt. "
             "Antworte auf Deutsch."),
    "de-2": ("Erklaere auf Deutsch, in vier Absaetzen, wie sich spekulatives Dekodieren von "
             "gewoehnlichem Dekodieren unterscheidet und warum die Ausgabe dabei identisch "
             "bleiben kann."),
}


class _Recording(Drafter):
    """Passes every call through to a real drafter and writes down what it proposed."""

    name = "recording"

    def __init__(self, inner: Drafter):
        self.inner = inner
        self.log: list[tuple[int, list[int]]] = []

    def propose(self, context: list[int], k: int) -> list[int]:
        draft = self.inner.propose(context, k)
        self.log.append((len(context), [int(t) for t in draft]))
        return draft

    def observe(self, tokens: list[int]) -> None:
        self.inner.observe(tokens)

    def reset(self) -> None:
        self.log = []
        self.inner.reset()

    def prime(self, tokens: list[int]) -> None:
        if hasattr(self.inner, "prime"):
            self.inner.prime(tokens)

    def sync(self, tokens, hidden, first_pos) -> None:
        if hasattr(self.inner, "sync"):
            self.inner.sync(tokens, hidden, first_pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--new", type=int, default=512, help="tokens of continuation per prompt")
    ap.add_argument("--depth", type=int, default=3, help="draft depth used to speed up recording")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason before answering; off by default, because the bench "
                         "row this program is measured against runs with thinking off, and because "
                         "128 tokens of reasoning is 128 tokens the edit and quote regimes never "
                         "reach")
    ap.add_argument("--only", default=None, help="comma separated trace names")
    ap.add_argument("--out", default=None, help="directory for the traces (default results/traces)")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = a.out or os.path.join(root, "results", "traces")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids

    wanted = set(a.only.split(",")) if a.only else None
    for name, text in PROMPTS.items():
        if wanted and name not in wanted:
            continue
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=a.think),
                  return_tensors="pt").input_ids[0].to(a.device)
        rec = _Recording(MTPDrafter(eng, max_len=a.max_len, hidden="post", depth=a.depth))
        t0 = time.perf_counter()
        out, st = generate_spec(eng, ids, a.new, rec, a.depth, eos)
        dt = time.perf_counter() - t0
        trace = {
            "name": name,
            "klass": name.split("-")[0],
            "model": os.path.basename(cfg.path),
            "nvfp4": os.path.basename(os.environ.get("QWEN38_NVFP4", "")) or None,
            "thinking": bool(a.think),
            "prompt_ids": [int(t) for t in ids.tolist()],
            "output_ids": [int(t) for t in out],
            "mtp_depth": a.depth,
            "mtp_proposals": [[pos, d] for pos, d in rec.log],
            "recorded_tok_s": st.tok_s,
            "recorded_accept_len": st.accept_len,
        }
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(trace, f)
        print(f"{name:8s} prompt {ids.numel():4d}  out {len(out):4d}  "
              f"{st.tok_s:5.2f} tok/s  acc/block {st.accept_len:4.2f}  {dt:5.1f} s  -> {path}")


if __name__ == "__main__":
    main()
