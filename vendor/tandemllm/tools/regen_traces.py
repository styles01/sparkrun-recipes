"""Re-continue recorded traces with the loaded target's own greedy decode.

The lattice recordings (`tools/record_lattice.py`) and every replay over them assume the trace's
`output_ids` are the target's greedy text: in a lossless engine the committed prefix is the greedy
prefix. A new weight set (an EXL3 size, a new NVFP4 build) writes a slightly different text, so the
traces of the old target must be re-continued before the lattices are re-recorded on it.

For every `*.json` trace in `--traces` this decodes `len(output_ids)` tokens (or `--max-new`) one
token per forward pass (`engine.spec.generate_greedy`, the reference every speculative run must
reproduce) from the trace's `prompt_ids`, and writes a copy with `output_ids` replaced into `--out`.
The copy also carries `gaps` (the top-1 minus top-2 logit at every decoded token, the near-tie
evidence the lossless gate reads), `ref_of` (the source trace) and `target` (the weight set's tag).
Token ids only, no text.

`--temperature T --top-p P` re-continues with the target's own SAMPLED decode instead (one token per
forward, `engine.sample.Sampler` exactly as a sampled request draws its tokens, unseeded or keyed by
`--seed` + the trace index). On such a trace the replays' "longest tree path the continuation
follows" is the sampled request's `--sampled-tree=det` walk (draw at each node, follow the child
carrying the draw) in distribution: the recorded token at a position IS a draw from the target there.
`gaps` is then empty; the copy carries `sampling` = {temperature, top_p, seed}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from engine.spec import _stop, generate_greedy  # noqa: E402


def generate_sampled(eng, prompt: torch.Tensor, max_new: int, eos: list[int],
                     sampler: Sampler) -> tuple[list[int], float]:
    """One token per forward, each drawn by `sampler` as the server draws a request's first token."""
    eng.reset()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
        pos = prompt.numel()
        tok = sampler(logits[0, -1], index=pos)
        out = [tok]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(out) < max_new and pos < eng.max_len and not _stop(tok, eos):
            logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                 last_only=True)
            pos += 1
            tok = sampler(logits[0, -1], index=pos)
            out.append(tok)
        torch.cuda.synchronize()
    return out, (len(out) - 1) / max(time.perf_counter() - t0, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--traces", required=True, help="dir of source *.json traces")
    ap.add_argument("--out", required=True, help="dir the re-continued traces go to")
    ap.add_argument("--only", default=None, help="comma list of trace names")
    ap.add_argument("--max-new", type=int, default=0, help="0: the source trace's length")
    ap.add_argument("--tag", default=os.environ.get("QWEN38_EXL3", "") or "nvfp4")
    ap.add_argument("--temperature", type=float, default=0.0, help=">0: the target's sampled decode")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None, help="keyed draws: seed + the trace's index")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traces, "*.json")))
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    if not files:
        raise SystemExit(f"no traces in {a.traces}")
    os.makedirs(a.out, exist_ok=True)

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    eos = list(cfg.eos_token_ids)
    tot_same = tot_n = 0
    for ti, path in enumerate(files):
        dst = os.path.join(a.out, os.path.basename(path))
        if os.path.exists(dst):
            print(f"{os.path.basename(path)} exists, skipped")
            continue
        tr = json.load(open(path))
        n = a.max_new or len(tr["output_ids"])
        prompt = torch.tensor(tr["prompt_ids"], device=a.device)
        t0 = time.perf_counter()
        if a.temperature > 0.0:
            smp = Sampler(a.temperature, a.top_p, seed=None if a.seed is None else a.seed + ti)
            out, rate = generate_sampled(eng, prompt, n, eos, smp)
            st = type("S", (), {"gaps": [], "tokens": len(out) - 1, "decode_s": (len(out) - 1) / rate})
        else:
            out, st = generate_greedy(eng, prompt, n, eos=eos, record_gaps=True)
        dt = time.perf_counter() - t0
        old = tr["output_ids"]
        same = 0
        while same < min(len(old), len(out)) and old[same] == out[same]:
            same += 1
        tot_same += same
        tot_n += len(out)
        new = dict(tr)
        new.update(output_ids=out, gaps=[round(g, 5) for g in st.gaps], ref_of=os.path.abspath(path),
                   target=a.tag, regen_tok_s=st.tokens / max(st.decode_s, 1e-9))
        if a.temperature > 0.0:
            new["sampling"] = {"temperature": a.temperature, "top_p": a.top_p,
                               "seed": None if a.seed is None else a.seed + ti}
        for k in ("mtp_depth", "mtp_proposals", "recorded_tok_s", "recorded_accept_len"):
            new.pop(k, None)
        with open(dst, "w") as f:
            json.dump(new, f)
        print(f"{tr['name']:16s} {len(out):4d} tokens  same prefix as source {same:4d}  "
              f"{dt:5.1f} s  ({st.tokens / max(st.decode_s, 1e-9):5.1f} tok/s)", flush=True)
    print(f"done: {tot_n} tokens, {tot_same} in the source's common prefix")


if __name__ == "__main__":
    main()
