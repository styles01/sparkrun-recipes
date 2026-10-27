"""The layer-major prefill against the chunk-major one, in one process (engine/model.py
`prefill_window`, engine/cache.py `_window_plan`).

For each prompt length in --lengths (held-out text, as tools/exl3_fastrun.py cuts it) and each case
(`cached`: the prefix grid of --chunk rows with a resident prefix and a state store, as a server with
the caches on runs it; `free`: --free-chunk rows and no cache, as a server without them), the prefill
runs with the windows off (`PREFILL_WINDOW = PREFILL_WINDOW_FREE = 0`, the chunk-major loop) and on
(--window / --window-free rows), and the tool compares bit for bit: the returned logits, the KV rows,
the recurrent and convolution state, every resident anchor, every store snapshot, both drafter arms'
context KV. Wall time of each prefill (synchronised) and the allocator peak are printed.

Usage: prefill_window_check.py --lengths 4096,32768 --window 8192 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.verify_spec import tree_router  # noqa: E402


def prompt_ids(tok, repo: Path, n: int) -> list[int]:
    ids: list[int] = []
    for name in ("heldout_prose.txt", "heldout_code.txt", "calib.txt"):
        p = repo / "bench" / name
        if p.exists():
            ids += tok.encode(p.read_text())
        if len(ids) >= n:
            break
    while len(ids) < n:
        ids += ids
    return ids[:n]


def digest(t: torch.Tensor) -> tuple:
    """A bit-sensitive digest of a tensor without a second copy of it: the 32-bit (or 16-bit)
    words summed plain and position-weighted, in int64, on the device."""
    t = t.contiguous()
    b = t.view(torch.uint8).view(-1)
    pad = (-b.numel()) % 4
    if pad:
        b = torch.cat([b, b.new_zeros(pad)])
    wds = b.view(torch.int32).view(-1)
    out = []
    for s0 in range(0, wds.numel(), 1 << 26):
        x = wds[s0:s0 + (1 << 26)].to(torch.int64)
        idx = torch.arange(s0, s0 + x.numel(), device=x.device, dtype=torch.int64) % 65521 + 1
        out.append((int(x.sum()), int((x * idx).sum())))
    return (tuple(t.shape), str(t.dtype), tuple(out))


def heads(lr):
    return [h for h in (getattr(lr, "head_small", None), getattr(lr, "head_large", None))
            if h is not None and getattr(h, "_ck", None) is not None]


def run(eng, lr, ids, chunk: int, cached: bool):
    eng.reset()
    lr.reset()
    res = cache.ResidentPrefix(8 << 30, chunk, tail=4) if cached else None
    store = cache.StateStore(4 << 30, chunk=chunk, max_entry_bytes=2 << 30) if cached else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    logits, reused, fwd = cache.prefill(eng, lr, ids, "cuda", store=store, chunk=chunk,
                                        checkpoint=cached, resident=res)[:3]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() - base
    n = len(ids)
    got = {"logits": logits.float().cpu(),
           "k": digest(eng.kv.k[:, :, :, :n]), "v": digest(eng.kv.v[:, :, :, :n]),
           "S": digest(eng.state.S), "conv": digest(eng.state.conv), "len": eng.kv.length}
    for j, h in enumerate(heads(lr)):
        got[f"ck{j}"] = digest(h._ck[:, :, :h.ctx_len])
        got[f"cv{j}"] = digest(h._cv[:, :, :h.ctx_len])
        got[f"ctx{j}"] = h.ctx_len
    if res is not None:
        for b, (S, conv) in res.anchors.items():
            got[f"anchor{b}.S"], got[f"anchor{b}.conv"] = digest(S), digest(conv)
    if store is not None:
        for key, e in store._d.items():
            s = e.snap
            got[f"snap{s.length}.S"], got[f"snap{s.length}.k"] = digest(s.S), digest(s.k)
            if s.drafter is not None:
                got[f"snap{s.length}.d"] = _tree_digest(s.drafter)
    return got, dt, peak


def _tree_digest(x):
    if torch.is_tensor(x):
        return digest(x)
    if isinstance(x, (list, tuple)):
        return [_tree_digest(y) for y in x]
    if isinstance(x, dict):
        return {k: _tree_digest(v) for k, v in x.items()}
    return x


def same(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and bool(
            torch.equal(a, b))
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096,32768")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--free-chunk", type=int, default=8192)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--window-free", type=int, default=32768)
    ap.add_argument("--cases", default="cached,free")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--ckpt8", default="train/ft-b8-v2")
    ap.add_argument("--ckpt16", default="train/ft-b16")
    ap.add_argument("--toggle", default="",
                    help="module.ATTR: the two arms flip this boolean instead of the windows "
                         "(windows on in both), e.g. tools.exl3_linear.MLP_BRIDGE")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ns = types.SimpleNamespace(dflash2_ckpt=a.ckpt8, dflash2_path="greedy", max_len=a.max_len,
                               lenrouter=a.ckpt16, corpus="", drop_idle=True, deep=0,
                               deep_after=2, len_fixed=0)
    lr = tree_router(eng, ns, narrow=False)
    out = []
    for n in [int(x) for x in a.lengths.split(",") if x]:
        ids = prompt_ids(tok, repo, n)
        for case in a.cases.split(","):
            cached = case == "cached"
            chunk = a.chunk if cached else a.free_chunk
            row = {"n": n, "case": case, "chunk": chunk}
            refs = {}
            for arm in ("off", "on", "off", "on")[:2 * a.reps]:
                if a.toggle:
                    import importlib
                    spec, _, vals = a.toggle.partition("=")
                    modname, attr = spec.rsplit(".", 1)
                    if vals:                     # module.ATTR=off_value:on_value (ints or strings)
                        lo, hi = vals.split(":")
                        conv = (lambda v: int(v)) if lo.lstrip("-").isdigit() else (lambda v: v)
                        val = conv(hi) if arm == "on" else conv(lo)
                    else:
                        val = arm == "on"
                    setattr(importlib.import_module(modname), attr, val)
                    cache.PREFILL_WINDOW, cache.PREFILL_WINDOW_FREE = a.window, a.window_free
                else:
                    cache.PREFILL_WINDOW = a.window if arm == "on" else 0
                    cache.PREFILL_WINDOW_FREE = a.window_free if arm == "on" else 0
                got, dt, peak = run(eng, lr, ids, chunk, cached)
                row.setdefault(f"s_{arm}", []).append(round(dt, 3))
                row.setdefault(f"peak_gib_{arm}", []).append(round(peak / (1 << 30), 2))
                if arm in refs:
                    continue
                refs[arm] = got
                del got
            off, on = refs["off"], refs["on"]
            bad = sorted(k for k in set(off) | set(on) if k not in off or k not in on
                         or not same(off[k], on[k]))
            row["keys"] = len(off)
            row["identical"] = not bad
            row["differ"] = bad[:20]
            print(f"[window] {json.dumps(row)}", flush=True)
            out.append(row)
            del refs, off, on
            torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
