"""On the board: do the DFlash2 drafters read only a bounded window of their context KV?

Both served arms (ft-b8-v2 and ft-b16, EXL3 K3 weights under the exl3-4.5 profile) have five
`sliding_attention` layers with `sliding_window = 2048`. The eager draft reads the context rows
[pos0 - 2047, ctx_len) (engine/drafters/dflash2.py `_one_block_steps`), the draft graph gathers
`lo .. lo + span` with lo = pos0 - 2047 and span <= 2048 (engine/drafters/draft_graph.py). If that is
all, a fixed ring of the window could replace the per-slot context KV that grows with the
conversation (40 KiB a token for the two arms, 10 GiB for a full 262,144-token slot).

The measurement, in-process, one engine, the served drafter stack (length router, both arms):
for real text prompts of 8k, 32k and 128k tokens, a greedy tree decode of `--new` tokens twice:

  A  as served;
  B  with every context row BELOW the window poisoned (a large constant) before each draft block:
     rows [0, pos0 - 2047) of both arms' K and V.

If the drafters read only the window, B's drafts are bit-identical to A's: the same tokens, the
same block count, the same accepted tokens per arm. Any read below the window changes them.

Run on the box, the box alone (:8000 stopped), with the served environment:
    set -a; . ~/tandem-live/ops/serve.env; . ~/tandem-live/ops/profiles/exl3-4.5.env; set +a
    python tests/gpu/test_drafter_window_gpu.py [--lens 8192,32768,131072] [--new 256]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

POISON = 37.0


def _text() -> str:
    parts = []
    for pat in ("engine/*.py", "engine/*/*.py", "server/*.py", "server/*/*.py", "tools/*.py",
                "tests/*.py", "docs/*.md"):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            try:
                parts.append(f"# {os.path.relpath(f, ROOT)}\n" + open(f).read())
            except OSError:
                pass
    return "\n\n".join(parts)


def _prompt(tok, text_ids, n: int) -> torch.Tensor:
    body = tok.decode(text_ids[:max(0, n - 120)])
    msgs = [{"role": "user", "content": body + "\n\nSummarise what the code above does, file by "
                                               "file, in plain words."}]
    r = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                tokenize=False)
    return torch.tensor(tok(r, add_special_tokens=False).input_ids, device="cuda")


def _drafters(lr):
    from server.memsplit import drafter_caches
    return drafter_caches(lr, depth=6)


class Poison:
    """At every draft block: the draft on clean rows, then the rows that have just fallen below
    the window poisoned and the same draft again; both must be bit-identical (tokens and the
    lattice). The poison stays, so later blocks run on a context whose whole past below the
    window is garbage. This compares call by call, so nothing timing-driven (the router's arm
    choice, the cut's prices) can make two runs differ."""

    def __init__(self, ds):
        self.ds = ds
        self.upto = {id(d): 0 for d in ds}
        self.on = False
        self.rows = 0
        self.checks = 0
        self.mismatch: list = []
        self.drafts: list = []
        self.orig = {}
        for d in ds:
            self.orig[id(d)] = d._one_block_steps
            d._one_block_steps = self._wrap(d)

    @staticmethod
    def _lat(d):
        lat = getattr(d, "_lattice", None)
        if lat is None:
            return None
        return tuple(t.clone() if torch.is_tensor(t) else t for t in lat)

    def _wrap(self, d):
        orig = self.orig[id(d)]
        from engine.drafters.dflash2 import run_steps

        def steps(m, anchor, pos0, carry, carry_pos, logp=False):
            if self.on and carry is None:
                W = int(d.cfg.sliding_window)
                lo = max(0, int(pos0) - W + 1)
                u = self.upto[id(d)]
                if d.ctx_len < u:
                    u = 0
                if lo > u:
                    clean = run_steps(orig(m, anchor, pos0, carry, carry_pos, logp))
                    lat_c = self._lat(d)
                    d._ck[:, :, u:lo].fill_(POISON)
                    d._cv[:, :, u:lo].fill_(POISON)
                    self.rows += lo - u
                    self.upto[id(d)] = lo
                    got = yield from orig(m, anchor, pos0, carry, carry_pos, logp)
                    lat_p = self._lat(d)
                    self.checks += 1
                    same = clean[0] == got[0]
                    if lat_c is not None or lat_p is not None:
                        same = same and lat_c is not None and lat_p is not None and all(
                            torch.equal(x, y) if torch.is_tensor(x) else x == y
                            for x, y in zip(lat_c, lat_p))
                    if not same:
                        self.mismatch.append((d.cfg.block_size, int(pos0), int(lo)))
                    self.drafts.append((d.cfg.block_size, int(pos0), [int(t) for t in got[0]]))
                    return got
            got = yield from orig(m, anchor, pos0, carry, carry_pos, logp)
            self.drafts.append((d.cfg.block_size, int(pos0), [int(t) for t in got[0]]))
            return got
        return steps

    def reset(self):
        for k in self.upto:
            self.upto[k] = 0


def chunked_prefill(eng, lr, chunk: int = 8192, over: int = 16384):
    """`generate_spec_tree` forwards the whole prompt at once; past ~100k rows one EXL3 GEMM
    launch is out of range (CUDA invalid argument at 131,072, 2026-10-10). Above `over` rows the
    prompt goes in `chunk`-row forwards, each followed by the router's sync of its rows, and the
    loop's own sync of the whole prompt is skipped once."""
    fwd, sync = eng.forward, lr.sync
    state = {"skip": False}

    def forward(x, start=0, last_only=False, **kw):
        T = int(x.shape[-1])
        if start != 0 or T <= over:
            return fwd(x, start=start, last_only=last_only, **kw)
        toks = x.tolist()
        out = None
        for s0 in range(0, T, chunk):
            out = fwd(x[s0:s0 + chunk], start=s0, last_only=True)
            sync(toks[s0:s0 + chunk], eng.hidden_post_norm[0], s0)
        state["skip"] = True
        return out

    def lsync(tokens, hidden, first_pos, rows=None):
        if state["skip"] and first_pos == 0:
            state["skip"] = False
            return None
        return sync(tokens, hidden, first_pos, rows)

    eng.forward, lr.sync = forward, lsync


def main() -> None:
    from engine.spec import generate_spec_tree
    from tools.verify_spec import tree_router
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="8192,32768,131072")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--out", default=os.path.expanduser("~/rd1-runs/drafter_window.json"))
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",") if x]
    M = max(lens) + 4096
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=os.environ.get("NV") or None,
                fp8_head=os.environ.get("HEAD") or None)
    eng = Qwen38Engine(cfg, w, max_len=M, paged=False)
    rp = os.path.expanduser("~/tandem-live")
    ns = types.SimpleNamespace(dflash2_ckpt=os.path.join(rp, "train/ft-b8-v2"),
                               dflash2_path="greedy", max_len=M,
                               lenrouter=os.path.join(rp, "train/ft-b16"), corpus="",
                               drop_idle=True,
                               deep=int(os.environ.get("QWEN38_DEEP", "0") or 0),
                               deep_after=int(os.environ.get("QWEN38_DEEP_AFTER", "2") or 2),
                               len_fixed=0)
    lr = tree_router(eng, ns, narrow=True)
    ds = _drafters(lr)
    assert len(ds) == 2, f"expected the two DFlash2 arms, found {len(ds)}"
    for d in ds:
        assert all(d.cfg.is_sliding(i) for i in range(d.cfg.num_hidden_layers)), "a full layer"
    win = {int(d.cfg.sliding_window) for d in ds}
    per_tok = sum(2 * d.cfg.num_hidden_layers * d.cfg.kv_dim * 2 for d in ds)
    print(f"[window] drafters: {[d.cfg.block_size for d in ds]} block sizes, windows {win}, "
          f"context KV {per_tok / 1024:.1f} KiB a token for both arms", flush=True)
    pz = Poison(ds)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text_ids = tok(_text(), add_special_tokens=False).input_ids
    print(f"[window] text {len(text_ids):,} tokens", flush=True)
    rows, bad = [], []
    chunked_prefill(eng, lr)
    for n in lens:
        ids = _prompt(tok, text_ids, n)
        # each arm pinned in turn, so both arms are measured at every length
        for fx in (8, 16):
            lr.fixed = fx
            pz.on = True
            pz.reset()
            pz.drafts, pz.mismatch, pz.checks, pz.rows = [], [], 0, 0
            eng.reset()
            t0 = time.time()
            with torch.no_grad():
                lr.attach()
                toks, st = generate_spec_tree(eng, ids, a.new, lr, 15)
            torch.cuda.synchronize()
            arm = [x for x in pz.drafts if x[0] == fx]
            row = {"n_prompt": int(ids.numel()), "arm": fx, "new": a.new, "tokens": len(toks),
                   "blocks": st.blocks, "tokens_per_block": round(len(toks) / max(1, st.blocks), 3),
                   "draft_calls": len(pz.drafts), "draft_calls_arm": len(arm),
                   "checked": pz.checks, "mismatch": list(pz.mismatch),
                   "poisoned_rows": pz.rows, "s": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  {n:>7} b{fx:<2}: {len(toks)} tokens in {st.blocks} blocks, {pz.checks} draft "
                  f"blocks checked clean vs poisoned, {len(pz.mismatch)} differ, "
                  f"{pz.rows:,} rows poisoned, {row['s']} s", flush=True)
            if pz.checks == 0:
                bad.append(f"{n} b{fx}: no draft block was checked")
            if pz.mismatch:
                bad.append(f"{n} b{fx}: {len(pz.mismatch)} draft blocks differ, first {pz.mismatch[0]}")
    pz.on = False
    out = {"windows": sorted(win), "context_kv_bytes_per_token": per_tok,
           "ring_rows": max(win), "rows": rows, "bad": bad,
           "env": {k: os.environ.get(k) for k in ("QWEN38_DRAFT_GRAPH", "QWEN38_HOST_WALK",
                                                   "QWEN38_VERIFY_GRAPH", "QWEN38_EXL3")}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[window] raw: {a.out}")
    if bad:
        print("WINDOW NOT BOUNDED (or not exact):\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"WINDOW BOUNDED: every draft block (tokens and lattice) bit-identical clean vs with every "
          f"context row below the {max(win)}-row window poisoned, both arms, at "
          f"{', '.join(sorted({str(r['n_prompt']) for r in rows}))} tokens")


if __name__ == "__main__":
    main()
