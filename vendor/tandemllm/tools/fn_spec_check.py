"""ENG-227's proof on the board: greedy speculative output == plain greedy output, token for token.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_spec_check.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 256 --modes oracle,oracle-eager,mtp --json out.json

Per class (tools/fn_generate.py's prompts at `--ctx`): the plain greedy text through the captured
decode step (`DecodeGraph`), then the same prompt through `SpecDecoder` with each draft source:

  oracle        trees built from the plain text with planted errors: a true chain of 0..`--depth`
                tokens, a wrong sibling subtree before a true child at random (so accepted rows sit
                past their depth and the commit moves them), a wrong token after the chain. Any
                drafter must give the plain text; this one makes long accepted paths through moved
                tree rows. Verify blocks from captured graphs.
  oracle-eager  the same trees, verify eager (no graphs)
  oracle-wide   wider trees: a wrong subtree of 1 to 3 nodes before most true children (blocks of up to
                32 rows with `--depth 8`)
  mtp           the pack's MTP layer as a greedy chain drafter of `--depth` (eager drafting)
  mtp-graph     the same chain from captured graphs (`MTPGraphDrafter`); its drafts should match
                `mtp`'s closely (the fused trunk's arithmetic in place of the reference's)

Recorded per class and mode: equal (the speculative text is the plain one), the first differing
index, rounds, rows verified, accepted drafts, rounds whose path was moved, and for `mtp` the
tokens a round. Speeds here are not measurements (the drafters are eager).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_specloop import Drafter, MTPChainDrafter, MTPGraphDrafter, SpecDecoder  # noqa: E402
from tools.fn_draft_trace import prompt_ids  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


class OracleDrafter(Drafter):
    """Trees from a known continuation `truth` (truth[0] = the first anchor) with planted errors."""

    def __init__(self, truth: list[int], depth: int, vocab: int, seed: int, wide: bool = False):
        self.truth, self.depth, self.vocab, self.wide = truth, depth, vocab, wide
        self.rng = random.Random(seed)
        self.i = 0                                    # the anchor's index in truth

    def _wrong(self, avoid: int) -> int:
        while True:
            t = self.rng.randrange(self.vocab)
            if t != avoid:
                return t

    def propose(self, anchor: int):
        P, i, rng = self.truth, self.i, self.rng
        toks, par = [anchor], [-1]

        def add(p, t):
            toks.append(t)
            par.append(p)
            return len(toks) - 1
        a = rng.randint(0, min(self.depth, len(P) - 1 - i))
        cur = 0
        for j in range(1, a + 1):
            if len(toks) > 26:                        # a block is at most 32 rows
                break
            if rng.random() < (0.9 if self.wide else 0.4):
                w = add(cur, self._wrong(P[i + j]))
                for _ in range(rng.randint(0, 2) if self.wide else int(rng.random() < 0.5)):
                    w = add(w, self._wrong(-1))
            cur = add(cur, P[i + j])
            a = j
        nxt = P[i + a + 1] if i + a + 1 < len(P) else -1
        add(cur, self._wrong(nxt))
        return toks, par

    def accepted(self, stack, nexts):
        self.i += len(nexts)


def plain(eng: FlashNextEngine, ids: list[int], new: int, eos: set) -> tuple[list[int], float]:
    global PLAIN_ST
    st = PLAIN_ST = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st)
    g = DecodeGraph(eng, st)
    g.capture()
    out = [int(logits[0].argmax())]
    torch.cuda.synchronize()
    t0 = time.time()
    while len(out) < new and out[-1] not in eos:
        out.append(g.greedy(out[-1]))
    return out, (len(out) - 1) / (time.time() - t0)


PLAIN_ST = None


@torch.no_grad()
def state_equal(eng: FlashNextEngine, a, b, run: list[int]) -> dict:
    """After both runs: the speculative state `a` against the plain one `b` (the rollback check:
    rejected rows leave no trace). The plain state first decodes on, one token at a time, over the
    tokens the speculative run committed past it (`run`: prompt + every emitted token), so both hold
    the same length; then the GDN and conv states, the PLE conv, the KV rows, the indexer keys and
    the pooled keys up to the length must be equal bit for bit."""
    eng.flush()
    while b.length < a.length:
        eng.decode_eager(run[b.length], b)
    L = min(a.length, b.length)
    out = {"lengths": [a.length, b.length]}
    out["kv"] = all(torch.equal(a.k[li][:, :L], b.k[li][:, :L]) and torch.equal(a.v[li][:, :L], b.v[li][:, :L])
                    for li in a.k)
    out["ik"] = all(torch.equal(a.ik[li][:L], b.ik[li][:L]) for li in a.ik)
    out["pool"] = all(torch.equal(a.pool[li][:L // 4], b.pool[li][:L // 4]) for li in a.pool)
    if a.length == b.length:
        out["gdn"] = all(torch.equal(a.S[li], b.S[li]) and torch.equal(a.conv[li], b.conv[li]) for li in a.S)
        out["ple"] = all(torch.equal(a.ple_conv[li], b.ple_conv[li]) for li in a.ple_conv)
    return out


class RecordingMTP(MTPGraphDrafter):
    """The graph MTP drafter, keeping every committed row's streams (for the MTP rollback check)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []

    def prime(self, stack, ids, first):
        super().prime(stack, ids, first)
        self.rows.append(stack[:len(ids)].clone())

    def accepted(self, stack, nexts):
        super().accepted(stack, nexts)
        self.rows.append(stack.clone())


@torch.no_grad()
def mtp_rollback(eng: FlashNextEngine, dr: "RecordingMTP", run: list[int]) -> dict:
    """The MTP cache after the run against a cache fed once, by the reference MTP forward, with the
    committed rows only: every committed position's K, V and indexer key within the fused-vs-reference
    arithmetic (a rejected draft left at a committed position would be an O(1) difference there)."""
    from engine.models.flashnext.ref import MTP
    st, m = dr.st, dr.st.length
    hist = torch.cat(dr.rows)[:m]
    ref = eng.new_mtp_state(eng.max_len)
    for i in range(0, m, 2048):
        j = min(i + 2048, m)
        eng.mtp_feed(hist[i:j], torch.tensor(run[i + 1:j + 1]), ref)
    out = {"positions": m}
    for name, a, b in (("k", st.k[MTP], ref.k[MTP]), ("v", st.v[MTP], ref.v[MTP]), ("ik", st.ik[MTP], ref.ik[MTP])):
        a, b = (a[:, :m] if a.dim() == 3 else a[:m]).float(), (b[:, :m] if b.dim() == 3 else b[:m]).float()
        err = ((a - b).abs().amax(dim=-1) / b.abs().amax(dim=-1).clamp_min(1e-3))
        err = err.amax(dim=0) if err.dim() == 2 else err
        out[name] = {"worst_rel": round(err.max().item(), 4), "at": int(err.argmax()), "p99_rel": round(err.quantile(0.99).item(), 4)}
    return out


def spec(eng: FlashNextEngine, ids: list[int], new: int, eos: set, mode: str, truth: list[int], depth: int,
         seed: int) -> tuple[list[int], dict]:
    st = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st, keep_stack=not mode.startswith("oracle"))
    first = int(logits[0].argmax())
    if mode.startswith("oracle"):
        dr = OracleDrafter(truth, depth, eng.cfg.vocab_size, seed, wide=mode == "oracle-wide")
    else:
        dr = (RecordingMTP if mode == "mtp-graph" else MTPChainDrafter)(eng, depth, eng.max_len)
        dr.prime(eng.prompt_stack, ids, first)
    sd = SpecDecoder(eng, st, dr, graphs=mode != "oracle-eager", log_union=True)
    out = sd.generate(first, new, eos)
    s = dict(sd.stats)
    s["union_experts_a_layer"] = {M: round(u[1] / u[0] / eng.cfg.num_hidden_layers, 1) for M, u in sorted(sd.union.items())}
    s["state_equal_plain"] = state_equal(eng, st, PLAIN_ST, list(ids) + sd.full)
    if mode == "mtp-graph":
        s["mtp_rollback"] = mtp_rollback(eng, dr, list(ids) + sd.full)
    s["tokens_a_round"] = round((len(out) - 1) / max(s["rounds"], 1), 3)
    for k in ("draft_s", "verify_s", "commit_s"):
        s[k] = round(s[k], 2)
    return out, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--modes", default="oracle,oracle-eager,mtp,mtp-graph")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    modes = a.modes.split(",")
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=any(m.startswith("mtp") for m in modes))
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.max_len)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": a.ctx, "new": a.new, "depth": a.depth, "modes": modes, "seed": a.seed, "greedy": True,
                     "thinking": False, "kv_dtype": "bf16", "max_len": a.max_len}, "classes": {}}
    sc = eng.self_check(class_prompt(tok, "chat", 0))
    res["self_check"] = sc
    for c in a.classes.split(","):
        ids = prompt_ids(tok, c, a.ctx)          # a class, or one of the extra code requests
        P, tps = plain(eng, ids, a.new, eos)
        row = {"prompt_tokens": len(ids), "plain_tokens": len(P), "plain_tok_s": round(tps, 2), "modes": {}}
        for m in modes:
            S, s = spec(eng, ids, a.new, eos, m, P, a.depth, a.seed)
            diff = next((i for i in range(min(len(S), len(P))) if S[i] != P[i]), None)
            if diff is None and len(S) != len(P):
                diff = min(len(S), len(P))
            s.update(equal=S == P, first_diff=diff, tokens=len(S))
            se = s["state_equal_plain"]
            s["state_ok"] = all(v for k, v in se.items() if k != "lengths")
            row["modes"][m] = s
            print(f"[spec] {c} {m}: {json.dumps(s)}", flush=True)
        row["text_head"] = tok.decode(P[:40], skip_special_tokens=True)
        res["classes"][c] = row
    res["mem_end_gib"] = mem_available_gib()
    res["all_equal"] = all(m["equal"] and m["state_ok"] for r in res["classes"].values() for m in r["modes"].values())
    print(f"[spec] all_equal={res['all_equal']}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
