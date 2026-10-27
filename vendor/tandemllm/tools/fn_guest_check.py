"""The resident prefix's guest path on Flash-Next, in one process (the P0 follow-up of 2026-10-12).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_guest_check.py --pack ~/flashnext-packs/3.05

The server answered a 6,000-token prompt differently from a cold prefill when it came after a
1,000-token guest of a 40,000-token resident conversation (the three prompts share a head). Here the
same sequence runs through engine/cache.py `prefill` with a resident prefix as the server builds it
(chunk 4,096, a stash, a small park and pin): A (40,000), B (1,000, a guest, then a few decode steps
as its answer), C (6,000). C's last logits are compared with a cold prefill's, and so is every section
of the state and every view row below C's length, so the first buffer that differs is named.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.models.flashnext.serving import build  # noqa: E402
from tools.fn_fit_run import prompt  # noqa: E402

GIB = 1 << 30


def ids_of(tok, pack, n):
    text, _ = prompt(pack, n)
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(chat, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def decode(eng, lg, n, steps):
    t = int(lg.view(-1).argmax())
    for i in range(steps):
        lg = eng.forward(torch.tensor([t], device="cuda"), n + i)
        t = int(lg.view(-1).argmax())


def state_of(eng, n):
    secs = {k: v.clone() for k, v in eng.cache_state().sections().items()}
    views = {name: t.narrow(ax, 0, n).clone() for name, t, ax in eng.cache_state().views()}
    return secs, views


def diff_sections(a, b) -> list:
    return [k for k in a if not torch.equal(a[k], b[k])]


def run(eng, tok, pack, A, B, C, cold, cs, cv, steps, guest=True, label=""):
    """A, then (B a guest with `steps` decode steps), then C through the resident prefix; C's logits,
    sections and views against the cold run's, and the anchor C resumes from against what A left."""
    res = cache.ResidentPrefix(int(2.5 * GIB), 4096, stash_bytes=GIB, guest_max=8192, pin_bytes=GIB // 4,
                               park_bytes=GIB // 4, end=True)
    store = cache.StateStore(2 * GIB, chunk=4096, max_entry_bytes=GIB, per_prefill=1)
    kw = dict(chunk=4096, resident=res, store=store, checkpoint=True)
    eng.reset()
    lg = cache.prefill(eng, None, A, "cuda", reach=len(A) + 120, **kw)[0]
    decode(eng, lg, len(A), 8)
    kept = {b: tuple(t.clone() for t in buf) for b, buf in res.anchors.items()}
    if guest:
        lg = cache.prefill(eng, None, B, "cuda", reach=len(B) + 120, **kw)[0]
        print(f"[guest] {label} B a guest: stash {res._stash is not None and (res._stash['start'], res._stash['rows'])}",
              flush=True)
        decode(eng, lg, len(B), steps)
        moved = [b for b in kept if b in res.anchors
                 and any(not torch.equal(x, y) for x, y in zip(kept[b], res.anchors[b]))]
        print(f"[guest] {label} anchors changed by the guest: {moved or 'none'} of {sorted(kept)}", flush=True)
    info = {}
    warm = cache.prefill(eng, None, C, "cuda", reach=len(C) + 120, info=info, **kw)
    lw, reused = warm[0], warm[1]
    ws, wv = state_of(eng, len(C))
    print(f"[guest] {label} C resumed at {reused} ({info.get('kind')}); logits max diff "
          f"{float((lw - cold).abs().max()):.4g}, argmax same {int(lw.argmax()) == int(cold.argmax())}", flush=True)
    bad = diff_sections(cs, ws)
    print(f"[guest] {label} sections differing: {len(bad)} of {len(cs)} {bad[:12]}", flush=True)
    for k in cv:
        d = (cv[k].float() - wv[k].float()).abs()
        if d.max() > 0:
            ax = 1 if k.startswith(("k.", "v.")) else 0
            rows = d.amax(dim=tuple(i for i in range(d.dim()) if i != ax)).nonzero().view(-1)
            print(f"[guest] {label} view {k}: rows {int(rows[0])}..{int(rows[-1])} differ ({rows.numel()} rows)",
                  flush=True)
            break
    return res, reused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--steps", type=int, default=64, help="the guest's answer, decode steps")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    pack = os.path.expanduser(a.pack)
    tok = AutoTokenizer.from_pretrained(pack)
    eng, _, _ = build(pack, 65536, tok, spec=False, log=lambda s: None)
    A, B, C = (ids_of(tok, pack, n) for n in (40000, 1000, 6000))
    # the truth: C cold
    eng.reset()
    cold = cache.prefill(eng, None, C, "cuda", chunk=4096)[0].clone()
    cs, cv = state_of(eng, len(C))
    # the state a cold prefill has at each anchor C can resume from
    at = {}
    for b in (4096,):
        eng.reset()
        cache.prefill(eng, None, C[:b], "cuda", chunk=4096)
        at[b] = {k: v.clone() for k, v in eng.cache_state().sections().items()}
    run(eng, tok, pack, A, B, C, cold, cs, cv, a.steps, guest=False, label="no-guest")
    run(eng, tok, pack, A, B, C, cold, cs, cv, 0, label="guest-0")
    res, reused = run(eng, tok, pack, A, B, C, cold, cs, cv, a.steps, label=f"guest-{a.steps}")
    if reused in at:
        # the anchor's state against the cold prefill's at the same length
        names = list(eng.cache_state().sections())
        anc = res.anchors.get(reused)
        if anc is not None:
            bad = [n for n, x in zip(names, anc) if not torch.equal(x, at[reused][n])]
            print(f"[guest] anchor {reused} against a cold prefill to {reused}: {len(bad)} sections differ {bad[:8]}",
                  flush=True)


if __name__ == "__main__":
    main()
