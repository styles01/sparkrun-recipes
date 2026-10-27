"""RND-35: the checkpoint's own MTP layer as a slot-0 scorer, recorded at the lattice positions.

For every trace of a lattice directory (`record_lattice.py` output: *.json + *.lattice.npz) the target
is teacher-forced over prompt + output, the MTP layer (engine/drafters/mtp.py's wiring) runs over the
whole sequence at once (causal), and at every recorded lattice position p (anchor = the token at p)
the MTP's log-probability of each of the lattice's 16 slot-0 candidates is stored -- the MTP row that
embeds the token at p with the target's hidden state at p - 1, i.e. the target-side prediction of
p + 1, exactly slot 0. Writes <trace>.mtp.npz {logp [N, k] fp32, top1 [N], in_cand [N]} beside each
trace. CPU replay: tools/mtp_blend.py.

Usage: mtp_slot0.py --traces DIR [--hidden post|pre]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.mtp import MTPDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine, head_logits, linear, rms_norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--hidden", default="post", choices=("post", "pre"))
    ap.add_argument("--max-len", type=int, default=4608)
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    mtp = MTPDrafter(eng, max_len=a.max_len, hidden=a.hidden)
    agree = tot = 0
    for path in sorted(glob.glob(os.path.join(a.traces, "*.json"))):
        npz = path[:-5] + ".lattice.npz"
        if not os.path.exists(npz):
            continue
        tr = json.load(open(path))
        z = np.load(npz)
        ids = list(tr["prompt_ids"]) + list(tr["output_ids"])
        P = len(tr["prompt_ids"])
        n = len(ids)
        with torch.no_grad():
            eng.reset()
            hs = []
            for s in range(0, n, a.chunk):
                blk = torch.tensor(ids[s:s + a.chunk], device="cuda")
                eng.forward(blk, start=s, last_only=True)
                h = eng.hidden_post_norm if a.hidden == "post" else eng.hidden_pre_norm
                hs.append(h[0].clone())
            hid = torch.cat(hs)                                            # [n, H]
            mtp.reset()
            emb = F.embedding(torch.tensor([ids[1:]], device="cuda"), w.norm("embed_tokens.weight"))
            xa = rms_norm(emb, w.norm("mtp.pre_fc_norm_embedding.weight"), cfg.rms_norm_eps)
            xb = rms_norm(hid[None, :n - 1], w.norm("mtp.pre_fc_norm_hidden.weight"), cfg.rms_norm_eps)
            x = linear(torch.cat([xa, xb], dim=-1), w.norm("mtp.fc.weight"))
            y = mtp._layer(x, 1, causal=True)                               # row r -> predicts r + 2
            y = rms_norm(y, w.norm("mtp.norm.weight"), cfg.rms_norm_eps)[0]
            at = z["at"].astype(np.int64)                                   # output indices
            rows = torch.tensor(P + at - 1, device="cuda")                  # embeds token at P + at
            logits = head_logits(y[rows], w.norm("lm_head.weight")).float()
            logp = torch.log_softmax(logits, dim=-1)
            cand0 = torch.tensor(z["cand"][:, 0, :].astype(np.int64), device="cuda")    # [N, k]
            lp = logp.gather(1, cand0).cpu().numpy()
            top1 = logits.argmax(-1).cpu().numpy()
        out = np.array(tr["output_ids"])
        true1 = out[np.minimum(at + 1, len(out) - 1)]
        ok = (at + 1) < len(out)
        agree += int(((top1 == true1) & ok).sum())
        tot += int(ok.sum())
        in_cand = (z["cand"][:, 0, :] == top1[:, None]).any(-1)
        np.savez_compressed(path[:-5] + ".mtp.npz", logp=lp.astype(np.float32), top1=top1,
                            in_cand=in_cand)
        print(f"[mtp] {tr['name']:16s} {len(at):4d} positions, top-1 = next token {((top1 == true1) & ok).mean():.3f}",
              flush=True)
    print(f"[mtp] all: MTP top-1 agrees with the true slot-0 token at {agree / max(tot, 1):.3f} of {tot}",
          flush=True)


if __name__ == "__main__":
    main()
