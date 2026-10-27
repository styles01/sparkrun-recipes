"""ENG-339: one quality comparison for every Bolt weight set (EXL3 levels, NVIDIA's IQ2 GGUF), through this
engine, against BF16 logits stored once.

    python tools/nh_quality.py eval --model <pack dir | GGUF dir> --bundle <dir> --json out.json [--rows 8]
                                    [--same <logits of another engine on the same pack>]
    python tools/nh_quality.py compare a.json b.json

Bolt's BF16 checkpoint (150 GB) does not fit the Spark beside anything, so the reference is a bundle made on
a cloud GPU (exllamav3 model_diff with BF16 as model B, 8 wikitext rows x 1,024):
`input_ids.safetensors` and `logits_bf16.safetensors` ("tensor.<row>", [1, T, vocab]). `eval` prefills each
row through the engine (teacher forced, every row's logits in fp32) and compares position by position:
KL(BF16 || set) of the softmax over the whole vocabulary, the top-1 agreement, how many disagreements are near
ties (the reference's log-prob margin between its top two under NEAR nats), and the set's perplexity on the
rows' next tokens. `--same` adds the top-1 agreement with another engine's logits of the same weights
(exllamav3's of an EXL3 pack: the bundle's `logits_exl3_<level>.safetensors`).

`compare`: the mean of the per-row KL differences between two sets and a bootstrap 95 % interval over the rows
(the same rows in both, so the difference is paired).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

#: a disagreement is a near tie when the reference's top-two log-prob margin is under this (nats)
NEAR = 0.5


def metrics(ref_logits: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor | None = None) -> dict:
    """ref_logits, logits [T, V] (any float type; compared over the narrower vocabulary), targets [T - 1] the
    next tokens -> KL(ref || set) per position, top-1 agreement, near-tie disagreements, the set's NLL."""
    V = min(ref_logits.shape[-1], logits.shape[-1])
    lr = torch.log_softmax(ref_logits[:, :V].float(), -1)
    lp = torch.log_softmax(logits[:, :V].float(), -1)
    kl = (lr.exp() * (lr - lp)).sum(-1)
    a, b = lr.argmax(-1), lp.argmax(-1)
    differ = a != b
    top2 = lr.topk(2, -1).values
    near = int((differ & ((top2[:, 0] - top2[:, 1]) < NEAR)).sum())
    out = {"positions": int(kl.numel()), "kl": float(kl.mean()), "kl_max": float(kl.max()),
           "top1": int((~differ).sum()), "differ": int(differ.sum()), "near_ties": near}
    if targets is not None:
        out["nll"] = float(-lp[:-1].gather(-1, targets.view(-1, 1).long()).mean())
    return out


def bundle(path: str) -> tuple[list, list]:
    """(rows of token ids [T], the BF16 logits [T, V] of each row) of a reference bundle."""
    from safetensors import safe_open
    with safe_open(os.path.join(path, "input_ids.safetensors"), "pt") as f:
        ids = [f.get_tensor(k).view(-1) for k in sorted(f.keys(), key=lambda k: int(k.rsplit(".", 1)[1]))]
    return ids, logits_file(os.path.join(path, "logits_bf16.safetensors"))


def logits_file(path: str) -> list:
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.rsplit(".", 1)[1]))
        return [f.get_tensor(k).view(-1, f.get_slice(k).get_shape()[-1]) for k in keys]


def _engine(path: str, max_len: int):
    from engine.models.nemotron_h.engine import NemotronHEngine
    from engine.models.nemotron_h.load import load_any
    idx, W, experts, timing = load_any(path, "cuda", mtp=False, log=None)
    return NemotronHEngine(idx.cfg, W, experts, device="cuda", max_len=max_len), timing


@torch.no_grad()
def cmd_eval(a) -> None:
    ids, refs = bundle(a.bundle)
    same = logits_file(a.same) if a.same else None
    n = min(a.rows or len(ids), len(ids))
    eng, timing = _engine(a.model, max(int(r.numel()) for r in ids[:n]) + 64)
    rows, t0 = [], time.time()
    for i in range(n):
        st = eng.new_state(ids[i].numel() + 64)
        lg = eng.prefill(ids[i].to("cuda"), st, head="all").float()
        m = metrics(refs[i].to("cuda"), lg, ids[i][1:].to("cuda"))
        if same is not None:
            m["top1_same"] = int((same[i].to("cuda")[:, :lg.shape[1]].float().argmax(-1) == lg.argmax(-1)).sum())
        rows.append(m)
        print(json.dumps({"row": i, **{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()},
                          "s": round(time.time() - t0)}), flush=True)
        del lg, st
    P = sum(r["positions"] for r in rows)
    out = {"model": a.model, "bundle": a.bundle, "rows": n, "positions": P,
           "kl": sum(r["kl"] * r["positions"] for r in rows) / P,
           "top1_pct": round(100 * sum(r["top1"] for r in rows) / P, 2),
           "near_tie_pct_of_differ": round(100 * sum(r["near_ties"] for r in rows) / max(1, sum(r["differ"] for r in rows)), 1),
           "ppl": float(torch.tensor([r["nll"] for r in rows]).mean().exp()),
           "row_kl": [r["kl"] for r in rows], "load": timing}
    if same is not None:
        out["top1_same_pct"] = round(100 * sum(r["top1_same"] for r in rows) / P, 2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("row_kl", "load")}))
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)


def cmd_compare(a) -> None:
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    g = torch.Generator().manual_seed(0)
    d = torch.tensor(y["row_kl"]) - torch.tensor(x["row_kl"])
    boot = torch.stack([d[torch.randint(0, len(d), (len(d),), generator=g)].mean() for _ in range(2000)])
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    print(json.dumps({"kl_a": round(x["kl"], 5), "kl_b": round(y["kl"], 5), "b_minus_a": round(float(d.mean()), 5),
                      "ci95": [round(lo, 5), round(hi, 5)], "b_lower_rows": int((d < 0).sum()), "rows": len(d)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--bundle", required=True)
    e.add_argument("--json", required=True)
    e.add_argument("--rows", type=int, default=0)
    e.add_argument("--same", default=None)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"eval": cmd_eval, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
