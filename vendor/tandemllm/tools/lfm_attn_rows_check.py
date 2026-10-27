"""LFM2.5's split attention (engine/models/lfm2_moe/kernels.py `attn_rows`) against another tree's: bytes.

ENG-363 sized the split partials for a class's largest row count and reads the first rows' share; the
kernel and its grid are the same. This runs both modules on the same random caches and queries (seed 0),
at LFM2.5's heads (32 query, 8 KV, 64 wide) and a small shape, two cache sizes, two classes each, M = 1..32
rows of a chain: every call alone with no buffers, then one buffer dict a class across M = 8, 4, 1, 16,
2, 32, 1 (as `Lfm2MoeEngine._rows` passes `rb["attn"].setdefault(ns, {})`). Every output must equal the
reference's lone call byte for byte. Needs a GPU and Triton.

    git show v0.11.1:engine/models/lfm2_moe/kernels.py > /tmp/lfm_kernels_ref.py
    python tools/lfm_attn_rows_check.py --ref /tmp/lfm_kernels_ref.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine.models.lfm2_moe import kernels as K  # noqa: E402

SHAPES = ((32, 8, 64), (8, 2, 64))           # (query heads, KV heads, head width)
CACHES = ((700, 600), (40000, 33000))        # (cache rows, block start)
ROWS = (1, 2, 4, 8, 16, 32)
SHARED = (8, 4, 1, 16, 2, 32, 1)


def _load_ref(path: str):
    spec = importlib.util.spec_from_file_location("lfm_kernels_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _outputs(mod) -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g).to(torch.bfloat16).cuda()

    out = {}
    for hq, hk, d in SHAPES:
        for cap, base in CACHES:
            kc, vc, q = rnd(hk, cap, d), rnd(hk, cap, d), rnd(max(ROWS), hq, d)
            anc = torch.zeros(max(ROWS), mod.ANC, dtype=torch.int32, device="cuda")
            ctl = torch.zeros(max(ROWS), 2, dtype=torch.int32, device="cuda")
            for i in range(max(ROWS)):           # a chain: row i sees rows 0..i of the block
                n = min(i + 1, mod.ANC)
                anc[i, :n] = base + torch.arange(n, dtype=torch.int32)
                ctl[i] = torch.tensor([base + n, base], dtype=torch.int32)
            c = mod.class_of(base + max(ROWS) + 1, cap)
            for ns in (c, 2 * c):
                for m in ROWS:
                    out[(hq, hk, d, cap, ns, m, "alone")] = mod.attn_rows(q[:m], kc, vc, ctl[:m], anc[:m], ns).cpu()
                bufs: dict = {}
                for j, m in enumerate(SHARED):
                    out[(hq, hk, d, cap, ns, m, f"shared{j}")] = mod.attn_rows(q[:m], kc, vc, ctl[:m], anc[:m], ns,
                                                                               bufs=bufs).cpu()
    torch.cuda.synchronize()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="another tree's engine/models/lfm2_moe/kernels.py")
    a = ap.parse_args()
    if not (K.HAVE_TRITON and torch.cuda.is_available()):
        print("needs Triton on a GPU")
        return 2
    new, ref = _outputs(K), _outputs(_load_ref(a.ref))
    alone = {k[:6]: v for k, v in ref.items() if k[6] == "alone"}
    diff = [list(k) for k in new if not torch.equal(new[k], ref[k]) or not torch.equal(new[k], alone[k[:6]])]
    print(json.dumps({"calls": len(new), "differ": len(diff), "first": diff[:5], "same": not diff}))
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
