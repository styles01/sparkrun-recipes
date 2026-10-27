"""The Qwen3.5 MoE trunk behind the server's own generation loop (ENG-391): Flash-Next's serving adapter
(engine/models/flashnext/serving.py) over `Qwen35MoeEngine`, so the request path is the 27B's code --
penalties, the think budget, pattern stops, logprobs, tool calls, the caches on the prefill's 4,096-row
grid -- and the drafter is the shared MTP tree drafter over the pack's own MTP layer, with the lookup
in the same cut (StairCut, priced online or from ops/prices/<model>/<profile>.<device>.json).

  forward, one token      the captured decode step (a graph per attention class)
  forward, more tokens    the prefill kernels, every row's post-norm hidden kept for the MTP layer
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks
  commit_tree             `commit` from captured graphs (the folded GDN commit included)
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.methods.staircut import PriceModel, StairCut
from engine.models.adapter import Capabilities
from engine.models.flashnext.serving import _KV, FlashNextServing
from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
from engine.settings import SETTINGS as _S

#: fingerprint v3: the files that compute this family's rows (Flash-Next's kernels and engine follow it:
#: the class is built on them)
MODULES = ("engine/models/qwen3_5_moe/__init__.py", "engine/models/qwen3_5_moe/config.py",
           "engine/models/qwen3_5_moe/engine.py", "engine/models/qwen3_5_moe/kernels.py",
           "engine/models/qwen3_5_moe/load.py", "engine/models/qwen3_5_moe/pack.py",
           "engine/models/qwen3_5_moe/ref.py", "engine/models/flashnext/engine.py",
           "engine/models/flashnext/load.py", "engine/models/flashnext/ref.py", "engine/models/flashnext/vision.py",
           "engine/settings.py")
TABLES = ("ops/fn-exl3-tiles.json", "ops/fn-dense-splits.json")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: the model's name in ops/prices/ and models/
MODEL_ID = "ornith-1.5-122b"


def manifest() -> Manifest:
    return Manifest(arch="qwen3_5_moe", modules=MODULES, data=TABLES)


class Qwen35MoeServing(FlashNextServing):
    """`Qwen35MoeEngine` with one state and its graphs, in the server's engine interface. One sequence
    slot for now: the joint verify of several slots is Flash-Next's mixer-shaped pass (not built here)."""

    caps = Capabilities(caches=True, side_slots=False, anchor_grid=4096, shadow=False, side_lane=False)

    def __init__(self, fn, max_len: int):
        super().__init__(fn, max_len)
        self.kv = _TapKV(self)
        self.dflash2 = None                 # the DFlash2 head, when the drafter is DFlash2's

    def restart(self) -> None:
        super().restart()
        self.st.tap_from, self.st.tap_gen = 0, getattr(self.st, "tap_gen", 0) + 1

    def price_class(self, n: int) -> int:
        """The draft prices' context class at length `n`: 0 up to 4,096, then 4,096 x 2^k (a verify's
        attention grows with the context)."""
        c = 4096
        if n <= c:
            return 0
        while c < n and c < 262_144:
            c *= 2
        return c

    def manifest(self) -> Manifest:
        return manifest()

    def cost(self):
        """Flash-Next's cost with this trunk's prefill workspace (SRV-655). The prompt's attention
        (`Qwen35MoeEngine._attn_pre`) repeats the cache's K and V to every query head for SDPA: Hq x
        head_dim x 2 bytes, twice, a token of context (32,768 B on Ornith), over the floor of a
        4,096-row chunk's activations. Flash-Next's line (76,309 B a token, capped at 10.75 GiB) priced
        Ornith's 148,184-token prefill at 11.5 GB, which the measured one never took (3.8 GB at
        104,807 tokens, live 2026-10-21), and refused it for memory twice."""
        c, base = self.cfg, super().cost()
        per = int(c.num_attention_heads) * int(c.head_dim) * 2 * 2
        floor = int(2.75 * (1 << 30))
        return replace(base, workspace_floor=floor, workspace_per_token=per,
                       workspace_cap=floor + per * int(self.max_len))

    def capture_cost(self, n: int) -> float:
        """0: a turn is not sent to the plain step for the graphs its class lacks. The idle captures take the
        class of the state's last length, which on a conductor box is a short side question's, so a class's
        graphs stayed missing and every 3-4k-token turn of the A/B took the plain step (2026-10-20); the
        first drafted rounds of a class capture its graphs and are not priced."""
        return 0.0


class _TapKV(_KV):
    """`kv` with the taps' bookkeeping: a length set by a restore (the caches) leaves the taps below it
    another sequence's, so the block drafter's context starts there."""

    @property
    def length(self) -> int:
        return _KV.length.fget(self)

    @length.setter
    def length(self, v: int) -> None:
        _KV.length.fset(self, v)
        if int(v):
            st = self._s.st
            st.tap_from, st.tap_gen = int(v), getattr(st, "tap_gen", 0) + 1


def profile_of(pack: str) -> str:
    return os.environ.get("QSE_WEIGHTS") or "ornith-1.5-122b-" + os.path.basename(os.path.normpath(pack))


def prices(pack: str, log=print) -> PriceModel:
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(pack)}.{device_id()}.json"),
                           log=log)


#: the DFlash2 gate's floor: the mean run of drafted tokens the trunk's greedy keeps (a draft wired wrong keeps none)
DFLASH2_GATE = 1.0
#: the lookup's shortest match beside a block drafter (as NemotronH's beside LiLiCorr)
LOOK_MIN_ORDER_BLOCK = 6


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 3, mark=lambda stage: None, log=print,
          corpus: str | None = None, kv_pool=None, dflash2: str | None = None):
    """The pack's weights, the engine in the loop's interface, and with `spec` the MTP tree drafter over
    the pack's MTP layer (the hot draft head of FN_DRAFT_HOT ids when the vocabulary is the 27B's), the
    lookup with Q35M_LOOKUP, gated by the load-time self-check of the expert kernels (a failed check
    serves plain decode). Returns (serving, drafter or None, info)."""
    from engine.models.qwen3_5_moe.load import load as q_load
    idx, W, banks, timing = q_load(pack, "cuda", mtp=spec and not dflash2, log=log)
    mark("weights")
    eng_ = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=max_len)
    d2 = None
    if spec and dflash2:
        from engine.models.qwen3_5_moe import dflash2 as D2
        try:
            dc = D2.config_of(dflash2)
            eng_.set_taps(dc["target_layer_ids"], dc["window"])
            d2 = {"path": dflash2}
        except Exception as e:                              # noqa: BLE001  (the lookup drafts alone)
            d2 = {"ok": False, "why": f"DFlash2 config: {type(e).__name__}: {e}"}
    if kv_pool is not None:
        kv_pool()
    eng = Qwen35MoeServing(eng_, max_len)
    mark("engine")
    drafter, check, hot = None, None, int(_S.get("FN_DRAFT_HOT"))
    if spec and dflash2:
        from engine.models.qwen3_5_moe import dflash2 as D2
        hot = int(_S.get("FN_DRAFT_HOT")) if D2.HOT and idx.cfg.vocab_size == 248_320 else 0
        if hot:
            eng_.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tok(text, add_special_tokens=False)["input_ids"]
        check = eng_.self_check(prompt, log=log)
        head = None
        if check["ok"] and "path" in d2:
            from engine.drafters.lilicorr import gate
            from engine.models.qwen3_5_moe import dflash2 as D2
            try:
                head = D2.build_head(eng, dflash2, log=log)
                g = gate(eng_, head, eng_.new_state(min(max_len, len(prompt) + 128)), prompt)
                ok = g["mean_run"] >= DFLASH2_GATE
                d2 = {"ok": ok, "gate": g, **({} if ok else {"why": f"DFlash2 gate: mean run {g['mean_run']}"})}
                log(f"[dflash2] gate {'PASS' if ok else 'FAIL'}: mean run {g['mean_run']} of {head.depth} "
                    f"over {g['positions']} positions")
            except Exception as e:                          # noqa: BLE001
                d2 = {"ok": False, "why": f"DFlash2 failed to load: {type(e).__name__}: {e}"}
            if not d2["ok"]:
                head = None
        if check["ok"]:
            from engine.drafters.ngram import NgramDrafter
            from engine.methods.mtp_drafter import NoHead
            from engine.tokfp import fingerprint
            ngram = None
            if _S.get("Q35M_LOOKUP") == "1":
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack),
                                     min_order=3 if head is None else LOOK_MIN_ORDER_BLOCK, max_depth=16,
                                     min_corpus_order=8)
            if head is None:
                log(f"[server] {d2.get('why', 'DFlash2 off')}: the lookup drafts alone")
            eng.dflash2 = head
            drafter = MTPTreeDrafter(eng, head if head is not None else NoHead(),
                                     StairCut(prices(pack, log), siblings=_S.get("Q35M_SIBLINGS") == "1"), ngram)
            drafter.anchor_backoff = int(_S.get("Q35M_ANCHOR_BACKOFF") or 0)
            drafter.capture_all()
        mark("drafters")
        info = {"pack": pack, "load": timing, "self_check": check,
                "draft_head": ("dflash2" + (" nvfp4" if D2.NVFP4 else "") + (f" hot {hot}" if hot else "")) if head else None,
                "dflash2": d2, "depth": head.depth if head is not None else 0}
        return eng, drafter, info
    spec = spec and timing.get("mtp", False)
    if spec:
        if hot and idx.cfg.vocab_size == 248_320:
            eng_.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        else:
            hot = 0
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check = eng_.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
        if check["ok"]:
            ngram = None
            if _S.get("Q35M_LOOKUP") == "1":
                from engine.drafters.ngram import NgramDrafter
                from engine.tokfp import fingerprint
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack), min_order=3,
                                     max_depth=16, min_corpus_order=8)
                log(f"[server] Qwen3.5 MoE lookup in the cut: corpus "
                    f"{(corpus or 'none') if ngram.corpus is not None else 'none'}")
            drafter = MTPTreeDrafter(eng, eng.add_mtp_head(depth),
                                     StairCut(prices(pack, log), siblings=_S.get("Q35M_SIBLINGS") == "1"), ngram)
            drafter.anchor_backoff = int(_S.get("Q35M_ANCHOR_BACKOFF") or 0)
            drafter.capture_all()
        else:
            log("[server] Qwen3.5 MoE self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec and hot else ("full" if spec else None),
            "depth": depth if drafter is not None else 0}
    return eng, drafter, info


def re_dflash2(model: str) -> str:
    """The DFlash2 checkpoint beside a pack: <models>/<model name before -exl3>-dflash2."""
    d = os.path.normpath(model)
    base = os.path.basename(d).split("-exl3")[0]
    return os.path.join(os.path.dirname(d), base + "-dflash2")


def serve(a, ctx) -> dict:
    """A Qwen3.5 MoE EXL3 pack for the server (`--model <pack>`, the registry's `serve`): the engine and
    its MTP drafter in the server loop's interface; the pack's EXL3 vision tower when it has one."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    spec = a.drafter != "none"
    if a.drafter not in ("mtp", "none"):
        ctx.log(f"[server] the Qwen3.5 MoE trunk drafts with its own MTP layer: --drafter {a.drafter} taken as mtp")
    tok = AutoTokenizer.from_pretrained(a.model)
    depth = a.depth if a.depth >= 2 else 3
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    # `--draft-head dflash2`: the DFlash2 checkpoint of QSE_ORN_DFLASH2 (the profile's), else <model>-dflash2
    d2 = None
    if spec and str(getattr(a, "draft_head", None) or "") == "dflash2":
        d2 = os.environ.get("QSE_ORN_DFLASH2") or re_dflash2(a.model)
    eng, drafter, info = build(a.model, a.max_len, tok, spec=spec, depth=depth, mark=ctx.mark, log=ctx.log,
                               corpus=a.corpus, kv_pool=pool, dflash2=d2)
    if d2:
        depth = info["depth"]

    def tower_of(vcfg):
        from engine.models.flashnext.vision import load_tower
        from engine.models.qwen3_5_moe.pack import Qwen35MoePackIndex
        return load_tower(Qwen35MoePackIndex.open(a.model), a.model)
    ctx.vision(a.model, tower_of)
    ctx.mark("vision")
    if drafter is not None:
        a.depth = depth
        if drafter.ngram is not None:
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
    what = ("DFlash2 block of " + str(depth) if d2 and info.get("draft_head") == "dflash2" else
            "lookup alone" if d2 else "MTP chain, depth " + str(a.depth))
    describe = (f"Qwen3.5 MoE {a.model}: load {info['load'].get('total_s')} s, drafter "
                f"{what + ' + StairCut' if drafter is not None else 'none'}"
                f"{' + lookup' if getattr(drafter, 'ngram', None) is not None else ''}, "
                f"max_len {a.max_len:,},")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "exl3", "weights": a.model,
                                         "drafters": (what if drafter is not None else None),
                                         "dflash2": info.get("dflash2"),
                                         "draft_head": info["draft_head"], "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"]}}}
