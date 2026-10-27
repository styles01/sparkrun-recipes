"""LFM2 MoE behind the server's own generation loop (SRV-425): an engine with the shapes `server/app.py`
drives every model through (the Flash-Next adapter's, engine/models/flashnext/serving.py).

  forward, one token      the captured decode step (`DecodeGraph`)
  forward, more tokens    `Lfm2MoeEngine.prefill` (chunks on the batched kernels)
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks
  commit_tree             `commit` (captured per shape)
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

so everything above it -- penalties, the think budget and its forced close, pattern stops, logprobs,
the reasoning split, the tool-call parser the model names (`tool_format`), the live view -- is the
server's code. A verified row is the decode step's row bit for bit (ENG-287), so greedy speculation
writes the plain greedy text. `caps` says what is on.
"""

from __future__ import annotations

import os
import time

import torch

from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.staircut import PriceModel
from engine.models.adapter import Capabilities, CostModel
from engine.models.graph_lru import joint_graphs
from engine.models.lfm2_moe import kernels as K
from engine.models.lfm2_moe.engine import CAPTURE_ROWS, PREFILL_CHUNK, DecodeGraph, Lfm2MoeEngine
from engine.models.state import SectionState

#: fingerprint v3: the package's files that compute rows
MODULES = ("engine/models/lfm2_moe/__init__.py", "engine/models/lfm2_moe/config.py",
           "engine/models/lfm2_moe/engine.py", "engine/models/lfm2_moe/kernels.py",
           "engine/models/lfm2_moe/load.py", "engine/models/lfm2_moe/pack.py",
           "engine/models/lfm2_moe/ref.py", "engine/settings.py")
#: the model's name in ops/prices/ and models/
MODEL_ID = "lfm2.5-24b-a2b"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


#: the tables the arithmetic reads (a dense split is a summation order)
TABLES = ("ops/lfm25-dense-splits.json",)


def manifest() -> Manifest:
    return Manifest(arch="lfm2_moe", modules=MODULES, data=TABLES)


def _tensors(o) -> list:
    """The tensors a weight entry holds: a tensor, an EXL3Block's, a layer's experts'."""
    if torch.is_tensor(o):
        return [o]
    if hasattr(o, "blocks"):
        return [t for e in o.blocks for b in e for t in (b.t, b.suh, b.svh)]
    if hasattr(o, "w13"):
        return [o.w13, o.w2]
    if hasattr(o, "trellis") or hasattr(o, "svh"):
        return [o.t, o.suh, o.svh]
    return []


class _KV:
    """The `kv` the server reads: the sequence's length (a pending chain commit lands first) and its
    capacity. Zero starts a new sequence in place (`restart`); any other length keeps the KV rows by
    length -- the conv states are the caller's to restore first (`state.restore`)."""

    def __init__(self, serving: "Lfm2MoeServing"):
        self._s = serving

    @property
    def pages(self):
        return self._s.st.pages

    @property
    def length(self) -> int:
        # what the state holds once its pending work commits (a chain block's rows, a joint plain
        # step's row), read without committing it: another thread reads it too (the health report,
        # the slot placement), and a commit from there while the slot's own loop sat between a
        # joint pass and its commit_tree counted the row twice (2026-10-14, three slots: "LFM2
        # forward at 53, the state holds 54 rows")
        s = self._s
        return s.st.length + (s._chain or 0) + (1 if s._plain_pending else 0)

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"the LFM2 state cannot hold {v} rows (capacity {self.max_len})")
        self._s.st.length = v

    @property
    def max_len(self) -> int:
        return self._s.max_len


class Lfm2CacheState(SectionState):
    """What the caches keep of a sequence (engine/models/state.py): sections the conv states of every
    conv layer ("conv.<layer>", [conv_L_cache - 1, H] bf16); views K and V of every attention layer
    ([kv heads, rows, D], axis 1)."""

    def __init__(self, serving: "Lfm2MoeServing"):
        self.s = serving

    def sections(self) -> dict:
        st = self.s.st
        return {f"conv.{li}": st.conv[li] for li in sorted(st.conv)}

    def views(self) -> list:
        st = self.s.st
        out = []
        for li in sorted(st.k):
            out += [(f"k.{li}", st.k[li], 1), (f"v.{li}", st.v[li], 1)]
        return out


# ENG-E10 / SRV-441: the conductor's router on this model (server/conductor/modelfit.py), measured on 2.5,
# 2026-10-14 (models/lfm2.5-24b-a2b/measurements/conductor-0.9.0.json). With thinking off the model answers a
# class prompt as a chat request (its one-id answers 0 to 2 of 8 right), so a short chat fell back to the
# default class and its notes; and a note at the tail of the user's message is read as part of the request:
# the same note texts cost 2.7x the raw thinking there and 1.38x as a system message. Every part that goes
# there (packages, effort sentences, rules, memory) raised the thinking; the class still sets the level
CONDUCTOR_FIT = {"served_classifier": False, "fallback": "lean", "packages": "core",
                 "no_package_classes": ("chat",), "effort_notes": False, "memory": False, "rules": False}


class Lfm2MoeServing:
    """`Lfm2MoeEngine` with one state and its graphs, in the server's engine interface."""

    # the caches on the prefill's 4,096-token grid (a restore answered as a cold prefill, logprobs equal:
    # tools/fn_cache_check.py on 4.0, 2026-10-14); side slots with joint passes, a plain step joining the
    # round (`joint_plain`: no draft head, so most rounds are plain steps); the shadow reader was
    # validated on Qwen text; the side lane copies the 27B's pending commit
    caps = Capabilities(caches=True, side_slots=True, anchor_grid=PREFILL_CHUNK, shadow=False, joint_verify=True,
                        side_lane=False, joint_plain=True, conductor=CONDUCTOR_FIT)

    def __init__(self, eng: Lfm2MoeEngine, max_len: int, slot=None):
        self.fn, self.cfg, self.max_len = eng, eng.cfg, max_len
        # slot 0 (None) keys its row buffers by row count; a side slot by (its tag, rows), so no slot
        # stages into a buffer another slot's graph reads
        self.slot = slot
        self.st = eng.new_state(max_len, paged=True)
        self.state = Lfm2CacheState(self)
        self.kv = _KV(self)
        self.dg = None
        self.vgraphs: dict = {}
        self.cgraphs: dict = {}
        self.picks = None
        self.tap = None
        self.mm = None                       # no vision tower: the server's image path is off
        self.pos_delta = 0
        self._hidden = None
        self._chain = None
        self._width = 0
        self.commit_ms = 0.0
        self.captured = False
        self.plain_captured = False
        self.heads: dict = {}
        self.sides: list = []
        self._plain_pending = False        # a joint plain step ran; its commit_tree([0]) moves the length
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    # --- the loop's interface -------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    def price_class(self, n: int) -> int:
        """The draft prices' context class of a round at context length `n`: the keys its attention
        programs cover (kernels.class_of: 2,048 keys, doubling to the window)."""
        return K.CHUNK * K.class_of(max(1, int(n)), self.max_len)

    def last_streams(self) -> torch.Tensor:
        """The last committed row's final normed state [1, H] (what a draft head would read next)."""
        return self._hidden[0, -1:] if self._hidden is not None else torch.zeros(1, self.cfg.hidden_size)

    def cost(self) -> CostModel:
        """A snapshot: the conv states whole; K and V of the 10 attention layers a token (bf16)."""
        sec = sum(t.numel() * t.element_size() for t in self.st.conv.values())
        return CostModel(section_bytes=sec, view_bytes_per_token=self.cfg.kv_bytes_per_token(2),
                         parts={"recurrent": 0, "conv": sec})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        """The decode graph of the first class, after the server's warm-up (whose rows are nobody's)."""
        if self.dg is None:
            self.dg = DecodeGraph(self.fn, self.st, self.slot)
        self.kv.length = 0
        return len(self.dg.graphs)

    def draft_heads(self) -> dict:
        """The draft heads this model carries, by name: none yet. Liquid's DSpark drafter for this
        model plugs in here (a head with the MTP tree drafter's interface, engine/methods/mtp_drafter.py)."""
        return self.heads

    def settle(self) -> None:
        self._settle_chain()

    def cache_state(self) -> Lfm2CacheState:
        return self.state

    def resume_at(self, n: int) -> None:
        self.discard_pending()
        self.kv.length = n

    def memory_parts(self) -> dict:
        """The dashboard's memory split (server/memsplit.py): weights (each storage once), the KV rows,
        the conv states."""
        fn = self.fn
        seen, w = set(), 0
        for o in list(fn.W.values()) + list(fn.experts.values()):
            for t in _tensors(o):
                s = t.untyped_storage()
                if s.data_ptr() not in seen:
                    seen.add(s.data_ptr())
                    w += s.nbytes()
        st = self.st
        kv = sum(int(t.numel() * t.element_size()) for d in (st.k, st.v) for t in d.values())
        return {"weights": w, "kv_pool": kv, "drafter_kv": 0,
                "gdn_state": sum(int(t.numel() * t.element_size()) for t in st.conv.values())}

    def resident_files(self) -> list:
        return []

    def discard_pending(self) -> None:
        self._chain = None
        self._plain_pending = False
        self.picks = None

    def trim_pending(self, n: int) -> bool:
        """Back to length `n` inside work not committed yet (a chain block whose commit waits)."""
        if self._plain_pending:
            self.commit_tree([0])
        n, L = int(n), self.st.length
        if self._chain is None:
            return n == L
        if not L <= n <= L + self._chain:
            return False
        self._chain = None
        if n > L:
            self.commit_tree(list(range(n - L)))
        return True

    def window_ok(self, i: int) -> bool:
        return False

    def reset(self) -> None:
        self.restart()

    def restart(self) -> None:
        """A new sequence in the same buffers: the conv states zeroed, the length 0 (KV rows stay)."""
        self._chain = None
        self._plain_pending = False
        self.picks = None
        self.st.reset_inplace()

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the decode graph, more through the
        prefill. Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        self._ensure(start + T + CAPTURE_ROWS)
        if T == 1 and start > 0:
            if self.dg is None:
                self.dg = DecodeGraph(self.fn, self.st, self.slot)
            n0 = self.dg.captures
            lg = self.dg(int(ids[0]))
            self.plain_captured = self.dg.captures != n0
            self._hidden = self.dg.rb["hidden"].view(1, 1, -1)
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st)
        self._hidden = self.fn.prompt_hidden.unsqueeze(0)
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32, its greedy pick in `picks`. Nothing but the block's own KV rows is written until
        `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        M = len(toks)
        self._ensure(start + M + CAPTURE_ROWS)
        n_graphs = len(self.vgraphs)
        self.fn.verify(toks, [int(p) for p in parents], self.st, self.vgraphs, slot=self.slot)
        self.captured = len(self.vgraphs) > n_graphs
        self._width = M
        rb = self.fn._rb[self.fn._key(self.slot, M)]
        self.picks = rb["next"][:M]
        self._hidden = rb["hidden"][:M].unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += M
        return rb["logits"][:M]

    def tree_batchable(self, parents) -> bool:
        """A block a joint pass may take: its rows within the row kernels' 32."""
        return len(parents) <= 32

    @staticmethod
    @torch.no_grad()
    def forward_tree_multi(items: list) -> list:
        """ENG-241: [(serving, block, parents, start)] of sequence slots over one engine, verified in one
        pass (`Lfm2MoeEngine.verify_multi`: the projections, experts and head read once); each slot's
        logits, picks and final rows as its `forward_tree` leaves them, its block committed by its
        `commit_tree`."""
        fn = items[0][0].fn
        if all(block.numel() == 1 for _, block, _, _ in items):
            # every slot takes a plain step (no draft this round, `joint_plain`): one joint decode
            # step from a graph, each state advanced at once; its commit_tree([0]) moves the length
            for s, block, _, start in items:
                s._settle_chain()
                s._at(start)
                s._ensure(start + 1 + CAPTURE_ROWS)
            rbs = fn.decode_multi([(int(block.view(-1)[0]), s.st, s.slot) for s, block, _, _ in items],
                                  graphs=joint_graphs(fn))
            out = []
            for (s, block, _, _), rb in zip(items, rbs):
                s.picks, s._hidden = rb["next"][:1], rb["hidden"][:1].unsqueeze(0)
                s.captured, s._width, s._plain_pending = False, 1, True
                s.stats["decode_steps"] += 1
                out.append(rb["logits"][:1])
            return out
        jobs = []
        for s, block, parents, start in items:
            s._settle_chain()
            s._at(start)
            s._ensure(start + block.numel() + CAPTURE_ROWS)
            jobs.append(([int(t) for t in block.view(-1).tolist()], [int(p) for p in parents], s.st))
        rbs = fn.verify_multi(jobs)
        out = []
        for (s, block, parents, start), rb in zip(items, rbs):
            M = block.numel()
            s.picks, s._hidden = rb["next"][:M], rb["hidden"][:M].unsqueeze(0)
            s.captured, s._width = False, M
            s.stats["verify_blocks"] += 1
            s.stats["verify_rows"] += M
            out.append(rb["logits"][:M])
        return out

    def side_slot(self, max_len: int, drafter0=None) -> tuple:
        """ENG-241: a side slot's engine over this engine's weights (`fn`, the one copy): its own state
        (paged as slot 0's when the pool is on), its own row buffers and graphs; with a drafter, the
        same lookup cut over slot 0's corpus (read-only) from the same price file."""
        e2 = Lfm2MoeServing(self.fn, max_len, slot=f"slot{id(self) % 9973}.{len(self.sides)}")
        self.sides.append(e2)
        d2 = None
        if drafter0 is not None:
            from engine.drafters.ngram import NgramDrafter
            from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead
            from engine.methods.staircut import PriceModel, StairCut
            ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
            if drafter0.ngram is not None:
                ng.corpus = drafter0.ngram.corpus
            c0 = drafter0.cut
            d2 = MTPTreeDrafter(e2, NoHead(), StairCut(PriceModel(c0.prices.table, c0.prices.step), siblings=False), ng)
        return e2, d2

    def side_slot_bytes(self, max_len: int, paged: bool) -> tuple:
        """(bytes a token, constant bytes) a side slot takes before it serves anything: its KV when the
        window is not paged; its conv states and the graphs' pools (SIDE_SLOT_CONST, an estimate)."""
        cost = self.cost()
        return (0 if paged else cost.view_bytes_per_token), cost.section_bytes + SIDE_SLOT_CONST

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        if self._plain_pending:
            # a joint plain step (`forward_tree_multi`): the state advanced in the pass itself
            if [int(p) for p in path] != [0]:
                raise ValueError(f"a plain step keeps its one row, not {path}")
            self._plain_pending = False
            self.st.length += 1
            self.commit_ms = 0.0
            return
        t0 = time.perf_counter()
        n_graphs = len(self.cgraphs)
        self.fn.commit(self.st, [int(p) for p in path], self.cgraphs)
        self.commit_ms = (time.perf_counter() - t0) * 1e3
        if len(self.cgraphs) > n_graphs:
            self.captured = True

    @torch.no_grad()
    def forward_block(self, block: torch.Tensor, start: int) -> torch.Tensor:
        n = block.numel()
        lg = self.forward_tree(block, list(range(-1, n - 1)), start)
        self._chain = n
        return lg

    @torch.no_grad()
    def rollback_to(self, keep: int) -> None:
        self._chain = None
        self.commit_tree(list(range(int(keep))))

    # --- internals -------------------------------------------------------------------------------
    def _settle_chain(self) -> None:
        if self._plain_pending:
            self.commit_tree([0])
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def _ensure(self, rows: int) -> None:
        """ENG-241: commit the KV pool's pages that hold rows [0, rows) before a write reaches them;
        never inside a graph capture (its caller committed the rows first)."""
        ps = self.st.pages
        rows = min(rows, self.max_len)
        if ps is None or rows + ps.margin() <= ps.rows:
            return
        if ps.on_device() and torch.cuda.is_current_stream_capturing():
            return
        ps.ensure(rows)

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"LFM2 forward at {start}, the state holds {self.st.length} rows")


#: a side slot's graph pools and buffers beyond its state, bytes (an estimate until measured, SPD-107)
SIDE_SLOT_CONST = int(0.5 * (1 << 30))


def profile_of(path: str) -> str:
    """The weight profile a checkpoint is served as: the running one (QSE_WEIGHTS), else lfm25-<dir>."""
    return os.environ.get("QSE_WEIGHTS") or "lfm25-" + os.path.basename(os.path.normpath(path))


def prices(path: str, log=print) -> PriceModel:
    """StairCut's prior for this profile on this device (ops/prices/lfm2.5-24b-a2b/<profile>.<device>.json)."""
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(path)}.{device_id()}.json"),
                           log=log)


def lookup_drafter(srv: "Lfm2MoeServing", path: str, corpus: str | None, log=print):
    """SPD-107: the lookup (the request's own text, `corpus` and the server's suffix store) cut by
    StairCut, through the shared tree drafter with no head (`NoHead`: this model has no draft layer
    yet; Liquid's drafter takes its place)."""
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead
    from engine.methods.staircut import StairCut
    from engine.tokfp import fingerprint
    ng = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(path), min_order=3, max_depth=16,
                      min_corpus_order=8)
    log(f"[server] LFM2 lookup drafter: corpus {(corpus or 'none') if ng.corpus is not None else 'none'}")
    return MTPTreeDrafter(srv, NoHead(), StairCut(prices(path, log), siblings=False), ng)


def build(path: str, max_len: int, tok, mark=lambda stage: None, log=print, kv_pool=None, check: bool = True):
    """LFM2 MoE for the server: the weights, the engine in the loop's interface, the load-time self-check
    (a verify block's rows against the decode step's). Returns (serving, info)."""
    from engine.models.lfm2_moe.load import load
    idx, W, experts, timing = load(path, "cuda", log=log)
    mark("weights")
    eng = Lfm2MoeEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
    if kv_pool is not None:
        kv_pool()
    srv = Lfm2MoeServing(eng, max_len)
    mark("engine")
    res = None
    if check:
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True)
        res = eng.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
    info = {"pack": path, "load": timing, "formats": idx.notes, "self_check": res, "exl3": idx.is_exl3,
            "quant": idx.quant}
    return srv, info


def serve(a, ctx) -> dict:
    """LiquidAI's LFM2 MoE from a BF16 checkpoint or an EXL3 pack for the server (`--model <dir>`, the
    registry's `serve`; server/app.py `_load`). Plain decode from the captured step; any `--drafter` but
    none serves the lookup cut by StairCut (`lookup_drafter`, gated by the load-time self-check). What
    the engine cannot hold yet is in `Lfm2MoeServing.caps`."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    if a.drafter not in ("none", "mtp"):
        ctx.log(f"[server] LFM2 MoE drafts with the lookup: --drafter {a.drafter} taken as the lookup")
    tok = AutoTokenizer.from_pretrained(a.model)
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    srv, info = build(a.model, a.max_len, tok, mark=ctx.mark, log=ctx.log, kv_pool=pool)
    drafter = None
    if a.drafter != "none":
        if info["self_check"] and info["self_check"]["ok"]:
            drafter = lookup_drafter(srv, a.model, a.corpus, ctx.log)
            # the loop hands a round k rows past the anchor: the lookup's copy runs up to what the cut admits
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
        else:
            ctx.log("[server] LFM2 self-check FAILED: speculation off, plain decode only")
    ctx.mark("drafters")
    fmt = "exl3" if info["exl3"] else "bf16"
    describe = (f"LFM2 MoE {a.model} ({fmt}): load {info['load'].get('total_s')} s, drafter "
                f"{'lookup + StairCut' if drafter is not None else 'none'}, max_len {a.max_len:,},")
    return {"engine": srv, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "tool_format": "pythonic",
                        "weights_info": {"format": fmt, "weights": a.model,
                                         "drafters": "lookup (no draft head)" if drafter is not None else None,
                                         "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"]}}}
