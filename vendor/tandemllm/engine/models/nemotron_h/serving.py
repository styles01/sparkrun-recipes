"""NemotronH behind the server's own generation loop: an engine with the shapes `server/app.py` drives
every model through (the LFM2 adapter's, engine/models/lfm2_moe/serving.py).

  forward, one token      the captured decode step (`DecodeGraph`)
  forward, more tokens    `NemotronHEngine.prefill` (chunks on the batched kernels)
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks
  commit_tree             `commit` (captured per shape)
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

so everything above it -- penalties, the think budget and its forced close, pattern stops, logprobs,
the reasoning split, the tool-call parser the model names (`tool_format`), the live view -- is the
server's code. A verified row is the decode step's row bit for bit and a commit leaves the decode
steps' SSM and conv states (engine.py), so greedy speculation writes the plain greedy text. The chat
template opens `<think>` itself and writes `<think></think>` for thinking off; tool calls are the XML
`<tool_call><function=NAME>` form the server's default parser reads. `caps` says what is on.
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
from engine.models.nemotron_h import kernels as K
from engine.models.nemotron_h.engine import CAPTURE_ROWS, PREFILL_CHUNK, DecodeGraph, NemotronHEngine
from engine.models.state import SectionState

#: fingerprint v3: the package's files that compute rows
MODULES = ("engine/models/nemotron_h/__init__.py", "engine/models/nemotron_h/config.py",
           "engine/models/nemotron_h/engine.py", "engine/models/nemotron_h/gguf.py",
           "engine/models/nemotron_h/gguf_load.py", "engine/models/nemotron_h/kernels.py",
           "engine/models/nemotron_h/load.py", "engine/models/nemotron_h/pack.py",
           "engine/models/nemotron_h/ref.py", "engine/models/nemotron_h/vision.py", "engine/models/lfm2_moe/kernels.py",
           "engine/models/lfm2_moe/pack.py", "engine/settings.py", "engine/vision.py")
#: the engine/ modules those files import (top level or inside a function) that compute no row, and why;
#: a change to one keeps the tier files. tests/test_tier_identity.py holds every such import to MODULES or
#: here (ENG-379)
LEFT_OUT = {
    "engine/__init__.py": "empty",
    "engine/capture_lock.py": "a lock held around graph capture (engine.py); no arithmetic",
    "engine/kvpool.py": "where K and V live, not what they hold: a paged buffer's tensor has the shape, dtype and "
                        "contiguous strides of the torch.zeros it replaces (new_state), and a page is zeroed when it is "
                        "mapped, so every kernel reads the same values at the same offsets",
    "engine/config.py": "resolve_snapshot only (config.py, lfm2_moe/config.py): the folder that holds config.json; the "
                        "weights in it count by content in the extra (server/diskcache.py fingerprint_extra)",
    "engine/models/lfm2_moe/__init__.py": "a docstring",
    "engine/models/lfm2_moe/config.py": "LFM2.5's config, read by lfm2_moe/pack.py's PackIndex; NemotronH takes only "
                                        "that file's constants, safetensors_header and TensorInfo (pack.py)",
}
#: the model's name in ops/prices/ and models/, by the served weight profile's name
MODEL_IDS = {"nemotron-bolt-": "nemotron-3.5-bolt-74b-a9b", "nemotron-lightning-": "nemotron-3.5-lightning-30b-a3b",
             "nemotron-super-vl-": "nemotron-3.5-super-vl-120b-a12b"}


def model_id(path: str) -> str:
    """The model of the served profile: by its name, or for a community pack (community/<id>, ENG-377) the base its
    profile names (QSE_NH_MODEL)."""
    p = profile_of(path)
    if p.startswith("community/") and os.environ.get("QSE_NH_MODEL"):
        return os.environ["QSE_NH_MODEL"]
    return next((m for k, m in MODEL_IDS.items() if p.startswith(k)), "nemotron-3.5-lightning-30b-a3b")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


#: the tables the arithmetic reads: none (the BF16 row kernels have fixed tiles)
TABLES = ()


def manifest() -> Manifest:
    return Manifest(arch="nemotron_h", modules=MODULES, data=TABLES)


def _tensors(o) -> list:
    """The tensors a weight entry holds: a tensor, a layer's stacked experts (bf16 or NVFP4), a ModelOpt
    projection's codes and scales."""
    if torch.is_tensor(o):
        return [o]
    if hasattr(o, "up") and hasattr(o, "down"):
        return [o.up, o.down]
    if hasattr(o, "uw"):
        return [o.uw, o.us, o.dw, o.ds]
    blk = getattr(o, "blk", None)
    if blk is not None and hasattr(blk, "w"):
        return [blk.w, blk.s]
    if hasattr(o, "w") and torch.is_tensor(o.w):
        return [o.w]
    return []


class _KV:
    """The `kv` the server reads: the sequence's length (a pending chain commit lands first) and its
    capacity. Zero starts a new sequence in place (`restart`); any other length keeps the KV rows by
    length -- the conv states are the caller's to restore first (`state.restore`)."""

    def __init__(self, serving: "NemotronHServing"):
        self._s = serving

    @property
    def pages(self):
        return self._s.st.pages

    @property
    def length(self) -> int:
        # what the state holds once its pending work commits (a chain block's rows, a joint plain
        # step's row), read without committing it: another thread reads it too (the health report,
        # the slot placement); a commit from there while the slot's own loop sits between a joint
        # pass and its commit_tree would count the row twice
        s = self._s
        return s.st.length + (s._chain or 0) + (1 if s._plain_pending else 0)

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"the NemotronH state cannot hold {v} rows (capacity {self.max_len})")
        st = self._s.st
        st.length = v
        # a restore (the caches, a probe's way back): the taps below v may be another sequence's
        st.tap_from, st.tap_gen = v, st.tap_gen + 1

    @property
    def max_len(self) -> int:
        return self._s.max_len


class NemotronHCacheState(SectionState):
    """What the caches keep of a sequence (engine/models/state.py): sections the conv state
    ("mamba.conv.<layer>", [conv_kernel - 1, conv_dim] bf16) and the SSM state ("mamba.S.<layer>",
    [heads, head_dim, state] fp32) of every Mamba layer; views K and V of every attention layer
    ([kv heads, rows, D], axis 1)."""

    def __init__(self, serving: "NemotronHServing"):
        self.s = serving

    def sections(self) -> dict:
        st = self.s.st
        out = {f"mamba.conv.{li}": st.conv[li] for li in sorted(st.conv)}
        out.update({f"mamba.S.{li}": st.ssm[li] for li in sorted(st.ssm)})
        return out

    def views(self) -> list:
        st = self.s.st
        out = []
        for li in sorted(st.k):
            out += [(f"k.{li}", st.k[li], 1), (f"v.{li}", st.v[li], 1)]
        return out


# the conductor's router on this model (server/conductor/modelfit.py): LFM2.5's measured fit as the start,
# until the class prompt, note placement and think budget are measured on NemotronH
CONDUCTOR_FIT = {"served_classifier": False, "fallback": "lean", "packages": "core",
                 "no_package_classes": ("chat",), "effort_notes": False, "memory": False, "rules": False}


class NemotronHServing:
    """`NemotronHEngine` with one state and its graphs, in the server's engine interface."""

    # the caches on the prefill's 4,096-token grid; side slots with joint passes, a plain step joining
    # the round (`joint_plain`); the shadow reader was validated on Qwen text; the side lane copies the
    # 27B's pending commit
    caps = Capabilities(caches=True, side_slots=True, anchor_grid=PREFILL_CHUNK, shadow=False, joint_verify=True,
                        side_lane=False, joint_plain=True, conductor=CONDUCTOR_FIT)

    def __init__(self, eng: NemotronHEngine, max_len: int, slot=None):
        self.fn, self.cfg, self.max_len = eng, eng.cfg, max_len
        # slot 0 (None) keys its row buffers by row count; a side slot by (its tag, rows), so no slot
        # stages into a buffer another slot's graph reads
        self.slot = slot
        self.st = eng.new_state(max_len, paged=True)
        self.state = NemotronHCacheState(self)
        self.kv = _KV(self)
        self.dg = None
        self.vgraphs: dict = {}
        self.cgraphs: dict = {}
        self.picks = None
        self.tap = None
        self.mm = None                       # the request's images (vision.py NhMMContext), set by the server
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
        """A snapshot: the SSM and conv states whole; K and V of the attention layers a token (bf16).
        The prefill workspace (ENG-363), measured on Super VL EXL3 4.0 (2026-10-16, the memory policy's
        `observed`): 3.34 GiB above the engine at a 131,054-token prefill and 3.18 GiB at 249,983. The
        prefill runs in 4,096-row chunks and attention reads the cache in place, so it does not grow with
        the prompt: a floor of 3.5 GiB, the line through the 131k point, capped at 4.0 GiB (the policy
        raises the cap if a prefill ever needs more). Without it the 27B's line (30 kB a token, no cap)
        refused a 1,000,000-token prefill as 29 GiB of workspace."""
        conv = sum(t.numel() * t.element_size() for t in self.st.conv.values())
        rec = sum(t.numel() * t.element_size() for t in self.st.ssm.values())
        return CostModel(section_bytes=conv + rec, view_bytes_per_token=self.cfg.kv_bytes_per_token(2),
                         workspace_floor=int(3.5 * (1 << 30)), workspace_per_token=27370,
                         workspace_cap=int(4.0 * (1 << 30)), parts={"recurrent": rec, "conv": conv})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        """The decode graph of the first class, after the server's warm-up (whose rows are nobody's)."""
        if self.dg is None:
            self.dg = DecodeGraph(self.fn, self.st, self.slot)
        self.kv.length = 0
        return len(self.dg.graphs)

    def draft_heads(self) -> dict:
        """The draft heads this model carries, by name: the checkpoint's MTP head ("mtp",
        engine/models/nemotron_h/mtp.py) when it was loaded, LiLiCorr ("lilicorr",
        engine/drafters/lilicorr.py) when its checkpoint was, else none."""
        return self.heads

    def add_mtp_head(self, depth: int):
        from engine.models.nemotron_h.mtp import NemotronHMTPHead
        self.heads["mtp"] = NemotronHMTPHead(self.fn, depth, self.max_len)
        return self.heads["mtp"]

    def add_lilicorr(self, model=None, path: str | None = None, log=print):
        """LiLiCorr over this state's taps: a new head over `model` (another slot's weights, shared) or
        over the checkpoint at `path`."""
        from engine.drafters.lilicorr import LiLiCorrHead, build_head
        self.heads["lilicorr"] = LiLiCorrHead(self, model) if model is not None else build_head(self, path, log=log)
        return self.heads["lilicorr"]

    def settle(self) -> None:
        self._settle_chain()

    def cache_state(self) -> NemotronHCacheState:
        return self.state

    def resume_at(self, n: int) -> None:
        self.discard_pending()
        self.kv.length = n

    def memory_parts(self) -> dict:
        """The dashboard's memory split (server/memsplit.py): weights (each storage once), the KV rows,
        the Mamba states."""
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
                "gdn_state": sum(int(t.numel() * t.element_size()) for d in (st.conv, st.ssm) for t in d.values())}

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
        """A new sequence in the same buffers: the conv and SSM states zeroed, the length 0 (KV rows stay)."""
        self._chain = None
        self._plain_pending = False
        self.picks = None
        self.st.reset_inplace()

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the decode graph, more (or an image row)
        through the prefill, with the request's image rows (`mm`). Logits [1, 1, vocab] of the last row
        (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        self._ensure(start + T + CAPTURE_ROWS)
        if T == 1 and start > 0 and not (self.mm is not None and self.mm.at(start)):
            if self.dg is None:
                self.dg = DecodeGraph(self.fn, self.st, self.slot)
            n0 = self.dg.captures
            lg = self.dg(int(ids[0]))
            self.plain_captured = self.dg.captures != n0
            self._hidden = self.dg.rb["hidden"].view(1, 1, -1)
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        # rows past the prompt hold no image: a forced close or a hidden note forwarded after the
        # prompt (2026-10-17: the conductor closed a looping reasoning on a request with a
        # screenshot, and the splice refused rows past the prompt; the request failed)
        mm = self.mm if self.mm is not None and start < self.mm.n else None
        lg = self.fn.prefill(ids, self.st, mm=mm)
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
        pass (`NemotronHEngine.verify_multi`: the projections, experts and head read once); each slot's
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
        e2 = NemotronHServing(self.fn, max_len, slot=f"slot{id(self) % 9973}.{len(self.sides)}")
        self.sides.append(e2)
        d2 = None
        if drafter0 is not None:
            from engine.drafters.ngram import NgramDrafter
            from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead
            from engine.methods.staircut import PriceModel, StairCut
            ng = None
            if drafter0.ngram is not None:
                ng = NgramDrafter(corpus_path="", min_order=drafter0.ngram.min_order, max_depth=16, min_corpus_order=8)
                ng.corpus = drafter0.ngram.corpus
            c0 = drafter0.cut
            if "lilicorr" in self.heads and drafter0.head is self.heads["lilicorr"]:
                head = e2.add_lilicorr(model=drafter0.head.m)
            else:
                head = e2.add_mtp_head(drafter0.head.depth) if drafter0.head.depth else NoHead()
            d2 = MTPTreeDrafter(e2, head, StairCut(PriceModel(c0.prices.table, c0.prices.step), siblings=False), ng)
        return e2, d2

    def side_slot_bytes(self, max_len: int, paged: bool) -> tuple:
        """(bytes a token, constant bytes) a side slot takes before it serves anything: its KV when the
        window is not paged; its Mamba states and the graphs' pools (SIDE_SLOT_CONST, an estimate)."""
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
            raise ValueError(f"NemotronH forward at {start}, the state holds {self.st.length} rows")


#: a side slot's graph pools and buffers beyond its state, bytes (an estimate until measured)
SIDE_SLOT_CONST = int(0.5 * (1 << 30))


def profile_of(path: str) -> str:
    """The weight profile a checkpoint is served as: the running one (QSE_WEIGHTS), else nemotron-<dir>."""
    return os.environ.get("QSE_WEIGHTS") or "nemotron-" + os.path.basename(os.path.normpath(path))


def prices(path: str, log=print) -> PriceModel:
    """StairCut's prior for this profile on this device (ops/prices/<model>/<profile>.<device>.json)."""
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", model_id(path), f"{profile_of(path)}.{device_id()}.json"),
                           log=log)


#: the lookup's shortest match beside a block drafter (LiLiCorr drafts 7 tokens a round): Bolt IQ2, 512 greedy tokens
#: a class, LiLiCorr + lookup at min order 3 / 6 / 8 / 12 (tok/s): code 71.4 / 73.5 / 74.1 / 73.7, chat 46.3 / 48.0 /
#: 50.2 / 49.1, edit 115.8 / 101.8 / 91.2 / 79.1, copy 147.4 / 147.9 / 146.5 / 105.6 (LiLiCorr alone: 74.0, 49.7,
#: 79.0, 106.6; the lookup alone: edit 103.4, copy 121.6): short matches pay on a rename and cost chat their rows
LOOK_MIN_ORDER_HEAD = 6


def lookup_drafter(srv: "NemotronHServing", path: str, corpus: str | None, log=print, depth: int = 0,
                   head=None):
    """The lookup (the request's own text, `corpus` and the server's suffix store) cut by StairCut,
    through the shared tree drafter: with `head` (LiLiCorr, its block a round), the checkpoint's MTP head
    drafting `depth` tokens a round, or no head (`NoHead`, depth 0)."""
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead
    from engine.methods.staircut import StairCut
    from engine.tokfp import fingerprint
    ng = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(path),
                      min_order=3 if head is None else LOOK_MIN_ORDER_HEAD, max_depth=16, min_corpus_order=8)
    if head is not None and os.environ.get("QSE_NH_LOOKUP", "on") == "off":
        ng = None                            # the head alone (a measurement's row: LiLiCorr without the lookup)
    what = ("LiLiCorr, block of " + str(head.depth) if head is not None else
            "MTP head depth " + str(depth) if depth else "no head")
    log(f"[server] NemotronH lookup drafter: corpus {(corpus or 'none') if ng is not None and ng.corpus is not None else 'none'}"
        f"{'' if ng is not None else ' (the lookup off)'}, {what}")
    if head is None:
        head = srv.add_mtp_head(depth) if depth else NoHead()
    return MTPTreeDrafter(srv, head, StairCut(prices(path, log), siblings=False), ng)


#: the load-time gate's floor: the mean run of drafted tokens the trunk's greedy keeps (a draft wired
#: wrong keeps about none)
LILICORR_GATE = 1.0


def build(path: str, max_len: int, tok, mark=lambda stage: None, log=print, kv_pool=None, check: bool = True,
          mtp: bool = False, lilicorr: str | None = None):
    """NemotronH for the server: the weights, the engine in the loop's interface, the load-time
    self-check (a verify block's rows and the committed states against the decode steps'). With
    `lilicorr` (a LiLiCorr checkpoint directory) the engine keeps its taps and the draft head is built
    and gated (`info["lilicorr"]`: the gate's result, or why the head is off). Returns (serving, info)."""
    from engine.drafters import lilicorr as LC
    from engine.models.nemotron_h.load import load_any
    idx, W, experts, timing = load_any(path, "cuda", mtp=mtp, log=log)
    mark("weights")
    eng = NemotronHEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
    lc = None
    if lilicorr is not None:
        missing = LC.missing_files(lilicorr)
        if missing:
            lc = {"ok": False, "why": f"LiLiCorr files missing: {', '.join(missing)}"}
        else:
            cfg = LC.load_config(lilicorr)
            eng.set_taps(cfg.target_layer_ids, cfg.window)
    if kv_pool is not None:
        kv_pool()
    srv = NemotronHServing(eng, max_len)
    mark("engine")
    res = None
    text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                   tokenize=False, add_generation_prompt=True)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    if check:
        res = eng.self_check(prompt, log=log)
    if lilicorr is not None and lc is None:
        try:
            head = srv.add_lilicorr(path=lilicorr, log=log)
            g = LC.gate(eng, head, eng.new_state(min(max_len, len(prompt) + 128)), prompt)
            ok = g["mean_run"] >= LILICORR_GATE
            lc = {"ok": ok, "gate": g, **({} if ok else {"why": f"LiLiCorr gate: mean run {g['mean_run']} "
                                                                  f"under {LILICORR_GATE}"})}
            log(f"[lilicorr] gate {'PASS' if ok else 'FAIL'}: mean run {g['mean_run']} of {head.depth} "
                f"over {g['positions']} positions")
        except Exception as e:                       # noqa: BLE001  (the drafter stays off, the trunk serves)
            lc = {"ok": False, "why": f"LiLiCorr failed to load: {type(e).__name__}: {e}"}
        if not lc["ok"]:
            srv.heads.pop("lilicorr", None)
        mark("lilicorr")
    if lc is not None and not lc["ok"]:
        log(f"[server] {lc['why']}: the lookup drafts alone")
    info = {"pack": path, "load": timing, "formats": idx.notes, "self_check": res, "mtp_in_checkpoint": idx.has_mtp,
            "exl3": bool(getattr(idx, "is_exl3", False)), "lilicorr": lc}
    return srv, info


def serve(a, ctx) -> dict:
    """NVIDIA's NemotronH from a BF16 checkpoint, an EXL3 pack or a GGUF for the server (`--model <dir>`, the registry's `serve`;
    server/app.py `_load`). Plain decode from the captured step; any `--drafter` but none serves the
    lookup cut by StairCut (`lookup_drafter`, gated by the load-time self-check), and `--draft-head mtp`
    adds the checkpoint's MTP head at `--depth` (its weights loaded only then). What the engine cannot
    hold yet is in `NemotronHServing.caps`."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    if a.drafter not in ("none", "mtp"):
        ctx.log(f"[server] NemotronH drafts with the lookup: --drafter {a.drafter} taken as the lookup")
    tok = AutoTokenizer.from_pretrained(a.model)
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    name = str(getattr(a, "draft_head", None) or "") if a.drafter != "none" else ""
    head = name == "mtp"
    # LiLiCorr's checkpoint: QSE_NH_LILICORR (the profile's), else the directory beside the model's
    lc_path = (os.environ.get("QSE_NH_LILICORR") or os.path.join(os.path.dirname(os.path.normpath(a.model)),
                                                                   "nemotron-3.5-bolt-lilicorr")) if name == "lilicorr" else None
    srv, info = build(a.model, a.max_len, tok, mark=ctx.mark, log=ctx.log, kv_pool=pool, mtp=head, lilicorr=lc_path)
    if hasattr(ctx, "vision"):
        # ENG-365: an Omni checkpoint's RADIO tower and projector (BF16 in every Super VL checkpoint), with
        # its processor and prompt rows (vision.py); `--vision off` or a checkpoint without one: text only
        from engine.models.nemotron_h.vision import load_radio_config, load_tower
        ctx.vision(a.model, lambda vcfg: load_tower(a.model), load_radio_config)
        ctx.mark("vision")
    drafter = None
    depth = max(1, int(getattr(a, "depth", 3) or 3)) if head and info["mtp_in_checkpoint"] else 0
    lc_head = srv.heads.get("lilicorr")
    if a.drafter != "none":
        if info["self_check"] and info["self_check"]["ok"]:
            drafter = lookup_drafter(srv, a.model, a.corpus, ctx.log, depth=depth, head=lc_head)
            # the loop hands a round k rows past the anchor: the lookup's copy runs up to what the cut admits
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
        else:
            ctx.log("[server] NemotronH self-check FAILED: speculation off, plain decode only")
    ctx.mark("drafters")
    from engine.models.nemotron_h.load import gguf_of
    fmt = "gguf" if gguf_of(a.model) else ("exl3" if info["exl3"] else "bf16")
    what = ("LiLiCorr + lookup + StairCut" if lc_head is not None else
            "MTP head depth " + str(depth) + " + lookup + StairCut" if depth else "lookup + StairCut")
    describe = (f"NemotronH {a.model} ({fmt}): load {info['load'].get('total_s')} s, drafter "
                f"{what if drafter is not None else 'none'}, max_len {a.max_len:,},")
    return {"engine": srv, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "tool_format": "xml",
                        "weights_info": {"format": fmt, "weights": a.model,
                                         "drafters": (what if drafter is not None else None),
                                         "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"],
                                         "lilicorr": _lc_info(info["lilicorr"])}}}


def _lc_info(lc: dict | None) -> dict | None:
    """What /health says of LiLiCorr: on or off, why it is off, the gate's mean run."""
    if lc is None:
        return None
    return {"ok": lc["ok"], **({"why": lc["why"]} if "why" in lc else {}),
            **({"gate_mean_run": lc["gate"]["mean_run"]} if "gate" in lc else {})}
