"""Flash-Next behind the server's own generation loop (SRV-87): an engine and a drafter with the shapes
`server/app.py` and `engine/cache.py` drive the 27B through.

The server's loop (`_generate_stream`) speaks a small interface: `forward` (a prompt chunk, a forced
phrase or one decode token at the state's length), `forward_tree` (a draft block; logits of every
row), `commit_tree` (keep a path), `hidden_post_norm` (the rows a drafter syncs from), `picks` (the
verify's own greedy tokens), `kv.length`, `reset`; the walk down a block is the shared one
(engine/methods/walk.py). The caches and the dashboard add the adapter's cold verbs (ENG-E8): `cost`,
`cache_state` (`FlashNextCacheState`: what the caches keep, ENG-157), `resume_at`, `settle`,
`discard_pending`, `trim_pending`. `FlashNextServing` maps it onto `FlashNextEngine`:

  forward, one token      the captured decode step (`DecodeGraph`), streams of that row
  forward, more tokens    the prefill kernels (`FlashNextEngine.prefill`), every row's streams
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks, streams
  commit_tree             `commit` from captured graphs (the folded GDN commit included)
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

so everything above it -- penalties, the think budget and its forced close, pattern stops, logprobs,
sampled requests walked down the greedy tree (`--sampled-tree det`), tool calls, the live view -- is
the 27B's code. A verified row is the decode step's row bit for bit (ENG-227), so greedy speculation
writes the plain greedy text and a sampled walk draws exactly what plain sampled decoding draws.

The drafter is the shared MTP tree drafter (engine/methods/mtp_drafter.py) over Flash-Next's MTP head
(engine/models/flashnext/mtp.py, `draft_heads()["mtp"]`), cut by the shared StairCut whose prior is the
price file of this pack on this device (ops/prices/qwen3.8-flash-next/<profile>.<device>.json).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.methods.staircut import PriceModel, StairCut
from engine.models.adapter import Capabilities, CostModel
from engine.models.flashnext.engine import CAPTURE_ROWS, CommitGraph, DecodeGraph, FlashNextEngine, VerifyGraph
from engine.models.flashnext.mtp import FlashNextMTPHead
from engine.models.state import SectionState
from engine.settings import SETTINGS as _S


#: fingerprint v3 (`FlashNextServing.manifest`): the package's files that compute rows, the tables
MODULES = ("engine/models/flashnext/__init__.py", "engine/models/flashnext/config.py",
           "engine/models/flashnext/engine.py", "engine/models/flashnext/load.py",
           "engine/models/flashnext/mtp.py", "engine/models/flashnext/pack.py",
           "engine/models/flashnext/ple.py", "engine/models/flashnext/ref.py",
           "engine/models/flashnext/vision.py", "engine/settings.py")
TABLES = ("ops/fn-exl3-tiles.json", "ops/fn-dense-splits.json")


def manifest() -> Manifest:
    """What Flash-Next's stored rows depend on (fingerprint v3, ENG-243): its package's arithmetic
    (not this adapter), the knob registry, its own tile tables; the `tools/` kernels follow."""
    return Manifest(arch="qwen4_exp", modules=MODULES, data=TABLES)


class _KV:
    """The `kv` the server reads: the length of the sequence the state holds (a pending chain commit
    lands first), its capacity. Setting it to zero starts a new sequence in place (`restart`: the rows
    stay, as the 27B's do); to any other length it keeps the KV rows by length and clamps the pooled
    keys -- the recurrent state is the caller's to restore first (`state.restore`), as on the 27B."""

    def __init__(self, serving: "FlashNextServing"):
        self._s = serving

    @property
    def pages(self):
        """The KV pool's paged set of the state's rows (the trunk's and the MTP head's), or None."""
        return self._s.st.pages

    @property
    def length(self) -> int:
        self._s._settle_chain()
        return self._s.st.length

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"Flash-Next's state cannot hold {v} rows (capacity {self.max_len})")
        self._s.st.length = v
        self._s.st.clamp_pool()

    @property
    def max_len(self) -> int:
        return self._s.max_len


def _tensors(obj) -> list:
    """The tensors an object of the weight dict holds: itself, or its attributes (an EXL3 block, an
    int8 row set, an expert bank's kept images)."""
    if torch.is_tensor(obj):
        return [obj]
    out = []
    for v in vars(obj).values() if hasattr(obj, "__dict__") else ():
        if torch.is_tensor(v):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [x for x in v if torch.is_tensor(x)]
    return out


class FlashNextCacheState(SectionState):
    """What the caches keep of a Flash-Next sequence (engine/models/state.py, ENG-157). Sections: the
    GDN states and conv tails, the PLE conv state and its token context (a fixed buffer, the end-of-
    text id where the sequence has none yet) and `mtp.h_last`, the last committed row's streams (the
    MTP layer's next input, which waits for its next token). Views: K, V and the indexer keys of every
    attention layer. The pooled indexer keys are derived and built again after a restore
    (`FlashNextServing.resume_at`)."""

    def __init__(self, serving: "FlashNextServing"):
        self.s = serving

    def sections(self) -> dict:
        s, st = self.s, self.s.st
        for li, buf in s._ctx.items():
            ctx = st.ple_ctx.get(li)
            if ctx is None:
                buf.fill_(s.cfg.eos_token_id)
            else:
                buf.copy_(ctx.view(1, -1))
        return self.buffers()

    def buffers(self) -> dict:
        """The sections' buffers as they are, without bringing the PLE context up to date: for what
        only counts their bytes (`cost`, `memory_parts`), which other threads call -- /health and the
        dashboard -- and which must issue no CUDA work: inside a graph capture (CUDA's global mode) a
        copy from another thread invalidates the capture and fails the request (2026-10-12, the MTP
        chain's capture at the 262,144-token class while /health was polled)."""
        s, st = self.s, self.s.st
        out = {}
        for li in sorted(st.S):
            out[f"gdn.S.{li}"] = st.S[li]
        for li in sorted(st.conv):
            out[f"gdn.conv.{li}"] = st.conv[li]
        for li in sorted(st.ple_conv):
            out[f"ple.conv.{li}"] = st.ple_conv[li]
        for li in sorted(s._ctx):
            out[f"ple.ctx.{li}"] = s._ctx[li]
        out["mtp.h_last"] = s._h_last
        return out

    def views(self) -> list:
        st = self.s.st
        out = []
        for li in sorted(st.k):
            out += [(f"k.{li}", st.k[li], 1), (f"v.{li}", st.v[li], 1)]
            if li in st.ik:
                out.append((f"ik.{li}", st.ik[li], 0))
        return out


class FlashNextServing:
    """`FlashNextEngine` with one state and its graphs, in the server's engine interface."""

    # the caches' grid: the prefill kernels' chunk (FN_PREFILL_CHUNK), so a resumed prefill runs the
    # chunks a cold one runs; side slots: ENG-241; the shadow reader: validated on the 27B's text only
    # side_lane: Borrow keeps the 27B's pending commit, not this engine's chain and folded commit
    # (lending it raised `'FlashNextCacheState' object has no attribute 'primed'`, rc-0.8.0 live)
    caps = Capabilities(caches=True, side_slots=True, anchor_grid=4096, shadow=False, side_lane=False)

    def __init__(self, fn: FlashNextEngine, max_len: int):
        self.fn, self.cfg, self.max_len = fn, fn.cfg, max_len
        self.st = fn.new_state(max_len, paged=True)
        c, dev = fn.cfg, fn.device
        self._ctx = {li: torch.full((1, fn.hash[li].context_len), c.eos_token_id, dtype=torch.long, device=dev)
                     for li in c.ple_layers}
        self._h_last = torch.zeros(1, c.hc_count * c.hidden_size, dtype=fn.dtype, device=dev)
        self.state = FlashNextCacheState(self)
        self.kv = _KV(self)
        self.dg = None                      # the decode step's graphs, made on the first decode token
        self.vgraphs: dict = {}
        self.cgraphs: dict = {}
        # verify width M / commit key (M, n, moved) -> rounds that used it, from a prior of one round
        # each for chains of 2-6 rows (a class's idle captures need shapes before the first rounds)
        self.uses: dict = dict(PRIOR_USES)
        self._fresh: set = set()            # graphs captured while idle and not replayed yet
        self.capture_ms = None              # one verify or commit graph's capture, ms (EMA 0.2)
        self._width = 0                     # the last verify block's rows (its commit's key)
        self.picks = None
        self.tap = None
        self._hidden = None                 # [1, rows, hc H]: the last forward's streams
        self._chain = None                  # a forward_block's rows awaiting their commit
        self.commit_ms = 0.0
        self.captured = False               # the last block captured a verify or commit graph
        self.plain_captured = False         # the last plain step captured its decode graph
        self.heads: dict = {}               # draft heads by name (`draft_heads`)
        self._weights = None                # the weights' bytes by part (`memory_parts`), once
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    # --- the loop's interface -------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    @property
    def mm(self):
        """VIS-97: the request's images (engine/vision.py `MMContext`), set by the server as on the 27B;
        kept by the state and the draft heads' states (`FNState.mm`), where the engine reads them."""
        return self.st.mm

    @mm.setter
    def mm(self, mm) -> None:
        for st in self._states():
            st.mm = mm

    @property
    def pos_delta(self) -> int:
        """The rotary offset of the rows after the prompt (position = slot + delta, VIS-97)."""
        return self.st.delta

    @pos_delta.setter
    def pos_delta(self, d: int) -> None:
        for st in self._states():
            st.delta = int(d)

    def price_class(self, n: int) -> int:
        """The draft prices' context class of a round at context length `n` (engine/methods/staircut.py):
        0 while the sparse attention is dense, else the next of 4,096 x 2^k and 6,144 x 2^k at or above
        n, up to 262,144. A verify's score rows grow with the context itself, not only with the sparse
        attention's class: on the P0 runs a block cost about 41 ms at 140,000 tokens and 53 at 260,800,
        both in the 262,144 class."""
        if not self.fn.qsa_class(n):
            return 0
        c = 4096
        while c < n and c < 262_144:
            c = c * 3 // 2 if (c & (c - 1)) == 0 else c * 4 // 3
        return min(c, 262_144)

    def cost(self) -> CostModel:
        """What a state snapshot holds: its sections whole (the recurrent, conv and PLE ones) and the
        cache rows a token adds (K, V and the indexer key of every attention layer, bf16)."""
        sec = {k: t.numel() * t.element_size() for k, t in self.state.buffers().items()}
        c = self.cfg
        per_tok = len(c.attention_layers) * (c.num_key_value_heads * c.head_dim * 2
                                             + (c.indexer_head_dim if c.has_indexer else 0)) * 2
        # the prefill workspace, measured 2026-10-12 on 3.05: 2.51 GiB at 8k and 2.40 GiB at 32k
        # (tools/fn_memory_check.py; the 4,096-row chunk's activations), 10.46 GiB above the engine at a
        # 147,149-token prefill and 10.47 GiB at 261,549 (SPD-99): it grows to about 140k tokens and then
        # stops -- a floor of 2.75 GiB, the line through the 147,149-token point (76,309 bytes a token),
        # capped at 10.75 GiB (the memory policy raises the cap if a prefill ever needs more)
        return CostModel(section_bytes=sum(sec.values()), view_bytes_per_token=per_tok,
                         workspace_floor=int(2.75 * (1 << 30)), workspace_per_token=76309,
                         workspace_cap=int(10.75 * (1 << 30)),
                         parts={"recurrent": sum(b for k, b in sec.items() if k.startswith("gdn.S")),
                                "conv": sum(b for k, b in sec.items() if k.startswith(("gdn.conv", "ple.conv")))})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        """After the server's warm-up: the warm-up's rows are nobody's conversation, so the state starts
        over and class 0's graphs (the shapes the rounds use, from the prior and the warm-up) are
        captured now, not in the first request (2026-10-12: the first short request after a restart
        drafted at 38.5 tok/s while it captured them, 58 after). Returns the graphs captured."""
        if drafter is None:
            return None
        self.kv.length = 0
        steps = self.idle_captures()
        return sum(1 for _ in steps) if steps is not None else 0

    def draft_heads(self) -> dict:
        """The draft heads this pack carries, by name: its MTP layer ("mtp", made by `add_mtp_head`)."""
        return self.heads

    def side_slot(self, max_len: int, drafter0=None) -> tuple:
        """ENG-241 step 2: a side slot's engine and drafter over this engine's weights (`fn`, the one
        copy): its own state (paged as slot 0's when the pool is on), its own graphs and MTP head; the
        drafter's StairCut from the same price file, its lookup over slot 0's corpus and suffix store
        (read-only). The slots share the engine's scratch buffers: a slot's work runs on its GPU turn,
        a call on one state applies another's parked GDN rows first (`FlashNextEngine.settle`), each
        MTP head drafts into its own buffers, and decoding slots verify their blocks in one pass
        (`forward_tree_multi`)."""
        e2 = FlashNextServing(self.fn, max_len)
        d2 = None
        if drafter0 is not None:
            ng = None
            if drafter0.ngram is not None:
                from engine.drafters.ngram import NgramDrafter
                ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
                ng.corpus = drafter0.ngram.corpus
            c0 = drafter0.cut
            d2 = MTPTreeDrafter(e2, e2.add_mtp_head(drafter0.depth),
                                StairCut(PriceModel(c0.prices.table, c0.prices.step), siblings=c0.siblings), ng)
            d2.capture_all()
        return e2, d2

    def side_slot_bytes(self, max_len: int, paged: bool) -> tuple:
        """(bytes a token, constant bytes) a side slot takes before it serves anything: its KV and the
        MTP head's when the window is not paged; the recurrent sections, and the graphs' pools a slot
        captures (SIDE_SLOT_CONST, measured)."""
        cost = self.cost()
        per_token = 0 if paged else cost.view_bytes_per_token * 13 // 12      # the MTP layer's 13th
        return per_token, cost.section_bytes + SIDE_SLOT_CONST

    def add_mtp_head(self, depth: int) -> FlashNextMTPHead:
        self.heads["mtp"] = FlashNextMTPHead(self.fn, depth, self.max_len, pages=self.st.pages)
        # the head's graphs are captured now (`capture_all`): with the KV pool its first rows must be
        # mapped -- a Triton launch refuses a buffer whose first page is not ("cannot be accessed from
        # Triton", 2026-10-12)
        self._ensure(CAPTURE_ROWS)
        return self.heads["mtp"]

    def settle(self) -> None:
        """Land what is pending, so the state's sections are the sequence's whole (before a
        snapshot): a chain block's commit, and the GDN rows the folded commit parked for the next call."""
        self._settle_chain()
        if self.fn._pend_owner is self.st:
            self.fn.flush()

    def cache_state(self) -> FlashNextCacheState:
        return self.state

    def resume_at(self, n: int) -> None:
        """The engine continues at length `n` from the state the caller just wrote into it: the PLE
        context from its section, the length, the pooled indexer keys built from the keys in place."""
        self.discard_pending()
        for li, buf in self._ctx.items():
            self.st.ple_ctx[li] = buf.view(1, -1).cpu().clone()
        self.kv.length = n
        if n > 0:
            self.fn.build_pool(self.st)

    def memory_parts(self) -> dict:
        """The dashboard's memory split (server/memsplit.py): the trunk's and the MTP layer's weights
        (each storage once), the trunk's KV, indexer keys and pooled keys, the drafter's MTP KV, the
        recurrent sections."""
        if self._weights is None:
            seen, by = set(), {"weights": 0, "mtp_weights": 0}
            items = [(k, v) for k, v in self.fn.W.items()] + [(f"bank.{li}", b) for li, b in self.fn.banks.items()]
            for name, obj in items:
                part = "mtp_weights" if name.startswith("mtp.") or name == "bank.-1" else "weights"
                for t in _tensors(obj):
                    s = t.untyped_storage()
                    if s.data_ptr() not in seen:
                        seen.add(s.data_ptr())
                        by[part] += s.nbytes()
            self._weights = by
        st = self.st

        from engine import kvpool as _kvp

        def nb(*ds):
            # a paged buffer's `nbytes` is its reserved address range: its committed pages are
            # the pool's figure (server/memsplit.py adds them as `kv_pool`)
            return sum(int(t.numel() * t.element_size()) for d in ds for t in d.values()
                       if not _kvp.paged(t))
        head = self.heads.get("mtp")
        mtp_kv = 0
        if head is not None:
            mtp_kv = nb(head.st.k, head.st.v, head.st.ik, head.st.pool)
        return {**self._weights, "kv_pool": nb(st.k, st.v, st.ik, st.pool), "drafter_kv": mtp_kv,
                "gdn_state": sum(int(t.numel() * t.element_size()) for t in self.state.buffers().values())}

    def resident_files(self) -> list:
        """The n-gram table: its policy, its file and page-cache bytes, the cold cache's counts."""
        return [t.report() for t in {id(t): t for t in self.fn.tables.values()}.values()]

    def last_streams(self) -> torch.Tensor:
        """The last committed row's streams [1, hc H] (`mtp.h_last`)."""
        return self._h_last

    def discard_pending(self) -> None:
        """Forget what is pending after the state was overwritten (a restore): the chain block, the
        last verify's picks and the GDN rows parked for the state that was there."""
        self._chain = None
        self.picks = None
        if self.fn._pend_owner is self.st:
            self.fn._pend_n.zero_()
            self.fn._pend_owner = None

    def trim_pending(self, n: int) -> bool:
        """Bring the state back to length `n` inside work not yet committed: a chain block
        (`forward_block`) whose commit waits keeps only its rows below `n` (none: the verify moved
        nothing but its own KV rows). False, and nothing changed, when the state already absorbed the
        rows past `n` -- a verified path is committed whole, conv tails included (cutting inside it is
        ENG-235's)."""
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
        return False                        # no layer-major prefill window (engine/cache.py)

    def reset(self) -> None:
        """A new sequence (`restart`), as the 27B's reset: its KV rows stay."""
        self.restart()

    def restart(self) -> None:
        """A new sequence in the same buffers: the recurrent sections cleared and the length 0, the KV,
        indexer-key and pooled-key rows left in place (nothing reads a row past the length). The
        resident prefix keeps a conversation's rows there while a short request runs from row 0 (a
        guest, engine/cache.py, whose cold prefill calls `reset`); zeroing every row there lost the
        rows above the guest's stash (ENG-157, 2026-10-12: rows 2,144-4,095 of a 40,000-token
        conversation came back as zeros, and a 6,000-token prompt resumed at 4,096 over them answered
        differently from a cold prefill, 67 of 75 sections wrong)."""
        self._chain = None
        self.picks = None
        fn = self.fn
        if fn._pend_owner is self.st:
            fn._pend_n.zero_()
            fn._pend_owner = None
        self.st.reset_inplace()

    # --- a context class's graphs, captured while the server is idle -----------------------------
    #: the shapes a class captures ahead: the verify widths and commit shapes that make up this share
    #: of the rounds so far, at most this many of each
    IDLE_SHARE, IDLE_MAX = 0.99, 24

    def _capture_took(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1e3
        self.capture_ms = ms if self.capture_ms is None else 0.8 * self.capture_ms + 0.2 * ms

    def _top(self, commits: bool) -> list:
        keys = sorted(((k, n) for k, n in self.uses.items() if isinstance(k, tuple) == commits),
                      key=lambda kn: -kn[1])
        total, out, got = sum(n for _, n in keys), [], 0
        for k, n in keys:
            if got >= self.IDLE_SHARE * total or len(out) >= self.IDLE_MAX:
                break
            out.append(k)
            got += n
        return out

    def _missing(self, cls: int) -> list:
        """What the class lacks of the shapes the rounds use (`_top`): ("v", M) verify widths,
        ("c", (M, n, moved)) commits of a captured or missing width, ("m", n) the MTP head's feeds."""
        widths = self._top(False)
        out = [("v", M) for M in widths if (M, cls) not in self.vgraphs]
        out += [("c", k) for k in self._top(True) if k[0] in widths and k + (cls,) not in self.cgraphs
                and (not k[2] or 2 <= k[1] < k[0])]
        for head in self.heads.values():
            out += [("m", n) for n in range(1, head.depth + 2) if (n, cls) not in head.graphs]
        return out

    def capture_cost(self, n: int) -> float:
        """What a drafted turn at context length `n` would spend capturing the graphs its class lacks,
        ms: the first drafted request in each class paid 3.4-5.1 s of captures (P0, 2026-10-12)
        until the idle captures below took them (`StairCut.turn` weighs it with the backlog)."""
        miss = self._missing(self.fn.qsa_class(n))
        return len(miss) * (self.capture_ms if self.capture_ms is not None else CAPTURE_MS)

    def idle_captures(self):
        """The graphs the class of the state's length lacks, one capture a step (a generator: the
        server stops between two when a request waits), or None when it lacks none. An empty state
        (a side slot after its warm-up) captures class 0's: its first drafted turn would pay them, so
        it would never draft (2026-10-12: 3 requests at once, the side slots drafted nothing). Every
        capture leaves the state as it was: a verify writes only rows past the length, a commit's and the
        MTP chain's warm-ups are undone (engine.py)."""
        cls = self.fn.qsa_class(self.st.length)
        miss = self._missing(cls)
        return self._capture_steps(cls, miss) if miss else None

    @torch.no_grad()
    def _capture_steps(self, cls: int, miss: list):
        self._settle_chain()
        fn, st = self.fn, self.st
        self._ensure(st.length + CAPTURE_ROWS)
        for kind, key in sorted(miss, key=lambda m: "vcm".index(m[0])):     # widths before their commits
            t0 = time.perf_counter()
            if kind == "v":
                g = VerifyGraph(fn, st, key, cls)
                g.capture()
                self.vgraphs[(key, cls)] = g
                self._fresh.add((key, cls))
            elif kind == "c":
                M, n, moved = key
                vg = self.vgraphs.get((M, cls))
                if vg is None:
                    continue
                parents, path = list(range(-1, M - 1)), list(range(n))
                if moved:
                    # the path leaves the chain at its last row: row M - 1, a second child of row n - 2
                    parents[M - 1], path[-1] = n - 2, M - 1
                fn.stage_rows([0] * M, parents, st)
                fn._vcls = cls
                vg.replay()
                fn.stage_commit(st, path)
                g = CommitGraph(fn, st, (M, n, moved, cls))
                g.capture()
                self.cgraphs[(M, n, moved, cls)] = g
                self._fresh.add((M, n, moved, cls))
            else:
                for head in self.heads.values():
                    head.capture_class(key, cls)
            torch.cuda.synchronize()
            self._capture_took(t0)
            yield

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the decode step, more through the prefill
        kernels. Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        self._ensure(start + T)
        if T == 1 and start > 0:
            if self.dg is None:
                self.dg = DecodeGraph(self.fn, self.st)
            cls = self.fn.qsa_class(start)
            n0 = self.dg.captures
            lg = self.dg(int(ids[0]))
            self.plain_captured = self.dg.captures != n0       # this step's time is a capture's
            self._hidden = self.dg.stacks[cls].view(1, 1, -1)
            self._h_last.copy_(self._hidden[0, -1:])
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st, keep_stack=True)
        self._hidden = self.fn.prompt_stack.unsqueeze(0)
        self._h_last.copy_(self._hidden[0, T - 1:T])
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32 (a static buffer), its greedy pick in `picks`, its streams in `hidden_post_norm`. Nothing
        but the block's own KV rows is written until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        M = len(toks)
        self._ensure(start + M)
        n_graphs = len(self.vgraphs)
        t0 = time.perf_counter()
        self.fn.verify(toks, [int(p) for p in parents], self.st, self.vgraphs)
        self.captured = len(self.vgraphs) > n_graphs
        self.uses[M] = self.uses.get(M, 0) + 1
        self._width = M
        cls = getattr(self.fn, "_vcls", 0)
        if (M, cls) in self._fresh:
            # an idle capture's first replay uploads the graph: not a price either
            self._fresh.discard((M, cls))
            self.captured = True
        if self.captured:
            self._capture_took(t0)
        rb = self.fn._rb[M]
        self.picks = rb["next"][:M]
        self._hidden = self.fn.rec["stack"][:M].unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += M
        return rb["logits"][:M]

    def tree_batchable(self, parents) -> bool:
        """A block a joint pass may take (`forward_tree_multi`): its rows within the widths whose
        projections and experts give a row its lone bits (the self-check's, up to 32)."""
        return len(parents) <= 32

    @staticmethod
    @torch.no_grad()
    def forward_tree_multi(items: list) -> list:
        """ENG-241 step 3: [(serving, block, parents, start)] of sequence slots over one engine, verified
        in one pass (`FlashNextEngine.verify_multi`: the weights read once); each slot's logits, picks
        and streams as its `forward_tree` leaves them, and its block committed by its `commit_tree`."""
        fn = items[0][0].fn
        jobs = []
        for s, block, parents, start in items:
            s._settle_chain()
            s._at(start)
            s._ensure(start + block.numel())
            jobs.append(([int(t) for t in block.view(-1).tolist()], [int(p) for p in parents], s.st))
        outs = fn.verify_multi(jobs)
        torch.cuda.current_stream().synchronize()
        res = []
        for (s, block, parents, start), (logits, picks, stack) in zip(items, outs):
            M = block.numel()
            s.picks, s._hidden = picks, stack.unsqueeze(0)
            s.captured, s._width = False, M
            s.uses[M] = s.uses.get(M, 0) + 1
            s.stats["verify_blocks"] += 1
            s.stats["verify_rows"] += M
            res.append(logits)
        return res

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        t0 = time.perf_counter()
        n_graphs = len(self.cgraphs)
        stack = self.fn.commit(self.st, [int(p) for p in path], self.cgraphs)
        self.commit_ms = (time.perf_counter() - t0) * 1e3
        if len(self.cgraphs) > n_graphs:
            self.captured = True
            self._capture_took(t0)
        k = (self._width, len(path), any(int(p) != j for j, p in enumerate(path)))
        self.uses[k] = self.uses.get(k, 0) + 1
        if k + (getattr(self.fn, "_vcls", 0),) in self._fresh:
            self._fresh.discard(k + (getattr(self.fn, "_vcls", 0),))
            self.captured = True
        if path:
            self._h_last.copy_(stack[len(path) - 1:len(path)])

    @torch.no_grad()
    def forward_block(self, block: torch.Tensor, start: int) -> torch.Tensor:
        """A chain block: its rows' logits; the rows are kept by `rollback_to(keep)`, or all of them
        by the next call on the engine."""
        n = block.numel()
        lg = self.forward_tree(block, list(range(-1, n - 1)), start)
        self._chain = n
        return lg

    @torch.no_grad()
    def rollback_to(self, keep: int) -> None:
        self._chain = None
        self.commit_tree(list(range(int(keep))))

    # --- internals -------------------------------------------------------------------------------
    def _states(self) -> list:
        return [self.st] + [h.st for h in self.heads.values()]

    def _settle_chain(self) -> None:
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def _ensure(self, rows: int) -> None:
        """ENG-241: commit the pages that hold rows [0, rows) and the pool's margin before a write
        reaches them (one int compare when they are); never inside a graph capture, whose caller
        committed the rows before it captured."""
        ps = self.st.pages
        if ps is None or rows + ps.margin() <= ps.rows:
            return
        if ps.on_device() and torch.cuda.is_current_stream_capturing():
            return
        ps.ensure(rows)

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"Flash-Next forward at {start}, the state holds {self.st.length} rows")


#: one graph's capture before the server has timed its own, ms (P0, 2026-10-12: 3,391 ms for 9 at the
#: 32,768 class, 5,121 for 12, 4,981 for 15 at 131,072)
CAPTURE_MS = 380.0
#: a side slot's graph pools and scratch beyond its state, bytes (ENG-241: an estimate until measured)
SIDE_SLOT_CONST = int(1.5 * (1 << 30))
#: the shapes a class captures ahead before any round was counted: chain blocks of 2-6 rows, each
#: path length (one round each, so the server's own rounds outweigh them at once)
PRIOR_USES = {**{M: 1 for M in range(2, 7)}, **{(M, n, False): 1 for M in range(2, 7) for n in range(1, M + 1)}}

#: the repository root (the hot draft vocabulary is ops/fn-hot-vocab.npy, the price files ops/prices/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: the model's name in ops/prices/ and models/
MODEL_ID = "qwen3.8-flash-next"


def profile_of(pack: str) -> str:
    """The weight profile a pack is served as: the running one (QSE_WEIGHTS), else flashnext-<pack dir>."""
    return os.environ.get("QSE_WEIGHTS") or "flashnext-" + os.path.basename(os.path.normpath(pack))


def prices(pack: str, log=print) -> PriceModel:
    """StairCut's prior for this pack on this device (ops/prices/<model>/<profile>.<device>.json)."""
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(pack)}.{device_id()}.json"),
                           log=log)


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 4, mark=lambda stage: None, log=print,
          corpus: str | None = None, kv_pool=None):
    """Flash-Next for the server (`serve` below): the pack's weights, the engine in the loop's
    interface, and with `spec` the shared MTP drafter over the pack's MTP head with the hot draft head
    (`FN_DRAFT_HOT` ids of ops/fn-hot-vocab.npy, 0 = the whole vocabulary), its graphs captured and
    StairCut's prior from the price file -- gated by the load-time self-check (a failed check serves
    plain decode); with FN_LOOKUP the lookup over the request's text and `corpus` joins the cut (ENG-229;
    the tokenizer is the 27B's, so its corpus and suffix store read as they are, checked by fingerprint).
    `mark(stage)` at "weights", "engine", "drafters" (the server's load ledger). `kv_pool()`: opens the
    KV pool after the weights, before the state is allocated, as the 27B's builder does (ENG-241: opened
    before the weights, the self-check's first Triton launch refused a pointer, "cannot be accessed
    from Triton", 2026-10-12).
    Returns (serving, drafter or None, info)."""
    from engine.models.flashnext.load import load as fn_load
    idx, W, banks, tables, timing = fn_load(pack, "cuda", mtp=spec)
    mark("weights")
    fn = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max_len)
    if kv_pool is not None:
        kv_pool()
    eng = FlashNextServing(fn, max_len)
    mark("engine")
    drafter, check, hot = None, None, int(_S.get("FN_DRAFT_HOT"))
    if spec:
        if hot:
            fn.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check = fn.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
        if check["ok"]:
            ngram = None
            if _S.get("FN_LOOKUP") == "1":
                from engine.drafters.ngram import NgramDrafter
                from engine.tokfp import fingerprint
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack), min_order=3,
                                     max_depth=16, min_corpus_order=8)
                log(f"[server] Flash-Next lookup in the cut: corpus "
                    f"{(corpus or 'none') if ngram.corpus is not None else 'none'}")
            drafter = MTPTreeDrafter(eng, eng.add_mtp_head(depth), StairCut(prices(pack, log), siblings=False), ngram)
            drafter.capture_all()
        else:
            log("[server] Flash-Next self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec else None, "depth": depth if drafter is not None else 0}
    return eng, drafter, info


def serve(a, ctx) -> dict:
    """Qwen3.8-Flash-Next from an EXL3 pack for the server (`--model <pack>`, the registry's `serve`;
    server/app.py `_load`): the engine and its MTP drafter in the server loop's interface, so the
    request path is the 27B's. The draft is the MTP chain from graphs cut by StairCut over the hot
    draft head, verified as a tree that a sampled request walks (`det`); the load-time self-check gates
    it. `ctx` is the server's: `keys(path, weights)` (the tokenizer's and the weights' fingerprints),
    `mark(stage)` (the load ledger), `log(line)`, `vision(path, tower_of)` (the server's tower, processor and
    image caches, with the pack's EXL3 tower: VIS-97). What it cannot hold yet is in `FlashNextServing.caps`."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    spec = a.drafter != "none"
    if a.drafter not in ("mtp", "none"):
        ctx.log(f"[server] Flash-Next drafts with its own MTP layer: --drafter {a.drafter} taken as mtp")
    tok = AutoTokenizer.from_pretrained(a.model)
    depth = a.depth if a.depth > 3 else 4
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    eng, drafter, info = build(a.model, a.max_len, tok, spec=spec, depth=depth, mark=ctx.mark, log=ctx.log,
                               corpus=a.corpus, kv_pool=pool)

    def tower_of(vcfg):
        from engine.models.flashnext.pack import PackIndex
        from engine.models.flashnext.vision import load_tower
        return load_tower(PackIndex.open(a.model, table=False), a.model)
    ctx.vision(a.model, tower_of)
    ctx.mark("vision")
    if drafter is not None:
        a.depth = depth
        if drafter.ngram is not None:
            # the loop hands a round k rows past the anchor: the lookup's copy may run past the head's
            # depth, up to what the cut admits
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
    describe = (f"Flash-Next {a.model}: load {info['load'].get('total_s')} s, drafter "
                f"{'MTP chain + StairCut, depth ' + str(a.depth) if drafter is not None else 'none'}"
                f"{' + lookup' if getattr(drafter, 'ngram', None) is not None else ''}, "
                f"max_len {a.max_len:,},")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "exl3", "weights": a.model,
                                         "drafters": "mtp (the pack's own layer)" if drafter is not None else None,
                                         "draft_head": info["draft_head"], "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"]}}}
