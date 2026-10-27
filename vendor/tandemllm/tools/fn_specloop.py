"""Greedy speculative decoding on Flash-Next outside the server, as a test driver (ENG-227, ENG-153;
moved out of the engine in ENG-245): draft, verify, walk, commit, round by round, for the measurement
and proof tools (tools/fn_spec_check.py, fn_spec_bench.py, fn_draft_trace.py, fn_serve.py). The server
drives Flash-Next through its own loop (server/app.py, engine/models/flashnext/serving.py).

A round starts at the anchor -- the last token the target chose, not yet run through the trunk -- at
the state's length L. A drafter proposes a block in DFS pre-order with the anchor as row 0
(`engine/tree.py`'s convention: `tokens`, `parents`). One verify (`FlashNextEngine.verify`) gives the
target's greedy token at every row; the walk (engine/methods/walk.py) follows the rows whose token is
the target's choice at their parent; the commit keeps that path. The round emits the path's drafted
tokens and the target's token after the path's last row, which is the next anchor.

Every verified row is the decode step's row at that position bit for bit, and the path's state is the
decode walk's, so the emitted text is the plain greedy text token for token whatever the drafter
proposes. `tools/fn_spec_check.py` proves it on the board.
"""

from __future__ import annotations

import time
import types

import torch

from engine.methods.staircut import StairCut
from engine.methods.walk import accept_tree
from engine.models.flashnext.engine import FlashNextEngine
from engine.models.flashnext.mtp import FlashNextMTPHead
from engine.models.flashnext.ref import FNState


def walk(tokens: list[int], parents: list[int], target: list[int]) -> list[int]:
    """The accepted path of a block (the shared walk's path)."""
    return accept_tree(types.SimpleNamespace(tokens=tokens, parents=parents), target)[0]


class Drafter:
    """What this driver asks of a draft source."""

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        """A block for the round: (tokens, parents), tokens[0] = the anchor, DFS pre-order."""
        raise NotImplementedError

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        """The round's committed rows: their streams [n, hc H] (positions L..L+n-1) and the token that
        follows each of them (the path's drafted tokens, then the new anchor)."""

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """The round's accepted path (rows of the block it proposed) and what the round cost."""


class SpecDecoder:
    """The round loop over one sequence. `graphs`: verify blocks from captured graphs (one per row
    count, captured on first use)."""

    def __init__(self, eng: FlashNextEngine, st: FNState, drafter: Drafter, graphs: bool = True,
                 log_union: bool = False):
        self.eng, self.st, self.drafter = eng, st, drafter
        self.log_union = log_union
        self.union: dict = {}          # rows -> [rounds, sum over layers of distinct experts read]
        self.graphs: dict | None = {} if graphs else None
        self.commit_graphs: dict | None = {} if graphs else None
        self.stats = {"rounds": 0, "rows": 0, "accepted": 0, "moved": 0, "draft_s": 0.0, "verify_s": 0.0,
                      "commit_s": 0.0}

    @torch.no_grad()
    def generate(self, first: int, new: int, eos: set, round_ms: list | None = None, emit=None) -> list[int]:
        """Tokens from `first` (the anchor after the prompt) until `new` tokens or an EOS, first included.
        `emit(tokens)` is called with each round's new tokens (after the cut at `new` or an EOS); a
        False return stops after that round. `self.full` keeps every emitted token uncut: the state
        has run (prompt + full)[:length]."""
        eng, st, s = self.eng, self.st, self.stats
        out = [first]
        self.full = [first]
        if emit is not None and emit([first]) is False:
            return out
        while len(out) < new and out[-1] not in eos:
            t0 = time.perf_counter()
            tokens, parents = self.drafter.propose(out[-1])
            t1 = time.perf_counter()
            target = eng.verify(tokens, parents, st, self.graphs)
            t2 = time.perf_counter()
            if self.log_union:
                self._union(len(tokens))
            path = walk(tokens, parents, target)
            stack = eng.commit(st, path, self.commit_graphs)
            emitted = [tokens[r] for r in path[1:]] + [target[path[-1]]]
            self.drafter.accepted(stack, emitted)
            torch.cuda.current_stream().synchronize()
            t3 = time.perf_counter()
            self.drafter.observe(path, (t3 - t1) * 1e3, (t1 - t0) * 1e3)
            s["rounds"] += 1
            s["rows"] += len(tokens)
            s["accepted"] += len(path) - 1
            s["moved"] += int(any(path[j] != j for j in range(len(path))))
            s["draft_s"] += t1 - t0
            s["verify_s"] += t2 - t1
            s["commit_s"] += t3 - t2
            if round_ms is not None:
                round_ms.append(((t3 - t0) * 1e3, len(emitted)))
            self.full += emitted
            n0 = len(out)
            for t in emitted:
                out.append(t)
                if t in eos or len(out) >= new:
                    break
            if emit is not None and emit(out[n0:]) is False:
                break
        return out

    def _union(self, M: int) -> None:
        """The distinct routed experts the verify block's rows read, summed over the MoE layers."""
        E = self.eng.cfg.num_experts
        ids = [v for k, v in self.eng.rec.items() if isinstance(k, tuple) and k[0] == "ids"]
        if not ids:
            return
        a = torch.stack([i[:M] for i in ids]).cpu()                    # [layers, M, k + 1]
        n = sum(len(set(x.tolist()) - {E}) for x in a.view(a.shape[0], -1))
        u = self.union.setdefault(M, [0, 0])
        u[0] += 1
        u[1] += n


class MTPChainDrafter(Drafter):
    """The pack's MTP layer as a greedy chain drafter (eager): each round feeds the committed rows
    (kept in its own cache) and drafts `depth` tokens from the last row's own streams, the deeper
    feeds rolled back."""

    def __init__(self, eng: FlashNextEngine, depth: int, cap: int):
        self.eng, self.depth = eng, depth
        self.st = FNState(cap=cap)
        self.pending = None

    def prime(self, stack: torch.Tensor, ids: list[int], first: int) -> None:
        """After the prompt: its streams [L, hc H] and tokens; `first` follows the last one."""
        L = len(ids)
        if L > 1:
            self.eng.mtp_feed(stack[: L - 1], torch.tensor(ids[1:L]), self.st)
        self.pending = (stack[L - 1:L], [first])

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        eng = self.eng
        rows, nexts = self.pending
        logits, h = eng.mtp_feed(rows, torch.tensor(nexts), self.st)
        keep = self.st.length
        out = [int(logits[-1].argmax())]
        for _ in range(self.depth - 1):
            logits, h = eng.mtp_feed(h[-1:], torch.tensor([out[-1]]), self.st)
            out.append(int(logits[-1].argmax()))
        self.st.length = keep
        self.st.clamp_pool()
        return [anchor] + out, list(range(-1, self.depth))

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        self.pending = (stack, nexts)


class MTPGraphDrafter(Drafter):
    """The MTP head from captured graphs (engine/models/flashnext/mtp.py) in this driver's protocol.
    Without a `cut` the block is the greedy chain of `depth`; with one (the shared `StairCut`) the
    block is the chain and the sibling leaves it keeps."""

    def __init__(self, eng: FlashNextEngine, depth: int, cap: int, cut: StairCut | None = None):
        self.eng, self.depth, self.cut = eng, depth, cut
        self.head = FlashNextMTPHead(eng, depth, cap)
        self.pending = None

    @property
    def st(self):
        return self.head.st

    def prime(self, stack: torch.Tensor, ids: list[int], first: int) -> None:
        L = len(ids)
        if L > 1:
            self.head.prime(stack[:L - 1], ids[1:L])
        self.pending = (stack[L - 1:L], [first])

    def capture_all(self) -> None:
        self.head.capture_all()

    def propose(self, anchor: int) -> tuple[list[int], list[int]]:
        rows, nexts = self.pending
        self.head.launch(rows, nexts)
        tok, prob = self.head.read()
        if self.cut is None:
            return [anchor] + [t[0] for t in tok], list(range(-1, self.depth))
        return self.cut.cut(anchor, tok, prob)

    def accepted(self, stack: torch.Tensor, nexts: list[int]) -> None:
        self.pending = (stack, nexts)

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        if self.cut is not None:
            self.cut.observe(path, round_ms, draft_ms)
