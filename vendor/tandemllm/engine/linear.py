"""One interface for every weight format the engine multiplies by.

    class Linear(Protocol):
        shape: (N, K)          nbytes: bytes read per use          matmul(x[..., K]) -> [..., N]

`FP8Block` (tools/fp8_linear.py, the checkpoint's e4m3 + 128x128 scales), `NVFP4Block`
(tools/nvfp4_linear.py), `FP8Head` (tools/head_gemv.py, fp32 logits) and `BF16Block` below implement
it; `engine.model.linear` calls `w.matmul(x)` and takes a plain tensor through `F.linear` as before.
Each `matmul` calls exactly the kernel the old `isinstance` chain called, with the same arguments,
so the bytes do not move (tests/test_linear_api.py, the refactor gate). Fused groups keep their own
entry (`engine.model.matmul_group`), because a group is split by its caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class Linear(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def nbytes(self) -> int: ...

    def matmul(self, x: torch.Tensor) -> torch.Tensor: ...


class BF16Block:
    """A plain bf16 projection as a Linear (the BF16 checkpoint target)."""

    __slots__ = ("w", "N", "K")

    def __init__(self, w: torch.Tensor):
        assert w.dim() == 2, w.shape
        self.w = w
        self.N, self.K = w.shape

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() * self.w.element_size()

    def dequant(self) -> torch.Tensor:
        return self.w

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w)
