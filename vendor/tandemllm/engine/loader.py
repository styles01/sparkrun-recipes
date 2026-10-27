"""Read the checkpoint into device tensors, text model only, fp8 left as codes.

The checkpoint is one safetensors file per layer plus `outside.safetensors` (embeddings, head, final
norm, and the vision tower) and `mtp.safetensors`. The vision tensors (0.92 GB of a 30.87 GB
checkpoint, bf16) are skipped by name here: the language model never reads them, and
engine/vision.py loads them as the vision tower when image input is on.

Quantised projections arrive as a pair -- `X.weight` (fp8 e4m3) and `X.weight_scale_inv` (bf16, one
value per 128x128 block) -- and are kept as that pair in an `FP8Block`. Nothing is dequantised at
load time: the decode step is bandwidth-bound on exactly these bytes.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fp8_linear import FP8Block, FP8Group  # noqa: E402
from tools.nvfp4_linear import NVFP4Block  # noqa: E402

LM_PREFIX = "model.language_model."
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


class Layout:
    """How a checkpoint of this model family names its tensors, in one place.

    The engine addresses every tensor by a canonical name -- `embed_tokens.weight`,
    `lm_head.weight`, `norm.weight`, `layers.N.<module>.<proj>` -- which is the language model's
    own naming with the checkpoint's wrapper prefix taken off. What differs between checkpoints of
    the family is only the wrapper: the vision-language checkpoint served today puts the language
    model under `model.language_model.` and carries a vision tower (`visual.` / `.visual.`) that
    engine/vision.py loads on its own; a text-only checkpoint puts it under `model.`, and `lm_head.weight`
    sits at the top level in both. The MTP layer is its own file. A head that is not in the
    checkpoint is the embedding when the config says the weights are tied.
    """

    prefixes = (LM_PREFIX, "model.")          # stripped, first match wins
    skip_inside, skip_start = ".visual.", "visual."
    mtp_file = "mtp.safetensors"
    mtp_prefix = "mtp."
    embed = "embed_tokens.weight"
    head = "lm_head.weight"
    final_norm = "norm.weight"
    # the projections the FP8 checkpoint stores as e4m3 + scales; a BF16 checkpoint stores the same
    # names as plain bf16 matrices, which the loader wraps as `BF16Block`s
    projections = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                   "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")

    @classmethod
    def plain_projection(cls, name: str, t) -> str | None:
        """The projection base name of a plain (bf16/fp16/fp32) 2-D weight, else None."""
        if not name.endswith(".weight") or t.dim() != 2 or t.dtype == torch.float8_e4m3fn:
            return None
        base = name[: -len(".weight")]
        return base if base.endswith(cls.projections) else None

    @classmethod
    def canonical(cls, key: str) -> str | None:
        """The engine's name for a checkpoint key, or None for a tensor the engine never reads."""
        if cls.skip_inside in key or key.startswith(cls.skip_start):
            return None
        for pre in cls.prefixes:
            if key.startswith(pre):
                return key[len(pre):]
        return key

# Projections that read the SAME activation and do not read each other's output, so their launches
# can be one launch. See `tools/nvfp4_linear_v2.NVFP4Group` for why that is worth doing; the short
# version is that a verify runs one kernel at a time and the widest of them puts 272 programs on
# 48 SMs, so the board spends every launch's tail idle and there are eleven launches a layer.
#
# These three cover 9.2 GB of the verify's 13.7 and take the step from 400 launches to 256.
PROJ_GROUPS = (
    ("mlp.gate_up", ("mlp.gate_proj", "mlp.up_proj")),
    ("self_attn.qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("linear_attn.qkvz", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")),
)


class Weights:
    """Every tensor the text model needs, addressed by its checkpoint name."""

    def __init__(self, snapshot: str, device: str = "cuda", *, skip_mtp: bool = False,
                 nvfp4: str | None = None, fp8_head: str | None = None, tied: bool | None = None,
                 exl3: str | None = None):
        self.snapshot = snapshot
        self.device = device
        self.t: dict[str, torch.Tensor] = {}
        self.q: dict[str, FP8Block | NVFP4Block] = {}
        self.g: dict = {}
        self.bytes_fp8 = 0
        self.bytes_other = 0
        self.nvfp4_source: str | None = None
        self.fp8_head_source: str | None = None
        self.exl3_source: str | None = None
        # ENG-197: an EXL3 checkpoint replaces every projection it quantises and the head. The
        # base snapshot still gives everything else (norms, embedding, the GDN gates and
        # convolutions in their original bf16), and its copies of the replaced projections are
        # never read off the disk. A standalone EXL3 pack identifies itself in config.json;
        # explicit exl3 paths (or an empty string to disable it) still take precedence.
        exl3 = exl3 if exl3 is not None else _S.get("EXL3")
        if exl3 is None and _config_exl3(snapshot):
            exl3 = snapshot
        self._skip: set[str] = set()
        if exl3:
            exl3 = os.path.expanduser(str(exl3))
            self._skip = exl3_bases(exl3)
        self._standalone_exl3 = bool(exl3 and os.path.realpath(exl3) == os.path.realpath(snapshot))
        self._standalone_bf16 = self._standalone_exl3 and _config_declares_bf16(snapshot)
        self._load(skip_mtp)
        if Layout.head not in self.t and Layout.embed in self.t and "lm_head" not in self._skip:
            if tied is None:
                tied = _config_tied(snapshot)
            if not tied:
                raise RuntimeError(f"{snapshot}: no {Layout.head} and the config does not tie it "
                                   f"to {Layout.embed}")
            # the same tensor under both names: nothing is copied; `load_fp8_head`/`build_fp8_head`
            # replace only the head's entry, the embedding stays what it is
            self.t[Layout.head] = self.t[Layout.embed]
        nvfp4 = nvfp4 if nvfp4 is not None else _S.get("NVFP4")
        fp8_head = fp8_head if fp8_head is not None else _S.get("FP8_HEAD")
        if exl3:
            if nvfp4 or fp8_head:
                print(f"[exl3] {exl3} replaces the projections and the head: "
                      f"--nvfp4 {nvfp4!r} and --fp8-head {fp8_head!r} are not loaded")
            nvfp4 = fp8_head = None
            self.load_exl3(exl3)
        if nvfp4:
            # A comma-separated list, because the quality gate decides which GROUPS of projections
            # are quantised and that decision has to be expressible without re-running the
            # quantiser: one file per group, and the ones that passed are the ones that are named.
            for part in str(nvfp4).split(","):
                if part.strip():
                    self.load_nvfp4_mlp(os.path.expanduser(part.strip()))
        if fp8_head:
            ratios = parse_head_build(fp8_head)
            if ratios is not None:
                self.build_fp8_head(ratios)
            else:
                self.load_fp8_head(os.path.expanduser(fp8_head))
        from tools.nvfp4_linear_v2 import FUSE_PROJ
        if exl3 and _S.get("EXL3_FUSE") == "1":
            layers = [int(k.split(".")[1]) for k in self.q if k.startswith("layers.")]
            if layers:
                self.fuse_exl3_groups(max(layers) + 1)
        elif FUSE_PROJ:
            layers = [int(k.split(".")[1]) for k in self.q if k.startswith("layers.")]
            if layers:
                self.fuse_nvfp4_groups(max(layers) + 1)

    def _files(self, skip_mtp: bool) -> list[str]:
        out = []
        for name in sorted(os.listdir(self.snapshot)):
            if not name.endswith(".safetensors"):
                continue
            if name == Layout.mtp_file and skip_mtp:
                continue
            out.append(os.path.join(self.snapshot, name))
        return out

    def _load(self, skip_mtp: bool) -> None:
        pending_scale: dict[str, torch.Tensor] = {}
        pending_code: dict[str, torch.Tensor] = {}
        for path in self._files(skip_mtp):
            with safe_open(path, framework="pt", device=self.device) as f:
                for key in f.keys():
                    name = Layout.canonical(key)
                    if name is None:
                        continue
                    if skip_mtp and name.startswith(Layout.mtp_prefix):
                        # the BF16 release keeps the MTP layer inside its last shard, not in a file
                        # of its own: skipped by name, as the FP8 release's file is skipped by name
                        continue
                    if self._skip and name.rsplit(".", 1)[0] in self._skip:
                        continue          # replaced by the EXL3 checkpoint: never read
                    if name.endswith(".weight_scale_inv"):
                        base = name[: -len(".weight_scale_inv")]
                        s = f.get_tensor(key)
                        if base in pending_code:
                            self._pair(base, pending_code.pop(base), s)
                        else:
                            pending_scale[base] = s
                        continue
                    t = f.get_tensor(key)
                    if self._standalone_exl3 and t.dtype == torch.float16:
                        if not self._standalone_bf16:
                            raise ValueError(f"{key}: standalone EXL3 FP16 copies require a declared BF16 "
                                             "source dtype (dtype or torch_dtype); use the original BF16 "
                                             "checkpoint as the base and this pack as the EXL3 overlay")
                        t = _stored_fp16_to_bf16(t, key)
                    if t.dtype == torch.float8_e4m3fn:
                        base = name[: -len(".weight")]
                        if base in pending_scale:
                            self._pair(base, t, pending_scale.pop(base))
                        else:
                            pending_code[base] = t
                        continue
                    base = Layout.plain_projection(name, t)
                    if base is not None:
                        # a BF16 checkpoint's projection, read through the Linear interface
                        from engine.linear import BF16Block
                        blk = BF16Block(t)
                        self.q[base] = blk
                        self.bytes_fp8 += blk.nbytes
                        continue
                    self.t[name] = t
                    self.bytes_other += t.numel() * t.element_size()
        if pending_code or pending_scale:
            raise RuntimeError(f"unpaired fp8 tensors: {sorted(pending_code) + sorted(pending_scale)}")

    def _pair(self, base: str, codes: torch.Tensor, scale: torch.Tensor) -> None:
        blk = FP8Block(codes, scale)
        self.q[base] = blk
        self.bytes_fp8 += blk.nbytes

    # --- accessors used by the model ---
    def norm(self, name: str) -> torch.Tensor:
        return self.t[name]

    def proj(self, name: str) -> FP8Block:
        return self.q[name]

    def group(self, name: str):
        """The fused form of a projection group, or None if it was not built for this layer."""
        return self.g.get(name)

    def fuse_nvfp4_groups(self, n_layers: int) -> int:
        """Lay each group's projections out as one weight, and leave the members as views of it.

        NVFP4 groups, and since FP8 groups (all members plain `FP8Block`s), and only
        complete, single-format ones: a group whose members are part NVFP4 and part fp8 (the
        quality gate left some of them in fp8) keeps its separate launches, and the model falls
        back to them by finding no group. Nothing is duplicated -- `torch.cat` along dim 0 leaves each member's
        rows contiguous, so after the copy every member points into the fused buffer and its own
        storage is dropped. Peak cost is one extra copy of the largest group, 89 MB.
        """
        from tools.nvfp4_linear_v2 import NVFP4Group
        made, fused_bytes = 0, 0
        for layer in range(n_layers):
            p = f"layers.{layer}"
            for key, members in PROJ_GROUPS:
                names = [f"{p}.{m}" for m in members]
                blocks = [self.q.get(n) for n in names]
                if all(isinstance(b, NVFP4Block) for b in blocks):
                    grp = NVFP4Group(blocks, names)
                elif all(type(b) is FP8Block for b in blocks):
                    # the plain-FP8 weight set gets the same launch count
                    grp = FP8Group(blocks, names)
                else:
                    continue
                self.g[f"{p}.{key}"] = grp
                fused_bytes += grp.nbytes
                made += 1
        torch.cuda.empty_cache()
        if made:
            print(f"[fuse] {made} projection groups, {fused_bytes / 1e9:.2f} GB in one launch each")
        return made

    def report(self) -> str:
        total = self.bytes_fp8 + self.bytes_other
        return (f"{len(self.q)} quantised projections ({self.bytes_fp8 / 2**30:.2f} GiB), "
                f"{len(self.t)} plain tensors ({self.bytes_other / 2**30:.2f} GiB), "
                f"{total / 2**30:.2f} GiB resident")

    def decode_step_bytes(self, n_layers: int) -> dict[str, float]:
        """Bytes a single autoregressive step has to read, by group."""
        per_layer: dict[int, int] = {}
        for base, blk in self.q.items():
            if not base.startswith("layers."):
                continue
            idx = int(base.split(".")[1])
            per_layer[idx] = per_layer.get(idx, 0) + blk.nbytes
        for name, t in self.t.items():
            if not name.startswith("layers."):
                continue
            idx = int(name.split(".")[1])
            per_layer[idx] = per_layer.get(idx, 0) + t.numel() * t.element_size()
        layers = sum(v for k, v in per_layer.items() if k < n_layers)
        head = self.t["lm_head.weight"]
        # the e4m3 head is an FP8Head (codes + per-row scales), not a tensor
        hb = head.numel() * head.element_size() if isinstance(head, torch.Tensor) else head.nbytes
        return {
            "layers_GB": layers / 1e9,
            "lm_head_GB": hb / 1e9,
            "total_GB": (layers + hb) / 1e9,
        }

    # ------------------------------------------------------------------ NVFP4 MLPs
    def load_nvfp4_mlp(self, path: str) -> None:
        """Replace every layer's three MLP projections with their NVFP4 form.

        Two sources are accepted and normalised to the same three tensor names, because the whole
        point of the exercise is to compare them: a file written by `tools/quant_nvfp4.py`, whose
        keys are already `layers.N.mlp.X.<...>`, and a published NVFP4 snapshot directory, whose
        keys carry the checkpoint's `model.language_model.` prefix and whose other tensors -- the
        attention and linear-attention projections, which that checkpoint quantises per tensor and
        this engine reads per 128x128 block -- are deliberately not read.

        The fp8 codes of the replaced projections are dropped as each one is swapped, so the peak
        footprint is one extra projection, not a second copy of the MLPs.
        """
        files = []
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.isfile(idx):
                import json
                names = sorted(set(json.load(open(idx))["weight_map"].values()))
                files = [os.path.join(path, n) for n in names]
            else:
                files = sorted(glob_safetensors(path))
        else:
            files = [path]
        found = 0
        freed = 0
        added = 0
        # A directory is somebody else's published snapshot, and this engine reads only its MLPs:
        # its attention and linear-attention tensors are quantised per tensor where this one reads
        # per 128x128 block, so they are a different format wearing the same names. A single file
        # is one this repository wrote, and whatever is in it is what was asked for -- since 13:20
        # of phase 4 that can be the GDN and attention projections as well.
        mlp_only = os.path.isdir(path)
        for fpath in files:
            with safe_open(fpath, framework="pt", device=self.device) as f:
                keys = [k for k in f.keys() if (not mlp_only or ".mlp." in k) and (
                    k.endswith(".weight") or k.endswith(".weight_scale")
                    or k.endswith(".weight_scale_2"))]
                bases = sorted({k.rsplit(".", 1)[0] for k in keys
                                if k.rsplit(".", 1)[1] in ("weight", "weight_scale")})
                if mlp_only:
                    bases = [b for b in bases if b.split(".")[-1] in MLP_PROJ]
                for base in bases:
                    if f"{base}.weight_scale_2" not in f.keys():
                        continue          # not NVFP4: a plain fp8 or bf16 MLP, leave it alone
                    name = base[len(LM_PREFIX):] if base.startswith(LM_PREFIX) else base
                    if name not in self.q:
                        continue
                    old = self.q[name]
                    blk = NVFP4Block(f.get_tensor(f"{base}.weight"),
                                     f.get_tensor(f"{base}.weight_scale"),
                                     f.get_tensor(f"{base}.weight_scale_2").float().item())
                    assert blk.shape == old.shape, (name, blk.shape, old.shape)
                    freed += old.nbytes
                    added += blk.nbytes
                    self.q[name] = blk
                    old.w = None
                    if hasattr(old, "s"):     # a BF16Block (the BF16 release) has no scale table
                        old.s = None
                    found += 1
        if not found:
            raise RuntimeError(f"no NVFP4 tensors found in {path}")
        torch.cuda.empty_cache()
        self.bytes_fp8 += added - freed
        self.nvfp4_source = path if not self.nvfp4_source else f"{self.nvfp4_source},{path}"
        print(f"[nvfp4] {found} projections from {path}: "
              f"{freed / 1e9:.2f} GB fp8 -> {added / 1e9:.2f} GB nvfp4")

    # ------------------------------------------------------------------ EXL3 (ENG-197)
    def load_exl3(self, path: str) -> None:
        """Every projection and the head from an EXL3 checkpoint (tools/exl3_linear.py).

        Read: `<base>.trellis/.suh/.svh/.mul1` of the language model's projections and of
        `lm_head`; the MTP layer, the vision tower and every plain tensor of the EXL3 checkpoint
        are not read (the base snapshot's originals serve: the converter stores some of them as
        fp16 copies of the bf16 originals, e.g. in_proj_a/b). A projection the base has and the
        EXL3 checkpoint does not quantise stays what the base loaded, and is named."""
        from tools.exl3_linear import read_block
        from contextlib import ExitStack
        files = _exl3_files(path)
        found, added, kbits = 0, 0, {}
        with ExitStack() as stack:
            owners = {}
            for fpath in files:
                shard = stack.enter_context(safe_open(fpath, framework="pt", device=self.device))
                for key in shard.keys():
                    if key in owners:
                        raise ValueError(f"{key}: duplicate tensor in EXL3 shards")
                    owners[key] = shard
            f = _EXL3ShardView(owners)
            bases = sorted({k[: -len(".trellis")] for k in f.keys() if k.endswith(".trellis")})
            for base in bases:
                name = Layout.canonical(base)
                if name is None or name.startswith(Layout.mtp_prefix):
                    continue
                if name == Layout.head[: -len(".weight")]:
                    blk = read_block(f, base, out_fp32=True)
                    old = self.t.get(Layout.head)
                    if old is not None:
                        assert tuple(blk.shape) == tuple(old.shape), (blk.shape, old.shape)
                        was = 0 if old is self.t.get(Layout.embed) else old.numel() * old.element_size()
                        self.bytes_other -= was
                    self.t[Layout.head] = blk
                    self.bytes_other += blk.nbytes
                else:
                    blk = read_block(f, base)
                    self.q[name] = blk
                    self.bytes_fp8 += blk.nbytes
                added += blk.nbytes
                kbits[blk.kb] = kbits.get(blk.kb, 0) + 1
                found += 1
        if not found:
            raise RuntimeError(f"no EXL3 tensors found in {path}")
        from tools.exl3_linear import EXL3Block
        left = sorted(n for n, b in self.q.items() if not isinstance(b, EXL3Block))
        if left:
            print(f"[exl3] {len(left)} projections not in the EXL3 checkpoint keep the base's "
                  f"format: {left[:6]}{' ...' if len(left) > 6 else ''}")
        torch.cuda.empty_cache()
        self.exl3_source = path
        hist = ", ".join(f"K{k}: {v}" for k, v in sorted(kbits.items()))
        print(f"[exl3] {found} tensors from {path}: {added / 1e9:.2f} GB ({hist})")

    def fuse_exl3_groups(self, n_layers: int) -> int:
        """QWEN38_EXL3_FUSE (on by default with EXL3; 0 = separate launches): each projection group (PROJ_GROUPS) whose members share the rate
        becomes one EXL3 group (one rotation launch, one GEMM launch), and the members become
        views into it (`EXL3Block.member`), so nothing is held twice."""
        from tools.exl3_linear import EXL3Block
        made, fused = 0, 0
        for layer in range(n_layers):
            p = f"layers.{layer}"
            for key, members in PROJ_GROUPS:
                names = [f"{p}.{m}" for m in members]
                blocks = [self.q.get(n) for n in names]
                if not all(isinstance(b, EXL3Block) for b in blocks):
                    continue
                if len({b.kb for b in blocks}) != 1:
                    continue
                grp = EXL3Block.group(blocks, names)
                for i, n in enumerate(names):
                    self.q[n] = grp.member(i)
                self.g[f"{p}.{key}"] = grp
                fused += grp.nbytes
                made += 1
        torch.cuda.empty_cache()
        if made:
            print(f"[fuse] {made} EXL3 projection groups, {fused / 1e9:.2f} GB in one launch each")
        return made

    # ------------------------------------------------------------------ fp8 lm_head
    def load_fp8_head(self, path: str) -> None:
        """Swap the bf16 `lm_head` for the e4m3 one written by `tools/quant_head.py`.

        The bf16 tensor is dropped as the codes arrive: the head is the single largest tensor the
        engine holds and keeping both would cost 3.8 GB for nothing. Everything downstream reaches
        the head through `Weights.norm("lm_head.weight")`, and `engine.model.head_logits`
        dispatches on the type, so nothing else has to know.
        """
        from safetensors import safe_open
        from tools.head_gemv import FP8Head
        with safe_open(path, framework="pt", device=self.device) as f:
            head = FP8Head(f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
        old = self.t["lm_head.weight"]
        assert tuple(head.shape) == tuple(old.shape), (head.shape, old.shape)
        # a tied head is the embedding's tensor, which stays resident: nothing is freed
        was = 0 if old is self.t.get(Layout.embed) else old.numel() * old.element_size()
        self.t["lm_head.weight"] = head
        del old
        torch.cuda.empty_cache()
        self.bytes_other += head.nbytes - was
        self.fp8_head_source = path
        print(f"[fp8-head] {was / 1e9:.2f} GB bf16 -> {head.nbytes / 1e9:.2f} GB e4m3 "
              f"(per-row scales) from {path}")

    def build_fp8_head(self, ratios=None) -> None:
        """The e4m3 head quantised here, from the checkpoint's bf16 `lm_head`, at load.

        The same function `tools/quant_head.py build` runs (`quantize_head_fp8`), with the ratio set
        the served file was built with (`HEAD_BUILD_RATIOS`, read from that file's metadata:
        `1.0,0.95,0.90`, the per-row search). So `--fp8-head build` needs no artifact and gives
        the bytes the file holds -- checked on the board against ~/nvfp4/head-fp8.safetensors.
        The transient is one 8,192-row chunk in fp32 per ratio (~0.2 GB each); the bf16 tensor
        is dropped once the codes exist, as `load_fp8_head` does.
        """
        from tools.head_gemv import quantize_head_fp8
        ratios = tuple(ratios or HEAD_BUILD_RATIOS)
        old = self.t["lm_head.weight"]
        head = quantize_head_fp8(old, ratios=ratios)
        # a tied head is the embedding's tensor, which stays resident: nothing is freed
        was = 0 if old is self.t.get(Layout.embed) else old.numel() * old.element_size()
        self.t["lm_head.weight"] = head
        del old
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.bytes_other += head.nbytes - was
        self.fp8_head_source = "build:" + ",".join(f"{r:g}" for r in ratios)
        print(f"[fp8-head] {was / 1e9:.2f} GB bf16 -> {head.nbytes / 1e9:.2f} GB e4m3 "
              f"(per-row scales) built at load, ratios {ratios}")


def _exl3_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        import json
        with open(idx) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(path, n) for n in names]
    return sorted(glob_safetensors(path))


def exl3_bases(path: str) -> set[str]:
    """The engine names of the projections an EXL3 checkpoint quantises (from its index, else its
    files' keys): the base snapshot skips reading those."""
    keys: list[str] = []
    idx = os.path.join(path, "model.safetensors.index.json") if os.path.isdir(path) else None
    if idx and os.path.isfile(idx):
        import json
        with open(idx) as f:
            keys = list(json.load(f)["weight_map"])
    else:
        for fpath in _exl3_files(path):
            with safe_open(fpath, framework="pt", device="cpu") as f:
                keys.extend(f.keys())
    out = set()
    for k in keys:
        if k.endswith(".trellis"):
            name = Layout.canonical(k[: -len(".trellis")])
            if name and not name.startswith(Layout.mtp_prefix):
                out.add(name)                 # "lm_head" too: the base's bf16 head is not read
    return out


class _EXL3ShardView:
    """A projection's trellis, signs and codebook may live in different indexed shards."""
    def __init__(self, owners):
        self.owners = owners

    def keys(self):
        return self.owners.keys()

    def get_tensor(self, key):
        return self.owners[key].get_tensor(key)


def _stored_fp16_to_bf16(t: torch.Tensor, key: str) -> torch.Tensor:
    """Preserve finite stored values, not a claim to recover the pre-conversion BF16 source.

    Only the original and the resulting tensor are full-size; comparison temporaries are
    bounded even when a pack leaves its vocabulary head or embedding unquantised.
    """
    out = torch.empty(t.shape, dtype=torch.bfloat16, device=t.device)
    src, dst = t.reshape(-1), out.reshape(-1)
    for i in range(0, src.numel(), 1 << 20):
        part = src[i:i + (1 << 20)]
        if not torch.isfinite(part).all().item():
            raise ValueError(f"{key}: standalone EXL3 plain FP16 tensor contains non-finite values")
        bf = part.to(torch.bfloat16)
        if not torch.equal(part, bf.to(torch.float16)):
            raise ValueError(f"{key}: standalone EXL3 stored FP16 values cannot be converted to BF16 "
                             "losslessly; use the original BF16 checkpoint as the base and this "
                             "pack as the EXL3 overlay")
        dst[i:i + (1 << 20)].copy_(bf)
    return out


def _config_declares_bf16(snapshot: str) -> bool:
    import json
    try:
        with open(os.path.join(snapshot, "config.json")) as f:
            raw = json.load(f)
    except OSError:
        return False
    t = raw.get("text_config") or raw
    values = [t[k] for k in ("dtype", "torch_dtype") if t.get(k) is not None]
    if not values:
        values = [raw[k] for k in ("dtype", "torch_dtype") if raw.get(k) is not None]
    return bool(values) and all(isinstance(v, str) and
                               v.lower().removeprefix("torch.") in ("bfloat16", "bf16") for v in values)


def _config_exl3(snapshot: str) -> bool:
    import json
    try:
        with open(os.path.join(snapshot, "config.json")) as f:
            raw = json.load(f)
    except OSError:
        return False
    t = raw.get("text_config") or raw
    q = raw.get("quantization_config") or t.get("quantization_config") or {}
    return q.get("quant_method") == "exl3"


def _config_tied(snapshot: str) -> bool:
    import json
    try:
        with open(os.path.join(snapshot, "config.json")) as f:
            raw = json.load(f)
    except OSError:
        return False
    t = raw.get("text_config") or raw
    return bool(t.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)))


# The ratio set of the served head file (its safetensors metadata: "ratios": "1.0,0.95,0.90"):
# per row, the scale is searched over amax * r / 448 for these r on squared error.
HEAD_BUILD_RATIOS = (1.0, 0.95, 0.90)


def parse_head_build(spec) -> tuple[float, ...] | None:
    """`build` or `build:1.0,0.95` -> the ratio tuple (the served set for plain `build`); a path -> None."""
    if spec is None:
        return None
    s = str(spec).strip()
    if s == "build":
        return HEAD_BUILD_RATIOS
    if s.startswith("build:"):
        vals = tuple(float(x) for x in s[len("build:"):].split(",") if x.strip())
        if not vals or any(not (0.0 < v <= 1.0) for v in vals):
            raise ValueError(f"--fp8-head {s!r}: ratios must be in (0, 1]")
        return vals
    return None


def glob_safetensors(d: str) -> list[str]:
    import glob as _g
    return _g.glob(os.path.join(d, "*.safetensors"))
