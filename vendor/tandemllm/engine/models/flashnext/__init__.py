"""Qwen3.8-Flash-Next (`qwen4_exp`) on the engine, from turboderp's EXL3 packs (ENG-E7).

  config.py   the text config, validated (ENG-133)
  pack.py     the pack: index, tensor kinds and K, byte budget, sha256, readers (ENG-219)
  ple.py      the PLE n-gram hash and the memory-mapped EXL3 n-gram table (ENG-145)
  ref.py      the reference forward in torch, checked against `transformers` (ENG-135)
  engine.py   the forward on the board's kernels, decode as a CUDA graph (ENG-221)

Ported from branch `flashnext` (20a82991) onto main; see the ENG-221 Memo note for what came from
which commit.
"""
