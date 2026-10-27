"""The Qwen3.5 mixture-of-experts trunk (`qwen3_5_moe`) on the engine, shape-generic (ENG-383, ENG-391):
Ornith-1.5-122B-A10B from our EXL3 packs, and any checkpoint of the family (MiniCPM-V 4.7's 35B-A3B text
trunk has the same layer at another shape).

  config.py   the text config, validated: an unknown arithmetic field stops the load by name
  ref.py      the reference forward in torch, checked against `transformers` (Qwen3_5Moe)
  pack.py     an EXL3 pack's index: the trunk, the MTP layer, the vision tower
  load.py     a pack onto the board (dense projections, expert banks, the MTP layer)
  kernels.py  dense attention over a row's visible keys at any context length (verify twins)
  engine.py   the row step (decode, verify blocks up to 32 rows) and the prefill on the board's kernels
  serving.py  the server's engine interface, the MTP drafter, the registry's builder
"""
