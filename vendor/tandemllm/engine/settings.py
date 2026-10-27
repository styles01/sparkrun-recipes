"""Every QWEN38_* knob the engine reads, in one registry, read through one object.

The engine is configured by environment variables (`ops/serve.env` sets them). They used to be read
by ~124 `os.environ.get("QWEN38_...", default)` calls spread over 22 modules, each with its own copy
of the default. This module is now the only place a knob's default is written down, and
`SETTINGS.get(name)` is the only way the engine reads one.

What it does not change: WHEN a knob is read. `SETTINGS` views the live process environment, so a
module that read its knob at import still does (into the same module global) and a function that
read one per call still does -- every value, every parse, every default is the one the old call
produced, and `serve.env` keeps working as it is. `tests/test_settings.py` pins that: no engine
module reads a QWEN38_* variable any other way, and every registry default equals the default the
removed call carried.

What it adds:
  * one table (`KNOBS`: name -> (default, the module that reads it)) to document, audit and diff;
  * `EngineSettings(env=..., prefix=...)`: the same reads over another mapping or another prefix
    (a `QSE_` prefix for the extracted package), e.g. to print what a profile file sets
    (`describe()`) or to build a second configuration's values in a test;
  * `describe()`: every knob with its effective value and whether the environment set it.

The step after this one (not done here): modules that capture a knob into a global at import take
it from a settings object handed to the engine instead, so two engines in one process can differ.
"""

from __future__ import annotations

import os
from typing import Mapping

_MISSING = object()

# name (without the prefix) -> (default, first module that reads it). Generated from the reads it
# replaced on 2026-09-27; the defaults are the literal defaults those reads carried.
KNOBS: dict[str, tuple] = {
    'FP8_TILES': (None, 'tools/fp8_linear.py'),                 #
    # read under a computed name (engine/router.py tree_nodes, tools/nvfp4_skinny.py _table)
    'TREE_NODES': ('0', 'engine/router.py'),
    'TREE_NODES_NARROW': ('0', 'engine/router.py'),
    'SKINNY_TILES_B': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_C': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_WIDE_B': (None, 'tools/nvfp4_skinny.py'),
    'BLOCKING_SYNC': ('0', 'server/app.py'),
    'COMMIT_IN_VERIFY': ('0', 'engine/model.py'),
    'CORPUS': ('', 'engine/drafters/ngram.py'),
    'DECODE_ATTN': ('0', 'engine/model.py'),
    'DECIDER_COLLATE': ('call', 'server/decider_api.py'),
    'DECIDER_DEVICE': ('auto', 'server/decider_api.py'),
    'DECIDER_DIR': ('~/.qwen38-spark-engine/deciders/laya', 'server/decider_api.py'),
    'DECIDER_MAX_BATCH': ('64', 'server/decider_api.py'),
    'DECIDER_MAX_QUEUE': ('256', 'server/decider_api.py'),
    'DECIDER_MODELS': ('', 'server/decider_api.py'),
    'DECIDER_WAIT_MS': ('2', 'server/decider_api.py'),
    'DEEP': ('0', 'server/app.py'),
    'DEEP_AFTER': ('2', 'server/app.py'),
    'DF2_TEMP': ('1.0', 'engine/drafters/dflash2.py'),
    'DF2_TREE_MODE': ('paths', 'engine/drafters/dflash2.py'),
    'DFLASH2': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_FC_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_GRAPH': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_HEAD': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_HEAD_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_HEAD_EXL3': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_EXL3': (None, 'engine/drafters/dflash2.py'),
    'DRAFT_NVFP4': ('0', 'engine/drafters/dflash2.py'),
    'DRAFT_TEMP': ('1.0', 'engine/drafters/dflash2.py'),
    'DSPARK': (None, 'engine/drafters/dspark.py'),
    'DSPARK_NO_YARN': (None, 'engine/drafters/dspark.py'),
    'FP8_HEAD': (None, 'engine/loader.py'),
    'FUSED_ADDNORM': ('0', 'engine/model.py'),
    'FUSED_ATTN': ('0', 'engine/model.py'),
    'FUSED_ATTN_PREP': ('0', 'engine/model.py'),
    'FUSED_COMMIT': ('0', 'engine/model.py'),
    'FUSED_GDN': ('1', 'engine/model.py'),
    'FUSED_GDNBLOCK': ('1', 'engine/model.py'),
    'FUSED_GDNPRE': ('1', 'engine/model.py'),
    'FUSED_GDNPREFILL': ('0', 'engine/model.py'),
    'FUSED_GDNTREE': ('1', 'engine/model.py'),
    'FUSED_GDNVERIFY': ('0', 'engine/model.py'),
    'FUSED_HEAD': ('1', 'engine/model.py'),
    'FUSED_NORM': ('1', 'engine/model.py'),
    'FUSE_PROJ': ('0', 'tools/nvfp4_linear_v2.py'),
    'GDNV_BV': ('16', 'tools/gdn_verify_kernels.py'),
    'GDNV_CONV_BLOCK': ('256', 'tools/gdn_verify_kernels.py'),
    'GDNV_CONV_WARPS': ('4', 'tools/gdn_verify_kernels.py'),
    'GDNV_ONE_WARP': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_TREE_PF': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WARPS': ('4', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_CHAIN_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_FUSED': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_FUSED_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_KC': ('0', 'tools/gdn_verify_kernels.py'),
    'GDNV_WY_MAXT': ('32', 'tools/gdn_verify_kernels.py'),
    'GDN_AB': ('0', 'engine/model.py'),
    'GDN_CHUNK': ('64', 'engine/model.py'),
    'GDN_PREFILL_NOREP': ('0', 'engine/model.py'),
    'ATTN_STAGES3': ('1', 'tools/attn_kernels.py'),
    'GDN_MM': ('fp32', 'engine/gdn.py'),
    'GDN_MM_FROM': ('64', 'engine/gdn.py'),
    'GDN_PREFILL_BV': ('32', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC': ('bf16x3', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC_A': ('tf32', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_PREC_SCAN': ('', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_SERIES': ('0', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_STAGES': ('1', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_WARPS_INTRA': ('8', 'tools/gdn_prefill_kernels.py'),
    'GDN_PREFILL_WARPS_SCAN': ('8', 'tools/gdn_prefill_kernels.py'),
    'GQA_FROM': ('64', 'engine/model.py'),
    'GRAPH_MAX_CTX': ('32768', 'engine/verify_graph.py'),
    'HEAD_GEMM': ('', 'tools/head_gemv.py'),
    'HOST_ASYNC': ('0', 'engine/model.py'),
    'HOST_WALK': ('0', 'engine/drafters/dflash2.py'),
    'KV_FP8': ('0', 'engine/model.py'),
    'LAUNCH_FIRST': ('0', 'server/app.py'),
    'LOWER_RIGHT_FROM': ('64', 'engine/model.py'),
    'MEMORY_WRITER_TEXT': ('', 'server/app.py'),                # a test engine's file of the memory writer's turns
    'MODEL': (None, 'engine/config.py'),
    'NARRATOR_DEVICE': ('auto', 'server/conductor/narrator.py'),
    # empty: ~/.qwen38-spark-engine/models/<QWEN38_NARRATOR_MODEL> (N2)
    'NARRATOR_DIR': ('', 'server/conductor/narrator.py'),
    'NARRATOR_MODEL': ('', 'server/conductor/narrator.py'),
    # SRV-191: the served model's name in the identity note and /v1/models ('' = Qwen3.8-27B)
    'MODEL_NAME': ('', 'server/conductor/schema.py'),
    # SRV-193: where the reasoning goes for a client kind that names no reasoning_format, when the
    # server's format is `tags` (kind=format,...; `*` any other kind; '' = tags for every client).
    # SRV-245 (2026-10-07): every client kind under a conductor profile gets `reasoning_content`
    # (an OpenAI SDK printed the model's thinking as its answer); the raw path keeps the server's.
    # Hermes keeps `tags`: its lines wait in the content for the block to close (S1)
    'CLIENT_REASONING': ('*=reasoning_content,hermes=tags', 'server/compat.py'),
    # SRV-237/249: the client kinds whose state lines before the first generated token (the
    # router's, the waiting and reading lines) go out as Open WebUI `status` events, outside the
    # reasoning it times as "Thought for N seconds" ('' = in the profile's channel, as before)
    'STATUS_CLIENTS': ('open-webui', 'server/conductor/states.py'),
    # SRV-197: which test says a request continues slot 0's conversation with --parallel N:
    # `view` (the client's rendered tokens, then the resident rows) or `resident` (the rows alone,
    # the old rule, for an A/B)
    'SIDE_CONTINUE': ('view', 'server/app.py'),
    # FX5 (2026-10-06): a short request unrelated to slot 0's conversation (the router's own
    # questions, a client's title or tags task) takes an idle side slot first; off: slot 0 as a
    # guest with the stash, as before
    'SIDE_GUESTS': ('on', 'server/app.py'),
    # A request continues slot 0's conversation only when the part it shares reaches
    # past its first user message (`on`); `off`: a client's fixed head alone may count
    'SIDE_CONV': ('on', 'server/app.py'),
    # A side slot reads the disk tier (`read`: a new chat's head from the NVMe, writes
    # stay slot 0's); `off`: a side slot prefills its whole prompt
    'SIDE_DISK': ('read', 'server/app.py'),
    # FX10 (0.4.1): a parked conversation a new park gives up is written to the NVMe spill first
    # (`on`); `off` gives it up as 0.4.0 did
    'SPILL_PARK': ('on', 'engine/cache.py'),
    # FX10 (0.4.1, SRV-213): the next turn of a conversation that ran on a side slot is carried
    # into slot 0 from the side slot's rows (`on`); `off` re-reads it on slot 0 as 0.4.0 did
    'SIDE_CARRY': ('on', 'server/app.py'),
    # FX16 (0.4.8): two chats beside each other. The least output a side slot's window must
    # leave a request (its max_tokens is cut to the room), whether a new conversation takes an
    # idle side slot before slot 0 when slot 0's conversation could never run there, how long a
    # turn waits for the side slot that holds its rows, and the side slot's own resident prefix
    # (anchors GB, stash GB; 0 builds none)
    'SIDE_MIN_ROOM': ('16384', 'server/app.py'),
    'SIDE_NEW_FIRST': ('on', 'server/app.py'),
    'SIDE_HOME_WAIT_S': ('10', 'server/app.py'),
    'SIDE_RESIDENT_GB': ('0.5', 'server/app.py'),
    'SIDE_STASH_GB': ('1.0', 'server/app.py'),
    # FX16: the end alignment's third pass (a block the client moved from an earlier message to
    # its last one is served where the model saw it), and the prompt length from which a turn
    # the end anchor did not serve says why in the log
    'END_ALIGN_MOVED': ('on', 'server/endalign.py'),
    # FX17: a router note stays live when its user message still ends with the same tokens
    'NOTE_ANCHOR': ('on', 'server/conductor/router.py'),
    # FX17: a short new chat waits this long for a busy side slot before it takes slot 0 and
    # evicts a long resident conversation
    'SIDE_NEW_WAIT_S': ('20', 'server/app.py'),
    # 0.8.2: a request whose conversation slot 0 holds (or reads back from the NVMe) waits this
    # long for slot 0, then takes a free side slot it fits in
    'SIDE_SLOT0_WAIT_S': ('3', 'server/app.py'),
    # SRV-629: a request with images may take a side slot (`on`), routed on its expanded key ids;
    # `off`: slot 0 only, as 0.12.0
    'SIDE_IMAGES': ('on', 'server/app.py'),
    # SRV-528: how long slot 0's conversation keeps new conversations on the side slots after the
    # last request that continued it (a cancelled prefill's rows: not at all)
    'SLOT0_KEEP_S': ('1800', 'server/app.py'),
    # FX17: the warm-up prefill at start (tokens; 0 skips)
    'WARM_PREFILL_TOKENS': ('2600', 'server/app.py'),
    'REREAD_SAY_TOKENS': ('8192', 'server/app.py'),
    'MM_END': ('on', 'server/app.py'),
    # SRV-225: a client that sends no conversation id (Open WebUI): chats that open
    # with the same message are separate conductor sessions (`on`); `off`: one session per head
    'SESSION_GENERATIONS': ('on', 'server/conductor/router.py'),
    # FX5: Open WebUI's live working directory moves from run_command's description (the system
    # prompt) into a router note (`note`), so a `cd` no longer re-reads the conversation; `off`
    'CLIENT_CWD': ('note', 'server/conductor/router.py'),
    # FX5b (2026-10-06): a class's note carries only the parts (effort sentence, packages, rules)
    # no live note of the session holds with the same text (`parts`); `block`: the whole note
    # whenever any part changed, as before (the general packages went in again at every
    # user message whose task skill differed, and the replay budget then cut the oldest note)
    'NOTE_DEDUP': ('parts', 'server/conductor/router.py'),
    # SRV-243 (2026-10-07): a class's packages whose steps need a tool the request does not offer
    # (a shell, a file write) stay out of the turn, their `fallback` goes in, and the note says
    # which tools the chat has (`on`); `off` keeps the packages whatever the tools
    'TOOL_FIT': ('on', 'server/conductor/router.py'),
    # ENG-E10: the router's fit to the served model (server/conductor/modelfit.py): model, off, k=v,...
    'CONDUCTOR_FIT': ('model', 'server/conductor/modelfit.py'),
    # SRV-239 (2026-10-07): the end alignment's second pass, for a copy that left an answer's
    # reasoning out (Open WebUI sends none back): `on` serves the rows' reasoning and resumes at
    # the end anchor; `off` re-reads from the grid line below the answer
    'END_ALIGN_REASONING': ('on', 'server/endalign.py'),
    # SR1 (2026-10-04): the shadow reader (server/conductor/shadow.py, engine/shadow.py)
    'SHADOW_MODEL': ('', 'server/conductor/shadow.py'),
    'SHADOW_DIR': ('', 'server/conductor/shadow.py'),
    'SHADOW_GB': ('4', 'server/conductor/shadow.py'),
    'SHADOW_GRID': ('512', 'server/conductor/shadow.py'),
    'SHADOW_DISK': ('', 'server/conductor/shadow.py'),
    'SHADOW_DISK_GB': ('20', 'server/conductor/shadow.py'),
    'SHADOW_POOL_SLACK_GIB': ('2', 'server/conductor/shadow.py'),
    'NVFP4': (None, 'engine/loader.py'),
    # ENG-E6: an EXL3 checkpoint (exllamav3 trellis) for every projection and the head; unset = off
    'EXL3': (None, 'engine/loader.py'),
    'EXL3_TILES': (None, 'tools/exl3_linear.py'),
    'EXL3_CHUNK_UNTIL': ('128', 'tools/exl3_linear.py'),
    'EXL3_ROWGRID': ('1', 'tools/exl3_linear.py'),
    'EXL3_FUSE': ('1', 'engine/loader.py'),
    'EXL3_SILU_FUSED': ('0', 'tools/exl3_linear.py'),
    'EXL3_MLP_BRIDGE': ('1', 'tools/exl3_linear.py'),
    'EXL3_STAGES': ('3', 'tools/exl3_linear.py'),
    'SAMPLE_TOPP_K': ('1024', 'engine/sample.py'),
    'SAMPLE_BATCH': ('1', 'engine/sample.py'),
    'SAMPLE_NUCLEUS': ('1', 'engine/sample.py'),
    'SAMPLED_CUT': ('0', 'engine/router.py'),
    # the layer-major prefill (engine/model.py `prefill_window`): rows a window takes when its chunk
    # lines hand out anchors/snapshots (each line holds a recurrent-state copy), and when none do;
    # 0 = chunk-major as before. Bit for bit the chunk-major forwards either way.
    'PREFILL_WINDOW': ('8192', 'engine/cache.py'),
    'PREFILL_WINDOW_FREE': ('32768', 'engine/cache.py'),
    'NVFP4_BLOCK_TILES': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_DEQUANT_FROM': ('512', 'tools/nvfp4_linear.py'),
    'NVFP4_PREFILL_V2': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_PREFILL_V2_UNTIL': ('1024', 'tools/nvfp4_linear.py'),
    'NVFP4_SKINNY': ('0', 'tools/nvfp4_skinny.py'),
    'NVFP4_SPLITK': ('0', 'tools/nvfp4_linear.py'),
    'NVFP4_V2': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_BN': ('0', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_DOTS': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_MAX': ('32', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_MIN': ('1', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_PREFETCH': ('0', 'tools/nvfp4_linear_v2.py'),
    'NVFP4_V2_W': ('0', 'tools/nvfp4_linear_v2.py'),
    'PREFILL_CAUSAL': ('1', 'engine/model.py'),
    'PROBE_DAY_MB': ('100', 'server/probe.py'),
    'PROBE_DIR': ('~/.qwen38-spark-engine/probe', 'server/probe.py'),
    'PROBE_EVERY': ('1024', 'server/probe.py'),
    'PROBE_SNAPSHOTS': ('off', 'server/probe.py'),
    'RANKK': ('1', 'engine/model.py'),
    'SCALE_ON': ('weight', 'tools/fp8_linear.py'),
    'SKINNY_LDW': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_PDL': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_SRUN': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_SSTUB': ('0', 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_TILES_WIDE': (None, 'tools/nvfp4_skinny.py'),
    'SKINNY_XSTUB': ('0', 'tools/nvfp4_skinny.py'),
    'SUFFIX_STORE': ('~/.qwen38-spark-engine/suffix', 'server/app.py'),
    'TREE_ALIAS_STATE': ('0', 'engine/model.py'),
    'TREE_CHAIN_DELEGATE': ('1', 'engine/model.py'),
    'TREE_HOST_DEPTH': ('0', 'engine/model.py'),
    'TREE_MS': ('', 'engine/router.py'),
    'TREE_WIDE_AFTER': ('0', 'engine/lenrouter.py'),
    'LATCH_PRICE': ('0', 'engine/lenrouter.py'),
    'LEN_SWITCH': ('0', 'engine/lenrouter.py'),
    'LEN_MODE': ('', 'engine/lenrouter.py'),
    'STAIR_TABLES': ('', 'engine/lenrouter.py'),
    'STAIR_RHO': ('', 'engine/lenrouter.py'),
    'STAIR_SKIP': ('0', 'engine/lenrouter.py'),
    'STAIR_MAX_NODES': ('', 'engine/lenrouter.py'),
    'TWO_STREAM': ('0', 'engine/model.py'),
    'UT_INVERSE': ('1', 'engine/gdn.py'),
    'VERIFY_GRAPH': ('0', 'engine/model.py'),
    'VERIFY_ROWS': ('16', 'engine/model.py'),
    'VERIFY_TILES': ('1', 'tools/nvfp4_verify_tiles.py'),
    'WY_BLOCK': ('8', 'tools/gdn_wy_kernels.py'),
    'WY_BV': ('32', 'tools/gdn_wy_kernels.py'),
    'WY_DBG': ('0', 'tools/gdn_wy_kernels.py'),
    'WY_PREC': ('ieee', 'tools/gdn_wy_kernels.py'),
    'WY_WARPS': ('4', 'tools/gdn_wy_kernels.py'),
    'WY_WARPS_PREP': ('4', 'tools/gdn_wy_kernels.py'),
    # SRV-241 (0.4.4-test): the regenerate rule (server/regenerate.py): `sample` = a regenerate of
    # an answered conversation under a TandemLLM plan samples with a fresh seed, `off` = no
    # detection; the defaults of --regenerate and its sampling flags (docs/server.md, Regenerate)
    'REGENERATE': ('sample', 'server/app.py'),
    'REGEN_TEMPERATURE': ('0.7', 'server/app.py'),
    'REGEN_TOP_P': ('0.95', 'server/app.py'),
    'REGEN_TOP_K': ('20', 'server/app.py'),
    # 2026-10-05: addresses besides loopback whose X-Tandem-User header names whose memory a
    # request reads and writes (comma-separated; '' = loopback only)
    'MEMORY_PROXIES': ('', 'server/app.py'),
    # SRV-163: the address `{tandem_url}` in a package becomes ('' = the request's own address)
    'TANDEM_URL': ('', 'server/conductor/packages.py'),
    # SRV-171: the session's scratch directory the notes name (`{token}` = the session token;
    # '' = `.tandem` in the project)
    'TANDEM_SCRATCH': ('', 'server/conductor/packages.py'),
    # ENG-E7: Qwen3.8-Flash-Next on turboderp's EXL3 packs (engine/models/flashnext/). The 27B's
    # disk-tier fingerprint leaves knobs owned under engine/models/ and tools/fn_ out (engine/disk_cache.py).
    # The hyper-connection mixers' served width: int8 rows with an fp32 scale, or the pack's fp16
    'FN_MIX': ('int8', 'engine/models/flashnext/pack.py'),
    # Flash-Next kernels launched with programmatic dependent launch (overlap a launch with its
    # predecessor's tail; the bits do not change)
    'FN_PDL': ('1', 'tools/fn_kernels.py'),
    # cp.async stages of a warp in the MoE kernels (2..4; the bits do not depend on it)
    'FN_MOE_ST': ('3', 'tools/fn_kernels.py'),
    # warps a MoE CTA splits its K range over (4 or 8; sets the order of the partial sums)
    'FN_MOE_WARPS': ('4', 'tools/fn_kernels.py'),
    # EXL3 projections of up to 32 rows on tools/fn_kernels.dense (input rotation fused, PDL) instead
    # of tools/exl3_linear's kernel; 2026-10-11 paired A/B at 1k: no difference (45.7 vs 45.4 prose)
    'FN_DENSE': ('0', 'engine/models/flashnext/engine.py'),
    # read the n-gram table into the page cache at load (POSIX_FADV_WILLNEED, in the background)
    'FN_WARM_TABLE': ('1', 'engine/models/flashnext/load.py'),
    # drop the shards' page cache after the load (the table's is kept); 0 = keep it (a warm reload)
    'FN_DROP_SHARD_CACHE': ('1', 'engine/models/flashnext/load.py'),
    # ENG-239: the n-gram table's rows from the page cache (warm, counted by every floor check) or read
    # with O_DIRECT through a hot-row cache of FN_TABLE_CACHE_GB (cold, never in the page cache)
    'FN_TABLE': ('warm', 'engine/models/flashnext/load.py'),
    'FN_TABLE_CACHE_GB': ('2', 'engine/models/flashnext/load.py'),
    # ENG-227: a speculative round's commit parks the accepted path's GDN rows and the next verify
    # block or decode step applies them as it loads each state tile (one read of the GDN state less
    # a round; the same arithmetic); 0 = the commit walks them itself
    'FN_FOLD_COMMIT': ('1', 'engine/models/flashnext/engine.py'),
    # ENG-225: a prompt chunk (and the MTP layer's catch-up) through the prefill kernels
    # (tools/fn_prefill_kernels.py: experts grouped by expert, mixers and the router on the library
    # GEMM, attention over each row's visible keys); "ref" = the reference's batched forward
    'FN_PREFILL': ('kernels', 'engine/models/flashnext/engine.py'),
    # rows a prompt chunk of the prefill kernels takes (the reference keeps 2,048)
    'FN_PREFILL_CHUNK': ('4096', 'engine/models/flashnext/engine.py'),
    # the prefill's mixers, router and small projections: "fp32" = exact weights and fp32 sums, "fp16" =
    # weights rounded to fp16 on the tensor cores
    'FN_PREFILL_MM': ('fp32', 'tools/fn_prefill_kernels.py'),
    # GEMVs of few output rows (mixer down, router, GDN a/b) with each row's K split over the warps of a
    # CTA (more loads in flight); sets their summation order, so a build keeps one value
    'FN_GEMV_SPLIT': ('1', 'tools/fn_kernels.py'),
    # SRV-87: the served Flash-Next's MTP draft head over the first N ids of ops/fn-hot-vocab.npy
    # (int8 rows; 0 = the full EXL3 head)
    'FN_DRAFT_HOT': ('32768', 'engine/models/flashnext/serving.py'),
    # ENG-229: the lookup (the request's text, --corpus and the server's suffix store) in the MTP round's
    # cut, one tree with the head's chain; 0 = the head alone
    'FN_LOOKUP': ('0', 'engine/models/flashnext/serving.py'),
    # ENG-391: the lookup in the Qwen3.5 MoE MTP round's cut (one tree with the MTP chain); 0 = the head alone
    'Q35M_LOOKUP': ('1', 'engine/models/qwen3_5_moe/serving.py'),
    # the DFlash2 head's projections in NVFP4 / its head over the hot vocabulary (drafter only: acceptance, not text)
    'Q35M_DFLASH2_NVFP4': ('0', 'engine/models/qwen3_5_moe/dflash2.py'),
    'Q35M_DFLASH2_HOT': ('0', 'engine/models/qwen3_5_moe/dflash2.py'),
    # the cut's second-choice leaf a depth (a tree block) for the Qwen3.5 MoE drafters
    'Q35M_SIBLINGS': ('0', 'engine/models/qwen3_5_moe/serving.py'),
    # rounds after an anchor-only cut that take the plain step (1, 2, 4 .. up to this; 0 = off)
    'Q35M_ANCHOR_BACKOFF': ('0', 'engine/models/qwen3_5_moe/serving.py'),
}


class EngineSettings:
    """The engine's knobs over an environment mapping (the live `os.environ` by default)."""

    def __init__(self, env: Mapping[str, str] | None = None, prefix: str = "QWEN38_"):
        self._env = env
        self.prefix = prefix

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def get(self, name: str, default=_MISSING):
        """The raw string the environment holds for `name`, else its registry default (or the
        explicit `default`, for the one read whose default is a module constant)."""
        if default is _MISSING:
            if name not in KNOBS:
                raise KeyError(f"{self.prefix}{name} is not a registered knob (engine/settings.py KNOBS)")
            default = KNOBS[name][0]
        return self.env.get(self.prefix + name, default)

    def is_set(self, name: str) -> bool:
        return (self.prefix + name) in self.env

    def describe(self) -> dict[str, dict]:
        return {k: {"value": self.get(k), "default": d, "set": self.is_set(k), "read_in": where}
                for k, (d, where) in sorted(KNOBS.items())}


#: The process's settings: the live environment, prefix QWEN38_.
SETTINGS = EngineSettings()
