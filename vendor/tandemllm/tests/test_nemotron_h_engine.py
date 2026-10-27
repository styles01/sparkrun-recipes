"""The NemotronH board engine on tiny BF16 models (with and without the latent pair, per-layer top k):
the row step against the reference forward, a verify block's rows against the decode step's (bit for
bit), commit and rollback (the SSM and conv states equal the decode steps', bit for bit), joint passes,
the chunked prefill against the row path, the load-time self-check. On the GPU, or on the CPU under
TRITON_INTERPRET=1 (whose bf16 casts truncate: the reference comparison is then looser; the engine
against itself is bit for bit either way)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import kernels as K  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import ExpertsRelu2  # noqa: E402
from engine.models.nemotron_h.ref import NemotronHRef, chain  # noqa: E402
from tests.test_nemotron_h_ref import PUZZLE, TINY, random_weights  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
# the row kernels' shapes: hidden, expert widths and the latent a multiple of 32, head_dim a power of two
SHAPE = dict(vocab_size=256, hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
             mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=24, n_groups=2, moe_shared_expert_intermediate_size=64,
             chunk_size=8)
TINY_B = {**TINY, **SHAPE, "n_routed_experts": 4, "moe_intermediate_size": 64}
PUZZLE_B = {**PUZZLE, **SHAPE, "moe_latent_size": 64,
            "block_configs": [dict(b, moe_intermediate_size=32 * (i % 3 + 1)) if b["block_type"] == "moe" else b
                              for i, b in enumerate(PUZZLE["block_configs"])]}
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88, 101, 7, 9, 200, 31])

_ENG: dict = {}


def tiny(name: str = "tiny"):
    if name in _ENG:
        return _ENG[name]
    from engine.models.nemotron_h.engine import NemotronHEngine
    raw = TINY_B if name == "tiny" else PUZZLE_B
    cfg = config_from_dict(raw, name)
    W0 = random_weights(cfg)
    f32 = ("e_score_correction_bias", "A_log", "dt_bias", ".D")
    W = {k: (v.float() if k.endswith(f32) else v.to(torch.bfloat16)).to(DEV) for k, v in W0.items()
         if ".experts." not in k or "shared_experts" in k}
    experts = {}
    for li in cfg.moe_layers:
        m = f"layers.{li}.mixer.experts."
        E = cfg.moe_experts[li]
        up = torch.stack([W0[m + f"{e}.up_proj.weight"] for e in range(E)]).to(torch.bfloat16).to(DEV).contiguous()
        down = torch.stack([W0[m + f"{e}.down_proj.weight"] for e in range(E)]).to(torch.bfloat16).to(DEV).contiguous()
        experts[li] = ExpertsRelu2(up, down)
    for li in cfg.attention_layers:
        m = f"layers.{li}.mixer."
        W[m + "qkv"] = torch.cat([W[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
    eng = NemotronHEngine(cfg, W, experts, device=DEV, max_len=256)
    ref = NemotronHRef(cfg, {k: v.to(torch.bfloat16).float() if not k.endswith(f32) else v.float()
                             for k, v in W0.items()})
    _ENG[name] = (cfg, eng, ref)
    return _ENG[name]


def _decode(eng, st, toks):
    out = []
    for t in toks:
        eng.stage(t, st)
        lg = eng.step(st).clone()
        st.length += 1
        out.append(lg[0])
    return torch.stack(out)


def _snap(st):
    return [t.clone() for t in st.conv.values()] + [t.clone() for t in st.ssm.values()]


def _put(st, snap):
    for t, s in zip(list(st.conv.values()) + list(st.ssm.values()), snap):
        t.copy_(s)


def _same_state(a, b):
    return all(torch.equal(x, y) for x, y in zip(_snap(a), _snap(b)))


def test_prefill_and_decode_near_the_reference():
    if SKIP:
        return SKIP
    tol = 0.05 if DEV == "cuda" else 0.3
    for name in ("tiny", "puzzle"):
        cfg, eng, ref = tiny(name)
        st = eng.new_state(256)
        lg = eng.prefill(PROMPT, st)
        rs = ref.new_state(64)
        want = ref.prefill(PROMPT, rs)[-1]
        assert (lg[0].cpu() - want).abs().max() < tol * want.abs().max(), (name, (lg[0].cpu() - want).abs().max())
        got = _decode(eng, st, [11, 50, 3]).cpu()
        for i, t in enumerate((11, 50, 3)):
            w = ref.decode(t, rs)[0]
            assert (got[i] - w).abs().max() < tol * w.abs().max(), (name, i)


def test_chunked_prefill_agrees_with_the_row_path():
    """The prompt through the chunked SSD form (chunk 8: the 19 rows cross three chunks) and through
    the row path (one row at a time) land on the same next token and near the same logits and state."""
    if SKIP:
        return SKIP
    tol = 0.05 if DEV == "cuda" else 0.25         # the interpreter's truncating casts compound over 19 rows
    for name in ("tiny", "puzzle"):
        cfg, eng, ref = tiny(name)
        a = eng.new_state(256)
        la = eng.prefill(PROMPT, a)[0]
        b = eng.new_state(256)
        eng.prefill(PROMPT[:1], b)
        lb = _decode(eng, b, PROMPT[1:].tolist())[-1]
        assert int(la.argmax()) == int(lb.argmax()), name
        assert (la - lb).abs().max() < tol * lb.abs().max(), (name, (la - lb).abs().max())
        for li in a.ssm:
            err = (a.ssm[li] - b.ssm[li]).abs().max() / (b.ssm[li].abs().max() + 1e-3)
            assert err < tol, (name, li, float(err))


def test_verify_rows_are_the_decode_steps():
    if SKIP:
        return SKIP
    for name in ("tiny", "puzzle"):
        cfg, eng, ref = tiny(name)
        st = eng.new_state(256)
        eng.prefill(PROMPT, st)
        L = st.length
        s0 = _snap(st)
        toks = [7, 19, 23, 44, 31, 60]
        want = _decode(eng, st, toks)
        st.length = L
        _put(st, s0)
        eng.stage_rows(toks, chain(len(toks)), st)
        got = eng.rows_step(st, len(toks))
        assert torch.equal(got, want), name
        # a tree: rows 0 -> {1 -> 2, 3}; row 3's path is (7, 19, 44)
        eng.stage_rows([7, 19, 23, 44], [-1, 0, 1, 1], st)
        tree = eng.rows_step(st, 4).clone()
        assert torch.equal(tree[:3], want[:3]), name
        s2 = eng.new_state(256)
        eng.prefill(PROMPT, s2)
        assert torch.equal(_decode(eng, s2, [7, 19, 44])[2], tree[3]), name


def test_verify_widths_share_attention_partials():
    """ENG-363: the verify widths of a power-of-two bucket share one set of attention partials per slot
    and class (5 to 8 rows: one set sized for 8), and each width's rows are still the decode steps'
    bits, a smaller block right after a larger one in the same set included."""
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    st = eng.new_state(256)
    eng.prefill(PROMPT, st)
    L, s0 = st.length, _snap(st)
    toks = [7, 19, 23, 44, 31, 60, 5, 9]
    want = _decode(eng, st, toks)
    for M in (8, 5, 3, 4, 6):
        st.length = L
        _put(st, s0)
        eng.stage_rows(toks[:M], chain(M), st)
        got = eng.rows_step(st, M)
        assert torch.equal(got[:M], want[:M]), M
    ns = K.class_of(L + 8, st.cap)
    assert eng._attn_bufs(None, 5, ns) is eng._attn_bufs(None, 8, ns)
    assert eng._attn_bufs(None, 3, ns) is eng._attn_bufs(None, 4, ns)
    assert eng._attn_bufs(None, 4, ns) is not eng._attn_bufs(None, 5, ns)
    assert eng._attn_bufs("s1", 8, ns) is not eng._attn_bufs(None, 8, ns)   # a side slot keeps its own
    b8 = eng._attn_bufs(None, 8, ns)
    assert b8["pacc"].shape[0] == 8 * cfg.num_key_value_heads * ns * 16
    assert eng._attn_bufs(None, 33, ns)["rows"] == 33 and eng._attn_bufs(None, 20, ns)["rows"] == 32
    return f"widths 8, 5, 3, 4, 6 from {len([k for k in eng._attn if k[0] is None and k[1] == ns])} sets"


def test_commit_a_moved_path_and_a_rollback():
    if SKIP:
        return SKIP
    for name in ("tiny", "puzzle"):
        cfg, eng, ref = tiny(name)
        st = eng.new_state(256)
        eng.prefill(PROMPT, st)
        alone = eng.new_state(256)
        eng.prefill(PROMPT, alone)
        eng.verify([7, 19, 23, 44], [-1, 0, 1, 1], st)
        eng.commit(st, [0, 1, 3])
        _decode(eng, alone, [7, 19, 44])
        assert st.length == alone.length and _same_state(st, alone), name
        assert torch.equal(_decode(eng, st, [5]), _decode(eng, alone, [5])), name
        eng.verify([9, 10, 11, 12], chain(4), st)
        eng.commit(st, [0, 1])
        _decode(eng, alone, [9, 10])
        assert _same_state(st, alone), name
        assert torch.equal(_decode(eng, st, [60]), _decode(eng, alone, [60])), name
        eng.verify([9, 10, 11], chain(3), st)
        eng.commit(st, [])                                   # nothing kept: the state stays
        assert torch.equal(_decode(eng, st, [61]), _decode(eng, alone, [61])), name


def test_a_joint_pass_is_each_states_lone_verify():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    a, b = eng.new_state(256), eng.new_state(256)
    eng.prefill(PROMPT, a)
    eng.prefill(PROMPT[:9], b)
    blocks = [([7, 19, 23, 44], [-1, 0, 1, 1]), ([5, 6, 7], [-1, 0, 1])]
    lone = []
    for (toks, par), st in zip(blocks, (a, b)):
        eng.verify(toks, par, st)
        lone.append(eng._rb[len(toks)]["logits"][:len(toks)].clone())
    rbs = eng.verify_multi([(toks, par, st) for (toks, par), st in zip(blocks, (a, b))])
    for rb, want, (toks, _) in zip(rbs, lone, blocks):
        assert torch.equal(rb["logits"][:len(toks)], want)
    eng.commit(a, [0, 1, 3])
    eng.commit(b, [0, 1])
    alone_a, alone_b = eng.new_state(256), eng.new_state(256)
    eng.prefill(PROMPT, alone_a)
    eng.prefill(PROMPT[:9], alone_b)
    _decode(eng, alone_a, [7, 19, 44])
    _decode(eng, alone_b, [5, 6])
    assert torch.equal(_decode(eng, a, [3]), _decode(eng, alone_a, [3]))
    assert torch.equal(_decode(eng, b, [3]), _decode(eng, alone_b, [3]))
    return "7 rows of 2 states: each its lone bits; each commit its own block"


def test_a_joint_decode_step_is_each_states_own():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    states, alone = [eng.new_state(256) for _ in range(3)], [eng.new_state(256) for _ in range(3)]
    for st, al, n in zip(states, alone, (14, 9, 5)):
        eng.prefill(PROMPT[:n], st)
        eng.prefill(PROMPT[:n], al)
    for tok in (7, 19):
        rbs = eng.decode_multi([(tok, st, f"s{j}") for j, st in enumerate(states)])
        for j, (st, al, rb) in enumerate(zip(states, alone, rbs)):
            want = _decode(eng, al, [tok])
            assert torch.equal(rb["logits"][:1], want), j
            st.length += 1
    for st, al in zip(states, alone):
        assert st.length == al.length and _same_state(st, al)
    return "3 states, 2 steps: each its own decode step's bits and state"


def test_graphs_replay_the_eager_bits():
    """The decode step, an M-row verify and a commit captured as CUDA graphs give the eager bits and
    states (GPU only)."""
    if SKIP or DEV != "cuda":
        return SKIP or "skipped: CUDA graphs need the GPU"
    from engine.models.nemotron_h.engine import DecodeGraph
    cfg, eng, ref = tiny()
    a, b = eng.new_state(256), eng.new_state(256)
    eng.prefill(PROMPT, a)
    eng.prefill(PROMPT, b)
    dg = DecodeGraph(eng, a)
    for t in (7, 19, 44):
        ga = dg(t).clone()
        assert torch.equal(ga[0], _decode(eng, b, [t])[0])
    assert _same_state(a, b)
    vg, cg = {}, {}
    for _ in range(2):                                   # the second round replays the captured graphs
        ta = eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], a, graphs=vg)
        tb = eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], b)
        assert ta == tb
        eng.commit(a, [0, 1, 3], graphs=cg)
        eng.commit(b, [0, 1, 3])
        assert a.length == b.length and _same_state(a, b)
    return f"{len(vg)} verify, {len(cg)} commit graphs"


def test_joint_decode_graphs_are_kept_by_recency_and_recaptured_with_the_same_bits():
    """ENG-363: the joint decode graphs live in a bounded cache (engine/models/graph_lru.py); with room
    for one, three sets of slots in turn drop and capture again, every row its state's lone decode
    step's bits (GPU only)."""
    if SKIP or DEV != "cuda":
        return SKIP or "skipped: CUDA graphs need the GPU"
    from engine.models.graph_lru import GraphLRU
    cfg, eng, ref = tiny()
    states, alone = [eng.new_state(256) for _ in range(3)], [eng.new_state(256) for _ in range(3)]
    for st, al, n in zip(states, alone, (14, 9, 5)):
        eng.prefill(PROMPT[:n], st)
        eng.prefill(PROMPT[:n], al)
    lru = GraphLRU(cap=1)
    sets = [(0, 1), (1, 2), (0, 1, 2), (0, 1)]
    for tok, pick in zip((7, 19, 23, 44), sets):
        rbs = eng.decode_multi([(tok, states[j], f"g{j}") for j in pick], graphs=lru)
        for j, rb in zip(pick, rbs):
            assert torch.equal(rb["logits"][:1], _decode(eng, alone[j], [tok])), (pick, j)
            states[j].length += 1
    for st, al in zip(states, alone):
        assert st.length == al.length and _same_state(st, al)
    assert len(lru) == 1 and lru.captures == 4 and lru.evicted == 3
    return f"4 rounds, {lru.captures} captures, {lru.evicted} dropped: the lone bits"


def test_mtp_head_drafts_and_resumes():
    """The MTP head on random weights of the checkpoint's layout: `launch` drafts `depth` tokens with
    their top-2 probabilities, `read` moves its length by the fed rows only, and a resume to an earlier
    length and the same feed drafts the same tokens (its KV is a view the caches restore)."""
    if SKIP:
        return SKIP
    from engine.models.nemotron_h.load import ExpertsRelu2
    from engine.models.nemotron_h.mtp import NemotronHMTPHead
    cfg, eng, ref = tiny()
    raw = dict(TINY_B, num_nextn_predict_layers=1, mtp_layers_block_type=["attention", "moe"])
    c = config_from_dict(raw, "tiny-mtp")
    g = torch.Generator().manual_seed(5)
    H = c.hidden_size

    def r(*shape, s=0.2):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16).to(DEV)
    W = dict(eng.W)
    W.update({"mtp.layers.0.enorm.weight": 1 + r(H, s=0.1), "mtp.layers.0.hnorm.weight": 1 + r(H, s=0.1),
              "mtp.layers.0.eh_proj.weight": r(H, 2 * H), "mtp.layers.0.norm.weight": 1 + r(H, s=0.1),
              "mtp.layers.0.mixer.q_proj.weight": r(c.q_dim, H), "mtp.layers.0.mixer.k_proj.weight": r(c.kv_dim, H),
              "mtp.layers.0.mixer.v_proj.weight": r(c.kv_dim, H), "mtp.layers.0.mixer.o_proj.weight": r(H, c.q_dim),
              "mtp.layers.1.norm.weight": 1 + r(H, s=0.1), "mtp.layers.1.final_layernorm.weight": 1 + r(H, s=0.1),
              "mtp.layers.1.mixer.gate.weight": r(4, H, s=0.5),
              "mtp.layers.1.mixer.gate.e_score_correction_bias": r(4, s=0.05).float(),
              "mtp.layers.1.mixer.shared_experts.up_proj.weight": r(c.shared_ffn, H),
              "mtp.layers.1.mixer.shared_experts.down_proj.weight": r(H, c.shared_ffn)})
    m = "mtp.layers.0.mixer."
    W[m + "qkv"] = torch.cat([W[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
    ex = dict(eng.experts)
    ex[("mtp", 1)] = ExpertsRelu2(r(4, 64, H).contiguous(), r(4, H, 64).contiguous())
    from engine.models.nemotron_h.engine import NemotronHEngine
    e2 = NemotronHEngine(c, W, ex, device=DEV, max_len=256)
    head = NemotronHMTPHead(e2, 3, 256)
    st = e2.new_state(256)
    e2.prefill(PROMPT, st)
    rows = e2.prompt_hidden
    head.prime(rows[:-1], PROMPT[1:].tolist())
    assert head.st.length == len(PROMPT) - 1
    head.launch(rows[-1:], [7])
    t1, p1 = head.read()
    assert head.st.length == len(PROMPT) and len(t1) == 3 and all(len(t) == 2 and p[0] >= p[1] for t, p in zip(t1, p1))
    head.resume(len(PROMPT) - 1)
    head.launch(rows[-1:], [7])
    t2, _ = head.read()
    assert t1 == t2
    return f"depth 3 drafts {[t[0] for t in t1]}, the same after a resume"


def test_self_check_passes():
    if SKIP:
        return SKIP
    for name in ("tiny", "puzzle"):
        cfg, eng, ref = tiny(name)
        res = eng.self_check(PROMPT.tolist(), steps=6, log=None)
        assert res["ok"], (name, res)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<46} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<46} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
