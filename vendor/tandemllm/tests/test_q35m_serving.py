"""ENG-391: the Qwen3.5 MoE serving adapter's cold verbs on a tiny CPU engine: a cache restore (`resume_at`, the end
anchor's and the park's way back) never touches Flash-Next's sparse-indexer pool, which this family has not
(2026-10-20 live: `build_pool` divided by the missing compress ratio on the second call of a tool turn); the taps'
bookkeeping moves with a restore and a restart; the cost and the cache state hold no indexer, PLE or stream parts.
Skipped without transformers' qwen3_5_moe (the tiny weights come from its random init)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _serving(taps=False):
    try:
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    except Exception:                                              # noqa: BLE001
        return None
    from engine.models.qwen3_5_moe.config import config_from_dict
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    from engine.models.qwen3_5_moe.serving import Qwen35MoeServing
    from tools.q35m_refcheck import tiny_config, weights_of
    raw = tiny_config("ornith")
    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**{k: v for k, v in raw.items() if k != "model_type"}))
    eng = Qwen35MoeEngine(config_from_dict(raw), weights_of(model), {}, device="cpu", max_len=64)
    if taps:
        eng.set_taps([1, 2], 16)
    return Qwen35MoeServing(eng, 64)


def test_resume_at_without_an_indexer():
    srv = _serving(taps=True)
    if srv is None:
        return "skipped: transformers without qwen3_5_moe"
    srv.resume_at(10)                       # the end anchor's restore: no indexer pool to build
    assert srv.kv.length == 10 and srv.st.tap_from == 10
    gen = srv.st.tap_gen
    srv.fn.build_pool(srv.st)
    srv.restart()
    assert srv.kv.length == 0 and srv.st.tap_from == 0 and srv.st.tap_gen == gen + 1
    srv.settle()
    srv.discard_pending()


def test_cost_and_state_have_no_flash_next_parts():
    srv = _serving()
    if srv is None:
        return "skipped: transformers without qwen3_5_moe"
    c = srv.cfg
    cost = srv.cost()
    assert cost.view_bytes_per_token == len(c.attention_layers) * c.num_key_value_heads * c.head_dim * 2 * 2
    secs = srv.state.sections()
    assert not [k for k in secs if k.startswith("ple.")]
    assert secs["mtp.h_last"].shape == (1, c.hidden_size)
    assert [v[0] for v in srv.state.views()] == [f"{x}.{li}" for li in c.attention_layers for x in "kv"]
    assert srv.price_class(100) == 0 and srv.price_class(5000) == 8192
    # SRV-655: the prompt's workspace is this trunk's (K and V repeated to every query head for
    # SDPA), not Flash-Next's line: 32,768 B a token at Ornith's 32 heads of 256
    assert cost.workspace_per_token == c.num_attention_heads * c.head_dim * 2 * 2
    assert cost.workspace_floor == int(2.75 * (1 << 30))
    assert cost.workspace_cap == cost.workspace_floor + cost.workspace_per_token * srv.max_len


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
