"""ENG-179 / ENG-180 row 1, the unit part: the Laya port on a tiny random checkpoint, on the CPU.

The real gate is tools/decider_equivalence.py on the box with the three real checkpoints (0.00
against the SDK on arbiter's 22 questions). Here a checkpoint of the same layout -- a 3-layer
ModernBERT with a word-level tokenizer, a 1-layer head, random weights from a fixed seed -- is
written to a temp directory and loaded the way the registry loads Laya's:

  * the rows: [CLS] head [SEP] one [MASK] per option [SEP] state [SEP], markers on the masks,
    the state cut to the room the head leaves; too many options for the head is an option-budget
    error;
  * batching changes nothing: every answer of one batched forward equals the one-row forward;
  * through the registry: an `auto` question goes through the Laya router and answers;
  * the router sends arbiter's 22-question cases where arbiter sent them (no model needed);
  * with the SDK importable (`DECIDER_SDK=<dir holding laya/>`), the port answers exactly as
    `laya.Agent.system_one` on the same checkpoint (skipped, and said so, without it).

Run: python tests/test_decider_laya.py      (DECIDER_SDK=~/deciders-equiv/sdk on the box)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402

from engine.decision import routers  # noqa: E402
from engine.decision.base import OptionBudgetError  # noqa: E402
from engine.decision.laya.model import LayaModel  # noqa: E402
from engine.decision.registry import Registry  # noqa: E402

WORDS = ("the a it is was to of and in on for with this that crashes start please fix bug plan "
         "work talk which kind what how urgent message charged twice refund support debug chat "
         "question noul choice score level false true yes no statement holds does not hold "
         "level 0 1 2 3 hello world urgent low high option").split()
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
CHOICE = {"cls": {"type": "choice", "instructions": "Which kind of work is this?",
                  "criteria": {"debug": "fix a bug", "plan": "plan the work", "chat": None}}}
MIXED = {**CHOICE,
         "urgent": {"type": "noul", "instructions": "The message is urgent.",
                    "criteria": {"true": "it is urgent"}},
         "level": {"type": "score", "instructions": "How bad is it?",
                   "criteria": ["low", "high", "urgent"]}}


def make_checkpoint(root: str, seed: int = 7) -> str:
    """A Laya-layout checkpoint (rl_agent_config.json, tokenizer/, encoder/, model.safetensors)."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
    from engine.decision.laya.head import LayaNet

    vocab = {t: i for i, t in enumerate(SPECIAL + sorted(set(WORDS)))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tok.save_pretrained(os.path.join(root, "tokenizer"))
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=96,
                           num_hidden_layers=3, num_attention_heads=4,
                           global_attn_every_n_layers=3, local_attention=16,
                           max_position_embeddings=256, pad_token_id=0, cls_token_id=2,
                           sep_token_id=3, bos_token_id=2, eos_token_id=3)
    cfg.save_pretrained(os.path.join(root, "encoder"))
    torch.manual_seed(seed)
    enc = ModernBertModel(cfg)
    net = LayaNet(enc, head_layers=1, n_act=2)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.2)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()},
              os.path.join(root, "model.safetensors"))
    with open(os.path.join(root, "rl_agent_config.json"), "w") as f:
        json.dump({"encoder": "tiny-modernbert", "head_layers": 1, "max_len": 64,
                   "head_max_len": 40, "amp_dtype": "bf16", "temperature": [1.0, 0.9, 1.1],
                   "temperature_by_options": {"choice:3-5": 0.8}, "act_costs": {"ask": 1.0}}, f)
    return root


CKPT = make_checkpoint(tempfile.mkdtemp(prefix="tiny-laya-"))


def _model() -> LayaModel:
    m = LayaModel("laya-english", CKPT)
    m.load(None)
    return m


def test_limits_come_from_the_checkpoint_before_load():
    m = LayaModel("laya-english", CKPT)
    assert m.limits() == {"context_tokens": 64, "question_tokens": 40, "max_options": 20}
    assert m.precision() is None and m.weight_hash() is None


def test_rows_have_the_sdk_layout():
    m = _model()
    tok = m.tok
    rows = m.build_rows("it crashes on start please fix " * 20, CHOICE)
    r = rows[0]
    ids, mk = r["ids"], r["markers"]
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert len(ids) == 64, "the state is cut to the room the head leaves in max_len"
    assert len(mk) == 3 and all(ids[p] == tok.mask_token_id for p in mk)
    assert ids[mk[0] - 1] == tok.sep_token_id, "the head ends with [SEP] before the first marker"
    assert r["qtype"] == 0 and r["q"]["crit"] == CHOICE["cls"]["criteria"]
    noul = m.build_rows("x", {"n": {"type": "noul", "instructions": "It holds."}})[0]
    assert len(noul["markers"]) == 2 and noul["qtype"] == 2
    many = {"q": {"type": "choice", "instructions": "pick",
                  "criteria": {f"o{i}": "fix a bug in the work please" for i in range(20)}}}
    try:
        m.build_rows("x", many)
        raise AssertionError("20 options of at least 4 tokens cannot all have a marker in 64")
    except OptionBudgetError as exc:
        assert "head_max_len=40" in str(exc)
    assert m.precision() == "fp32" and len(m.weight_hash()) == 64 and m.device_bytes() > 0


def test_batching_changes_nothing():
    m = _model()
    states = ["it crashes on start please fix", "hello world plan the work",
              {"message": "charged twice refund", "urgent": True}]
    rows, single = [], []
    for st in states:
        rs = m.build_rows(st, MIXED)
        rows += rs
        single += m.predict_rows(rs)[0]
    batched, tokens, path = m.predict_rows(rows)
    assert path == "eager" and len(batched) == len(single) == 9
    assert tokens == [len(r["ids"]) for r in rows]
    assert batched == single, "a row's answer depends on its batch"
    a = batched[0]
    assert a["type"] == "choice" and set(a["probabilities"]) == {"debug", "plan", "chat"}
    assert abs(sum(a["probabilities"].values()) - 1.0) < 2e-3 and 0 <= a["confidence"] <= 1
    assert batched[1]["type"] == "noul" and 0 <= batched[1]["noul"] <= 1
    assert batched[2]["type"] == "score" and set(batched[2]["legend"]) == {"0", "1", "2"}
    assert 0 <= batched[0]["action"]["act_probability"] <= 1


def test_through_the_registry_with_the_laya_router():
    m = LayaModel("laya-english", CKPT)
    reg = Registry([m], gate=threading.Lock(), routers=routers([m]), log=lambda s: None)
    reg.load("laya-english", wait=True)
    assert reg.state["laya-english"] == "loaded", reg.error
    out = reg.ask("it crashes on start please fix the bug", CHOICE, model="auto").result(30)
    assert out["model"] == "laya-english" and out["routing"]["reason"] == "English Latin text"
    assert out["answers"]["cls"]["choice"] in ("debug", "plan", "chat")
    out = reg.ask("Привет, мир", CHOICE, model="jev-latest").result(30)
    assert out["routing"]["requested"] == "laya-multilingual", "Cyrillic wants the multilingual one"
    assert "not served here" in out["routing"]["reason"] and out["model"] == "laya-english"
    assert reg.limits("laya") == {"context_tokens": 64, "question_tokens": 40, "max_options": 20}
    assert reg.unload("laya-english") and m.net is None


def test_the_router_sends_arbiters_cases_where_arbiter_did():
    from decider_equivalence import routing_check
    assert routing_check() == []


def test_the_port_answers_as_the_sdk():
    sdk = os.environ.get("DECIDER_SDK")
    if not sdk:
        return ("skipped: DECIDER_SDK not set (the box run sets it; the gate itself is "
                "tools/decider_equivalence.py)")
    sys.path.insert(0, os.path.expanduser(sdk))
    from laya.agent import Agent
    agent = Agent(CKPT, device="cpu")
    m = _model()
    from decider_equivalence import compare
    for st in ("it crashes on start please fix", {"message": "charged twice refund"},
               [{"role": "user", "text": "hello world"}]):
        ref = agent.system_one(st, MIXED)["answers"]
        got = dict(zip(MIXED, m.predict_rows(m.build_rows(st, MIXED))[0]))
        w, agree, total, moved = compare(ref, got)
        assert w == 0.0 and agree == total, (st, w, moved)
        assert ref == got, "the whole answer object, not only the probabilities"
    return "the SDK and the port agree exactly on the tiny checkpoint"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            note = fn()
            print(f"  {name:72s} ok" + (f"  ({note})" if note else ""))
            passed += 1
    print(f"{passed} passed")
