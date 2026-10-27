"""The narrator sums up each thought (N2, 2026-10-03): Qwen3.5 on the engine's own forward.

The owner: "for each thinking stretch something like 'thought about 2 min about X and Y' in two short
sentences; the small model does not need the whole conversation, only the thought text."

The test matrix:

  the model (engine/generative/qwen35.py, a random tiny Qwen3.5 checkpoint written here, CPU):
    1  it loads through the slot like the 0.8B's directory (the vision and MTP tensors left out,
       the head tied), and its forward is transformers' Qwen3_5ForCausalLM on the same weights
                                                     test_the_model_is_the_engines_forward
    2  a job advanced in slices (a deadline before every chunk and step) writes what one call
       writes; the prompt's fixed head is kept and the next job prefills only the rest, with the
       same answer as a cold cache; a generation in between makes a job start over
                                                     test_slices_and_the_kept_head
    3  the token cap, an end token, a prompt that does not fit    test_the_cap_and_the_end
    4  Qwen's ids pass as they are when the tokenizers agree (the added tokens dropped), and
       `same_ids` says why not when they do not      test_same_ids
    5  the thought prompt: the head and tail around the placeholder are the rendered template,
       thinking off                                  test_the_prompt_around_the_thought
  the narrator's thoughts (server/conductor/narrator.py Thoughts, the fake model, a fake clock):
    6  the clean: two sentences, a cut answer keeps what ended, markup and length refused; the
       duration as the line says it                  test_the_clean_and_the_span
    7  a stretch that ends gets "Thought <span>: <summary>" with the engine's facts; a short one
       gets no line; the window is the last THOUGHT_WINDOW ids of the thought alone
                                                     test_a_thought_gets_its_summary
    8  the budget: the slices hold Qwen at most THOUGHT_SHARE of its time; the turn's end gets
       THOUGHT_END_MS; past it, and on a refusal, an error or a refused answer, the template
                                                     test_the_budget_and_the_templates
    9  the progress mark in a long thinking: the summary of the thought since the last mark
       ("Thinking ..."), superseded by the thought's own; shadow keeps the line in the journal
                                                     test_the_mark_and_shadow
  the handler (the fake engine in process, the monitor on, a conductor; test_ui_states' serve):
   10  thoughts: the stream carries the thought's line after the thinking, the text, the calls
       and the usage are the stream's without it, the journal row keeps the line and the totals
                                                     test_the_stream_carries_the_thought
   11  the line never reaches Qwen: turn 2 with turn 1's reasoning sent back (with the line,
       trimmed; in the reasoning, the content and, merged with X1, the inline channel) reaches the
       engine as the same prompt
                                                     test_the_line_never_reaches_qwen
   12  off, absent, `on` (P1's lines) and no slot: byte for byte the stream without thoughts and
       the model never asked; a failing model: the template line   test_off_* / test_a_failing_*
   13  a long thinking with the progress lines: its marks are the narrator's summaries
                                                     test_a_long_thinking_is_summed_up_on_the_way
   14  the contract: ui.narrator's modes, the role's model, the routes' status
                                                     test_the_contract_and_the_status

Run: python tests/test_thoughts.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import threading

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ[f"QWEN38_FUSED_{_k}"] = "0"          # the CPU forward

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.generative.fake import FakeGenerative  # noqa: E402
from engine.generative.slot import Slot  # noqa: E402
from server.conductor import narrator as narr_mod  # noqa: E402
from server.conductor import schema, states  # noqa: E402

M = states.MARKER
SENT = narr_mod.SENTINEL
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
TEMPLATE = ("{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + "
            "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}"
            "{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
            "enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
            "{%- endif %}")


# ------------------------------------------------------------------ a tiny Qwen3.5 checkpoint

def tiny_dir(seed: int = 5) -> str:
    """A random Qwen3.5 checkpoint laid out as the 0.8B's release: the language model under
    `model.language_model.`, one value head per key head, the head tied to the embedding, a
    visual and an MTP tensor in the same file, and a byte-level tokenizer with Qwen's specials
    (vocabulary 300, the specials at 256-260)."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from engine.loader import LM_PREFIX
    from test_refactor_gate import TinyWeights, config
    cfg = dataclasses.replace(config("fp8"), linear_num_value_heads=2, vocab_size=300)
    tw = TinyWeights(cfg, "bf16", seed=seed)
    d = tempfile.mkdtemp(prefix="tiny-qwen35-")
    text = {"model_type": "qwen3_5_text", "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size, "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size, "rms_norm_eps": cfg.rms_norm_eps,
            "max_position_embeddings": cfg.max_position_embeddings, "layer_types": cfg.layer_types,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads, "tie_word_embeddings": True,
            "eos_token_id": 256,
            "rope_parameters": {"rope_type": "default", "rope_theta": cfg.rope_theta,
                                "partial_rotary_factor": cfg.partial_rotary_factor,
                                "mrope_section": cfg.mrope_section, "mrope_interleaved": True}}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5", "text_config": text, "tie_word_embeddings": True}, f)
    out = {}
    for name, t in tw.t.items():
        if name == "lm_head.weight":
            continue
        key = LM_PREFIX + name + (".weight" if not name.endswith(".weight") and t.dim() == 2
                                  else "")
        out[key] = t.to(torch.bfloat16).contiguous()
    out["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    out["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(out, os.path.join(d, "model.safetensors-00001-of-00001.safetensors"))
    alpha = pre_tokenizers.ByteLevel.alphabet()
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(alpha))}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    tok.save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
                   "pad_token": "<|endoftext|>", "chat_template": TEMPLATE,
                   "model_max_length": 4096}, f)
    return d


_DIR: list = []


def tiny():
    if not _DIR:
        _DIR.append(tiny_dir())
    return _DIR[0]


def loaded(**kw):
    from engine.generative.qwen35 import Qwen35Model
    m = Qwen35Model("tiny", tiny(), max_ctx=1024, claim_gib=0.1, **kw)
    m.load(None)
    return m


def test_the_model_is_the_engines_forward():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5ForCausalLM,
                                                              Qwen3_5TextRotaryEmbedding)
    from engine.generative.qwen35 import Qwen35Model
    from safetensors import safe_open
    from engine.loader import Layout
    m = Qwen35Model("tiny", tiny(), max_ctx=1024, claim_gib=0.1)
    slot = Slot(m, role="narrator", gate=threading.Lock(), log=lambda s: None,
                contract=lambda f, n: schema.role_errors(f, n))
    slot.load(wait=True)
    assert slot.state == "loaded", slot.error
    eng = m.eng
    assert eng.w.t["lm_head.weight"] is eng.w.t["embed_tokens.weight"]
    assert not any(k.startswith(("mtp", "visual")) for k in list(eng.w.t) + list(eng.w.q))
    assert m.eos >= {258} and m.added == set(range(256, 261)) and m.vocab == 300
    # transformers' own model on the same file
    hf = AutoConfig.from_pretrained(tiny()).text_config
    with torch.device("meta"):
        ref = Qwen3_5ForCausalLM(hf)
    sd = {}
    for name in os.listdir(tiny()):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(tiny(), name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    c = Layout.canonical(key)
                    if c is not None and not c.startswith("mtp"):
                        sd["model." + c] = f.get_tensor(key)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    ref.load_state_dict(sd, strict=False, assign=True)
    ref.model.rotary_emb = Qwen3_5TextRotaryEmbedding(hf, device="cpu")
    ref.eval()
    ids = list(range(3, 40))
    with torch.inference_mode():
        mine = eng.forward(torch.tensor(ids), start=0)[0].float()
        theirs = ref(input_ids=torch.tensor(ids)[None]).logits[0].float()
    scale = theirs.abs().max()
    assert (mine - theirs).abs().max() < 0.05 * scale, float((mine - theirs).abs().max() / scale)
    assert (mine.argmax(-1) == theirs.argmax(-1)).float().mean() >= 0.9
    st = slot.status()
    assert st["precision"].startswith("bfloat16") and "eager decode" in st["precision"]
    assert st["device_bytes"] > 0 and st["weight_hash"]
    assert slot.unload() and m.eng is None


def _greedy(m, ids, n):
    """The engine's plain forward, greedy, one call: the reference for the slices."""
    import torch
    eng = m.eng
    eng.reset()
    with torch.inference_mode():
        lg = eng.forward(torch.tensor(ids), start=0, last_only=True)
        out, pos = [], len(ids)
        for _ in range(n):
            t = int(lg[0, -1].argmax())
            out.append(t)
            lg = eng.forward(torch.tensor([t]), start=pos, last_only=True)
            pos += 1
    return out


def test_slices_and_the_kept_head():
    m = loaded()
    m.eos = set()                                    # a random model: no end token, 12 tokens
    head = [10 + i % 240 for i in range(300)]        # more than one prefill chunk of 256
    a = head + [40, 41, 42, 43, 44]
    want = _greedy(m, a, 12)
    whole = m.begin(a, 12, keep=len(head))
    m.advance(whole, None)
    assert whole.out == want and whole.finish == "length" and whole.reused == 0
    # the head is kept: a second job with the same head prefills only its rest
    b = head + [50, 51, 52, 53, 54, 55, 56]
    cold = _greedy(m, b, 12)
    m.snap = None
    m.advance(m.begin(a, 4, keep=len(head)), None)   # the head kept again
    warm = m.begin(b, 12, keep=len(head))
    m.advance(warm, None)
    assert warm.reused == len(head) and warm.out == cold, (warm.reused, warm.out, cold)
    # in slices: a clock that moves 1 s a read and a deadline 2.5 s after each slice starts
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]

    m.clock = clock
    job = m.begin(b, 12, keep=len(head))
    slices = 0
    while not job.done:
        m.advance(job, t[0] + 2.5)
        slices += 1
        assert slices < 40
    assert job.out == cold and slices > 3, (slices, job.out)
    # another generation in between: the job starts its prompt over and still writes the same
    j2 = m.begin(a, 12, keep=len(head))
    m.advance(j2, t[0] + 2.5)
    m.advance(m.begin(list(range(100, 140)), 3), None)
    while not j2.done:
        m.advance(j2, t[0] + 2.5)
    assert j2.out == want


def test_the_cap_and_the_end():
    m = loaded()
    ids = list(range(20, 60))
    free = _greedy(m, ids, 6)
    m.eos = {free[3]} - set(free[:3])
    j = m.begin(ids, 20)
    m.advance(j, None)
    if m.eos:
        assert j.finish == "stop" and j.out == free[:3], (j.out, free)
    m.eos = set()
    r = m.generate(ids, 5, None)
    assert r["ids"] == free[:5] and r["finish"] == "length"
    try:
        m.begin(list(range(1000)), 40)
    except ValueError:
        pass
    else:
        raise AssertionError("1,000 + 40 rows do not fit 1,024")


def test_same_ids():
    from engine.generative.qwen35 import same_ids
    m = loaded(engine_dir=tiny())
    assert m.same_ids is None
    assert m.text_ids([1, 2, 256, 299, 300, 258, 7]) == [1, 2, 299, 7]   # added and out-of-range out
    other = tempfile.mkdtemp()
    with open(os.path.join(tiny(), "tokenizer.json")) as f:
        tj = json.load(f)
    tj["added_tokens"][0]["content"] = "<|audio|>"
    with open(os.path.join(other, "tokenizer.json"), "w") as f:
        json.dump(tj, f)
    assert "added token 256" in same_ids(tiny(), other)
    tj["model"]["vocab"] = {"a": 0}
    with open(os.path.join(other, "tokenizer.json"), "w") as f:
        json.dump(tj, f)
    assert same_ids(tiny(), other) == "the vocabulary differs"
    assert same_ids(tiny(), None) and same_ids(tiny(), tempfile.mkdtemp()).endswith("missing")
    # without the check the ids go through the text
    m.same_ids = "the vocabulary differs"
    assert m.text_ids([104, 105], decode=lambda xs: "hi") == m.tok("hi")["input_ids"]


def test_the_prompt_around_the_thought():
    m = loaded()
    head, tail = m.encode_around(narr_mod.thought_messages(), narr_mod.MARK)
    whole = m.tok.decode(head + m.tok("THE THOUGHT")["input_ids"] + tail)
    assert whole.startswith("<|im_start|>system\nYou summarise the private reasoning")
    assert whole.endswith("<|im_start|>user\nReasoning:\nTHE THOUGHT<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n"), whole[-120:]
    assert whole.count("<|im_start|>assistant\n") == len(narr_mod.THOUGHT_EXAMPLES) + 1
    msgs = narr_mod.thought_messages("x")
    assert [x["role"] for x in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    # the narrator's own: the answer begun with "It", as the examples begin theirs
    slot = Slot(m, role="narrator", gate=None, log=lambda s: None)
    slot.state, slot.loaded_at = "loaded", 1.0
    h2, t2 = narr_mod.Narrator(slot).around()
    assert h2 == head and m.tok.decode(t2).endswith("</think>\n\nIt")
    assert all(a.startswith(narr_mod.THOUGHT_LEAD + " ") for _, a in narr_mod.THOUGHT_EXAMPLES)


# ------------------------------------------------------------------ the narrator's thoughts

def test_the_clean_and_the_span():
    c = narr_mod.clean_summary
    room = 300
    assert c("Planned the loop. Chose a fixed step. Then more.", "stop", room) == (
        "planned the loop. Chose a fixed step", None)
    assert c("Summary: compared X and Y; next it writes a.py.", "stop", room) == (
        "compared X and Y; next it writes a.py", None)
    assert c("HTML first, then the loop", "stop", room)[0] == "HTML first, then the loop"
    # cut at the token cap: what ended stays, a half clause goes
    assert c("compared X and Y; next it wri", "length", room) == ("compared X and Y", None)
    assert c("checked the test. Then it start", "length", room) == ("checked the test", None)
    assert c("a b c d", "length", room) == (None, "length")
    assert c("", "stop", room) == (None, "empty")
    assert c("used <think> tags", "stop", room) == (None, "markup")
    assert c(" ".join(["word"] * 50), "stop", room) == (None, "words")
    assert c("x" * 40, "stop", 30) == (None, "chars")
    assert c("\n\n  first line\nsecond", "stop", room) == ("first line", None)
    s = narr_mod.span
    assert [s(0.4), s(43), s(103), s(600), s(659), s(3900)] == [
        "0s", "43s", "1m 43s", "10m", "10m", "1h 5m"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _thoughts(model=None, *, opened=True, mode="on", budget_ms=150.0, planner=None):
    """A Thoughts over a loaded fake slot whose clock is a fake one, for a request's States."""
    clk = Clock()
    m = model or FakeGenerative(delay_s=0.01, clock=clk, sleep=lambda s: setattr(
        clk, "t", clk.t + s))
    if model is not None:
        m.clock, m.sleep = clk, (lambda s: setattr(clk, "t", clk.t + s))
    slot = Slot(m, role="narrator", gate=None, log=lambda s: None, clock=clk,
                planner=planner)
    if planner is None:
        slot.load(wait=True)
    narr = narr_mod.Narrator(slot, role=lambda: {"mode": mode, "latency_budget_ms": budget_ms})
    ui = states.States()
    narr_mod.THOUGHT_MIN_TOKENS = 128                # the handler tests lower it
    th = narr_mod.Thoughts(narr, ui, opened=opened, close_id=0x110002, budget_ms=budget_ms,
                           shadow=mode == "shadow", clock=clk, log=lambda *a, **k: None)
    return th, ui, m, clk


def _think(th, clk, ids, words, *, per_token_s=0.02, in_think=True):
    """Qwen writes `words` (one id a character), `per_token_s` apart."""
    for ch in words:
        clk.t += per_token_s
        ids.append(ord(ch))
        th.token(ids, in_think)


def test_a_thought_gets_its_summary():
    th, ui, m, clk = _thoughts()
    ids: list[int] = []
    thought = ("Let me plan the level layout first. " * 8)[:200]
    _think(th, clk, ids, thought)                     # 200 tokens over 4 s
    ids.append(0x110002)                              # </think>
    clk.t += 0.02
    th.token(ids, False)                              # the block closed: its summary starts
    _think(th, clk, ids, "Here is the plan." * 40, in_think=False)
    th.finish(ids)
    got = [e for e in ui.log if e["kind"] == "thought"]
    assert len(got) == 1, ui.log
    e = got[0]
    assert e["summary"] == "Thought 4s: weighed Let me plan; next it answers", e
    assert e["text"] == "thought 4 s, 200 tokens"     # the template, the engine's facts
    assert e["narrator"]["why"] is None and e["narrator"]["thought_tokens"] == 200
    assert e["narrator"]["window"] == 200 and e["narrator"]["tokens"] == 7
    # the model read the thought alone: the head, the thought's ids (no close tag), the tail
    from engine.generative.fake import HEAD, TAIL
    assert m.jobs_seen[0] == ([ord(c) for c in HEAD] + [ord(c) for c in thought]
                              + [ord(c) for c in TAIL])
    shown = ui.take()
    assert shown == [narr_mod.frame(e["summary"])] and shown[0].endswith(SENT + "\n\n")
    assert states.strip_text("a" + shown[0] + "b") == "ab"
    tot = th.totals()
    assert tot["lines"] == 1 and tot["templates"] == 0 and tot["ms"] > 0
    # a short thought: no line at all; a long one: only its last THOUGHT_WINDOW ids
    th, ui, m, clk = _thoughts()
    ids = []
    _think(th, clk, ids, "x" * (narr_mod.THOUGHT_MIN_TOKENS - 1))
    th.token(ids + [0x110002], False)
    th.finish(ids)
    assert not ui.log and th.totals()["none"] == 1 and not m.jobs_seen
    th, ui, m, clk = _thoughts(opened=False)
    ids = [ord(c) for c in "<think>"]                 # Qwen opened the block itself
    th.token(ids, True)
    _think(th, clk, ids, "ab" * 1000, per_token_s=0.0)
    th.finish(ids)                                    # the turn ended inside the thinking
    e = [x for x in ui.log if x["kind"] == "thought"][0]
    assert e["narrator"]["thought_tokens"] == 2000 and e["narrator"]["window"] == 1500
    assert len(m.jobs_seen[0]) == len(HEAD) + narr_mod.THOUGHT_WINDOW + len(TAIL)


def test_the_budget_and_the_templates():
    # a model at 50 ms a token: its 7 tokens are 350 ms; the share earns 30 ms a second of Qwen
    slow = FakeGenerative(delay_s=0.05)
    th, ui, m, clk = _thoughts(slow)
    ids: list[int] = []
    _think(th, clk, ids, "y" * 200, per_token_s=0.01)    # 2 s of thinking: 60 ms earned
    ids.append(0x110002)
    th.token(ids, False)
    assert th.job is not None and th.spent_ms <= 0.03 * 2.01 * 1e3 + 50 + 1e-6, th.spent_ms
    # Qwen writes the answer: 10 s, the share pays slices as they are earned
    _think(th, clk, ids, "z" * 500, per_token_s=0.02, in_think=False)
    assert th.job is None and ui.log and ui.log[0]["summary"].startswith("Thought 2s: ")
    held = th.spent_ms
    assert held <= 0.03 * (clk.t) * 1e3 + 50 + 1e-6, (held, clk.t)
    assert ui.log[0]["narrator"]["slices"] > 1
    # a thought at the turn's very end: the end allowance finishes it
    th, ui, m, clk = _thoughts(FakeGenerative(delay_s=0.05))
    ids = []
    _think(th, clk, ids, "y" * 200, per_token_s=0.0)      # no time earned
    th.finish(ids)
    e = ui.log[0]
    assert e["summary"] and th.totals()["end_ms"] >= 250, th.totals()
    # a model too slow for the end allowance: the template, why time
    th, ui, m, clk = _thoughts(FakeGenerative(delay_s=0.2))
    ids = []
    _think(th, clk, ids, "y" * 300, per_token_s=0.0)
    th.finish(ids)
    e = ui.log[0]
    assert e["summary"] is None and e["narrator"]["why"] == "time", e
    assert ui.take() == [states.frame("thought 0 s, 300 tokens")]
    assert th.totals()["end_ms"] <= narr_mod.THOUGHT_END_MS + 200 + 1
    # an error, a refused answer, an unloaded model whose load the planner refuses
    bad = FakeGenerative()
    bad.fail = True
    th, ui, _, clk = _thoughts(bad)
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    assert ui.log[0]["summary"] is None and ui.log[0]["narrator"]["why"] == "error"
    th, ui, _, clk = _thoughts(FakeGenerative(summary=lambda ids: "used <b>bold</b> words"))
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    assert ui.log[0]["narrator"]["why"] == "markup" and "raw" in ui.log[0]["narrator"]
    th, ui, m, clk = _thoughts(planner=lambda name, **kw: {"allowed": False,
                                                          "reason": "refused_hold"})
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    assert ui.log[0]["narrator"]["why"] == "refused" and not m.jobs_seen
    # a secret the model repeats from the thought never reaches the line
    key = "sk-" + "a1b2c3d4" * 4
    th, ui, _, clk = _thoughts(FakeGenerative(summary=lambda ids: f"tried the key {key} first"))
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    assert key not in ui.log[0]["summary"] and "[redacted:api_key]" in ui.log[0]["summary"]
    # a broken hook never breaks the stream: it switches itself off
    th, ui, m, clk = _thoughts()
    th.ui = None
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    assert th.broken and th.totals()["broken"]


def test_the_mark_and_shadow():
    th, ui, m, clk = _thoughts()
    ids: list[int] = []
    _think(th, clk, ids, "alpha beta gamma " * 20)       # 340 tokens, 6.8 s
    assert th.mark(ids, 6.8)                             # the progress line's mark
    e = ui.log[-1]
    assert e["kind"] == "thinking" and e["summary"] == "Thinking 7s: weighed alpha beta gamma; " \
        "next it answers", e
    assert e["text"] == "thinking 7 s, 340 tokens"
    _think(th, clk, ids, "delta epsilon zeta " * 20)
    assert th.mark(ids, 14.0)
    # the second mark reads the thought since the first one
    assert ui.log[-1]["summary"].startswith("Thinking 14s: weighed delta epsilon zeta")
    # a mark while a summary is still being written: the caller says the template
    th.job = {"kind": "thinking"}
    assert not th.mark(ids, 20.0)
    th.job = None
    # the thought ends: its own line; a mark's summary still running is dropped for it
    slow = FakeGenerative(delay_s=0.5)
    th, ui, m, clk = _thoughts(slow)
    ids = []
    _think(th, clk, ids, "alpha beta gamma " * 20)
    th.mark(ids, 6.8)
    assert th.job is not None and th.job["kind"] == "thinking"
    ids.append(0x110002)
    th.token(ids, False)
    th.finish(ids)
    assert [e["kind"] for e in ui.log] == ["thought"] and th.totals()["superseded"] == 1
    # shadow: the template goes out, the summary is in the journal record
    th, ui, m, clk = _thoughts(mode="shadow")
    ids = []
    _think(th, clk, ids, "y" * 200)
    th.finish(ids)
    e = ui.log[0]
    assert e["summary"] is None and e["narrator"]["shadow"].startswith("Thought ")
    assert ui.take() == [states.frame("thought 4 s, 200 tokens")]


# ------------------------------------------------------------------ the handler

THINKING = {"mode": "on", "thinking_every_tokens": 100000, "thinking_every_s": 3600}


def fake_slot(**kw):
    m = FakeGenerative(**kw)
    slot = Slot(m, role="narrator", gate=threading.Lock(), log=lambda s: None)
    slot.load(wait=True)
    return slot, m


def _serve(narrator, slot=None, channel=None, progress=None, min_tokens=16):
    import test_ui_states as ui
    from server import app
    cond, seen = ui.serve("on")
    from test_narrator import narrator_on
    narrator_on(cond)               # SRV-534: the role is off by default
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        u = dict(doc["ui"])
        if narrator is not None:
            u["narrator"] = narrator
        if channel is not None:
            u["states_channel"] = channel
        if progress is not None:
            u["progress"] = progress
        doc["ui"] = u
        cond.store.put("profiles", name, doc, note="thoughts", author="test")
    app.STATE.pop("narrator", None)
    if slot is not None:
        app.STATE["narrator"] = narr_mod.Narrator(
            slot, role=lambda: narr_mod.role_doc(app.STATE),
            labels=lambda: narr_mod.class_labels(app.STATE))
    narr_mod.THOUGHT_MIN_TOKENS = min_tokens         # the fake's thought is ~50 characters
    return cond, seen


def _call():
    return ("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
            json.dumps({"name": "bash", "arguments": {"command": "ls -la"}}))


def _thought_lines(text):
    return [ln[len(M):] for ln in text.split("\n") if ln.startswith(M + "Thought ")]


def test_the_stream_carries_the_thought():
    import test_ui_states as ui
    _serve("off")
    off = ui.opencode(_call())
    s, m = fake_slot()
    cond, _ = _serve("thoughts", s)
    on = ui.opencode(_call())
    got = _thought_lines(on["reasoning"] + on["content"])
    assert len(got) == 1 and got[0].startswith("Thought 0s: weighed The question has"), got
    assert got[0].endswith(SENT)
    # the model read the thought alone: no system prompt, no tool, no user text
    from engine.generative.fake import HEAD, TAIL
    seen = "".join(chr(i) for i in m.jobs_seen[0])
    body = seen[len(HEAD):-len(TAIL)]
    assert body.startswith("The question has") and "ls -la" not in body and "bash" not in body
    # the line came after the thinking (the channel puts it where state lines go)
    assert (on["reasoning"] + on["content"]).index("Thought ") > 0
    # the rest is the stream without it: the other lines templates, the same text, calls, usage
    assert states.strip_text(on["reasoning"]) == states.strip_text(off["reasoning"])
    assert states.strip_text(on["content"]) == states.strip_text(off["content"])
    assert [c["function"] for c in on["calls"]] == [c["function"] for c in off["calls"]]
    ui.same_usage(on, off)
    rid = json.loads(on["raw"].splitlines()[0][6:])["id"]
    row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
    th = [e for e in row["extra"]["states"] if e["kind"] == "thought"][0]
    assert th["summary"].startswith("Thought ") and th["narrator"]["why"] is None
    assert th["text"].startswith("thought ") and th["narrator"]["thought_tokens"] > 16
    assert row["extra"]["narrator"]["lines"] == 1, row["extra"]["narrator"]
    # the other lines stayed templates: the route line is the template's
    assert ui.lines(on["reasoning"] + on["content"])[0] == "deciding: code_edit, xhigh effort"


def test_the_line_never_reaches_qwen():
    import test_ui_states as ui
    s, _ = fake_slot()
    prompts = {}
    # X1's inline channel (a reasoning part titled by the line) once the branches are merged
    channels = ("reasoning", "content") + (("inline",) if "inline" in states.CHANNELS else ())
    for channel in channels:
        for shape in ("clean", "lines", "trimmed"):
            _, seen = _serve("thoughts", s, channel=channel)
            first = ui.opencode("Write the lexer.")
            r, c = first["reasoning"], first["content"]
            assert "Thought " in r + c and SENT in r + c, (r, c)
            pick = {"lines": lambda x: x, "trimmed": lambda x: x.strip(),
                    "clean": states.strip_text}[shape]
            asst = {"role": "assistant", "content": pick(c), "reasoning_content": pick(r)}
            seen.clear()
            ui.opencode("Now add a test.",
                        history=[{"role": "user", "content": "Write the lexer."}, asst])
            prompts[(channel, shape)] = seen[-1]
        assert prompts[(channel, "lines")] == prompts[(channel, "clean")] == \
            prompts[(channel, "trimmed")], channel
    # and the inline title of a thought's line comes out of any text
    if "inline" in states.CHANNELS:
        line = narr_mod.frame("Thought 1m 43s: planned the loop; next it writes game.js")
        t = states.inline(line)
        assert states.strip_text("before" + t + "after") == "beforeafter"
    # a long summary (two sentences, the most the clean lets through) is still one line the
    # strip takes
    long = narr_mod.frame("Thought 12m: " + "x" * (states.NARRATED_MAX_CHARS - 14))
    assert states.strip_text("a" + long + "b") == "ab"


def test_off_absent_on_and_no_slot_are_the_stream_without_thoughts():
    import test_ui_states as ui
    _serve(None)
    base = ui._strip_ids(ui.opencode(_call())["raw"])
    for narrator, with_slot in (("off", True), (None, True), ("thoughts", False)):
        s = m = None
        if with_slot:
            s, m = fake_slot()
        _serve(narrator, s)
        assert ui._strip_ids(ui.opencode(_call())["raw"]) == base, (narrator, with_slot)
        assert m is None or not m.jobs_seen
    # P1's `on`: its lines, never a thought's
    s, m = fake_slot()
    _serve("on", s)
    on = ui.opencode(_call())
    assert not _thought_lines(on["reasoning"] + on["content"]) and not m.jobs_seen and m.calls


def test_a_failing_narrator_says_the_template():
    import test_ui_states as ui
    s, m = fake_slot()
    m.fail = True
    _serve("thoughts", s)
    on = ui.opencode(_call())
    got = [x for x in ui.lines(on["reasoning"] + on["content"]) if x.startswith("thought ")]
    assert len(got) == 1 and got[0].endswith(" tokens"), ui.lines(on["reasoning"])
    # the role's mode off: nothing asked, no line
    from server import app
    s, m = fake_slot()
    _serve("thoughts", s)
    app.STATE["narrator"] = narr_mod.Narrator(s, role=lambda: {"mode": "off",
                                                               "latency_budget_ms": 150})
    on = ui.opencode(_call())
    assert "thought" not in " ".join(ui.lines(on["reasoning"] + on["content"])).lower()
    assert not m.jobs_seen


def test_a_long_thinking_is_summed_up_on_the_way():
    import test_ui_states as ui
    s, m = fake_slot()
    _serve("thoughts", s, progress={"mode": "on", "thinking_every_tokens": 1000,
                                    "thinking_every_s": 3600})
    # the fake engine writes 200,000 tokens a second: no share would pay a slice in time, so
    # here the share is everything (the budget's own test is above)
    share, least = narr_mod.THOUGHT_SHARE, narr_mod.THOUGHT_MIN_SLICE_MS
    narr_mod.THOUGHT_SHARE, narr_mod.THOUGHT_MIN_SLICE_MS = 1.0, 0
    try:
        on = ui.opencode("FAKE_LOOP write the pagoda")
    finally:
        narr_mod.THOUGHT_SHARE, narr_mod.THOUGHT_MIN_SLICE_MS = share, least
    said = ui.lines(on["reasoning"] + on["content"])
    marks = [x for x in said if x.startswith("Thinking ")]
    assert marks and all(x.endswith(SENT) for x in marks), said
    templ = lambda xs: [x for x in xs if x.startswith("thinking ") and x.endswith(" tokens")]  # noqa: E731
    assert not templ(said), said                       # no template beside them
    assert "weighed I am going" in marks[0]
    # with the narrator off the same request says the templates
    _serve("off", progress={"mode": "on", "thinking_every_tokens": 1000,
                            "thinking_every_s": 3600})
    off = ui.opencode("FAKE_LOOP write the pagoda")
    assert templ(ui.lines(off["reasoning"] + off["content"]))


def test_the_contract_and_the_status():
    prof = schema.SCHEMAS["profiles"]
    ui_s = prof["properties"]["ui"]["properties"]
    assert ui_s["narrator"]["enum"] == ["off", "on", "thoughts", "live"]       # v3: live
    assert "thought" in ui_s["kinds"]["properties"]
    for name in schema.PROFILES:
        assert (schema.seed("profiles", name).get("ui") or {}).get("narrator", "off") == "off"
    r = schema.seed("roles")["narrator"]
    assert r["model"] == "qwen3.5-0.8b" and r["window"] == "between_rounds"
    assert not schema.role_errors({k: r[k] for k in schema.ROLE_CONTRACT}, "narrator")
    from server import app
    s, _ = fake_slot()
    app.STATE["narrator"] = narr_mod.Narrator(s, role=lambda: {"mode": "on",
                                                               "latency_budget_ms": 150})
    st = narr_mod.status(app.STATE)
    assert st["thoughts"]["jobs"] and st["thoughts"]["qwen_ids"] is True
    assert st["thoughts"]["share"] == narr_mod.THOUGHT_SHARE
    # the Qwen3.5 names resolve to the engine's forward, the directory by the name
    from engine import generative
    q = generative.model("qwen3.5-0.8b", None)
    assert type(q).__name__ == "Qwen35Model" and q.dir.endswith("models/qwen3.5-0.8b")
    assert q.memory_claim_gib == 2.5 and q.jobs
    try:
        generative.model("qwen9", None)
    except ValueError as exc:
        assert "qwen3.5-0.8b" in str(exc)
    else:
        raise AssertionError("an unknown name stops the start")


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
