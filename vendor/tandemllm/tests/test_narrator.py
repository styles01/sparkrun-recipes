"""The narrator (P1, 2026-10-02): a small generative model inside the engine writes the state lines.

The test matrix:

  the model (engine/generative, a random tiny Llama checkpoint written here, CPU, fp32):
    1  the forward equals transformers' LlamaForCausalLM on the same weights: the logits, and
       greedy text                                  test_the_forward_is_transformers_llama
    2  a prompt that shares a prefix with the cache reuses its rows and generates what a cold
       cache does                                   test_the_cache_reuses_a_common_prefix
    3  the token cap and the time budget end the generation (the budget is checked before every
       step)                                        test_the_token_cap_and_the_deadline
    4  the checkpoint directory loads through the slot (planner, placement on the CPU), its chat
       template closes thinking, it answers, it unloads    test_the_slot_loads_and_runs_a_checkpoint
    5  MiniCPM5-2B's own tokenizer on the box: the generation prompt ends in an empty, closed
       think block (skipped where the weights are not)    test_minicpm5_template_thinking_off
  the narrator (server/conductor/narrator.py, the fake generative model):
    6  a line inside the budget is the narrator's; every failure is the template with its reason:
       loading (the first line starts the load), refused, time, length, words, markup, empty,
       error, busy                                   test_every_failure_is_the_template
    7  the budget is hard: a slow model is cut within one step of it; a request spends at most
       REQUEST_LINES budgets                         test_the_budget_is_hard
    8  the roles document's mode: off asks nothing, shadow journals the line and shows the
       template                                      test_role_mode_off_and_shadow
    9  a narrated line is stripped wherever it sits, as the templates are; a `›` line of Qwen's
       own stays                                     test_a_narrated_line_is_stripped
  the handler (the fake engine in process, the monitor on, a conductor; test_ui_states' serve):
   10  ui.narrator off, absent, or no narrator configured: the stream is byte for byte the one
       without a narrator                            test_off_is_byte_identical
   11  on: the lines are the narrator's, the text, the calls and the usage are the stream's
       without it; the reasoning sent back reaches the engine as the prompt without the lines;
       the journal keeps the summary and the narrator's numbers   test_on_*
   12  a failing narrator: the stream is the template stream byte for byte   test_a_failing_*
  the contract and the residency:
   13  the seed's narrator role signs the contract; the rules for generative roles; a roles
       document stored before it gets the built-in one; the residency planner knows its claim
       and never loads it beside a decider load     test_the_role_*
   14  the engine's ceiling leaves the narrator's bytes out; the routes answer
                                                    test_the_ceiling_*, test_the_routes

Run: python tests/test_narrator.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.generative.fake import FakeGenerative  # noqa: E402
from engine.generative.slot import Slot  # noqa: E402
from server.conductor import narrator as narr_mod  # noqa: E402
from server.conductor import residency, schema, states  # noqa: E402

M = states.MARKER
TEMPLATE = ("{{- bos_token }}{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + "
            "m.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}"
            "{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
            "enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
            "{%- endif %}")
SPECIALS = ["<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


# ------------------------------------------------------------------ a tiny checkpoint

def tiny_dir(seed: int = 0, tied: bool = False) -> str:
    """A random Llama checkpoint of 2 layers with a byte-level tokenizer, as MiniCPM5's directory
    is laid out (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    d = tempfile.mkdtemp(prefix="tiny-llama-")
    alpha = pre_tokenizers.ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(alpha))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    tok.save(os.path.join(d, "tokenizer.json"))
    n_vocab = 272
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>",
                   "eos_token": "<|im_end|>", "pad_token": "</s>", "unk_token": "<unk>",
                   "chat_template": TEMPLATE, "model_max_length": 4096}, f)
    im_end = tok.token_to_id("<|im_end|>")
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
           "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 16, "num_hidden_layers": 2, "vocab_size": n_vocab, "rms_norm_eps": 1e-6,
           "rope_theta": 5000000.0, "rope_scaling": None, "tie_word_embeddings": tied,
           "hidden_act": "silu", "max_position_embeddings": 4096, "bos_token_id": 0,
           "eos_token_id": [im_end], "torch_dtype": "float32"}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=0.08):
        return (torch.randn(*shape, generator=g) * s).contiguous()

    w = {"model.embed_tokens.weight": r(n_vocab, 64, s=1.0), "model.norm.weight": 1 + r(64)}
    if not tied:
        w["lm_head.weight"] = r(n_vocab, 64, s=0.3)
    for i in range(2):
        p = f"model.layers.{i}."
        w.update({p + "input_layernorm.weight": 1 + r(64),
                  p + "post_attention_layernorm.weight": 1 + r(64),
                  p + "self_attn.q_proj.weight": r(64, 64), p + "self_attn.k_proj.weight": r(32, 64),
                  p + "self_attn.v_proj.weight": r(32, 64), p + "self_attn.o_proj.weight": r(64, 64),
                  p + "mlp.gate_proj.weight": r(128, 64), p + "mlp.up_proj.weight": r(128, 64),
                  p + "mlp.down_proj.weight": r(64, 128)})
    save_file(w, os.path.join(d, "model.safetensors"))
    return d


def _llama(d):
    import torch
    from safetensors.torch import load_file
    from engine.generative.llama import Config, Llama
    return Llama(Config.read(d), {k: v for k, v in load_file(
        os.path.join(d, "model.safetensors")).items()}, max_ctx=256), torch


def _reference(d):
    """transformers' own LlamaForCausalLM on the same directory (fp32, eager attention)."""
    import torch
    from transformers import LlamaForCausalLM
    m = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.float32,
                                         attn_implementation="eager")
    return m.eval()


def test_the_forward_is_transformers_llama():
    import torch
    for tied in (False, True):
        d = tiny_dir(seed=3, tied=tied)
        net, _ = _llama(d)
        ref = _reference(d)
        ids = [5, 77, 140, 3, 99, 200, 12, 64, 31, 250, 7]
        with torch.inference_mode():
            want = ref(torch.tensor([ids])).logits[0, -1].float()
            got = net.forward(ids, 0)
        assert torch.allclose(got, want, atol=2e-5, rtol=1e-4), (got - want).abs().max()
        # greedy, 12 tokens: transformers recomputing the whole sequence each step
        seq = list(ids)
        with torch.inference_mode():
            for _ in range(12):
                seq.append(int(ref(torch.tensor([seq])).logits[0, -1].argmax()))
        net.ids = []
        r = net.generate(ids, 12, eos=set())
        assert seq[len(ids):] == r["ids"], (seq[len(ids):], r["ids"])
        assert r["finish"] == "length" and r["prefilled"] == len(ids) and r["reused"] == 0


def test_the_cache_reuses_a_common_prefix():
    d = tiny_dir(seed=4)
    warm, _ = _llama(d)
    system = [10, 20, 30, 40, 50, 60, 70, 80]
    a, b = system + [1, 2, 3], system + [9, 8, 7, 6]
    warm.generate(a, 6, eos=set())
    rb = warm.generate(b, 8, eos=set())
    assert rb["reused"] == len(system) and rb["prefilled"] == 4, rb
    cold, _ = _llama(d)
    assert cold.generate(b, 8, eos=set())["ids"] == rb["ids"]
    # the same prompt again: everything but its last row is reused, the same text comes out
    again = warm.generate(b, 8, eos=set())
    assert again["reused"] == len(b) - 1 and again["ids"] == rb["ids"]
    # a prompt that does not fit is refused before anything is written
    try:
        warm.generate(list(range(250)), 24, eos=set())
    except ValueError:
        pass
    else:
        raise AssertionError("250 + 24 rows do not fit 256")


def test_the_token_cap_and_the_deadline():
    d = tiny_dir(seed=5)
    net, _ = _llama(d)
    r = net.generate([1, 2, 3], 5, eos=set())
    assert len(r["ids"]) == 5 and r["finish"] == "length"
    # an end token stops it, and is not returned
    first = net.generate([1, 2, 3], 1, eos=set())["ids"][0]
    assert net.generate([1, 2, 3], 5, eos={first}) == {"ids": [], "finish": "stop",
                                                       "prefilled": 1, "reused": 2}
    # the deadline: a clock that moves 1 s a read, a deadline at 3.5: read 1 before the prefill,
    # 2 and 3 after the first two tokens, 4 after the third -> three tokens, then time
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]

    r = net.generate([4, 5, 6], 20, eos=set(), deadline=3.5, clock=clock)
    assert r["finish"] == "time" and len(r["ids"]) == 3, r
    assert net.generate([4, 5, 6], 20, eos=set(), deadline=0.5, clock=clock) == {
        "ids": [], "finish": "time", "prefilled": 0, "reused": 0}


def test_the_slot_loads_and_runs_a_checkpoint():
    from engine.generative.model import LlamaModel
    d = tiny_dir(seed=6)
    m = LlamaModel("tiny", d, max_ctx=512, claim_gib=0.1)
    asked = []

    def planner(name, **kw):
        asked.append((name, kw))
        return {"allowed": True, "reason": None, "role": "narrator"}

    rec = []
    mem = iter([40.0, 39.5])
    slot = Slot(m, role="narrator", gate=threading.Lock(), planner=planner, log=lambda s: None,
                contract=lambda f, n: schema.role_errors(f, n), journal=rec.append,
                mem_available=lambda: next(mem))
    assert slot.state == "not_loaded" and m.tokenizer_sha256 != "none"
    msgs = [{"role": "system", "content": "You narrate."}, {"role": "user", "content": "hi"}]
    assert slot.run(msgs, max_new=8, budget_ms=1000) == {"why": "loading"} or \
        slot.state == "loaded"
    for _ in range(200):
        if slot.state == "loaded":
            break
        time.sleep(0.02)
    assert slot.state == "loaded" and asked[0][0] == "tiny", (slot.state, slot.error)
    assert rec and rec[0]["kind"] == "loaded" and rec[0]["took_gib"] == 0.5
    ids = m.encode(msgs)
    assert m.tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert m.tok.decode(ids).startswith("<s><|im_start|>system\nYou narrate.<|im_end|>")
    r = slot.run(msgs, max_new=8, budget_ms=5000)
    assert r["finish"] in ("stop", "length") and r["tokens"] <= 8 and isinstance(r["text"], str)
    # the second call reuses the system turn's rows
    r2 = slot.run(msgs[:1] + [{"role": "user", "content": "another event"}], max_new=8,
                  budget_ms=5000)
    assert r2["reused"] > 10, r2
    st = slot.status()
    assert st["device_bytes"] > 0 and st["precision"].startswith("float32")
    assert st["weight_hash"] and st["calls"]
    assert slot.unload() and slot.state == "not_loaded" and m.net is None


def test_minicpm5_template_thinking_off():
    d = os.path.expanduser("~/.qwen38-spark-engine/models/minicpm5-2b")
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        print("    (skipped: no MiniCPM5-2B on this machine)")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d)
    msgs = narr_mod.messages({"kind": "route", "text": "deciding: code_edit, high effort",
                              "facts": {"cls": "code_edit", "effort": "high effort"}})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), text[-80:]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert ids[0] == tok.bos_token_id and len(ids) < 400, len(ids)


# ------------------------------------------------------------------ the narrator

def fake_slot(reply=None, delay_s=0.0, planner=None, gate=None):
    m = FakeGenerative(reply=reply, delay_s=delay_s)
    s = Slot(m, role="narrator", gate=gate or threading.Lock(), planner=planner,
             log=lambda s: None)
    return s, m


def loaded(reply=None, delay_s=0.0, role=None):
    s, m = fake_slot(reply, delay_s)
    s.load(wait=True)
    assert s.state == "loaded"
    return narr_mod.Narrator(s, role=lambda: role or {"mode": "on",
                                                      "latency_budget_ms": 150}), m


ENTRY = {"kind": "route", "text": "deciding: code_edit, high effort",
         "facts": {"cls": "code_edit", "effort": "high effort"}}


def said(kind: str) -> str:
    """The fake model's line for an event of `kind`."""
    return "now " + " ".join(narr_mod.EVENTS[kind].split()[:5])


def test_the_prompt_and_the_clean():
    msgs = narr_mod.messages({"kind": "check", "text": "checking the tool call",
                              "facts": {"tools": ["edit", "bash"], "files": ["src/lexer.ts"],
                                        "token": "ghp_" + "a" * 40}})
    assert msgs[0] == {"role": "system", "content": narr_mod.SYSTEM}
    u = msgs[1]["content"]
    assert u.startswith("Event: " + narr_mod.EVENTS["check"] + "\nFacts: tools: edit, bash; "
                        "files: src/lexer.ts; token: ") and u.endswith("\nLine:"), u
    assert "ghp_" not in u and "checking the tool call" not in u, u     # redacted; no template
    # the facts in words: the router's label for a class, the effort's words, the facts' names
    u = narr_mod.messages(ENTRY, {"code_edit": "Code edit"})[1]["content"]
    assert "Facts: task: Code edit; effort: high effort" in u, u
    u = narr_mod.messages({"kind": "route", "text": "x", "facts": {
        "cls": "three_d", "effort": "xhigh effort"}})[1]["content"]
    assert "Facts: task: three d; effort: very high effort" in u, u
    assert narr_mod.messages({"kind": "compaction", "text": "compaction seen"})[1][
        "content"].endswith("Facts: none\nLine:")
    assert narr_mod.clean('"Checking the edit to src/lexer.ts."') == (
        "Checking the edit to src/lexer.ts", None)
    assert narr_mod.clean("Line one\nline two") == ("Line one", None)
    assert narr_mod.clean("Line: Pulling in the skill") == ("Pulling in the skill", None)
    assert narr_mod.clean("") == (None, "empty")
    assert narr_mod.clean(" ".join(["word"] * 15)) == (None, "words")
    assert narr_mod.clean("a <think> tag") == (None, "markup")
    assert narr_mod.clean("› nested marker") == (None, "markup")


def test_every_failure_is_the_template():
    n, m = loaded()
    line, info = n.line(ENTRY, budget_ms=500)
    assert line == said("route") and info["why"] is None, info
    # the first line of an unloaded model starts its load and is the template
    s, _ = fake_slot(planner=lambda name, **kw: {"allowed": True, "reason": None})
    n2 = narr_mod.Narrator(s)
    assert n2.line(ENTRY, budget_ms=150) == (None, {"why": "loading"})
    for _ in range(100):
        if s.state == "loaded":
            break
        time.sleep(0.01)
    assert n2.line(ENTRY, budget_ms=150)[0] is not None
    # refused by the planner: the template, and the refusal is reused for a minute
    calls = []
    s, _ = fake_slot(planner=lambda name, **kw: calls.append(1) or
                     {"allowed": False, "reason": "refused_memory"})
    n3 = narr_mod.Narrator(s)
    assert n3.line(ENTRY, budget_ms=150)[1]["why"] == "refused"
    assert n3.line(ENTRY, budget_ms=150)[1]["why"] == "refused" and len(calls) == 1
    # the token cap, too many words, markup, nothing, an error
    for reply, cap, why in ((" ".join(f"w{i}" for i in range(30)), 24, "length"),
                            (" ".join(f"w{i}" for i in range(16)), 24, "words"),
                            ("see <think> here", 24, "markup"), ("", 24, "empty")):
        n, m = loaded(reply=lambda msgs, r=reply: r)
        line, info = n.line(ENTRY, budget_ms=500, max_tokens=cap)
        assert line is None and info["why"] == why, (why, info)
    n, m = loaded()
    m.fail = True
    assert n.line(ENTRY, budget_ms=150)[1]["why"] == "error"
    # busy: a second call while one runs
    n, m = loaded(delay_s=0.05)
    got = []
    t = threading.Thread(target=lambda: got.append(n.line(ENTRY, budget_ms=2000)))
    t.start()
    time.sleep(0.02)
    assert n.line(ENTRY, budget_ms=150)[1]["why"] == "busy"
    t.join()
    assert got[0][0] is not None
    # three errors in a minute: the model is unloaded and failed
    n, m = loaded()
    m.fail = True
    for _ in range(3):
        n.line(ENTRY, budget_ms=150)
    for _ in range(100):
        if n.slot.state == "failed":
            break
        time.sleep(0.01)
    assert n.slot.state == "failed" and n.line(ENTRY, budget_ms=150)[1]["why"] == "failed"


def test_the_budget_is_hard():
    # 8 words at 40 ms each, a 150 ms budget: cut after the 4th word (the check before a step)
    n, m = loaded(reply=lambda msgs: "one two three four five six seven eight", delay_s=0.04)
    t0 = time.monotonic()
    line, info = n.line(ENTRY, budget_ms=150)
    took = (time.monotonic() - t0) * 1e3
    assert line is None and info["why"] == "time", info
    assert 150 <= took < 150 + 40 + 30, took                 # the budget and at most one step
    # a request spends at most REQUEST_LINES budgets, then its lines are templates
    lines = n.for_request()
    whys = [lines(dict(ENTRY))[1]["why"] for _ in range(narr_mod.REQUEST_LINES + 2)]
    assert whys[:narr_mod.REQUEST_LINES] == ["time"] * narr_mod.REQUEST_LINES, whys
    assert whys[-1] == "request_budget", whys


def test_role_mode_off_and_shadow():
    n, _ = loaded(role={"mode": "off", "latency_budget_ms": 150})
    assert n.for_request() is None
    n, m = loaded(role={"mode": "shadow", "latency_budget_ms": 150})
    ui = states.States(narrator=n.for_request())
    ui.add("route", cls="code_edit", effort="high effort")
    assert ui.take() == [states.frame("deciding: code_edit, high effort")]
    e = ui.log[0]
    assert e["summary"] is None and e["narrator"]["shadow"] == said("route")
    # on: the narrated frame goes out, once (a second render does not ask again)
    n, m = loaded()
    ui = states.States(narrator=n.for_request())
    ui.add("check", {"tools": ["bash"]})
    shown = ui.take()
    assert shown == [narr_mod.frame(said("check"))] and len(m.calls) == 1
    assert ui.shown(ui.log[0]) == shown[0] and len(m.calls) == 1
    assert ui.log[0]["facts"] == {"tools": ["bash"]}


def test_a_narrated_line_is_stripped():
    a = narr_mod.frame("Reading the lexer before editing it")
    b = states.frame("checking the tool call")
    r = "The user wants a lexer.\nI read it first."
    for c in (a + r, r + a, a + r + b, r[:7] + a + r[7:], b + a + r):
        assert states.strip_text(c) == r, c
    for c in (a + r, r + a):
        assert states.strip_text(c.strip()).strip() == r.strip()
    # Qwen's own lines that start with the marker stay, with or without a word joiner inside
    for c in ("x\n› a quote of mine\n\ny", "x\n› mine⁠ and more\n\ny"):
        assert states.strip_text(c) == c
    msgs, n = states.strip_messages([{"role": "assistant", "content": "",
                                      "reasoning_content": a + r}])
    assert msgs[0]["reasoning_content"] == r and n == len(a)


# ------------------------------------------------------------------ the handler

def narrator_on(cond) -> None:
    """The roles document's narrator switched on by hand (0.11.3: off by default)."""
    _, roles = cond.store.get("roles")
    roles["narrator"] = dict(roles.get("narrator") or schema.seed("roles")["narrator"], mode="on")
    cond.store.put("roles", None, roles, note="narrator on", author="test")


def _serve(narrator: str | None, slot=None, states_on: str = "on"):
    import test_ui_states as ui
    from server import app
    cond, seen = ui.serve(states_on)
    narrator_on(cond)
    if narrator is not None:
        for name in ("coding", "chat"):
            _, doc = cond.store.get("profiles", name)
            doc["ui"] = dict(doc["ui"], narrator=narrator)
            cond.store.put("profiles", name, doc, note="narrator", author="test")
    if slot is not None:
        app.STATE["narrator"] = narr_mod.Narrator(
            slot, role=lambda: narr_mod.role_doc(app.STATE),
            labels=lambda: narr_mod.class_labels(app.STATE))
    return cond, seen


def _call():
    return ("FAKE_DROP_CLOSER FAKE_CALL_JSON " +
            json.dumps({"name": "bash", "arguments": {"command": "ls -la"}}))


def test_off_is_byte_identical():
    import test_ui_states as ui
    _serve(None)
    base = ui._strip_ids(ui.opencode(_call())["raw"])
    for narrator, with_slot in (("off", True), (None, True), ("on", False)):
        s = None
        if with_slot:
            s, _ = fake_slot()
            s.load(wait=True)
        _serve(narrator, s)
        got = ui._strip_ids(ui.opencode(_call())["raw"])
        assert got == base, (narrator, with_slot)
        if s is not None:
            assert not s.model.calls, "the narrator was asked with its switch off"
    # the narrator on, the state lines off: no line, nothing asked
    s, _ = fake_slot()
    s.load(wait=True)
    _serve("on", s, states_on="off")
    off = ui.opencode(_call())
    assert M not in off["raw"] and not s.model.calls


def test_on_the_lines_are_the_narrators():
    import test_ui_states as ui
    _serve("off")
    off = ui.opencode(_call())
    s, m = fake_slot()
    s.load(wait=True)
    cond, _ = _serve("on", s)
    on = ui.opencode(_call())
    got = ui.lines(on["reasoning"])
    assert got[0] == said("route") + narr_mod.SENTINEL, got
    assert said("check") + narr_mod.SENTINEL in got, got
    assert said("repair") + narr_mod.SENTINEL in got, got
    # the facts reached the model: the class's label and the effort, the check's tool
    asked = [c["messages"][1]["content"] for c in m.calls]
    assert "Facts: task: Code edit; effort: very high effort" in asked[0], asked
    assert any("tools: bash" in a for a in asked), asked
    # the rest is the stream with the template lines, and the same usage
    assert states.strip_text(on["reasoning"]) == states.strip_text(off["reasoning"])
    assert on["content"] == off["content"]
    ui.same_usage(on, off)
    rid = json.loads(on["raw"].splitlines()[0][6:])["id"]
    row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
    first = row["extra"]["states"][0]
    assert first["text"] == "deciding: code_edit, xhigh effort"
    assert first["summary"] == said("route")
    assert first["narrator"]["why"] is None and "ms" in first["narrator"]


def test_on_a_replayed_reasoning_reaches_the_engine_without_its_lines():
    import test_ui_states as ui
    s, _ = fake_slot()
    s.load(wait=True)
    prompts = {}
    for shape in ("clean", "lines", "trimmed"):
        _, seen = _serve("on", s)
        first = ui.opencode("Write the lexer.")
        r = first["reasoning"]
        assert narr_mod.SENTINEL in r
        sent = {"lines": r, "trimmed": r.strip(), "clean": states.strip_text(r)}[shape]
        asst = {"role": "assistant", "content": first["content"], "reasoning_content": sent}
        seen.clear()
        ui.opencode("Now add a test.",
                    history=[{"role": "user", "content": "Write the lexer."}, asst])
        prompts[shape] = seen[-1]
    assert prompts["lines"] == prompts["clean"] == prompts["trimmed"]


def test_on_non_streamed():
    import test_ui_states as ui
    s, _ = fake_slot()
    s.load(wait=True)
    _serve("on", s)
    on = ui.opencode("Write the lexer.", stream=False)
    assert on["reasoning"].startswith(narr_mod.frame(said("route")))


def test_a_failing_narrator_streams_the_templates():
    import test_ui_states as ui
    _serve("off")
    off = ui._strip_ids(ui.opencode(_call())["raw"])
    s, m = fake_slot()
    s.load(wait=True)
    m.fail = True
    _serve("on", s)
    assert ui._strip_ids(ui.opencode(_call())["raw"]) == off
    # an unloaded narrator whose load the planner refuses: the same
    s, m = fake_slot(planner=lambda name, **kw: {"allowed": False, "reason": "refused_hold"})
    _serve("on", s)
    assert ui._strip_ids(ui.opencode(_call())["raw"]) == off and not m.calls


# ------------------------------------------------------------------ the contract, the residency

def test_the_role_signs_the_contract():
    roles = schema.seed("roles")
    r = roles["narrator"]
    assert r["kind"] == "generative" and r["window"] == "between_rounds"
    assert r["writes"] == "state_lines" and r["preemptible"] is True
    # SRV-534: off by default
    assert r["latency_budget_ms"] == 150 and r["contract"] == "generate" and r["mode"] == "off"
    assert schema.doc_errors("roles", roles) == []
    assert schema.role_errors(FakeGenerative().role(), "fake") == []
    from engine.generative.model import LlamaModel
    assert schema.role_errors(LlamaModel("minicpm5-2b", "/nonexistent").role(), "m") == []
    bad = dict(r, kind="clerk")
    errs = schema.role_rules(bad)
    assert any("between_rounds is for a generative" in e for e in errs), errs
    assert any("state_lines are written by a generative" in e for e in errs), errs
    assert any("1000 ms" in e for e in schema.role_rules(dict(r, latency_budget_ms=1500)))
    assert any("honesty" in e for e in schema.role_rules(dict(r, writes="client_visible")))
    # the box's roles document of 2026-10-01 (version 2, before SRV-133 and the narrator)
    v2 = {"decider": {"endpoint": "self", "model": "auto", "contract": "jev", "timeout_ms": 400,
                      "memory_gb": 5.1, "mode": "on"}}
    doc, applied = schema.upgrade("roles", v2)
    assert doc["narrator"] == roles["narrator"] and "narrator=built-in" in applied
    assert schema.doc_errors("roles", doc) == []
    # a document that has it keeps its own
    mine = copy.deepcopy(roles)
    mine["narrator"]["latency_budget_ms"] = 400
    assert schema.upgrade("roles", mine)[0]["narrator"]["latency_budget_ms"] == 400
    # the profile's switch
    prof = schema.seed("profiles", "coding")
    prof["ui"] = {"states": "on", "narrator": "on"}
    assert schema.validate(prof, schema.SCHEMAS["profiles"]) == []
    prof["ui"]["narrator"] = "maybe"
    assert schema.validate(prof, schema.SCHEMAS["profiles"])


def test_the_role_and_the_residency_planner():
    doc, roles = schema.seed("residency"), schema.seed("roles")
    assert doc["claims"]["narrator"]["gib"] == 6.0 and "narrator" in doc["order"]
    # 2026-10-01 at start: MemAvailable 55.2 GiB with the engine at 57.45 of its 77 GiB peak; the
    # first Laya checkpoint took 4.4 GiB of it, the decider's claim is 8.2 (three)
    d = residency.check("narrator", 55.2 - 4.4, doc, roles, engine_peak_gib=77.0,
                        engine_now_gib=57.45)
    assert d["allowed"] and d["after_gib"] == round(55.2 - 4.4 - 19.55 - 6.0, 2), d
    d = residency.check("narrator", 55.2 - 8.2, doc, roles, engine_peak_gib=77.0,
                        engine_now_gib=57.45)
    assert d["allowed"] and d["after_gib"] < 22.0, d             # all three: 0.45 GiB to spare
    assert residency.check("narrator", 40.0, doc, roles, engine_peak_gib=77.0,
                           engine_now_gib=57.45)["reason"] == "refused_memory"
    # the planner the slot asks: never beside a decision model's load
    from server import decider_api

    class Reg:
        def loading(self):
            return True

    state = {"deciders": Reg()}
    os.environ["FAKE_MEM_AVAILABLE_GIB"] = "64"
    try:
        check = decider_api.planner(state, fake=True, role="narrator")
        d = check("minicpm5-2b", loading=False, service_start=False, hold_restart=True)
        assert d["reason"] == "refused_load_in_progress" and d["role"] == "narrator", d
        state["deciders"] = None
        d = check("minicpm5-2b", loading=False, service_start=False, hold_restart=True)
        assert d["allowed"] and d["claim_gib"] == 6.0, d
        # and a decider load waits while the narrator loads
        s, _ = fake_slot()
        s.state = "loading"
        state["narrator"] = narr_mod.Narrator(s)
        d = decider_api.planner(state, fake=True)("laya-english", loading=False,
                                                   service_start=False, hold_restart=True)
        assert d["reason"] == "refused_load_in_progress", d
    finally:
        os.environ.pop("FAKE_MEM_AVAILABLE_GIB", None)


def test_the_ceiling_leaves_the_narrator_out():
    from server import memlimits
    s, m = fake_slot()
    state = {"narrator": narr_mod.Narrator(s)}
    assert memlimits._role_bytes(state) == 0
    s.load(wait=True)
    m.device_bytes = lambda: 5 << 30
    assert memlimits._role_bytes(state) == 5 << 30
    s.unload()
    assert memlimits._role_bytes(state) == 0


def test_the_routes():
    import io
    from server import app
    from server.conductor import routes
    import test_ui_states as ui
    cond, _ = ui.serve("on")
    app.STATE.pop("narrator", None)

    class H:
        def __init__(self, path):
            self.path, self.wfile, self.sent = path, io.BytesIO(), None

        def _allowed(self, rule):
            return True

        def _json(self, status, payload, extra_headers=()):
            self.sent = (status, payload)

    def call(method, path, body=None):
        h = H(path)
        routes.handle(h, cond, method, body or {}, None)
        return h.sent

    st, out = call("GET", "/v1/conductor/narrator/status")
    assert st == 200 and out["enabled"] is False and "QWEN38_NARRATOR_MODEL" in out["why"]
    assert call("POST", "/v1/conductor/narrator/try", {"events": [{"kind": "route"}]})[0] == 404
    s, m = fake_slot(planner=lambda name, **kw: {"allowed": True, "reason": None})
    app.STATE["narrator"] = narr_mod.Narrator(s, role=lambda: narr_mod.role_doc(app.STATE))
    # SRV-534: the role off (the default) loads nothing and keeps the keeper away
    app.STATE.pop("narrator_keep", None)
    st, out = call("POST", "/v1/conductor/narrator/load", {"service_start": True})
    assert st == 409 and out["error"]["type"] == "role_off", out
    assert s.state == "not_loaded" and not app.STATE.get("narrator_keep")
    narrator_on(cond)
    st, out = call("POST", "/v1/conductor/narrator/load")
    assert st == 202, out
    for _ in range(100):
        if s.state == "loaded":
            break
        time.sleep(0.01)
    st, out = call("POST", "/v1/conductor/narrator/try", {
        "events": [{"kind": "route", "facts": {"cls": "code_edit", "effort": "high effort"}},
                   {"kind": "cap", "facts": {"cap": "16,384"}},
                   {"kind": "nope", "facts": {}}], "budget_ms": 2000})
    assert st == 200, out
    a, b, c = out["lines"]
    assert a["template"] == "deciding: code_edit, high effort" and a["line"] == said("route")
    assert b["shown"] == said("cap") and "error" in c
    assert call("POST", "/v1/conductor/narrator/try", {"events": [{"kind": "route"}],
                                                        "budget_ms": 0})[0] == 400
    st, out = call("GET", "/v1/conductor/narrator/status")
    assert out["enabled"] and out["slot"]["state"] == "loaded" and out["settings"] == {
        "mode": "on", "budget_ms": 150}
    health = cond.roles_health(force=True)["narrator"]
    assert health["up"] and "fake-narrator: loaded" in health["detail"], health
    t = cond.role_test("narrator")
    assert t["ok"] and t["answer"]["line"] == said("route"), t
    assert call("POST", "/v1/conductor/narrator/unload")[0] == 200 and s.state == "not_loaded"


def test_the_small_models_never_take_the_served_weight_overrides():
    """The narrator / shadow load (engine/generative/qwen35.py) turns NVFP4, the FP8 head and EXL3 off
    by argument: QWEN38_EXL3 of an EXL3 weight profile names the 27B's checkpoint (2026-10-09:
    'EXL3Block' object has no attribute 'to' on every narrator load)."""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "engine/generative/qwen35.py")).read()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "Weights"]
    assert calls, "no Weights(...) load found"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        for name in ("nvfp4", "fp8_head", "exl3"):
            assert name in kw and isinstance(kw[name], ast.Constant) and kw[name].value == "", (name, ast.dump(c))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
