"""Structured outputs, on a CPU: the automaton, the masks, and the loop under a constraint.

  * the regex engine accepts exactly what Python's `re.fullmatch` accepts, over random strings;
  * a code point range becomes UTF-8 byte sequences that cover it exactly (brute force);
  * a JSON schema compiles to a language that holds its valid instances -- compact or indented --
    and none of the invalid ones;
  * a state's token mask is exactly the brute-force walk of every token's bytes;
  * in the served loop a constrained greedy request decodes to the same tokens with no drafter, a
    chain and a tree (drafting the unconstrained continuation, so the mask is where drafts die),
    and every answer that ends is in the language;
  * reasoning is not constrained; the closing tag starts the constraint;
  * through the handler: json_schema, json_object, a choice list and a regex give answers in their
    languages on both transports; a bad schema and a constraint the vocabulary cannot write are
    400s that say so; `--no-structured-outputs` refuses as before;
  * SRV-619, SRV-620: a document with tools under auto is the calls or the document alone (the
    language, the call markers' own ids, the handler on both transports); tool_choice none is the
    document, required the calls; the model's own first answer token picks the branch (the
    product automaton's sides, the row's argmax, exact under speculation).

Run: python tests/test_grammar.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import grammar as G  # noqa: E402
from server import app  # noqa: E402
import test_compat as C  # noqa: E402

PATTERNS = [
    r"a|bc", r"(ab)*c?", r"[0-9]{2,4}", r"x[a-c]+y|z{0,2}", r"(?:ab|a)(?:bc|c)", r"[^ab]\d",
    r"\w+\s\w*", r"(a|b){3}", r"[a-e]{1,}q{2,}", r"\.[\]\\]", r"(?:[a-z]\d)?[A-Z]",
]


def _match(p, s):
    d = G.compile_regex(p)
    return bool(d.accept[d.walk(d.start, s.encode())])


def test_the_regex_engine_agrees_with_python():
    rng = random.Random(3)
    alphabet = "abcxyzqAZ019 .]\\\t"
    n = 0
    for p in PATTERNS:
        d = G.compile_regex(p)
        py = re.compile(p)
        for _ in range(400):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
            want = py.fullmatch(s) is not None
            got = bool(d.accept[d.walk(d.start, s.encode())])
            assert got == want, (p, s, got, want)
            n += 1
    for bad in ("(a", "a{3,2}", "a*?", "(?=a)", "\\q", "[a", "^a"):
        try:
            G.compile_regex(bad)
            raise AssertionError(f"{bad!r} must be refused")
        except G.GrammarError:
            pass
    return f"{len(PATTERNS)} patterns x 400 random strings == re.fullmatch; 7 bad patterns refused"


def test_utf8_ranges_cover_exactly():
    rng = random.Random(5)
    for _ in range(60):
        a = rng.choice([0, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, rng.randint(0, 0x10FFFF)])
        b = min(0x10FFFF, a + rng.choice([0, 1, 63, 64, 2047, 70000]))
        seqs = G.utf8_sequences(a, b)
        for cp in list(range(max(0, a - 3), a + 4)) + list(range(b - 3, b + 4)) + \
                [rng.randint(a, b) for _ in range(50)]:
            if cp < 0 or cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
                continue
            enc = chr(cp).encode()
            hit = any(len(s) == len(enc) and all(x <= y <= z for (x, z), y in zip(s, enc))
                      for s in seqs)
            assert hit == (a <= cp <= b), (hex(a), hex(b), hex(cp))
    d = G.compile_regex("[^a]")
    for ch in ("é", "🤦", "中", "\x7f"):
        assert d.accept[d.walk(0, ch.encode())], ch
    half = "🤦".encode()[:2]
    assert d.walk(0, half) != d.dead and not d.accept[d.walk(0, half)], "half a character is a live prefix"
    return "60 random ranges, boundaries and samples; half a 4-byte character is live, not accepting"


SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "city": {"enum": ["Berlin", "Paris"]}, "ok": {"type": "boolean"},
    "score": {"type": "number"}, "extra": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/p"}]}},
    "required": ["name", "age"], "$defs": {"p": {"type": "object", "properties": {
        "x": {"type": "integer"}}, "required": ["x"]}}}


def test_a_schema_holds_its_instances_and_nothing_else():
    d = G.compile_regex(G.WS + G.schema_regex(SCHEMA) + G.WS)
    good = [{"name": "Ann", "age": 31}, {"name": "Ann", "age": -2, "city": "Paris"},
            {"name": "a\"b\\u00e9", "age": 0, "tags": ["x", "y"], "ok": True, "score": 1.5e-3,
             "extra": {"x": 4}},
            {"name": "Ünïcødé 🙂", "age": 7, "extra": None}]
    bad = [{"age": 31}, {"name": "Ann"}, {"name": "Ann", "age": 1.5},
           {"name": "Ann", "age": 1, "city": "Rome"}, {"name": "Ann", "age": 1, "tags": ["a"] * 4},
           {"name": "Ann", "age": 1, "unknown": 1}, {"age": 1, "name": "Ann"},
           {"name": "Ann", "age": 1, "extra": {"y": 1}}]
    for inst in good:
        for text in (json.dumps(inst), json.dumps(inst, indent=2), json.dumps(inst, ensure_ascii=False)):
            assert d.accept[d.walk(0, text.encode())], text
    for inst in bad:
        assert not d.accept[d.walk(0, json.dumps(inst).encode())], inst
    for bad_schema in ({"type": "object", "properties": {"a": {"$ref": "#/$defs/a"}},
                        "$defs": {"a": {"$ref": "#/$defs/a"}}}, {"type": "tuple"}, [1]):
        try:
            G.schema_regex(bad_schema)
            raise AssertionError(f"{bad_schema} must be refused")
        except G.GrammarError:
            pass
    obj = G.compile_regex(G.json_object_regex(3))
    for text in ('{}', '{"a": [1, {"b": null}], "c": "d"}', ' {"a":1} '):
        assert obj.accept[obj.walk(0, text.encode())], text
    for text in ('[]', '"x"', '{"a": 1', '{"a": [[[[1]]]]}'):
        assert not obj.accept[obj.walk(0, text.encode())], text
    return (f"{len(good)} instances x 3 spellings in, {len(bad)} out; a $ref cycle and an unknown "
            f"type refused; json_object: {obj.states} states")


def _toy_vocab():
    rng = random.Random(9)
    toks = [bytes([b]) for b in range(32, 127)] + [None]          # the last one: special
    for _ in range(300):
        toks.append("".join(rng.choice('{}":, abc123') for _ in range(rng.randint(2, 5))).encode())
    toks += ["é".encode(), "é".encode()[:1], "é".encode()[1:]]
    return G.Vocab(toks, len(toks) + 3), toks


def test_a_mask_is_the_brute_force_walk():
    v, toks = _toy_vocab()
    d = G.compile_regex(r'\{"a": ?[0-9]+(, ?"b": ?"[a-cé]*")?\}')
    for state in range(d.states):
        m = v.mask(d, state)
        for i, b in enumerate(toks):
            want = b is not None and d.walk(state, b) != d.dead
            assert m[i] == want, (state, i, b)
        assert not m[len(toks):].any(), "ids past the tokenizer are never allowed"
    return f"{d.states} states x {len(toks)} tokens, split UTF-8 tokens included"


# ------------------------------------------------------------------ the loop under a constraint

class TreeOracle(C.TreeOracle):
    pass


def _grammar(pattern):
    return G.Grammar(pattern, G.Vocab.from_tokenizer(C.Tok(), 97))


def _run(pattern, drafter=None, tree=False, n=40, in_think=False, prompt=None, bias=None):
    C.serve(drafter, tree=tree)
    cons = G.Constraint(_grammar(pattern), {C.EOS}, "cpu", think_end=90, in_think=in_think)
    pen = cons
    if bias:
        from engine.penalty import PenaltySpec, PenaltyState
        pen = G.LogitChain([PenaltyState(PenaltySpec(bias=bias), 97, "cpu"), cons])
    return list(app.generate_stream(torch.tensor(prompt or C.PROMPT), n, {C.EOS}, pen=pen))


LANG = r'\{"k": "[a-z]{2,6}", "n": [0-9]{1,3}\}'


def test_a_constrained_request_is_exact_under_speculation():
    free = C._run(n=40)
    ref = _run(LANG)
    text = C.Tok().decode(ref)
    assert re.fullmatch(LANG, text) and ref[-1] == C.EOS, (text, ref[-3:])
    runs, took = {}, {}
    for label, dr, tree in (("chain, free drafts", C.Oracle(free), False),
                            ("chain, constrained drafts", C.Oracle(ref), False),
                            ("tree, free drafts", C.TreeOracle(free), True),
                            ("tree, constrained drafts", C.TreeOracle(ref), True)):
        runs[label] = _run(LANG, dr, tree)
        took[label] = C._accepted()
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less constrained run {ref}: {bad}"
    assert took["chain, constrained drafts"] and took["tree, constrained drafts"], took
    return f"{text!r}: 4 speculative runs == no drafter; drafts accepted {took}"


def test_reasoning_is_not_constrained():
    # a prompt that ends inside the reasoning block: nothing is masked until the closing token
    # (id 90 here) -- a forced close is exactly that token -- and then the language starts
    cons = G.Constraint(_grammar(LANG), {C.EOS}, "cpu", think_end=90, in_think=True)
    cons.seed([1, 2])
    row = torch.zeros(97)
    cons.apply_single(row)
    assert torch.isfinite(row).all(), "inside the reasoning every token is allowed"
    lg = torch.zeros(3, 97)
    cons.apply_chain(lg, [5, 90])
    assert torch.isfinite(lg[:2]).all() and not torch.isfinite(lg[2]).all()
    assert torch.isfinite(lg[2][(ord("{") - 32) % 96]), "after the close, the answer opens with {"
    cons.commit([5, 90])
    assert not cons.thinking
    return "rows before the close are free, the row after it is the language's first state"


def test_the_impossible_state_fails_loudly():
    g = _grammar("é")                         # no token of the toy vocabulary writes 0xC3
    try:
        g.mask(g.dfa.start, frozenset({C.EOS}), "cpu")
        raise AssertionError("an empty mask must raise")
    except G.GrammarError as exc:
        assert "no legal token" in str(exc)
    return "a state no token can leave raises GrammarError"


def _ask(stream=False, **extra):
    C.serve()
    body = dict({"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 60,
                 "stream": stream}, **extra)
    head, raw = C.Req("/v1/chat/completions", body).response()
    if head.startswith("HTTP/1.1 200") and stream:
        ev = C._events(raw)
        text = "".join(c["delta"].get("content") or "" for e in ev for c in e["choices"])
        finish = [c["finish_reason"] for e in ev for c in e["choices"] if c["finish_reason"]]
        err = [e.get("error") for e in ev if e.get("error")]
        return head, text, finish[-1] if finish else None, err
    if head.startswith("HTTP/1.1 200"):
        c = json.loads(raw)["choices"][0]
        return head, c["message"]["content"], c["finish_reason"], []
    return head, raw, None, []


def test_the_handler_serves_every_constraint_form():
    schema = {"type": "object", "properties": {"k": {"type": "string", "maxLength": 4},
                                               "n": {"type": "integer"}}, "required": ["k", "n"]}
    forms = {
        "json_schema": ({"response_format": {"type": "json_schema",
                                             "json_schema": {"name": "s", "schema": schema}}},
                        lambda t: set(json.loads(t)) >= {"k", "n"}),
        "json_object": ({"response_format": {"type": "json_object"}},
                        lambda t: isinstance(json.loads(t), dict)),
        "choice": ({"structured_outputs": {"choice": ["red", "green", "blue"]}},
                   lambda t: t in ("red", "green", "blue")),
        "regex": ({"structured_outputs": {"regex": r"[0-9]{3}-[a-z]{2}"}},
                  lambda t: re.fullmatch(r"[0-9]{3}-[a-z]{2}", t)),
    }
    seen = {}
    for label, (extra, ok) in forms.items():
        for stream in (False, True):
            head, text, finish, err = _ask(stream, **extra)
            assert head.startswith("HTTP/1.1 200"), (label, head, text[:200])
            if finish == "stop":
                assert ok(text), (label, stream, text)
            else:
                assert finish == "length", (label, finish)
            seen[f"{label}/{'stream' if stream else 'json'}"] = finish
    return f"8 answers in their languages ({sorted(set(seen.values()))})"


def test_the_handler_refuses_what_it_cannot_serve():
    tools = [{"type": "function", "function": {"name": "f"}}]
    cases = [({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "x"}}}},
              "response_format"),
             ({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "x"}}},
               "tools": tools}, "response_format"),
             ({"structured_outputs": {"regex": "(a"}}, "structured_outputs"),
             ({"structured_outputs": {"grammar": "root ::= x"}}, "structured_outputs"),
             ({"structured_outputs": {"choice": ["a"]}, "response_format": {"type": "json_object"}},
              "structured_outputs")]
    for extra, param in cases:
        head, raw, _, _ = _ask(**extra)
        assert head.startswith("HTTP/1.1 400") and json.loads(raw)["error"]["param"] == param, \
            (extra, head, raw[:200])
    head, raw, _, _ = _ask(response_format={"type": "json_object"}, tools=tools, tool_choice="none")
    assert head.startswith("HTTP/1.1 200"), "tool_choice none takes the tools out of the request"
    # SRV-619: a model whose calls are not the XML format keeps the 400 with tools
    C.serve()
    app.STATE["tool_format"] = "pythonic"
    head, raw = C.Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                              "response_format": {"type": "json_object"},
                                              "tools": tools}).response()
    assert head.startswith("HTTP/1.1 400") and json.loads(raw)["error"]["param"] == "tools", raw
    # a constraint the vocabulary cannot write: non-streamed, a 400 that names it; streamed, the
    # stream closes with finish_reason error and the reason
    head, raw, _, _ = _ask(structured_outputs={"regex": "é+"})
    assert head.startswith("HTTP/1.1 400") and "no legal token" in raw, (head, raw[:300])
    head, text, finish, err = _ask(True, structured_outputs={"regex": "é+"})
    assert finish == "error" and err and "no legal token" in err[0]["message"], (finish, err)
    C.serve()
    app.STATE["structured_outputs"] = False
    head, raw = C.Req("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                              "response_format": {"type": "json_object"}}).response()
    assert head.startswith("HTTP/1.1 400"), "--no-structured-outputs refuses as rc5 did"
    return f"{len(cases)} refusals by name; the impossible constraint on both transports; the flag off"


def test_an_unconstrained_request_never_builds_a_vocabulary():
    C.serve()
    C.chat()
    assert "grammar_vocab" not in app.STATE, "a plain request must not pay for the constraint"
    return "no vocabulary, no grammar, the same processors"


def test_the_mask_cache_is_bounded_and_forgets_dropped_grammars():
    v, _ = _toy_vocab()
    old = G.MASK_CACHE
    G.MASK_CACHE = 5
    try:
        G._MASKS.clear()
        g = G.Grammar(r"[a-c]{8}", v)
        first = g.mask(0, frozenset(), "cpu")
        for s in range(1, 8):
            g.mask(s, frozenset(), "cpu")
        assert len(G._MASKS) == 5 and not g.cached(0, "cpu"), "the least recently used goes first"
        assert torch.equal(g.mask(0, frozenset(), "cpu"), first), "and comes back the same"
        G._GRAMMARS.clear()
        for i in range(3):
            G.grammar_for(f"x{i}", v, keep=2).mask(0, frozenset(), "cpu")
        live = {x.serial for x in G._GRAMMARS.values()}
        assert {k[0] for k in G._MASKS} - {g.serial} <= live, "a dropped grammar leaves no masks"
    finally:
        G.MASK_CACHE = old
        G._MASKS.clear()
        G._GRAMMARS.clear()
    return "5 masks kept of 8, LRU order; a grammar dropped from the cache takes its masks along"


def test_a_new_grammar_never_reads_a_freed_grammars_masks():
    """The masks were keyed by `id(grammar)`, and CPython hands a freed object's id to the next
    one: a grammar built after another was freed read its masks (a mask for a 97-token
    vocabulary on a 99-token one: `test_the_handler_forces_the_call` and two more failed in
    most runs on 0.12.2). Built and dropped in a loop, the ids repeat; the keys must not."""
    import gc
    G._MASKS.clear()
    G._GRAMMARS.clear()
    try:
        v, _ = _toy_vocab()
        seen_ids, serials = set(), set()
        reused = False
        for i in range(50):
            g = G.Grammar(r"[a-c]{4}" if i % 2 else r"[ab]{3}", v)
            reused |= id(g) in seen_ids
            seen_ids.add(id(g))
            assert g.serial not in serials
            serials.add(g.serial)
            want = g.vocab.mask(g.dfa, g.dfa.start)
            got = g.mask(g.dfa.start, frozenset(), "cpu").numpy()
            assert (got == want).all(), f"grammar {i} read another grammar's mask"
            del g
            gc.collect()
        return f"50 grammars, ids reused: {reused}, every mask its own"
    finally:
        G._MASKS.clear()
        G._GRAMMARS.clear()


def test_the_same_pattern_on_another_vocabulary_drops_the_old_masks():
    G._MASKS.clear()
    G._GRAMMARS.clear()
    try:
        v1, _ = _toy_vocab()
        v2, _ = _toy_vocab()
        g1 = G.grammar_for(r"[a-c]{4}", v1)
        g1.mask(g1.dfa.start, frozenset(), "cpu")
        g2 = G.grammar_for(r"[a-c]{4}", v2)
        assert g2 is not g1 and g2.vocab is v2
        assert not any(k[0] == g1.serial for k in G._MASKS), "the replaced grammar's masks left"
    finally:
        G._MASKS.clear()
        G._GRAMMARS.clear()


# ------------------------------------------------------------------ tool_choice as a mask

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
             "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "set", "parameters": {
             "type": "object", "properties": {"n": {"type": "integer"}, "mode": {"enum": ["a", "b"]},
                                              "tags": {"type": "array", "items": {"type": "string"}}},
             "required": ["n"]}}},
         {"type": "function", "function": {"name": "now"}}]


def test_the_closer_free_text_is_exactly_the_strings_without_it():
    d = G.compile_regex(G.avoiding("</parameter>"))
    rng = random.Random(1)
    for _ in range(20000):
        s = "".join(rng.choice("</parameter>xyz\n") for _ in range(rng.randint(0, 30)))
        if rng.random() < 0.2:
            cut = rng.randint(0, len(s))
            s = s[:cut] + rng.choice(["</parameter>", "</paramete", "<</parameter",
                                      "</param</parameter>"]) + s[cut:]
        assert bool(d.accept[d.walk(0, s.encode())]) == ("</parameter>" not in s), s
    return "20,000 random strings: accepted exactly when `</parameter>` is absent"


def test_the_tool_call_language():
    pat, typed = G.tool_call_regex(TOOLS, ["read_file", "set", "now"])
    assert typed == {"set": {"n", "tags"}}
    d = G.compile_regex(pat)
    ok = ["<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n"
          "</function>\n</tool_call>",
          "\n<tool_call>\n<function=set>\n<parameter=n>\n3\n</parameter>\n<parameter=mode>\nb\n"
          "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n"
          "</tool_call>",
          '<tool_call><function=set><parameter=n>-2</parameter><parameter=tags>["x", "y"]'
          "</parameter></function></tool_call>",
          "<tool_call>\n<function=read_file>\n<parameter=path>\na </par b\n</parameter>\n"
          "</function>\n</tool_call>"]
    bad = ["I will read it.<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n"
           "</parameter>\n</function>\n</tool_call>",
           "<tool_call>\n<function=write_file>\n</function>\n</tool_call>",
           "<tool_call>\n<function=set>\n<parameter=mode>\na\n</parameter>\n</function>\n"
           "</tool_call>",
           "<tool_call>\n<function=set>\n<parameter=n>\nthree\n</parameter>\n</function>\n"
           "</tool_call>",
           "<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n</parameter>\n"
           "<parameter=mode>\na\n</parameter>\n</function>\n</tool_call>",
           "<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n</parameter>\n</function>\n"
           "</tool_call> and then some text", ""]
    for t in ok:
        assert d.accept[d.walk(0, t.encode())], t
    for t in bad:
        assert not d.accept[d.walk(0, t.encode())], t
    one = G.compile_regex(G.tool_call_regex(TOOLS, ["now"], many=False)[0])
    two = "<tool_call><function=now></function></tool_call>"
    assert one.accept[one.walk(0, two.encode())] and not one.accept[one.walk(0, (two * 2).encode())]
    assert one.accept[one.walk(0, (two + "\n" * 8).encode())]
    assert one.walk(0, (two + "\n" * 9).encode()) == one.dead, "trailing whitespace is bounded"
    return (f"{len(ok)} calls in; prose first, an unknown function, a missing required parameter, "
            f"a mistyped integer, an undeclared parameter, trailing text and nothing at all out; "
            f"many=False: one call")


# an enum only: the random model, free to add digits to an integer, never stops writing one
PICK = [{"type": "function", "function": {"name": "pick", "parameters": {
            "type": "object", "properties": {"color": {"enum": ["red", "green"]}},
            "required": ["color"]}}},
        {"type": "function", "function": {"name": "stop", "parameters": {
            "type": "object", "properties": {}}}}]


def test_the_tool_constraint_is_exact_under_speculation():
    # the random model, left a choice of whitespace between the tags, takes a space forever; a bias
    # against the space (id 0) walks it through the call -- and chains a bias before the mask
    pat, _ = G.tool_call_regex(PICK, ["pick"], many=False)
    free = C._run(n=120)
    ref = _run(pat, n=120, bias={0: -50.0})
    text = C.Tok().decode(ref)
    assert ref[-1] == C.EOS and re.fullmatch(r"[ \t\n\r]*<tool_call>.*</tool_call>[ \t\n\r]*",
                                             text), text
    runs = {label: _run(pat, dr, tree, n=120, bias={0: -50.0})
            for label, dr, tree in (("chain, free drafts", C.Oracle(free), False),
                                    ("chain, constrained drafts", C.Oracle(ref), False),
                                    ("tree, free drafts", C.TreeOracle(free), True),
                                    ("tree, constrained drafts", C.TreeOracle(ref), True))}
    bad = {k: v for k, v in runs.items() if v != ref}
    assert not bad, f"differs from the drafter-less run: {bad}"
    return f"{text!r}: no drafter, chain and tree agree"


def test_the_handler_forces_the_call():
    from server.toolcall import parse_tool_calls  # noqa: F401 -- the path under test
    seen = {}
    for label, extra, allowed in (("required", {"tool_choice": "required"}, {"pick", "stop"}),
                                  ("named", {"tool_choice": {"type": "function",
                                                             "function": {"name": "pick"}}},
                                   {"pick"}),
                                  ("many", {"tool_choice": "required", "parallel_tool_calls": True},
                                   {"pick", "stop"})):
        for stream in (False, True):
            C.serve()
            body = dict({"messages": [{"role": "user", "content": "hello there"}],
                         "max_tokens": 120, "stream": stream, "tools": PICK,
                         "logit_bias": {"0": -50}}, **extra)
            head, raw = C.Req("/v1/chat/completions", body).response()
            assert head.startswith("HTTP/1.1 200"), (label, head, raw[:200])
            if stream:
                ev = C._events(raw)
                names = [d["function"]["name"] for e in ev for c in e["choices"]
                         for d in (c["delta"].get("tool_calls") or []) if d.get("function", {}).get("name")]
                finish = [c["finish_reason"] for e in ev for c in e["choices"] if c["finish_reason"]][-1]
                args = None
            else:
                c = json.loads(raw)["choices"][0]
                names = [t["function"]["name"] for t in c["message"].get("tool_calls") or []]
                finish = c["finish_reason"]
                args = [json.loads(t["function"]["arguments"]) for t in c["message"].get("tool_calls") or []]
            assert finish == "tool_calls", (label, stream, finish, raw[-300:])
            if finish == "tool_calls":
                assert names and set(names) <= allowed, (label, names)
                if label != "many":
                    assert len(names) == 1, ("one call unless parallel_tool_calls is true", names)
                for a in args or []:
                    if "color" in a:
                        assert a["color"] in ("red", "green"), a
            seen[f"{label}/{'stream' if stream else 'json'}"] = (finish, names)
    return f"{seen}"


# ------------------------------------------------------------------ SRV-619: calls or the document

SIG = {"type": "object", "properties": {"signal": {"enum": ["buy", "hold"]},
                                        "ok": {"type": "boolean"}},
       "required": ["signal", "ok"], "additionalProperties": False}
MARK = [{"type": "function", "function": {"name": "pick", "parameters": {
            "type": "object", "properties": {"color": {"enum": ["red", "green"]}},
            "required": ["color"]}}},
        {"type": "function", "function": {"name": "mark", "parameters": {
            "type": "object", "properties": {"level": {"enum": ["lo", "hi"]}},
            "required": ["level"]}}}]
JS = {"type": "json_schema", "json_schema": {"name": "sig", "strict": True, "schema": SIG}}


def _union(tools=TOOLS, rf=JS, **extra):
    a = C.compat.answer_constraint(dict({"messages": [], "tools": tools, "response_format": rf},
                                        **extra))
    return a, G.compile_regex(a.pattern)


def _in(d, text):
    return bool(d.accept[d.walk(d.start, text.encode())])


def test_the_calls_or_the_document_language():
    a, d = _union()
    assert a.kind == "response_format" and a.calls and a.types == {"set": {"n", "tags"}}, a
    call = ("<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n"
            "</function>\n</tool_call>")
    now = "<tool_call>\n<function=now>\n</function>\n</tool_call>"
    doc = '{"signal": "buy", "ok": true}'
    ok = [call, "\n\n" + call + "\n" + now, doc, "\n\n" + doc + "\n",
          '{\n  "signal": "hold",\n  "ok": false\n}']
    bad = ["```json\n" + doc + "\n```", "Here it is: " + doc, call + "\n" + doc, doc + "\n" + call,
           '{"signal": "sell", "ok": true}', '{"signal": "buy", "ok": true, "x": 1}',
           '{"ok": true, "signal": "buy"}', "I will call it.\n" + call, ""]
    for t in ok:
        assert _in(d, t), t
    for t in bad:
        assert not _in(d, t), t
    # one copy of the call language for many calls: the automaton is about the two parts' sum
    parts = (G.compile_regex(G.tool_call_regex(TOOLS, ["read_file", "set", "now"], many=False)[0])
             .states + G.compile_regex(G.WS + G.schema_regex(SIG) + G.WS).states)
    assert d.states <= parts + 16, (d.states, parts)
    _, one = _union(parallel_tool_calls=False)
    assert _in(one, call) and _in(one, doc) and not _in(one, call + "\n" + now)
    jo, dj = _union(rf={"type": "json_object"})
    assert jo.calls and _in(dj, call) and _in(dj, '{"any": [1, {"x": null}]}')
    assert not _in(dj, "[1, 2]") and not _in(dj, "```json\n{}\n```")
    return (f"{len(ok)} in (a call, two, the document compact and indented), {len(bad)} out "
            f"(fences, prose, a call and the document, enum, extra key, order, empty); "
            f"parallel false: one call; json_object: the same with any object")


def test_tool_choice_decides_the_branches():
    none, dn = _union(tool_choice="none")
    assert not none.calls and none.types is None and _in(dn, '{"signal": "buy", "ok": true}')
    assert not _in(dn, "<tool_call>\n<function=now>\n</function>\n</tool_call>")
    for tc in ("required", {"type": "function", "function": {"name": "now"}}):
        req, dr = _union(tool_choice=tc)
        assert req.kind == "tool_choice" and not req.calls
        assert _in(dr, "<tool_call>\n<function=now>\n</function>\n</tool_call>")
        assert not _in(dr, '{"signal": "buy", "ok": true}'), tc
    # the document is still checked: a bad schema is the 400 that names response_format
    try:
        _union(rf={"type": "json_schema", "json_schema": {"schema": {"type": "x"}}},
               tool_choice="required")
        raise AssertionError("a bad schema beside required must be refused")
    except C.compat.Refusal as exc:
        assert exc.param == "response_format", exc.param
    # the flat tool form is a function too; structured_outputs joins the calls the same way
    flat, df = _union(tools=[{"name": "now"}])
    assert flat.calls and _in(df, "<tool_call><function=now></function></tool_call>")
    so = C.compat.answer_constraint({"messages": [], "tools": TOOLS,
                                     "structured_outputs": {"choice": ["yes", "no"]}})
    ds = G.compile_regex(so.pattern)
    assert so.calls and _in(ds, "yes") and _in(ds, "<tool_call><function=now></function></tool_call>")
    # the completions endpoint has no tools: the document alone
    comp = C.compat.answer_constraint({"prompt": "x", "tools": TOOLS, "response_format": JS},
                                      chat=False)
    assert not comp.calls
    try:
        C.compat.answer_constraint({"messages": [], "tools": TOOLS, "response_format": JS},
                                   xml_calls=False)
        raise AssertionError("pythonic calls cannot be a branch")
    except C.compat.Refusal as exc:
        assert exc.param == "tools"
    return ("none: the document; required and named: the calls; a bad schema beside them a 400; "
            "flat tools, structured_outputs, completions, pythonic")


class MarkerTok(C.Tok):
    """The toy tokenizer with `<tool_call>` and `</tool_call>` as added ids 97 and 98, text in the
    answer as in Qwen's tokenizer."""

    added_tokens_decoder = {97: "<tool_call>", 98: "</tool_call>"}

    def __len__(self):
        return 99

    def convert_tokens_to_ids(self, text):
        return {"<tool_call>": 97, "</tool_call>": 98}.get(text, -1)

    def decode(self, seq, skip_special_tokens=True):
        return "".join(self.added_tokens_decoder[int(t)] if int(t) in self.added_tokens_decoder
                       else super(MarkerTok, self).decode([t], skip_special_tokens) for t in seq)


def test_the_call_markers_keep_their_own_ids():
    tok = MarkerTok()
    plain = G.Vocab.from_tokenizer(tok, 99)
    calls = G.Vocab.from_tokenizer(tok, 99, markers=G.CALL_MARKERS)
    assert plain.bytes[97] is None and plain.bytes[98] is None
    assert calls.bytes[97] == b"<tool_call>" and calls.bytes[98] == b"</tool_call>"
    assert calls.bytes[C.EOS] is None, "the end token stays special"
    a, _ = _union(tools=MARK)
    g = G.Grammar(a.pattern, calls)
    start = g.mask(g.dfa.start, frozenset({C.EOS}), "cpu")
    assert start[97] and not start[98] and start[(ord("{") - 32) % 96], "the call or the document"
    assert not G.Grammar(a.pattern, plain).mask(g.dfa.start, frozenset({C.EOS}), "cpu")[97]
    # a whole call written with the marker ids walks to an accepting state
    body = "<function=mark><parameter=level>hi</parameter></function>"
    ids = [97] + [(ord(c) - 32) % 96 for c in body] + [98]
    cons = G.Constraint(g, {C.EOS}, "cpu")
    cons.commit(ids)
    assert g.dfa.accept[cons.state], "the markers' bytes are the language's"
    return "plain vocabulary: the markers masked (as before); the calls vocabulary: their own ids"


BRANCHES: list = []


def _answer(stream, **extra):
    C.serve()
    body = dict({"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 120,
                 "stream": stream}, **extra)
    real = getattr(G.Grammar, "branch", None)

    def branch(self, *a, **k):
        BRANCHES.append(real(self, *a, **k))
        return BRANCHES[-1]

    BRANCHES.clear()
    if real is not None:
        G.Grammar.branch = branch
    try:
        head, raw = C.Req("/v1/chat/completions", body).response()
    finally:
        if real is not None:
            G.Grammar.branch = real
    assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
    if stream:
        ev = C._events(raw)
        content = "".join(c["delta"].get("content") or "" for e in ev for c in e["choices"])
        names = [d["function"]["name"] for e in ev for c in e["choices"]
                 for d in (c["delta"].get("tool_calls") or []) if d.get("function", {}).get("name")]
        finish = [c["finish_reason"] for e in ev for c in e["choices"] if c["finish_reason"]][-1]
        return content, names, finish
    c = json.loads(raw)["choices"][0]
    return (c["message"]["content"] or "",
            [t["function"]["name"] for t in c["message"].get("tool_calls") or []],
            c["finish_reason"])


def _valid(text):
    v = json.loads(text)
    assert text == text.strip() and set(v) == {"signal", "ok"} and v["signal"] in ("buy", "hold") \
        and isinstance(v["ok"], bool), text
    return v


def test_the_handler_answers_tools_and_a_schema():
    # the random model: biased to `{` it writes the document, to `<` a call; the space is biased
    # out (the toy vocabulary has no other whitespace); json_object: `{` first, then `}` (its
    # strings are free text, where the random model never stops)
    to_doc, to_call = {"0": -50, "91": 50}, {"0": -50, "28": 50}
    to_obj = {"0": -50, "91": 30, "93": 60}
    seen = {}
    for stream in (False, True):
        for rf, bias in ((JS, to_doc), ({"type": "json_object"}, to_obj)):
            content, names, finish = _answer(stream, tools=MARK, response_format=rf,
                                             logit_bias=bias)
            assert finish == "stop" and names == [], (rf, stream, finish, names, content)
            if rf is JS:
                _valid(content)
            else:
                assert isinstance(json.loads(content), dict) and content == content.strip()
            seen[f"{rf['type']}/{'stream' if stream else 'json'}"] = "document"
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias=to_call, parallel_tool_calls=False)
        assert finish == "tool_calls" and len(names) == 1 and names[0] in ("pick", "mark"), \
            (stream, finish, names, content)
        assert "{" not in content and "<tool_call>" not in content, content
        # tool_choice none: the document, whatever the bias; required: the calls
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias=to_call, tool_choice="none")
        assert finish == "stop" and names == [], (finish, names)
        _valid(content)
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias=to_doc, tool_choice="required")
        assert finish == "tool_calls" and len(names) == 1, (finish, names, content)
        seen[f"call/{'stream' if stream else 'json'}"] = "tool_calls"
    return f"{seen}; tool_choice none: the document, required: the call"


# ------------------------------------------------------------------ SRV-619: the model picks the branch

BACKTICK, LT, LBRACE, RBRACE, H = 64, 28, 91, 93, 40          # "`", "<", "{", "}", "H"


def test_the_product_knows_its_branches():
    a, _ = _union(tools=MARK)
    g = G.Grammar(a.pattern, G.Vocab.from_tokenizer(C.Tok(), 97), a.parts)
    d, side = g.dfa, g.side
    assert side[d.start] == G.BOTH and g.open(d.start)
    for text, want in (("  ", G.BOTH), ("<tool", G.FIRST), ("{", G.SECOND), (' {"sig', G.SECOND),
                       ("```", G.NEITHER), ("Here", G.NEITHER)):
        assert side[d.walk(d.start, text.encode())] == want, text
    # the same language as the union regex: the product changes how the mask is chosen, not what
    # it allows
    ref = G.compile_regex(a.pattern)
    rng = random.Random(7)
    pieces = ["<tool_call>", "<function=pick>", "<function=mark>", "<parameter=color>",
              "<parameter=level>", "red", "hi", "</parameter>", "</function>", "</tool_call>",
              "{", "}", '"signal"', '"buy"', ":", ",", '"ok"', "true", " ", "\n", "x"]
    for _ in range(3000):
        t = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 14)))
        assert _in(ref, t) == bool(d.accept[d.walk(d.start, t.encode())]), t
    assert g.branch([(ord(c) - 32) % 96 for c in '{"signal": "buy", "ok": true}']) == "document"
    assert g.branch([(ord(c) - 32) % 96 for c in "<tool_call><function=mark>"]) == "calls"
    assert g.branch([0, 0]) == "open" and G.Grammar("x", g.vocab).branch([1]) is None
    return f"{d.states} states; sides at the start, a call, the document; 3,000 strings == the union"


def test_the_row_argmax_picks_the_branch():
    # the live case: the model's own first token is a fence (or prose); among what the union
    # leaves, the call marker outranks `{`. The document wins; the marker only as the argmax
    tok = MarkerTok()
    v = G.Vocab.from_tokenizer(tok, 99, markers=G.CALL_MARKERS)
    a, _ = _union(tools=MARK)
    g = G.Grammar(a.pattern, v, a.parts)
    eos = frozenset({C.EOS})
    for top, second, want in ((BACKTICK, 97, "document"), (H, 97, "document"),
                              (C.EOS, 97, "document"), (97, LBRACE, "calls"),
                              (LT, LBRACE, "calls"), (0, 97, "all")):
        row = torch.zeros(99)
        row[second], row[top] = 5.0, 9.0
        assert g.pick(g.dfa.start, top) == want, (top, want)
        cons = G.Constraint(g, eos, "cpu")
        cons.apply_single(row)
        pick = int(row.argmax())
        if want == "document":
            assert pick != 97 and not torch.isfinite(row[97]) and torch.isfinite(row[LBRACE]), top
        elif want == "calls":
            assert pick == top and not torch.isfinite(row[LBRACE]), top
        else:
            assert pick == 0 and torch.isfinite(row[97]) and torch.isfinite(row[LBRACE])
    # committed, the side is fixed: after `{` no call; after the marker no document
    cons = G.Constraint(g, eos, "cpu")
    cons.commit([LBRACE])
    row = torch.zeros(99)
    row[97] = 9.0
    cons.apply_single(row)
    assert not torch.isfinite(row[97]), "after `{` the marker is not legal"
    # a plain grammar ignores the row (tool_choice none / required as before)
    plain = G.Grammar(G.WS + G.schema_regex(SIG) + G.WS, v)
    row = torch.zeros(99)
    row[97] = 9.0
    G.Constraint(plain, eos, "cpu").apply_single(row)
    assert not torch.isfinite(row[97]) and torch.isfinite(row[LBRACE])
    return "fence, prose, the end token -> the document; the marker or `<` -> the calls; ws open"


def test_a_picked_branch_is_exact_under_speculation():
    # the row-dependent mask is a function of the prefix and the target's row: drafts change no
    # token. The random model biased to a fence first: the document
    a, _ = _union(tools=MARK)
    bias = {0: -50.0, BACKTICK: 50.0, LT: 20.0}

    def run(dr=None, tree=False):
        C.serve(dr, tree=tree)
        from engine.penalty import PenaltySpec, PenaltyState
        g = G.Grammar(a.pattern, G.Vocab.from_tokenizer(C.Tok(), 97), a.parts)
        pen = G.LogitChain([PenaltyState(PenaltySpec(bias=bias), 97, "cpu"),
                            G.Constraint(g, {C.EOS}, "cpu")])
        return list(app.generate_stream(torch.tensor(C.PROMPT), 60, {C.EOS}, pen=pen))

    ref = run()
    text = C.Tok().decode(ref)
    assert ref[-1] == C.EOS and _valid(text), text
    free = C._run(n=60)
    bad = {}
    for label, dr, tree in (("chain, free drafts", C.Oracle(free), False),
                            ("chain, constrained drafts", C.Oracle(ref), False),
                            ("tree, free drafts", C.TreeOracle(free), True),
                            ("tree, constrained drafts", C.TreeOracle(ref), True)):
        got = run(dr, tree)
        if got != ref:
            bad[label] = got
    assert not bad, f"differs from the drafter-less run: {bad}"
    return f"{text!r}: no drafter, chain and tree agree"


def test_the_handler_lets_the_model_pick():
    # the model's first token a fence or prose, `<` ranked above `{`: before, the union took the
    # call; now the document. `<` first: the call
    seen = {}
    for stream in (False, True):
        tag = "stream" if stream else "json"
        for label, bias in (("fence", {"0": -50, str(BACKTICK): 50, str(LT): 30}),
                            ("prose", {"0": -50, str(H): 50, str(LT): 30})):
            content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                             logit_bias=bias)
            assert finish == "stop" and names == [], (label, stream, finish, names, content)
            _valid(content)
            assert BRANCHES == ["document"], BRANCHES
            seen[f"{label}/{tag}"] = "document"
        content, names, finish = _answer(stream, tools=MARK, response_format={"type": "json_object"},
                                         logit_bias={"0": -50, str(BACKTICK): 50, str(LT): 30,
                                                     str(RBRACE): 40})
        assert finish == "stop" and names == [] and json.loads(content) == {}, (content, names)
        assert BRANCHES == ["document"]
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias={"0": -50, str(LT): 50, str(LBRACE): 30},
                                         parallel_tool_calls=False)
        assert finish == "tool_calls" and len(names) == 1 and BRANCHES == ["calls"], \
            (finish, names, BRANCHES)
        # tool_choice none and required: one language, no branch to pick
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias={"0": -50, str(LT): 50}, tool_choice="none")
        assert finish == "stop" and names == [] and BRANCHES == [], (finish, BRANCHES)
        _valid(content)
        content, names, finish = _answer(stream, tools=MARK, response_format=JS,
                                         logit_bias={"0": -50, str(BACKTICK): 50},
                                         tool_choice="required")
        assert finish == "tool_calls" and len(names) == 1 and BRANCHES == [], (finish, names)
        seen[f"call/{tag}"] = "calls"
    return f"{seen}; json_object: the document; none and required: unchanged"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:58s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
