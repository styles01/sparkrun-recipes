"""A fake engine for the dashboard's end-to-end tests: the real server, no model.

    QSE_ADMIN_TOKEN=$(python3 -c 'print("e2e-" + "0" * 40)') \\
    python server/app.py --fake-engine --port 8011 --served-model qwen38-spark-engine \\
        --usage-ledger /tmp/qse-e2e/ledger.sqlite3 --dashboard-dir ../qwen38-spark-engine-dash/dashboard/dist

Everything a browser can reach is the real code -- the handler, the streaming, usage and timings,
the usage ledger, the dashboard API, the log stream, /metrics, the access control, the static
files. Only `generate_stream` and the tokenizer are replaced: a character-level tokenizer and a
deterministic writer that answers every prompt with the same text for the same prompt, at
`--fake-tps` tokens a second in blocks of four, and fills the loop's own counters (blocks, the
first-miss histogram, the prefill's reused/forwarded) so every number the dashboard shows is
populated. It needs CPU torch and nothing else: no weights, no triton, no transformers, no GPU.
It refuses the production ledger (it is a test run).

Words in the last user message steer it, for the e2e's error and edge states:

    FAKE_ERROR    the generation raises after five tokens (finish_reason error)
    FAKE_CALL_NOTOOLS  the answer calls a tool named `tool`, offered or not (SRV-247)
    FAKE_SLOW     ten tokens a second
    FAKE_SLOW_PREFILL  the prefill runs in 512-token chunks at `--fake-prefill-tps` (default 2,000)
                  and calls the handler's chunk hook after each, as the real prefill does, so the
                  live view shows its progress and a client that leaves mid-prefill is caught;
                  the prompt is padded to at least 8,192 tokens of prefill
    FAKE_WARM_PREFILL  the same, with a first chunk that takes `WARM_FIRST_CHUNK_S` longer (the
                  real first forward's allocation and warm-up), for the live rate that leaves
                  the first chunk out (VIS-40)
    FAKE_LONG     an answer about ten times as long
    a message containing "tool" with `tools` in the request: one call to the first tool
    FAKE_TWO_CALLS  with `tools`: two calls, to the first tool and the last
    FAKE_EMPTY_THINK   the end token inside the reasoning block: no `</think>`, no answer
    FAKE_EMPTY_ANSWER  the reasoning closes and the end token follows with no answer
    FAKE_EMPTY_ALWAYS  with either of the two: every continuation ends empty again
    FAKE_CALL_IN_THINK  with `tools`: the reasoning ends with a complete call to the first tool and
                  the end token follows inside it (SRV-199); FAKE_CALL_IN_THINK_CLOSED the
                  reasoning closes after the call, then the end token, no answer;
                  FAKE_CALL_IN_THINK_OTHER the call names a tool the request does not offer
    FAKE_LOOP     the reasoning repeats one paragraph for about 6,000 tokens
    FAKE_JSON_ANSWER {...}  the answer is exactly that JSON document, to the end of its line
                  (SRV-619)
    FAKE_CALL_JSON {"name": ..., "arguments": {...}}   with `tools`: exactly that call (the rest of
                  the line is the JSON), for the tool-call checks' scenarios; one such line a
                  call, in order (SRV-623). After a hidden action, FAKE_AFTER_FIRST writes the
                  first of them alone, FAKE_AFTER_SAME all of them again
    FAKE_ANSWER <text>   the answer is that text, to the end of its line (`\\n` a newline); with
                  FAKE_CALL_JSON on a line of its own, the text and then that call (SRV-625)
    FAKE_REASONING <text>  with thinking on, the reasoning is that text, to the end of its line
    FAKE_ATTR_CALL  with `tools`: a call to the first tool in the attribute style, its
                  `offset` written as the text "150"
    FAKE_DROP_CLOSER  with FAKE_CALL_JSON: the last parameter's `</parameter>` left out
    FAKE_SECOND_BAD   with FAKE_TWO_CALLS: the second block is one the parser cannot read
    FAKE_BAD_CALL   with `tools`: a block the parser cannot read; after a hidden action, a call
    FAKE_BAD_ALWAYS with `tools`: a block the parser cannot read, after every hidden action too
    FAKE_ARG_LOOP   with `tools`: a `write` whose content falls into the 2026-10-04 loop (a
                  tile-map row repeated until the token limit; the call never closes)
    FAKE_MEMORY   the answer ends with a memory block (server/memblock.py): a user rule
                  "short answers"; FAKE_MEMORY_PROJECT a project gotcha instead; FAKE_MEMORY_BAD
                  a block of a kind that does not exist; FAKE_MEMORY_OPEN one that never closes;
                  with FAKE_CALL_JSON the block sits before the call
    FAKE_STRAY_CLOSE  the answer starts with a sentence and a bare `</think>` line (SRV-187)
    FAKE_PYTHONIC   with `tools`: a sentence, then LFM2's Pythonic call block to the first tool,
                  `<|tool_call_start|>[get_weather(city='Berlin', days=3)]<|tool_call_end|>`
                  (SRV-425; the server reads it when `STATE["tool_format"]` is "pythonic", as the
                  LFM2.5 adapter sets it); FAKE_PYTHONIC_BAD the block's list never closes
    FAKE_CLASS <id>   the in-context stage (server/conductor/incontext.py) answers that class
    FAKE_CONTEXT_ERROR  the in-context stage fails

`QSE_FAKE_PROMPTS` in the environment names a file: every prompt the writer is given is appended to
it as one JSON line, `{"sha256": ..., "tokens": N}` of its ids, so a test compares the engine input
of two requests (SRV-621: the `tools` profile's prompt against the raw path's).

`FAKE_MEM_GIB` in the environment gives it the engine's memory policy (OPS-47) on an allocator that
holds that many GiB: 76 with the default limits refuses every prompt with a 503 (the peak is 77 and
a prefill's workspace is at least 1.15 GiB); 60 serves them.

With thinking on (the chat template's default), the answer is preceded by a short reasoning block.
The writer feeds its blocks to the request's `ThinkBudget` as the real loop does (before yielding
them), so the live view's thinking/writing states come from the same object on both engines.
A second request that repeats a prompt's first 64+ characters reports them as a prefix-cache hit.

With the conductor's monitor (`sup`), the writer stops at the same safe points as the real loop,
through the same helpers: after a block (SP1, `app._sp1`) and at an end token (SP2, `app._sp2`).
An action's tokens join the writer's context -- only `act.visible` is yielded -- and the writer
then answers from there: a call to the first tool when the request offered tools, else its text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import types

import torch

from engine.models.adapter import CostModel

THINK_OPEN, THINK_CLOSE, EOS = 0x110001, 0x110002, 0x110003
# the tool-call markers are single ids too, as in Qwen's tokenizer (SRV-199's lift reads them)
CALL_OPEN, CALL_CLOSE = 0x110004, 0x110005
SPECIAL = {"<think>": THINK_OPEN, "</think>": THINK_CLOSE, "<|im_end|>": EOS,
           "<tool_call>": CALL_OPEN, "</tool_call>": CALL_CLOSE}
WARM_FIRST_CHUNK_S = 1.5
WORDS = ("the engine reads one sequence at a time and writes several tokens a block when the "
         "draft agrees with the target so the speed depends on the text more than on the prompt "
         "length a quotation runs at a hundred tokens a second and fresh prose at forty").split()


class FakeTokenizer:
    """One token per character; `<think>`, `</think>`, `<|im_end|>`, `<tool_call>` and
    `</tool_call>` are single ids."""

    unk_token_id = -1
    eos_token_id = EOS
    def convert_tokens_to_ids(self, text):
        return SPECIAL.get(text, -1)

    def encode_text(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for s, t in SPECIAL.items():
                if text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode_text(text or "")
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, seq, skip_special_tokens=True):
        inv = {v: k for k, v in SPECIAL.items()}
        out = []
        for t in seq:
            t = int(t)
            if t == EOS:
                if not skip_special_tokens:
                    out.append("<|im_end|>")
            elif t in inv:
                out.append(inv[t])
            else:
                out.append(chr(t))
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=True,
                            tools=None, return_tensors=None, return_dict=False, **kw):
        parts = []
        if tools:
            names = [((t.get("function") or {}).get("name") or "?") for t in tools
                     if isinstance(t, dict)]
            parts.append(f"<|im_start|>system\n# Tools: {', '.join(names)}<|im_end|>\n")
        cont = bool(kw.get("continue_final_message"))
        for i, m in enumerate(messages):
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if cont and i == len(messages) - 1:
                # the real template's shape: the last answer open after its closed think block
                parts.append(f"<|im_start|>{m.get('role', 'user')}\n<think>\n\n</think>\n\n{c or ''}")
                continue
            parts.append(f"<|im_start|>{m.get('role', 'user')}\n{c or ''}<|im_end|>\n")
        if add_generation_prompt and not cont:
            parts.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking
                                                        else "<think>\n\n</think>\n\n"))
        ids = self.encode_text("".join(parts))
        return {"input_ids": torch.tensor([ids])}


class _FakeCfg:
    vocab_size = EOS + 1
    attention_layers: list = []
    num_key_value_heads = 0
    head_dim = 0


class _FakeState:
    S = torch.zeros(1)
    conv = torch.zeros(1)


class FakeEngine:
    """What `server/app.py` reads off an engine outside the decode loop."""

    cfg = _FakeCfg()
    state = _FakeState()
    kv = types.SimpleNamespace(length=0)
    w = types.SimpleNamespace(nvfp4_source="", fp8_head_source=None)

    def cost(self) -> CostModel:
        return CostModel(section_bytes=8, view_bytes_per_token=0, parts={"recurrent": 4, "conv": 4})

    def discard_pending(self) -> None:
        pass

    def resident_files(self) -> list:
        return []

    def memory_parts(self) -> dict:
        return {}


def _call(name: str, user: str) -> str:
    return (f"<tool_call>\n<function={name}>\n<parameter=query>\n{user[:40]}\n"
            f"</parameter>\n</function>\n</tool_call>")


BAD_CALL = "<tool_call>\n<function>\n{\"broken\": \n</tool_call>"


def _json_call(user: str, first: bool = False) -> str | None:
    found = re.findall(r"FAKE_CALL_JSON (\{.*\})\s*$", user, flags=re.M)
    if not found:
        return None
    out = []
    for raw in found[:1] if first else found:
        c = json.loads(raw)
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n"
                         f"</parameter>\n" for k, v in (c.get("arguments") or {}).items())
        if "FAKE_DROP_CLOSER" in user and params.endswith("</parameter>\n"):
            params = params[:-len("</parameter>\n")]
        out.append(f"<tool_call>\n<function={c['name']}>\n{params}</function>\n</tool_call>")
    return "\n".join(out)


MEMORY_BLOCK = ('<tandem_memory op="add" scope="{scope}" kind="{kind}">\ntitle: {title}\n\n'
                '{body}\n</tandem_memory>')


def _memory_block(user: str, answer: str) -> str:
    """The answer with a memory block (FAKE_MEMORY*): after the text, before a call."""
    if "FAKE_MEMORY_PROJECT" in user:
        b = MEMORY_BLOCK.format(scope="project", kind="gotcha",
                                title="the checker wants the summer scene within 1.0",
                                body="Gate (a) compares the summer scene with night 1; offerings "
                                     "recolour it by season.")
    else:
        b = MEMORY_BLOCK.format(scope="user", kind="poem" if "FAKE_MEMORY_BAD" in user else "rule",
                                title="short answers",
                                body="The user wants answers of at most five sentences.")
    if "FAKE_MEMORY_OPEN" in user:
        b = b[:b.index("</tandem_memory>")]
    if answer.startswith("<tool_call>"):
        return "Noted.\n\n" + b + "\n\n" + answer
    return answer + "\n\n" + b


def _reply(user: str, tools: list[str], think: bool) -> tuple[str, str]:
    """(reasoning, answer) for a prompt, the same every time."""
    h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
    words = [WORDS[(h >> (3 * i)) % len(WORDS)] for i in range(24 + h % 40)]
    if "FAKE_LONG" in user:
        words = words * 10
    answer = " ".join(words).capitalize() + "."
    if tools and "tool" in user.lower():
        answer = _call(tools[0], user)
        if "FAKE_TWO_CALLS" in user:
            answer += "\n" + (BAD_CALL if "FAKE_SECOND_BAD" in user
                              else _call(tools[-1], user[::-1]))
    doc = re.search(r"FAKE_JSON_ANSWER (\{.*\})\s*$", user, flags=re.M)
    if doc is not None:
        answer = doc.group(1)
    if tools and _json_call(user):
        answer = _json_call(user)
    elif tools and "FAKE_ATTR_CALL" in user:
        answer = (f'<tool_call>\n<function name="{tools[0]}">\n<parameter name="filePath">a.js'
                  f'</parameter>\n<parameter name="offset">150</parameter>\n</function>\n'
                  f'</tool_call>')
    elif tools and ("FAKE_BAD_CALL" in user or "FAKE_BAD_ALWAYS" in user):
        answer = BAD_CALL
    elif tools and "FAKE_ARG_LOOP" in user:
        head = "".join(f"const C{i} = {i * 7 % 13};\n" for i in range(40))
        row = '"' + " " * 94 + '",\n'
        answer = ("<tool_call>\n<function=write>\n<parameter=filePath>\n/tmp/mario.html\n"
                  "</parameter>\n<parameter=content>\n" + head + "const LEVEL = [\n"
                  + row * 3000)
    if tools and "FAKE_PYTHONIC" in user:
        inner = f"[{tools[0]}(city='Berlin', days=3" + ("" if "FAKE_PYTHONIC_BAD" in user else ")]")
        answer = "Let me look that up.<|tool_call_start|>" + inner + "<|tool_call_end|>"
    if "FAKE_CALL_NOTOOLS" in user:
        # SRV-247: a call the request did not offer a tool for (the memory search as `tool`)
        answer = "Let me check my memory for that.\n\n" + _call("tool", user)
    if "FAKE_MEMORY" in user:
        answer = _memory_block(user, answer)
    if "FAKE_STRAY_CLOSE" in user:
        # SRV-187: the model closes its reasoning a second time in the answer
        answer = "Next I run it.\n</think>\n\n" + answer
    said = re.search(r"FAKE_ANSWER (.*)$", user, flags=re.M)
    if said is not None:
        # SRV-625: an answer that quotes what the conversation's tool results held
        answer = said.group(1).replace("\\n", "\n")
        if tools and _json_call(user):
            answer += "\n\n" + _json_call(user)
    reasoning = f"The question has {len(user)} characters; answer it plainly.\n" if think else ""
    thought = re.search(r"FAKE_REASONING (.*)$", user, flags=re.M)
    if thought is not None and think:
        reasoning = thought.group(1).replace("\\n", "\n") + "\n"
    if "FAKE_LOOP" in user and think:
        para = ("I am going to try a completely different approach. The pagoda roof needs the "
                "colour of the summer island, so I compute the difference again from the start "
                "and compare it with the reference value the checker printed last time. ")
        reasoning = para * (6000 // len(para) + 1)
    return reasoning, answer


#: OPS-175: the real engine's load lines, in its order, for `--fake-load-s` (ops/engine_status.py
#: reads the phases from them): the share of the load each takes, the line
LOAD_LINES = ((0.15, "[fake-load] 8.0 GB of 20.0 GB of weights read"),
              (0.15, "[fake-load] 16.0 GB of 20.0 GB of weights read"),
              (0.10, "[server] 400 quantised projections (17.00 GiB), 451 plain tensors (3.60 GiB), 20.61 GiB resident"),
              (0.25, "[server] drafter=fake depth=0 k=0  loaded in 1.0s"),
              (0.15, "[server] warm-up prefill of 2600 tokens in 1.0s"),
              (0.20, "[server] verify graphs: 0 captured in 0.0s"))


def fake_load(seconds: float, sleep=time.sleep) -> None:
    """The phases of a real load over `seconds`, printed as the engine prints them."""
    for share, line in LOAD_LINES:
        sleep(max(0.0, seconds) * share)
        print(line, flush=True)


def load(app, a) -> None:
    """Put the fake in `app.STATE` and replace `app.generate_stream`. Called from `main()`."""
    if float(getattr(a, "fake_load_s", 0.0) or 0.0) > 0:
        fake_load(float(a.fake_load_s))
    tok = FakeTokenizer()
    tps = max(1.0, float(getattr(a, "fake_tps", 200.0)))
    ptps = float(getattr(a, "fake_prefill_tps", 0.0) or 0.0) or 2000.0
    seen: list[list[int]] = []

    def generate_stream(prompt, max_new, eos, think=None, conv_id=None, deadline=None, pen=None,
                        pstop=None, sampler=None, lpr=None, on_prefill=None, sup=None,
                        compacting=False):
        ctx = [int(t) for t in prompt.tolist()]
        log = os.environ.get("QSE_FAKE_PROMPTS")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"sha256": hashlib.sha256(json.dumps(ctx).encode()).hexdigest(),
                                    "tokens": len(ctx)}) + "\n")
        text = tok.decode(ctx, skip_special_tokens=False)
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
        user = users[-1] if users else text
        tools = re.findall(r"# Tools: (.*?)<\|im_end\|>", text)
        tools = [t.strip() for t in tools[0].split(",")] if tools else []
        opened = text.rstrip().endswith("<think>")
        # the prefill: a prefix hit when an earlier prompt shares 64+ tokens, on a 64 grid
        best = max((next((i for i, (x, y) in enumerate(zip(p, ctx)) if x != y), min(len(p), len(ctx)))
                    for p in seen[-32:]), default=0)
        reused = (min(best, len(ctx) - 1) // 64) * 64
        if think is not None:
            think.start(ctx)
        t0 = time.perf_counter()
        info = getattr(on_prefill, "info", None)
        warm = "FAKE_WARM_PREFILL" in user
        chunked = warm or "FAKE_SLOW_PREFILL" in user
        if info is not None:
            info.update(kind="prefix" if reused else None, start=reused, t0=t0,
                        chunk=512 if chunked else 0)
        if chunked:
            # the real prefill's shape: chunks, the hook after every chunk but the last
            total = max(len(ctx), reused + 8192)
            i = reused
            while i < total:
                step = min(512 - (i % 512), total - i)
                app._par_sleep(step / ptps + (WARM_FIRST_CHUNK_S if warm and i == reused else 0.0))
                i += step
                if on_prefill is not None and i < total:
                    on_prefill(i, total)
                if i < total:
                    _lane("prefill")
        else:
            app._par_sleep(min(2.0, 0.02 + 0.0001 * (len(ctx) - reused)))
        app.STATE["last_prefill"] = {"reused": reused, "forwarded": len(ctx) - reused,
                                     "ms": (time.perf_counter() - t0) * 1e3,
                                     "kind": "prefix" if reused else None}
        seen.append(ctx)
        reasoning, answer = _reply(user, tools, opened)
        close = [THINK_CLOSE] + tok.encode_text("\n\n") if opened else []
        if tools and "FAKE_CALL_IN_THINK" in user and opened:
            # SRV-199: the call written inside the reasoning, then the end token
            name = "not_offered" if "FAKE_CALL_IN_THINK_OTHER" in user else tools[0]
            if "FAKE_CALL_IN_THINK_CLOSED" in user:
                ids = tok.encode_text(reasoning + "\n" + _call(name, user) + "\n") + close + [EOS]
            else:
                # the end token right after `</tool_call>`, as the last of a 4-token block, so the
                # call's last characters reach the client after SP2 decided (the live shape)
                ids = tok.encode_text(reasoning + "\n" + _call(name, user)) + [EOS]
                ids = tok.encode_text(" " * (-len(ids) % 4)) + ids
        elif "FAKE_EMPTY_THINK" in user and opened:
            ids = tok.encode_text(reasoning) + [EOS]
        elif "FAKE_EMPTY_ANSWER" in user:
            ids = tok.encode_text(reasoning) + close + [EOS]
        else:
            ids = tok.encode_text(reasoning) + close + tok.encode_text(answer) + [EOS]
        # what the writer answers after a hidden action: a call when tools were offered
        after = ([EOS] if "FAKE_EMPTY_ALWAYS" in user else
                 tok.encode_text(BAD_CALL if tools and "FAKE_BAD_ALWAYS" in user else
                                 _json_call(user, first=True) if tools and
                                 "FAKE_AFTER_FIRST" in user and _json_call(user) else
                                 _json_call(user) if tools and "FAKE_AFTER_SAME" in user and
                                 _json_call(user) else
                                 _call(tools[0], user) if tools else answer) + [EOS])
        rate = 10.0 if "FAKE_SLOW" in user else tps
        bs = app.STATE["blocks"] = app.BlockStats()
        app.STATE["last_ctx"] = ctx
        start = len(ctx)
        n, i = 0, 0
        while i < len(ids):
            block = ids[i:i + 4]
            i += len(block)
            if sup is not None and EOS in block:
                j = block.index(EOS)
                act = app._sp2(sup, think, ctx, start, n + j, 1 << 30, extra=block[:j])
                if act is not None:
                    # SP2: the end token is not taken; the action's hidden tokens join the
                    # context, and what the client sees of them is written next
                    head, hidden = block[:j], act.ids[:len(act.ids) - len(act.visible)]
                    if think is not None:
                        think.observe(head)
                    for t in head:
                        ctx.append(t)
                        n += 1
                        yield t
                    app._release_held(sup)    # SP2's line after the block's head, as the engine
                    ctx.extend(hidden)
                    if think is not None:
                        think.observe(hidden)
                    # after a lifted call (E1 case C) the model ends its turn
                    ids, i = list(act.visible) + ([EOS] if act.detail.get("case") == "C"
                                                  else after), 0
                    continue
            _lane("decode")                   # a block boundary: the side lane's safe point
            if n:
                app._par_sleep(len(block) / rate)
                bs.block(15, len(block) - 1)
            if think is not None:
                think.observe(block)          # as the real loop: the block, before it is yielded
            for t in block:
                if "FAKE_ERROR" in user and n == 5:
                    raise RuntimeError("the fake engine failed on purpose (FAKE_ERROR)")
                if n == 0:
                    bs.first()
                if lpr is not None:
                    lpr.forced([t])            # no model, no rows: every token is certain
                ctx.append(t)
                n += 1
                yield t
                if t in eos or n >= max_new:
                    return
            if sup is not None:
                act = app._sp1(sup, think, n, 1 << 30)
                if act is not None:
                    if act.kind == app.GIVE_UP:
                        return
                    # SP1: a hidden close; what is left of the reasoning is not written
                    hidden = act.ids[:len(act.ids) - len(act.visible)]
                    ctx.extend(hidden)
                    think.observe(hidden)
                    ids, i = list(act.visible) + after, 0
            if deadline is not None and deadline.expired():
                return

    def _lane(where):
        lane = app.STATE.get("lane")
        if lane is not None and lane.jobs:
            lane.serve(where)

    def lane_run(job):
        """A side-lane job on the fake (server/preempt.py): the writer's answer to its prompt,
        thinking off, at the writer's rate, its prefill counted cold."""
        text = tok.decode(job.ids, skip_special_tokens=False)
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
        user = users[-1] if users else text
        _, answer = _reply(user, [], False)
        t0 = time.perf_counter()
        time.sleep(min(0.5, 0.01 + len(job.ids) / ptps))
        job.prefill = {"reused": 0, "forwarded": len(job.ids), "kind": None,
                       "ms": (time.perf_counter() - t0) * 1e3}
        bs = job.blocks = app.BlockStats()
        for t in tok.encode_text(answer) + [EOS]:
            time.sleep(1.0 / tps)
            if job.lpr is not None:
                job.lpr.forced([t])
            job.n_out += 1
            if job.n_out == 1:
                bs.first()
            else:
                bs.block()
            job.out.put(t)
            if t in job.eos or job.n_out >= job.max_new:
                break

    def context_choice(ids, letters, at, conv_id=None, on_chunk=None, images=None):
        """The in-context stage on the fake: the letter of the class named by `FAKE_CLASS <id>`
        in the conversation's latest real user message (the one before the hidden question),
        else the first class, at logit 8 against 0; FAKE_CONTEXT_ERROR fails it. The prefill's
        counts as `generate_stream` reports them, and its prompt joins what later prompts reuse."""
        ctx = [int(t) for t in ids]
        text = tok.decode(ctx, skip_special_tokens=False)
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
        user = users[-2] if len(users) > 1 else ""
        if "FAKE_CONTEXT_ERROR" in user:
            raise RuntimeError("the fake in-context stage failed on purpose")
        listed = re.findall(r"^([A-Z]): ([a-z][a-z0-9_-]*) - ", users[-1] if users else "",
                            flags=re.M)
        want = re.search(r"FAKE_CLASS (\S+)", user)
        pick = next((c for c, cid in listed if want and cid == want.group(1)),
                    listed[0][0] if listed else None)
        best = max((next((i for i, (x, y) in enumerate(zip(p, ctx)) if x != y), min(len(p), len(ctx)))
                    for p in seen[-32:]), default=0)
        reused = (min(best, len(ctx) - 1) // 64) * 64
        seen.append(ctx)
        logits = [8.0 if chr(t) == pick else 0.0 for t in letters]
        return logits, {"reused": reused, "forwarded": len(ctx) - reused, "model": "fake"}

    app.STATE.update(
        engine=FakeEngine(), tok=tok, drafter=None, k=15, device="cpu", tree=False,
        sampled_tree=False, relax=app.Relax(1.0, 1), model=a.served_model,
        started=int(time.time()), verbose=a.verbose, max_len=int(a.max_len),
        default_max_tokens=int(a.default_max_tokens), reasoning_format=a.reasoning_format,
        request_timeout=float(a.request_timeout), max_queue=int(a.max_queue),
        queue_timeout=float(a.queue_timeout), draining=False, cfg_eos=EOS,
        think_budget=a.think_budget, think_stall=a.think_stall, reasoning_effort=a.reasoning_effort,
        answer_reserve=app.thinking_mod.parse_reserve(a.answer_reserve),
        pen_spec=app.PenaltySpec(a.rep_penalty, a.presence_penalty, a.frequency_penalty,
                                 a.no_repeat_ngram),
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, pattern_stop=None,
        state_store=None, session_cache=False, prefix_cache=False, prefix_chunk=0,
        response_cache=None, suffix_store=None, suffix_scope="all",
        usage_default=(a.usage_default == "on"),
        # SRV-475: a test's checkpoint sampling (a directory with a generation_config.json)
        card_sampling=app.loopescape_mod.card_sampling(os.environ.get("QSE_FAKE_CARD_DIR")))
    app.generate_stream = generate_stream
    app.context_choice = context_choice
    if int(getattr(a, "parallel", 1) or 1) > 1:
        # parallel requests on the fake: side slots with fake engines, the GPU turn handed over
        # while the writer sleeps (`app._par_sleep`), the admission on FAKE_MEMAVAILABLE_GIB
        app._par_fake_slots(a, FakeEngine)
    app.STATE["lane_run"] = lane_run
    held = float(os.environ.get("FAKE_MEM_GIB") or 0)
    if held:
        # OPS-47 over HTTP: the memory policy on an allocator that holds FAKE_MEM_GIB and never
        # moves, so the gate's 503 and the limits view run through the real handler
        from engine import cache
        gib = 1 << 30
        b = int(held * gib)
        app.STATE["mem"] = cache.Budget(int(a.mem_ceiling_gib * gib), int(a.mem_peak_gib * gib),
                                        int(a.reserve_slack_gib * gib), probe=lambda: (b, b),
                                        empty=lambda: None, peak_probe=lambda reset=False: 0)
    print(f"[server] FAKE ENGINE: no model, a deterministic character writer at {tps:g} tok/s "
          f"(server/fake_engine.py) -- for tests only", flush=True)
