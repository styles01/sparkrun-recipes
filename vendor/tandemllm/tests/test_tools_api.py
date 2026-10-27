"""Tool calling through the real handler, on a CPU (parallel_tool_calls).

The prompt side renders the checkpoint's own chat template (`tests/fixtures/`, read from the model
directory) with transformers' renderer, so what is asserted is the text the model is given:

  * `tool_choice` auto and absent render the same prompt, and `none` renders the tool-free one;
  * `required` and a named function end the system turn with a one-sentence directive, after the
    tool definitions and after the client's own system prompt;
  * a request without tools renders byte for byte what rc5 rendered;
  * a bad `tool_choice` is a 400 that names it.

The output side scripts the model's text and reads both transports: the reasoning never calls
(under all three reasoning formats), a block closed by the end token is a call and one cut by the
token limit is not, the whole-answer JSON form converts only behind its gate, `tool_choice: none`
reads no calls, `parallel_tool_calls: false` keeps one, and the `[req]` line says `tools=parsed:N`
-- through the metrics wrapper, as the served process has it.

Run: python tests/test_tools_api.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from engine.penalty import PenaltySpec  # noqa: E402
from engine.spec import Relax  # noqa: E402
from server import app, metrics  # noqa: E402

TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                             "qwen38_chat_template.jinja")).read()
EOS = 3
TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"{n} a path",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}} for n in ("read_file", "list_dir")]
USER = [{"role": "user", "content": "What is in /etc/hosts?"}]


class TemplateTok:
    """The checkpoint's template, rendered by transformers; one id per character."""

    unk_token_id = -1
    eos_token_id = EOS

    def convert_tokens_to_ids(self, text):
        return -1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) for c in text]
        return type("R", (), {"input_ids": torch.tensor([ids]) if return_tensors else ids})()

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None,
                            return_dict=False, tools=None, **kw):
        text = render_jinja_template([messages], tools=tools, chat_template=TEMPLATE,
                                     add_generation_prompt=add_generation_prompt, **kw)[0][0]
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, seq, skip_special_tokens=True):
        return "".join("" if int(t) == EOS and skip_special_tokens else chr(int(t)) for t in seq)


class _Eng:
    cfg = type("C", (), {"vocab_size": 0x110000})()


def serve(fmt="tags"):
    app.STATE.clear()
    app.STATE.update(engine=_Eng(), drafter=None, k=1, tree=False, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                     prefix_chunk=0, session_cache=False, tok=TemplateTok(), device="cpu",
                     max_len=1 << 16, default_max_tokens=4096, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=EOS, think_budget=0, think_stall=False, reasoning_format=fmt,
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None, reasoning_effort="low")


def rendered(body: dict) -> str:
    serve()
    ids, _, _ = app.build_prompt(body)
    return app.STATE["tok"].decode(ids.tolist(), skip_special_tokens=False)


def direct(messages, tools=None, **kw) -> str:
    """What transformers renders on its own, with the server's own template arguments."""
    return render_jinja_template([messages], tools=tools, chat_template=TEMPLATE,
                                 add_generation_prompt=True, enable_thinking=True,
                                 reasoning_effort="low", **kw)[0][0]


# ------------------------------------------------------------------ the prompt side

def test_auto_is_the_default_and_none_is_the_tool_free_prompt():
    absent = rendered({"messages": USER, "tools": TOOLS})
    assert absent == rendered({"messages": USER, "tools": TOOLS, "tool_choice": "auto"})
    assert absent == direct(USER, TOOLS), "the tools reach the template as left them"
    assert "# Tools" in absent and '"name": "read_file"' in absent
    none = rendered({"messages": USER, "tools": TOOLS, "tool_choice": "none"})
    assert none == rendered({"messages": USER}) == direct(USER)
    assert "# Tools" not in none
    return "auto == absent == the template's own rendering; none == no tools at all"


def test_a_request_without_tools_renders_what_it_did():
    for msgs in (USER, [{"role": "system", "content": "Be brief."}] + USER):
        assert rendered({"messages": msgs}) == direct(msgs)
    return "with and without a system prompt"


def test_required_and_named_end_the_system_turn_with_a_directive():
    auto = rendered({"messages": USER, "tools": TOOLS})
    end = "</IMPORTANT><|im_end|>"
    assert auto.count(end) == 1
    req = rendered({"messages": USER, "tools": TOOLS, "tool_choice": "required"})
    assert req == auto.replace(end, "</IMPORTANT>\n\nYou must call at least one of the functions "
                                    "above in this reply.<|im_end|>")
    named = rendered({"messages": USER, "tools": TOOLS,
                      "tool_choice": {"type": "function", "function": {"name": "list_dir"}}})
    assert named == auto.replace(end, "</IMPORTANT>\n\nYou must call the function list_dir in "
                                      "this reply.<|im_end|>")
    # after the client's own system prompt, whether it is a string or a list of parts
    for sys_content in ("Answer in German.", [{"type": "text", "text": "Answer in German."}]):
        msgs = [{"role": "system", "content": sys_content}] + USER
        got = rendered({"messages": msgs, "tools": TOOLS, "tool_choice": "required"})
        assert "</IMPORTANT>\n\nAnswer in German.\n\nYou must call at least one of the functions " \
               "above in this reply.<|im_end|>" in got, got[-600:]
    return "the directive is the system turn's last sentence, after the tools and the client's prompt"


def test_a_bad_tool_choice_is_refused_by_name():
    for tc, tools in (("required", None), ({"type": "function", "function": {"name": "nope"}},
                                           TOOLS),
                      ("always", TOOLS), ({"type": "function"}, TOOLS)):
        head, body = Req("/v1/chat/completions", {"messages": USER, "tools": tools,
                                                  "tool_choice": tc}).response()
        assert head.startswith("HTTP/1.1 400"), (tc, head)
        assert json.loads(body)["error"]["param"] == "tool_choice"
    return "4 bad values"


# ------------------------------------------------------------------ the output side

class Req(app.Handler):
    def __init__(self, path, body):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.path, self.command, self.request_version = path, "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST " + path, ("127.0.0.1", 0)
        self.close_connection = True
        self.log = ""

    def response(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.do_POST()
        self.log = out.getvalue()
        head, _, body = self.wfile.getvalue().partition(b"\r\n\r\n")
        return head.decode(), body.decode()


THOUGHT = ("I could call <tool_call>\n<function=read_file>\n<parameter=path>\n/secret\n"
           "</parameter>\n</function>\n</tool_call> but no.\n</think>\n\n")
CALL = ("<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n"
        "</function>\n</tool_call>")


def ask(answer: str, stream: bool, eos: bool = True, fmt="tags", **extra):
    """One request whose model writes THOUGHT + answer (+ the end token). Returns
    (content, calls, finish, the [req] line) from either transport."""
    serve(fmt)
    script = [ord(c) for c in THOUGHT + answer] + ([EOS] if eos else [])
    real = app.generate_stream
    app.generate_stream = lambda *a, **k: iter(script)
    try:
        body = dict({"messages": USER, "tools": TOOLS, "stream": stream}, **extra)
        req = Req("/v1/chat/completions", body)
        head, raw = req.response()
    finally:
        app.generate_stream = real
    assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
    line = next(ln for ln in req.log.splitlines() if ln.startswith("[req]"))
    if not stream:
        c = json.loads(raw)["choices"][0]
        calls = [(t["function"]["name"], json.loads(t["function"]["arguments"]))
                 for t in c["message"].get("tool_calls") or []]
        return c["message"]["content"], calls, c["finish_reason"], line
    content, parts, finish = "", {}, None
    for ln in raw.splitlines():
        if not ln.startswith("data: {"):
            continue
        for ch in json.loads(ln[6:])["choices"]:
            d = ch["delta"]
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                p = parts.setdefault(tc["index"], {"name": "", "args": ""})
                p["name"] += (tc.get("function") or {}).get("name") or ""
                p["args"] += (tc.get("function") or {}).get("arguments") or ""
            finish = ch["finish_reason"] or finish
    calls = [(p["name"], json.loads(p["args"])) for _, p in sorted(parts.items())]
    return content, calls, finish, line


def both(answer, **kw):
    a, b = ask(answer, False, **kw), ask(answer, True, **kw)
    assert a[1:3] == b[1:3], ("the transports disagree", a, b)
    return a, b


def test_only_the_answer_calls_under_every_reasoning_format():
    for fmt in ("tags", "reasoning_content", "both"):
        (content, calls, finish, _), (s_content, *_rest) = both("Reading it.\n" + CALL, fmt=fmt)
        assert calls == [("read_file", {"path": "/etc/hosts"})], (fmt, calls)
        assert finish == "tool_calls"
        assert "/secret" not in json.dumps(calls)
        assert "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts" not in content
        assert s_content.endswith("Reading it.\n"), (fmt, s_content[-80:])
    return "tags, reasoning_content, both: one call, the answer's"


def test_a_block_the_end_token_closed_is_a_call_and_one_the_limit_cut_is_not():
    open_block = CALL[:-len("</tool_call>")]
    (content, calls, finish, _), _ = both("Reading it.\n" + open_block)
    assert calls == [("read_file", {"path": "/etc/hosts"})] and finish == "tool_calls"
    assert "<tool_call>" not in content.split("</think>")[-1]
    # Cut by the token limit, the block is content without a stream and the finish is length.
    # A stream has already sent the call's deltas while the model wrote it (rc3's live
    # arguments), and a delta cannot be taken back. Since 0.4.0 (c78353e) the block's text
    # is not sent again as content there: the function the stream closed stays the call it was
    # sent as, so the client holds it once.
    content, calls, finish, _ = ask("Reading it.\n" + open_block, False, eos=False)
    assert calls == [] and finish == "length" and open_block in content
    content, calls, finish, _ = ask("Reading it.\n" + open_block, True, eos=False)
    assert finish == "length" and open_block not in content and "<tool_call>" not in \
        content.split("</think>")[-1], content[-200:]
    assert calls == [("read_file", {"path": "/etc/hosts"})], calls
    return "EOS: tool_calls; the token limit: content (no stream) or the streamed call once, finish length"


def test_the_whole_answer_json_form_converts_behind_its_gate():
    js = '{"name": "list_dir", "arguments": {"path": "/etc"}}'
    (content, calls, finish, _), _ = both(js)
    assert calls == [("list_dir", {"path": "/etc"})] and finish == "tool_calls"
    assert js not in content
    (content, calls, finish, _), _ = both('{"name": "rm_rf", "arguments": {"path": "/"}}')
    assert calls == [] and finish == "stop", "a name that is not one of the request's tools"
    (content, calls, finish, _), _ = both(js, tools=None)
    assert calls == [] and js in content, "a request without tools has no gate to pass"
    return "tool name + whole answer: tool_calls; otherwise the JSON is the answer"


def test_tool_choice_none_reads_no_calls():
    (content, calls, finish, _), _ = both(CALL, tool_choice="none")
    assert calls == [] and finish == "stop" and CALL in content
    return "the block stays text, finish stop"


def test_parallel_calls_and_the_cap_of_one():
    two = CALL + "\n" + CALL.replace("read_file", "list_dir").replace("/etc/hosts", "/etc")
    (_, calls, finish, line), (_, _, _, s_line) = both(two)
    assert calls == [("read_file", {"path": "/etc/hosts"}), ("list_dir", {"path": "/etc"})]
    assert " tools=parsed:2 " in line + " " and " tools=parsed:2" in s_line
    (_, calls, finish, line), (_, _, _, s_line) = both(two, parallel_tool_calls=False)
    assert calls == [("read_file", {"path": "/etc/hosts"})] and finish == "tool_calls"
    assert "tools=parsed:2,dropped:1" in line and "tools=parsed:2,dropped:1" in s_line, s_line
    return "two calls; with parallel_tool_calls false the first, and the line counts the drop"


def test_the_req_line_names_tools_only_when_there_are_some():
    _, _, _, line = ask("Plain answer.", False, tools=None)
    assert " tools=" not in line, line
    _, _, _, line = ask("Plain answer.", False)
    assert "tools=parsed:0" in line, line
    return "tool-free: the line as it was; with tools: parsed:0"



def test_opencode_arguments_come_back_as_their_schema_types():
    # opencode refused `"offset": "150"`, `"timeout": "120000"` and a todo list
    # sent as a string ("expected number, received string"); the handler types them by the
    # request's own schemas, on both transports, and leaves the string parameters alone
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fixtures", "opencode_tools.json")) as f:
        oc = json.load(f)["tools"]
    call = ("<tool_call>\n<function=read>\n<parameter=filePath>\n/p/index.html\n</parameter>\n"
            "<parameter=offset>\n150\n</parameter>\n<parameter=limit>\n120\n</parameter>\n"
            "</function>\n</tool_call>")
    (_, calls, finish, _), _ = both("Reading.\n" + call, tools=oc, tool_choice="auto",
                                    stream_options={"include_usage": True})
    assert calls == [("read", {"filePath": "/p/index.html", "offset": 150, "limit": 120})], calls
    assert finish == "tool_calls"
    todo = ('<tool_call>\n<function=todowrite>\n<parameter=todos>\n[{"content": "x", "status": '
            '"pending", "priority": "high"}]\n</parameter>\n</function>\n</tool_call>')
    (_, calls, _, _), _ = both(todo, tools=oc)
    assert calls == [("todowrite", {"todos": [{"content": "x", "status": "pending",
                                               "priority": "high"}]})], calls
    (_, calls, _, _), _ = both(CALL)                  # a string-only schema: exactly as before
    assert calls == [("read_file", {"path": "/etc/hosts"})]
    return "read offset/limit ints, todowrite a list, both transports; string tools unchanged"


if __name__ == "__main__":
    # the served process wraps the handler, the loop and the log line (server/metrics.py); so do
    # these tests, so a wrapper that drops a new keyword fails here and not on the box
    metrics.install(app)
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:66s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
