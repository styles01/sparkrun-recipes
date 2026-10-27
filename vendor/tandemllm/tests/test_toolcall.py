"""The Qwen XML tool-call parser, unit-tested on a CPU."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.toolcall import ToolCallBuffer, parse_tool_calls  # noqa: E402

BLOCK = ("<tool_call>\n<function=write_file>\n<parameter=path>\n/app/a.html\n</parameter>\n"
         "<parameter=content>\n<p>hi</p>\n</parameter>\n</function>\n</tool_call>")


def test_parse_single_block():
    content, calls = parse_tool_calls("I'll write it now.\n\n" + BLOCK)
    assert content == "I'll write it now.\n\n"
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "write_file"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "/app/a.html", "content": "<p>hi</p>"}
    assert calls[0]["id"].startswith("call_")


def test_consecutive_duplicate_calls_are_dropped():
    content, calls = parse_tool_calls(BLOCK + "\n" + BLOCK)
    assert content == "\n"
    assert len(calls) == 1, "the echoed duplicate must not be executed twice"


def test_parse_two_blocks_and_text_between():
    text = "<tool_call><function=one><parameter=a>1</parameter></function></tool_call> and " + BLOCK
    content, calls = parse_tool_calls(text)
    assert content == " and "
    assert [c["function"]["name"] for c in calls] == ["one", "write_file"]


def test_unparseable_block_stays_in_content():
    bad = "<tool_call>no function here</tool_call>"
    content, calls = parse_tool_calls("x " + bad)
    assert calls == [] and bad in content


def test_buffer_releases_content_and_collects_calls():
    # The buffer holds up to len("<tool_call>")-1 characters back so a split opener is never
    # shown; content is what feed() releases plus flush() at the end.
    b = ToolCallBuffer()
    content = "".join(b.feed("hello ") + b.feed("world"))
    assert content == "hello world", "plain content flows with no delay"
    content += "".join(b.feed(" <tool_c"))
    content += "".join(b.feed("all>"))
    assert b.feed("<function=write_file><parameter=path>/x</parameter></function>") == []
    assert b.feed("</tool_call>") == []
    assert len(b.calls) == 1
    assert json.loads(b.calls[0]["function"]["arguments"]) == {"path": "/x"}
    content += b.flush()
    assert content.strip() == "hello world", "the held tail comes out at flush"


def test_buffer_does_not_delay_plain_content():
    # Certification regression: holding a fixed 9-char tail delayed the first delta and moved the
    # row's TTFT +14 %. With no partial opener, everything flows immediately.
    b = ToolCallBuffer()
    assert "".join(b.feed("the quick brown fox")) == "the quick brown fox"
    assert b.feed("x") == ["x"]
    assert b.feed("<") == []                      # a lone '<' could begin an opener
    assert b.feed("t") == []                      # still could
    assert "".join(b.feed("he answer")) == "<the answer"  # dis-proven: released whole


def test_buffer_flushes_an_unclosed_block_as_content():
    b = ToolCallBuffer()
    assert b.feed("text <tool_call><function=foo>") == ["text "]
    left = b.flush()
    assert left.startswith("<tool_call>") and "foo" in left
    assert b.calls == []


def test_buffer_releases_an_unparseable_block():
    b = ToolCallBuffer()
    out = "".join(b.feed("a <tool_call>junk</tool_call> b"))
    out += b.flush()
    assert out == "a <tool_call>junk</tool_call> b"
    assert b.calls == []


def _stream_charwise(b: ToolCallBuffer, text: str) -> tuple[str, list[dict]]:
    """Feed one character at a time -- the harshest split -- and collect content + deltas."""
    content, deltas = "", []
    for ch in text:
        content += "".join(b.feed(ch))
        deltas += b.drain_deltas()
    return content, deltas


def _arguments_from(deltas: list[dict]) -> str:
    return "".join(d["function"]["arguments"] for d in deltas if "arguments" in d["function"])


def test_streams_the_call_while_the_block_is_written():
    # The rolex_svg fix: a whole-file call used to be withheld and then dumped whole at the very
    # end. The call's name and argument fragments must flow as the model writes them.
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, "I'll write it now.\n\n" + BLOCK + " done")
    assert content.startswith("I'll write it now.\n\n")
    assert len(deltas) > 5, "arguments must arrive as many fragments, not one lump"
    start = deltas[0]
    assert start["index"] == 0 and start["id"].startswith("call_")
    assert start["function"] == {"name": "write_file", "arguments": ""}
    assert json.loads(_arguments_from(deltas)) == {"path": "/app/a.html", "content": "<p>hi</p>"}
    assert b.calls[0]["id"] == start["id"], "the collected call keeps the streamed id"
    assert start["id"] in b.streamed_ids, "and the end-of-stream sweep must skip it"
    assert content == "I'll write it now.\n\n done"


def test_streamed_values_are_trimmed_and_escaped():
    value = 'he said "hi" then \\ and\nnew line  '
    block = ("<tool_call><function=note><parameter=text>\n" + value + "\n</parameter>"
             "</function></tool_call>")
    b = ToolCallBuffer()
    _, deltas = _stream_charwise(b, block)
    assert json.loads(_arguments_from(deltas)) == {"text": 'he said "hi" then \\ and\nnew line'}


def test_streamed_value_with_angle_bracket():
    # A content parameter is HTML; its `<` must not end the value.
    block = ("<tool_call><function=write_file><parameter=content>\n<!DOCTYPE html>\n</parameter>"
             "</function></tool_call>")
    b = ToolCallBuffer()
    _, deltas = _stream_charwise(b, block)
    assert json.loads(_arguments_from(deltas)) == {"content": "<!DOCTYPE html>"}


def test_unknown_shape_never_streams_partially():
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, "a <tool_call>junk</tool_call> b")
    assert deltas == [], "an unrecognised block must not vend half a call"
    assert content + b.flush() == "a <tool_call>junk</tool_call> b"
    assert b.calls == []


def test_two_blocks_stream_with_increasing_index():
    b = ToolCallBuffer()
    text = ("<tool_call><function=one><parameter=a>1</parameter></function></tool_call>"
            "<tool_call><function=two><parameter=b>2</parameter></function></tool_call>")
    _, deltas = _stream_charwise(b, text)
    starts = [d for d in deltas if "id" in d]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["function"]["name"] for s in starts] == ["one", "two"]
    assert len(b.calls) == 2 and b.streamed_ids == {s["id"] for s in starts}


# --- the pre-merge review of rc3 (five defects, one test each) ----------------------------------

def _drive(pieces: list[str]) -> tuple[str, list[dict]]:
    """The server's stream loop, the way `server/app.py` runs it.

    Content comes out of `feed()`, tool-call deltas out of `drain_deltas()`, and at the end of the
    generation `finish()` returns what is still held -- which goes to the wire AS IS. Feeding it
    back through `feed()` is defect 2: the opener re-opens the block and the arguments are streamed
    a second time at a new index.
    """
    b = ToolCallBuffer()
    content, deltas = "", []
    for piece in pieces:
        content += "".join(b.feed(piece))
        deltas += b.drain_deltas()
    left, sweep = b.finish()
    return content + left, deltas + sweep


def _starts(deltas: list[dict]) -> list[dict]:
    return [d for d in deltas if "id" in d]


def test_a_zero_argument_call_is_a_call():
    # `get_time` takes no parameters. It was dropped: the parser refused a function without
    # parameters, so the block stayed in the content as raw XML.
    text = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
    content, calls = parse_tool_calls("one moment " + text)
    assert content == "one moment "
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_time"
    assert calls[0]["function"]["arguments"] == "{}"


def test_a_zero_argument_call_streams_an_empty_object():
    # Streamed, the same call sent `arguments: "}"` -- invalid JSON -- and then vanished, because
    # nothing was collected and the content fallback was skipped as "already streamed".
    text = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, text)
    start = _starts(deltas)[0]
    assert start["function"]["name"] == "get_time"
    assert json.loads(_arguments_from(deltas)) == {}
    assert content == "" and len(b.calls) == 1, "the call is collected, so finish_reason is tool_calls"
    assert b.calls[0]["id"] == start["id"] and start["id"] in b.streamed_ids


def test_the_flushed_tail_is_not_streamed_a_second_time():
    # Defect 2: the flush used to be routed back through the same buffer. The opener re-opened the
    # block, the argument stream went out again at a new index, and the fallback text was eaten.
    content, deltas = _drive(["I'll write it. ",
                              "<tool_call>\n<function=write_file>\n<parameter=path>\n"
                              "/app/a.html\n</parameter>\n</function>\n"])
    assert len(_starts(deltas)) == 1, "one call streamed once, not once per flush"
    assert json.loads(_arguments_from(deltas)) == {"path": "/app/a.html"}
    # 2026-10-05: the block streamed live, so the client has it as the call; its text is not content
    # a second time (it was, until 2026-10-05: once as arguments, once as text)
    assert content == "I'll write it. ", content


def test_a_partial_opener_at_the_end_is_content():
    # The other half of defect 2, with nothing but a truncated opener: pure content loss.
    content, deltas = _drive(["hello <tool"])
    assert content == "hello <tool" and deltas == []


def test_two_functions_in_one_block_each_go_out_once():
    # Defect 3: only the first function was streamed, the id was not recorded because the block
    # parsed to two calls, and the end-of-stream sweep then sent the first one a second time.
    text = ("<tool_call>\n<function=one>\n<parameter=a>\n1\n</parameter>\n</function>\n"
            "<function=two>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>")
    _, deltas = _drive(list(text))
    starts = _starts(deltas)
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["function"]["name"] for s in starts] == ["one", "two"]
    per_call = {}
    for d in deltas:
        per_call.setdefault(d["index"], []).append(d["function"].get("arguments", ""))
    assert json.loads("".join(per_call[0])) == {"a": "1"}
    assert json.loads("".join(per_call[1])) == {"b": "2"}


def test_an_unknown_tag_between_parameters_keeps_the_arguments_valid():
    # Defect 4: a stray tag stopped the argument stream where it was -- `{"a": "1"` -- while the
    # parser read the whole block, so the id was reused and the truncation was never corrected.
    text = ("<tool_call><function=f><parameter=a>1</parameter><note/>"
            "<parameter=b>2</parameter></function></tool_call>")
    b = ToolCallBuffer()
    _, deltas = _stream_charwise(b, text)
    per_call = {}
    for d in deltas:
        per_call.setdefault(d["index"], []).append(d["function"].get("arguments", ""))
    assert len(per_call) == 1
    assert json.loads("".join(per_call[0])) == {"a": "1", "b": "2"}
    assert json.loads(b.calls[0]["function"]["arguments"]) == {"a": "1", "b": "2"}


def test_a_streamed_call_whose_arguments_were_truncated_is_resent_whole():
    # The safety net under defect 4: if what went out is NOT what the parser read, the streamed
    # call is not credited and the sweep sends the complete one rather than hiding the difference.
    text = ("<tool_call><function=f><parameter=a>1</parameter><broken"
            "<parameter=b>2</parameter></function></tool_call>")
    _, deltas = _drive(list(text))
    last = _starts(deltas)[-1]
    assert json.loads(last["function"]["arguments"]) == {"a": "1", "b": "2"}


def test_an_echoed_duplicate_call_reaches_the_client_once():
    # Defect 5: the duplicate drop applied to the collected calls but not to what had already been
    # streamed, so streaming delivered two identical write_file calls and JSON returned one.
    _, calls = parse_tool_calls(BLOCK + "\n" + BLOCK)
    content, deltas = _drive(list(BLOCK + "\n" + BLOCK))
    starts = _starts(deltas)
    assert len(starts) == len(calls) == 1, "both transports deliver the same one call"
    assert json.loads(_arguments_from(deltas)) == json.loads(calls[0]["function"]["arguments"])
    assert content == "\n"


def test_a_repeated_name_with_different_arguments_is_two_calls():
    # The duplicate drop is exact-match only: the same tool called twice with different arguments
    # is two calls on both transports (the second is not streamed live, it is swept at the end).
    second = BLOCK.replace("/app/a.html", "/app/b.html")
    _, calls = parse_tool_calls(BLOCK + second)
    content, deltas = _drive(list(BLOCK + second))
    starts = _starts(deltas)
    assert len(calls) == 2 and len(starts) == 2
    assert [s["index"] for s in starts] == [0, 1]
    paths = [json.loads(s["function"]["arguments"])["path"] if s["function"]["arguments"]
             else None for s in starts]
    assert paths[1] == "/app/b.html"
    assert content == ""


# --- the tiered parser (the ticket's unit matrix) --------------------------------------

def _one_start(deltas: list[dict]) -> dict:
    starts = _starts(deltas)
    assert len(starts) == 1, starts
    return starts[0]


def _args_by_index(deltas: list[dict]) -> dict[int, dict]:
    per = {}
    for d in deltas:
        per.setdefault(d["index"], []).append(d["function"].get("arguments", ""))
    return {i: json.loads("".join(a)) for i, a in per.items()}


def test_lenient_whitespace_parses_and_streams_live():
    text = ("<tool_call>\n\n<function = write_file >\n\n<parameter = path >\n/x\n</parameter>\n\n"
            "</function>\n\n</tool_call>")
    content, calls = parse_tool_calls(text)
    assert content == "" and len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/x"}
    b = ToolCallBuffer()
    content, deltas = _stream_charwise(b, text)
    start = _one_start(deltas)
    assert start["function"]["name"] == "write_file" and start["id"] in b.streamed_ids
    assert json.loads(_arguments_from(deltas)) == {"path": "/x"} and content == ""


def test_attribute_style_parses_and_streams_live():
    text = ('<tool_call><function name="write_file"><parameter name="path">/x</parameter>'
            '<parameter name="content">\nhi\n</parameter></function></tool_call>')
    _, calls = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/x", "content": "hi"}
    content, deltas = _drive(list(text))
    start = _one_start(deltas)
    assert start["function"]["name"] == "write_file"
    assert json.loads(_arguments_from(deltas)) == {"path": "/x", "content": "hi"} and content == ""


def test_a_function_without_its_closer_ends_at_the_next_one_and_at_the_block_end():
    text = ("<tool_call>\n<function=one>\n<parameter=a>\n1\n</parameter>\n"
            "<function=two>\n<parameter=b>\n2\n</parameter>\n</tool_call>")
    _, calls = parse_tool_calls(text)
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == \
        [("one", {"a": "1"}), ("two", {"b": "2"})]
    content, deltas = _drive(list(text))
    assert [s["function"]["name"] for s in _starts(deltas)] == ["one", "two"], "each streamed once"
    assert _args_by_index(deltas) == {0: {"a": "1"}, 1: {"b": "2"}} and content == ""


def test_parallel_blocks_are_parallel_calls_on_both_paths():
    blocks = "".join(f"<tool_call>\n<function=read_file>\n<parameter=path>\n/{n}\n</parameter>\n"
                     f"</function>\n</tool_call>\n" for n in ("a", "b", "c"))
    content, calls = parse_tool_calls(blocks)
    assert [json.loads(c["function"]["arguments"])["path"] for c in calls] == ["/a", "/b", "/c"]
    _, deltas = _drive(list(blocks))
    assert [s["index"] for s in _starts(deltas)] == [0, 1, 2]
    assert [a["path"] for a in _args_by_index(deltas).values()] == ["/a", "/b", "/c"]


def test_json_inside_the_tags_is_a_call():
    for inner in ('{"name": "read_file", "arguments": {"path": "/x"}}',
                  '{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}',
                  '{"type": "function", "function": {"name": "read_file", "arguments": {"path": "/x"}}}'):
        text = "Reading it.\n<tool_call>\n" + inner + "\n</tool_call>"
        content, calls = parse_tool_calls(text)
        assert content == "Reading it.\n", (inner, content)
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == \
            [("read_file", {"path": "/x"})], inner
        content, deltas = _drive(list(text))
        start = _one_start(deltas)
        assert json.loads(start["function"]["arguments"]) == {"path": "/x"}
        assert content == "Reading it.\n"


def test_a_whole_answer_json_call_converts_behind_the_gate():
    names = {"read_file", "write_file"}
    for text in ('{"name": "read_file", "arguments": {"path": "/x"}}',
                 '\n  {"name": "read_file", "parameters": {"path": "/x"}}\n',
                 '```json\n{"name": "read_file", "arguments": {"path": "/x"}}\n```'):
        content, calls = parse_tool_calls(text, names=names)
        assert content == "" and len(calls) == 1, text
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/x"}
        b = ToolCallBuffer(names=names)
        content, deltas = "", []
        for ch in text:
            content += "".join(b.feed(ch))
            deltas += b.drain_deltas()
        left, sweep = b.finish(eos=True)
        assert content + left == "", (text, content + left)
        start = _one_start(deltas + sweep)
        assert start["function"]["name"] == "read_file"
        assert json.loads(start["function"]["arguments"]) == {"path": "/x"}
    two = ('[{"name": "write_file", "arguments": {"path": "/a", "content": "alpha"}}, '
           '{"name": "write_file", "arguments": {"path": "/b", "content": "beta"}}]')
    _, calls = parse_tool_calls(two, names=names)
    assert [json.loads(c["function"]["arguments"])["path"] for c in calls] == ["/a", "/b"]


def test_the_json_gate_leaves_ordinary_json_answers_alone():
    names = {"read_file"}
    call = '{"name": "read_file", "arguments": {"path": "/x"}}'
    cases = {
        "no tools in the request": (call, None),
        "a name that is not a tool": ('{"name": "delete_all", "arguments": {}}', names),
        "a key a call does not have": ('{"name": "read_file", "arguments": {}, "why": "x"}', names),
        "no arguments at all": ('{"name": "read_file"}', names),
        "inside prose": ("Here is the call: " + call, names),
        "followed by prose": (call + " and then I stop.", names),
        "an ordinary answer": ('{"answer": 42, "name": "read_file", "arguments": {}}', names),
        "a code fence of another language": ("```python\n" + call + "\n```", names),
    }
    for label, (text, gate) in cases.items():
        content, calls = parse_tool_calls(text, names=gate)
        assert calls == [] and content == text, label
        b = ToolCallBuffer(names=gate)
        out = "".join(b.feed(text))
        left, sweep = b.finish(eos=True)
        assert out + left == text and sweep == [] and b.calls == [], label


def test_the_json_gate_releases_an_answer_as_soon_as_it_cannot_be_a_call():
    # A request with tools must not have its ordinary answers held: prose flows at once, and a
    # JSON answer whose first key is not a call's goes out when that key is complete.
    b = ToolCallBuffer(names={"read_file"})
    assert b.feed("\n") == [] and b.feed("The") == ["\nThe"]
    assert b.feed(" answer") == [" answer"]
    b = ToolCallBuffer(names={"read_file"})
    assert b.feed('{"ans') == []
    assert "".join(b.feed('wer": 4')) == '{"answer": 4'
    assert b.feed("2}") == ["2}"]


def test_a_block_closed_by_eos_is_a_call_and_one_cut_short_is_content():
    head = "I'll read it.\n"
    block = "<tool_call>\n<function=read_file>\n<parameter=path>\n/x\n</parameter>\n</function>\n"
    content, calls = parse_tool_calls(head + block, eos=True)
    assert content == head and len(calls) == 1
    content, calls = parse_tool_calls(head + block)                  # the token limit cut it
    assert content == head + block and calls == []
    b = ToolCallBuffer()
    content = "".join(b.feed(head + block))
    deltas = b.drain_deltas()
    left, sweep = b.finish(eos=True)
    assert content + left == head and sweep == []
    start = _one_start(deltas)
    assert start["id"] in b.streamed_ids and json.loads(_arguments_from(deltas)) == {"path": "/x"}
    assert len(b.calls) == 1
    # a value that never closed is not a call even at EOS: content, as before
    cut = "<tool_call>\n<function=write_file>\n<parameter=content>\n<html>"
    content, calls = parse_tool_calls(head + cut, eos=True)
    assert calls == [] and content == head + cut


def test_a_live_call_cut_by_the_token_limit_is_not_sent_again_as_content():
    # 2026-10-05 (an open item): a `write` that streamed live and was cut by the token limit
    # went out twice, once as arguments and once as text in `finish`, and the next prompt was
    # 157,806 tokens. The client keeps the call as it was sent; nothing of the block is content.
    head = "Writing it.\n"
    cut = ("<tool_call>\n<function=write>\n<parameter=filePath>\n/a.js\n</parameter>\n"
           "<parameter=content>\nconst LEVEL = [\n" + '"' + " " * 40 + '",\n' * 50)
    b = ToolCallBuffer()
    content, deltas = "", []
    for ch in head + cut:
        content += "".join(b.feed(ch))
        deltas += b.drain_deltas()
    left, sweep = b.finish()                                       # no EOS: the token limit
    assert content + left == head, (content + left)[:80]
    assert sweep == [] and b.calls == [] and b.cut_live
    start = _one_start(deltas)
    assert start["function"]["name"] == "write"
    args = _arguments_from(deltas)
    assert args.startswith('{"filePath": "/a.js", "content": "const LEVEL = [')
    try:
        json.loads(args)
        raise AssertionError("a cut call must not look whole")
    except ValueError:
        pass
    # not streamed live (stop strings): the block is the text it is, as before
    b = ToolCallBuffer(live=False)
    out = "".join(b.feed(head + cut))
    left, sweep = b.finish()
    assert out + left == head + cut and sweep == [] and not b.cut_live
    # a block cut before its function's name went out sent nothing: it stays content
    b = ToolCallBuffer()
    out = "".join(b.feed(head + "<tool_call>\n<funct"))
    left, _ = b.finish()
    assert out + left == head + "<tool_call>\n<funct" and not b.cut_live


def test_a_cut_block_keeps_the_calls_it_closed_and_sends_none_twice():
    one = "<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>\n"
    two = "<function=write>\n<parameter=content>\nabc"
    b = ToolCallBuffer()
    content, deltas = "", []
    for ch in "<tool_call>\n" + one + two:
        content += "".join(b.feed(ch))
        deltas += b.drain_deltas()
    left, sweep = b.finish()
    assert content + left == "" and sweep == []
    starts = _starts(deltas)
    assert [s["function"]["name"] for s in starts] == ["read", "write"]
    assert [c["function"]["name"] for c in b.calls] == ["read"]
    assert b.calls[0]["id"] == starts[0]["id"] and starts[0]["id"] in b.streamed_ids
    assert json.loads(b.calls[0]["function"]["arguments"]) == {"path": "/x"}
    # an end token inside an open value: not a call, and not sent twice either
    b = ToolCallBuffer()
    for ch in "<tool_call>\n" + two:
        b.feed(ch)
    left, sweep = b.finish(eos=True)
    assert left == "" and sweep == [] and b.calls == [] and b.cut_live


def test_the_first_closer_ends_a_value():
    # The boundary rule: XML-ish, not XML. A value cannot contain `</parameter>`; the first one ends it.
    text = "<tool_call><function=f><parameter=a>x</parameter>y</parameter></function></tool_call>"
    _, calls = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": "x"}
    _, deltas = _drive(list(text))
    assert json.loads(_arguments_from(deltas)) == {"a": "x"}


def test_parallel_calls_capped_at_one_on_the_stream():
    two = BLOCK + BLOCK.replace("/app/a.html", "/app/b.html")
    b = ToolCallBuffer(max_calls=1)
    content, deltas = "", []
    for ch in two:
        content += "".join(b.feed(ch))
        deltas += b.drain_deltas()
    left, sweep = b.finish(eos=True)
    start = _one_start(deltas + sweep)
    assert json.loads(_arguments_from(deltas + sweep))["path"] == "/app/a.html"
    assert len(b.calls) == 1 and b.dropped == 1 and content + left == ""
    assert start["id"] == b.calls[0]["id"]


def test_a_long_json_call_is_read_in_linear_time():
    # A whole-file argument through the whole-answer gate: 400 kB in 4-character pieces. The gate
    # must scan each character once; re-reading the held buffer per piece is 50k x 400 kB.
    import time
    body = "x" * 400_000
    text = json.dumps({"name": "write_file", "arguments": {"path": "/a", "content": body}})
    b = ToolCallBuffer(names={"write_file"})
    t0 = time.perf_counter()
    out = []
    for i in range(0, len(text), 4):
        out += b.feed(text[i:i + 4])
    left, sweep = b.finish(eos=True)
    took = time.perf_counter() - t0
    assert out == [] and left == "" and len(sweep) == 1
    assert json.loads(sweep[0]["function"]["arguments"])["content"] == body
    assert took < 3.0, f"{took:.2f} s for 400 kB"


def test_typed_values_are_returned_typed_on_both_transports():
    # under a tool_choice constraint the mask writes non-string values as JSON literals;
    # both transports return them as values, and the live stream sends them unquoted
    types = {"set": {"n", "tags", "ok"}}
    text = ('<tool_call>\n<function=set>\n<parameter=n>\n-3\n</parameter>\n<parameter=tags>\n'
            '["a b", "c"]\n</parameter>\n<parameter=ok>\ntrue\n</parameter>\n<parameter=note>\n'
            '42\n</parameter>\n</function>\n</tool_call>')
    _, calls = parse_tool_calls(text, types=types)
    want = {"n": -3, "tags": ["a b", "c"], "ok": True, "note": "42"}
    assert json.loads(calls[0]["function"]["arguments"]) == want
    b = ToolCallBuffer(types=types)
    content, deltas = _stream_charwise(b, text)
    start = _one_start(deltas)
    assert json.loads(_arguments_from(deltas)) == want and start["id"] in b.streamed_ids
    assert '"n": -3' in _arguments_from(deltas), "streamed unquoted, as the value"
    _, calls = parse_tool_calls(text)                   # no constraint: strings, as before
    assert json.loads(calls[0]["function"]["arguments"])["n"] == "-3"
    return "integer, array, boolean typed; a string parameter stays a string; no types: all strings"


# ------------------------------------------------------------------ schema-typed values

def _opencode_tools():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fixtures", "opencode_tools.json")) as f:
        return json.load(f)["tools"]


def _xml(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_schema_types_name_only_the_parameters_a_string_does_not_fit():
    from server.toolcall import schema_types
    t = schema_types(_opencode_tools())
    assert t["read"] == {"offset": {"integer"}, "limit": {"integer"}}
    assert t["bash"] == {"timeout": {"integer"}} and t["edit"] == {"replaceAll": {"boolean"}}
    assert t["todowrite"] == {"todos": {"array"}}
    assert t["chrome-devtools_select_page"] == {"pageId": {"number"}, "bringToFront": {"boolean"}}
    assert "write" not in t and "glob" not in t, "all-string tools have nothing to type"
    mixed = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {
        "a": {"type": ["integer", "null"]}, "b": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "c": {"enum": ["x", "y"]}, "d": {"enum": [1, 2]}, "e": {}, "f": {"type": "object"}}}}}]
    assert schema_types(mixed) == {"f": {"a": {"integer", "null"}, "d": {"integer"},
                                         "f": {"object"}}}
    assert schema_types(None) == {} and schema_types([{"type": "web_search"}]) == {}
    return "opencode's read/bash/edit/todowrite + MCP pageId; a string-admitting schema stays text"


def test_opencode_calls_come_back_as_the_schema_types_on_both_transports():
    # 2026-09-26: every one of these reached opencode as a string and was refused by its validator
    # ("expected number, received string"): read offset/limit, bash timeout, todowrite todos, the
    # chrome-devtools pageId and bringToFront
    from server.toolcall import schema_types
    types = schema_types(_opencode_tools())
    todos = '[{"content": "a", "status": "pending", "priority": "high"}]'
    cases = [
        (_xml("read", filePath="/p/index.html", offset="150", limit="120"),
         {"filePath": "/p/index.html", "offset": 150, "limit": 120}),
        (_xml("bash", command="node --check f.mjs", timeout="120000"),
         {"command": "node --check f.mjs", "timeout": 120000}),
        (_xml("todowrite", todos=todos), {"todos": json.loads(todos)}),
        (_xml("edit", filePath="/a", oldString="x", newString="y", replaceAll="true"),
         {"filePath": "/a", "oldString": "x", "newString": "y", "replaceAll": True}),
        (_xml("chrome-devtools_select_page", pageId="2", bringToFront="True"),
         {"pageId": 2, "bringToFront": True}),
        (_xml("chrome-devtools_navigate_page", pageId="1", type="url", url="file:///a.html",
              timeout="60000"),
         {"pageId": 1, "type": "url", "url": "file:///a.html", "timeout": 60000}),
    ]
    for text, want in cases:
        _, calls = parse_tool_calls("Doing it.\n" + text, names={"read"}, types=types)
        assert json.loads(calls[0]["function"]["arguments"]) == want, calls
        b = ToolCallBuffer(types=types)
        content, deltas = _stream_charwise(b, "Doing it.\n" + text)
        left, sweep = b.finish(eos=True)
        assert sweep == [], "streamed live and credited, never sent twice"
        assert json.loads(_arguments_from(deltas)) == want, _arguments_from(deltas)
        assert _one_start(deltas)["id"] in b.streamed_ids
    return f"{len(cases)} opencode shapes typed, stream == parse"


def test_a_value_that_is_not_its_type_keeps_its_text():
    # never guessed: `60s` for an integer stays "60s" (the client's validator says so, as before)
    from server.toolcall import convert, schema_types
    types = schema_types(_opencode_tools())
    text = _xml("bash", command="sleep 1", timeout="60s")
    _, calls = parse_tool_calls(text, types=types)
    assert json.loads(calls[0]["function"]["arguments"])["timeout"] == "60s"
    b = ToolCallBuffer(types=types)
    _, deltas = _stream_charwise(b, text)
    assert json.loads(_arguments_from(deltas)) == {"command": "sleep 1", "timeout": "60s"}
    assert convert("150.0", frozenset({"integer"})) == 150
    assert convert("1.5", frozenset({"integer"})) == "1.5"
    assert convert("1.5", frozenset({"number"})) == 1.5
    assert convert("true", frozenset({"integer"})) == "true"
    assert convert("null", frozenset({"integer", "null"})) is None
    assert convert("{'a': 1}", frozenset({"object"})) == {"a": 1}, "a Python literal"
    assert convert("[1, 2", frozenset({"array"})) == "[1, 2"
    assert convert("FALSE", frozenset({"boolean"})) is False
    return "60s stays text; 150.0 -> 150 for an integer; Python literals read"


def test_a_json_form_call_with_string_numbers_is_typed_too():
    from server.toolcall import schema_types
    types = schema_types(_opencode_tools())
    text = ('<tool_call>\n{"name": "read", "arguments": {"filePath": "/a", "offset": "5"}}\n'
            '</tool_call>')
    _, calls = parse_tool_calls(text, types=types)
    assert json.loads(calls[0]["function"]["arguments"]) == {"filePath": "/a", "offset": 5}
    return "tier 3 inside the tags"


def test_a_string_parameter_is_never_converted():
    # `content` of write is a string even when it reads as JSON: the file is `42`, not a number
    from server.toolcall import schema_types
    types = schema_types(_opencode_tools())
    text = _xml("write", filePath="/n.json", content='{"a": 1}')
    _, calls = parse_tool_calls(text, types=types)
    assert json.loads(calls[0]["function"]["arguments"]) == {"filePath": "/n.json",
                                                              "content": '{"a": 1}'}
    b = ToolCallBuffer(types=types)
    _, deltas = _stream_charwise(b, text)
    assert json.loads(_arguments_from(deltas))["content"] == '{"a": 1}'
    return "write.content stays text"


# ------------------------------------------------------------------ hold (SRV-109, SRV-110 row 6)

TWO = ("<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n"
       "</tool_call>\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls -la\n</parameter>\n"
       "</function>\n</tool_call>")


def _feed_held(buf, text, step=3):
    """Feed in small pieces; nothing may stream while holding. The held calls as they close."""
    held = []
    for i in range(0, len(text), step):
        out = buf.feed(text[i:i + step])
        assert "".join(out).strip() == "", out
        assert buf.drain_deltas() == [], "a holding buffer streams nothing"
        held.extend(buf.take_held())
    return held


def test_hold_releases_each_call_whole_with_a_fresh_id_and_the_next_index():
    buf = ToolCallBuffer(names=["read", "bash"], hold=True)
    held = _feed_held(buf, TWO)
    assert [c["function"]["name"] for c in held] == ["read", "bash"]
    for c in held:
        buf.release(c)
    d = buf.drain_deltas()
    assert [x["index"] for x in d] == [0, 1]
    assert [json.loads(x["function"]["arguments"]) for x in d] == [{"path": "a.txt"},
                                                                    {"cmd": "ls -la"}]
    assert d[0]["id"] != held[0]["id"], "the id is made at release"
    assert [c["id"] for c in buf.calls] == [x["id"] for x in d]
    left, sweep = buf.finish(eos=True)
    assert sweep == [] and left.strip() == "", "a released call is not swept a second time"


def test_hold_a_discarded_call_leaves_no_index_behind():
    buf = ToolCallBuffer(names=["read", "bash"], hold=True)
    held = _feed_held(buf, TWO)
    buf.discard(held[0])
    buf.release(held[1])
    d = buf.drain_deltas()
    assert [(x["index"], x["function"]["name"]) for x in d] == [(0, "bash")]
    assert buf.discarded == 1 and [c["function"]["name"] for c in buf.calls] == ["bash"]


def test_hold_keeps_the_caps_and_the_echo_rule():
    buf = ToolCallBuffer(names=["read"], max_calls=1, hold=True)
    held = _feed_held(buf, TWO)
    assert buf.release(held[0]) is not None and buf.release(held[1]) is None
    assert buf.dropped == 1
    buf = ToolCallBuffer(names=["read"], hold=True)
    one = TWO.split("\n<tool_call>")[0]
    held = _feed_held(buf, one + "\n" + one)
    assert buf.release(held[0]) is not None and buf.release(held[1]) is None, "the echo"


def test_hold_a_block_closed_by_the_end_token_is_held_too():
    buf = ToolCallBuffer(names=["read"], hold=True)
    _feed_held(buf, TWO.split("</tool_call>")[0])
    left, sweep = buf.finish(eos=True)
    held = buf.take_held()
    assert [c["function"]["name"] for c in held] == ["read"] and sweep == [] and left == ""


def test_not_live_a_block_goes_out_whole_or_as_text():
    """`live=False` (a request with stop strings): no argument streams while the block is open.
    A block that closes is a call for the end-of-stream sweep; one cut before its closer (a stop
    string) is the text it was, and no delta ever went out for it."""
    blk = ("<tool_call>\n<function=read>\n<parameter=filePath>\nsrc/a.js\n</parameter>\n"
           "</function>\n</tool_call>")
    b = ToolCallBuffer(names=["read"], live=False)
    assert "".join(b.feed("ok " + blk)) == "ok "
    assert b.drain_deltas() == [] and len(b.calls) == 1
    left, sweep = b.finish(eos=True)
    assert left == "" and len(sweep) == 1
    assert json.loads(sweep[0]["function"]["arguments"]) == {"filePath": "src/a.js"}
    b = ToolCallBuffer(names=["read"], live=False)
    cut = blk.split("</parameter>")[0]
    out = "".join(b.feed("ok " + cut))
    assert b.drain_deltas() == [], "nothing of an open block streams"
    left, sweep = b.finish(eos=False)
    assert sweep == [] and b.calls == [] and out + left == "ok " + cut
    b = ToolCallBuffer(names=["read"])
    b.feed("ok " + cut)
    assert b.drain_deltas(), "live (no stop strings): the arguments stream as before"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:52s} ok")
            passed += 1
    print(f"{passed} passed")
