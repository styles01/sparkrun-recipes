"""SRV-161 (QA SRV-162): a tool's media in a user message is a tool-result turn.

opencode sends an image a tool returned (a `read` of a PNG, a browser screenshot) to a provider
without media in tool results as a user message right after the tool results: the caption
"Attached media from tool result:" and the image. Before this, the router took it for a new user
message, the decider was unsure, Qwen said `chat` at 0.96 (above the 0.8 switch), and every
photo3d TandemLLM session of eval night 2 and round 1 ran about 90 % of its turns as chat/low
under a 1,024-token cap (night agent A's read of the ledger, 2026-10-02 05:40).

The test matrix is SRV-161's Gherkin:

  1  a screenshot returned as a user message keeps the class, no model asked
                                                test_the_photo3d_shapes_*
  2  a person's message with an image (text of its own, or no tool result before it) classifies
                                                test_a_person_*
  3  the journal says why the class was kept    (in 1: kept = tool_media)
  -  the structure decides, the caption alone does not
                                                test_the_caption_alone_*

Run: python tests/test_tool_media_turn.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import SYSTEM, TOOLS, setup  # noqa: E402

from server.conductor import signals  # noqa: E402

# opencode 1.18.33's caption, as the journal holds it (the digest of the misrouted turns)
CAPTION = "Attached media from tool result:"
PNG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
PHOTO3D = ("Tonight you rebuild `ref/photo.jpg` as a Three.js scene in this folder, alone. From "
           "the hero camera your render must look like the photo. The checker decides when it does.")


def media(*parts, caption=True):
    return {"role": "user",
            "content": ([{"type": "text", "text": CAPTION}] if caption else []) + list(parts)}


def call(i, name, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(i, text):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


def plan(cond, msgs, sid="ses-media", **extra):
    b = {"model": "TandemLLM", "messages": list(msgs), "tools": TOOLS, **extra}
    return cond.plan(b, {"x-session-id": sid}, client_kind="opencode", request_id="r", chat=True)


def start(label="three_d", conf=0.88):
    """A three_d session after its first message. On the caption Laya is unsure (design_ui
    at 0.33) and Qwen says `chat` at 0.96, as the journal of 2026-10-01 shows."""
    cond, fakes = setup(decider=lambda t, o: ("design_ui", 0.33) if CAPTION in t else (label, conf),
                        qwen=lambda t, o: ("chat", 0.96) if CAPTION in t else (label, 0.9))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PHOTO3D}]
    p = plan(cond, msgs)
    assert (p.row["class_id"], p.row["effort"]) == (label, "high"), p.row
    return cond, fakes, msgs


# ------------------------------------------------------------------ 1 and 3

SHAPES = {
    # a `read` of the stage's shot: the tool's text, then the image as a user message
    "read a shot": lambda: [call(1, "read", {"filePath": "shots/s1.png"}),
                            result(1, "Image read successfully"), media(PNG)],
    # the browser's screenshot, the same way
    "browser screenshot": lambda: [call(2, "chrome-devtools_take_screenshot", {}),
                                   result(2, "Took a screenshot of the current page."),
                                   media(PNG)],
    # two calls in one step, the second a read of two shots: both results, then the media
    "two results, two images": lambda: [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c3", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "c4", "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"filePath": "shots/s2.png"})}}]},
        result(3, "scenecheck shots/s2.png\nRESULT: FAIL <- (a) de8 12.1 > 10.0"),
        result(4, "Image read successfully"), media(PNG, PNG)],
    # one media message per result, back to back, and one without a caption
    "media after media": lambda: [call(5, "read", {"filePath": "shots/s3.png"}),
                                  result(5, "Image read successfully"), media(PNG),
                                  media(PNG, caption=False)],
}


def test_the_photo3d_shapes_keep_the_class_and_ask_no_model():
    for name, tail in SHAPES.items():
        cond, fakes, msgs = start()
        asked = (len(fakes["decider"].calls), len(fakes["qwen"].calls))
        msgs += tail()
        p = plan(cond, msgs)
        row = p.row
        assert (row["class_id"], row["source"], row["kept"]) == ("three_d", "sticky",
                                                                 "tool_media"), (name, row)
        # the class's effort and its cap; the checker FAIL among the results raises it a level
        want = ("xhigh", None) if "FAIL" in json.dumps(tail()) else ("high", 16384)
        assert (row["effort"], p.budget) == want, (name, row)
        assert (len(fakes["decider"].calls), len(fakes["qwen"].calls)) == asked, \
            f"{name}: no classifier for a tool's media"
        # and the session stays three_d for the tool turns after it
        msgs += [call(9, "bash", {"command": "ls"}), result(9, "ok")]
        p = plan(cond, msgs)
        assert (p.row["class_id"], p.row["source"], p.row["kept"]) == \
            ("three_d", "sticky", None), (name, p.row)


def test_before_the_fix_the_same_shape_switched_to_chat():
    """The control: the same session with the media message written by a person (text of its
    own) is classified, and Qwen's chat at 0.96 is above the switch, as on 2026-10-01."""
    cond, fakes, msgs = start()
    msgs += [call(1, "read", {"filePath": "shots/s1.png"}), result(1, "Image read successfully"),
             {"role": "user", "content": [{"type": "text", "text": CAPTION + " what do you see?"},
                                          PNG]}]
    p = plan(cond, msgs)
    assert (p.row["class_id"], p.row["source"], p.row["effort"]) == ("chat", "qwen", "low"), p.row
    assert any(c[0].startswith(CAPTION) for c in fakes["qwen"].calls)


def test_a_tool_media_turn_with_the_clients_budget_fields():
    """0.3.9 (client budgets on SRV-161): opencode sends `reasoning_effort: medium` on every
    turn. On a kept media turn the word yields to the session's level (three_d, high, 16,384),
    and a client's number in any shape still wins over the class cap."""
    for kw in ({"reasoning_effort": "medium"}, {"reasoning": {"effort": "low"}}):
        cond, fakes, msgs = start()
        msgs += SHAPES["read a shot"]()
        p = plan(cond, msgs, **kw)
        assert (p.row["class_id"], p.row["kept"], p.row["effort"], p.budget,
                p.row["budget_source"]) == ("three_d", "tool_media", "high", 16384, "class"), \
            (kw, p.row)
        assert p.decides_effort
    for kw in ({"max_reasoning_tokens": 777}, {"reasoning": {"max_tokens": 777}},
               {"thinking": {"type": "enabled", "budget_tokens": 777}}):
        cond, fakes, msgs = start()
        msgs += SHAPES["read a shot"]()
        p = plan(cond, msgs, **kw)
        assert (p.row["kept"], p.row["budget_source"], p.budget) == \
            ("tool_media", "client", None), (kw, p.row)
        assert p.think_budget(777) == 777


# ------------------------------------------------------------------ 2

def test_a_person_with_an_image_and_words_is_classified():
    cond, fakes, msgs = start()
    asked = len(fakes["decider"].calls)
    msgs += [call(1, "read", {"filePath": "shots/s1.png"}), result(1, "Image read successfully"),
             {"role": "user", "content": [{"type": "text", "text": "Make the fjord darker."},
                                          PNG]}]
    p = plan(cond, msgs)
    assert p.row["source"] == "decider" and p.row["kept"] is None, p.row
    assert fakes["decider"].calls[asked][0].startswith("Make the fjord darker."), \
        "a person's message is asked about"


def test_a_person_sending_only_an_image_after_the_model_spoke_is_classified():
    """An image-only user message that does not follow a tool result (the model answered, then
    the user pasted a picture) is a person's turn: it is classified as before."""
    cond, fakes, msgs = start()
    asked = len(fakes["decider"].calls)
    msgs += [{"role": "assistant", "content": "Done: the scene is in index.html."}, media(PNG)]
    p = plan(cond, msgs)
    assert p.row["source"] == "qwen" and p.row["kept"] is None, p.row
    assert fakes["decider"].calls[asked][0] == CAPTION, "classified as before"


def test_the_caption_alone_does_not_decide():
    """The caption with no media is text a person wrote; media alone after a tool result needs
    no caption."""
    assert signals.TOOL_MEDIA_CAPTION == CAPTION
    msgs = [{"role": "user", "content": "x"}, call(1, "read", {}), result(1, "ok")]
    assert not signals.is_tool_media(msgs + [{"role": "user", "content": CAPTION}], 3)
    assert signals.is_tool_media(msgs + [media(PNG, caption=False)], 3)
    assert signals.is_tool_media(msgs + [media({"type": "file", "file": {"file_data": "x"}})], 3)
    assert not signals.is_tool_media([{"role": "user", "content": [PNG]}], 0)
    sh = signals.shape({"messages": msgs + [media(PNG)], "tools": TOOLS})
    assert sh.tool_turn and sh.tool_media and sh.user_text == "x" and sh.user_index == 0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
