"""SRV-193 (QA: SRV-194): Open WebUI's live view shows no reasoning tag. CPU only.

FX2's live check of a 0.4.0 test build (2026-10-06): in a long Open WebUI agent chat, 15 of 70 page scans
showed a literal `</think>` while the answer streamed; the saved chat was clean. The engine's
stream was well formed. The cause is Open WebUI 0.11.3's live path (backend
`open_webui/utils/middleware.py`, frontend `structuredOutput.ts`, read in the container): it folds
`<think>` blocks out of the content after the fact, and forwards each content piece RAW as a
`response.output_text.delta` to the output item that is last AFTER the fold. The piece that carries
`</think>` closes the reasoning item and opens a new message item, so that raw piece, tag and all,
becomes the new message's text in the browser (`ensureOutputItem` creates the item from the event),
and stays there until the round's stream ends and the full output is sent (with a tool call: after
its arguments, which for a file write is minutes). Its `reasoning_content` path has no tag to show:
each round's reasoning is its own item, closed (and timed) by the first content.

The fix (server/compat.py `answer_format`, `QWEN38_CLIENT_REASONING`): under a conductor plan, a
client kind that names no `reasoning_format` gets its own; Open WebUI gets `reasoning_content`, with
the template's blank lines as one content piece where the reasoning ends (server/stream.py
`Reasoning(boundary=...)`), so a round that goes from its thinking straight to a tool call is timed
too. The raw path keeps the release's shape.

The test matrix (SRV-193's Gherkin):

  1  the rule: Open WebUI under a plan gets reasoning_content; its own reasoning_format wins; a
     JSON answer keeps SRV-169's rule; other kinds, the raw path and an empty setting keep tags
                                                      test_the_rule
  2  the splitter: the boundary piece at the close, wherever the tag falls in the pieces, never
     twice, nothing in `tags`                          test_the_boundary
  3  the live view, replayed through a port of Open WebUI 0.11.3's live path (backend fold and
     events, frontend `applyResponseStreamEvent`), on this server's own stream of a two-round
     agent turn (a tool call, its result, the answer):
       - `tags` (the old default) shows the literal `</think>`: the bug, reproduced
       - Open WebUI's default now: no tag in any live state, one reasoning item per round, each
         closed with a duration, a worded answer's text exactly as the server sent it
                                                      test_the_live_view_*
  4  the stored shape: every round's reasoning its own item, no tag in a message item
                                                      (in test_the_live_view_is_clean)

Run: python tests/test_owui_live_render.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server import compat  # noqa: E402
from server.stream import Reasoning  # noqa: E402
from test_ui_states import TOOLS, _go, serve  # noqa: E402

OWUI_UA = "Python/3.11 aiohttp/3.12.15"


# ------------------------------------------------------------------ 1 the rule

def test_the_rule():
    own = {"open-webui": "reasoning_content"}
    f = compat.answer_format
    assert f({}, "tags", False, "open-webui", own) == "reasoning_content"
    assert f({"reasoning_format": "tags"}, "tags", False, "open-webui", own) == "tags"
    assert f({}, "tags", False, "opencode", own) == "tags"
    assert f({}, "tags", False, None, own) == "tags"                 # the raw path passes none
    assert f({}, "tags", True, "hermes", own) == "reasoning_content"  # SRV-169 stays
    assert f({}, "reasoning_content", False, "open-webui", {"open-webui": "tags"}) == "reasoning_content"
    assert compat.client_formats("") == {}
    assert compat.client_formats("open-webui=reasoning_content, hermes=both,x=nonsense,=tags") == \
        {"open-webui": "reasoning_content", "hermes": "both"}
    from engine.settings import KNOBS
    # SRV-245 (2026-10-07): every kind under a plan, Hermes on tags; Open WebUI as before
    assert KNOBS["CLIENT_REASONING"][0] == "*=reasoning_content,hermes=tags"
    assert compat.client_formats() == {"*": "reasoning_content", "hermes": "tags"} or \
        os.environ.get("QWEN38_CLIENT_REASONING") is not None
    dflt = compat.client_formats("*=reasoning_content,hermes=tags")
    assert f({}, "tags", False, "open-webui", dflt) == "reasoning_content"
    assert f({}, "tags", False, "curl", dflt) == "reasoning_content"
    assert f({}, "tags", False, "hermes", dflt) == "tags"
    assert f({"reasoning_format": "tags"}, "tags", False, "curl", dflt) == "tags"
    return "Open WebUI -> reasoning_content under a plan; its own format, SRV-169, raw kept"


# ------------------------------------------------------------------ 2 the splitter

def _split(pieces, **kw):
    r = Reasoning("reasoning_content", **kw)
    out = []
    for p in pieces:
        out += r.push(p)
    return out + r.finish()


def test_the_boundary():
    for pieces in (["Let me look.", "</think>", "\n\n", "Done."], ["Let me look.</think>\n\nDone."],
                   ["Let me look.</th", "ink>\n", "\nDone."], ["Let me look.</think>"]):
        got = _split(pieces, boundary="\n\n")
        reasoning = "".join(t for f, t in got if f == "reasoning")
        content = [t for f, t in got if f == "content"]
        assert reasoning == "Let me look.", (pieces, got)
        assert content[0] == "\n\n" and "".join(content[1:]) in ("Done.", ""), (pieces, got)
        assert sum(1 for t in content if t == "\n\n") == 1, (pieces, got)
    assert _split(["a</think>\n\nb"]) == [("reasoning", "a"), ("content", "b")]     # no boundary
    t = Reasoning("tags", boundary="\n\n")
    assert t.boundary == "" and t.push("a</think>\n\nb") == [("tagged", "a</think>"), ("content", "\n\nb")]
    return "one blank piece where the reasoning ends; none in tags or without the option"


# ------------------------------------------------------------------ 3 Open WebUI 0.11.3, ported
#
# The parts of the live path a chat-completions stream goes through, in the order the backend
# runs them per chunk. The fold's search positions (`scanned_length`) are a speed-up and left out:
# the text searched is the same. Kept: which item a piece lands in, the event the browser gets for
# it (the RAW piece, at the item last after the fold), how the browser applies it, and the full
# output the backend sends when a round's stream ends.

START, END = "<think>", "</think>"


class Owui:
    def __init__(self):
        self.prior: list = []          # the rounds before (the backend's `prior_output`)
        self.output: list = []         # this round's items
        self.browser: list = []        # the frontend's `message.output`
        self.n = 0
        self.clock = 0.0
        self.live_tags = 0             # browser states with a tag in a shown message
        self.states = 0

    def _id(self, p):
        self.n += 1
        return f"{p}{self.n}"

    # --- backend: one round's stream
    def round(self, deltas):
        self.output = []
        for d in deltas:
            self.clock += 1.0
            self._delta(d)
        # the end of a round: an open reasoning item last is closed; the full output is sent
        if self.output and self.output[-1]["type"] == "message":
            self.output[-1]["content"][0]["text"] = self.output[-1]["content"][0]["text"].strip()
            if not self.output[-1]["content"][0]["text"]:
                self.output.pop()
        if self.output and self.output[-1]["type"] == "reasoning" and "ended_at" not in self.output[-1]:
            self._close(self.output[-1])
        self.browser = copy.deepcopy(self.prior + self.output)
        self._look()
        self.prior = self.prior + self.output

    def _close(self, item):
        item["ended_at"] = self.clock
        item["duration"] = int(item["ended_at"] - item["started_at"])
        item["status"] = "completed"

    def _emit(self, ev):
        ev = dict(ev, output_index=ev["output_index"] + len(self.prior))
        self._browser_apply(ev)
        self._look()

    def _delta(self, d):
        for tc in d.get("tool_calls") or ():
            item = {"type": "function_call", "id": tc.get("id") or self._id("fc"), "call_id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name", ""), "arguments": "",
                    "status": "in_progress"}
            self.output.append(item)
            self._emit({"type": "response.output_item.added", "output_index": len(self.output) - 1,
                        "item": dict(item)})
        rc = d.get("reasoning_content") or d.get("reasoning")
        value = d.get("content")
        if rc:
            existing = next((i for i in reversed(self.output) if i["type"] == "reasoning"), None)
            mi = next((k for k, i in enumerate(self.output) if i["type"] == "message"), None)
            item = existing if mi is not None else None
            if item is None:
                if not self.output or self.output[-1]["type"] != "reasoning":
                    item = {"type": "reasoning", "id": self._id("r"), "status": "in_progress",
                            "attributes": {"type": "reasoning_content"}, "content": [],
                            "started_at": self.clock}
                    if mi is not None:
                        item.update(ended_at=self.clock, duration=0, status="completed")
                        self.output.insert(mi, item)
                    else:
                        self.output.append(item)
                else:
                    item = self.output[-1]
            parts = item["content"]
            if parts:
                parts[-1]["text"] += rc
            else:
                item["content"] = [{"type": "output_text", "text": rc}]
            self._emit({"type": "response.reasoning_text.delta", "item_id": item["id"],
                        "output_index": self.output.index(item), "content_index": 0, "delta": rc})
        if value:
            last = self.output[-1] if self.output else None
            if last and last["type"] == "reasoning" and last.get("attributes", {}).get("type") == "reasoning_content":
                self._close(last)
                self.output.append({"type": "message", "id": self._id("msg"), "status": "in_progress",
                                    "content": [{"type": "output_text", "text": ""}]})
                last = self.output[-1]
            inside = (last is not None and last.get("status") == "in_progress"
                      and last.get("attributes", {}).get("type") != "reasoning_content"
                      and last["type"] == "reasoning")
            if inside:
                if last["content"]:
                    last["content"][-1]["text"] += value
                else:
                    last["content"] = [{"type": "output_text", "text": value}]
            else:
                if not self.output or self.output[-1]["type"] != "message":
                    self.output.append({"type": "message", "id": self._id("msg"), "status": "in_progress",
                                        "content": [{"type": "output_text", "text": ""}]})
                self.output[-1]["content"][-1]["text"] += value
            self._fold()
            target = self.output[-1]
            kind = "response.reasoning_text.delta" if target["type"] == "reasoning" else "response.output_text.delta"
            self._emit({"type": kind, "item_id": target["id"], "output_index": len(self.output) - 1,
                        "content_index": max(len(target["content"]) - 1, 0), "delta": value})

    def _fold(self):
        """`tag_output_handler('reasoning', [('<think>', '</think>')], output)`"""
        last = self.output[-1]
        if last["type"] == "message":
            text = last["content"][-1]["text"]
            k = text.find(START)
            if k < 0:
                return
            before, after = text[:k], text[k + len(START):]
            last["content"][-1]["text"] = before
            if not before.strip():
                self.output.pop()
            self.output.append({"type": "reasoning", "id": self._id("r"), "status": "in_progress",
                                "attributes": {}, "content": [], "started_at": self.clock})
            if after:
                self.output[-1]["content"] = [{"type": "output_text", "text": after}]
                self._fold()
        elif last["type"] == "reasoning" and last.get("attributes", {}).get("type") != "reasoning_content":
            text = (last["content"] or [{"text": ""}])[-1]["text"]
            if END not in text:
                return
            block, _, leftover = text.replace(START, "").strip().partition(END)
            block, leftover = block.strip(), leftover.strip()
            if block:
                last["content"] = [{"type": "output_text", "text": block}]
                self._close(last)
            else:
                self.output.pop()
            self.output.append({"type": "message", "id": self._id("msg"), "status": "in_progress",
                                "content": [{"type": "output_text", "text": leftover}]})

    # --- frontend: `applyResponseStreamEvent`
    def _browser_apply(self, ev):
        out = self.browser
        idx = next((k for k, i in enumerate(out) if i.get("id") == ev.get("item_id")
                    or (i.get("call_id") and i.get("call_id") == ev.get("item_id"))), -1)
        idx = idx if idx >= 0 else ev["output_index"]
        if ev["type"] == "response.output_item.added":
            item = dict(ev["item"])
            if idx < len(out):
                out.insert(idx, item)
            else:
                out.append(item)
            return
        while len(out) <= idx:                         # `ensureOutputItem`
            fallback = {"id": ev.get("item_id"), "type": "reasoning" if "reasoning" in ev["type"] else "message",
                        "status": "in_progress", "content": []}
            out.append(fallback if len(out) == idx else {"type": "message", "status": "in_progress", "content": []})
        item = out[idx]
        item.setdefault("content", [])
        while len(item["content"]) <= ev["content_index"]:
            item["content"].append({"type": "output_text", "text": ""})
        item["content"][ev["content_index"]]["text"] += ev["delta"]

    def _look(self):
        """What the page shows: the text of every message item with text (`buildOutputDisplayItems`)."""
        self.states += 1
        shown = " ".join(i["content"][0]["text"] for i in self.browser
                         if i.get("type") == "message" and i.get("content") and i["content"][0]["text"].strip())
        if START in shown or END in shown:
            self.live_tags += 1


def _deltas(out):
    ds = []
    for ln in out["raw"].splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:]).get("choices") or ():
                if ch.get("delta"):
                    ds.append(ch["delta"])
    return ds


def _agent_turn(fmt=None):
    """Two rounds of an Open WebUI agent turn through this server: a tool call, then its answer."""
    serve("on")
    sysm = {"role": "system", "content": "You are a helpful agent."}
    user = {"role": "user", "content": "Write a game. FAKE_TOOL"}
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 3000, "tools": TOOLS,
            "messages": [sysm, user]}
    if fmt:
        body["reasoning_format"] = fmt
    r1 = _go(body, {"User-Agent": OWUI_UA})
    calls = [c for c in r1["calls"] if c.get("id")]
    assert calls, r1["content"][-300:]
    hist = [sysm, user, {"role": "assistant", "content": "", "tool_calls": [
                {"id": calls[0]["id"], "type": "function", "function": calls[0]["function"]}]},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "index.html written"}]
    r2 = _go(dict(body, messages=hist), {"User-Agent": OWUI_UA})
    return r1, r2


def test_the_live_view_shows_the_tag_with_tags():
    r1, r2 = _agent_turn(fmt="tags")
    ow = Owui()
    for r in (r1, r2):
        ow.round(_deltas(r))
    assert ow.live_tags > 0, "the bug: the piece with </think> shown raw until the round ends"
    stored = [i for i in ow.prior if i["type"] == "message"]
    assert not any("think>" in i["content"][0]["text"] for i in stored), stored   # saved chat clean
    return f"tags: {ow.live_tags} of {ow.states} live states show a tag (the bug, reproduced)"


def test_the_live_view_is_clean():
    r1, r2 = _agent_turn()
    for r in (r1, r2):
        assert "think>" not in r["content"], r["content"][-200:]
        assert r["reasoning"], "the reasoning goes to reasoning_content"
        assert r["content"].startswith("\n\n"), repr(r["content"][:40])        # the boundary
    ow = Owui()
    for r in (r1, r2):
        ow.round(_deltas(r))
    assert ow.live_tags == 0, f"{ow.live_tags} of {ow.states} live states show a tag"
    rs = [i for i in ow.prior if i["type"] == "reasoning"]
    assert len(rs) == 2 and all(i["status"] == "completed" and i["duration"] >= 1 for i in rs), rs
    assert not any("think>" in i["content"][0]["text"] for i in ow.prior if i["type"] == "message")
    assert sum(1 for i in ow.prior if i["type"] == "function_call") >= 2, \
        [(i["type"], i.get("name")) for i in ow.prior]
    # a round that answers in words: the text the browser shows is the server's answer, untagged
    r3 = _go({"model": "TandemLLM", "stream": True, "max_tokens": 2000,
              "messages": [{"role": "user", "content": "Say hello."}]}, {"User-Agent": OWUI_UA})
    ow.round(_deltas(r3))
    assert ow.live_tags == 0, f"{ow.live_tags} of {ow.states} live states show a tag"
    shown = [i["content"][0]["text"] for i in ow.browser if i["type"] == "message"
             and i["content"][0]["text"].strip()]
    assert shown == [r3["content"].strip()] and shown[0], (shown, r3["content"][-200:])
    assert len([i for i in ow.browser if i["type"] == "reasoning"]) == 3
    return f"default: 0 of {ow.states} live states show a tag; 3 rounds, 3 timed reasoning items"


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:50s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
