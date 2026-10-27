"""chatd's context on a long tool run (rc-0.8.0, the prompt "Create a clock in a single html").

THE CASE (live, 2026-10-13): a 21-step run on Flash-Next failed with "This conversation is about
319,136 tokens and the model's window is 262,144" while the engine had 16,980 tokens. Re-rendered
from the store: 2,122,473 request characters, 2,077,842 of them (97.9 %) the base64 of five 300 KB
render_page screenshots, counted as text at the learned 0.15 tokens a character. Each screenshot
also came as its own user message, six user turns in all, and the model wrote "The user is
repeating the same message for the 4th time". No compaction: the one user turn left nothing before
the last two turns to summarize.

  1  a tool's image rides in its tool message (one user turn). Images are append-only (SRV-546):
     all of them up to TOOL_IMAGES_KEPT, then the first ones and the newest, so a new screenshot
     never changes the prompt before the newest-but-one; that one becomes a line saying a newer
     screenshot follows. Other backends get the images in a user message after the results
  2  the estimate counts an image by its pixels (32 x 32 a token), never by its base64, and the
     tokens-per-character ratio is learned on the text alone
  3  a long run compacts its earlier steps between two steps, keeps its user message and goes on
  4  the Context panel after such a compaction counts what the next request carries (live
     2026-10-17: 45,778 tokens, "conversation 0", while the engine got 215,247-token prompts)
  5  file texts stay out of the request (live 2026-10-17, chat 01M556JVWDQVDCAMBMVFZCT7XF: a run
     grew from 25k to 162k tokens of write_file, edit_file and read_file texts, and its
     compaction kept 138,673 tokens): a finished write's content is a stub from the next request
     on (only the tail changes), long reads stay whole until a compaction and are stubs after
     it, and a compaction lands under half the limit that set it off. A worker's write and edit
     texts stay whole until a compaction (SRV-593: a stub inside the last answer made the next
     call read that answer again)

CPU only. Run: python -m pytest -q tests/test_chatd_context.py
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib

from chatd_util import Chat
from test_chatd_agent import calls, requests, small_window

from chatd import compaction as comp, context as ctx_mod, render


def png(w: int, h: int, pad: int = 0) -> bytes:
    """A real PNG of w x h (one colour), with `pad` bytes of noise in an ancillary chunk: a 300 KB
    screenshot's bytes without drawing one."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x20" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"tEXt", b"x\x00" + os.urandom(pad))
            + chunk(b"IEND", b""))


def run_with_screenshots(n: int = 5, pad: int = 300_000):
    """The live case's shape: one user message, then n steps that each rendered the page."""
    shots = {f"img{k}": png(800, 900, pad) for k in range(n)}
    parts = []
    for k in range(n):
        parts += [{"type": "thinking", "text": f"check {k}", "step": 2 * k + 1},
                  {"type": "tool_call", "id": f"call_{k}", "name": "render_page", "step": 2 * k + 1,
                   "input": json.dumps({"path": "/workspace/clock.html"}), "status": "done",
                   "output": {"content": f"rendered file:///workspace/clock.html; screenshot "
                                         f"/workspace/.shots/call_{k}.png\nno console errors",
                              "images": [{"id": f"img{k}", "mime": "image/png"}]}},
                  {"type": "tool_call", "id": f"edit_{k}", "name": "edit_file", "step": 2 * k + 2,
                   "input": "{}", "status": "done", "output": {"content": "edited clock.html"}}]
    branch = [{"id": "u", "role": "user", "parts": [{"type": "text",
                                                      "text": "Create a clock in a single html"}]},
              {"id": "a", "role": "assistant", "parts": parts}]
    return branch, (lambda a: shots.get(a["id"]))


# ---------------------------------------------------------------------- 1. the images in a request

def _shots(body):
    msgs = body["messages"]
    return [m for m in msgs if m["role"] == "tool" and m["tool_call_id"].startswith("call_")]


def _images(shots):
    return [p for m in shots if isinstance(m["content"], list) for p in m["content"]
            if p["type"] == "image_url"]


def test_tool_screenshots_ride_in_their_tool_message_and_stay_where_they_are():
    branch, blob = run_with_screenshots()
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    msgs = body["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) == 1 and users[0]["content"] == "Create a clock in a single html"
    shots = _shots(body)
    assert len(_images(shots)) == 5 and all(isinstance(m["content"], list) for m in shots)
    assert shots[-1]["content"][0]["text"].endswith("no console errors")
    # the same branch renders to the same bytes (the engine's cache finds its prefix)
    assert render.encode(body) == render.encode(render.body("m", {}, branch, system="s",
                                                            vision=True, blob=blob))
    # another backend: the images in a user message after the results
    other = render.body("m", {}, branch, system="s", vision=True, blob=blob, inline_images=False)
    after = [m for m in other["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(after) == 5 and all(len(m["content"]) == 2 for m in after)
    # a model without vision: no image at all
    text = render.body("m", {}, branch, system="s", vision=False, blob=blob)
    assert not any(isinstance(m["content"], list) for m in text["messages"])


def test_srv546_a_new_screenshot_only_appends_to_the_prompt():
    """SRV-546 (live 2026-10-17 10:18:36): with only the newest screenshot kept, each new one turned
    the previous into a line and the engine read 90,186 tokens again. Every step's request now
    starts with the previous step's request, byte for byte, up to the cap; past it only the
    newest-but-one changes, and everything before it stays."""
    # the pagoda run's seven screenshots: every request starts with the one before it
    branch7, blob7 = run_with_screenshots(7, pad=10)
    msgs = [render.body("m", {}, [branch7[0], {"id": "a", "role": "assistant", "parts": [
        p for p in branch7[1]["parts"] if p["step"] <= step]}], system="s", vision=True,
        blob=blob7)["messages"] for step in range(1, 15)]
    for step in range(1, 14):
        assert msgs[step][:len(msgs[step - 1])] == msgs[step - 1], f"step {step + 1} rewrote it"
    k = render.TOOL_IMAGES_KEPT
    n = k + 3
    branch, blob = run_with_screenshots(n, pad=10)
    full = branch[1]["parts"]

    def at(step):                       # the request after `step` model calls
        b = [branch[0], {"id": "a", "role": "assistant",
                         "parts": [p for p in full if p["step"] <= step]}]
        return render.body("m", {}, b, system="s", vision=True, blob=blob)["messages"]

    prev = None
    for step in range(1, 2 * n + 1):
        cur = at(step)
        if prev is not None:
            shots_now = (step + 1) // 2         # odd steps add a screenshot
            if shots_now <= k or step % 2 == 0:
                assert cur[:len(prev)] == prev, f"step {step} rewrote the prompt"
            else:
                # past the cap: the messages before the newest-but-one's result stay as they were
                ids = [m.get("tool_call_id") for m in prev]
                cut = ids.index(f"call_{shots_now - 2}")
                assert cur[:cut] == prev[:cut], f"step {step} changed an earlier image"
                assert cur[cut] != prev[cut] and render.LEFT_OUT in cur[cut]["content"]
        prev = cur
    shots = _shots({"messages": at(2 * n)})
    kept = [m["tool_call_id"] for m in shots if isinstance(m["content"], list)]
    assert kept == [f"call_{i}" for i in range(k - 1)] + [f"call_{n - 1}"]
    assert all(render.LEFT_OUT in m["content"] for m in shots if not isinstance(m["content"], list))


def test_srv546_the_now_line_names_a_re_read_and_its_reason():
    """SRV-546: the card said "Prefilling 114,336 tokens" beside the run's 81,843 written tokens and
    its 38 min, which read as 36 tok/s; the engine's live row says what the cache held and why
    the rest is read again."""
    from chatd.service import prefill_words
    why = "the chat's text differs from what the engine read, at token 88,640 of 112,642"
    assert prefill_words({"total": 114_336, "cached": 29_696, "why": why}) == \
        f"Re-reading 84,640 of 114,336 tokens: {why}"
    assert prefill_words({"total": 114_336, "cached": 112_642, "why": None}) == \
        "Reading 1,694 new tokens (112,642 cached)"
    assert prefill_words({"total": 30_443, "cached": 0}) == "Prefilling 30,443 tokens"


# ---------------------------------------------------------------------- 2. the estimate

def test_the_estimate_counts_an_image_by_its_pixels_not_its_base64():
    branch, blob = run_with_screenshots()
    url = "data:image/png;base64," + base64.b64encode(blob({"id": "img0"})).decode()
    assert len(url) > 400_000
    assert comp.image_tokens(url) == 25 * 29 + 2                     # 800 x 900 at 32 pixels
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    chars, img = comp.measure(body)
    assert img == 5 * (25 * 29 + 2) and chars < 10_000, (chars, img)
    est = comp.estimate(body, 0.15)
    assert est < 6_000, est
    # the old count: every image's base64 as text, all five kept
    real = render.TOOL_IMAGES_KEPT
    render.TOOL_IMAGES_KEPT = 99
    try:
        old = render.body("m", {}, branch, system="s", vision=True, blob=blob, inline_images=False)
    finally:
        render.TOOL_IMAGES_KEPT = real
    old_chars = len(json.dumps(old["messages"], ensure_ascii=False))
    assert old_chars * 0.15 > 50 * est
    # the ratio is learned on text: the engine's count less the images' tokens
    assert comp.learn(1000 + img, chars, img) == max(0.15, min(0.8, 1000 / chars))
    assert comp.learn(img, chars, img) is None
    # JPEG and GIF sizes; an unknown format counts as IMAGE_UNKNOWN
    jpg = b"\xff\xd8\xff\xe0\x00\x10" + b"JFIF\x00" + b"\x00" * 9 + \
        b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 640, 480) + b"\x03" + b"\x00" * 9
    assert comp.image_tokens("data:image/jpeg;base64," + base64.b64encode(jpg).decode()) == \
        20 * 15 + 2
    gif = b"GIF89a" + struct.pack("<HH", 64, 64) + b"\x00" * 8
    assert comp.image_tokens("data:image/gif;base64," + base64.b64encode(gif).decode()) == 4 + 2
    assert comp.image_tokens("data:image/webp;base64,AAAA") == comp.IMAGE_UNKNOWN


# ---------------------------------------------------------------------- 3. a long run compacts

def test_step_boundary():
    def run(steps):
        return [{"role": "user", "parts": []},
                {"role": "assistant", "parts": [{"type": "text", "step": s} for s in steps]}]
    assert comp.step_boundary(run([1, 2, 3])) is None
    assert comp.step_boundary(run([1, 2, 3, 4, 5])) == (1, 3)
    assert comp.step_boundary(run([1, 2, 3, 4, 5]), prior_step=3) is None
    assert comp.step_boundary(run([1, 2, 3, 4, 5, 6, 7]), prior_step=3) == (1, 5)
    assert comp.step_boundary([{"role": "user", "parts": []}]) is None


def test_a_long_run_compacts_between_steps_and_keeps_its_user_message():
    c = Chat()
    try:
        small_window(c, window=6000)
        c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        ws = c.svc.workspace(cid)
        ws.write("a.txt", ("alpha line\n" * 20).encode())
        ws.write("b.txt", ("beta line\n" * 20).encode())
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 16})
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        s = c.wait_idle(cid, 60)
        assert s["compactions"], "compacted inside the run"
        cp = s["compactions"][0]
        assert "@" in cp["boundary_id"] and cp["boundary_id"].split("@")[0] == s["messages"][1]["id"]
        reqs = requests(c)
        after = [b for b in reqs if "Summary of the earlier part" in
                 (b["messages"][0].get("content") or "")]
        assert after, "the next steps carry the summary"
        for b in after:
            users = [m for m in b["messages"] if m["role"] == "user"]
            assert len(users) == 1 and "FAKE_CALLS" in users[0]["content"]
            assert comp.estimate(b, 0.5) + comp.MIN_ANSWER <= 6000
        assert len(reqs) > 8, "the run went on past the window"
        assert "no room left" not in json.dumps(s["messages"][-1])
    finally:
        c.close()


# ---------------------------------------------------------------------- 4. the panel after it

def test_the_panel_counts_a_run_compacted_mid_way_and_continued():
    """The live case's shape: run 1 compacted at step 43 and stopped at its step cap (100); a
    Continue (run 2) went on in the same message to step 113, and the engine refused step 114.
    The panel took run 1's stats (step 100's prompt less the whole turn's thinking, the 42 steps'
    in the summary too), counted none of run 2's calls and showed 0 for the conversation."""
    from chatd.ids import now_ms, ulid
    from chatd.runs import Run
    from chatd.service import DEFAULT_SETTINGS
    c = Chat()
    try:
        cid = c.new()
        conv = c.svc.conv(cid)

        def step(k, out=400):
            return [{"type": "thinking", "text": "t" * 8000, "step": k, "done": True},
                    {"type": "tool_call", "id": f"call_{k}", "name": "bash", "step": k,
                     "input": "{}", "status": "done", "output": {"content": "ok"},
                     "model_content": "o" * out}]
        parts = [p for k in range(1, 113) for p in step(k)] + step(113, out=4000)

        def usage(k, prompt):
            return {"step": k, "prompt_tokens": prompt, "cached_tokens": prompt,
                    "cache_source": "end", "completion_tokens": 2100, "reasoning_tokens": 2000}
        run1 = {"finish": "length", "prompt_tokens": 178_858, "completion_tokens": 210_000,
                "reasoning_tokens": 200_000, "steps": 100, "cached_tokens": 178_858,
                "step_usage": [usage(k, 178_858 - (100 - k) * 400) for k in range(51, 101)]}
        branch = [{"id": "u1", "role": "user", "parts": [{"type": "text", "text": "a pagoda"}]},
                  {"id": "a1", "role": "assistant", "parts": parts, "stats": run1}]
        c.svc.store.run("INSERT INTO compactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ulid(), cid, "a1@43", "the summary", 195_532, 18_750, now_ms(), 1000,
                         None))
        run = Run(c.svc, conv, user_message_id="u1", message_id="a1", max_steps=200,
                  started_by=None)
        run.step_usage = [usage(k, 180_795 + (k - 101) * 1867) for k in range(101, 114)]
        run.step_usage[-1].update(prompt_tokens=203_199, completion_tokens=2_795)
        run.step, run.thinking_tokens, run.call_base = 114, 26_000, (30_000, 26_000)
        # the next request: step 113's prompt and answer, and its tool's result (4,000 characters)
        want = 203_199 + 2_795 + 1_000
        # the thinking it carries: steps 43 to 113, 2,000 tokens each
        thinking = (113 - 43 + 1) * 2000
        ctx = c.svc._context(conv, dict(DEFAULT_SETTINGS), branch, run=run)
        got = {p["kind"]: p["tokens"] for p in ctx["parts"]}
        assert ctx["used"] == want, (ctx["used"], got)
        assert got["thinking"] == thinking and got["messages"] == want - thinking, got
        assert ctx["cache"]["warm"] == 203_199               # step 113's, not step 100's
        # run 2 ended with its last call refused: its stats have no prompt, its calls' do
        branch[1]["stats"] = {"finish": "error", "prompt_tokens": 0, "reasoning_tokens": 26_000,
                              "steps": 114, "interrupted": "engine",
                              "step_usage": run.step_usage}
        ctx = c.svc._context(conv, dict(DEFAULT_SETTINGS), branch)
        got = {p["kind"]: p["tokens"] for p in ctx["parts"]}
        assert ctx["used"] == want and got["thinking"] == thinking, (ctx["used"], got)
    finally:
        c.close()


# ---------------------------------------------------------------------- 5. file texts

def _prefix_kept(prev: dict, cur: dict) -> bool:
    """`cur` starts with `prev`, byte for byte: its head and every message `prev` had."""
    n = len(prev["messages"])
    return (render.encode(cur["messages"][:n]) == render.encode(prev["messages"]) and
            render.encode(cur.get("tools") or []) == render.encode(prev.get("tools") or []))


def test_a_write_stays_whole_in_the_next_request_and_a_note_sent_as_content_is_refused():
    """The live case (2026-10-20, chat 01M5DEZ7BKRZJH3T2JJ2AG4H2W): a stub in place of the
    content made the model write effects.js three times and copy the stub as new files. A write
    stays whole in the requests after it (they only append), and a stub or a folded call's note
    sent as a file's text is refused with what is on disk."""
    c = Chat()
    try:
        cid = c.new()
        page = "".join(f"const wave{i} = Math.sin({i} * t) * 0.25;\n" for i in range(150))
        assert len(page) > render.FOLD_ARG_CHARS
        c.send(cid, calls(("write_file", {"path": "ocean.js", "content": page})))
        c.wait_idle(cid, 20)
        note = render.folded_note({"name": "write_file", "status": "done", "input": json.dumps(
            {"path": "sky.js", "content": page})}).strip()
        c.send(cid, calls(("write_file", {"path": "ocean.js",
                                          "content": render.wrote_stub("ocean.js", page)}),
                          ("write_file", {"path": "sky.js", "content": note})))
        c.wait_idle(cid, 20)
        c.send(cid, calls(("edit_file", {"path": "ocean.js", "old": page[:2500],
                                         "new": page[:2500].replace("0.25", "0.5")})))
        s = c.wait_idle(cid, 20)
        reqs = requests(c)
        # every request after the write carries it as the model sent it, and starts with the
        # request before it (the engine's prefix cache and end anchor hold the turn)
        for b in reqs[1:]:
            call = next(tc for m in b["messages"] if m.get("tool_calls")
                        for tc in m["tool_calls"])
            assert json.loads(call["function"]["arguments"]) == {"path": "ocean.js",
                                                                 "content": page}
        for prev, cur in zip(reqs, reqs[1:]):
            assert _prefix_kept(prev, cur)
        made = [p for m in s["messages"] if m["role"] == "assistant" for p in m["parts"]
                if p["type"] == "tool_call"]
        assert json.loads(made[0]["input"])["content"] == page
        stub, noted = made[1], made[2]
        assert stub["status"] == "failed" and noted["status"] == "failed"
        out = stub["output"]["content"]
        assert "nothing was written" in out and "/workspace/ocean.js is on disk" in out and \
            "correct as it is, so there is nothing to write again" in out, out
        out = noted["output"]["content"]
        assert "There is no file at /workspace/sky.js yet" in out, out
        assert "sky.js" not in os.listdir(c.svc.workspace(cid).root)
        assert c.svc.workspace(cid).read("ocean.js").decode().count("0.5") > 0
    finally:
        c.close()


def test_an_edit_of_text_the_model_never_read_tells_it_to_read_the_file():
    c = Chat()
    try:
        cid = c.new()
        c.svc.workspace(cid).write("a.js", b"let x = 1;\n")
        c.send(cid, calls(("edit_file", {"path": "a.js", "old": "let y = 2;", "new": "z"})))
        s = c.wait_idle(cid, 20)
        out = next(p for m in s["messages"] if m["role"] == "assistant" for p in m["parts"]
                   if p["type"] == "tool_call")["output"]["content"]
        assert "read it again with read_file" in out and "stub" in out, out
    finally:
        c.close()


def _read_call(k, path, text, ended="2026-10-17T19:00:00.000Z", name="read_file", code=None):
    return {"type": "tool_call", "id": f"r{k}", "name": name, "step": k, "status": "done",
            "input": json.dumps({"path": path} if name == "read_file" else {"command": path}),
            "ended_at": ended, "output": {"content": text, "exit_code": code}}


def test_long_reads_stay_whole_until_a_compaction_and_are_stubs_after_it():
    lines = [f"  vec3 c{i} = mix(deep, foam, {i}.0 / 64.0);" for i in range(200)]
    read = "\n".join(f"{i:6d}\t{ln}" for i, ln in enumerate(lines, 1))
    odd = "/workspace/a [b] \"c\".js"
    log = "build start\n" + "".join(f"compiling module {i}\n" for i in range(400)) + "done in 3 s"
    branch = [{"id": "u", "role": "user", "parts": [{"type": "text", "text": "an ocean"}]},
              {"id": "a", "role": "assistant", "parts": [
                  _read_call(1, "ocean.frag", read), _read_call(2, odd, read),
                  _read_call(3, "npm run build", log, name="bash", code=0),
                  _read_call(4, "late.frag", read, ended="2026-10-17T19:45:00.000Z")]}]

    def results(fold):
        body = render.body("m", {}, branch, system="s", fold_before=fold)
        return [m["content"] for m in body["messages"] if m["role"] == "tool"]
    whole = results(None)
    assert whole[0] == read and whole[2] == log and whole[3] == read
    out = results("2026-10-17T19:44:00.000Z")          # a compaction at 19:44
    assert out[0] == ('[read /workspace/ocean.frag lines 1-200: %s bytes, left out at a '
                      'compaction; read_file {"path": "/workspace/ocean.frag", "offset": 1, '
                      '"limit": 200} reads it again]' % f"{len(read.encode()):,}"), out[0]
    # an odd path: the call in the stub is exact JSON
    call = json.loads(out[1][out[1].index("read_file ") + 10:out[1].rindex(" reads it again")])
    assert call == {"path": odd, "offset": 1, "limit": 200}
    # a command's output is not read again faithfully: its ends stay, the stub does not pretend
    assert out[2].startswith("build start\ncompiling module 0\n") and out[2].endswith(
        "compiling module 399\ndone in 3 s") and "exit code 0" in out[2] and \
        "the output is not stored, run the command again only if needed" in out[2], out[2]
    assert len(out[2]) < 1200
    # a read after the compaction stays whole
    assert out[3] == read
    # a write stub for an odd path parses as the arguments it sits in, and the guard knows it
    stub = render.wrote_stub(odd, "x\n" * 3000)
    assert json.loads(json.dumps({"content": stub}))["content"] == stub
    assert render.WROTE_STUB.fullmatch(stub) and render.WROTE_STUB.fullmatch(
        render.wrote_stub("/workspace/[draft] ocean v2.js", "y"))
    assert not render.WROTE_STUB.fullmatch("[wrote the intro] and more text")


def test_writes_and_edits_leave_the_request_only_at_a_fold_point(monkeypatch=None):
    """A write's content and an edit's long texts stay whole while the rendered calls' whole
    texts stay under FILES_HIGH_CHARS (chat and worker alike); past it the oldest leave until
    FILES_KEEP_CHARS remain, and at a compaction the calls before it are cut down to it. A folded
    call keeps its path, its result says what left; between fold points the request only
    appends."""
    page = "".join(f"const wave{i} = Math.sin({i} * t) * 0.25;\n" for i in range(150))
    old, new = page[:2500], page[:2500].replace("0.25", "0.5")
    short = {"path": "a.txt", "content": "hi"}
    assert len(page) > 5000

    def call(k, name, args, ended):
        return {"type": "tool_call", "id": f"w{k}", "name": name, "step": k, "status": "done",
                "input": json.dumps(args), "ended_at": ended,
                "output": {"content": f"wrote {args['path']}"}}
    calls_ = [call(1, "write_file", {"path": "ocean.js", "content": page},
                   "2026-10-18T10:00:00.000Z"),
              call(2, "edit_file", {"path": "ocean.js", "old": old, "new": new},
                   "2026-10-18T10:01:00.000Z"),
              call(3, "write_file", short, "2026-10-18T10:02:00.000Z")]
    calls_ += [call(k, "write_file", {"path": f"p{k}.js", "content": page.replace("wave", f"w{k}")},
                    f"2026-10-18T10:0{k}:00.000Z") for k in range(4, 8)]

    def branch(n):
        return [{"id": "u", "role": "user", "parts": [{"type": "text", "text": "an ocean"}]},
                {"id": "a", "role": "assistant", "parts": calls_[:n]}]
    high, keep, share = render.FILES_HIGH_CHARS, render.FILES_KEEP_CHARS, render.FILES_KEEP_SHARE
    render.FILES_HIGH_CHARS, render.FILES_KEEP_CHARS = 3 * len(page) + 100, len(page) + 100
    render.FILES_KEEP_SHARE = 1.0
    try:
        def body(n, fold=None):
            return render.body("m", {}, branch(n), system="s", fold_before=fold)

        def args(b):
            return [json.loads(tc["function"]["arguments"]) for m in b["messages"]
                    for tc in m.get("tool_calls") or []]
        # nothing leaves while the whole texts stay under the high mark: the calls as written
        assert args(body(3)) == [{"path": "ocean.js", "content": page},
                                 {"path": "ocean.js", "old": old, "new": new}, short]
        assert render.folded_calls(branch(4)) == set()
        # the fifth write passes it: the oldest leave until the keep mark, the newest stays
        assert render.folded_calls(branch(5)) == {"w1", "w2", "w4"}
        a = args(body(5))
        assert a == [{"path": "ocean.js"}, {"path": "ocean.js"}, short, {"path": "p4.js"},
                     json.loads(calls_[4]["input"])], a
        tool = [m["content"] for m in body(5)["messages"] if m["role"] == "tool"]
        assert tool[0] == ("wrote ocean.js\n[the content of this call (150 lines, "
                           f"{len(page.encode()):,} bytes, sha256 {render._sha(page)}) is left out "
                           "of this conversation to save room; the file was written with it as "
                           'sent. read_file {"path": "/workspace/ocean.js"} shows the file as it '
                           "is now]"), tool[0]
        assert tool[1].startswith("wrote ocean.js\n[the old and new text of this call (2,500 and "
                                  f"{len(new):,} bytes) are left out"), tool[1]
        assert tool[2] == "wrote a.txt" and tool[4] == "wrote p5.js"
        # between fold points the request only appends; a fold point rewrites it once
        changed = []
        for n in range(1, len(calls_) + 1):
            prev, cur = body(n - 1), body(n)
            k = len(prev["messages"])
            if render.encode(cur["messages"][:k]) != render.encode(prev["messages"]):
                changed.append(n)
        assert changed == [5], changed
        # a compaction is a fold point: the calls before it go down to the keep mark, newest kept
        assert render.folded_calls(branch(4), "2026-10-18T10:05:00.000Z") == {"w1", "w2"}
        assert render.folded_calls(branch(2), "2026-10-18T10:05:00.000Z") == {"w1"}
        assert render.folded_calls(branch(2), "2026-10-18T10:00:30.000Z") == set()
        # the same branch folds the same calls on every render (no clock, no state)
        assert render.encode(body(7)) == render.encode(body(7))
        # the Context panel counts what the request carries
        whole = ctx_mod.file_tokens(branch(4))
        assert whole == sum(ctx_mod.estimate_tokens(t) for t in
                            (page, old, new, "hi", json.loads(calls_[3]["input"])["content"]))
        assert ctx_mod.file_tokens(branch(4), fold_before="2026-10-18T10:05:00.000Z") == \
            ctx_mod.estimate_tokens("hi") + ctx_mod.estimate_tokens(
                json.loads(calls_[3]["input"])["content"])
    finally:
        render.FILES_HIGH_CHARS, render.FILES_KEEP_CHARS = high, keep
        render.FILES_KEEP_SHARE = share
    # a small window lowers the marks: 4 characters a token, FILES_WINDOW_SHARE of it
    assert render.files_marks(None) == (render.FILES_HIGH_CHARS, render.FILES_KEEP_CHARS)
    assert render.files_marks(262_144)[0] <= render.FILES_HIGH_CHARS
    assert render.files_marks(32_768) == (19_660, 5_898)


def _first_call(body: dict) -> str | None:
    """The id of a request's first tool call: a compaction inside the run moves it."""
    return next((tc["id"] for m in body["messages"] for tc in m.get("tool_calls") or []), None)


def test_a_long_command_folds_like_a_write_and_keeps_its_output():
    """A heredoc that writes a file is a write in a bash call (the reef run wrote its first page in
    six of them): past the high mark its command leaves the call too, its output and exit code
    stay, and a failed command folds as well (it ran); a refused write does not (nothing was
    written, the note would say it was)."""
    page = "".join(f"const reef{i} = Math.cos({i} * t);\n" for i in range(160))
    cmd = f"cat > /workspace/reef.js <<'EOF'\n{page}EOF\nnode --check /workspace/reef.js"

    def call(k, name, args, status="done", out="ok"):
        return {"type": "tool_call", "id": f"c{k}", "name": name, "step": k, "status": status,
                "input": json.dumps(args), "ended_at": f"2026-10-20T21:0{k}:00.000Z",
                "output": {"content": out, "exit_code": 1 if status == "failed" else 0}}
    parts = [call(1, "bash", {"command": cmd}, "failed", "SyntaxError\n[exit code 1]"),
             call(2, "write_file", {"path": "a.js", "content": page}, "failed", "refused"),
             call(3, "write_file", {"path": "b.js", "content": page})]
    branch = [{"id": "u", "role": "user", "parts": [{"type": "text", "text": "a reef"}]},
              {"id": "a", "role": "assistant", "parts": parts}]
    assert render.folded_calls(branch, "2026-10-20T21:02:30.000Z", window=20_000) == {"c1"}
    body = render.body("m", {}, branch, system="s", fold_before="2026-10-20T21:02:30.000Z",
                       window=20_000)
    args = [json.loads(tc["function"]["arguments"]) for m in body["messages"]
            for tc in m.get("tool_calls") or []]
    assert args[0] == {} and args[1]["content"] == page and args[2]["content"] == page, args
    tool = [m["content"] for m in body["messages"] if m["role"] == "tool"]
    assert tool[0] == ("SyntaxError\n[exit code 1]\n[the command of this call (163 lines, "
                       f"{len(cmd.encode()):,} bytes) is left out of this conversation to save "
                       "room; it ran as sent, its output is above]"), tool[0]
    assert tool[1] == "refused"


def _ocean_run(c, window: int, max_steps: int = 10):
    """The live case's shape: one user message, then steps that each read two long shader files
    and write one back."""
    small_window(c, window=window)
    c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
    cid = c.new()
    c.svc.conv(cid).tok_ratio = 0.5
    ws = c.svc.workspace(cid)
    text = "".join(f"  float h{i} = sin(p.x * {i}.0 + t) * cos(p.y * {i}.0);\n" for i in range(110))
    ws.write("ocean.frag", text.encode())
    ws.write("sky.frag", text.replace("h", "s").encode())
    c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": max_steps})
    c.send(cid, calls(("read_file", {"path": "ocean.frag"}), ("read_file", {"path": "sky.frag"}),
                      ("write_file", {"path": "ocean.js", "content": text}))
           + " FAKE_CALLS_ALWAYS")
    return cid


def test_a_compaction_lands_under_half_its_trigger_with_big_kept_steps():
    c = Chat()
    try:
        window = 50_000
        cid = _ocean_run(c, window)
        s = c.wait_idle(cid, 90)
        at = comp.compact_at(window)
        assert s["compactions"], "the run compacted"
        cp = s["compactions"][0]
        reqs = requests(c)
        first = next(i for i, b in enumerate(reqs) if "Summary of the earlier part" in
                     (b["messages"][0].get("content") or ""))
        before, after = (comp.estimate(reqs[k], 0.5) for k in (first - 1, first))
        assert after <= at * comp.TARGET_SHARE, (after, at)
        # the row's figures are the chat's request sizes, before and after
        assert cp["before_tokens"] > at * 0.9 and cp["after_tokens"] <= at * comp.TARGET_SHARE, cp
        assert abs(cp["after_tokens"] - after) <= 0.1 * after, (cp, after)
        # the reads it kept are stubs; the steps after it read whole again, and from there on each
        # request starts with the one before, byte for byte
        kept = [m["content"] for m in reqs[first]["messages"] if m["role"] == "tool"]
        assert kept and all(not k.startswith("     1\t") for k in kept), kept
        # (the writes stay whole, so this small window compacts again later, and a fold point of
        # the written files rewrites it once: each is a new start)
        def notes(b):
            return sum("is left out of this conversation" in (m.get("content") or "")
                       for m in b["messages"] if m["role"] == "tool")
        for k in range(first + 1, len(reqs)):
            again = _first_call(reqs[k]) != _first_call(reqs[k - 1])
            assert again or _prefix_kept(reqs[k - 1], reqs[k]) or \
                notes(reqs[k]) > notes(reqs[k - 1]), f"request {k} rewrote the prompt"
        line = next(ln for ln in c.lines if ln.startswith("[compact]"))
        assert line.startswith(f"[compact] {cid} {cp['before_tokens']} -> {cp['after_tokens']} "
                               f"(summary call ") and line.endswith(" ms"), line
    finally:
        c.close()


def test_a_run_whose_kept_steps_have_no_stubs_keeps_fewer_steps():
    """Results that are no file text (grep's lines) stay whole: the compaction keeps fewer than
    KEEP_STEPS steps until the request lands under half its trigger."""
    c = Chat()
    try:
        window = 24_000
        small_window(c, window=window)
        c.svc.learn_ratio = lambda *a: None
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        c.svc.workspace(cid).write("waves.txt", "".join(
            f"wave {i}: amplitude 0.25 phase {i}\n" for i in range(60)).encode())
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 10})
        c.send(cid, calls(("grep", {"pattern": "wave", "include": "*.txt"}),
                          ("grep", {"pattern": "phase", "include": "*.txt"})) + " FAKE_CALLS_ALWAYS")
        s = c.wait_idle(cid, 90)
        at = comp.compact_at(window)
        assert s["compactions"], "the run compacted"
        for cp in s["compactions"]:
            assert cp["after_tokens"] <= at * comp.TARGET_SHARE, (cp, at)
        cp = s["compactions"][0]
        reqs = requests(c)
        first = next(i for i, b in enumerate(reqs) if "Summary of the earlier part" in
                     (b["messages"][0].get("content") or ""))
        kept = {m["tool_call_id"] for m in reqs[first]["messages"] if m["role"] == "tool"}
        assert 0 < len(kept) < 2 * comp.KEEP_STEPS, kept     # two calls a step
        assert comp.estimate(reqs[first], 0.5) <= at * comp.TARGET_SHARE
    finally:
        c.close()


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
