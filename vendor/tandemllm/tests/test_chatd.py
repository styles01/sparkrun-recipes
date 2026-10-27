"""chatd (SRV-343, SRV-345, SRV-349): the store, runs that outlive the browser, the event log with
snapshot plus tail, resume after Last-Event-ID, idempotent sends, 409, stop, steer, send now,
interrupted and Continue, retries, branches, the byte-stable request, the context report.

CPU only: chatd talks to tests/fake_backend.py, the standard library's fake OpenAI server.

Run: python -m pytest -q tests/test_chatd.py   (or python tests/test_chatd.py)
"""

from __future__ import annotations

import json
import os
import random
import socket
import sqlite3
import tempfile
import threading
import time

import fake_backend
from chatd_util import Chat, apply, is_done, run_all, text_of

from chatd import render
from chatd.events import RESET_BEHIND
from chatd.store import MIGRATIONS, SCHEMA_V1, Store, StoreError

ANSWER24 = "".join(f"w{i} " for i in range(24))


# ------------------------------------------------------------------ the store

def test_the_store_migrates_forward_once_and_keeps_every_row():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "chat.sqlite3")
    s1 = Store(path, {1: SCHEMA_V1})
    assert s1.migrated == [1] and s1.schema_version == 1
    s1.run("INSERT INTO conversations (id, title, created, updated, settings, ui) VALUES "
           "('c1', 't', 1, 1, '{}', '{}')")
    s1.db.close()
    v2 = {1: SCHEMA_V1, 2: "ALTER TABLE conversations ADD COLUMN extra TEXT"}
    s2 = Store(path, v2)
    assert s2.migrated == [2] and s2.schema_version == 2
    assert s2.one("SELECT title FROM conversations WHERE id = 'c1'")[0] == "t"
    s2.db.close()
    s3 = Store(path, v2)
    assert s3.migrated == [] and s3.schema_version == 2
    s3.db.close()
    try:
        Store(path, {1: SCHEMA_V1})
        raise AssertionError("a newer file must be refused")
    except StoreError:
        pass


def test_the_daily_backup_keeps_seven():
    d = tempfile.mkdtemp()
    s = Store(os.path.join(d, "chat.sqlite3"))
    out = os.path.join(d, "backups")
    for day in range(1, 10):
        assert s.backup(out, today=f"202610{day:02d}") is not None
    assert s.backup(out, today="20261009") is None          # once a day
    kept = sorted(os.listdir(out))
    assert len(kept) == 7 and kept[0] == "chat-20261003.sqlite3", kept
    assert sqlite3.connect(os.path.join(out, kept[-1])).execute(
        "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "8"


def test_v7_drops_every_stored_conductor_profile():
    """VIS-203: a chat or a preset that held `coding`, `chat` or a test profile from the old picker
    runs under chat-app (the client kind) after the upgrade; the rest of its settings stay."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "chat.sqlite3")
    old = {v: sql for v, sql in MIGRATIONS.items() if v < 7}
    s = Store(path, old)
    for cid, st in (("c1", {"model": "TandemLLM", "profile": "coding", "effort": "high"}),
                    ("c2", {"model": "TandemLLM", "profile": None}), ("c3", {"model": "x"})):
        s.run("INSERT INTO conversations (id, title, created, updated, settings, ui) VALUES "
              "(?, 't', 1, 1, ?, '{}')", (cid, json.dumps(st)))
    s.run("INSERT INTO presets VALUES ('p1', 'n', 'TandemLLM', NULL, ?)",
          (json.dumps({"profile": "ui-check", "effort": "low"}),))
    s.db.close()
    s = Store(path, {v: sql for v, sql in MIGRATIONS.items() if v <= 7})
    assert s.migrated == [7]
    got = {r[0]: json.loads(r[1]) for r in s.all("SELECT id, settings FROM conversations")}
    assert got["c1"] == {"model": "TandemLLM", "profile": None, "effort": "high"}
    assert got["c2"] == {"model": "TandemLLM", "profile": None} and got["c3"] == {"model": "x"}
    assert json.loads(s.one("SELECT settings FROM presets")[0]) == {"profile": None,
                                                                    "effort": "low"}
    s.db.close()


def test_capabilities_list_models_only_each_by_its_own_name():
    """VIS-203: the picker's rows. The engine's model by its label; a backend's model by the
    provider's name (the card's label names the backend: two DeepSeek models were both `DeepSeek
    (cloud)`), `(cloud)` for a cloud backend, the id when the provider name is unknown; no
    conductor profile in any model."""
    from types import SimpleNamespace

    from chatd.service import Service
    cards = [
        {"id": "TandemLLM", "owned_by": "tandemllm", "tandem": {
            "backend": "engine", "label": "Nemotron 3.5 Super VL 120B-A12B",
            "pack": "nemotron-super-vl-exl3-4.0", "context_length": 262144}},
        {"id": "deepseek-flash", "owned_by": "deepseek", "tandem": {
            "backend": "deepseek", "kind": "deepseek", "label": "DeepSeek (cloud)",
            "upstream_id": "deepseek-flash", "name": "DeepSeek V4 Flash", "cloud": True,
            "context_length": 1048576}},
        {"id": "deepseek-v4-pro", "owned_by": "deepseek", "tandem": {
            "backend": "deepseek", "kind": "deepseek", "label": "DeepSeek (cloud)",
            "upstream_id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "cloud": True,
            "context_length": 1048576}},
        {"id": "Qwen/Qwen3.5-4B", "owned_by": "vllm1", "tandem": {
            "backend": "vllm1", "kind": "vllm", "label": "vLLM box", "name": None,
            "cloud": False, "upstream_id": "Qwen/Qwen3.5-4B", "context_length": 32768}},
    ]
    stub = SimpleNamespace(engine=SimpleNamespace(models=lambda: cards),
                           permission_default=lambda: "auto", agents_default=lambda: False)
    caps = Service._build_caps(stub, {"engine": {"max_len": 262144}})
    rows = {m["id"]: m for m in caps["models"]}
    assert [m["name"] for m in caps["models"]] == [
        "Nemotron 3.5 Super VL 120B-A12B", "DeepSeek V4 Flash (cloud)", "DeepSeek V4 Pro (cloud)",
        "Qwen/Qwen3.5-4B"]
    assert all(m["profiles"] == [] for m in caps["models"])
    assert rows["TandemLLM"]["backend"] == "engine"
    assert rows["deepseek-flash"]["backend"] == "cloud" and rows["deepseek-flash"]["weights"] == ""
    assert rows["Qwen/Qwen3.5-4B"]["backend"] == "proxy"
    assert caps["defaults"]["model"] == "TandemLLM" and caps["defaults"]["profile"] is None


def test_a_backend_model_gets_max_tokens_only_from_its_output_limit():
    """VIS-203: DeepSeek through the proxy has a 1,048,576 window and refuses a max_tokens above
    393,216; the window minus the prompt was sent, and the first DeepSeek chat failed at once."""
    c = Chat()
    try:
        info = {"id": "fake-a", "max_len": 1048576, "weights": "", "backend": "cloud",
                "max_output": None, "vision": False, "tools": True, "effort_budgets": {}}
        c.svc.model_info = lambda model_id: dict(info)
        cid = c.new()
        c.send(cid, "hello")
        c.wait_idle(cid)
        assert "max_tokens" not in c.last_body()
        info["max_output"] = 393216
        c.send(cid, "again")
        c.wait_idle(cid)
        assert c.last_body()["max_tokens"] == 393216
    finally:
        c.close()


def test_a_relay_without_a_window_gets_the_operators_output_limit():
    """VIS-227: a llama.cpp, Ollama or manual model's card has no context length; the output
    limit an operator set under Backends still goes out as max_tokens."""
    c = Chat()
    try:
        info = {"id": "fake-a", "max_len": 0, "weights": "", "backend": "proxy",
                "max_output": None, "vision": False, "tools": True, "effort_budgets": {}}
        c.svc.model_info = lambda model_id: dict(info)
        cid = c.new()
        c.send(cid, "hello")
        c.wait_idle(cid)
        assert "max_tokens" not in c.last_body()
        info["max_output"] = 8192
        c.send(cid, "again")
        c.wait_idle(cid)
        assert c.last_body()["max_tokens"] == 8192
    finally:
        c.close()


def test_a_settings_patch_never_stores_a_profile():
    c = Chat()
    try:
        cid = c.new()
        st, obj, _ = c.call("PATCH", f"/conversations/{cid}/settings",
                            {"profile": "ui-check", "effort": "low"})
        assert st == 200, obj
        st, conv, _ = c.call("GET", f"/conversations/{cid}")
        assert conv["settings"]["profile"] is None and conv["settings"]["effort"] == "low"
    finally:
        c.close()


# ------------------------------------------------------------------ the forward key

def test_chatd_answers_the_forward_only():
    c = Chat()
    try:
        st, obj, _ = c.call("GET", "/capabilities", key=None)
        assert st == 403 and obj["error"]["type"] == "forbidden"
        st, obj, _ = c.call("GET", "/capabilities", key="wrong" * 10)
        assert st == 403
        st, obj, h = c.call("GET", "/capabilities")
        assert st == 200 and "json" in h["Content-Type"]
        assert obj["models"][0]["id"] == "fake-a" and obj["session"]["password_set"] is False
        assert obj["efforts"][0] == "auto" and obj["defaults"]["model"] == "fake-a"
        assert not any(k.startswith("_") for k in obj)
        st, obj, _ = c.call("GET", "/session", headers={"X-Tandem-Password-Set": "1",
                                                        "X-Tandem-Signed-In": "0"})
        assert st == 200 and obj["password_set"] is True and obj["signed_in"] is False
        conn = socket.create_connection(("127.0.0.1", c.port))      # /health needs no key
        conn.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        raw = b""
        while True:                      # the head and the body may come in two segments
            got = conn.recv(4096)
            if not got:
                break
            raw += got
        assert b'"status": "ok"' in raw, raw
        conn.close()
    finally:
        c.close()


# ------------------------------------------------------------------ runs that outlive the browser

def test_an_answer_completes_with_no_client_attached():
    c = Chat()
    try:
        cid = c.new()
        r = c.send(cid, "hello there")
        assert r["message_id"] and r["run_id"]
        s = c.wait_idle(cid)
        a = s["messages"][-1]
        assert a["role"] == "assistant" and a["status"] == "done", a
        assert text_of(a) == ANSWER24
        assert a["stats"]["prompt_tokens"] > 0 and a["stats"]["completion_tokens"] == 24
        assert a["stats"]["finish"] == "stop"
        assert s["title"] == "hello there" and s["turns"] == 1 and s["status"] == "idle"
        body = c.last_body()
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert c.fake.last["headers"]["x-session-id"] == f"chat-{cid}"
        assert s["usage"]["turns"] == 1 and s["context"]["used"] > 0
    finally:
        c.close()


def test_a_late_client_sees_the_current_token_then_every_next_one():
    """The phone case: the snapshot in the middle of a run, then the tail after its seq; applying
    the events to the snapshot gives exactly the stored message, no gap and no duplicate."""
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW FAKE_LONG 40 tell me")
        c.wait(lambda: len(text_of(c.snap(cid)["messages"][-1])) > 15)
        snap = c.snap(cid)
        mid = snap["messages"][-1]["id"]
        assert snap["run"] is not None and snap["messages"][-1]["status"] == "streaming"
        partial = text_of(snap["messages"][-1])
        sse = c.events(cid, after=snap["seq"])
        frames = sse.until(is_done)
        sse.close()
        assert frames[0][1] == "current"
        ids = [int(f[0]) for f in frames if f[0] is not None]
        assert ids == list(range(int(snap["seq"]) + 1, int(snap["seq"]) + 1 + len(ids))), ids
        state = snap
        for f in frames:
            if f[0] is not None:
                state = apply(state, json.loads(f[2]))
        final = c.wait_idle(cid)
        m = next(x for x in state["messages"] if x["id"] == mid)
        stored = next(x for x in final["messages"] if x["id"] == mid)
        assert text_of(m) == text_of(stored) == "".join(f"w{i} " for i in range(40))
        assert text_of(m).startswith(partial) and m["status"] == "done"
    finally:
        c.close()


def test_reconnect_with_last_event_id_replays_from_the_next_event():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "hi")
        c.wait_idle(cid)
        last = int(c.snap(cid)["seq"])
        assert last > 8
        sse = c.events(cid, last_event_id="5")
        frames = sse.until(lambda f: f[1] == "current")
        sse.close()
        ids = [int(f[0]) for f in frames if f[0] is not None]
        assert ids == list(range(6, last + 1)), ids
        sse = c.events(cid, after="5")                 # ?after= is the same thing
        assert [int(f[0]) for f in sse.until(lambda f: f[1] == "current") if f[0]][0] == 6
        sse.close()
    finally:
        c.close()


def test_a_client_too_far_behind_gets_reset():
    c = Chat()
    try:
        cid = c.new()
        conv = c.svc.conv(cid)
        for _ in range(RESET_BEHIND + 5):
            c.svc.emit(conv, {"type": "queue.changed", "queue": []})
        sse = c.events(cid, after="1")
        f = sse.until(lambda f: True)
        assert f[0][1] == "reset" and f[0][0] is None
        sse.close()
        sse = c.events(cid, after=str(int(c.snap(cid)["seq"]) + 50))     # ahead of the log
        assert sse.until(lambda f: True)[0][1] == "reset"
        sse.close()
        sse = c.events(cid, after=str(int(c.snap(cid)["seq"]) - 10))     # close enough
        frames = sse.until(lambda f: f[1] == "current")
        assert len(frames) == 11
        sse.close()
    finally:
        c.close()


def test_the_replay_to_live_switch_has_no_gap_and_no_duplicate():
    """10,000 attaches at random moments from four threads while events are emitted without a
    pause; every subscriber's replay plus live part is contiguous from its `after`."""
    c = Chat()
    try:
        cid = c.new()
        conv = c.svc.conv(cid)
        start_seq = conv.last_seq
        attaches = 10_000
        stop = threading.Event()

        def emitter():
            while not stop.is_set():
                c.svc.emit(conv, {"type": "queue.changed", "queue": []})
        results, resets = [], []
        lock = threading.Lock()

        def attacher(seed, n):
            rng = random.Random(seed)
            for _ in range(n):
                last = conv.last_seq
                after = max(start_seq, last - rng.randint(0, 200))
                sub, replay = c.svc.attach(cid, after)
                if sub is None:
                    # a reset: the emitter wrote more than RESET_BEHIND events between the read of
                    # last_seq and the attach (a thread descheduled that long: the Spark at nice
                    # 19 under the CPU suite, 1 in 10,000)
                    assert conv.last_seq - after > RESET_BEHIND, (after, conv.last_seq)
                    with lock:
                        resets.append(after)
                    continue
                got = [r[0] for r in replay]
                for _ in range(rng.randint(0, 3)):
                    time.sleep(rng.random() * 0.002)
                    got += [i[0] for i in (sub.take(0) or []) if i[0] is not None]
                c.svc.detach(cid, sub)
                got += [i[0] for i in (sub.take(0) or []) if i[0] is not None]
                with lock:
                    results.append((after, got))
        em = threading.Thread(target=emitter)
        em.start()
        threads = [threading.Thread(target=attacher, args=(k, attaches // 4)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        em.join()
        assert len(results) + len(resets) == attaches and len(resets) < attaches // 100, \
            (len(results), len(resets))
        live = sum(1 for after, got in results if got and got[-1] > after + 200)
        for after, got in results:
            assert got == list(range(after + 1, after + 1 + len(got))), (after, got[:5])
        assert conv.last_seq - start_seq > 1000 and live > 0
    finally:
        c.close()


def test_deltas_are_coalesced():
    """A 10,000-token answer is under 500 stored events (one delta every 50 ms)."""
    c = Chat(tps=2000.0)
    try:
        cid = c.new()
        c.send(cid, "FAKE_LONG 10000 go")
        s = c.wait_idle(cid, 30)
        assert len(text_of(s["messages"][-1]).split()) == 10_000
        n = c.svc.store.one("SELECT COUNT(*) FROM events WHERE conversation_id = ?", (cid,))[0]
        assert n < 500, n
    finally:
        c.close()


# ------------------------------------------------------------------ stop, steer, send now

def test_stop_keeps_the_partial_answer_and_closes_the_engine_stream():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW FAKE_LONG 80 long one")
        c.wait(lambda: len(text_of(c.snap(cid)["messages"][-1])) > 10)
        c.ok("POST", f"/conversations/{cid}/stop")
        s = c.wait_idle(cid)
        a = s["messages"][-1]
        assert a["status"] == "stopped" and a["stats"]["finish"] == "stopped", a["status"]
        assert 0 < len(text_of(a).split()) < 80
        c.wait(lambda: c.fake.closed_early >= 1, 5)     # the engine saw the socket close
        assert s["status"] == "idle"
    finally:
        c.close()


def test_a_queued_message_goes_in_after_the_answer():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW first")
        q = c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "q1", "text": "second"})
        assert q["text"] == "second" and q["client_msg_id"] == "q1"
        assert c.snap(cid)["queue"][0]["id"] == q["id"]
        c.wait(lambda: len(c.snap(cid)["messages"]) == 4 and c.snap(cid)["run"] is None, 20)
        s = c.snap(cid)
        assert [m["role"] for m in s["messages"]] == ["user", "assistant", "user", "assistant"]
        assert text_of(s["messages"][2]) == "second" and s["queue"] == []
        sent = [m for m in c.last_body()["messages"] if m["role"] != "system"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]
        assert sent[-1]["content"] == "second"
    finally:
        c.close()


def test_a_send_during_a_run_is_queued():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW first")
        r = c.send(cid, "from the phone", cmid="phone-1")
        assert r["queued"]["text"] == "from the phone"
        c.wait(lambda: len(c.snap(cid)["messages"]) == 4 and c.snap(cid)["run"] is None, 20)
    finally:
        c.close()


def test_send_now_stops_the_answer_keeps_its_text_and_sends_the_queued_one():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW FAKE_LONG 80 first")
        q = c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "q1", "text": "steer"})
        c.wait(lambda: len(text_of(c.snap(cid)["messages"][-1])) > 6)
        c.ok("POST", f"/conversations/{cid}/queue/{q['id']}/now")
        c.wait(lambda: len(c.snap(cid)["messages"]) == 4 and c.snap(cid)["run"] is None, 20)
        s = c.snap(cid)
        first = s["messages"][1]
        assert first["status"] == "stopped" and text_of(first)
        assert text_of(s["messages"][2]) == "steer"
        sent = [m for m in c.last_body()["messages"] if m["role"] != "system"]
        assert sent[1]["role"] == "assistant" and sent[1]["content"] == text_of(first)
        assert sent[-1] == {"role": "user", "content": "steer"}
    finally:
        c.close()


def test_unqueue():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW first")
        q = c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "q1", "text": "x"})
        st, _, _ = c.call("DELETE", f"/conversations/{cid}/queue/{q['id']}")
        assert st == 204 and c.snap(cid)["queue"] == []
        s = c.wait_idle(cid)
        assert len(s["messages"]) == 2
    finally:
        c.close()


# ------------------------------------------------------------------ idempotency and 409

def test_a_double_send_is_one_message():
    c = Chat()
    try:
        cid = c.new()
        a = c.send(cid, "hello", cmid="dev-A")
        b = c.send(cid, "hello", cmid="dev-A")
        assert a == b
        s = c.wait_idle(cid)
        assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
        c.send(cid, "FAKE_SLOW again", cmid="dev-C")
        q1 = c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "dev-B", "text": "x"})
        q2 = c.ok("POST", f"/conversations/{cid}/queue", {"client_msg_id": "dev-B", "text": "x"})
        assert q1["id"] == q2["id"] and len(c.snap(cid)["queue"]) == 1
        st, obj, _ = c.call("POST", f"/conversations/{cid}/messages", {"text": "no id"})
        assert st == 400 and "client_msg_id" in obj["error"]["message"]
    finally:
        c.close()


def test_a_stale_version_gets_409_and_a_ui_patch_does_not():
    c = Chat()
    try:
        cid = c.new()
        v = c.snap(cid)["version"]
        s = c.ok("PATCH", f"/conversations/{cid}", {"version": v, "title": "Renamed"})
        assert s["title"] == "Renamed" and s["version"] == v + 1
        st, obj, _ = c.call("PATCH", f"/conversations/{cid}", {"version": v, "pinned": True})
        assert st == 409 and obj["error"]["type"] == "version_conflict"
        assert obj["current"]["version"] == v + 1 and obj["current"]["title"] == "Renamed"
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        ui = {"panes": [{"kind": "terminal", "ref": "shell", "title": "Terminal"}],
              "active": "terminal:shell", "split": 0.9}
        s = c.ok("PATCH", f"/conversations/{cid}", {"version": 1, "ui": ui})
        f = sse.until(lambda f: f[1] == "ui.changed")[-1]
        got = json.loads(f[2])["ui"]
        assert got["panes"][0]["id"] == "terminal:shell" and got["split"] == 0.75
        assert c.snap(cid)["ui"] == got and c.snap(cid)["version"] == v + 1
        c.ok("PATCH", f"/conversations/{cid}", {"version": v + 1, "pinned": True})
        f = sse.until(lambda f: f[1] == "conversation.updated")[-1]
        assert json.loads(f[2])["patch"]["version"] == v + 2       # devices learn the version
        sse.close()
    finally:
        c.close()


# ------------------------------------------------------------------ interrupted, Continue, retry

def test_a_chatd_restart_interrupts_the_run_and_continue_resumes_it():
    fk = fake_backend.start(models=["fake-a"])
    c = Chat(fake=fk)
    data = c.data
    try:
        cid = c.new()
        c.send(cid, "FAKE_SLOW FAKE_LONG 80 long")
        c.wait(lambda: len(text_of(c.snap(cid)["messages"][-1]).split()) >= 5)
    finally:
        c.close(keep_data=True)                         # SIGTERM: the run ends interrupted
    c2 = Chat(data=data, fake=(c.fake_srv, c.fake_port))
    try:
        s = c2.snap(cid)
        a = s["messages"][-1]
        assert a["status"] == "interrupted" and s["run"] is None and s["status"] == "error"
        partial = text_of(a)
        c2.ok("POST", f"/conversations/{cid}/continue")
        s = c2.wait_idle(cid)
        a = s["messages"][-1]
        assert a["status"] == "done" and len(s["messages"]) == 2
        assert text_of(a) == partial + "".join(f"closed{i} " for i in range(6))
        body = c2.last_body()
        assert body["continue_final_message"] is True and body["add_generation_prompt"] is False
        assert body["messages"][-1] == {"role": "assistant", "content": partial}
        st, obj, _ = c2.call("POST", f"/conversations/{cid}/continue")
        assert st == 409
    finally:
        c2.close()
        c.fake_srv.shutdown()


def test_a_crash_leaves_runs_that_the_next_start_interrupts():
    """No graceful stop at all: the rows say `running`; the next chatd marks them interrupted and
    tells every client with a run.done event."""
    c = Chat()
    data = c.data
    cid = c.new()
    c.send(cid, "FAKE_SLOW FAKE_LONG 80 long")
    c.wait(lambda: len(text_of(c.snap(cid)["messages"][-1]).split()) >= 3)
    c.svc.stopping.set()
    c.srv.shutdown()
    c.svc.finish_run = lambda *a: None                  # the old process writes nothing more
    c.svc.conv(cid).run.stop("shutdown")
    db = sqlite3.connect(os.path.join(data, "chat.sqlite3"))
    run = db.execute("SELECT id, state FROM runs").fetchone()
    db.execute("UPDATE runs SET state = 'running' WHERE id = ?", (run[0],))
    db.execute("UPDATE messages SET status = 'streaming' WHERE role = 'assistant'")
    db.commit()
    c2 = Chat(data=data, fake=(c.fake_srv, c.fake_port))
    try:
        assert c2.svc.interrupted_at_start == 1
        s = c2.snap(cid)
        assert s["messages"][-1]["status"] == "interrupted"
        ev = c2.svc.store.one("SELECT type, data FROM events WHERE conversation_id = ? ORDER BY "
                              "seq DESC LIMIT 1", (cid,))
        assert ev[0] == "run.done" and json.loads(ev[1])["status"] == "interrupted"
    finally:
        c2.close()
        c.fake_srv.shutdown()


def test_an_engine_that_is_down_is_retried_then_the_run_is_interrupted():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()                                        # nothing listens there
    c = Chat(engine_url=f"http://127.0.0.1:{port}", retry_s=2.5)
    try:
        cid = c.new()
        c.send(cid, "anyone?")
        c.wait(lambda: "retry in" in (c.snap(cid)["run"] or {}).get("now", ""), 5)
        s = c.wait_idle(cid, 10)
        assert s["messages"][-1]["status"] == "interrupted"
    finally:
        c.close()


def test_a_503_is_retried_and_a_400_fails_the_run():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_503_ONCE hello")
        s = c.wait_idle(cid, 15)
        assert s["messages"][-1]["status"] == "done", s["messages"][-1]
        c.fake.refuse = 400
        cid2 = c.new()
        c.send(cid2, "hello")
        s = c.wait_idle(cid2)
        a = s["messages"][-1]
        assert a["status"] == "error" and a["parts"][-1]["type"] == "error"
        assert s["status"] == "error"
    finally:
        c.close()


def test_a_stream_that_drops_is_continued_not_repeated():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_DROP_ONCE hello")
        s = c.wait_idle(cid, 15)
        a = s["messages"][-1]
        assert a["status"] == "done"
        assert text_of(a) == "w0 w1 w2 w3 w4 " + "".join(f"closed{i} " for i in range(6))
        assert c.last_body().get("continue_final_message") is True
    finally:
        c.close()


def test_an_error_frame_fails_the_run_with_its_message():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_ERROR_FRAME hi")
        s = c.wait_idle(cid)
        a = s["messages"][-1]
        assert a["status"] == "error" and "fake upstream failure" in a["parts"][-1]["message"]
    finally:
        c.close()


# ------------------------------------------------------------------ branches

def test_an_edit_makes_a_branch_and_switching_goes_back():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "first question")
        s = c.wait_idle(cid)
        u1 = s["messages"][0]["id"]
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.ok("POST", f"/conversations/{cid}/messages/{u1}/edit",
             {"client_msg_id": "e1", "text": "better question"})
        f = sse.until(lambda f: f[1] == "branch.changed")[-1]
        s = c.wait_idle(cid)
        assert json.loads(f[2])["head_id"] == s["messages"][0]["id"]
        sse.close()
        assert [text_of(m) for m in s["messages"]][0] == "better question"
        assert s["messages"][0]["siblings"] == {"index": 1, "count": 2}
        assert len(s["messages"]) == 2
        b = c.ok("POST", f"/conversations/{cid}/branch", {"message_id": s["messages"][0]["id"],
                                                          "index": 0})
        assert text_of(b["messages"][0]) == "first question" and len(b["messages"]) == 2
        assert b["messages"][0]["siblings"] == {"index": 0, "count": 2}
        assert c.snap(cid)["head_id"] == b["messages"][-1]["id"]
    finally:
        c.close()


def test_regenerate_makes_a_sibling_answer():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "q")
        s = c.wait_idle(cid)
        a1 = s["messages"][1]["id"]
        c.ok("POST", f"/conversations/{cid}/messages/{a1}/regenerate", {})
        s = c.wait_idle(cid)
        assert len(s["messages"]) == 2 and s["messages"][1]["id"] != a1
        assert s["messages"][1]["siblings"] == {"index": 1, "count": 2}
        st, obj, _ = c.call("POST", f"/conversations/{cid}/branch", {"message_id": a1, "index": 0})
        assert st == 200 and obj["messages"][1]["id"] == a1
    finally:
        c.close()


# ------------------------------------------------------------------ the request

def test_the_request_is_byte_stable_across_calls_and_restarts():
    c = Chat()
    data = c.data
    try:
        cid = c.new()
        c.send(cid, "FAKE_REASONING one")
        c.wait_idle(cid)
        c.send(cid, "two")
        c.wait_idle(cid)
        run_row = c.svc.store.one("SELECT id FROM runs ORDER BY started DESC LIMIT 1")

        def bytes_now(svc):
            from chatd.runs import Run
            conv = svc.conv(cid)
            snap = svc.snapshot(cid)
            r = Run(svc, conv, user_message_id=snap["messages"][2]["id"],
                    message_id=snap["messages"][3]["id"], max_steps=50, started_by=None)
            r.id, r.step = run_row[0], 1
            saved, conv.live = conv.live, None
            try:
                return svc.request_for(r)
            finally:
                conv.live = saved
        b1, h1 = bytes_now(c.svc)
        b2, h2 = bytes_now(c.svc)
        assert b1 == b2 and h1 == h2
        assert json.loads(b1)["messages"][2]["reasoning_content"] == "".join(
            f"think{i} " for i in range(6))
    finally:
        c.close(keep_data=True)
    c2 = Chat(data=data)
    try:
        b3, h3 = bytes_now(c2.svc)
        assert b3 == b1 and h3 == h1
        assert b"chat-" not in b1 and str(int(time.time()))[:6].encode() not in b1
    finally:
        c2.close()


def test_render_effort_and_thinking_switches():
    branch = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
    auto = render.body("m", {"effort": "auto", "thinking": True}, branch)
    assert "reasoning_effort" not in auto and "chat_template_kwargs" not in auto
    high = render.body("m", {"effort": "high", "thinking": True}, branch)
    assert high["reasoning_effort"] == "high"
    off = render.body("m", {"effort": "high", "thinking": False}, branch)
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in off
    assert render.headers("C1") == {"x-session-id": "chat-C1", "X-QSE-Client": "chat-app",
                                    "X-Tandem-Cost": "1"}


def test_think_tags_and_reasoning_content_become_thinking_parts():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_THINK_TAGS hi")
        a = c.wait_idle(cid)["messages"][-1]
        assert [p["type"] for p in a["parts"]] == ["thinking", "text"], a["parts"]
        assert text_of(a, "thinking") == "plan more " and text_of(a) == ANSWER24
        assert a["parts"][0]["done"] is True
        c.send(cid, "FAKE_REASONING again")
        a = c.wait_idle(cid)["messages"][-1]
        assert [p["type"] for p in a["parts"]] == ["thinking", "text"]
        assert a["parts"][0]["tokens"] == 6 and a["stats"]["reasoning_tokens"] == 6
    finally:
        c.close()


def test_a_tag_cut_across_chunks_is_still_split():
    from chatd.runs import Run

    got = []

    class Svc:
        def append(self, run, kind, text):
            got.append((kind, text))

        def close_thinking(self, run):
            got.append(("close", ""))

    r = Run.__new__(Run)
    r.svc, r.carry, r.in_tag, r.phase, r.tokens = Svc(), "", False, "writing", 0
    r.thinking_tokens, r.marks, r.t_first, r.call_first = 0, [], None, None
    r.set_phase = lambda *a, **k: None
    r.trim = False
    for piece in ["<th", "ink>\nabc</thi", "nk>", "\n\n", "tail <", "b>"]:
        r._content(piece)
    kinds = [k for k, _ in got]
    assert "".join(t for k, t in got if k == "thinking") == "abc"
    assert "".join(t for k, t in got if k == "text") == "tail <b>"
    assert kinds.index("close") < kinds.index("text")


# ------------------------------------------------------------------ the rest of the contract

def test_the_list_search_folders_and_delete():
    c = Chat()
    try:
        a, b = c.new(), c.new()
        c.send(a, "apples and pears")
        c.wait_idle(a)
        f = c.ok("POST", "/folders", {"name": "Fruit"})
        c.ok("PATCH", f"/conversations/{a}", {"folder_id": f["id"]})
        lst = c.ok("GET", "/conversations?archived=0")
        assert {x["id"] for x in lst["conversations"]} == {a, b}
        assert lst["folders"] == [f]
        row = next(x for x in lst["conversations"] if x["id"] == a)
        assert row["preview"] == "apples and pears" and row["turns"] == 1
        assert row["status"] == "idle" and row["folder_id"] == f["id"]
        assert [x["id"] for x in c.ok("GET", "/conversations?q=pear")["conversations"]] == [a]
        st, _, _ = c.call("DELETE", f"/conversations/{b}")
        assert st == 204
        assert c.call("GET", f"/conversations/{b}")[0] == 404
        assert [x["id"] for x in c.ok("GET", "/conversations")["conversations"]] == [a]
        c.svc.sweep_once(now=int(time.time() * 1000) + 8 * 86_400_000)
        assert c.svc.store.one("SELECT COUNT(*) FROM conversations WHERE id = ?", (b,))[0] == 0
    finally:
        c.close()


def test_routes_without_a_sandbox_and_what_comes_later():
    """On a box without the sandbox image: the tools that need it are not offered, the sandbox
    routes say why, terminals list nothing; MCP, skills, prompts and presets are later."""
    c = Chat()
    try:
        cid = c.new()
        names = [t["id"] for t in c.ok("GET", "/tools")]
        assert names[:3] == ["read_file", "write_file", "edit_file"] and "bash" not in names
        for p in ("/packages", "/mcp/servers", "/prompts", "/presets"):
            assert c.ok("GET", p) == []
        assert c.ok("GET", f"/conversations/{cid}/terminals") == []
        st, obj, _ = c.call("POST", f"/conversations/{cid}/sandbox/start")
        assert st == 503 and obj["error"]["type"] == "sandbox_unavailable"
        st, obj, _ = c.call("POST", f"/conversations/{cid}/compact")
        assert st == 202
        snap = c.snap(cid)
        assert snap["sandbox"]["state"] == "error" and "not installed" in snap["sandbox"]["error"]
        assert snap["compactions"] == []
        assert c.ok("GET", "/capabilities")["session"]["sandbox"] == "not_installed"
    finally:
        c.close()


def test_settings_patch():
    c = Chat()
    try:
        cid = c.new()
        s = c.ok("PATCH", f"/conversations/{cid}/settings", {"effort": "high", "bogus": 1})
        assert s["effort"] == "high" and "bogus" not in s
        st, _, _ = c.call("PATCH", f"/conversations/{cid}/settings", {"effort": "huge"})
        assert st == 400
        c.send(cid, "hi")
        c.wait_idle(cid)
        assert c.last_body()["reasoning_effort"] == "high"
    finally:
        c.close()


def test_attachments_upload_and_go_to_the_model():
    c = Chat()
    try:
        cid = c.new()
        boundary = "XyZ"
        raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"notes."
               f"txt\"\r\nContent-Type: text/plain\r\n\r\nline one\nline two\r\n--{boundary}--\r\n"
               ).encode()
        att = c.ok("POST", "/attachments", raw=raw,
                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert att["kind"] == "text" and att["bytes"] == len(b"line one\nline two")
        st, data, h = c.call("GET", f"/attachments/{att['id']}")
        assert st == 200 and data == b"line one\nline two" and h["Content-Type"] == "text/plain"
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": "a1", "text": "read it", "attachments": [att["id"]]})
        s = c.wait_idle(cid)
        assert s["messages"][0]["attachments"][0]["id"] == att["id"]
        assert "line two" in c.last_body()["messages"][-1]["content"]
        st, _, _ = c.call("POST", f"/conversations/{cid}/messages",
                          {"client_msg_id": "a2", "text": "x", "attachments": ["nope"]})
        assert st == 400
    finally:
        c.close()


def test_the_global_stream():
    c = Chat()
    try:
        from chatd_util import SseReader
        g = SseReader(c.port, "/v1/chat-app/events")
        g.until(lambda f: f[1] == "current")
        cid = c.new()
        f = g.until(lambda f: f[1] == "conversation.created")[-1]
        assert json.loads(f[2])["conversation"]["id"] == cid
        c.send(cid, "hi")
        g.until(lambda f: f[1] == "run.done")
        g.close()
    finally:
        c.close()


def test_two_runs_at_most_the_third_waits():
    c = Chat(max_runs=2)
    try:
        ids = [c.new() for _ in range(3)]
        for cid in ids:
            c.send(cid, "FAKE_SLOW FAKE_LONG 20 x")
        c.wait(lambda: "free run slot" in (c.snap(ids[2])["run"] or {}).get("now", ""), 5)
        assert c.fake.max_streams <= 2
        for cid in ids:
            c.wait_idle(cid, 20)
        assert c.fake.max_streams == 2
    finally:
        c.close()


def test_a_cloud_models_run_takes_no_run_slot():
    # live (2026-10-17): two chats on the engine held both run slots and a third on DeepSeek
    # (cloud) waited "for a free run slot"; a cloud call runs at the provider, not on the box
    srv, port = fake_backend.start(models=["fake-a", "fake-cloud"])
    srv.fake.cards["fake-cloud"] = {"tandem": {"kind": "deepseek", "cloud": True}}
    c = Chat(max_runs=2, fake=(srv, port))
    try:
        ids = [c.new() for _ in range(3)]
        c.ok("PATCH", f"/conversations/{ids[2]}/settings", {"model": "fake-cloud"})
        for cid in ids:
            c.send(cid, "FAKE_SLOW FAKE_LONG 20 x")
        c.wait(lambda: c.fake.max_streams == 3, 10)     # the cloud run beside the two
        for cid in ids:
            c.wait_idle(cid, 20)
        assert "free run slot" not in ((c.snap(ids[2])["run"] or {}).get("now") or "")
        assert not c.svc.slots.holders
    finally:
        c.close()
        srv.shutdown()


def test_the_title_call_names_the_chat_and_a_rambling_answer_is_not_a_title():
    c = Chat(titles=True)
    try:
        cid = c.new()
        c.send(cid, "FAKE_LONG 3 a short one")         # the fake's title answer: three words
        c.wait_idle(cid)
        c.wait(lambda: c.snap(cid)["title"] == "w0 w1 w2", 10)
        body = [b for b in c.fake.bodies if b.get("stream") is False][-1]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] == 48
        cid2 = c.new()
        c.send(cid2, "tell me something long")         # 24 words back: not a title
        c.wait_idle(cid2)
        time.sleep(1.0)
        assert c.snap(cid2)["title"] == "tell me something long"
        renamed = c.new()
        c.ok("PATCH", f"/conversations/{renamed}", {"title": "Mine"})
        c.send(renamed, "FAKE_LONG 3 x")
        c.wait_idle(renamed)
        time.sleep(1.0)
        assert c.snap(renamed)["title"] == "Mine"       # a person's title stays
    finally:
        c.close()


def test_logs_carry_no_message_text():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "my secret plan for the garden")
        c.wait_idle(cid)
        c.ok("POST", f"/conversations/{cid}/stop")
        text = "\n".join(c.lines)
        assert "[run]" in text
        for word in ("secret", "garden", "w0", "w1 "):
            assert word not in text, word
    finally:
        c.close()


def test_a_big_snapshot_is_fast():
    c = Chat()
    try:
        cid = c.new()
        st = c.svc.store
        parent = None
        with st.tx() as db:
            for i in range(2000):
                mid = f"M{i:06d}"
                db.execute("INSERT INTO messages (id, conversation_id, parent_id, role, created, "
                           "status, parts) VALUES (?, ?, ?, ?, ?, 'done', ?)",
                           (mid, cid, parent, "user" if i % 2 == 0 else "assistant", i,
                            json.dumps([{"type": "text", "text": "x" * 400}])))
                parent = mid
            db.execute("UPDATE conversations SET head_id = ? WHERE id = ?", (parent, cid))
        c.svc.conv(cid).context = None
        t0 = time.monotonic()
        s = c.snap(cid)
        dt = time.monotonic() - t0
        assert len(s["messages"]) == 2000 and dt < 1.0, dt
        t0 = time.monotonic()
        c.ok("GET", "/conversations")
        assert time.monotonic() - t0 < 1.0
    finally:
        c.close()


def test_the_context_report_follows_the_last_answer():
    c = Chat()
    try:
        cid = c.new()
        c.send(cid, "FAKE_CACHED 3 hi there")
        s = c.wait_idle(cid)
        ctx = s["context"]
        stats = s["messages"][-1]["stats"]
        assert ctx["used"] == sum(p["tokens"] for p in ctx["parts"] if p["kind"] != "reserved")
        msgs = next(p for p in ctx["parts"] if p["kind"] == "messages")["tokens"]
        assert msgs == stats["prompt_tokens"] + stats["completion_tokens"]
        assert ctx["cache"]["warm"] == 3 and ctx["cache"]["cold"] == stats["prompt_tokens"] - 3
        assert ctx["model"]["name"] == "fake-a" and ctx["model"]["backend"] == "proxy"
        assert "hi there" not in json.dumps(ctx) and "w0" not in json.dumps(ctx)
    finally:
        c.close()


def test_the_peek_asks_with_the_next_requests_head_and_the_usage_frame_fills_the_panel():
    """SRV-349: the usage frame's `tandem` gives the panel the harness's choices and the system and
    tool tokens; on attach chatd peeks with the body its next request would send -- the same system
    block and tools, the chat's session -- and prewarms a chat the engine has on the NVMe."""
    c = Chat()
    try:
        c.fake.peek = {"tier": "nvme_spill", "cached_tokens": 40, "reread_tokens": 3,
                       "reread_eta_ms": 12.5, "tokens": 43}
        cid = c.new()
        c.send(cid, "FAKE_TANDEM hi there")
        s = c.wait_idle(cid)
        ctx = s["context"]
        assert ctx["harness"] == {"class": "code_edit", "effort": "xhigh",
                                  "packages": [{"name": "coding", "tokens": 7}]}, ctx["harness"]
        parts = {p["kind"]: p["tokens"] for p in ctx["parts"]}
        assert parts["system"] == 2 and parts["tools"] == 3, parts
        assert ctx["used"] == sum(p["tokens"] for p in ctx["parts"] if p["kind"] != "reserved")
        sent = c.last_body()
        c.wait(lambda: any("session" in b for b in c.fake.bodies), 10)
        peek = [b for b in c.fake.bodies if "session" in b][-1]
        assert peek["session"] == f"chat-{cid}" and "stream" not in peek
        assert peek["messages"][0] == sent["messages"][0] and peek["messages"][0]["role"] == "system", (peek["messages"][:1], sent["messages"][:1])
        assert [t["function"]["name"] for t in peek["tools"]] == \
            [t["function"]["name"] for t in sent["tools"]] and peek["tools"]
        c.wait(lambda: c.fake.prewarms == 1, 10)
        c.wait(lambda: (c.snap(cid)["context"] or {}).get("cache", {}).get("source") ==
               "nvme_spill", 10)
        cache = c.snap(cid)["context"]["cache"]
        assert cache["warm"] == 40 and cache["cold"] == 3 and cache["cold_eta_ms"] == 12.5, cache
        assert "hi there" not in json.dumps(peek.get("session")) and \
            "hi there" not in json.dumps(c.snap(cid)["context"])
    finally:
        c.close()


def test_the_reports_profile_is_the_served_weight_profile():
    """rc-0.8.0 live: the Context panel's "Model and pack" said profile none on Flash-Next 3.05 (it
    showed the chat's conductor profile, which a raw server does not have). The profile the engine
    serves comes first; a server that names none leaves the chat's own."""
    from chatd import context as ctx_mod
    model = {"id": "TandemLLM", "max_len": 262144, "weights": "flashnext-3.05", "profile": "flashnext-3.05"}
    r = ctx_mod.report(model=model, settings={"profile": None}, branch_tokens=10, files_tokens=0,
                       stats=None, run=None, row=None, facts={}, peek=None, harness=None)
    assert r["model"]["profile"] == "flashnext-3.05" and r["window"] == 262144
    r = ctx_mod.report(model=dict(model, profile=None), settings={"profile": "coding"}, branch_tokens=10,
                       files_tokens=0, stats=None, run=None, row=None, facts={}, peek=None, harness=None)
    assert r["model"]["profile"] == "coding"


def test_run_metrics_and_context_are_ephemeral():
    c = Chat()
    try:
        cid = c.new()
        sse = c.events(cid)
        sse.until(lambda f: f[1] == "current")
        c.send(cid, "FAKE_SLOW FAKE_LONG 30 x")
        frames = sse.until(is_done, 20)
        sse.close()
        m = [f for f in frames if f[1] == "run.metrics"]
        assert m and all(f[0] is None for f in m)
        assert all(f[0] is None for f in frames if f[1] == "context.changed")
        stored = {r[0] for r in c.svc.store.all("SELECT DISTINCT type FROM events")}
        assert "run.metrics" not in stored and "context.changed" not in stored
        ev = json.loads(m[-1][2])
        assert ev["tokens"] > 0 and "thinking_used" in ev
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
