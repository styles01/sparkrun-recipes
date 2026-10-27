"""The conductor's configuration (SRV-119, QA SRV-120): versions, hot reload, validation, locked
profiles, restore, the role test, a corrupt file, and the contract files.

The Gherkin of SRV-119 is the matrix:
  * a write creates the next version with its note, and the next reader (another Store on the same
    directory, as a second server process would be) runs it without a restart;
  * invalid config is refused with the reason and the current version is unchanged, including a
    Tier 1 task set to act while its role is offline;
  * raw and bench are read-only (409);
  * a restore is a new version equal to the old one, noted "restore of vN";
  * a role test answers with the latency and the answer, or why it failed;
  * a corrupt current file keeps the last good version and says so in the status.
Plus: a note is required, two editors cannot overwrite each other (base_version), the diff names
the changed leaves, and docs/contract/conductor-v1 holds exactly these schemas.

Run: python tests/test_conductor_config.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import fake_decider, schema  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from server.conductor.store import ConfigError, Store, diff  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")


def _store(**kw) -> Store:
    return Store(tempfile.mkdtemp(prefix="cond-"), **kw)


def _refused(fn, status):
    try:
        fn()
    except ConfigError as exc:
        assert exc.status == status, (exc.status, str(exc))
        return exc
    raise AssertionError(f"expected a {status}")


def test_the_seeds_are_valid_and_version_0():
    st = _store()
    for kind in ("settings", "router", "roles", "packages"):
        v, doc = st.get(kind)
        assert v == 0 and st.check(kind, None, doc) == [], kind
    for n in ("chat", "coding", "overnight"):
        v, doc = st.get("profiles", n)
        assert v == 0 and st.check("profiles", n, doc) == [], n
    assert st.names() == ["bench", "chat", "coding", "overnight", "raw"]


def test_a_write_creates_a_version_and_hot_reloads():
    """Given coding at version 3, a PUT with D1 to act and a note makes version 4, and the next
    reader runs it: a second Store on the same directory, as a restart would, sees it at once."""
    st = _store()
    for i in range(3):
        _, doc = st.get("profiles", "coding")
        doc["description"] = f"edit {i}"
        st.put("profiles", "coding", doc, note=f"edit {i}", author="test")
    assert st.get("profiles", "coding")[0] == 3
    other = Store(st.root)
    assert other.get("profiles", "coding")[1]["description"] == "edit 2"
    _, doc = st.get("profiles", "coding")
    doc["tasks"]["D1"]["mode"] = "act"
    v = st.put("profiles", "coding", doc, note="D1 to act", author="test")
    assert v == 4
    got_v, got = other.get("profiles", "coding")
    assert got_v == 4 and got["tasks"]["D1"]["mode"] == "act", "the other reader reloaded"
    assert st.versions("profiles", "coding")[0]["note"] == "D1 to act"
    assert st.meta("profiles", "coding")["author"] == "test"


def test_a_document_held_by_a_request_never_changes():
    """A request keeps the document it was given; a write makes a new object."""
    st = _store()
    _, held = st.get("router")
    doc = json.loads(json.dumps(held))
    doc["stickiness"]["switch"] = 0.95
    st.put("router", None, doc, note="stickier", author="t")
    assert held["stickiness"]["switch"] == 0.8
    assert st.get("router")[1]["stickiness"]["switch"] == 0.95


def test_invalid_config_is_refused_with_the_reason():
    st = _store()
    _, doc = st.get("profiles", "coding")
    bad = json.loads(json.dumps(doc))
    bad["decisions"]["effort"] = "maybe"
    exc = _refused(lambda: st.put("profiles", "coding", bad, note="x", author="t"), 422)
    assert any("decisions.effort" in e for e in exc.errors), exc.errors
    bad = json.loads(json.dumps(doc))
    bad["surprise"] = 1
    exc = _refused(lambda: st.put("profiles", "coding", bad, note="x", author="t"), 422)
    assert any("unknown key 'surprise'" in e for e in exc.errors)
    bad = json.loads(json.dumps(doc))
    bad["routing"]["nope"] = {"effort": "low"}
    exc = _refused(lambda: st.put("profiles", "coding", bad, note="x", author="t"), 422)
    assert any("no class 'nope'" in e for e in exc.errors)
    assert st.get("profiles", "coding")[0] == 0, "the current version is unchanged"
    # the router: a package that does not exist, a wrapper without {text}, two classes, one id
    _, r = st.get("router")
    bad = json.loads(json.dumps(r))
    bad["classes"][0]["policy"]["packages"] = ["missing-pkg"]
    bad["injection"]["wrapper"] = "no placeholder here"
    bad["classes"][1]["id"] = bad["classes"][0]["id"]
    errs = st.check("router", None, bad)
    assert any("no package 'missing-pkg'" in e for e in errs)
    assert any("{text}" in e for e in errs)
    assert any("share an id" in e for e in errs)
    # a package a class uses cannot be removed
    _, p = st.get("packages")
    bad = json.loads(json.dumps(p))
    del bad["packages"]["ui-design"]
    assert any("used by a class" in e for e in st.check("packages", None, bad))


def test_act_on_an_offline_role_is_refused():
    """When a PUT sets D3 to act while the decider role is offline: 422 with the reason. (E2,
    the claim check, asks Qwen in place and needs no role since 2026-10-02: act is accepted.)"""
    st = _store(roles_up=lambda: {"decider": False, "classifier": True})
    _, doc = st.get("profiles", "coding")
    doc["tasks"]["E2"]["mode"] = "act"
    assert st.put("profiles", "coding", doc, note="E2 to act", author="t") == 1
    doc["tasks"]["D3"]["mode"] = "act"
    exc = _refused(lambda: st.put("profiles", "coding", doc, note="D3 to act", author="t"), 422)
    assert "decider" in str(exc) and "offline" in str(exc), str(exc)
    st.roles_up = lambda: {"decider": True}
    assert st.put("profiles", "coding", doc, note="D3 to act", author="t") == 2


def test_raw_and_bench_are_read_only():
    st = _store()
    for n in ("raw", "bench"):
        _, doc = st.get("profiles", n)
        _refused(lambda: st.put("profiles", n, doc, note="try", author="t"), 409)
        _refused(lambda: st.restore("profiles", n, 0, author="t"), 409)


def test_a_note_is_required_and_editors_do_not_overwrite_each_other():
    st = _store()
    _, doc = st.get("settings")
    _refused(lambda: st.put("settings", None, doc, note="  ", author="t"), 422)
    st.put("settings", None, doc, note="first", author="a", base=0)
    exc = _refused(lambda: st.put("settings", None, doc, note="second", author="b", base=0), 409)
    assert "version 1" in str(exc)


def test_restore():
    """POST restore with version 2 makes version 5 current, equal to 2, noted "restore of v2"."""
    st = _store()
    for i in range(4):
        _, doc = st.get("router")
        doc["stickiness"]["switch"] = 0.5 + i / 10
        st.put("router", None, doc, note=f"switch {i}", author="t")
    v = st.restore("router", None, 2, author="t")
    assert v == 5
    assert st.get("router")[1] == st.version("router", None, 2)
    assert st.versions("router")[0]["note"] == "restore of v2"
    assert st.versions("router")[0]["restored_from"] == 2
    assert st.restore("router", None, 0, author="t") == 6
    assert st.get("router")[1] == schema.seed("router")


def test_a_corrupt_file_keeps_the_last_good_version():
    st = _store()
    _, doc = st.get("roles")
    doc["decider"]["timeout_ms"] = 350
    st.put("roles", None, doc, note="350 ms", author="t")
    # version 2 is written by hand and broken; the pointer moves to it
    d = os.path.join(st.root, "roles")
    with open(os.path.join(d, "0002.json"), "w") as f:
        f.write("{not json")
    time.sleep(0.01)
    with open(os.path.join(d, "current"), "w") as f:
        f.write("2\n")
    v, got = st.get("roles")
    assert v == 1 and got["decider"]["timeout_ms"] == 350, "the last good version stays"
    status = st.status()["documents"]["roles"]
    assert status["error"] and "0002.json" in status["error"]["file"], status
    # a schema-invalid version is refused the same way; a fresh reader falls back to v1
    with open(os.path.join(d, "0002.json"), "w") as f:
        json.dump({"version": 2, "doc": {"decider": {"endpoint": 5}}}, f)
    os.utime(os.path.join(d, "current"))
    fresh = Store(st.root)
    assert fresh.get("roles")[0] == 1
    # the pointer missing means the built-in defaults
    os.remove(os.path.join(d, "current"))
    assert Store(st.root).get("roles")[0] == 0


def test_the_diff_names_the_changed_leaves():
    a = schema.seed("router")
    b = json.loads(json.dumps(a))
    b["classes"][0]["policy"]["effort"] = "high"
    b["stickiness"]["switch"] = 0.7
    b["classes"].append({"id": "data", "label": "Data", "description": "analyse a table of data",
                         "policy": {"effort": "high"}})
    paths = {c["path"] for c in diff(a, b)}
    assert "$.classes[code_edit].policy.effort" in paths
    assert "$.stickiness.switch" in paths
    assert "$.classes[data]" in paths


def _fake_decider(**cfg):
    fake_decider.H.cfg = {"model": "laya-english", "context": 512, "question": 192,
                          "delay_ms": 0, **cfg}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fake_decider.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_a_role_test_answers_or_says_why():
    srv = _fake_decider()
    try:
        c = Conductor(tempfile.mkdtemp(), journal=False)
        _, roles = c.store.get("roles")
        roles["decider"]["endpoint"] = f"http://127.0.0.1:{srv.server_address[1]}"
        c.store.put("roles", None, roles, note="the fake", author="t")
        out = c.role_test("decider")
        assert out["ok"] and out["answer"]["label"] == "code_edit", out
        assert out["ms"] >= 0 and out["answer"]["stage"] == "decider"
        roles["decider"]["endpoint"] = "http://127.0.0.1:9"      # nothing listens
        c.store.put("roles", None, roles, note="gone", author="t")
        out = c.role_test("decider")
        assert not out["ok"] and out["error"] == "decider_unreachable", out
        assert c.role_test("nobody")["error"] == "no such role"
        # the health view: up, down, off
        roles["decider"]["endpoint"] = f"http://127.0.0.1:{srv.server_address[1]}"
        c.store.put("roles", None, roles, note="back", author="t")
        h = c.roles_health(force=True)
        assert h["decider"]["up"] and h["librarian"]["mode"] == "off" and not h["librarian"]["up"]
    finally:
        srv.shutdown()


def test_the_packaged_policy_copy_is_the_policy_file():
    """An installed proxy reads server/conductor/monitor/policy.default.json (no ops/, maybe no
    PyYAML); it must be ops/supervisor.yaml, as `tools/conductor_policy.py --json` writes it."""
    from server.conductor.monitor import policy as mpolicy
    with open(mpolicy.DEFAULT_JSON) as f:
        copy_ = json.load(f)
    assert copy_ == json.loads(json.dumps(mpolicy.load().data)), \
        "policy.default.json is stale; run tools/conductor_policy.py --json"
    mpolicy.Policy(copy_)


def test_the_contract_files_are_these_schemas():
    """docs/contract/conductor-v1/<kind>.config.schema.json equals schema.SCHEMAS[kind]
    (tools/conductor_contract.py --write regenerates them)."""
    for kind, sch in schema.SCHEMAS.items():
        path = os.path.join(CONTRACT, f"{kind}.config.schema.json")
        with open(path) as f:
            on_disk = json.load(f)
        on_disk.pop("$schema", None)
        on_disk.pop("title", None)
        assert on_disk == json.loads(json.dumps(sch)), f"{path} is stale; run " \
                                                       f"tools/conductor_contract.py --write"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
