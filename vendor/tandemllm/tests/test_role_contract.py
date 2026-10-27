"""The role contract fields (SRV-133, QA SRV-134): kind, window, preemptible, latency budget,
fallback, tokenizer hash, writes and sharing in the roles document, with their validation, the
defaults a document stored before them loads with, and the decision runtime's registry checking
its models against the same contract.

The Gherkin of SRV-133 is the matrix:
  1 a complete role validates          test_1_* (unit, and the HTTP API on the fake engine)
  2 a missing claim or window is refused  test_2_* (unit, one test per field; HTTP: the version
                                        does not move)
  3 only the reasoner writes client-visible tokens   test_3_*
  4 the live v1 document still loads   test_4_* (the box's roles document of 2026-10-01 as the
                                        fixture: defaults, the journal row, a restore)
Plus: in_decode is for a decider or a probe, the tokenizer hash decides the forms of sharing, an
unknown value is refused, the planner reads the claim, the registry refuses a model that breaks
the contract.

Run: python tests/test_role_contract.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import residency, schema  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from server.conductor.store import ConfigError, Store  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "roles-v2-box.json")
QWEN_SHA = "ab" * 32


def _store(**kw) -> Store:
    return Store(tempfile.mkdtemp(prefix="roles-"), **kw)


def _refused(fn, status=422) -> ConfigError:
    try:
        fn()
    except ConfigError as exc:
        assert exc.status == status, (exc.status, str(exc))
        return exc
    raise AssertionError(f"expected a {status}")


def _clerk(**over) -> dict:
    r = {"endpoint": "http://127.0.0.1:8031", "model": "gemma-4-e4b", "contract": "openai-chat",
         "timeout_ms": 20000, "memory_claim_gib": 11.0, "mode": "shadow", "kind": "clerk",
         "window": "idle", "preemptible": True, "latency_budget_ms": 1500, "fallback": "skip",
         "tokenizer_sha256": "none", "writes": "hidden_notes", "sharing": ["text"]}
    r.update(over)
    return r


def _with(st: Store, name: str, role: dict) -> dict:
    _, doc = st.get("roles")
    doc[name] = role
    return doc


def _put_v1(root: str, doc: dict, v: int = 2) -> None:
    """A roles version written the way the box's store holds it, before SRV-133."""
    d = os.path.join(root, "roles")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{v:04d}.json"), "w") as f:
        json.dump({"version": v, "note": "0.3.6-test", "author": "admin token 127.0.0.1",
                   "created_at": "2026-10-01T16:59:47Z", "restored_from": None, "doc": doc}, f)
    with open(os.path.join(d, "current"), "w") as f:
        f.write(f"{v}\n")


# ------------------------------------------------------------------------- 1 a complete role
def test_1_a_complete_role_validates_and_reads_back_every_field():
    st = _store()
    doc = _with(st, "clerk", _clerk())
    assert st.put("roles", None, doc, note="the clerk in shadow", author="t") == 1
    v, got = st.get("roles")
    assert v == 1 and got["clerk"] == _clerk()
    for name, r in got.items():
        assert set(schema.ROLE_CONTRACT) <= set(r), name
    assert "memory_gb" not in json.dumps(got), "the old field name is gone from a new version"


def test_1_the_seed_signs_the_contract():
    doc = schema.seed("roles")
    assert schema.check("roles", doc) == []
    assert doc["decider"]["kind"] == "decider" and doc["decider"]["window"] == "idle"
    assert doc["classifier"]["kind"] == "reasoner" and doc["classifier"]["window"] == "any"
    assert all(r["writes"] == "nothing" and r["sharing"] == ["text"] for n, r in doc.items()
               if n != "narrator")
    # P1: the narrator writes the state lines, between Qwen's rounds, text only
    assert doc["narrator"]["writes"] == "state_lines" and doc["narrator"]["sharing"] == ["text"]
    assert doc["narrator"]["window"] == "between_rounds" and doc["narrator"]["kind"] == "generative"


# ------------------------------------------------------------------------- 2 a missing field
def _missing(field: str) -> None:
    st = _store()
    role = _clerk()
    del role[field]
    exc = _refused(lambda: st.put("roles", None, _with(st, "clerk", role), note="x", author="t"))
    assert f"$.clerk: missing {field!r}" in exc.errors, exc.errors
    assert st.get("roles")[0] == 0, "the version did not change"


def test_2_a_role_without_a_claim_is_refused():
    _missing("memory_claim_gib")


def test_2_a_role_without_a_window_is_refused():
    _missing("window")


def test_2_every_contract_field_is_required_and_named():
    for field in schema.ROLE_CONTRACT:
        _missing(field)


def test_2_an_unknown_value_anywhere_is_refused():
    st = _store()
    for k, bad in (("window", "sometimes"), ("kind", "oracle"), ("fallback", "retry"),
                   ("writes", "everything"), ("sharing", ["kv"]), ("preemptible", "yes"),
                   ("latency_budget_ms", 0), ("tokenizer_sha256", "ABC"),
                   ("memory_claim_gib", -1)):
        exc = _refused(lambda: st.put("roles", None, _with(st, "clerk", _clerk(**{k: bad})),
                                      note="x", author="t"))
        assert any(e.startswith(f"$.clerk.{k}") for e in exc.errors), (k, exc.errors)
    role = dict(_clerk(), colour="blue")
    exc = _refused(lambda: st.put("roles", None, _with(st, "clerk", role), note="x", author="t"))
    assert "$.clerk: unknown key 'colour'" in exc.errors


# ------------------------------------------------------------------------- 3 the honesty rule
def test_3_only_the_reasoner_writes_client_visible_tokens():
    st = _store()
    exc = _refused(lambda: st.put("roles", None, _with(st, "clerk", _clerk(
        writes="client_visible")), note="x", author="t"))
    assert "honesty rule" in exc.errors[0] and exc.errors[0].startswith("$.clerk.writes"), exc.errors
    for kind in ("decider", "probe", "librarian", "tool"):
        assert schema.role_rules(_clerk(kind=kind, writes="client_visible")), kind
    ok = _clerk(kind="reasoner", writes="client_visible", window="any", preemptible=False)
    assert st.put("roles", None, _with(st, "writer", ok), note="the reasoner", author="t") == 1


def test_in_decode_is_for_a_decider_or_a_probe():
    for kind in ("clerk", "librarian", "tool", "reasoner"):
        errs = schema.role_rules(_clerk(kind=kind, window="in_decode"), "$.r")
        assert errs and errs[0].startswith("$.r.window"), kind
    for kind in ("decider", "probe"):
        assert schema.role_rules(_clerk(kind=kind, window="in_decode")) == [], kind


# ------------------------------------------------------------------------- the tokenizer hash
def test_the_tokenizer_hash_decides_the_forms_of_sharing():
    # token ids only with the engine's own hash; states only for a probe; text always
    st = _store(engine_tokenizer=lambda: QWEN_SHA)
    same = _clerk(tokenizer_sha256=QWEN_SHA, sharing=["text", "tokens"])
    assert st.put("roles", None, _with(st, "small_qwen", same), note="same tokenizer",
                  author="t") == 1
    other = _clerk(tokenizer_sha256="cd" * 32, sharing=["text", "tokens"])
    exc = _refused(lambda: st.put("roles", None, _with(st, "gemma", other), note="x", author="t"))
    assert "not the engine's tokenizer" in exc.errors[0], exc.errors
    none = _clerk(sharing=["text", "tokens"])
    exc = _refused(lambda: st.put("roles", None, _with(st, "gemma", none), note="x", author="t"))
    assert exc.errors[0].startswith("$.gemma.tokenizer_sha256"), exc.errors
    blind = _store(engine_tokenizer=lambda: None)
    exc = _refused(lambda: blind.put("roles", None, _with(blind, "small_qwen", same), note="x",
                                     author="t"))
    assert "recorded no tokenizer hash" in exc.errors[0], exc.errors
    exc = _refused(lambda: st.put("roles", None, _with(st, "c", _clerk(sharing=["states"])),
                                  note="x", author="t"))
    assert "probe" in exc.errors[0]
    exc = _refused(lambda: st.put("roles", None, _with(st, "c", _clerk(sharing=["text", "text"])),
                                  note="x", author="t"))
    assert "twice" in exc.errors[0]
    assert schema.sharing_forms(same, QWEN_SHA) == ["text", "tokens"]
    assert schema.sharing_forms(other, QWEN_SHA) == ["text"]
    assert schema.sharing_forms(same, None) == ["text"]
    assert schema.sharing_forms(_clerk(kind="probe"), QWEN_SHA) == ["text", "states"]


def test_the_conductor_reports_what_each_role_may_share():
    c = Conductor(tempfile.mkdtemp(prefix="roles-"), journal=False,
                  state={"tokenizer_sha": QWEN_SHA})
    try:
        _, doc = c.store.get("roles")
        doc["small_qwen"] = _clerk(tokenizer_sha256=QWEN_SHA, sharing=["text", "tokens"])
        doc["gemma"] = _clerk(tokenizer_sha256="cd" * 32)
        c.store.put("roles", None, doc, note="two clerks", author="t")
        got = c.roles_contract()
        assert got["engine_tokenizer_sha256"] == QWEN_SHA
        assert got["roles"]["small_qwen"] == {"sharing_allowed": ["text", "tokens"],
                                              "tokenizer_match": True}
        assert got["roles"]["gemma"] == {"sharing_allowed": ["text"], "tokenizer_match": False}
        assert got["roles"]["decider"]["tokenizer_match"] is None
    finally:
        c.close()


# ------------------------------------------------------------------------- 4 the live v1 document
def _box_doc() -> dict:
    with open(FIXTURE) as f:
        return json.load(f)


def test_4_the_box_roles_document_loads_with_defaults_and_journals_them():
    root = tempfile.mkdtemp(prefix="roles-")
    _put_v1(root, _box_doc())
    c = Conductor(root, journal=True)
    try:
        v, doc = c.store.get("roles")
        assert v == 2, "the stored version is the one loaded, not the seed"
        dec = doc["decider"]
        assert dec["memory_claim_gib"] == 5.1 and "memory_gb" not in dec
        assert (dec["kind"], dec["window"], dec["preemptible"], dec["fallback"], dec["writes"],
                dec["tokenizer_sha256"], dec["sharing"], dec["latency_budget_ms"]) == \
            ("decider", "idle", True, "PASS", "nothing", "none", ["text"], 400)
        assert doc["classifier"]["kind"] == "reasoner" and doc["reader"]["kind"] == "clerk"
        assert doc["librarian"]["kind"] == "librarian"
        meta = c.store.meta("roles")
        assert meta["error"] is None and meta["version"] == 2
        assert "decider.window=idle" in meta["defaults"] and \
            "decider.memory_claim_gib=5.1 (memory_gb)" in meta["defaults"]
        # the server start loaded it and journaled which defaults applied: names, never a value
        # the user wrote
        c.journal.flush()
        conn = sqlite3.connect(c.journal.path)
        rows = conn.execute("SELECT task, decision, detail FROM conductor_events WHERE "
                            "kind = 'config'").fetchall()
        conn.close()
        assert len(rows) == 1 and rows[0][:2] == ("roles", "defaults"), rows
        detail = json.loads(rows[0][2])
        assert detail["version"] == 2 and detail["applied"] == meta["defaults"]
        # the planner reads the claim under its new name
        assert residency.claims({"claims": {}}, {"reader": dict(doc["reader"],
                                                               memory_claim_gib=12.0)}) \
            ["reader"]["gib"] == 12.0
        # the file on disk is untouched; saving it makes the defaults explicit
        with open(os.path.join(root, "roles", "0002.json")) as f:
            assert "memory_gb" in json.load(f)["doc"]["decider"]
        assert c.store.put("roles", None, doc, note="the contract fields, explicit",
                           author="t", base=2) == 3
        assert "defaults" not in c.store.meta("roles")
    finally:
        c.close()


def test_4_a_restore_of_a_version_before_the_contract_works():
    # the 0.3.6 rollback note says "restore v1": an old version comes back whole, with defaults
    root = tempfile.mkdtemp(prefix="roles-")
    old = _box_doc()
    _put_v1(root, old, v=1)
    st = Store(root)
    doc = st.get("roles")[1]
    doc["decider"]["mode"] = "off"
    assert st.put("roles", None, doc, note="decider off", author="t") == 2
    assert st.restore("roles", None, 1, author="t") == 3
    v, got = st.get("roles")
    assert v == 3 and got["decider"]["mode"] == "on" and got["decider"]["window"] == "idle"
    assert "memory_gb" not in json.dumps(st.version("roles", None, 1))


def test_4_a_stored_version_that_breaks_a_rule_is_not_loaded():
    root = tempfile.mkdtemp(prefix="roles-")
    st = Store(root)
    doc = st.get("roles")[1]
    st.put("roles", None, doc, note="good", author="t")
    bad = copy.deepcopy(doc)
    bad["reader"]["writes"] = "client_visible"         # edited by hand on the box
    _put_v1(root, bad, v=2)
    v, _ = st.get("roles")
    assert v == 1 and "honesty rule" in st.meta("roles")["error"]["message"]


# ------------------------------------------------------------------------- the registry
def test_the_registry_checks_its_models_against_the_same_contract():
    from engine.decision.fake import FakeModel
    from engine.decision.registry import Registry
    import threading

    def contract(fields, name):
        return schema.role_errors(fields, name, QWEN_SHA)

    reg = Registry([FakeModel("fake-decider")], gate=threading.Lock(), contract=contract)
    assert reg.status()[0]["role"]["kind"] == "decider"
    assert reg.status()[0]["role"]["memory_claim_gib"] == 0.0

    class NoClaim(FakeModel):
        memory_claim_gib = None

    class Talks(FakeModel):
        writes = "client_visible"

    for cls, word in ((NoClaim, "memory_claim_gib"), (Talks, "honesty rule")):
        try:
            Registry([cls("fake-x")], gate=threading.Lock(), contract=contract)
        except ValueError as exc:
            assert word in str(exc), str(exc)
        else:
            raise AssertionError(f"{cls.__name__} was served")
    from engine.decision.base import DecisionModel
    assert schema.role_errors(DecisionModel().role()), "the base class declares no claim"


def test_the_meta_carries_the_contract_values():
    m = schema.meta()["roles"]
    assert m["kinds"] == list(schema.ROLE_KINDS) and m["windows"] == [
        "idle", "in_decode", "any", "between_rounds"]
    assert m["fallbacks"] == ["PASS", "stale", "skip", "next-role"]
    assert m["writes"] == ["client_visible", "hidden_notes", "memory_files", "nothing",
                           "state_lines"]
    assert m["sharing"] == ["text", "tokens", "states"] and m["defaults"]["window"] == "idle"


# ------------------------------------------------------------------------- HTTP on the fake engine
def test_1_2_4_the_roles_api_on_the_fake_engine():
    from tests.test_conductor_e2e import _call, _check, _port, _server, _wait
    root = tempfile.mkdtemp(prefix="qse-roles-")
    cdir = os.path.join(root, "conductor")
    _put_v1(cdir, _box_doc())
    port = _port()
    proc, ledger = _server(port, cdir)
    try:
        _wait(port, proc)
        st, _, got = _call(port, "GET", "/v1/conductor/roles")
        assert st == 200, got
        _check(got, "roles")
        assert got["version"] == 2 and got["doc"]["decider"]["window"] == "idle"
        assert "decider.window=idle" in got["meta"]["defaults"]
        assert got["contract"]["roles"]["decider"]["sharing_allowed"] == ["text"]
        st, _, d = _call(port, "GET", "/v1/conductor/roles/versions/2")
        assert d["doc"]["decider"]["memory_claim_gib"] == 5.1
        # 2: a role without a window -- 422 naming the field, the version stays
        doc = got["doc"]
        doc["clerk"] = _clerk()
        del doc["clerk"]["window"]
        st, _, err = _call(port, "PUT", "/v1/conductor/roles",
                           {"doc": doc, "note": "no window", "base_version": 2})
        assert st == 422 and "$.clerk: missing 'window'" in err["error"]["errors"], err
        _check(err, "error")
        assert _call(port, "GET", "/v1/conductor/roles")[2]["version"] == 2
        # 1: the complete role is a new version and reads back whole
        doc["clerk"] = _clerk()
        st, _, w = _call(port, "PUT", "/v1/conductor/roles",
                         {"doc": doc, "note": "a clerk in shadow", "base_version": 2})
        assert st == 200 and w["version"] == 3, w
        got = _call(port, "GET", "/v1/conductor/roles")[2]
        _check(got, "roles")
        assert got["doc"]["clerk"] == _clerk() and "defaults" not in got["meta"]
        meta = _call(port, "GET", "/v1/conductor/meta")[2]
        _check(meta, "meta")
        assert meta["roles"]["windows"] == ["idle", "in_decode", "any", "between_rounds"]
        # the start journaled the defaults into the ledger's journal
        proc.send_signal(2)
        proc.wait(30)
        conn = sqlite3.connect(ledger)
        n = conn.execute("SELECT count(*) FROM conductor_events WHERE kind = 'config' AND "
                         "task = 'roles' AND decision = 'defaults'").fetchone()[0]
        conn.close()
        assert n >= 1
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
