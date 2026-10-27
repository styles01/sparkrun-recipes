"""The disk tier in the server (C1): its settings document, the tag a request hands the prefill, the
Memory screen's routes, /v1/cache/clear?tier=disk, the planner's declaration and the metrics.

The tier's own behaviour -- what it writes, the restore's bit equality, crash safety, eviction --
is tests/test_disk_cache.py; this file holds the server to the contract around it:

  * `settings.memory.disk_cache` validates (mode, budget, TTL, admission, a floor of at least
    50 GB), its seed is the engine's defaults, and a new version applies at the next request;
  * the request's tag: the client, the notes the router spliced in (their served positions and
    package names), where the last user message starts;
  * GET /v1/conductor/memory/disk answers the contract whether the tier runs or not, with a
    preview only for opencode heads and packages; POST .../clear empties it;
  * the residency planner counts the staging buffer until it exists;
  * the start refuses a directory inside a git work tree, the fake engine, a server without the
    prefix cache;
  * qse_disk_cache_* on /metrics.

Run: python tests/test_disk_cache_server.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from engine import cache, disk_cache as dc  # noqa: E402
from server import diskcache, memlimits, metrics as M  # noqa: E402
from server.conductor import schema  # noqa: E402
from tools import contract_check  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_disk_cache import CH, HUGE, Pos  # noqa: E402

CONTRACT = os.path.join(ROOT, "docs", "contract", "conductor-v1")


def _check(obj, name):
    errs = contract_check.validate(obj, contract_check.load(name, CONTRACT), root=CONTRACT)
    assert not errs, (name, errs[:5])


class Store:
    def __init__(self, v, doc):
        self.v, self.doc = v, doc

    def peek(self, kind):
        assert kind == "settings"
        return self.v, self.doc


class Cond:
    def __init__(self, state, v=1, doc=None):
        self.state = state
        self.store = Store(v, doc or {})


def _tier(d, **kw):
    kw.setdefault("settings", {"admit_after": 1})
    return dc.DiskTier(d, chunk=CH, fingerprint="t", min_tokens=CH, head_anchors=2,
                       free_bytes=lambda p: HUGE, **kw).open()


def _doc(**disk):
    d = schema.seed("settings")
    d["memory"]["disk_cache"].update(disk)
    return d


# ------------------------------------------------------------------ the settings document

def test_the_settings_document_takes_the_disk_tier_and_holds_its_floor():
    assert schema.SETTINGS["memory"]["disk_cache"] == dc.SETTINGS     # one set of defaults
    assert schema.check("settings", _doc()) == []
    assert schema.check("settings", _doc(budget_gb=120.5, ttl_days=30, admit_after=1)) == []
    assert any("min_free_gb" in e for e in schema.check("settings", _doc(min_free_gb=10)))
    assert any("mode" in e for e in schema.check("settings", _doc(mode="sometimes")))
    assert any("ttl_days" in e for e in schema.check("settings", _doc(ttl_days=0)))
    old = schema.seed("settings")
    del old["memory"]["disk_cache"]
    assert schema.check("settings", old) == []                # a version stored before C1
    assert diskcache.settings_of(old) == dc.SETTINGS
    assert diskcache.settings_of(_doc(budget_gb=20))["budget_gb"] == 20


def test_a_new_settings_version_applies_at_the_next_request():
    d = tempfile.mkdtemp(prefix="qse-disk-")
    try:
        t = _tier(d, settings={})
        state = {"disk_cache": t}
        state["conductor"] = Cond(state, 3, _doc(budget_gb=20, ttl_days=2))
        assert sorted(diskcache.refresh(state)) == ["budget_gb=20", "ttl_days=2"]
        assert t.budget == 20 * dc.GB and t.ttl == 2 * 86400
        assert diskcache.refresh(state) == []                 # the same version: nothing
        state["conductor"].store = Store(4, _doc(mode="off"))
        changed = diskcache.refresh(state)
        assert "mode=off" in changed and not t.on and "budget_gb=16" in changed
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ the request's tag

class Tok:
    def convert_tokens_to_ids(self, text):
        return {"<|im_start|>": 7}[text]


class PlanStub:
    spans = [(10, 30, ["packages"], ["ui-design"])]


def test_the_tag_names_the_client_the_notes_and_where_the_user_message_starts():
    ids = [7, 1, 2, 3, 7, 4, 5, 9, 7, 6]       # system, user, the answer's opening
    tag = diskcache.request_tag(PlanStub(), "opencode", ids, Tok())
    assert tag == {"client": "opencode", "spans": [(10, 30, ["packages"], ["ui-design"])],
                   "head_end": 4, "first_answer": 8, "side": False}
    assert diskcache.request_tag(None, None, [7, 1, 2], Tok())["head_end"] is None
    # a later turn: the user message is the last one, the first answer stays where it started
    later = diskcache.request_tag(None, "opencode", ids + [1, 7, 2, 7, 3], Tok())
    assert (later["head_end"], later["first_answer"]) == (11, 8)
    assert diskcache.request_tag(None, None, [7, 1, 7, 2], Tok())["first_answer"] is None


class SidePlan:
    spans = []
    side = {"shape": "opencode_title", "kind": "title", "max_tokens": 64}


def test_a_side_request_is_read_and_admits_nothing():
    """S1 x C1: a client's title or summary call writes nothing to the disk and is not counted
    towards admission; the same prompt as a turn is admitted (admit_after 1)."""
    ids = [7, 1, 2, 3, 7, 4, 5, 9, 7, 6]
    assert diskcache.request_tag(SidePlan(), "opencode", ids, Tok())["side"] is True
    d = tempfile.mkdtemp(prefix="qse-disk-")
    try:
        t = _tier(d)
        prompt = tokens(100, seed=3)
        side = {"client": "opencode", "head_end": 80, "side": True}
        cache.prefill(fresh(), Pos(), prompt, "cpu", chunk=CH, disk=t, conv_id="ses_s",
                      disk_tag=side)
        assert t.flush()
        assert not t.anchors and not t.chunks and not t.counts and not t.sessions
        cache.prefill(fresh(), Pos(), prompt, "cpu", chunk=CH, disk=t, conv_id="ses_s",
                      disk_tag=dict(side, side=False))
        assert t.flush()
        assert t.anchors and t.chunks
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_router_records_where_its_notes_sit_in_the_served_prompt():
    from test_conductor_router import SYSTEM, QTok, body, by_label, setup, turn
    cond, _ = setup(decider=by_label("design_ui", 0.84))
    plan, ids, clean = turn(cond, body([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": "Make the settings page "
                                                                    "look calmer."}]))
    (start, end, kinds, names), = plan.spans
    assert "packages" in kinds and names == ["ui-design"]
    assert ids[start:end] == QTok().encode(plan.injected[0]["text"])
    assert len(ids) - len(clean) == end - start


# ------------------------------------------------------------------ the Memory screen

def test_the_disk_route_answers_the_contract_running_or_not():
    state = {"args": {"disk_cache": "off"}}
    state["conductor"] = Cond(state, None, {})
    v = diskcache.view(state)
    assert v["running"] is False and v["settings"] == dc.SETTINGS and v["floor_gb"] == 50
    _check({**v, "contract_version": "1.0"}, "memory-disk")
    d = tempfile.mkdtemp(prefix="qse-disk-")
    try:
        t = _tier(d, preview=lambda keys, L, kind, spans: "You are opencode, an interactive CLI")
        tag = {"client": "opencode", "head_end": 80}
        cache.prefill(fresh(), Pos(), tokens(100, seed=1), "cpu", chunk=CH, disk=t,
                      conv_id="ses_1", disk_tag=tag)
        cache.prefill(fresh(), Pos(), tokens(100, seed=2), "cpu", chunk=CH, disk=t)
        assert t.flush()
        cache.prefill(fresh(), Pos(), tokens(100, seed=2) + [3], "cpu", chunk=CH, disk=t)
        state = {"disk_cache": t, "args": {"disk_cache": d}}
        state["conductor"] = Cond(state, 2, _doc())
        v = diskcache.view(state)
        _check({**v, "contract_version": "1.0"}, "memory-disk")
        rows = {r["kind"]: r for r in v["top"]}
        assert rows["opencode-head"]["preview"].startswith("You are opencode")
        assert rows["prefix"]["preview"] is None and rows["prefix"]["hits"] == 1
        # three prefills the disk could have shortened, the third one a hit
        assert v["running"] and (v["lookups"], v["hits"], v["hit_rate"]) == (3, 1, 0.3333)
        from server.conductor import routes
        st, got, _ = routes._memory(state["conductor"], "POST", ["disk", "clear"], {}, {})
        # the session's head (lines 64 and 80, under its user message, and its longest line 96
        # admitted after one prefill) and the other prompt's line 96
        assert st == 200 and got["cleared"] and got["entries"] == 4, got["entries"]
        _check({**got, "contract_version": "1.0", "disk": {**got["disk"],
                                                           "contract_version": "1.0"}},
               "memory-disk-clear")
        assert got["disk"]["entries"] == 0
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_clear_without_a_tier_is_a_conflict():
    from server.conductor import routes
    state = {"args": {}}
    try:
        routes._memory(Cond(state), "POST", ["disk", "clear"], {}, {})
        raise AssertionError("no 409")
    except routes.ApiError as exc:
        assert exc.status == 409


# ------------------------------------------------------------------ the start

def _args(**kw):
    a = argparse.Namespace(disk_cache="off", fake_engine=False, disk_cache_head_anchors=2,
                           disk_cache_min_tokens=CH, prefix_chunk=CH, drafter="test",
                           narrow_arm="on", nvfp4=None, fp8_head=None, model=None,
                           dflash2_ckpt=None, dflash2_ckpt16=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_the_start_refuses_what_it_must_and_opens_otherwise():
    d = tempfile.mkdtemp(prefix="qse-disk-")
    try:
        eng, dr = fresh(), Pos()
        state = {"engine": eng, "drafter": dr, "prefix_cache": True, "prefix_chunk": CH,
                 "tok": None}
        assert diskcache.start(state, _args()) is None                          # off
        assert diskcache.start(state, _args(disk_cache=d, fake_engine=True)) is None
        assert diskcache.start(dict(state, prefix_cache=False), _args(disk_cache=d)) is None
        os.makedirs(os.path.join(d, "repo", ".git"))
        inside = os.path.join(d, "repo", "cache")
        assert diskcache.start(state, _args(disk_cache=inside)) is None
        assert not os.path.exists(inside)                    # refused, never created
        events = []

        class J:
            def note(self, kind, task, decision=None, detail=None):
                events.append((kind, task, decision))
        t = diskcache.start(state, _args(disk_cache=os.path.join(d, "disk")), J())
        assert t is not None and t.on and t.staging_planned > 0 and t.staging_bytes == 0
        assert oct(os.stat(os.path.join(d, "disk")).st_mode & 0o777) == "0o700"
        cache.prefill(fresh(), dr, tokens(80, seed=3), "cpu", chunk=CH, disk=t)
        cache.prefill(fresh(), dr, tokens(80, seed=3), "cpu", chunk=CH, disk=t)
        assert t.flush()
        assert ("memory", "disk_cache", "write") in events
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_fingerprint_leaves_out_what_cannot_change_a_row():
    os.environ["QWEN38_PROBE_SNAPSHOTS"] = "on"
    os.environ["QWEN38_FUSED_GDNPREFILL"] = "1"
    try:
        x = diskcache.fingerprint_extra(_args())
        assert "QWEN38_PROBE_SNAPSHOTS" not in x["env"] and "QWEN38_FUSED_GDNPREFILL" in x["env"]
        assert x["chunk"] == CH and x["torch"]
        a = dc.fingerprint(ROOT, x)
        os.environ["QWEN38_FUSED_GDNPREFILL"] = "0"
        assert dc.fingerprint(ROOT, diskcache.fingerprint_extra(_args())) != a
        os.environ["QWEN38_PROBE_SNAPSHOTS"] = "off"
    finally:
        os.environ.pop("QWEN38_PROBE_SNAPSHOTS", None)
        os.environ["QWEN38_FUSED_GDNPREFILL"] = "0"


# ------------------------------------------------------------------ memory and metrics

def test_the_planner_counts_the_staging_buffer_until_it_exists():
    d = tempfile.mkdtemp(prefix="qse-disk-")
    try:
        gib = 1 << 30
        mem = cache.Budget(70 * gib, 77 * gib, 2 * gib, probe=lambda: (60 * gib, 61 * gib),
                           empty=lambda: None, peak_probe=lambda reset=False: 0)
        t = _tier(d)
        t.staging_planned = gib // 2
        state = {"mem": mem, "disk_cache": t}
        assert memlimits.engine_gib(state) == (77.5, 61.0)
        cache.prefill(fresh(), Pos(), tokens(80, seed=4), "cpu", chunk=CH, disk=t)
        assert t.flush() and t.staging_bytes > 0
        assert memlimits.engine_gib(state) == (77.0, 61.0)    # allocated: MemAvailable shows it
        assert memlimits.engine_gib({"mem": mem}) == (77.0, 61.0)
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_metrics_are_on_the_scrape():
    d = tempfile.mkdtemp(prefix="qse-disk-")
    saved = M._SOURCES.get("cache_stats")
    try:
        t = _tier(d)
        ids = tokens(80, seed=5)
        cache.prefill(fresh(), Pos(), ids, "cpu", chunk=CH, disk=t)
        assert t.flush()
        cache.prefill(fresh(), Pos(), ids + [1], "cpu", chunk=CH, disk=t)   # a hit at 64, and
        assert t.flush()                                     # line 80 written (admit_after 1)
        M._SOURCES["cache_stats"] = lambda: {"disk_cache": t.report(top=10)}
        text = M.render()
        assert "qse_disk_cache_hits_total 1" in text and "qse_disk_cache_lookups_total 2" in text
        assert 'qse_disk_cache_entries{kind="anchors"} 2' in text
        assert 'qse_disk_cache_writes_total{kind="chunk"} 5' in text
        assert "qse_disk_cache_tokens_restored_total 64" in text
        assert 'qse_disk_cache_bytes{part="floor"} 50000000000' in text
        M._SOURCES["cache_stats"] = lambda: {"disk_cache": None}
        assert "qse_disk_cache_hits_total 1" not in M.render()
        t.close()
    finally:
        M._SOURCES["cache_stats"] = saved
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ the served loop

def _serve_loop(t):
    """The server's state around `generate_stream`, on the random model: the prefix cache's grid,
    the resident prefix, the disk tier `t` (or None)."""
    import torch
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok
    app.STATE.clear()
    app.STATE.update(engine=fresh(), drafter=None, k=4, tree=False, sampled_tree=False,
                     relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=True,
                     prefix_chunk=CH, session_cache=False, tok=FakeTok(), device="cpu",
                     max_len=512, default_max_tokens=32, pen_spec=PenaltySpec(), model="t",
                     cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                     response_cache=None, suffix_store=None,
                     resident=cache.ResidentPrefix(1 << 30, CH, stash_bytes=1 << 30),
                     disk_cache=t)

    def gen(ids, session=None, tag=None):
        out = list(app.generate_stream(torch.tensor(ids), 24, set(), None, session,
                                       disk_tag=tag))
        return out, dict(app.STATE["last_prefill"])
    return gen


def test_the_served_loop_writes_the_head_and_a_restart_reads_it_with_the_same_text():
    """The first long request of an opencode session is written; after a restart (a new engine,
    a reopened tier) the next session's first request starts from the disk and writes the same
    greedy tokens as a server without the tier."""
    head = tokens(120, seed=40)
    s1, s2 = head + tokens(25, seed=41), head + tokens(30, seed=42)
    tag = {"client": "opencode", "head_end": 120}
    d = tempfile.mkdtemp(prefix="qse-disk-loop-")
    try:
        t = _tier(d, settings={})                              # admit_after 2: the head only
        gen = _serve_loop(t)
        gen(s1, "ses_1", tag)
        assert t.flush() and t.report()["entries"] == 2      # lines 96 and 112, under 120
        t.close()
        t2 = _tier(d, settings={})
        gen = _serve_loop(t2)
        out_disk, pre = gen(s2, "ses_2", tag)
        assert pre["kind"] == "disk" and pre["reused"] == 112, pre
        out_cold, pre_cold = _serve_loop(None)(s2, "ses_2", tag)
        assert pre_cold["reused"] == 0 and out_disk == out_cold and len(out_cold) == 24
        t2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ over HTTP (the fake engine)

def test_the_routes_on_the_fake_engine():
    from test_conductor_e2e import PROXY, _call, _port, _server, _wait
    root = tempfile.mkdtemp(prefix="qse-disk-http-")
    port = _port()
    proc, _ = _server(port, os.path.join(root, "conductor"),
                      extra=("--disk-cache", os.path.join(root, "disk")))
    try:
        _wait(port, proc)
        st, _, got = _call(port, "GET", "/v1/conductor/memory/disk")
        assert st == 200 and got["running"] is False, got        # the fake engine runs none
        _check(got, "memory-disk")
        assert got["settings"]["budget_gb"] == 16 and got["flag"] == os.path.join(root, "disk")
        assert _call(port, "POST", "/v1/conductor/memory/disk/clear", {})[0] == 409
        assert _call(port, "GET", "/v1/conductor/memory/disk", headers=PROXY, token=None)[0] == 401
        st, _, stats = _call(port, "POST", "/v1/cache/clear?tier=disk", {})
        assert st == 200 and stats["disk_cache"] is None
        assert _call(port, "POST", "/v1/cache/clear?tier=attic", {})[0] == 400
        # a settings version with the section: stored, then refused below the floor
        st, _, cur = _call(port, "GET", "/v1/conductor/settings")
        doc = cur["doc"]
        doc["memory"]["disk_cache"] = dict(dc.SETTINGS, budget_gb=80)
        st, _, put = _call(port, "PUT", "/v1/conductor/settings",
                           {"doc": doc, "note": "disk tier 80 GB", "base_version": cur["version"]})
        assert st == 200, put
        doc["memory"]["disk_cache"]["min_free_gb"] = 20
        st, _, err = _call(port, "PUT", "/v1/conductor/settings",
                           {"doc": doc, "note": "below the floor", "base_version": put["version"]})
        assert st in (400, 422) and "min_free_gb" in str(err), (st, err)
        st, _, got = _call(port, "GET", "/v1/conductor/memory/disk")
        assert got["settings"]["budget_gb"] == 80 and got["settings_version"] == put["version"]
    finally:
        proc.kill()
        shutil.rmtree(root, ignore_errors=True)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
