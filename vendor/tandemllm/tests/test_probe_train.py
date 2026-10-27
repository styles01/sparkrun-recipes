"""tools/probe_train.py: the join, the fit, the source-confound report, the probe file (ENG-181).

The test matrix (a ring written by server/probe.py and a journal built with the ledger's DDL, both
in a temp directory, synthetic rows of width 16):

    1  the join: class from Qwen's accepted no-think or in-context answer and the fixture's labels
       by request id (a decider's or a below-threshold answer is not a label); stuck from the D1
       guard's act within the horizon, negatives from turns it never acted in; done from the next
       checker RESULT                              test_the_join_*
    2  a probe on rows that carry the class beside a source offset: held-out accuracy and the
       deconfounded accuracy above the gate, the source told apart, not confounded, pass
                                                    test_a_real_class_signal_passes
    3  rows that carry only the source, each source mostly one class: the class accuracy is the
       source's, flagged confounded, fail            test_a_source_only_signal_is_flagged
    4  stuck: precision at the guard's act rate; under 30 positives, insufficient
                                                    test_stuck_*
    5  the probe file: the server loads it and answers as the fit does; held out by session
                                                    test_the_probe_file_*, test_held_out_by_session
    6  growth: a night's bytes from the ledger's requests, raw ones never counted
                                                    test_growth_from_the_ledger

Run: python tests/test_probe_train.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from server import probe as probe_mod  # noqa: E402
from server.ledger import CONDUCTOR_DDL  # noqa: E402
from tools import probe_train as pt  # noqa: E402

H = 16
CLASSES = ["code_edit", "debug", "plan"]


class Src:
    def __init__(self, v: torch.Tensor):
        self.hidden_post_norm = v.view(1, 1, H)

    def tapped(self):
        return None


def _write(ring, vecs, metas, point="prompt"):
    for v, m in zip(vecs, metas):
        cap = ring.capture(0)
        s = Src(torch.as_tensor(v, dtype=torch.float32))
        cap._take(point, s, s, slice(0, 1), at=m.get("at", 0))
        ring.submit(cap, m, tool_calls=0)


def _journal(path, routes=(), monitors=()):
    conn = sqlite3.connect(path)
    conn.executescript(CONDUCTOR_DDL)
    for r in routes:
        conn.execute("INSERT INTO conductor_events (id, ts_ms, request_id, kind, stages, "
                     "next_checker, source) VALUES (?, ?, ?, 'route', ?, ?, ?)",
                     (r["id"], 1, r.get("request_id"), json.dumps(r.get("stages") or []),
                      json.dumps(r["next_checker"]) if r.get("next_checker") else None,
                      r.get("source")))
    for i, m in enumerate(monitors):
        conn.execute("INSERT INTO conductor_events (id, ts_ms, request_id, kind, task, mode, "
                     "action, tokens_at_cut) VALUES (?, 1, ?, 'monitor', ?, ?, ?, ?)",
                     (f"m{i}", m["request_id"], m["task"], m.get("mode", "act"),
                      m.get("action", "CLOSE_REASONING"), m.get("at")))
    conn.commit()
    conn.close()


def _stage(kind, label, conf=0.9, below=False, error=None):
    d = {"stage": kind, "label": label, "confidence": conf, "error": error}
    if below:
        d["below"] = True
    return d


def test_the_join_takes_only_qwens_answers_and_the_fixture():
    d = tempfile.mkdtemp(prefix="pt-join-")
    ring = probe_mod.Ring(d, hidden=H, weights="w1", slots=8)
    metas = [{"request_id": f"r{i}", "row_id": f"row{i}", "session": f"s{i}", "ts": time.time()}
             for i in range(6)]
    _write(ring, np.eye(6, H), metas)
    _journal(os.path.join(d, "j.sqlite"), routes=[
        {"id": "row0", "stages": [_stage("decider", "debug", 0.95)]},              # Laya: no
        {"id": "row1", "stages": [_stage("decider", "x", 0.2, below=True),
                                  _stage("qwen", "plan", 0.8)]},                    # qwen
        {"id": "row2", "stages": [_stage("qwen_context", "debug", 0.99)]},          # in context
        {"id": "row3", "stages": [_stage("qwen_context", "plan", 0.3, below=True)]},  # below
        {"id": "row4", "stages": [_stage("qwen", None, error="qwen_timeout")]}])    # failed
    conn = pt.open_journal(os.path.join(d, "j.sqlite"))
    lab = pt.labelled("class", pt.read_index(d), conn, {"r5": "code_edit"}, 1024)
    got = {r["request_id"]: (r["y"], r["source"]) for r in lab}
    assert got == {"r1": ("plan", "qwen"), "r2": ("debug", "qwen_context"),
                   "r5": ("code_edit", "fixture")}, got


def test_the_join_of_stuck_and_done():
    d = tempfile.mkdtemp(prefix="pt-join2-")
    ring = probe_mod.Ring(d, hidden=H, weights="w1", slots=8)
    metas = [{"request_id": "a", "row_id": "ra", "at": 2048, "finish": "stop"},   # act at 2,500
             {"request_id": "b", "row_id": "rb", "at": 1024, "finish": "stop"},   # act at 9,000
             {"request_id": "c", "row_id": "rc", "at": 1024, "finish": "tool_calls"},
             {"request_id": "e", "row_id": "re", "at": 1024, "finish": "length"}]
    _write(ring, np.eye(4, H), metas, point="think")
    _write(ring, np.eye(3, H), [{"request_id": x, "row_id": "r" + x} for x in "fgh"],
           point="finish")
    _journal(os.path.join(d, "j.sqlite"), routes=[
        {"id": "rf", "next_checker": [{"checker": "calccheck", "verdict": "PASS"}]},
        {"id": "rg", "next_checker": [{"verdict": "PASS"}, {"verdict": "FAIL"}]},
        {"id": "rh"}],
        monitors=[{"request_id": "a", "task": "D1", "at": 2500},
                  {"request_id": "b", "task": "D1", "at": 9000},
                  {"request_id": "c", "task": "D1", "mode": "shadow", "at": 1500}])
    conn = pt.open_journal(os.path.join(d, "j.sqlite"))
    recs = pt.read_index(d)
    stuck = {r["request_id"]: r["y"] for r in pt.labelled("stuck", recs, conn, {}, 1024)}
    # a: the guard closed 452 tokens later; b: it acted, but 7,976 later (no label); c: only a
    # shadow decision, the turn ended in a call (a negative); e: cut by the length (no label)
    assert stuck == {"a": 1, "c": 0}, stuck
    done = {r["request_id"]: r["y"] for r in pt.labelled("done", recs, conn, {}, 1024)}
    assert done == {"f": 1, "g": 0}, done


def _class_data(n_per=60, sources=("fixture", "qwen", "qwen_context"), signal=3.0, skew=None,
                seed=0):
    """Rows = a class direction * signal + a source direction * 3 + noise. `skew`: the share of
    each source that is its own class (the class then follows the source)."""
    rng = np.random.default_rng(seed)
    cdir = rng.standard_normal((len(CLASSES), H))
    sdir = rng.standard_normal((len(sources), H))
    X, y, src, ses = [], [], [], []
    for si, s in enumerate(sources):
        for i in range(n_per):
            if skew is None:
                c = i % len(CLASSES)
            else:
                c = si % len(CLASSES) if rng.random() < skew else int(rng.integers(len(CLASSES)))
            X.append(cdir[c] * signal + sdir[si] * 3 + rng.standard_normal(H))
            y.append(CLASSES[c])
            src.append(s)
            ses.append(f"{s}-{i // 2}")
    return np.array(X, np.float32), y, src, ses


def _recs(d, X, y, src, ses):
    ring = probe_mod.Ring(d, hidden=H, weights="w1", slots=8)
    metas = [{"request_id": f"r{i}", "row_id": f"row{i}", "session": s, "ts": time.time()}
             for i, s in enumerate(ses)]
    _write(ring, X, metas)
    recs = pt.read_index(d)
    return [dict(r, y=y[i], source=src[i]) for i, r in enumerate(recs)]


def test_a_real_class_signal_passes():
    X, y, src, ses = _class_data()
    recs = _recs(tempfile.mkdtemp(prefix="pt-real-"), X, y, src, ses)
    rep, m = pt.fit_question("class", recs, ["post"])
    c = rep["confound"]
    assert rep["split"] == "by_session" and rep["accuracy"] >= 0.9, rep
    assert c["deconfounded"] >= 0.9 and c["source_accuracy"] >= 0.9, c
    assert c["source_only"] < 0.5 and not rep["confounded"] and rep["verdict"] == "pass", rep
    assert set(c["leave_one_source"]) == {"fixture", "qwen", "qwen_context"}
    return (f"accuracy {rep['accuracy']}, deconfounded {c['deconfounded']}, source "
            f"{c['source_accuracy']}, source only {c['source_only']}")


def test_a_source_only_signal_is_flagged():
    X, y, src, ses = _class_data(signal=0.0, skew=0.9)
    recs = _recs(tempfile.mkdtemp(prefix="pt-conf-"), X, y, src, ses)
    rep, _ = pt.fit_question("class", recs, ["post"])
    c = rep["confound"]
    assert c["source_accuracy"] >= 0.9
    assert rep["confounded"] and rep["verdict"] == "fail", rep
    assert c["deconfounded"] < 0.6, c
    return f"accuracy {rep['accuracy']} = source only {c['source_only']}, deconfounded " \
        f"{c['deconfounded']}"


def _stuck_recs(d, n_pos, n_neg, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(H)
    X = [w * 4 + rng.standard_normal(H) for _ in range(n_pos)] + \
        [rng.standard_normal(H) for _ in range(n_neg)]
    y = [1] * n_pos + [0] * n_neg
    return _recs(d, np.array(X, np.float32), y, ["coding"] * len(y),
                 [f"s{i}" for i in range(len(y))])


def test_stuck_precision_at_the_guards_rate():
    recs = _stuck_recs(tempfile.mkdtemp(prefix="pt-stuck-"), 120, 600)
    rep, _ = pt.fit_question("stuck", recs, ["post"])
    assert rep["positives"] == 120 and 0.1 < rep["act_rate"] < 0.25
    assert rep["precision"] >= 0.9 and rep["verdict"] == "pass", rep
    small = _stuck_recs(tempfile.mkdtemp(prefix="pt-stuck2-"), 12, 200)
    rep2, _ = pt.fit_question("stuck", small, ["post"])
    assert rep2["verdict"] == "insufficient", rep2
    return f"precision {rep['precision']} at act rate {rep['act_rate']}"


def test_held_out_by_session():
    ses = [f"s{i // 3}" for i in range(300)]
    t = pt.held_out(ses, 0.25)
    by = {}
    for s, v in zip(ses, t):
        by.setdefault(s, set()).add(bool(v))
    assert all(len(v) == 1 for v in by.values()), "a session on both sides"
    assert 0.15 < t.mean() < 0.35


def test_the_probe_file_answers_as_the_fit():
    d = tempfile.mkdtemp(prefix="pt-file-")
    X, y, src, ses = _class_data()
    recs = _recs(d, X, y, src, ses)
    rep, m = pt.fit_question("class", recs, ["post"])
    path = os.path.join(d, "probes", "class.pt")
    pt.save_probe(path, m, rep, recs, ["post"])
    assert os.stat(path).st_mode & 0o777 == 0o600
    stage = probe_mod.ProbeStage(d, weights="w1", hidden=H)
    probe, why = stage.get({"file": "class.pt"})
    assert why is None and probe.classes == sorted(CLASSES)
    Xr = pt.vectors(recs[:40], ["post"])
    want = m.proba(Xr)
    for i in range(40):
        label, conf, probs, ms = probe.answer(torch.from_numpy(Xr[i]).view(1, H), ["post"])
        assert label == m.classes[int(want[i].argmax())]
        assert abs(conf - float(want[i].max())) < 1e-4, (conf, want[i].max())
    assert probe_mod.ProbeStage(d, weights="w2", hidden=H).get({"file": "class.pt"}) == \
        (None, "weights")
    assert probe_mod.ProbeStage(d, weights="w1", hidden=H * 2).get({"file": "class.pt"}) == \
        (None, "dim")


def test_growth_from_the_ledger():
    d = tempfile.mkdtemp(prefix="pt-growth-")
    path = os.path.join(d, "ledger.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE requests (ts_ms INTEGER, conductor_mode TEXT, client_kind TEXT, "
                 "reasoning_tokens INTEGER, tool_calls INTEGER)")
    t = int(time.mktime(time.strptime("2026-10-01 22:00", "%Y-%m-%d %H:%M")) * 1000)
    rows = [(t, "on", "opencode", 3000, 1), (t, "on", "opencode", 500, 0),
            (t, "raw", "opencode", 9000, 1), (t, "on", "open-webui", 4000, 0)]
    conn.executemany("INSERT INTO requests VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pt.main(["growth", "--ledger", path, "--since", "2026-10-01 21:00",
                 "--until", "2026-10-02 09:00"])
    got = json.loads(out.getvalue())
    assert (got["tandem_tool_requests"], got["think_points"], got["finish_points"],
            got["raw_tool_requests_never_written"]) == (2, 2, 1, 1), got
    assert got["bytes"] == 5 * (6 * 5120 * 2 + probe_mod.INDEX_LINE_BYTES) and got["pass"]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:62s} ok" + (f"  ({got})" if isinstance(got, str) else ""))
            passed += 1
    print(f"{passed} passed")
