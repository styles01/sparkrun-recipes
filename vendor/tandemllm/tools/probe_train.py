"""The first probes on Qwen's own rows (ENG-181): join the snapshots to labels, fit, report.

    # fit and report one question on the box CPU (nothing on the GPU, no server touched)
    nice -n 19 python tools/probe_train.py fit --question class \\
        [--dir ~/.qwen38-spark-engine/probe] [--journal ~/.qwen38-spark-engine/usage/ledger.sqlite3] \\
        [--fixture-labels ~/.qwen38-spark-engine/probe/fixture/labels.jsonl] [--rows all|taps|post] \\
        [--out ~/.qwen38-spark-engine/probe/probes/class.pt] [--report out.json]

    # the 216-prompt fixture through a TandemLLM server with the snapshots on (needs the GPU)
    python tools/probe_train.py fixture --server http://127.0.0.1:8000 \\
        --out ~/.qwen38-spark-engine/probe/fixture/labels.jsonl [--live]

    # what a night's traffic writes, from the usage ledger (read-only)
    python tools/probe_train.py growth --since "2026-10-01 21:00" --until "2026-10-02 09:00"

THE JOIN. Each snapshot's index line (server/probe.py) carries the request id and the router's
journal row id; the labels arrive later in the journal (the usage ledger's `conductor_events`,
opened read-only):

  class   `prompt` points. The class Qwen decided: the route row's accepted stage, `qwen` (the
          no-think digest answer) or `qwen_context` (the in-context answer); a row decided by
          anyone else (Laya, stickiness, a fallback) is not a label. Plus the 216-prompt fixture
          run through the engine (`fixture`), labelled from the fixture file by request id.
  stuck   `think` points. Positive: the D1 loop guard closed the reasoning later in the same turn,
          within `--horizon` (1,024) reasoning tokens of the point. Negative: a turn the guard never
          acted in that ended normally (an answer or a call). A point the guard acted on further
          away is not a label.
  done    `finish` points (done claims: candidates of a turn that did not end in a call). The
          next request's checker RESULT line on the route row (`next_checker`): any FAIL is not
          done, all PASS is done; no result, no label.

THE FIT. A linear probe per question: multinomial logistic regression on the standardised rows,
L2 (`--l2`), full-batch accelerated gradient descent, no library beyond numpy. Held out by
session (a stable hash of the session key, `--test` 0.25), so no conversation is on both sides;
the temperature is fitted on the held-out set (the probe's confidence, not its accuracy).

THE CONFOUND (arXiv 2606.02907: a probe at 100 % on "reasoning mode" fell to chance once the data
source and format were regressed out). For the class question every report carries, next to the
class accuracy:

  source_accuracy     a probe on the same rows predicting the source (fixture, qwen,
                      qwen_context), against the majority baseline. Near 1.0 means the rows tell the
                      sources apart, which is what makes the next three necessary
  source_only         the class from the source alone (each source's majority class)
  deconfounded        the class accuracy after each source's mean row (from the training side) is
                      subtracted: what is left is not the source
  leave_one_source    trained on the other sources, tested on one (where it has two classes)

and `confounded: true` when the class accuracy is within 5 points of `source_only`.

GATES (Harmony 2): class at or above 90 % held-out (on the deconfounded split); stuck precision at
or above 0.9 at the D1 guard's act rate (the probe acts at most as often as the guard did on the
training side, and never below even odds), over at least 30 positives. Below its data minimum a question
reports `insufficient`, never a pass.

The probe file (`--out`, torch.save: W, b, mu, sd, classes, rows, dim, weights, temperature, the
report) is what `classifier.probe_stage.file` names; the server loads it for the weights it was
trained on only. Activations stay on the box; nothing here prints or writes a prompt's text.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import sqlite3
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HOME = os.path.expanduser("~/.qwen38-spark-engine")
DEFAULT_DIR = os.path.join(HOME, "probe")
DEFAULT_JOURNAL = os.path.join(HOME, "usage", "ledger.sqlite3")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "conductor", "task_prompts.jsonl")
QUESTIONS = {"class": "prompt", "stuck": "think", "done": "finish"}
CLASS_STAGES = ("qwen", "qwen_context")
LOOP_TASKS = ("D1", "D1b")
GATE_CLASS = 0.90
GATE_STUCK = 0.90
MIN_STUCK_POSITIVES = 30
MIN_CLASS = 40


# --------------------------------------------------------------------------- the ring
def read_index(root: str, weights: str | None = None) -> list[dict]:
    """Every snapshot's index line, oldest first, with `path` (its record file) and `day`."""
    snap = os.path.join(root, "snapshots")
    out = []
    try:
        days = sorted(os.listdir(snap))
    except OSError:
        return out
    for day in days:
        idx = os.path.join(snap, day, "index.jsonl")
        if not os.path.exists(idx):
            continue
        with open(idx, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                  # a line cut by a crash: the records stay valid
                if weights and r.get("weights") != weights:
                    continue
                r["day"], r["path"] = day, os.path.join(snap, day, r["file"])
                out.append(r)
    return out


def vectors(recs: list[dict], rows: list[str]) -> np.ndarray:
    """The chosen rows of each record, flattened: [n, len(rows) * dim] float32."""
    maps: dict[str, np.memmap] = {}
    out = []
    for r in recs:
        R, H = len(r["rows"]), int(r["dim"])
        m = maps.get(r["path"])
        if m is None:
            m = maps[r["path"]] = np.memmap(r["path"], dtype=np.float16, mode="r")
        start = int(r["off"]) // 2
        v = np.asarray(m[start:start + R * H], dtype=np.float32).reshape(R, H)
        out.append(v[[r["rows"].index(x) for x in rows]].reshape(-1))
    return np.stack(out) if out else np.zeros((0, 0), np.float32)


def pick_rows(recs: list[dict], spec: str) -> list[str]:
    names = recs[0]["rows"] if recs else []
    if spec == "all":
        return list(names)
    if spec == "taps":
        return [n for n in names if n.startswith("tap")]
    if spec == "post":
        return ["post"]
    return [x.strip() for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------- the journal
def open_journal(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)


def journal_rows(conn, row_ids, request_ids) -> tuple[dict, dict]:
    """(route rows by id, monitor acts by request id) for the snapshots' keys."""
    routes, acts = {}, collections.defaultdict(list)
    ids = sorted({x for x in row_ids if x})
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        for rid, stages, nc, outcome, src in conn.execute(
                f"SELECT id, stages, next_checker, outcome, source FROM conductor_events "
                f"WHERE id IN ({', '.join('?' * len(part))})", part):
            routes[rid] = {"stages": _j(stages) or [], "next_checker": _j(nc),
                           "outcome": _j(outcome) or {}, "source": src}
    reqs = sorted({x for x in request_ids if x})
    for i in range(0, len(reqs), 500):
        part = reqs[i:i + 500]
        for req, task, action, mode, at in conn.execute(
                f"SELECT request_id, task, action, mode, tokens_at_cut FROM conductor_events "
                f"WHERE kind = 'monitor' AND request_id IN ({', '.join('?' * len(part))})", part):
            if action and mode == "act":
                acts[req].append({"task": task, "action": action, "at": at})
    return routes, acts


def _j(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def class_label(route: dict | None) -> tuple[str, str] | None:
    """(label, source) when Qwen decided the class: its accepted no-think or in-context answer."""
    for s in (route or {}).get("stages") or []:
        if s.get("stage") in CLASS_STAGES and not s.get("error") and not s.get("below") \
                and s.get("label"):
            return s["label"], s["stage"]
    return None


def stuck_label(rec: dict, acts: list[dict], horizon: int) -> int | None:
    loops = [a for a in acts if a["task"] in LOOP_TASKS and a.get("at") is not None]
    if any(rec["at"] < a["at"] <= rec["at"] + horizon for a in loops):
        return 1
    if not loops and not any(a["task"] == "D2" for a in acts) and \
            rec.get("finish") in ("stop", "tool_calls"):
        return 0
    return None


def done_label(route: dict | None) -> int | None:
    nc = (route or {}).get("next_checker")
    if not isinstance(nc, list) or not nc:
        return None
    verdicts = [str(r.get("verdict") or "").upper() for r in nc if isinstance(r, dict)]
    if "FAIL" in verdicts:
        return 0
    if verdicts and all(v == "PASS" for v in verdicts):
        return 1
    return None


def labelled(question: str, recs: list[dict], conn, fixture: dict, horizon: int) -> list[dict]:
    """The records of the question's point, each with `y` and `source`; unlabelled ones dropped."""
    point = QUESTIONS[question]
    recs = [r for r in recs if r.get("point") == point]
    routes, acts = (journal_rows(conn, [r.get("row_id") for r in recs],
                                 [r.get("request_id") for r in recs])
                    if conn is not None else ({}, {}))
    out = []
    for r in recs:
        y, src = None, None
        if question == "class":
            if r.get("request_id") in fixture:
                y, src = fixture[r["request_id"]], "fixture"
            else:
                got = class_label(routes.get(r.get("row_id")))
                if got is not None:
                    y, src = got
        elif question == "stuck":
            y, src = stuck_label(r, acts.get(r.get("request_id"), []), horizon), r.get("profile")
        else:
            y, src = done_label(routes.get(r.get("row_id"))), r.get("profile")
        if y is not None:
            out.append(dict(r, y=y, source=src or "-"))
    return out


def read_fixture_labels(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["request_id"]] = d["label"]
    return out


# --------------------------------------------------------------------------- the fit
def held_out(sessions: list[str], frac: float) -> np.ndarray:
    """True for the held-out side: a stable hash of the session, so a session is on one side."""
    def u(s):
        return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return np.array([u(s) < frac for s in sessions])


class Logistic:
    """Multinomial logistic regression, standardised inputs, L2, accelerated gradient descent."""

    def __init__(self, l2: float = 1e-3, iters: int = 400):
        self.l2, self.iters = float(l2), int(iters)

    def fit(self, X: np.ndarray, y: np.ndarray, classes: list) -> "Logistic":
        self.classes = list(classes)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        c = len(self.classes)
        Y = np.zeros((n, c), np.float32)
        Y[np.arange(n), [self.classes.index(v) for v in y]] = 1.0
        # the step from a bound on the curvature: the largest singular value of Z, squared, over
        # n (power iteration), plus the L2 term
        v = np.random.default_rng(0).standard_normal(d).astype(np.float32)
        for _ in range(30):
            v = Z.T @ (Z @ v)
            v /= np.linalg.norm(v) + 1e-12
        L = 0.5 * float(np.linalg.norm(Z @ v) ** 2) / n + self.l2
        step = 1.0 / L
        W = np.zeros((c, d), np.float32)
        b = np.zeros(c, np.float32)
        Wm, bm, t = W.copy(), b.copy(), 1.0
        for _ in range(self.iters):
            P = _softmax(Z @ Wm.T + bm)
            G = (P - Y) / n
            gW = G.T @ Z + self.l2 * Wm
            gb = G.sum(0)
            W2, b2 = Wm - step * gW, bm - step * gb
            t2 = (1 + (1 + 4 * t * t) ** 0.5) / 2
            Wm = W2 + ((t - 1) / t2) * (W2 - W)
            bm = b2 + ((t - 1) / t2) * (b2 - b)
            W, b, t = W2, b2, t2
        self.W, self.b, self.T = W, b, 1.0
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.W.T + self.b

    def proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(X) / self.T)

    def predict(self, X: np.ndarray) -> list:
        return [self.classes[i] for i in self.logits(X).argmax(1)]

    def fit_temperature(self, X: np.ndarray, y) -> float:
        """The temperature that minimises the held-out log loss (golden section on [0.25, 8])."""
        z = self.logits(X)
        idx = np.array([self.classes.index(v) for v in y])

        def nll(T):
            p = _softmax(z / T)
            return -float(np.log(p[np.arange(len(idx)), idx] + 1e-12).mean())
        lo, hi = 0.25, 8.0
        g = (5 ** 0.5 - 1) / 2
        for _ in range(40):
            a, c = hi - g * (hi - lo), lo + g * (hi - lo)
            if nll(a) < nll(c):
                hi = c
            else:
                lo = a
        self.T = (lo + hi) / 2
        return self.T


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def accuracy(pred, y) -> float:
    return float(np.mean([a == b for a, b in zip(pred, y)])) if len(y) else float("nan")


def _fit_acc(Xtr, ytr, Xte, yte, l2, iters) -> float | None:
    if len(set(ytr)) < 2 or not len(yte):
        return None
    m = Logistic(l2, iters).fit(Xtr, ytr, sorted(set(ytr)))
    return accuracy(m.predict(Xte), yte)


def confound(X, y, src, test, l2, iters) -> dict:
    """The source-confound report of one question (see the module's header)."""
    y, src = np.asarray(y, dtype=object), np.asarray(src, dtype=object)
    tr = ~test
    out = {"sources": dict(collections.Counter(src.tolist()))}
    maj = collections.Counter(src[tr].tolist()).most_common(1)[0][1] / max(1, tr.sum())
    out["source_majority"] = round(maj, 4)
    sa = _fit_acc(X[tr], src[tr].tolist(), X[test], src[test].tolist(), l2, iters)
    out["source_accuracy"] = None if sa is None else round(sa, 4)
    per = {s: collections.Counter(y[tr][src[tr] == s].tolist()).most_common(1)[0][0]
           for s in set(src[tr].tolist())}
    base = [per.get(s) for s in src[test].tolist()]
    out["source_only"] = round(accuracy(base, y[test].tolist()), 4) if test.any() else None
    means = {s: X[tr][src[tr] == s].mean(0) for s in per}
    Xd = X - np.stack([means.get(s, 0.0 * X[0]) for s in src.tolist()])
    dc = _fit_acc(Xd[tr], y[tr].tolist(), Xd[test], y[test].tolist(), l2, iters)
    out["deconfounded"] = None if dc is None else round(dc, 4)
    loso = {}
    for s in sorted(set(src.tolist())):
        held = src == s
        if len(set(y[held].tolist())) < 2 or len(set(y[~held].tolist())) < 2:
            continue
        a = _fit_acc(X[~held], y[~held].tolist(), X[held], y[held].tolist(), l2, iters)
        if a is not None:
            loso[s] = round(a, 4)
    out["leave_one_source"] = loso
    return out


def fit_question(question: str, recs: list[dict], rows: list[str], *, test_frac=0.25, l2=1e-3,
                 iters=400) -> tuple[dict, Logistic | None]:
    """The probe and its report for one question's labelled records."""
    rep = {"question": question, "n": len(recs), "rows": rows,
           "labels": dict(collections.Counter(str(r["y"]) for r in recs))}
    if not recs:
        rep["verdict"] = "insufficient"
        return rep, None
    X = vectors(recs, rows)
    y = [r["y"] for r in recs]
    test = held_out([r.get("session") or r.get("request_id") for r in recs], test_frac)
    if test.all() or not test.any():
        test = np.arange(len(recs)) % 4 == 0           # too few sessions to split by session
        rep["split"] = "by_record"
    else:
        rep["split"] = "by_session"
    tr = ~test
    rep["train"], rep["held_out"] = int(tr.sum()), int(test.sum())
    classes = sorted(set(y), key=str)
    if len(set(np.asarray(y, dtype=object)[tr].tolist())) < 2:
        rep["verdict"] = "insufficient"
        return rep, None
    m = Logistic(l2, iters).fit(X[tr], [y[i] for i in np.where(tr)[0]], classes)
    yte = [y[i] for i in np.where(test)[0]]
    pred = m.predict(X[test])
    rep["accuracy"] = round(accuracy(pred, yte), 4)
    rep["temperature"] = round(m.fit_temperature(X[test], yte), 4) if yte else 1.0
    if question == "class":
        rep["confound"] = confound(X, y, [r["source"] for r in recs], test, l2, iters)
        c = rep["confound"]
        ref = c["deconfounded"] if c["deconfounded"] is not None else rep["accuracy"]
        rep["confounded"] = bool(c["source_only"] is not None
                                 and rep["accuracy"] <= c["source_only"] + 0.05
                                 and len(c["sources"]) > 1)
        if len(recs) < MIN_CLASS:
            rep["verdict"] = "insufficient"
        else:
            rep["verdict"] = ("pass" if ref >= GATE_CLASS and not rep["confounded"] else "fail")
        rep["gate"] = {"held_out_min": GATE_CLASS, "on": "deconfounded"}
    elif question == "stuck":
        pos = int(sum(1 for v in y if v == 1))
        rate = float(np.mean([v == 1 for v in np.asarray(y, dtype=object)[tr]]))
        p1 = m.proba(X[test])[:, classes.index(1)] if 1 in classes else np.zeros(test.sum())
        # the probe acts at most as often as the guard did on the training side, and never
        # below even odds
        thr = max(0.5, float(np.quantile(p1, 1 - rate))) if len(p1) and rate > 0 else 1.0
        acts = p1 >= thr
        tp = int(sum(1 for a, v in zip(acts, yte) if a and v == 1))
        rep.update(positives=pos, act_rate=round(rate, 4), threshold=round(thr, 4),
                   acts=int(acts.sum()), precision=round(float(tp / acts.sum()), 4)
                   if acts.sum() else None,
                   recall=round(tp / max(1, sum(1 for v in yte if v == 1)), 4))
        rep["gate"] = {"precision_min": GATE_STUCK, "positives_min": MIN_STUCK_POSITIVES}
        if pos < MIN_STUCK_POSITIVES or rep["precision"] is None:
            rep["verdict"] = "insufficient"
        else:
            rep["verdict"] = "pass" if rep["precision"] >= GATE_STUCK else "fail"
    else:
        rep["verdict"] = "report"                      # no gate yet for done (Harmony 2)
    return rep, m


def save_probe(path: str, m: Logistic, rep: dict, recs: list[dict], rows: list[str]) -> None:
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), mode=0o700, exist_ok=True)
    torch.save({"question": rep["question"], "classes": [str(c) for c in m.classes],
                "rows": rows, "dim": int(recs[0]["dim"]), "weights": recs[0].get("weights"),
                "temperature": float(m.T), "W": torch.from_numpy(m.W.astype(np.float32)),
                "b": torch.from_numpy(m.b.astype(np.float32)),
                "mu": torch.from_numpy(m.mu.astype(np.float32)),
                "sd": torch.from_numpy(m.sd.astype(np.float32)),
                "report": json.dumps(rep)}, path)
    os.chmod(path, 0o600)


# --------------------------------------------------------------------------- commands
def cmd_fit(a) -> int:
    recs = read_index(os.path.expanduser(a.dir))
    if a.weights == "auto" and recs:
        a.weights = collections.Counter(r.get("weights") for r in recs).most_common(1)[0][0]
    if a.weights:
        recs = [r for r in recs if r.get("weights") == a.weights]
    conn = open_journal(os.path.expanduser(a.journal)) if os.path.exists(
        os.path.expanduser(a.journal)) else None
    fixture = read_fixture_labels(a.fixture_labels and os.path.expanduser(a.fixture_labels))
    report = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "weights": a.weights,
              "snapshots": len(recs), "questions": {}}
    for q in (QUESTIONS if a.question == "all" else [a.question]):
        lab = labelled(q, recs, conn, fixture, a.horizon)
        rows = pick_rows(lab, a.rows)
        t0 = time.perf_counter()
        rep, m = fit_question(q, lab, rows, test_frac=a.test, l2=a.l2, iters=a.iters)
        rep["fit_s"] = round(time.perf_counter() - t0, 2)
        report["questions"][q] = rep
        print(f"[probe] {q}: n={rep['n']} labels={rep['labels']} verdict={rep['verdict']}"
              + (f" accuracy={rep.get('accuracy')}" if "accuracy" in rep else "")
              + (f" precision={rep.get('precision')}" if "precision" in rep else ""), flush=True)
        if "confound" in rep:
            c = rep["confound"]
            print(f"[probe]   source: accuracy {c['source_accuracy']} (majority "
                  f"{c['source_majority']}), class from the source alone {c['source_only']}, "
                  f"deconfounded {c['deconfounded']}, leave one source out "
                  f"{c['leave_one_source']}, confounded {rep['confounded']}", flush=True)
        if m is not None and a.out and (a.question != "all"):
            save_probe(os.path.expanduser(a.out), m, rep, lab, rows)
            print(f"[probe]   probe written: {a.out}", flush=True)
    if a.report:
        with open(os.path.expanduser(a.report), "w") as f:
            json.dump(report, f, indent=1)
    return 0


def cmd_fixture(a) -> int:
    """The labelled prompts through a TandemLLM server whose snapshots are on: one request with
    tools each, one output token, its own session; the request ids and the labels to `--out`."""
    import urllib.parse
    import urllib.request
    port = urllib.parse.urlparse(a.server).port
    if port == 8000 and not a.live:
        print("[probe] refusing :8000 without --live (the live engine)", flush=True)
        return 2
    tool = {"type": "function", "function": {
        "name": "noop", "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    rows = [json.loads(line) for line in open(a.set, encoding="utf-8") if line.strip()]
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), mode=0o700, exist_ok=True)
    tok = os.environ.get(a.token_env) if a.token_env else None
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            body = {"model": "TandemLLM", "max_tokens": 1, "stream": False, "tools": [tool],
                    "messages": [{"role": "user", "content": r["text"]}]}
            req = urllib.request.Request(a.server.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("x-session-id", f"probe-fixture-{i:03d}")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=600) as resp:
                rid = json.loads(resp.read())["id"]
            f.write(json.dumps({"request_id": rid, "label": r["label"], "i": i}) + "\n")
            n += 1
    print(f"[probe] fixture: {n} requests, labels in {out}", flush=True)
    return 0


def cmd_growth(a) -> int:
    """A night's snapshot bytes from the usage ledger: TandemLLM requests (the router on) of
    opencode (every one offers tools), their reasoning tokens at one point every `--every`, and
    their done claims (no tool call)."""
    from server.probe import INDEX_LINE_BYTES, MAX_ROWS, night_bytes

    def ms(s):
        return int(_dt.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp() * 1000)
    conn = open_journal(os.path.expanduser(a.ledger))
    q = ("SELECT count(*), coalesce(sum(reasoning_tokens / ?), 0), "
         "coalesce(sum(tool_calls = 0), 0), coalesce(sum(reasoning_tokens), 0) FROM requests "
         "WHERE ts_ms >= ? AND ts_ms < ? AND conductor_mode = 'on' AND client_kind = 'opencode'")
    n, think, done, rt = conn.execute(q, (a.every, ms(a.since), ms(a.until))).fetchone()
    raw = conn.execute("SELECT count(*) FROM requests WHERE ts_ms >= ? AND ts_ms < ? AND "
                       "conductor_mode = 'raw' AND client_kind = 'opencode'",
                       (ms(a.since), ms(a.until))).fetchone()[0]
    per = MAX_ROWS * a.hidden * 2 + INDEX_LINE_BYTES
    total = night_bytes(n, think, done, MAX_ROWS, a.hidden)
    print(json.dumps({"window": [a.since, a.until], "tandem_tool_requests": n,
                      "raw_tool_requests_never_written": raw, "reasoning_tokens": rt,
                      "think_points": think, "finish_points": done, "prompt_points": n,
                      "bytes_per_point": per, "bytes": total, "mb": round(total / 1e6, 1),
                      "gate_mb": 100, "pass": total < 100e6}, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--question", choices=list(QUESTIONS) + ["all"], default="class")
    f.add_argument("--dir", default=DEFAULT_DIR)
    f.add_argument("--journal", default=DEFAULT_JOURNAL)
    f.add_argument("--fixture-labels", default=os.path.join(DEFAULT_DIR, "fixture",
                                                            "labels.jsonl"))
    f.add_argument("--weights", default="auto", help="the weights key to train on (auto: the "
                   "most common in the ring)")
    f.add_argument("--rows", default="all")
    f.add_argument("--horizon", type=int, default=1024)
    f.add_argument("--test", type=float, default=0.25)
    f.add_argument("--l2", type=float, default=1e-3)
    f.add_argument("--iters", type=int, default=400)
    f.add_argument("--out")
    f.add_argument("--report")
    x = sub.add_parser("fixture")
    x.add_argument("--server", required=True)
    x.add_argument("--set", default=FIXTURE)
    x.add_argument("--out", default=os.path.join(DEFAULT_DIR, "fixture", "labels.jsonl"))
    x.add_argument("--token-env", default="")
    x.add_argument("--live", action="store_true", help="allow :8000")
    g = sub.add_parser("growth")
    g.add_argument("--ledger", default=DEFAULT_JOURNAL)
    g.add_argument("--since", required=True)
    g.add_argument("--until", required=True)
    g.add_argument("--every", type=int, default=1024)
    g.add_argument("--hidden", type=int, default=5120)
    a = p.parse_args(argv)
    return {"fit": cmd_fit, "fixture": cmd_fixture, "growth": cmd_growth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
