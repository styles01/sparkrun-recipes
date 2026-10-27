"""Laya for the model's own scripts (L1, 2026-10-03).

The owner, 2026-10-02 18:45 and 2026-10-03 00:20: the model must be able to reach Laya through our
provider, from the scripts it writes (a test that runs the app, collects its log, console or state
and asks Laya typed questions in loops and assertions; a batch check of many outputs), for any
software. `/v1/systemone` is the decision models' full Jev contract behind the admin token; this
is the small, stable shape a script needs, open like the chat API it sits beside:

    GET  /v1/laya          what the route takes and answers, its limits, whether it is on
    POST /v1/laya/ask      {question, choices | scale | neither (yes or no), text | texts,
                            threshold?, model?, deadline_ms?}

    choices   a list of labels, or {label: what it means}: 2 to the model's `max_options` (20);
              a label is a word of letters, digits, `_`, `-` or `.`, at most 40 characters
    scale     a list of 2 to 10 level descriptions, lowest first: the answer is the level, its
              index and the expected level (`score`, a float)
    neither   `question` is a statement or a yes-no question: the answer is `yes` or `no`, with
              `yes` the probability that it holds

    one text  -> {label, p, sure, probs, model, ms, chars, read_chars, clipped}
               (+ level and score for a scale, + yes for a yes-no)
    texts     -> {answers: [one per text, in order], model, ms}

`p` is the probability of the label and `sure` is `p >= threshold` (the request's, else the
`deciders` document's `api.threshold`, 0.8: Laya's routing answers at 0.80 were right 97 % of the
time; on test evidence it is untrained, so a script treats `sure: false` as "look yourself").
Laya reads a window of about 1,100 characters (laya-english: 512 tokens less the question's 192):
a longer text is cut to its head (a quarter) and its tail around a `[...]` line, by code, and the
answer says so (`clipped`, `read_chars`).

Laya answers in the engine's idle window only (stage A), which is where a script runs: between
two requests of the session that wrote it. A caller that arrives while a completion holds the
engine waits up to its deadline (`deadline_ms`, the document's 15 s by default, at most 60 s),
then gets 503 `busy` with `Retry-After`.

Errors are the OpenAI API's shape, `{"error": {"message", "type", "param"?}}`, so a script that
reads the chat API reads these: 422 `invalid_request_error`, 429 `rate_limited` (the document's
`api.rate_per_min` texts a minute over every caller), 503 `busy` / `loading` / `not_loaded` /
`refused` / `failed` with `Retry-After`, 404 when the route is off or the engine serves no
decision model.

Auth is the chat API's: none at the engine (server/auth.py: the OpenAI routes stay open; the
service is LAN-only behind the reverse proxy). A script sends nothing secret, and a bearer
header it sends anyway is ignored, as the chat routes ignore it. The answers carry
`Access-Control-Allow-Origin: *` and `OPTIONS` answers the preflight, so a test the model runs
inside the page it builds (the browser tool's evaluate script) can ask Laya with `fetch` too: the
route holds no credentials and reads no cookie.

Every call is one journal row (`kind: laya_api`, task `ask`): the type, the labels (names), the
number of texts, their sizes, what was cut, the answers' labels and probabilities, the model and
the time; the question as a hash and a length, never a text, as every row the journal keeps
outside the router's own.

DRIVER SESSIONS (X1, 2026-10-03). The intended use of Laya: it PLAYS the software under test. A driver
(the playtest kit, `GET /v1/laya/kit/<file>`: `laya_drive.py`, `laya-drive.js`) sends the state of a
game, a UI, a CLI or an API every ~250 ms and applies the action Laya picks from a fixed set; its
log goes back to Qwen. A call with `session` (a name of letters, digits and `_ . : -`, at most 64)
is one step of such a loop:

  - its rate is the session's own, `api.drive_rate_per_s` texts a second (10: one decision every
    100 ms), not the scripts' minute (`rate_per_min`), so a playtest never starves the scripts and
    the reverse; at most `api.drive_sessions` (4) live at once, `api.drive_max_texts` (36,000, an
    hour at 10 a second) each; over any of them: 429 with Retry-After;
  - no journal row a call (10 a second would be the journal); one row a session (`kind:
    laya_drive`, task `session`) when it ends: `POST /v1/laya/drive/end {session, summary?}`, or
    `api.drive_idle_s` (30 s) after its last call, written at the next call of any Laya route.
    The row: the calls, the texts, the errors by kind, the labels' counts (names), the mean and
    lowest p, how many were not sure, the time p50/p95/max, the seconds, how it ended, the
    question as a hash, and the driver's own summary (episodes, deaths, wins, timeouts, stuck,
    steps: numbers only, anything else is dropped). Never a state or a question's words.
  - the answer is the route's (the same contract); the session adds nothing to it.

Latency: a step is the HTTP call, the batcher's 2 ms collation and one forward. The journal's
`laya_api` rows of 2026-10-03 give 18.7 to 48.7 ms for one text on the box (laya-english, idle
GPU), 122.9 ms for 7 texts. The idle rule holds for a playtest: the driver runs inside a tool call
(opencode's bash), and opencode sends its next request only when the tool returns. The request
timeline of 2026-10-03 (`requests` against `laya_api` rows, 16 calls): 14 met an idle engine, 2
arrived while the request that wrote the tool call was still finishing (opencode runs a tool as
soon as its call is complete in the stream; the call came 96 and 115 ms before that request's
end) and waited for it: 148.5 and 187.8 ms. So the first step of a playtest may wait about 0.2 s,
the rest meet an idle engine unless another client (Hermes, a second opencode session) runs a
request meanwhile; then a step waits up to its `deadline_ms` and is 503 `busy`, which the kit
logs and answers with its fallback action.
"""

from __future__ import annotations

import collections
import hashlib
import os
import re
import threading
import time

LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
SESSION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# the driver's summary in its session row: these numbers, nothing else
SUMMARY_KEYS = ("episodes", "steps", "deaths", "wins", "timeouts", "stuck", "busy", "fallbacks")
DRIVE_MS_KEPT = 4096          # a session keeps its last this many call times for p50/p95
MAX_QUESTION = 400
MAX_DESCRIPTION = 200
MAX_DEADLINE_MS = 60_000
MIN_LEVELS, MAX_LEVELS = 2, 10
CHARS_PER_TOKEN = 3.5
HEAD = 0.25                  # the share of Laya's window the head of a long text gets
FORWARD_GRACE_S = 30.0


class Refused(ValueError):
    """A request the route answers with an error before any model runs."""

    def __init__(self, status: int, kind: str, message: str, param: str | None = None,
                 retry: int | None = None):
        super().__init__(message)
        self.status, self.kind, self.param, self.retry = status, kind, param, retry

    def body(self) -> dict:
        err = {"message": str(self), "type": self.kind}
        if self.param:
            err["param"] = self.param
        return {"error": err}


class Rate:
    """Texts a minute over every caller: a sliding window of the last minute's stamps."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._t: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def take(self, n: int, per_min: int) -> int | None:
        """None when `n` texts fit in the window (they are counted), else the seconds to wait."""
        now = self.clock()
        with self._lock:
            while self._t and now - self._t[0] >= 60.0:
                self._t.popleft()
            if len(self._t) + n > per_min:
                if not self._t or n > per_min:
                    return 60
                return max(1, int(60.0 - (now - self._t[0])) + 1)
            self._t.extend([now] * n)
            return None


class Drive:
    """The driver sessions (DRIVER SESSIONS): each one's own rate and its compact stats, and its
    journal row when it ends."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._s: dict[str, dict] = {}
        self._lock = threading.Lock()

    def live(self) -> int:
        with self._lock:
            return len(self._s)

    def sweep(self, cfg: dict) -> list[dict]:
        """The rows of the sessions idle past `drive_idle_s`, now closed."""
        now = self.clock()
        with self._lock:
            gone = [k for k, v in self._s.items() if now - v["last"] >= float(cfg["drive_idle_s"])]
            return [self._row(self._s.pop(k), "idle") for k in gone]

    def take(self, sid: str, n: int, cfg: dict) -> None:
        """Count `n` texts for session `sid` (opened here); raises `Refused` over a limit."""
        now = self.clock()
        with self._lock:
            s = self._s.get(sid)
            if s is None:
                if len(self._s) >= int(cfg["drive_sessions"]):
                    raise Refused(429, "rate_limited", f"at most {cfg['drive_sessions']} driver "
                                  f"sessions at once; end one (POST /v1/laya/drive/end) or wait "
                                  f"{cfg['drive_idle_s']} s", "session", retry=5)
                s = self._s[sid] = {
                    "id": sid, "t0": now, "last": now, "stamps": collections.deque(),
                    "calls": 0, "texts": 0, "errors": collections.Counter(),
                    "labels": collections.Counter(), "p_sum": 0.0, "p_n": 0, "p_min": None,
                    "unsure": 0, "ms": collections.deque(maxlen=DRIVE_MS_KEPT), "q": set()}
            s["last"] = now
            if s["texts"] + n > int(cfg["drive_max_texts"]):
                raise Refused(429, "rate_limited", f"the session used its {cfg['drive_max_texts']} "
                              f"texts; end it and start another", "session", retry=1)
            st = s["stamps"]
            while st and now - st[0] >= 1.0:
                st.popleft()
            per_s = int(cfg["drive_rate_per_s"])
            if len(st) + n > per_s:
                raise Refused(429, "rate_limited", f"at most {per_s} texts a second in a driver "
                              f"session", "session", retry=1)
            st.extend([now] * n)
            s["texts"] += n

    def record(self, sid: str, body, out: dict | None, err: Refused | None) -> None:
        """One call's numbers into its session (a refused call before `take` has none)."""
        with self._lock:
            s = self._s.get(sid)
            if s is None:
                return
            s["calls"] += 1
            q = body.get("question") if isinstance(body, dict) else None
            if isinstance(q, str) and len(s["q"]) < 16:
                s["q"].add(hashlib.sha256(q.encode()).hexdigest()[:12])
            if err is not None:
                s["errors"][err.kind] += 1
                return
            s["ms"].append(out["ms"])
            for a in out.get("answers") or [out]:
                s["labels"][str(a["label"]) if "level" not in a else f"level_{a['level']}"] += 1
                s["p_sum"] += a["p"]
                s["p_n"] += 1
                s["p_min"] = a["p"] if s["p_min"] is None else min(s["p_min"], a["p"])
                s["unsure"] += not a["sure"]

    def end(self, sid: str) -> dict | None:
        with self._lock:
            s = self._s.pop(sid, None)
        return None if s is None else self._row(s, "end")

    def _row(self, s: dict, how: str) -> dict:
        ms = sorted(s["ms"])

        def pct(q):
            return ms[min(len(ms) - 1, int(q * len(ms)))] if ms else None
        return {"session": s["id"], "ended": how, "calls": s["calls"], "texts": s["texts"],
                "errors": dict(s["errors"]), "labels": dict(s["labels"].most_common(20)),
                "p_mean": round(s["p_sum"] / s["p_n"], 4) if s["p_n"] else None,
                "p_min": s["p_min"], "unsure": s["unsure"], "ms_p50": pct(0.5),
                "ms_p95": pct(0.95), "ms_max": ms[-1] if ms else None,
                "seconds": round(s["last"] - s["t0"], 1), "q": sorted(s["q"])}


def summary_of(body) -> dict:
    """The driver's own summary, numbers only (SUMMARY_KEYS)."""
    sm = body.get("summary") if isinstance(body, dict) else None
    if not isinstance(sm, dict):
        return {}
    return {k: int(sm[k]) for k in SUMMARY_KEYS
            if isinstance(sm.get(k), (int, float)) and not isinstance(sm.get(k), bool)
            and 0 <= sm[k] < 10 ** 9}


RATE = Rate()
DRIVE = Drive()
# what the Deciders screen shows: calls, texts, errors, the last call's time and when it was
STATS = {"calls": 0, "texts": 0, "errors": 0, "rate_limited": 0, "last_ms": None,
         "last_at": None}
_STATS_LOCK = threading.Lock()


def stats() -> dict:
    with _STATS_LOCK:
        out = dict(STATS)
    out["drive_sessions"] = DRIVE.live()
    return out


def _count(out: dict | None, err) -> None:
    with _STATS_LOCK:
        STATS["calls"] += 1
        STATS["last_at"] = round(time.time(), 3)
        if err is not None:
            STATS["errors"] += 1
            STATS["rate_limited"] += err.status == 429
        else:
            STATS["texts"] += len(out.get("answers") or [out])
            STATS["last_ms"] = out["ms"]


def clip(text: str, limit: int) -> str:
    """`text` within `limit` characters: its head (a quarter) and its tail around `[...]`."""
    if len(text) <= limit:
        return text
    h = int(limit * HEAD)
    return text[:h] + "\n[...]\n" + text[-(limit - h - 7):]


def _labels(choices, max_options: int) -> dict[str, str]:
    if isinstance(choices, list):
        if not all(isinstance(c, str) for c in choices):
            raise Refused(422, "invalid_request_error", "choices: a list of labels (strings)",
                          "choices")
        out = {c: c for c in choices}
        if len(out) != len(choices):
            raise Refused(422, "invalid_request_error", "choices: a label is listed twice",
                          "choices")
    elif isinstance(choices, dict):
        out = {}
        for k, v in choices.items():
            if not isinstance(v, str) or not v.strip():
                raise Refused(422, "invalid_request_error",
                              f"choices[{k!r}]: what the label means, a non-empty string",
                              "choices")
            out[k] = v.strip()
    else:
        raise Refused(422, "invalid_request_error",
                      "choices: a list of labels or an object {label: what it means}", "choices")
    if not 2 <= len(out) <= max_options:
        raise Refused(422, "invalid_request_error",
                      f"choices: 2 to {max_options} labels, got {len(out)}", "choices")
    for k, v in out.items():
        if not LABEL_RE.match(k):
            raise Refused(422, "invalid_request_error",
                          f"choices: {k!r} is not a label (letters, digits, _ - ., at most 40)",
                          "choices")
        if len(v) > MAX_DESCRIPTION:
            raise Refused(422, "invalid_request_error",
                          f"choices[{k!r}]: at most {MAX_DESCRIPTION} characters", "choices")
    return out


def parse(body, cfg: dict, max_options: int) -> dict:
    """The request, checked before any model runs: `{type, question, criteria, labels, texts,
    threshold, model, deadline_ms}`."""
    if not isinstance(body, dict):
        raise Refused(422, "invalid_request_error", "the body is a JSON object")
    q = body.get("question")
    if not isinstance(q, str) or not q.strip():
        raise Refused(422, "invalid_request_error", "question: a non-empty string", "question")
    if len(q) > MAX_QUESTION:
        raise Refused(422, "invalid_request_error",
                      f"question: at most {MAX_QUESTION} characters", "question")
    if "text" in body and "texts" in body:
        raise Refused(422, "invalid_request_error", "send text or texts, not both", "texts")
    if "texts" in body:
        texts = body["texts"]
        if not isinstance(texts, list) or not texts or \
                not all(isinstance(t, str) for t in texts):
            raise Refused(422, "invalid_request_error", "texts: a non-empty list of strings",
                          "texts")
        if len(texts) > int(cfg["max_texts"]):
            raise Refused(422, "invalid_request_error",
                          f"texts: at most {cfg['max_texts']} in one call, got {len(texts)}",
                          "texts")
    else:
        t = body.get("text")
        if not isinstance(t, str):
            raise Refused(422, "invalid_request_error",
                          "text: the evidence to judge, a string (or texts: a list)", "text")
        texts = [t]
    for i, t in enumerate(texts):
        if not t.strip():
            # FX13: say what an empty text means, so the caller can fix it (the model piped its
            # evidence into a heredoc script whose stdin was the script, and sent "")
            which = "text" if "texts" not in body else f"texts[{i}]"
            raise Refused(422, "invalid_request_error",
                          f"{which} is empty: there is nothing to judge; send the evidence itself "
                          f"(a log tail, an output, a state dump), not an empty string", "texts")
        if len(t) > int(cfg["max_chars"]):
            raise Refused(422, "invalid_request_error",
                          f"text {i}: at most {cfg['max_chars']} characters; send the part to "
                          f"judge (a log's tail, the output, the state)", "texts")
    if "choices" in body and "scale" in body:
        raise Refused(422, "invalid_request_error", "send choices or scale, not both", "scale")
    if "choices" in body:
        labels = _labels(body["choices"], max_options)
        qtype, crit = "choice", dict(labels)
    elif "scale" in body:
        sc = body["scale"]
        if not isinstance(sc, list) or not all(isinstance(x, str) and x.strip() for x in sc):
            raise Refused(422, "invalid_request_error",
                          "scale: a list of level descriptions, lowest first", "scale")
        if not MIN_LEVELS <= len(sc) <= MAX_LEVELS:
            raise Refused(422, "invalid_request_error",
                          f"scale: {MIN_LEVELS} to {MAX_LEVELS} levels, got {len(sc)}", "scale")
        if any(len(x) > MAX_DESCRIPTION for x in sc):
            raise Refused(422, "invalid_request_error",
                          f"scale: a level is at most {MAX_DESCRIPTION} characters", "scale")
        qtype, crit, labels = "score", [x.strip() for x in sc], None
    else:
        qtype, crit, labels = "noul", None, None
    th = body.get("threshold", cfg["threshold"])
    if isinstance(th, bool) or not isinstance(th, (int, float)) or not 0.0 <= th <= 1.0:
        raise Refused(422, "invalid_request_error", "threshold: a number from 0 to 1",
                      "threshold")
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise Refused(422, "invalid_request_error", "model: a decision model's name or auto",
                      "model")
    dl = body.get("deadline_ms", cfg["deadline_ms"])
    if isinstance(dl, bool) or not isinstance(dl, (int, float)) or not 1 <= dl <= MAX_DEADLINE_MS:
        raise Refused(422, "invalid_request_error",
                      f"deadline_ms: 1 to {MAX_DEADLINE_MS}", "deadline_ms")
    sid = body.get("session")
    if sid is not None and (not isinstance(sid, str) or not SESSION_RE.match(sid)):
        raise Refused(422, "invalid_request_error", "session: a driver session's name (letters, "
                      "digits, _ . : -, at most 64)", "session")
    return {"type": qtype, "question": q.strip(), "criteria": crit, "labels": labels,
            "texts": texts, "threshold": float(th), "model": model, "deadline_ms": float(dl),
            "session": sid}


def answer_of(a: dict, req: dict) -> dict:
    """One model answer in the route's shape."""
    th = req["threshold"]
    if req["type"] == "noul":
        yes = float(a["noul"])
        label, p = ("yes", yes) if yes >= 0.5 else ("no", 1.0 - yes)
        return {"label": label, "p": round(p, 4), "sure": p >= th, "yes": round(yes, 4),
                "probs": {"yes": round(yes, 4), "no": round(1.0 - yes, 4)}}
    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
    if req["type"] == "choice":
        label = a["choice"]
        p = probs.get(label, 0.0)
        return {"label": label, "p": round(p, 4), "sure": p >= th,
                "probs": {k: round(v, 4) for k, v in probs.items()}}
    levels = req["criteria"]
    level = max(range(len(levels)), key=lambda i: probs.get(str(i), 0.0))
    p = probs.get(str(level), 0.0)
    return {"label": levels[level], "level": level, "score": round(float(a["score"]), 4),
            "p": round(p, 4), "sure": p >= th,
            "probs": {str(i): round(probs.get(str(i), 0.0), 4) for i in range(len(levels))}}


def limits(reg, model) -> dict:
    lim = dict(reg.limits(model or "auto") or {})
    ctx, qt = int(lim.get("context_tokens") or 512), int(lim.get("question_tokens") or 192)
    return {"context_tokens": ctx, "question_tokens": qt,
            "max_options": int(lim.get("max_options") or 20),
            "read_chars": max(200, int((ctx - qt) * CHARS_PER_TOKEN))}


def _error(exc) -> Refused:
    """A runtime error as the route's answer."""
    from engine.decision.base import DeadlineMissed, DecisionError
    from engine.decision.registry import UnknownModel
    if isinstance(exc, UnknownModel):
        return Refused(422, "invalid_request_error", str(exc), "model")
    if isinstance(exc, DeadlineMissed):
        return Refused(503, "busy", f"{exc}; the engine served a request the whole time; ask "
                                    f"again, or send a longer deadline_ms", retry=exc.retry or 2)
    if isinstance(exc, DecisionError):
        kind = {422: "invalid_request_error", 529: "overloaded"}.get(exc.status, exc.why)
        return Refused(exc.status if exc.status != 529 else 503, kind, str(exc),
                       retry=(exc.retry or 2) if exc.status >= 500 else None)
    if isinstance(exc, TimeoutError):
        return Refused(504, "timeout", "the forward did not finish in time", retry=5)
    return Refused(500, "error", f"{type(exc).__name__}: {exc}")


def ask(reg, body, cfg: dict, default_model: str = "auto", rate: Rate = RATE,
        drive: Drive = DRIVE) -> dict:
    """The answer of one call; raises `Refused`. `cfg` is the deciders document's `api`. A call
    with `session` counts against its driver session's rate (DRIVER SESSIONS), not the minute's."""
    t0 = time.perf_counter()
    model = body.get("model") if isinstance(body, dict) and isinstance(body.get("model"), str) \
        else default_model
    try:
        lim = limits(reg, model)
    except Exception as exc:                                       # noqa: BLE001
        raise _error(exc) from exc
    req = parse(body, cfg, lim["max_options"])
    if req["session"] is not None:
        drive.take(req["session"], len(req["texts"]), cfg)
        wait = None
    else:
        wait = rate.take(len(req["texts"]), int(cfg["rate_per_min"]))
    if wait is not None:
        raise Refused(429, "rate_limited", f"at most {cfg['rate_per_min']} texts a minute; "
                                           f"wait {wait} s", retry=wait)
    model = req["model"] or default_model
    read = lim["read_chars"]
    shown = [clip(t, read) for t in req["texts"]]
    q = {"q": {"type": req["type"], "instructions": req["question"],
               **({"criteria": req["criteria"]} if req["criteria"] is not None else {})}}
    futs = []
    try:
        for s in shown:
            futs.append(reg.ask(s, q, model=model, window="idle",
                                deadline_ms=req["deadline_ms"], source="laya_api"))
        outs = [f.result(timeout=req["deadline_ms"] / 1000.0 + FORWARD_GRACE_S) for f in futs]
    except Exception as exc:                                       # noqa: BLE001
        raise _error(exc) from exc
    answers = []
    for t, s, o in zip(req["texts"], shown, outs):
        a = answer_of(o["answers"]["q"], req)
        a.update(chars=len(t), read_chars=len(s), clipped=len(t) > len(s))
        answers.append(a)
    ms = round((time.perf_counter() - t0) * 1e3, 1)
    used = outs[0]["model"] if outs else model
    out = (dict(answers[0], model=used, ms=ms) if "text" in body
           else {"answers": answers, "model": used, "ms": ms})
    out["_req"] = req
    return out


def card(reg, cfg: dict, default_model: str = "auto") -> dict:
    """`GET /v1/laya`: what a script needs to know, from the code and the document."""
    lim = limits(reg, default_model)
    return {
        "route": "POST /v1/laya/ask", "on": cfg["mode"] == "on", "ready": bool(reg.ready()),
        "model": default_model, "loaded": reg.loaded(),
        "request": {"question": f"string, at most {MAX_QUESTION} characters",
                    "choices": f"[label, ...] or {{label: meaning}}, 2 to {lim['max_options']}",
                    "scale": f"[lowest level, ..., highest], {MIN_LEVELS} to {MAX_LEVELS}",
                    "neither": "a yes-no question: label yes or no",
                    "text": "the evidence (a log tail, an output, a state dump)",
                    "texts": f"or a list, at most {cfg['max_texts']}",
                    "threshold": f"optional, default {cfg['threshold']}",
                    "deadline_ms": f"optional, default {cfg['deadline_ms']}, at most "
                                   f"{MAX_DEADLINE_MS}",
                    "session": f"optional: a driver session's name; its own rate, "
                               f"{cfg['drive_rate_per_s']} texts a second; end it with POST "
                               f"/v1/laya/drive/end"},
        "response": {"label": "the answer", "p": "its probability",
                     "sure": "p >= threshold", "probs": "every option's probability",
                     "level, score": "for a scale", "yes": "for a yes-no question",
                     "read_chars, clipped": "what Laya read of the text: the head and the tail",
                     "answers": "for texts, one per text"},
        "limits": {"read_chars": lim["read_chars"], "max_chars": cfg["max_chars"],
                   "max_texts": cfg["max_texts"], "rate_per_min": cfg["rate_per_min"],
                   "drive_rate_per_s": cfg["drive_rate_per_s"],
                   "drive_sessions": cfg["drive_sessions"]},
        "kit": {f: f"GET /v1/laya/kit/{f}" for f in KIT_FILES},
        "auth": "none at the engine, as the chat API",
    }


def journal_detail(out: dict | None, err: Refused | None, body) -> tuple[str, dict]:
    """The journal row's decision and detail: names, counts and numbers, never a text."""
    q = body.get("question") if isinstance(body, dict) else None
    qh = hashlib.sha256(q.encode()).hexdigest()[:12] if isinstance(q, str) else None
    detail = {"q": qh, "q_chars": len(q) if isinstance(q, str) else None}
    if err is not None:
        detail.update(status=err.status, error=err.kind, param=err.param)
        return err.kind, detail
    req = out["_req"]
    answers = out.get("answers") or [out]
    detail.update(type=req["type"], options=list(req["labels"]) if req["labels"] else
                  (len(req["criteria"]) if req["criteria"] else 2),
                  texts=len(answers), chars=sum(a["chars"] for a in answers),
                  clipped=sum(1 for a in answers if a["clipped"]),
                  labels=[a["label"] if req["type"] != "score" else a["level"]
                          for a in answers][:16],
                  p=[a["p"] for a in answers][:16], sure=sum(1 for a in answers if a["sure"]),
                  model=out["model"], ms=out["ms"], threshold=req["threshold"])
    first = answers[0]
    return (str(first["label"]) if req["type"] != "score" else f"level_{first['level']}"), detail


# ---------------------------------------------------------------------- server/app.py's routes
def _cfg(state: dict) -> tuple[dict, str]:
    from server import decider_api
    from server.conductor import schema
    doc = decider_api._deciders_doc(state)
    return schema.decider_api(doc), (doc.get("questions") or {}).get("laya_api") or "auto"


CORS = (("Access-Control-Allow-Origin", "*"),)
# the playtest kit the model copies into a project (`GET /v1/laya/kit/<file>`)
KIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laya_kit")
KIT_FILES = {"laya_drive.py": "text/x-python; charset=utf-8",
             "laya-drive.js": "text/javascript; charset=utf-8"}


def handle_options(handler) -> None:
    """The browser's preflight for a page's `fetch` with a JSON content type."""
    handler.send_response(204)
    for k, v in CORS + (("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                        ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
                        ("Access-Control-Max-Age", "600"), ("Content-Length", "0")):
        handler.send_header(k, v)
    handler.end_headers()


def _off(handler) -> None:
    handler._json(404, Refused(404, "not_found", "the Laya route is off (the deciders "
                                                 "document's api.mode)").body(), extra_headers=CORS)


def handle_get(handler, reg, state: dict) -> None:
    """`GET /v1/laya`."""
    cfg, model = _cfg(state)
    if cfg["mode"] != "on":
        return _off(handler)
    handler._json(200, card(reg, cfg, model), extra_headers=CORS)


def _drive_rows(state: dict, rows: list[dict], summary: dict | None = None) -> None:
    """The session rows (`kind: laya_drive`)."""
    cond = state.get("conductor")
    for r in rows:
        if summary:
            r["summary"] = summary
        if cond is not None:
            cond.journal.note("laya_drive", "session", decision=r["ended"], detail=r)


def handle_post(handler, reg, state: dict, body) -> None:
    """`POST /v1/laya/ask`: the answer, the journal row (a driver session's: when it ends)."""
    cfg, model = _cfg(state)
    if cfg["mode"] != "on":
        return _off(handler)
    _drive_rows(state, DRIVE.sweep(cfg))
    out, err = None, None
    try:
        out = ask(reg, body, cfg, model)
    except Refused as exc:
        err = exc
    _count(out, err)
    sid = body.get("session") if isinstance(body, dict) else None
    cond = state.get("conductor")
    # The conversation the address's token names (server/session_tokens.py)
    sess, tagged = getattr(handler, "_sess", None), getattr(handler, "_stok", None) is not None
    if tagged and state.get("session_tokens") is not None:
        state["session_tokens"].note(sess, "laya")
    if isinstance(sid, str) and SESSION_RE.match(sid):
        DRIVE.record(sid, body, out, err)
    elif cond is not None:
        decision, detail = journal_detail(out, err, body)
        if tagged:
            detail["session_key"] = sess
        try:
            from server import ledger as ledger_mod
            detail["client"] = ledger_mod.client_of(handler.headers)[1]
        except Exception:                                          # noqa: BLE001
            pass
        cond.journal.note("laya_api", "ask", decision=decision, detail=detail,
                          **({"session_key": sess} if sess is not None else {}))
    if err is not None:
        return handler._json(err.status, err.body(), extra_headers=CORS + (
            (("Retry-After", str(err.retry)),) if err.retry else ()))
    out.pop("_req", None)
    handler._json(200, out, extra_headers=CORS)


def handle_drive_end(handler, state: dict, body) -> None:
    """`POST /v1/laya/drive/end {session, summary?}`: the session's row, and its numbers back."""
    cfg, _ = _cfg(state)
    if cfg["mode"] != "on":
        return _off(handler)
    _drive_rows(state, DRIVE.sweep(cfg))
    sid = body.get("session") if isinstance(body, dict) else None
    if not isinstance(sid, str) or not SESSION_RE.match(sid):
        return handler._json(422, Refused(422, "invalid_request_error", "session: the driver "
                                          "session's name", "session").body(), extra_headers=CORS)
    row = DRIVE.end(sid)
    if row is None:
        return handler._json(404, Refused(404, "not_found", f"no live driver session {sid!r} "
                                          f"(ended, or idle past {cfg['drive_idle_s']} s)",
                                          "session").body(), extra_headers=CORS)
    _drive_rows(state, [row], summary_of(body))
    handler._json(200, row, extra_headers=CORS)


def handle_kit(handler, state: dict, name: str) -> None:
    """`GET /v1/laya/kit/<file>`: a file of the playtest kit, as text."""
    cfg, _ = _cfg(state)
    if cfg["mode"] != "on":
        return _off(handler)
    if name not in KIT_FILES:
        return handler._json(404, Refused(404, "not_found", f"the kit has {', '.join(KIT_FILES)}")
                             .body(), extra_headers=CORS)
    with open(os.path.join(KIT_DIR, name), "rb") as f:
        data = f.read()
    handler.send_response(200)
    for k, v in CORS + (("Content-Type", KIT_FILES[name]), ("Content-Length", str(len(data))),
                        ("Cache-Control", "no-cache")):
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)
