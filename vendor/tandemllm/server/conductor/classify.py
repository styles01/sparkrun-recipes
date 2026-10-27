"""The two models that answer "which of these is it": a decider and Qwen itself.

Both answer a `choice` over options `{id: description}` and return an `Answer`; the router's
cascade (`router.py`) does not care which one spoke. Neither is ever awaited past its timeout,
and every failure comes back as an `Answer` with `error` set, never as an exception, so the
cascade fails open by construction.

THE DECIDER (role `decider`) is an interface, not a service: anything with
`ask(state, questions) -> {qid: answer}` in the Jev shape (questions `choice`, `noul`, `score`;
answers with `choice`/`noul`/`score`, `probabilities` and `confidence`), plus `limits()` and
`health()`. `Decider.choice()` builds one `choice` question on top of it. Two backends:
  * `InProcess`: the decision models this engine serves itself (engine/decision, SRV-131), asked
    through the registry with no HTTP hop. The server registers it with `register_backend` as
    `self` and as `engine:<family>` (`engine:laya`) when `QWEN38_DECIDER_MODELS` names models; a
    server without them registers nothing, and the stage answers `decider_no_endpoint`;
  * `JevHttp`: the Jev contract over HTTP, `POST {endpoint}/v1/systemone` and `GET /v1/models` for
    the limits, for a decider on another box (an engine's `decision_models`, or arbiter).
The model per question id is the conductor's `deciders` document; the router resolves it into the
role's `model` before it asks (`router.py`), and names the question in `ASKING` for the journal of
picks the Deciders screen shows.
Model-agnostic either way: the limits come from the backend (the fallbacks 512/192/20 when it
does not say), the digest is sized to `context_tokens - question_tokens` at 3.5 characters a
token, and no checkpoint name or model limit is written here.

QWEN (role `classifier`): the engine's own `/v1/chat/completions` with `X-Tandem: raw` (the model
name is the role's; the raw engine's old name is a hidden alias of `TandemLLM`, so the header is
what keeps the call raw), thinking off (`enable_thinking: false`), greedy, and
`structured_outputs: {"choice": [ids]}`, so the grammar lets it write only an option id. The
confidence is the probability of the path it wrote, `exp(sum(logprob))` over the answer's tokens,
under the constrained distribution. It is a short request unrelated to the conversation, so the
engine treats it as a guest: the rows it overwrites are stashed and restored (docs/caches.md).
`endpoint: self` means this server's own port. It must be called before the calling request takes
the engine lock, or the two would wait on each other. (The in-process decider has the same rule
for the same reason: it asks in the idle window, the time no completion holds the engine lock;
a call from under the lock ends at its deadline as `decider_deadline_missed`.)
"""

from __future__ import annotations

import contextvars
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_LIMITS = {"context_tokens": 512, "question_tokens": 192, "max_options": 20}
CHARS_PER_TOKEN = 3.5
LIMITS_TTL_S = 600.0
USER_AGENT = "tandem-conductor/1"
# the question id being asked (classify, packages, ...), set by the router around a cascade
ASKING: contextvars.ContextVar = contextvars.ContextVar("asking", default=None)
# VIS-60: the request the router is planning; its question to the engine names it in
# `X-Tandem-Parent`, so the ledger row of the question points at the turn it served
PARENT: contextvars.ContextVar = contextvars.ContextVar("parent", default=None)


class Answer:
    __slots__ = ("label", "confidence", "probs", "ms", "error", "model", "stage", "guard")

    def __init__(self, stage: str, label=None, confidence=0.0, probs=None, ms=0.0, error=None,
                 model=None):
        self.stage, self.label, self.confidence = stage, label, float(confidence or 0.0)
        self.probs = probs or {}
        self.ms, self.error, self.model = round(float(ms), 2), error, model
        self.guard = None                # SRV-189: the code guard's change (router._code_guard)

    def as_dict(self) -> dict:
        return {"stage": self.stage, "label": self.label, "confidence": round(self.confidence, 4),
                "probs": {k: round(float(v), 4) for k, v in self.probs.items()}, "ms": self.ms,
                "error": self.error, "model": self.model}


def _post(url: str, body: dict, timeout_s: float, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    parent = PARENT.get()
    if parent:
        req.add_header("X-Tandem-Parent", str(parent)[:80])
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read() or b"{}")


def _get(url: str, timeout_s: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read() or b"{}")


def _why(exc: BaseException) -> str:
    """A failure reason for the journal: timeout, unreachable or http_<code>."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    why = getattr(exc, "why", None)          # the in-process runtime's own reason
    if isinstance(why, str):
        return why
    reason = getattr(exc, "reason", exc)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError) or \
            "timed out" in str(reason):
        return "timeout"
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return "unreachable"
    return "error"


class Decider:
    """A typed-decision model behind the Jev question shape. Subclasses give `ask`, and may give
    `limits` and `health`; everything the conductor asks goes through `choice` (and later `noul`)."""

    name = "decider"

    def ask(self, state: str, questions: dict) -> dict:
        raise NotImplementedError

    def limits(self) -> dict:
        return dict(DEFAULT_LIMITS)

    def health(self) -> dict:
        return {"up": True, "ms": 0.0}

    def state_chars(self) -> int:
        lim = self.limits()
        return max(200, int((lim["context_tokens"] - lim["question_tokens"]) * CHARS_PER_TOKEN))

    def choice(self, state: str, question: str, options: dict[str, str]) -> Answer:
        """One `choice` question: the option it picks, at the probability it gives that option."""
        t0 = time.perf_counter()
        lim = self.limits()
        opts = dict(list(options.items())[:lim["max_options"]])
        try:
            got = self.ask(state[:self.state_chars()], {"q": {
                "type": "choice", "instructions": question, "criteria": opts}})
            a = (got.get("answers") or {}).get("q") or {}
            probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
            label = a.get("choice")
            conf = probs.get(label, a.get("confidence", 0.0)) if label else 0.0
            return Answer("decider", label, conf, probs, (time.perf_counter() - t0) * 1e3,
                          model=got.get("model"))
        except Exception as exc:                                   # noqa: BLE001
            return Answer("decider", ms=(time.perf_counter() - t0) * 1e3,
                          error=f"decider_{_why(exc)}")


class JevHttp(Decider):
    """The Jev contract over HTTP (the arbiter sidecar today). Safe to share between threads."""

    def __init__(self, endpoint: str, model: str = "auto", timeout_ms: int = 400,
                 api_key: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model, self.timeout_ms, self.api_key = model or "auto", int(timeout_ms), api_key
        self._limits: dict | None = None
        self._limits_at = 0.0
        self._lock = threading.Lock()

    def limits(self) -> dict:
        """The limits of the model `self.model` resolves to; the fallbacks when not reported."""
        now = time.monotonic()
        if self._limits is not None and now - self._limits_at < LIMITS_TTL_S:
            return self._limits
        with self._lock:
            got = dict(DEFAULT_LIMITS)
            try:
                data = _get(self.endpoint + "/v1/models", max(0.2, self.timeout_ms / 1000))
                want = "jev-latest" if self.model in ("auto", "default", "jev", "") else self.model
                # an engine lists its decision models beside its chat model; arbiter in `data`
                for m in (data.get("data") or []) + (data.get("decision_models") or []):
                    if m.get("id") == want or want in (m.get("aliases") or []):
                        for k, v in (m.get("limits") or {}).items():
                            if k in got and isinstance(v, int) and v > 0:
                                got[k] = v
            except Exception:                                      # noqa: BLE001
                pass
            self._limits, self._limits_at = got, now
            return got

    def ask(self, state: str, questions: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        return _post(self.endpoint + "/v1/systemone",
                     {"state": state, "model": self.model, "questions": questions},
                     self.timeout_ms / 1000, headers)

    def health(self) -> dict:
        t0 = time.perf_counter()
        try:
            r = _get(self.endpoint + "/readyz", max(0.3, self.timeout_ms / 1000))
            return {"up": r.get("status") == "ready",
                    "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "detail": {k: r.get(k) for k in ("engine", "models", "device", "version")},
                    "limits": self.limits()}
        except Exception as exc:                                   # noqa: BLE001
            return {"up": False, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "error": _why(exc)}


class InProcess(Decider):
    """The engine's own decision models (engine/decision/registry.py): one `ask` in the idle
    window with the stage's timeout as the deadline. The registry is duck-typed here, so the
    conductor stays standard-library only."""

    name = "in-process"

    def __init__(self, registry, model: str = "auto", timeout_ms: int = 400):
        self.registry, self.model, self.timeout_ms = registry, model or "auto", int(timeout_ms)

    def ask(self, state: str, questions: dict) -> dict:
        fut = self.registry.ask(state, questions, model=self.model, window="idle",
                                deadline_ms=self.timeout_ms, source=ASKING.get())
        return fut.result(timeout=self.timeout_ms / 1000.0)

    def limits(self) -> dict:
        got = dict(DEFAULT_LIMITS)
        try:
            for k, v in (self.registry.limits(self.model) or {}).items():
                if k in got and isinstance(v, int) and v > 0:
                    got[k] = v
        except Exception:                                          # noqa: BLE001
            pass
        return got

    def health(self) -> dict:
        loaded = self.registry.loaded()
        return {"up": bool(loaded), "ms": 0.0,
                "detail": {"engine": "in-process", "models": loaded,
                           "device": str(self.registry.device.device)
                           if self.registry.device is not None else None},
                "limits": self.limits(),
                **({} if loaded else {"error": "no decision model loaded"})}


# in-process deciders: name -> factory(model, timeout_ms) -> Decider
BACKENDS: dict = {}


def register_backend(name: str, factory) -> None:
    """A decider the server itself provides, reachable as role endpoint `engine:<name>`."""
    BACKENDS[name] = factory


def decider_for(endpoint: str, model: str, timeout_ms: int) -> Decider | None:
    """The decider a role's endpoint names: `self` or `engine:<name>` an in-process backend, an
    http(s) URL the Jev contract over HTTP; None when nothing answers to it."""
    if endpoint == "self":
        f = BACKENDS.get("self")
        return f(model, timeout_ms) if f is not None else None
    if endpoint.startswith("engine:"):
        f = BACKENDS.get(endpoint[7:])
        return f(model, timeout_ms) if f is not None else None
    if endpoint.startswith(("http://", "https://")):
        return JevHttp(endpoint, model or "auto", timeout_ms)
    return None


def distribution(toks: list, ids: list[str], label: str, conf: float) -> dict[str, float]:
    """SRV-189: every option's probability from the answer's first token and its alternatives
    (`top_logprobs`, under the grammar only an option's prefix can come first): an alternative
    goes to the options its text starts, shared evenly when it starts several (`d` of `debug` and
    `design_ui`); the chosen option keeps the probability of its whole path. Before, the stage
    returned the chosen option alone, so 0.57 for code_edit hid chat at 0.29 and writing at
    0.10 (the UT2 essay, 2026-10-06)."""
    out = {label: conf}
    first = toks[0] if toks else {}
    alts = first.get("top_logprobs") or []
    chosen = str(first.get("token") or "")
    for a in alts:
        t = str(a.get("token") or "").strip()
        if not t or t == chosen.strip():
            continue
        hit = [o for o in ids if o.startswith(t) and o != label]
        if not hit:
            continue
        p = math.exp(float(a.get("logprob", -1e9))) / len(hit)
        for o in hit:
            out[o] = out.get(o, 0.0) + p
    return out


class Qwen:
    """The classifier role: the engine itself on the raw model name, thinking off."""

    def __init__(self, endpoint: str, model: str, timeout_ms: int = 8000):
        self.endpoint = endpoint.rstrip("/")
        self.model, self.timeout_ms = model, int(timeout_ms)

    def choice(self, state: str, prompt: str, options: dict[str, str],
               max_chars: int = 2000) -> Answer:
        t0 = time.perf_counter()
        ids = list(options)
        body = {"model": self.model, "stream": False, "temperature": 0, "max_tokens": 24,
                "messages": [{"role": "system", "content": prompt},
                             {"role": "user", "content": state[:max_chars]}],
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"choice": ids},
                # SRV-189: the first token's alternatives give every class its probability
                "logprobs": True, "top_logprobs": min(20, max(3, len(ids)))}
        try:
            r = _post(self.endpoint + "/v1/chat/completions", body, self.timeout_ms / 1000,
                      {"X-Tandem": "raw"})
            ch = (r.get("choices") or [{}])[0]
            label = str((ch.get("message") or {}).get("content") or "").strip()
            toks = ((ch.get("logprobs") or {}).get("content")) or []
            if toks:
                conf = math.exp(sum(float(t.get("logprob", 0.0)) for t in toks))
            else:
                conf = 0.5
            if label not in ids:
                return Answer("qwen", None, 0.0, {}, (time.perf_counter() - t0) * 1e3,
                              error="qwen_unknown_label", model=r.get("model"))
            probs = distribution(toks, ids, label, conf)
            return Answer("qwen", label, conf, probs, (time.perf_counter() - t0) * 1e3,
                          model=r.get("model"))
        except Exception as exc:                                   # noqa: BLE001
            return Answer("qwen", ms=(time.perf_counter() - t0) * 1e3,
                          error=f"qwen_{_why(exc)}")

    def health(self) -> dict:
        t0 = time.perf_counter()
        try:
            r = _get(self.endpoint + "/v1/models", 2.0)
            names = [m.get("id") for m in r.get("data") or []]
            up = self.model in names
            if not up and "TandemLLM" in names:
                # one model since 2026-10-01: the raw name is a hidden alias of TandemLLM, which the
                # list names alone; the call itself goes out with `X-Tandem: raw`
                try:
                    up = _get(self.endpoint + "/v1/models/" + urllib.parse.quote(self.model, safe=""),
                              2.0).get("id") == "TandemLLM"
                except Exception:                                  # noqa: BLE001
                    up = False
            return {"up": up, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "detail": {"models": names}}
        except Exception as exc:                                   # noqa: BLE001
            return {"up": False, "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "error": _why(exc)}
