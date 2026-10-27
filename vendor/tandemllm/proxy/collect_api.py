"""Layer L2 (P6): per-request internals from the plugins inside vLLM and SGLang, joined to the ledger.

    POST /v1/collect/<backend>     Authorization: Bearer <collect token>
        a batch from tandemllm-vllm-plugin or tandemllm-sglang-plugin (plugins/*/_push.py):
        {"plugin", "plugin_version", "engine", "engine_version", "instance", "pid", "sent_at",
         "dropped", "requests": [...], "steps": [...]}
        -> 200 {"accepted": {"requests": n, "steps": n}, "joined": n}

THE TOKEN. One shared token per proxy: `TANDEM_COLLECT_TOKEN` when set, else a random one made at
the first need and kept in the config database's settings (`collect_token`); `tandem-dash launch`
hands the same one to the engine. Every batch needs it, from loopback too; a wrong one is a 401.
It only lets its holder post numbers for a backend that exists.

THE JOIN. The relay tags each request it sends to a `vllm` or `sglang` backend with an id the
engine reports back: vLLM takes `X-Request-Id: px-<hex>` and reports `chatcmpl-px-<hex>` (the
proxy's own id for the row) on the chat route, `cmpl-px-<hex>` on completions; SGLang takes the
body's `rid` and reports it unchanged. A client's own `X-Request-Id` or `rid` is left alone and
becomes the key. When the plugin's item and the row meet (either may come first: the row is
written when the proxy's stream ends, the item within a second of the engine's last step), the
row gets the engine's numbers:

    queue_ms, prompt_ms (prefill), decode_ms   the engine's own split
    decode_tps        (completion tokens - 1) over the engine's decode time
    prefill_tps       uncached prompt tokens over the engine's prefill time
    cached_tokens     the engine's own count
    preemptions, engine_finish, corrupted       new columns (ledger schema 4)
    blocks, draft_tokens, draft_accepted        SGLang's verify passes and drafts, per request
    l2_join           `exact`, or `approximate` (below)

A field the item leaves null leaves the row's value as it was. An item that arrives before its row
waits for it; a row without its item, and an item no row asked for, wait 10 minutes and are then
dropped and counted. An item whose id matches no tag (a client's id the engine changed, a route
the proxy cannot tag such as the Responses API) is matched to an untagged row of the same backend
that finished within 3 s of it with the same prompt and completion token counts, when exactly one
such row exists; the row says `l2_join: approximate` and the log has one line for it.

THE BACKEND'S STATE. The scheduler samples of the last five minutes are kept per backend for the
backend's view (KV use, running and waiting, prefix hits, evictions, preemptions, speculation per
draft position). A backend whose plugin has posted nothing for 60 s reads "L1 only"; the view says
which plugin and engine versions it last saw. An idle plugin posts an empty batch every 10 s.

PULL. A vLLM with `tandem_stats_endpoint` loaded also serves `GET /tandem/stats`; when no batch
has been pushed for 25 s the proxy asks that route every 2 s instead (404: asked again after
120 s). Items carry sequence numbers per plugin instance, so a batch seen twice counts once.
"""

from __future__ import annotations

import collections
import hmac
import json
import secrets
import threading
import time
import urllib.error
import urllib.request

L2_KINDS = ("vllm", "sglang")
TAGGED_PATHS = ("/v1/chat/completions", "/v1/completions")
WAIT_S = 600.0                     # a row or an item waits this long for the other
EXPECT_MAX_S = 3600.0              # a tag whose request never wrote a row
APPROX_AFTER_S = 5.0               # a tagged row becomes a candidate for an approximate match
APPROX_WINDOW_S = 3.0
L1_AFTER_S = 60.0                  # no batch for this long: "L1 only"
STEPS_KEEP = 300
RECENT_KEEP = 1000                 # the last request items per backend (the night checks' sums)
PULL_AFTER_S = 25.0
PULL_EVERY_S = 2.0
PULL_OFF_S = 120.0
PULL_TIMEOUT_S = 2.0
BODY_MAX = 8 * 1024 * 1024
ITEMS_MAX = 20_000                 # items and rows waiting, per proxy
INSTANCE_TTL_S = 3600.0            # an instance id carries the pid: every engine restart
                                   # added a row no refresh could ever stale out before

# the item's fields and the ledger columns they fill
NUMERIC = {"queue_ms": "queue_ms", "prefill_ms": "prompt_ms", "decode_ms": "decode_ms",
           "cached_tokens": "cached_tokens", "preemptions": "preemptions",
           "spec_verify_ct": "blocks", "spec_draft_tokens": "draft_tokens",
           "spec_accepted": "draft_accepted"}
AMEND_COLUMNS = ("queue_ms", "prompt_ms", "decode_ms", "decode_tps", "prefill_tps",
                 "cached_tokens", "preemptions", "engine_finish", "corrupted", "blocks",
                 "draft_tokens", "draft_accepted", "l2_join")


class CollectError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.kind}}


def token_of(cfg, env) -> str:
    """The proxy's collect token: the environment's, else the stored one, else a new stored one."""
    t = (env.get("TANDEM_COLLECT_TOKEN") or "").strip()
    if t:
        return t
    t = cfg.setting("collect_token")
    if isinstance(t, str) and len(t) >= 16:
        return t
    t = secrets.token_urlsafe(32)
    cfg.set_setting("collect_token", t, who="collect")
    return t


def upstream_key(kind: str, path: str, value: str) -> str:
    """The id the engine reports for a request tagged with `value`."""
    if kind == "vllm":
        return ("chatcmpl-" if path.endswith("/chat/completions") else "cmpl-") + value
    return value


def fields_of(item: dict, prompt_tokens=None, completion_tokens=None) -> dict:
    """The ledger columns an item fills; only what it measured."""
    out = {}
    for k, col in NUMERIC.items():
        v = item.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[col] = v
    if isinstance(item.get("finish_reason"), str):
        out["engine_finish"] = item["finish_reason"][:32]
    if isinstance(item.get("corrupted"), bool):
        out["corrupted"] = 1 if item["corrupted"] else 0
    n_out = item.get("completion_tokens")
    if not isinstance(n_out, int):
        n_out = completion_tokens
    d = out.get("decode_ms")
    if isinstance(n_out, int) and n_out > 1 and d and d > 0:
        out["decode_tps"] = round((n_out - 1) / (d / 1e3), 2)
    n_in = item.get("prompt_tokens")
    if not isinstance(n_in, int):
        n_in = prompt_tokens
    p = out.get("prompt_ms")
    if isinstance(n_in, int) and p and p > 0:
        cached = out.get("cached_tokens") or 0
        if n_in - cached > 0:
            out["prefill_tps"] = round((n_in - cached) / (p / 1e3), 2)
    return out


class _Backend:
    def __init__(self):
        self.last_batch = None          # monotonic time of the last batch, pushed or pulled
        self.plugin = self.plugin_version = self.engine = self.engine_version = None
        self.instances: dict = {}       # instance -> {"requests": seq, "steps": seq}
        self.inst_t: dict = {}          # instance -> last batch seen for it (eviction uses it)
        self.was_configured = False     # the pull round has seen this id in the config
        self.steps: collections.deque = collections.deque(maxlen=STEPS_KEEP)
        self.recent: collections.deque = collections.deque(maxlen=RECENT_KEEP)   # checks, sums
        self.counts = collections.Counter()
        self.dropped = {"requests": 0, "steps": 0}
        self.pull = {"state": "unknown", "next": 0.0, "via": None}


class Collector:
    def __init__(self, service, *, env=None, clock=time.monotonic, wall=time.time):
        import os
        self.svc = service
        if env is None:                     # the environment the config was read with
            env = getattr(getattr(service, "cfg", None), "env", None)
        self.env = os.environ if env is None else env
        self.clock, self.wall = clock, wall
        self._token: str | None = None
        self._token_lock = threading.Lock()   # not _lock: token_of may write the config,
                                              # and the collector lock is the hot path
        self._lock = threading.Lock()
        self.backends: dict[str, _Backend] = {}
        self.expect: dict = {}          # (bid, key) -> rec, while the request runs
        self.done: dict = {}            # (bid, key or row id) -> row info, after the row
        # (bid, prompt_tokens, completion_tokens) -> done keys, so an untagged item finds its
        # candidate rows without scanning every row: tick() tried each loose item against
        # every done row — both near ITEMS_MAX that is the ingest lock held for seconds
        self.done_by: dict = {}
        self.items: dict = {}           # (bid, key) -> (item, t), an item before its row
        self.loose: dict = {}           # (bid, n) -> (item, t), an item with no tag at all
        self._n = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log = getattr(service, "log", None) or (lambda m: print(m, flush=True))

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> "Collector":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="proxy-collect", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception as exc:                               # noqa: BLE001
                self.log(f"[collect] tick failed: {type(exc).__name__}: {exc}")

    def token(self) -> str:
        if self._token is None:
            # two first requests used to race here: each generated a different token and
            # each wrote it to the config; whoever lost the write is now refused forever,
            # because self._token holds the winner and is never re-read
            with self._token_lock:
                if self._token is None:
                    self._token = token_of(self.svc.cfg, self.env)
        return self._token

    def _b(self, bid: str) -> _Backend:
        b = self.backends.get(bid)
        if b is None:
            b = self.backends[bid] = _Backend()
        return b

    # ----------------------------------------------------------------- the relay's side
    def tag(self, backend: dict, path: str, body: dict, up_body: bytes, headers: list, rec
            ) -> tuple[bytes, list]:
        """Give an upstream request to a vLLM or SGLang backend an id its plugin will report.
        Returns the body and the headers to send."""
        kind = backend.get("kind")
        if kind not in L2_KINDS or path not in TAGGED_PATHS:
            return up_body, headers
        if kind == "vllm":
            have = next((v for k, v in headers if k.lower() == "x-request-id"), None)
            value = have or (rec.request_id[len("chatcmpl-"):]
                             if rec.request_id.startswith("chatcmpl-") else rec.request_id)
            if have is None:
                headers = headers + [("X-Request-Id", value)]
        else:
            n = body.get("n")
            if isinstance(n, int) and n > 1:          # SGLang makes `<rid>_<i>` of it: no tag
                return up_body, headers
            if body.get("rid") is not None and not isinstance(body.get("rid"), str):
                return up_body, headers
            have = body.get("rid")
            value = have or rec.request_id
            if have is None:
                try:
                    obj = json.loads(up_body)
                    if isinstance(obj, dict):
                        obj["rid"] = value
                        up_body = json.dumps(obj, ensure_ascii=False).encode()
                except ValueError:
                    return up_body, headers
        key = upstream_key(kind, path, value)
        rec.l2_key = key
        with self._lock:
            self.expect[(backend["id"], key)] = (rec, self.clock())
        return up_body, headers

    def on_row(self, rec, row: dict) -> None:
        """Called with the row before it goes to the ledger: fill it when the item came first,
        else remember the row for the item."""
        if rec.backend_kind not in L2_KINDS or not rec.backend:
            return
        bid = rec.backend
        key = getattr(rec, "l2_key", None)
        with self._lock:
            if key is not None:
                self.expect.pop((bid, key), None)
                got = self.items.pop((bid, key), None)
                if got is not None:
                    row.update(fields_of(got[0], row.get("prompt_tokens"),
                                         row.get("completion_tokens")))
                    row["l2_join"] = "exact"
                    self._b(bid).counts["exact"] += 1
                    return
            if len(self.done) >= ITEMS_MAX:
                self._expire(force=True)
            dk = (bid, key or f"row:{rec.request_id}")
            self._done_pop(dk)            # a re-used key must leave its old (p, c) bucket first
            self.done[dk] = {
                "request_id": rec.request_id, "ts_ms": row.get("ts_ms"), "t": self.clock(),
                "end": self.wall(), "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"), "tagged": key is not None}
            self.done_by.setdefault((bid, row.get("prompt_tokens"),
                                     row.get("completion_tokens")), set()).add(dk)

    def _done_pop(self, dk):
        """Take a row out of `done` and out of the (bid, p, c) index with it."""
        row = self.done.pop(dk, None)
        if row is not None:
            sig = (dk[0], row["prompt_tokens"], row["completion_tokens"])
            s = self.done_by.get(sig)
            if s is not None:
                s.discard(dk)
                if not s:
                    self.done_by.pop(sig, None)
        return row

    # ----------------------------------------------------------------- the plugin's side
    def check_token(self, authorization: str | None) -> bool:
        got = (authorization or "").strip()
        got = got[7:].strip() if got[:7].lower() == "bearer " else ""
        return bool(got) and hmac.compare_digest(got.encode(), self.token().encode())

    def ingest(self, bid: str, batch: dict, via: str = "push") -> dict:
        if not isinstance(batch, dict):
            raise CollectError(400, "the batch must be a JSON object")
        reqs = batch.get("requests") or []
        steps = batch.get("steps") or []
        if not isinstance(reqs, list) or not isinstance(steps, list):
            raise CollectError(400, "requests and steps must be lists")
        inst = str(batch.get("instance") or batch.get("pid") or "?")[:40]
        joined = 0
        amends = []
        with self._lock:
            b = self._b(bid)
            b.last_batch = self.clock()
            for k in ("plugin", "plugin_version", "engine", "engine_version"):
                v = batch.get(k)
                if isinstance(v, str) and v:
                    setattr(b, k, v[:64])
            if via == "push":
                b.pull["via"] = "push"
            cur = b.instances.setdefault(inst, {"requests": 0, "steps": 0})
            b.inst_t[inst] = self.clock()
            dr = batch.get("dropped") if isinstance(batch.get("dropped"), dict) else {}
            for k in ("requests", "steps"):
                if isinstance(dr.get(k), int):
                    b.dropped[k] = max(b.dropped[k], dr[k])
            n_req = n_step = 0
            for it in reqs:
                if not isinstance(it, dict):
                    continue
                seq = it.get("seq")
                if isinstance(seq, int):
                    if seq <= cur["requests"]:
                        continue
                    cur["requests"] = seq
                n_req += 1
                b.recent.append(it)
                got = self._take(bid, b, it)
                if got is not None:
                    amends.append(got)
                    joined += 1
            for it in steps:
                if not isinstance(it, dict):
                    continue
                seq = it.get("seq")
                if isinstance(seq, int):
                    if seq <= cur["steps"]:
                        continue
                    cur["steps"] = seq
                n_step += 1
                it["_rx"] = self.wall()
                b.steps.append(it)
            b.counts["batches"] += 1
        led = getattr(self.svc, "ledger", None)
        for request_id, ts_ms, fields in amends:
            if led is not None and hasattr(led, "amend"):
                led.amend(request_id, fields, since_ms=(ts_ms or 0) - 60_000)
        return {"accepted": {"requests": n_req, "steps": n_step}, "joined": joined}

    def _take(self, bid: str, b: _Backend, it: dict):
        """One request item: an amend for a row already written, or kept for later."""
        key = it.get("id") if isinstance(it.get("id"), str) else None
        now = self.clock()
        if key is not None:
            row = self._done_pop((bid, key))
            if row is not None:
                b.counts["exact"] += 1
                f = fields_of(it, row["prompt_tokens"], row["completion_tokens"])
                f["l2_join"] = "exact"
                return row["request_id"], row["ts_ms"], f
            if (bid, key) in self.expect:
                self.items[(bid, key)] = (it, now)
                return None
        row_key = self._approx(bid, it)
        if row_key is not None:
            row = self._done_pop(row_key)
            b.counts["approximate"] += 1
            f = fields_of(it, row["prompt_tokens"], row["completion_tokens"])
            f["l2_join"] = "approximate"
            self.log(f"[collect] join approximate: {row['request_id']} <- {bid} engine id "
                     f"{key or '?'} (same token counts, {APPROX_WINDOW_S:.0f} s window)")
            return row["request_id"], row["ts_ms"], f
        if len(self.loose) >= ITEMS_MAX:
            self._expire(force=True)
        self._n += 1
        self.loose[(bid, self._n)] = (it, now)
        return None

    def _approx(self, bid: str, it: dict):
        """The one row an untagged item matches, or None."""
        p, c, t = it.get("prompt_tokens"), it.get("completion_tokens"), it.get("t_end")
        if not isinstance(p, int) or not isinstance(c, int) or not isinstance(t, (int, float)):
            return None
        now = self.clock()
        found = None
        for k in self.done_by.get((bid, p, c), ()):
            row = self.done.get(k)
            if row is None:
                continue
            if row["prompt_tokens"] != p or row["completion_tokens"] != c:
                continue                  # the scan's token equality, kept on the indexed path
            if row["tagged"] and now - row["t"] < APPROX_AFTER_S:
                continue
            if abs(row["end"] - t) > APPROX_WINDOW_S:
                continue
            if found is not None:
                return None                       # two candidates: no guess
            found = k
        return found

    # ----------------------------------------------------------------- housekeeping
    def tick(self) -> None:
        amends = []
        with self._lock:
            # untagged items that waited: try the rows that became candidates since
            for k, (it, t) in list(self.loose.items()):
                rk = self._approx(k[0], it)
                if rk is not None:
                    row = self._done_pop(rk)
                    self.loose.pop(k)
                    b = self._b(k[0])
                    b.counts["approximate"] += 1
                    f = fields_of(it, row["prompt_tokens"], row["completion_tokens"])
                    f["l2_join"] = "approximate"
                    self.log(f"[collect] join approximate: {row['request_id']} <- {k[0]} engine "
                             f"id {it.get('id') or '?'}")
                    amends.append((row["request_id"], row["ts_ms"], f))
            self._expire()
        led = getattr(self.svc, "ledger", None)
        for request_id, ts_ms, fields in amends:
            if led is not None and hasattr(led, "amend"):
                led.amend(request_id, fields, since_ms=(ts_ms or 0) - 60_000)
        self._pull_round()

    def _expire(self, force: bool = False) -> None:
        now = self.clock()
        limit = 60.0 if force else WAIT_S
        for name, store, idx in (("row_unmatched", self.done, None),
                                 ("item_unmatched", self.items, 1),
                                 ("item_untagged", self.loose, 1)):
            for k, v in list(store.items()):
                t = v["t"] if idx is None else v[idx]
                if now - t > (limit if name != "item_untagged" else min(limit, 60.0)):
                    if idx is None:                    # done rows leave through the index
                        self._done_pop(k)
                        self._b(k[0]).counts[name] += 1
                        continue
                    store.pop(k, None)
                    self._b(k[0]).counts[name] += 1
        for k, (_rec, t) in list(self.expect.items()):
            if now - t > EXPECT_MAX_S:            # a request that never wrote its row
                self.expect.pop(k, None)
        for b in self.backends.values():          # restarts left the old pid's row behind
            for inst, t in list(b.inst_t.items()):
                if now - t > INSTANCE_TTL_S:
                    b.inst_t.pop(inst, None)
                    b.instances.pop(inst, None)

    # ----------------------------------------------------------------- pull (vLLM)
    def _pull_round(self) -> None:
        try:
            configured = self.svc.cfg.backends()
        except Exception:                                          # noqa: BLE001
            return
        # a backend deleted from the config had a card here forever. Only a card that this
        # round actually configured leaves: a card that exists purely because traffic named
        # it stays counted (ingest of an unknown backend id keeps its counters as before)
        alive = {b.get("id") for b in configured}
        with self._lock:
            for bid in list(self.backends):
                if bid not in alive and self.backends[bid].was_configured:
                    self.backends.pop(bid, None)
                    # the row keys of a gone card go with it, so a late expire has
                    # nothing to count onto a resurrected card
                    for store in (self.done, self.items, self.loose, self.expect):
                        for key in [x for x in store if x[0] == bid]:
                            store.pop(key, None)
        for bk in configured:
            if bk.get("id") in self.backends:
                self._b(bk["id"]).was_configured = True
        backends = [b for b in configured if b["kind"] == "vllm" and b["enabled"]]
        now = self.clock()
        for bk in backends:
            with self._lock:
                st = self._b(bk["id"])
                pushed = st.last_batch is not None and st.pull["via"] == "push" and \
                    now - st.last_batch < PULL_AFTER_S
                if pushed or now < st.pull["next"]:
                    continue
                st.pull["next"] = now + PULL_EVERY_S
                cur = {"requests": 0, "steps": 0}
                if st.instances:
                    last = max(st.instances.items(), key=lambda kv: kv[1]["requests"])[1]
                    cur = dict(last)
            try:
                self._pull(bk, st, cur)
            except Exception as exc:                                   # noqa: BLE001
                # a sample that fails outside `_pull`'s own net (a bad url, an unexpected
                # ingest error) must not cost the other backends their round; the failing
                # one keeps its own error state and the PULL_OFF_S backoff between tries
                if st.pull["state"] != "error":
                    self.log(f"[collect] pull {bk['id']} failed: {type(exc).__name__}: {exc}")
                st.pull.update(state="error", next=self.clock() + PULL_OFF_S)

    def _pull(self, bk: dict, st: _Backend, cur: dict) -> None:
        from proxy import engines
        url = (engines.root_of(bk["url"]) + f"/tandem/stats?after_requests={cur['requests']}"
               f"&after_steps={cur['steps']}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token()}"})
        try:
            with urllib.request.urlopen(req, timeout=PULL_TIMEOUT_S) as r:
                data = r.read(BODY_MAX + 1)
            batch = json.loads(data)
        except urllib.error.HTTPError as e:
            st.pull.update(state="off" if e.code in (401, 403, 404) else "error",
                           next=self.clock() + PULL_OFF_S)
            return
        except (OSError, ValueError):
            st.pull.update(state="error", next=self.clock() + PULL_OFF_S)
            return
        st.pull.update(state="on", via="pull")
        try:
            self.ingest(bk["id"], batch, via="pull")
        except CollectError:
            pass

    # ----------------------------------------------------------------- the views
    def age(self, bid: str) -> float | None:
        b = self.backends.get(bid)
        if b is None or b.last_batch is None:
            return None
        return round(self.clock() - b.last_batch, 1)

    def view(self, bid: str) -> dict | None:
        """The backend's `plugin` block (contract 1.3, optional), or None before any batch."""
        b = self.backends.get(bid)
        if b is None or b.last_batch is None:
            return None
        age = self.clock() - b.last_batch
        steps = list(b.steps)
        last = steps[-1] if steps else None
        out = {"state": "l2" if age < L1_AFTER_S else "l1_only", "age_s": round(age, 1),
               "via": b.pull["via"] or "push", "plugin": b.plugin,
               "plugin_version": b.plugin_version, "engine": b.engine,
               "engine_version": b.engine_version,
               "joined": b.counts["exact"], "approximate": b.counts["approximate"],
               "unmatched": b.counts["row_unmatched"] + b.counts["item_unmatched"],
               "untagged": b.counts["item_untagged"], "dropped": dict(b.dropped),
               "last": None, "window": None}
        if last is not None:
            out["last"] = {k: last.get(k) for k in ("running", "waiting", "kv_usage",
                                                     "cache_hit_rate", "gen_tps")
                           if last.get(k) is not None}
        win = [s for s in steps if s.get("_rx", 0) >= self.wall() - 300.0]
        if win:
            q = sum(s.get("prefix_queries") or 0 for s in win)
            h = sum(s.get("prefix_hits") or 0 for s in win)
            spec = [s["spec"] for s in win if isinstance(s.get("spec"), dict)]
            w = {"samples": len(win),
                 "preempted": sum((s.get("preempted") or 0) + (s.get("retracted") or 0)
                                  for s in win),
                 "evictions": sum(s.get("evictions") or 0 for s in win),
                 "prefix_hit_rate": round(h / q, 4) if q else None, "spec": None}
            if spec and any("drafts" in s for s in spec):
                drafts = sum(s.get("drafts") or 0 for s in spec)
                dtok = sum(s.get("draft_tokens") or 0 for s in spec)
                acc = sum(s.get("accepted") or 0 for s in spec)
                n = max((len(s.get("accepted_per_pos") or []) for s in spec), default=0)
                pos = [sum((s.get("accepted_per_pos") or [0] * n)[i]
                           if i < len(s.get("accepted_per_pos") or []) else 0 for s in spec)
                       for i in range(n)]
                w["spec"] = {"acceptance": round(acc / dtok, 4) if dtok else None,
                             "tokens_per_step": round(1 + acc / drafts, 3) if drafts else None,
                             "per_position": [round(x / drafts, 4) if drafts else None
                                              for x in pos]}
            elif spec:
                ln = [s["accept_length"] for s in spec if s.get("accept_length")]
                rt = [s["accept_rate"] for s in spec if s.get("accept_rate") is not None]
                w["spec"] = {"acceptance": round(sum(rt) / len(rt), 4) if rt else None,
                             "tokens_per_step": round(sum(ln) / len(ln), 3) if ln else None,
                             "per_position": None}
            out["window"] = w
        return out


def handle(handler, svc, path: str, payload) -> None:
    """`POST /v1/collect/<backend>` on either server (proxy/app.py, server/app.py). `payload` is
    the raw body, or the body already parsed by the server."""
    col: Collector | None = getattr(svc, "collect", None)

    def send(code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        try:
            handler.send_response(code)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            if code == 401:
                handler.send_header("WWW-Authenticate", 'Bearer realm="TandemLLM collect"')
            handler.end_headers()
            handler.wfile.write(data)
        except OSError:
            pass

    if col is None:
        return send(404, {"error": {"message": "no collector on this server",
                                    "type": "not_found"}})
    bid = path[len("/v1/collect/"):].strip("/") if path.startswith("/v1/collect/") else ""
    if not col.check_token(handler.headers.get("Authorization")):
        return send(401, {"error": {"message": "the collect token is needed (Authorization: "
                                               "Bearer <TANDEM_COLLECT_TOKEN>)",
                                    "type": "unauthorized"}})
    b = svc.cfg.backend(bid) if bid else None
    if b is None:
        return send(404, {"error": {"message": f"no backend {bid!r}", "type": "not_found"}})
    if isinstance(payload, (bytes, bytearray)) and len(payload) > BODY_MAX:
        return send(413, {"error": {"message": "the batch is too large",
                                    "type": "invalid_request_error"}})
    try:
        batch = (json.loads(payload or b"{}") if isinstance(payload, (bytes, bytearray))
                 else payload)
        return send(200, col.ingest(bid, batch))
    except ValueError as exc:
        return send(400, {"error": {"message": f"bad json: {exc}",
                                    "type": "invalid_request_error"}})
    except CollectError as exc:
        return send(exc.status, exc.body())
