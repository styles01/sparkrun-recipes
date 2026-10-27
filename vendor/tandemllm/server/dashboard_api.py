"""The dashboard's query API, contract v1: summary, usage, requests, system.

The contract is docs/contract/dashboard-v1/README.md, and its
machine-readable form is `docs/contract/dashboard-v1/*.schema.json` -- the dashboard is built
against those, first on mocks, so a response that does not validate is a bug here, not there
(`tools/contract_check.py`, and every test in `tests/test_dashboard_api.py` runs it).

Read-only by construction: every query runs on the ledger's per-thread read-only connection and
nothing here touches the engine lock. The one number that needs a subprocess, the GPU's
temperature and power, is sampled at most every 5 s with a 2 s timeout, and a failure gives nulls.

DEFINITIONS, because the numbers are only as good as these:

  * day buckets are LOCAL days in the request's `tz` (IANA name, default Europe/Berlin): a DST
    day is 23 or 25 hours long. Hour buckets are local hours; a fall-back night has 25 of them.
  * `buckets` is dense: every bucket of the range is present, zero-filled, percentiles null.
  * percentiles are nearest-rank -- the value at rank ceil(p * n) of the sorted values -- so they
    are exact, observed values. They cover DECODED requests only: status 200, not an error, not
    a refusal, not a response-cache replay (a replay's "speed" is a dictionary lookup), at least
    two tokens, and nothing the conductor changed (`conductor_acted`, SRV-115). Token sums
    include every row.
  * `tokens_per_block_mean` is committed tokens over blocks summed across the decoded rows (the
    engine's tuning number, weighted the way the row weights it); `draft_acceptance` is accepted
    over drafted, summed the same way.
  * `errors` = `finish_reason` error, or a 5xx that was not a refusal. `refused` = 503/429 before
    the engine. `total_tokens` = prompt + completion (cached is part of prompt, reasoning of
    completion).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:                                               # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception
try:
    # Order statistics over half a million values: np.partition is linear and in C. The box's
    # interpreter has numpy (torch's); without it the same numbers come from sorted().
    import numpy as np
except ImportError:                                               # pragma: no cover
    np = None

from server.activity import REPLAY  # noqa: E402  (answered from memory, nothing decoded)

# 1.3 (the proxy, P1): rows and dimensions gain the backend and the user, the totals a cost, and
# `backend=` / `user=` filter usage and requests. Every new field is optional in the schemas.
CONTRACT = "1.3"
# the log endpoints did not change: their answers stay at 1.0
LOGS_CONTRACT = "1.0"
# `GET /v1/dashboard/session` has its own: 1.1 added `login` (false when the dashboard login is
# off) and the version field itself; `expires_at` is null when there is no login
SESSION_CONTRACT = "1.1"
DEFAULT_TZ = "Europe/Berlin"
HOUR_LIMIT_DAYS = 31
MAX_LIMIT = 500
CACHE_TTL_S = 60.0
CLIENTS_JSON = "~/.qwen38-spark-engine/clients.json"

# A request the conductor acted in is not a speed sample (SRV-115), nor a regenerate (SRV-241: it
# samples, the speed sample is the greedy default's), nor a loop escape's or a profile's card
# sampling (SRV-475): its time holds hidden
# forwards its token counts leave out, and its prompt a note the client did not send.
DECODED = ("status = 200 AND (finish_reason IS NULL OR finish_reason NOT IN ('error', 'refused')) "
           "AND (cache_source IS NULL OR cache_source NOT IN ('response', 'prepared')) "
           "AND decode_tps IS NOT NULL AND completion_tokens > 1 "
           "AND COALESCE(conductor_acted, 0) = 0 AND lane IS NULL "
           "AND COALESCE(sampling, '') NOT IN ('regenerate', 'loop_escape', 'card')")
ERROR = ("(finish_reason = 'error' OR (status >= 500 AND (finish_reason IS NULL "
         "OR finish_reason != 'refused')))")


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message

    def body(self) -> dict:
        return {"error": {"type": self.kind, "message": self.message}}


def _flag(v) -> bool:
    """A query switch: 1, true, yes, on."""
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _bad(msg: str) -> ApiError:
    return ApiError(400, "bad_request", msg)


# ----------------------------------------------------------------- time

def iso_utc(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    t = _dt.datetime.fromtimestamp(ms / 1000.0, _dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def iso_utc_s(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def zone(name: str | None):
    name = name or DEFAULT_TZ
    if ZoneInfo is None:
        raise _bad("time zones are not available on this server")
    if not re.fullmatch(r"[A-Za-z0-9_+\-/]{1,64}", name):
        raise _bad(f"tz must be an IANA time zone name, got {name!r}")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise _bad(f"unknown time zone {name!r}")


def local_midnight_ms(day: _dt.date, tz) -> int:
    return int(_dt.datetime(day.year, day.month, day.day, tzinfo=tz).timestamp() * 1000)


def parse_day(text: str | None, tz, name: str) -> _dt.date | None:
    """`YYYY-MM-DD` (a local date) or RFC 3339 (its local date in `tz`)."""
    if not text:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return _dt.date.fromisoformat(text)
        t = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=tz)
        return t.astimezone(tz).date()
    except ValueError:
        raise _bad(f"{name} must be YYYY-MM-DD or RFC 3339, got {text!r}")


def offset_segments(start_ms: int, end_ms: int, tz) -> list[tuple[int, int]]:
    """[(from_ms, utc offset in ms)] covering [start_ms, end_ms): where the zone's offset changes.

    Found by stepping an hour at a time and bisecting each change to the second, so a range of a
    year costs 8,760 offset lookups once per query, and the SQL can then bucket rows by a CASE
    over two or three segments instead of calling back into Python per row.
    """
    def off(ms):
        return int(_dt.datetime.fromtimestamp(ms / 1000, tz).utcoffset().total_seconds() * 1000)

    segs = [(start_ms, off(start_ms))]
    t = start_ms
    while t < end_ms:
        n = min(t + 3_600_000, end_ms)
        if off(n) != segs[-1][1] and n > t:
            lo, hi = t, n
            while hi - lo > 1000:
                mid = (lo + hi) // 2
                if off(mid) == segs[-1][1]:
                    lo = mid
                else:
                    hi = mid
            segs.append((hi, off(n)))
        t = n
    return segs


def _case_offset(segs) -> tuple[str, list]:
    if len(segs) == 1:
        return "?", [segs[0][1]]
    sql, args = "CASE", []
    for (_, o), (nxt, _) in zip(segs, segs[1:]):
        sql += " WHEN ts_ms < ? THEN ?"
        args += [nxt, o]
    sql += " ELSE ? END"
    args.append(segs[-1][1])
    return sql, args


def _rank(p: float, n: int) -> int:
    """Nearest rank, 0-based: the value at rank ceil(p * n) of the sorted values."""
    return max(1, math.ceil(p * n)) - 1


def _values(text: str | None):
    """A `group_concat` of REALs back into numbers, unsorted."""
    if not text:
        return np.empty(0) if np is not None else []
    return (np.array(text.split(","), dtype=np.float64) if np is not None
            else [float(x) for x in text.split(",")])


def _join(parts):
    parts = [x for x in parts if len(x)]
    if np is not None:
        return np.concatenate(parts) if parts else np.empty(0)
    return [v for x in parts for v in x]


def percentiles(vals, ps=(0.5, 0.9)) -> list:
    """Exact nearest-rank percentiles of unsorted values; None when there are none."""
    n = len(vals)
    if not n:
        return [None] * len(ps)
    ks = [_rank(p, n) for p in ps]
    if np is not None:
        part = np.partition(vals, sorted(set(ks)))
        return [round(float(part[k]), 2) for k in ks]
    srt = sorted(vals)
    return [round(float(srt[k]), 2) for k in ks]


# ----------------------------------------------------------------- the ranges

class Range:
    """A query range: buckets, their edges in UTC ms, and the SQL that maps a row to its bucket."""

    def __init__(self, first: _dt.date, last: _dt.date, bucket: str, tz):
        if last < first:
            raise _bad("from is after to")
        self.first, self.last, self.bucket, self.tz = first, last, bucket, tz
        self.start_ms = local_midnight_ms(first, tz)
        self.end_ms = local_midnight_ms(last + _dt.timedelta(days=1), tz)
        if bucket == "day":
            days = (last - first).days + 1
            self.starts = [local_midnight_ms(first + _dt.timedelta(days=i), tz)
                           for i in range(days)]
            segs = offset_segments(self.start_ms, self.end_ms, tz)
            case, args = _case_offset(segs)
            # local epoch-day of the row, minus the first bucket's
            base = (self.start_ms + segs[0][1]) // 86_400_000
            self.idx_sql = f"((ts_ms + {case}) / 86400000 - ?)"
            self.idx_args = args + [base]
        else:
            n = (self.end_ms - self.start_ms) // 3_600_000
            self.starts = [self.start_ms + i * 3_600_000 for i in range(n)]
            self.idx_sql = "((ts_ms - ?) / 3600000)"
            self.idx_args = [self.start_ms]

    def label(self, i: int) -> str:
        t = _dt.datetime.fromtimestamp(self.starts[i] / 1000, self.tz)
        return t.isoformat(timespec="seconds")


ENGINE = "engine"     # the `backend=` filter's value for the engine's own rows (backend IS NULL)


def _filters(model: str | None, client: str | None, backend: str | None = None,
             user: str | None = None, internal: bool = False) -> tuple[str, list]:
    """The rows a view counts. VIS-60: TandemLLM's own calls (client kind `internal`, the router's
    question to the engine) are left out unless `internal` (the views' "show internal" switch)."""
    sql, args = "", []
    if not internal:
        sql += " AND client_kind IS NOT 'internal'"
    if model:
        sql += " AND model = ?"
        args.append(model)
    if client:
        sql += " AND client_id = ?"
        args.append(client)
    if backend == ENGINE:
        sql += " AND backend IS NULL"
    elif backend:
        sql += " AND backend = ?"
        args.append(backend)
    if user:
        sql += " AND user = ?"
        args.append(user)
    return sql, args


_SUMS = (f"COUNT(*), SUM({ERROR}), SUM(finish_reason = 'refused'), "
         "SUM(COALESCE(prompt_tokens, 0)), SUM(COALESCE(cached_tokens, 0)), "
         "SUM(COALESCE(completion_tokens, 0)), SUM(COALESCE(reasoning_tokens, 0)), "
         "SUM(tool_calls > 0), SUM(thinking), "
         "SUM(COALESCE(draft_tokens, 0)), SUM(COALESCE(draft_accepted, 0)), "
         "SUM(COALESCE(cost_micros, 0)), COUNT(cost_micros)")
# the decoded rows only: their speed values, and committed tokens and blocks for tokens/block
_SPEEDS = ("group_concat(decode_tps), group_concat(ttft_ms), group_concat(prefill_tps), "
           "SUM(completion_tokens - 1), SUM(COALESCE(blocks, 0))")
_COUNTS = ("n", "err", "ref", "p", "c", "o", "r", "tc", "th", "drafted", "accepted",
           "cost", "cost_n", "committed", "blocks")


class Agg:
    """One bucket's rows: the sums, and the decoded rows' speed values for the percentiles.

    Two scans fill it -- the sums over every row, then a `group_concat` of the decoded rows'
    speeds, unsorted -- and the percentiles are order statistics taken when asked. Measured on
    500,000 synthetic rows (2026-09-24, Mac CPU): the old one-list-a-row version spent 0.8 s in
    Python appends alone.
    """

    __slots__ = _COUNTS + ("dec", "ttft", "pre")

    def __init__(self):
        for k in _COUNTS:
            setattr(self, k, 0)
        empty = np.empty(0) if np is not None else []
        self.dec, self.ttft, self.pre = empty, empty, empty

    @classmethod
    def merge(cls, aggs) -> "Agg":
        out = cls()
        for k in _COUNTS:
            setattr(out, k, sum(getattr(a, k) for a in aggs))
        out.dec = _join([a.dec for a in aggs])
        out.ttft = _join([a.ttft for a in aggs])
        out.pre = _join([a.pre for a in aggs])
        return out

    def totals(self, active_days: int | None) -> dict:
        d50, d90 = percentiles(self.dec)
        t50, t90 = percentiles(self.ttft)
        (p50,) = percentiles(self.pre, (0.5,))
        out = {"requests": self.n, "errors": self.err, "refused": self.ref,
               "prompt_tokens": self.p, "cached_tokens": self.c, "completion_tokens": self.o,
               "reasoning_tokens": self.r, "total_tokens": self.p + self.o,
               "tool_call_requests": self.tc, "thinking_requests": self.th,
               "decode_tps_p50": d50, "decode_tps_p90": d90, "ttft_ms_p50": t50,
               "ttft_ms_p90": t90, "prefill_tps_p50": p50,
               "tokens_per_block_mean": round(self.committed / self.blocks, 2)
               if self.blocks else None,
               "draft_acceptance": round(self.accepted / self.drafted, 4)
               if self.drafted else None,
               # 1.3: micro-dollars at the models' prices; null when no row in range had a price
               "cost_micros": self.cost if self.cost_n else None}
        if active_days is not None:
            out["active_days"] = active_days
        return out


class DashboardAPI:
    """The four read endpoints. `ledger` may be None (the ledger is off): everything is empty."""

    def __init__(self, ledger=None, *, live=None, system=None, clock=time.time,
                 clients_path: str = CLIENTS_JSON, cache_ttl: float = CACHE_TTL_S,
                 backends=None):
        self.ledger = ledger
        # 1.3: the configured backends (proxy/config.py), for their labels and kinds; None on a
        # server without the proxy
        self.backends_fn = backends
        self.live = live or (lambda: {"status": "ok", "running": 0, "waiting": 0,
                                      "uptime_s": 0.0, "last_request_at": None})
        self.system_fn = system
        self.clock = clock
        self.clients_path = clients_path
        self.cache_ttl = cache_ttl
        self._cache: dict = {}
        self._memo: dict = {}
        self._memo_gen = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- plumbing
    def _db(self):
        return self.ledger.reader() if self.ledger is not None else None

    def _cached(self, key, fn):
        now = self.clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]
        out = fn()
        with self._lock:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = (now + self.cache_ttl, out)
        return out

    def labels(self) -> dict:
        try:
            with open(os.path.expanduser(self.clients_path)) as f:
                got = json.load(f)
            return {str(k): str(v) for k, v in got.items()} if isinstance(got, dict) else {}
        except (OSError, ValueError):
            return {}

    def _today(self, tz) -> _dt.date:
        return _dt.datetime.fromtimestamp(self.clock(), tz).date()

    # ----------------------------------------------------------------- the aggregation
    def _scan(self, rng: Range, model=None, client=None, backend=None, user=None,
              internal=False) -> list:
        """Every bucket of a range from the ledger: two scans, the sums and the decoded rows."""
        aggs = [Agg() for _ in rng.starts]
        db = self._db()
        if db is None:
            return aggs
        fsql, fargs = _filters(model, client, backend, user, internal)
        where = f"ts_ms >= ? AND ts_ms < ?{fsql}"
        args = rng.idx_args + [rng.start_ms, rng.end_ms] + fargs
        for row in db.execute(f"SELECT {rng.idx_sql} AS i, {_SUMS} FROM requests "
                              f"WHERE {where} GROUP BY i", args):
            i = int(row[0])
            if 0 <= i < len(aggs):
                a = aggs[i]
                (a.n, a.err, a.ref, a.p, a.c, a.o, a.r, a.tc, a.th, a.drafted,
                 a.accepted, a.cost, a.cost_n) = (int(x or 0) for x in row[1:])
        for i, dec, ttft, pre, committed, blocks in db.execute(
                f"SELECT {rng.idx_sql} AS i, {_SPEEDS} FROM requests "
                f"WHERE {where} AND {DECODED} GROUP BY i", args):
            if 0 <= i < len(aggs):
                a = aggs[i]
                a.dec, a.ttft, a.pre = _values(dec), _values(ttft), _values(pre)
                a.committed, a.blocks = int(committed or 0), int(blocks or 0)
        return aggs

    def _days(self, first: _dt.date, last: _dt.date, tz, tz_name, model, client,
              backend=None, user=None, internal=False) -> list:
        """One `Agg` per local day, from the memo where the day is final.

        A day is final two hours after it ended: a row carries its request's ARRIVAL time and is
        written when the request ends, which for the longest ones (the queue wait plus the
        request timeout) is about an hour later. Final days are kept per (tz, filters) and
        forgotten when the ledger prunes, so the year view pays for the year once and then only
        for today.
        """
        gen = (self.ledger.path, self.ledger.stats.get("pruned")) if self.ledger else None
        with self._lock:
            if self._memo_gen != gen:
                self._memo.clear()
                self._memo_gen = gen
            memo = self._memo.setdefault((tz_name, model, client, backend, user, internal), {})
            while len(self._memo) > 8:
                self._memo.pop(next(iter(self._memo)))
        dates = [first + _dt.timedelta(days=i) for i in range((last - first).days + 1)]
        out = [memo.get(d) for d in dates]
        cutoff = self.clock() * 1000 - 2 * 3_600_000
        i = 0
        while i < len(out):
            if out[i] is not None:
                i += 1
                continue
            j = i                                      # one scan for each run of missing days
            while j + 1 < len(out) and out[j + 1] is None:
                j += 1
            for k, a in enumerate(self._scan(Range(dates[i], dates[j], "day", tz), model,
                                             client, backend, user, internal)):
                d = dates[i + k]
                out[i + k] = a
                if local_midnight_ms(d + _dt.timedelta(days=1), tz) < cutoff:
                    memo[d] = a
            i = j + 1
        return out

    def _top_client(self, day: _dt.date, tz, model, client, backend=None,
                    user=None, internal=False) -> str | None:
        """The client with the most tokens on one local day (the ts index makes it cheap)."""
        fsql, fargs = _filters(model, client, backend, user, internal)
        row = self._db().execute(
            "SELECT client_id, SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) "
            f"AS t FROM requests WHERE ts_ms >= ? AND ts_ms < ?{fsql} GROUP BY client_id "
            "ORDER BY t DESC, client_id LIMIT 1",
            [local_midnight_ms(day, tz), local_midnight_ms(day + _dt.timedelta(days=1), tz)]
            + fargs).fetchone()
        return row[0] if row else None

    # ----------------------------------------------------------------- endpoints
    def usage(self, q: dict) -> dict:
        tz_name = q.get("tz") or DEFAULT_TZ
        tz = zone(tz_name)
        bucket = q.get("bucket") or "day"
        if bucket not in ("day", "hour"):
            raise _bad(f"bucket must be day or hour, got {bucket!r}")
        last = parse_day(q.get("to"), tz, "to") or self._today(tz)
        first = parse_day(q.get("from"), tz, "from") or (
            last - _dt.timedelta(days=364 if bucket == "day" else 2))
        if bucket == "hour" and (last - first).days + 1 > HOUR_LIMIT_DAYS:
            raise _bad(f"bucket=hour spans at most {HOUR_LIMIT_DAYS} days; "
                       f"this range is {(last - first).days + 1}")
        if (last - first).days > 3660:
            raise _bad("a range may span at most ten years")
        model, client = q.get("model") or None, q.get("client") or None
        backend, user = q.get("backend") or None, q.get("user") or None
        internal = _flag(q.get("internal"))
        key = ("usage", tz_name, bucket, first, last, model, client, backend, user, internal)
        return self._cached(key, lambda: self._usage(first, last, bucket, tz, tz_name, model,
                                                     client, backend, user, internal))

    def _usage(self, first, last, bucket, tz, tz_name, model, client, backend=None,
               user=None, internal=False) -> dict:
        rng = Range(first, last, bucket, tz)
        aggs = (self._days(first, last, tz, tz_name, model, client, backend, user, internal)
                if bucket == "day" else self._scan(rng, model, client, backend, user, internal))
        buckets = []
        for i, a in enumerate(aggs):
            b = a.totals(None)
            b.pop("refused")
            b.pop("thinking_requests")
            buckets.append({"start": rng.label(i), **b})
        top_days = []
        if bucket == "day":
            order = sorted((i for i, a in enumerate(aggs) if a.n),
                           key=lambda i: (-(aggs[i].p + aggs[i].o), i))[:10]
            for i in order:
                d = first + _dt.timedelta(days=i)
                top_days.append({"date": d.isoformat(), "total_tokens": aggs[i].p + aggs[i].o,
                                 "requests": aggs[i].n,
                                 "top_client": self._top_client(d, tz, model, client, backend,
                                                                user, internal)})
        filters = {"model": model, "client": client}
        if internal:
            filters["internal"] = True
        if backend or user:
            filters.update(backend=backend, user=user)
        return {"contract_version": CONTRACT, "from": first.isoformat(), "to": last.isoformat(),
                "bucket": bucket, "tz": tz_name, "filters": filters,
                "buckets": buckets,
                "totals": Agg.merge(aggs).totals(sum(1 for a in aggs if a.n)),
                "top_days": top_days, "dimensions": self.dimensions(internal)}

    def dimensions(self, internal: bool = False) -> dict:
        """Every model and client the ledger has seen (the filter lists), from the covering
        index on (model, client_id, ts_ms); a client's kind is the one on its latest row. A
        model or client seen only on internal rows is listed with `internal` alone (VIS-60)."""
        db = self._db()
        models, clients = set(), {}
        if db is not None:
            labels = self.labels()
            for model, cid, ts in db.execute(
                    "SELECT model, client_id, MAX(ts_ms) FROM requests GROUP BY model, client_id"
                    ).fetchall():
                if not internal and self._internal_only(db, model, cid):
                    continue
                models.add(model)
                if cid not in clients or ts > clients[cid][0]:
                    kind = db.execute("SELECT client_kind FROM requests WHERE model = ? AND "
                                      "client_id = ? AND ts_ms = ? LIMIT 1",
                                      (model, cid, ts)).fetchone()[0]
                    clients[cid] = (ts, {"id": cid, "label": labels.get(cid), "kind": kind})
        out = {"models": sorted(models), "clients": [clients[c][1] for c in sorted(clients)]}
        extra = self._proxy_dimensions(db)
        if extra is not None:
            out.update(extra)
        return out

    @staticmethod
    def _internal_only(db, model: str, cid: str) -> bool:
        """VIS-60: every row of this model and client is `internal`. Two seeks on requests_dim_kind
        (model, client_id, client_kind; client_kind is NOT NULL): a kind sorting before
        'internal' or one after it. It replaced a GROUP BY ... HAVING over every row on each usage
        call (165 ms of 1.1 s over 500k rows, FX14)."""
        q = ("SELECT 1 FROM requests WHERE model = ? AND client_id = ? AND client_kind {} 'internal' "
             "LIMIT 1")
        return not any(db.execute(q.format(op), (model, cid)).fetchone() for op in ("<", ">"))

    @staticmethod
    def _distinct_users(db) -> list:
        """The users the ledger has seen, one seek each on the partial index requests_user (a
        DISTINCT read every row: 67 ms over 500k rows, FX14)."""
        out = []
        u = db.execute("SELECT MIN(user) FROM requests WHERE user IS NOT NULL").fetchone()[0]
        while u is not None:
            out.append(u)
            u = db.execute("SELECT MIN(user) FROM requests WHERE user > ?", (u,)).fetchone()[0]
        return sorted(out)

    def _proxy_dimensions(self, db) -> dict | None:
        """1.3: the backends and users the ledger has seen, with the configured ones; None on a
        server without the proxy and a ledger with no relayed row (the 1.0 answer)."""
        rows, users = [], []
        if db is not None:
            try:
                rows = [r[0] for r in db.execute("SELECT DISTINCT backend FROM requests")]
                users = self._distinct_users(db)
            except sqlite3.Error:
                rows, users = [], []
        conf = {}
        if self.backends_fn is not None:
            try:
                conf = {b["id"]: b for b in self.backends_fn()}
            except Exception:                                      # noqa: BLE001
                conf = {}
        seen = [b for b in rows if b is not None]
        if not seen and not conf:
            return None
        backends = []
        if None in rows:
            backends.append({"id": ENGINE, "kind": "engine", "label": "this engine"})
        for bid in sorted(set(seen) | set(conf)):
            b = conf.get(bid) or {}
            backends.append({"id": bid, "kind": b.get("kind"), "label": b.get("label") or bid})
        return {"backends": backends, "users": users}

    def summary(self, q: dict) -> dict:
        tz_name = q.get("tz") or DEFAULT_TZ
        tz = zone(tz_name)
        internal = _flag(q.get("internal"))
        key = ("summary", tz_name, internal)
        out = dict(self._cached(key, lambda: self._summary(tz, tz_name, internal)))
        out["live"] = self.live()                      # never cached: it is the "now" pill
        out["generated_at"] = iso_utc_s(self.clock())
        return out

    def _summary(self, tz, tz_name, internal=False) -> dict:
        today = self._today(tz)
        led = self.ledger.info() if self.ledger is not None else None
        oldest_ms = (led or {}).get("oldest_ms")
        first_day = (_dt.datetime.fromtimestamp(oldest_ms / 1000, tz).date()
                     if oldest_ms is not None else today)
        first_day = min(first_day, today - _dt.timedelta(days=364))
        aggs = self._days(first_day, today, tz, tz_name, None, None, internal=internal)
        n = len(aggs)
        windows = {}
        for name, days in (("today", 1), ("7d", 7), ("30d", 30), ("365d", 365), ("all", n)):
            part = aggs[max(0, n - days):]
            windows[name] = Agg.merge(part).totals(sum(1 for a in part if a.n))
        active = [a.n > 0 for a in aggs]
        cur, i = 0, n - 1
        if i >= 0 and not active[i]:
            i -= 1                                     # today may not have started yet
        while i >= 0 and active[i]:
            cur += 1
            i -= 1
        longest = run = 0
        for a in active:
            run = run + 1 if a else 0
            longest = max(longest, run)
        ledger = {"enabled": led is not None,
                  "since": iso_utc((led or {}).get("since_ms")),
                  "rows": (led or {}).get("rows") or 0, "bytes": (led or {}).get("bytes") or 0,
                  "retention_days": (led or {}).get("retention_days")}
        return {"contract_version": CONTRACT, "generated_at": iso_utc_s(self.clock()),
                "tz": tz_name, "windows": windows,
                "streak": {"current_days": cur, "longest_days": longest}, "ledger": ledger}

    def requests(self, q: dict) -> dict:
        try:
            limit = int(q.get("limit") or 50)
            before = int(q["before"]) if q.get("before") else None
        except ValueError:
            raise _bad("limit and before must be integers")
        if not 1 <= limit <= MAX_LIMIT:
            raise _bad(f"limit must be 1..{MAX_LIMIT}, got {limit}")
        db = self._db()
        rows = []
        if db is not None:
            sql, args = "SELECT * FROM requests WHERE 1=1", []
            if before is not None:
                sql += " AND id < ?"
                args.append(before)
            fsql, fargs = _filters(q.get("model"), q.get("client"), q.get("backend"),
                                   q.get("user"), _flag(q.get("internal")))
            sql += fsql
            args += fargs
            if q.get("finish"):
                sql += " AND finish_reason = ?"
                args.append(q["finish"])
            cur = db.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit + 1])
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        more = len(rows) > limit
        rows = rows[:limit]
        labels = self.labels()
        shadow = _shadow_counts(db, [r["request_id"] for r in rows]) if rows else {}
        conf = {}
        if self.backends_fn is not None:
            try:
                conf = {b["id"]: b for b in self.backends_fn()}
            except Exception:                                      # noqa: BLE001
                conf = {}
        return {"contract_version": CONTRACT,
                "next_before": rows[-1]["id"] if more and rows else None,
                "requests": [self._request(r, labels, shadow, conf) for r in rows]}

    @staticmethod
    def _request(r: dict, labels: dict, shadow: dict | None = None,
                 conf: dict | None = None) -> dict:
        c, b = r["completion_tokens"], r["blocks"]
        replay = r["cache_source"] in REPLAY
        n_shadow = (shadow or {}).get(r["request_id"], 0)
        mode = r.get("conductor_mode")
        # VIS-44: the row's badge -- the monitor's acts (filled) or its shadow decisions
        # (outlined); null for a request the conductor did not run on
        cond = ({"mode": mode, "acts": int(r.get("conductor_acts") or 0), "shadow": n_shadow,
                 "hidden_tokens": int(r.get("conductor_hidden_tokens") or 0),
                 "acted": bool(r.get("conductor_acted"))}
                if (mode not in (None, "raw") or n_shadow or r.get("conductor_acts")) else None)
        return {"id": r["id"], "request_id": r["request_id"], "ts": iso_utc(r["ts_ms"]),
                "model": r["model"],
                "client": {"id": r["client_id"], "label": labels.get(r["client_id"]),
                           "kind": r["client_kind"]},
                "endpoint": r["endpoint"], "stream": bool(r["stream"]), "status": r["status"],
                "finish_reason": r["finish_reason"], "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"], "completion_tokens": c,
                "reasoning_tokens": r["reasoning_tokens"], "queue_ms": r["queue_ms"],
                "prompt_ms": r["prompt_ms"], "ttft_ms": r["ttft_ms"], "decode_ms": r["decode_ms"],
                "total_ms": r["total_ms"],
                # a replay (the response cache, a summary written ahead) decoded nothing: its
                # rate is no speed, also on rows written before 0.4.0 recorded none
                "decode_tps": None if replay else r["decode_tps"],
                "prefill_tps": None if replay else r["prefill_tps"], "blocks": b,
                "tokens_per_block": round((c - 1) / b, 2) if b and c else None,
                "draft_tokens": r["draft_tokens"], "draft_accepted": r["draft_accepted"],
                "tool_calls": r["tool_calls"], "thinking": bool(r["thinking"]),
                "cache_source": r["cache_source"], "error_type": r["error_type"],
                "conductor": cond,
                # Served in the side lane (server/preempt.py), null otherwise
                "lane": r.get("lane"),
                # SRV-197: the sequence slot with parallel requests (0 main, 1+ side), else null
                "slot": r.get("slot"),
                # SRV-241: how it sampled (null greedy, `client`, `regenerate`)
                "sampling": r.get("sampling"),
                # VIS-60: an internal row's request (the turn the router's question served)
                **({"parent_request": r.get("parent_request")}
                   if r["client_kind"] == "internal" else {}),
                # 1.3, the proxy: null for the engine's own rows
                **_proxy_fields(r, conf or {})}

    def system(self) -> dict:
        out = {"contract_version": CONTRACT, "generated_at": iso_utc_s(self.clock())}
        out.update(self.system_fn() if self.system_fn is not None else {})
        return out


def _proxy_fields(r: dict, conf: dict) -> dict:
    """A request row's 1.3 fields: the backend that answered, the user's label, and what only a
    relayed row has. `measured_at` says where the timings come from: the engine's own loop or the
    proxy (TTFT then holds the network and the backend's queue)."""
    bid = r.get("backend")
    b = conf.get(bid) or {}
    return {"backend": ({"id": bid, "kind": r.get("backend_kind") or b.get("kind"),
                         "label": b.get("label") or bid} if bid else None),
            "user": r.get("user"),
            "measured_at": "proxy" if bid else "engine",
            "backend_model": r.get("backend_model"),
            "usage_source": r.get("usage_source"),
            "cache_write_tokens": r.get("cache_write_tokens"),
            "cost_micros": r.get("cost_micros"),
            "proxy_ms": r.get("proxy_ms"),
            "retries": r.get("retries"),
            "fallback_from": r.get("fallback_from"),
            "upstream_status": r.get("upstream_status"),
            # VIS-207 (ledger schema 10): the backend configuration the answer named
            "system_fingerprint": r.get("system_fingerprint"),
            # P6: the engine's own numbers came from its plugin (ledger schema 4)
            "l2": ({"join": r["l2_join"], "preemptions": r.get("preemptions"),
                    "engine_finish": r.get("engine_finish"),
                    "corrupted": None if r.get("corrupted") is None else bool(r["corrupted"])}
                   if r.get("l2_join") else None)}


def _shadow_counts(db, ids: list[str]) -> dict:
    """request id -> the session monitor's shadow decisions that were not a pass, from the
    conductor's journal in the ledger's file (SRV-115); {} with no journal table (an older
    ledger) or no database."""
    ids = [i for i in ids if i][:900]
    if db is None or not ids:
        return {}
    try:
        cur = db.execute("SELECT request_id, COUNT(*) FROM conductor_events WHERE kind = 'monitor' "
                         "AND mode = 'shadow' AND decision IS NOT NULL AND decision != 'PASS' "
                         f"AND request_id IN ({', '.join('?' * len(ids))}) GROUP BY request_id",
                         ids)
        return {rid: int(n) for rid, n in cur.fetchall()}
    except sqlite3.Error:
        return {}


# ----------------------------------------------------------------- the system snapshot's parts

# a secret by its NAME: a whole `_`-separated part, so QSE_ADMIN_TOKEN is one and
# default_max_tokens is not
SECRET = re.compile(r"(^|_)(TOKEN|SECRET|KEY|PASSWORD|PASS|APIKEY)(_|$)", re.I)


def redact_env(env: dict, home: str | None = None) -> dict:
    """`QWEN38_*` / `QSE_*` variables, secrets as `<redacted>`, the home directory as `~`."""
    home = home or os.path.expanduser("~")
    out = {}
    for k in sorted(env):
        if not (k.startswith("QWEN38_") or k.startswith("QSE_")):
            continue
        v = env[k]
        out[k] = "<redacted>" if SECRET.search(k) else _tilde(v, home)
    return out


def _tilde(v, home: str):
    if isinstance(v, str) and home and home != "/":
        return v.replace(home, "~")
    return v


def redact_args(args: dict, home: str | None = None) -> dict:
    home = home or os.path.expanduser("~")
    out = {}
    for k, v in sorted(args.items()):
        if SECRET.search(k):
            out[k] = "<redacted>"
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = _tilde(v, home)
        else:
            out[k] = _tilde(str(v), home)
    return out


class GpuSampler:
    """`nvidia-smi` at most every `period` s, `timeout` s each; a failed source is all nulls."""

    QUERY = "name,temperature.gpu,power.draw,clocks.sm,utilization.gpu"

    def __init__(self, period: float = 5.0, timeout: float = 2.0, cmd: str = "nvidia-smi",
                 clock=time.time):
        self.period, self.timeout, self.cmd, self.clock = period, timeout, cmd, clock
        self._at, self._last = 0.0, None
        self._lock = threading.Lock()

    def sample(self) -> dict:
        with self._lock:
            now = self.clock()
            if self._last is not None and now - self._at < self.period:
                return self._last
            self._at, self._last = now, self._read(now)
            return self._last

    def _read(self, now: float) -> dict:
        empty = {"name": None, "temperature_c": None, "power_w": None, "sm_clock_mhz": None,
                 "utilization": None, "source": None, "sampled_at": iso_utc_s(now)}
        try:
            r = subprocess.run([self.cmd, f"--query-gpu={self.QUERY}",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=self.timeout)
            if r.returncode != 0 or not r.stdout.strip():
                return empty
            parts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        except (OSError, subprocess.SubprocessError):
            return empty

        def num(x, scale=1.0):
            try:
                return round(float(x) * scale, 3)
            except ValueError:
                return None                             # "[N/A]" on GB10 for some fields
        return {"name": parts[0] if parts else None,
                "temperature_c": num(parts[1]) if len(parts) > 1 else None,
                "power_w": num(parts[2]) if len(parts) > 2 else None,
                "sm_clock_mhz": num(parts[3]) if len(parts) > 3 else None,
                "utilization": num(parts[4], 0.01) if len(parts) > 4 else None,
                "source": "nvidia-smi", "sampled_at": iso_utc_s(now)}


def meminfo() -> dict:
    out = {"unified_total_bytes": None, "unified_available_bytes": None, "process_rss_bytes": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    out["unified_total_bytes"] = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    out["unified_available_bytes"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["process_rss_bytes"] = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return out


def disk_free(path: str = "~/.qwen38-spark-engine") -> int | None:
    p = os.path.expanduser(path)
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    try:
        return shutil.disk_usage(p or "/").free
    except OSError:
        return None
