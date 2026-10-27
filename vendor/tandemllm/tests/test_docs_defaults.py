"""The docs name the cache budgets the code and the shipped serve.env use (OPS-77). CPU only.

UX2's W06 (2026-10-07) pasted the docs into a chat and asked for contradictions; the model found
real ones: the state store's budget was "8 GiB in the served profile" in caches.md's table and in
operations.md but 2 GiB in caches.md's text and on the live /health; the guest rule was "8,192
tokens" in one place and "a quarter of it" in another (the code needs both); and "93.4 ms and 4.58
tokens, which gives 49.89 tok/s" (the quotient is 49.0; 49.89 is the mean of the requests' rates).

  1  the server's flag defaults equal ops/serve.env for every cache budget the docs name
                                                        test_1_flags_and_serve_env_agree
  2  operations.md's state store and resident prefix rows and caches.md's table and text name
     those values; no state-store row says 8 GiB         test_2_the_docs_name_them
  3  the guest rule is stated with both conditions, and the code applies both
                                                        test_3_the_guest_rule
  4  a sentence that puts 49.89 tok/s beside 93.4 ms and 4.58 tokens says it is a mean of rates
                                                        test_4_derived_numbers

Run: python tests/test_docs_defaults.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# serve.env name -> the server flag's dest
KNOBS = {"CACHE_GB": "cache_budget_gb", "STORE_ENTRY_GB": "store_entry_gb",
         "RESIDENT_GB": "resident_gb", "RESIDENT_STASH_GB": "resident_stash_gb",
         "RESIDENT_GUEST_MAX": "resident_guest_max", "RESIDENT_PIN_GB": "resident_pin_gb",
         "RESIDENT_PARK_GB": "resident_park_gb"}


def _read(*parts) -> str:
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _serve_env() -> dict:
    out = {}
    for line in _read("ops", "serve.env").splitlines():
        m = re.match(r"^([A-Z_]+)=([^\s#]*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _num(v) -> str:
    """2.0 -> "2", 2.5 -> "2.5", 8192 -> "8,192": the way the docs write it."""
    f = float(v)
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:g}"


def _defaults() -> dict:
    from server import app
    a = app.parser().parse_args([])
    return {k: getattr(a, d) for k, d in KNOBS.items()}


def _row(text: str, name: str) -> str:
    rows = [ln for ln in text.splitlines() if ln.startswith(f"| {name} |")]
    assert len(rows) == 1, (name, rows)
    return rows[0]


def test_1_flags_and_serve_env_agree():
    env, flags = _serve_env(), _defaults()
    for k, v in flags.items():
        assert k in env, f"ops/serve.env sets no {k}"
        assert float(env[k]) == float(v), (k, env[k], v)
    return ", ".join(f"{k}={_num(v)}" for k, v in flags.items())


def test_2_the_docs_name_them():
    d = {k: _num(v) for k, v in _defaults().items()}
    ops, caches = _read("docs", "operations.md"), _read("docs", "caches.md")
    store = _row(ops, "state store")
    assert f"{d['CACHE_GB']} GiB budget (`CACHE_GB`)" in store, store
    assert f"{d['STORE_ENTRY_GB']} GiB a snapshot (`STORE_ENTRY_GB`" in store, store
    res = _row(ops, "resident prefix")
    for k, what in (("RESIDENT_GB", "of anchors"), ("RESIDENT_STASH_GB", "guest stash"),
                    ("RESIDENT_PIN_GB", "pin"), ("RESIDENT_PARK_GB", "parked session")):
        assert f"{d[k]} GiB" in res and f"`{k}`" in res and what in res, (k, res)
    table = _row(caches, "state store")
    assert f"{d['CACHE_GB']} GiB (`CACHE_GB`)" in table, table
    assert f"The served budget is {d['CACHE_GB']} GiB (`CACHE_GB`)" in caches
    # no state-store budget other than the served one anywhere in the two pages, except a row
    # dated as another day's configuration
    for page, text in (("operations.md", ops), ("caches.md", caches)):
        for ln in text.splitlines():
            if "state store" in ln and re.search(r"\b8 GiB\b", ln):
                assert "that day's budgets" in ln, (page, ln[:200])
    return f"state store {d['CACHE_GB']} GiB, entry {d['STORE_ENTRY_GB']} GiB, anchors {d['RESIDENT_GB']} GiB"


def test_3_the_guest_rule():
    caches = _read("docs", "caches.md")
    g = _num(_defaults()["RESIDENT_GUEST_MAX"])
    assert f"at most `--resident-guest-max` tokens, {g}, and under a quarter of the resident " \
           f"prompt: both must hold" in caches
    assert "(`--resident-guest-max`, a quarter of it)" not in caches
    # the code: both conditions (engine/cache.py ResidentPrefix.is_guest), a quarter by default
    from engine.cache import ResidentPrefix
    import inspect
    src = inspect.getsource(ResidentPrefix.is_guest)
    assert "n <= self.guest_max" in src and "n < self.guest_frac * self.valid" in src, \
        "the guest rule in the code changed: update caches.md and this test"
    assert inspect.signature(ResidentPrefix.__init__).parameters["guest_frac"].default == 0.25


def test_4_derived_numbers():
    for page in ("architecture.md", "speculative-decoding.md"):
        text = _read("docs", page)
        for para in text.split("\n\n"):
            if "93.4 ms" in para and "4.58 tokens" in para:
                assert "49.89 tok/s" not in para or "mean of the" in para, (page, para[:300])
                assert "which gives 49.89" not in para, (page, para[:300])
    assert "4.58 tokens in 93.4 ms is 49.0 tok/s" in _read("docs", "architecture.md")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:50s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
