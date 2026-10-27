"""Collect keeps no ghosts: a restarted engine's rows and a deleted backend's card expire.

Instance ids embed the engine's pid, so every restart added a row to the per-backend
instance table that nothing ever removed; a backend deleted from the config kept its
_Bcard forever too. Instances now expire after an hour without a batch, and the pull
round — which re-reads the config every round anyway — drops cards the config no
longer names.
"""
import os
import sys
import threading
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy.collect_api import Collector, _Backend, INSTANCE_TTL_S  # noqa: E402


def _collector(cfg_backends):
    col = Collector.__new__(Collector)
    col._lock = threading.Lock()
    col.done, col.items, col.loose, col.expect = {}, {}, {}, {}
    col.backends = {}
    col._n = 0
    col.log = lambda *a: None
    col._t = [1000.0]
    col.clock = lambda: col._t[0]
    col.svc = SimpleNamespace(cfg=SimpleNamespace(backends=cfg_backends), ledger=None)
    col._pull = lambda *a: None            # no network in a unit test
    return col


def test_stale_instances_expire():
    col = _collector(lambda: [])
    b = _Backend()
    b.instances["eng:111"] = {"requests": 9, "steps": 4}
    b.inst_t["eng:111"] = 1000.0
    col.backends["eng"] = b
    col._t[0] = 1000.0 + INSTANCE_TTL_S + 1.0
    col._expire()
    assert "eng:111" not in b.instances and "eng:111" not in b.inst_t


def test_fresh_instances_stay_and_a_deleted_backend_goes():
    col = _collector(lambda: [{"id": "kept", "kind": "vllm", "enabled": True}])
    kept, gone = _Backend(), _Backend()
    kept.instances["k:1"] = {"requests": 1, "steps": 0}
    kept.inst_t["k:1"] = 1000.0
    col.backends = {"kept": kept, "gone": gone}
    kept.was_configured = True
    gone.was_configured = True   # it used to be in the config; now it is gone from it
    traffic = _Backend()        # a card that only ingest ever named: stays counted
    col.backends["traffic"] = traffic
    col._pull_round()
    assert "kept" in col.backends and "gone" not in col.backends
    assert "traffic" in col.backends, "a card the config never named is not a ghost yet"
    assert "k:1" in kept.instances, "a fresh instance under TTL stays"


def test_a_deleted_cards_row_does_not_resurrect_it_in_expire():
    # The maintainer's probe: a deleted backend's pending rows expire up to WAIT_S
    # after eviction; counting them through _b() used to re-create the card with
    # was_configured=False, where it then stayed.
    col = _collector(lambda: [])
    b = _Backend()
    b.was_configured = True
    col.backends["gone"] = b
    col.done[("gone", "row:1")] = {"request_id": 1, "t": 1000.0, "end": 1000.0,
                                   "prompt_tokens": 5, "completion_tokens": 2,
                                   "tagged": False}
    col._pull_round()                       # the config names nothing: the card is evicted
    assert "gone" not in col.backends
    assert col.done == {}, "the evicted card's row keys go with it"
    col._t[0] = 2000.0
    col._expire()                           # any leftovers must not re-create the card
    assert "gone" not in col.backends


if __name__ == "__main__":
    test_stale_instances_expire()
    test_fresh_instances_stay_and_a_deleted_backend_goes()
    test_a_deleted_cards_row_does_not_resurrect_it_in_expire()
    print("3 passed")
