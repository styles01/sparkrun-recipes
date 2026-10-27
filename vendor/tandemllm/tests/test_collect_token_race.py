"""One collector token, even when the first two requests race for it.

token() generated lazily with no guard: two first-use requests each produced a
different token and each wrote it to the config. A launcher that read the
stored value between the two writes holds the loser, which the collector then
refuses with 401 forever — self._token cached the winner and was never re-read.
"""
import os
import sys
import threading
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy.collect_api import Collector  # noqa: E402


class _Cfg:
    def __init__(self):
        self.store = {}
        self.writes = []

    def setting(self, key):
        return self.store.get(key)

    def set_setting(self, key, value, who=None):
        time.sleep(0.05)                    # widen the window the old code raced in
        self.writes.append(value)
        self.store[key] = value


def test_first_use_writes_one_token():
    col = Collector.__new__(Collector)
    col._token = None
    col._token_lock = threading.Lock()
    col.env = {}
    cfg = _Cfg()
    col.svc = SimpleNamespace(cfg=cfg)

    got = []
    ts = [threading.Thread(target=lambda: got.append(col.token())) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert len(cfg.writes) == 1, f"first use wrote {len(cfg.writes)} tokens"
    assert got[0] == got[1] == cfg.store["collect_token"]


if __name__ == "__main__":
    test_first_use_writes_one_token()
    print("1 passed")
