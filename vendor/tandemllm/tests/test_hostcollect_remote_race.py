"""Hosts.view() builds one _Remote per key, never two.

view() ran the get-or-create of a _Remote as two unlocked steps; the dashboard
serves requests from ThreadingHTTPServer threads, so two requests missing the
same key could each construct a _Remote — the replaced object's fetch thread
then fetched against a discarded entry, doubling the remote fetch budget.
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proxy import hostcollect as hc  # noqa: E402


def test_two_concurrent_views_build_one_remote():
    created = []
    real = hc._Remote

    def counting(*a, **k):
        time.sleep(0.05)                       # widen the window the old code raced in
        r = real(*a, **k)
        created.append(r)
        return r

    hc._Remote = counting
    try:
        hosts = hc.Hosts(fetch=lambda *a, **k: b"")
        backends = [{"id": "sg", "url": "http://192.0.2.11:30000/v1", "cloud": False,
                     "enabled": True,
                     "host": {"mode": "exporter", "url": "http://192.0.2.11:9100/node"}}]
        ts = [threading.Thread(target=lambda: hosts.view(backends, None, wait=0.0))
              for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        assert len(created) == 1, f"one key built {len(created)} remotes"
    finally:
        hc._Remote = real


if __name__ == "__main__":
    test_two_concurrent_views_build_one_remote()
    print("1 passed")
