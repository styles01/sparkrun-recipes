"""One lock between a CUDA graph capture and the CUDA work of threads that are not the engine's.

A capture runs in CUDA's global mode: while the engine thread captures a graph (the drafter's at
its first use for a context class, a verify graph), a CUDA call another thread makes that a
capture cannot hold -- a copy and its synchronize, a pinned allocation -- fails with
`cudaErrorStreamCaptureUnsupported` and invalidates the capture, whose request then fails. On
2026-10-02 at 21:09 (the first request after the restart) the disk tier's writer
copied a session head's rows while the drafter captured: the writer stopped after one of the
head's 15 chunks, so the head was never readable, and the request ended in an error from the
caching allocator (`info.status != cudaStreamCaptureStatusInvalidated`).

Every capture holds `LOCK` (engine/drafters/draft_graph.py, engine/verify_graph.py), and every
thread that calls CUDA outside the engine's own lock holds it around those calls (the disk
tier's writer, the probe snapshots' writer). The decision models and the narrator run under the
engine's lock, which a capture also holds, and need nothing more. A capture is rare and a
background copy takes milliseconds, so neither side waits long. `tests/test_capture_lock.py`
holds every capture site to it; `tools/capture_race_check.py` checks it on the GPU.
"""

from __future__ import annotations

import threading

LOCK = threading.RLock()
